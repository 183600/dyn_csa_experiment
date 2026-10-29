"""CPU checks for the local-fusion control ladder and single-factor wiring.

The fusion on/off ablation changes both whether the depth-wise convolution is
trained AND whether its parameters exist, and constructing the convolution can
shift the global RNG stream for every later block parameter.  These checks pin
the capacity-matched frozen-random placebo, the shared-initialization contract,
and the audit wiring.  No result panel is read.
"""
import contextlib
import copy
import io
import json
import os
import sys as _sys
import tempfile
import unittest.mock

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def _tiny(variant, *, vocab=32, d=16, n_layers=2, n_heads=2, d_head=8,
          seq_len=16):
    import torch
    torch.manual_seed(1234)
    return L.SmallGPT(vocab, d, n_layers, n_heads, d_head, seq_len,
                      L.make_layer_cfgs(n_layers, variant))


def _named(model):
    return dict(model.named_parameters())


def test_fusion_control_ladder_accepts_shipped_design():
    rep = L.fusion_control_audit()
    assert rep['status'] == 'ok', rep['violations']
    assert rep['treatment_component'] == 'local_fusion_learning'
    assert rep['total_effect_component'] == 'local_fusion'
    roles = {k: v['role'] for k, v in rep['levels'].items()}
    assert roles == {'learned': 'treatment',
                     'randomized': 'capacity_placebo',
                     'absent': 'absence_control'}, roles
    assert set(rep['bases']) == {'csa', 'hybrid'}
    for base, rec in rep['base_checks'].items():
        assert rec['params_checked'] is True, (base, rec)
        assert rec['treatment_placebo_matched'] is True, (base, rec)
        assert rec['shared_init_matched'] is True, (base, rec)
        assert rec['treatment_absence_delta'] == rec['expected_absence_delta']
        assert rec['randomized']['status'] == 'ok'
        assert rec['absent']['status'] == 'ok'
    for arm in ('csa_dyn_randfuse', 'hybrid_csa_dyn_randfuse'):
        assert arm in L.RUN_ABL['variants'], arm
    assert ('csa_dyn_fuse', 'csa_dyn_randfuse') in {
        (a, b) for a, b, *_ in L.ABLATION_DESIGN['local_fusion_learning']}
    assert ('csa_dyn_fuse', 'csa_dynamic') in {
        (a, b) for a, b, *_ in L.ABLATION_DESIGN['local_fusion']}
    assert 'local_fusion_learning' in L.HYPOTHESIS_DESIGN
    assert 'local_fusion_learning' in L.ABLATION_SESOI
    assert L.single_factor_design_audit(verbose=False)['status'] == 'ok'
    init = L.single_factor_initialization_audit()
    assert init['status'] == 'ok', init['violations']
    assert L.ablation_completeness_audit(
        L.ABLATION_DESIGN, L.RUN_ABL['variants'])['complete'] is True
    assert L.hypothesis_coverage_audit()['status'] == 'ok'


def test_fusion_control_ladder_fails_closed_on_drift():
    # A missing placebo level is the original confound, not a pass.
    ladder = copy.deepcopy(L.FUSION_CONTROL_DESIGN)
    ladder['levels'].pop('randomized')
    assert L.fusion_control_audit(ladder=ladder)['status'] == 'fail'
    # An unscheduled arm.
    ladder = copy.deepcopy(L.FUSION_CONTROL_DESIGN)
    ladder['levels']['randomized']['arms']['csa'] = 'csa_dyn_missing'
    rep = L.fusion_control_audit(ladder=ladder)
    assert rep['status'] == 'fail' and any(
        'not scheduled' in v for v in rep['violations']), rep
    # The placebo must not be a relabelled treatment.
    ladder = copy.deepcopy(L.FUSION_CONTROL_DESIGN)
    ladder['levels']['randomized']['arms']['csa'] = 'csa_dyn_fuse'
    rep = L.fusion_control_audit(ladder=ladder)
    assert rep['status'] == 'fail' and any(
        'relabelled copy' in v or 'cfg diff' in v for v in rep['violations']), rep
    # A second changed knob is not the fusion placebo.
    ladder = copy.deepcopy(L.FUSION_CONTROL_DESIGN)
    ladder['levels']['absent']['arms']['csa'] = 'csa_fix_randidx'
    rep = L.fusion_control_audit(ladder=ladder)
    assert rep['status'] == 'fail' and any(
        'cfg diff' in v for v in rep['violations']), rep
    # Duplicate roles cannot silently drop a level from the checks.
    ladder = copy.deepcopy(L.FUSION_CONTROL_DESIGN)
    ladder['levels']['absent_copy'] = copy.deepcopy(ladder['levels']['absent'])
    rep = L.fusion_control_audit(ladder=ladder)
    assert rep['status'] == 'fail' and any(
        'exactly once' in v for v in rep['violations']), rep


def test_fusion_arms_share_non_fusion_initialization_and_placebo_is_frozen():
    import torch
    base = _named(_tiny('csa_dynamic'))
    treatment = _named(_tiny('csa_dyn_fuse'))
    placebo = _named(_tiny('csa_dyn_randfuse'))
    # The only extra tensors are the fusion kernels.
    assert {n for n in treatment if n not in base} == {
        'blocks.0.attn.fuse_conv.weight', 'blocks.1.attn.fuse_conv.weight'}
    assert {n for n in placebo if n not in base} == {
        'blocks.0.attn.fuse_conv.weight', 'blocks.1.attn.fuse_conv.weight'}
    for name, value in base.items():
        assert torch.equal(value, treatment[name]), name
        assert torch.equal(value, placebo[name]), name
    # Learned is trainable; the same-shape random placebo is frozen and differs.
    assert all(placebo[n].requires_grad is False
               for n in placebo if 'fuse_conv' in n)
    assert all(treatment[n].requires_grad is True
               for n in treatment if 'fuse_conv' in n)
    assert not torch.equal(treatment['blocks.0.attn.fuse_conv.weight'],
                           placebo['blocks.0.attn.fuse_conv.weight'])
    # Hybrid has the same property on its fusable (CSA) layers.
    hbase = _named(_tiny('hybrid_csa_dyn'))
    htreat = _named(_tiny('hybrid_csa_dyn_fuse'))
    hplac = _named(_tiny('hybrid_csa_dyn_randfuse'))
    for name, value in hbase.items():
        assert torch.equal(value, htreat[name]), name
        assert torch.equal(value, hplac[name]), name


def test_single_factor_initialization_audit_detects_rng_shift():
    base_factory = L.make_layer_cfgs
    design = {'fake_component': [('csa_fixed', 'csa_fix_randidx', 'fake')]}

    def shifted_factory(n_layers, arm):
        if arm == 'csa_fix_randidx':
            torch.rand(7)  # simulate an optional module consuming global RNG
        return base_factory(n_layers, arm)

    rep = L.single_factor_initialization_audit(
        design, factory=shifted_factory,
        intervention_keys={'fake_component': {
            ('csa_fixed', 'csa_fix_randidx'): frozenset({'indexer_mode'})}},
        baseline_design={})
    assert rep['status'] == 'fail', rep
    assert any('initialization' in v or 'differ' in v
               for v in rep['violations']), rep['violations']


def test_single_factor_gate_reports_fusion_and_dynamic_study():
    A = __import__('audit_gates')
    fusion = A.audit_fusion_controls()
    assert fusion['status'] == 'ok', fusion
    sf = A.audit_single_factor()
    assert sf['fusion_controls']['status'] == 'ok', sf['fusion_controls']
    assert sf['initialization']['status'] == 'ok', sf['initialization']
    assert sf['dynamic_study']['status'] == 'ok', sf['dynamic_study']
    assert sf['status'] == 'ok', sf
    # The report builder includes the new block.
    import inspect
    src = inspect.getsource(A.run_report)
    assert 'fusion_controls' in src


def test_single_factor_gate_refuses_multi_factor_dynamic_hypothesis():
    A = __import__('audit_gates')
    with open(os.path.join(A.REPO, 'confirmation_frozen_dynamic.json'),
              encoding='utf-8') as f:
        dyn = json.load(f)
    h1 = next(c for c in dyn['primary_comparisons']
              if c.get('hypothesis_id') == 'H1_segmentation_quality'
              and c.get('a') == 'csa_dynamic_rope_randidx_seeded')
    # H1 is mechanism-claim eligible; an operator swap makes it multi-factor.
    h1['a'], h1['b'] = 'hybrid_dynamic_rope', 'csa_dynamic_rope'
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, 'confirmation_frozen_dynamic.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(dyn, f)
        with unittest.mock.patch.object(A, 'REPO', td):
            with contextlib.redirect_stdout(io.StringIO()):
                rep = A.audit_single_factor()
    assert rep['dynamic_study']['status'] == 'fail', rep['dynamic_study']
    assert rep['status'] == 'fail', rep['status']
