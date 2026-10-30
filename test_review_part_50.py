"""CPU checks for the learned/frozen boundary-gate single-factor control.

The dynamic-vs-fixed contrast changes the cut geometry, adds the per-layer
``delta_logit`` offset and applies the compression objective.  These checks
pin the capacity- and objective-matched placebo that keeps the parameter, its
initial value and the comp_reg term fixed while freezing only its gradient, so
``learned - frozen`` is a true single-factor learned-boundary contrast.  No
result panel is read.
"""
import contextlib
import copy
import dataclasses as _dc
import inspect
import io
import os
import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)

import pytest as _pytest
import torch


def _tiny(variant, *, vocab=32, d=16, n_layers=2, n_heads=2, d_head=8,
          seq_len=16, seed=1234):
    torch.manual_seed(seed)
    return L.SmallGPT(vocab, d, n_layers, n_heads, d_head, seq_len,
                      L.make_layer_cfgs(n_layers, variant))


def test_gate_control_ladder_accepts_shipped_design():
    rep = L.gate_control_audit()
    assert rep['status'] == 'ok', rep['violations']
    assert rep['treatment_component'] == 'dynamic_blocking_learning'
    assert rep['total_effect_component'] == 'dynamic_blocking'
    roles = {k: v['role'] for k, v in rep['levels'].items()}
    assert roles == {'learned': 'treatment', 'frozen': 'capacity_placebo'}, roles
    assert set(rep['bases']) == {'csa', 'hybrid'}
    for base, rec in rep['base_checks'].items():
        assert rec['learned']['status'] == 'ok', (base, rec)
        assert rec['frozen']['status'] == 'ok', (base, rec)
        cap = rec['capacity_check']
        assert cap['status'] == 'ok', (base, cap)
        assert cap['a_params'] == cap['b_params'], (base, cap)
        assert cap['delta_params_a'] == cap['delta_params_b'] > 0, (base, cap)
    for arm in ('csa_dyn_frozengate', 'hybrid_dyn_frozengate'):
        assert arm in L.RUN_ABL['variants'], arm
    pairs = {(a, b) for a, b, *_ in L.ABLATION_DESIGN['dynamic_blocking_learning']}
    assert ('csa_dynamic', 'csa_dyn_frozengate') in pairs, pairs
    assert ('hybrid_dynamic', 'hybrid_dyn_frozengate') in pairs, pairs
    assert 'dynamic_blocking_learning' in L.ABLATION_SESOI
    assert 'dynamic_blocking_learning' in L.HYPOTHESIS_DESIGN
    assert L.single_factor_design_audit(verbose=False)['status'] == 'ok'
    init = L.single_factor_initialization_audit()
    assert init['status'] == 'ok', init['violations']
    assert ('dynamic_blocking_learning: csa_dynamic vs csa_dyn_frozengate'
            in init['pairs']), init['pairs']
    assert L.ablation_completeness_audit(
        L.ABLATION_DESIGN, L.RUN_ABL['variants'])['complete'] is True
    assert L.hypothesis_coverage_audit()['status'] == 'ok'


def test_frozen_gate_matches_capacity_objective_and_initialization():
    for base, placebo in (('csa_dynamic', 'csa_dyn_frozengate'),
                          ('hybrid_dynamic', 'hybrid_dyn_frozengate')):
        ma = _tiny(base)
        mb = _tiny(placebo)
        sa = dict(ma.named_parameters())
        sb = dict(mb.named_parameters())
        assert set(sa) == set(sb)
        assert (sum(p.numel() for p in sa.values())
                == sum(p.numel() for p in sb.values()))
        delta_a = [n for n in sa if 'delta_logit' in n]
        delta_b = [n for n in sb if 'delta_logit' in n]
        assert delta_a and delta_a == delta_b
        for n in sa:
            if n in delta_a:
                # Same shape and same frozen initial value: only trainability
                # differs, so the control cannot be a different intervention
                # value.
                assert sa[n].shape == sb[n].shape
                assert torch.equal(sa[n], sb[n]), n
                assert sa[n].requires_grad is True, n
                assert sb[n].requires_grad is False, n
            else:
                assert torch.equal(sa[n], sb[n]), n
        oa = L.train_objective(ma, comp_lambda=0.05, delta_lr_mult=10.0)
        ob = L.train_objective(mb, comp_lambda=0.05, delta_lr_mult=10.0)
        # Capacity and the compression objective are matched.
        for k in ('comp_reg_applies', 'comp_lambda', 'comp_reg_layers',
                  'comp_reg_configured_layers', 'comp_lambda_mults',
                  'comp_lambda_effective', 'delta_param_count',
                  'boundary_policy', 'loss_terms'):
            assert oa[k] == ob[k], (base, k, oa[k], ob[k])
        assert oa['delta_trainable'] is True and ob['delta_trainable'] is False
        assert oa['delta_trainable_param_count'] > 0
        assert ob['delta_trainable_param_count'] == 0
        assert oa['delta_lr_effective'] == 10.0
        assert ob['delta_lr_effective'] == 1.0
        assert oa['comp_reg_applies'] is True


def test_frozen_gate_is_the_same_function_at_initialization():
    # A frozen offset is the same forward function at init; the intervention is
    # whether that parameter can move during training, not a different value.
    ids = torch.randint(0, 32, (1, 16))
    for base, placebo in (('csa_dynamic', 'csa_dyn_frozengate'),
                          ('hybrid_dynamic', 'hybrid_dyn_frozengate')):
        ma = _tiny(base)
        mb = _tiny(placebo)
        ma.eval()
        mb.eval()
        with torch.no_grad():
            la = ma(ids)
            lb = mb(ids)
        assert torch.equal(la, lb), base


def test_gate_mode_validation_is_fail_closed():
    with _pytest.raises(ValueError):
        L.AttnCfg(gate_mode='bogus')
    with _pytest.raises(ValueError):
        L.AttnCfg(gate_mode='frozen', chunking='fixed', dynamic=False)
    # A full layer never constructs delta_logit, even if the chunking field is
    # left at its learnable value: freezing it would silently do nothing.
    with _pytest.raises(ValueError):
        L.AttnCfg(kind='full', chunking='cosine_learnable', gate_mode='frozen')
    # A frozen gate needs a learnable offset to freeze.
    cfg = L.AttnCfg(gate_mode='frozen', chunking='cosine_learnable', dynamic=True)
    assert cfg.gate_mode == 'frozen'


def test_gate_objective_reports_the_declared_trainability_difference():
    a = _tiny('csa_dynamic')
    b = _tiny('csa_dyn_frozengate')
    oa = L.train_objective(a, comp_lambda=0.05, delta_lr_mult=10.0)
    ob = L.train_objective(b, comp_lambda=0.05, delta_lr_mult=10.0)
    reason, keys = L._objective_diff({'train_objective': oa},
                                     {'train_objective': ob})
    assert reason and keys == ['delta_lr_effective', 'delta_trainable'], keys
    assert set(keys) <= L.OBJECTIVE_INTERVENTION_KEYS[
        'dynamic_blocking_learning'], keys
    # A one-sided optional field is unverifiable, not silently "matched".
    legacy = {k: v for k, v in oa.items() if k != 'delta_trainable'}
    reason, keys = L._objective_diff({'train_objective': legacy},
                                     {'train_objective': oa})
    assert reason and reason.startswith('unverifiable') and keys == [], reason


def test_gate_control_audit_validates_labels_and_keeps_base_diagnostics():
    ladder = copy.deepcopy(L.GATE_CONTROL_DESIGN)
    ladder['component'] = 'something_else'
    assert L.gate_control_audit(ladder=ladder)['status'] == 'fail'
    ladder = copy.deepcopy(L.GATE_CONTROL_DESIGN)
    ladder['total_effect_component'] = 'something_else'
    assert L.gate_control_audit(ladder=ladder)['status'] == 'fail'
    # The total-effect family the learned contrast is conditioned on must be
    # declared, not silently assumed.
    design = dict(L.ABLATION_DESIGN)
    design.pop('dynamic_blocking')
    assert L.gate_control_audit(design=design)['status'] == 'fail'
    # A missing arm is still diagnosable per base instead of dropping the base.
    ladder = copy.deepcopy(L.GATE_CONTROL_DESIGN)
    ladder['levels']['frozen']['arms']['csa'] = 'csa_dyn_missing'
    rep = L.gate_control_audit(ladder=ladder)
    assert rep['status'] == 'fail'
    assert rep['base_checks']['csa']['frozen']['status'] == 'missing', rep


def test_gate_control_ladder_fails_closed():
    # Missing placebo level.
    ladder = copy.deepcopy(L.GATE_CONTROL_DESIGN)
    ladder['levels'].pop('frozen')
    assert L.gate_control_audit(ladder=ladder)['status'] == 'fail'
    # Unscheduled placebo arm.
    ladder = copy.deepcopy(L.GATE_CONTROL_DESIGN)
    ladder['levels']['frozen']['arms']['csa'] = 'csa_dyn_missing'
    rep = L.gate_control_audit(ladder=ladder)
    assert rep['status'] == 'fail' and any(
        'not scheduled' in v for v in rep['violations']), rep
    # A relabelled treatment is not a placebo.
    ladder = copy.deepcopy(L.GATE_CONTROL_DESIGN)
    ladder['levels']['frozen']['arms']['csa'] = 'csa_dynamic'
    rep = L.gate_control_audit(ladder=ladder)
    assert rep['status'] == 'fail' and any(
        'relabelled' in v or 'reused' in v for v in rep['violations']), rep
    # A second changed knob is not the gate placebo.
    ladder = copy.deepcopy(L.GATE_CONTROL_DESIGN)
    ladder['levels']['frozen']['arms']['csa'] = 'csa_dyn_nocomp'
    rep = L.gate_control_audit(ladder=ladder)
    assert rep['status'] == 'fail' and any(
        'cfg diff' in v for v in rep['violations']), rep
    # Duplicate roles cannot silently drop a level from the checks.
    ladder = copy.deepcopy(L.GATE_CONTROL_DESIGN)
    ladder['levels']['frozen_copy'] = copy.deepcopy(ladder['levels']['frozen'])
    rep = L.gate_control_audit(ladder=ladder)
    assert rep['status'] == 'fail' and any(
        'exactly once' in v or 'exactly treatment' in v
        for v in rep['violations']), rep
    # The declaration must say the placebo uses the frozen mode.
    ladder = copy.deepcopy(L.GATE_CONTROL_DESIGN)
    ladder['levels']['frozen']['gate_mode'] = 'learned'
    rep = L.gate_control_audit(ladder=ladder)
    assert rep['status'] == 'fail' and any(
        'gate_mode' in v for v in rep['violations']), rep
    # Removing the consuming single-factor declaration fails closed.
    design = dict(L.ABLATION_DESIGN)
    design.pop('dynamic_blocking_learning')
    rep = L.gate_control_audit(design=design)
    assert rep['status'] == 'fail' and any(
        'no dynamic_blocking_learning contrast' in v
        for v in rep['violations']), rep
    # Injection of an extra config knob while keeping the declared mode label.
    def widened(n_layers, arm):
        cfgs = L.make_layer_cfgs(n_layers, arm)
        if arm == 'csa_dyn_frozengate':
            return [_dc.replace(c, target_block_tokens=8) for c in cfgs]
        return cfgs
    rep = L.gate_control_audit(factory=widened)
    assert rep['status'] == 'fail' and any(
        'cfg diff' in v for v in rep['violations']), rep


def test_single_factor_gate_reports_gate_controls():
    A = __import__('audit_gates')
    with contextlib.redirect_stdout(io.StringIO()):
        gate = A.audit_gate_controls()
        sf = A.audit_single_factor()
    assert gate['status'] == 'ok', gate['violations']
    assert sf['gate_controls']['status'] == 'ok', sf['gate_controls']
    assert sf['initialization']['status'] == 'ok', sf['initialization']
    assert sf['status'] == 'ok', sf
    # The design gate counts the new declared pairs and classifies them as
    # knob-level single factors (the frozen gate changes only gate_mode).
    design = sf['design']
    assert design['status'] == 'ok', design['violations']
    for label in design['single_factor_pairs']:
        if 'dynamic_blocking_learning' in label:
            assert design['pairs'][label]['actual_keys'] == ['gate_mode']
            break
    else:  # pragma: no cover - the shipped design always has the two pairs
        raise AssertionError('gate-learning pairs missing from single_factor_pairs')
    src = inspect.getsource(A.run_report)
    assert 'gate_controls' in src
