"""CPU checks for the functional manipulation gate on ablation designs.

The config-diff and initialization gates can both pass while the declared
intervention is inert: a selector can cover every block, a content table can
be unused, or a "frozen" parameter can still receive a gradient through a
second path.  These checks pin the design-time manipulation audit: it must
require real forward evidence for architectural contrasts, real
parameter/gradient/objective evidence for training-only contrasts, fail closed
on inert arms, and restore the caller's RNG state.  No result panel is read.
"""
import contextlib
import copy
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


def test_shipped_manipulation_design_passes():
    rep = L.ablation_manipulation_audit()
    assert rep['status'] == 'ok', rep['violations']
    assert rep['kind'] == 'ablation_manipulation'
    declared = sum(len(v) for v in L.ABLATION_DESIGN.values())
    assert rep['n_contrasts'] == declared, (rep['n_contrasts'], declared)
    assert (rep['n_forward_mode'] + rep['n_training_mode']
            == rep['n_contrasts'])
    assert rep['n_forward_evidence'] > 0
    assert rep['n_training_evidence'] > 0
    assert not rep['inert'] and not rep['not_verifiable']
    for label, rec in rep['contrasts'].items():
        assert rec['status'] == 'ok', (label, rec)
        assert rec['mode'] in ('forward', 'training')
        assert rec['evidence'] in ('forward', 'training'), (label, rec)
        if rec['mode'] == 'forward':
            assert rec['forward']['changed'] is True, (label, rec)
        else:
            assert rec['training_changed'] is True, (label, rec)
    # Every declared component is covered by the closed declaration table.
    comps = {c for c, _a, _b in
             L._manip_pair_items(L.ABLATION_DESIGN)[0]}
    assert comps == set(L.ABLATION_MANIPULATION_DESIGN), comps


def test_training_only_contrasts_are_real_path_interventions():
    # Identity fusion, the frozen gate and the loss-term-only regularizer are
    # deliberately the same forward function at initialization.  Their
    # intervention must still be visible on the training path.
    rep = L.ablation_manipulation_audit()
    for label in (
            'dynamic_blocking_learning: csa_dynamic vs csa_dyn_frozengate',
            'compression_regularizer: csa_dynamic vs csa_dyn_nocomp',
            'local_fusion: csa_dyn_fuse vs csa_dynamic',
            'task_learned_boundary: csa_dynamic_task vs csa_dynamic'):
        rec = rep['contrasts'][label]
        assert rec['status'] == 'ok', (label, rec)
        assert rec['mode'] == 'training', (label, rec)
        assert rec['forward']['changed'] is False, (label, rec)
        assert rec['training_changed'] is True, (label, rec)
    gate = rep['contrasts'][
        'dynamic_blocking_learning: csa_dynamic vs csa_dyn_frozengate']
    assert gate['parameter']['trainability_diff'], gate
    reg = rep['contrasts'][
        'compression_regularizer: csa_dynamic vs csa_dyn_nocomp']
    assert reg['objective']['changed'] is True, reg
    assert any('delta_logit' in n for n in reg['gradient']['names']), reg
    fuse = rep['contrasts']['local_fusion: csa_dyn_fuse vs csa_dynamic']
    assert fuse['parameter']['names_only_a'], fuse
    assert any('fuse_conv' in n for n in fuse['parameter']['names_only_a']), fuse


def test_inert_factory_fails_closed():
    def same_fixed(n_layers, _arm):
        return L.make_layer_cfgs(n_layers, 'csa_fixed')

    rep = L.ablation_manipulation_audit(factory=same_fixed)
    assert rep['status'] == 'fail'
    assert rep['inert']
    assert len(rep['inert']) == rep['n_contrasts'], rep['inert']
    assert any('inert' in v for v in rep['violations']), rep['violations']


def test_declaration_is_a_closed_contract_and_mode_is_a_hard_gate():
    # Missing declaration for a live contrast.
    decl = copy.deepcopy(L.ABLATION_MANIPULATION_DESIGN)
    decl.pop('hybrid_hierarchy')
    rep = L.ablation_manipulation_audit(declarations=decl)
    assert rep['status'] == 'fail'
    assert any('no manipulation declaration' in v for v in rep['violations'])

    # Stale declaration for a component that is no longer in the design.
    decl = copy.deepcopy(L.ABLATION_MANIPULATION_DESIGN)
    decl['bogus_component'] = {'mode': 'forward'}
    rep = L.ablation_manipulation_audit(declarations=decl)
    assert rep['status'] == 'fail'
    assert any('stale declaration' in v for v in rep['violations'])

    # Unknown mode is not a pass.
    decl = copy.deepcopy(L.ABLATION_MANIPULATION_DESIGN)
    decl['hybrid_hierarchy'] = {'mode': 'maybe'}
    rep = L.ablation_manipulation_audit(declarations=decl)
    assert rep['status'] == 'fail'

    # The frozen gate is the same function at init; declaring it 'forward'
    # must expose the inertness instead of silently accepting the mismatch.
    decl = copy.deepcopy(L.ABLATION_MANIPULATION_DESIGN)
    decl['dynamic_blocking_learning'] = {'mode': 'forward'}
    rep = L.ablation_manipulation_audit(declarations=decl)
    assert rep['status'] == 'fail'
    assert ('dynamic_blocking_learning: csa_dynamic vs csa_dyn_frozengate'
            in rep['inert']), rep['inert']


def test_malformed_design_and_arguments_fail_closed():
    design = copy.deepcopy(L.ABLATION_DESIGN)
    design['broken'] = [('only_one_arm',)]
    rep = L.ablation_manipulation_audit(design=design)
    assert rep['status'] == 'fail'
    assert rep['violations']
    for bad in (
            {'n_layers': 0},
            {'probe_tokens': -1},
            {'seed': -2},
            {'forward_tol': -1.0},
    ):
        rep = L.ablation_manipulation_audit(**bad)
        assert rep['status'] == 'fail', (bad, rep)
    rep = L.ablation_manipulation_audit(design={})
    assert rep['status'] == 'fail'
    rep = L.ablation_manipulation_audit(declarations={})
    assert rep['status'] == 'fail'


def test_audit_gate_and_report_wire_the_manipulation_block():
    A = __import__('audit_gates')
    with contextlib.redirect_stdout(io.StringIO()):
        rep = A.audit_manipulation()
        sf = A.audit_single_factor()
    assert rep['status'] == 'ok', rep['violations']
    assert sf['manipulation']['status'] == 'ok', sf['manipulation']
    assert sf['status'] == 'ok', sf['status']
    src = inspect.getsource(A.run_report)
    assert "'manipulation'" in src
    src_cli = inspect.getsource(A._run_cli)
    for alias in ('manipulation', 'intervention_fidelity',
                  'ablation_manipulation', 'functional_ablation'):
        assert alias in src_cli, alias


def test_manipulation_probe_restores_rng_state():
    torch.manual_seed(987654)
    before = torch.get_rng_state().clone()
    L.ablation_manipulation_audit(n_layers=1, probe_tokens=64)
    after = torch.get_rng_state()
    assert torch.equal(before, after), 'the design probe leaked RNG state'
