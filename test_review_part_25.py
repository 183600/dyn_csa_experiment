"""CPU review checks (split group 25): capacity-matched ablation placebos.

An on/off ablation changes both the mechanism and the parameter/capacity
budget.  The sink ladder adds a same-shape, same-count, frozen random sink so
`learned - random` isolates the learned component while `learned - absent`
remains the total effect.  These checks cover the forward semantics, the
design-time gate, and the audit wiring.
"""
import copy
import json
import os
import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def _tiny(variant, *, vocab=32, d=16, n_layers=2, n_heads=2, d_head=8, seq_len=16):
    import torch
    torch.manual_seed(1234)
    return L.SmallGPT(vocab, d, n_layers, n_heads, d_head, seq_len,
                      L.make_layer_cfgs(n_layers, variant))


def _named(model):
    return dict(model.named_parameters())


def test_random_sink_is_capacity_matched_frozen_and_rng_neutral():
    import torch
    torch.manual_seed(1234)
    learned = _tiny('csa_fixed')
    state_after_learned = torch.get_rng_state().clone()
    torch.manual_seed(1234)
    placebo = _tiny('csa_fix_randomsink')
    state_after_placebo = torch.get_rng_state().clone()
    # The placebo must not draw from the global RNG stream: otherwise it would
    # shift the initialization of every later tensor and stop being a placebo.
    assert torch.equal(state_after_learned, state_after_placebo)
    pl, pp = _named(learned), _named(placebo)
    assert sum(p.numel() for p in pl.values()) == sum(p.numel() for p in pp.values())
    sink_l = {k: v for k, v in pl.items() if k == 'sink' or k.endswith('.sink')}
    sink_p = {k: v for k, v in pp.items() if k == 'sink' or k.endswith('.sink')}
    assert set(sink_l) == set(sink_p) and sink_l
    assert sum(p.numel() for p in sink_l.values()) == sum(p.numel() for p in sink_p.values())
    assert all(p.requires_grad for p in sink_l.values())
    assert all(not p.requires_grad for p in sink_p.values())
    # Shape and the rest of the module are unchanged; the values differ.
    assert all(pl[k].shape == pp[k].shape for k in sink_l)
    ids = torch.arange(16).reshape(1, 16) % 32
    with torch.no_grad():
        assert not torch.allclose(learned(ids), placebo(ids))
    # Deterministic replay.
    again = _tiny('csa_fix_randomsink')
    with torch.no_grad():
        assert torch.allclose(again(ids), placebo(ids))
    # Unknown modes and contradictory configurations fail closed instead of
    # silently running as learned (or as a no-op).
    for bad in ('pooled', '', None, 'Random'):
        try:
            L.AttnCfg(kind='csa', sink_mode=bad)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f'bad sink_mode accepted: {bad!r}')
    for bad in (dict(kind='full', sink_mode='random'),
                dict(kind='csa', use_sink=False, sink_mode='random')):
        try:
            L.AttnCfg(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'random sink without an active sink accepted: {bad!r}')
    # The dense control has the same property.
    dense_learned = _tiny('full_sink')
    dense_placebo = _tiny('full_randomsink')
    assert sum(p.numel() for p in dense_learned.parameters()) == \
        sum(p.numel() for p in dense_placebo.parameters())
    with torch.no_grad():
        assert not torch.allclose(dense_learned(ids), dense_placebo(ids))


def test_random_sink_keeps_prefix_causality():
    import torch
    torch.manual_seed(7)
    m = _tiny('csa_dyn_randomsink', seq_len=16)
    ids = torch.arange(16).reshape(1, 16) % 32
    with torch.no_grad():
        full = m(ids)
        prefix = m(ids[:, :8])
    assert torch.allclose(full[:, :8], prefix, atol=1e-5), \
        (full[:, :8] - prefix).abs().max()
    # The placebo collapses neither to 'off' nor to the learned arm.
    with torch.no_grad():
        off = _tiny('csa_dyn_nosink', seq_len=16)(ids)
        learned = _tiny('csa_dynamic', seq_len=16)(ids)
    assert not torch.allclose(full, off)
    assert not torch.allclose(full, learned)


def test_sink_control_ladder_accepts_shipped_design():
    rep = L.sink_control_audit()
    assert rep['status'] == 'ok', rep['violations']
    roles = {k: v['role'] for k, v in rep['levels'].items()}
    assert roles == {'learned': 'treatment',
                     'randomized': 'capacity_placebo',
                     'absent': 'absence_control'}, roles
    assert set(rep['bases']) == {'fixed', 'dynamic', 'dense'}
    for base, rec in rep['base_checks'].items():
        assert rec['params_checked'] is True, (base, rec)
        assert rec['treatment_placebo_matched'] is True, (base, rec)
        assert rec['treatment_absence_delta'] == rec['expected_absence_delta']
        assert rec['randomized']['status'] == 'ok'
        assert rec['absent']['status'] == 'ok'
    for arm in ('csa_fix_randomsink', 'csa_dyn_randomsink', 'full_randomsink'):
        assert arm in L.RUN_ABL['variants'], arm
    assert ('csa_fixed', 'csa_fix_randomsink') in {
        (a, b) for a, b, *_ in L.ABLATION_DESIGN['attention_sink_learning']}
    assert ('csa_fixed', 'csa_fix_nosink') in {
        (a, b) for a, b, *_ in L.ABLATION_DESIGN['attention_sink']}
    assert L.single_factor_design_audit(verbose=False)['status'] == 'ok'
    assert L.ablation_completeness_audit(
        L.ABLATION_DESIGN, L.RUN_ABL['variants'])['complete'] is True
    assert L.hypothesis_coverage_audit()['status'] == 'ok'
    assert L.factorial_design_audit()['status'] == 'ok'
    assert L.interaction_sesoi_audit()['status'] == 'ok'
    # The learned-vs-placebo factorial is declared and capacity-matched.
    fac = L.FACTORIAL_DESIGN['dynamic_blocking_x_attention_sink_learning']
    assert fac['factors'] == ('dynamic_blocking', 'attention_sink_learning')
    assert fac['cells'][('control', 'control')] == 'csa_fix_randomsink'
    assert fac['cells'][('active', 'active')] == 'csa_dynamic'
    assert 'attention_sink_learning' in L.HYPOTHESIS_DESIGN
    assert 'attention_sink_learning' in L.ABLATION_SESOI
    assert 'dynamic_blocking_x_attention_sink_learning' in L.INTERACTION_SESOI


def test_sink_control_ladder_fails_closed_on_drift():
    # Missing absence level: a learned-vs-random contrast alone cannot show
    # that the sink parameter is useful at all.
    ladder = copy.deepcopy(L.SINK_CONTROL_DESIGN)
    ladder['levels'].pop('absent')
    assert L.sink_control_audit(ladder=ladder)['status'] == 'fail'
    # A level claiming an already-used role must not be silently dropped.
    ladder = copy.deepcopy(L.SINK_CONTROL_DESIGN)
    ladder['levels']['absent_copy'] = copy.deepcopy(ladder['levels']['absent'])
    rep = L.sink_control_audit(ladder=ladder)
    assert rep['status'] == 'fail' and any(
        'exactly once' in v for v in rep['violations']), rep
    # An unscheduled placebo.
    ladder = copy.deepcopy(L.SINK_CONTROL_DESIGN)
    ladder['levels']['randomized']['arms']['fixed'] = 'csa_fix_missing'
    rep = L.sink_control_audit(ladder=ladder)
    assert rep['status'] == 'fail' and any(
        'not scheduled' in v for v in rep['violations']), rep
    # A relabelled copy is not an independent control level.
    ladder = copy.deepcopy(L.SINK_CONTROL_DESIGN)
    ladder['levels']['randomized']['arms']['fixed'] = 'csa_fixed'
    rep = L.sink_control_audit(ladder=ladder)
    assert rep['status'] == 'fail' and any(
        'reused' in v or 'capacity placebo must' in v for v in rep['violations']), rep
    # A placebo that also moves a second knob is confounded.
    def _factory_extra(n_layers, variant):
        cfgs = L.make_layer_cfgs(n_layers, variant)
        if variant == 'csa_fix_randomsink':
            import dataclasses
            return [dataclasses.replace(c, index_topk=8) for c in cfgs]
        return cfgs
    rep = L.sink_control_audit(factory=_factory_extra)
    assert rep['status'] == 'fail' and any(
        'index_topk' in v for v in rep['violations']), rep
    # An absence arm that removes more than the sink parameters.
    ladder = copy.deepcopy(L.SINK_CONTROL_DESIGN)
    ladder['levels']['absent']['arms']['fixed'] = 'csa_fix_topk8'
    rep = L.sink_control_audit(ladder=ladder)
    assert rep['status'] == 'fail', rep
    # Malformed declarations do not crash.
    for bad in ([], 'x', {'base_arms': {}, 'levels': {}}):
        rep = L.sink_control_audit(ladder=bad)
        assert rep['status'] == 'fail', rep
    # An undeclared leave-one-out family for the placebo is refused.
    design = dict(L.ABLATION_DESIGN)
    design.pop('attention_sink_learning')
    rep = L.sink_control_audit(design=design)
    assert rep['status'] == 'fail' and any(
        'no attention_sink_learning contrast' in v for v in rep['violations']), rep


def test_audit_gate_entry_reports_sink_controls():
    A = __import__('audit_gates')
    rep = A.audit_sink_controls()
    assert rep['status'] == 'ok', rep
    sf = A.audit_single_factor()
    assert sf['sink_controls']['status'] == 'ok', sf['sink_controls']
    assert sf['status'] == 'ok', sf
    # The report builder includes the new block.
    import inspect
    src = inspect.getsource(A.run_report)
    assert 'sink_controls' in src
