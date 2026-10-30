"""CPU checks for the within-checkpoint lesion ladder.

A between-model on/off ablation shows that *training with* a component changed
the model; it does not show that the trained model *uses* the component at
inference.  These checks cover the surgical lesion context manager, the
forward manipulation check (including the no-op initialisation trap), the
paired PPL/TOST verdict, the design gate and the audit wiring.  No result
panel or checkpoint is read.
"""
import contextlib
import io
import inspect
import json
import os
import sys as _sys
import tempfile

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)

import torch


def _tiny(variant, *, vocab=32, d=16, n_layers=2, n_heads=2, d_head=8,
          seq_len=16):
    torch.manual_seed(1234)
    return L.SmallGPT(vocab, d, n_layers, n_heads, d_head, seq_len,
                      L.make_layer_cfgs(n_layers, variant))


def _ids(seq_len=16, vocab=32):
    return torch.arange(seq_len).reshape(1, seq_len) % vocab


def _named(model):
    return dict(model.named_parameters())


def _set_sink(model, value=0.37):
    with torch.no_grad():
        for name, p in _named(model).items():
            if name == 'sink' or name.endswith('.sink'):
                p.copy_(torch.linspace(0.1, 0.1 + value, p.numel()).reshape_as(p))


def _randomise_fusion(model, seed=9):
    with torch.no_grad():
        for name, p in _named(model).items():
            if name.endswith('.fuse_conv.weight'):
                g = torch.Generator().manual_seed(seed)
                p.copy_(torch.randn(p.shape, generator=g) * 0.2)


def test_lesion_design_accepts_shipped_components():
    torch.manual_seed(7)
    rng_before = torch.get_rng_state().clone()
    rep = L.lesion_design_audit()
    assert torch.equal(torch.get_rng_state(), rng_before), \
        'the design audit must not consume/redirect the caller RNG stream'
    assert rep['status'] == 'ok', rep['violations']
    assert set(rep['components']) == set(L.LESION_COMPONENTS)
    for component, rec in rep['components'].items():
        assert rec['status'] == 'ok', (component, rec)
        assert rec['arms'], component
        for arm, arec in rec['arms'].items():
            assert arec['scheduled'] is True, (component, arm)
            assert arec['in_leave_one_out_family'] is True, (component, arm)
            assert arec['applied'] is True, (component, arm)
            assert (arec['changed_fields']
                    or arec.get('module_overrides')), (component, arm)
            assert (arec['state'] or {}).get('status') in (
                'present', 'trivial', 'not_parameterized'), (component, arm)
    # No result directory is present, so the panel half must not fabricate a
    # pass; it still returns the design block.
    import audit_gates as A
    gate = A.audit_lesion(verbose=False)
    assert gate['status'] == 'not_verifiable_locally', gate
    assert gate['design']['status'] == 'ok'
    assert 'lesion' in inspect.getsource(A.run_report)
    assert "'lesion': audit_lesion" in inspect.getsource(A._run_cli)


def test_lesion_context_restores_config_and_module_on_exception():
    m = _tiny('csa_dyn_fuse')
    attn = m.blocks[0].attn
    original_cfg = attn.cfg
    original_conv = attn.fuse_conv
    original_fuse_mode = attn.cfg.fuse_mode
    before = L.state_dict_sha256(m.state_dict())
    assert original_conv is not None
    with contextlib.suppress(RuntimeError):
        with L.component_lesion(m, 'local_fusion') as rec:
            assert rec['component'] == 'local_fusion'
            assert rec['module_overrides'] == ['fuse_conv']
            assert attn.cfg is not original_cfg
            assert attn.fuse_conv is None
            assert attn.cfg.fuse_mode == original_fuse_mode
            raise RuntimeError('deliberate')
    assert attn.cfg is original_cfg
    assert attn.cfg.fuse_mode == original_fuse_mode
    assert attn.fuse_conv is original_conv
    assert L.state_dict_sha256(m.state_dict()) == before
    # Unknown and already-absent components fail closed instead of silently
    # yielding an empty context.
    for bad in ('not_a_component', '', None):
        try:
            with L.component_lesion(m, bad):
                pass
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f'unknown lesion component accepted: {bad!r}')
    absent = _tiny('csa_fix_nosink')
    try:
        with L.component_lesion(absent, 'attention_sink'):
            pass
    except ValueError:
        pass
    else:
        raise AssertionError('a sink lesion on a sink-free arm must fail closed')


def test_sink_lesion_is_a_real_manipulation_check():
    m = _tiny('csa_fixed')
    ids = _ids()
    # A freshly initialised learned sink is exactly zero, but a zero sink is
    # still a uniform extra attention slot: it is functionally active, not a
    # no-op.  This is the opposite trap from an identity-initialised fusion.
    state0 = L.component_lesion_state(m, 'attention_sink')
    assert state0['status'] == 'present'
    assert state0['n_nonzero'] == 0 and state0['zero_valued'] is True
    assert state0['trivial'] is False
    sens0 = L.lesion_sensitivity(m, ids, 'attention_sink')
    assert sens0['status'] == 'changed' and sens0['changed'] is True
    assert sens0['weights_unchanged'] is True
    # A non-trivial sink is a genuine forward intervention.
    _set_sink(m, 0.37)
    state1 = L.component_lesion_state(m, 'attention_sink')
    assert state1['status'] == 'present' and state1['n_nonzero'] > 0
    before = L.state_dict_sha256(m.state_dict())
    sens1 = L.lesion_sensitivity(m, ids, 'attention_sink')
    assert sens1['status'] == 'changed' and sens1['changed'] is True
    assert sens1['max_abs_logit_delta'] > sens1['threshold']
    assert sens1['weights_unchanged'] is True
    assert L.state_dict_sha256(m.state_dict()) == before
    # Under the lesion the sink is disconnected: changing its values cannot
    # change the output, while the intact pass still depends on them.
    with torch.no_grad():
        intact = m(ids)
        with L.component_lesion(m, 'attention_sink'):
            a = m(ids)
            _set_sink(m, 0.91)
            b = m(ids)
        assert torch.allclose(a, b)
        changed = m(ids)
    assert not torch.allclose(intact, changed)


def test_fusion_lesion_is_a_real_manipulation_check():
    m = _tiny('csa_dyn_fuse')
    ids = _ids()
    state0 = L.component_lesion_state(m, 'local_fusion')
    assert state0['status'] == 'trivial'
    assert state0['distance_from_identity'] == 0.0
    sens0 = L.lesion_sensitivity(m, ids, 'local_fusion')
    assert sens0['status'] == 'unchanged'
    _randomise_fusion(m, seed=9)
    state1 = L.component_lesion_state(m, 'local_fusion')
    assert state1['status'] == 'present'
    assert state1['distance_from_identity'] > 0.0
    sens1 = L.lesion_sensitivity(m, ids, 'local_fusion')
    assert sens1['status'] == 'changed'
    assert sens1['weights_unchanged'] is True
    # Removing fuse_conv disconnects the kernel without changing cfg.fuse_mode
    # (which participates in backend dispatch).  Hold a reference to the live
    # kernel: the module is detached from the model during the lesion, so it
    # cannot be found through named_parameters() inside the context.
    kernels = [blk.attn.fuse_conv.weight for blk in m.blocks
               if getattr(blk.attn, 'fuse_conv', None) is not None]
    assert len(kernels) == 2
    with torch.no_grad():
        intact = m(ids)
        with L.component_lesion(m, 'local_fusion'):
            a = m(ids)
            for p in kernels:
                p.copy_(torch.randn_like(p) * 0.2)
            b = m(ids)
        assert torch.allclose(a, b)
        changed = m(ids)
    assert not torch.allclose(intact, changed)


def test_content_lesion_changes_forward_without_touching_weights():
    m = _tiny('csa_fixed')
    ids = _ids()
    state = L.component_lesion_state(m, 'content_pooling')
    assert state['status'] == 'not_parameterized'
    before = L.state_dict_sha256(m.state_dict())
    sens = L.lesion_sensitivity(m, ids, 'content_pooling')
    assert sens['status'] == 'changed', sens
    assert sens['changed'] is True
    assert sens['weights_unchanged'] is True
    assert L.state_dict_sha256(m.state_dict()) == before


def test_lesion_effect_verdicts_and_fail_closed_inputs():
    ok = {'changed': True, 'weights_unchanged': True}
    base = [10.0] * 6
    necessary = L.lesion_effect(base, [11.0] * 6, sensitivity=ok,
                                component='attention_sink')
    assert necessary['verdict'] == 'necessary', necessary
    assert necessary['claim_allowed'] is True
    assert necessary['mean_log_ratio'] > necessary['sesoi_log_margin']
    assert abs(necessary['mean_pct'] - 10.0) < 1e-9
    harmful = L.lesion_effect(base, [9.0] * 6, sensitivity=ok)
    assert harmful['verdict'] == 'harmful'
    assert harmful['mean_pct'] < 0
    redundant = L.lesion_effect(base, [10.05] * 6, sensitivity=ok)
    assert redundant['verdict'] == 'redundant', redundant
    assert redundant['tost']['equivalent'] is True
    exact_zero = L.lesion_effect(base, base, sensitivity=ok)
    assert exact_zero['verdict'] == 'redundant'
    assert exact_zero['mean_log_ratio'] == 0.0
    inert = L.lesion_effect(base, [11.0] * 6,
                            sensitivity={'changed': False,
                                         'weights_unchanged': True})
    assert inert['verdict'] == 'inert' and inert['claim_allowed'] is True
    unchecked = L.lesion_effect(base, [11.0] * 6, sensitivity=None)
    assert unchecked['verdict'] == 'not_checked'
    assert unchecked['claim_allowed'] is False
    invalid = L.lesion_effect(base, [11.0] * 6,
                              sensitivity={'changed': True,
                                           'weights_unchanged': False})
    assert invalid['verdict'] == 'invalid'
    under = L.lesion_effect([10.0] * 3, [11.0] * 3, sensitivity=ok)
    assert under['verdict'] == 'underpowered', under
    # Paired input contract: no silent complete-case subset.
    for kwargs in (dict(intact_ppl=[10.0] * 6, lesion_ppl=[11.0] * 5),
                   dict(intact_ppl=[10.0, True], lesion_ppl=[11.0, 11.0]),
                   dict(intact_ppl=[10.0, 0.0], lesion_ppl=[11.0, 11.0]),
                   dict(intact_ppl=[10.0] * 6, lesion_ppl=[11.0] * 6,
                        seeds=[0] * 6),
                   dict(intact_ppl=[10.0] * 6, lesion_ppl=[11.0] * 6,
                        relative_pct=0.0),
                   dict(intact_ppl=[10.0] * 6, lesion_ppl=[11.0] * 6,
                        alpha=1.5)):
        try:
            L.lesion_effect(**kwargs)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f'invalid lesion_effect input accepted: {kwargs!r}')


def _records(value=11.0, *, component='attention_sink', arm='csa_dynamic',
             n=6, ckpt='ckpt-a', eval_fp='eval-a', changed=True,
             weights=True):
    return [{'component': component, 'arm': arm, 'seed': s,
             'checkpoint_sha256': f'{ckpt}-{s}', 'eval_fp': eval_fp,
             'intact_ppl': 10.0, 'lesion_ppl': value,
             'sensitivity': {'changed': changed,
                             'weights_unchanged': weights}}
            for s in range(n)]


def test_lesion_panel_audit_groups_and_fails_closed():
    rep = L.lesion_panel_audit({'records': _records()})
    assert rep['status'] == 'ok', rep
    assert rep['n_panels'] == 1
    panel = rep['panels']['attention_sink/csa_dynamic']
    assert panel['verdict'] == 'necessary', panel
    assert panel['n'] == 6 and panel['mean_pct'] > 1.0
    assert rep['necessary_panels'] == ['attention_sink/csa_dynamic']
    # A no-op lesion is a valid "inert" reading but is not a clean pass.
    inert = L.lesion_panel_audit({'records': _records(changed=False)})
    assert inert['panels']['attention_sink/csa_dynamic']['verdict'] == 'inert'
    assert inert['status'] == 'warn', inert
    # Different evaluation rows inside one panel are a second factor.
    mixed = _records()
    mixed[1]['eval_fp'] = 'eval-b'
    assert L.lesion_panel_audit({'records': mixed})['status'] == 'fail'
    # Reusing one checkpoint across training seeds is pseudo-replication, not
    # independent lesion evidence; it is surfaced as a warning.
    shared = _records()
    for rec in shared:
        rec['checkpoint_sha256'] = 'one-checkpoint'
    shared_rep = L.lesion_panel_audit({'records': shared})
    assert shared_rep['status'] == 'warn'
    assert any('pseudo-replication' in w for w in shared_rep['warnings'])
    # Duplicate seed, missing checkpoint, changed weights and an undeclared
    # arm all fail closed.
    dup = _records()
    dup[1]['seed'] = dup[0]['seed']
    assert L.lesion_panel_audit({'records': dup})['status'] == 'fail'
    no_ckpt = _records()
    no_ckpt[0].pop('checkpoint_sha256')
    assert L.lesion_panel_audit({'records': no_ckpt})['status'] == 'fail'
    bad_w = _records(weights=False)
    assert L.lesion_panel_audit({'records': bad_w})['status'] == 'fail'
    bad_arm = _records(arm='csa_fix_nosink')
    assert L.lesion_panel_audit({'records': bad_arm})['status'] == 'fail'
    bad_ppl = _records()
    bad_ppl[0]['lesion_ppl'] = float('nan')
    assert L.lesion_panel_audit({'records': bad_ppl})['status'] == 'fail'
    empty = L.lesion_panel_audit([])
    assert empty['status'] == 'not_verifiable'
    # Gate parameters are validated even when the panel is empty; they are not
    # measurements that can be silently defaulted.
    for bad in (dict(relative_pct=0.0), dict(alpha=1.5), dict(min_n=True),
                dict(require_manipulation='yes')):
        try:
            L.lesion_panel_audit([], **bad)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f'invalid panel parameter accepted: {bad!r}')


def test_audit_lesion_reads_and_replays_an_artifact():
    import audit_gates as A
    with tempfile.TemporaryDirectory() as td:
        good = os.path.join(td, 'lesion.json')
        with open(good, 'w', encoding='utf-8') as f:
            json.dump({'records': _records()}, f)
        with contextlib.redirect_stdout(io.StringIO()):
            rep = A.audit_lesion(good)
        assert rep['status'] == 'ok', rep
        _panel_doc = next(iter(rep['panels'].values()))
        assert 'attention_sink/csa_dynamic' in _panel_doc['panels']
        assert _panel_doc['panels']['attention_sink/csa_dynamic']['verdict'] == 'necessary'
        # A tampered panel is a failure, not a missing-evidence warning.
        bad = os.path.join(td, 'bad.json')
        with open(bad, 'w', encoding='utf-8') as f:
            json.dump({'records': _records(weights=False)}, f)
        with contextlib.redirect_stdout(io.StringIO()):
            rep_bad = A.audit_lesion(bad)
        assert rep_bad['status'] == 'fail', rep_bad


def test_lesion_panel_flags_partially_attested_eval_fingerprints():
    # One record carries the rows fingerprint, the rest do not.  This is not a
    # conflicting panel, but "same rows" is unverifiable; the audit must not
    # treat it as a clean, fully attested pass.
    records = _records()
    for rec in records[1:]:
        rec.pop('eval_fp', None)
    rep = L.lesion_panel_audit(records)
    assert rep['status'] == 'warn', rep
    assert any('only 1 of 6 records carry eval_fp' in w
               for w in rep['warnings']), rep['warnings']
    assert rep['panels']['attention_sink/csa_dynamic']['verdict'] == 'necessary'


def test_lesion_design_reports_unhashable_source_without_crashing():
    # Hand-written declarations can carry a list/dict where the registry key
    # belongs; the gate must fail closed instead of raising TypeError from the
    # dictionary membership/lookup.
    for bad_source in ([], {'component': 'attention_sink'}):
        design = dict(L.LESION_DESIGN)
        design['attention_sink'] = dict(design['attention_sink'],
                                        source_component=bad_source)
        rep = L.lesion_design_audit(design)
        assert rep['status'] == 'fail', rep
        assert any('source_component' in v for v in rep['violations']), \
            rep['violations']
