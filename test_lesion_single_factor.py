"""Regression checks: a one-component lesion must be a single-factor design.

``lesion_design_audit`` checked that the declared ``mechanism_keys`` were a
subset of every key ever declared for the component's source contrast.  That
is a union test: a lesion that also moves a second real AttnCfg knob (or a
hand-written declaration that combines two declared families) could still be
reported as one component.  The gate now diffs the actual intact/lesioned cfg,
removes the sink switch that is inert for the layer kind, and applies the same
``cfg_keys_single_factor`` family test used by the between-arm gates.
"""
import contextlib
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def test_shipped_lesions_are_single_factor():
    rep = L.lesion_design_audit(build_models=False)
    assert rep['status'] == 'ok', rep['violations']
    assert set(rep['components']) == set(L.LESION_COMPONENTS)
    expected = {
        ('attention_sink', 'csa_fixed'): {'use_sink'},
        ('attention_sink', 'csa_dynamic'): {'use_sink'},
        ('attention_sink', 'full_sink'): {'full_sink'},
        ('content_pooling', 'csa_fixed'): {'content_mode'},
        ('content_pooling', 'csa_dynamic'): {'content_mode'},
        ('local_fusion', 'csa_dyn_fuse'): set(),
        ('local_fusion', 'hybrid_csa_dyn_fuse'): set(),
    }
    for component, rec in rep['components'].items():
        assert rec['status'] == 'ok', (component, rec)
        for arm, arec in rec['arms'].items():
            assert arec['single_factor_family'] is True, (component, arm, arec)
            assert set(arec['effective_cfg_keys']) == expected[component, arm], \
                (component, arm, arec)
            assert set(arec['effective_cfg_keys']) <= set(
                L.LESION_DESIGN[component]['mechanism_keys']), (component, arm)
            assert L.cfg_keys_single_factor(set(arec['effective_cfg_keys'])), \
                (component, arm)
    # `use_sink` is never consulted on a full layer, so the shared declaration
    # that turns both sink switches off is not a second factor there.  It is
    # recorded as inert instead of being silently dropped.
    dense = rep['components']['attention_sink']['arms']['full_sink']
    assert dense['inert_cfg_keys'] == ['use_sink'], dense
    assert dense['level'] == 'single_factor', dense


def test_lesion_with_a_second_real_knob_is_refused():
    design = {k: dict(v) for k, v in L.LESION_DESIGN.items()}
    design['content_pooling'] = dict(
        design['content_pooling'],
        cfg_overrides={'content_mode': 'zero', 'index_topk': 8})
    rep = L.lesion_design_audit(design, build_models=False)
    assert rep['status'] == 'fail', rep
    arec = rep['components']['content_pooling']['arms']['csa_fixed']
    assert arec['single_factor_family'] is False, arec
    assert 'index_topk' in arec['effective_cfg_keys'], arec
    assert any('index_topk' in v and 'undeclared' in v for v in rep['violations']), \
        rep['violations']


def test_lesion_spanning_two_declared_families_is_refused():
    design = {k: dict(v) for k, v in L.LESION_DESIGN.items()}
    design['attention_sink'] = dict(
        design['attention_sink'],
        cfg_overrides={'use_sink': False, 'full_sink': False,
                       'sink_mode': 'random'})
    rep = L.lesion_design_audit(design, build_models=False)
    assert rep['status'] == 'fail', rep
    for arm in ('csa_fixed', 'full_sink'):
        arec = rep['components']['attention_sink']['arms'][arm]
        assert arec['single_factor_family'] is False, (arm, arec)
    assert any('span more than one declared single-factor family' in v
               for v in rep['violations']), rep['violations']


def test_lesion_missing_the_active_dense_sink_switch_is_refused():
    design = {k: dict(v) for k, v in L.LESION_DESIGN.items()}
    design['attention_sink'] = dict(
        design['attention_sink'], cfg_overrides={'use_sink': False})
    rep = L.lesion_design_audit(design, build_models=False)
    assert rep['status'] == 'fail', rep
    dense = rep['components']['attention_sink']['arms']['full_sink']
    assert dense['effective_cfg_keys'] == [], dense
    assert any('full_sink' in v and 'do not change' in v
               for v in rep['violations']), rep['violations']


def test_aggregate_single_factor_gate_includes_lesion():
    import audit_gates as A
    with contextlib.redirect_stdout(io.StringIO()):
        rep = A.audit_single_factor()
    assert rep['status'] == 'ok', rep['status']
    assert rep['lesion']['status'] == 'ok', rep['lesion']
    for component, rec in rep['lesion']['components'].items():
        for arm, arec in rec['arms'].items():
            assert arec['single_factor_family'] is True, (component, arm, arec)


def test_zero_model_lesion_layer_never_builds_a_model(monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError('build_models=False must not construct a model')

    monkeypatch.setattr(L, 'SmallGPT', _boom)
    rep = L.lesion_design_audit(build_models=False)
    assert rep['status'] == 'ok', rep['violations']


def test_aggregate_single_factor_gate_reflects_a_confounded_lesion(monkeypatch):
    design = {k: dict(v) for k, v in L.LESION_DESIGN.items()}
    design['content_pooling'] = dict(
        design['content_pooling'],
        cfg_overrides={'content_mode': 'zero', 'index_topk': 8})
    monkeypatch.setattr(L, 'LESION_DESIGN', design)
    import audit_gates as A
    with contextlib.redirect_stdout(io.StringIO()):
        rep = A.audit_single_factor()
    assert rep['status'] == 'fail', rep['status']
    assert rep['lesion']['status'] == 'fail', rep['lesion']
    assert any('index_topk' in v and 'undeclared' in v
               for v in rep['lesion']['violations']), rep['lesion']['violations']
