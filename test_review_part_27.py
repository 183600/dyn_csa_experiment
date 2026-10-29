"""CPU checks for the crossed seed x unit variance decomposition (part 38).

The decomposition is a descriptive result-analysis readout: it says how much
of the per-cell variance is training-seed, evaluation-unit or residual noise,
and which extra cell would reduce the mean variance most.  These checks pin
the ANOVA identities, the non-negativity/truncation convention, the
scale-free readouts, the paired-NLL wrapper, the report renderer and the
artifact audit gate.
"""
import copy
import json
import math
import os
import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def _additive_panel():
    a = np.array([-2.0, -1.0, 0.0, 2.0])
    b = np.array([1.0, 0.0, -1.0])
    return a[:, None] + b[None, :], a, b


def test_exact_additive_panel_recovers_components():
    y, a, b = _additive_panel()
    vc = L.crossed_variance_components(y)
    assert vc['status'] == 'ok', vc
    assert vc['n_seeds'] == 4 and vc['n_units'] == 3 and vc['n_cells'] == 12
    # With an exactly additive panel the residual is zero and the two
    # components are the sample variances of the row/column effects.
    assert math.isclose(vc['components']['seed']['estimate'],
                        float(np.var(a, ddof=1)), rel_tol=1e-12)
    assert math.isclose(vc['components']['unit']['estimate'],
                        float(np.var(b, ddof=1)), rel_tol=1e-12)
    assert abs(vc['components']['residual']['estimate']) < 1e-12
    assert math.isclose(vc['total_variance'],
                        sum(vc['components'][k]['estimate']
                            for k in ('seed', 'unit', 'residual')),
                        rel_tol=1e-12)
    assert math.isclose(sum(vc['icc'].values()), 1.0, rel_tol=1e-12)
    assert math.isclose(vc['mean_variance']['total'],
                        vc['components']['seed']['estimate'] / 4.0
                        + vc['components']['unit']['estimate'] / 3.0,
                        rel_tol=1e-12)
    assert vc['allocation']['status'] == 'ok'
    assert vc['allocation']['next_cell'] in ('seed', 'unit')
    assert vc['effective_n']['equivalent'] is not None
    assert 0.0 < vc['effective_n']['equivalent'] <= 12.0
    ci = vc['mean_ci']
    assert ci['low'] <= vc['grand_mean'] <= ci['high']
    assert set(vc['prediction_intervals']) == {
        'new_cell', 'new_seed_same_units', 'new_unit_same_seeds'}


def test_allocation_ratio_and_gains_match_the_fixed_cell_objective():
    # For a fixed number of cells N=S*D, the variance-minimising ratio is
    # s/u, not sqrt(s/u) (the latter is the fixed S+D rule).  The next-cell
    # advice also includes the residual term e/(S*D).
    y, _a, _b = _additive_panel()
    vc = L.crossed_variance_components(y)
    s = vc['components']['seed']['estimate']
    u = vc['components']['unit']['estimate']
    e = vc['components']['residual']['estimate']
    S, D = vc['n_seeds'], vc['n_units']
    alloc = vc['allocation']
    assert math.isclose(alloc['recommended_seed_to_unit_ratio'], s / u,
                        rel_tol=1e-12)
    seed_gain = (s + e / D) / (S * (S + 1))
    unit_gain = (u + e / S) / (D * (D + 1))
    assert alloc['next_cell'] == ('seed' if seed_gain > unit_gain else 'unit')
    # The additive panel has zero residual, but the ratio check above already
    # distinguishes s/u from sqrt(s/u) for these components.
    assert not math.isclose(s / u, math.sqrt(s / u), rel_tol=1e-6)


def test_negative_moments_are_reported_raw_and_truncated():
    # A checkerboard has zero row/column means and all variance in the
    # residual, so both variance components have negative raw estimates.
    y = np.array([[(-1.0) ** (s + d) for d in range(4)] for s in range(4)])
    vc = L.crossed_variance_components(y)
    assert vc['status'] == 'ok', vc
    assert set(vc['negative_components']) == {'seed', 'unit'}, vc
    for key in ('seed', 'unit'):
        assert vc['components'][key]['raw'] < 0.0
        assert vc['components'][key]['estimate'] == 0.0
        assert vc['components'][key]['share'] == 0.0
    assert vc['components']['residual']['estimate'] > 0.0
    assert any('truncated' in w for w in vc['warnings'])


def test_constant_panel_is_not_estimable_not_a_zero_width_claim():
    vc = L.crossed_variance_components(np.ones((3, 4)))
    assert vc['status'] == 'not_estimable', vc
    assert vc['total_variance'] == 0.0
    assert vc['mean_ci'] is None and vc['mean_ci_conditional_on_units'] is None
    assert vc['prediction_intervals'] == {}
    assert any('constant' in w for w in vc['warnings'])


def test_scale_free_readouts_and_component_scaling():
    y, _a, _b = _additive_panel()
    base = L.crossed_variance_components(y)
    scaled = L.crossed_variance_components(y * 1e6)
    assert scaled['components']['seed']['share'] == base['components']['seed']['share']
    assert scaled['icc'] == base['icc']
    assert scaled['effective_n']['equivalent'] == base['effective_n']['equivalent']
    assert scaled['effective_n']['design_effect'] == base['effective_n']['design_effect']
    for key in ('seed', 'unit', 'residual'):
        assert math.isclose(scaled['components'][key]['estimate'],
                            base['components'][key]['estimate'] * 1e12,
                            rel_tol=1e-12)
    assert scaled['mean_ci']['halfwidth'] == base['mean_ci']['halfwidth'] * 1e6


def test_invalid_inputs_are_refused():
    import pytest
    for bad in ([1.0, 2.0, 3.0],
                [[1.0, 2.0]],
                [[1.0], [2.0]],
                [[1.0, float('nan')], [2.0, 3.0]],
                [[1.0, float('inf')], [2.0, 3.0]],
                [[1.0, 2.0], [3.0]],
                [[True, 2.0], [3.0, 4.0]],
                'not a matrix'):
        with pytest.raises(ValueError):
            L.crossed_variance_components(bad)
    y, _a, _b = _additive_panel()
    for conf in (0.0, 1.0, float('nan'), 'x'):
        with pytest.raises(ValueError):
            L.crossed_variance_components(y, conf=conf)


def test_paired_nll_wrapper_uses_per_token_cells_and_reports_both_means():
    A = np.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]])
    B = np.array([[12.0, 22.0, 33.0], [13.0, 23.0, 34.0]])
    tok = np.array([10.0, 20.0, 30.0])
    delta = (A - B) / tok[None, :]
    vc = L.paired_delta_variance_components(A, B, tok)
    ref = L.crossed_variance_components(delta)
    assert vc['kind'] == 'paired_delta_variance_components'
    assert math.isclose(vc['grand_mean'], ref['grand_mean'], rel_tol=1e-12)
    assert math.isclose(vc['components']['seed']['estimate'],
                        ref['components']['seed']['estimate'], rel_tol=1e-12)
    expected_pooled = float(np.mean(np.sum(A - B, axis=1) / np.sum(tok)))
    assert math.isclose(vc['token_pooled_delta_nll_per_token'],
                        expected_pooled, rel_tol=1e-12)
    # Unequal unit lengths make the equal-cell and token-pooled means differ.
    assert not math.isclose(vc['mean_delta_nll_per_token_equal_units'],
                            vc['token_pooled_delta_nll_per_token'],
                            rel_tol=1e-12)
    import pytest
    with pytest.raises(ValueError):
        L.paired_delta_variance_components(A, B[:1], tok)
    with pytest.raises(ValueError):
        L.paired_delta_variance_components(A, B, tok[:2])
    with pytest.raises(ValueError):
        L.paired_delta_variance_components(A, B, [0.0, 1.0, 1.0])


def test_markdown_renderer_shows_shares_and_flags_missing_blocks():
    y, _a, _b = _additive_panel()
    vc = L.crossed_variance_components(y)
    ft = {'comparisons': {
        'main': {'doc_sampling': {'unit': 'article', 'n_seeds': 4,
                                  'n_units': 3, 'variance_components': vc}},
        'legacy': {'doc_sampling': {'unit': 'article', 'n_seeds': 4,
                                    'n_units': 3,
                                    'variance_components_unavailable_reason':
                                        'NLL vectors missing'}},
    }}
    lines = V12._variance_components_markdown(ft)
    text = '\n'.join(lines)
    assert '方差分解' in text
    assert '74%' in text, text
    assert '加种子' in text or '加评测单元' in text
    assert '未验证' in text and 'legacy' in text


def _valid_artifact():
    y, _a, _b = _additive_panel()
    vc = L.crossed_variance_components(y)
    return {'finaltest': {'comparisons': {'main': {
        'doc_sampling': {'unit': 'article', 'n_seeds': 4, 'n_units': 3,
                         'variance_components': vc}}}}}


def test_audit_replays_identities_and_fails_tampering():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'test_stats.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(_valid_artifact(), f)
        got = A.audit_variance_components(p)
        assert got['status'] == 'ok', got
        assert got['n_checked'] == 1 and got['n_ok'] == 1, got

        bad = _valid_artifact()
        vc = bad['finaltest']['comparisons']['main']['doc_sampling']['variance_components']
        vc['total_variance'] = vc['total_variance'] * 2.0
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(bad, f)
        got = A.audit_variance_components(p)
        assert got['status'] == 'fail', got
        assert any('total_variance' in x for x in got['problems']), got

        bad = _valid_artifact()
        vc = bad['finaltest']['comparisons']['main']['doc_sampling']['variance_components']
        vc['mean_variance']['seed'] = 0.0
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(bad, f)
        got = A.audit_variance_components(p)
        assert got['status'] == 'fail', got
        assert any('mean_variance' in x for x in got['problems']), got


def test_audit_replays_the_allocation_advice():
    """The next-cell recommendation is computed from the components.

    A descriptive table is still quotable; an edited ``next_cell`` (or a
    deleted allocation block) must not pass as a replayed decomposition.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'test_stats.json')

        def run(vc):
            art = _valid_artifact()
            art['finaltest']['comparisons']['main']['doc_sampling'][
                'variance_components'] = vc
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(art, f)
            return A.audit_variance_components(p)

        good = _valid_artifact()['finaltest']['comparisons']['main'][
            'doc_sampling']['variance_components']
        assert run(good)['status'] == 'ok'

        bad = copy.deepcopy(good)
        cur = bad['allocation']['next_cell']
        bad['allocation']['next_cell'] = 'unit' if cur == 'seed' else 'seed'
        got = run(bad)
        assert got['status'] == 'fail', got
        assert any('next_cell' in x for x in got['problems']), got

        bad = copy.deepcopy(good)
        bad['allocation']['recommended_seed_to_unit_ratio'] = (
            float(bad['allocation']['recommended_seed_to_unit_ratio']) * 2.0)
        got = run(bad)
        assert got['status'] == 'fail', got
        assert any('recommended_seed_to_unit_ratio' in x
                   for x in got['problems']), got

        bad = copy.deepcopy(good)
        bad['allocation']['current']['n_seeds'] += 1
        got = run(bad)
        assert got['status'] == 'fail', got
        assert any('allocation.current.n_seeds' in x
                   for x in got['problems']), got

        bad = copy.deepcopy(good)
        bad.pop('allocation')
        got = run(bad)
        assert got['status'] == 'fail', got
        assert any('allocation block missing' in x
                   for x in got['problems']), got


def test_audit_is_not_verifiable_without_a_decomposition_block():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'test_stats.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'finaltest': {'comparisons': {
                'main': {'doc_sampling': {'n_seeds': 4, 'n_units': 3}}}}}, f)
        got = A.audit_variance_components(p)
        assert got['status'] == 'not_verifiable_locally', got
        assert got['n_legacy'] == 1
