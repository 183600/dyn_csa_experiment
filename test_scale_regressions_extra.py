"""Regression checks for extreme-scale guards and the fusion control ladder.

These are zero-GPU checks.  They cover paths where a finite input can produce a
non-finite report value, a design gate that must propagate an upstream
rejection, or a deterministic placebo that must not depend on uninitialised
memory or on the expected norm of a Gaussian vector.
"""
import json
import math
import os
import sys as _sys

import numpy as np
import torch

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403


def test_initialization_audit_propagates_definition_site_failure():
    # No declared intervention keys: the definition-site audit must refuse this
    # contrast, and the initialization report must not call it "ok" merely
    # because no pair reached the parameter comparison loop.
    design = {'content_alignment': [('csa_fixed', 'csa_dynamic')]}
    rep = L.single_factor_initialization_audit(design, verbose=False)
    assert rep['design_status'] == 'fail', rep
    assert rep['status'] == 'fail', rep
    assert any('definition-site' in v for v in rep['violations']), rep


def test_random_fuse_kernel_is_deterministic_and_magnitude_matched():
    class _BadWeight:
        def detach(self):
            raise RuntimeError('force the deterministic fallback')

    w1 = L._frozen_random_fuse_weight(_BadWeight(), 8, 3, torch.float32)
    w2 = L._frozen_random_fuse_weight(_BadWeight(), 8, 3, torch.float32)
    assert torch.equal(w1, w2), 'the fallback placebo must not read uninitialised memory'
    norms = w1.reshape(8, -1).norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), norms

    # A hashed kernel has the same per-channel radius as the identity kernel
    # used by the learned treatment, for every kernel width (including 1).
    for kernel in (1, 2, 5):
        w = L._frozen_random_fuse_weight(torch.randn(16, 16), 8, kernel,
                                         torch.float32)
        norms = w.reshape(8, -1).norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), (
            kernel, norms)


def test_paired_effect_stats_withholds_unrepresentable_sample_sd():
    # The true sample SD is sqrt(2)*1.7e308, beyond float64 even though every
    # delta is finite.  The mean and the exact test survive; the SD-derived
    # fields become unavailable instead of infinite.
    rep = L.paired_effect_stats([1.7e308, -1.7e308])
    assert rep['std'] is None and rep['std_available'] is False, rep
    assert rep['sem'] is None and rep['cohens_dz'] is None, rep
    assert math.isfinite(rep['mean']) and math.isfinite(rep['p_exact_signflip']), rep
    for key in ('std', 'sem', 'cohens_dz'):
        assert rep[key] is None or math.isfinite(rep[key]), rep


def test_paired_delta_token_pooled_mean_is_finite_for_extreme_nll():
    # A-B itself overflows the row sum, but the token-pooled per-token mean is
    # representable.  The wider path must recover it; ordinary input stays on
    # the historical NumPy path.
    A = np.array([[1e308, 1.5e308], [1.2e308, 1.7e308]])
    B = np.array([[1e307, 1.5e307], [1.2e307, 1.7e307]])
    rep = L.paired_delta_variance_components(A, B, np.array([10.0, 20.0]))
    assert math.isfinite(rep['token_pooled_delta_nll_per_token']), rep
    json.dumps(rep, allow_nan=False)

    ordinary_a = np.array([[10.0, 20.0], [11.0, 21.0]])
    ordinary_b = np.array([[12.0, 22.0], [13.0, 23.0]])
    tok = np.array([10.0, 20.0])
    got = L.paired_delta_variance_components(ordinary_a, ordinary_b, tok)
    expected = float(np.mean(np.sum(ordinary_a - ordinary_b, axis=1)
                             / np.sum(tok)))
    assert math.isclose(got['token_pooled_delta_nll_per_token'], expected,
                        rel_tol=1e-12)


def test_scale_ladder_unrepresentable_budget_ratio_is_not_infinite():
    def tier(name, steps, params):
        return {'name': name,
                'cfg': {'steps': steps, 'batch_size': 1, 'seq_len': 1,
                        'n_train_tokens': steps, 'lr': 3e-4, 'warmup': 0,
                        'seeds': [0, 1]},
                'params': params}

    out = L.scale_ladder_audit([tier('S1', 1, 10 ** 300),
                                tier('S2', 10 ** 300, 1)])
    assert out['tokens_per_param_ratio'] is None, out
    assert out['budget_matched'] is False, out
    assert any('representable float range' in w for w in out['warnings']), out
    json.dumps(out, allow_nan=False)


def test_meta_analysis_opposite_extreme_effects_is_json_safe():
    points = [
        {'label': 'a', 'scale': 1.0, 'seeds': [0, 1],
         'effects': [1.7e308, -1.7e308]},
        {'label': 'b', 'scale': 2.0, 'seeds': [0, 1],
         'effects': [1.6e308, -1.6e308]},
    ]
    rep = L.meta_analyze_effects(points, min_seeds=2)
    for row in rep['per_point']:
        for key in ('std', 'se'):
            assert row[key] is None or math.isfinite(row[key]), (row, key)
        assert row['ci'] is None or all(math.isfinite(v) for v in row['ci']), row
    json.dumps(rep, allow_nan=False)


def test_unit_precision_design_reorders_an_overflowing_ci_product():
    base = [-1.0, -0.8, -1.2, -0.9, -1.1, -0.95]
    rep = L.unit_precision_design(
        10000, pilot_unit_deltas=[x * 1e307 for x in base])
    assert rep['status'] == 'ok', rep
    for key in ('ci_halfwidth', 'ci95_halfwidth'):
        assert math.isfinite(rep['precision'][key]), (key, rep)
    assert math.isfinite(rep['mde']['absolute']), rep


def test_scale_interaction_withholds_nonrepresentable_and_undefined_fields():
    # An all-zero interaction has no non-zero rank for the Wilcoxon
    # sensitivity test, and a constant non-zero interaction has an unbounded
    # Cohen's d_z.  Both must be unavailable (None), not NaN/Infinity, because
    # the report is serialised with allow_nan=False.
    zero = L.scale_interaction([0.0] * 6, [0.0] * 6)
    assert zero['p_wilcoxon'] is None, zero
    assert zero['p_exact_signflip'] == 1.0, zero
    json.dumps(zero, allow_nan=False)
    constant = L.scale_interaction([0.0] * 6, [1.0] * 6)
    assert constant['cohens_dz'] is None, constant
    assert math.isfinite(constant['mean']), constant
    json.dumps(constant, allow_nan=False)
    # An interval endpoint beyond float64 is likewise withheld.
    extreme = L.scale_interaction([1e308, -1e308, 5e307],
                                  [1.7e308, -1.7e308, 1.7e308])
    for key in ('ci95_low', 'ci95_high'):
        assert extreme[key] is None or math.isfinite(extreme[key]), (key, extreme)
    json.dumps(extreme, allow_nan=False)
    # Ordinary panels keep the historical finite values.
    ordinary = L.scale_interaction([1.0] * 6, [1.5] * 6)
    assert ordinary['mean'] == 0.5 and ordinary['cohens_dz'] is None
    # A numeric string is not a measurement.  The old float() path accepted
    # '1.0' as if it were a recorded effect.
    strings = L.scale_interaction(['1.0'] * 6, ['2.0'] * 6)
    assert strings['n'] == 0 and strings['n_dropped'] == 6, strings
    json.dumps(strings, allow_nan=False)
    for bad in (('1.0', '2.0'), (None, [1.0])):
        try:
            L.scale_interaction(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'non-sequence effects accepted: {bad!r}')


def test_seed_and_measurement_iterables_are_not_strings_or_mappings():
    # `list("012")` is ['0','1','2'] and the legacy integer reader accepts
    # those characters as seeds 0,1,2.  A string is one malformed descriptor,
    # never a seed panel (or a per-seed effect vector).
    bad_calls = (
        lambda: L._scale_points([{'label': 'a', 'scale': 1,
                                  'effects': [1.0, 2.0, 3.0],
                                  'seeds': '012'}]),
        lambda: L._scale_points([{'label': 'a', 'scale': 1,
                                  'effects': [1.0, 2.0, 3.0],
                                  'planned_seeds': '012'}]),
        lambda: L.scale_interaction([1, 2, 3], [1, 2, 3], seeds='012'),
        lambda: L.paired_effect_stats([1, 2, 3], seeds='012'),
        lambda: L.paired_effect_stats('abc'),
        lambda: L.paired_log_ratio([1, 2], [2, 1], seeds='01'),
        lambda: L.paired_log_ratio('12', '12'),
        lambda: L.loglog_regression('abc'),
        lambda: L.crossover_scaling_fit('abc'),
        lambda: L.missingness_sensitivity([1, 2], seeds='01',
                                          planned_seeds=[0, 1]),
        lambda: L.missingness_sensitivity([1, 2], seeds=[0, 1],
                                          planned_seeds=[0, 1],
                                          delta_range='12'),
        lambda: L.lesion_effect([1, 1], [1, 1], seeds='01'),
        lambda: L.tost_paired('abc', 0.1),
    )
    for call in bad_calls:
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError(f'malformed iterable accepted: {call}')

    # Numeric strings are serialisation artefacts, not measurements.  A
    # per-seed effect pair is dropped (and counted), never float()-converted;
    # axis points are skipped for the same reason.
    ratio = L.paired_log_ratio(['1.0', '2.0'], ['2.0', '1.0'])
    assert ratio['n'] == 0 and ratio['n_dropped'] == 2, ratio
    assert L.loglog_regression([('1', '2'), ('2', '4')])['n_points'] == 0
    assert L.crossover_scaling_fit([
        {'d': '1', 'crossover_tokens': '100', 'train_tokens': '1000'}
        for _ in range(4)])['n_input'] == 0
    for bad in ([{'label': 'a', 'scale': 1,
                  'effects': {'s0': 1.0}, 'seeds': [0]}],):
        try:
            L._scale_points(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'mapping effects accepted: {bad!r}')
    point = L._scale_points([{'label': 'a', 'scale': '1',
                              'effects': [1.0], 'seeds': [0]}])[0]
    assert math.isnan(point['scale']), point


def test_crossover_volume_ratio_is_not_infinite():
    # Finite token budgets can still have an unrepresentable ratio.  The
    # confound is real, but the report must not carry an `inf` ratio or a
    # "x inf" reason string, and it must not let `tmin*(1+tol)` overflow and
    # hide a genuine mismatch.
    pts = [{'label': str(i), 'd': 2 ** i, 'crossover_tokens': 100,
            'train_tokens': tokens}
           for i, tokens in enumerate([1e-300, 1.0, 1e308, 1.7e308])]
    rep = L.crossover_scaling_fit(pts)
    assert rep['status'] == 'training_volume_confounded', rep
    assert rep['volume_ratio'] is None, rep
    assert 'inf' not in ' '.join(rep['reasons']), rep['reasons']
    json.dumps(rep, allow_nan=False)


def test_scaling_trend_withholds_unrepresentable_slope_but_keeps_direction():
    # Every effect is finite, but mapping a steep unit-scaled slope back to
    # those units exceeds float64.  The slope fields are unavailable; the
    # direction and the lack-of-fit p remain finite and the report is strict-
    # JSON safe (previously `direction_slope=inf`, `heterogeneity_paired.p=nan`).
    pts = []
    for i, scale in enumerate((1.0, 1.000001, 1.000002)):
        effect = 1.4e308 * (0.7 + 0.15 * i)
        pts.append({'label': str(i), 'scale': scale, 'seeds': [0, 1, 2],
                    'effects': [effect, effect * 0.99, effect * 1.01]})
    rep = L.scaling_trend(pts, min_seeds=2, min_points=2)
    assert rep['direction'] == 'increases', rep
    assert rep['direction_slope'] is None, rep
    assert rep['direction_slope_available'] is False, rep
    assert rep['wls']['slope'] is None and rep['paired']['slope_mean'] is None, rep
    assert rep['heterogeneity_paired']['p'] is not None
    assert math.isfinite(rep['heterogeneity_paired']['p'])
    json.dumps(rep, allow_nan=False)


def test_scale_interaction_overflowing_pair_does_not_report_a_kept_seed():
    # Both endpoints are finite, but their difference is beyond float64.  The
    # pair is unavailable; `n` and the reported `seeds` must agree (the old
    # bookkeeping kept the seed while paired_effect_stats dropped its value).
    rep = L.scale_interaction([-1.7e308, 0.0, 1.0], [1.7e308, 1.0, 2.0])
    assert rep['n'] == len(rep['seeds']) == 2, rep
    assert rep['n_dropped'] == 1 and rep['complete_pairs'] is False, rep
    json.dumps(rep, allow_nan=False)


def test_zero_paired_panel_has_a_json_safe_wilcoxon_field():
    # A signed-rank test has no rank distribution when every delta is zero.
    # The old NaN sentinel made a perfectly valid all-zero panel impossible to
    # serialise with allow_nan=False.
    rep = L.paired_effect_stats([0.0] * 3)
    assert rep['p_exact_signflip'] == 1.0, rep
    assert rep['p_wilcoxon'] is None, rep
    json.dumps(rep, allow_nan=False)


def test_trajectory_step_span_beyond_float64_withholds_slope_only():
    # Step labels are exact integers and may exceed float64.  A synthetic
    # look at 10**400 must not raise OverflowError while converting the OLS
    # x-axis; the slope becomes unavailable and the rest of the block survives.
    huge = 10 ** 400
    with_hist = {0: [(0, 1.0), (huge, 2.0), (huge + 1, 3.0)]}
    without_hist = {0: [(0, 1.0), (huge, 1.5), (huge + 1, 2.0)]}
    rep = L.paired_trajectory_stability(with_hist, without_hist,
                                        primary_delta=0.1)
    assert rep['status'] == 'stable', rep
    for key in ('tail_slope_log_per_1k_with',
                'tail_slope_log_per_1k_without',
                'tail_gap_slope_log_per_1k'):
        assert rep[key] is None, (key, rep)
    json.dumps(rep, allow_nan=False)


def test_trajectory_y_beyond_float64_withholds_slope_only():
    # The x-axis conversion was hardened first; a y measurement that is an
    # exact integer beyond float64 must likewise make the slope unavailable
    # instead of escaping from the float conversion.
    huge = 10 ** 400
    assert L._traj_slope([(0, 1.0), (1, huge)]) is None
    # Ordinary panels keep the slope.
    assert math.isclose(L._traj_slope([(0, 1.0), (1000, 3.0)]), 2.0)


def test_dose_response_extreme_adjacent_steps_are_strict_json():
    # Adjacent differences can overflow even though each level mean is finite.
    # Report the unavailable step as None (not inf/NaN) and keep the endpoint
    # difference, which is representable because the two level means coincide.
    levels = [
        {'label': 'l1', 'dose': 1,
         'values': {0: 1.7e308, 1: 1.5e308, 2: 1.6e308}},
        {'label': 'l2', 'dose': 2,
         'values': {0: -1.7e308, 1: -1.5e308, 2: -1.6e308}},
        {'label': 'l3', 'dose': 3,
         'values': {0: 1.7e308, 1: 1.5e308, 2: 1.6e308}},
    ]
    rep = L.dose_response_trend(levels, direction='increasing', min_n=2)
    for step in rep['monotonicity']['adjacent_steps']:
        assert step['mean_delta'] is None, step
        assert step['direction_match'] is None, step
        assert step['zero'] is None, step
    assert rep['endpoint_delta']['first_to_last_mean'] == 0.0, rep
    assert all(v == 0.0 for v in
               rep['endpoint_delta']['per_seed'].values()), rep
    for value in rep['per_seed_slopes'].values():
        assert math.isfinite(float(value)), rep
    json.dumps(rep, allow_nan=False)


def test_pooled_ppl_survives_totals_beyond_float64():
    # Every record is finite, but the NLL total overflows and the exact token
    # total is beyond float64.  The pooled per-token ratio (1.0) is still
    # representable: the old path raised OverflowError converting the token
    # total, and the NLL total was inf.
    huge = 10 ** 308
    recs = [{'nll_sum': 1e308, 'n_tokens': huge} for _ in range(3)]
    pooled, nll_total, tok_total = L._pooled_ppl_from_records(recs)
    assert math.isclose(pooled, math.e, rel_tol=1e-15), pooled
    assert nll_total is None, nll_total
    assert tok_total == 3 * huge and isinstance(tok_total, int), tok_total
    json.dumps([pooled, nll_total])

    # The same holds when the record carries only per-row arrays: their sums
    # overflow, but the row-weighted ratio is representable.
    nested = {'ppl_estimation_rows': 'nested_validation_holdout',
              'nll_per_row_holdout': [1e308, 1e308],
              'n_tokens_per_row_holdout': [huge, huge]}
    pooled, nll_total, tok_total = L._pooled_ppl_from_records([nested])
    assert math.isclose(pooled, math.e, rel_tol=1e-15), pooled
    assert nll_total is None and tok_total == 2 * huge, (nll_total, tok_total)

    # A finite token total above 2**53 must stay exact instead of being
    # rounded through float64.
    n_big = 2 ** 53 + 1
    pooled, _nll, tok_total = L._pooled_ppl_from_records(
        [{'nll_sum': 1e-300, 'n_tokens': n_big} for _ in range(2)])
    assert tok_total == 2 * n_big, tok_total
    assert pooled == 1.0, pooled





def test_cross_panel_leave_one_out_rescales_each_subset():
    # The LOO row cache is only valid while every subset stays on the unscaled
    # meta-analysis branch.  Dropping the only ordinary-magnitude panel can
    # leave only sub-floor effects, and dropping the only tiny panel can leave
    # only >1e100 effects; either flip changes whether mean/SE are raw or
    # divided by the family scale.  Reusing the old rows then double-scales the
    # pooled estimate (the first case below was 3.255e202 instead of 1.55e101).
    cases = [
        [
            {'label': 'small', 'scale': 1.0, 'seeds': [0, 1],
             'effects': [1.0, 1.0]},
            {'label': 'big1', 'scale': 2.0, 'seeds': [0, 1],
             'effects': [1e101, 1.1e101]},
            {'label': 'big2', 'scale': 3.0, 'seeds': [0, 1],
             'effects': [2e101, 2.1e101]},
        ],
        [
            {'label': 'tiny1', 'scale': 1.0, 'seeds': [0, 1],
             'effects': [1e-90, 1.1e-90]},
            {'label': 'tiny2', 'scale': 2.0, 'seeds': [0, 1],
             'effects': [1.2e-90, 1.3e-90]},
            {'label': 'ordinary', 'scale': 3.0, 'seeds': [0, 1],
             'effects': [1.0, 1.1]},
        ],
    ]
    for points in cases:
        loo = L._cps_leave_one_out(points, alpha=0.05, min_n=2,
                                   full_mean=0.0)
        for i, point in enumerate(points):
            rest = points[:i] + points[i + 1:]
            meta = L._cps_pool(rest, alpha=0.05, min_n=2,
                               pair_detail_limit=0)
            re_block = meta.get('random_effects') or {}
            fe_block = meta.get('fixed_effect') or {}
            expected = re_block.get('mean')
            if expected is None:
                expected = fe_block.get('mean')
            got = loo[point['label']]['pooled_mean']
            assert got is not None and expected is not None, (point, meta)
            assert math.isclose(got, expected, rel_tol=1e-12), (
                point['label'], got, expected)


def test_exact_json_key_integer_does_not_round_through_float64():
    # JSON object keys are strings.  A large exact key must remain the same
    # integer identity, not the nearest float64 (2**53 + 1 used to become
    # 2**53), and a key beyond the float exponent range must still parse.
    assert A._exact_int(str(2 ** 53 + 1)) == 2 ** 53 + 1
    assert A._exact_int(str(10 ** 400)) == 10 ** 400
    # Integer-valued decimal spellings keep the historical acceptance;
    # fractional / non-finite / boolean values are still not counts.
    assert A._exact_int('1.0') == 1
    assert A._exact_int('1e3') == 1000
    for bad in ('1.5', 'nan', 'inf', None, True):
        assert A._exact_int(bad) is None, bad


def test_meta_analysis_mixed_subfloor_and_ordinary_effects_share_a_scale():
    # A legitimate effect at 1e-200 must not be excluded merely because a
    # sibling scale point has an ordinary magnitude.  Max-only normalisation
    # left 1/se**2 over float64, so the tiny point became "non-estimable" and
    # the pooled mean silently became the ordinary point.  Both points now
    # enter the inverse-variance pool on one common work scale.
    points = [
        {'label': 'tiny', 'scale': 1.0, 'seeds': [0, 1],
         'effects': [1e-200, 1.1e-200]},
        {'label': 'ordinary', 'scale': 2.0, 'seeds': [0, 1],
         'effects': [1.0, 1.1]},
    ]
    meta = L.meta_analyze_effects(points, min_seeds=2)
    assert meta['n_usable'] == 2, meta
    assert meta['non_estimable_points'] == [], meta
    assert meta['zero_variance_points'] == [], meta
    assert math.isclose(meta['fixed_effect']['mean'], 1.05e-200,
                        rel_tol=1e-12), meta['fixed_effect']
    assert math.isfinite(meta['fixed_effect']['se']), meta['fixed_effect']
    json.dumps(meta, allow_nan=False)


def test_paired_nll_recovers_a_finite_ratio_when_the_row_sum_overflows():
    # Each per-document NLL is finite, but the three-document row sum is
    # beyond float64.  The token-pooled CE (and therefore the PPL) is still
    # representable: 1e308 / 1e306 ~ 275.  The cluster interval and the
    # delete-one-unit influence must not collapse to "unavailable" merely
    # because NumPy's accumulator overflowed.
    from fractions import Fraction

    A = np.array([[1e308, 1.1e308, 1.2e308],
                  [1.05e308, 1.15e308, 1.25e308]])
    B = np.array([[0.9e308, 1.0e308, 1.1e308],
                  [0.95e308, 1.05e308, 1.15e308]])
    tokens = [3 * 10 ** 305, 4 * 10 ** 305, 5 * 10 ** 305]
    with np.errstate(over='ignore', invalid='ignore'):
        assert not np.isfinite(A.sum(axis=1)).all()

    def _ce_rows(matrix):
        total = sum(Fraction(int(t)) for t in tokens)
        return [sum((Fraction(float(v)) for v in row), Fraction(0, 1)) / total
                for row in matrix]

    ce_a = _ce_rows(A)
    ce_b = _ce_rows(B)
    expected_lr = float(sum((a - b for a, b in zip(ce_a, ce_b)),
                            Fraction(0, 1)) / len(ce_a))
    ci = L._paired_nll_doc_ci(A, B, tokens)
    assert ci is not None, 'a representable pooled ratio was discarded'
    assert math.isclose(ci['delta_log_ratio'], expected_lr,
                        rel_tol=1e-12, abs_tol=1e-12), ci
    assert math.isfinite(ci['delta_ppl']), ci
    assert 'row NLL sum' in ci['method'], ci
    json.dumps(ci, allow_nan=False)

    infl = L.paired_nll_unit_influence(A, B, tokens)
    assert infl['available'] is True, infl
    assert math.isclose(infl['full_delta_log_ratio'], expected_lr,
                        rel_tol=1e-12, abs_tol=1e-12), infl
    assert infl['n_units_loo_available'] == len(tokens), infl
    json.dumps(infl, allow_nan=False)

    two = L.paired_nll_two_way_bootstrap(
        A, B, tokens, n_boot=40, seed=7)
    assert two['status'] == 'ok', two
    assert math.isclose(two['delta_log_ratio'], expected_lr,
                        rel_tol=1e-12, abs_tol=1e-12), two
    assert two['n_boot_valid'] >= 20, two
    json.dumps(two, allow_nan=False)

    # The order-dependence readout shares the same point estimator and must
    # use the exact-rational influence path instead of withholding the point.
    D = 8
    A8 = np.array([[1e308 * (1.0 + 0.01 * i) for i in range(D)],
                   [1.05e308 * (1.0 + 0.01 * i) for i in range(D)]])
    B8 = np.array([[0.9e308 * (1.0 + 0.01 * i) for i in range(D)],
                   [0.95e308 * (1.0 + 0.01 * i) for i in range(D)]])
    tokens8 = [10 ** 306 * (i + 1) for i in range(D)]
    order = L.pooled_order_inference(
        A8, B8, tokens8, n_boot=40, n_perm=0)
    assert order['status'] == 'ok', order
    assert math.isfinite(order['point_ppl']), order
    assert order['point_log_ratio'] is not None, order
    assert 'row NLL sum' in order['upstream'], order
    json.dumps(order, allow_nan=False)


def test_paired_readouts_accept_an_exact_token_total_beyond_float64():
    # A single unit can hold more tokens than float64 can represent, and the
    # panel total can overflow even when each unit's count is representable
    # (3 * 10**308).  The token-pooled CE is still a representable ratio, so
    # every paired readout must take its exact-ratio path instead of raising
    # from the token conversion or calling a representable point "invalid".
    from fractions import Fraction

    A = np.array([[1e308, 1.1e308, 1.2e308],
                  [1.05e308, 1.15e308, 1.25e308]])
    B = np.array([[0.9e308, 1.0e308, 1.1e308],
                  [0.95e308, 1.05e308, 1.15e308]])
    tokens = [10 ** 308, 2 * 10 ** 308, 3 * 10 ** 308]
    total = sum(tokens)

    def _ce(row):
        return sum((Fraction(float(v)) for v in row), Fraction(0, 1)) / total

    expected_lr = float(sum((_ce(A[s]) - _ce(B[s]) for s in range(A.shape[0])),
                            Fraction(0, 1)) / A.shape[0])

    ci = L._paired_nll_doc_ci(A, B, tokens)
    assert ci is not None and math.isfinite(ci['delta_log_ratio']), ci
    assert math.isclose(ci['delta_log_ratio'], expected_lr,
                        rel_tol=1e-12, abs_tol=1e-12), ci
    json.dumps(ci, allow_nan=False)

    infl = L.paired_nll_unit_influence(A, B, tokens)
    assert infl['available'] is True, infl
    assert math.isclose(infl['full_delta_log_ratio'], expected_lr,
                        rel_tol=1e-12, abs_tol=1e-12), infl
    assert infl['n_units_loo_available'] == len(tokens), infl
    json.dumps(infl, allow_nan=False)

    two = L.paired_nll_two_way_bootstrap(A, B, tokens, n_boot=40, seed=11)
    assert two['status'] == 'ok', two
    assert math.isclose(two['delta_log_ratio'], expected_lr,
                        rel_tol=1e-12, abs_tol=1e-12), two
    assert math.isclose(two['delta_ppl'], ci['delta_ppl'],
                        rel_tol=1e-12, abs_tol=1e-300), two
    assert two['n_boot_valid'] >= 20, two
    json.dumps(two, allow_nan=False)

    # A tiny but nonzero log difference must not become a fake zero PPL
    # difference (exp(a)-exp(b) rounds both sides to the same float).  The
    # two-level point and its percentile interval must agree with the document
    # interval's expm1 path.
    tiny_a = np.full((2, 3), 100.0)
    tiny_b = np.full((2, 3), 100.0 + 1e-10)
    tiny_tokens = [10 ** 6] * 3
    tiny_doc = L._paired_nll_doc_ci(tiny_a, tiny_b, tiny_tokens)
    tiny = L.paired_nll_two_way_bootstrap(
        tiny_a, tiny_b, tiny_tokens, n_boot=40, seed=13)
    assert tiny['status'] == 'ok' and tiny['delta_ppl'] != 0.0, tiny
    assert math.isclose(tiny['delta_ppl'], tiny_doc['delta_ppl'],
                        rel_tol=1e-12, abs_tol=0.0), (tiny, tiny_doc)
    assert (tiny['delta_ppl_ci95'][0] <= tiny['delta_ppl']
            <= tiny['delta_ppl_ci95'][1]), tiny
    json.dumps(tiny, allow_nan=False)

    # The variance decomposition uses the wider machine float for the
    # per-token cells; on a platform without a wider type this assertion is
    # intentionally skipped rather than pretending the ratio is unmeasurable.
    if np.finfo(np.longdouble).max > np.finfo(float).max:
        vc = L.paired_delta_variance_components(A, B, tokens)
        assert vc['status'] == 'ok', vc
        assert math.isfinite(vc['token_pooled_delta_nll_per_token']), vc
        json.dumps(vc, allow_nan=False)


def test_row_bootstrap_pools_token_columns_beyond_int64_and_nll_float64():
    # Four records of three 3e18-token rows: each record's own total is below
    # int64, but the pooled column total (1.2e19) wraps negative in an int64
    # reduction.  The old code then treated every document as zero-token and
    # withheld both intervals.  Every row has the same CE (1/3), so the
    # resampling distribution is a point mass at exp(1/3).
    expected = math.exp(1.0 / 3.0)
    recs = [{'variant': 'a', 'seed': s, 'ppl': expected,
             'nll_per_row': [1e18, 1e18, 1e18],
             'n_tokens_per_row': [3 * 10 ** 18] * 3} for s in range(4)]
    doc = L._doc_bootstrap_ppl_ci(recs, n_boot=60, seed=5)
    assert doc is not None, 'a representable pooled CE was discarded'
    assert math.isclose(doc[0], expected, rel_tol=1e-12), doc
    assert math.isclose(doc[1], expected, rel_tol=1e-12), doc
    combined = L._combined_seed_doc_ci(recs, n_boot=60, seed=5)
    assert combined['status'] == 'ok', combined
    assert combined['n_docs'] == 3 and combined['n_docs_dropped_zero_token'] == 0, combined
    assert math.isclose(combined['ci95_ppl'][0], expected, rel_tol=1e-12), combined
    json.dumps(combined, allow_nan=False)

    # The same records can also have a finite row NLL whose float64 column
    # sum overflows (3 * 1e308).  The CE is still 100, so the interval must be
    # reported instead of collapsing to "unrepresentable".
    big = [{'variant': 'a', 'seed': s, 'ppl': math.exp(100.0),
            'nll_per_row': [1e308] * 3,
            'n_tokens_per_row': [10 ** 306] * 3} for s in range(3)]
    doc_big = L._doc_bootstrap_ppl_ci(big, n_boot=60, seed=5)
    assert doc_big is not None, 'a finite CE with an overflowing NLL sum was lost'
    assert math.isclose(doc_big[0], math.exp(100.0), rel_tol=1e-12), doc_big
    combined_big = L._combined_seed_doc_ci(big, n_boot=60, seed=5)
    assert combined_big['status'] == 'ok', combined_big
    assert math.isclose(combined_big['ci95_ppl'][0], math.exp(100.0),
                        rel_tol=1e-12), combined_big
    json.dumps(combined_big, allow_nan=False)

    # The paired row bootstrap shares the same denominator/numerator hazards.
    a = [{'variant': 'a', 'seed': s, 'ppl': math.exp(100.0),
          'nll_per_row': [1e308] * 3,
          'n_tokens_per_row': [10 ** 306] * 3} for s in range(3)]
    b = [{'variant': 'b', 'seed': s, 'ppl': math.exp(150.0),
          'nll_per_row': [1.5e308] * 3,
          'n_tokens_per_row': [10 ** 306] * 3} for s in range(3)]
    row = L.paired_row_bootstrap_delta(list(zip(a, b)), n_boot=60, seed=5)
    assert row['status'] == 'ok', row
    expected_delta = math.exp(100.0) - math.exp(150.0)
    assert math.isclose(row['mean_delta'], expected_delta,
                        rel_tol=1e-12), row
    assert math.isclose(row['ci95_delta_ppl'][0], expected_delta,
                        rel_tol=1e-12), row
    json.dumps(row, allow_nan=False)


def test_combined_seed_doc_ci_scales_components_without_squaring_overflow():
    # A finite but huge cross-entropy panel has a representable sample SD and
    # combined SE, yet the ordinary sqrt(s**2 + d**2) and np.std paths square
    # values around 1e200 and overflow.  The components must stay finite (and
    # strict-JSON safe); the exponentiated CI itself is correctly unavailable
    # because no finite PPL endpoint exists for a CE of that size.
    for scale in (1e200, 1e308):
        recs = []
        for s, off in enumerate((0.0, 0.05, 0.1)):
            nll = [scale * (1.0 + off + 0.01 * j) for j in range(4)]
            recs.append({'variant': 'a', 'seed': s, 'ppl': 1.0,
                         'nll_per_row': nll,
                         'n_tokens_per_row': [1] * 4})
        r = L._combined_seed_doc_ci(recs, n_boot=60, seed=5)
        for key in ('se_seed_logppl', 'se_doc_logppl',
                    'se_combined_logppl'):
            assert r[key] is not None and math.isfinite(r[key]), (scale, key, r)
        assert r['se_combined_logppl'] >= max(r['se_seed_logppl'],
                                              r['se_doc_logppl']), r
        assert r['status'] == 'ci_unavailable' and r['ci95_ppl'] is None, r
        json.dumps(r, allow_nan=False)


def test_paired_nll_doc_ci_mean_survives_three_maximum_ppl_differences():
    # exp(709) is the largest finite PPL.  Three seed differences near that
    # value are each representable, but np.mean sums them to inf.  The point
    # estimate (and the zero-variance interval) must stay finite and strict-
    # JSON safe; the same values exercise the exact-ratio fallback.
    A = np.full((3, 3), 709.0)
    B = np.zeros((3, 3))
    tokens = [1, 1, 1]
    expected = math.exp(709.0) - 1.0
    assert math.isfinite(expected)
    for got in (L._paired_nll_doc_ci(A, B, tokens),
                L._paired_nll_doc_ci_extreme(A, B, tokens, 0.95)):
        assert got is not None, 'a finite maximum-PPL panel was discarded'
        assert math.isclose(got['delta_ppl'], expected, rel_tol=1e-15), got
        assert got['delta_log_ratio'] == 709.0, got
        assert got['delta_ppl_ci95'] == [expected, expected], got
        json.dumps(got, allow_nan=False)


def test_pooled_order_and_unit_influence_mean_survives_maximum_ppl_differences():
    # Three per-seed PPL differences near exp(709) overflow a plain column
    # mean although the mean itself is representable.  Both the pooled-order
    # readout and the leave-one-unit-out influence must keep the finite point
    # estimate instead of declaring the panel unmeasurable.
    pattern = np.array([-0.2, -0.1, 0.0, 0.1, 0.2, -0.15, 0.05, 0.1])
    assert math.isclose(float(pattern.sum()), 0.0, abs_tol=1e-15)
    A = 709.0 + np.tile(pattern, (3, 1))
    B = np.zeros((3, 8))
    tokens = [1] * 8
    expected = math.exp(709.0) - 1.0

    block = L.pooled_order_inference(A, B, tokens, n_boot=40, n_perm=0)
    assert block['status'] == 'ok', block
    assert math.isclose(block['point_ppl'], expected, rel_tol=1e-15), block
    assert math.isclose(block['point_log_ratio'], 709.0,
                        rel_tol=0.0, abs_tol=1e-12), block
    json.dumps(block, allow_nan=False)

    inf = L.paired_nll_unit_influence(A, B, tokens)
    assert inf['available'] is True, inf
    assert math.isclose(inf['full_delta_ppl'], expected,
                        rel_tol=1e-15), inf
    finite = [v for v in inf['loo_delta_ppl'] if v is not None]
    assert len(finite) == 8 and all(math.isfinite(v) for v in finite), inf
    json.dumps(inf, allow_nan=False)

    # The helper preserves the ordinary NumPy column mean bit-for-bit.
    ordinary = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert np.array_equal(L._scaled_mean_axis0(ordinary),
                          np.mean(ordinary, axis=0))


def test_tuning_selection_stability_recovers_overflowing_token_totals():
    # Each seed's NLL and token count is finite, but three of them overflow a
    # plain float sum.  The pooled ratio (1.0 here) is still representable, so
    # the leave-one-seed-out choice must be a finite PPL and strict-JSON safe;
    # a NaN would also make min()/max() arbitrary.
    trials = [
        {'config': {'arm': i},
         'seed_ppls': {0: math.e, 1: math.e, 2: math.e},
         'seed_nll_sum': {0: 1e308, 1: 1e308, 2: 1e308},
         'seed_n_tokens': {0: 1e308, 1: 1e308, 2: 1e308}}
        for i in range(3)
    ]
    rep = L.tuning_selection_stability(trials)
    assert rep['status'] == 'ok', rep
    assert all(math.isclose(c['metric'], math.e, rel_tol=1e-15)
               for c in rep['choices']), rep
    json.dumps(rep, allow_nan=False)

    # A fractional token count has no exact pooled denominator; the scan must
    # fail closed instead of truncating it to an integer.
    bad = [{'config': {'arm': 0},
            'seed_ppls': {0: 2.0, 1: 2.0, 2: 2.0},
            'seed_nll_sum': {0: 1e308, 1: 1e308, 2: 1e308},
            'seed_n_tokens': {0: 1.5, 1: 1.5, 2: 1.5}}]
    rep_bad = L.tuning_selection_stability(bad)
    assert rep_bad['status'] == 'unavailable', rep_bad
    json.dumps(rep_bad, allow_nan=False)
