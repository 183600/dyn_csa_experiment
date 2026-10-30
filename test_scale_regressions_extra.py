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



