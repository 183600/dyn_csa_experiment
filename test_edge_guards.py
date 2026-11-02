"""Edge guards for local-RNG seeds and extreme normal tails.

Two failures used to escape as uncaught exceptions or silently wrong numbers:

* every bootstrap helper passed a caller-supplied seed straight to
  ``numpy.random.default_rng``; a negative seed raised ``ValueError`` instead
  of returning the helper's documented "unavailable" result;
* the Type S/M lower tail and the design-sensitivity normal quantile used
  ``0.5 * (1 + erf(x))`` and a fixed [-9, 9] bracket, so probabilities below
  ~1e-19 became exactly zero and a quantile beyond the bracket edge was
  silently clamped.

These checks are CPU-only and do not read a split or start a run.
"""
import math

import pytest

import exp_lib as L


def test_bootstrap_helpers_reject_a_negative_seed_without_raising():
    vals = [1.0, 2.0, 3.0, 4.0]
    assert L.bootstrap_mean_ci(vals, n_boot=64, seed=-1) is None
    assert L.paired_bootstrap_ci(vals, n_boot=64, seed=-1) is None
    assert L.unpaired_bootstrap_ci(vals, vals, n_boot=64, seed=-1) is None
    assert L._bootstrap_t_interval(
        vals, n_boot=64, seed=-1, conf=0.95)['ci_low'] is None

    recs = [{'variant': 'a', 'seed': s, 'ppl': 30.0,
             'nll_per_row': [1.0, 1.2, 0.8, 1.1],
             'n_tokens_per_row': [10, 10, 10, 10]}
            for s in (0, 1)]
    assert L._doc_bootstrap_ppl_ci(recs, n_boot=40, seed=-1) is None
    combined = L._combined_seed_doc_ci(recs, n_boot=40, seed=-1)
    assert combined['status'] == 'unverifiable'
    assert 'seed' in combined['reason']
    row = L.paired_row_bootstrap_delta(
        [({'nll_per_row': [1.0, 2.0], 'n_tokens_per_row': [10, 10]},
          {'nll_per_row': [0.9, 2.1], 'n_tokens_per_row': [10, 10]})],
        n_boot=20, seed=-1)
    assert row['status'] == 'unverifiable'
    assert 'seed' in row['reason']

    with pytest.raises(ValueError, match='seed'):
        L.paired_nll_two_way_bootstrap(
            [[1.0, 2.0], [1.1, 2.1]], [[0.9, 2.0], [1.0, 2.1]],
            [10, 10], n_boot=20, seed=-1)

    # The ordinary path is unchanged and still deterministic.
    assert L.bootstrap_mean_ci(vals, n_boot=64, seed=7) is not None
    assert L.paired_bootstrap_ci(vals, n_boot=64, seed=7) is not None


def test_type_sm_lower_tail_keeps_a_tiny_but_real_power():
    # P(Z < -10) is 7.6e-24, not zero.  The old cdf rounded it to 0 and the
    # whole conditional readout became "not computable".
    assert L._tsm_cdf(-10.0) == pytest.approx(7.619853024160526e-24, rel=1e-12)
    assert L._tsm_sf(10.0) == pytest.approx(L._tsm_cdf(-10.0), rel=1e-15)
    met = L._tsm_metrics(12, 0.1, 20.0)
    assert met is not None and met['power'] > 0.0
    out = L.type_sm_design(12, 1.0, 0.1, critical_value=20.0)
    assert out['status'] in ('ok', 'warnings'), out
    assert out['power'] is not None and out['power'] > 0.0, out
    assert out['type_s'] is not None and out['type_m'] is not None, out


def test_normal_quantile_does_not_clamp_at_the_old_bracket_edge():
    assert L._normal_quantile(0.5) == pytest.approx(0.0, abs=1e-15)
    assert L._normal_quantile(1e-20) < -9.0
    assert L._normal_quantile(1e-300) == pytest.approx(
        -37.0470962993612, abs=1e-10)
    assert L._normal_quantile(1.0 - 1e-16) > 8.0
    assert L._normal_quantile(0.0) != L._normal_quantile(0.0)  # NaN


def test_distribution_tails_reject_absurd_arguments_without_raising():
    # ``float(10**400)`` raises OverflowError.  The distribution helpers must
    # map that valid integer to its mathematical limit (an infinite statistic
    # or infinite df), or return NaN for a non-measurement, never escape.
    assert L.student_t_two_sided_p(10 ** 400, 10) == 0.0
    assert L.student_t_two_sided_p(1.0, 10 ** 400) == pytest.approx(
        math.erfc(1.0 / math.sqrt(2.0)), rel=1e-14)
    assert L.chi2_sf(10 ** 400, 10) == 0.0
    assert L.chi2_sf(-(10 ** 400), 10) == 1.0
    assert L.chi2_sf(1.0, 10 ** 400) == 1.0
    assert L._t_critical(0.95, 10 ** 400) == pytest.approx(
        1.959963984540054, rel=1e-14)
    for bad in (True, '1', b'1', None):
        assert math.isnan(L.student_t_two_sided_p(bad, 10))
        assert math.isnan(L.chi2_sf(1.0, bad))
        assert math.isnan(L._t_critical(0.95, bad))
    # Ordinary values are unchanged.
    assert L.student_t_two_sided_p(0.0, 10) == 1.0
    assert L.chi2_sf(3.841458820694124, 1.0) == pytest.approx(0.05, abs=1e-12)


def test_global_fdr_huge_tamper_is_a_reported_failure_not_a_crash():
    import copy
    import json

    art = L.build_global_fdr([{'name': 'a', 'p': 0.01, 'n_nonzero': 12}])
    for path in (('result', 'n_reject'), ('result', 'm_declared'),
                 ('result', 'alpha'), ('result', 'claims', 'a', 'q'),
                 ('schema',), ('kind',), ('result', 'warnings')):
        bad = copy.deepcopy(art)
        node = bad
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = [10 ** 10000] if path[-1] == 'warnings' else 10 ** 10000
        verdict = L.validate_global_fdr(bad)
        assert verdict['status'] == 'fail', (path, verdict)
        assert verdict['problems'], (path, verdict)
        # Formatting the diagnostic must not hit Python's integer-digit limit.
        json.dumps(verdict, allow_nan=False)


def test_global_fdr_treats_numeric_strings_as_non_measurements():
    got = L.global_fdr({'a': '0.01', 'b': b'0.02'})
    assert got['status'] == 'not_applicable', got
    assert got['claims']['a']['p'] is None
    assert got['claims']['b']['p'] is None
    with pytest.raises(ValueError, match='p_floor'):
        L.global_fdr({'a': {'p': 0.2, 'p_floor': '0.1'}})


def test_frontier_cap_accepts_an_arbitrary_precision_max_points():
    cells = []
    for i in range(20):
        cells.append({'label': f'a::seed{i}', 'variant': 'a', 'block': str(i)})
        cells.append({'label': f'b::seed{i}', 'variant': 'b', 'block': str(i)})
    comps = [{'name': 'a-b', 'a': 'a', 'b': 'b'}]
    out = L.truncation_frontier(
        cells, comps, min_n=2, complete_coverage=False,
        required_blocks=[str(i) for i in range(20)],
        max_points=10 ** 400)
    assert out['status'] == 'ok', out['problems']
    assert len(out['points']) <= len(cells)


def _ablation_cells(seeds, comps=('a', 'b')):
    import itertools
    out = {}
    effects = {'a': -0.6, 'b': -0.3}
    for size in range(len(comps) + 1):
        for combo in itertools.combinations(comps, size):
            out[frozenset(combo)] = {
                seed: math.exp(3.0 + sum(effects[c] for c in combo))
                for seed in seeds}
    return out


def test_ablation_ranking_floor_does_not_overflow_at_scale():
    # 2 / 2**1100 cannot be evaluated as 2.0 / (1 << n): converting the
    # 1100-bit integer denominator to float raises OverflowError.  The floor
    # is mathematically 0.0 at this scale and must not abort the ranking.
    cells = _ablation_cells(range(1100))
    out = L.ablation_ranking(cells, components=['a', 'b'],
                             scales=('log_ppl',))
    assert out['status'] in ('ok', 'warnings'), out['problems']
    assert out['p_floor'] == 0.0
    import json
    json.dumps(out, allow_nan=False)


def test_ablation_ranking_large_seed_label_replays():
    import copy
    huge = 10 ** 400
    arts = L.build_ablation_ranking(
        _ablation_cells([huge, 0, 1]), components=['a', 'b'])
    assert L.validate_ablation_ranking(arts)['status'] == 'ok'
    assert L.validate_ablation_ranking(copy.deepcopy(arts))['status'] == 'ok'


def test_ablation_ranking_huge_tamper_is_reported_not_a_crash():
    import copy
    import json
    art = L.build_ablation_ranking(_ablation_cells(range(4)),
                                   components=['a', 'b'])
    for path in (('kind',), ('schema',), ('inputs', 'min_n'),
                 ('inputs', 'max_components'), ('result', 'n_seeds'),
                 ('result', 'p_floor')):
        bad = copy.deepcopy(art)
        node = bad
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = 10 ** 10000
        verdict = L.validate_ablation_ranking(bad)
        assert verdict['status'] == 'fail', (path, verdict)
        assert verdict['problems'], (path, verdict)
        # Formatting the diagnostic must not hit the integer-digit limit.
        json.dumps(verdict, allow_nan=False)
    # Mixed non-string keys must be sorted/formatted through a bounded repr,
    # not by comparing int with str or by calling repr(10**10000).
    for where in ('top', 'inputs', 'result'):
        bad = copy.deepcopy(art)
        if where == 'top':
            bad[10 ** 10000] = 1
        elif where == 'inputs':
            bad['inputs'][10 ** 10000] = 1
        else:
            bad['result'][10 ** 10000] = 1
        verdict = L.validate_ablation_ranking(bad)
        assert verdict['status'] == 'fail', (where, verdict)
        json.dumps(verdict, allow_nan=False)


def test_ablation_lattice_huge_status_is_reported_not_a_crash():
    import copy
    import json
    art = L.build_ablation_lattice(_ablation_cells(range(3)),
                                   components=['a', 'b'], scale='ppl')
    bad = copy.deepcopy(art)
    bad['status'] = 10 ** 10000
    verdict = L.validate_ablation_lattice(bad)
    assert verdict['status'] == 'fail', verdict
    assert verdict['problems'], verdict
    json.dumps(verdict, allow_nan=False)
