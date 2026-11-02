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
