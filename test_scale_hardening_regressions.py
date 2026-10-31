"""Scale-safety regressions for exact enumeration and provenance helpers.

These checks cover finite-but-unbounded inputs that the existing ordinary
fixtures do not reach:

* a caller-supplied ``max_n`` must not disable the exact-sign-flip allocation
  guard (``2**n`` is materialised in memory);
* malformed stored p-values/frozen Holm maps are a ``not_estimable`` readout,
  not a raw exception from a report builder;
* optional per-seed PPL provenance is validated before it can leak ``inf`` /
  ``NaN`` / strings into a strict-JSON block;
* a non-constant extreme variance panel still reports its scale-free readout
  (shares/ICC/effective N) instead of being labelled "not_estimable" merely
  because the absolute variance does not fit float64.
"""
import json
import math
import os
import sys as _sys

import numpy as np

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def _family(*, n_seeds):
    rows = {'c0': [float(i + 1) for i in range(n_seeds)]}
    comps = {name: {'per_seed': vals,
                    'seeds': list(range(n_seeds)),
                    'p_exact_signflip': L.exact_signflip_p(vals)['p']}
             for name, vals in rows.items()}
    return comps, list(rows)


def test_maxt_exact_enumeration_has_an_absolute_memory_guard():
    hard = int(L.JOINT_SIGNFLIP_HARD_MAX_N)
    deltas = [float(i + 1) for i in range(hard + 1)]
    try:
        L.maxt_signflip_adjust({'c': deltas}, max_n=hard + 1)
    except ValueError as e:
        assert 'hard cap' in str(e), e
    else:
        raise AssertionError('a caller-controlled max_n disabled the memory guard')

    # The grouping wrapper turns the refusal into a named not_estimable group
    # instead of attempting the allocation or crashing.
    rep = L.maxt_joint_correction(
        {'c': {'deltas': deltas, 'seeds': list(range(hard + 1))}},
        family_size=1, max_n=hard + 1)
    assert rep['status'] == 'not_estimable', rep
    assert any('hard cap' in w for w in rep['warnings']), rep['warnings']
    json.dumps(rep, allow_nan=False)


def test_confirmation_family_exact_enumeration_has_an_absolute_guard():
    hard = int(L.FAMILY_SIGNFLIP_HARD_MAX_N)
    comps, names = _family(n_seeds=hard + 1)
    rep = L.confirmation_family_correction(
        comps, family_names=names, max_n=hard + 1)
    assert rep['status'] == 'not_estimable', rep
    assert 'hard cap' in (rep.get('reason') or ''), rep


def test_family_correction_rejects_malformed_replay_fields_as_not_estimable():
    comps, names = _family(n_seeds=2)
    for bad in ({'c0': 'not-a-p'}, {'c0': float('inf')}):
        rep = L.confirmation_family_correction(
            comps, family_names=names, frozen_holm=bad)
        assert rep['status'] == 'not_estimable', (bad, rep)
        assert 'frozen_holm' in (rep.get('reason') or ''), rep

    bad_comps = {name: dict(rec) for name, rec in comps.items()}
    bad_comps['c0']['p_exact_signflip'] = 'not-a-p'
    rep = L.confirmation_family_correction(bad_comps, family_names=names)
    assert rep['status'] == 'not_estimable', rep
    assert 'mean p-value' in (rep.get('reason') or ''), rep


def test_practical_significance_validates_optional_ppl_provenance():
    base = dict(deltas=[-0.05, -0.06], margin=0.1,
                seeds=[0, 1], expected_sign='negative')
    good = L.confirmation_practical_significance(
        **base, ppl_a=[50.0, 50.1], ppl_b=[51.0, 51.2])
    assert good['per_seed_ppl_a'] == [50.0, 50.1], good
    json.dumps(good, allow_nan=False)

    bad_calls = (
        dict(ppl_a=[1.0, float('inf')], ppl_b=[1.0, 1.0]),
        dict(ppl_a=[1.0, float('nan')], ppl_b=[1.0, 1.0]),
        dict(ppl_a=[1.0, None], ppl_b=[1.0, 1.0]),
        dict(ppl_a=[1.0, '1.0'], ppl_b=[1.0, 1.0]),
        dict(ppl_a=[1.0, 0.0], ppl_b=[1.0, 1.0]),
        dict(ppl_a=[1.0], ppl_b=[1.0, 1.0]),
        dict(ppl_a=[1.0, 1.0]),
    )
    for extra in bad_calls:
        try:
            L.confirmation_practical_significance(**base, **extra)
        except ValueError:
            pass
        else:
            raise AssertionError(f'malformed PPL provenance was accepted: {extra!r}')

    try:
        L.confirmation_practical_significance(**base, scale='bogus')
    except ValueError as e:
        assert 'scale' in str(e), e
    else:
        raise AssertionError('an unknown effect scale silently fell back')


def test_extreme_variance_panel_keeps_scale_free_readouts():
    base = np.array([[-2.0, -1.0, 0.0, 2.0],
                     [1.0, 0.0, -1.0, 0.5],
                     [-0.5, 1.0, 2.0, -1.0]])
    rep = L.crossed_variance_components(base * 1e200)
    assert rep['status'] == 'ok', rep
    assert rep['component_scale']['absolute_available'] is False, rep
    assert math.isfinite(rep['grand_mean']), rep
    assert abs(sum(rep['icc'].values()) - 1.0) < 1e-12, rep['icc']
    assert abs(sum(rep['components'][k]['share']
                   for k in ('seed', 'unit', 'residual')) - 1.0) < 1e-12, rep
    assert rep['effective_n']['equivalent'] is not None, rep
    assert rep['total_variance'] is None, rep
    json.dumps(rep, allow_nan=False)


def test_bootstrap_helpers_refuse_an_unbounded_draw_count():
    """A caller-controlled n_boot must not become an unbounded allocation.

    These legacy helpers materialise at least one float per draw.  The newer
    resampling readouts already declare a 200k hard cap; the shared constant
    keeps the older helpers from accepting ``n_boot=10**9`` and exhausting
    memory before any interval is reported.
    """
    hard = int(L.BOOTSTRAP_HARD_MAX_DRAWS)
    assert hard >= 1000
    vals = [1.0, 2.0, 3.0, 4.0]
    tint = L._bootstrap_t_interval(vals, n_boot=hard + 1, seed=7, conf=0.95)
    assert tint['ci_low'] is None and tint['ci_high'] is None, tint
    assert 'n_boot' in tint['method'], tint
    assert L.paired_bootstrap_ci(vals, n_boot=hard + 1) is None
    assert L.unpaired_bootstrap_ci(vals, vals, n_boot=hard + 1) is None
    assert L.bootstrap_mean_ci(vals, n_boot=hard + 1) is None

    recs = [{'variant': 'a', 'seed': s, 'ppl': 30.0,
             'nll_per_row': [1.0, 1.2, 0.8, 1.1],
             'n_tokens_per_row': [10, 10, 10, 10]}
            for s in (0, 1)]
    assert L._doc_bootstrap_ppl_ci(recs, n_boot=hard + 1) is None
    combined = L._combined_seed_doc_ci(recs, n_boot=hard + 1)
    assert combined['status'] == 'unverifiable', combined
    assert 'n_boot' in combined['reason'], combined

    # The ordinary path still returns the same seeded interval.
    assert L.paired_bootstrap_ci(vals, n_boot=64, seed=7) is not None
    assert L._combined_seed_doc_ci(recs, n_boot=40)['status'] == 'ok'
