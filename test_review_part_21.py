"""Scale-invariance regressions for preprocessing and look histories.

Both paths consume values whose unit is arbitrary.  A feature can be all
around ``1e-200`` without being constant, and a paired-effect look can be
rescaled by any positive constant without changing whether it is the same
look as the current record.  Absolute epsilons (``eps=1e-12``,
``abs(d) > 1e-9``, ``round(v, 12)``) silently made those decisions depend on
the unit instead of the data.
"""
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403
from test_support import _factorial_records


def test_preprocessor_scaling_is_invariant_under_a_units_only_rescaling():
    import data_guard as DG

    base = np.array([[0.0], [0.5], [1.0], [2.0]])
    for mode in ('standard', 'robust', 'minmax'):
        ref = DG.LeakageSafePreprocessor(scale=mode).fit(base).transform(base)
        assert np.all(np.isfinite(ref)), (mode, ref)
        for factor in (1e-6, 1e-12, 1e-13, 1e-50, 1e-150, 1e-300,
                       1e150, 1e300):
            X = base * factor
            got = DG.LeakageSafePreprocessor(scale=mode).fit(X).transform(X)
            # A rescaling must not change the fitted transform's values.
            assert np.allclose(got, ref, rtol=1e-10, atol=1e-12), (
                mode, factor, got.ravel(), ref.ravel())


def test_look_stale_detection_and_dedup_use_one_relative_scale():
    name = 'dynamic_blocking_x_learned_selection'
    key = L._interaction_look_key(name)
    recs = _factorial_records(0.05, n=5)
    snap = L.factorial_look_snapshot(recs)[key]

    for factor in (1e-150, 1e-12, 1.0, 1e150):
        scaled = copy.deepcopy(snap)
        scaled['deltas'] = [x * factor for x in snap['deltas']]
        history = {'looks': [{'look': 1, 'contrasts': {key: scaled}}]}

        # The same values at the same unit are one look, not optional stopping.
        seq, stale, n = L._interaction_look_analysis(
            L.looks_by_contrast(history), key,
            scaled['seeds'], scaled['deltas'],
            final_tags=scaled['tags'], final_cells=scaled['cells'])
        assert n == 1 and not stale, (factor, seq, stale, n)

        # A real edit to a shared seed is stale at every unit.  The edit is
        # 1% of the largest delta, so it is well above rounding noise after
        # normalisation but would be invisible to an absolute 1e-9 rule once
        # the whole panel is tiny.
        changed = copy.deepcopy(scaled)
        _peak = max(abs(v) for v in changed['deltas'])
        changed['deltas'][0] += 0.01 * _peak
        history_bad = {'looks': [{'look': 1, 'contrasts': {key: changed}}]}
        seq_bad, stale_bad, n_bad = L._interaction_look_analysis(
            L.looks_by_contrast(history_bad), key,
            scaled['seeds'], scaled['deltas'],
            final_tags=scaled['tags'], final_cells=scaled['cells'])
        assert n_bad == 1 and stale_bad, (factor, seq_bad, stale_bad, n_bad)
        assert stale_bad[0]['reason'] == 'seed delta changed across looks', (
            factor, stale_bad)
