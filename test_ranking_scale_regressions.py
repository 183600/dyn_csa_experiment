"""Scale/identity regressions for the ranking-sensitivity and data-guard paths.

These are CPU-only checks written after a review of the newest ranking readout.
They pin four behaviours that were wrong:

* the two-way bootstrap must actually resample seeds as well as units;
* a legitimate finite panel of ``1e308`` token counts must not raise
  ``OverflowError`` in a partial sum, and an underflowing display factor must
  yield ``None`` original-unit means rather than misleading zeros;
* competition ranks may tie, so the unit-ranking audit must accept
  ``sum(p_best) > 1`` for a legitimate all-identical panel;
* a finite ``Decimal`` outside the float64 range must not share a row identity
  with ``float('inf')``, and a large exact seed label must survive replay.
"""
import contextlib
import io
import json
import math
import os
import sys as _sys
import tempfile
from decimal import Decimal

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)

import data_guard as DG


def _unit_payload(vals, tokens, seeds=range(5), ids=True):
    out = {}
    for arm, per_unit in vals.items():
        out[arm] = {}
        for s in seeds:
            cell = {'nll': [v * t for v, t in zip(per_unit, tokens)],
                    'tokens': list(tokens)}
            if ids:
                cell['unit_ids'] = [f'u{i}' for i in range(len(tokens))]
            out[arm][s] = cell
    return out


def test_two_way_really_resamples_seeds():
    # Arm a alternates between a very good and a very bad seed; arm b is
    # constant.  Unit-only therefore always prefers a (averaged over all
    # seeds), while a genuine two-way draw sometimes selects only a-bad seeds.
    tokens = [100.0] * 8
    payload = {'a': {}, 'b': {}}
    for s in range(6):
        ppl_a = 10.0 if s % 2 == 0 else 100.0
        payload['a'][s] = {'nll': [math.log(ppl_a) * t for t in tokens],
                           'tokens': tokens}
        payload['b'][s] = {'nll': [math.log(60.0) * t for t in tokens],
                           'tokens': tokens}
    r = L.arm_rank_unit_sensitivity(payload, n_boot=1000)
    assert r['unit_only']['p_best']['a'] > 0.99, r['unit_only']['p_best']
    # With seed resampling, a loses whenever the draw is dominated by its bad
    # seeds, so its two-way P(best) cannot stay at the unit-only value.
    assert r['two_way']['p_best']['a'] < 0.99, r['two_way']['p_best']
    assert r['two_way']['p_best']['b'] > 0.01, r['two_way']['p_best']


def test_huge_token_counts_and_underflowing_display_scale():
    tokens = [1e308] * 8
    payload = _unit_payload({'a': [1.0] * 8, 'b': [1.01] * 8}, tokens,
                            ids=False)
    r = L.arm_rank_unit_sensitivity(payload, n_boot=100)
    assert r['status'] != 'not_estimable', r
    json.dumps(r, allow_nan=False)
    # A panel whose original PPL underflows float64 still ranks in the shifted
    # log scale; only the unrepresentable linear/mean readouts are withheld.
    payload = _unit_payload({'a': [-1000.0] * 8, 'b': [-1010.0] * 8},
                            [1.0] * 8, ids=False)
    r = L.arm_rank_unit_sensitivity(payload, n_boot=100)
    assert r['observed_means'] == {'a': None, 'b': None}, r['observed_means']
    assert r['observed_log_means']['a'] == -1000.0, r['observed_log_means']
    assert r['observed_log_means']['b'] == -1010.0, r['observed_log_means']
    assert r['observed_winner'] == 'b', r


def test_identical_unit_panel_passes_the_replay_audit():
    payload = _unit_payload({'a': [1.0] * 8, 'b': [1.0] * 8,
                             'c': [1.0] * 8}, [100.0] * 8)
    r = L.arm_rank_unit_sensitivity(payload, n_boot=50)
    assert r['status'] == 'undifferentiated', r
    # Competition ranks make every arm rank 1 in every replicate.
    assert sum(r['unit_only']['p_best'].values()) == 3.0, r['unit_only']
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'stats.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'finaltest': {'arm_rank_units': r}}, f)
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_arm_rank_units(path=path)
    assert out['status'] == 'ok', out


def test_huge_decimal_row_is_not_float_infinity():
    huge = np.array([[Decimal('1e400')]], dtype=object)
    inf = np.array([[float('inf')]], dtype=object)
    assert DG._row_keys(huge)[0] != DG._row_keys(inf)[0]
    assert DG._matrix_fingerprint(huge) != DG._matrix_fingerprint(inf)
    assert DG.duplicate_mask(np.array(
        [[Decimal('1e400')], [float('inf')], [Decimal('Infinity')]],
        dtype=object)).tolist() == [False, False, True]


def test_large_exact_seed_label_replays_through_the_audit():
    huge = 10 ** 400
    cells = {'a': {huge: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0},
             'b': {huge: 2.0, 1: 3.0, 2: 4.0, 3: 5.0, 4: 6.0}}
    r = L.arm_rank_inference(cells, n_boot=200, seed=huge, min_seeds=2)
    assert r['bootstrap_seed'] == huge, r['bootstrap_seed']
    assert huge in r['seeds'], r['seeds']
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'stats.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'finaltest': {'arm_ranking': r}}, f)
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_arm_ranking(path=path)
    assert out['status'] == 'ok', out
