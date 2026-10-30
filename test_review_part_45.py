"""Scale regressions for exact step labels and bounded context scans.

Both checks target paths that are documented as scale-up safe but still did
work proportional to the *unbounded* input size:

* the evaluation-trajectory reader round-tripped step labels through float64
  and the OLS slope converted absolute steps to float64, so a long run whose
  step counter exceeds 2**53 (or float64's exponent range) was rejected or
  silently produced a wrong slope;
* ``_context_target_counts`` materialised ``list(range(n_rows))`` before the
  ``max_positions`` branch replaced it, so a bounded scan of a multi-million
  row split still allocated one Python integer per row.

These are zero-GPU checks.
"""
import contextlib
import io
import json
import math
import os
import sys as _sys
import tempfile

import numpy as np

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403


def _history(base):
    return ({0: [[base, 50.0], [base + 100, 40.0], [base + 200, 30.0]]},
            {0: [[base, 52.0], [base + 100, 43.0], [base + 200, 34.0]]})


def _expected_slope():
    y = [math.log(50.0) - math.log(52.0),
         math.log(40.0) - math.log(43.0),
         math.log(30.0) - math.log(34.0)]
    x = [0.0, 100.0, 200.0]
    xm = sum(x) / len(x)
    ym = sum(y) / len(y)
    den = sum((v - xm) ** 2 for v in x)
    return sum((v - xm) * (w - ym) for v, w in zip(x, y)) / den * 1000.0


def test_trajectory_step_labels_stay_exact_at_scale():
    # Exact integers above float64's integer range and exponent range are
    # still valid step labels.  The slope is translation invariant, so the
    # same history must give the same slope at every origin.
    expected = _expected_slope()
    for base in (0, 10 ** 18, 2 ** 60, 10 ** 400):
        ha, hb = _history(base)
        r = L.paired_trajectory_stability(ha, hb)
        assert r['status'] == 'stable', (base, r)
        assert r['final_step'] == base + 200, (base, r['final_step'])
        assert r['shared_steps'] == [base, base + 100, base + 200], r
        assert math.isclose(r['tail_gap_slope_log_per_1k'], expected,
                            rel_tol=1e-12, abs_tol=1e-15), (base, r)
        json.dumps(r, allow_nan=False)


def test_trajectory_step_parser_still_rejects_non_integral_labels():
    for bad in (True, 1.5, '100', b'100', None, -1):
        assert L._traj_step_int(bad) is None, bad
    assert L._traj_step_int(np.int64(2 ** 62)) == 2 ** 62
    assert L._traj_step_int(10 ** 400) == 10 ** 400


def test_bounded_context_scan_does_not_materialise_every_row(monkeypatch):
    import builtins

    class _GuardedRange:
        def __init__(self, n):
            self.n = int(n)

        def __iter__(self):
            raise AssertionError(
                f'unbounded range({self.n}) iterated for a bounded scan')

        def __len__(self):
            raise AssertionError(
                f'unbounded range({self.n}) sized for a bounded scan')

    def _guarded_range(*args):
        # A plain range object is lazy and harmless; *iterating* the full row
        # range (or sizing it into a list) is the O(n_rows) work the bounded
        # path must avoid.  Small ranges (the position loop) pass normally.
        if len(args) == 1 and int(args[0]) > 1000:
            return _GuardedRange(args[0])
        return builtins.range(*args)

    class _HugeRows:
        shape = (10_000_000, 8)

        def __getitem__(self, row):
            return np.arange(8, dtype=np.int64)

    monkeypatch.setattr(L, 'range', _guarded_range, raising=False)
    out, ctx_len, meta = L._context_target_counts(
        _HugeRows(), 3, max_positions=2)
    assert ctx_len == 3, ctx_len
    assert meta['n_rows_total'] == 10_000_000, meta
    assert meta['n_rows_scanned'] == 1, meta
    assert meta['n_positions_scanned'] == 2, meta
    assert meta['complete'] is False, meta
    assert out, out


def test_trajectory_audit_replays_exact_large_step_labels():
    base = 10 ** 400
    ha, hb = _history(base)
    tr = L.paired_trajectory_stability(ha, hb)
    panel = {
        'n_contrasts': 1, 'n_stable': 1, 'n_late_crossing': 0,
        'n_zero_final': 0, 'n_final_point_only': 0,
        'n_insufficient_history': 0, 'n_invalid': 0,
        'n_gap_narrowing': 0, 'n_primary_history_disagreement': 0,
        'n_scale_sensitive_final': 0, 'policy': tr['policy'],
    }
    data = {
        'trajectory_stability': panel,
        'components': {'comp': [{
            'with': 'a', 'without': 'b', 'trajectory_stability': tr}]},
    }
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'methodology_audit.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, allow_nan=False)
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_trajectory(td)
    assert out['status'] == 'ok', out
    assert out['n_ok'] == 1, out
