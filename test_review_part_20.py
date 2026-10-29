"""CPU review checks (split group 20): seed x unit total-uncertainty interval.

The cluster-robust Δ CI conditions on the trained seeds, while the seed-level
sign-flip test says nothing about evaluation-unit sampling.  Reporting either
one as "the" uncertainty understates the total.  These checks cover the
two-way bootstrap: it must reproduce the frozen pooled point estimate, widen
when seed-to-seed variation exists even if documents do not vary, stay
deterministic/confidence-aware, refuse malformed accounting, and be wired into
the v12 analysis, report cell and audit gate without changing any p-value.
"""
import json
import os
import sys as _sys
import tempfile

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403


def _close(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def _seed_only_panel(n_seeds=6, n_units=20, token=100):
    """Constant across units within a seed, but different across seeds.

    The document cluster interval is exactly zero (each seed's per-unit NLL is
    proportional to the unit's tokens), so any positive joint width is the
    seed component this change exists to expose.
    """
    tokens = [token] * n_units
    a, b = [], []
    for s in range(n_seeds):
        ce_a = 1.0 + 0.10 * s
        ce_b = 1.03
        a.append([ce_a * t for t in tokens])
        b.append([ce_b * t for t in tokens])
    return a, b, tokens


def test_two_way_bootstrap_reproduces_the_pooled_point_and_adds_seed_variation():
    a, b, tokens = _seed_only_panel()
    got = L.paired_nll_two_way_bootstrap(a, b, tokens, n_boot=600, seed=7)
    doc = L._paired_nll_doc_ci(a, b, tokens)
    assert got['status'] == 'ok', got
    assert got['n_seeds'] == 6 and got['n_units'] == 20, got
    assert _close(got['delta_ppl'], doc['delta_ppl'], 1e-15), (got, doc)
    assert got['delta_ppl_ci95'][0] <= got['delta_ppl'] <= got['delta_ppl_ci95'][1], got
    # Documents carry no within-seed variation here, so the conditional
    # cluster interval is degenerate; the joint interval must not be.
    assert doc['se_ppl'] is not None and doc['se_ppl'] < 1e-12, doc
    assert (got['delta_ppl_ci95'][1] - got['delta_ppl_ci95'][0]) > 0.05, got
    assert got['se_ppl'] is not None and got['se_ppl'] > 0.0, got
    assert got['ratio_ci95'] is not None and got['ratio_ci95'][0] < got['ratio_ci95'][1], got
    assert 'two-way' in got['method'] and got['role'], got


def test_two_way_bootstrap_is_deterministic_and_respects_confidence():
    a, b, tokens = _seed_only_panel()
    r1 = L.paired_nll_two_way_bootstrap(a, b, tokens, n_boot=400, seed=11)
    r2 = L.paired_nll_two_way_bootstrap(a, b, tokens, n_boot=400, seed=11)
    assert r1['delta_ppl_ci95'] == r2['delta_ppl_ci95'], (r1, r2)
    r50 = L.paired_nll_two_way_bootstrap(a, b, tokens, n_boot=800, seed=11,
                                         conf=0.50)
    r95 = L.paired_nll_two_way_bootstrap(a, b, tokens, n_boot=800, seed=11,
                                         conf=0.95)
    assert r50['status'] == r95['status'] == 'ok', (r50, r95)
    assert (r50['delta_ppl_ci95'][1] - r50['delta_ppl_ci95'][0]
            < r95['delta_ppl_ci95'][1] - r95['delta_ppl_ci95'][0]), (r50, r95)
    # A different seed changes the draws but not the point estimate.
    r3 = L.paired_nll_two_way_bootstrap(a, b, tokens, n_boot=400, seed=12)
    assert _close(r1['delta_ppl'], r3['delta_ppl'], 1e-15), (r1, r3)


def test_two_way_bootstrap_refuses_malformed_accounting_and_tiny_panels():
    a, b, tokens = _seed_only_panel(n_seeds=4, n_units=5)
    for bad in (
        lambda: L.paired_nll_two_way_bootstrap(a[:-1], b, tokens),
        lambda: L.paired_nll_two_way_bootstrap([[1.0, 2.0]], [[1.0]], [1, 1]),
        lambda: L.paired_nll_two_way_bootstrap([[1.0, True]], [[1.0, 2.0]], [1, 1]),
        lambda: L.paired_nll_two_way_bootstrap([[1.0, float('nan')]], [[1.0, 2.0]], [1, 1]),
        lambda: L.paired_nll_two_way_bootstrap(a, b, [1, 1, 1, 1, 0]),
        lambda: L.paired_nll_two_way_bootstrap(a, b, [1, 1, 1, 1, 1.5]),
        lambda: L.paired_nll_two_way_bootstrap(a, b, [1, 1, 1, 1, 1], conf=1.0),
        lambda: L.paired_nll_two_way_bootstrap(a, b, [1, 1, 1, 1, 1], n_boot=10),
        lambda: L.paired_nll_two_way_bootstrap(a, b, [1, 1, 1, 1, 1], seed='x'),
    ):
        try:
            bad()
            raise AssertionError('malformed two-way input was accepted')
        except ValueError:
            pass
    one_seed = L.paired_nll_two_way_bootstrap(a[:1], b[:1], tokens)
    assert one_seed['status'] == 'unverifiable' and 'one-seed' in one_seed['reason'], one_seed
    one_unit = L.paired_nll_two_way_bootstrap([[1.0]] * 4, [[0.5]] * 4, [10])
    assert one_unit['status'] == 'unverifiable' and 'one-unit' in one_unit['reason'], one_unit


def test_v12_doc_analysis_attaches_the_two_level_interval_without_changing_the_estimate():
    docs = ['0000000000000001', '0000000000000002',
            '0000000000000003', '0000000000000004']
    toks = [10, 20, 30, 70]
    nll_a = [2.0, 6.0, 12.0, 30.0]
    nll_b = [3.0, 5.0, 13.0, 31.0]
    recs_a = {s: {'by_len': {512: _document_cell(nll_a, toks, docs)}}
              for s in (0, 1, 2)}
    recs_b = {s: {'by_len': {512: _document_cell(nll_b, toks, docs)}}
              for s in (0, 1, 2)}
    got = V12._doc_level_analysis(recs_a, recs_b, [0, 1, 2], 512)
    assert got is not None and got['granularity'] == 'document', got
    assert 'pooled_two_level_ci95' in got, got
    lo, hi = got['pooled_two_level_ci95']
    assert lo <= got['pooled_two_level_delta_ppl'] <= hi, got
    assert _close(got['pooled_two_level_delta_ppl'],
                  got['pooled_delta_ppl'], 1e-12), got
    assert got['pooled_two_level_n_seeds'] == 3, got
    assert got['pooled_two_level_n_units'] == 4, got
    assert got['pooled_two_level_n_boot'] >= 20, got
    assert 'two-way' in got['pooled_two_level_method'], got
    assert 'confidence' not in got['pooled_two_level_note'].lower(), got
    cell = V12._doc_ci_markdown_cell(got)
    assert 'cluster' in cell and '两层 bootstrap' in cell, cell
    # The cluster interval remains the primary estimand-specific interval;
    # the joint block is additional, not a replacement.
    assert got['pooled_delta_ppl_ci95'] is not None, got


def _artifact(joint=True, point=0.5, joint_point=None, ci=(-2.0, 2.0),
              doc_ci=(-1.0, 1.0), n_seeds=12, n_units=48, n_boot=2000):
    ds = {
        'pooled_delta_ppl': point,
        'pooled_delta_ppl_ci95': list(doc_ci),
        'pooled_two_level_delta_ppl': (point if joint_point is None else joint_point),
        'pooled_two_level_ci95': list(ci),
        'pooled_two_level_n_seeds': n_seeds,
        'pooled_two_level_n_units': n_units,
        'pooled_two_level_n_boot': n_boot,
        'pooled_two_level_method': ('two-way (seed x evaluation-unit) '
                                    'percentile bootstrap'),
    }
    if not joint:
        for k in list(ds):
            if k.startswith('pooled_two_level'):
                ds.pop(k)
    return {'comparisons': {'main': {'doc_sampling': ds}}}


def test_two_level_audit_ok_fail_warning_and_legacy_states():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'test_stats.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(_artifact(), f)
        ok = A.audit_two_level_ci(p)
        assert ok['status'] == 'ok' and ok['n_ok'] == 1, ok
        # A point that does not reproduce the pooled Δ is a hard failure.
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(_artifact(point=0.5, joint_point=0.9, ci=(-2.0, 2.0)), f)
        bad = A.audit_two_level_ci(p)
        assert bad['status'] == 'fail' and bad['problems'], bad
        # A joint interval narrower than the conditional cluster interval is a
        # warning (the seed component may be missing), not a numeric failure.
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(_artifact(ci=(-0.4, 0.6), doc_ci=(-1.0, 1.0)), f)
        narrow = A.audit_two_level_ci(p)
        assert narrow['status'] == 'warnings' and any(
            'narrower' in w for w in narrow['warnings']), narrow
        # Old artifacts keep the pooled CI but carry no joint block.
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(_artifact(joint=False), f)
        old = A.audit_two_level_ci(p)
        assert old['status'] == 'warnings' and old['n_missing'] == 1, old
        # No artifact at all is an explicit not-verifiable state.
        missing = A.audit_two_level_ci(os.path.join(td, 'absent.json'))
        assert missing['status'] == 'not_verifiable_locally', missing
