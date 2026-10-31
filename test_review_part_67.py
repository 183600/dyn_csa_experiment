"""CPU review checks (split group 67): arm-graph identity and replay scale.

These pin three defects in ``contrast_network_audit`` that a pairwise-only
fixture does not reach:

* a third/fourth duplicate edge could re-enter the graph after the first
  contradictory duplicate had been removed;
* duplicates with the same log ratio but different arm PPLs were treated as
  identical, so two different node measurements were silently collapsed;
* the indirect-contrast search scanned every edge for every node pair and ran
  a BFS for every non-direct pair even though only 128 rows are published.

The replay gate also has to name a malformed/tampered stored block instead of
raising before it can report a verdict.
"""
import contextlib
import copy as _copy
import io as _io
import json as _json
import math as _math
import os as _os
import sys as _sys
import tempfile as _tempfile

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def _entry(with_arm, without_arm, with_vals, without_vals, comp, *,
           kind='component', tag_with='', tag_without=''):
    seeds = list(range(len(with_vals)))
    return {
        'with': with_arm, 'without': without_arm,
        'controlled': True, 'fully_paired': True,
        'component': comp, 'kind': kind,
        'tag_with': tag_with, 'tag_without': tag_without,
        'seeds': seeds,
        'per_seed_with': list(with_vals),
        'per_seed_without': list(without_vals),
        'deltas': [w - wo for w, wo in zip(with_vals, without_vals)],
    }


def _chain(n):
    vals = [45.0 + 0.01 * i for i in range(n)]
    return [_entry('a%d' % i, 'a%d' % (i + 1), [vals[i]] * 2,
                   [vals[i + 1]] * 2, 'c%d' % i)
            for i in range(n - 1)]


def test_conflicting_duplicate_edge_cannot_be_reintroduced_by_a_third_copy():
    rows = [
        _entry('a', 'b', [45.0, 45.0], [50.0, 50.0], 'x'),
        _entry('a', 'b', [44.0, 44.0], [50.0, 50.0], 'y'),
        _entry('a', 'b', [43.0, 43.0], [50.0, 50.0], 'z'),
    ]
    r = L.contrast_network_audit(rows)
    assert r['status'] == 'warn', r
    assert r['n_edges'] == 0 and r['edges'] == {}, r
    assert len(r['conflicting_edges']) == 1, r['conflicting_edges']
    assert r['conflicting_edges'][0]['components'] == ['x', 'y', 'z'], r
    assert r['cross_component_ranking_allowed'] is False, r


def test_duplicate_with_equal_ratio_but_different_ppls_is_not_identical():
    # 25/50 and 50/100 have the same log ratio, but they are different arm
    # measurements.  Collapsing them would fabricate one shared node.
    rows = [
        _entry('a', 'b', [25.0, 25.0], [50.0, 50.0], 'x'),
        _entry('a', 'b', [50.0, 50.0], [100.0, 100.0], 'y'),
    ]
    r = L.contrast_network_audit(rows)
    assert r['status'] == 'warn', r
    assert r['n_edges'] == 0 and len(r['conflicting_edges']) == 1, r


def test_self_contrast_is_named_and_does_not_create_a_self_loop():
    rows = [_entry('a', 'a ', [45.0, 45.0], [50.0, 50.0], 'x')]
    r = L.contrast_network_audit(rows)
    assert r['status'] == 'not_estimable', r
    assert r['n_edges'] == 0 and r['n_nodes'] == 0, r
    assert any('self-contrast' in e['reason'] for e in r['ignored_entries']), r


def test_malformed_labels_do_not_break_strict_json_output():
    rows = _chain(3)
    rows[0]['component'] = _math.inf
    rows[1]['kind'] = _math.nan
    r = L.contrast_network_audit(rows)
    _json.dumps(r, allow_nan=False)
    assert any(label in ('inf', 'nan') for label in
               [e.get('component') for e in r['edges'].values()]
               + [e.get('kind') for e in r['edges'].values()]), r


def test_indirect_search_is_capped_per_component_not_per_pair(monkeypatch):
    # A 300-arm chain has ~45k connected non-direct pairs; the published table
    # is capped at 128.  The search must not run one BFS for every pair.
    rows = _chain(300)
    calls = []
    original = L._cn_shortest_path

    def _counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(L, '_cn_shortest_path', _counted)
    r = L.contrast_network_audit(rows)
    assert r['status'] == 'ok', r
    assert r['n_edges'] == 299, r
    assert len(r['indirect_contrasts']) == int(
        L.CONTRAST_NETWORK_POLICY['max_indirect_pairs']), r
    assert len(calls) <= int(L.CONTRAST_NETWORK_POLICY['max_indirect_pairs']), (
        'the path search was not capped: %d BFS calls' % len(calls))


def _network_report(td):
    fp = ('steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_'
          'dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX')
    summary = {}
    for i in range(4):
        for arm, ppl in (('a', 45.0), ('b', 50.0), ('c', 52.0)):
            summary['%s::seed%d' % (arm, i)] = dict(
                variant=arm, seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                params=100, ppl=ppl)
    design = {'ab': [('a', 'b', 'A vs B')], 'bc': [('b', 'c', 'B vs C')]}
    L.methodology_audit(summary, design=design, baselines={},
                        outdir=td, verbose=False)
    return _os.path.join(td, 'methodology_audit.json')


def test_gate_rejects_tampered_edge_metadata_and_indirect_rows():
    with _tempfile.TemporaryDirectory() as td:
        path = _network_report(td)
        raw = _json.load(open(path, encoding='utf-8'))
        assert A.audit_contrast_network(path)['status'] == 'ok'

        bad = _copy.deepcopy(raw)
        bad['contrast_network']['edges']['b -> a']['seeds'] = []
        with open(path, 'w', encoding='utf-8') as f:
            _json.dump(bad, f, allow_nan=False)
        out = A.audit_contrast_network(path)
        assert out['status'] == 'fail', out
        assert any('edge_meta' in v for v in out['violations']), out

        bad = _copy.deepcopy(raw)
        bad['contrast_network']['indirect_contrasts']['a -> c']['path'] = []
        with open(path, 'w', encoding='utf-8') as f:
            _json.dump(bad, f, allow_nan=False)
        out = A.audit_contrast_network(path)
        assert out['status'] == 'fail', out
        assert any('indirect_meta' in v for v in out['violations']), out


def test_gate_fails_closed_on_malformed_block_fields():
    with _tempfile.TemporaryDirectory() as td:
        path = _network_report(td)
        raw = _json.load(open(path, encoding='utf-8'))
        for field, replacement in (('node_conflicts', 1),
                                   ('edges', 'not-an-object'),
                                   ('indirect_contrasts', [])):
            bad = _copy.deepcopy(raw)
            bad['contrast_network'][field] = replacement
            with open(path, 'w', encoding='utf-8') as f:
                _json.dump(bad, f, allow_nan=False)
            with contextlib.redirect_stdout(_io.StringIO()):
                out = A.audit_contrast_network(path)
            assert out['status'] == 'fail', (field, out)
            assert any(field in v for v in out['violations']), (field, out)
