"""CPU checks for the order-dependence / effective-n readout.

The document interval resamples evaluation units i.i.d., but the frozen slice
is a contiguous source-order prefix.  These checks pin the new descriptive
diagnostic: an ordered sequence with positive serial dependence is flagged,
the order-robust interval is no narrower than the i.i.d. interval, the
effective n is below the raw n, a shuffled/i.i.d. sequence is not flagged, the
permutation p is seeded, extreme panels stay strict-JSON, and the replay audit
catches a hand edit while refusing to call a missing/legacy block a pass.
No result panel and no test split is read.
"""
import contextlib
import copy
import inspect
import io
import json
import math
import os
import tempfile

from test_support import *  # noqa: F401,F403

import pytest as _pytest


def _ar(n=64, rho=0.8, seed=3):
    rng = np.random.default_rng(seed)
    out = []
    value = 0.0
    for shock in rng.normal(size=n):
        value = rho * value + shock
        out.append(value)
    return out


def _iid(n=64, seed=4):
    return np.random.default_rng(seed).normal(size=n).tolist()


def _stored(block, *, name='c1'):
    comp = {'doc_sampling': {'order_inference': block,
                             'n_docs': block.get('n_units'),
                             'pooled_ci_n_units': block.get('n_units')}}
    return {'finaltest': {'comparisons': {name: comp}}}


def _audit(obj):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'test_stats.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(obj, fh)
        with contextlib.redirect_stdout(io.StringIO()):
            return A.audit_order_dependence(path)


def test_policy_status_and_boundary_are_declared():
    assert L.ORDER_DEPENDENCE_POLICY['version'] == 'order-dependence-1'
    assert L.ORDER_DEPENDENCE_STATUSES == ('ok', 'not_estimable')
    assert 'p-value' in L.ORDER_DEPENDENCE_POLICY['boundary']
    assert 'cannot' in L.ORDER_DEPENDENCE_POLICY['boundary']
    block = L.ordered_unit_inference(_iid(12, seed=11))
    assert block['kind'] == 'ordered_unit_inference'
    assert block['status'] == 'ok'
    assert block['n_units'] == 12
    assert not ({'confirmed', 'p_holm', 'verified', 'verdict'} & set(block))
    stats = block['sufficient_statistics']
    assert stats['values'] == _iid(12, seed=11)
    assert stats['block_lengths'] and stats['n_batches']
    assert stats['primary_n_batches'] in stats['n_batches']
    # A valid but too-small panel is an explicit not_estimable readout.
    small = L.ordered_unit_inference(_iid(7, seed=12))
    assert small['status'] == 'not_estimable'
    assert small['kind'] == 'ordered_unit_inference'
    assert small['reason']


def test_ordered_sequence_flags_design_effect_and_shuffle_does_not():
    ordered = L.ordered_unit_inference(_ar())
    assert ordered['order_dependence_detected'] is True
    assert ordered['order_dependence_p'] <= 0.05
    assert ordered['design_effect_max'] > 1.5
    assert ordered['n_effective_min'] < ordered['n_units']
    assert ordered['robust_ci_width'] >= ordered['iid_ci_width']
    assert ordered['inflation_detected'] is True
    assert 'understate' in ordered['wording']
    shuffled = L.ordered_unit_inference(
        np.random.default_rng(5).permutation(np.asarray(_ar())).tolist())
    assert shuffled['order_dependence_detected'] is False
    assert shuffled['inflation_detected'] is False
    assert 'not proof' in shuffled['wording']
    iid = L.ordered_unit_inference(_iid())
    assert iid['order_dependence_detected'] is False
    assert iid['iid_understates'] is False


def test_batch_and_block_tables_are_reported_with_effective_n():
    block = L.ordered_unit_inference([0.5 + 0.02 * i for i in range(48)])
    assert block['status'] == 'ok'
    assert block['batch_means']
    assert block['block_bootstrap']
    for row in block['batch_means']:
        assert row['n_batches'] * row['batch_size'] == row['n_used']
        assert row['design_effect'] is not None
        assert row['n_effective'] is not None
    for row in block['block_bootstrap']:
        assert row['block_length'] >= 2
        assert row['ci'] is not None
        assert row['design_effect'] is not None
    assert block['design_effect_min'] <= block['design_effect_max']
    assert block['n_effective_min'] <= block['n_effective_max']
    # The reported robust interval is the widest declared interval, not a
    # post-hoc choice of the block length that looks best.
    widths = [abs(block['iid_ci'][1] - block['iid_ci'][0])]
    for row in block['block_bootstrap']:
        widths.append(abs(row['ci'][1] - row['ci'][0]))
    assert abs(block['robust_ci_width'] - max(widths)) < 1e-12
    assert block['robust_ci_source'] in (
        ['iid_t'] + [f'block_length_{r["block_length"]}'
                     for r in block['block_bootstrap']])


def test_permutation_and_bootstrap_are_seeded():
    a = L.ordered_unit_inference(_ar(48), n_boot=500, n_perm=500, seed=99)
    b = L.ordered_unit_inference(_ar(48), n_boot=500, n_perm=500, seed=99)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    c = L.ordered_unit_inference(_ar(48), n_boot=500, n_perm=500, seed=100)
    assert c['iid_ci'] == a['iid_ci']
    assert c['permutation']['design_effect_p'] != a['permutation']['design_effect_p'] \
        or c['block_bootstrap'][0]['ci'] != a['block_bootstrap'][0]['ci']


def test_invalid_inputs_fail_closed():
    with _pytest.raises(ValueError):
        L.ordered_unit_inference([[1.0, 2.0], [3.0, 4.0]])
    with _pytest.raises(ValueError):
        L.ordered_unit_inference([1.0, True, 3.0])
    with _pytest.raises(ValueError):
        L.ordered_unit_inference([1.0, '2', 3.0])
    with _pytest.raises(ValueError):
        L.ordered_unit_inference([1.0, float('nan'), 3.0])
    with _pytest.raises(ValueError):
        L.ordered_unit_inference('123456789')
    with _pytest.raises(ValueError):
        L.ordered_unit_inference(_iid(12), conf=1.0)
    with _pytest.raises(ValueError):
        L.ordered_unit_inference(_iid(12), alpha=0.0)
    with _pytest.raises(ValueError):
        L.ordered_unit_inference(_iid(12), seed=-1)
    with _pytest.raises(ValueError):
        L.ordered_unit_inference(_iid(12), n_boot=1)
    with _pytest.raises(ValueError):
        L.ordered_unit_inference(_iid(12), n_perm=-1)
    with _pytest.raises(ValueError):
        L.ordered_unit_inference(_iid(12), block_lengths=[1])
    with _pytest.raises(ValueError):
        L.ordered_unit_inference(_iid(12), block_lengths=[12])
    with _pytest.raises(ValueError):
        L.ordered_unit_inference(_iid(12), n_batches=[100])
    with _pytest.raises(ValueError):
        L.ordered_unit_inference(_iid(12), design_effect_tolerance=0.0)
    with _pytest.raises(ValueError):
        L.ordered_unit_inference(_iid(12), material_ci_inflation=0.9)
    with _pytest.raises(ValueError):
        L.ordered_unit_inference(_iid(12), unit_ids='abc')
    with _pytest.raises(ValueError):
        L.ordered_unit_inference(_iid(12), unit_ids=['a', 'b'])
    with _pytest.raises(ValueError):
        L.ordered_unit_inference(_iid(12), unit_ids=['a'] * 12)


def test_extreme_panels_are_strict_json_and_never_infinite():
    alternating = L.ordered_unit_inference([1e308, -1e308] * 6)
    assert alternating['status'] == 'ok'
    json.dumps(alternating, allow_nan=False)
    for key in ('mean', 'sd', 'se', 'iid_ci_width'):
        assert alternating[key] is None or math.isfinite(float(alternating[key]))
    constant = L.ordered_unit_inference([1.7e308] * 12)
    assert constant['status'] == 'not_estimable'
    json.dumps(constant, allow_nan=False)


def test_doc_level_analysis_stores_the_ordered_block():
    rng = np.random.default_rng(21)
    n_rows, seeds = 12, [0, 1, 2]
    recs_a, recs_b = {}, {}
    base = 50.0
    for s in seeds:
        pa = [base + 0.1 * math.sin(i / 2.0) + 0.01 * s for i in range(n_rows)]
        pb = [p + (0.5 if i < 6 else -0.2) + 0.01 * rng.normal()
              for i, p in enumerate(pa)]
        recs_a[s] = {'by_len': {512: {'ppl_per_seq': pa,
                                      'ppl': float(np.mean(pa)),
                                      'n_tok': n_rows * 511}}}
        recs_b[s] = {'by_len': {512: {'ppl_per_seq': pb,
                                      'ppl': float(np.mean(pb)),
                                      'n_tok': n_rows * 511}}}
    out = V12._doc_level_analysis(recs_a, recs_b, seeds, 512)
    assert out['order_inference']['status'] == 'ok'
    assert out['order_inference']['n_units'] == n_rows
    assert out['order_inference']['sufficient_statistics']['values'] == [
        float(np.mean([recs_a[s]['by_len'][512]['ppl_per_seq'][i]
                       - recs_b[s]['by_len'][512]['ppl_per_seq'][i]
                       for s in seeds]))
        for i in range(n_rows)]


def test_audit_replays_and_detects_tampering():
    block = L.ordered_unit_inference(_ar(48), n_boot=400, n_perm=400)
    good = _stored(block)
    out = _audit(good)
    assert out['status'] == 'ok', out['problems']
    assert out['n_replayed'] == 1 and out['n_ok'] == 1

    for mutate in ('iid_ci', 'design_effect_max', 'robust_ci', 'n_units',
                   'values'):
        obj = copy.deepcopy(good)
        target = obj['finaltest']['comparisons']['c1']['doc_sampling'][
            'order_inference']
        if mutate == 'iid_ci':
            target['iid_ci'][0] += 1.0
        elif mutate == 'design_effect_max':
            target['design_effect_max'] *= 2.0
        elif mutate == 'robust_ci':
            target['robust_ci'][1] += 1.0
        elif mutate == 'n_units':
            target['n_units'] += 1
        else:
            target['sufficient_statistics']['values'][0] += 1.0
        bad = _audit(obj)
        assert bad['status'] == 'fail', (mutate, bad)
        assert bad['problems'], mutate

    obj = copy.deepcopy(good)
    obj['finaltest']['comparisons']['c1']['doc_sampling'][
        'order_inference']['p_holm'] = 0.01
    bad = _audit(obj)
    assert bad['status'] == 'fail'
    assert any('confirmatory' in p for p in bad['problems'])

    obj = copy.deepcopy(good)
    obj['finaltest']['comparisons']['c1']['doc_sampling'][
        'order_inference']['status'] = 'not_estimable'
    na = _audit(obj)
    assert na['status'] == 'not_applicable'

    obj = copy.deepcopy(good)
    del obj['finaltest']['comparisons']['c1']['doc_sampling'][
        'order_inference']
    legacy = _audit(obj)
    assert legacy['status'] == 'not_verifiable_locally'
    assert legacy['warnings']

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'test_stats.json')
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write('{not json')
        with contextlib.redirect_stdout(io.StringIO()):
            broken = A.audit_order_dependence(path)
        assert broken['status'] == 'fail'
        assert any('not a readable JSON object' in p
                   for p in broken['problems'])

    with contextlib.redirect_stdout(io.StringIO()):
        missing = A.audit_order_dependence(
            os.path.join(REPO, 'does-not-exist.json'))
    assert missing['status'] == 'not_verifiable_locally'


def test_audit_never_reads_the_test_split(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError('order-dependence audit must not read test rows')
    monkeypatch.setattr(L, 'load_wikitest', _boom, raising=False)
    out = _audit(_stored(L.ordered_unit_inference(_ar(24), n_boot=300,
                                                  n_perm=300)))
    assert out['status'] == 'ok', out['problems']


def test_markdown_and_cli_wiring():
    block = L.ordered_unit_inference(_ar(32), n_boot=300, n_perm=300)
    ft = {'comparisons': {'c1': {
        'doc_sampling': {'order_inference': block,
                         'pooled_delta_ppl_ci95': [0.1, 0.2]}}}}
    lines = V12._order_inference_markdown(ft)
    text = '\n'.join(lines)
    assert '顺序依赖' in text
    assert 'n_eff' in text
    assert '不改变' in text
    assert '设计效应' in text
    assert V12._order_inference_markdown({'comparisons': {}}) == []
    src = inspect.getsource(A._run_cli)
    for alias in ('order_dependence', 'order', 'serial_dependence',
                  'design_effect', 'effective_n'):
        assert alias in src, alias
    assert 'order_dependence' in inspect.getsource(A.run_report)
    assert '_order_inference_markdown' in inspect.getsource(
        V12.build_report)


def test_audit_replays_default_and_explicit_batch_layouts():
    # The declared default list contains a count that is infeasible at n=12
    # (8 batches of size 1).  The stored block records the raw declaration and
    # the skipped count, so the replay must reproduce the same filtering
    # instead of treating the filtered list as the declaration.
    block = L.ordered_unit_inference(_iid(12, seed=31))
    assert block['skipped_batch_counts']
    assert block['sufficient_statistics']['n_batches_defaulted'] is True
    assert _audit(_stored(block))['status'] == 'ok'
    explicit = L.ordered_unit_inference(_iid(12, seed=32),
                                        n_batches=[4, 6])
    assert explicit['skipped_batch_counts'] == []
    assert explicit['sufficient_statistics']['n_batches_defaulted'] is False
    assert _audit(_stored(explicit))['status'] == 'ok'
