"""CPU checks for the token-pooled order-dependence / effective-n readout.

The companion token-pooled cluster interval treats evaluation units as
independent clusters.  The equal-unit order readout cannot be copied onto the
weighted estimator, so this suite pins the estimator-specific cluster
influence readout: the i.i.d. interval reproduces the existing cluster CI, a
serially dependent unit panel is flagged, an independent panel is not, the
audit replays and catches hand edits, the report/CLI are wired, and no test
split or result panel is read.
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
    return np.asarray(out)


def _panel(n=48, rho=None, seed=7):
    rng = np.random.default_rng(seed)
    tokens = rng.integers(200, 800, size=n)
    base = rng.normal(100.0, 5.0, size=(6, n))
    noise = rng.normal(4.0, 0.05, size=(6, n))
    delta = np.full(n, 0.03)
    if rho is not None:
        delta = delta + 0.15 * _ar(n, rho=rho, seed=seed + 1)
    a = base + tokens[None, :] * noise
    b = base + tokens[None, :] * (noise + delta[None, :])
    return a, b, tokens.tolist()


def _stored(block, *, name='c1'):
    comp = {'doc_sampling': {'pooled_order_inference': block,
                             'pooled_delta_ppl_ci95': [0.0, 1.0],
                             'unit': 'article'}}
    return {'finaltest': {'comparisons': {name: comp}}}


def _audit(obj):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'test_stats.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(obj, fh)
        with contextlib.redirect_stdout(io.StringIO()):
            return A.audit_pooled_order(path)


def _fixture_records(n=16, seeds=(0, 1, 2), seed=41):
    rng = np.random.default_rng(seed)
    tokens = rng.integers(150, 650, size=n).tolist()
    ids = [f'd{i}' for i in range(n)]
    # A nontrivial corpus order relative to the canonical sorted-id layout.
    perm = list(range(0, n, 2)) + list(range(1, n, 2))
    positions = np.cumsum(
        [0] + [tokens[i] for i in perm][:-1]).astype(int).tolist()
    order = L.unit_order_attestation(
        [ids[i] for i in perm], unit_kind='document', permutation=perm,
        first_positions=positions, n_tokens=int(sum(tokens)))
    recs_a, recs_b = {}, {}
    for s in seeds:
        base = rng.normal(70.0, 4.0, size=n)
        a = base + np.asarray(tokens) * rng.normal(4.0, 0.08, size=n)
        b = base + np.asarray(tokens) * (rng.normal(4.0, 0.08, size=n) + 0.02)
        for rec, nll in ((recs_a, a), (recs_b, b)):
            ppl = [math.exp(x / t) for x, t in zip(nll, tokens)]
            rec[s] = {'by_len': {512: {
                'ppl': math.exp(float(nll.sum()) / float(sum(tokens))),
                'nll_sum': float(nll.sum()), 'n_tok': int(sum(tokens)),
                'ppl_per_doc': ppl, 'nll_per_doc': nll.tolist(),
                'n_tokens_per_doc': tokens,
                'doc_ids': ids, 'unit_order': dict(order),
                'n_seq': n}}}
    return recs_a, recs_b, list(seeds)


def test_policy_status_and_boundary_are_declared():
    assert L.POOLED_ORDER_POLICY['version'] == 'pooled-order-dependence-1'
    assert L.POOLED_ORDER_STATUSES == ('ok', 'not_estimable')
    assert 'cannot' in L.POOLED_ORDER_POLICY['boundary']
    assert 'p-value' in L.POOLED_ORDER_POLICY['boundary']
    a, b, t = _panel(n=24)
    block = L.pooled_order_inference(a, b, t, n_boot=200, n_perm=200)
    assert block['kind'] == 'pooled_order_inference'
    assert block['status'] == 'ok'
    assert block['n_units'] == 24 and block['n_seeds'] == 6
    assert not ({'confirmed', 'p_holm', 'verified', 'verdict'} & set(block))
    stats = block['sufficient_statistics']
    assert len(stats['ppl_influence']) == 24
    assert len(stats['log_ratio_influence']) == 24
    assert abs(math.fsum(stats['ppl_influence'])) < 1e-12
    assert abs(math.fsum(stats['log_ratio_influence'])) < 1e-12
    small = L.pooled_order_inference(*_panel(n=7)[:2], _panel(n=7)[2],
                                     n_boot=100, n_perm=0)
    assert small['status'] == 'not_estimable'
    assert small['reason']
    a24, b24, t24 = _panel(n=24)
    one_seed = L.pooled_order_inference(a24[:1], b24[:1], t24,
                                       n_boot=100, n_perm=0)
    assert one_seed['status'] == 'not_estimable'
    assert 'seed' in one_seed['reason']


def test_iid_interval_matches_the_existing_cluster_ci():
    a, b, t = _panel(n=48)
    block = L.pooled_order_inference(a, b, t, n_boot=200, n_perm=0)
    pc = L._paired_nll_doc_ci(a, b, t)
    assert block['status'] == 'ok'
    for key, ci_key in (('ppl', 'delta_ppl_ci95'),
                        ('log_ratio', 'delta_log_ratio_ci95')):
        assert abs(block[key]['point'] - pc[
            'delta_ppl' if key == 'ppl' else 'delta_log_ratio']) < 1e-9
        assert abs(block[key]['iid_se'] - pc[
            'se_ppl' if key == 'ppl' else 'se_log_ratio']) < 1e-9
        for got, want in zip(block[key]['iid_ci'], pc[ci_key]):
            assert abs(got - want) < 1e-9


def test_ordered_panel_flags_design_effect_and_independent_does_not():
    a, b, t = _panel(n=64, rho=0.8)
    ordered = L.pooled_order_inference(a, b, t, n_boot=400, n_perm=400)
    for key in ('ppl', 'log_ratio'):
        ep = ordered[key]
        assert ep['order_dependence_detected'] is True
        assert ep['order_dependence_p'] <= 0.05
        assert ep['design_effect_max'] > 1.2
        assert ep['n_effective_min'] < ep['n_units'] if 'n_units' in ep else True
        assert ep['robust_ci_width'] >= ep['iid_ci_width']
        assert ep['inflation_detected'] is True
    assert ordered['inflation_detected'] is True
    assert 'does not sort' in ordered['notes'][0]
    independent = L.pooled_order_inference(*_panel(n=64, seed=9)[:2],
                                           _panel(n=64, seed=9)[2],
                                           n_boot=400, n_perm=400)
    assert independent['ppl']['order_dependence_detected'] is False
    assert independent['ppl']['inflation_detected'] is False
    assert 'not proof' in independent['ppl']['wording']


def test_batch_block_and_robust_interval_selection():
    a, b, t = _panel(n=48)
    block = L.pooled_order_inference(a, b, t, n_boot=200, n_perm=0)
    for key in ('ppl', 'log_ratio'):
        ep = block[key]
        assert ep['batch_means'] and ep['block_bootstrap']
        for row in ep['batch_means']:
            assert row['n_batches'] * row['batch_size'] == row['n_used']
            assert row['design_effect'] is not None
            assert row['n_effective'] is not None
        for row in ep['block_bootstrap']:
            assert row['block_length'] >= 2
            assert row['ci'] is not None
            assert row['design_effect'] is not None
        widths = [abs(ep['iid_ci'][1] - ep['iid_ci'][0])]
        widths += [abs(r['ci'][1] - r['ci'][0]) for r in ep['block_bootstrap']]
        assert abs(ep['robust_ci_width'] - max(widths)) < 1e-12
        assert ep['robust_ci_source'] in (
            ['iid_cluster_t']
            + [f'block_length_{r["block_length"]}'
               for r in ep['block_bootstrap']])


def test_replay_from_influence_is_bit_identical():
    a, b, t = _panel(n=32)
    block = L.pooled_order_inference(a, b, t, n_boot=200, n_perm=200)
    stats = block['sufficient_statistics']
    replay = L.pooled_order_inference_from_influence(
        stats['ppl_influence'], stats['log_ratio_influence'],
        delta_ppl=stats['delta_ppl'],
        delta_log_ratio=stats['delta_log_ratio'],
        unit_ids=stats['unit_ids'], seeds=stats['seeds'],
        n_seeds=stats['n_seeds'], block_lengths=stats['block_lengths'],
        n_batches=stats['n_batches'], n_boot=stats['n_boot'],
        n_perm=stats['n_perm'], conf=stats['conf'], alpha=stats['alpha'],
        seed=stats['seed'],
        design_effect_tolerance=stats['design_effect_tolerance'],
        material_ci_inflation=stats['material_ci_inflation'])
    trimmed_block = {k: v for k, v in block.items()
                     if k not in ('upstream', 'sufficient_statistics')}
    trimmed_replay = {k: v for k, v in replay.items()
                      if k != 'sufficient_statistics'}
    assert json.dumps(trimmed_replay, sort_keys=True) == json.dumps(
        trimmed_block, sort_keys=True)


def test_seeded_reproducibility_and_invalid_inputs_fail_closed():
    a, b, t = _panel(n=24)
    one = L.pooled_order_inference(a, b, t, n_boot=150, n_perm=150, seed=99)
    two = L.pooled_order_inference(a, b, t, n_boot=150, n_perm=150, seed=99)
    assert json.dumps(one, sort_keys=True) == json.dumps(two, sort_keys=True)
    with _pytest.raises(ValueError):
        L.pooled_order_inference(a, b[:-1], t)
    with _pytest.raises(ValueError):
        bad = b.copy()
        bad[0, 0] = float('nan')
        L.pooled_order_inference(a, bad, t)
    with _pytest.raises(ValueError):
        L.pooled_order_inference(a, b, [True] * len(t))
    with _pytest.raises(ValueError):
        L.pooled_order_inference(a, b, [0] + t[1:])
    with _pytest.raises(ValueError):
        L.pooled_order_inference(a, b, t, unit_ids='abc')
    with _pytest.raises(ValueError):
        L.pooled_order_inference(a, b, t, unit_ids=['x', 'x'] + [
            f'u{i}' for i in range(len(t) - 2)])
    with _pytest.raises(ValueError):
        L.pooled_order_inference(a, b, t, seeds=[0, 0, 1, 2, 3, 4])
    with _pytest.raises(ValueError):
        L.pooled_order_inference(a, b, t, block_lengths=[1])
    with _pytest.raises(ValueError):
        L.pooled_order_inference(a, b, t, block_lengths=[len(t)])
    with _pytest.raises(ValueError):
        L.pooled_order_inference(a, b, t, n_batches=[100])
    with _pytest.raises(ValueError):
        L.pooled_order_inference(a, b, t, n_boot=1)
    with _pytest.raises(ValueError):
        L.pooled_order_inference(a, b, t, n_perm=-1)
    with _pytest.raises(ValueError):
        L.pooled_order_inference(a, b, t, conf=1.0)
    with _pytest.raises(ValueError):
        L.pooled_order_inference(a, b, t, alpha=0.0)
    with _pytest.raises(ValueError):
        L.pooled_order_inference(a, b, t, seed=-1)
    with _pytest.raises(ValueError):
        L.pooled_order_inference(a, b, t, design_effect_tolerance=0.0)
    with _pytest.raises(ValueError):
        L.pooled_order_inference(a, b, t, material_ci_inflation=0.9)
    with _pytest.raises(ValueError):
        L.pooled_order_inference_from_influence(
            [0.1, 0.2], [0.1, 0.2], delta_ppl=0.0, delta_log_ratio=0.0)
    with _pytest.raises(ValueError):
        L.pooled_order_inference_from_influence(
            [1.0, -1.0], [1.0], delta_ppl=0.0, delta_log_ratio=0.0)


def test_extreme_panels_are_strict_json_and_never_infinite():
    a, b, t = _panel(n=24)
    alternating = a.copy()
    alternating[0, :] = b[0, :] + np.where(np.arange(len(t)) % 2, 1e308, -1e308)
    block = L.pooled_order_inference(alternating, b, t, n_boot=100, n_perm=0)
    json.dumps(block, allow_nan=False)
    for key in ('point_ppl', 'point_log_ratio'):
        assert block.get(key) is None or math.isfinite(float(block[key]))
    constant = a.copy()
    constant[:, :] = b
    flat = L.pooled_order_inference(constant, b, t, n_boot=100, n_perm=0)
    assert flat['status'] == 'not_estimable'
    json.dumps(flat, allow_nan=False)


def test_doc_level_analysis_stores_the_pooled_block():
    recs_a, recs_b, seeds = _fixture_records()
    out = V12._doc_level_analysis(recs_a, recs_b, seeds, 512)
    block = out['pooled_order_inference']
    assert block['status'] == 'ok'
    assert block['n_units'] == len(recs_a[0]['by_len'][512]['doc_ids'])
    assert block['n_seeds'] == len(seeds)
    assert block['point_ppl'] is not None
    assert block['ppl']['iid_ci'] is not None
    assert 'order_inference' in out
    assert 'pooled_delta_ppl_ci95' in out


def test_audit_replays_and_detects_tampering():
    a, b, t = _panel(n=32)
    block = L.pooled_order_inference(a, b, t, n_boot=200, n_perm=200)
    good = _stored(block)
    out = _audit(good)
    assert out['status'] == 'ok', out['problems']
    assert out['n_replayed'] == 1 and out['n_ok'] == 1

    for mutate in ('iid_ci', 'design_effect', 'robust_ci', 'n_units',
                   'influence', 'point_ppl', 'permutation_p'):
        obj = copy.deepcopy(good)
        target = obj['finaltest']['comparisons']['c1']['doc_sampling'][
            'pooled_order_inference']
        if mutate == 'iid_ci':
            target['ppl']['iid_ci'][0] += 1.0
        elif mutate == 'design_effect':
            target['ppl']['batch_means'][0]['design_effect'] *= 2.0
        elif mutate == 'robust_ci':
            target['ppl']['robust_ci'][1] += 1.0
        elif mutate == 'n_units':
            target['n_units'] += 1
        elif mutate == 'influence':
            target['sufficient_statistics']['ppl_influence'][0] += 0.01
        elif mutate == 'point_ppl':
            target['point_ppl'] += 0.1
        else:
            target['ppl']['order_dependence_p'] = 0.0001
        bad = _audit(obj)
        assert bad['status'] == 'fail', (mutate, bad)
        assert bad['problems'], mutate

    obj = copy.deepcopy(good)
    obj['finaltest']['comparisons']['c1']['doc_sampling'][
        'pooled_order_inference']['p_holm'] = 0.01
    bad = _audit(obj)
    assert bad['status'] == 'fail'
    assert any('confirmatory' in p for p in bad['problems'])

    obj = copy.deepcopy(good)
    obj['finaltest']['comparisons']['c1']['doc_sampling'][
        'pooled_order_inference']['status'] = 'not_estimable'
    na = _audit(obj)
    assert na['status'] == 'not_applicable'

    obj = copy.deepcopy(good)
    del obj['finaltest']['comparisons']['c1']['doc_sampling'][
        'pooled_order_inference']
    legacy = _audit(obj)
    assert legacy['status'] == 'not_verifiable_locally'
    assert legacy['warnings']

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'test_stats.json')
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write('{not json')
        with contextlib.redirect_stdout(io.StringIO()):
            broken = A.audit_pooled_order(path)
        assert broken['status'] == 'fail'
        assert any('not a readable JSON object' in p
                   for p in broken['problems'])

    with contextlib.redirect_stdout(io.StringIO()):
        missing = A.audit_pooled_order(
            os.path.join(REPO, 'does-not-exist.json'))
    assert missing['status'] == 'not_verifiable_locally'


def test_audit_never_reads_the_test_split(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError('pooled-order audit must not read test rows')
    monkeypatch.setattr(L, 'load_wikitest', _boom, raising=False)
    a, b, t = _panel(n=24)
    out = _audit(_stored(L.pooled_order_inference(
        a, b, t, n_boot=150, n_perm=150)))
    assert out['status'] == 'ok', out['problems']


def test_markdown_and_cli_wiring():
    a, b, t = _panel(n=32)
    block = L.pooled_order_inference(a, b, t, n_boot=200, n_perm=200)
    ft = {'comparisons': {'c1': {
        'doc_sampling': {'pooled_order_inference': block,
                         'pooled_delta_ppl_ci95': [0.1, 0.2],
                         'unit': 'article'}}}}
    lines = V12._pooled_order_markdown(ft)
    text = '\n'.join(lines)
    assert 'token 池化' in text
    assert 'n_eff' in text
    assert '不改变' in text
    assert '设计效应' in text
    assert V12._pooled_order_markdown({'comparisons': {}}) == []
    src = inspect.getsource(A._run_cli)
    for alias in ('pooled_order', 'pooled_dependence',
                  'pooled_design_effect', 'pooled_effective_n'):
        assert alias in src, alias
    assert 'pooled_order' in inspect.getsource(A.run_report)
    assert '_pooled_order_markdown' in inspect.getsource(V12.build_report)
