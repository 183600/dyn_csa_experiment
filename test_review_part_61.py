"""CPU checks for first-occurrence unit-order attestation.

The evaluation aggregator stores per-unit vectors in the canonical sorted-id
layout (``np.unique``).  Corpus order is a separate measurement; the latest
order-dependence readouts must apply that permutation or refuse.  This suite
pins the attestation primitive, the aggregation metadata, the end-to-end
document analysis, and the audit that prevents a sorted vector from being
silently analyzed as corpus order.  It reads no corpus and no test split.
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


def _doc_cell(nll, tokens, ids, *, unit_order=None, ppl=None, n_tok=None):
    nll = np.asarray(nll, dtype=float)
    provided_ppl = ppl is not None
    ppl = ([math.exp(float(v) / int(t)) for v, t in zip(nll, tokens)]
           if not provided_ppl else list(ppl))
    return {
        'ppl': (math.exp(float(nll.sum()) / int(sum(tokens)))
                if not provided_ppl else float(ppl[0])),
        'nll_sum': float(nll.sum()),
        'n_tok': int(sum(tokens) if n_tok is None else n_tok),
        'ppl_per_doc': ppl,
        'nll_per_doc': [float(v) for v in nll],
        'n_tokens_per_doc': list(tokens),
        'doc_ids': list(ids),
        'unit_order': (dict(unit_order) if unit_order is not None else None),
        'n_seq': len(tokens),
    }


def _panel(n=24, seeds=(0, 1, 2), slope=0.2):
    rng = np.random.default_rng(20261031)
    tokens = rng.integers(120, 360, size=n).tolist()
    ids = [f'd{i:03d}' for i in range(n)]
    # A monotone unit effect: reversing the order flips lag-1 correlation, so
    # an analysis on sorted ids cannot masquerade as the corpus-order one.
    effect = np.asarray([slope * i for i in range(n)], dtype=float)
    effect = effect + rng.normal(0.0, 0.005, size=n)
    recs_a, recs_b = {}, {}
    for s in seeds:
        base = rng.normal(80.0, 3.0, size=n)
        nll_a = (base + 0.5 * effect) * np.asarray(tokens)
        nll_b = (base - 0.5 * effect) * np.asarray(tokens)
        recs_a[s] = {'by_len': {512: _doc_cell(nll_a, tokens, ids)}}
        recs_b[s] = {'by_len': {512: _doc_cell(nll_b, tokens, ids)}}
    return recs_a, recs_b, tokens, ids


def _attestation(ids, tokens, perm, *, unit_kind='document', n_tokens=None):
    ordered = [ids[i] for i in perm]
    positions = np.cumsum([0] + [tokens[i] for i in perm][:-1]).astype(int)
    return L.unit_order_attestation(
        ordered, unit_kind=unit_kind, permutation=list(perm),
        first_positions=[int(x) for x in positions],
        n_tokens=(int(sum(tokens)) if n_tokens is None else int(n_tokens)))


def _attach(recs, attestation):
    for rec in recs.values():
        for cell in rec['by_len'].values():
            cell['unit_order'] = dict(attestation)
    return recs


def _write(path, obj):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh)


def _audit(obj):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'artifact.json')
        _write(path, obj)
        with contextlib.redirect_stdout(io.StringIO()):
            return A.audit_unit_order(path)


def test_policy_digest_and_attestation_primitives():
    assert L.UNIT_ORDER_POLICY['version'] == 'unit-order-1'
    assert L.UNIT_ORDER_POLICY['schema'] == 'unit_order_v1'
    assert L.UNIT_ORDER_POLICY['basis'] == (
        'first_occurrence_in_scored_targets')
    assert 'attestation' in L.ORDER_DEPENDENCE_POLICY['boundary']
    assert 'attestation' in L.POOLED_ORDER_POLICY['boundary']
    # Length prefixing: concatenation alone would collide.
    h1 = L.unit_order_sha256(['a', 'bc'])
    h2 = L.unit_order_sha256(['ab', 'c'])
    assert h1 != h2 and len(h1) == 64
    att = L.unit_order_attestation(
        ['u2', 'u0', 'u1'], unit_kind='document',
        permutation=[1, 2, 0], first_positions=[0, 4, 9], n_tokens=20)
    assert att['sha256'] == L.unit_order_sha256(['u2', 'u0', 'u1'])
    assert L.unit_order_problems(
        att, ['u2', 'u0', 'u1'], unit_kind='document') == []
    # A sorted/other sequence is not this attestation.
    assert L.unit_order_problems(att, ['u0', 'u1', 'u2'])
    # Malformed primitives fail closed.
    for bad in ([], ['x', 'x'], [True], ['']):
        with _pytest.raises(ValueError):
            L.unit_order_sha256(bad)
    with _pytest.raises(ValueError):
        L.unit_order_attestation(['a'], unit_kind='row')
    with _pytest.raises(ValueError):
        L.unit_order_attestation(['a', 'b'], unit_kind='document',
                                 permutation=[0, 0])
    with _pytest.raises(ValueError):
        L.unit_order_attestation(['a', 'b'], unit_kind='document',
                                 first_positions=[2, 2])


def test_aggregate_records_first_occurrence_not_sorted_ids():
    ids = np.asarray([[30, 30, 10, 10, 20, 20]], dtype=np.uint64)
    nll = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    doc = L.aggregate_nll_by_document(nll, ids)
    # Backward-compatible canonical layout.
    assert doc['doc_ids'] == ['000000000000000a', '0000000000000014',
                              '000000000000001e']
    uo = doc['unit_order']
    assert uo['unit_kind'] == 'document'
    assert uo['permutation'] == [2, 0, 1]
    assert uo['first_positions'] == [0, 2, 4]
    ordered = [doc['doc_ids'][i] for i in uo['permutation']]
    assert ordered == ['000000000000001e', '000000000000000a',
                       '0000000000000014']
    assert uo['sha256'] == L.unit_order_sha256(ordered)
    assert uo['n_tokens'] == 6
    assert L.unit_order_problems(uo, ordered, unit_kind='document') == []
    art = L.aggregate_nll_by_article(nll, ids)
    assert art['unit_order']['unit_kind'] == 'article'
    assert art['unit_order']['sha256'] == uo['sha256']


def test_aggregate_shorter_prefix_recomputes_slice_order():
    full = np.asarray([[3, 3, 1, 1, 2, 2, 4, 4]], dtype=np.uint64)
    nll = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    keys, inv = L.document_index(full)
    short = L.aggregate_nll_by_document(
        nll[:, :6], full[:, :6], document_index_map=(keys, inv[:, :6]))
    # The returned vectors stay sorted; first occurrence within the slice is
    # 3, 1, 2, not the sorted 1, 2, 3.
    assert short['doc_ids'] == ['0000000000000001', '0000000000000002',
                                '0000000000000003']
    assert short['unit_order']['permutation'] == [2, 0, 1]
    assert short['unit_order']['first_positions'] == [0, 2, 4]
    assert short['unit_order']['n_tokens'] == 6


def test_order_functions_verify_or_refuse_the_attestation():
    n, m = 32, 3
    rng = np.random.default_rng(7)
    t = rng.integers(100, 400, size=n).tolist()
    a = rng.normal(50.0, 2.0, size=(m, n)) * np.asarray(t)
    b = a + 0.02 * np.asarray(t)
    ids_sorted = [f'u{i}' for i in range(n)]
    perm = list(range(n - 1, -1, -1))
    ordered = [ids_sorted[i] for i in perm]
    att = _attestation(ids_sorted, t, perm)
    values = rng.normal(0.0, 1.0, size=n)
    ok = L.ordered_unit_inference(values, unit_ids=ordered,
                                  order_attestation=att)
    assert ok['unit_order']['status'] == 'verified'
    assert ok['unit_order']['sha256'] == att['sha256']
    assert ok['sufficient_statistics']['unit_order_attestation'] == att
    unverified = L.ordered_unit_inference(values, unit_ids=ordered)
    assert unverified['unit_order']['status'] == 'unverified'
    pooled = L.pooled_order_inference(a, b, t, unit_ids=ordered,
                                      order_attestation=att)
    assert pooled['unit_order']['status'] == 'verified'
    # A sorted vector with the reversed-order attestation is refused.
    with _pytest.raises(ValueError):
        L.ordered_unit_inference(values, unit_ids=ids_sorted,
                                 order_attestation=att)
    with _pytest.raises(ValueError):
        L.pooled_order_inference(a, b, t, unit_ids=ids_sorted,
                                 order_attestation=att)
    with _pytest.raises(ValueError):
        L.ordered_unit_inference(
            values, unit_ids=ordered,
            order_attestation=dict(att, basis='sorted_id'))


def test_doc_level_analysis_uses_the_attested_permutation():
    recs_a, recs_b, tokens, ids = _panel()
    perm = list(range(len(ids) - 1, -1, -1))
    att = _attestation(ids, tokens, perm)
    _attach(recs_a, att)
    _attach(recs_b, att)
    out = V12._doc_level_analysis(recs_a, recs_b, [0, 1, 2], 512)
    assert out['unit_order']['status'] == 'ok'
    assert out['unit_order']['sha256'] == att['sha256']
    assert out['unit_order']['permutation_verified'] is True
    oi = out['order_inference']
    assert oi['status'] == 'ok'
    assert oi['unit_order']['status'] == 'verified'
    # Recompute the equal-unit seed-mean vector and its reversed order.
    vals = [float(np.mean([recs_a[s]['by_len'][512]['ppl_per_doc'][i]
                           - recs_b[s]['by_len'][512]['ppl_per_doc'][i]
                           for s in (0, 1, 2)]))
            for i in range(len(ids))]
    ordered_vals = [vals[i] for i in perm]
    ordered_ids = [ids[i] for i in perm]
    ref = L.ordered_unit_inference(ordered_vals, unit_ids=ordered_ids,
                                   order_attestation=att)
    for key in ('iid_ci', 'robust_ci', 'design_effect_max',
                'lag1_autocorr', 'batch_means', 'block_bootstrap'):
        assert oi[key] == ref[key], key
    # The readout is bound to the reversed corpus order, not the sorted-id
    # sequence that is stored as the canonical vector layout.
    assert oi['unit_order']['sha256'] != L.unit_order_sha256(ids)
    assert oi['unit_order']['sha256'] == L.unit_order_sha256(ordered_ids)
    sorted_ref = L.ordered_unit_inference(vals, unit_ids=ids)
    assert oi['batch_means'] != sorted_ref['batch_means']
    assert oi['design_effect_max'] != sorted_ref['design_effect_max']
    assert out['pooled_order_inference']['unit_order']['status'] == 'verified'


def test_eval_length_gen_attaches_first_occurrence_order():
    import copy
    torch.manual_seed(3)
    dev = torch.device('cpu')
    cfgs = L.make_layer_cfgs(2, 'full_rope')
    model = L.SmallGPT(256, 64, 2, 4, 16, 64,
                       copy.deepcopy(cfgs)).to(dev)
    model.eval()
    rng = np.random.default_rng(11)
    ids = np.asarray(rng.integers(0, 256, size=(2, 33)), dtype=np.int64)
    doc = np.empty_like(ids, dtype=np.uint64)
    doc[:, :12] = 30
    doc[:, 12:24] = 10
    doc[:, 24:] = 20
    try:
        cell = V7.eval_length_gen(model, ids, [32], dev, max_pos=None,
                                  doc_ids=doc)[32]
    finally:
        del model
    assert cell['doc_granularity'] == 'document'
    uo = cell['unit_order']
    assert uo['unit_kind'] == 'document'
    assert uo['permutation'] == [2, 0, 1]
    assert uo['first_positions'] == [0, 11, 23]
    ordered = [cell['doc_ids'][i] for i in uo['permutation']]
    assert ordered == ['000000000000001e', '000000000000000a',
                       '0000000000000014']
    assert L.unit_order_problems(uo, ordered, unit_kind='document') == []


def test_doc_level_analysis_withholds_legacy_and_tampered_order():
    recs_a, recs_b, tokens, ids = _panel(n=16)
    legacy = V12._doc_level_analysis(recs_a, recs_b, [0, 1, 2], 512)
    assert legacy['unit_order']['status'] == 'not_verifiable_locally'
    assert 'order_inference' not in legacy
    assert 'order_inference_unavailable_reason' in legacy
    assert 'pooled_order_inference' not in legacy
    perm = list(range(len(ids) - 1, -1, -1))
    att = _attestation(ids, tokens, perm)
    bad = copy.deepcopy(att)
    bad['sha256'] = '0' * 64
    _attach(recs_a, bad)
    _attach(recs_b, bad)
    out = V12._doc_level_analysis(recs_a, recs_b, [0, 1, 2], 512)
    assert out['unit_order']['status'] == 'fail'
    assert out['unit_order']['problems']
    assert 'order_inference' not in out


def test_audit_records_clean_legacy_and_mixed():
    ids = np.asarray([[30, 30, 10, 10, 20, 20]], dtype=np.uint64)
    nll = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    cell = L.aggregate_nll_by_document(nll, ids)
    cell.update({'ppl': 100.0, 'nll_sum': float(nll.sum()), 'n_tok': 6,
                 'n_seq': 1})
    clean = _audit({'runs': [{'variant': 'a', 'seed': 0,
                              'by_len': {'512': cell}}]})
    assert clean['status'] == 'ok'
    assert clean['n_ok'] == 1 and not clean['problems']
    legacy_cell = {k: v for k, v in cell.items() if k != 'unit_order'}
    legacy = _audit({'runs': [{'variant': 'a', 'seed': 0,
                               'by_len': {'512': legacy_cell}}]})
    assert legacy['status'] == 'not_verifiable_locally'
    assert legacy['n_legacy'] == 1
    mixed = _audit({'runs': [
        {'variant': 'a', 'seed': 0, 'by_len': {'512': cell}},
        {'variant': 'a', 'seed': 1, 'by_len': {'512': legacy_cell}}]})
    assert mixed['status'] == 'fail'
    assert mixed['n_ok'] == 1 and mixed['n_legacy'] == 1


def test_audit_records_catch_each_tamper():
    ids = np.asarray([[30, 30, 10, 10, 20, 20]], dtype=np.uint64)
    nll = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    cell = L.aggregate_nll_by_document(nll, ids)
    cell.update({'ppl': 100.0, 'nll_sum': float(nll.sum()), 'n_tok': 6,
                 'n_seq': 1})

    def _run(mutate):
        bad = copy.deepcopy(cell)
        mutate(bad)
        return _audit({'runs': [{'variant': 'a', 'seed': 0,
                                 'by_len': {'512': bad}}]})
    cases = [
        ('sha256', lambda c: c['unit_order'].__setitem__('sha256', '0' * 64)),
        ('permutation', lambda c: c['unit_order'].__setitem__(
            'permutation', [0, 0, 1])),
        ('first_positions', lambda c: c['unit_order'].__setitem__(
            'first_positions', [0, 0, 4])),
        ('n_tokens', lambda c: c['unit_order'].__setitem__('n_tokens', 999)),
        ('unit_kind', lambda c: c['unit_order'].__setitem__(
            'unit_kind', 'article')),
        ('basis', lambda c: c['unit_order'].__setitem__('basis', 'sorted_id')),
        ('n_units', lambda c: c['unit_order'].__setitem__('n_units', 4)),
    ]
    for label, mutate in cases:
        out = _run(mutate)
        assert out['status'] == 'fail', label
        assert out['problems'], label


def _analysis_artifact():
    recs_a, recs_b, tokens, ids = _panel(n=20)
    perm = list(range(len(ids) - 1, -1, -1))
    att = _attestation(ids, tokens, perm)
    _attach(recs_a, att)
    _attach(recs_b, att)
    out = V12._doc_level_analysis(recs_a, recs_b, [0, 1, 2], 512)
    return {'finaltest': {'comparisons': {'c1': {'doc_sampling': out}}}}


def test_audit_stats_clean_and_unverified_readout():
    art = _analysis_artifact()
    clean = _audit(art)
    assert clean['status'] == 'ok', clean
    assert clean['n_ok'] == 1 and not clean['problems']
    bad = copy.deepcopy(art)
    bad['finaltest']['comparisons']['c1']['doc_sampling']['order_inference'][
        'unit_order']['sha256'] = '0' * 64
    out = _audit(bad)
    assert out['status'] == 'fail'
    assert any('disagrees' in p or 'verified' in p for p in out['problems'])
    # A readout without the cell attestation is not legacy: it must fail.
    bad2 = copy.deepcopy(art)
    bad2['finaltest']['comparisons']['c1']['doc_sampling'].pop('unit_order')
    out2 = _audit(bad2)
    assert out2['status'] == 'fail'
    assert any('without' in p for p in out2['problems'])
    # Legacy (no readout, no attestation) is not_verifiable.
    legacy = {'finaltest': {'comparisons': {'c1': {'doc_sampling': {
        'pooled_delta_ppl_ci95': [0.0, 1.0]}}}}}
    out3 = _audit(legacy)
    assert out3['status'] == 'not_verifiable_locally'
    # An explicit fail status is a failure even when no readout was produced.
    failed = copy.deepcopy(art)
    dsm = failed['finaltest']['comparisons']['c1']['doc_sampling']
    dsm.pop('order_inference', None)
    dsm.pop('pooled_order_inference', None)
    dsm['unit_order'] = {'status': 'fail', 'problems': ['tampered']}
    out4 = _audit(failed)
    assert out4['status'] == 'fail'


def test_audit_allows_row_level_panel_order_readout():
    n_rows, seeds = 12, [0, 1, 2]
    recs_a, recs_b = {}, {}
    for s in seeds:
        pa = [50.0 + 0.1 * math.sin(i / 2.0) + 0.01 * s
              for i in range(n_rows)]
        pb = [p + (0.5 if i < 6 else -0.2) for i, p in enumerate(pa)]
        recs_a[s] = {'by_len': {512: {
            'ppl_per_seq': pa, 'ppl': float(np.mean(pa)),
            'n_tok': n_rows * 511}}}
        recs_b[s] = {'by_len': {512: {
            'ppl_per_seq': pb, 'ppl': float(np.mean(pb)),
            'n_tok': n_rows * 511}}}
    out = V12._doc_level_analysis(recs_a, recs_b, seeds, 512)
    assert out['unit_order']['status'] == 'not_applicable'
    assert out['order_inference']['status'] == 'ok'
    assert out['order_inference']['unit_order']['status'] == 'unverified'
    art = {'finaltest': {'comparisons': {'row': {'doc_sampling': out}}}}
    checked = _audit(art)
    assert checked['status'] == 'warnings', checked
    assert not checked['problems']


def test_audit_directory_and_no_test_split(monkeypatch):
    cell = L.aggregate_nll_by_document(
        torch.arange(6, dtype=torch.float32).reshape(1, 6),
        np.asarray([[30, 30, 10, 10, 20, 20]], dtype=np.uint64))
    cell.update({'ppl': 100.0, 'nll_sum': 15.0, 'n_tok': 6, 'n_seq': 1})

    def _boom(*_a, **_k):
        raise AssertionError('unit-order audit must not read the test split')

    monkeypatch.setattr(L, 'load_wikitest', _boom)
    with tempfile.TemporaryDirectory() as td:
        _write(os.path.join(td, 'summary.json'),
               {'runs': [{'variant': 'a', 'seed': 0,
                          'by_len': {'512': cell}}]})
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_unit_order(td)
        assert out['status'] == 'ok', out


def test_report_markdown_flags_verified_and_unverified():
    art = _analysis_artifact()
    ft = art['finaltest']
    order_md = '\n'.join(V12._order_inference_markdown(ft))
    pooled_md = '\n'.join(V12._pooled_order_markdown(ft))
    assert 'first-occurrence' in order_md and 'first-occurrence' in pooled_md
    assert '未核验' not in order_md and '未核验' not in pooled_md
    legacy = {'comparisons': {'c1': {'doc_sampling': {
        'unit_order': {'status': 'not_verifiable_locally'},
        'pooled_delta_ppl_ci95': [0.0, 1.0]}}}}
    assert '未核验' in '\n'.join(V12._order_inference_markdown(legacy))
    assert '未核验' in '\n'.join(V12._pooled_order_markdown(legacy))


def test_cli_and_report_wiring_is_present():
    assert 'unit_order' in (A.__doc__ or '')
    src = inspect.getsource(A._run_cli)
    assert "'unit_order': audit_unit_order" in src
    assert "'order_attestation': audit_unit_order" in src
    report_src = inspect.getsource(A.run_report)
    assert "'unit_order': audit_unit_order()" in report_src


def test_audit_rejects_deleted_attestation_fields():
    # An optional low-level field is not optional in a stored cell: deleting
    # permutation/first_positions/n_tokens/policy_version must not silently
    # skip the corresponding check.
    ids = np.asarray([[30, 30, 10, 10, 20, 20]], dtype=np.uint64)
    cell = L.aggregate_nll_by_document(
        torch.arange(6, dtype=torch.float32).reshape(1, 6), ids)
    cell.update({'ppl': 100.0, 'nll_sum': 15.0, 'n_tok': 6, 'n_seq': 1})
    for key in ('policy_version', 'first_positions', 'n_tokens',
                'n_units', 'algorithm', 'permutation'):
        bad = copy.deepcopy(cell)
        del bad['unit_order'][key]
        out = _audit({'runs': [{'variant': 'a', 'seed': 0,
                                'by_len': {'512': bad}}]})
        assert out['status'] == 'fail', (key, out)
    for key in ('n_docs', 'n_tok', 'n_tokens_per_doc'):
        bad = copy.deepcopy(cell)
        del bad[key]
        out = _audit({'runs': [{'variant': 'a', 'seed': 0,
                                'by_len': {'512': bad}}]})
        assert out['status'] == 'fail', (key, out)


def test_audit_stats_replays_the_full_attestation_and_reports_only_verified():
    art = _analysis_artifact()
    for key in ('n_units', 'n_tokens', 'permutation', 'first_positions'):
        bad = copy.deepcopy(art)
        del bad['finaltest']['comparisons']['c1']['doc_sampling'][
            'unit_order'][key]
        out = _audit(bad)
        assert out['status'] == 'fail', (key, out)
    for key, value in (('policy_version', 'old'), ('sha256', '0' * 64),
                       ('algorithm', 'md5'), ('unit_kind', 'row')):
        bad = copy.deepcopy(art)
        for rk in ('order_inference', 'pooled_order_inference'):
            block = bad['finaltest']['comparisons']['c1']['doc_sampling'][rk]
            block['sufficient_statistics']['unit_order_attestation'][key] = value
        out = _audit(bad)
        assert out['status'] == 'fail', (key, out)
    # A readout that bypassed the attestation is not rendered as corpus order,
    # even when the comparison-level summary still says ok.
    bad = copy.deepcopy(art)
    dsm = bad['finaltest']['comparisons']['c1']['doc_sampling']
    for rk in ('order_inference', 'pooled_order_inference'):
        dsm[rk]['unit_order'] = {'status': 'unverified'}
    ft = bad['finaltest']
    order_md = '\n'.join(V12._order_inference_markdown(ft))
    pooled_md = '\n'.join(V12._pooled_order_markdown(ft))
    assert '未核验' in order_md and '\n|' not in order_md, order_md
    assert '未核验' in pooled_md and '\n|' not in pooled_md, pooled_md
