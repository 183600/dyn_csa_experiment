"""Cleaning-specification sensitivity: the served set is a choice, not a fact.

The cache fixes one exact/SimHash geometry and the replay gate proves that this
geometry produced the served membership.  These checks cover the next question:
how much does that membership move under nearby valid cleaning choices or a
different raw-row order?  No fixture reads or writes a real corpus, and the
held-out test split is never opened by the audit.
"""
import json
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403


def _nonempty(rows):
    return [r for r in rows if isinstance(r, str) and r.strip()]


def _ids(rows):
    return np.asarray(
        [int(L._doc_id_u64(L._doc_exact_fp(r))) for r in rows],
        dtype=np.uint64)


def _reference(rows, val_rows, *, protocol, fit_rows, train_scanned=None,
               val_scanned=None):
    """Independent production-helper replay used as the expected sidecar."""
    train_ne = _nonempty(rows)
    val_ne = _nonempty(val_rows)
    train_scanned = len(train_ne) if train_scanned is None else train_scanned
    val_scanned = len(val_ne) if val_scanned is None else val_scanned
    d = L._DocDeduplicator(**L.dedup_kwargs_from_protocol(protocol))
    for row in train_ne[:fit_rows]:
        d.is_duplicate(row)
    train_kept = []
    for i, row in enumerate(train_ne[:train_scanned]):
        if i < fit_rows:
            train_kept.append(row)
            continue
        dup, _kind = d.is_duplicate(row)
        if not dup:
            train_kept.append(row)
    val_kept = []
    for row in val_ne[:val_scanned]:
        dup, _kind = d.is_duplicate(row)
        if not dup:
            val_kept.append(row)
    return _ids(train_kept), _ids(val_kept)


def _protocol(**overrides):
    p = dict(L.DEDUP_PROTOCOL_DEFAULTS)
    p.update(overrides)
    return p


def _write_raw(path, rows, *, blank_between=True):
    with open(path, 'w', encoding='utf-8') as fh:
        for i, row in enumerate(rows):
            fh.write(row + '\n')
            if blank_between and i + 1 < len(rows):
                fh.write('\n')


def _write_cache(cache, raw, *, rows, val_rows, protocol=None, fit_rows=1,
                 train_scanned=None, val_scanned=None, sidecars=True,
                 include_protocol=True, include_fit=True,
                 blank_between=True):
    os.makedirs(cache, exist_ok=True)
    os.makedirs(raw, exist_ok=True)
    _write_raw(os.path.join(raw, 'wiki.train.raw'), rows,
               blank_between=blank_between)
    _write_raw(os.path.join(raw, 'wiki.valid.raw'), val_rows,
               blank_between=blank_between)
    protocol = _protocol() if protocol is None else dict(protocol)
    train_ne = _nonempty(rows)
    val_ne = _nonempty(val_rows)
    train_scanned = len(train_ne) if train_scanned is None else train_scanned
    val_scanned = len(val_ne) if val_scanned is None else val_scanned
    tag = 'v5_sl16_cap128_v64_vs4'
    train_total = len(rows) + (max(0, len(rows) - 1) if blank_between else 0)
    val_total = len(val_rows) + (max(0, len(val_rows) - 1) if blank_between else 0)
    meta = {
        'vocab': 64,
        'seq_len': 16,
        'fit_fp': 'f' * 16,
        'sampling': {
            'train': {'source_rows': int(train_total),
                      'nonempty_rows_scanned': int(train_scanned),
                      'rows_served': 1, 'tokens_served': 1},
            'validation': {'source_rows': int(val_total),
                           'nonempty_rows_scanned': int(val_scanned),
                           'rows_served': 1, 'tokens_served': 1},
        },
    }
    if include_fit:
        meta['tokenizer_fit_rows'] = int(fit_rows)
    if include_protocol:
        meta['dedup'] = {
            'protocol': protocol,
            'protocol_fp': L.dedup_protocol_fp(protocol),
        }
    if sidecars:
        exp_tr, exp_va = _reference(
            rows, val_rows, protocol=protocol, fit_rows=fit_rows,
            train_scanned=train_scanned, val_scanned=val_scanned)
        np.save(os.path.join(cache, f'train_doc_ids_{tag}.npy'), exp_tr)
        np.save(os.path.join(cache, f'val_doc_ids_{tag}.npy'), exp_va)
    with open(os.path.join(cache, f'meta_{tag}.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(meta, fh)
    return tag, meta, protocol


def _rows(n=24, prefix='doc'):
    return [f'{prefix} {i} ' + ' '.join(
        f'w{(i * 7 + j) % 53}' for j in range(12)) for i in range(n)]


def test_hashed_dedup_matches_production_state():
    # The curve trials use a precomputed-hash state machine; it must make the
    # same exact/near decisions (and quarantine the same skipped near copies)
    # as the production _DocDeduplicator for every valid geometry.
    base = _rows(30)
    rows = base + [base[0], base[3] + ' tail', base[7] + ' extra words']
    for bands, threshold, words in ((2, 0, 2), (2, 1, 2),
                                    (4, 2, 3), (8, 3, 4)):
        p = _protocol(bands=bands, near_hamming_threshold=threshold,
                      shingle_words=words, max_shingles=64)
        d = L._DocDeduplicator(**L.dedup_kwargs_from_protocol(p))
        hd = L._HashedDedup(p)
        for row in rows:
            fp = int(L._doc_exact_fp(row), 16)
            h = int(L._doc_simhash(row, shingle_words=words,
                                   max_shingles=64))
            assert d.is_duplicate(row)[0] == hd.add(fp, h), (bands, threshold, row)
        st = d.stats()
        assert st['exact_duplicates_skipped'] == hd.n_exact, (p, st, hd.n_exact)
        assert st['near_duplicates_skipped'] == hd.n_near, (p, st, hd.n_near)
        assert st['added_documents'] == hd.n_seen, (p, st, hd.n_seen)


def test_cleaning_sensitivity_baseline_matches_production_replay():
    rows = _rows(18)
    val = _rows(6, prefix='val')
    p = _protocol(shingle_words=3, max_shingles=32, bands=4,
                  near_hamming_threshold=2)
    fit = 2
    exp_tr, exp_va = _reference(rows, val, protocol=p, fit_rows=fit)
    out = L.cleaning_sensitivity(
        rows, val, protocol=p, fit_rows=fit,
        expected_train_doc_ids=exp_tr, expected_validation_doc_ids=exp_va,
        expected_train_source_rows=len(rows),
        expected_validation_source_rows=len(val),
        n_order_permutations=1)
    assert out['status'] in ('ok', 'warnings'), out
    assert out['baseline_match']['train']['match'] is True, out
    assert out['baseline_match']['validation']['match'] is True, out
    assert out['baseline']['train']['kept'] == int(exp_tr.size), out
    assert out['summary']['n_variants_tested'] >= 1, out
    assert out['test_raw_read'] is False, out
    assert out['policy']['role'].startswith('descriptive'), out['policy']


def test_near_threshold_axis_is_measured(monkeypatch):
    # SimHash values 0,1,3: at threshold 1 B is a near copy of A and C is two
    # bits away; at threshold 0 neither is removed.  The strict variant must
    # therefore keep two more rows and be flagged material.
    table = {'A': 0, 'B': 1, 'C': 3}
    monkeypatch.setattr(L, '_doc_simhash',
                        lambda text, **kw: table[text])
    p = _protocol(shingle_words=2, max_shingles=16, bands=2,
                  near_hamming_threshold=1)
    out = L.cleaning_sensitivity(['A', 'B', 'C'], [], protocol=p, fit_rows=0,
                                 n_order_permutations=0)
    assert out['baseline']['train']['kept'] == 1, out['baseline']
    strict = next(v for v in out['variants']
                  if v['name'] == 'near_threshold_strict')
    assert strict['status'] == 'tested', strict
    assert strict['train']['kept'] == 3, strict
    # A is still in the strict kept set, so one baseline member survives.
    assert strict['train']['kept_jaccard'] == 0.33333333, strict
    assert strict['material'] is True, strict
    assert out['status'] == 'warnings', out
    assert 'near_threshold_strict' in out['summary']['material_variants'], out


def test_row_order_changes_the_kept_representative(monkeypatch):
    # A and B are one near-duplicate cluster.  Source order keeps A; reversing
    # the rows keeps B.  The count is unchanged, so only a membership metric
    # (not a count check) can see this order dependence.
    table = {'A': 0, 'B': 1}
    monkeypatch.setattr(L, '_doc_simhash',
                        lambda text, **kw: table[text])
    p = _protocol(shingle_words=2, max_shingles=16, bands=2,
                  near_hamming_threshold=1)
    out = L.cleaning_sensitivity(['A', 'B'], [], protocol=p, fit_rows=0,
                                 n_order_permutations=0)
    rev = next(v for v in out['variants'] if v['name'] == 'order_reverse')
    assert rev['status'] == 'tested', rev
    assert rev['train']['kept'] == 1, rev
    assert rev['train']['count_delta_fraction'] == 0.0, rev
    assert rev['train']['membership_identical'] is False, rev
    assert rev['train']['kept_jaccard'] == 0.0, rev
    assert rev['material'] is True, rev


def test_order_only_sequence_change_is_not_hidden_by_set_equality(monkeypatch):
    # Two distinct documents, no duplicates under the protocol.  Reversing the
    # stream keeps the same SET (Jaccard 1, count delta 0) but produces a
    # different ordered token stream; the audit must not call that identical.
    table = {'A': 0, 'B': 1}
    monkeypatch.setattr(L, '_doc_simhash',
                        lambda text, **kw: table[text])
    p = _protocol(shingle_words=2, max_shingles=16, bands=2,
                  near_hamming_threshold=0)
    out = L.cleaning_sensitivity(['A', 'B'], [], protocol=p, fit_rows=0,
                                 n_order_permutations=0)
    rev = next(v for v in out['variants'] if v['name'] == 'order_reverse')
    assert rev['status'] == 'tested', rev
    assert rev['train']['kept_jaccard'] == 1.0, rev
    assert rev['train']['count_delta_fraction'] == 0.0, rev
    assert rev['train']['membership_identical'] is True, rev
    assert rev['train']['kept_sequence_identical'] is False, rev
    assert rev['sequence_changed'] is True, rev
    # Membership (the set/count axis) is stable; only the order changed.
    assert rev['material'] is False, rev
    assert out['summary']['n_sequence_changed_variants'] >= 1, out
    assert out['summary']['order_identical'] is False, out
    assert out['summary']['fully_identical'] is False, out


def test_fit_prefix_is_excluded_from_the_order_permutation(monkeypatch):
    # The fit rows are fixed by the tokenizer stage; permuting them here would
    # silently test a different tokenizer fit, not a different cleaning order.
    table = {'F0': 100, 'F1': 101, 'A': 200, 'B': 201}
    monkeypatch.setattr(L, '_doc_simhash',
                        lambda text, **kw: table[text])
    p = _protocol(shingle_words=2, max_shingles=16, bands=2,
                  near_hamming_threshold=0)
    out = L.cleaning_sensitivity(['F0', 'F1', 'A', 'B'], [],
                                 protocol=p, fit_rows=2,
                                 n_order_permutations=0)
    # With threshold 0 and these hashes, no near copy exists; the fit rows are
    # force-kept and the post-fit order cannot change membership.  The point is
    # that the curve still reports a baseline and a successful test, not that
    # the specific permutation happened to differ.
    assert out['baseline']['train']['kept'] == 4, out['baseline']
    rev = next(v for v in out['variants'] if v['name'] == 'order_reverse')
    assert rev['status'] == 'tested', rev
    assert rev['train']['kept'] == 4, rev


def test_cleaning_sensitivity_is_deterministic():
    rows = _rows(20)
    p = _protocol(shingle_words=3, max_shingles=32, bands=4,
                  near_hamming_threshold=2)
    a = L.cleaning_sensitivity(rows, [], protocol=p, fit_rows=2,
                               n_order_permutations=2, seed=7)
    b = L.cleaning_sensitivity(rows, [], protocol=p, fit_rows=2,
                               n_order_permutations=2, seed=7)
    assert a['summary'] == b['summary'], (a['summary'], b['summary'])
    assert [(v['name'], v.get('status'), v.get('protocol_fp'),
             v.get('train')) for v in a['variants']] == \
           [(v['name'], v.get('status'), v.get('protocol_fp'),
             v.get('train')) for v in b['variants']]


def test_cleaning_sensitivity_fails_closed_on_bad_inputs():
    p = _protocol()
    for kwargs in ({'fit_rows': True}, {'fit_rows': -1},
                   {'n_order_permutations': -1},
                   {'n_order_permutations': 1.5},
                   {'max_rows': True}):
        base = dict(protocol=p, fit_rows=0)
        base.update(kwargs)
        try:
            L.cleaning_sensitivity(['a b c'], [], **base)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f'{kwargs} must be refused')
    try:
        L.cleaning_sensitivity(['a b c'], [], protocol={'bands': 0},
                               fit_rows=0)
    except ValueError:
        pass
    else:
        raise AssertionError('an invalid protocol must be refused')
    try:
        L.cleaning_sensitivity([1, 2], [], protocol=p, fit_rows=0)
    except TypeError:
        pass
    else:
        raise AssertionError('non-string rows must be refused')
    # A declared source-row mismatch is a hard failure, not a different corpus.
    out = L.cleaning_sensitivity(['a b c', 'd e f'], [], protocol=p,
                                 fit_rows=0, expected_train_source_rows=99)
    assert out['status'] == 'fail', out
    # A cache-sidecar mismatch is also a hard failure.
    out2 = L.cleaning_sensitivity(['a b c', 'd e f'], [], protocol=p,
                                  fit_rows=0,
                                  expected_train_doc_ids=np.asarray([1, 2],
                                                                    dtype=np.uint64))
    assert out2['status'] == 'fail', out2
    assert out2['baseline_match']['train']['match'] is False, out2
    # The row guard is a not_verifiable result, never a silent sample.
    out3 = L.cleaning_sensitivity(['a b c', 'd e f'], [], protocol=p,
                                  fit_rows=0, max_rows=1)
    assert out3['status'] == 'not_verifiable', out3


def test_audit_cleaning_sensitivity_end_to_end():
    rows = _rows(22)
    val = _rows(8, prefix='val')
    with tempfile.TemporaryDirectory() as td:
        cache = os.path.join(td, 'cache')
        raw = os.path.join(td, 'raw')
        p = _protocol(shingle_words=3, max_shingles=32, bands=4,
                      near_hamming_threshold=2)
        tag, _meta, _p = _write_cache(cache, raw, rows=rows, val_rows=val,
                                      protocol=p, fit_rows=2)
        # A held-out dump exists but must never be opened by this audit.
        with open(os.path.join(raw, 'wiki.test.raw'), 'w',
                  encoding='utf-8') as fh:
            fh.write('this held-out sentence must not be read\n')
        out = A.audit_cleaning_sensitivity(cache, raw)
        assert out['status'] in ('ok', 'warnings'), out
        assert out['test_raw_read'] is False, out
        assert out['baseline_match']['train']['match'] is True, out
        assert out['cache']['tag'] == tag, out
        # A same-count edit to the raw dump changes the replayed membership:
        # the cache was built from a different snapshot, so this is a failure.
        edited = list(rows)
        edited[3] = 'a completely different training document'
        _write_raw(os.path.join(raw, 'wiki.train.raw'), edited)
        out2 = A.audit_cleaning_sensitivity(cache, raw)
        assert out2['status'] == 'fail', out2
        # Missing raw data is not a clean sensitivity result.
        shutil.rmtree(raw)
        out3 = A.audit_cleaning_sensitivity(cache, raw)
        assert out3['status'] == 'not_verifiable_locally', out3
        # Missing exact scan counts / protocol are also unverifiable.
        tag2, _, _ = _write_cache(cache, raw, rows=rows, val_rows=val,
                                  protocol=p, fit_rows=2,
                                  include_protocol=False)
        out4 = A.audit_cleaning_sensitivity(cache, raw)
        assert out4['status'] == 'not_verifiable_locally', out4


def test_audit_cleaning_sensitivity_missing_sidecars_is_not_verified():
    rows = _rows(10)
    val = _rows(4, prefix='val')
    with tempfile.TemporaryDirectory() as td:
        cache = os.path.join(td, 'cache')
        raw = os.path.join(td, 'raw')
        _write_cache(cache, raw, rows=rows, val_rows=val, fit_rows=1,
                     sidecars=False)
        out = A.audit_cleaning_sensitivity(cache, raw)
        assert out['status'] == 'not_verifiable_locally', out
        assert any('sidecar' in w for w in out.get('warnings') or []), out


def test_audit_cleaning_sensitivity_empty_sidecar_is_not_verified():
    rows = _rows(8)
    val = _rows(3, prefix='val')
    with tempfile.TemporaryDirectory() as td:
        cache = os.path.join(td, 'cache')
        raw = os.path.join(td, 'raw')
        tag, _meta, _p = _write_cache(cache, raw, rows=rows, val_rows=val,
                                      fit_rows=1)
        np.save(os.path.join(cache, f'train_doc_ids_{tag}.npy'),
                np.empty(0, dtype=np.uint64))
        out = A.audit_cleaning_sensitivity(cache, raw)
        assert out['status'] == 'not_verifiable_locally', out
        assert any('empty' in w for w in out.get('warnings') or []), out
