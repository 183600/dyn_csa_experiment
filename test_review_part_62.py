"""Cleaning-decision replay: counts/protocol are not served membership.

The cache metadata records *how* many documents the exact/SimHash filter
removed and *which geometry* it used.  These checks recompute the decision
sequence from the raw rows and compare the kept document ids with the cache
sidecars.  The point is that a cache can be internally self-consistent yet
serve a different set of documents; only a replay against the source settles
which membership the declared cleaning produces.

No test reads the real test split: every fixture is a tiny synthetic corpus.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403


A_BASE = ' '.join('word%d' % i for i in range(100))
A_NEAR = A_BASE + ' extra'
B_DOC = ' '.join('beta%d' % i for i in range(30))
C_DOC = ' '.join('gamma%d' % i for i in range(30))
T_DOC = ' '.join('theta%d' % i for i in range(30))


def _nonempty(rows):
    return [r for r in rows if isinstance(r, str) and r.strip()]


def _ids(rows):
    return np.asarray([int(L._doc_id_u64(L._doc_exact_fp(r))) for r in rows],
                      dtype=np.uint64)


def _build_reference(train_rows, val_rows, *, protocol=None, fit_rows=1,
                     train_scanned=None, val_scanned=None):
    """Independent test-side reference builder, following the loader order."""
    protocol = dict(L.DEDUP_PROTOCOL_DEFAULTS) if protocol is None else dict(protocol)
    kwargs = L.dedup_kwargs_from_protocol(protocol)
    train_ne = _nonempty(train_rows)
    val_ne = _nonempty(val_rows)
    if train_scanned is None:
        train_scanned = len(train_ne)
    if val_scanned is None:
        val_scanned = len(val_ne)
    d = L._DocDeduplicator(**kwargs)
    for row in train_ne[:fit_rows]:
        d.is_duplicate(row)
    seed = d.stats()
    train_kept = []
    for i, row in enumerate(train_ne[:train_scanned]):
        if i < fit_rows:
            train_kept.append(row)
            continue
        dup, _kind = d.is_duplicate(row)
        if not dup:
            train_kept.append(row)
    train_stats = d.stats()
    train_counts = {
        'exact': int(train_stats['exact_duplicates_skipped'])
        - int(seed['exact_duplicates_skipped']),
        'near': int(train_stats['near_duplicates_skipped'])
        - int(seed['near_duplicates_skipped']),
    }
    val_kept = []
    for row in val_ne[:val_scanned]:
        dup, _kind = d.is_duplicate(row)
        if not dup:
            val_kept.append(row)
    val_stats = d.stats()
    val_counts = {
        'exact': int(val_stats['exact_duplicates_skipped'])
        - int(train_stats['exact_duplicates_skipped']),
        'near': int(val_stats['near_duplicates_skipped'])
        - int(train_stats['near_duplicates_skipped']),
    }
    meta = {
        'vocab': 8192,
        'fit_fp': 'f' * 16,
        'seq_len': 64,
        'tokenizer_fit_rows': int(fit_rows),
        'sampling': {
            'schema': 'wikitext_slice_sampling_v1',
            'train': {
                'source_rows': len(train_rows),
                'nonempty_rows_scanned': int(train_scanned),
                'rows_served': len(train_kept),
                'tokens_served': 123,
                'duplicate_documents_skipped': train_counts,
            },
            'validation': {
                'source_rows': len(val_rows),
                'nonempty_rows_scanned': int(val_scanned),
                'rows_served': len(val_kept),
                'tokens_served': 50,
                'duplicate_documents_skipped': val_counts,
            },
        },
        'dedup': {
            'accounting_schema': 'wikitext_dedup_accounting_v1',
            'protocol': protocol,
            'protocol_fp': L.dedup_protocol_fp(protocol),
            'splits': {'train': train_counts, 'validation': val_counts},
        },
        'doc_ids': {
            'train_count': len(train_kept),
            'validation_count': len(val_kept),
        },
    }
    return meta, _ids(train_kept), _ids(val_kept)


def _build_test_reference(train_rows, val_rows, test_rows, *, protocol=None):
    protocol = dict(L.DEDUP_PROTOCOL_DEFAULTS) if protocol is None else dict(protocol)
    kwargs = L.dedup_kwargs_from_protocol(protocol)
    train_ne = _nonempty(train_rows)
    val_ne = _nonempty(val_rows)
    train_index = L._DocDeduplicator(**kwargs)
    for row in train_ne:
        train_index.is_duplicate(row)
    val_only = L._DocDeduplicator(**kwargs)
    indexed = 0
    for row in val_ne:
        dup, _kind = val_only.is_duplicate(row)
        if not dup:
            indexed += 1
    val_screen = L._DocDeduplicator(**kwargs)
    for row in train_ne:
        val_screen.is_duplicate(row)
    val_kept_vs_train = 0
    for row in val_ne:
        dup, _kind = val_screen.is_duplicate(row)
        if not dup:
            val_kept_vs_train += 1
    kept, excluded, _index = L._filter_test_documents(
        train_ne, val_only, list(test_rows), protocol=protocol, verbose=False)
    test_doc_ids = _ids(kept)  # every kept row is served in this tiny fixture
    meta = {
        'vocab': 8192,
        'fit_fp': 'f' * 16,
        'seq_len': 64,
        'sampling': {
            'schema': 'wikitext_slice_sampling_v1',
            'test': {
                'source_rows': len(test_rows),
                'rows_after_dedup': len(kept),
                'rows_served': len(test_doc_ids),
                'tokens_served': 40,
            },
        },
        'near_duplicate_cleanup': {
            'accounting_schema': 'wikitext_dedup_accounting_v1',
            'protocol': protocol,
            'protocol_fp': L.dedup_protocol_fp(protocol),
            'train_exact_removed': int(excluded['train_exact']),
            'train_near_removed': int(excluded['train_near']),
            'test_exact_removed': int(excluded['test_exact']),
            'test_near_removed': int(excluded['test_near']),
            'source_documents': int(len(list(test_rows))),
            'source_documents_nonempty': int(len(_nonempty(test_rows))),
            'kept_documents': int(len(kept)),
            'validation_cleanup': {
                'source_documents': int(len(val_ne)),
                'kept_documents': int(val_kept_vs_train),
                'indexed_documents': int(indexed),
                'duplicate_documents_removed': int(len(val_ne) - val_kept_vs_train),
                'exact_removed_from_test': int(excluded['validation_exact']),
                'near_removed_from_test': int(excluded['validation_near']),
            },
        },
        'doc_ids': {'test_count': int(test_doc_ids.size)},
    }
    return meta, test_doc_ids


def _write_rows(path, rows):
    with open(path, 'w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(str(row) + '\n')


def _write_cache(td, raw_dir):
    train_rows = [' = Alpha = ', A_BASE, A_NEAR, B_DOC, B_DOC, '']
    val_rows = [A_BASE, C_DOC, '']
    test_rows = [A_BASE, A_NEAR, C_DOC, B_DOC, T_DOC, T_DOC, '']
    _write_rows(os.path.join(raw_dir, 'wiki.train.raw'), train_rows)
    _write_rows(os.path.join(raw_dir, 'wiki.valid.raw'), val_rows)
    _write_rows(os.path.join(raw_dir, 'wiki.test.raw'), test_rows)
    meta, train_ids, val_ids = _build_reference(train_rows, val_rows, fit_rows=1)
    test_meta, test_ids = _build_test_reference(train_rows, val_rows, test_rows)
    with open(os.path.join(td, 'meta_v5_sl64.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh)
    with open(os.path.join(td, 'test_meta_v5_sl64.json'), 'w', encoding='utf-8') as fh:
        json.dump(test_meta, fh)
    np.save(os.path.join(td, 'train_doc_ids_v5_sl64.npy'), train_ids)
    np.save(os.path.join(td, 'val_doc_ids_v5_sl64.npy'), val_ids)
    np.save(os.path.join(td, 'test_doc_ids_v5_sl64.npy'), test_ids)
    return train_rows, val_rows, test_rows, meta, train_ids, val_ids, test_meta, test_ids


def test_replay_train_validation_membership_is_reproducible():
    train_rows = [' = Alpha = ', A_BASE, A_NEAR, B_DOC, B_DOC, '']
    val_rows = [A_BASE, C_DOC, '']
    meta, train_ids, val_ids = _build_reference(train_rows, val_rows, fit_rows=1)
    rep = L.replay_train_validation_cleanup(
        train_rows, val_rows, train_meta=meta,
        train_doc_ids=train_ids, validation_doc_ids=val_ids)
    assert rep['status'] == 'ok', rep
    assert rep['problems'] == [], rep
    assert rep['train']['match'] is True, rep['train']
    assert rep['validation']['match'] is True, rep['validation']
    # One near copy of the fit-prefix document and one exact repeated row.
    assert rep['train']['removed'] == {'exact': 1, 'near': 1}, rep['train']
    assert rep['validation']['removed'] == {'exact': 1, 'near': 0}, rep['validation']


def test_replay_detects_a_different_served_membership():
    train_rows = [' = Alpha = ', A_BASE, A_NEAR, B_DOC, B_DOC, '']
    val_rows = [A_BASE, C_DOC, '']
    meta, train_ids, val_ids = _build_reference(train_rows, val_rows, fit_rows=1)
    tampered = np.array(train_ids, copy=True)
    tampered[0] = np.uint64(int(tampered[0]) + 7)
    rep = L.replay_train_validation_cleanup(
        train_rows, val_rows, train_meta=meta,
        train_doc_ids=tampered, validation_doc_ids=val_ids)
    assert rep['status'] == 'fail', rep
    assert rep['train']['first_mismatch']['position'] == 0, rep['train']


def test_replay_detects_a_different_dedup_geometry():
    train_rows = [' = Alpha = ', A_BASE, A_NEAR, B_DOC, B_DOC, '']
    val_rows = [A_BASE, C_DOC, '']
    meta, train_ids, val_ids = _build_reference(train_rows, val_rows, fit_rows=1)
    # The cache keeps the ids produced by threshold=3.  Replaying with
    # threshold=0 makes the near copy no longer a duplicate, so a different
    # document set is served even though every recorded count is unchanged.
    strict = dict(L.DEDUP_PROTOCOL_DEFAULTS)
    strict['near_hamming_threshold'] = 0
    tampered_meta = dict(meta)
    tampered_meta['dedup'] = dict(meta['dedup'])
    tampered_meta['dedup']['protocol'] = strict
    tampered_meta['dedup']['protocol_fp'] = L.dedup_protocol_fp(strict)
    rep = L.replay_train_validation_cleanup(
        train_rows, val_rows, train_meta=tampered_meta,
        train_doc_ids=train_ids, validation_doc_ids=val_ids)
    assert rep['status'] == 'fail', rep
    assert any('position' in p for p in rep['problems']), rep


def test_replay_detects_a_wrong_fit_prefix():
    train_rows = [' = Alpha = ', A_BASE, A_NEAR, B_DOC, B_DOC, '']
    val_rows = [A_BASE, C_DOC, '']
    meta, train_ids, val_ids = _build_reference(train_rows, val_rows, fit_rows=1)
    tampered = json.loads(json.dumps(meta))
    tampered['tokenizer_fit_rows'] = 10
    rep = L.replay_train_validation_cleanup(
        train_rows, val_rows, train_meta=tampered,
        train_doc_ids=train_ids, validation_doc_ids=val_ids)
    assert rep['status'] == 'fail', rep


def test_replay_allows_the_validation_trailing_trim():
    train_rows = [' = Alpha = ', A_BASE, A_NEAR, B_DOC, B_DOC, '']
    val_rows = [A_BASE, C_DOC, '']
    meta, train_ids, val_ids = _build_reference(train_rows, val_rows, fit_rows=1)
    # The loader tokenises one extra validation sequence and trims the served
    # sidecar.  Dropping the last replayed validation document is legitimate.
    trimmed = val_ids[:-1]
    rep = L.replay_train_validation_cleanup(
        train_rows, val_rows, train_meta=meta,
        train_doc_ids=train_ids, validation_doc_ids=trimmed)
    assert rep['status'] == 'ok', rep
    assert any('served prefix' in w for w in rep['warnings']), rep['warnings']


def test_replay_is_not_verifiable_without_protocol_or_fit_rows():
    train_rows = [A_BASE, B_DOC]
    val_rows = [C_DOC]
    meta, train_ids, val_ids = _build_reference(train_rows, val_rows, fit_rows=1)
    no_proto = json.loads(json.dumps(meta))
    del no_proto['dedup']
    rep = L.replay_train_validation_cleanup(
        train_rows, val_rows, train_meta=no_proto,
        train_doc_ids=train_ids, validation_doc_ids=val_ids)
    assert rep['status'] == 'not_verifiable', rep
    no_fit = json.loads(json.dumps(meta))
    del no_fit['tokenizer_fit_rows']
    rep = L.replay_train_validation_cleanup(
        train_rows, val_rows, train_meta=no_fit,
        train_doc_ids=train_ids, validation_doc_ids=val_ids)
    assert rep['status'] == 'not_verifiable', rep


def test_heldout_replay_membership_and_counts():
    train_rows = [' = Alpha = ', A_BASE, A_NEAR, B_DOC, B_DOC, '']
    val_rows = [A_BASE, C_DOC, '']
    test_rows = [A_BASE, A_NEAR, C_DOC, B_DOC, T_DOC, T_DOC, '']
    test_meta, test_ids = _build_test_reference(train_rows, val_rows, test_rows)
    rep = L.replay_heldout_cleanup(
        train_rows, val_rows, test_rows, test_meta=test_meta,
        test_doc_ids=test_ids)
    assert rep['status'] == 'ok', rep
    assert rep['problems'] == [], rep
    assert rep['test']['match'] is True, rep['test']
    assert rep['test']['removed']['train_exact'] == 2, rep['test']
    assert rep['test']['removed']['train_near'] == 1, rep['test']
    assert rep['test']['removed']['test_exact'] == 1, rep['test']
    assert rep['test']['validation_indexed_documents'] == 2, rep['test']


def test_heldout_replay_detects_a_count_or_membership_tamper():
    train_rows = [' = Alpha = ', A_BASE, A_NEAR, B_DOC, B_DOC, '']
    val_rows = [A_BASE, C_DOC, '']
    test_rows = [A_BASE, A_NEAR, C_DOC, B_DOC, T_DOC, T_DOC, '']
    test_meta, test_ids = _build_test_reference(train_rows, val_rows, test_rows)
    bad_count = json.loads(json.dumps(test_meta))
    bad_count['near_duplicate_cleanup']['train_near_removed'] = 0
    rep = L.replay_heldout_cleanup(
        train_rows, val_rows, test_rows, test_meta=bad_count,
        test_doc_ids=test_ids)
    assert rep['status'] == 'fail', rep
    bad_ids = np.array(test_ids, copy=True)
    bad_ids[0] = np.uint64(int(bad_ids[0]) + 1)
    rep2 = L.replay_heldout_cleanup(
        train_rows, val_rows, test_rows, test_meta=test_meta,
        test_doc_ids=bad_ids)
    assert rep2['status'] == 'fail', rep2


def test_file_level_replay_and_explicit_test_opt_in():
    with tempfile.TemporaryDirectory() as td:
        cache = os.path.join(td, 'cache')
        raw = os.path.join(td, 'raw')
        os.makedirs(cache)
        os.makedirs(raw)
        _write_cache(cache, raw)
        out = L.replay_wikitext_cleaning(cache_dir=cache, raw_dir=raw)
        assert out['status'] == 'ok', out
        assert out['test'] is None and out['test_raw_read'] is False, out
        out2 = L.replay_wikitext_cleaning(cache_dir=cache, raw_dir=raw,
                                          include_test=True)
        assert out2['status'] == 'ok', out2
        assert out2['test_raw_read'] is True, out2
        assert out2['test'] and out2['test']['match'] is True, out2
        # The held-out gate is opt-in: removing wiki.test.raw must not make the
        # train/validation replay fail, but an explicit include_test must.
        os.unlink(os.path.join(raw, 'wiki.test.raw'))
        out3 = L.replay_wikitext_cleaning(cache_dir=cache, raw_dir=raw)
        assert out3['status'] == 'ok', out3
        out4 = L.replay_wikitext_cleaning(cache_dir=cache, raw_dir=raw,
                                          include_test=True)
        assert out4['status'] == 'not_verifiable', out4


def test_file_level_replay_reports_missing_inputs_honestly():
    with tempfile.TemporaryDirectory() as td:
        out = L.replay_wikitext_cleaning(cache_dir=td, raw_dir=td)
        assert out['status'] == 'not_verifiable', out
        cache = os.path.join(td, 'cache')
        raw = os.path.join(td, 'raw')
        os.makedirs(cache)
        os.makedirs(raw)
        _write_cache(cache, raw)
        os.unlink(os.path.join(cache, 'val_doc_ids_v5_sl64.npy'))
        out = L.replay_wikitext_cleaning(cache_dir=cache, raw_dir=raw)
        assert out['status'] == 'not_verifiable', out


def test_replay_max_rows_guard_is_explicit():
    train_rows = [A_BASE, B_DOC, B_DOC]
    val_rows = [C_DOC]
    meta, train_ids, val_ids = _build_reference(train_rows, val_rows, fit_rows=1)
    rep = L.replay_train_validation_cleanup(
        train_rows, val_rows, train_meta=meta,
        train_doc_ids=train_ids, validation_doc_ids=val_ids, max_rows=1)
    assert rep['status'] == 'not_verifiable', rep
    assert any('max_rows' in p for p in rep['problems']), rep


def test_replay_max_rows_guard_precedes_the_raw_read():
    with tempfile.TemporaryDirectory() as td:
        cache = os.path.join(td, 'cache')
        raw = os.path.join(td, 'raw')
        os.makedirs(cache)
        os.makedirs(raw)
        _write_cache(cache, raw)
        original = L._read_raw_rows
        def _boom(_path):
            raise AssertionError('the guard must reject before reading the raw dump')
        L._read_raw_rows = _boom
        try:
            out = L.replay_wikitext_cleaning(cache_dir=cache, raw_dir=raw,
                                             max_rows=1)
        finally:
            L._read_raw_rows = original
        assert out['status'] == 'not_verifiable', out
        assert any('max_rows' in p for p in out['problems']), out


def test_audit_gate_wraps_the_replay_result():
    with tempfile.TemporaryDirectory() as td:
        cache = os.path.join(td, 'cache')
        raw = os.path.join(td, 'raw')
        os.makedirs(cache)
        os.makedirs(raw)
        _write_cache(cache, raw)
        out = A.audit_cleaning_replay(cache, raw)
        assert out['status'] == 'ok', out
        assert out['kind'] == 'cleaning_replay_audit'
        assert out['scope'] == {'train_validation': True, 'heldout_test': False}
        out2 = A.audit_cleaning_replay(cache, raw, include_test=True)
        assert out2['status'] == 'ok', out2
        assert out2['scope'] == {'train_validation': True, 'heldout_test': True}


def test_replay_allows_the_train_chunk_prefix():
    train_rows = [' = Alpha = ', A_BASE, A_NEAR, B_DOC, B_DOC, '']
    val_rows = [A_BASE, C_DOC, '']
    meta, train_ids, val_ids = _build_reference(train_rows, val_rows, fit_rows=1)
    # The loader deduplicates the whole final 10k-row chunk and only then stops
    # at the token cap, so train_doc_ids is a prefix of the replayed kept rows.
    prefix = train_ids[:1]
    rep = L.replay_train_validation_cleanup(
        train_rows, val_rows, train_meta=meta,
        train_doc_ids=prefix, validation_doc_ids=val_ids)
    assert rep['status'] == 'ok', rep
    assert rep['train']['match'] is True, rep['train']
    assert rep['train']['replayed'] > rep['train']['expected'], rep['train']


def test_replay_rejects_non_string_rows_and_malformed_counts():
    train_rows = [' = Alpha = ', A_BASE, B_DOC]
    val_rows = [C_DOC]
    meta, train_ids, val_ids = _build_reference(train_rows, val_rows, fit_rows=1)
    rep = L.replay_train_validation_cleanup(
        train_rows + [None], val_rows, train_meta=meta,
        train_doc_ids=train_ids, validation_doc_ids=val_ids)
    assert rep['status'] == 'fail', rep
    bad = json.loads(json.dumps(meta))
    bad['sampling']['train']['nonempty_rows_scanned'] = '3'
    rep2 = L.replay_train_validation_cleanup(
        train_rows, val_rows, train_meta=bad,
        train_doc_ids=train_ids, validation_doc_ids=val_ids)
    assert rep2['status'] == 'not_verifiable', rep2
    # A malformed seq_len in a sibling meta must not crash meta selection.
    with tempfile.TemporaryDirectory() as td:
        cache = os.path.join(td, 'cache')
        raw = os.path.join(td, 'raw')
        os.makedirs(cache)
        os.makedirs(raw)
        _write_cache(cache, raw)
        with open(os.path.join(cache, 'meta_v5_sl64.json'), encoding='utf-8') as fh:
            m = json.load(fh)
        m['seq_len'] = 'not-an-int'
        with open(os.path.join(cache, 'meta_v5_sl64.json'), 'w', encoding='utf-8') as fh:
            json.dump(m, fh)
        out = L.replay_wikitext_cleaning(cache_dir=cache, raw_dir=raw,
                                         train_seq_len=64)
        assert out['status'] == 'not_verifiable', out
