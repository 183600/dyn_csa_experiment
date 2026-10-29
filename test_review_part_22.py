"""Data-processing identity: near-duplicate cleaning + cleaning ledger.

Token ids do not identify how a corpus was cleaned.  These checks cover the
new protocol object (exact/SimHash geometry), its binding into cache metadata,
the cross-split geometry check, and the row/token accounting that makes
"source rows -> served rows" auditable rather than a narrative.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403


def _protocol(**changes):
    p = dict(L.DEDUP_PROTOCOL_DEFAULTS)
    p.update(changes)
    return p


def _train_meta(protocol=None, *, fit_fp='f' * 16, vocab=8192):
    protocol = _protocol() if protocol is None else protocol
    return {
        'vocab': vocab,
        'fit_fp': fit_fp,
        'seq_len': 512,
        'val_sequences': 2,
        'sampling': {
            'schema': 'wikitext_slice_sampling_v1',
            'train': {
                'design': 'source_order_prefix',
                'nonempty_rows_scanned': 10,
                'rows_served': 7,
                'tokens_served': 100,
                'duplicate_documents_skipped': {'exact': 2, 'near': 1},
            },
            'validation': {
                'design': 'source_order_prefix',
                'nonempty_rows_scanned': 6,
                'rows_served': 4,
                'tokens_served': 50,
                'duplicate_documents_skipped': {'exact': 1, 'near': 1},
            },
        },
        'doc_ids': {
            'train_count': 7,
            'validation_count': 4,
            'validation_token_count': 50,
        },
        'dedup': {
            'accounting_schema': 'wikitext_dedup_accounting_v1',
            'protocol': protocol,
            'protocol_fp': L.dedup_protocol_fp(protocol),
            'near_hamming_threshold': protocol['near_hamming_threshold'],
            'bands': protocol['bands'],
            'shingle_words': protocol['shingle_words'],
            'max_shingles': protocol['max_shingles'],
            'splits': {
                'train': {'exact': 2, 'near': 1},
                'validation': {'exact': 1, 'near': 1},
            },
            'validation_cleanup': {
                'protocol': protocol,
                'protocol_fp': L.dedup_protocol_fp(protocol),
                'exact_removed': 1,
                'near_removed': 1,
            },
        },
    }


def _test_meta(protocol=None, *, fit_fp='f' * 16, vocab=8192, **overrides):
    protocol = _protocol() if protocol is None else protocol
    block = {
        'accounting_schema': 'wikitext_dedup_accounting_v1',
        'protocol': protocol,
        'protocol_fp': L.dedup_protocol_fp(protocol),
        'near_hamming_threshold': protocol['near_hamming_threshold'],
        'bands': protocol['bands'],
        'shingle_words': protocol['shingle_words'],
        'max_shingles': protocol['max_shingles'],
        'source_documents': 20,
        'source_documents_nonempty': 15,
        'kept_documents': 10,
        'train_exact_removed': 3,
        'train_near_removed': 1,
        'test_exact_removed': 1,
        'test_near_removed': 0,
        'validation_cleanup': {
            'protocol': protocol,
            'protocol_fp': L.dedup_protocol_fp(protocol),
            'source_documents': 6,
            'kept_documents': 4,
            'duplicate_documents_removed': 2,
        },
    }
    block.update(overrides)
    return {
        'vocab': vocab,
        'fit_fp': fit_fp,
        'sampling': {
            'schema': 'wikitext_slice_sampling_v1',
            'test': {
                'design': 'source_order_prefix',
                'source_rows': 20,
                'rows_after_dedup': 10,
                'rows_served': 8,
                'tokens_served': 80,
            },
        },
        'doc_ids': {'test_count': 8, 'test_token_count': 80},
        'near_duplicate_cleanup': block,
    }


def _write(td, name, obj):
    path = os.path.join(td, name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f)
    return path


def test_dedup_protocol_is_complete_and_fingerprinted():
    d = L._DocDeduplicator()
    assert d.protocol() == L.DEDUP_PROTOCOL_DEFAULTS, d.protocol()
    assert d.protocol_fp() == L.dedup_protocol_fp(d.protocol())
    st = d.stats()
    assert st['protocol'] == d.protocol() and st['protocol_fp'] == d.protocol_fp()
    # A non-default but valid geometry is a different data identity.
    d2 = L._DocDeduplicator(near_threshold=2, bands=8,
                            shingle_words=4, max_shingles=64)
    assert d2.protocol()['near_hamming_threshold'] == 2
    assert d2.protocol()['bands'] == 8
    assert d2.protocol()['shingle_words'] == 4
    assert d2.protocol_fp() != d.protocol_fp()
    d3 = L._DocDeduplicator(scope='document')
    assert d3.protocol()['scope'] == 'document'
    assert d3.protocol_fp() != d.protocol_fp()
    # The band index is only exhaustive when fewer bits differ than bands.
    for bad in (
        {'near_threshold': 4, 'bands': 4},
        {'near_threshold': 5, 'bands': 4},
        {'near_threshold': 2, 'bands': 5},
        {'shingle_words': 0},
        {'shingle_words': 65, 'max_shingles': 64},
        {'max_shingles': True},
        {'shingle_words': 1.5},
        {'scope': 'paragraph'},
    ):
        try:
            L._DocDeduplicator(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'_DocDeduplicator({bad!r}) must be refused')


def test_dedup_protocol_validator_and_metadata_reader():
    ok = _protocol()
    assert L.dedup_protocol_problems(ok) == [], ok
    for bad in (
        None,
        {'schema': 'other'},
        _protocol(near_hamming_threshold=4, bands=4),
        _protocol(bands=5),
        _protocol(shingle_words=200, max_shingles=128),
        _protocol(simhash_bits=32),
        _protocol(scope='paragraph'),
    ):
        assert L.dedup_protocol_problems(bad), bad
    assert L.dedup_kwargs_from_protocol(ok)['near_threshold'] == 3
    assert L.dedup_protocol_from_meta({'dedup': {'protocol': ok}}) == ok
    assert L.dedup_protocol_from_meta(
        {'near_duplicate_cleanup': {'protocol': ok}}) == ok
    assert L.dedup_protocol_from_meta({'dedup': {'near_hamming_threshold': 3}}) is None


def test_data_processing_audit_accepts_a_consistent_clean_ledger():
    with tempfile.TemporaryDirectory() as td:
        _write(td, 'meta_v5_sl512.json', _train_meta())
        _write(td, 'test_meta_v5_sl4096.json', _test_meta())
        # Sidecar lengths are part of the ledger; the audit reads headers only.
        np.save(os.path.join(td, 'train_ids_v5_sl512.npy'),
                np.zeros(100, dtype=np.int32))
        np.save(os.path.join(td, 'val_batch_v5_sl512.npy'),
                np.zeros((2, 512), dtype=np.int64))
        np.save(os.path.join(td, 'train_doc_ids_v5_sl512.npy'),
                np.zeros(7, dtype=np.uint64))
        np.save(os.path.join(td, 'val_doc_ids_v5_sl512.npy'),
                np.zeros(4, dtype=np.uint64))
        np.save(os.path.join(td, 'test_ids_v5_sl4096.npy'),
                np.zeros((8, 10), dtype=np.int64))
        np.save(os.path.join(td, 'test_doc_ids_v5_sl4096.npy'),
                np.zeros(8, dtype=np.uint64))
        np.save(os.path.join(td, 'test_token_doc_ids_v5_sl4096.npy'),
                np.zeros(80, dtype=np.uint64))
        out = A.audit_data_processing(td)
        assert out['status'] == 'ok', out
        assert out['problems'] == [], out
        train = out['records']['meta_v5_sl512.json']
        test = out['records']['test_meta_v5_sl4096.json']
        assert train['protocol_fp'] == L._DocDeduplicator().protocol_fp()
        assert all(train['file_checks'].values()), train['file_checks']
        assert all(test['file_checks'].values()), test['file_checks']
        assert test['accounting']['removed_total'] == 5
        assert test['accounting']['source_documents_nonempty'] == 15
        # A sidecar that no longer matches its metadata is not a clean cache.
        np.save(os.path.join(td, 'test_ids_v5_sl4096.npy'),
                np.zeros((7, 10), dtype=np.int64))
        broken = A.audit_data_processing(td)
        assert broken['status'] == 'fail', broken
        assert any('metadata accounts for 80' in p for p in broken['problems']), broken


def test_data_processing_audit_fails_on_tampered_or_mixed_geometry():
    with tempfile.TemporaryDirectory() as td:
        good = _train_meta()
        _write(td, 'meta_v5_sl512.json', good)
        # (a) test cache cleaned with a different (but internally valid)
        # geometry: same vocab/fit_fp, different band layout.
        other = _protocol(near_hamming_threshold=2, bands=4)
        _write(td, 'test_meta_v5_sl4096.json',
               _test_meta(other, fit_fp=good['fit_fp'], vocab=good['vocab']))
        out = A.audit_data_processing(td)
        assert out['status'] == 'fail', out
        assert any('does not match the paired train protocol' in p
                   for p in out['problems']), out['problems']
    with tempfile.TemporaryDirectory() as td:
        bad = _train_meta()
        bad['dedup']['protocol'] = _protocol(shingle_words=4)
        # fingerprint still names the old object
        _write(td, 'meta_v5_sl512.json', bad)
        _write(td, 'test_meta_v5_sl4096.json', _test_meta())
        out = A.audit_data_processing(td)
        assert out['status'] == 'fail', out
        assert any('protocol_fp does not match' in p for p in out['problems']), out


def test_data_processing_audit_fails_when_the_cleaning_ledger_does_not_close():
    with tempfile.TemporaryDirectory() as td:
        _write(td, 'meta_v5_sl512.json', _train_meta())
        _write(td, 'test_meta_v5_sl4096.json',
               _test_meta(kept_documents=9))  # 9 + 5 != 15
        out = A.audit_data_processing(td)
        assert out['status'] == 'fail', out
        assert any('does not close' in p for p in out['problems']), out
    with tempfile.TemporaryDirectory() as td:
        train = _train_meta()
        train['sampling']['train']['rows_served'] = 6  # 10 - 6 != 3
        _write(td, 'meta_v5_sl512.json', train)
        _write(td, 'test_meta_v5_sl4096.json', _test_meta())
        out = A.audit_data_processing(td)
        assert out['status'] == 'fail', out
        assert any('train cleaning ledger does not close' in p
                   for p in out['problems']), out


def test_data_processing_audit_reports_legacy_and_missing_caches_honestly():
    with tempfile.TemporaryDirectory() as td:
        train = _train_meta()
        test = _test_meta()
        for m, key in ((train, 'dedup'), (test, 'near_duplicate_cleanup')):
            block = m[key]
            for k in ('protocol', 'protocol_fp', 'bands', 'shingle_words',
                      'max_shingles'):
                block.pop(k, None)
            block['near_hamming_threshold'] = 3
        _write(td, 'meta_v5_sl512.json', train)
        _write(td, 'test_meta_v5_sl4096.json', test)
        out = A.audit_data_processing(td)
        assert out['status'] == 'not_verifiable_locally', out
        assert out['warnings'], out
        assert any('legacy' in w for w in out['warnings']), out['warnings']
    with tempfile.TemporaryDirectory() as td:
        out = A.audit_data_processing(td)
        assert out['status'] == 'not_verifiable_locally', out
    out = A.audit_data_processing(os.path.join(tempfile.gettempdir(), 'no-such-cache'))
    assert out['status'] == 'not_verifiable_locally', out


def test_test_document_filter_accepts_the_paired_protocol():
    protocol = _protocol(near_hamming_threshold=2, bands=4,
                         shingle_words=4, max_shingles=64)
    val = L._DocDeduplicator(**L.dedup_kwargs_from_protocol(protocol))
    val.is_duplicate('validation one')
    kept, excluded, test_index = L._filter_test_documents(
        ['train one'], val, ['new test'], protocol=protocol)
    assert kept == ['new test'], kept
    assert excluded['train_exact'] == 0 and excluded['validation_exact'] == 0
    assert test_index.protocol() == protocol
