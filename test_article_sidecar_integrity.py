"""Integrity guards for persisted article-id sidecars.

A cache that declares an article sidecar must also attest its bytes.  Without
the recorded digest an edited ``.npy`` would pass every structural check (the
length, id fingerprint and document nesting all come from the same metadata
file), so the reader must fail closed rather than treat the missing hash as a
legacy cache.  No split or model is read here.
"""
import json
import os
import tempfile

import numpy as np

from test_support import *  # noqa: F401,F403


def _write_case(cache, *, sha=True):
    seq_len, vocab, n_seqs = 4, 16, 2
    tag = f'v2_sl{seq_len}_v{vocab}_ns{n_seqs}'
    docs = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.uint64)
    art = np.asarray([10, 10, 11, 11, 10, 10, 11, 11], dtype=np.uint64)
    sidecar_name = f'test_token_article_ids_{tag}.npy'
    sidecar_path = os.path.join(cache, sidecar_name)
    np.save(sidecar_path, art)
    np.save(os.path.join(cache, f'test_token_doc_ids_{tag}.npy'), docs)
    block = L.article_sidecar_block(art, split='test', source='unit_test',
                                    extra={'n_lines': 4})
    block['sidecar_file'] = sidecar_name
    if sha:
        block['sidecar_sha256'] = L.file_sha256(sidecar_path)
    meta = {'article_sidecars': True, 'document_unit': 'line',
            'document_unit_is_article': False, 'article_unit': block}
    with open(os.path.join(cache, f'test_meta_{tag}.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(meta, fh)
    return seq_len, vocab, n_seqs


def test_declared_sidecar_without_digest_is_not_a_legacy_cache():
    with tempfile.TemporaryDirectory() as td:
        seq_len, vocab, n_seqs = _write_case(td, sha=False)
        arr, diag = L.load_cached_article_ids(
            td, split='test', seq_len=seq_len, vocab_size=vocab,
            n_seqs=n_seqs)
    assert arr is None, arr
    assert diag['status'] == 'article_sidecar_invalid', diag
    assert any('sidecar_sha256' in p for p in diag['problems']), diag


def test_matching_digest_still_loads_and_malformed_digest_fails():
    with tempfile.TemporaryDirectory() as td:
        seq_len, vocab, n_seqs = _write_case(td, sha=True)
        arr, diag = L.load_cached_article_ids(
            td, split='test', seq_len=seq_len, vocab_size=vocab,
            n_seqs=n_seqs)
        assert diag['status'] == 'ok', diag
        assert arr is not None and int(np.asarray(arr).size) == 8, diag
    with tempfile.TemporaryDirectory() as td:
        seq_len, vocab, n_seqs = _write_case(td, sha=True)
        tag = f'v2_sl{seq_len}_v{vocab}_ns{n_seqs}'
        path = os.path.join(td, f'test_meta_{tag}.json')
        with open(path, encoding='utf-8') as fh:
            meta = json.load(fh)
        meta['article_unit']['sidecar_sha256'] = 'not-a-digest'
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh)
        arr, diag = L.load_cached_article_ids(
            td, split='test', seq_len=seq_len, vocab_size=vocab,
            n_seqs=n_seqs)
    assert arr is None, arr
    assert diag['status'] == 'article_sidecar_invalid', diag
    assert any('sidecar_sha256' in p for p in diag['problems']), diag


def test_cache_hit_refuses_a_declared_but_unattested_sidecar():
    # The full cache path must apply the same attestation rule, not only the
    # narrow reader.  Reuse the synthetic-corpus builder from the sidecar
    # regression module so no real held-out data is touched.
    import test_review_part_80 as sidecar_fixture

    with tempfile.TemporaryDirectory() as td:
        _, cache = sidecar_fixture._build_cache(td)
        path = os.path.join(cache, 'test_meta_v2_sl16_v512_ns4.json')
        with open(path, encoding='utf-8') as fh:
            meta = json.load(fh)
        assert meta['article_unit'].pop('sidecar_sha256', None), meta
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh)
        try:
            L.load_wikitest(16, n_seqs=4, vocab_size=512, cache_dir=cache)
        except RuntimeError as exc:
            assert 'sidecar_sha256' in str(exc), exc
        else:
            raise AssertionError(
                'cache hit served a declared article sidecar without a digest')
