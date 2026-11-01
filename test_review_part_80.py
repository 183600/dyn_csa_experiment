"""Persisted article-level evaluation unit (zero GPU, no real test read).

These checks exercise the additive article sidecar that is written next to the
frozen WikiText line map:

* the cache-build path (raw rows + per-row lengths) and the decode path
  (already-served token stream + per-token line map) produce the SAME article
  clustering;
* the sidecar is bound by length, digest and document->article nesting;
* a legacy cache without it stays readable with an explicit fallback, while a
  declared-but-tampered/misaligned sidecar is refused;
* the analyst-facing reader and the P5T resolver prefer the persisted map;
* the document-unit audit verifies the file against its own reconstruction.

All fixtures are synthetic local raw dumps; ``wiki.test.raw`` is generated in a
temporary directory, never the real held-out corpus.  No token, split, PPL,
seed, frozen p/Holm value or manifest byte is changed.
"""
import json
import os
import shutil
import tempfile

import numpy as np
import pytest

from test_support import *  # noqa: F401,F403

REPO = os.path.dirname(os.path.abspath(__file__))


def _write_raw(root, *, n_train=300, n_valid=200, n_test=200, seed=0):
    """Write deterministic unique WikiText-like raw rows with headings."""
    import random
    rng = random.Random(seed)
    os.makedirs(root, exist_ok=True)
    vocab = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta']
    for split, n in (('train', n_train), ('valid', n_valid), ('test', n_test)):
        with open(os.path.join(root, f'wiki.{split}.raw'), 'w',
                  encoding='utf-8') as fh:
            for i in range(n):
                words = ' '.join(
                    rng.choice(vocab) + str(rng.randrange(10 ** 6))
                    for _ in range(12))
                fh.write(f'= {split} article {i} =\n\n{words}\n\n')


def _build_cache(td, *, seed=0):
    raw = os.path.join(td, 'raw')
    _write_raw(raw, seed=seed)
    cache = os.path.join(td, 'cache')
    os.makedirs(cache, exist_ok=True)
    old = os.environ.get('WT103_RAW_TXT')
    os.environ['WT103_RAW_TXT'] = raw
    try:
        L.load_wikitext(16, 1024, 512, val_seqs=4, cache_dir=cache)
        L.load_wikitest(16, n_seqs=4, vocab_size=512, cache_dir=cache)
    finally:
        if old is None:
            os.environ.pop('WT103_RAW_TXT', None)
        else:
            os.environ['WT103_RAW_TXT'] = old
    return raw, cache


@pytest.fixture(scope='module')
def built():
    with tempfile.TemporaryDirectory() as td:
        raw, cache = _build_cache(td)
        yield raw, cache


def _test_meta(cache):
    path = os.path.join(cache, 'test_meta_v2_sl16_v512_ns4.json')
    with open(path, encoding='utf-8') as fh:
        return path, json.load(fh)


def _val_meta(cache):
    path = os.path.join(cache, 'meta_v5_sl16_cap1024_v512_vs4.json')
    with open(path, encoding='utf-8') as fh:
        return path, json.load(fh)


def _same_partition(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.size != b.size or a.size == 0:
        return False
    pairs = np.unique(np.stack([a, b], axis=1), axis=0)
    return (pairs.shape[0] == np.unique(a).size == np.unique(b).size)


def test_build_and_decode_paths_agree_on_the_article_unit(built):
    _, cache = built
    ids, _, _, _, docs = L.load_wikitest(
        16, n_seqs=4, vocab_size=512, cache_dir=cache, return_doc_ids=True)
    tok = L.load_wikitext_test_tokenizer(cache, n_seqs=4, seq_len=16,
                                         vocab_size=512)
    cold = L.build_article_sidecar(ids, docs, tok, split='test',
                                   source='decoded_line_map')
    assert cold['status'] == 'ok' and cold['available'] is True, cold
    assert cold['token_article_ids'].size == ids.size
    assert np.unique(cold['token_article_ids']).size == cold['n_articles']
    loaded, diag = L.load_cached_article_ids(
        cache, split='test', seq_len=16, vocab_size=512, n_seqs=4)
    assert diag['status'] == 'ok', diag
    assert loaded.shape == ids.shape
    # Article ids are cluster LABELS.  The persisted raw-row path and the
    # decode path must agree on the partition of served tokens; a tiny
    # synthetic tokenizer may decode a heading differently, which relabels a
    # cluster without changing its membership.
    assert _same_partition(loaded.reshape(-1), cold['token_article_ids'])


def test_values_are_article_clusters_not_raw_line_ids(built):
    _, cache = built
    ids, _, _, _, docs = L.load_wikitest(
        16, n_seqs=4, vocab_size=512, cache_dir=cache, return_doc_ids=True)
    art, diag = L.load_cached_article_ids(
        cache, split='test', seq_len=16, vocab_size=512, n_seqs=4)
    assert diag['status'] == 'ok'
    # A heading and its paragraph are two line ids but one article.
    assert np.unique(art).size < np.unique(docs).size
    _, meta = _test_meta(cache)
    block = meta['article_unit']
    assert meta['article_sidecars'] is True
    assert meta['document_unit'] == 'line'       # frozen unit unchanged
    assert meta['document_unit_is_article'] is False
    assert block['schema'] == L.ARTICLE_SIDECAR_SCHEMA
    assert block['unit'] == 'article'
    assert block['valid_for_clustering'] is True
    assert block['token_article_count'] == int(ids.size)
    assert block['token_article_fp'] == L._ids_fp(art.reshape(-1))
    assert block['n_lines'] == int(np.unique(docs).size)
    assert block['n_articles'] == int(np.unique(art).size)
    assert block['sidecar_sha256'] == L.file_sha256(
        os.path.join(cache, block['sidecar_file']))


def test_validation_sidecar_is_written_and_readable(built):
    _, cache = built
    _, meta = _val_meta(cache)
    assert meta['article_sidecars'] is True
    assert meta['document_unit'] == 'line'
    art, diag = L.load_cached_article_ids(
        cache, split='validation', seq_len=16, vocab_size=512,
        n_train_tokens=1024, val_seqs=4)
    assert diag['status'] == 'ok', diag
    assert art.shape == (4, 16)
    assert meta['article_unit']['token_article_count'] == int(art.size)


def test_problems_reject_a_line_split_across_articles():
    rows = [' = A = ', ' a text ', ' = B = ', ' b text ']
    lengths = [3, 4, 3, 4]
    rep = L.build_article_sidecar_from_rows(
        rows, lengths, sum(lengths), split='test')
    art = rep['token_article_ids']
    docs = np.repeat(np.arange(len(rows), dtype=np.uint64), lengths)
    meta = {'article_sidecars': True,
            'article_unit': dict(rep['block'])}
    assert L.article_sidecar_problems(meta, art, docs, split='test',
                                      n_tokens=art.size) == []
    bad = art.copy()
    # Split the first line's tokens between the two articles.  ``np.unique``
    # sorts labels, so pick the label that is not the first token's article.
    other = [x for x in np.unique(art) if x != art[0]][0]
    bad[0] = other
    problems = L.article_sidecar_problems(meta, bad, docs, split='test',
                                          n_tokens=art.size)
    assert any('split a document' in p for p in problems), problems


def test_problems_reject_count_fp_and_missing_attestation():
    rows = [' = A = ', ' a text ', ' = B = ', ' b text ']
    lengths = [3, 4, 3, 4]
    rep = L.build_article_sidecar_from_rows(
        rows, lengths, sum(lengths), split='test')
    art = rep['token_article_ids']
    docs = np.repeat(np.arange(len(rows), dtype=np.uint64), lengths)
    meta = {'article_sidecars': True,
            'article_unit': dict(rep['block'])}
    _, updated = _test_meta_copy(meta)
    updated['article_unit']['n_articles'] = int(np.unique(art).size) + 1
    assert any('n_articles' in p for p in L.article_sidecar_problems(
        updated, art, docs, split='test', n_tokens=art.size))
    updated = json.loads(json.dumps(meta))
    updated['article_unit']['token_article_fp'] = '0' * 64
    assert any('token_article_fp' in p for p in L.article_sidecar_problems(
        updated, art, docs, split='test', n_tokens=art.size))
    declared_only = {'article_sidecars': True}
    assert any('no article_unit block' in p for p in
               L.article_sidecar_problems(declared_only, art, docs,
                                          split='test', n_tokens=art.size))
    # An undeclared/legacy cache is readable, not a problem.
    assert L.article_sidecar_problems({}, art, docs, split='test',
                                      n_tokens=art.size) == []


def _test_meta_copy(meta):
    import copy
    return meta, copy.deepcopy(meta)


def test_no_heading_records_unavailable_not_a_fake_article():
    rep = L.build_article_sidecar_from_rows(
        [' plain one ', ' plain two '], [3, 3], 6, split='test')
    assert rep['status'] == 'not_identifiable'
    assert rep['available'] is False
    assert 'heading' in rep['reason']
    assert 'token_article_ids' not in rep


def test_legacy_cache_is_explicit_and_audit_stays_a_warning(built):
    raw, cache = built
    with tempfile.TemporaryDirectory() as td:
        copy = os.path.join(td, 'cache')
        shutil.copytree(cache, copy)
        path, meta = _test_meta(copy)
        block = meta.pop('article_unit')
        meta.pop('article_sidecars')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh)
        # An orphan sidecar (file present, metadata no longer attests it) is
        # unbound evidence, not a legacy cache.
        rep_orphan = A.audit_document_unit(copy, raw_dir=raw)
        assert rep_orphan['status'] == 'fail', rep_orphan.get('problems')
        os.unlink(os.path.join(copy, block['sidecar_file']))
        art, diag = L.load_cached_article_ids(
            copy, split='test', seq_len=16, vocab_size=512, n_seqs=4)
        assert art is None
        assert diag['status'] == 'legacy_no_article_sidecar'
        rep = A.audit_document_unit(copy, raw_dir=raw)
        trec = rep['caches'][os.path.abspath(copy)]['test']
        assert trec['article_sidecar']['status'] == 'legacy_absent'
        # The missing additive sidecar itself is not a hard failure; this
        # legacy cache only keeps the explicit line-level fallback.
        assert not any('article-id sidecar' in p for p in rep['problems']), rep


def test_tampered_test_sidecar_is_refused_on_the_cache_hit(built):
    _, cache = built
    with tempfile.TemporaryDirectory() as td:
        copy = os.path.join(td, 'cache')
        shutil.copytree(cache, copy)
        path = os.path.join(copy, 'test_token_article_ids_v2_sl16_v512_ns4.npy')
        art = np.load(path)
        art[0] += 1
        np.save(path, art)
        with pytest.raises(RuntimeError, match='article-id sidecar'):
            L.load_wikitest(16, n_seqs=4, vocab_size=512, cache_dir=copy)
        strict, _ = L.load_cached_article_ids(
            copy, split='test', seq_len=16, vocab_size=512, n_seqs=4)
        assert strict is None
        with pytest.raises(RuntimeError, match='cannot be verified'):
            L.load_cached_article_ids(
                copy, split='test', seq_len=16, vocab_size=512, n_seqs=4,
                strict=True)


def test_audit_detects_a_sidecar_that_disagrees_with_the_decode(built):
    raw, cache = built
    ids, _, _, _, docs = L.load_wikitest(
        16, n_seqs=4, vocab_size=512, cache_dir=cache, return_doc_ids=True)
    with tempfile.TemporaryDirectory() as td:
        copy = os.path.join(td, 'cache')
        shutil.copytree(cache, copy)
        path = os.path.join(copy, 'test_token_article_ids_v2_sl16_v512_ns4.npy')
        art = np.load(path)
        # Split a WHOLE line document into its own article.  The sidecar is
        # still internally valid (every line maps to one article) and the
        # metadata is recomputed, so only the decode comparison can catch it.
        new_label = np.uint64(int(np.max(art)) + 1)
        last_doc = np.unique(docs.reshape(-1))[-1]
        art[docs.reshape(-1) == last_doc] = new_label
        np.save(path, art)
        mpath, meta = _test_meta(copy)
        meta['article_unit']['token_article_fp'] = L._ids_fp(art)
        meta['article_unit']['n_articles'] = int(np.unique(art).size)
        meta['article_unit']['sidecar_sha256'] = L.file_sha256(path)
        with open(mpath, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh)
        # The cache-hit validator accepts the self-consistent file ...
        got, gdiag = L.load_cached_article_ids(
            copy, split='test', seq_len=16, vocab_size=512, n_seqs=4)
        assert gdiag['status'] == 'ok', gdiag
        # ... but the decode-based audit sees a different clustering.
        rep = A.audit_document_unit(copy, raw_dir=raw)
        assert rep['status'] == 'fail', rep.get('problems')
        trec = rep['caches'][os.path.abspath(copy)]['test']
        assert trec['article_sidecar']['status'] == 'fail'
        assert any('different clustering' in p
                   for p in trec['article_sidecar']['problems'])


def test_p5t_prefers_the_persisted_sidecar(built):
    _, cache = built
    ids, _, _, _, docs = L.load_wikitest(
        16, n_seqs=4, vocab_size=512, cache_dir=cache, return_doc_ids=True)
    loaded, ldiag = L.load_cached_article_ids(
        cache, split='test', seq_len=16, vocab_size=512, n_seqs=4)
    old = os.environ.get('WT103_CACHE_DIR')
    os.environ['WT103_CACHE_DIR'] = cache
    try:
        art, diag = V12._resolve_test_article_ids(
            ids, docs, n_seqs=4, seq_len=16, vocab_size=512)
    finally:
        if old is None:
            os.environ.pop('WT103_CACHE_DIR', None)
        else:
            os.environ['WT103_CACHE_DIR'] = old
    assert diag['source'] == 'cache_sidecar', diag
    assert np.array_equal(art, loaded)
    assert ldiag['status'] == 'ok'


def test_shape_and_absent_cache_diagnostics(built):
    _, cache = built
    art, diag = L.load_cached_article_ids(
        cache, split='test', seq_len=16, vocab_size=512, n_seqs=4,
        shape=(3, 16))
    assert art is None and diag['status'] == 'shape_mismatch', diag
    with tempfile.TemporaryDirectory() as td:
        art, diag = L.load_cached_article_ids(
            td, split='test', seq_len=16, vocab_size=512, n_seqs=4)
        assert art is None and diag['status'] == 'cache_absent'


def test_registration_and_document_unit_contract(built):
    _, cache = built
    with open(os.path.join(REPO, 'explib', '__init__.py'),
              encoding='utf-8') as fh:
        init = fh.read()
    assert "'part_75_article_sidecar.py'" in init
    assert 'ARTICLE_SIDECAR_SCHEMA' in init or hasattr(
        L, 'ARTICLE_SIDECAR_SCHEMA')
    _, meta = _test_meta(cache)
    assert meta['document_unit'] == 'line'
    assert meta['article_sidecars'] is True
    # The article sidecar is additive evidence; the frozen line-level fields
    # still describe the same served tokens.
    assert meta['doc_ids']['test_token_count'] == 64
    assert meta['split_fingerprints']['test']['token_documents_fp'] is not None
