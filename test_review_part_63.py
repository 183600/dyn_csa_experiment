"""CPU review checks (split group 63): article-level held-out contamination.

The WikiText cache unit is a raw line, so line-level exact/SimHash cleaning
cannot see an article whose lines were all reworded or reformatted.  These
checks cover the new served-test-cache scan and the coverage honesty of an
article-level audit sample; they never read ``wiki.test.raw``.
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def _long_body():
    return ' '.join(['Alpha', 'is', 'a', 'long', 'article', 'about',
                     'data', 'processing'] * 10)


def test_article_scan_sampling_is_not_a_prefix():
    items = list(range(10))
    got, meta = A._sample_articles(items, 3)
    assert meta['truncated'] is True and meta['n_scanned'] == 3, meta
    assert meta['coverage'] == 0.3 and meta['design'] == 'deterministic_even_spacing', meta
    # The sample spans the whole list instead of taking the first three.
    assert got[0] == 0 and got[-1] == 9, got
    got_all, meta_all = A._sample_articles(items, None)
    assert got_all == items and meta_all['truncated'] is False, meta_all
    assert meta_all['coverage'] == 1.0 and meta_all['design'] == 'full', meta_all


def test_article_leakage_scan_flags_line_invisible_near_copy():
    body = _long_body()
    ref_text = ' = Long = \n' + body
    # Same word sequence, different raw bytes: the exact line hash changes but
    # the SimHash shingles do not, which is exactly the line-index blind spot.
    test_text = ' =  Long  = \n' + body.replace(' ', '  ')
    units = [{
        'title': ' =  Long  = ', 'text': test_text,
        'line_ids': [11, 12], 'n_lines': 2,
        'n_words': A._article_word_count(test_text),
        'has_heading': True, 'truncated': False,
    }]
    scan = A._article_leakage_scan(
        {'train': [(' = Long = ', ref_text)]}, units,
        reference_line_hashes={'train': {1, 2, 3}},
        near_threshold=L.DEDUP_PROTOCOL_DEFAULTS['near_hamming_threshold'])
    assert scan['status'] == 'fail', scan
    assert scan['matches'] and scan['matches'][0]['kind'] in ('near', 'exact'), scan
    assert scan['matches'][0]['reference_split'] == 'train', scan
    assert scan['test_articles_scanned'] == 1, scan

    # An exact line hit is line-visible: the line index can see it, so this
    # scan must not double-count it as a new article-level failure.
    hit = A._article_leakage_scan(
        {'train': [(' = Long = ', ref_text)]}, units,
        reference_line_hashes={'train': {11}},
        near_threshold=L.DEDUP_PROTOCOL_DEFAULTS['near_hamming_threshold'])
    assert hit['status'] == 'not_verifiable', hit
    assert hit['test_articles_line_visible'] == 1 and not hit['matches'], hit


def test_article_leakage_scan_coverage_is_honest():
    body = _long_body()
    ref_text = ' = Long = \n' + body
    # A short article cannot be compared by whole-article SimHash.  It is
    # reported as uncovered, not silently certified as clean.
    short = [{
        'title': ' = Tiny = ', 'text': ' = Tiny = \n tiny body ',
        'line_ids': [21], 'n_lines': 2,
        'n_words': 3, 'has_heading': True, 'truncated': False,
    }]
    scan = A._article_leakage_scan(
        {'train': [(' = Long = ', ref_text)]}, short,
        reference_line_hashes={'train': {1}})
    assert scan['status'] == 'not_verifiable', scan
    assert scan['test_articles_too_short'] == 1, scan

    # A truncated final article is excluded from the comparison and reported,
    # rather than compared as if it were a complete article.
    truncated = [{
        'title': ' =  Long  = ', 'text': ' =  Long  = \n' + body,
        'line_ids': [31, 32], 'n_lines': 2,
        'n_words': A._article_word_count(body), 'has_heading': True,
        'truncated': True,
    }]
    scan2 = A._article_leakage_scan(
        {'train': [(' = Long = ', ref_text)]}, truncated,
        reference_line_hashes={'train': {1}})
    assert scan2['status'] == 'not_verifiable', scan2
    assert scan2['test_articles_truncated'] == 1, scan2


def test_document_unit_audit_fails_on_served_test_article_copy():
    td = tempfile.mkdtemp(prefix='testarticle_')
    try:
        info = _write_document_unit_fixture(td)
        tag, ttag = info['tag'], info['ttag']
        body = _long_body()
        # The model actually saw this long training article (append its line
        # ids to the served train sidecar and its rows to the raw split).
        long_lines = [' = Long = ', body, body]
        with open(os.path.join(info['raw'], 'wiki.train.raw'), 'a',
                  encoding='utf-8') as f:
            f.write('\n'.join(long_lines) + '\n')
        trd_path = os.path.join(td, f'train_doc_ids_{tag}.npy')
        trd = np.load(trd_path)
        new_ids = np.asarray(
            [int(L._doc_id_u64(L._doc_exact_fp(r))) for r in long_lines],
            dtype=np.uint64)
        np.save(trd_path, np.concatenate([trd, new_ids]))

        # Rebuild only the served test cache with a reformatted copy: every
        # line's exact hash differs (so the line index misses it) while the
        # word-shingle SimHash still identifies the same article.
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(td, f'tok_{tag}.json'))
        eos = tok.token_to_id('<eos>')
        test_rows = [' =  Long  = ',
                     body.replace(' ', '  '),
                     body.replace(' ', '   '),
                     ' = Tail = ', ' tail padding article. ']
        lengths = []
        ids = L._tokenise_row_chunks([test_rows], 10 ** 9, 'test', tok, eos,
                                     lengths_out=lengths)
        sl = 8
        nt = len(ids) // sl
        batch = ids[:nt * sl].reshape(nt, sl)
        doc = L._doc_ids_for_capped_rows(test_rows, lengths, nt * sl)
        tdoc = L._token_doc_ids_for_capped_rows(test_rows, lengths, nt * sl)
        np.save(os.path.join(td, f'test_ids_{ttag}.npy'), batch.astype(np.int64))
        np.save(os.path.join(td, f'test_doc_ids_{ttag}.npy'), doc)
        np.save(os.path.join(td, f'test_token_doc_ids_{ttag}.npy'), tdoc)
        with contextlib.redirect_stdout(io.StringIO()):
            rep = A.audit_document_unit(td, raw_dir=info['raw'])
        rec = rep['caches'][os.path.abspath(td)]
        scan = rec['test_article_leakage']
        assert rep['status'] == 'fail', rep
        assert scan['status'] == 'fail' and scan['matches'], scan
        assert any('test article leakage' in p for p in rep['problems']), rep
        assert rec['document_level_claims_allowed'] is False, rec
    finally:
        shutil.rmtree(td, ignore_errors=True)

def test_document_unit_audit_records_truncated_article_scan():
    td = tempfile.mkdtemp(prefix='articlecap_')
    try:
        info = _write_document_unit_fixture(td)
        with contextlib.redirect_stdout(io.StringIO()):
            rep = A.audit_document_unit(td, raw_dir=info['raw'], max_articles=1)
        rec = rep['caches'][os.path.abspath(td)]
        scan = rec['article_leakage']['article_scan']['train']
        assert scan['truncated'] is True and scan['design'] == 'deterministic_even_spacing', scan
        assert scan['n_scanned'] == 1 and scan['coverage'] == 0.5, scan
        assert rec['article_leakage']['status'] == 'not_verifiable_locally', rec
        assert any('truncated' in p for p in rec['article_leakage']['problems']), rec
    finally:
        shutil.rmtree(td, ignore_errors=True)

def test_article_leakage_scan_does_not_fail_a_clean_article():
    ref = ' = Long = \n' + _long_body()
    clean_text = (' = Different = \n' + ' '.join(
        ['gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota',
         'kappa'] * 8))
    units = [{
        'title': ' = Different = ', 'text': clean_text,
        'line_ids': [41, 42], 'n_lines': 2,
        'n_words': A._article_word_count(clean_text),
        'has_heading': True, 'truncated': False,
    }]
    scan = A._article_leakage_scan(
        {'train': [(' = Long = ', ref)]}, units,
        reference_line_hashes={'train': {1, 2, 3}})
    assert scan['status'] != 'fail' and not scan['matches'], scan
    assert scan['test_articles_scanned'] == 1, scan
