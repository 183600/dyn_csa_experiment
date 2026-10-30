
def _tokenise_row_chunks(row_chunks, cap_tokens, label, tok, eos_id,
                         lengths_out=None):
    """Standalone twin of load_wikitext's nested stream_tokenise with the exact
    same chunking / eos / cap / flush behaviour (1<<22 chunks, row-chunk 10000,
    [:cap_tokens] truncation) so the served ids obey the same discipline.

    ``lengths_out`` optionally receives ``len(ids) + 1`` (the EOS) for every
    encoded non-empty row, in row order.  The produced ids are byte-identical
    with or without it; the lengths let the caller reconstruct which documents
    actually contributed tokens to the capped slice (for a document-id
    sidecar) without tokenising the corpus a second time.
    """
    chunks = []
    cur = []
    n_flushed = 0

    def _flush():
        nonlocal n_flushed
        if cur:
            n_flushed += len(cur)
            chunks.append(np.array(cur, dtype=np.int64))
            cur.clear()
    for rows in row_chunks:
        rows = [r for r in rows if r.strip()]
        for enc in tok.encode_batch(rows):
            cur.extend(enc.ids)
            cur.append(eos_id)
            if lengths_out is not None:
                lengths_out.append(len(enc.ids) + 1)
        if len(cur) >= (1 << 22):
            _flush()
        n_have = n_flushed + len(cur)
        if n_have >= cap_tokens:
            break
        print(f'    [{label}] {n_have} tokens so far')
    _flush()
    if not chunks:
        return np.empty(0, dtype=np.int64)
    ids = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
    return ids[:cap_tokens]


def _doc_ids_for_capped_rows(rows, lengths, cap_tokens):
    """Document ids of the rows that contributed to a capped token stream.

    ``lengths`` is the per-row ``len(ids) + 1`` list produced by
    :func:`_tokenise_row_chunks`; it is aligned with the non-empty ``rows``.
    A row is recorded when at least one of its tokens (or its EOS) falls before
    ``cap_tokens``, matching the truncated id array exactly.
    """
    acc = array.array('Q')
    cursor = 0
    cap_tokens = max(0, int(cap_tokens))
    for row, length in zip(rows, lengths):
        if cursor >= cap_tokens:
            break
        take = min(int(length), cap_tokens - cursor)
        if take <= 0:
            break
        acc.append(int(_doc_id_u64(_doc_exact_fp(row))))
        cursor += take
    return (np.frombuffer(acc, dtype=np.uint64).copy()
            if acc else np.empty(0, dtype=np.uint64))


def _token_doc_ids_for_capped_rows(rows, lengths, cap_tokens):
    """Per-TOKEN document ids of a capped token stream.

    The document-level sidecar answers "which documents contributed", but not
    "which token belongs to which document" — fixed evaluation rows are
    arbitrary windows that may split or merge documents.  Without the latter
    mapping a "per-document" statistic is really a per-row statistic and its
    document bootstrap is pseudoreplicated (a document split over two rows is
    counted twice, while one row can mix two documents).

    ``lengths`` is the per-row ``len(ids) + 1`` list produced by
    :func:`_tokenise_row_chunks` (aligned with the non-empty rows).  The
    returned array is exactly as long as the served token slice; the EOS of a
    document carries that document's id, so the boundary token is attributed
    to the document that emitted it.
    """
    doc_ids = []
    takes = []
    cursor = 0
    cap_tokens = max(0, int(cap_tokens))
    for row, length in zip(rows, lengths):
        if cursor >= cap_tokens:
            break
        take = min(int(length), cap_tokens - cursor)
        if take <= 0:
            break
        doc_ids.append(int(_doc_id_u64(_doc_exact_fp(row))))
        takes.append(take)
        cursor += take
    if not takes:
        return np.empty(0, dtype=np.uint64)
    # One repeat rather than a Python list of `take` ids: the per-token
    # sidecar is O(tokens), so a single long document must not materialise an
    # equally long intermediate list.
    return np.repeat(np.asarray(doc_ids, dtype=np.uint64),
                     np.asarray(takes, dtype=np.int64))


def _token_doc_ids_from_eos(token_ids, doc_ids, eos_id):
    """Recover per-token document ids for a cache built by the EOS discipline.

    ``_tokenise_row_chunks`` appends exactly one EOS to every encoded row, so
    the served stream contains one EOS per COMPLETE document (the last document
    may be cut by the token cap and have no EOS).  The document-level sidecar
    lists the contributing documents in stream order, so the boundaries can be
    reconstructed without re-reading the raw test split — which matters after
    the one-shot test access, when rebuilding the frozen cache is forbidden.

    Returns ``(ids, None)`` or ``(None, reason)``.  The caller must still check
    that the reconstructed document set equals the sidecar's; this function
    only enforces the EOS-count/order invariants.
    """
    arr = np.asarray(token_ids).reshape(-1)
    docs = np.asarray(doc_ids).reshape(-1)
    if arr.size == 0 or docs.size == 0:
        return None, 'empty token or document sidecar'
    if eos_id is None or isinstance(eos_id, (bool, np.bool_)):
        return None, 'no integer EOS id available'
    try:
        _eos = int(eos_id)
    except (TypeError, ValueError, OverflowError):
        return None, 'EOS id is not an integer'
    if _eos < 0:
        return None, 'EOS id is negative'
    eos_pos = np.flatnonzero(arr == _eos)
    n_docs = int(docs.size)
    n_eos = int(eos_pos.size)
    if n_eos not in (n_docs, n_docs - 1):
        return None, (f'{n_eos} EOS tokens for {n_docs} documents; '
                       'the stream does not follow the one-EOS-per-document '
                       'construction')
    # Every document but (possibly) the capped final one ends at its EOS.
    # Build the token counts per document and repeat the ids; the old loop
    # assigned one Python slice per document, which is O(documents) Python
    # calls and is needlessly slow on a long-context test cache.
    ends = np.empty(n_docs, dtype=np.int64)
    ends[:n_eos] = eos_pos + 1
    if n_eos < n_docs:
        # The final document was truncated by the token cap; it has no EOS.
        ends[-1] = int(arr.size)
    starts = np.empty(n_docs, dtype=np.int64)
    starts[0] = 0
    starts[1:] = ends[:-1]
    lengths = ends - starts
    if bool((lengths < 0).any()) or int(lengths.sum(dtype=np.int64)) != int(arr.size):
        # This includes the n_docs == n_eos case with trailing tokens after
        # the last EOS, which has no document id left to carry them.
        return None, 'EOS boundaries do not partition the token stream'
    out = np.repeat(docs, lengths)
    if int(out.size) != int(arr.size):
        return None, 'document sidecar does not cover the token stream'
    return out, None

TEST_READ_PURPOSES = ('final_evaluation',)


def load_wikitest(seq_len, n_seqs=256, vocab_size=None, cache_dir=None,
                  purpose='final_evaluation', return_doc_ids=False):
    """Load the WikiText-103 **test** split as (n_seqs, seq_len) int64 rows.

    只读 wiki.test.raw（本地 dump 或 HF test split）作为最终评估数据
    （design_notes §17 的「未参与方法选择的数据」）；train/validation 只用于
    重建同一个冻结 tokenizer 和构建去重索引，绝不把 test 行用于拟合或选择。
    与 load_wikitext 同一套纪律：npy 缓存 + sha256 内容指纹 + 原子写；
    分词器使用与主语料完全相同的冻结 train-prefix token 上限（上限记录在
    train cache meta 中）重建，并把拟合语料指纹存进 meta 以供审计。构建
    test 缓存时，与 training **以及 validation（选择集）** 文档完全相同或
    近重复（SimHash）的 test 文档会在 tokenisation 前被剔除并记录数量；
    test 文档 id 会作为 sidecar 落盘，缓存命中时重新计算 train/test 与
    val/test 的文档交集。这样既防止 test 文档穿越到训练集，也防止它穿越到
    用于调参/早停的验证集。缓存命中会校验内容指纹、tokenizer 拟合来源和
    去重元数据；train 语料被更换时应删除整个 wt103_cache/，否则 test 缓存
    不会跟随重建——这是显式记录的边界，不是静默的失效路径。

    ``vocab_size=None`` resolves the width from the paired train cache instead
    of falling back to the historical 8192.  ``purpose`` is an allow-list:
    only the frozen final evaluation may read this split, so a diagnostic or
    tuning caller cannot silently turn test into an implicit validation set.
    ``return_doc_ids=True`` returns a fifth element: the per-token document-id
    map shaped like the test rows (or ``None`` when neither the sidecar nor
    EOS reconstruction can verify it).  The default four-tuple is unchanged.
    """
    if purpose not in TEST_READ_PURPOSES:
        raise ValueError(
            f'load_wikitest: purpose={purpose!r} is not an allowed test-split '
            f'read ({TEST_READ_PURPOSES}).  Tuning, model selection, diagnostics '
            'and aggregation must use the validation split; only the frozen '
            'final evaluation may read test.')
    from datasets import load_dataset
    cache_dir = _wt_cache_dir(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    if vocab_size is None:
        vocab_size = _unique_cached_vocab(cache_dir, label='load_wikitest')
    vocab_size = _protocol_pos_int('vocab_size', vocab_size)
    seq_len = _protocol_pos_int('seq_len', seq_len)
    n_seqs = _protocol_pos_int('n_seqs', n_seqs)
    tag = f'v2_sl{seq_len}_v{vocab_size}_ns{n_seqs}'
    te_path = os.path.join(cache_dir, f'test_ids_{tag}.npy')
    tb_path = os.path.join(cache_dir, f'test_bnd_{tag}.npy')
    td_path = os.path.join(cache_dir, f'test_doc_ids_{tag}.npy')
    ttd_path = os.path.join(cache_dir, f'test_token_doc_ids_{tag}.npy')
    me_path = os.path.join(cache_dir, f'test_meta_{tag}.json')
    tk_path = os.path.join(cache_dir, f'test_tok_{tag}.json')
    if all((os.path.exists(p) for p in (te_path, tb_path, me_path))):
        _sig_paths = {'test': te_path, 'bnd': tb_path, 'meta': me_path}
        if os.path.exists(td_path):
            _sig_paths['test_doc_ids'] = td_path
        if os.path.exists(ttd_path):
            _sig_paths['test_token_doc_ids'] = ttd_path
        if os.path.exists(tk_path):
            _sig_paths['tokenizer'] = tk_path
        # The verification path also reads every paired train meta and the
        # train/validation document-id sidecars, and it compares the recorded
        # dataset source.  Those files were absent from the memo key, so a
        # replaced/deleted sidecar or a different declared source could keep
        # serving an "already verified" test tuple from this process.  Bind
        # them to the key as well.
        import glob as _glob_memo
        for _pm in sorted(_glob_memo.glob(os.path.join(
                cache_dir, 'meta_v5_*.json'))):
            _sig_paths['paired_meta:' + os.path.basename(_pm)] = _pm
        for _pat in ('train_doc_ids_v5_*.npy', 'val_doc_ids_*.npy'):
            for _p in sorted(_glob_memo.glob(os.path.join(cache_dir, _pat))):
                _sig_paths['paired_docs:' + os.path.basename(_p)] = _p
        _sig = _memo_file_sig(_sig_paths)
        _src_key = json.dumps(wikitext_dataset_source(), sort_keys=True,
                              default=str)
        _mkey = (os.path.abspath(cache_dir), tag, _sig, _src_key)
        _hit = _WT103_TEST_MEMO.get(_mkey)
        if _hit is not None:
            print('[data] serving TEST token ids from the in-process memo (same cache files, already verified) ...')
            return _hit if return_doc_ids else _hit[:4]
        print('[data] loading cached TEST token ids ...')
        test_ids = np.load(te_path)
        meta = _read_cache_meta(me_path, 'WikiText TEST')
        if meta is None:
            # The test split is frozen and must never be silently rebuilt by a
            # mere metadata read: fail closed and require an explicit rebuild.
            raise RuntimeError(
                'load_wikitest: the cached TEST metadata is unreadable; the '
                'frozen test slice cannot be verified — delete the test cache '
                'and rebuild it explicitly (V12_REFREEZE rules apply after '
                'the first test access)')
        test_bnd = np.load(tb_path)
        test_doc_ids = None
        if os.path.exists(td_path):
            try:
                test_doc_ids = np.load(td_path)
            except Exception as _de:
                _problems_td = f'test document-id sidecar unreadable ({type(_de).__name__}: {_de})'
            else:
                _problems_td = None
        else:
            _problems_td = None
        # Per-token document ids are the mapping the document-level bootstrap
        # needs; a legacy cache without them keeps the row-level fallback, but
        # a cache whose metadata DECLARES them and whose sidecar is missing is
        # corrupt and must not silently degrade the analysis granularity.
        test_token_doc_ids = None
        _problems_ttd = None
        if os.path.exists(ttd_path):
            try:
                test_token_doc_ids = np.load(ttd_path)
            except Exception as _te:
                _problems_ttd = f'test token document-id sidecar unreadable ({type(_te).__name__}: {_te})'
        elif 'test_token_doc_ids' in (meta.get('file_sigs') or {}):
            _problems_ttd = ('test metadata binds a per-token document-id sidecar '
                            'that is missing')
        _problems = []
        if _problems_td:
            _problems.append(_problems_td)
        if _problems_ttd:
            _problems.append(_problems_ttd)
        # Upstream identity: a frozen test slice is only a snapshot if the
        # corpus commit is the pinned one (a local raw dump is content-bound by
        # the fingerprints and gets a warning, not a hard failure).  Compare it
        # with every paired train cache whose metadata records a revision.
        _src_problems, _src_warnings = _dataset_source_problems(
            meta, wikitext_dataset_source(), 'WikiText TEST')
        _problems.extend(_src_problems)
        _warn_dataset_source('WikiText TEST', _src_warnings)
        for _skey in ('train_dataset_source', 'validation_dataset_source'):
            _sub = meta.get(_skey)
            if isinstance(_sub, dict):
                _sub_p, _sub_w = _dataset_source_problems(
                    {'dataset_source': _sub}, wikitext_dataset_source(),
                    f'WikiText TEST {_skey}')
                _problems.extend(_sub_p)
                _warn_dataset_source(f'WikiText TEST {_skey}', _sub_w)
        import glob as _glob_src
        _paired_train_metas = []
        for _mpath in sorted(_glob_src.glob(os.path.join(cache_dir, 'meta_v5_*.json'))):
            try:
                _mobj = json.load(open(_mpath, encoding='utf-8'))
            except Exception:
                continue
            if isinstance(_mobj, dict):
                # Pair only train caches that describe the same tokenizer-fit
                # corpus; an unrelated protocol in the same directory is not a
                # reference snapshot for this test slice.
                if (meta.get('fit_fp') and _mobj.get('fit_fp')
                        and meta.get('fit_fp') != _mobj.get('fit_fp')):
                    continue
                _paired_train_metas.append(_mobj)
        _pair_problems, _pair_warnings = dataset_source_pair_problems(
            meta, _paired_train_metas, label='test/train')
        _problems.extend(_pair_problems)
        _warn_dataset_source('test/train pair', _pair_warnings)
        _sigs = _cache_file_sigs(_sig_paths)
        _fast = _cache_file_sigs_match(meta, _sigs)
        try:
            _meta_vocab = int(meta.get('vocab'))
        except (TypeError, ValueError, OverflowError):
            _meta_vocab = None
        if _meta_vocab is None or _meta_vocab != int(vocab_size):
            _problems.append(
                f'test metadata vocab {meta.get("vocab")!r} != requested '
                f'{int(vocab_size)} — the cache tag and its content disagree')
        if meta.get('src_fp') != _ids_fp(test_ids):
            _problems.append('test id fingerprint does not match the test meta file')
        if meta.get('bnd_fp') is not None and meta['bnd_fp'] != _ids_fp(
                np.ascontiguousarray(test_bnd).reshape(-1)):
            _problems.append('test boundary-mask fingerprint does not match the test meta file')
        # Redundant explicit per-split field (new caches): a reader that looks
        # for `test_fp` must see the same content digest as `src_fp`.
        if meta.get('test_fp') is not None and meta['test_fp'] != _ids_fp(test_ids):
            _problems.append('test_fp does not match the cached test ids')
        _sf = meta.get('split_fingerprints')
        if isinstance(_sf, dict) and isinstance(_sf.get('test'), dict):
            _declared_test_fp = _sf['test'].get('ids_fp')
            if _declared_test_fp is not None and _declared_test_fp != _ids_fp(test_ids):
                _problems.append('test split_fingerprints.ids_fp does not match the cached test ids')
            _declared_bnd_fp = _sf['test'].get('boundary_fp')
            if _declared_bnd_fp is not None and _declared_bnd_fp != _ids_fp(
                    np.ascontiguousarray(test_bnd).reshape(-1)):
                _problems.append('test split_fingerprints.boundary_fp does not match the cached boundary mask')
        if np.asarray(test_bnd).shape != np.asarray(test_ids).shape:
            _problems.append(f'test boundary mask shape {np.asarray(test_bnd).shape} != {np.asarray(test_ids).shape}')
        if meta.get('shuffle_at_split') is True:
            _problems.append('metadata records a shuffled split')
        _rp_meta = meta.get('read_purpose')
        if _rp_meta is not None and _rp_meta != 'final_evaluation':
            _problems.append(
                f'test cache was built for read_purpose={_rp_meta!r}, not the '
                'frozen final evaluation')
        if meta.get('tokenizer_fit_split') not in (None, 'train'):
            _problems.append(f"tokenizer was fitted on {meta.get('tokenizer_fit_split')!r}, not train")
        if meta.get('tokenizer_fit_protocol') != 'train_prefix_token_budget':
            _problems.append('test cache does not attest the frozen train-prefix tokenizer fit')
        _ndc = meta.get('near_duplicate_cleanup')
        if not isinstance(_ndc, dict):
            _problems.append('test cache does not attest near-duplicate cleanup against train/validation')
        else:
            if _ndc.get('policy') != 'test_documents_deduplicated_against_train_and_validation':
                _problems.append(
                    'test cache does not attest duplicate cleanup against BOTH the '
                    'train and validation (selection) splits; a test document may '
                    'cross the val/test group boundary')
            _vc = _ndc.get('validation_cleanup')
            if (not isinstance(_vc, dict)
                    or _vc.get('policy') != 'validation_documents_deduplicated_against_train'):
                _problems.append(
                    'test cache does not record validation-document cleanup provenance')
        # Test document groups must not cross train or validation.  A metadata
        # flag is not enough: recompute the intersections from the persisted
        # sidecars on every hit, exactly like the train/val cache does.
        import glob as _glob2
        _mdoc_raw = meta.get('doc_ids')
        if _mdoc_raw is not None and not isinstance(_mdoc_raw, dict):
            _problems.append(
                'test cache doc_ids metadata is a '
                f'{type(_mdoc_raw).__name__}, not an object; group isolation '
                'cannot be verified')
        _mdoc = _mdoc_raw if isinstance(_mdoc_raw, dict) else {}
        if not _mdoc and test_doc_ids is None:
            # Neither the metadata ledger nor the sidecar names the served
            # documents.  The train/test and val/test intersections cannot be
            # recomputed, so the frozen slice has no evidence that its groups
            # are disjoint from the selection/training splits.  A row-level
            # fallback is not isolation evidence; fail closed and require an
            # explicit rebuild (legacy caches without a doc-id sidecar must be
            # rebuilt before the one-shot final evaluation).
            _problems.append(
                'test cache has no document-id evidence (neither the metadata '
                'doc_ids block nor test_doc_ids_*.npy is present); '
                'train/validation/test group isolation cannot be verified')
        if _mdoc or test_doc_ids is not None:
            if test_doc_ids is None:
                _problems.append('test cache metadata records document ids but the sidecar is missing')
            else:
                _declared_count = None
                if _mdoc:
                    try:
                        _declared_count = int(_mdoc.get('test_count', -1))
                    except (TypeError, ValueError, OverflowError):
                        _problems.append(
                            'test cache metadata has a non-integer test_count; '
                            'the document-id sidecar cannot be verified')
                if _declared_count is not None and int(np.asarray(test_doc_ids).size) != _declared_count:
                    _problems.append(
                        'test document-id sidecar count does not match the cache metadata')
                if _mdoc.get('test_fp') and _mdoc['test_fp'] != _ids_fp(test_doc_ids):
                    _problems.append(
                        'test document-id sidecar does not match the fingerprint in the cache metadata')
                _overlap_bad = None
                if _mdoc:
                    try:
                        _tr_ov = int(_mdoc.get('train_overlap', 0) or 0)
                        _va_ov = int(_mdoc.get('validation_overlap', 0) or 0)
                    except (TypeError, ValueError, OverflowError):
                        _problems.append(
                            'test cache metadata has non-integer train/validation '
                            'overlap counts; group isolation cannot be verified')
                    else:
                        _overlap_bad = (_mdoc.get('disjoint') is not True
                                        or _tr_ov != 0 or _va_ov != 0)
                if _overlap_bad:
                    _problems.append(
                        'test cache metadata does not attest document-disjoint '
                        'train/validation/test groups')
                _shared = {'train': 0, 'validation': 0}
                _tr_sidecars = _glob2.glob(os.path.join(cache_dir, 'train_doc_ids_v5_*.npy'))
                _va_sidecars = _glob2.glob(os.path.join(cache_dir, 'val_doc_ids_*.npy'))
                # A zero intersection computed from a missing sidecar is not
                # evidence of isolation.
                if not _tr_sidecars:
                    _problems.append('train document-id sidecar is missing; '
                                     'test/train group isolation cannot be verified')
                if not _va_sidecars:
                    _problems.append('validation document-id sidecar is missing; '
                                     'test/validation group isolation cannot be verified')
                for _dp in _tr_sidecars:
                    try:
                        _shared['train'] += int(np.intersect1d(
                            np.load(_dp), test_doc_ids, assume_unique=False).size)
                    except Exception as _de:
                        _problems.append(
                            f'train document-id sidecar unreadable ({os.path.basename(_dp)}: {_de})')
                for _dp in _va_sidecars:
                    try:
                        _shared['validation'] += int(np.intersect1d(
                            np.load(_dp), test_doc_ids, assume_unique=False).size)
                    except Exception as _de:
                        _problems.append(
                            f'validation document-id sidecar unreadable ({os.path.basename(_dp)}: {_de})')
                if _shared['train'] or _shared['validation']:
                    _problems.append(
                        f'{_shared["train"]} train / {_shared["validation"]} validation '
                        'document ids also occur in the test split (group leakage)')
        # Per-token document ids, when present, are part of the cache identity.
        # They must cover every served token exactly; a short/long sidecar
        # would silently shift document boundaries in the evaluation.
        _ttd_declared = bool(_mdoc and (
            _mdoc.get('test_token_count') is not None
            or _mdoc.get('test_token_fp') is not None))
        if test_token_doc_ids is not None:
            _ttd = np.asarray(test_token_doc_ids)
            if _ttd.ndim != 1:
                _problems.append(
                    f'test token document-id sidecar must be 1-D, got shape {_ttd.shape}')
            if _ttd.dtype.kind not in 'iu':
                _problems.append(
                    'test token document-id sidecar must contain integers, got '
                    f'dtype {_ttd.dtype}')
            elif _ttd.size and int(_ttd.min()) < 0:
                _problems.append(
                    'test token document-id sidecar contains negative ids')
            if _ttd.size != int(np.asarray(test_ids).size):
                _problems.append(
                    f'test token document-id sidecar has {_ttd.size} ids but the '
                    f'served slice has {int(np.asarray(test_ids).size)} tokens')
            if _mdoc.get('test_token_count') is not None:
                try:
                    _want_ttd_n = int(_mdoc['test_token_count'])
                except (TypeError, ValueError, OverflowError):
                    _problems.append(
                        'test cache metadata has a non-integer test_token_count')
                else:
                    if _want_ttd_n != int(_ttd.size):
                        _problems.append(
                            'test token document-id sidecar count does not match '
                            'the cache metadata')
            if _mdoc.get('test_token_fp') is not None:
                if str(_mdoc['test_token_fp']) != str(_ids_fp(_ttd)):
                    _problems.append(
                        'test token document-id sidecar does not match the '
                        'fingerprint in the cache metadata')
            _sf_all_ttd = meta.get('split_fingerprints')
            _sf_test_ttd = (_sf_all_ttd.get('test')
                            if isinstance(_sf_all_ttd, dict) else None)
            if (isinstance(_sf_test_ttd, dict)
                    and _sf_test_ttd.get('token_documents_fp') is not None
                    and str(_sf_test_ttd['token_documents_fp']) != str(_ids_fp(_ttd))):
                _problems.append(
                    'split_fingerprints.test.token_documents_fp does not match '
                    'the per-token document-id sidecar')
            # The per-token map is what the document-level evaluation actually
            # groups by; the per-document sidecar is what the train/val/test
            # group-isolation check intersects.  If they name different
            # document sets, isolation can be verified for one sidecar while
            # the scored statistic uses an unverified document.  The cold
            # build enforces this; a cache hit must not silently relax it.
            if (test_doc_ids is not None and _ttd.ndim == 1
                    and _ttd.dtype.kind in 'iu'):
                try:
                    if not np.array_equal(
                            np.unique(_ttd),
                            np.unique(np.asarray(test_doc_ids).reshape(-1))):
                        _problems.append(
                            'per-token and per-document test sidecars name '
                            'different document sets; the document-level '
                            'evaluation would not match the verified groups')
                except (TypeError, ValueError):
                    _problems.append(
                        'per-token and per-document test sidecars cannot be '
                        'compared for document identity')
        elif _ttd_declared:
            _problems.append(
                'test cache metadata records per-token document ids but the '
                'sidecar is missing; document-level evaluation would silently '
                'fall back to row granularity')
        # A cached test set is only pairable if its tokenizer was fitted on the
        # SAME train prefix as the train/validation cache.  Verifying this on
        # the cold path is not enough: a swapped test cache could otherwise
        # serve a tokenizer that saw test text.
        _train_fit_fps = set()
        _train_semantics = set()
        _train_vocab_shas = set()
        _train_dedup_fps = []
        for _mp in _glob2.glob(os.path.join(cache_dir, 'meta_v5_*.json')):
            try:
                _mt = json.load(open(_mp, encoding='utf-8'))
            except Exception:
                continue
            if not isinstance(_mt, dict):
                continue
            if _mt.get('fit_fp'):
                _train_fit_fps.add(_mt['fit_fp'])
            if _mt.get('code_semantics') is not None:
                _train_semantics.add(str(_mt['code_semantics']))
            if _mt.get('tokenizer_vocab_sha256'):
                _train_vocab_shas.add(str(_mt['tokenizer_vocab_sha256']))
            _train_proto = dedup_protocol_from_meta(_mt)
            if _train_proto is not None:
                _train_dedup_fps.append({
                    'meta': os.path.basename(_mp),
                    'fit_fp': _mt.get('fit_fp'),
                    'vocab': _mt.get('vocab'),
                    'protocol_fp': dedup_protocol_fp(_train_proto),
                })
        if _train_fit_fps and meta.get('fit_fp') not in _train_fit_fps:
            raise RuntimeError(
                'load_wikitest: the cached test split was tokenised with a fit '
                f"corpus fingerprint {meta.get('fit_fp')!r} that is absent from "
                'the train cache metas — this test cache cannot be paired with '
                'the current train/validation split; delete the test cache and '
                'rebuild it from the same train tokenizer')
        # Cleaning geometry must also match: if train and test were built with
        # different shingle widths / LSH bands / Hamming thresholds, one split
        # may retain documents the other removed even though both policy
        # strings say "deduplicated".  A missing protocol is legacy and is
        # reported, never treated as a match.
        _test_proto = dedup_protocol_from_meta(meta)
        if _test_proto is not None:
            _tpp = dedup_protocol_problems(_test_proto)
            if _tpp:
                raise RuntimeError(
                    'load_wikitest: the cached test split records an invalid '
                    'dedup protocol (' + '; '.join(_tpp) + ') — the cleaning '
                    'geometry cannot be trusted; rebuild the test cache')
            _test_dedup_fp = dedup_protocol_fp(_test_proto)
            # Compare only caches that describe the SAME corpus protocol.
            # A cache directory may hold several scale/vocab variants; using
            # the union of every train protocol would make a correctly paired
            # test cache fail merely because another variant was present.
            _paired_dedup = [
                x for x in _train_dedup_fps
                if (x.get('fit_fp') in (None, meta.get('fit_fp'))
                    and x.get('vocab') in (None, meta.get('vocab')))]
            _train_dedup_fp_set = {x['protocol_fp'] for x in _paired_dedup}
            if not _train_dedup_fp_set and _train_dedup_fps:
                # No cache declared a matching corpus identity; fall back to
                # the union but report that the pairing is ambiguous.
                _train_dedup_fp_set = {x['protocol_fp'] for x in _train_dedup_fps}
            if _train_dedup_fp_set and _test_dedup_fp not in _train_dedup_fp_set:
                raise RuntimeError(
                    'load_wikitest: the test cache was deduplicated with a '
                    f'different protocol ({_test_dedup_fp[:16]}…) than the '
                    f'paired train cache(s) {sorted(x[:16] for x in _train_dedup_fp_set)} '
                    '— the two splits did not use the same near-duplicate '
                    'cleaning geometry; delete and rebuild the test cache from '
                    'the paired train protocol')
        elif _train_dedup_fps:
            print('[data] NOTE: the test cache is legacy (no full dedup '
                  'protocol); its cleaning geometry cannot be compared with '
                  'the train cache.  Rebuild the test cache to bind it.')
        if meta.get('fit_fp') and not _train_fit_fps:
            print('[data] NOTE: no train-cache meta found; the cached TEST fit provenance cannot be cross-checked locally')
        # Code-semantics binding across the pair.  The data-construction stamp
        # is deliberately separate from the kernel variant: it says how the
        # split/dedup/tokenisation were produced.  Two caches built by
        # different semantics are not a matched train/test pair.
        _sem_meta = meta.get('code_semantics')
        if _sem_meta is not None and _train_semantics and str(_sem_meta) not in _train_semantics:
            raise RuntimeError(
                f'load_wikitest: the test cache was built under code semantics '
                f'{_sem_meta!r}, but the paired train cache(s) declare '
                f'{sorted(_train_semantics)} — the train/test split construction '
                'semantics differ, so the split cannot be treated as one frozen '
                'pair')
        if not _train_semantics:
            print('[data] NOTE: train cache meta has no data code-semantics stamp; the train/test construction binding is unverified (rebuild both caches with the current loader to bind it)')
        # Full id-space identity across the pair (in addition to the tokenizer
        # file bytes below).  A cache that records no vocab hash is legacy.
        _vsha_meta = meta.get('tokenizer_vocab_sha256')
        if _vsha_meta is not None and _train_vocab_shas and str(_vsha_meta) not in _train_vocab_shas:
            raise RuntimeError(
                'load_wikitest: the test tokenizer id-space hash does not match '
                f'the train cache tokenizer hash ({str(_vsha_meta)[:16]}… vs '
                f'{sorted(str(x)[:16] for x in _train_vocab_shas)}) — the two '
                'splits do not share one tokenizer')
        if _vsha_meta is None:
            print('[data] NOTE: test cache has no tokenizer_vocab_sha256; only the tokenizer file bytes can be compared')
        # Bind the actual test tokenizer bytes on EVERY cache hit.
        # `fit_fp` hashes the fit TEXT, which two different BPE merge sets can
        # share; it is not a tokenizer identity.  A missing identity in a
        # legacy cache is a hard refusal for the frozen confirmation (delete
        # the test cache and rebuild it from the same train tokenizer).
        _tok_sha_now = None
        if os.path.exists(tk_path):
            try:
                _tok_sha_now = file_sha256(tk_path)
            except Exception as _te:
                _problems.append(f'test tokenizer file unreadable ({type(_te).__name__}: {_te})')
        _tok_sha_meta = meta.get('tokenizer_file_sha256')
        if _tok_sha_meta:
            if _tok_sha_now is None:
                _problems.append('test meta binds a tokenizer file that is missing')
            elif _tok_sha_now != _tok_sha_meta:
                raise RuntimeError(f'load_wikitest: the cached test tokenizer bytes (sha256 {_tok_sha_now[:16]}…) do not match the identity recorded in test meta ({str(_tok_sha_meta)[:16]}…) — the test cache and its tokenizer are not a matched pair; delete the test cache and rebuild it')
        else:
            _train_toks = sorted(_glob2.glob(os.path.join(cache_dir, 'tok_v5_*.json')))
            if _tok_sha_now is None or not _train_toks:
                raise RuntimeError('load_wikitest: this is a legacy test cache without a tokenizer identity and without a matched train tokenizer file; refusing to serve an unpaired id space — rebuild the test cache with the current loader')
            try:
                from tokenizers import Tokenizer as _Tok
                _tv = dict(_Tok.from_file(tk_path).get_vocab())
                _trv = dict(_Tok.from_file(_train_toks[0]).get_vocab())
            except Exception as _te:
                raise RuntimeError(f'load_wikitest: could not compare the legacy test tokenizer with the train tokenizer ({type(_te).__name__}: {_te}); rebuild the test cache') from _te
            if _tv != _trv:
                raise RuntimeError('load_wikitest: the legacy test tokenizer vocabulary is not identical to the train tokenizer vocabulary — the id spaces cannot be paired; delete the test cache and rebuild it')
        if np.asarray(test_ids).shape != (int(n_seqs), int(seq_len)):
            raise RuntimeError(f'load_wikitest: cached test slice shape {np.asarray(test_ids).shape} != the frozen ({int(n_seqs)}, {int(seq_len)}); the frozen slice is part of the pre-registration, so a different row count must be re-frozen explicitly')
        if _problems:
            # Unlike the train/validation cache (which can be rebuilt), the
            # held-out test split is frozen: serving it while any content,
            # tokenizer or provenance check is unverified would silently make
            # an un-auditable panel look like a confirmation.  Fail closed.
            raise RuntimeError(
                'load_wikitest: the cached TEST split cannot be verified ('
                + '; '.join(_problems)
                + ') — refusing to serve it.  Repair the cache pair explicitly '
                '(never by relaxing the check); a frozen test slice is read once '
                'and cannot be silently redefined')
        if not _fast:
            _validate_token_ids(test_ids, meta['vocab'], 'cached test_ids')
        _out_docs = None
        if test_token_doc_ids is not None:
            _out_docs = np.asarray(test_token_doc_ids).reshape(
                int(n_seqs), int(seq_len))
        elif test_doc_ids is not None:
            # Legacy cache: recover the token→document mapping from the
            # construction contract (one EOS per complete document) instead of
            # forcing a rebuild of a frozen test cache.  If the invariant does
            # not hold, the caller gets the explicit row-level fallback.
            _eos_hit = meta.get('eos_id')
            if _eos_hit is None and os.path.exists(tk_path):
                try:
                    from tokenizers import Tokenizer as _TokE
                    _eos_hit = _TokE.from_file(tk_path).token_to_id('<eos>')
                except Exception:
                    _eos_hit = None
            if _eos_hit is not None:
                _recon, _why = _token_doc_ids_from_eos(
                    np.asarray(test_ids).reshape(-1), test_doc_ids, _eos_hit)
                if (_recon is not None
                        and np.array_equal(np.unique(_recon),
                                           np.unique(test_doc_ids))):
                    _out_docs = _recon.reshape(int(n_seqs), int(seq_len))
                    print('[data] reconstructed the per-token document mapping '
                          'from the EOS boundaries and the document sidecar '
                          '(legacy cache; no rebuild needed)')
                elif return_doc_ids:
                    print(f'[data] NOTE: the legacy test cache has no per-token '
                          f'document sidecar and EOS reconstruction was not '
                          f'possible ({_why}); the by-length readout stays '
                          'row-level')
        out = (test_ids, meta['vocab'], None, test_bnd, _out_docs)
        _memo_put(_WT103_TEST_MEMO, _mkey, out, _WT103_TEST_MEMO_CAP)
        return out if return_doc_ids else out[:4]
    # cold build: resolve the train tokenizer FIRST, then touch the test rows.
    # Resolve the tokenizer before touching the test rows, so a tokenizer/fit
    # mismatch fails before the held-out split is read.  `test_rows` is
    # populated after the validation block below, immediately before
    # tokenisation.
    test_rows = None
    _test_source = local_dataset_source('wikitext_test_raw')
    ds_test = None
    ds_train = None
    raw_txt = os.environ.get('WT103_RAW_TXT')
    _cands = []
    if raw_txt:
        _cands.append(raw_txt)
    _cands.append(os.path.join(os.path.dirname(os.path.abspath(cache_dir)), 'wt103_raw'))
    _cands.append('/root/wt103_raw')
    # tokenizer: identical frozen-train-prefix fit to load_wikitext.  The
    # token cap is taken from the paired train cache when it is unique, so
    # the test tokenizer is an exact function of the same train region.
    import glob as _glob
    _fit_cap = int(TOKENIZER_FIT_TOKENS)
    _fit_max_rows = 50000
    _train_meta_objs = []
    _train_meta_paths = []
    for _mp in _glob.glob(os.path.join(cache_dir, 'meta_v5_*.json')):
        try:
            _mobj = json.load(open(_mp, encoding='utf-8'))
        except Exception:
            continue
        if not isinstance(_mobj, dict):
            # A valid-JSON list/scalar is not train-cache metadata.  Keeping
            # it in the list made the tokenizer-cap/fingerprint loops call
            # .get() on a list and crash the test-cache build.
            print(f'[data] WARNING: train-cache metadata {_mp} is a '
                  f'{type(_mobj).__name__}, not an object — ignored for the '
                  'test-cache binding')
            continue
        _train_meta_objs.append(_mobj)
        _train_meta_paths.append(_mp)
    _caps = {int(_m['tokenizer_fit_token_cap']) for _m in _train_meta_objs
             if _m.get('tokenizer_fit_token_cap') is not None}
    _maxrows = {int(_m['tokenizer_fit_max_rows']) for _m in _train_meta_objs
                if _m.get('tokenizer_fit_max_rows') is not None}
    if len(_caps) == 1:
        _fit_cap = _caps.pop()
    if len(_maxrows) == 1:
        _fit_max_rows = _maxrows.pop()
    train_rows_source = None
    _train_source = local_dataset_source('wikitext_train_raw')
    for cand in _cands:
        _f = os.path.join(cand, 'wiki.train.raw')
        if os.path.exists(_f):
            train_rows_source = _read_raw_rows(_f)
            break
    if train_rows_source is None:
        _wt_source = wikitext_dataset_source()
        last_err = None
        for repo in (WIKITEXT_HF_REPO, WIKITEXT_HF_REPO_ALIAS):
            try:
                # TRAIN split only, for the frozen tokenizer fit.  Never reuse
                # the test dataset here: fitting the tokenizer on test text
                # would be leakage.  The revision pin is the same immutable
                # Hub commit used by the train/validation cache.
                ds_train = load_dataset(repo, WIKITEXT_HF_CONFIG, split='train', cache_dir=cache_dir, revision=_wt_source['revision'])
                _train_source = wikitext_dataset_source(repo=repo)
                break
            except Exception as e:
                last_err = e
        if ds_train is None:
            raise last_err if last_err is not None else RuntimeError('load_wikitest: no train source to fit the shared tokenizer')
        train_rows_source = list(ds_train['text'])
    # Same blank-row filtering as load_wikitext, otherwise the two tokenizer
    # fit corpora (and therefore the vocabularies) can differ.
    train_rows_source = [r for r in train_rows_source
                         if isinstance(r, str) and r.strip()]
    train_fit_rows, tok, n_fit_tokens = _wikitext_fit_rows(
        train_rows_source, vocab_size, _fit_cap, max_rows=_fit_max_rows)
    if tok is None or not train_fit_rows:
        raise RuntimeError('load_wikitest: the train split has no non-empty text rows to fit the shared tokenizer')
    # Bind the cold build to the EXISTING train cache before any test row is
    # read.  On the cache-hit path this check already exists; without it here,
    # a local dump / dataset revision that happens to fit the same vocab could
    # produce a test cache whose dedup index came from a different train
    # corpus, and the first (frozen) test evaluation would consume it.
    _fit_fp_build = _text_fp(train_fit_rows)
    _train_fit_fps_build = {_m['fit_fp'] for _m in _train_meta_objs
                            if _m.get('fit_fp')}
    if _train_fit_fps_build and _fit_fp_build not in _train_fit_fps_build:
        raise RuntimeError(
            'load_wikitest: the train source used for the test tokenizer has '
            f'fit-corpus fingerprint {_fit_fp_build!r}, which is absent from '
            'the train cache metas — the test split cannot be deduplicated '
            'against the corpus that was actually trained on; point '
            'WT103_RAW_TXT at the matching train dump or delete/rebuild both '
            'caches from one source')
    if _train_fit_fps_build:
        print(f'[data] test tokenizer fit is bound to the train cache (fit_fp={_fit_fp_build})')
    else:
        print('[data] NOTE: the train cache has no fit fingerprint; the cold test build cannot independently bind its train source')
    vocab = tok.get_vocab_size()
    eos_id = tok.token_to_id('<eos>')
    # vocabulary consistency with any existing train-side cache: a mismatch means
    # the two corpora builds cannot be paired — refuse rather than serve it.
    # A persisted train tokenizer is compared by FULL vocab identity (id->token
    # mapping), not merely by size: two BPE fits can share a vocab size with
    # different merges, in which case the id spaces are un-pairable.  Legacy
    # caches without a persisted tokenizer fall back to the size check.
    import glob as _glob
    _test_vocab = dict(tok.get_vocab())
    _train_tok_paths = _glob.glob(os.path.join(cache_dir, 'tok_v5_*.json'))
    for _tp in _train_tok_paths:
        try:
            from tokenizers import Tokenizer as _Tok
            _tt = _Tok.from_file(_tp)
        except Exception:
            continue
        if dict(_tt.get_vocab()) != _test_vocab:
            raise RuntimeError(f'load_wikitest: the freshly fitted test tokenizer is not identical to the cached train tokenizer ({_tp}) — same-size vocabularies can still use different merges, so the splits cannot be paired; delete wt103_cache/ and rebuild both')
    for _mp in _glob.glob(os.path.join(cache_dir, 'meta_v5_*.json')):
        try:
            _mt = json.load(open(_mp, encoding='utf-8'))
        except Exception:
            continue
        if not isinstance(_mt, dict):
            # A valid-JSON list/scalar is not train-cache metadata.  The
            # earlier builder loop already skips these; this cross-check must
            # use the same guard instead of crashing on ``.get``.
            continue
        _mv = _mt.get('vocab')
        if _mv is None:
            continue
        try:
            _mv = int(_mv)
        except (TypeError, ValueError, OverflowError):
            raise RuntimeError(
                f'load_wikitest: cached train corpus {_mp} records a '
                f'non-integer vocab ({_mt.get("vocab")!r}); the test/train '
                'tokenizer pairing cannot be verified — rebuild the cache')
        if _mv != int(vocab):
            raise RuntimeError(f'load_wikitest: the freshly fitted tokenizer (vocab={vocab}) disagrees with the cached train corpus (vocab={_mt.get("vocab")!r} in {_mp}) — the splits cannot be paired; delete wt103_cache/ and rebuild both')
    if not _train_meta_objs:
        raise RuntimeError('load_wikitest: no train/validation meta exists in the cache directory — build (or copy) the frozen train cache FIRST; the test tokenizer and the dedup index are functions of that train split, and touching test before it exists cannot be audited')
    # Validation is the SELECTION split.  A test document that duplicates a
    # validation document would leak selection data into the final evaluation,
    # so the held-out subset must be cleaned against validation as well as
    # train.  Read validation (already consumed by earlier phases, never test)
    # before the held-out rows, so the dedup index is complete first.
    val_rows_source = None
    _val_source = local_dataset_source('wikitext_validation_raw')
    for cand in _cands:
        _f = os.path.join(cand, 'wiki.valid.raw')
        if os.path.exists(_f):
            val_rows_source = _read_raw_rows(_f)
            print(f'[data] local wikitext-103 VALIDATION split at {_f} ({len(val_rows_source)} rows)')
            break
    if val_rows_source is None:
        _wt_source = wikitext_dataset_source()
        for repo in (WIKITEXT_HF_REPO, WIKITEXT_HF_REPO_ALIAS):
            try:
                ds_val = load_dataset(repo, WIKITEXT_HF_CONFIG, split='validation', cache_dir=cache_dir, revision=_wt_source['revision'])
                val_rows_source = list(ds_val['text'])
                _val_source = wikitext_dataset_source(repo=repo)
                break
            except Exception as e:
                last_err = e
        if val_rows_source is None:
            raise RuntimeError('load_wikitest: no validation-split source is available; the test subset cannot be proven disjoint from the selection split — provide wiki.valid.raw or the HF validation split')
    val_rows_source = [r for r in val_rows_source
                       if isinstance(r, str) and r.strip()]
    # Pair the held-out cleanup with the TRAIN cache's declared cleaning
    # protocol.  Using the current code defaults here would silently rebuild
    # the test index under a different shingle/band geometry if the code
    # default ever changed after the train cache was frozen.
    _paired_train_metas = [
        _m for _m in _train_meta_objs
        if not _m.get('fit_fp') or _m.get('fit_fp') == _fit_fp_build]
    _paired_protocols = {}
    for _m in _paired_train_metas:
        _p = dedup_protocol_from_meta(_m)
        if _p is not None:
            _pfp = dedup_protocol_fp(_p)
            _paired_protocols.setdefault(_pfp, _p)
    if len(_paired_protocols) > 1:
        raise RuntimeError(
            'load_wikitest: the paired train metas declare '
            f'{len(_paired_protocols)} different dedup protocols '
            f'({sorted(k[:16] for k in _paired_protocols)}) for the same fit '
            'corpus — cannot choose the held-out cleaning geometry; rebuild '
            'the train cache so one protocol defines the split')
    _pair_protocol = (next(iter(_paired_protocols.values()))
                      if _paired_protocols else None)
    if _pair_protocol is not None:
        _pp = dedup_protocol_problems(_pair_protocol)
        if _pp:
            raise RuntimeError(
                'load_wikitest: the paired train cache records an invalid dedup '
                'protocol (' + '; '.join(_pp) + '); rebuild it before touching '
                'the held-out split')
        _dedup_kwargs = dedup_kwargs_from_protocol(_pair_protocol)
        print('[data] held-out cleanup uses the train cache dedup protocol '
              f'({dedup_protocol_fp(_pair_protocol)[:16]}…)')
    else:
        _dedup_kwargs = {}
        print('[data] NOTE: no paired train dedup protocol found; the held-out '
              'cleanup uses the current code defaults (legacy cache pair)')
    # Screen the raw validation documents against train exactly as the
    # validation split itself was screened, then index only the documents that
    # could actually have been served for model selection.  Doing this BEFORE
    # the test read means a validation/train source mismatch is caught before
    # the held-out split is touched.
    _val_screen = _DocDeduplicator(**_dedup_kwargs)
    for _r in train_rows_source:
        if isinstance(_r, str) and _r.strip():
            _val_screen.is_duplicate(_r)
    _kept_val_docs = []
    for _r in val_rows_source:
        _dup, _kind = _val_screen.is_duplicate(_r)
        if not _dup:
            _kept_val_docs.append(_r)
    # The test/train check above is conservative (it uses the whole raw train
    # split), but the val/test index must contain EVERY raw validation document
    # that could have been served by the selection split.  Indexing only the
    # documents that survive our train screen could miss a validation document
    # the real (capped) validation stream retained, so index all raw
    # validation documents (de-duplicated among themselves) instead.
    _val_only = _DocDeduplicator(**_dedup_kwargs)
    _val_indexed_docs = 0
    for _r in val_rows_source:
        _dup_v, _kind_v = _val_only.is_duplicate(_r)
        if not _dup_v:
            _val_indexed_docs += 1
    # Bind the validation source to the cached validation documents.  The
    # cached val_doc_ids sidecar for every train meta with the SAME fit corpus
    # must be a subset of the documents this source yields; otherwise the
    # val/test dedup would be built from a different validation split than the
    # one used for model selection.
    _raw_val_ids = set(
        int(_doc_id_u64(_doc_exact_fp(_r))) for _r in val_rows_source)
    for _mpath in _train_meta_paths:
        try:
            _mtag = os.path.basename(_mpath)[len('meta_'):-len('.json')]
            _mobj = json.load(open(_mpath, encoding='utf-8'))
        except Exception:
            continue
        if not isinstance(_mobj, dict):
            continue
        if _mobj.get('fit_fp') and _mobj['fit_fp'] != _fit_fp_build:
            continue
        _vpath = os.path.join(cache_dir, f'val_doc_ids_{_mtag}.npy')
        if not os.path.exists(_vpath):
            continue
        try:
            _cached_val_ids = set(int(x) for x in np.load(_vpath).tolist())
        except Exception as _ve:
            raise RuntimeError(
                f'load_wikitest: cached validation document-id sidecar {_vpath} '
                f'is unreadable ({type(_ve).__name__}: {_ve}); cannot bind the '
                'test dedup index to the selection split') from _ve
        _missing_val = _cached_val_ids - _raw_val_ids
        if _missing_val:
            raise RuntimeError(
                f'load_wikitest: the validation source yields {len(val_rows_source)} '
                f'documents but does not contain {len(_missing_val)} document(s) '
                f'from the cached validation split {_vpath} — the test dedup index '
                'would be built from a different selection split; point '
                'WT103_RAW_TXT / the HF source at the split used to build the '
                'train cache, or delete/rebuild both caches from one source')
    # Tokenizer and fit-cap are now validated.  ONLY NOW read the held-out rows.
    for cand in _cands:
        _f = os.path.join(cand, 'wiki.test.raw')
        if os.path.exists(_f):
            test_rows = _read_raw_rows(_f)
            print(f'[data] local wikitext-103 TEST split at {_f} ({len(test_rows)} rows)')
            break
    if test_rows is None:
        _wt_source = wikitext_dataset_source()
        last_err = None
        for repo in (WIKITEXT_HF_REPO, WIKITEXT_HF_REPO_ALIAS):
            try:
                # Only the TEST split: never materialise train/validation here.
                ds_test = load_dataset(repo, WIKITEXT_HF_CONFIG, split='test', cache_dir=cache_dir, revision=_wt_source['revision'])
                _test_source = wikitext_dataset_source(repo=repo)
                break
            except Exception as e:
                last_err = e
        if ds_test is None:
            raise last_err if last_err is not None else RuntimeError('load_wikitest: no test-split source available (local dump wiki.test.raw or the HF dataset)')
        test_rows = ds_test['text']
    print(f'[data] tokenising TEST split (cap {n_seqs * seq_len + seq_len} tokens) ...')
    _trows = list(test_rows) if not isinstance(test_rows, list) else list(test_rows)
    # Freeze a near-duplicate-clean evaluation subset: any test document that
    # exactly or near-duplicates a TRAIN or VALIDATION document is excluded
    # before tokenisation.  Both indices are built from the earlier splits
    # only; test rows never decide what counts as a duplicate of train/val.
    _kept_test, _excluded, _test_dedup = _filter_test_documents(
        train_rows_source, _val_only, _trows, label='test',
        protocol=_pair_protocol)
    _removed_test = _excluded['test_exact'] + _excluded['test_near']
    if _removed_test:
        # Test-to-test cleanup is not train/val leakage, but it changes the
        # frozen slice composition and must be visible in its own record.
        print(f'[data] NOTE: {_removed_test} additional test documents were '
              'near/exact duplicates of earlier test documents')
    _trows = _kept_test
    _cap_test = n_seqs * seq_len + seq_len
    _test_lengths = []
    ids = _tokenise_row_chunks(
        [_trows[s:s + 10000] for s in range(0, len(_trows), 10000)],
        _cap_test, 'test', tok, eos_id, lengths_out=_test_lengths)
    n_have = min(len(ids) // seq_len, n_seqs)
    if n_have < n_seqs:
        # Hard failure.  The frozen plan declares n_seqs (16) as part
        # of the slice; silently evaluating 14 rows would change the estimand
        # and could still pass every downstream admission check.
        raise RuntimeError(f'load_wikitest: the test split yields only {n_have} full rows but {n_seqs} are required by the frozen slice; refusing to shrink the held-out panel silently (investigate the near-duplicate cleanup / raw dump or re-freeze explicitly)')
    test_batch = ids[:n_have * seq_len].reshape(n_have, seq_len)
    _validate_token_ids(test_batch, vocab, 'wikitext test_batch')
    print(f'[data] TEST rows = {n_have} (seq_len={seq_len}, vocab={vocab})')
    bnd_ids = sorted(boundary_token_ids(tok, eos_id))
    test_bnd = np.isin(test_batch, bnd_ids)
    # Reconstruct the document-id sidecar from the same capped stream, but only
    # up to the tokens actually scored (n_have full sequences).  A document is
    # recorded exactly when it contributed at least one evaluated token; this
    # is what makes test/train and test/val group disjointness independently
    # checkable on a later cache hit.
    _served_cap = int(n_have) * int(seq_len)
    _served_rows = [r for r in _trows if r.strip()]
    test_doc_ids = _doc_ids_for_capped_rows(
        _served_rows, _test_lengths, _served_cap)
    # The per-token mapping is what makes a document-level statistic honest:
    # a fixed row can split a document or merge two, so aggregating losses by
    # ROW is pseudoreplication.  This sidecar is tiny (one id per served
    # token) relative to the id cache.
    test_token_doc_ids = _token_doc_ids_for_capped_rows(
        _served_rows, _test_lengths, _served_cap)
    if int(test_token_doc_ids.size) != int(_served_cap):
        raise RuntimeError(
            f'load_wikitest: per-token document-id sidecar covers '
            f'{int(test_token_doc_ids.size)} tokens but the served test slice '
            f'has {int(_served_cap)}; refusing to write a misaligned mapping')
    if not np.array_equal(np.unique(test_token_doc_ids),
                          np.unique(test_doc_ids)):
        raise RuntimeError(
            'load_wikitest: per-token and per-document test sidecars name '
            'different document sets; the document-level evaluation mapping '
            'would be inconsistent')
    # Prove group isolation against every train/val cache sidecar present.  The
    # build-time dedup should already guarantee zero; this turns any silent
    # dedup/index mismatch into a hard refusal before the cache is written.
    _shared_train_docs = 0
    _shared_val_docs = 0
    _doc_scan_errors = []
    import glob as _glob3
    for _dp in _glob3.glob(os.path.join(cache_dir, 'train_doc_ids_v5_*.npy')):
        try:
            _shared_train_docs += int(np.intersect1d(np.load(_dp), test_doc_ids).size)
        except Exception as _de:
            _doc_scan_errors.append(f'{os.path.basename(_dp)}: {_de}')
    for _dp in _glob3.glob(os.path.join(cache_dir, 'val_doc_ids_*.npy')):
        try:
            _shared_val_docs += int(np.intersect1d(np.load(_dp), test_doc_ids).size)
        except Exception as _de:
            _doc_scan_errors.append(f'{os.path.basename(_dp)}: {_de}')
    if _doc_scan_errors:
        raise RuntimeError(
            'load_wikitest: cannot verify test/document group isolation because '
            f'a train/validation document-id sidecar is unreadable ({_doc_scan_errors[:3]})')
    if _shared_train_docs or _shared_val_docs:
        raise RuntimeError(
            f'load_wikitest: {_shared_train_docs} train and {_shared_val_docs} '
            'validation document ids occur in the held-out test slice after '
            'deduplication — refusing to write a group-leaking test cache')
    for _p, _a in ((te_path, test_batch), (tb_path, test_bnd),
                   (td_path, test_doc_ids), (ttd_path, test_token_doc_ids)):
        _fd, _tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(_p)), prefix=os.path.basename(_p) + '.', suffix='.tmp')
        try:
            with os.fdopen(_fd, 'wb') as _fh:
                np.save(_fh, _a)
            os.replace(_tmp, _p)
        except BaseException:
            try:
                os.unlink(_tmp)
            except OSError:
                pass
            raise
    _file_sigs = _cache_file_sigs({'test': te_path, 'bnd': tb_path,
                                   'test_doc_ids': td_path,
                                   'test_token_doc_ids': ttd_path})
    # Persist the test tokenizer BEFORE writing the meta, so the meta can bind
    # its exact bytes.  The cache identity therefore includes the full-file
    # sha256 (not only a hash of the fit TEXT, which two different BPE merge
    # sets can share), so a swapped tokenizer cannot serve an unpaired id
    # space.
    _tok_sha = None
    try:
        tok.save(tk_path)
        _tok_sha = file_sha256(tk_path)
    except Exception as _e:
        print(f'[data] NOTE: could not persist the test tokenizer ({type(_e).__name__}: {_e}) — a rebuild will re-fit it deterministically')
    # The test rows were read only after every earlier check; before writing the
    # frozen slice, refuse a mixed-upstream build (two different Hub commits
    # cannot define one held-out snapshot).
    _cold_revs = {str(s.get('revision')).lower()
                  for s in (_train_source, _val_source, _test_source)
                  if isinstance(s, dict) and s.get('provider') == 'huggingface'
                  and s.get('revision')}
    if len(_cold_revs) > 1:
        raise RuntimeError(
            'load_wikitest: train/validation/test were loaded from different '
            f'upstream revisions {sorted(_cold_revs)} — refusing to write a '
            'test cache that spans more than one corpus snapshot')
    _cold_pair_problems, _cold_pair_warnings = dataset_source_pair_problems(
        {'dataset_source': _test_source},
        [m for m in _train_meta_objs
         if not (m.get('fit_fp') and m.get('fit_fp') != _fit_fp_build)],
        label='test')
    _warn_dataset_source('test/train pair', _cold_pair_warnings)
    if _cold_pair_problems:
        raise RuntimeError('load_wikitest: ' + '; '.join(_cold_pair_problems))
    atomic_write_json(me_path, {
        'vocab': vocab,
        'src_fp': _ids_fp(test_batch),
        'test_fp': _ids_fp(test_batch),
        'bnd_fp': _ids_fp(np.ascontiguousarray(test_bnd).reshape(-1)),
        'fit_fp': _text_fp(train_fit_rows),
        'code_semantics': CODE_SEMANTICS,
        'read_purpose': purpose,
        # Which immutable upstream snapshot (or local raw dump) produced each
        # split.  The audit recomputes pair consistency instead of trusting the
        # build-time check.
        'dataset_source': _test_source,
        'train_dataset_source': _train_source,
        'validation_dataset_source': _val_source,
        'tokenizer_file': os.path.basename(tk_path),
        'tokenizer_file_sha256': _tok_sha,
        'tokenizer_vocab_sha256': _tokenizer_vocab_sha256(tok),
        'file_sigs': _file_sigs,
        'id_range': {'test': _array_id_range(test_batch)},
        'split_strategy': 'official_dataset_test_split_near_duplicate_filtered',
        # The held-out panel is a source-order prefix of the
        # near-duplicate-cleaned test split.  It is a frozen evaluation
        # snapshot, not a probability sample of WikiText test articles; the
        # sampling audit reports its coverage without ever re-reading raw test.
        'sampling': {
            'schema': 'wikitext_slice_sampling_v1',
            'test': {
                'design': 'source_order_prefix',
                'selection_rule': 'first_nonempty_rows_of_deduplicated_test_split_until_token_cap',
                'random_seed': None,
                'representativeness_claim_allowed': False,
                'population_claim_allowed': False,
                'source_rows': int(len(test_rows)),
                'rows_after_dedup': int(len(_trows)),
                'rows_served': int(test_doc_ids.size),
                'tokens_served': int(n_have) * int(seq_len),
                'tokens_tokenised': int(len(ids)),
                'token_cap': int(_cap_test),
                'cap_reached': bool(len(ids) >= int(_cap_test)),
                'sequences_served': int(test_batch.shape[0]),
                'sequences_requested': int(n_seqs),
                'claim_reason': (
                    'the frozen panel is the first evaluated rows of the test '
                    'split in source order, not a random sample of test '
                    'articles; it supports the frozen comparison, not an '
                    'unconditional population claim'),
            },
        },
        # See load_wikitext: the per-token map groups by raw LINE, not by
        # article.  The document-level readout must name/regroup that unit.
        'document_unit': 'line',
        'document_unit_is_article': False,
        'document_unit_definition': (
            'one non-empty raw WikiText row (heading/paragraph line); an '
            'article spans multiple rows and is delimited by top-level '
            '" = Title = " headings'),
        'split_fingerprints': {
            'fingerprint_algorithm': _IDS_FP_ALGO,
            'test': {'file': os.path.basename(te_path),
                     'ids_fp': _ids_fp(test_batch),
                     'boundary_fp': _ids_fp(np.ascontiguousarray(test_bnd).reshape(-1)),
                     'documents_fp': _ids_fp(test_doc_ids),
                     'token_documents_fp': _ids_fp(test_token_doc_ids),
                     'sequences': int(test_batch.shape[0]),
                     'seq_len': int(seq_len)},
            'paired_train': {
                'fit_fps': sorted({str(_m.get('fit_fp')) for _m in _train_meta_objs
                                   if _m.get('fit_fp')}),
                'code_semantics': sorted({str(_m.get('code_semantics'))
                                          for _m in _train_meta_objs
                                          if _m.get('code_semantics') is not None}),
                'tokenizer_vocab_sha256': sorted(
                    {str(_m.get('tokenizer_vocab_sha256'))
                     for _m in _train_meta_objs if _m.get('tokenizer_vocab_sha256')}),
            },
        },
        'shuffle_at_split': False,
        'tokenizer_fit_split': 'train',
        'tokenizer_fit_protocol': 'train_prefix_token_budget',
        'tokenizer_fit_token_cap': int(_fit_cap),
        'tokenizer_fit_max_rows': int(_fit_max_rows),
        'tokenizer_fit_tokens': int(n_fit_tokens),
        'near_duplicate_cleanup': {
            'accounting_schema': 'wikitext_dedup_accounting_v1',
            'policy': 'test_documents_deduplicated_against_train_and_validation',
            'protocol': _test_dedup.protocol(),
            'protocol_fp': _test_dedup.protocol_fp(),
            'train_exact_removed': int(_excluded['train_exact']),
            'train_near_removed': int(_excluded['train_near']),
            'test_exact_removed': int(_excluded['test_exact']),
            'test_near_removed': int(_excluded['test_near']),
            'test_self_cleanup_policy': 'test_documents_deduplicated_among_themselves',
            'near_hamming_threshold': int(_test_dedup.near_threshold),
            'bands': int(_test_dedup.bands),
            'shingle_words': int(_test_dedup.shingle_words),
            'max_shingles': int(_test_dedup.max_shingles),
            'source_documents': int(len(test_rows)),
            # Blank raw rows never enter the dedup index.  Recording the
            # non-empty count makes the cleaning ledger close exactly:
            # nonempty = kept + train/val/test exact+near removals.
            'source_documents_nonempty': int(sum(
                1 for _r in test_rows if isinstance(_r, str) and _r.strip())),
            'kept_documents': int(len(_trows)),
            'validation_cleanup': {
                'policy': 'validation_documents_deduplicated_against_train',
                'protocol': _test_dedup.protocol(),
                'protocol_fp': _test_dedup.protocol_fp(),
                'source_documents': int(len(val_rows_source)),
                'kept_documents': int(len(_kept_val_docs)),
                'indexed_documents': int(_val_indexed_docs),
                'duplicate_documents_removed': int(
                    len(val_rows_source) - len(_kept_val_docs)),
                'exact_removed_from_test': int(_excluded['validation_exact']),
                'near_removed_from_test': int(_excluded['validation_near']),
            },
        },
        'doc_ids': {
            'test_count': int(test_doc_ids.size),
            'test_fp': _ids_fp(test_doc_ids),
            'test_token_count': int(test_token_doc_ids.size),
            'test_token_fp': _ids_fp(test_token_doc_ids),
            'token_policy': 'per_token_document_ids_for_document_level_evaluation',
            'train_overlap': int(_shared_train_docs),
            'validation_overlap': int(_shared_val_docs),
            'disjoint': True,
            'policy': 'test_whole_documents_disjoint_from_train_and_validation',
            'paired_train_doc_fps': sorted(
                {(_m.get('doc_ids') or {}).get('train_fp')
                 for _m in _train_meta_objs
                 if (_m.get('doc_ids') or {}).get('train_fp')}),
            'paired_validation_doc_fps': sorted(
                {(_m.get('doc_ids') or {}).get('validation_fp')
                 for _m in _train_meta_objs
                 if (_m.get('doc_ids') or {}).get('validation_fp')}),
        },
        'test_sequences': int(test_batch.shape[0]),
        'seq_len': int(seq_len),
        'eos_id': int(eos_id),
    }, indent=0)
    print('[data] cached TEST token ids to disk (held-out split; final evaluation only)')
    _out = (test_batch, vocab, tok, test_bnd,
            test_token_doc_ids.reshape(int(n_have), int(seq_len)))
    return _out if return_doc_ids else _out[:4]

_FWEB_MEMO = {}
_FWEB_MEMO_CAP = 2

def _fineweb_disk_dtype(vocab):
    if vocab <= 65536:
        return np.uint16
    if vocab < 2 ** 31:
        return np.int32
    raise ValueError(f'vocab {vocab} exceeds the id range representable by the on-disk corpus cache')

def _disk_ids_chunk(buf, disk_dt, vocab):
    # Range check must run on the pre-cast Python ints: once the array is
    # uint16 its max is <= 65535 by construction, so checking it afterwards
    # can never fire (and an out-of-range id would already have wrapped on
    # numpy builds that silently overflow).  `_fineweb_disk_dtype` returns
    # uint16 only for vocab <= 65536, so this is a defensive guard.
    if buf:
        # Convert once to int64 and take the extrema with NumPy's C loops:
        # scanning a multi-million-element Python list twice is the slow part
        # of the staging write at scale.  The range check still runs on the
        # pre-cast values, so a uint16 wrap can never hide a bad id.
        _a = np.asarray(buf, dtype=np.int64)
        _mn, _mx = int(_a.min()), int(_a.max())
        if _mn < 0 or _mx >= int(vocab):
            raise ValueError(f'token id range [{_mn}, {_mx}] is outside vocab [0, {int(vocab)}) — refusing to cache a corrupted corpus')
        if disk_dt is np.uint16 and _mx > 65535:
            raise ValueError(f'token id {_mx} exceeds the uint16 cache range (vocab={vocab}) — widen _fineweb_disk_dtype before rebuilding')
        return _a if disk_dt is np.int64 else _a.astype(disk_dt)
    return np.empty(0, dtype=disk_dt)

def _i32_sidecar_path(path):
    """Identity-bound int32 sidecar path for a wide on-disk token cache."""
    st = os.stat(path)
    return (f'{path}.i32-{int(st.st_size)}-'
            f'{int(getattr(st, "st_mtime_ns", 0))}-'
            f'{int(getattr(st, "st_ctime_ns", int(st.st_ctime * 1e9)))}-'
            f'{int(getattr(st, "st_ino", 0))}.npy')


def _npy_serve_int32(path):
    mm = np.load(path, mmap_mode='r')
    src = np.asarray(mm)
    if not np.issubdtype(src.dtype, np.integer):
        raise TypeError(f'_npy_serve_int32: expected an integer token cache at {path}, got dtype {src.dtype}')
    if src.dtype in (np.uint8, np.uint16, np.int32):
        return src
    # Wider integer cache: narrow to int32 only after proving every id fits.
    # A silent wrap (e.g. 2**32 + 5 -> 5) would pass the downstream
    # vocab-range check for small vocabularies and corrupt the corpus.
    if src.size:
        _mn, _mx = int(src.min()), int(src.max())
        if _mn < 0 or _mx > 2 ** 31 - 1:
            raise ValueError(f'_npy_serve_int32: token ids in {path} span [{_mn}, {_mx}], outside the int32 transport range')
    # Serve an identity-bound int32 sidecar so workers share one mapped copy:
    # the returned values are the same exact integers and the fingerprint
    # stays on the int32 canonical form.
    try:
        _side = _i32_sidecar_path(path)
        if os.path.exists(_side):
            _n = np.load(_side, mmap_mode='r')
            if (int(_n.size) == int(src.size)
                    and tuple(_n.shape) == tuple(src.shape)
                    and np.dtype(_n.dtype) == np.dtype(np.int32)):
                return _n
        _dir = os.path.dirname(os.path.abspath(_side)) or '.'
        _fd, _tmp = tempfile.mkstemp(dir=_dir, prefix=os.path.basename(_side) + '.', suffix='.tmp')
        os.close(_fd)
        try:
            # Preserve the source shape: the sidecar is a drop-in replacement
            # for the loaded cache, and a same-size but differently shaped
            # cache must never be silently flattened to 1-D.
            _mm = np.lib.format.open_memmap(_tmp, mode='w+', dtype=np.int32, shape=tuple(src.shape))
            _flat_mm = _mm.reshape(-1)
            _flat_src = src.reshape(-1)
            _step = 1 << 24
            for _s in range(0, int(src.size), _step):
                _e = min(_s + _step, int(src.size))
                _flat_mm[_s:_e] = _flat_src[_s:_e]
            _mm.flush()
            del _flat_mm, _mm
            os.replace(_tmp, _side)
        except BaseException:
            try:
                os.unlink(_tmp)
            except OSError:
                pass
            raise
        return np.load(_side, mmap_mode='r')
    except (OSError, ValueError, TypeError):
        out = np.empty(src.shape, dtype=np.int32)
        for s in range(0, src.size, 1 << 24):
            e = min(s + (1 << 24), src.size)
            out[s:e] = src[s:e]
        return out

def _fineweb_local_files(env_value):
    """Resolve FWEDU_LOCAL_JSONL to the ordered list of local documents files."""
    import glob
    files = []
    for cand in str(env_value).split(os.pathsep):
        cand = cand.strip()
        if not cand:
            continue
        if os.path.isdir(cand):
            for pat in ('*.jsonl', '*.json', '*.txt'):
                files.extend(sorted(glob.glob(os.path.join(cand, pat))))
        elif any(ch in cand for ch in '*?['):
            files.extend(sorted(glob.glob(cand)))
        else:
            files.append(cand)
    return [f for f in files if os.path.isfile(f)]

def _fineweb_local_source_fingerprint(env_value):
    """Identity of the local stand-in source (paths, size, mtime, ctime, inode).

    The token cache's own `src_fp` only proves that the cached ids match the
    cache meta; it cannot see that the *source* files changed.  The signature
    changes whenever a source file is added/removed/replaced — ctime/inode
    catch even a same-size replacement that preserves mtime — so a rebuilt
    cache is non-confusable with a stale one."""
    files = _fineweb_local_files(env_value)
    if not files:
        return None
    h = hashlib.sha1()
    for path in files:
        try:
            st = os.stat(path)
        except OSError:
            return None
        h.update(os.path.abspath(path).encode('utf-8', 'replace'))
        h.update(b'\0')
        h.update(str(int(st.st_size)).encode('ascii'))
        h.update(b'\0')
        h.update(str(int(getattr(st, 'st_mtime_ns', int(st.st_mtime * 1e9)))).encode('ascii'))
        h.update(b'\0')
        h.update(str(int(getattr(st, 'st_ctime_ns', int(st.st_ctime * 1e9)))).encode('ascii'))
        h.update(b'\0')
        h.update(str(int(getattr(st, 'st_ino', 0))).encode('ascii'))
        h.update(b'\0')
    return h.hexdigest()[:16]

def _doc_exact_fp(text):
    h = hashlib.blake2b(digest_size=16)
    h.update(str(text).strip().encode('utf-8', errors='replace'))
    return h.hexdigest()


# The near-duplicate cleaning geometry is part of the DATA identity, not an
# implementation detail.  Two caches whose token ids are identical but whose
# exact/near filters used different shingle widths, band layouts or Hamming
# thresholds are different datasets: one may contain evaluation rows that the
# other removed.  Keep every parameter in one declared protocol so the cache
# metadata, the loader pairing checks and `audit_gates.py data_processing` can
# all compare the same object instead of trusting a policy string.
DEDUP_SIMHASH_SHINGLE_WORDS = 5
DEDUP_SIMHASH_MAX_SHINGLES = 128
DEDUP_PROTOCOL_SCHEMA = 'wikitext_exact_blake2b128_simhash64_lsh_v1'
DEDUP_PROTOCOL_DEFAULTS = {
    'schema': DEDUP_PROTOCOL_SCHEMA,
    'exact_hash': 'blake2b-128',
    'simhash_bits': 64,
    'shingle_unit': 'word',
    'shingle_words': DEDUP_SIMHASH_SHINGLE_WORDS,
    'max_shingles': DEDUP_SIMHASH_MAX_SHINGLES,
    'bands': 4,
    'near_hamming_threshold': 3,
    'scope': 'raw_line',
}


def _dedup_jsonable(value):
    """Normalise protocol values so a fingerprint is stable across containers.

    JSON metadata yields Python ints/floats, but a caller may hand the helper a
    NumPy scalar (e.g. from a parsed array).  Normalising before hashing keeps
    the same protocol from producing two different identities.
    """
    if isinstance(value, dict):
        return {str(k): _dedup_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dedup_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.dtype.kind in 'Mm':
            # `tolist()` would turn ns-resolution values into raw integers.
            return _json_clean(value)
        return _dedup_jsonable(value.tolist())
    if isinstance(value, np.generic):
        # `.item()` alone is not JSON-safe for every NumPy scalar type:
        # np.timedelta64 is an np.integer subclass that raises on int(),
        # np.longdouble returns another NumPy scalar, and np.datetime64
        # returns a Python date.  Reuse the artifact writer's canonical
        # converter so the protocol identity is stable for all of them.
        return _json_clean(value)
    # The remaining scalar types (bytes, Decimal/Fraction, non-finite floats)
    # also need the canonical JSON-safe image used by the artifact writer.
    return _json_clean(value)


def dedup_protocol_fp(protocol):
    """Stable SHA-256 digest of a dedup protocol object.

    The digest covers every key, so adding/changing any cleaning parameter
    changes the data identity even when the policy string stays the same.
    """
    if not isinstance(protocol, dict):
        raise ValueError(f'dedup protocol must be an object, got {type(protocol).__name__}')
    blob = json.dumps(_dedup_jsonable(protocol), sort_keys=True,
                      separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def dedup_protocol_problems(protocol):
    """Validate a recorded dedup protocol; returns a list of problems.

    ``near_hamming_threshold < bands`` is required: the band index is only
    exhaustive for pairs within the threshold when fewer bits differ than
    there are bands (pigeonhole).  A larger threshold would silently lose
    near duplicates instead of failing closed.
    """
    if not isinstance(protocol, dict):
        return [f'protocol must be an object, got {type(protocol).__name__}']
    problems = []
    required = {
        'schema', 'exact_hash', 'simhash_bits', 'shingle_unit',
        'shingle_words', 'max_shingles', 'bands', 'near_hamming_threshold',
        'scope',
    }
    missing = sorted(required - set(protocol))
    if missing:
        problems.append('missing field(s): ' + ', '.join(missing))
    if protocol.get('schema') != DEDUP_PROTOCOL_SCHEMA:
        problems.append(
            f"schema must be {DEDUP_PROTOCOL_SCHEMA!r}, got {protocol.get('schema')!r}")
    if protocol.get('exact_hash') != 'blake2b-128':
        problems.append(
            f"exact_hash must be 'blake2b-128', got {protocol.get('exact_hash')!r}")
    if protocol.get('shingle_unit') != 'word':
        problems.append(
            f"shingle_unit must be 'word', got {protocol.get('shingle_unit')!r}")
    if protocol.get('scope') not in ('raw_line', 'document'):
        problems.append(
            "scope must be 'raw_line' or 'document', got "
            f"{protocol.get('scope')!r}")

    def _int_field(name, lo, hi):
        value = protocol.get(name)
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            problems.append(f'{name} must be an integer, got {value!r}')
            return None
        value = int(value)
        if not (lo <= value <= hi):
            problems.append(f'{name} must be in [{lo}, {hi}], got {value}')
            return None
        return value

    _bits = _int_field('simhash_bits', 64, 64)
    if _bits is None and 'simhash_bits' in protocol:
        # The SimHash implementation is fixed at 64 bits; report the concrete
        # constraint rather than a generic range failure above.
        problems.append('simhash_bits must be exactly 64')
    _words = _int_field('shingle_words', 1, 1 << 20)
    _max_sh = _int_field('max_shingles', 1, 1 << 20)
    _bands = _int_field('bands', 1, 8)
    _near = _int_field('near_hamming_threshold', 0, 64)
    if _bands is not None and 64 % _bands:
        problems.append(
            f'bands must divide 64 so every bit is covered (got {_bands}); '
            'a partial band layout can miss near duplicates in the uncovered '
            'high bits')
    if _bands is not None and _near is not None and _near >= _bands:
        problems.append(
            'near_hamming_threshold must be < bands for an exhaustive LSH '
            f'candidate search (got threshold={_near}, bands={_bands})')
    if _words is not None and _max_sh is not None and _words > _max_sh:
        problems.append(
            f'shingle_words ({_words}) must not exceed max_shingles ({_max_sh})')
    return problems


def dedup_kwargs_from_protocol(protocol):
    """Translate a recorded protocol into ``_DocDeduplicator`` constructor args.

    Rebuilding a test/validation index from the TRAIN cache's protocol (rather
    than the current code defaults) is what makes the pair use one cleaning
    geometry.  Unknown extra keys are ignored here; the protocol digest still
    binds them.
    """
    problems = dedup_protocol_problems(protocol)
    if problems:
        raise ValueError('invalid dedup protocol: ' + '; '.join(problems))
    return {
        'near_threshold': int(protocol['near_hamming_threshold']),
        'bands': int(protocol['bands']),
        'shingle_words': int(protocol['shingle_words']),
        'max_shingles': int(protocol['max_shingles']),
        'scope': str(protocol['scope']),
    }


def dedup_protocol_from_meta(meta):
    """Return a cache's full dedup protocol object, or ``None`` for legacy.

    Train/validation metadata stores it under ``dedup.protocol``; held-out
    metadata stores the same object under ``near_duplicate_cleanup.protocol``.
    A legacy cache has only scattered count/threshold fields, so it cannot be
    compared for exact geometry.
    """
    if not isinstance(meta, dict):
        return None
    for _key in ('dedup', 'near_duplicate_cleanup'):
        _block = meta.get(_key)
        if isinstance(_block, dict):
            _proto = _block.get('protocol')
            if isinstance(_proto, dict):
                return _proto
    return None


def _doc_simhash(text, *, shingle_words=DEDUP_SIMHASH_SHINGLE_WORDS,
                 max_shingles=DEDUP_SIMHASH_MAX_SHINGLES):
    """Deterministic 64-bit SimHash over word shingles (near-duplicate key).

    The shingle set is sub-sampled on an evenly spaced grid so very long
    documents stay bounded; the same text always yields the same hash
    across processes (blake2b, no Python salted hash).
    """
    # Only the first 1M words can ever be used: streaming the matches with
    # islice keeps the working set bounded on pathologically long documents
    # (a full findall would materialise every word before truncating).
    import itertools as _it
    words = [m.group() for m in _it.islice(
        re.finditer(r"[\w']+", str(text).lower()), 1 << 20)]
    if not words:
        words = [str(text)[:512]]
    if len(words) < shingle_words:
        shingles = [' '.join(words)]
    else:
        count = len(words) - shingle_words + 1
        if count <= max_shingles:
            pos = range(count)
        else:
            pos = np.unique(np.linspace(0, count - 1, int(max_shingles), dtype=np.int64))
        shingles = [' '.join(words[int(i):int(i) + shingle_words]) for i in pos]
    hashes = np.fromiter(
        (int.from_bytes(hashlib.blake2b(sh.encode('utf-8', errors='replace'),
                                        digest_size=8).digest(), 'little')
         for sh in shingles),
        dtype=np.uint64, count=len(shingles))
    if hashes.size == 0:
        # Keep the pre-vectorised semantics: an empty shingle set has all-zero
        # per-bit sums, and `v >= 0` sets every bit (not the all-zero hash).
        return (1 << 64) - 1
    bits = ((hashes[:, None] >> np.arange(64, dtype=np.uint64)[None, :]) & 1)
    vec = bits.astype(np.int64) * 2 - 1
    out = 0
    for b, v in enumerate(vec.sum(axis=0).tolist()):
        if v >= 0:
            out |= (1 << b)
    return out


def _hamming64(a, b):
    return int((int(a) ^ int(b)).bit_count())


def _doc_id_u64(doc_id):
    h = hashlib.blake2b(str(doc_id).encode('utf-8', errors='replace'), digest_size=8)
    return np.uint64(int.from_bytes(h.digest(), 'little'))


# ---------------------------------------------------------------------------
# WikiText row units: line vs article
# ---------------------------------------------------------------------------
# The HF `wikitext-103-raw-v1` builder yields ONE ROW PER RAW LINE, including
# section headings and blank lines (see the upstream builder's
# `_generate_examples`).  A WikiText *article* therefore spans many rows: its
# title is a top-level heading (` = Title = `), its sections are inner headings
# (` = = Section = = `), and blank lines occur *inside* an article between
# sections.  The loaders have historically called a row a "document", which is
# a unit-of-analysis error: line-level dedup cannot see an article whose lines
# were reworded/reordered, and a line-level bootstrap resamples paragraphs,
# not independent articles.  These helpers make the unit explicit and let the
# audits (re)group served lines back into articles without touching token ids.
_WIKITEXT_TOP_HEADING_RE = re.compile(
    r'^\s*=\s*([^=\s](?:[^=]*[^=\s])?)\s*=\s*$')
# The same heading at the START of a decoded run.  A whole-article run
# decodes to " = Title =  body ..." (the per-line EOS tokens are skipped by
# decode, so there need not be a newline after the title); the full-line form
# above would reject it.  This prefix form deliberately excludes section
# headings (the character after the first "=" is another "=").
_WIKITEXT_TOP_HEADING_PREFIX_RE = re.compile(
    r'^\s*=\s*[^=\s](?:[^=]*?[^=\s])?\s*=')


def wikitext_article_groups(rows):
    """Group raw WikiText rows into articles by top-level headings.

    ``rows`` are the raw strings a loader sees (HF rows or a local
    ``wiki.*.raw`` dump).  Blank rows are ignored; every non-blank row that
    matches a top-level `` = Title = `` heading starts a new article, and all
    following non-heading rows (including `` = = Section = = `` headings)
    belong to it until the next top-level heading.  This is the actual
    WikiText segmentation: blank lines separate *sections/paragraph groups*
    inside an article, not articles (verified by the 60 top-level headings in
    the 60-article WikiText-2 validation split).

    Returns a dict with ``row_group`` (-1 for blank rows), ``groups`` (row
    index lists), ``n_articles``, and diagnostics.  A corpus without any
    top-level heading is not silently cut into one-line "articles": it is
    returned as a single group with ``valid=False`` so the caller can refuse
    or fall back explicitly.
    """
    try:
        _rows = list(rows)
    except TypeError:
        raise TypeError(
            f'wikitext_article_groups: rows must be iterable, got {type(rows).__name__}') from None
    n_rows = len(_rows)
    group_of = np.full(n_rows, -1, dtype=np.int64)
    groups = []
    heading_rows = []
    cur = []
    for _i, _raw in enumerate(_rows):
        if not isinstance(_raw, str):
            raise TypeError(
                'wikitext_article_groups: every row must be a string, got '
                f'{type(_raw).__name__} at row {_i}')
        _s = _raw.rstrip('\r\n')
        if not _s.strip():
            continue
        if _WIKITEXT_TOP_HEADING_RE.match(_s):
            if cur:
                groups.append(cur)
            heading_rows.append(_i)
            cur = [_i]
        else:
            if not cur:
                # A preamble before the first title (rare) is its own article
                # but carries no heading, which the diagnostics expose.
                cur = [_i]
            else:
                cur.append(_i)
    if cur:
        groups.append(cur)
    for _g, _idxs in enumerate(groups):
        group_of[np.asarray(_idxs, dtype=np.int64)] = _g
    group_sizes = [int(len(_g)) for _g in groups]
    group_has_heading = [
        bool(_WIKITEXT_TOP_HEADING_RE.match(_rows[int(_g[0])].rstrip('\r\n')))
        for _g in groups
    ]
    preamble_groups = int(sum(1 for _h in group_has_heading if not _h))
    has_headings = bool(heading_rows)
    problems = []
    if n_rows and not has_headings:
        problems.append(
            'no top-level " = Title = " heading was found; the corpus cannot '
            'be segmented into articles and is returned as one group')
    if preamble_groups and has_headings:
        # Rows before the first title are not part of any article; report them
        # instead of folding them into the first article silently.
        problems.append(
            f'{preamble_groups} headerless group(s) precede the first title')
    valid = bool(has_headings and not problems)
    return {
        'unit': 'article',
        'boundary_rule': 'top_level_heading',
        'row_group': group_of,
        'groups': groups,
        'heading_rows': heading_rows,
        'n_rows': int(n_rows),
        'n_nonempty_rows': int(sum(group_sizes)),
        'n_articles': int(len(groups)),
        'group_sizes': group_sizes,
        'group_has_heading': group_has_heading,
        'preamble_groups': int(preamble_groups),
        'has_headings': bool(has_headings),
        'max_group_rows': int(max(group_sizes)) if group_sizes else 0,
        'mean_group_rows': (float(np.mean(group_sizes)) if group_sizes else 0.0),
        'valid': bool(valid),
        'problems': problems,
    }
