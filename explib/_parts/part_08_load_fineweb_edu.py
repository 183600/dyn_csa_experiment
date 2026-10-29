
def load_fineweb_edu(seq_len, n_train_tokens, vocab_size=None, val_seqs=512,
                     cache_dir='./fineweb_cache', source='sample-10BT',
                     fit_docs=20000):
    """B-tier corpus: FineWeb-Edu, self-trained BPE.

    Discipline mirrors load_wikitext exactly: npy cache keyed by the full
    protocol tag, atomic writes, sha256 content fingerprint in the meta file,
    in-process memo keyed on (file size, mtime).  Differences forced by the
    scale band:
      - token ids are cached on disk at 2 bytes/token (uint16, guarded by the
        actual vocab) and served at that width; every consumer widens ids back
        to int64 per batch exactly as the int64 corpora always did, so yielded
        batches are bit-identical to the int64-storage form;
      - the corpus is built in one streaming pass: the tokenizer-fit documents
        are the TRAIN PREFIX (bounded by a frozen token cap), then train
        tokens continue up to n_train_tokens + seq_len, then the validation
        holdout draws val_seqs*seq_len + seq_len tokens from documents strictly
        after the train region.  Exact and near-duplicate documents are removed
        as whole groups across all phases, document ids are written next to
        the cache and their disjointness is re-checked on every hit.  The
        validation region is closed per protocol tag (never reused across
        caps); the partial final train batch is discarded between regions;
      - val_batch is int64 (eval paths feed targets straight to cross entropy).
    """
    if vocab_size is None:
        vocab_size = int(DATA_PROTOCOL_DEFAULTS['fineweb']['vocab_size'])
    vocab_size = _protocol_pos_int('vocab_size', vocab_size)
    os.makedirs(cache_dir, exist_ok=True)
    # A local stand-in corpus must never share a cache tag with the hub source:
    # otherwise a cache built from FWEDU_LOCAL_JSONL would be served on a later
    # hub run (the self-fingerprint cannot tell the two corpora apart).  The
    # marker only appears when the local path is active, so existing hub caches
    # keep their exact filenames.
    _local_env = os.environ.get('FWEDU_LOCAL_JSONL', '').strip()
    _local_tag = f'_loc{hashlib.sha1(_local_env.encode("utf-8")).hexdigest()[:8]}' if _local_env else ''
    _local_sig = _fineweb_local_source_fingerprint(_local_env) if _local_env else None
    _fw_source = fineweb_cache_source(source)
    tag = f'v2_sl{seq_len}_cap{n_train_tokens}_v{vocab_size}_vs{val_seqs}_fw{source}_fd{fit_docs}{_local_tag}'
    tr_path = os.path.join(cache_dir, f'train_ids_{tag}.npy')
    va_path = os.path.join(cache_dir, f'val_batch_{tag}.npy')
    vp_path = os.path.join(cache_dir, f'val_bnd_{tag}.npy')
    me_path = os.path.join(cache_dir, f'meta_{tag}.json')
    trd_path = os.path.join(cache_dir, f'train_doc_ids_{tag}.npy')
    vad_path = os.path.join(cache_dir, f'val_doc_ids_{tag}.npy')
    tk_path = os.path.join(cache_dir, f'tok_{tag}.json')
    # Only validation needs token-level groups; a 5B-token TRAIN sidecar
    # would be tens of GB and has no consumer.  Keep train document ids at the
    # document level.
    vad_tok_path = os.path.join(cache_dir, f'val_token_doc_ids_{tag}.npy')
    if all((os.path.exists(p) for p in (tr_path, va_path, vp_path, me_path,
                                        trd_path, vad_path))):
        _sig = _memo_file_sig({'train': tr_path, 'val': va_path, 'bnd': vp_path,
                               'meta': me_path, 'train_doc_ids': trd_path,
                               'val_doc_ids': vad_path, 'tokenizer': tk_path})
        _mkey = (os.path.abspath(cache_dir), tag, _sig, _local_sig)
        _hit = _FWEB_MEMO.get(_mkey)
        if _hit is not None:
            print('[data] serving fineweb token ids from the in-process memo (same cache files, already verified) ...')
            return _hit
        print('[data] loading cached fineweb token ids ...')
        meta = _read_cache_meta(me_path, 'FineWeb train/validation')
        if meta is None:
            print('[data] WARNING: the FineWeb train/validation metadata is '
                  'unreadable — REBUILDING the cache so a mixed or stale split '
                  'is never silently served')
            meta = {}
        train_ids = _npy_serve_int32(tr_path)
        _sigs = _cache_file_sigs({'train': tr_path, 'val': va_path, 'bnd': vp_path,
                                  'train_doc_ids': trd_path, 'val_doc_ids': vad_path})
        _fast = _cache_file_sigs_match(meta, _sigs)
        if meta.get('src_fp') != _ids_fp(train_ids):
            print('[data] WARNING: the cached fineweb train_ids do not match the fingerprint in the meta file — the cache is corrupt or was mixed across builds; REBUILDING it')
        elif _local_sig is not None and meta.get('local_source_sig') != _local_sig:
            print('[data] WARNING: the local FineWeb-Edu stand-in source changed since this cache was built — REBUILDING the cache (a stale corpus must not keep its old fingerprint)')
        else:
            val_batch = np.load(va_path).astype(np.int64)
            val_bnd = np.load(vp_path)
            _problems = _cached_split_problems(meta, train_ids, val_batch, val_bnd,
                                               meta.get('vocab'), seq_len=seq_len,
                                               fingerprints_trusted=False,
                                               ranges_trusted=_fast)
            _problems.extend(_tokenizer_file_problems(
                meta, tk_path, 'FineWeb train/validation tokenizer'))
            _src_problems, _src_warnings = _dataset_source_problems(
                meta, _fw_source, 'FineWeb train/validation')
            _problems.extend(_src_problems)
            _warn_dataset_source('FineWeb train/validation', _src_warnings)
            try:
                _trd = np.load(trd_path)
                _vad = np.load(vad_path)
                _shared_docs = int(np.intersect1d(_trd, _vad, assume_unique=True).size)
                if _shared_docs:
                    _problems.append(f'{_shared_docs} document ids occur in both train and validation')
                if meta.get('doc_ids', {}).get('disjoint') is not True:
                    _problems.append('cache metadata does not attest document-level disjointness')
                _mdoc = meta.get('doc_ids') or {}
                if _mdoc.get('train_fp') and _mdoc['train_fp'] != _ids_fp(_trd):
                    _problems.append(
                        'train document-id sidecar does not match the fingerprint in the cache metadata')
                if _mdoc.get('validation_fp') and _mdoc['validation_fp'] != _ids_fp(_vad):
                    _problems.append(
                        'validation document-id sidecar does not match the fingerprint in the cache metadata')
            except Exception as _de:
                _problems.append(f'document-id sidecars unreadable ({type(_de).__name__}: {_de})')
            _vr = meta.get('validation_region')
            if (not isinstance(_vr, dict)
                    or _vr.get('policy') != 'per_cap_closed_no_cross_cap_reuse'):
                _problems.append('cache does not declare a closed per-cap validation region')
            if (meta.get('tokenizer_fit_protocol') != 'train_prefix_token_budget'
                    or meta.get('tokenizer_fit_docs_in_train_prefix') is not True):
                _problems.append('cache does not attest a tokenizer fit inside the training prefix')
            if not meta.get('val_fp'):
                _problems.append(
                    'cache does not record a validation fingerprint; holdout '
                    'content cannot be bound to the split it was audited with')
            _dd_meta = meta.get('dedup')
            _vc = _dd_meta.get('validation_cleanup') if isinstance(_dd_meta, dict) else None
            if (not isinstance(_vc, dict)
                    or _vc.get('policy') != 'validation_docs_deduped_against_fit_and_train'):
                _problems.append(
                    'cache does not attest exact/near-duplicate cleanup of the '
                    'validation split against the tokenizer-fit and training documents')
            if _problems:
                _cache_warning('FineWeb train/validation split', _problems)
            else:
                if 'val_fp' not in meta:
                    print('[data] NOTE: this is a legacy FineWeb cache without a validation fingerprint; shape/range checks passed but split membership cannot be independently re-derived. New caches record val_fp.')
                audit_corpus_split(train_ids, val_batch, int(meta['vocab']),
                                   dataset='fineweb', meta=meta)
                _out = (train_ids, val_batch, int(meta['vocab']), None, val_bnd)
                _memo_put(_FWEB_MEMO, _mkey, _out, _FWEB_MEMO_CAP)
                return _out
    rows = _fineweb_docs(source)
    _dedup = _DocDeduplicator()
    _skipped = {'fit': {'exact': 0, 'near': 0},
                'train': {'exact': 0, 'near': 0},
                'validation': {'exact': 0, 'near': 0}}
    fit_pairs = []
    while len(fit_pairs) < int(fit_docs):
        try:
            doc_id, doc = next(rows)
        except StopIteration:
            break
        _dup, _kind = _dedup.is_duplicate(doc, fp=doc_id)
        if _dup:
            _skipped['fit'][_kind] += 1
            continue
        fit_pairs.append((doc_id, doc))
    _fit_cap = min(int(TOKENIZER_FIT_TOKENS), int(n_train_tokens))
    _fit_candidates = [d for _, d in fit_pairs]
    fit_text, tok, n_fit_tokens = _wikitext_fit_rows(
        _fit_candidates, vocab_size, _fit_cap, max_rows=len(_fit_candidates))
    if tok is None or not fit_text:
        raise RuntimeError('load_fineweb_edu: no non-empty documents are available for the tokenizer fit')
    # Keep only the documents that survived the token cap; they form the
    # train prefix below, so the tokenizer's fit corpus is a subset of the
    # corpus the model actually sees.
    fit_pairs = fit_pairs[:len(fit_text)]
    print(f'[data] training BPE tokenizer on {len(fit_text)} unique fineweb docs '
          f'({n_fit_tokens} tokens <= fit cap {_fit_cap}, vocab_size={vocab_size}) ...')
    # _wikitext_fit_rows already built ``tok`` on exactly this fit_text.
    vocab = tok.get_vocab_size()
    eos_id = tok.token_to_id('<eos>')
    disk_dt = _fineweb_disk_dtype(vocab)
    cap_train = int(n_train_tokens) + int(seq_len)
    cap_val = int(val_seqs) * int(seq_len) + int(seq_len)
    raw_dir = os.path.dirname(os.path.abspath(tr_path))

    def _raw_tmp(label):
        fd, tmp = tempfile.mkstemp(dir=raw_dir, prefix=f'.fineweb_{label}_{tag}.', suffix='.raw')
        os.close(fd)
        return tmp

    raw_train = _raw_tmp('train')
    raw_val = _raw_tmp('val')
    n_train_total = 0
    n_val_total = 0
    _marks = 0
    # Compact document-id accumulators: an unsigned-64 array converts to
    # NumPy without an object-by-object pass.
    import array as _array_mod
    train_doc_ids = _array_mod.array('Q')
    val_doc_ids = _array_mod.array('Q')

    try:
        buf = []
        # Documents already deduplicated for the TRAIN batch but left after the
        # token cap is reached.  They are not emitted by train and must seed
        # the validation stream instead of vanishing with the batch (which
        # could make a small source "run dry" even though unique docs remain).
        carry_pairs = []
        print(f'[data] tokenising fineweb train (cap {n_train_tokens} tokens, source {source}) ...')
        with open(raw_train, 'wb') as fh:
            def _write_train_pairs(pairs):
                nonlocal buf, n_train_total, _marks, carry_pairs
                carry_pairs = []
                if not pairs:
                    return False
                for _i, ((doc_id, _), enc) in enumerate(
                        zip(pairs, tok.encode_batch([d for _, d in pairs]))):
                    buf.extend(enc.ids)
                    buf.append(eos_id)
                    train_doc_ids.append(int(_doc_id_u64(doc_id)))
                    if len(buf) >= (1 << 22):
                        _disk_ids_chunk(buf, disk_dt, vocab).tofile(fh)
                        n_train_total += len(buf)
                        buf = []
                        if (n_train_total >> 26) > _marks:
                            _marks = n_train_total >> 26
                            print(f'    [train] {n_train_total} tokens')
                    if n_train_total + len(buf) >= cap_train:
                        carry_pairs = list(pairs[_i + 1:])
                        return True
                return n_train_total + len(buf) >= cap_train

            # The tokenizer-fit documents are the PREFIX of the training
            # stream, not a separate region: the tokenizer sees exactly the
            # documents the model's first tokens come from.
            stop = _write_train_pairs(fit_pairs)
            del fit_text, fit_pairs
            gc.collect()
            while not stop:
                docs = list(itertools.islice(rows, 4096))
                if not docs:
                    break
                pairs = []
                for doc_id, doc in docs:
                    # Whole-document dedup: a document that already appeared
                    # (fit or train) is never emitted again.
                    _dup, _kind = _dedup.is_duplicate(doc, fp=doc_id)
                    if _dup:
                        _skipped['train'][_kind] += 1
                    else:
                        pairs.append((doc_id, doc))
                stop = _write_train_pairs(pairs)
            if buf:
                _disk_ids_chunk(buf, disk_dt, vocab).tofile(fh)
                n_train_total += len(buf)
                buf = []
        n_train = min(n_train_total, cap_train)
        print(f'[data] train tokens = {n_train}  vocab = {vocab}')
        if n_train < cap_train:
            print(f'[data] WARNING: the source ran dry at {n_train} train tokens (< requested cap {cap_train}); runs configured for the advertised token target will train on fewer tokens than advertised')
        print('[data] tokenising fineweb validation holdout ...')
        with open(raw_val, 'wb') as fh:
            def _write_val_pairs(pairs):
                nonlocal buf, n_val_total
                if not pairs:
                    return False
                for (doc_id, _), enc in zip(pairs, tok.encode_batch([d for _, d in pairs])):
                    buf.extend(enc.ids)
                    buf.append(eos_id)
                    val_doc_ids.append(int(_doc_id_u64(doc_id)))
                    if len(buf) >= (1 << 22):
                        _disk_ids_chunk(buf, disk_dt, vocab).tofile(fh)
                        n_val_total += len(buf)
                        buf = []
                    if n_val_total + len(buf) >= cap_val:
                        return True
                return n_val_total + len(buf) >= cap_val

            stop = False
            while not stop:
                if carry_pairs:
                    # Already vetted by the train dedup pass; re-running the
                    # deduper would classify them as duplicates and skip them.
                    pairs = carry_pairs
                    carry_pairs = []
                else:
                    docs = list(itertools.islice(rows, 4096))
                    if not docs:
                        break
                    pairs = []
                    for doc_id, doc in docs:
                        _dup, _kind = _dedup.is_duplicate(doc, fp=doc_id)
                        if _dup:
                            _skipped['validation'][_kind] += 1
                        else:
                            pairs.append((doc_id, doc))
                stop = _write_val_pairs(pairs)
            if buf:
                _disk_ids_chunk(buf, disk_dt, vocab).tofile(fh)
                n_val_total += len(buf)
                buf = []
        n_val_tokens = min(n_val_total, cap_val)
        # 与 load_wikitext 同口径：块缓冲区含 +1 条序列的余量，钳到 val_seqs。
        n_val = min(n_val_tokens // seq_len, int(val_seqs))
        print(f'[data] val sequences = {n_val} (seq_len={seq_len}, contiguous holdout after the train region)')
        if n_val < int(val_seqs):
            print(f'[data] WARNING: the FineWeb validation holdout filled only '
                  f'{n_val} of {int(val_seqs)} requested sequence(s) '
                  f'({n_val_tokens} tokens for seq_len={seq_len}); downstream '
                  'panels configured for val_seqs rows will see a smaller '
                  'holdout — enlarge the holdout or lower val_seqs and record '
                  'the change')
        if n_val == 0:
            raise RuntimeError('load_fineweb_edu: the corpus ran dry before a single validation sequence was filled — raise n_train_tokens or provide a larger source')
        if n_train < seq_len + 1:
            raise RuntimeError(f'load_fineweb_edu: only {n_train} train tokens available (need >= seq_len+1 = {seq_len + 1}) — the source is too small for the requested cap')

        def _finalise_npy(raw_path, dst_path, count):
            # Write through a unique same-directory temp file, then replace
            # atomically.
            _fd, npy_tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(dst_path)),
                                            prefix=os.path.basename(dst_path) + '.', suffix='.tmp')
            os.close(_fd)
            try:
                mm = np.lib.format.open_memmap(npy_tmp, mode='w+', dtype=disk_dt, shape=(int(count),))
                done = 0
                with open(raw_path, 'rb') as rf:
                    while done < count:
                        take = min(1 << 22, count - done)
                        chunk = np.fromfile(rf, dtype=disk_dt, count=take)
                        if chunk.size < take:
                            raise RuntimeError(f'load_fineweb_edu: raw staging file {raw_path} is shorter than the encoded count — aborting the cache write')
                        mm[done:done + take] = chunk
                        done += take
                mm.flush()
                del mm
                os.replace(npy_tmp, dst_path)
            except BaseException:
                try:
                    os.unlink(npy_tmp)
                except OSError:
                    pass
                raise

        _finalise_npy(raw_train, tr_path, n_train)
        val_disk = np.fromfile(raw_val, dtype=disk_dt, count=n_val * seq_len)
        val_batch = val_disk.astype(np.int64).reshape(n_val, seq_len)
        _validate_token_ids(val_batch, vocab, 'fineweb val_batch')
        bnd_ids = sorted(boundary_token_ids(tok, eos_id))
        val_bnd = np.isin(val_batch, bnd_ids)
        print(f'[data] boundary tokens: {int(val_bnd.sum())} ({val_bnd.mean() * 100:.1f}% of val tokens)')
        _train_doc_arr = np.frombuffer(train_doc_ids, dtype=np.uint64).copy() if train_doc_ids else np.empty(0, dtype=np.uint64)
        _val_doc_arr = np.frombuffer(val_doc_ids, dtype=np.uint64).copy() if val_doc_ids else np.empty(0, dtype=np.uint64)
        _shared_docs = int(np.intersect1d(_train_doc_arr, _val_doc_arr, assume_unique=True).size)
        if _shared_docs:
            raise RuntimeError(f'load_fineweb_edu: {_shared_docs} document ids occur in both train and validation — refusing to write a leaking split')
        for _p, _a in ((va_path, val_batch), (vp_path, val_bnd),
                       (trd_path, _train_doc_arr), (vad_path, _val_doc_arr)):
            _fd, _tmp = tempfile.mkstemp(dir=raw_dir, prefix=os.path.basename(_p) + '.', suffix='.tmp')
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
        train_ids = _npy_serve_int32(tr_path)
        _validate_token_ids(train_ids, vocab, 'fineweb train_ids')
        _file_sigs = _cache_file_sigs({'train': tr_path, 'val': va_path, 'bnd': vp_path,
                                       'train_doc_ids': trd_path, 'val_doc_ids': vad_path})
        try:
            tok.save(tk_path)
        except Exception as _te:
            print(f'[data] NOTE: could not persist the FineWeb tokenizer ({type(_te).__name__}: {_te})')
        _tok_ident = tokenizer_identity(tok, tk_path if os.path.exists(tk_path) else None)
        atomic_write_json(me_path, {
            'vocab': vocab,
            'tokenizer_file': os.path.basename(tk_path),
            'tokenizer_file_sha256': _tok_ident.get('file_sha256'),
            'tokenizer_vocab_sha256': _tok_ident.get('vocab_sha256'),
            'code_semantics': CODE_SEMANTICS,
            'src_fp': _ids_fp(train_ids),
            'val_fp': _ids_fp(np.ascontiguousarray(val_batch).reshape(-1)),
            'split_fingerprints': {
                'fingerprint_algorithm': _IDS_FP_ALGO,
                'train': {'file': os.path.basename(tr_path),
                          'ids_fp': _ids_fp(train_ids),
                          'documents_fp': _ids_fp(_train_doc_arr),
                          'tokens': int(train_ids.size)},
                'validation': {'file': os.path.basename(va_path),
                               'ids_fp': _ids_fp(np.ascontiguousarray(val_batch).reshape(-1)),
                               'boundary_fp': _ids_fp(np.ascontiguousarray(val_bnd).reshape(-1)),
                               'documents_fp': _ids_fp(_val_doc_arr),
                               'sequences': int(val_batch.shape[0]),
                               'seq_len': int(seq_len)},
            },
            'file_sigs': _file_sigs,
            'id_range': {'train': _array_id_range(train_ids), 'val': _array_id_range(val_batch)},
            'doc_ids': {
                'train_count': int(_train_doc_arr.size),
                'validation_count': int(_val_doc_arr.size),
                'train_fp': _ids_fp(_train_doc_arr),
                'validation_fp': _ids_fp(_val_doc_arr),
                'overlap': _shared_docs,
                'disjoint': True,
                'policy': 'whole_document_exact_and_near_duplicate_filtered',
            },
            'dedup': {
                'fit': dict(_skipped['fit']),
                'train': dict(_skipped['train']),
                'validation': dict(_skipped['validation']),
                'validation_cleanup': {
                    'policy': 'validation_docs_deduped_against_fit_and_train',
                    'exact_removed': int(_skipped['validation'].get('exact', 0)),
                    'near_removed': int(_skipped['validation'].get('near', 0)),
                },
                **{'simhash_'+k: v for k, v in _dedup.stats().items() if k.startswith('near_') or k == 'policy'},
            },
            'local_source_sig': _local_sig,
            'dataset_source': _fw_source,
            'source': source,
            'fit_docs': fit_docs,
            'disk_dtype': np.dtype(disk_dt).str,
            'seq_len': int(seq_len),
            'n_train_tokens': int(n_train_tokens),
            'n_train_actual': int(n_train),
            'val_seqs': int(val_seqs),
            'val_sequences': int(val_batch.shape[0]),
            'split_strategy': 'stream_order_document_disjoint_deduplicated',
            'split_order': ['tokenizer_fit_prefix', 'train', 'validation'],
            'shuffle_at_split': False,
            'tokenizer_fit_split': 'train',
            'tokenizer_fit_protocol': 'train_prefix_token_budget',
            'tokenizer_fit_token_cap': int(_fit_cap),
            'tokenizer_fit_tokens': int(n_fit_tokens),
            'tokenizer_fit_docs_in_train_prefix': True,
            'train_region_end_tokens': int(n_train),
            'validation_region': {
                'policy': 'per_cap_closed_no_cross_cap_reuse',
                'start_token_approx': int(n_train_total),
                'end_token_approx': int(n_train_total + n_val_tokens),
                'note': ('this validation region is closed for this protocol tag; '
                         'never reuse it as another cap\'s validation and never '
                         'compare runs whose val_fp differs'),
            },
        }, indent=0)
        print('[data] cached fineweb token ids to disk (resume-safe)')
        audit_corpus_split(train_ids, val_batch, vocab, dataset='fineweb',
                           meta=json.load(open(me_path, encoding='utf-8')))
        _out = (train_ids, val_batch, vocab, tok, val_bnd)
        _memo_put(_FWEB_MEMO, (os.path.abspath(cache_dir), tag, _memo_file_sig({'train': tr_path, 'val': va_path, 'bnd': vp_path, 'meta': me_path, 'train_doc_ids': trd_path, 'val_doc_ids': vad_path, 'tokenizer': tk_path}), _local_sig), _out, _FWEB_MEMO_CAP)
        return _out
    finally:
        for _raw in (raw_train, raw_val):
            try:
                os.unlink(_raw)
            except OSError:
                pass

_CORPUS_AUDIT_MEMO = {}


def audit_corpus_split(train_ids, val_batch, vocab, *, dataset='wikitext',
                       meta=None, max_scan_tokens=2_000_000, sample_rows=64,
                       shift_warn=0.25, shift_fail=0.75, strict=True,
                       passage_shingle=32, min_scan_fraction=0.25,
                       max_val_shingle_overlap=0.25,
                       max_passage_excess_overlap=0.10,
                       max_val_row_shingle_overlap=0.50):
    """Hard pre-training gate for a cached/served token split.

    Runs before any model fitting and raises on a hard leakage/data-quality
    failure: out-of-vocabulary ids, non-finite or empty holdouts, duplicate
    rows *across* splits (exact match on a deterministic sample of the train
    stream), long near-duplicate passages shared by train and validation,
    non-disjoint document ids when the cache records them, and materially
    shifted holdout token distributions.  A clean report is cached by the two
    content fingerprints, so resume phases do not rescan.

    The exact-row and passage scans cover at most ``max_scan_tokens`` of
    train.  If that is less than ``min_scan_fraction``, a clean sample is
    reported as ``warnings`` (not ``ok``): it is not a proof about the
    unscanned region.  ``max_passage_excess_overlap`` subtracts the corpus's
    own repeated-passage baseline before the set-level passage threshold is
    applied, so boilerplate does not masquerade as split leakage.
    """
    import data_guard as _DG

    def _gate_float(name, value):
        # NaN compares False against every threshold, silently disabling the
        # gate (and a negative threshold is not a meaningful probability).
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f'{name} must be a finite non-negative number, got {value!r}')
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f'{name} must be a finite non-negative number, got {value!r}') from None
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f'{name} must be a finite non-negative number, got {value!r}')
        return value

    def _gate_int(name, value):
        parsed = _as_seed_int(value)
        if parsed is None or parsed < 1:
            raise ValueError(f'{name} must be a positive integer, got {value!r}')
        return parsed

    shift_warn = _gate_float('shift_warn', shift_warn)
    shift_fail = _gate_float('shift_fail', shift_fail)
    if shift_warn > shift_fail:
        raise ValueError(
            f'shift_warn ({shift_warn}) must not exceed shift_fail ({shift_fail})')
    sample_rows = _gate_int('sample_rows', sample_rows)
    max_scan_tokens = _gate_int('max_scan_tokens', max_scan_tokens)
    passage_shingle = _gate_int('passage_shingle', passage_shingle)
    min_scan_fraction = _gate_float('min_scan_fraction', min_scan_fraction)
    max_val_shingle_overlap = _gate_float(
        'max_val_shingle_overlap', max_val_shingle_overlap)
    max_passage_excess_overlap = _gate_float(
        'max_passage_excess_overlap', max_passage_excess_overlap)
    max_val_row_shingle_overlap = _gate_float(
        'max_val_row_shingle_overlap', max_val_row_shingle_overlap)
    for _name, _val in (('min_scan_fraction', min_scan_fraction),
                        ('max_val_shingle_overlap', max_val_shingle_overlap),
                        ('max_passage_excess_overlap', max_passage_excess_overlap),
                        ('max_val_row_shingle_overlap', max_val_row_shingle_overlap)):
        if _val > 1.0:
            raise ValueError(f'{_name} must be a fraction in [0, 1], got {_val!r}')
    tr = np.asarray(train_ids).reshape(-1)
    va = np.asarray(val_batch)
    report = {
        'dataset': dataset,
        'train_tokens': int(tr.size),
        'val_shape': list(va.shape),
        'vocab': int(vocab),
        'hard_failures': [],
        'warnings': [],
        'meta_split_strategy': (meta or {}).get('split_strategy'),
    }
    # The cached verdict depends on the split attestation and on every gate
    # parameter, not only on the two token arrays.  Omitting `meta` let a
    # clean report for one document-isolation attestation be served for a
    # different (or missing) one; omitting `strict`/thresholds let a lenient
    # call satisfy a later strict one.
    _meta_fp = (hashlib.sha256(
        json.dumps(meta, sort_keys=True, default=str).encode('utf-8')).hexdigest()[:16]
        if isinstance(meta, dict) else None)
    key = (dataset, int(vocab), _ids_fp(tr),
           _ids_fp(np.ascontiguousarray(va).reshape(-1)) if va.size else None,
           _meta_fp, float(shift_warn), float(shift_fail),
           int(sample_rows), int(max_scan_tokens), bool(strict),
           int(passage_shingle), float(min_scan_fraction),
           float(max_val_shingle_overlap), float(max_passage_excess_overlap),
           float(max_val_row_shingle_overlap))
    _hit = _CORPUS_AUDIT_MEMO.get(key)
    if _hit is not None:
        return _hit
    try:
        _validate_token_ids(tr, vocab, f'{dataset} train_ids')
        _validate_token_ids(va, vocab, f'{dataset} val_batch')
    except (TypeError, ValueError, OverflowError) as e:
        report['hard_failures'].append(str(e))
    if va.ndim != 2 or va.shape[0] == 0 or va.size == 0:
        report['hard_failures'].append(f'validation split is empty or not 2-D (shape {va.shape})')
    if tr.size < 2:
        report['hard_failures'].append('training stream has fewer than 2 tokens')
    # No NaN/Inf can hide in an integer token cache, but a float cache (future
    # feature matrix) is checked explicitly.
    if not np.issubdtype(tr.dtype, np.integer):
        _bad = int(np.size(tr) - np.isfinite(tr).sum())
        report['nonfinite_train'] = _bad
        if _bad:
            report['hard_failures'].append(f'{_bad} NaN/Inf values in train')
    if not np.issubdtype(va.dtype, np.integer):
        _bad = int(np.size(va) - np.isfinite(va).sum())
        report['nonfinite_val'] = _bad
        if _bad:
            report['hard_failures'].append(f'{_bad} NaN/Inf values in validation')

    if va.ndim == 2 and va.size:
        _dup = int(_DG.duplicate_mask(va).sum())
        report['duplicate_rows_within_validation'] = _dup
        if _dup:
            report['warnings'].append(f'{_dup} duplicate validation rows')

    # Exact cross-split row overlap on a uniformly-spaced sample of the train
    # stream.  Small caches are scanned completely; large ones report the
    # scanned fraction instead of pretending the sample is a proof.
    overlap = None
    if tr.size and va.ndim == 2 and va.shape[0]:
        import audit_gates as _AG
        _rows = np.unique(np.linspace(0, va.shape[0] - 1,
                                      min(int(sample_rows), va.shape[0]),
                                      dtype=np.int64))
        _sample = np.asarray(va)[_rows]
        _nseg = 16
        _seg_len = max(int(max_scan_tokens) // _nseg,
                       int(_sample.shape[1]) + 1)
        if tr.size <= max(int(max_scan_tokens), _nseg * _seg_len):
            _segments = [tr]
            _scanned = int(tr.size)
        else:
            # 16 equally spaced windows are disjoint because the start spacing
            # is >= _seg_len once tr.size >= _nseg * _seg_len; otherwise the
            # same validation row could be counted by several windows.
            _starts = np.linspace(0, tr.size - _seg_len, _nseg, dtype=np.int64)
            _segments = [tr[int(s):int(s) + _seg_len] for s in _starts]
            _scanned = int(sum(s.size for s in _segments))
        # Count the UNION of matched validation rows over the windows.  A
        # repeated training row can occur in several disjoint windows, so
        # summing per-window unique counts would report more overlapping
        # validation rows than were sampled.  Join the windows with a token id
        # that cannot occur in either side: any window spanning the separator
        # then fails to equal a validation row, while each real occurrence is
        # still seen.
        if len(_segments) == 1:
            _scan = _segments[0]
        else:
            _sentinel = max(int(np.max(tr)) if tr.size else -1,
                            int(np.max(_sample)) if _sample.size else -1) + 1
            _scan = np.concatenate([
                np.concatenate([np.asarray(_seg).reshape(-1),
                                np.asarray([_sentinel], dtype=np.int64)])
                for _seg in _segments])
        _shared = int(_AG._row_overlap_count(_scan, _sample))
        overlap = _shared
        report['exact_row_overlap_in_sample'] = int(_shared)
        report['sampled_val_rows'] = int(_rows.size)
        report['train_tokens_scanned'] = int(_scanned)
        report['train_tokens_scanned_fraction'] = round(
            float(_scanned) / float(tr.size), 6) if tr.size else 1.0
        report['exact_row_scan_sufficient'] = bool(
            report['train_tokens_scanned_fraction'] >= float(min_scan_fraction))
        if _shared:
            report['hard_failures'].append(
                f'{_shared} sampled validation rows occur verbatim in train')
        elif not report['exact_row_scan_sufficient']:
            report['warnings'].append(
                'exact-row overlap scan covered only '
                f'{report["train_tokens_scanned_fraction"]:.1%} of train; '
                'a clean sample is not a clean-split proof')
        # A validation row can reuse a long training passage without being an
        # exact row match.  Audit near-duplicate shingles on the same train
        # windows and on (up to) 4096 validation rows.  The helper subtracts a
        # train-internal repetition baseline and reports coverage explicitly.
        _pva = np.asarray(va)
        if _pva.shape[0] > 4096:
            _pidx = np.unique(np.linspace(0, _pva.shape[0] - 1, 4096,
                                          dtype=np.int64))
            _pva = _pva[_pidx]
            report['passage_val_rows_total'] = int(va.shape[0])
            report['passage_val_rows_audited'] = int(_pidx.size)
        try:
            _passage = _AG._passage_overlap_from_segments(
                _segments, _pva, shingle=passage_shingle,
                scanned_tokens=_scanned, total_train_tokens=int(tr.size),
                min_scan_fraction=min_scan_fraction,
                max_val_shingle_overlap=max_val_shingle_overlap,
                max_excess_overlap=max_passage_excess_overlap,
                max_val_row_overlap=max_val_row_shingle_overlap)
            report['passage_overlap'] = _passage
            if _passage.get('status') == 'fail':
                report['hard_failures'].append(str(_passage.get(
                    'note', 'near-duplicate train/validation passage overlap')))
            elif _passage.get('status') == 'not_verifiable':
                report['warnings'].append(str(_passage.get(
                    'note', 'passage overlap scan incomplete')))
        except Exception as _pe:
            report['hard_failures'].append(
                'passage-overlap audit failed '
                f'({type(_pe).__name__}: {_pe})')

    # Document-level proof, when the cache carries it.
    _docs = (meta or {}).get('doc_ids') or {}
    if _docs:
        report['documents'] = {k: _docs.get(k) for k in
                               ('train_count', 'validation_count', 'overlap',
                                'disjoint', 'policy')}
        if _docs.get('disjoint') is not True or int(_docs.get('overlap', 0) or 0) != 0:
            report['hard_failures'].append('cache does not attest document-disjoint train/validation splits')
    elif dataset == 'fineweb':
        report['warnings'].append('legacy FineWeb cache has no document-id disjointness proof')

    # Training-only reference distribution; the holdout never defines it.
    if tr.size and va.size:
        try:
            import audit_gates as _AG
            _tr_s = _AG._sample_tokens(tr, cap=200000, seed=0)
            _va_s = _AG._sample_tokens(va, cap=200000, seed=1)
            _shift = _DG.categorical_shift_report(_tr_s, _va_s)
            _js = float(_shift.get('js_divergence', 0.0))
            report['train_val_token_shift'] = _shift
            if _js >= float(shift_fail):
                report['hard_failures'].append(
                    f'train/validation token JS divergence {_js:.3f} >= {shift_fail}')
            elif _js >= float(shift_warn):
                report['warnings'].append(
                    f'train/validation token JS divergence {_js:.3f} >= {shift_warn}')
        except Exception as _se:
            report['warnings'].append(f'token-shift audit unavailable ({type(_se).__name__}: {_se})')

    report['status'] = 'fail' if report['hard_failures'] else (
        'warnings' if report['warnings'] else 'ok')
    if report['hard_failures'] and strict:
        raise RuntimeError(
            f'[{dataset}] DATA GATE REFUSED this split: ' +
            '; '.join(report['hard_failures']) +
            ' — fix the split/cache before training (no model has been fitted).')
    _CORPUS_AUDIT_MEMO[key] = report
    return report


def assert_shared_validation_fp(records, *, context='comparison'):
    """Refuse a cross-run comparison whose records were scored on different
    validation slices (e.g. reusing a small cap's validation region under a
    larger cap).  Returns the single shared fingerprint."""
    fps = set()
    for r in records:
        if isinstance(r, dict):
            # Provenance is untrusted: a list/string in the old nested field
            # used to raise AttributeError on .get() and abort the comparison
            # before the fingerprints could be checked.
            _prov = r.get('provenance')
            _data = _prov.get('data') if isinstance(_prov, dict) else None
            _nested_fp = _data.get('val_fp') if isinstance(_data, dict) else None
            _fp = r.get('val_fp') or _nested_fp
            if _fp:
                fps.add(str(_fp))
    if len(fps) > 1:
        raise ValueError(f'{context}: records use {len(fps)} different validation '
                         f'slices {sorted(fps)}; their numbers are not comparable')
    return next(iter(fps)) if fps else None


def load_run_corpus(cfg):
    """Serve the corpus tuple for a run cfg (data-source switch).

    A cfg without a 'dataset' key keeps WikiText-103 with exactly the arguments
    the drivers have always passed — behaviour is bit-identical to the former
    hard-wired load_wikitext call, so no existing panel, fingerprint or record
    is touched.  'dataset': 'fineweb' selects the B-tier corpus
    with its own cache directory, tokenizer width,
    source name and fit protocol."""
    name = cfg.get('dataset') or 'wikitext'
    # Resolve every scale knob from declared fields.  The loader receives the
    # resolved values, so a missing/typoed seq_len or vocab fails here instead
    # of silently training the historical current-scale shape.
    proto = resolve_data_protocol(cfg, dataset=name)
    if name == 'wikitext':
        return load_wikitext(proto['seq_len'], proto['n_train_tokens'],
                             vocab_size=proto['vocab_size'],
                             val_seqs=proto['val_seqs'],
                             cache_dir=proto['cache_dir'])
    if name == 'fineweb':
        return load_fineweb_edu(proto['seq_len'], proto['n_train_tokens'],
                                vocab_size=proto['vocab_size'],
                                val_seqs=proto['val_seqs'],
                                cache_dir=proto['cache_dir'],
                                source=cfg.get('fw_source', 'sample-10BT'),
                                fit_docs=_protocol_pos_int(
                                    'fw_fit_docs', cfg.get('fw_fit_docs', 20000)))
    raise ValueError(f"load_run_corpus: unknown dataset {name!r} (expected 'wikitext' or 'fineweb')")

def _json_clean(obj):
    """Recursively replace non-finite floats with null and make NumPy values
    JSON-safe.  Standard JSON has no NaN/Infinity tokens; writing them makes a
    damaged curve indistinguishable from a real value on strict parsers."""
    if isinstance(obj, dict):
        return {str(k): _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_clean(obj.tolist())
    if isinstance(obj, np.generic):
        return _json_clean(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def atomic_write_json(path, obj, indent=2):
    text = json.dumps(_json_clean(obj), indent=indent, default=float, allow_nan=False)
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=os.path.basename(path) + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

def atomic_write_text(path, text):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=os.path.basename(path) + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

def locked_json_read(path):
    """Consistent read of a shared JSON file under the same flock the writers
    use ({} when absent).  Pure read: never creates or rewrites the file."""
    try:
        import fcntl
    except ImportError:
        fcntl = None
    lock_f = None
    if fcntl is not None:
        try:
            lock_f = open(path + '.lock', 'a')
            fcntl.flock(lock_f, fcntl.LOCK_SH)
        except OSError:
            # A missing/unwritable parent directory means the file is absent
            # (or unreadable); a read must return {} rather than crash.
            lock_f = None
    try:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, encoding='utf-8') as f:
                obj = json.load(f)
        except Exception as _e:
            print(f'[resume] FATAL: {path} exists but cannot be parsed ({type(_e).__name__}: {_e}).  Refusing to start with a silently RESET panel — move it aside to start fresh.')
            raise
        if not isinstance(obj, dict):
            # Every caller indexes the result by cell key.  Returning a parsed
            # list/scalar here made the first .get() crash (or, worse, be
            # treated as an empty panel); fail with the same contract as the
            # update path instead.
            raise ValueError(
                f'{path} parsed as a {type(obj).__name__}, not a JSON object; '
                'refusing to resume from it')
        return obj
    finally:
        if lock_f is not None:
            try:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
            finally:
                lock_f.close()


def locked_json_update(path, mutfn, indent=2):
    """Load-modify-write a JSON file under an exclusive flock.

    Parallel cell workers (par.py) all funnel their summary.json mutations
    through this helper, so two children finishing at the same moment can
    never lose each other's records.  mutfn(obj) mutates the loaded object in
    place (or returns a replacement); a missing file starts from {}.  The
    write itself stays atomic (atomic_write_json), so a crash mid-update
    leaves the previous content intact.
    """
    try:
        import fcntl
    except ImportError:
        fcntl = None
    lock_f = None
    # The lock file (and later the JSON) must be creatable for a nested path
    # that does not exist yet.
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if fcntl is not None:
        lock_f = open(path + '.lock', 'a')
        fcntl.flock(lock_f, fcntl.LOCK_EX)
    try:
        obj = {}
        if os.path.exists(path):
            try:
                with open(path, encoding='utf-8') as f:
                    obj = json.load(f)
            except Exception as _e:
                print(f'[resume] FATAL: {path} exists but cannot be parsed ({type(_e).__name__}: {_e}).  Refusing to overwrite it with a fresh object — move it aside to start fresh.')
                raise
            if not isinstance(obj, dict):
                # A valid-JSON list/scalar has no keys for the updater's
                # mutfn.  The old path called it anyway (crashing inside .get)
                # and could overwrite whatever the operator stored there.
                raise ValueError(
                    f'{path} parsed as a {type(obj).__name__}, not a JSON '
                    'object; refusing to update/overwrite it')
        rep = mutfn(obj)
        obj = obj if rep is None else rep
        atomic_write_json(path, obj, indent=indent)
        return obj
    finally:
        if lock_f is not None:
            try:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
            finally:
                lock_f.close()


def atomic_write_csv(path, header, rows):
    import io as _io
    # A one-shot iterator cannot be scanned for header inference and then
    # iterated again; materialise it once so no rows are silently dropped.
    if not isinstance(rows, (list, tuple)):
        rows = list(rows)
    if header is None:
        header = []
        seen = set()
        for r in rows:
            for k in r.keys() if isinstance(r, dict) else r:
                if k not in seen:
                    seen.add(k)
                    header.append(k)
    cols = list(header)
    buf = _io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in rows:
        if isinstance(r, dict):
            w.writerow([r.get(k, '') for k in cols])
        else:
            w.writerow(list(r))
    atomic_write_text(path, buf.getvalue())

def new_experiment_id(prefix='run'):
    """A collision-resistant id for one experiment attempt.

    Uses wall-clock time + pid + OS entropy (never the Python/NumPy/Torch RNG
    streams, so taking an id cannot perturb a seeded run).  The id is written
    into the record and the append-only experiment log, making repeated
    attempts of the same (variant, seed, run_cfg) distinguishable."""
    return f'{prefix}-{time.strftime("%Y%m%dT%H%M%S", time.gmtime())}-{os.getpid()}-{os.urandom(4).hex()}'

def append_jsonl(path, record):
    """Append one JSON object to a shared JSONL log under an exclusive flock.

    Append-only tracking is deliberately separate from summary.json: summary's
    keys are the resume identity `variant::seed`, so a rerun legitimately
    overwrites an old value.  The JSONL ledger keeps every attempt (including
    failures and cap truncations), which is what makes the paper trail
    auditable after the fact.  A write failure is the caller's to report; this
    helper never swallows a partially written line (single write + flush)."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    line = json.dumps(_json_clean(_jsonable(record)), sort_keys=True, ensure_ascii=False, default=float, allow_nan=False) + '\n'
    try:
        import fcntl
    except ImportError:
        fcntl = None
    with open(path, 'a', encoding='utf-8') as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def log_experiment_event(outdir, event):
    """Append a trace event to `outdir/experiments.jsonl` (never raises).

    Tracking is diagnostic infrastructure: a full disk or a read-only archive
    must not destroy a completed training cell.  The warning names the path so
    the missing ledger is visible rather than silent."""
    try:
        path = os.path.join(outdir, 'experiments.jsonl')
        append_jsonl(path, event)
        return path
    except Exception as _e:
        print(f'[track] WARNING: could not append to {os.path.join(outdir, "experiments.jsonl")} ({type(_e).__name__}: {_e}) — the training record itself is unaffected')
        return None

# Keys that describe *when/where/with which code* an artifact was produced,
# rather than the analysis result itself.  They are excluded from the content
# digest so two identical analyses run on two machines compare equal, while a
# changed statistic/verdict does not.
_ANALYSIS_VOLATILE_KEYS = frozenset({
    'kind', 'created_at', 'code', 'provenance', 'content_sha256', 'inputs',
})


def _analysis_repo_root():
    """Repository root used to store input paths relative to the checkout."""
    p = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        if os.path.exists(os.path.join(p, 'pyproject.toml')):
            return p
        p = os.path.dirname(p)
    return os.getcwd()


def analysis_content_sha256(record):
    """Canonical digest of the RESULT body of a stamped analysis.

    The digest is deliberately independent of timestamp, git state, installed
    environment and absolute paths: a reproduced analysis must be comparable
    across machines.  Unknown values are converted with the same helpers the
    JSON writer uses, and the serialisation is key-sorted and separator-stable,
    so a dict reordering cannot change the digest.
    """
    if not isinstance(record, dict):
        raise TypeError(f'analysis record must be a dict, got {type(record).__name__}')
    body = {k: v for k, v in record.items() if k not in _ANALYSIS_VOLATILE_KEYS}
    text = json.dumps(_json_clean(_jsonable(body)), sort_keys=True,
                      separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False, default=float)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def analysis_input_manifest(inputs, *, base=None):
    """Content hashes of the files an analysis consumed.

    ``inputs`` is either an iterable of paths or a ``label -> path`` mapping.
    Relative paths resolve against ``base`` (default: the repository root) and
    are stored relative to that root when possible, so an artifact moved
    between checkouts still points at the same logical file.  A missing input
    is recorded as ``exists: false`` rather than raising: the analysis may
    legitimately have run on an absent optional probe, and the verifier must be
    able to say so explicitly instead of crashing on the provenance block.
    """
    if inputs is None:
        return {}
    base = os.path.abspath(base or _analysis_repo_root())
    if isinstance(inputs, dict):
        items = list(inputs.items())
    else:
        if isinstance(inputs, (str, bytes, os.PathLike)):
            inputs = [inputs]
        items = [(f'input_{i}', p) for i, p in enumerate(inputs)]
    out = {}
    for label, raw in items:
        label = str(label)
        if raw is None:
            out[label] = {'path': None, 'exists': False, 'sha256': None,
                          'bytes': None, 'error': 'no path supplied'}
            continue
        raw_s = os.fspath(raw)
        path = (os.path.abspath(raw_s) if os.path.isabs(raw_s)
                else os.path.abspath(os.path.join(base, raw_s)))
        try:
            rel = os.path.relpath(path, base)
            shown = rel.replace(os.sep, '/')
            if shown == '..' or shown.startswith('../'):
                shown = path
        except (OSError, ValueError):
            shown = path
        ent = {'path': shown, 'abs_path': path, 'exists': os.path.isfile(path),
               'sha256': None, 'bytes': None}
        if ent['exists']:
            try:
                st = os.stat(path)
                ent['sha256'] = file_sha256(path)
                ent['bytes'] = int(st.st_size)
            except OSError as e:
                ent['error'] = f'{type(e).__name__}: {e}'
        else:
            ent['error'] = 'file does not exist at provenance time'
        out[label] = ent
    return out


def verify_stamped_analysis(artifact, *, base=None, check_inputs=True):
    """Verify a stamped analysis against its own content and input hashes.

    This is the local reproducibility check for a derived number: the body
    digest detects an edited statistic/verdict/curve, and the input hashes
    detect a changed ``bylen.json``/summary/manifest that the artifact was
    actually derived from.  A legacy artifact without ``content_sha256`` is
    reported as not verifiable, never silently accepted; an input that is no
    longer present is a warning (an archived artifact is still valid evidence)
    rather than a false failure.
    """
    path = None
    if isinstance(artifact, (str, bytes, os.PathLike)):
        path = os.path.abspath(os.fspath(artifact))
        try:
            with open(path, encoding='utf-8') as fh:
                rec = json.load(fh)
        except (OSError, ValueError) as e:
            return {'kind': 'analysis_integrity', 'status': 'not_verifiable_locally',
                    'path': path, 'problems': [],
                    'warnings': [f'cannot read/parse the artifact: {type(e).__name__}: {e}'],
                    'note': 'no readable stamped artifact'}
    else:
        rec = artifact
    if not isinstance(rec, dict):
        return {'kind': 'analysis_integrity', 'status': 'fail', 'path': path,
                'problems': [f'artifact is a {type(rec).__name__}, not a JSON object'],
                'warnings': [], 'note': 'malformed artifact'}
    out = {'kind': 'analysis_integrity', 'status': 'ok', 'path': path,
           'artifact_kind': rec.get('kind'), 'problems': [], 'warnings': []}
    digest = rec.get('content_sha256')
    if digest is None:
        out['status'] = 'not_verifiable_locally'
        out['note'] = ('legacy artifact: no content_sha256, so a body edit '
                       'cannot be detected; regenerate it with the current code')
    elif (not isinstance(digest, str) or len(digest) != 64
          or any(c not in '0123456789abcdef' for c in digest.lower())):
        out['status'] = 'fail'
        out['problems'].append(f'content_sha256 is not a SHA-256 hex digest: {digest!r}')
    else:
        try:
            want = analysis_content_sha256(rec)
        except (TypeError, ValueError, OverflowError) as e:
            out['status'] = 'fail'
            out['problems'].append(
                f'content body cannot be canonicalised: {type(e).__name__}: {e}')
        else:
            if want != digest.lower():
                out['status'] = 'fail'
                out['problems'].append(
                    f'content_sha256 does not reproduce (stored {digest[:12]}…, '
                    f'recomputed {want[:12]}…): the analysis body was edited '
                    'after it was written')
    if check_inputs:
        inputs = rec.get('inputs')
        if inputs is not None and not isinstance(inputs, dict):
            out['status'] = 'fail'
            out['problems'].append(
                f'inputs is a {type(inputs).__name__}, not an object')
        elif isinstance(inputs, dict):
            base_abs = os.path.abspath(base or _analysis_repo_root())
            for label, ent in sorted(inputs.items()):
                if not isinstance(ent, dict):
                    out['status'] = 'fail'
                    out['problems'].append(f'input {label!r} is not an object')
                    continue
                raw = ent.get('path')
                if not isinstance(raw, str) or not raw:
                    raw = ent.get('abs_path')
                if not isinstance(raw, str) or not raw:
                    out['status'] = 'fail'
                    out['problems'].append(f'input {label!r} has no path')
                    continue
                p = raw if os.path.isabs(raw) else os.path.join(base_abs, raw)
                want = ent.get('sha256')
                if not isinstance(want, str) or len(want) != 64:
                    out['status'] = 'fail'
                    out['problems'].append(
                        f'input {label!r} does not record a SHA-256 digest')
                    continue
                if ent.get('exists') is False or not os.path.isfile(p):
                    out['warnings'].append(
                        f'input {label!r} is absent locally ({raw}); its '
                        'content cannot be re-verified from this checkout')
                    continue
                try:
                    got = file_sha256(p)
                except OSError as e:
                    out['warnings'].append(
                        f'input {label!r} cannot be read ({type(e).__name__}: {e})')
                    continue
                if got != want.lower():
                    out['status'] = 'fail'
                    out['problems'].append(
                        f'input {label!r} changed after the analysis '
                        f'({raw}: recorded {want[:12]}…, now {got[:12]}…)')
    if out['status'] == 'ok' and out['warnings']:
        out['status'] = 'warnings'
    if out['status'] == 'ok':
        out['note'] = ('content digest and every locally available input '
                       'match the recorded provenance')
    elif out['status'] == 'warnings':
        out['note'] = ('content digest matches; some inputs are not available '
                       'locally, so the full derivation is not re-verifiable')
    return out


def stamped_analysis(kind, payload, *, outdir=None, name=None, event=True,
                     inputs=None):
    """Return `payload` merged with the standard traceability stamp — and
    optionally persist it plus an experiments.jsonl entry.

    A derived number (statistic, audit verdict, analytic FLOPs table, kernel
    validation report, parameter count) is quotable in a report only if the
    artifact records WHICH code produced it, when, and under which
    environment/git state — the same provenance spine a training record
    carries.  Without this, re-deriving an analysis after a code change is
    indistinguishable from re-deriving the original, and a stale artifact
    silently masquerades as current.  `kind` identifies the artifact family;
    `payload` is the analysis body; `outdir`+`name` persist the stamped record
    atomically; `event` additionally appends a ledger entry so analysis runs
    appear in the same experiments.jsonl trace as training panels.

    ``inputs`` optionally binds the exact input bytes the analysis consumed
    (path + SHA-256 + size).  `content_sha256` binds the result body itself,
    excluding volatile provenance fields; `verify_stamped_analysis` replays
    both checks, so a hand-edited number or a changed input cannot silently
    masquerade as the recorded derivation.

    The stamp overwrites its own keys (`kind`/`created_at`/`code`/
    `provenance`), so re-stamping a loaded record on rewrite updates the
    trace instead of accumulating stale copies."""
    rec = dict(payload) if isinstance(payload, dict) else {'result': payload}
    rec['kind'] = kind
    rec['created_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    rec['code'] = code_stamp()
    rec['provenance'] = code_provenance()
    if inputs is not None:
        rec['inputs'] = analysis_input_manifest(inputs)
    rec['content_sha256'] = analysis_content_sha256(rec)
    if outdir is not None and name:
        path = os.path.join(outdir, name)
        atomic_write_json(path, rec)
        if event:
            log_experiment_event(outdir, {'kind': 'analysis', 'tracking_schema': 2, 'artifact': name, 'created_at': rec['created_at'], 'code': rec['code'], 'label': kind, 'status': 'ok', 'content_sha256': rec['content_sha256']})
    return rec

_PINNED_HOST_CACHE = {}
_PINNED_HOST_CAP_BYTES = 256 << 20
_GPU_CORPUS_CACHE_BYTES = 64 << 20


def _gpu_corpus_cache_fits(nbytes, device):
    """Reserve the corpus cache only when it also fits the current free device capacity."""
    if int(nbytes) > _GPU_CORPUS_CACHE_BYTES:
        return False
    try:
        free, _total = torch.cuda.mem_get_info(device)
        return int(nbytes) <= max(0, int(free) // 4)
    except Exception:
        return True

_NARROW_MEMO = {}
_NARROW_MEMO_CAP = 4


def _narrow_memo_put(key, ref, arr):
    """Store one narrowed corpus with a bounded, dead-entry-pruned cache.

    Keying on ``id(train_ids)`` without eviction retained every corpus for the
    process lifetime, which is a real leak when a scale sweep loads successive
    multi-million-token arrays.
    """
    for _k in [k for k, (r, _v) in _NARROW_MEMO.items() if r() is None]:
        _NARROW_MEMO.pop(_k, None)
    if key not in _NARROW_MEMO and len(_NARROW_MEMO) >= _NARROW_MEMO_CAP:
        _NARROW_MEMO.pop(next(iter(_NARROW_MEMO)))
    _NARROW_MEMO[key] = (ref, arr)


def _cache_id_upper_bound(path):
    """Recorded token-id upper bound for a cache file, when the sibling meta
    file provides one.  Avoids a full-corpus max() scan on legacy int64 caches."""
    if not path:
        return None
    base = os.path.basename(str(path))
    d = os.path.dirname(os.path.abspath(str(path)))
    if base.startswith('train_ids_'):
        meta_name = 'meta_' + base[len('train_ids_'):]
    elif base.startswith('test_ids_'):
        meta_name = 'test_meta_' + base[len('test_ids_'):]
    else:
        return None
    meta_path = os.path.join(d, os.path.splitext(meta_name)[0] + '.json')
    try:
        with open(meta_path, encoding='utf-8') as fh:
            meta = json.load(fh)
    except (OSError, ValueError, TypeError):
        return None
    for key in ('max_id', 'max_token_id', 'token_max'):
        raw = meta.get(key)
        if isinstance(raw, (bool, np.bool_)):
            continue
        if isinstance(raw, numbers.Integral):
            v = int(raw)
        elif isinstance(raw, numbers.Real):
            f = float(raw)
            if not math.isfinite(f) or f != math.floor(f):
                continue
            v = int(f)
        else:
            continue
        if 0 <= v < 2 ** 31:
            return v
    raw_vocab = meta.get('vocab')
    if isinstance(raw_vocab, (bool, np.bool_)):
        return None
    if isinstance(raw_vocab, numbers.Integral):
        vocab = int(raw_vocab)
    elif isinstance(raw_vocab, numbers.Real):
        f = float(raw_vocab)
        if not math.isfinite(f) or f != math.floor(f):
            return None
        vocab = int(f)
    else:
        return None
    if 0 < vocab <= 2 ** 31:
        return vocab - 1
    return None


def _ids_narrow(train_ids):
    if train_ids.dtype in (np.uint8, np.uint16, np.int32):
        return train_ids
    import weakref as _weakref
    _key = id(train_ids)
    _hit = _NARROW_MEMO.get(_key)
    if _hit is not None and _hit[0]() is train_ids:
        return _hit[1]

    class _TransportRangeError(ValueError):
        pass

    def _copy_i32_checked(src, dst):
        """Copy in chunks, refusing any value outside the int32 transport range.

        The metadata/`max()` fast path only proves an UPPER bound; a large
        negative sentinel would otherwise wrap to a positive id on assignment
        to int32 (`-2**31 - 1 -> 2**31 - 1`) and the model would train on a
        different token than the cache contains."""
        _s_flat = np.asarray(src).reshape(-1)
        _d_flat = np.asarray(dst).reshape(-1)
        _step = 1 << 24
        for _s in range(0, int(_s_flat.size), _step):
            _e = min(_s + _step, int(_s_flat.size))
            _chunk = _s_flat[_s:_e]
            if _chunk.size:
                _mn, _mx = int(_chunk.min()), int(_chunk.max())
                if _mn < -(2 ** 31) or _mx > 2 ** 31 - 1:
                    raise _TransportRangeError(
                        f'token ids span [{_mn}, {_mx}] outside the int32 '
                        'transport range; refusing a lossy narrowing')
            _d_flat[_s:_e] = _chunk
        return dst

    _path = getattr(train_ids, 'filename', None)
    if _path and train_ids.size:
        # A disk-backed corpus is mapped read-only.  Workers share one int32
        # sidecar next to the cache.
        try:
            _st = os.stat(_path)
            # Bind the sidecar to the same identity fields the cache loaders
            # use.  A same-size rewrite can preserve mtime (``cp -p``), so
            # size+mtime alone could silently serve a stale int32 copy.
            _side = _i32_sidecar_path(_path)
            if os.path.exists(_side):
                _n = np.load(_side, mmap_mode='r')
                if (int(_n.size) == int(train_ids.size)
                        and tuple(_n.shape) == tuple(train_ids.shape)
                        and np.dtype(_n.dtype) == np.dtype(np.int32)):
                    _narrow_memo_put(_key, _weakref.ref(train_ids), _n)
                    return _n
            _mx = _cache_id_upper_bound(_path)
            if _mx is None:
                _mx = int(train_ids.max()) if train_ids.size else -1
            if _mx < 2 ** 31:
                _dir = os.path.dirname(os.path.abspath(_side)) or '.'
                _fd, _tmp = tempfile.mkstemp(dir=_dir, prefix=os.path.basename(_side) + '.', suffix='.tmp')
                os.close(_fd)
                try:
                    _mm = np.lib.format.open_memmap(_tmp, mode='w+', dtype=np.int32, shape=tuple(train_ids.shape))
                    _copy_i32_checked(train_ids, _mm)
                    _mm.flush()
                    del _mm
                    os.replace(_tmp, _side)
                except BaseException:
                    try:
                        os.unlink(_tmp)
                    except OSError:
                        pass
                    raise
                _n = np.load(_side, mmap_mode='r')
                _narrow_memo_put(_key, _weakref.ref(train_ids), _n)
                return _n
        except (OSError, ValueError, TypeError) as _e:
            # A range failure is a data error, not a reason to fall through to
            # a second lossy conversion; other sidecar/filesystem failures keep
            # the historical in-memory fallback.
            if isinstance(_e, _TransportRangeError):
                raise
            pass
    _mx = _cache_id_upper_bound(getattr(train_ids, 'filename', None))
    if _mx is None:
        _mx = int(train_ids.max()) if train_ids.size else -1
    if train_ids.size and _mx < 2 ** 31:
        _n = np.empty(train_ids.shape, dtype=np.int32)
        _copy_i32_checked(train_ids, _n)
        _narrow_memo_put(_key, _weakref.ref(train_ids), _n)
        return _n
    return train_ids

def batch_iter(train_ids, seq_len, batch_size, device, seed=0):
    # A zero/negative window or batch is not a smaller training stream: it
    # yields empty input/target tensors (or crashes inside NumPy allocation).
    # Refuse it explicitly instead of feeding a silently empty batch.
    _seq = _exact_count(seq_len, minimum=1)
    _bs = _exact_count(batch_size, minimum=1)
    if _seq is None:
        raise ValueError(f'batch_iter needs a positive integer seq_len, got {seq_len!r}')
    if _bs is None:
        raise ValueError(f'batch_iter needs a positive integer batch_size, got {batch_size!r}')
    seq_len, batch_size = int(_seq), int(_bs)
    rng = np.random.default_rng(_seed_value(seed))
    # A corpus of exactly seq_len+1 tokens still contains one causal window
    # (start=0), so only a shortfall is invalid.
    n = len(train_ids) - seq_len - 1
    if n < 0:
        raise ValueError(f'batch_iter needs len(train_ids) >= seq_len+1 to sample causal windows, got len={len(train_ids)} seq_len={seq_len}. Increase n_train_tokens or reduce seq_len.')
    train_ids = np.asarray(train_ids)
    _narrow = _ids_narrow(train_ids)
    _devkey = _dev_str(device)
    host = None
    host_np = None
    cached = _PINNED_HOST_CACHE.get('entry')
    if (cached is not None and cached[0] is train_ids
            and cached[1].shape[0] == train_ids.shape[0]
            and cached[2] == _devkey):
        host = cached[1]
    elif device.type == 'cuda' and _gpu_corpus_cache_fits(train_ids.nbytes, device):
        # The residency budget is defined on the ORIGINAL token array (design
        # notes §6.1).  The narrowed view is a transport form; using its
        # smaller size would admit up to 4x more corpus than the contract for
        # a uint16/uint8 cache, so the execution path would silently change
        # with the storage width at scale.
        try:
            host = torch.from_numpy(_narrow).to(device)
            _PINNED_HOST_CACHE['entry'] = (train_ids, host, _devkey)
        except RuntimeError:
            host = None
    if host is None:
        if device.type == 'cuda':
            try:
                if _narrow.dtype in (np.uint8, np.uint16):
                    host_np = _narrow
                elif _narrow.nbytes <= _PINNED_HOST_CAP_BYTES:
                    host = torch.from_numpy(_narrow).pin_memory()
                    _PINNED_HOST_CACHE['entry'] = (train_ids, host, _devkey)
                else:
                    host_np = _narrow
            except RuntimeError:
                host_np = _narrow
        else:
            if _narrow.dtype in (np.uint8, np.uint16):
                # PyTorch's CPU index kernel lacks uint8/uint16 gathers.  Gather
                # the narrow array with NumPy per batch and widen only the
                # [B, seq_len+1] result — the same exact token ids.
                host_np = _narrow
            else:
                host = torch.from_numpy(_narrow)
    # Transport-only narrowing (design notes §6.1): every gather index is an
    # exact integer bounded by n + seq_len, so when that bound fits in int32
    # the index tensor is uploaded / consumed as int32 and the gathered ids
    # are widened back to int64 right after the gather exactly as before —
    # bit-identical batches.  The broadcast addition itself is then also done
    # in int32: every operand and the sum are < 2**31, so the int32 result
    # holds exactly the same integers as the int64 form.
    _idx_i32 = n + seq_len + 1 < 2 ** 31
    _idx_dt = np.int32 if _idx_i32 else np.int64
    cols = np.arange(seq_len + 1, dtype=_idx_dt)
    idx_np = np.empty((batch_size, seq_len + 1), dtype=_idx_dt)

    def _gather(idx):
        if host_np is not None:
            return torch.from_numpy(host_np[idx])
        if host.device.type == 'cuda':
            idx_t = torch.from_numpy(idx)
            return host[_h2d_async(idx_t, idx_t.dtype, host.device)]
        return host[idx]

    def _prepare(ids):
        if ids.dtype != torch.int64:
            # widen back right after the gather: the yielded ids are
            # bit-identical to the int64 form.
            ids = ids.long()
        if ids.device != device:
            ids = _h2d_async(ids, ids.dtype, device)
        return ids

    # A small bounded producer queue runs the identical seeded draw and gather
    # one step ahead.  Values and order are unchanged (same RNG, same indices).
    _prefetch = device.type == 'cuda' and (host_np is not None or (host is not None and host.device.type == 'cpu'))
    if not _prefetch:
        while True:
            starts = rng.integers(0, n + 1, size=batch_size, dtype=_idx_dt)
            np.add(starts[:, None], cols[None, :], out=idx_np)
            ids = _prepare(_gather(idx_np))
            yield (ids[:, :-1], ids[:, 1:])

    import queue as _queue
    import threading as _threading
    q = _queue.Queue(maxsize=2)
    stop = _threading.Event()

    def _producer():
        try:
            while not stop.is_set():
                starts = rng.integers(0, n + 1, size=batch_size, dtype=_idx_dt)
                np.add(starts[:, None], cols[None, :], out=idx_np)
                item = _gather(idx_np)
                while not stop.is_set():
                    try:
                        q.put((item, None), timeout=0.2)
                        break
                    except _queue.Full:
                        continue
        except BaseException as exc:
            import traceback as _tb
            try:
                q.put((None, (exc, _tb.format_exc())))
            except BaseException:
                pass

    th = _threading.Thread(target=_producer, name='batch-prefetch', daemon=True)
    th.start()
    try:
        while True:
            try:
                item, err = q.get(timeout=0.2)
            except _queue.Empty:
                if stop.is_set() or not th.is_alive():
                    return
                continue
            if err is not None:
                exc, tb = err
                if tb:
                    raise RuntimeError(f'batch prefetch failed: {exc!r}\n{tb}') from exc
                raise exc
            ids = _prepare(item)
            yield (ids[:, :-1], ids[:, 1:])
    finally:
        stop.set()
        th.join(timeout=5.0)

def batch_contract_check(train_ids, seq_len, batch_size, device, seed=0, vocab=None, n_batches=2):
    """Validate the ACTUAL training batch iterator, not a re-derived sample.

    The small-batch overfit probe trains on a fixed batch; that alone cannot
    prove the production loader shifts inputs/targets correctly, keeps every
    id inside the vocabulary, or repeats the same stream for the same seed.
    This check pulls a few real batches from ``batch_iter`` twice and reports
    the loader-level contract separately.  It consumes no global RNG state and
    is meant to run before a panel starts."""
    _seq = _exact_count(seq_len, minimum=1)
    if _seq is None:
        raise ValueError(f'batch_contract_check: seq_len must be a positive integer, got {seq_len!r}')
    _bs = _exact_count(batch_size, minimum=1)
    if _bs is None:
        raise ValueError(f'batch_contract_check: batch_size must be a positive integer, got {batch_size!r}')
    _nb = _exact_count(n_batches, minimum=1)
    if _nb is None:
        raise ValueError(f'batch_contract_check: n_batches must be a positive integer, got {n_batches!r}')
    _vocab = None if vocab is None else _exact_count(vocab, minimum=1)
    if vocab is not None and _vocab is None:
        raise ValueError(f'batch_contract_check: vocab must be a positive integer or None, got {vocab!r}')
    seq_len, batch_size, n_batches = int(_seq), int(_bs), int(_nb)
    seed = _seed_value(seed)
    device = device if isinstance(device, torch.device) else torch.device(device)
    report = {
        'kind': 'batch_iterator_contract',
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'code': code_stamp(),
        'seq_len': seq_len,
        'batch_size': batch_size,
        'seed': int(seed),
        'vocab_size': (None if _vocab is None else int(_vocab)),
        'n_batches': n_batches,
        'device': str(device),
        'checks': {},
        'batches': [],
        'passed': False,
    }
    checks = {
        'shapes_ok': True,
        'dtypes_int64': True,
        'labels_are_next_token_shift': True,
        'ids_within_vocab': True,
        'same_seed_repeats_stream': True,
        'transport_matches_int64_reference': True,
        'nonempty': True,
    }
    # Transport-equivalence reference.  `batch_iter` may narrow its gather
    # indices to int32, pin/upload the corpus, or run a prefetch thread.  All
    # of those are transport-only claims; this second stream is built from the
    # SAME seeded starts with plain int64 NumPy indexing and no prefetch, so
    # comparing it element by element checks the claim instead of restating it.
    _n_windows = len(train_ids) - seq_len - 1
    _idx_dtype = 'int32' if (_n_windows + seq_len + 1 < 2 ** 31) else 'int64'
    _ref_rng = np.random.default_rng(int(seed))
    _ref_cols = np.arange(seq_len + 1, dtype=np.int64)
    _ref_arr = np.asarray(train_ids)
    report['transport'] = {
        'index_dtype': _idx_dtype,
        'reference': 'plain int64 numpy gather from the same seeded starts, no pin/prefetch/narrowing',
        'corpus_dtype': np.dtype(_ref_arr.dtype).str,
        'corpus_writeable': bool(getattr(_ref_arr, 'flags', None) is not None
                                 and _ref_arr.flags.writeable),
    }
    try:
        it_a = batch_iter(train_ids, seq_len, batch_size, device, seed=seed)
        it_b = batch_iter(train_ids, seq_len, batch_size, device, seed=seed)
        for index in range(n_batches):
            xa, ya = next(it_a)
            xb, yb = next(it_b)
            _starts = _ref_rng.integers(0, _n_windows + 1, size=batch_size,
                                        dtype=np.int64)
            _ref_idx = _starts[:, None] + _ref_cols[None, :]
            _ref_ids = torch.from_numpy(np.asarray(_ref_arr[_ref_idx])).long()
            _ref_x, _ref_y = _ref_ids[:, :-1], _ref_ids[:, 1:]
            checks['nonempty'] = bool(checks['nonempty'] and xa.numel() and ya.numel())
            checks['shapes_ok'] = bool(
                checks['shapes_ok']
                and tuple(xa.shape) == (batch_size, seq_len)
                and tuple(ya.shape) == (batch_size, seq_len)
                and tuple(xb.shape) == tuple(xa.shape)
                and tuple(yb.shape) == tuple(ya.shape))
            checks['dtypes_int64'] = bool(
                checks['dtypes_int64']
                and xa.dtype == torch.int64 and ya.dtype == torch.int64
                and xb.dtype == torch.int64 and yb.dtype == torch.int64)
            checks['labels_are_next_token_shift'] = bool(
                checks['labels_are_next_token_shift']
                and ya.shape[1] == xa.shape[1]
                and torch.equal(ya[:, :-1], xa[:, 1:]))
            checks['same_seed_repeats_stream'] = bool(
                checks['same_seed_repeats_stream']
                and torch.equal(xa, xb) and torch.equal(ya, yb))
            checks['transport_matches_int64_reference'] = bool(
                checks['transport_matches_int64_reference']
                and torch.equal(xa.cpu(), _ref_x)
                and torch.equal(ya.cpu(), _ref_y))
            if _vocab is not None and xa.numel() and ya.numel():
                _v = int(_vocab)
                checks['ids_within_vocab'] = bool(
                    checks['ids_within_vocab']
                    and int(xa.min()) >= 0 and int(xa.max()) < _v
                    and int(ya.min()) >= 0 and int(ya.max()) < _v)
            report['batches'].append({
                'index': index,
                'input_shape': [int(_n) for _n in xa.shape],
                'target_shape': [int(_n) for _n in ya.shape],
                'input_min': int(xa.min()) if xa.numel() else None,
                'input_max': int(xa.max()) if xa.numel() else None,
                'target_min': int(ya.min()) if ya.numel() else None,
                'target_max': int(ya.max()) if ya.numel() else None,
                'transport_matches_int64_reference': torch.equal(
                    xa.cpu(), _ref_x) and torch.equal(ya.cpu(), _ref_y),
            })
        del it_a, it_b
    except Exception as e:
        report['error'] = f'{type(e).__name__}: {e}'
        report['checks'] = checks
        report['diagnosis'] = 'batch iterator failed before producing the requested batches'
        return report
    report['checks'] = checks
    failed = [k for k, ok in checks.items() if not ok]
    report['passed'] = not failed
    report['diagnosis'] = 'ok' if not failed else 'failed checks: ' + ', '.join(failed)
    return report

def count_params(m):
    return sum((p.numel() for p in m.parameters()))
