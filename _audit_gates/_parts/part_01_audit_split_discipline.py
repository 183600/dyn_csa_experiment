
def _unique_int_set(values):
    """Set of the distinct integer values, without a per-token Python list.

    The document-id sidecars are indexed per TOKEN, not per document: a
    100M-token corpus yields a 100M-element array that may hold only ~800k
    distinct document ids.  ``arr.tolist()`` materialises one Python int per
    token before the set discards the duplicates.  Collapsing to the unique
    values first keeps the same set and only pays for the distinct ids.  An
    object/mixed array falls back to the historical conversion so malformed
    sidecars keep their existing error behaviour.
    """
    arr = np.asarray(values)
    if arr.dtype.kind in 'iufb':
        arr = np.unique(arr.reshape(-1))
    return set(int(x) for x in arr.reshape(-1).tolist())


def audit_split_discipline(cache_dirs=None, max_rows=50000, js_warn=0.05,
                           js_fail=0.25, min_shift_tokens=5000,
                           preferred_train_seq_len=None,
                           passage_shingle=32, max_passage_scan_tokens=4_000_000,
                           min_passage_scan_fraction=0.25,
                           max_val_shingle_overlap=0.25,
                           max_passage_excess_overlap=0.10,
                           max_val_row_shingle_overlap=0.50):
    """Audit the *split contract* for every cached corpus on disk.

    Unlike ``audit_leakage`` (tokenizer provenance and document overlap), this
    gate checks the split itself:

      * the metadata declares an ordered/non-shuffled split and a train-only
        tokenizer fit;
      * validation and test contain no exact row duplicated from train;
      * validation contains no long near-duplicate passage from train (a
        shingle-overlap scan calibrated against train-internal repetition);
      * no group/document spans two sides (for FineWeb, document-disjoint
        chronological construction is recorded and the cache metadata keeps
        the validation fingerprint);
      * train vs holdout token distributions are compared with Jensen-Shannon
        divergence (a shift is reported explicitly rather than silently
        assumed away).

     Exact duplicate rows across splits are a hard failure.  Distribution or
    legacy-metadata issues are warnings because they need a scientific
    judgement, not an automatic rebuild.

    ``preferred_train_seq_len`` restricts the audit to the train cache that
    actually served a panel when a directory holds several protocol variants
    (e.g. a 512-token training cache and a 4097-token probe cache).  A caller
    that names the training length must not be silently audited against a
    different stream.
    """
    if preferred_train_seq_len is not None:
        preferred_train_seq_len = _audit_positive_int(
            'preferred_train_seq_len', preferred_train_seq_len)
    js_warn = _audit_nonneg_float('js_warn', js_warn)
    js_fail = _audit_nonneg_float('js_fail', js_fail)
    if js_warn > js_fail:
        raise ValueError(f'js_warn ({js_warn}) must not exceed js_fail ({js_fail})')
    min_shift_tokens = _audit_positive_int('min_shift_tokens', min_shift_tokens)
    max_rows = _audit_positive_int('max_rows', max_rows)
    passage_shingle = _audit_positive_int('passage_shingle', passage_shingle)
    max_passage_scan_tokens = _audit_positive_int(
        'max_passage_scan_tokens', max_passage_scan_tokens)
    min_passage_scan_fraction = _audit_nonneg_float(
        'min_passage_scan_fraction', min_passage_scan_fraction)
    max_val_shingle_overlap = _audit_nonneg_float(
        'max_val_shingle_overlap', max_val_shingle_overlap)
    max_passage_excess_overlap = _audit_nonneg_float(
        'max_passage_excess_overlap', max_passage_excess_overlap)
    max_val_row_shingle_overlap = _audit_nonneg_float(
        'max_val_row_shingle_overlap', max_val_row_shingle_overlap)
    for _name, _val in (('min_passage_scan_fraction', min_passage_scan_fraction),
                        ('max_val_shingle_overlap', max_val_shingle_overlap),
                        ('max_passage_excess_overlap', max_passage_excess_overlap),
                        ('max_val_row_shingle_overlap', max_val_row_shingle_overlap)):
        if _val > 1.0:
            raise ValueError(f'{_name} must be a fraction in [0, 1], got {_val!r}')
    import glob
    import numpy as np
    sys.path.insert(0, REPO)
    import data_guard as DG

    if cache_dirs is None:
        cache_dirs = [os.path.join(REPO, 'wt103_cache'),
                      os.path.join(REPO, 'fineweb_cache')]
    out = {'status': 'ok', 'caches': {}, 'warnings': []}

    def _fail(msg):
        out['warnings'].append(msg)
        out['status'] = 'fail'

    def _warn(msg):
        out['warnings'].append(msg)
        if out['status'] == 'ok':
            out['status'] = 'warnings'

    def _audit_row_sample(arr, label, rec):
        """Deterministically cap a 2-D panel at ``max_rows`` for duplicate scans.

        ``max_rows`` was validated but unused, so a 1e8-row cache still built
        giant row-key tables.  The cap is recorded; it is an audit sample, not
        a proof about the unsampled rows.
        """
        a = np.asarray(arr)
        if a.ndim == 2 and a.shape[0] > max_rows:
            _ix = np.unique(np.linspace(0, a.shape[0] - 1, num=max_rows,
                                        dtype=np.int64))
            rec[f'{label}_rows_total'] = int(a.shape[0])
            rec[f'{label}_rows_audited'] = int(_ix.size)
            return a[_ix]
        return a

    for cd in cache_dirs:
        if not os.path.isdir(cd):
            continue
        cache = {'status': 'ok', 'corpora': {}}
        out['caches'][os.path.abspath(cd)] = cache
        for mp in sorted(glob.glob(os.path.join(cd, 'meta_*.json'))):
            if os.path.basename(mp).startswith('test_meta_'):
                continue
            _nwarn_before = len(out['warnings'])
            meta = _load(mp)
            if not isinstance(meta, dict):
                continue
            if preferred_train_seq_len is not None:
                try:
                    _meta_sl = int(meta.get('seq_len'))
                except (TypeError, ValueError, OverflowError):
                    continue
                if _meta_sl != int(preferred_train_seq_len):
                    continue
            tag = os.path.basename(mp)[len('meta_'):-len('.json')]
            trp = os.path.join(cd, f'train_ids_{tag}.npy')
            vap = os.path.join(cd, f'val_batch_{tag}.npy')
            if not (os.path.exists(trp) and os.path.exists(vap)):
                continue
            rec = {'status': 'ok', 'meta': os.path.basename(mp)}
            try:
                tr = np.load(trp, mmap_mode='r')
                va = np.load(vap, mmap_mode='r')
            except Exception as e:
                rec['status'] = 'fail'
                rec['error'] = f'{type(e).__name__}: {e}'
                _fail(f'{cd}/{tag}: cache unreadable ({rec["error"]})')
                cache['corpora'][tag] = rec
                continue
            rec['train_shape'] = list(tr.shape)
            rec['val_shape'] = list(va.shape)
            _va_scan = _audit_row_sample(va, 'validation', rec)
            rec['declared_split_strategy'] = meta.get('split_strategy') or 'legacy_unrecorded'
            rec['shuffle_at_split'] = meta.get('shuffle_at_split')
            rec['tokenizer_fit_split'] = meta.get('tokenizer_fit_split')
            rec['val_fingerprint_recorded'] = bool(meta.get('val_fp'))
            # The cache files must still BE the stream the metadata names.
            # Checking only "val_fp is present" (or only the runtime loader's
            # check) lets a replaced/edited .npy be certified by this offline
            # audit.  Recompute both recorded fingerprints from the actual
            # bytes; a missing fingerprint stays a legacy warning.
            _src_recorded = meta.get('src_fp') or meta.get('train_fp')
            _val_recorded = meta.get('val_fp')
            if _src_recorded is not None or _val_recorded is not None:
                try:
                    _Lfp = _import_exp_lib()
                    if _src_recorded is not None:
                        _src_actual = _Lfp._ids_fp(tr)
                        rec['train_fingerprint_verified'] = bool(
                            str(_src_actual) == str(_src_recorded))
                        if not rec['train_fingerprint_verified']:
                            rec['status'] = 'fail'
                            _fail(f'{cd}/{tag}: train id-cache fingerprint does not '
                                  'match metadata src_fp (the cache was edited or '
                                  'replaced)')
                    if _val_recorded is not None:
                        _val_actual = _Lfp._ids_fp(np.asarray(va).reshape(-1))
                        rec['val_fingerprint_verified'] = bool(
                            str(_val_actual) == str(_val_recorded))
                        if not rec['val_fingerprint_verified']:
                            rec['status'] = 'fail'
                            _fail(f'{cd}/{tag}: validation id-cache fingerprint does '
                                  'not match metadata val_fp (a stale/mixed '
                                  'validation file cannot be certified)')
                except Exception as _fpe:
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: id-cache fingerprint audit failed '
                          f'({type(_fpe).__name__}: {_fpe})')
            if meta.get('shuffle_at_split') is True:
                rec['status'] = 'fail'
                _fail(f'{cd}/{tag}: metadata declares a shuffled split')
            if meta.get('tokenizer_fit_split') not in (None, 'train'):
                rec['status'] = 'fail'
                _fail(f'{cd}/{tag}: tokenizer fit split is {meta.get("tokenizer_fit_split")!r}, not train')
            if meta.get('split_strategy') is None:
                _warn(f'{cd}/{tag}: legacy cache has no split_strategy/tokenizer provenance metadata; rebuild it to make the split auditable')

            # Data-cleaning coverage: non-finite values are a hard failure (an
            # embedding index or loss built from them is undefined), and exact
            # duplicates inside the holdout are reported.  Integer token caches
            # cannot represent NaN/Inf, so only float caches are scanned.
            for _name, _arr in (('train', tr), ('val', va)):
                if np.issubdtype(np.asarray(_arr).dtype, np.integer):
                    rec[f'nonfinite_{_name}'] = 0
                    continue
                _bad = int(np.size(_arr) - np.isfinite(_arr).sum())
                rec[f'nonfinite_{_name}'] = _bad
                if _bad:
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: {_bad} NaN/Inf values in the {_name} split')
            if np.asarray(va).ndim == 2:
                # ``_va_scan`` is the max_rows-capped readout.  Hashing the
                # full validation cache here bypassed the audit's own scale
                # guard (the test-side scan already uses its sample).
                _dup_val = int(DG.duplicate_mask(np.asarray(_va_scan)).sum())
                rec['duplicate_rows_within_val'] = _dup_val
                if _dup_val:
                    _warn(f'{cd}/{tag}: {_dup_val} duplicate validation rows '
                          '(deduplicate before scoring, or report it as a '
                          'sampling artefact)')

            # Whole-document (group) integrity.  The loaders write one doc-id
            # per emitted document; checking the sidecars proves no group
            # straddles train/validation instead of trusting the build flag.
            _trd_path = os.path.join(cd, f'train_doc_ids_{tag}.npy')
            _vad_path = os.path.join(cd, f'val_doc_ids_{tag}.npy')
            _trd_exists = os.path.exists(_trd_path)
            _vad_exists = os.path.exists(_vad_path)
            _mdoc = meta.get('doc_ids') or {}
            if _trd_exists != _vad_exists:
                rec['status'] = 'fail'
                _fail(f'{cd}/{tag}: only one train/validation document-id '
                      'sidecar exists; group isolation is half-attested')
            if _trd_exists and _vad_exists:
                try:
                    _trd = np.load(_trd_path)
                    _vad = np.load(_vad_path)
                    _shared_docs = int(np.intersect1d(_trd, _vad).size)
                    rec['documents'] = {
                        'train_count': int(np.asarray(_trd).size),
                        'validation_count': int(np.asarray(_vad).size),
                        'overlap': _shared_docs,
                    }
                    if _shared_docs:
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: {_shared_docs} document ids occur in '
                              'both train and validation (group leakage)')
                    if _mdoc and _mdoc.get('disjoint') is not True:
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: cache metadata does not attest '
                              'document-disjoint splits')
                    if _mdoc and int(_mdoc.get('overlap', 0) or 0) != 0:
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: cache metadata records a nonzero '
                              'train/validation document overlap')
                except Exception as _de:
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: document-id sidecars unreadable ({type(_de).__name__}: {_de})')
            elif _mdoc:
                rec['status'] = 'fail'
                _fail(f'{cd}/{tag}: cache metadata records document ids but the '
                      'sidecar files are missing')

            # Exact duplicate rows across train/val/test.  This is row-level
            # leakage regardless of how the split was generated.
            shared_val = _row_overlap_count(tr, _va_scan)
            rec['exact_row_overlap_train_val'] = int(shared_val)
            if shared_val:
                rec['status'] = 'fail'
                _fail(f'{cd}/{tag}: {shared_val} validation rows exactly duplicate a training row')
            # Near-duplicate passage overlap catches a validation row that
            # reuses a long training passage while differing elsewhere.  The
            # scan is sampled, so a clean result on a small train fraction is
            # recorded as not_verifiable rather than certified.
            try:
                _passage = _passage_overlap_stats(
                    tr, _va_scan, shingle=passage_shingle,
                    max_scan_tokens=max_passage_scan_tokens,
                    min_scan_fraction=min_passage_scan_fraction,
                    max_val_shingle_overlap=max_val_shingle_overlap,
                    max_excess_overlap=max_passage_excess_overlap,
                    max_val_row_overlap=max_val_row_shingle_overlap)
                rec['passage_overlap'] = _passage
                if _passage.get('status') == 'fail':
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: ' + str(_passage.get(
                        'note', 'near-duplicate passage overlap')))
                elif _passage.get('status') == 'not_verifiable':
                    if rec['status'] == 'ok':
                        rec['status'] = 'warnings'
                    _warn(f'{cd}/{tag}: ' + str(_passage.get(
                        'note', 'passage overlap scan incomplete')))
            except Exception as _pe:
                rec['status'] = 'fail'
                _fail(f'{cd}/{tag}: passage-overlap audit failed '
                      f'({type(_pe).__name__}: {_pe})')
            # The test cache filename carries its own seq_len/vocab/ns tag.
            # P5T deliberately loads test at a longer sequence length than the
            # training cache, so seq_len is NOT part of the cache identity.
            # Pair on vocab + tokenizer fit provenance (as audit_leakage
            # does); a mis-paired test file would make both false positives
            # and false negatives in the overlap audit.
            _cands = []
            for _tmp in sorted(glob.glob(os.path.join(cd, 'test_meta_*.json'))):
                _tm = _load(_tmp)
                if not isinstance(_tm, dict):
                    continue
                _ttag = os.path.basename(_tmp)[len('test_meta_'):-len('.json')]
                _tfp = os.path.join(cd, f'test_ids_{_ttag}.npy')
                if not os.path.exists(_tfp):
                    continue
                if meta.get('vocab') is not None and _tm.get('vocab') is not None and int(_tm['vocab']) != int(meta['vocab']):
                    continue
                if meta.get('fit_fp') and _tm.get('fit_fp') and meta['fit_fp'] != _tm['fit_fp']:
                    continue
                _cands.append((_tfp, _ttag, _tm))
            te = None
            if len(_cands) == 1:
                _tfp, _ttag, _tmeta = _cands[0]
                rec['test_split_strategy'] = _tmeta.get('split_strategy') or 'legacy_unrecorded'
                rec['test_shuffle_at_split'] = _tmeta.get('shuffle_at_split')
                rec['test_tokenizer_fit_split'] = _tmeta.get('tokenizer_fit_split')
                if _tmeta.get('shuffle_at_split') is True:
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: test metadata declares a shuffled split')
                if _tmeta.get('tokenizer_fit_split') not in (None, 'train'):
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: test tokenizer was fitted on {_tmeta.get("tokenizer_fit_split")!r}, not train')
                if _tmeta.get('split_strategy') is None:
                    _warn(f'{cd}/{tag}: legacy test cache has no split_strategy/tokenizer provenance metadata')
                _rp = _tmeta.get('read_purpose')
                if _rp is not None and _rp != 'final_evaluation':
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: test cache records read_purpose={_rp!r}, '
                          'not the frozen final evaluation; a diagnostic/tuning '
                          'read would turn test into an implicit validation set')
                _tndc = _tmeta.get('near_duplicate_cleanup')
                if isinstance(_tndc, dict):
                    if _tndc.get('policy') != 'test_documents_deduplicated_against_train_and_validation':
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: test metadata does not attest duplicate cleanup against both train and validation')
                try:
                    te = np.load(_tfp, mmap_mode='r')
                    rec['test_file'] = os.path.basename(_tfp)
                    rec['test_tag'] = _ttag
                except Exception:
                    te = None
            elif len(_cands) > 1:
                rec['test_pairing'] = 'ambiguous'
                rec['test_candidates'] = sorted(os.path.basename(p) for p, _, _ in _cands)
                _warn(f'{cd}/{tag}: {len(_cands)} test caches match the same protocol; refusing to guess which test split to audit')
            else:
                rec['test_pairing'] = 'not_found'
            if te is not None:
                rec['test_shape'] = list(te.shape)
                _te_scan = _audit_row_sample(te, 'test', rec)
                # Per-split content fingerprints are recomputed from the
                # actual bytes, not trusted from the JSON.  The audit already
                # did this for train/validation; without the same check on
                # test, a replaced/stale test file could be certified by a
                # metadata-only reader and then become an implicit validation
                # set through repeated evaluation.
                try:
                    _Lf = _import_exp_lib()
                    _t_actual = _Lf._ids_fp(np.asarray(te))
                    _t_recorded = _tmeta.get('src_fp')
                    _t_explicit = _tmeta.get('test_fp')
                    _sf_all = _tmeta.get('split_fingerprints')
                    _t_split = _sf_all.get('test') if isinstance(_sf_all, dict) else None
                    if _t_split is not None and not isinstance(_t_split, dict):
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: test split_fingerprints is malformed')
                        _t_split = {}
                    _t_split = _t_split or {}
                    rec['test_fingerprint_recorded'] = bool(_t_recorded)
                    _fp_ok = (_t_recorded is not None
                              and str(_t_actual) == str(_t_recorded))
                    if _t_explicit is not None and str(_t_explicit) != str(_t_actual):
                        _fp_ok = False
                    if _t_split.get('ids_fp') is not None and str(_t_split['ids_fp']) != str(_t_actual):
                        _fp_ok = False
                    rec['test_fingerprint_verified'] = bool(_fp_ok)
                    if _t_recorded is None and _t_explicit is None and not _t_split:
                        # Legacy test cache: there is no recorded fingerprint to
                        # disagree with.  Report the missing evidence (like the
                        # train/validation reader does) instead of inventing a
                        # hard failure; a cache that RECORDS a value must match.
                        _warn(f'{cd}/{tag}: legacy test cache records no content '
                              'fingerprint; the held-out id bytes cannot be '
                              'bound to the audited stream')
                    elif _t_recorded is not None and str(_t_actual) != str(_t_recorded):
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: test id-cache fingerprint does not '
                              'match metadata src_fp (the held-out cache was '
                              'edited or replaced)')
                    if _t_explicit is not None and str(_t_explicit) != str(_t_actual):
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: test_fp does not match the actual '
                              'held-out ids')
                    if _t_split.get('ids_fp') is not None and str(_t_split['ids_fp']) != str(_t_actual):
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: split_fingerprints.test.ids_fp does '
                              'not match the actual held-out ids')
                except Exception as _tfe:
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: test id-cache fingerprint audit failed '
                          f'({type(_tfe).__name__}: {_tfe})')
                # Split-construction semantics and tokenizer identity must be
                # the SAME across the train/test pair.  A cache pair built by
                # different code semantics or different tokenizers is not one
                # frozen split, even if every individual file is intact.
                _train_sem = meta.get('code_semantics')
                _test_sem = _tmeta.get('code_semantics')
                rec['code_semantics'] = {'train': _train_sem, 'test': _test_sem}
                if _train_sem is not None and _test_sem is not None and str(_train_sem) != str(_test_sem):
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: train/test data code semantics differ '
                          f'({_train_sem!r} vs {_test_sem!r}); the split '
                          'construction is not one frozen pair')
                import hashlib as _hl
                _train_tok_sha = meta.get('tokenizer_file_sha256')
                _test_tok_sha = _tmeta.get('tokenizer_file_sha256')
                # The BINDING identity is the full id->token map.  File bytes
                # can differ for equivalent serialisations, so use the vocab
                # hash for equality and the file hash only to prove each
                # persisted artefact is the one its own metadata names.
                _train_vsha = meta.get('tokenizer_vocab_sha256')
                _test_vsha = _tmeta.get('tokenizer_vocab_sha256')
                if _train_vsha and _test_vsha and str(_train_vsha) != str(_test_vsha):
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: train/test tokenizer id-space hashes '
                          'differ; the id spaces are not one paired tokenizer')
                if _train_tok_sha and _test_tok_sha and str(_train_tok_sha) != str(_test_tok_sha):
                    if _train_vsha and _test_vsha and str(_train_vsha) == str(_test_vsha):
                        _warn(f'{cd}/{tag}: train/test tokenizer files are not '
                              'byte-identical, but their full vocab identity is; '
                              'the binding equality holds')
                    else:
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: train/test tokenizer file identities '
                              'differ without a matching vocab hash; the id '
                              'spaces cannot be shown to be one paired tokenizer')
                _train_tok_path = os.path.join(cd, f'tok_{tag}.json')
                if _train_tok_sha:
                    if not os.path.exists(_train_tok_path):
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: train metadata binds a tokenizer file '
                              'that is missing')
                    else:
                        try:
                            with open(_train_tok_path, 'rb') as _tfh2:
                                _actual_train_tok_sha = _hl.sha256(_tfh2.read()).hexdigest()
                        except Exception as _tke2:
                            rec['status'] = 'fail'
                            _fail(f'{cd}/{tag}: train tokenizer file unreadable '
                                  f'({type(_tke2).__name__}: {_tke2})')
                        else:
                            if str(_actual_train_tok_sha) != str(_train_tok_sha):
                                rec['status'] = 'fail'
                                _fail(f'{cd}/{tag}: train tokenizer file sha256 '
                                      'does not match its metadata (tokenizer '
                                      'was replaced)')
                _test_tok_path = os.path.join(cd, f"test_tok_{_ttag}.json")
                if _test_tok_sha:
                    if not os.path.exists(_test_tok_path):
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: test metadata binds a tokenizer file '
                              'that is missing')
                    else:
                        try:
                            with open(_test_tok_path, 'rb') as _tfh:
                                _actual_tok_sha = _hl.sha256(_tfh.read()).hexdigest()
                        except Exception as _tke:
                            rec['status'] = 'fail'
                            _fail(f'{cd}/{tag}: test tokenizer file unreadable '
                                  f'({type(_tke).__name__}: {_tke})')
                        else:
                            if str(_actual_tok_sha) != str(_test_tok_sha):
                                rec['status'] = 'fail'
                                _fail(f'{cd}/{tag}: test tokenizer file sha256 '
                                      'does not match its metadata (tokenizer '
                                      'was replaced)')
                _test_fit = _tmeta.get('fit_fp')
                _train_fit = meta.get('fit_fp')
                if _train_fit and _test_fit and str(_train_fit) != str(_test_fit):
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: test fit_fp differs from the train '
                          'cache fit_fp (different tokenizer-fit corpus)')
                if np.issubdtype(np.asarray(te).dtype, np.integer):
                    rec['nonfinite_test'] = 0
                else:
                    _bad_te = int(np.size(te) - np.isfinite(te).sum())
                    rec['nonfinite_test'] = _bad_te
                    if _bad_te:
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: {_bad_te} NaN/Inf values in the test split')
                if np.asarray(te).ndim == 2:
                    _dup_te = int(DG.duplicate_mask(np.asarray(_te_scan)).sum())
                    rec['duplicate_rows_within_test'] = _dup_te
                    if _dup_te:
                        _warn(f'{cd}/{tag}: {_dup_te} duplicate test rows')
                shared_test = _row_overlap_count(tr, _te_scan)
                shared_val_test = _row_overlap_count(_va_scan, _te_scan)
                rec['exact_row_overlap_train_test'] = int(shared_test)
                rec['exact_row_overlap_val_test'] = int(shared_val_test)
                if shared_test:
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: {shared_test} test rows exactly duplicate a training row')
                if shared_val_test:
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: {shared_val_test} test rows exactly duplicate a validation row')
                _ted_path = os.path.join(cd, f'test_doc_ids_{_ttag}.npy')
                if os.path.exists(_ted_path):
                    if not (os.path.exists(_trd_path) and os.path.exists(_vad_path)):
                        # The old code substituted empty arrays for the
                        # missing train/val sidecars, which reported zero
                        # overlap and certified a test group split that was
                        # never checked.
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: test document-id sidecar is present '
                              'but the train/validation doc-id sidecars are '
                              'missing; test group isolation cannot be verified')
                    try:
                        _ted = np.load(_ted_path)
                        _trd = np.load(_trd_path)
                        _vad = np.load(_vad_path)
                        _dtrain = int(np.intersect1d(_trd, _ted).size)
                        _dval = int(np.intersect1d(_vad, _ted).size)
                        rec['test_documents'] = int(_ted.size)
                        rec['test_train_document_overlap'] = _dtrain
                        rec['test_validation_document_overlap'] = _dval
                        _md_t = _tmeta.get('doc_ids') or {}
                        if _md_t.get('test_fp') is not None:
                            _Lfp3 = _import_exp_lib()
                            if str(_Lfp3._ids_fp(_ted)) != str(_md_t['test_fp']):
                                rec['status'] = 'fail'
                                _fail(f'{cd}/{tag}: test document-id sidecar does '
                                      'not match the fingerprint in its metadata (a '
                                      'stale/mixed sidecar cannot certify isolation)')
                        if _dtrain:
                            rec['status'] = 'fail'
                            _fail(f'{cd}/{tag}: {_dtrain} test document ids also occur in train (group leakage)')
                        if _dval:
                            rec['status'] = 'fail'
                            _fail(f'{cd}/{tag}: {_dval} test document ids also occur in validation (selection/test group leakage)')
                    except Exception as _de:
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: test document-id sidecar unreadable ({type(_de).__name__}: {_de})')
                else:
                    rec['test_document_isolation'] = 'not_verified (no test_doc_ids sidecar)'
            else:
                rec['test_file'] = None

            # Token-distribution shift: compare holdout marginals with the
            # training reference (never the other way round).
            tr_tok = _sample_tokens(tr, cap=1000000, seed=0)
            va_tok = _sample_tokens(va, cap=1000000, seed=1)
            rec['train_val_shift'] = DG.categorical_shift_report(tr_tok, va_tok)
            # A handful of holdout tokens cannot estimate a distribution; mark
            # the reading as unreliable instead of emitting a huge JS value.
            rec['train_val_shift']['reliable'] = bool(
                tr_tok.size >= int(min_shift_tokens) and
                va_tok.size >= int(min_shift_tokens))
            jsv = rec['train_val_shift']['js_divergence']
            if rec['train_val_shift']['reliable']:
                if jsv >= float(js_fail):
                    _warn(f'{cd}/{tag}: train/val token JS divergence {jsv:.3f} >= {js_fail} — material distribution shift')
                elif jsv >= float(js_warn):
                    _warn(f'{cd}/{tag}: train/val token JS divergence {jsv:.3f} >= {js_warn}')
            else:
                _warn(f'{cd}/{tag}: train/val distribution shift not estimable from {va_tok.size} holdout tokens (need >= {min_shift_tokens})')
            if te is not None:
                te_tok = _sample_tokens(te, cap=1000000, seed=2)
                rec['train_test_shift'] = DG.categorical_shift_report(tr_tok, te_tok)
                rec['train_test_shift']['reliable'] = bool(
                    tr_tok.size >= int(min_shift_tokens) and
                    te_tok.size >= int(min_shift_tokens))
                jst = rec['train_test_shift']['js_divergence']
                if rec['train_test_shift']['reliable']:
                    if jst >= float(js_fail):
                        _warn(f'{cd}/{tag}: train/test token JS divergence {jst:.3f} >= {js_fail} — material distribution shift')
                    elif jst >= float(js_warn):
                        _warn(f'{cd}/{tag}: train/test token JS divergence {jst:.3f} >= {js_warn}')
                else:
                    _warn(f'{cd}/{tag}: train/test distribution shift not estimable from {te_tok.size} test tokens (need >= {min_shift_tokens})')
            # A warning is a property of this corpus, not only of the global
            # audit: mark the per-cache record so a caller inspecting one cache
            # does not read it as clean while the top-level status says warnings.
            if rec['status'] == 'ok' and len(out['warnings']) > _nwarn_before:
                rec['status'] = 'warnings'
            cache['corpora'][tag] = rec
            if rec['status'] == 'fail':
                cache['status'] = 'fail'
                out['status'] = 'fail'
            elif rec['status'] == 'warnings':
                # _warn already raised out['status'] to 'warnings', so gating
                # this branch on out['status'] == 'ok' would leave the cache
                # record labelled clean.
                if cache['status'] == 'ok':
                    cache['status'] = 'warnings'
                if out['status'] == 'ok':
                    out['status'] = 'warnings'
        if not cache['corpora']:
            cache['status'] = 'not_verifiable_locally'
    # An existing but empty cache directory proves nothing: -- like "no cache
    # at all" -- the split contract was not verified.  Do not leave the
    # top-level verdict as 'ok' in that case.
    if (not out['caches']
            or not any(c.get('corpora') for c in out['caches'].values())):
        if out['status'] == 'ok':
            out['status'] = 'not_verifiable_locally'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
# 1b) row unit: line vs article (the WikiText "document" is a raw line)
# ---------------------------------------------------------------------------

def _unit_meta_int(meta, key):
    """Integer metadata field or None (bool/fraction/NaN are not ints).

    Integral values are returned without a float round-trip: a JSON integer
    above 2**53 would otherwise be silently rounded to a different count.
    """
    v = (meta or {}).get(key)
    if v is None or isinstance(v, (bool, np.bool_)):
        return None
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, (str, bytes, bytearray)):
        # Keep the historical acceptance of integral numeric strings (JSON
        # object keys / hand-written metadata) while still rejecting '3.5'.
        # Parse an exact decimal-integer spelling first so a count above
        # 2**53 is not rounded through float64.
        try:
            return int(v)
        except (TypeError, ValueError, OverflowError):
            pass
        try:
            f = float(v)
        except (TypeError, ValueError, OverflowError):
            return None
        return int(f) if math.isfinite(f) and f == math.floor(f) else None
    try:
        i = int(v)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        if v != i:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return int(i)


def _unit_decode_runs(ids, ttd, tok, L, label, rec):
    """Decode a per-token document map into ``[(span, text, id)]`` runs."""
    import numpy as np
    flat = np.asarray(ids).reshape(-1)
    doc = np.asarray(ttd).reshape(-1)
    if doc.size < flat.size:
        rec['problems'].append(
            f'{label}: per-token document map covers {doc.size} tokens but the '
            f'id array has {flat.size}; the served slice is not fully mapped')
        return None
    if doc.size > flat.size:
        # The loaders cap the validation stream at val_seqs*seq_len + seq_len
        # but serve only the first n_val*seq_len tokens.  The sidecar therefore
        # carries one extra (unserved) sequence of ids by construction.  The
        # map is written in stream order, so the served rows are its prefix;
        # trim the unserved tail and say so instead of rejecting a normal cache.
        # A shorter map stays a hard error (above): that would mean some served
        # token has no document identity at all.
        rec['trimmed_unserved_tokens'] = int(doc.size - flat.size)
        doc = doc[:flat.size]
    try:
        spans = L.line_spans_from_token_doc_ids(doc)
    except (TypeError, ValueError, OverflowError) as e:
        rec['problems'].append(f'{label}: {type(e).__name__}: {e}')
        return None
    texts, dprobs = L.decode_token_runs(flat, spans, tok)
    rec['decoded_runs'] = int(len(spans))
    if dprobs:
        rec['problems'].extend(f'{label}: {p}' for p in dprobs)
    return list(zip(spans, texts))


def _unit_classify_runs(runs, L):
    """Classify a served slice as line-unit, article-unit or unknown.

    A run is a maximal stretch of equal ids in the per-token map, so its id is
    the hash of the *unit* the cache grouped by.  Both a line cache and an
    article cache reproduce their own text hash, so hashing alone cannot tell
    them apart; the structural signal is that a line cache's runs include many
    non-heading lines, while an article cache has one run per article and
    every run starts with a top-level heading.  The id/text hash is still
    checked, so a map that does not describe the decoded bytes is refused.
    """
    n = 0
    hash_ok = 0
    heading_runs = 0
    multiline_runs = 0
    failures = []
    for (_s, _e, _doc), _text in runs:
        if _text is None:
            continue
        if int(L._doc_id_u64(L._doc_exact_fp(_text))) == int(_doc):
            hash_ok += 1
        else:
            failures.append(n)
        n += 1
        _s_body = _text.strip()
        if _s_body:
            _first = _s_body.split('\n', 1)[0].rstrip('\r')
            # Full-line form covers a single decoded raw line; the prefix form
            # covers a whole-article run whose title and body are separated by
            # spaces (per-line EOS tokens are skipped by decode).
            if (L._WIKITEXT_TOP_HEADING_RE.match(_first)
                    or L._WIKITEXT_TOP_HEADING_PREFIX_RE.match(_s_body)):
                heading_runs += 1
            if '\n' in _s_body:
                multiline_runs += 1
    if n == 0:
        return {'unit': 'unknown', 'reason': 'no decoded runs'}
    # The served slice ends at a row/token boundary, so the FINAL line can be
    # cut mid-line (its doc id is the hash of the full raw line).  That is an
    # expected cap artefact, not a unit mismatch; allow exactly that one run.
    truncated_final = bool(failures == [n - 1])
    _common = {'hash_matches': hash_ok, 'n': n,
               'heading_runs': heading_runs,
               'multiline_runs': multiline_runs,
               'truncated_final_run': truncated_final}
    # An article-unit run is a whole article and therefore starts with a
    # top-level heading.  Its stored id may hash a different canonical form of
    # the article text (e.g. raw lines joined with '\n' vs the decoded token
    # run), so the id hash is diagnostic here, not the unit test.  The line
    # case still requires the exact decode hash (apart from the cap-truncated
    # final run): that is what proves the map describes these bytes.
    if heading_runs == n:
        if failures and not truncated_final:
            return dict(_common, unit='unknown', failures=failures[:8],
                        reason=('every run is headed like an article, but its '
                                'stored id does not hash the decoded article '
                                'text (or the cap-truncated final run)'))
        return dict(_common, unit='article',
                    reason='every served run starts with a top-level article heading')
    if failures and not truncated_final:
        return dict(_common, unit='unknown', failures=failures[:8],
                    reason=('served ids do not hash the decoded run text; the '
                            'cache and its tokenizer/sidecar do not describe the '
                            'same bytes'))
    if heading_runs > 0:
        return dict(_common, unit='line',
                    reason='runs include non-heading raw lines')
    return dict(_common, unit='unknown',
                reason='no top-level heading in the served runs; line/article '
                       'unit is not self-describing')


def _unit_group_lines(runs, L):
    """Regroup decoded line runs into articles (without re-reading the corpus).

    ``runs`` is the ``[((start, stop, id), text), ...]`` form returned by
    ``_unit_decode_runs``.
    """
    import numpy as np
    texts = [r[1] for r in runs]
    if any(t is None for t in texts):
        return None, 'a decoded line is missing; cannot regroup into articles'
    g = L.wikitext_article_groups(texts)
    if not g.get('has_headings'):
        return None, ('no top-level heading in the decoded line text; the '
                      'article unit cannot be recovered from this slice')
    line_to_group = np.full(len(texts), -1, dtype=np.int64)
    for _gi, _g in enumerate(g['groups']):
        line_to_group[np.asarray(_g, dtype=np.int64)] = _gi
    if int((line_to_group < 0).sum()):
        # A run decoded to empty/whitespace and was skipped by the grouping;
        # indexing art_ids[-1] would silently attach it to the last article.
        return None, (f'{int((line_to_group < 0).sum())} decoded run(s) are '
                      'empty and cannot be assigned to an article')
    art_ids = np.asarray(L.wikitext_article_ids(texts, g['groups']), dtype=np.uint64)
    lengths = np.asarray([int(r[0][1]) - int(r[0][0]) for r in runs],
                         dtype=np.int64)
    token_art = np.repeat(art_ids[line_to_group], lengths)
    return {'groups': g, 'line_to_group': line_to_group,
            'article_ids': art_ids, 'token_article_ids': token_art}, None


def _article_word_count(text):
    """Whitespace/punctuation-delimited word count used by the article scan."""
    import re
    return int(len(re.findall(r"[\w']+", str(text).lower())))


def _sample_articles(items, cap):
    """Deterministically scan all or an evenly spaced subset of articles.

    A clean prefix is not a clean split: it can miss an article-level copy
    that occurs later in the corpus.  When a caller imposes a cap, sample
    evenly spaced positions across the **whole** list (not the first N) and
    return coverage metadata, so a truncated scan can be reported honestly
    instead of being certified as clean.
    """
    import numpy as np
    if cap is not None:
        cap = _audit_positive_int('max_articles', cap)
    total = len(items)
    if cap is None or total <= int(cap):
        return list(items), {
            'design': 'full', 'n_total': int(total), 'n_scanned': int(total),
            'truncated': False, 'coverage': 1.0,
        }
    idx = np.unique(np.linspace(0, total - 1, int(cap), dtype=np.int64))
    return [items[int(i)] for i in idx], {
        'design': 'deterministic_even_spacing',
        'n_total': int(total), 'n_scanned': int(idx.size),
        'truncated': True,
        'coverage': round(float(idx.size) / float(total), 6),
        'note': ('a clean subset is not a clean split; the audit is not a '
                 'full-corpus proof when truncated'),
    }


def _article_units_from_runs(runs, grouped=None, *, truncated_last=False):
    """Turn decoded line runs into article units for leakage comparison.

    ``runs`` is ``[((start, stop, id), text), ...]``.  ``grouped`` is the
    result of :func:`_unit_group_lines` for a line-unit cache (so served lines
    are regrouped by top-level heading); ``None`` means every run is already
    one whole-article run (article-unit cache).  Only served runs enter a
    unit, which is exactly the text the model or evaluator saw.
    """
    units = []
    if grouped is None:
        seqs = [[i] for i in range(len(runs))]
        has_heading = [True] * len(runs)
    else:
        seqs = grouped['groups']['groups']
        has_heading = list(grouped['groups']['group_has_heading'])
    for _gi, _g in enumerate(seqs):
        _lines = []
        _ids = []
        for _pos in _g:
            _pos = int(_pos)
            _text = runs[_pos][1]
            if _text is None:
                continue
            _lines.append(str(_text).rstrip('\r\n'))
            _ids.append(int(runs[_pos][0][2]))
        if not _lines:
            continue
        _text = '\n'.join(_lines)
        _is_last = bool(_g and int(_g[-1]) == len(runs) - 1)
        units.append({
            'title': _lines[0].strip(),
            'text': _text,
            'line_ids': _ids,
            'n_lines': int(len(_lines)),
            'n_words': _article_word_count(_text),
            'has_heading': bool(has_heading[_gi]),
            'truncated': bool(truncated_last and _is_last),
        })
    return units


def _article_leakage_scan(reference_articles, test_articles, *,
                          reference_line_hashes=None,
                          near_threshold=None, min_words=20):
    """Find article-level test contamination the line index cannot see.

    ``reference_articles`` maps ``train``/``validation`` to
    ``[(title, served_article_text), ...]``: the text actually served to the
    model/selection stage.  ``test_articles`` are units reconstructed from the
    **served test cache** (never from ``wiki.test.raw``).  A test unit whose
    served lines are all absent from the raw train/validation line indexes but
    whose whole article is an exact/near copy of a reference article is the
    case line-level dedup cannot see; it is a hard failure.  A unit with an
    exact line hit is counted as ``line_visible`` and left to the line-index
    audits instead of being double-counted here.

    Short units below ``min_words`` are not comparable by whole-article
    SimHash; they are reported as uncovered, not silently certified.  Title
    collisions alone are a warning, because a heading is not content proof.
    """
    L = _import_exp_lib()
    if reference_line_hashes is None:
        reference_line_hashes = {}
    threshold = (int(L.DEDUP_PROTOCOL_DEFAULTS['near_hamming_threshold'])
                 if near_threshold is None else int(near_threshold))
    if isinstance(min_words, bool) or not isinstance(min_words, int) or min_words < 1:
        raise ValueError(f'min_words must be a positive integer, got {min_words!r}')
    out = {
        'status': 'ok',
        'test_raw_read': False,
        'reference_scope': 'served_train_validation_articles',
        'line_index_scope': 'all_raw_train_validation_lines',
        'near_hamming_threshold': threshold,
        'min_article_words': int(min_words),
        'test_articles_scanned': 0,
        'test_articles_line_visible': 0,
        'test_articles_too_short': 0,
        'test_articles_without_heading': 0,
        'test_articles_truncated': 0,
        'matches': [],
        'title_collisions': [],
        'references': {},
        'problems': [],
        'warnings': [],
    }
    refs = {}
    ref_titles = {}
    # A scale-up corpus has millions of reference articles.  Comparing every
    # test article with every reference is O(test * train) 64-bit hamming
    # operations, which is not a usable audit on WikiText-103, let alone a
    # larger corpus.  Build the same exact-hash map plus SimHash LSH bands the
    # cleaning protocol uses, and query only the candidate buckets.  The LSH
    # guarantee (every pair within the threshold shares a whole band) needs
    # ``threshold < bands`` and a band layout that covers all 64 bits; when a
    # caller overrides either, fall back to the exhaustive scan so the result
    # stays exact rather than silently missing a near copy.
    _bands = int(L.DEDUP_PROTOCOL_DEFAULTS['bands'])
    _use_lsh = (_bands > 0 and 0 <= threshold < _bands
                and 64 % _bands == 0)
    _band_bits = (64 // _bands) if _use_lsh else 0
    _band_mask = ((1 << _band_bits) - 1) if _use_lsh else 0
    _exact_by_split = {}
    _flat = []
    _buckets = {}
    for _split, _items in reference_articles.items():
        _entries = []
        _titles = set()
        _exact = _exact_by_split.setdefault(_split, {})
        for _title, _text in _items:
            if not _text:
                continue
            _titles.add(str(_title))
            _nw = _article_word_count(_text)
            if _nw < int(min_words):
                continue
            _entry = {
                'title': str(_title),
                'text_hash': L._doc_exact_fp(_text),
                'simhash': int(L._doc_simhash(_text)),
                'n_words': int(_nw),
            }
            _entry_pos = int(len(_flat))
            _flat.append((_split, _entry))
            _exact.setdefault(_entry['text_hash'], _entry)
            if _use_lsh:
                _h = int(_entry['simhash'])
                for _b in range(_bands):
                    _key = (_b, (_h >> (_b * _band_bits)) & _band_mask)
                    _buckets.setdefault(_key, []).append(_entry_pos)
            _entries.append(_entry)
        refs[_split] = _entries
        ref_titles[_split] = _titles
        out['references'][_split] = {
            'n_articles': int(len(_items)),
            'n_comparable': int(len(_entries)),
            'n_titles': int(len(_titles)),
        }
    if not any(refs.values()):
        out['status'] = 'not_verifiable'
        out['problems'].append(
            'no reference article has at least the minimum word count; '
            'whole-article test contamination cannot be compared')
        return out
    for _art in test_articles:
        if _art.get('truncated'):
            out['test_articles_truncated'] += 1
            continue
        if not _art.get('has_heading'):
            out['test_articles_without_heading'] += 1
            continue
        if int(_art.get('n_words') or 0) < int(min_words):
            out['test_articles_too_short'] += 1
            continue
        _visible = False
        for _split, _hashes in reference_line_hashes.items():
            if any(int(_h) in _hashes for _h in _art.get('line_ids') or ()):
                _visible = True
                break
        if _visible:
            out['test_articles_line_visible'] += 1
            continue
        out['test_articles_scanned'] += 1
        _text = _art.get('text') or ''
        _h = int(L._doc_simhash(_text))
        _text_hash = L._doc_exact_fp(_text)
        # Best near candidate per split; the split order determines which
        # reference is reported, matching the historical exhaustive scan.
        _near = {}
        if _use_lsh:
            _seen = set()
            for _b in range(_bands):
                _key = (_b, (_h >> (_b * _band_bits)) & _band_mask)
                for _pos in _buckets.get(_key, ()):
                    if _pos in _seen:
                        continue
                    _seen.add(_pos)
                    _split, _e = _flat[_pos]
                    _dist = int((_h ^ int(_e['simhash'])).bit_count())
                    if _dist > threshold:
                        continue
                    _prev = _near.get(_split)
                    if _prev is None or (_dist, _pos) < (_prev[0], _prev[1]):
                        _near[_split] = (_dist, _pos, _e)
        else:
            for _split, _entries in refs.items():
                for _pos, _e in enumerate(_entries):
                    _dist = int((_h ^ int(_e['simhash'])).bit_count())
                    _prev = _near.get(_split)
                    if _prev is None or _dist < _prev[0]:
                        _near[_split] = (_dist, _pos, _e)
        _found = False
        for _split in refs:
            _e = _exact_by_split.get(_split, {}).get(_text_hash)
            if _e is not None:
                out['matches'].append({
                    'kind': 'exact', 'reference_split': _split,
                    'test_title': _art.get('title'),
                    'reference_title': _e['title'],
                    'test_line_ids': list(_art.get('line_ids') or ()),
                    'hamming': 0,
                })
                _found = True
                break
            _best = _near.get(_split)
            if _best is not None and _best[0] <= threshold:
                out['matches'].append({
                    'kind': 'near', 'reference_split': _split,
                    'test_title': _art.get('title'),
                    'reference_title': _best[2]['title'],
                    'test_line_ids': list(_art.get('line_ids') or ()),
                    'hamming': int(_best[0]),
                })
                _found = True
                break
        if not _found:
            for _split, _titles in ref_titles.items():
                if str(_art.get('title')) in _titles:
                    out['title_collisions'].append({
                        'reference_split': _split,
                        'test_title': _art.get('title'),
                    })
    if out['matches']:
        out['status'] = 'fail'
        out['problems'].append(
            f'{len(out["matches"])} test article(s) are exact/near duplicates '
            'of a served train/validation article although none of their '
            'served lines is an exact line-index hit; line-level dedup cannot '
            'see this')
        out['problems'].append(
            'examples: ' + ', '.join(
                f'{m["kind"]}({m["test_title"]!r}->{m["reference_title"]!r}, '
                f'{m["reference_split"]}, hamming={m["hamming"]})'
                for m in out['matches'][:5]))
    if out['title_collisions']:
        if out['status'] == 'ok':
            out['status'] = 'warnings'
        out['warnings'].append(
            f'{len(out["title_collisions"])} test article title(s) also occur '
            'in a served reference article, but no line-invisible article copy '
            'was found; a title is not content proof')
    if out['test_articles_scanned'] == 0:
        out['status'] = 'not_verifiable'
        out['problems'].append(
            'no test article was eligible for whole-article comparison '
            f'(line_visible={out["test_articles_line_visible"]}, '
            f'too_short={out["test_articles_too_short"]}, '
            f'without_heading={out["test_articles_without_heading"]}, '
            f'truncated={out["test_articles_truncated"]})')
    elif (out['test_articles_too_short'] or out['test_articles_truncated']
          or out['test_articles_without_heading']):
        if out['status'] == 'ok':
            out['status'] = 'warnings'
        out['warnings'].append(
            'the whole-article scan did not cover every test article '
            f'(too_short={out["test_articles_too_short"]}, '
            f'truncated={out["test_articles_truncated"]}, '
            f'without_heading={out["test_articles_without_heading"]}); a clean '
            'scanned subset is not a clean split')
    return out


def audit_document_unit(cache_dir=None, *, seq_len=None, raw_dir=None,
                        max_articles=None):
    """Audit the *unit of analysis* of the cached WikiText splits.

    The HF ``wikitext-103-raw-v1`` builder yields one row per raw **line**:
    every article is many rows (a `` = Title = `` heading plus its paragraphs
    and `` = = Section = = `` headings).  The loaders therefore persist line
    ids, and every downstream "per-document" statistic is a per-*line*
    statistic unless the rows are regrouped by their top-level headings.

    This zero-GPU audit makes that distinction executable:

      * it decodes the cached per-token map and classifies the served unit as
        ``line``, ``article`` or ``unknown`` from the run structure, checking
        the decoded bytes against the stored ids (a missing map is
        ``not_verifiable_locally``; test raw is never read to replace it);
      * for a line-unit slice it regroups the decoded lines by top-level
        headings and reports ``n_lines`` / ``n_articles`` and the
        pseudoreplication factor, plus the number of distinct articles in the
        frozen evaluation prefix (the honest denominator for the repo's
        64-document population floor);
      * when the raw train/validation dumps are locally available it checks
        **article-level** leakage the line index cannot see: an article whose
        lines straddle train and validation, and near-duplicate articles whose
        individual lines were all reworded;
      * it reconstructs the served **test** articles from the decoded cache
        (never from ``wiki.test.raw``) and checks them against the served
        train/validation articles.  A test article whose served lines all miss
        the raw line index but whose whole text is an exact/near copy is the
        leak that line-level dedup cannot see; it is a hard failure.  Test
        articles with an exact line hit are counted as ``line_visible`` and
        left to the line-index audits, so the two checks are not
        double-counted;
      * an article scan is full by default.  When ``max_articles`` caps it,
        the sample is evenly spaced over the whole list (not a prefix), and
        the coverage/truncation is recorded.  A clean truncated scan is
        ``not_verifiable_locally``, never a clean-split proof.

    A line-unit cache is reported as ``warnings`` with
    ``document_level_claims_allowed=False``; it is not silently relabelled.
    Article-level leakage is a hard failure.  Missing caches/raw data are
    ``not_verifiable_locally``, never "clean".

    ``seq_len`` optionally disambiguates a cache directory holding several
    training sequence lengths (it selects the train meta with that
    ``seq_len``); without it, an ambiguous train/test pairing is refused rather
    than guessed.  ``raw_dir`` overrides the ``WT103_RAW_TXT`` / ``./wt103_raw``
    discovery used only for the train/validation article-leakage section.
    ``max_articles=None`` (the default) scans every served article; a finite
    cap is an explicit audit sample and is reported as such.
    """
    L = _import_exp_lib()
    import glob
    import numpy as np
    if seq_len is not None:
        seq_len = _audit_positive_int('seq_len', seq_len)
    if max_articles is not None:
        max_articles = _audit_positive_int('max_articles', max_articles)
    if cache_dir in (None, ''):
        cache_dir = os.environ.get('WT103_CACHE_DIR') or './wt103_cache'
    out = {'kind': 'document_unit_audit', 'status': 'ok', 'caches': {},
           'warnings': [], 'problems': [],
           'note': ('WikiText rows are raw lines, not articles; a per-row '
                    '"document" statistic is per-line unless regrouped by the '
                    'top-level " = Title = " headings.  This audit never reads '
                    'wiki.test.raw.')}
    if not os.path.isdir(cache_dir):
        out['status'] = 'not_verifiable_locally'
        out['note'] = f'{cache_dir!r} does not exist — no cache to audit'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    def _warn(msg):
        out['warnings'].append(msg)
        if out['status'] == 'ok':
            out['status'] = 'warnings'

    def _fail(msg):
        out['problems'].append(msg)
        out['status'] = 'fail'

    _tm = sorted(glob.glob(os.path.join(cache_dir, 'test_meta_*.json')))
    _vm = sorted(p for p in glob.glob(os.path.join(cache_dir, 'meta_*.json'))
                 if not os.path.basename(p).startswith('test_meta_'))
    if not _tm or not _vm:
        out['status'] = 'not_verifiable_locally'
        out['note'] = (f'{cache_dir}/ lacks test_meta_*.json or meta_*.json — '
                       'build the caches once before auditing the unit')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    _pairs = []
    for _tp in _tm:
        _t = _load(_tp)
        if not isinstance(_t, dict):
            continue
        for _mp in _vm:
            _m = _load(_mp)
            if not isinstance(_m, dict):
                continue
            _tv, _mv = _unit_meta_int(_t, 'vocab'), _unit_meta_int(_m, 'vocab')
            if _tv is not None and _mv is not None and _tv != _mv:
                continue
            if (_t.get('fit_fp') and _m.get('fit_fp')
                    and _t['fit_fp'] != _m['fit_fp']):
                continue
            _pairs.append((_mp, _m, _tp, _t))
    if seq_len is not None:
        _pref = [(mp, m, tp, t) for mp, m, tp, t in _pairs
                 if _unit_meta_int(m, 'seq_len') == int(seq_len)]
        if _pref:
            _pairs = _pref
    if len(_pairs) != 1:
        out['status'] = 'not_verifiable_locally'
        out['note'] = (f'{len(_pairs)} compatible train/test cache pair(s) '
                       'found; refusing to guess which one defines the served '
                       'unit' if _pairs else
                       'no train/test cache pair shares vocab + tokenizer-fit provenance')
        if _pairs:
            out['checks'] = {'candidate_pairs': [
                [os.path.basename(p[0]), os.path.basename(p[2])] for p in _pairs]}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    _mp, _mmeta, _tp, _tmeta = _pairs[0]
    _ds = _mmeta.get('dataset') or _tmeta.get('dataset')
    if _ds not in (None, '', 'wikitext'):
        out['status'] = 'not_verifiable_locally'
        out['note'] = (f'the paired cache declares dataset={_ds!r}; this audit '
                       'defines the WikiText raw-line/article unit and does not '
                       'assume it for another corpus')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    _train_tag = os.path.basename(_mp)[len('meta_'):-len('.json')]
    _test_tag = os.path.basename(_tp)[len('test_meta_'):-len('.json')]
    rec = {'train_meta': os.path.basename(_mp), 'test_meta': os.path.basename(_tp)}
    out['caches'][os.path.abspath(cache_dir)] = rec
    rec['declared_units'] = {'train': _mmeta.get('document_unit'),
                             'test': _tmeta.get('document_unit')}

    # --- test slice: decode line runs and regroup into articles ------------
    _te_path = os.path.join(cache_dir, f'test_ids_{_test_tag}.npy')
    _ttd_path = os.path.join(cache_dir, f'test_token_doc_ids_{_test_tag}.npy')
    _tok_path = os.path.join(cache_dir, f'test_tok_{_test_tag}.json')
    if not os.path.exists(_tok_path):
        _tok_path = os.path.join(cache_dir, f'tok_{_train_tag}.json')
    trec = {'status': 'ok', 'problems': []}
    rec['test'] = trec
    _test_article_units = None
    if not (os.path.exists(_te_path) and os.path.exists(_ttd_path)
            and os.path.exists(_tok_path)):
        trec['status'] = 'not_verifiable_locally'
        trec['problems'].append(
            'test id cache, per-token document map or persisted tokenizer is '
            'missing; the served unit cannot be decoded')
        _warn('test: ' + trec['problems'][-1])
    else:
        try:
            from tokenizers import Tokenizer
            _te = np.load(_te_path)
            _ttd = np.load(_ttd_path)
            _tok = Tokenizer.from_file(_tok_path)
        except Exception as e:
            trec['status'] = 'not_verifiable_locally'
            trec['problems'].append(
                f'test artifacts unreadable ({type(e).__name__}: {e})')
            _warn('test: ' + trec['problems'][-1])
        else:
            _runs = _unit_decode_runs(_te, _ttd, _tok, L, 'test', trec)
            if _runs is None:
                trec['status'] = 'not_verifiable_locally'
                _warn('test: ' + '; '.join(trec['problems']))
            else:
                _cls = _unit_classify_runs(_runs, L)
                trec['unit_classification'] = _cls
                trec['unit'] = _cls.get('unit')
                _n_rows = int(_te.shape[0]) if np.asarray(_te).ndim == 2 else 0
                _sl = _unit_meta_int(_tmeta, 'seq_len') or (
                    int(_te.shape[1]) if _n_rows else 0)
                _prefix_n = int(np.asarray(_te).size)
                _token_article = None
                if _cls.get('unit') == 'article':
                    trec['n_units'] = int(len(_runs))
                    trec['n_articles'] = int(len(_runs))
                    trec['n_lines'] = None
                    _token_article = np.repeat(
                        np.asarray([int(r[0][2]) for r in _runs],
                                   dtype=np.uint64),
                        np.asarray([int(r[0][1]) - int(r[0][0])
                                    for r in _runs], dtype=np.int64))
                    _test_article_units = _article_units_from_runs(
                        _runs, None,
                        truncated_last=bool(_cls.get('truncated_final_run')))
                elif _cls.get('unit') == 'line':
                    _grp, _gerr = _unit_group_lines(_runs, L)
                    if _grp is None:
                        trec['unit'] = 'unknown'
                        trec['problems'].append(_gerr)
                        _warn(f'test: {_gerr}')
                    else:
                        _token_article = _grp['token_article_ids']
                        trec['n_units'] = int(len(_runs))
                        trec['n_lines'] = int(len(_runs))
                        trec['n_articles'] = int(len(_grp['groups']['groups']))
                        trec['article_grouping'] = {
                            'boundary_rule': _grp['groups']['boundary_rule'],
                            'preamble_groups': int(_grp['groups']['preamble_groups']),
                            'max_lines_per_article': int(_grp['groups']['max_group_rows']),
                            'mean_lines_per_article': round(
                                float(_grp['groups']['mean_group_rows']), 3),
                        }
                        _test_article_units = _article_units_from_runs(
                            _runs, _grp,
                            truncated_last=bool(
                                _cls.get('truncated_final_run')))
                else:
                    trec['status'] = 'fail'
                    _fail('test: ' + (_cls.get('reason') or 'unknown unit'))
                _expected_te = int(_n_rows) * int(_sl) if _n_rows and _sl else 0
                if (_token_article is not None and _n_rows and _sl
                        and int(np.asarray(_te).size) != _expected_te):
                    trec['status'] = 'fail'
                    trec['problems'].append(
                        f'test id cache has {int(np.asarray(_te).size)} '
                        f'token(s) but metadata declares {_n_rows} rows x '
                        f'{_sl} seq_len = {_expected_te}; the served prefix '
                        'geometry is inconsistent')
                    _fail('test: ' + trec['problems'][-1])
                elif _token_article is not None and _n_rows and _sl:
                    _token_article = np.asarray(_token_article)
                    _prefix_n = min(_prefix_n, _n_rows * _sl)
                    _art_pref = _token_article[:_prefix_n]
                    _line_pref = np.asarray(_ttd).reshape(-1)[:_prefix_n]
                    trec['frozen_prefix'] = {
                        'rows': _n_rows, 'seq_len': _sl,
                        'tokens': int(_prefix_n),
                        'distinct_lines': int(np.unique(_line_pref).size)
                        if trec.get('unit') == 'line' else None,
                        'distinct_articles': int(np.unique(_art_pref).size),
                    }
                    _art2d = _art_pref.reshape(_n_rows, _sl)
                    _per_row = [int(np.unique(_art2d[i]).size) for i in range(_n_rows)]
                    trec['frozen_prefix'].update({
                        'mean_articles_per_row': round(float(np.mean(_per_row)), 3),
                        'max_articles_per_row': int(max(_per_row)),
                        'rows_with_multiple_articles': int(sum(1 for x in _per_row if x > 1)),
                    })
                    _n_art = int(np.unique(_art_pref).size)
                    trec['claims'] = {
                        'document_level_claims_allowed': bool(
                            trec.get('unit') == 'article'),
                        'article_population_floor': 64,
                        'population_claim_allowed': bool(_n_art >= 64),
                        'n_independent_articles_in_slice': _n_art,
                        'reason': (
                            'the served unit is a raw line, not an article; a '
                            'line bootstrap resamples paragraphs and cannot '
                            'support a document-level interval or a 64-document '
                            'population claim' if trec.get('unit') == 'line'
                            else 'served unit is article-level'),
                    }
                if trec.get('unit') == 'line':
                    _warn(
                        'test: the cache serves %s line units for %s articles '
                        'in the frozen prefix (pseudoreplication factor %.1fx); '
                        'per-document claims must say "line" or be regrouped'
                        % (trec.get('n_lines'), trec.get('n_articles'),
                           (float(trec.get('n_lines') or 0)
                            / max(1.0, float(trec.get('n_articles') or 1)))))
                _decl = _tmeta.get('document_unit')
                if _decl is not None and trec.get('unit') not in (None, 'unknown', _decl):
                    trec['status'] = 'fail'
                    _fail(f'test: metadata declares document_unit={_decl!r} but '
                          f'the decoded served unit is {trec.get("unit")!r}')
                if trec.get('problems'):
                    trec['status'] = 'fail' if trec.get('status') == 'fail' else 'warnings'

    # --- validation slice: same decode (selection split) -------------------
    _vad = os.path.join(cache_dir, f'val_token_doc_ids_{_train_tag}.npy')
    _va = os.path.join(cache_dir, f'val_batch_{_train_tag}.npy')
    vrec = {'status': 'ok', 'problems': []}
    rec['validation'] = vrec
    if os.path.exists(_vad) and os.path.exists(_va):
        try:
            from tokenizers import Tokenizer
            _vids = np.load(_va)
            _vmap = np.load(_vad)
            _vtok = Tokenizer.from_file(_tok_path)
        except Exception as e:
            vrec['status'] = 'not_verifiable_locally'
            vrec['problems'].append(f'validation artifacts unreadable ({type(e).__name__}: {e})')
        else:
            _vruns = _unit_decode_runs(_vids, _vmap, _vtok, L, 'validation', vrec)
            if _vruns is not None:
                _vcls = _unit_classify_runs(_vruns, L)
                vrec['unit'] = _vcls.get('unit')
                vrec['unit_classification'] = _vcls
                vrec['n_units'] = int(len(_vruns))
                if _vcls.get('unit') == 'line':
                    _vg, _verr = _unit_group_lines(_vruns, L)
                    if _vg is not None:
                        vrec['n_articles'] = int(len(_vg['groups']['groups']))
                        vrec['article_grouping'] = {
                            'boundary_rule': _vg['groups']['boundary_rule'],
                            'max_lines_per_article': int(_vg['groups']['max_group_rows']),
                        }
                    else:
                        vrec['problems'].append(_verr)
                elif _vcls.get('unit') == 'article':
                    vrec['n_articles'] = int(len(_vruns))
            if vrec['problems']:
                vrec['status'] = 'warnings'
                _warn('validation: ' + '; '.join(vrec['problems']))
    else:
        vrec['status'] = 'not_verifiable_locally'
        vrec['problems'].append('validation per-token document map is missing')

    # --- train: line count from the sidecar, article count needs raw ------
    _trd = os.path.join(cache_dir, f'train_doc_ids_{_train_tag}.npy')
    trrec = {'status': 'ok', 'problems': []}
    rec['train'] = trrec
    _served_train = None
    _served_val = None
    if os.path.exists(_trd):
        try:
            _trd_arr = np.load(_trd)
            trrec['served_units'] = int(np.asarray(_trd_arr).size)
            trrec['unit'] = _mmeta.get('document_unit') or (
                vrec.get('unit') or trec.get('unit'))
            _served_train = _unique_int_set(_trd_arr)
        except Exception as e:
            trrec['status'] = 'not_verifiable_locally'
            trrec['problems'].append(f'train document sidecar unreadable ({type(e).__name__}: {e})')
    else:
        trrec['status'] = 'not_verifiable_locally'
        trrec['problems'].append('train document-id sidecar is missing')
    if os.path.exists(os.path.join(cache_dir, f'val_doc_ids_{_train_tag}.npy')):
        try:
            _served_val = _unique_int_set(np.load(
                os.path.join(cache_dir, f'val_doc_ids_{_train_tag}.npy')))
        except Exception:
            _served_val = None

    # --- raw train/validation: article-level leakage the line index misses -
    if raw_dir in (None, ''):
        raw_dir = os.environ.get('WT103_RAW_TXT') or './wt103_raw'
    _raw_tr = os.path.join(raw_dir, 'wiki.train.raw')
    _raw_va = os.path.join(raw_dir, 'wiki.valid.raw')
    rawrec = {'status': 'ok', 'problems': [], 'test_raw_read': False}
    rec['article_leakage'] = rawrec
    terec = None
    if not (os.path.exists(_raw_tr) and os.path.exists(_raw_va)):
        rawrec['status'] = 'not_verifiable_locally'
        rawrec['problems'].append(
            f'raw train/validation dumps not found under {raw_dir!r}; '
            'article-level leakage cannot be tested (test raw is never read)')
        out.setdefault('notes', []).append(rawrec['problems'][-1])
        terec = {
            'status': 'not_verifiable_locally',
            'problems': [
                'raw train/validation dumps are unavailable, so the served '
                'test articles have no line-index reference to compare against'
            ],
            'warnings': [],
        }
        rec['test_article_leakage'] = terec
    elif _served_train is None or _served_val is None:
        rawrec['status'] = 'not_verifiable_locally'
        rawrec['problems'].append(
            'served train/validation line ids are unavailable; cannot map raw '
            'articles to the served streams')
        terec = {
            'status': 'not_verifiable_locally',
            'problems': [
                'served train/validation document ids are unavailable, so the '
                'served test articles have no reference set to compare against'
            ],
            'warnings': [],
        }
        rec['test_article_leakage'] = terec
    else:
        try:
            _tr_rows = L._read_raw_rows(_raw_tr)
            _va_rows = L._read_raw_rows(_raw_va)
            _tr_g = L.wikitext_article_groups(_tr_rows)
            _va_g = L.wikitext_article_groups(_va_rows)
        except Exception as e:
            rawrec['status'] = 'not_verifiable_locally'
            rawrec['problems'].append(f'raw dump unreadable ({type(e).__name__}: {e})')
            terec = {
                'status': 'not_verifiable_locally',
                'problems': [
                    'the raw train/validation dumps are unreadable, so the '
                    'served test articles have no line-index reference to '
                    'compare against'],
                'warnings': [],
            }
            rec['test_article_leakage'] = terec
        else:
            if not (_tr_g.get('valid') and _va_g.get('valid')):
                rawrec['status'] = 'not_verifiable_locally'
                rawrec['problems'].append(
                    'raw dumps are not segmented by top-level headings; '
                    'article-level leakage is unverifiable')
                terec = {
                    'status': 'not_verifiable_locally',
                    'problems': [
                        'the raw train/validation dumps cannot be segmented '
                        'into articles, so the served test articles have no '
                        'article-level reference to compare against'],
                    'warnings': [],
                }
                rec['test_article_leakage'] = terec
            else:
                rawrec.update({
                    'raw_train_lines': int(_tr_g['n_nonempty_rows']),
                    'raw_train_articles': int(_tr_g['n_articles']),
                    'raw_validation_lines': int(_va_g['n_nonempty_rows']),
                    'raw_validation_articles': int(_va_g['n_articles']),
                    'lines_per_article_train': round(
                        float(_tr_g['mean_group_rows']), 3),
                })

                def _served_articles(rows, groups, served):
                    # (article title, article text built from served lines).
                    # The title is the group's FIRST row even when that row was
                    # not itself served: the title is the article's identity,
                    # and a split article is exactly the case where the train
                    # side saw one line and the validation side another.
                    _out = []
                    for _g in groups['groups']:
                        if not _g:
                            continue
                        _title = rows[int(_g[0])].strip()
                        _kept = []
                        for _i in _g:
                            _row = rows[int(_i)]
                            _h = int(L._doc_id_u64(L._doc_exact_fp(_row)))
                            if _h not in served:
                                continue
                            _kept.append(_row.rstrip('\r\n'))
                        if _kept:
                            _out.append((_title, '\n'.join(_kept)))
                    return _out

                _tr_served_all = _served_articles(
                    _tr_rows, _tr_g, _served_train)
                _va_served_all = _served_articles(
                    _va_rows, _va_g, _served_val)
                _tr_served, _tr_scan = _sample_articles(
                    _tr_served_all, max_articles)
                _va_served, _va_scan = _sample_articles(
                    _va_served_all, max_articles)
                rawrec['served_train_articles'] = int(len(_tr_served_all))
                rawrec['served_validation_articles'] = int(len(_va_served_all))
                rawrec['article_scan'] = {
                    'train': _tr_scan, 'validation': _va_scan,
                }
                if _tr_scan.get('truncated') or _va_scan.get('truncated'):
                    rawrec['status'] = 'not_verifiable_locally'
                    rawrec['problems'].append(
                        'the article-level scan is truncated '
                        f'(train {_tr_scan.get("n_scanned")}/'
                        f'{_tr_scan.get("n_total")}, validation '
                        f'{_va_scan.get("n_scanned")}/'
                        f'{_va_scan.get("n_total")}); a clean subset is not a '
                        'clean split — pass max_articles=None for a full scan')
                # Same title on both sides with different served lines is an
                # article split; identical line hashes would already have been
                # an exact-row failure, so the title is the cross-line identity.
                _tr_titles = {t for t, _txt in _tr_served}
                _va_titles = {t for t, _txt in _va_served}
                _shared_titles = _tr_titles & _va_titles
                rawrec['articles_with_served_lines_in_both_splits'] = int(
                    len(_shared_titles))
                if _shared_titles:
                    rawrec['status'] = 'fail'
                    _fail('article leakage: %d article title(s) have served '
                          'lines in BOTH train and validation (line-level exact '
                          'dedup cannot see this): %s'
                          % (len(_shared_titles),
                             sorted(_shared_titles)[:5]))
                # Near-duplicate articles whose individual lines were reworded
                # are invisible to the exact line index.  Build the same
                # SimHash LSH the loader uses and query validation articles
                # against the served train articles.
                try:
                    _ded = L._DocDeduplicator()
                    _n_tr_seen = 0
                    for _t, _txt in _tr_served:
                        if _txt:
                            _ded.is_duplicate(_txt)
                            _n_tr_seen += 1
                    _near = 0
                    _exact = 0
                    _examples = []
                    for _t, _txt in _va_served:
                        if not _txt:
                            continue
                        _dup, _kind = _ded.find(_txt)
                        if _dup:
                            _near += int(_kind == 'near')
                            _exact += int(_kind == 'exact')
                            if len(_examples) < 5:
                                _examples.append({'title': _t, 'kind': _kind})
                    rawrec['article_near_duplicates'] = int(_near)
                    rawrec['article_exact_duplicates'] = int(_exact)
                    rawrec['article_duplicate_examples'] = _examples
                    rawrec['near_hamming_threshold'] = int(_ded.near_threshold)
                    rawrec['train_articles_indexed'] = int(_n_tr_seen)
                    if _near or _exact:
                        if rawrec['status'] != 'fail':
                            rawrec['status'] = 'warnings'
                        _warn(
                            'article-level duplicates missed by the line index: '
                            f'{_exact} exact / {_near} near-duplicate '
                            'validation article(s) (SimHash <= '
                            f'{int(_ded.near_threshold)} bits) against the '
                            'served train articles')
                except Exception as e:
                    rawrec['problems'].append(
                        f'article near-duplicate scan failed ({type(e).__name__}: {e})')
                    rawrec['status'] = 'not_verifiable_locally'
                # Held-out article contamination that the line index cannot
                # see.  The test slice is reconstructed from the SERVED cache
                # (its decoded line runs), so this does not read
                # ``wiki.test.raw``; it compares the served test articles with
                # the train/validation articles the model and selector saw.
                # A unit whose lines all miss the raw line index but whose
                # whole article matches is exactly the leak that line-level
                # dedup cannot detect.
                terec = {'status': 'ok', 'problems': [], 'warnings': []}
                rec['test_article_leakage'] = terec
                if _test_article_units is None:
                    terec['status'] = 'not_verifiable_locally'
                    terec['problems'].append(
                        'served test article units could not be reconstructed '
                        'from the cache (missing/misaligned test sidecar or '
                        'no top-level heading), so test/article-line-index '
                        'contamination cannot be tested without reading '
                        'wiki.test.raw')
                    _warn('test article leakage: ' + terec['problems'][-1])
                else:
                    # The sets are only needed when test units exist; building
                    # them first would hash every raw train/validation line on
                    # a scale-up dump even when the scan cannot run.
                    _tr_line_hashes = {
                        int(L._doc_id_u64(L._doc_exact_fp(_r)))
                        for _r in _tr_rows
                        if isinstance(_r, str) and _r.strip()
                    }
                    _va_line_hashes = {
                        int(L._doc_id_u64(L._doc_exact_fp(_r)))
                        for _r in _va_rows
                        if isinstance(_r, str) and _r.strip()
                    }
                    try:
                        _test_scan = _article_leakage_scan(
                            {'train': _tr_served,
                             'validation': _va_served},
                            _test_article_units,
                            reference_line_hashes={
                                'train': _tr_line_hashes,
                                'validation': _va_line_hashes,
                            },
                            near_threshold=L.DEDUP_PROTOCOL_DEFAULTS[
                                'near_hamming_threshold'])
                    except Exception as _tse:
                        terec['status'] = 'not_verifiable_locally'
                        terec['problems'].append(
                            'test article leakage scan failed '
                            f'({type(_tse).__name__}: {_tse})')
                        _warn('test article leakage: ' + terec['problems'][-1])
                    else:
                        terec.update(_test_scan)
                        if _test_scan.get('status') == 'fail':
                            _fail('test article leakage: ' + '; '.join(
                                _test_scan.get('problems') or []))
                        elif _test_scan.get('status') == 'warnings':
                            _warn('test article leakage: ' + '; '.join(
                                _test_scan.get('warnings') or []))
                        elif _test_scan.get('status') == 'not_verifiable':
                            terec['status'] = 'not_verifiable_locally'
                            _warn('test article leakage: ' + '; '.join(
                                _test_scan.get('problems') or []))
                    if ((_tr_scan.get('truncated')
                         or _va_scan.get('truncated'))
                            and terec.get('status') != 'fail'):
                        # The test scan is only as complete as the reference
                        # sample it was given.  A clean result against a
                        # truncated train/validation sample must not be
                        # reported as a clean per-cache test-article record.
                        terec['status'] = 'not_verifiable_locally'
                        terec.setdefault('problems', []).append(
                            'the train/validation reference scan is truncated, '
                            'so the clean test-article result is not a full-'
                            'corpus proof; pass max_articles=None')
                if rawrec['problems']:
                    # A truncated scan is not the same claim as a clean sample:
                    # keep the explicit not_verifiable status instead of
                    # downgrading it to an ordinary warning.
                    if rawrec['status'] != 'not_verifiable_locally':
                        rawrec['status'] = 'warnings'
                    _warn('article leakage: ' + '; '.join(rawrec['problems']))
                if (_tr_scan.get('truncated') or _va_scan.get('truncated')
                        or terec.get('status') == 'not_verifiable_locally'):
                    if out['status'] == 'ok':
                        out['status'] = 'not_verifiable_locally'

    # --- aggregate verdict -------------------------------------------------
    _test_unit = trec.get('unit')
    rec['unit'] = _test_unit
    rec['document_level_claims_allowed'] = bool(
        _test_unit == 'article' and rawrec.get('status') != 'fail'
        and (rec.get('test_article_leakage') or {}).get('status') != 'fail')
    rec['line_unit_warning'] = bool(_test_unit == 'line')
    if _test_unit == 'line':
        _warn('the served evaluation unit is a raw LINE, not an article; '
              'per-document intervals/population claims are not supported '
              'until the rows are regrouped by top-level headings')
    elif _test_unit not in (None, 'article'):
        _fail(f'test unit is {_test_unit!r}, not line/article')
    if trec.get('claims', {}).get('population_claim_allowed') is False:
        _warn('the frozen prefix contains fewer than the 64 independent '
              'articles the repo requires for a population claim')
    if out['status'] == 'ok' and any(
            r is not None and r.get('status') == 'not_verifiable_locally'
            for r in (trec, vrec, trrec, rawrec, terec)):
        out['status'] = 'not_verifiable_locally'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
# 1b) sampling：前缀切片 vs 概率样本 / 文章覆盖 / 可外推性
# ---------------------------------------------------------------------------

def _sample_ks(a, b):
    """Two-sample Kolmogorov-Smirnov statistic without SciPy."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    if a.size == 0 or b.size == 0:
        return None
    grid = np.unique(np.concatenate([a, b]))
    ca = np.searchsorted(a, grid, side='right') / float(a.size)
    cb = np.searchsorted(b, grid, side='right') / float(b.size)
    return float(np.max(np.abs(ca - cb)))
