
def _dp_int(value):
    """Exact non-negative integer from cache metadata, else ``None``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        return None
    value = int(value)
    return value if value >= 0 else None


def _dp_block(meta, is_test):
    if not isinstance(meta, dict):
        return None
    key = 'near_duplicate_cleanup' if is_test else 'dedup'
    block = meta.get(key)
    return block if isinstance(block, dict) else None


def _dp_protocol_check(protocol, block, problems, warnings, label):
    """Validate one cache's recorded protocol and redundant fields."""
    L = _import_exp_lib()
    if protocol is None:
        threshold = _dp_int((block or {}).get('near_hamming_threshold'))
        default_thr = int(L.DEDUP_PROTOCOL_DEFAULTS['near_hamming_threshold'])
        _partial = [k for k in ('bands', 'shingle_words', 'max_shingles',
                                'protocol_fp')
                    if (block or {}).get(k) is not None]
        if _partial:
            problems.append(
                f'{label}: dedup geometry fields {_partial} are recorded '
                'without a full protocol; the cleaning identity is incomplete')
        if block is None:
            warnings.append(f'{label}: no dedup block; cleaning protocol is unrecorded')
        elif threshold is None:
            warnings.append(f'{label}: legacy dedup block has no full protocol or threshold')
        elif threshold != default_thr:
            problems.append(
                f'{label}: legacy dedup block records threshold={threshold}, but '
                f'the declared protocol threshold is {default_thr}; the shingle/'
                'band geometry is unknown, so this cache cannot be certified')
        else:
            warnings.append(
                f'{label}: legacy dedup metadata (no full protocol); threshold '
                'matches the default but shingle width/bands are unrecorded')
        return None
    pp = L.dedup_protocol_problems(protocol)
    if pp:
        problems.append(f'{label}: invalid dedup protocol: ' + '; '.join(pp))
        return None
    recorded_fp = (block or {}).get('protocol_fp')
    try:
        actual_fp = L.dedup_protocol_fp(protocol)
    except (TypeError, ValueError, OverflowError) as e:
        # A syntactically valid object can still contain an extra
        # non-JSON-safe value (e.g. NaN).  The protocol is then not a stable
        # identity; report it instead of crashing the audit.
        problems.append(
            f'{label}: dedup protocol cannot be fingerprinted '
            f'({type(e).__name__}: {e})')
        return None
    if recorded_fp is None:
        problems.append(f'{label}: dedup protocol has no protocol_fp binding')
    elif str(recorded_fp) != str(actual_fp):
        problems.append(
            f'{label}: dedup protocol_fp does not match its recorded parameters '
            '(edited/mixed metadata)')
    for field in ('near_hamming_threshold', 'bands', 'shingle_words',
                  'max_shingles'):
        recorded = (block or {}).get(field)
        if recorded is None:
            continue
        # The loader writes plain ints.  ``int("4")``/``int(True)``/``int(4.9)``
        # would silently coerce a malformed redundant field into a match, so
        # use the same exact-integer contract as the ledger fields.
        recorded_i = _dp_int(recorded)
        if recorded_i is None:
            problems.append(
                f'{label}: dedup field {field}={recorded!r} is not an exact '
                'non-negative integer; the redundant protocol field is '
                'unverifiable')
            continue
        try:
            expected_i = int(protocol[field])
        except (TypeError, ValueError, OverflowError, KeyError):
            expected_i = None
        if expected_i is None or recorded_i != expected_i:
            problems.append(
                f'{label}: dedup field {field}={recorded!r} disagrees with the '
                f'protocol ({protocol.get(field)!r})')
    if protocol != L.DEDUP_PROTOCOL_DEFAULTS:
        warnings.append(
            f'{label}: dedup protocol differs from the current code default; '
            'this is a different cleaning geometry and must be paired by its '
            'protocol_fp, not by the policy string alone')
    return actual_fp


def _dp_check_file_size(path, expected, problems, warnings, label):
    # ``expected is None`` means the metadata records no count for this
    # sidecar, so there is nothing to bind.  A count with a missing file is a
    # different case: the audit must not report ``ok`` while a declared
    # sidecar cannot be checked.  Keep it a warning (a metadata-only snapshot
    # can still have auditable accounting) but make the false green visible.
    if expected is None:
        return False
    if not os.path.exists(path):
        warnings.append(
            f'{label}: {os.path.basename(path)} is missing; the metadata '
            f'accounts for {int(expected)} value(s) but the sidecar length '
            'cannot be verified')
        return False
    try:
        n = int(np.load(path, mmap_mode='r').size)
    except Exception as e:
        problems.append(f'{label}: cannot read {os.path.basename(path)} '
                        f'({type(e).__name__}: {e})')
        return False
    if n != int(expected):
        problems.append(
            f'{label}: {os.path.basename(path)} holds {n} values but the '
            f'metadata accounts for {int(expected)}')
        return False
    return True


def audit_data_processing(cache_dir=None):
    """Audit the data-processing ledger and the dedup cleaning protocol.

    Token ids alone do not identify a processed dataset.  Two caches can share
    the same `src_fp` yet differ in *how* near-duplicate documents were removed
    (exact hash, SimHash shingle width, LSH bands, Hamming threshold) or in how
    many source rows became served rows.  This zero-GPU audit makes the
    cleaning step part of the auditable data identity:

      * every train/validation/test cache that records a full dedup protocol
        is checked for an intact `protocol_fp`, consistent redundant fields,
        and a geometry that is valid (threshold < bands so the LSH candidate
        search is exhaustive; shingle width <= shingle cap);
      * train/validation and held-out caches paired by
        ``(vocab, tokenizer-fit fingerprint)`` must declare the **same**
        ``protocol_fp`` — a test slice cleaned with different parameters is a
        different dataset and is a hard failure;
      * the cleaning ledger must close arithmetically:
        ``nonempty_rows_scanned - rows_served`` equals the recorded exact+near
        removals, document-sidecar counts equal the sampling counts, and
        ``source_documents_nonempty = kept + all removed categories`` for the
        held-out split;
      * when the ``.npy`` sidecars are present, their lengths are checked
        against the metadata (headers only; no token stream is loaded).

    Legacy caches without a full protocol are reported as warnings (or a hard
    failure when they record a non-default threshold that cannot be audited).
    A missing cache is ``not_verifiable_locally``, never "clean".
    """
    import glob
    L = _import_exp_lib()
    if cache_dir in (None, ''):
        cache_dir = os.environ.get('WT103_CACHE_DIR') or './wt103_cache'
    out = {
        'kind': 'data_processing_audit',
        'status': 'ok',
        'cache_dir': os.path.abspath(cache_dir) if os.path.isdir(cache_dir) else cache_dir,
        'expected_protocol': L.DEDUP_PROTOCOL_DEFAULTS,
        'records': {},
        'problems': [],
        'warnings': [],
        'note': ('Token ids do not identify the cleaning step.  This audit '
                 'binds the exact/SimHash near-duplicate geometry and the '
                 'row/token accounting into the cache identity.'),
    }
    if not os.path.isdir(cache_dir):
        out['status'] = 'not_verifiable_locally'
        out['note'] = f'{cache_dir!r} does not exist — no processed cache to audit'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    def _fail(msg):
        out['problems'].append(msg)
        out['status'] = 'fail'

    def _warn(msg):
        out['warnings'].append(msg)
        if out['status'] == 'ok':
            out['status'] = 'warnings'

    paths = sorted(glob.glob(os.path.join(cache_dir, 'meta_*.json')))
    paths += sorted(glob.glob(os.path.join(cache_dir, 'test_meta_*.json')))
    # meta_*.json and test_meta_*.json overlap only if a train tag starts with
    # "test_"; de-duplicate by absolute path so one file is audited once.
    seen_paths = set()
    metas = []
    for path in paths:
        ap = os.path.abspath(path)
        if ap in seen_paths:
            continue
        seen_paths.add(ap)
        is_test = os.path.basename(path).startswith('test_meta_')
        meta = _load(path)
        if not isinstance(meta, dict):
            _warn(f'{os.path.basename(path)} is not a readable metadata object')
            continue
        tag = (os.path.basename(path)[len('test_meta_'):-len('.json')]
               if is_test else
               os.path.basename(path)[len('meta_'):-len('.json')])
        metas.append((path, tag, is_test, meta))
    if not metas:
        out['status'] = 'not_verifiable_locally'
        out['note'] = f'{cache_dir!r} contains no cache metadata'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    train_rows = []
    test_rows = []
    n_legacy = 0
    n_protocol = 0
    for path, tag, is_test, meta in metas:
        label = f'{os.path.basename(path)}'
        block = _dp_block(meta, is_test)
        protocol = block.get('protocol') if isinstance(block, dict) else None
        rec = {
            'kind': 'test' if is_test else 'train_validation',
            'tag': tag,
            'vocab': _dp_int(meta.get('vocab')),
            'fit_fp': meta.get('fit_fp'),
            'block_recorded': bool(block),
            'protocol': protocol,
            'protocol_fp': (block.get('protocol_fp')
                            if isinstance(block, dict) else None),
            'problems': [],
            'warnings': [],
            'file_checks': {},
            'accounting': {},
        }
        out['records'][label] = rec
        # `(vocab, fit_fp)` is the cross-split pairing key, so these fields
        # must be usable as an identity.  A JSON list/object in either slot
        # would be unhashable (crashing the grouping below) and cannot name a
        # corpus; report it as a metadata problem instead.
        if meta.get('vocab') is not None and rec['vocab'] is None:
            rec['problems'].append(
                f"{label}: vocab={meta.get('vocab')!r} is not a non-negative "
                'integer; the corpus identity is malformed')
        _fit_fp = meta.get('fit_fp')
        if _fit_fp is not None and not isinstance(_fit_fp, str):
            rec['problems'].append(
                f'{label}: fit_fp={_fit_fp!r} is not a string; the corpus '
                'identity is malformed')
            rec['fit_fp'] = None
        actual_fp = _dp_protocol_check(
            protocol, block, rec['problems'], rec['warnings'], label)
        if protocol is not None and actual_fp is not None:
            n_protocol += 1
            rec['protocol_fp'] = str(actual_fp)
        elif block is not None:
            n_legacy += 1

        # ---- cleaning ledger arithmetic ---------------------------------
        if is_test:
            sampling = meta.get('sampling') or {}
            st = sampling.get('test') if isinstance(sampling, dict) else None
            st = st if isinstance(st, dict) else {}
            doc_ids = meta.get('doc_ids') if isinstance(meta.get('doc_ids'), dict) else {}
            src = _dp_int((block or {}).get('source_documents'))
            src_ne = _dp_int((block or {}).get('source_documents_nonempty'))
            kept = _dp_int((block or {}).get('kept_documents'))
            excluded = {}
            for key in ('train_exact_removed', 'train_near_removed',
                        'test_exact_removed', 'test_near_removed'):
                excluded[key] = _dp_int((block or {}).get(key))
            rec['accounting'] = {
                'source_documents': src,
                'source_documents_nonempty': src_ne,
                'kept_documents': kept,
                'removed': excluded,
            }
            if src is not None and kept is not None and src < kept:
                rec['problems'].append(
                    f'{label}: kept_documents={kept} exceeds source_documents={src}')
            if src_ne is not None and kept is not None and src_ne < kept:
                rec['problems'].append(
                    f'{label}: kept_documents={kept} exceeds '
                    f'source_documents_nonempty={src_ne}')
            if kept is not None and all(v is not None for v in excluded.values()):
                removed = sum(int(v) for v in excluded.values())
                rec['accounting']['removed_total'] = int(removed)
                if src_ne is not None:
                    if int(kept) + removed != int(src_ne):
                        rec['problems'].append(
                            f'{label}: held-out cleaning ledger does not close: '
                            f'{src_ne} nonempty source docs != {kept} kept + '
                            f'{removed} removed')
                elif src is not None:
                    if removed > int(src) - int(kept):
                        rec['problems'].append(
                            f'{label}: removed={removed} exceeds '
                            f'source-kept={int(src) - int(kept)} (no nonempty '
                            'source count recorded)')
                    else:
                        rec['warnings'].append(
                            f'{label}: source_documents_nonempty is not recorded; '
                            'blank rows make the cleaning ledger inexact')
            rows_after = _dp_int(st.get('rows_after_dedup'))
            if rows_after is not None and kept is not None and rows_after != kept:
                rec['problems'].append(
                    f'{label}: sampling.rows_after_dedup={rows_after} disagrees '
                    f'with kept_documents={kept}')
            rows_served = _dp_int(st.get('rows_served'))
            tokens_served = _dp_int(st.get('tokens_served'))
            if rows_served is not None and _dp_int(doc_ids.get('test_count')) is not None:
                if rows_served != _dp_int(doc_ids.get('test_count')):
                    rec['problems'].append(
                        f'{label}: sampling.rows_served={rows_served} disagrees '
                        f"with doc_ids.test_count={doc_ids.get('test_count')}")
            if (tokens_served is not None
                    and _dp_int(doc_ids.get('test_token_count')) is not None):
                if tokens_served != _dp_int(doc_ids.get('test_token_count')):
                    rec['problems'].append(
                        f'{label}: sampling.tokens_served={tokens_served} '
                        'disagrees with doc_ids.test_token_count='
                        f"{doc_ids.get('test_token_count')}")
            if src is not None and _dp_int(st.get('source_rows')) is not None:
                if src != _dp_int(st.get('source_rows')):
                    rec['problems'].append(
                        f'{label}: sampling.source_rows={st.get("source_rows")} '
                        f'disagrees with source_documents={src}')
            vc = (block or {}).get('validation_cleanup')
            if isinstance(vc, dict):
                vsrc = _dp_int(vc.get('source_documents'))
                vkept = _dp_int(vc.get('kept_documents'))
                vrem = _dp_int(vc.get('duplicate_documents_removed'))
                if None not in (vsrc, vkept, vrem) and vsrc != vkept + vrem:
                    rec['problems'].append(
                        f'{label}: validation cleanup ledger does not close: '
                        f'{vsrc} source != {vkept} kept + {vrem} removed')
            # File-length binding (headers only).
            base = os.path.dirname(path)
            if tokens_served is not None:
                te = os.path.join(base, f'test_ids_{tag}.npy')
                rec['file_checks']['test_ids_size'] = _dp_check_file_size(
                    te, tokens_served, rec['problems'], rec['warnings'], label)
            if rows_served is not None:
                td = os.path.join(base, f'test_doc_ids_{tag}.npy')
                rec['file_checks']['test_doc_ids_size'] = _dp_check_file_size(
                    td, rows_served, rec['problems'], rec['warnings'], label)
            if tokens_served is not None:
                ttd = os.path.join(base, f'test_token_doc_ids_{tag}.npy')
                rec['file_checks']['test_token_doc_ids_size'] = _dp_check_file_size(
                    ttd, tokens_served, rec['problems'], rec['warnings'], label)
            # Cross-split geometry is a property of the declared protocol, not
            # of whether the (optional for legacy) document-count ledger is
            # complete: keep the record so an edited/missing kept_documents
            # cannot also hide a protocol mismatch.
            if protocol is not None:
                test_rows.append(rec)
        else:
            sampling = meta.get('sampling') or {}
            dedup = block or {}
            splits = dedup.get('splits') if isinstance(dedup.get('splits'), dict) else {}
            doc_ids = meta.get('doc_ids') if isinstance(meta.get('doc_ids'), dict) else {}
            for split in ('train', 'validation'):
                st = sampling.get(split) if isinstance(sampling, dict) else None
                st = st if isinstance(st, dict) else {}
                counts = splits.get(split) if isinstance(splits.get(split), dict) else {}
                skipped = st.get('duplicate_documents_skipped')
                skipped = skipped if isinstance(skipped, dict) else {}
                rec['accounting'][split] = {
                    'nonempty_rows_scanned': _dp_int(st.get('nonempty_rows_scanned')),
                    'rows_served': _dp_int(st.get('rows_served')),
                    'tokens_served': _dp_int(st.get('tokens_served')),
                    'skipped': {k: _dp_int(skipped.get(k))
                                for k in ('exact', 'near')},
                    'dedup_counts': {k: _dp_int(counts.get(k))
                                     for k in ('exact', 'near')},
                }
                for kind in ('exact', 'near'):
                    if (rec['accounting'][split]['skipped'][kind] is not None
                            and rec['accounting'][split]['dedup_counts'][kind] is not None
                            and (rec['accounting'][split]['skipped'][kind]
                                 != rec['accounting'][split]['dedup_counts'][kind])):
                        rec['problems'].append(
                            f'{label}: {split} skipped.{kind}='
                            f"{rec['accounting'][split]['skipped'][kind]} "
                            f"disagrees with dedup.splits.{split}.{kind}="
                            f"{rec['accounting'][split]['dedup_counts'][kind]}")
                scanned = rec['accounting'][split]['nonempty_rows_scanned']
                served = rec['accounting'][split]['rows_served']
                sk = rec['accounting'][split]['skipped']
                if (None not in (scanned, served, sk['exact'], sk['near'])):
                    if int(scanned) - int(served) != int(sk['exact']) + int(sk['near']):
                        rec['problems'].append(
                            f'{label}: {split} cleaning ledger does not close: '
                            f'{scanned} scanned - {served} served != '
                            f"{int(sk['exact']) + int(sk['near'])} removed")
                # Sidecar counts.
                key = 'train_count' if split == 'train' else 'validation_count'
                expected_docs = _dp_int(doc_ids.get(key))
                if expected_docs is not None and served is not None and expected_docs != served:
                    rec['problems'].append(
                        f'{label}: doc_ids.{key}={expected_docs} disagrees with '
                        f'sampling.{split}.rows_served={served}')
                base = os.path.dirname(path)
                if split == 'train':
                    tr = os.path.join(base, f'train_ids_{tag}.npy')
                    rec['file_checks']['train_ids_size'] = _dp_check_file_size(
                        tr, rec['accounting'][split]['tokens_served'],
                        rec['problems'], rec['warnings'], label)
                    trd = os.path.join(base, f'train_doc_ids_{tag}.npy')
                    rec['file_checks']['train_doc_ids_size'] = _dp_check_file_size(
                        trd, expected_docs, rec['problems'], rec['warnings'], label)
                else:
                    va = os.path.join(base, f'val_batch_{tag}.npy')
                    va_expected = None
                    if (meta.get('val_sequences') is not None
                            or meta.get('seq_len') is not None):
                        _va_n = _dp_int(meta.get('val_sequences'))
                        _va_len = _dp_int(meta.get('seq_len'))
                        if _va_n is None or _va_len is None:
                            rec['problems'].append(
                                f'{label}: val_sequences={meta.get("val_sequences")!r} '
                                f'and seq_len={meta.get("seq_len")!r} must both be '
                                'non-negative integers to size the validation '
                                'batch; the sidecar length is unverifiable')
                        else:
                            va_expected = _va_n * _va_len
                    rec['file_checks']['val_batch_size'] = _dp_check_file_size(
                        va, va_expected, rec['problems'], rec['warnings'], label)
                    vad = os.path.join(base, f'val_doc_ids_{tag}.npy')
                    rec['file_checks']['val_doc_ids_size'] = _dp_check_file_size(
                        vad, expected_docs, rec['problems'], rec['warnings'], label)
            # Validation cleanup in the train meta must match the split ledger.
            vc = dedup.get('validation_cleanup')
            if isinstance(vc, dict):
                vcounts = splits.get('validation') if isinstance(splits.get('validation'), dict) else {}
                for src_key, dst_key in (('exact_removed', 'exact'),
                                         ('near_removed', 'near')):
                    a = _dp_int(vc.get(src_key))
                    b = _dp_int(vcounts.get(dst_key))
                    if a is not None and b is not None and a != b:
                        rec['problems'].append(
                            f'{label}: validation_cleanup.{src_key}={a} '
                            f'disagrees with dedup.splits.validation.{dst_key}={b}')
            if protocol is not None:
                train_rows.append(rec)

        if rec['problems']:
            for msg in rec['problems']:
                _fail(msg)
        if rec['warnings']:
            for msg in rec['warnings']:
                _warn(msg)

    # ---- cross-split protocol consistency --------------------------------
    def _pair_key(rec):
        fit = rec.get('fit_fp')
        # Malformed metadata was converted to ``None`` above; keep this
        # helper total so an unexpected caller cannot crash the grouping.
        if fit is not None and not isinstance(fit, str):
            fit = repr(fit)
        return (rec.get('vocab'), fit)
    by_pair = {}
    for rec in train_rows:
        by_pair.setdefault(_pair_key(rec), set()).add(rec.get('protocol_fp'))
    for key, fps in sorted(by_pair.items(), key=lambda kv: str(kv[0])):
        if len(fps) > 1:
            _fail(
                'train/validation caches sharing vocab/fit_fp '
                f'{key!r} declare different dedup protocols '
                f'{sorted(str(x)[:16] for x in fps)}; one corpus cannot have '
                'two cleaning geometries')
    for rec in test_rows:
        key = _pair_key(rec)
        train_fps = by_pair.get(key)
        if not train_fps:
            rec['warnings'].append(
                f"{rec['tag']}: no train cache with the same vocab/fit_fp "
                'declares a dedup protocol; cross-split geometry is unverified')
            _warn(rec['warnings'][-1])
        elif rec.get('protocol_fp') not in train_fps:
            _fail(
                f"{rec['tag']}: test dedup protocol "
                f"{str(rec.get('protocol_fp'))[:16]}… does not match the paired "
                f"train protocol(s) {sorted(str(x)[:16] for x in train_fps)}; "
                'the held-out split was cleaned differently from the corpus it '
                'is paired with')
    if n_protocol == 0:
        if n_legacy:
            # Legacy means "cannot certify the geometry", not "overwrite a
            # hard failure with not_verifiable".  A legacy cache that records
            # a non-default threshold is a fail-closed problem above; keep
            # that fail so a reader cannot mistake a corrupt/unknown cache
            # for an honestly-unverifiable one.
            if out['status'] != 'fail':
                out['status'] = 'not_verifiable_locally'
            out['note'] = (
                'all caches are legacy (no full dedup protocol); the cleaning '
                'geometry cannot be certified.  Rebuild the caches with the '
                'current loader to make the data identity auditable.')
        elif out['status'] == 'ok':
            out['status'] = 'not_verifiable_locally'
            out['note'] = 'no cache records a dedup protocol'
    elif n_legacy:
        out['warnings'].append(
            f'{n_legacy} legacy cache(s) have no full dedup protocol; their '
            'cleaning geometry is unverified')
        if out['status'] == 'ok':
            out['status'] = 'warnings'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
