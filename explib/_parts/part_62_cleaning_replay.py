# Replay the corpus-cleaning decisions from raw rows and verify the cache.
#
# The cache metadata records a cleaning *protocol* and *counts* (``source rows ->
# served rows + exact/near removals``).  Neither proves that the documents named
# by the served sidecars are the ones that protocol actually keeps: a cache
# built with a different index order, a different fit prefix or a subtly
# different dedup geometry can carry self-consistent counts yet a different
# dataset.
#
# This module makes the cleaning step reproducible instead of merely
# self-consistent.  Given the raw WikiText rows, the recorded protocol, the
# recorded tokenizer-fit row count and the recorded scanned-row counts, it
# replays the loader's exact decision sequence (fit-prefix seeding, then the
# per-row exact/SimHash check) and compares the resulting kept document ids with
# the cache sidecars.  Test cleanup is replayed through the production
# ``_filter_test_documents`` helper, so the train/validation/self attributions
# are the same objects the builder used.
#
# What this proves: the served document membership is the deterministic output
# of the declared cleaning geometry on the supplied raw rows.  What it does not
# prove: that the raw rows are the upstream split (the cache's ``dataset_source``
# and row-count preflight bind that separately) or that the tokenizer produced a
# particular token stream (``src_fp``/``fit_fp`` cover that).  It also does not
# read ``wiki.test.raw`` unless the caller explicitly opts in, because that is a
# held-out access.
#
# No token ids, split membership, PPL, seed or frozen decision is changed here.
CLEANING_REPLAY_SCHEMA = 'wikitext_cleaning_replay_v1'
CLEANING_REPLAY_MAX_ROWS = 1_000_000


def _replay_count(value, *, name, problems, minimum=0, allow_none=False):
    """Exact non-negative count from cache metadata, else ``None``.

    Booleans, floats and numeric strings are refused: a malformed count must
    not be silently truncated into a different replay length.
    """
    if value is None:
        if allow_none:
            return None
        problems.append(f'{name} is missing; the replay length is unknown')
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        problems.append(f'{name}={value!r} is not an exact integer count')
        return None
    value = int(value)
    if value < int(minimum):
        problems.append(f'{name}={value} is below {int(minimum)}')
        return None
    return value


def _replay_nonempty(rows, *, name):
    """Return ``(non-empty rows in source order, total raw row count)``.

    The total includes blank rows so it can be compared with the cache's
    ``source_rows``; the filtered list is what the dedup loop consumes.  The
    loaders call ``r.strip()`` on every row, so a non-string must fail here for
    the same reason it would fail there rather than being skipped silently and
    shifting every subsequent index.
    """
    if isinstance(rows, (str, bytes)):
        raise TypeError(
            f'{name} must be an iterable of rows, not a single string')
    try:
        seq = list(rows)
    except TypeError:
        raise TypeError(
            f'{name} must be an iterable of strings, got {type(rows).__name__}') from None
    out = []
    for i, row in enumerate(seq):
        if not isinstance(row, str):
            raise TypeError(
                f'{name}[{i}] must be a string, got {type(row).__name__}')
        if row.strip():
            out.append(row)
    return out, int(len(seq))


def _replay_check_rows(rows, *, name):
    """Type-check every raw row without dropping blanks (order must be exact)."""
    if isinstance(rows, (str, bytes)):
        raise TypeError(
            f'{name} must be an iterable of rows, not a single string')
    for i, row in enumerate(rows):
        if not isinstance(row, str):
            raise TypeError(
                f'{name}[{i}] must be a string, got {type(row).__name__}')
    return rows


def _replay_id_array(values, *, name, problems):
    """Validate a persisted uint64 document-id vector without silent casting."""
    try:
        arr = np.asarray(values)
    except Exception as exc:
        problems.append(f'{name} is unreadable ({type(exc).__name__}: {exc})')
        return None
    flat = arr.reshape(-1)
    if flat.size == 0:
        return np.empty(0, dtype=np.uint64)
    if flat.dtype.kind not in 'iu':
        problems.append(f'{name} has dtype {flat.dtype}; expected an integer document-id array')
        return None
    if flat.dtype.kind == 'i' and bool((flat < 0).any()):
        problems.append(f'{name} contains negative document ids')
        return None
    if flat.dtype.kind == 'u' and flat.dtype.itemsize > 8:
        problems.append(f'{name} has dtype {flat.dtype}, wider than uint64')
        return None
    return flat.astype(np.uint64, copy=False)


def _replay_protocol(meta, *, name, problems):
    protocol = dedup_protocol_from_meta(meta)
    if protocol is None:
        problems.append(
            f'{name} records no full dedup protocol (legacy cache); the '
            'cleaning geometry cannot be replayed')
        return None
    bad = dedup_protocol_problems(protocol)
    if bad:
        problems.append(f'{name} dedup protocol is invalid: ' + '; '.join(bad))
        return None
    return protocol


def _replay_sampling_block(meta, *, split, problems):
    sampling = meta.get('sampling') if isinstance(meta, dict) else None
    block = sampling.get(split) if isinstance(sampling, dict) else None
    if not isinstance(block, dict):
        problems.append(f'{split}: sampling.{split} block is missing')
        return None
    return block


def _compare_kept_ids(replayed, expected, *, split, allow_trailing=False,
                      problems=None, warnings=None):
    """Compare replayed kept ids with the persisted sidecar.

    ``allow_trailing`` handles the loader's chunk/tokenisation boundary: it
    deduplicates a whole 10k-row chunk before stopping at the token cap, so
    kept rows after the last tokenised row are in (and quarantine) the index
    but never enter ``*_doc_ids``.  Validation additionally tokenises one extra
    sequence and trims the served sidecar.  The expected sidecar must still be
    an exact prefix of the replayed kept sequence.
    """
    problems = [] if problems is None else problems
    warnings = [] if warnings is None else warnings
    _problems_before = len(problems)
    rep = np.asarray(replayed, dtype=np.uint64).reshape(-1)
    exp = np.asarray(expected, dtype=np.uint64).reshape(-1)
    detail = {'replayed': int(rep.size), 'expected': int(exp.size)}
    if rep.size < exp.size:
        problems.append(
            f'{split}: replay kept {rep.size} documents but the cache sidecar '
            f'lists {exp.size}; the cache contains documents this cleaning '
            'never serves')
    elif rep.size > exp.size:
        if allow_trailing:
            _why = ('the loader tokenises one extra validation sequence then '
                    'trims the served sidecar'
                    if split == 'validation' else
                    'the loader deduplicates a whole row chunk and then stops '
                    'at the token cap')
            warnings.append(
                f'{split}: {rep.size - exp.size} replayed document(s) fall after '
                f'the served prefix ({_why}); they are not compared')
        else:
            problems.append(
                f'{split}: replay kept {rep.size} documents but the cache sidecar '
                f'lists {exp.size}; the cache is missing served documents')
    n = min(int(rep.size), int(exp.size))
    if n:
        diff = np.flatnonzero(rep[:n] != exp[:n])
        if diff.size:
            i = int(diff[0])
            detail['first_mismatch'] = {
                'position': i,
                'replayed': int(rep[i]),
                'cached': int(exp[i]),
            }
            problems.append(
                f'{split}: kept document id at position {i} is '
                f'{int(rep[i])} from the replay but {int(exp[i])} in the cache; '
                'the served membership is not the output of this cleaning '
                'protocol on these rows')
    detail['match'] = len(problems) == _problems_before
    return detail


def _compare_count(label, got, expected, *, problems, warnings):
    if expected is None:
        warnings.append(f'{label}: no recorded count; membership was replayed but the count is unverified')
        return None
    if int(got) != int(expected):
        problems.append(f'{label}: replay found {int(got)}, cache records {int(expected)}')
        return False
    return True


def _meta_protocol_counts(meta, *, split, problems):
    """Recorded exact/near removal counts for a train/validation split.

    A *missing* count is a warning downstream (the membership replay still
    stands), but a malformed one is appended to ``problems``: a value that is
    not an exact integer is not a count and must not be compared as if it
    were.
    """
    sampling = meta.get('sampling') if isinstance(meta, dict) else None
    block = sampling.get(split) if isinstance(sampling, dict) else None
    skipped = block.get('duplicate_documents_skipped') if isinstance(block, dict) else None
    if isinstance(skipped, dict):
        exact = _replay_count(skipped.get('exact'), name=f'sampling.{split}.duplicate_documents_skipped.exact',
                              problems=problems, allow_none=True)
        near = _replay_count(skipped.get('near'), name=f'sampling.{split}.duplicate_documents_skipped.near',
                             problems=problems, allow_none=True)
        if exact is not None or near is not None:
            return exact, near
    dedup = meta.get('dedup') if isinstance(meta, dict) else None
    splits = dedup.get('splits') if isinstance(dedup, dict) else None
    counts = splits.get(split) if isinstance(splits, dict) else None
    if isinstance(counts, dict):
        return (_replay_count(counts.get('exact'), name=f'dedup.splits.{split}.exact',
                              problems=problems, allow_none=True),
                _replay_count(counts.get('near'), name=f'dedup.splits.{split}.near',
                              problems=problems, allow_none=True))
    return None, None


def replay_train_validation_cleanup(train_rows, validation_rows, *, train_meta,
                                    train_doc_ids, validation_doc_ids,
                                    max_rows=CLEANING_REPLAY_MAX_ROWS):
    """Replay the train/validation dedup sequence and verify kept membership.

    The order mirrors ``load_wikitext.stream_tokenise`` exactly:

    1. seed the index with the first ``tokenizer_fit_rows`` non-empty train
       rows (including any duplicates among them);
    2. process exactly ``sampling.train.nonempty_rows_scanned`` non-empty
       train rows; the first ``tokenizer_fit_rows`` are force-kept without a
       duplicate check, later rows go through the mutating deduper;
    3. continue the same index over exactly
       ``sampling.validation.nonempty_rows_scanned`` non-empty validation
       rows.

    The replayed kept ids must have ``train_doc_ids`` and ``validation_doc_ids``
    as exact served prefixes; additional trailing kept rows are the loader's
    chunk/token-cap boundary and are reported, not compared.
    """
    problems = []
    warnings = []
    out = {
        'schema': CLEANING_REPLAY_SCHEMA,
        'status': 'ok',
        'protocol_fp': None,
        'fit_rows': None,
        'train': {},
        'validation': {},
        'problems': problems,
        'warnings': warnings,
    }
    if not isinstance(train_meta, dict):
        problems.append('train metadata is not an object')
        out['status'] = 'not_verifiable'
        return out
    protocol = _replay_protocol(train_meta, name='train/validation cache', problems=problems)
    if protocol is None:
        out['status'] = 'not_verifiable'
        return out
    out['protocol_fp'] = dedup_protocol_fp(protocol)
    fit_rows = _replay_count(train_meta.get('tokenizer_fit_rows'),
                             name='tokenizer_fit_rows', problems=problems)
    train_block = _replay_sampling_block(train_meta, split='train', problems=problems)
    val_block = _replay_sampling_block(train_meta, split='validation', problems=problems)
    if fit_rows is None or train_block is None or val_block is None:
        out['status'] = 'not_verifiable'
        return out
    train_scanned = _replay_count(train_block.get('nonempty_rows_scanned'),
                                  name='sampling.train.nonempty_rows_scanned',
                                  problems=problems, minimum=1)
    val_scanned = _replay_count(val_block.get('nonempty_rows_scanned'),
                                name='sampling.validation.nonempty_rows_scanned',
                                problems=problems, minimum=0)
    if train_scanned is None or val_scanned is None:
        out['status'] = 'not_verifiable'
        return out
    out['fit_rows'] = int(fit_rows)
    if max_rows and int(train_scanned) > int(max_rows):
        problems.append(
            f'train replay would hash {train_scanned} rows, above the '
            f'max_rows={int(max_rows)} guard; rerun explicitly with a larger '
            'max_rows to certify this cache')
        out['status'] = 'not_verifiable'
        return out
    try:
        train_ne, train_total = _replay_nonempty(train_rows, name='train_rows')
        val_ne, val_total = _replay_nonempty(validation_rows, name='validation_rows')
    except TypeError as exc:
        problems.append(str(exc))
        out['status'] = 'fail'
        return out
    # Row-count preflight binds the replay source to the split the cache
    # declares; a wrong dump must not be reported as a cleaning mismatch.
    for _split, _rows, _total, _block in (
            ('train', train_ne, train_total, train_block),
            ('validation', val_ne, val_total, val_block)):
        expected_rows = _replay_count(_block.get('source_rows'),
                                      name=f'sampling.{_split}.source_rows',
                                      problems=problems, allow_none=True)
        if expected_rows is None:
            warnings.append(
                f'{_split}: sampling.{_split}.source_rows is missing; the raw '
                'dump is not bound to the cache\'s declared source row count')
        elif expected_rows != _total:
            problems.append(
                f'{_split}: raw source has {_total} rows but the cache records '
                f'source_rows={expected_rows}; refusing to replay a different dump')
    if any('refusing to replay a different dump' in p for p in problems):
        out['status'] = 'fail'
        return out
    if len(train_ne) < int(train_scanned):
        problems.append(
            f'train: cache scanned {train_scanned} non-empty rows but the raw '
            f'source has only {len(train_ne)}')
        out['status'] = 'fail'
        return out
    if len(val_ne) < int(val_scanned):
        problems.append(
            f'validation: cache scanned {val_scanned} non-empty rows but the '
            f'raw source has only {len(val_ne)}')
        out['status'] = 'fail'
        return out
    try:
        dedup = _DocDeduplicator(**dedup_kwargs_from_protocol(protocol))
    except (TypeError, ValueError) as exc:
        problems.append(f'cannot construct the recorded deduplicator: {exc}')
        out['status'] = 'fail'
        return out
    # 1) fit-prefix seeding.  These calls mutate the index (and a duplicate
    # among fit rows is counted there), but the train loop force-keeps them
    # below, exactly like the loader.
    for row in train_ne[:int(fit_rows)]:
        dedup.is_duplicate(row)
    seeded = dedup.stats()
    seed_exact = int(seeded['exact_duplicates_skipped'])
    seed_near = int(seeded['near_duplicates_skipped'])
    # 2) train stream.
    train_kept = []
    for i, row in enumerate(train_ne[:int(train_scanned)]):
        if i < int(fit_rows):
            train_kept.append(row)
            continue
        dup, _kind = dedup.is_duplicate(row)
        if not dup:
            train_kept.append(row)
    train_stats = dedup.stats()
    train_removed = {
        'exact': int(train_stats['exact_duplicates_skipped']) - seed_exact,
        'near': int(train_stats['near_duplicates_skipped']) - seed_near,
    }
    train_kept_ids = np.asarray(
        [int(_doc_id_u64(_doc_exact_fp(row))) for row in train_kept],
        dtype=np.uint64)
    expected_train = _replay_id_array(train_doc_ids, name='train_doc_ids', problems=problems)
    exp_counts = _meta_protocol_counts(train_meta, split='train', problems=problems)
    train_detail = {
        'scanned_nonempty_rows': int(train_scanned),
        'fit_rows_force_kept': int(min(int(fit_rows), int(train_scanned))),
        'removed': train_removed,
        'replayed_kept_fp': _ids_fp(train_kept_ids),
    }
    if expected_train is not None:
        train_detail['expected_kept_fp'] = _ids_fp(expected_train)
        train_detail.update(_compare_kept_ids(
            train_kept_ids, expected_train, split='train',
            allow_trailing=True, problems=problems, warnings=warnings))
    _compare_count('train exact removals', train_removed['exact'], exp_counts[0],
                   problems=problems, warnings=warnings)
    _compare_count('train near removals', train_removed['near'], exp_counts[1],
                   problems=problems, warnings=warnings)
    out['train'] = train_detail
    # 3) validation stream on the SAME index, like load_wikitext.
    val_kept = []
    for row in val_ne[:int(val_scanned)]:
        dup, _kind = dedup.is_duplicate(row)
        if not dup:
            val_kept.append(row)
    val_stats = dedup.stats()
    val_removed = {
        'exact': int(val_stats['exact_duplicates_skipped']) - int(train_stats['exact_duplicates_skipped']),
        'near': int(val_stats['near_duplicates_skipped']) - int(train_stats['near_duplicates_skipped']),
    }
    val_kept_ids = np.asarray(
        [int(_doc_id_u64(_doc_exact_fp(row))) for row in val_kept],
        dtype=np.uint64)
    expected_val = _replay_id_array(validation_doc_ids, name='validation_doc_ids', problems=problems)
    val_counts = _meta_protocol_counts(train_meta, split='validation', problems=problems)
    val_detail = {
        'scanned_nonempty_rows': int(val_scanned),
        'removed': val_removed,
        'replayed_kept_fp': _ids_fp(val_kept_ids),
    }
    if expected_val is not None:
        val_detail['expected_kept_fp'] = _ids_fp(expected_val)
        val_detail.update(_compare_kept_ids(
            val_kept_ids, expected_val, split='validation', allow_trailing=True,
            problems=problems, warnings=warnings))
    _compare_count('validation exact removals', val_removed['exact'], val_counts[0],
                   problems=problems, warnings=warnings)
    _compare_count('validation near removals', val_removed['near'], val_counts[1],
                   problems=problems, warnings=warnings)
    out['validation'] = val_detail
    out['status'] = 'fail' if problems else 'ok'
    return out


def replay_heldout_cleanup(train_rows, validation_rows, test_rows, *, test_meta,
                           test_doc_ids, max_rows=CLEANING_REPLAY_MAX_ROWS):
    """Replay the held-out test cleanup and verify its served prefix.

    This calls the production ``_filter_test_documents`` helper with the same
    train/validation/test row order as ``load_wikitest``.  The first
    ``len(test_doc_ids)`` replayed kept documents must equal the cached test
    document ids; documents after that point are beyond the token cap and are
    not part of the served slice.  The recorded train/validation/self removal
    counts and the validation-cleanup ledger are checked too.
    """
    problems = []
    warnings = []
    out = {
        'schema': CLEANING_REPLAY_SCHEMA,
        'status': 'ok',
        'protocol_fp': None,
        'test': {},
        'problems': problems,
        'warnings': warnings,
    }
    if not isinstance(test_meta, dict):
        problems.append('test metadata is not an object')
        out['status'] = 'not_verifiable'
        return out
    protocol = _replay_protocol(test_meta, name='test cache', problems=problems)
    if protocol is None:
        out['status'] = 'not_verifiable'
        return out
    out['protocol_fp'] = dedup_protocol_fp(protocol)
    cleanup = test_meta.get('near_duplicate_cleanup')
    if not isinstance(cleanup, dict):
        problems.append('test metadata has no near_duplicate_cleanup ledger')
        out['status'] = 'not_verifiable'
        return out
    sampling = _replay_sampling_block(test_meta, split='test', problems=problems)
    if sampling is None:
        out['status'] = 'not_verifiable'
        return out
    try:
        train_ne, _train_total = _replay_nonempty(train_rows, name='train_rows')
        val_ne, _val_total = _replay_nonempty(validation_rows, name='validation_rows')
        test_seq = _replay_check_rows(list(test_rows), name='test_rows')
    except TypeError as exc:
        problems.append(str(exc))
        out['status'] = 'fail'
        return out
    if max_rows and len(train_ne) > int(max_rows):
        problems.append(
            f'test replay needs the full raw train split ({len(train_ne)} rows), '
            f'above the max_rows={int(max_rows)} guard; rerun explicitly with a '
            'larger max_rows to certify this cache')
        out['status'] = 'not_verifiable'
        return out
    expected_rows = _replay_count(sampling.get('source_rows'),
                                  name='sampling.test.source_rows',
                                  problems=problems, allow_none=True)
    if expected_rows is None:
        warnings.append('test: sampling.test.source_rows is missing; the raw '
                        'dump is not bound to the declared source row count')
    elif expected_rows != len(test_seq):
        problems.append(
            f'test: raw source has {len(test_seq)} rows but the cache records '
            f'source_rows={expected_rows}; refusing to replay a different dump')
    if problems:
        out['status'] = 'fail'
        return out
    # Build the two reference indices exactly as load_wikitest does: a
    # train-seeded screen for the validation ledger, and a self-deduplicated
    # validation index for the test screen itself.
    kwargs = dedup_kwargs_from_protocol(protocol)
    try:
        # The production helper rebuilds its own read-only train index; the
        # extra one here is only the train-seeded screen used to check the
        # validation-cleanup ledger.
        val_screen = _DocDeduplicator(**kwargs)
        for row in train_ne:
            val_screen.is_duplicate(row)
        val_kept_vs_train = []
        for row in val_ne:
            dup, _kind = val_screen.is_duplicate(row)
            if not dup:
                val_kept_vs_train.append(row)
        val_only = _DocDeduplicator(**kwargs)
        val_indexed = 0
        for row in val_ne:
            dup, _kind = val_only.is_duplicate(row)
            if not dup:
                val_indexed += 1
        kept, excluded, _test_index = _filter_test_documents(
            train_ne, val_only, list(test_seq), label='test',
            protocol=protocol, verbose=False)
    except (TypeError, ValueError) as exc:
        problems.append(f'cannot replay the test cleanup: {exc}')
        out['status'] = 'fail'
        return out
    kept_ids = np.asarray(
        [int(_doc_id_u64(_doc_exact_fp(row))) for row in kept],
        dtype=np.uint64)
    expected_ids = _replay_id_array(test_doc_ids, name='test_doc_ids', problems=problems)
    detail = {
        'source_rows': int(len(test_seq)),
        'source_nonempty_rows': int(sum(1 for r in test_seq
                                        if isinstance(r, str) and r.strip())),
        'kept_documents': int(len(kept)),
        'removed': {k: int(v) for k, v in excluded.items()},
        'validation_indexed_documents': int(val_indexed),
        'validation_kept_vs_train': int(len(val_kept_vs_train)),
        'replayed_kept_fp': _ids_fp(kept_ids),
    }
    if expected_ids is not None:
        detail['expected_kept_fp'] = _ids_fp(expected_ids)
        detail.update(_compare_kept_ids(
            kept_ids, expected_ids, split='test', allow_trailing=True,
            problems=problems, warnings=warnings))
    count_checks = (
        ('source_documents', detail['source_rows']),
        ('source_documents_nonempty', detail['source_nonempty_rows']),
        ('kept_documents', detail['kept_documents']),
        ('train_exact_removed', detail['removed']['train_exact']),
        ('train_near_removed', detail['removed']['train_near']),
        ('test_exact_removed', detail['removed']['test_exact']),
        ('test_near_removed', detail['removed']['test_near']),
    )
    for key, got in count_checks:
        expected = _replay_count(cleanup.get(key), name=f'test.{key}',
                                 problems=problems, allow_none=True)
        _compare_count(f'test {key}', got, expected, problems=problems,
                       warnings=warnings)
    # rows_after_dedup / rows_served are ledger forecasts of the replayed
    # kept/served counts; check them when present.
    expected_after = _replay_count(sampling.get('rows_after_dedup'),
                                   name='sampling.test.rows_after_dedup',
                                   problems=problems, allow_none=True)
    _compare_count('test rows_after_dedup', detail['kept_documents'],
                   expected_after, problems=problems, warnings=warnings)
    expected_served = _replay_count(sampling.get('rows_served'),
                                    name='sampling.test.rows_served',
                                    problems=problems, allow_none=True)
    if expected_ids is not None:
        _compare_count('test rows_served', int(expected_ids.size),
                       expected_served, problems=problems, warnings=warnings)
    vc = cleanup.get('validation_cleanup')
    if isinstance(vc, dict):
        for key, got in (
                ('source_documents', int(len(val_ne))),
                ('kept_documents', int(len(val_kept_vs_train))),
                ('indexed_documents', int(val_indexed)),
                ('duplicate_documents_removed',
                 int(len(val_ne) - len(val_kept_vs_train))),
                ('exact_removed_from_test', int(excluded['validation_exact'])),
                ('near_removed_from_test', int(excluded['validation_near']))):
            expected = _replay_count(vc.get(key), name=f'test.validation_cleanup.{key}',
                                     problems=problems, allow_none=True)
            _compare_count(f'test validation_cleanup.{key}', got, expected,
                           problems=problems, warnings=warnings)
    else:
        warnings.append('test validation_cleanup block is missing; the '
                        'held-out cleanup cannot be checked against the '
                        'selection split')
    out['test'] = detail
    out['status'] = 'fail' if problems else 'ok'
    return out


# ---------------------------------------------------------------------------
# File/cache wrapper
# ---------------------------------------------------------------------------

def _replay_cache_dir(cache_dir=None):
    return str(cache_dir) if cache_dir not in (None, '') else (
        os.environ.get('WT103_CACHE_DIR') or './wt103_cache')


def _replay_raw_dir(raw_dir=None, cache_dir=None):
    """Resolve a directory holding wiki.train.raw / wiki.valid.raw / wiki.test.raw."""
    if raw_dir not in (None, ''):
        # An explicit source is authoritative: silently falling back to another
        # dump would replay a different corpus than the caller named.
        cand = str(raw_dir)
        if (os.path.exists(os.path.join(cand, 'wiki.train.raw'))
                and os.path.exists(os.path.join(cand, 'wiki.valid.raw'))):
            return cand
        return None
    candidates = []
    env = os.environ.get('WT103_RAW_TXT')
    if env:
        candidates.append(env)
    if cache_dir not in (None, ''):
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(cache_dir)),
                                       'wt103_raw'))
    candidates.append(os.path.join(_REPO, 'wt103_raw'))
    candidates.append('/root/wt103_raw')
    for cand in candidates:
        if (os.path.exists(os.path.join(cand, 'wiki.train.raw'))
                and os.path.exists(os.path.join(cand, 'wiki.valid.raw'))):
            return cand
    return None


def _replay_load_json(path):
    try:
        with open(path, encoding='utf-8') as fh:
            obj = json.load(fh)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _replay_meta_seq_len(meta):
    value = meta.get('seq_len') if isinstance(meta, dict) else None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        return None
    return int(value)


def _replay_meta_tag(path):
    name = os.path.basename(path)
    if name.startswith('test_meta_') and name.endswith('.json'):
        return name[len('test_meta_'):-len('.json')]
    if name.startswith('meta_') and name.endswith('.json'):
        return name[len('meta_'):-len('.json')]
    return None


def _replay_select_train_meta(cache_dir, *, tag=None, train_seq_len=None,
                              problems=None, warnings=None):
    problems = [] if problems is None else problems
    warnings = [] if warnings is None else warnings
    if tag:
        path = os.path.join(cache_dir, f'meta_{tag}.json')
        meta = _replay_load_json(path)
        if meta is None:
            problems.append(f'cannot read train metadata {path}')
            return None, None, None
        return path, meta, _replay_meta_tag(path)
    import glob as _glob
    candidates = []
    for path in sorted(_glob.glob(os.path.join(cache_dir, 'meta_*.json'))):
        if os.path.basename(path).startswith('test_meta_'):
            continue
        meta = _replay_load_json(path)
        if meta is None:
            warnings.append(f'{os.path.basename(path)} is not a readable metadata object; ignored')
            continue
        candidates.append((path, meta, _replay_meta_tag(path)))
    if not candidates:
        problems.append(f'{cache_dir} contains no readable train metadata')
        return None, None, None
    if train_seq_len is not None:
        want = int(train_seq_len)
        filtered = [c for c in candidates
                    if _replay_meta_seq_len(c[1]) == want]
        if not filtered:
            problems.append(f'no train metadata declares seq_len={want}; refusing to guess')
            return None, None, None
        candidates = filtered
    if len(candidates) > 1:
        problems.append(
            'multiple train metadata files match '
            f'{[c[0] for c in candidates]}; pass tag= or train_seq_len= to name one')
        return None, None, None
    return candidates[0]


def _replay_select_test_meta(cache_dir, train_meta, *, problems, warnings):
    """Pair the test metadata with the train cache by (vocab, fit_fp)."""
    import glob as _glob
    pairs = []
    for path in sorted(_glob.glob(os.path.join(cache_dir, 'test_meta_*.json'))):
        meta = _replay_load_json(path)
        if meta is None:
            warnings.append(f'{os.path.basename(path)} is not a readable test metadata object; ignored')
            continue
        if meta.get('vocab') != train_meta.get('vocab'):
            continue
        if train_meta.get('fit_fp') and meta.get('fit_fp'):
            if meta['fit_fp'] != train_meta['fit_fp']:
                continue
        pairs.append((path, meta, _replay_meta_tag(path)))
    if not pairs:
        problems.append('no test metadata shares the train cache vocab/fit_fp; cannot pair the held-out split')
        return None, None, None
    if len(pairs) > 1:
        problems.append(
            'multiple test metadata files match the train cache '
            f'{[p[0] for p in pairs]}; refusing to guess')
        return None, None, None
    return pairs[0]


def replay_wikitext_cleaning(cache_dir=None, raw_dir=None, *,
                             train_seq_len=None, include_test=False, tag=None,
                             max_rows=CLEANING_REPLAY_MAX_ROWS):
    """Replay a local WikiText cache's cleaning decisions from the raw dump.

    Returns a JSON-ready report.  ``not_verifiable`` means the inputs needed
    for a replay are absent (raw dump, full protocol, fit-row count, sidecar);
    it is never reported as ``ok``.  ``include_test`` reads ``wiki.test.raw``
    and is therefore an explicit held-out access; the default never opens it.
    """
    problems = []
    warnings = []
    cache_dir = _replay_cache_dir(cache_dir)
    out = {
        'schema': CLEANING_REPLAY_SCHEMA,
        'status': 'not_verifiable',
        'cache_dir': os.path.abspath(cache_dir) if os.path.isdir(cache_dir) else cache_dir,
        'raw_dir': None,
        'train_tag': None,
        'protocol_fp': None,
        'train': {},
        'validation': {},
        'test': None,
        'include_test': bool(include_test),
        'test_raw_read': False,
        'problems': problems,
        'warnings': warnings,
        'note': ('Replays the dedup decision sequence from raw rows and verifies '
                 'the kept document ids; counts alone are not cleaning identity.'),
    }
    if not os.path.isdir(cache_dir):
        problems.append(f'cache directory {cache_dir!r} does not exist')
        return out
    resolved_raw = _replay_raw_dir(raw_dir, cache_dir)
    if resolved_raw is None:
        if raw_dir not in (None, ''):
            problems.append(
                f'the explicit raw_dir {raw_dir!r} does not contain both '
                'wiki.train.raw and wiki.valid.raw; the cleaning decisions '
                'cannot be replayed against that dump')
        else:
            problems.append(
                'no raw WikiText dump with wiki.train.raw and wiki.valid.raw was '
                'found (set WT103_RAW_TXT or pass raw_dir); the cleaning '
                'decisions cannot be replayed')
        return out
    out['raw_dir'] = os.path.abspath(resolved_raw)
    path, meta, tag = _replay_select_train_meta(
        cache_dir, tag=tag, train_seq_len=train_seq_len,
        problems=problems, warnings=warnings)
    if meta is None:
        return out
    out['train_tag'] = tag
    train_doc_path = os.path.join(cache_dir, f'train_doc_ids_{tag}.npy')
    val_doc_path = os.path.join(cache_dir, f'val_doc_ids_{tag}.npy')
    for p, label in ((train_doc_path, 'train_doc_ids'), (val_doc_path, 'val_doc_ids')):
        if not os.path.exists(p):
            problems.append(f'{label} sidecar {os.path.basename(p)} is missing; '
                            'the served membership cannot be replayed')
    if problems:
        return out
    # Apply the row guard before materialising a multi-million-row raw dump:
    # the replay itself would reject it, but reading the file first would waste
    # the time/memory the guard is meant to save.
    _scan_problems = []
    _sampling_probe = (meta.get('sampling')
                       if isinstance(meta.get('sampling'), dict) else {})
    _train_probe = (_sampling_probe.get('train')
                    if isinstance(_sampling_probe.get('train'), dict) else {})
    _train_scanned_probe = _replay_count(
        _train_probe.get('nonempty_rows_scanned'),
        name='sampling.train.nonempty_rows_scanned',
        problems=_scan_problems, allow_none=True)
    if (max_rows and _train_scanned_probe is not None
            and int(_train_scanned_probe) > int(max_rows)):
        problems.append(
            f'train replay would hash {_train_scanned_probe} rows, above the '
            f'max_rows={int(max_rows)} guard; set CSA_REPLAY_MAX_ROWS to a '
            'larger value (0 disables the guard) to certify this cache')
        return out
    try:
        train_rows = _read_raw_rows(os.path.join(resolved_raw, 'wiki.train.raw'))
        val_rows = _read_raw_rows(os.path.join(resolved_raw, 'wiki.valid.raw'))
        train_doc_ids = np.load(train_doc_path)
        val_doc_ids = np.load(val_doc_path)
    except Exception as exc:
        problems.append(f'cannot read replay inputs ({type(exc).__name__}: {exc})')
        return out
    base = replay_train_validation_cleanup(
        train_rows, val_rows, train_meta=meta,
        train_doc_ids=train_doc_ids, validation_doc_ids=val_doc_ids,
        max_rows=max_rows)
    out['train'] = base.get('train', {})
    out['validation'] = base.get('validation', {})
    out['protocol_fp'] = base.get('protocol_fp')
    problems.extend(base.get('problems') or [])
    warnings.extend(base.get('warnings') or [])
    out['status'] = base.get('status', 'not_verifiable')
    if include_test and out['status'] != 'fail':
        test_path, test_meta, test_tag = _replay_select_test_meta(
            cache_dir, meta, problems=problems, warnings=warnings)
        test_doc_path = (os.path.join(cache_dir, f'test_doc_ids_{test_tag}.npy')
                         if test_tag else None)
        if test_meta is None or test_doc_path is None or not os.path.exists(test_doc_path):
            if test_meta is not None and test_doc_path is not None:
                problems.append(f'test_doc_ids sidecar {os.path.basename(test_doc_path)} is missing')
            out['status'] = 'fail' if any(
                'cannot pair' in p for p in problems) else 'not_verifiable'
        else:
            test_raw = os.path.join(resolved_raw, 'wiki.test.raw')
            if not os.path.exists(test_raw):
                problems.append(
                    'include_test=True but wiki.test.raw is absent; the held-out '
                    'cleanup cannot be replayed')
                out['status'] = 'not_verifiable'
            else:
                try:
                    test_rows = _read_raw_rows(test_raw)
                    test_doc_ids = np.load(test_doc_path)
                except Exception as exc:
                    problems.append(f'cannot read test replay inputs ({type(exc).__name__}: {exc})')
                    out['status'] = 'fail'
                else:
                    out['test_raw_read'] = True
                    test_rep = replay_heldout_cleanup(
                        train_rows, val_rows, test_rows, test_meta=test_meta,
                        test_doc_ids=test_doc_ids, max_rows=max_rows)
                    out['test'] = test_rep.get('test', {})
                    problems.extend(test_rep.get('problems') or [])
                    warnings.extend(test_rep.get('warnings') or [])
                    if (base.get('protocol_fp')
                            and test_rep.get('protocol_fp')
                            and base['protocol_fp'] != test_rep['protocol_fp']):
                        problems.append(
                            'test cache cleaning protocol '
                            f'{test_rep["protocol_fp"][:16]}… differs from the '
                            f'paired train protocol {base["protocol_fp"][:16]}…; '
                            'the held-out split is not the same cleaning geometry')
                        out['status'] = 'fail'
                    if test_rep.get('status') == 'fail':
                        out['status'] = 'fail'
                    elif test_rep.get('status') == 'not_verifiable' and out['status'] == 'ok':
                        out['status'] = 'not_verifiable'
    if out['status'] == 'ok' and problems:
        out['status'] = 'fail'
    return out
