# Persisted ARTICLE-level sidecar for the WikiText evaluation caches.
#
# The frozen caches already persist a per-token LINE map (one WikiText raw row
# per id).  That unit is not the independent corpus unit: one article spans a
# title line plus its paragraphs, so a line-level bootstrap resamples several
# correlated rows of the same article and reports an interval that is too
# narrow (design notes §53/§54).
#
# The article clustering was previously reconstructed at analysis time by
# decoding the served line runs.  That works, but it is not part of the cache
# identity: a legacy/edited tokenizer, a decode difference or a later code
# change could change the cluster labels without any fingerprint catching it.
# This fragment makes the article unit a persisted, fingerprinted artifact:
#
#   * cache builds write one per-token article-id sidecar next to the line map;
#   * the metadata records its length, digest, article/line counts and the
#     heading rule, so the sidecar is part of the data identity;
#   * the cache-hit path verifies it (length, digest, document->article
#     nesting, article count) before serving it;
#   * `load_cached_article_ids` is the reader for analysis code;
#   * `article_sidecar_problems` is the single validator reused by the audit
#     gate, so a tampered file is reported as `fail`, not silently decoded
#     around.
#
# This is ADDITIVE evidence.  It changes no token id, split member, training
# slice, pooled PPL, seed, alpha/Holm slot, frozen manifest bytes or
# `CODE_SEMANTICS`.  `document_unit` stays 'line'; the article ids are an
# additional clustering of exactly the served tokens.  A cache without the
# sidecar remains readable (legacy) and the analysis keeps its explicit
# line-level fallback.
#
# This fragment intentionally has no module docstring: the split loader
# executes fragments in the package globals, and a string expression there
# would overwrite exp_lib.__doc__.

ARTICLE_SIDECAR_SCHEMA = 'wikitext_article_sidecar_v1'

ARTICLE_SIDECAR_POLICY = {
    'version': 'wikitext-article-sidecar-1',
    'schema': ARTICLE_SIDECAR_SCHEMA,
    'unit': 'article',
    'boundary_rule': 'top_level_heading',
    'id_rule': (
        'article cluster id = u64(blake2b128(group_ordinal, normalized '
        'top-level title)); a headerless preamble is its own cluster'),
    'token_policy': 'one article id per served token, aligned with the line map',
    'role': (
        'persist the independent evaluation unit next to the frozen line map '
        'so article-level uncertainty does not depend on re-decoding the cache '
        'and cannot change silently after the slice is bound'),
    'boundary': (
        'the sidecar is an additive clustering of the SAME served tokens; it '
        'does not replace document ids, change document_unit, or participate '
        'in the frozen line-level PPL/decision contract'),
    'missing_sidecar': (
        'legacy cache: readable, but article-level uncertainty is an explicit '
        'fallback, never silently relabelled as article-level'),
}


def _as_sidecar_int(value, name, *, minimum=0, allow_none=False):
    """Exact non-negative integer from metadata, else ``None``.

    Booleans, floats and numeric strings are refused: a malformed count must
    not be silently truncated into a different sidecar length.
    """
    if value is None:
        if allow_none:
            return None
        raise ValueError(f'{name} is missing; the sidecar length is unknown')
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, numbers.Integral):
        raise ValueError(f'{name}={value!r} is not an exact integer')
    value = int(value)
    if value < int(minimum):
        raise ValueError(f'{name}={value} is below {int(minimum)}')
    return value


def _as_sidecar_ids(values, name, *, expected=None, allow_empty=False):
    """Validate an integer id vector without silent casting."""
    try:
        arr = np.asarray(values)
    except Exception as exc:
        raise ValueError(f'{name} is unreadable ({type(exc).__name__}: {exc})') from None
    flat = arr.reshape(-1)
    if flat.size == 0:
        if allow_empty:
            return np.empty(0, dtype=np.uint64)
        raise ValueError(f'{name} is empty')
    if flat.dtype.kind not in 'iu':
        raise ValueError(
            f'{name} has dtype {flat.dtype}; expected an integer id array')
    if flat.dtype.kind == 'i' and bool((flat < 0).any()):
        raise ValueError(f'{name} contains negative ids')
    if flat.dtype.kind == 'u' and flat.dtype.itemsize > 8:
        raise ValueError(f'{name} has dtype {flat.dtype}, wider than uint64')
    if expected is not None and int(flat.size) != int(expected):
        raise ValueError(
            f'{name} has {int(flat.size)} id(s) but {int(expected)} were '
            'expected')
    return flat.astype(np.uint64, copy=False)


def _article_summary(rep):
    """JSON-safe diagnostics copied from an article builder result."""
    out = {}
    for key in ('n_lines', 'n_articles', 'mean_lines_per_article',
                'max_lines_per_article', 'pseudoreplication_factor',
                'preamble_groups', 'has_headings', 'grouping_valid',
                'last_run_partial', 'valid_for_clustering'):
        if key in rep:
            value = rep[key]
            if isinstance(value, (bool, np.bool_)):
                out[key] = bool(value)
            elif isinstance(value, (int, np.integer)):
                out[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                out[key] = (float(value) if math.isfinite(float(value))
                            else None)
            elif isinstance(value, str) or value is None:
                out[key] = value
    if 'problems' in rep:
        out['problems'] = [str(x) for x in (rep.get('problems') or [])]
    return out


def article_sidecar_block(token_article_ids, *, split, source,
                          extra=None):
    """Metadata block describing a persisted per-token article vector.

    The caller writes the array separately and then adds ``sidecar_file`` /
    ``sidecar_sha256``.  Keeping the block a pure function of the vector makes
    it replayable from the raw rows or from the decoded line map.
    """
    split = str(split)
    if split not in ('validation', 'test'):
        raise ValueError(
            f'article_sidecar_block: split={split!r} must be '
            "'validation' or 'test'")
    arr = _as_sidecar_ids(token_article_ids, 'token_article_ids')
    articles = np.unique(arr)
    block = {
        'schema': ARTICLE_SIDECAR_SCHEMA,
        'policy': ARTICLE_SIDECAR_POLICY['version'],
        'split': split,
        'unit': 'article',
        'boundary_rule': 'top_level_heading',
        'source': str(source),
        'available': True,
        'valid_for_clustering': True,
        'token_article_count': int(arr.size),
        'token_article_fp': _ids_fp(arr),
        'n_articles': int(articles.size),
        'token_policy': ARTICLE_SIDECAR_POLICY['token_policy'],
        'claim': ('additive article clustering of the served tokens; the '
                  'frozen document_unit remains line'),
    }
    if isinstance(extra, dict):
        for key, value in extra.items():
            if key not in block and key != 'problems':
                block[key] = value
    return block


def build_article_sidecar(token_ids, token_doc_ids, tok, *,
                          split='test', source='decoded_line_map'):
    """Build the article vector from an already loaded line map.

    Returns a dict with ``status``, ``available``, ``token_article_ids`` and
    ``block`` (when available).  A decode/grouping failure is returned as an
    explicit unavailable result, never guessed.
    """
    rep = wikitext_token_article_ids_from_line_map(
        token_ids, token_doc_ids, tok)
    return _article_result(rep, split=split, source=source)


def build_article_sidecar_from_rows(rows, lengths, cap_tokens, *,
                                    split='test', source='raw_rows'):
    """Build the article vector from raw rows + per-row token lengths.

    This is the cache-build path: it never decodes the token stream, so the
    grouping is derived from exactly the rows that produced the served ids.
    """
    rep = wikitext_token_article_ids_for_rows(rows, lengths, cap_tokens)
    return _article_result(rep, split=split, source=source)


def _article_result(rep, *, split, source):
    if not isinstance(rep, dict):
        return {'status': 'invalid_builder_result', 'available': False,
                'problems': ['the article builder did not return a mapping'],
                'split': str(split), 'source': str(source)}
    base = {'split': str(split), 'source': str(source),
            'available': False, 'problems': list(rep.get('problems') or []),
            **_article_summary(rep)}
    status = rep.get('status')
    base['status'] = status
    if status != 'ok':
        base.setdefault('reason', 'article grouping did not complete')
        return base
    if rep.get('valid_for_clustering') is not True:
        base['status'] = 'not_identifiable'
        base['reason'] = (
            'the served slice has no evidenced top-level " = Title = " '
            'heading; article boundaries are not identifiable')
        return base
    arr = np.asarray(rep.get('token_article_ids'))
    try:
        arr = _as_sidecar_ids(arr, 'token_article_ids')
    except ValueError as exc:
        base['status'] = 'invalid_article_ids'
        base['problems'].append(str(exc))
        return base
    block = article_sidecar_block(arr, split=split, source=source,
                                  extra=_article_summary(rep))
    base.update({'status': 'ok', 'available': True,
                 'token_article_ids': arr, 'block': block})
    return base


def article_sidecar_problems(meta, token_article_ids, token_doc_ids, *,
                             split, n_tokens=None):
    """Reasons a persisted article sidecar cannot describe this cache.

    An undeclared/legacy sidecar returns ``[]``: it is a readable cache with
    an explicit line-level fallback.  A cache that DECLARES the sidecar but
    whose block/array is missing, malformed or inconsistent is a hard problem.
    """
    problems = []
    if not isinstance(meta, dict):
        return ['cache metadata is not an object']
    declared = meta.get('article_sidecars')
    block = meta.get('article_unit')
    if block is None:
        if declared is True:
            problems.append(
                'cache declares article_sidecars=true but records no '
                'article_unit block')
        elif declared not in (None, False):
            problems.append(
                f'article_sidecars={declared!r} is not a boolean')
        return problems
    if not isinstance(block, dict):
        problems.append('article_unit is not an object')
        return problems
    if declared is not True:
        problems.append(
            'article_unit block is present but article_sidecars is not true; '
            'the article attestation is ambiguous')
    if block.get('schema') != ARTICLE_SIDECAR_SCHEMA:
        problems.append(
            f"article_unit.schema={block.get('schema')!r} is not "
            f'{ARTICLE_SIDECAR_SCHEMA!r}')
    if block.get('unit') != 'article':
        problems.append(
            f"article_unit.unit={block.get('unit')!r} is not 'article'")
    if block.get('split') not in (None, str(split)):
        problems.append(
            f"article_unit.split={block.get('split')!r} does not match "
            f'{split!r}')
    if block.get('valid_for_clustering') is not True:
        # A cache that could not evidence article boundaries records an
        # unavailable block and no sidecar; that is a truthful fallback.
        if block.get('available') in (None, False):
            return problems
        problems.append(
            'article_unit claims an available sidecar but is not '
            'valid_for_clustering')
    if block.get('available') is False:
        # Explicit "no article sidecar" block.  It must not claim a file.
        if block.get('sidecar_file') is not None:
            problems.append(
                'unavailable article_unit block names a sidecar file')
        return problems
    try:
        arr = _as_sidecar_ids(token_article_ids, 'token_article_ids')
    except ValueError as exc:
        problems.append(str(exc))
        return problems
    if n_tokens is not None and int(arr.size) != int(n_tokens):
        problems.append(
            f'token article vector has {int(arr.size)} id(s) but the served '
            f'slice has {int(n_tokens)} token(s)')
    for key in ('token_article_count', 'n_articles'):
        try:
            got = _as_sidecar_int(block.get(key), f'article_unit.{key}',
                                  minimum=1)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        actual = (int(arr.size) if key == 'token_article_count'
                  else int(np.unique(arr).size))
        if got != actual:
            problems.append(
                f'article_unit.{key}={got} does not match the vector '
                f'({actual})')
    recorded_fp = block.get('token_article_fp')
    if not isinstance(recorded_fp, str) or not recorded_fp:
        problems.append('article_unit.token_article_fp is missing')
    elif recorded_fp != _ids_fp(arr):
        problems.append(
            'article_unit.token_article_fp does not match the article vector')
    try:
        docs = _as_sidecar_ids(token_doc_ids, 'token_doc_ids',
                               expected=int(arr.size))
    except ValueError as exc:
        problems.append(str(exc))
        return problems
    # Each served raw line belongs to exactly one article.  The article unit
    # is coarser than the line unit, so one article may own many line ids, but
    # one line may never be split across two articles by a bad sidecar.
    pairs = np.unique(np.stack([docs, arr], axis=1), axis=0)
    n_docs = int(np.unique(docs).size)
    n_articles = int(np.unique(arr).size)
    if int(pairs.shape[0]) != n_docs:
        problems.append(
            'a served line document maps to more than one article; the '
            'article sidecar split a document')
    try:
        declared_lines = _as_sidecar_int(
            block.get('n_lines'), 'article_unit.n_lines', minimum=1)
    except ValueError as exc:
        problems.append(str(exc))
    else:
        if declared_lines != n_docs:
            problems.append(
                f'article_unit.n_lines={declared_lines} does not match the '
                f'{n_docs} distinct line ids in the served slice')
    try:
        declared_articles = _as_sidecar_int(
            block.get('n_articles'), 'article_unit.n_articles', minimum=1)
    except ValueError:
        pass
    else:
        if declared_articles != n_articles:
            problems.append(
                f'article_unit.n_articles={declared_articles} does not match '
                f'the {n_articles} distinct article ids in the served slice')
    return problems


def _article_sidecar_diag(meta, split, cache_dir):
    """Locate and describe the article block without reading the array."""
    block = meta.get('article_unit') if isinstance(meta, dict) else None
    diag = {'status': 'ok', 'split': str(split), 'unit': 'article',
            'source': 'cache_sidecar'}
    if not isinstance(block, dict):
        diag.update(status='legacy_no_article_sidecar',
                    reason=('the cache predates persisted article ids; '
                            'article clustering must be decoded at analysis '
                            'time'))
        return None, diag
    if block.get('available') is False or block.get('valid_for_clustering') is not True:
        diag.update(status='article_sidecar_unavailable',
                    reason=str(block.get('reason') or
                               'the cache records no identifiable article unit'))
        return None, diag
    name = block.get('sidecar_file')
    if not isinstance(name, str) or not name:
        diag.update(status='article_sidecar_unavailable',
                    reason='article_unit does not name a sidecar file')
        return None, diag
    if name != os.path.basename(name) or name in ('.', '..'):
        diag.update(status='article_sidecar_invalid',
                    reason=('article_unit.sidecar_file is not a plain '
                            'basename; refusing a path outside the cache'))
        return None, diag
    diag.update(sidecar_file=name,
                sidecar_sha256=block.get('sidecar_sha256'),
                n_articles=block.get('n_articles'),
                n_lines=block.get('n_lines'),
                valid_for_clustering=block.get('valid_for_clustering'),
                boundary_rule=block.get('boundary_rule'),
                pseudoreplication_factor=block.get('pseudoreplication_factor'))
    return os.path.join(str(cache_dir), name), diag


def load_cached_article_ids(cache_dir=None, *, split='test', seq_len,
                            vocab_size, n_seqs=None, n_train_tokens=None,
                            val_seqs=None, shape=None, strict=False):
    """Read the persisted per-token ARTICLE ids for a cached WikiText split.

    ``split='test'`` addresses the frozen test cache; ``split='validation'``
    addresses the train/validation cache.  The reader verifies the recorded
    length, digest and document->article nesting before returning, so analysis
    code never gets an unverified cluster label.

    Returns ``(ids_or_None, diagnostics)``.  A declared-but-corrupt sidecar is
    reported in ``diagnostics['problems']``; with ``strict=True`` it raises a
    ``RuntimeError`` instead of returning ``None``.  A legacy cache returns
    ``None`` with ``status='legacy_no_article_sidecar'`` and the caller keeps
    its explicit fallback.
    """
    cache_dir = _wt_cache_dir(cache_dir)
    seq_len = _protocol_pos_int('seq_len', seq_len)
    vocab_size = _protocol_pos_int('vocab_size', vocab_size)
    if split not in ('test', 'validation'):
        raise ValueError(
            f'load_cached_article_ids: split={split!r} must be '
            "'test' or 'validation'")
    if split == 'test':
        n_seqs = _protocol_pos_int('n_seqs', n_seqs)
        tag = f'v2_sl{seq_len}_v{vocab_size}_ns{n_seqs}'
        meta_path = os.path.join(cache_dir, f'test_meta_{tag}.json')
        line_path = os.path.join(cache_dir, f'test_token_doc_ids_{tag}.npy')
        expected = int(n_seqs) * int(seq_len)
    else:
        n_train_tokens = _protocol_pos_int('n_train_tokens', n_train_tokens)
        val_seqs = _protocol_pos_int('val_seqs', val_seqs)
        tag = (f'v5_sl{seq_len}_cap{n_train_tokens}_v{vocab_size}_vs{val_seqs}')
        meta_path = os.path.join(cache_dir, f'meta_{tag}.json')
        line_path = os.path.join(cache_dir, f'val_token_doc_ids_{tag}.npy')
        expected = int(val_seqs) * int(seq_len)
    diag = {'status': 'cache_absent', 'split': split, 'unit': 'article',
            'meta': os.path.basename(meta_path),
            'source': 'cache_sidecar'}
    if not os.path.exists(meta_path):
        return None, diag
    try:
        with open(meta_path, encoding='utf-8') as fh:
            meta = json.load(fh)
    except (OSError, ValueError, TypeError) as exc:
        diag.update(status='meta_unreadable',
                    reason=f'{type(exc).__name__}: {exc}')
        return None, diag
    if not isinstance(meta, dict):
        diag.update(status='meta_unreadable', reason='metadata is not an object')
        return None, diag
    path, diag = _article_sidecar_diag(meta, split, cache_dir)
    if path is None:
        return None, diag
    block = meta.get('article_unit') or {}
    problems = []
    try:
        art = np.load(path)
    except Exception as exc:
        problems.append(f'article sidecar unreadable ({type(exc).__name__}: {exc})')
        art = None
    recorded_sha = block.get('sidecar_sha256')
    if not (isinstance(recorded_sha, str) and len(recorded_sha) == 64
            and all(ch in '0123456789abcdefABCDEF' for ch in recorded_sha)):
        # A declared-but-unattested sidecar is not bound to the cache: without
        # the digest an edited array would pass every structural check.
        problems.append(
            'article_unit.sidecar_sha256 is missing or not a 64-character '
            'hex digest; the persisted article vector is not attested')
        recorded_sha = None
    if art is not None and recorded_sha:
        try:
            actual_sha = file_sha256(path)
        except Exception as exc:
            problems.append(
                f'article sidecar hash unreadable ({type(exc).__name__}: {exc})')
        else:
            if str(actual_sha) != str(recorded_sha):
                problems.append(
                    'article sidecar sha256 does not match the cache metadata')
    try:
        lines = np.load(line_path)
    except Exception as exc:
        problems.append(f'line sidecar unreadable ({type(exc).__name__}: {exc})')
        lines = None
    if art is not None and lines is not None:
        problems.extend(article_sidecar_problems(
            meta, art, lines, split=split, n_tokens=expected))
    if problems:
        diag['problems'] = problems
        diag['status'] = 'article_sidecar_invalid'
        if strict:
            raise RuntimeError(
                'load_cached_article_ids: the persisted article sidecar cannot '
                'be verified (' + '; '.join(problems) + ')')
        return None, diag
    arr = _as_sidecar_ids(art, 'token_article_ids', expected=expected)
    diag.update(status='ok', n_tokens=int(arr.size),
                n_articles=int(np.unique(arr).size))
    if shape is None and split == 'test':
        shape = (int(n_seqs), int(seq_len))
    elif shape is None:
        shape = (int(val_seqs), int(seq_len))
    if shape is not None:
        if int(np.prod(shape)) != int(arr.size):
            diag['status'] = 'shape_mismatch'
            diag['reason'] = (f'article vector has {int(arr.size)} tokens, '
                              f'cannot reshape to {tuple(shape)}')
            return None, diag
        arr = arr.reshape(tuple(int(x) for x in shape))
    return arr, diag


def article_sidecar_summary(meta, *, split='test'):
    """Small JSON-safe description of the recorded article unit."""
    block = meta.get('article_unit') if isinstance(meta, dict) else None
    if not isinstance(block, dict):
        return {'status': 'legacy_no_article_sidecar', 'split': split}
    out = {'status': 'recorded', 'split': split}
    for key in ('available', 'valid_for_clustering', 'n_articles', 'n_lines',
                'token_article_count', 'token_article_fp', 'sidecar_file',
                'sidecar_sha256', 'boundary_rule', 'source',
                'pseudoreplication_factor'):
        if key in block:
            value = block[key]
            if isinstance(value, (np.integer,)):
                value = int(value)
            out[key] = value
    return out
