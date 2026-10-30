

def wikitext_article_text(rows, group):
    """Canonical text of one article group (used for article-level dedup)."""
    _parts = []
    for _i in group:
        _s = rows[int(_i)]
        if not isinstance(_s, str):
            raise TypeError(
                f'wikitext_article_text: row {int(_i)} is not a string')
        _s = _s.rstrip('\r\n')
        if _s.strip():
            _parts.append(_s)
    return '\n'.join(_parts)


def wikitext_article_ids(rows, groups):
    """Stable article ids: content hash of the concatenated article text.

    The loader's existing document id is a hash of ONE row; an article id is a
    hash of the article's rows joined in order, so it is stable across caches
    built from the same raw source and it changes if any line changes.
    """
    return [int(_doc_id_u64(_doc_exact_fp(wikitext_article_text(rows, _g))))
            for _g in groups]


def line_spans_from_token_doc_ids(token_doc_ids):
    """Runs of equal ids in a per-token document map -> ``(start, stop, id)``.

    The per-token sidecar written by the loader repeats one line/document id
    for every token of that row (EOS included).  A run is therefore one served
    raw row; decoding the run recovers the line text without re-reading the
    corpus.  Used by the document-unit audit to regroup a frozen slice.
    """
    _arr = np.asarray(token_doc_ids)
    if _arr.ndim != 1:
        raise ValueError(
            f'line_spans_from_token_doc_ids: expected a 1-D map, got shape {_arr.shape}')
    if _arr.dtype.kind not in 'iu':
        raise ValueError(
            'line_spans_from_token_doc_ids: token document ids must be '
            f'integers, got dtype {_arr.dtype}')
    if _arr.size and int(_arr.min()) < 0:
        raise ValueError(
            'line_spans_from_token_doc_ids: token document ids must be non-negative')
    if _arr.size == 0:
        return []
    _change = np.flatnonzero(_arr[1:] != _arr[:-1]) + 1
    _starts = np.concatenate((np.asarray([0], dtype=np.int64), _change))
    _stops = np.concatenate((_change, np.asarray([_arr.size], dtype=np.int64)))
    return [(int(_s), int(_e), int(_arr[int(_s)]))
            for _s, _e in zip(_starts, _stops)]


def ensure_bytelevel_decoder(tok):
    """Attach the inverse of the byte-level pre-tokenizer, if it is missing.

    Legacy caches were persisted before `build_tokenizer` set the decoder, so
    their JSON has `decoder: null`.  Encode/ids are unaffected; this repairs
    only the in-memory decode path used by audits.  Returns True when a decoder
    was attached by this call.
    """
    try:
        if getattr(tok, 'decoder', None) is not None:
            return False
    except Exception:
        return False
    try:
        from tokenizers import decoders as _decoders
        tok.decoder = _decoders.ByteLevel()
        return True
    except Exception:
        return False


def decode_token_runs(token_ids, spans, tok, *, chunk=4096):
    """Decode each ``(start, stop, id)`` run of a token stream to text.

    Returns ``(texts, problems)``.  Decoding is only used for diagnostics and
    grouping; it does not rewrite any token or cache byte.  A run that fails to
    decode becomes ``None`` and is counted in ``problems`` instead of aborting
    the whole audit.  A legacy tokenizer with no decoder gets the byte-level
    inverse attached in memory first, so old caches remain auditable without a
    rebuild.
    """
    ensure_bytelevel_decoder(tok)
    _ids = np.asarray(token_ids).reshape(-1)
    texts = []
    problems = []
    try:
        _spans = list(spans)
    except TypeError:
        raise TypeError(
            f'decode_token_runs: spans must be iterable, got {type(spans).__name__}') from None
    _step = max(1, int(chunk))
    for _off in range(0, len(_spans), _step):
        _batch = _spans[_off:_off + _step]
        _encoded = []
        for _s, _e, _doc in _batch:
            if (not isinstance(_s, int) or not isinstance(_e, int)
                    or _s < 0 or _e < _s or _e > int(_ids.size)):
                raise ValueError(
                    f'decode_token_runs: invalid span ({_s!r}, {_e!r}) for a '
                    f'{int(_ids.size)}-token stream')
            _encoded.append([int(_x) for _x in _ids[_s:_e]])
        try:
            _decoded = tok.decode_batch(_encoded)
        except (AttributeError, TypeError, ValueError) as _e:
            problems.append(f'decode_batch failed: {type(_e).__name__}: {_e}')
            texts.extend([None] * len(_batch))
            continue
        texts.extend(_decoded)
    return texts, problems


def _article_group_summary(group, *, n_lines):
    """Serialisable diagnostics shared by the article-unit builders."""
    n_articles = int(group.get('n_articles') or 0)
    n_lines = int(n_lines)
    return {
        'unit': 'article',
        'boundary_rule': 'top_level_heading',
        'n_lines': n_lines,
        'n_articles': n_articles,
        'mean_lines_per_article': (float(n_lines) / n_articles
                                   if n_articles else None),
        'max_lines_per_article': int(group.get('max_group_rows') or 0),
        'pseudoreplication_factor': (float(n_lines) / n_articles
                                     if n_articles else None),
        'preamble_groups': int(group.get('preamble_groups') or 0),
        'has_headings': bool(group.get('has_headings')),
        'grouping_valid': bool(group.get('valid')),
        'problems': list(group.get('problems') or []),
    }


def _normalise_article_unit_text(text):
    return re.sub(r'\s+', ' ', str(text or '')).strip()


def _article_cluster_ids(rows, groups):
    """Stable CLUSTER ids for article grouping.

    Unlike :func:`wikitext_article_ids` (a content hash used for leakage
    audits), a cluster label must be identical when the same article is
    recovered from raw rows or from decoded token runs.  The raw line and its
    decoded form can differ in insignificant whitespace, so the key is the
    top-level title (normalised) plus the group's ordinal position.  A
    headerless preamble uses its whitespace-normalised text instead.  The
    ordinal keeps two same-titled articles distinct within one slice.
    """
    out = []
    for _gid, _idxs in enumerate(groups):
        _first = str(rows[int(_idxs[0])]).rstrip('\r\n')
        _m = _WIKITEXT_TOP_HEADING_RE.match(_first)
        if _m:
            _key = f'title:{_m.group(1).strip()}'
        else:
            _key = ('text:' + _normalise_article_unit_text(
                wikitext_article_text(rows, _idxs)))
        out.append(int(_doc_id_u64(_doc_exact_fp(f'{_gid}\0{_key}'))))
    return out


def wikitext_article_row_ids(rows):
    """Map raw WikiText rows to stable ARTICLE ids.

    The loader's ``doc_id`` is a hash of one raw row (a title, a paragraph or
    a section heading).  An article spans several rows and is delimited by a
    top-level `` = Title = `` heading.  This helper returns one article id per
    row, so a cache can persist the unit a document-level claim actually needs.

    Rows that are blank (or any row outside every group) get their own
    line-level id rather than being merged into a neighbouring article: a
    blank line must never make two articles one cluster.

    Returns a dict with ``status`` in {'ok', 'empty'}, ``row_article_id``
    (uint64, one entry per row), the article ids in group order, and the
    grouping diagnostics.  A corpus with no top-level heading is still
    returned (one group) but ``valid_for_clustering`` is False, because the
    article boundary is not evidenced; callers fall back explicitly.
    """
    _rows = list(rows)
    group = wikitext_article_groups(_rows)
    n_rows = int(group.get('n_rows') or 0)
    if not group.get('groups'):
        return {
            'status': 'empty',
            'row_article_id': np.empty(0, dtype=np.uint64),
            'article_ids': [],
            'valid_for_clustering': False,
            **_article_group_summary(group, n_lines=0),
        }
    article_ids = _article_cluster_ids(_rows, group['groups'])
    row_aid = np.empty(n_rows, dtype=np.uint64)
    # Start with a line-level id for every row, then overwrite the rows that
    # belong to an evidenced article.  Blank rows all have the same stripped
    # content, so a plain content hash made every blank row one cluster; add
    # the row position to keep the documented "each blank row is its own
    # cluster" contract.  Non-blank rows keep the loader's content line id.
    for _i, _row in enumerate(_rows):
        if isinstance(_row, str) and _row.strip():
            row_aid[_i] = _doc_id_u64(_doc_exact_fp(_row))
        else:
            row_aid[_i] = _doc_id_u64(_doc_exact_fp(f'blank\x00{_i}\x00{_row}'))
    for _gid, _idxs in enumerate(group['groups']):
        row_aid[np.asarray(_idxs, dtype=np.int64)] = np.uint64(article_ids[_gid])
    return {
        'status': 'ok',
        'row_article_id': row_aid,
        'article_ids': [int(x) for x in article_ids],
        'valid_for_clustering': bool(group.get('has_headings')),
        **_article_group_summary(
            group, n_lines=int(group.get('n_nonempty_rows') or n_rows)),
    }


def wikitext_token_article_ids_from_line_map(token_ids, token_doc_ids, tok,
                                             *, chunk=4096):
    """Recover per-token ARTICLE ids from a frozen cache without reading raw.

    ``token_doc_ids`` is the per-token line mapping persisted by the loader.
    Each maximal run of equal ids is one served raw row; decoding the run
    recovers its text (the tokenizer's ByteLevel decoder is attached in memory
    for legacy caches).  Rows are then regrouped by top-level headings.

    This is the path a frozen test cache uses on a cache hit: the held-out raw
    file is not re-read and no token/cache byte is rewritten.  The recovered
    ids are used only to choose the resampling CLUSTER, so the pooled PPL and
    every line-level field stay bit-identical.

    Returns a dict.  ``status`` is 'ok' only when every run decoded and at
    least one group exists; ``valid_for_clustering`` additionally requires an
    evidenced top-level heading.  On a decode failure the caller must fall
    back to line-level uncertainty, never guess the article.
    """
    ids = np.asarray(token_ids)
    line_ids = np.asarray(token_doc_ids)
    if tuple(ids.shape) != tuple(line_ids.shape):
        # A cache stores the per-token map flat while the served panel is 2-D.
        # Accept the unambiguous total-size match, but never a different
        # length (that would shift every line boundary silently).
        if int(ids.size) == int(line_ids.size):
            line_ids = line_ids.reshape(ids.shape)
        else:
            return {'status': 'shape_mismatch',
                    'reason': (f'token_ids shape {tuple(ids.shape)} != line map '
                               f'shape {tuple(line_ids.shape)}')}
    if line_ids.size == 0:
        return {'status': 'empty', 'reason': 'empty token/line map'}
    flat_line = line_ids.reshape(-1)
    flat_ids = ids.reshape(-1)
    spans = line_spans_from_token_doc_ids(flat_line)
    # The loader cuts the served per-token map at a fixed token cap, so the
    # final run can be a PREFIX of a raw row.  A partial BODY line still
    # belongs to the preceding article and can be decoded/grouped normally.
    # A partial HEADING, however, does not match the whole-heading regex and
    # would be folded into the previous article, disagreeing with the
    # cold-build helper that sees the full heading.  Keep a partial
    # heading-like run at line identity (in both paths) instead of guessing an
    # article from incomplete text.
    try:
        _eos = tok.token_to_id('<eos>')
    except Exception:
        _eos = None
    _last_partial = bool(spans) and _eos is not None and int(flat_ids[-1]) != int(_eos)
    texts, problems = decode_token_runs(flat_ids, spans, tok, chunk=chunk)
    if problems:
        return {'status': 'decode_failed',
                'reason': '; '.join(problems[:3]),
                'problems': problems}
    if any(t is None for t in texts):
        return {'status': 'decode_failed',
                'reason': 'one or more line runs could not be decoded'}
    _partial_heading = bool(
        _last_partial
        and re.match(r'^\s*=\s*(?!=)', str(texts[-1] or '')))
    _group_texts = texts[:-1] if _partial_heading else texts
    group = wikitext_article_groups(_group_texts)
    if not group.get('groups') and not (_last_partial and _partial_heading):
        return {'status': 'empty_after_grouping',
                'reason': 'decoded line runs grouped to zero articles'}
    article_ids = (_article_cluster_ids(_group_texts, group['groups'])
                   if group.get('groups') else [])
    # One line-level id for every served run; a partial heading keeps that
    # identity instead of being merged into the previous article.
    line_aid = np.empty(len(spans), dtype=np.uint64)
    for _i, (_s, _e, _lid) in enumerate(spans):
        line_aid[_i] = np.uint64(_lid)
    for _gid, _idxs in enumerate(group['groups']):
        line_aid[np.asarray(_idxs, dtype=np.int64)] = np.uint64(article_ids[_gid])
    out = np.empty(flat_line.size, dtype=np.uint64)
    for (_s, _e, _lid), _aid in zip(spans, line_aid):
        out[int(_s):int(_e)] = _aid
    # Report the same served-cluster accounting as the cold-build helper:
    # the partial final run contributes one line-level cluster, not an
    # unmeasured article.
    _n_served_articles = int(np.unique(line_aid).size)
    _served_counts = (np.unique(line_aid, return_counts=True)[1]
                      if _n_served_articles else np.empty(0, dtype=np.int64))
    _summary = _article_group_summary(group, n_lines=len(spans))
    _summary.update(
        n_lines=int(len(spans)),
        n_articles=_n_served_articles,
        mean_lines_per_article=(float(len(spans)) / _n_served_articles
                                if _n_served_articles else None),
        pseudoreplication_factor=(float(len(spans)) / _n_served_articles
                                  if _n_served_articles else None),
        max_lines_per_article=(int(_served_counts.max())
                               if _served_counts.size else 0),
    )
    return {
        'status': 'ok',
        'token_article_ids': out.reshape(ids.shape),
        'line_article_ids': line_aid,
        'line_ids': [int(s[2]) for s in spans],
        'last_run_partial': bool(_last_partial),
        'valid_for_clustering': bool(group.get('has_headings')),
        **_summary,
    }


def wikitext_token_article_ids_for_rows(rows, lengths, cap_tokens):
    """Build per-token ARTICLE ids for a capped token stream from raw rows.

    ``rows`` are the non-empty raw rows aligned with ``lengths`` (each
    ``len(encoded_ids) + 1``, the EOS included), exactly as produced by
    ``_tokenise_row_chunks``.  This is the cache-build counterpart of
    :func:`wikitext_token_article_ids_from_line_map`; it writes the same
    grouping without ever needing to decode the token stream.
    """
    rows = list(rows)
    lengths = list(lengths)
    if len(rows) != len(lengths):
        raise ValueError(
            f'wikitext_token_article_ids_for_rows: {len(rows)} rows but '
            f'{len(lengths)} lengths')
    takes = []
    lens_used = []
    cursor = 0
    cap_tokens = max(0, int(cap_tokens))
    for _i, (_row, _length) in enumerate(zip(rows, lengths)):
        _len_i = _exact_count(_length, minimum=1)
        if _len_i is None:
            raise ValueError(
                'wikitext_token_article_ids_for_rows: lengths['
                f'{_i}]={_length!r} is not a positive integer token count')
        if cursor >= cap_tokens:
            break
        take = min(int(_len_i), cap_tokens - cursor)
        if take <= 0:
            break
        takes.append(take)
        lens_used.append(int(_len_i))
        cursor += take
    # The cap can cut the final row.  A partial BODY line stays in the
    # preceding article (its full text does not change the grouping).  A
    # partial top-level HEADING must instead keep line identity: the decoded
    # prefix does not match the whole-heading rule, and grouping the full
    # heading would create an article the cache-hit path cannot reconstruct.
    _last_truncated = bool(takes) and int(takes[-1]) < int(lens_used[-1])
    _partial_heading = bool(
        _last_truncated
        and _WIKITEXT_TOP_HEADING_RE.match(str(rows[len(takes) - 1])))
    _n_complete = len(takes) - (1 if _partial_heading else 0)
    _group_rows = rows[:_n_complete] if _partial_heading else rows
    rid = wikitext_article_row_ids(_group_rows)
    if rid['status'] != 'ok' and _n_complete:
        return {'status': rid['status'],
                'token_article_ids': np.empty(0, dtype=np.uint64),
                **{k: v for k, v in rid.items()
                   if k not in ('status', 'row_article_id', 'article_ids')}}
    row_aid = np.asarray(rid['row_article_id'], dtype=np.uint64)
    if not takes:
        out = np.empty(0, dtype=np.uint64)
        served_aid = np.empty(0, dtype=np.uint64)
    else:
        served_aid = np.empty(len(takes), dtype=np.uint64)
        _copy_n = min(len(takes), int(row_aid.size))
        if _copy_n:
            served_aid[:_copy_n] = row_aid[:_copy_n]
        if _partial_heading:
            # Use the same content line id the loader wrote into the sidecar;
            # the partial heading is not assigned to a neighbouring article.
            served_aid[-1] = _doc_id_u64(
                _doc_exact_fp(rows[len(takes) - 1]))
        out = np.repeat(served_aid, np.asarray(takes, dtype=np.int64))
    _n_served_articles = int(np.unique(served_aid).size)
    _served_counts = (np.unique(served_aid, return_counts=True)[1]
                      if _n_served_articles else np.empty(0, dtype=np.int64))
    _served = {k: v for k, v in rid.items()
               if k not in ('status', 'row_article_id', 'article_ids')}
    # The diagnostics describe the SERVED (capped) stream, not every raw row
    # that happened to be passed in: an article beyond the cap is not an
    # evaluation unit of this cell.
    _served.update(
        n_lines=int(len(takes)),
        n_articles=_n_served_articles,
        mean_lines_per_article=(float(len(takes)) / _n_served_articles
                                if _n_served_articles else None),
        pseudoreplication_factor=(float(len(takes)) / _n_served_articles
                                  if _n_served_articles else None),
        max_lines_per_article=(int(_served_counts.max())
                               if _served_counts.size else 0),
    )
    return {
        'status': 'ok',
        'token_article_ids': out,
        'line_article_ids': served_aid,
        'n_tokens': int(out.size),
        'last_run_partial': bool(_last_truncated),
        'valid_for_clustering': bool(rid['valid_for_clustering']),
        **_served,
    }


def _sampling_int(name, value, *, minimum=0, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f'{name} must be an integer, got {type(value).__name__}')
    out = int(value)
    if out < int(minimum):
        raise ValueError(f'{name} must be >= {int(minimum)}, got {out}')
    if maximum is not None and out > int(maximum):
        raise ValueError(f'{name} must be <= {int(maximum)}, got {out}')
    return out


def _sampling_plan_fp(plan):
    """Canonical fingerprint of a sampling plan (excluding ``plan_fp``)."""
    body = {k: v for k, v in plan.items() if k != 'plan_fp'}
    blob = json.dumps(body, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def _sampling_finite_sum(values):
    """Sum a finite size vector, or return ``None`` when truly unrepresentable.

    Ordinary corpora keep the exact NumPy sum (so persisted plans do not
    change).  A syntactically valid scale-up declaration may carry sizes near
    ``float.max``; summing those overflows even though every element is finite,
    and a NaN/inf coverage fraction makes the whole plan non-JSON.  Normalise
    by the largest element only on that overflow path.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    with np.errstate(over='ignore', invalid='ignore'):
        total = float(arr.sum())
    if math.isfinite(total):
        return total
    scale = float(np.max(np.abs(arr)))
    if not (math.isfinite(scale) and scale > 0.0):
        return None
    with np.errstate(over='ignore', invalid='ignore'):
        unit = math.fsum((arr / scale).tolist())
    if not (math.isfinite(unit) and unit > 0.0):
        return None
    try:
        log_total = math.log(scale) + math.log(unit)
    except (ValueError, OverflowError):
        return None
    if log_total > math.log(sys.float_info.max):
        return None
    out = math.exp(log_total)
    return out if math.isfinite(out) else None


def _sampling_allocation(n_units, n_select, stratum_sizes, allocation,
                         min_per_stratum, stratum_size_sums=None):
    """Integer allocation of ``n_select`` units over non-empty strata.

    ``proportional`` allocates in proportion to each stratum's unit count,
    ``equal`` gives every stratum the same quota, and ``size_weighted`` uses
    the sum of the per-unit sizes.  Floors are followed by a deterministic
    largest-remainder pass; a final capped pass moves units between strata
    that cannot supply or absorb their quota.  The result is exact,
    non-negative, never exceeds a stratum's population and honours
    ``min_per_stratum`` (an infeasible floor is an error, not silently
    lowered).

    The remainder is distributed in bulk rather than one unit at a time:
    when such a capped allocation is asked for millions of units, a
    per-unit loop turns into millions of full-array argmax passes.
    """
    S = len(stratum_sizes)
    counts = np.asarray(stratum_sizes, dtype=np.int64)
    if allocation == 'proportional':
        weights = counts.astype(np.float64)
    elif allocation == 'equal':
        weights = np.ones(S, dtype=np.float64)
    elif allocation == 'size_weighted':
        weights = np.asarray(
            stratum_size_sums if stratum_size_sums is not None else stratum_sizes,
            dtype=np.float64)
    else:
        raise ValueError(
            f'allocation must be one of proportional/equal/size_weighted, got {allocation!r}')
    with np.errstate(over='ignore', invalid='ignore'):
        _wsum = float(weights.sum())
    if not np.isfinite(weights).all() or not math.isfinite(_wsum) or _wsum <= 0:
        if (np.isfinite(weights).all() and np.any(weights > 0.0)):
            # The elements are representable but their sum is not.  Only the
            # relative weights enter the allocation; normalise by the largest
            # weight so a scale-up size declaration keeps size weighting
            # instead of silently falling back to unit-count weighting.
            weights = weights / float(np.max(weights))
            _wsum = float(weights.sum())
        if not (math.isfinite(_wsum) and _wsum > 0):
            weights = counts.astype(np.float64)
            _wsum = float(weights.sum())
    min_per_stratum = int(min_per_stratum)
    if min_per_stratum < 0:
        raise ValueError(
            f'min_per_stratum must be non-negative, got {min_per_stratum}')
    if S and int(counts.min()) < min_per_stratum:
        raise ValueError(
            f'min_per_stratum={min_per_stratum} exceeds the smallest stratum '
            f'({int(counts.min())} unit(s)); the per-stratum floor cannot be met')
    if n_select < S * min_per_stratum:
        raise ValueError(
            f'n_select={n_select} cannot give {min_per_stratum} unit(s) '
            f'to each of {S} non-empty strata; lower min_per_stratum or select more')
    ideal = weights / _wsum * float(n_select)
    alloc = np.floor(ideal).astype(np.int64)
    alloc = np.minimum(alloc, counts)
    # Guarantee the per-stratum minimum before spending the remainder.
    alloc = np.maximum(alloc, min_per_stratum)
    alloc = np.minimum(alloc, counts)
    total = int(alloc.sum())
    target = int(n_select)
    while total < target:
        # Prefer strata still below their ideal quota, then any stratum with
        # room.  Each pass adds an equal share to the eligible set and caps at
        # the stratum's population; a capped stratum drops out next pass, so
        # the number of passes is bounded by the number of strata, not by the
        # number of remaining units.
        eligible = np.flatnonzero(alloc < counts)
        if not eligible.size:
            break
        positive = eligible[ideal[eligible] - alloc[eligible] > 0]
        if positive.size:
            eligible = positive
        missing = target - total
        share, extra = divmod(missing, int(eligible.size))
        if share:
            add = np.minimum(counts[eligible] - alloc[eligible], share)
            alloc[eligible] += add
            total += int(add.sum())
        else:
            deficit = ideal[eligible] - alloc[eligible]
            order = np.lexsort((eligible, -deficit))
            chosen = eligible[order[:extra]]
            alloc[chosen] += 1
            total += int(extra)
    while total > target:
        # Remove from strata furthest above their ideal quota, never below the
        # declared floor; bulk-remove like the addition pass above.
        eligible = np.flatnonzero(alloc > min_per_stratum)
        if not eligible.size:
            break
        positive = eligible[alloc[eligible] - ideal[eligible] > 0]
        if positive.size:
            eligible = positive
        surplus = total - target
        share, extra = divmod(surplus, int(eligible.size))
        if share:
            sub = np.minimum(alloc[eligible] - min_per_stratum, share)
            alloc[eligible] -= sub
            total -= int(sub.sum())
        else:
            excess = alloc[eligible] - ideal[eligible]
            order = np.lexsort((eligible, -excess))
            chosen = eligible[order[:extra]]
            alloc[chosen] -= 1
            total -= int(extra)
    if total != target:
        raise ValueError(
            f'could not allocate n_select={n_select} over strata sizes '
            f'{counts.tolist()} with min_per_stratum={min_per_stratum}')
    return alloc


def document_stratified_plan(sizes, n_select, *, seed=0, n_strata=4,
                             allocation='proportional', min_per_stratum=1,
                             source_fp=None):
    """Deterministic stratified sample of independent units by size.

    The units are sorted by ``(size, index)`` and cut into ``n_strata``
    equal-count blocks.  A seeded, without-replacement sample is drawn inside
    each block.  Because the strata are equal-count, proportional allocation
    is also equal allocation across strata: every selected unit has the same
    inclusion probability *conditional on its stratum*, and every stratum is
    represented when ``n_select >= n_strata * min_per_stratum``.

    This is a *sampling-design* helper, not a loader default: it enables a
    future freeze to draw a probability sample instead of taking the first
    rows in source order.  ``source_fp`` binds the plan to a corpus snapshot;
    changing it changes the selected units even when the sizes and seed are
    unchanged.
    """
    try:
        arr = np.asarray(sizes)
    except Exception as e:
        raise TypeError(f'sizes must be a 1-D numeric sequence ({type(e).__name__}: {e})') from None
    if arr.ndim != 1:
        raise ValueError(f'sizes must be 1-D, got shape {arr.shape}')
    if arr.size == 0:
        raise ValueError('sizes must contain at least one unit')
    if not (np.issubdtype(arr.dtype, np.number) and not np.iscomplexobj(arr)):
        raise TypeError(f'sizes must be numeric, got dtype {arr.dtype}')
    sizes_f = arr.astype(np.float64, copy=False)
    if not np.isfinite(sizes_f).all():
        raise ValueError('sizes must be finite (no NaN/inf)')
    if (sizes_f < 0).any():
        raise ValueError('sizes must be non-negative')
    if not np.any(sizes_f > 0.0):
        raise ValueError('sizes must have a positive total')
    seed = _sampling_int('seed', seed, minimum=0)
    n_strata = _sampling_int('n_strata', n_strata, minimum=1)
    n_select = _sampling_int('n_select', n_select, minimum=1)
    min_per_stratum = _sampling_int('min_per_stratum', min_per_stratum, minimum=0)
    n_units = int(sizes_f.size)
    if n_select > n_units:
        raise ValueError(f'n_select={n_select} exceeds the {n_units} available units')
    if source_fp is not None and not isinstance(source_fp, str):
        raise TypeError(f'source_fp must be a string or None, got {type(source_fp).__name__}')
    n_strata = min(int(n_strata), n_units)
    order = np.lexsort((np.arange(n_units, dtype=np.int64), sizes_f))
    stratum_members = [np.asarray(x, dtype=np.int64)
                       for x in np.array_split(order, n_strata)]
    stratum_sizes = [int(m.size) for m in stratum_members]
    _size_sums = [_sampling_finite_sum(sizes_f[m]) for m in stratum_members]
    if any(v is None for v in _size_sums):
        # Only the relative weights matter for `size_weighted`; use a common
        # scale so an overflow in one stratum cannot silently fall back to the
        # unit-count weighting.
        _sum_scale = float(np.max(np.abs(sizes_f)))
        _size_sums = [math.fsum((sizes_f[m] / _sum_scale).tolist())
                      if math.isfinite(_sum_scale) and _sum_scale > 0.0 else 1.0
                      for m in stratum_members]
    alloc = _sampling_allocation(n_units, n_select, stratum_sizes,
                                 allocation, min_per_stratum,
                                 stratum_size_sums=_size_sums)
    selected = []
    strata = []
    for _i, (_members, _k) in enumerate(zip(stratum_members, alloc)):
        _k = int(_k)
        if _k:
            # Hash the plan parameters into a per-stratum seed so the draw is
            # independent of NumPy's global random state and of the order in
            # which callers request plans.
            _key = json.dumps({'schema': 'document_stratified_sample_v1',
                               'seed': seed, 'stratum': _i,
                               'source_fp': source_fp, 'n_units': n_units,
                               'n_select': n_select, 'n_strata': n_strata},
                              sort_keys=True, separators=(',', ':'))
            _digest = hashlib.sha256(_key.encode('utf-8')).digest()
            _rng = np.random.default_rng(int.from_bytes(_digest[:16], 'little'))
            _take = np.sort(_rng.choice(_members, size=_k, replace=False))
            selected.extend(int(x) for x in _take.tolist())
        else:
            _take = np.empty(0, dtype=np.int64)
        _sizes = sizes_f[_members]
        strata.append({
            'index': int(_i),
            'n_units': int(_members.size),
            'n_selected': _k,
            'size_sum': _sampling_finite_sum(_sizes),
            'min_size': float(_sizes.min()) if _members.size else None,
            'max_size': float(_sizes.max()) if _members.size else None,
            'selected_fraction': (float(_k) / float(_members.size)
                                  if _members.size else 0.0),
        })
    selected = sorted(selected)
    if len(selected) != n_select or len(set(selected)) != n_select:
        raise RuntimeError('internal error: stratified sample is not a '
                           f'without-replacement sample of size {n_select}')
    all_represented = all(s['n_selected'] > 0 for s in strata)
    total_size = _sampling_finite_sum(sizes_f)
    selected_size = _sampling_finite_sum(
        sizes_f[np.asarray(selected, dtype=np.int64)])
    if (total_size is not None and selected_size is not None
            and total_size > 0.0):
        size_coverage = selected_size / total_size
    else:
        # The absolute sums can overflow while their ratio is a well-defined
        # fraction in [0, 1].  Compute that ratio on a common unit scale.
        _size_scale = float(np.max(np.abs(sizes_f)))
        if math.isfinite(_size_scale) and _size_scale > 0.0:
            _all_units = math.fsum((sizes_f / _size_scale).tolist())
            _sel_units = math.fsum(
                (sizes_f[np.asarray(selected, dtype=np.int64)] / _size_scale).tolist())
            size_coverage = (_sel_units / _all_units
                             if _all_units > 0.0 else 0.0)
        else:
            size_coverage = 0.0
    # Hash the size vector incrementally: the same digest as
    # ','.join(format(float(x), '.17g') ...), without materialising a string
    # with one entry per corpus unit (which is large for a scale-up corpus).
    _sizes_hash = hashlib.sha256()
    for _i, _x in enumerate(sizes_f):
        if _i:
            _sizes_hash.update(b',')
        _sizes_hash.update(format(float(_x), '.17g').encode('ascii'))
    plan = {
        'schema': 'document_stratified_sample_v1',
        'design': 'stratified_document',
        'seed': int(seed),
        'n_units': n_units,
        'n_select': int(n_select),
        'n_strata': int(n_strata),
        'allocation': str(allocation),
        'min_per_stratum': int(min_per_stratum),
        'source_fp': source_fp,
        'sizes_fp': _sizes_hash.hexdigest(),
        'selected': selected,
        'selection_without_replacement': True,
        'all_strata_represented': bool(all_represented),
        'unit_coverage_fraction': float(n_select) / float(n_units),
        'size_coverage_fraction': float(size_coverage),
        'strata': strata,
    }
    plan['plan_fp'] = _sampling_plan_fp(plan)
    return plan


def sampling_plan_problems(plan):
    """Structural/consistency problems with a persisted sampling plan.

    This deliberately re-derives the fingerprint and the stratum totals from
    the plan itself: a plan edited after the freeze (or one whose selected
    units no longer sum to ``n_select``) is rejected instead of being quoted
    as a probability sample.
    """
    if not isinstance(plan, dict):
        return [f'plan is not an object (got {type(plan).__name__})']
    problems = []
    try:
        if plan.get('schema') != 'document_stratified_sample_v1':
            problems.append(f"schema={plan.get('schema')!r}")
        if plan.get('design') != 'stratified_document':
            problems.append(f"design={plan.get('design')!r}")
        n_units = _sampling_int('n_units', plan.get('n_units'), minimum=1)
        n_select = _sampling_int('n_select', plan.get('n_select'), minimum=1)
        n_strata = _sampling_int('n_strata', plan.get('n_strata'), minimum=1)
        seed = _sampling_int('seed', plan.get('seed'), minimum=0)
        if n_select > n_units:
            problems.append(f'n_select={n_select} > n_units={n_units}')
        if n_strata > n_units:
            problems.append(f'n_strata={n_strata} > n_units={n_units}; '
                            'a stratum cannot be empty')
        selected = plan.get('selected')
        if not isinstance(selected, list):
            problems.append('selected is not a list')
            selected = []
        if len(selected) != n_select:
            problems.append(f'selected has {len(selected)} entries, expected {n_select}')
        _sel = []
        for i, x in enumerate(selected):
            try:
                _xi = _sampling_int(f'selected[{i}]', x, minimum=0, maximum=n_units - 1)
            except (TypeError, ValueError) as e:
                problems.append(str(e))
                continue
            _sel.append(_xi)
        if len(set(_sel)) != len(_sel):
            problems.append('selected contains duplicate unit indices')
        if _sel != sorted(_sel):
            problems.append('selected is not sorted by source index')
        if plan.get('selection_without_replacement') is not True:
            problems.append('selection_without_replacement is not true')
        strata = plan.get('strata')
        if not isinstance(strata, list) or len(strata) != n_strata:
            problems.append('strata is not a list of length n_strata')
            strata = []
        _sum_units = 0
        _sum_selected = 0
        _all_repr = True
        for i, s in enumerate(strata):
            if not isinstance(s, dict):
                problems.append(f'strata[{i}] is not an object')
                continue
            try:
                _nu = _sampling_int(f'strata[{i}].n_units', s.get('n_units'), minimum=0)
                _ns = _sampling_int(f'strata[{i}].n_selected', s.get('n_selected'), minimum=0)
            except (TypeError, ValueError) as e:
                problems.append(str(e))
                continue
            if _nu <= 0:
                problems.append(f'strata[{i}] has no units; strata must be non-empty')
            if _ns > _nu:
                problems.append(f'strata[{i}] selects {_ns} of {_nu} units')
            _sum_units += _nu
            _sum_selected += _ns
            _all_repr = _all_repr and _ns > 0
        if strata and _sum_units != n_units:
            problems.append(f'stratum n_units sum {_sum_units} != n_units {n_units}')
        if strata and _sum_selected != n_select:
            problems.append(f'stratum n_selected sum {_sum_selected} != n_select {n_select}')
        if bool(plan.get('all_strata_represented')) != bool(_all_repr):
            problems.append('all_strata_represented disagrees with the stratum counts')
        if seed < 0:
            problems.append('seed must be non-negative')
        _fp = plan.get('plan_fp')
        if not isinstance(_fp, str) or _fp != _sampling_plan_fp(plan):
            problems.append('plan_fp does not match the plan contents')
        if plan.get('source_fp') is not None and not isinstance(plan.get('source_fp'), str):
            problems.append('source_fp is not a string or null')
    except (TypeError, ValueError, KeyError) as e:
        problems.append(f'invalid plan ({type(e).__name__}: {e})')
    return problems


def wikitext_article_sampling_plan(rows, n_articles, *, seed=0, n_strata=4,
                                   allocation='proportional', size_unit='rows',
                                   min_per_stratum=1, source_fp=None):
    """Stratified article sample for a raw WikiText split.

    Unlike the loaders' source-order prefix, this returns whole articles drawn
    without replacement from equal-count size strata.  ``size_unit='rows'``
    stratifies by article length in raw rows; ``size_unit='chars'`` by the
    concatenated article character count.  Whole articles are selected, so a
    caller can materialise their rows without splitting an article across
    selected and unselected units.

    The returned ``selected_row_indices`` are sorted and every row of a
    selected article is included.  ``valid_for_article_sampling`` is false
    for corpora without top-level headings; the function refuses instead of
    silently treating lines as articles.
    """
    _rows = list(rows)
    groups = wikitext_article_groups(_rows)
    if not groups.get('valid'):
        raise ValueError(
            'wikitext_article_sampling_plan: the rows cannot be segmented into '
            'articles by top-level headings (' + '; '.join(groups.get('problems') or []) + ')')
    _groups = groups['groups']
    if size_unit == 'rows':
        sizes = [int(len(g)) for g in _groups]
    elif size_unit == 'chars':
        sizes = [int(len(wikitext_article_text(_rows, g))) for g in _groups]
    else:
        raise ValueError(f"size_unit must be 'rows' or 'chars', got {size_unit!r}")
    if source_fp is None:
        _src_hash = hashlib.sha256()
        for _r in _rows:
            _src_hash.update(_r.encode('utf-8', errors='replace'))
            _src_hash.update(b'\n')
        source_fp = _src_hash.hexdigest()
    plan = document_stratified_plan(
        sizes, n_articles, seed=seed, n_strata=n_strata,
        allocation=allocation, min_per_stratum=min_per_stratum,
        source_fp=source_fp)
    selected_article_indices = list(plan['selected'])
    selected_row_indices = sorted(
        int(i) for _a in selected_article_indices for i in _groups[_a])
    article_ids = wikitext_article_ids(_rows, _groups)
    return {
        'status': 'ok',
        'schema': 'wikitext_article_sample_v1',
        'design': 'stratified_article',
        'size_unit': size_unit,
        'n_articles_available': int(len(_groups)),
        'n_articles_selected': int(len(selected_article_indices)),
        'selected_article_indices': selected_article_indices,
        'selected_article_ids': [int(article_ids[i]) for i in selected_article_indices],
        'selected_article_sizes': [int(sizes[i]) for i in selected_article_indices],
        'selected_row_indices': selected_row_indices,
        'article_grouping': {
            'boundary_rule': groups.get('boundary_rule'),
            'n_nonempty_rows': int(groups.get('n_nonempty_rows', 0)),
            'n_articles': int(groups.get('n_articles', 0)),
            'max_rows_per_article': int(groups.get('max_group_rows', 0)),
            'mean_rows_per_article': float(groups.get('mean_group_rows', 0.0)),
        },
        'plan': plan,
        'plan_fp': plan['plan_fp'],
    }


def select_sampled_items(items, plan):
    """Materialise the units named by a validated ``document_stratified_plan``.

    Indexable containers (lists, arrays, ``range``) are never copied in full:
    only the selected units are materialised, so a plan over a scale-up corpus
    does not allocate a second copy of the whole corpus just to pick a subset.
    """
    problems = sampling_plan_problems(plan)
    if problems:
        raise ValueError('select_sampled_items: invalid plan: ' + '; '.join(problems))
    n_units = int(plan['n_units'])
    # Only containers whose ``__getitem__`` is positional may use the fast
    # path.  A dict, for example, would interpret the integer indices as keys,
    # so it must go through the generic sequence fallback instead.
    if isinstance(items, (list, tuple, range, np.ndarray)):
        n_items = len(items)
        if n_items != n_units:
            raise ValueError(
                f'select_sampled_items: {n_items} items but the plan describes '
                f'{n_units} units')
        return [items[int(i)] for i in plan['selected']]
    seq = list(items)
    if len(seq) != n_units:
        raise ValueError(
            f'select_sampled_items: {len(seq)} items but the plan describes '
            f'{n_units} units')
    return [seq[int(i)] for i in plan['selected']]


class _DocDeduplicator:
    """Exact + near-duplicate document filter with a deterministic LSH index.

    A document is a group: once seen (tokenizer fit, train or validation) any
    exact copy or a near copy within ``near_threshold`` SimHash bits is skipped
    on every later split.  Skipped near copies are quarantined in the index as
    well, so a chain A~T, B~A cannot slip B through when B is just outside the
    threshold for T: membership is closed under the LSH matches already made.
    This is what keeps a document (or a lightly edited copy) from appearing on
    both sides of the split.
    """

    def __init__(self, *, near_threshold=None, bands=None,
                 shingle_words=None, max_shingles=None, scope=None):
        # Every control defines the LSH/similarity geometry and therefore the
        # cleaned corpus.  int(2.5) would silently run a different band layout,
        # and a negative threshold would disable near-duplicate detection while
        # looking like a validated setting.  ``None`` resolves to the declared
        # project protocol so the loader and the audit share one default.
        near_threshold = (DEDUP_PROTOCOL_DEFAULTS['near_hamming_threshold']
                          if near_threshold is None else near_threshold)
        bands = DEDUP_PROTOCOL_DEFAULTS['bands'] if bands is None else bands
        shingle_words = (DEDUP_PROTOCOL_DEFAULTS['shingle_words']
                         if shingle_words is None else shingle_words)
        max_shingles = (DEDUP_PROTOCOL_DEFAULTS['max_shingles']
                        if max_shingles is None else max_shingles)
        scope = DEDUP_PROTOCOL_DEFAULTS['scope'] if scope is None else scope
        _near = _as_seed_int(near_threshold)
        if _near is None or not (0 <= _near <= 64):
            raise ValueError(f'near_threshold must be an integer in [0, 64], got {near_threshold!r}')
        _bands = _as_seed_int(bands)
        if _bands is None or not (1 <= _bands <= 8):
            raise ValueError(f'bands must be an integer in [1, 8], got {bands!r}')
        if _near >= _bands:
            # With fewer differing bits than bands, some pair within the
            # threshold must share a full band.  Once threshold >= bands that
            # guarantee is gone and the index can silently miss a near copy;
            # fail closed instead of reporting a weaker cleanup as exact.
            raise ValueError(
                'near_threshold must be < bands for exhaustive LSH candidate '
                f'search (got near_threshold={_near}, bands={_bands})')
        if 64 % _bands:
            # The band hash is built from ``64 // bands``-bit chunks.  A band
            # count that does not divide 64 leaves high bits out of every
            # bucket key, so even a zero-bit pair can differ only in those
            # uncovered bits and evade the candidate search.
            raise ValueError(
                'bands must divide 64 so every SimHash bit is indexed '
                f'(got bands={_bands})')
        _words = _as_seed_int(shingle_words)
        if _words is None or not (1 <= _words <= (1 << 20)):
            raise ValueError(
                f'shingle_words must be a positive integer, got {shingle_words!r}')
        _max_sh = _as_seed_int(max_shingles)
        if _max_sh is None or not (1 <= _max_sh <= (1 << 20)):
            raise ValueError(
                f'max_shingles must be a positive integer, got {max_shingles!r}')
        if _words > _max_sh:
            raise ValueError(
                f'shingle_words ({_words}) must not exceed max_shingles ({_max_sh})')
        if scope not in ('raw_line', 'document'):
            raise ValueError(
                f"scope must be 'raw_line' or 'document', got {scope!r}")
        self.near_threshold = _near
        self.bands = _bands
        self.shingle_words = _words
        self.max_shingles = _max_sh
        self.scope = str(scope)
        self.bits = 64 // self.bands
        self.mask = (1 << self.bits) - 1
        self.exact = set()
        self.buckets = {}
        self.n_exact_skipped = 0
        self.n_near_skipped = 0
        self.n_seen = 0

    def _bucket_keys(self, h):
        return [(b, (int(h) >> (b * self.bits)) & self.mask)
                for b in range(self.bands)]

    def _match(self, text, fp):
        """Return ``(duplicate, kind, fp_key, simhash)`` without mutating.

        The SimHash is returned even when no match is found so callers can add
        the document once, exactly like :meth:`is_duplicate` does."""
        if fp is None:
            fp = _doc_exact_fp(text)
        # The content hash is a 128-bit value; storing it as a Python int
        # keeps exactly the same identity as the hex string while representing
        # each document with one integer.
        fp_key = int(fp, 16) if isinstance(fp, (str, bytes)) else int(fp)
        if fp_key in self.exact:
            return True, 'exact', fp_key, None
        # Keep the historical single-argument call on the default geometry:
        # callers/tests that substitute ``_doc_simhash`` still work, while a
        # non-default protocol passes its shingle geometry explicitly.
        if (self.shingle_words == DEDUP_SIMHASH_SHINGLE_WORDS
                and self.max_shingles == DEDUP_SIMHASH_MAX_SHINGLES):
            h = _doc_simhash(text)
        else:
            h = _doc_simhash(text, shingle_words=self.shingle_words,
                             max_shingles=self.max_shingles)
        for key in self._bucket_keys(h):
            for other in self.buckets.get(key, ()):
                if _hamming64(h, other) <= self.near_threshold:
                    return True, 'near', fp_key, h
        return False, None, fp_key, h

    def find(self, text, fp=None):
        """Read-only duplicate lookup (never adds the document or counters).

        Use this to test a document against an earlier split's index without
        letting the document itself become part of that split's reference
        set.  ``is_duplicate`` remains the mutating form used when building a
        split."""
        dup, kind, _fp_key, _h = self._match(text, fp)
        return dup, kind

    def is_duplicate(self, text, fp=None):
        dup, kind, fp_key, h = self._match(text, fp)
        if dup:
            if kind == 'exact':
                self.n_exact_skipped += 1
            else:
                self.n_near_skipped += 1
                # Keep the skipped member in the near-duplicate index too.
                # Otherwise a chain A~T (A skipped), B~A (B not close
                # enough to T to be found separately) would let B into a
                # later split: the same document cluster would cross the
                # boundary through the skipped copy.  Quarantining the
                # hash makes the group filter closed under the LSH matches
                # it has already made, without counting the copy as seen.
                for _key in self._bucket_keys(h):
                    self.buckets.setdefault(_key, []).append(h)
            return True, kind
        self.exact.add(fp_key)
        self.n_seen += 1
        for key in self._bucket_keys(h):
            self.buckets.setdefault(key, []).append(h)
        return False, None

    def protocol(self):
        """The complete cleaning geometry, suitable for cache metadata."""
        return {
            'schema': DEDUP_PROTOCOL_SCHEMA,
            'exact_hash': 'blake2b-128',
            'simhash_bits': 64,
            'shingle_unit': 'word',
            'shingle_words': int(self.shingle_words),
            'max_shingles': int(self.max_shingles),
            'bands': int(self.bands),
            'near_hamming_threshold': int(self.near_threshold),
            'scope': self.scope,
        }

    def protocol_fp(self):
        return dedup_protocol_fp(self.protocol())

    def stats(self):
        protocol = self.protocol()
        return {'policy': 'exact_content_hash+simhash_lsh',
                'simhash_bits': 64,
                'near_hamming_threshold': self.near_threshold,
                'shingle_words': int(self.shingle_words),
                'max_shingles': int(self.max_shingles),
                'bands': int(self.bands),
                'protocol': protocol,
                'protocol_fp': dedup_protocol_fp(protocol),
                'added_documents': int(self.n_seen),
                'exact_duplicates_skipped': int(self.n_exact_skipped),
                'near_duplicates_skipped': int(self.n_near_skipped)}


def _filter_test_documents(train_rows, val_index, test_rows, *, label='test',
                          protocol=None):
    """Remove test documents that duplicate an earlier split or each other.

    ``train_rows`` are the raw training documents and ``val_index`` is the
    already-built validation index.  Both are consulted READ-ONLY: a test
    document must not enter the training/validation reference set, otherwise a
    later test document could be labelled as a train/val duplicate merely
    because it resembles an earlier *test* document.  Test-to-test duplicates
    are closed through a separate self index, which is also updated for test
    documents that are excluded on an earlier-split match, so transitive
    near-duplicate chains stay together.

    ``protocol`` is the dedup protocol recorded by the paired train cache.
    Rebuilding the test index from that object (instead of the current code
    defaults) makes the held-out cleanup use the same shingle/band geometry as
    the training split.

    Returns ``(kept_rows, excluded_counts, test_index)``.  ``excluded_counts``
    keys are ``train_exact``, ``train_near``, ``validation_exact``,
    ``validation_near``, ``test_exact`` and ``test_near``.  The set of kept
    documents is the same as the historical mutating form; only the
    attribution of a self/train/validation match is made honest.
    """
    _kwargs = dedup_kwargs_from_protocol(protocol) if protocol is not None else {}
    train_index = _DocDeduplicator(**_kwargs)
    for _r in train_rows:
        if isinstance(_r, str) and _r.strip():
            train_index.is_duplicate(_r)
    test_index = _DocDeduplicator(**_kwargs)
    kept = []
    excluded = {'train_exact': 0, 'train_near': 0,
                'validation_exact': 0, 'validation_near': 0,
                'test_exact': 0, 'test_near': 0}
    for _r in test_rows:
        if not isinstance(_r, str) or not _r.strip():
            continue
        # Index every test document (even one excluded below) so a later
        # near copy cannot slip through a transitive chain.
        _self_dup, _self_kind = test_index.is_duplicate(_r)
        _dup, _kind = train_index.find(_r)
        if _dup:
            excluded[f'train_{_kind}'] += 1
            continue
        _dup, _kind = val_index.find(_r)
        if _dup:
            excluded[f'validation_{_kind}'] += 1
            continue
        if _self_dup:
            excluded[f'test_{_self_kind}'] += 1
            continue
        kept.append(_r)
    print(f'[data] {label} duplicate cleanup: excluded '
          f'{excluded["train_exact"]} exact and {excluded["train_near"]} '
          f'near-duplicate documents from train, '
          f'{excluded["validation_exact"]} exact and '
          f'{excluded["validation_near"]} near-duplicate documents from '
          f'validation, and {excluded["test_exact"]} exact / '
          f'{excluded["test_near"]} near-duplicate {label} documents')
    return kept, excluded, test_index


TOKENIZER_FIT_TOKENS = 8_000_000


# ---- Pinned upstream corpus revisions --------------------------------------
# A token-content fingerprint binds a run to the bytes it actually saw, but it
# cannot tell a *fresh build* that the upstream Hub dataset changed under the
# same configuration.  Two builds of "WikiText-103-raw" from different Hub
# revisions would then be compared as if they were the same corpus.  Pin the
# upstream commit (the exact value already backing the frozen wt103_cache) and
# record it in every cache metadata file; a mismatch is a hard refusal, not a
# silent rebuild.
#
# The environment overrides exist for mirrors/offline copies of the same
# immutable commit.  They accept a full 40-hex commit only: a branch name or a
# short prefix can move and is not a reproducible data identity.
WIKITEXT_HF_REPO = 'Salesforce/wikitext'
WIKITEXT_HF_REPO_ALIAS = 'wikitext'  # mirror alias of the same repository
WIKITEXT_HF_CONFIG = 'wikitext-103-raw-v1'
WIKITEXT_HF_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
FINEWEB_EDU_HF_REPO = 'HuggingFaceFW/fineweb-edu'
FINEWEB_EDU_HF_REVISION = '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
_HF_REVISION_RE = re.compile(r'^[0-9a-f]{40}$')


def _pinned_hf_revision(env_name, default, label):
    """Resolve a full-commit data revision, refusing moving references."""
    raw = os.environ.get(env_name)
    rev = (raw if raw not in (None, '') else default)
    rev = str(rev).strip().lower()
    if not _HF_REVISION_RE.match(rev):
        raise ValueError(
            f'{env_name} must be a full 40-hex commit SHA (got {rev!r}); a '
            f'branch/tag/short prefix can move and cannot pin the {label} '
            'corpus for reproducible builds')
    return rev


def wikitext_dataset_source(repo=None):
    """Machine-readable identity of the WikiText-103 upstream source."""
    return {
        'provider': 'huggingface',
        'repo': str(repo or WIKITEXT_HF_REPO),
        'config': WIKITEXT_HF_CONFIG,
        'revision': _pinned_hf_revision(
            'WT103_DATASET_REVISION', WIKITEXT_HF_REVISION, 'WikiText-103'),
        'revision_pinned': True,
    }


def fineweb_edu_dataset_source(source='sample-10BT'):
    """Machine-readable identity of the FineWeb-Edu upstream source."""
    return {
        'provider': 'huggingface',
        'repo': FINEWEB_EDU_HF_REPO,
        'config': str(source),
        'revision': _pinned_hf_revision(
            'FWEDU_DATASET_REVISION', FINEWEB_EDU_HF_REVISION, 'FineWeb-Edu'),
        'revision_pinned': True,
    }


def local_dataset_source(label, paths=None):
    """Identity of a local raw/jsonl stand-in (no movable upstream revision)."""
    rec = {
        'provider': 'local',
        'label': str(label),
        'revision': None,
        'revision_pinned': False,
    }
    if paths:
        rec['paths'] = sorted({os.path.abspath(os.fspath(p)) for p in paths})
    return rec


def fineweb_cache_source(source='sample-10BT'):
    """Source record actually used by the FineWeb-Edu loader."""
    env_value = os.environ.get('FWEDU_LOCAL_JSONL', '')
    if env_value.strip():
        return local_dataset_source('fineweb_local_jsonl',
                                    _fineweb_local_files(env_value))
    return fineweb_edu_dataset_source(source)


def _dataset_source_problems(meta, expected, label):
    """Return ``(problems, warnings)`` for a cache's dataset provenance.

    A missing record is a legacy-cache warning (old caches must stay readable),
    while a present-but-wrong record is a hard problem.  A genuine local raw
    dump is not an error: its content is bound by the token fingerprints, and
    the warning says the upstream revision could not be attested.
    """
    problems, warnings = [], []
    if not isinstance(meta, dict):
        return [f'{label}: cache metadata is not an object'], []
    src = meta.get('dataset_source')
    if src is None:
        warnings.append(
            f'{label}: cache predates the pinned dataset-source record; the '
            'upstream commit cannot be attested locally (rebuild to record it)')
        return problems, warnings
    if not isinstance(src, dict):
        return [f'{label}: dataset_source is a {type(src).__name__}, not an object'], []
    provider = src.get('provider')
    if provider not in ('huggingface', 'local'):
        problems.append(
            f'{label}: dataset_source.provider={provider!r} is not one of '
            '"huggingface"/"local"')
        return problems, warnings
    if provider == 'huggingface':
        rev = src.get('revision')
        if not isinstance(rev, str) or not _HF_REVISION_RE.match(rev.strip().lower()):
            problems.append(
                f'{label}: dataset_source.revision={rev!r} is not a full 40-hex '
                'commit; the corpus cannot be pinned')
        if src.get('revision_pinned') is not True:
            problems.append(
                f'{label}: dataset_source claims the Hub but does not attest '
                'revision_pinned=true')
        if expected is not None and expected.get('provider') == 'huggingface':
            want = expected.get('revision')
            if isinstance(rev, str) and want and rev.strip().lower() != want:
                problems.append(
                    f'{label}: cached corpus revision {rev[:12]}… differs from '
                    f'the pinned {want[:12]}… — rebuild the cache from the '
                    'pinned commit (or set the documented WT103_DATASET_REVISION/'
                    'FWEDU_DATASET_REVISION override deliberately)')
    else:
        warnings.append(
            f'{label}: cache was built from a local raw dump; content is bound '
            'by the token fingerprints, but the upstream commit is not attested')
    return problems, warnings


def dataset_source_pair_problems(test_meta, train_metas, label='test'):
    """Check that a test cache and its paired train caches share one upstream source."""
    problems, warnings = [], []
    ts = test_meta.get('dataset_source') if isinstance(test_meta, dict) else None
    if not isinstance(ts, dict):
        warnings.append(
            f'{label}: test cache has no dataset_source record; the upstream '
            'revision cannot be compared with the train caches')
        return problems, warnings
    trev = ts.get('revision') if ts.get('provider') == 'huggingface' else None
    seen = set()
    for _m in train_metas or []:
        if not isinstance(_m, dict):
            continue
        s = _m.get('dataset_source')
        if not isinstance(s, dict):
            warnings.append(
                f'{label}: a paired train cache has no dataset_source record; '
                'the upstream revision cannot be compared')
            continue
        if s.get('provider') == 'huggingface' and s.get('revision'):
            seen.add(str(s['revision']).strip().lower())
    if trev and seen and seen != {str(trev).strip().lower()}:
        problems.append(
            f'{label}: test corpus revision {str(trev)[:12]}… does not match the '
            f'paired train revision(s) {sorted(r[:12] for r in seen)} — the '
            'split cannot be described as one upstream snapshot')
    return problems, warnings


_DATA_SOURCE_WARNED = set()


def _warn_dataset_source(label, warnings):
    for w in warnings or []:
        if w in _DATA_SOURCE_WARNED:
            continue
        _DATA_SOURCE_WARNED.add(w)
        print(f'[data] NOTE: {w}')



def _wikitext_fit_rows(rows, vocab_size, token_cap, max_rows=50000):
    """Select a deterministic train-prefix fit corpus within ``token_cap``.

    The candidate prefix is measured with the tokenizer it is used to build;
    rows (or, for a single oversized row, the row text) beyond the cap are
    shortened and the tokenizer is rebuilt until the final fit text itself
    fits inside the cap.  The result is a prefix of the train split, so it can
    never include validation/test text.
    """
    total = min(len(rows), int(max_rows))
    if total <= 0 or int(token_cap) <= 0:
        return [], None, 0
    fit_text = list(rows[:total])
    tok = None
    n_tok = 0
    # BPE training + a full encode are linear in the fit corpus.  The first
    # measurement already yields the token/row (or token/char) ratio; a
    # conservative prefix from that ratio converges in one rebuild for
    # realistic text, and a small pass cap guards against a pathological
    # vocabulary.  Never return a fit corpus that exceeds the advertised
    # token cap.
    for _pass in range(8):
        tok = build_tokenizer(fit_text, vocab_size)
        n_tok = int(sum(len(enc.ids) + 1 for enc in tok.encode_batch(fit_text))) if fit_text else 0
        if n_tok <= int(token_cap):
            return fit_text, tok, n_tok
        ratio = float(token_cap) / max(n_tok, 1) * (0.9 if _pass else 0.95)
        if len(fit_text) > 1:
            keep = max(1, int(len(fit_text) * ratio))
            fit_text = fit_text[:min(len(fit_text) - 1, keep)]
        else:
            text = str(fit_text[0])
            new_len = max(1, int(len(text) * ratio))
            if new_len >= len(text):
                new_len = len(text) - 1
            if new_len <= 0:
                break
            fit_text = [text[:new_len]]
    raise RuntimeError(
        f'_wikitext_fit_rows: could not shrink the tokenizer-fit corpus to the '
        f'{int(token_cap)}-token cap (last measured {int(n_tok)} tokens)')

def _fineweb_local_rows(env_value):
    """Offline FineWeb-Edu stand-in: FWEDU_LOCAL_JSONL lists files or
    directories (os.pathsep-separated; directories expand to their sorted
    *.jsonl / *.json / *.txt).  Every non-empty line is a document — either a
    JSON object with a "text" field (the FineWeb-Edu row shape) or plain text.
    Pure availability path for air-gapped kernel-dev machines; the corpus a
    panel trains on is still bound by the content fingerprint, so a local
    stand-in can never be confused with a genuine FineWeb-Edu cache."""
    files = _fineweb_local_files(env_value)
    if not files:
        raise RuntimeError(f'FWEDU_LOCAL_JSONL is set ({env_value!r}) but resolves to no readable file — point it at FineWeb-Edu style jsonl/txt files, or unset it to stream from the hub')
    print(f'[data] local FineWeb-Edu stand-in corpus: {len(files)} file(s) ({files[0]}{" ..." if len(files) > 1 else ""})')
    for path in files:
        with open(path, encoding='utf-8') as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except ValueError:
                    yield ln
                    continue
                text = obj.get('text') if isinstance(obj, dict) else (obj if isinstance(obj, str) else None)
                if isinstance(text, str) and text.strip():
                    yield text
                elif not isinstance(obj, dict):
                    yield ln

def _fineweb_rows(source):
    """Ordered, non-empty FineWeb-Edu documents (streaming; deterministic row
    order per dataset revision).  The token cache plus its content fingerprint
    binds a panel to the corpus.  The Hub commit is also pinned
    (`fineweb_edu_dataset_source`) and passed to `load_dataset`, so a fresh
    build cannot silently stream a newer snapshot than the one recorded in the
    cache metadata.  A src_fp mismatch would still drop stale records, but the
    pin makes the change visible *before* the cache is rebuilt."""
    env_value = os.environ.get('FWEDU_LOCAL_JSONL', '')
    if env_value.strip():
        yield from _fineweb_local_rows(env_value)
        return
    from datasets import load_dataset
    _rev = fineweb_edu_dataset_source(source)['revision']
    ds = load_dataset(FINEWEB_EDU_HF_REPO, name=source, split='train',
                      streaming=True, revision=_rev)
    n_docs = 0
    for rec in ds:
        text = rec.get('text') if isinstance(rec, dict) else None
        if text is None or not str(text).strip():
            continue
        n_docs += 1
        if n_docs % 50000 == 0:
            print(f'    [fineweb] {n_docs} docs streamed')
        yield str(text)

def _fineweb_docs(source):
    """Ordered ``(doc_id, text)`` pairs.

    ``doc_id`` is a stable content fingerprint (the only identifier available
    for both the hub and the offline JSONL stand-in), so train/validation
    metadata can prove document-level disjointness instead of merely claiming
    that stream positions were different.
    """
    for text in _fineweb_rows(source):
        txt = str(text)
        yield _doc_exact_fp(txt), txt
