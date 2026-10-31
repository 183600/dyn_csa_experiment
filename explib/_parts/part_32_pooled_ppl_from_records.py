
def _pooled_nll_ratio(nlls, nts):
    """``sum(nll) / sum(tokens)`` without overflowing either total.

    ``nll_sum`` values are finite floats and token counts are exact integers,
    so at scale-up sizes the raw NLL sum can overflow and an int total beyond
    float64 cannot even be converted.  The ratio itself is still representable
    in that regime: write ``nll_i = max_nll * a_i`` and
    ``tokens_i = max_tok * w_i``, then

        sum(nll) / sum(tokens) = (max_nll / max_tok)
                                 * sum(a_i) / sum(w_i).

    The two unitless sums are O(n) even when the absolute totals overflow.
    Ordinary panels never call this path (the caller keeps the historical
    float totals).
    """
    if not nlls or len(nlls) != len(nts):
        return None
    try:
        max_nll = max(abs(float(v)) for v in nlls)
        max_tok = max(int(v) for v in nts)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(max_nll) or max_tok <= 0:
        return None
    if max_nll == 0.0:
        return 0.0
    try:
        # int/int yields a float ratio in [0, 1] even when the operands are
        # far beyond float64's exponent range.
        weights = [int(v) / max_tok for v in nts]
        total_w = math.fsum(weights)
        if not (math.isfinite(total_w) and total_w > 0.0):
            return None
        total_a = math.fsum(float(v) / max_nll for v in nlls)
        if not math.isfinite(total_a):
            return None
        # numerator = max_nll * sum(a_i), denominator = max_tok * sum(w_i).
        # Build the scale exactly first: ``max_nll / max_tok`` would raise
        # OverflowError when the token total is beyond float64 even though
        # the (possibly tiny) quotient is representable.
        from fractions import Fraction as _Fraction
        try:
            scale = float(_Fraction(max_nll) / max_tok)
        except OverflowError:
            return None
        ratio = scale * (total_a / total_w)
    except (OverflowError, ValueError, ZeroDivisionError):
        return None
    return ratio if math.isfinite(ratio) else None


def _nll_per_token_ratio(nll, n_tokens):
    """One finite NLL total divided by one exact token count.

    A token count above float64's exponent range cannot be converted to float
    by the ordinary division operator, even when the quotient is tiny and
    representable.  Fall back to the scaled helper only on that path so
    ordinary records keep their historical division.
    """
    try:
        nt = int(n_tokens)
    except (TypeError, ValueError, OverflowError):
        return None
    if nt <= 0:
        return None
    try:
        return float(nll / nt)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return _pooled_nll_ratio([nll], [nt])


def _pooled_ppl_from_records(recs, *, estimand=None):
    """Token-pooled PPL: exp(sum NLL / sum N tokens) over all seeds.

    This is the desired estimator; the arithmetic mean of per-seed
    PPLs is kept as `ppl_mean_arithmetic` for backward compatibility but is no
    longer the primary aggregate when token counts exist.  Records that predate
    `nll_sum`/`n_tokens` cannot be pooled and fall back explicitly.

    ``estimand`` defaults to each record's own declaration.  A group that mixes
    nested-holdout and full-validation records has no single estimand: pooling
    would sum NLLs from two different samples and then label the result as one
    of them, so it returns ``(None, None, None)``.  A caller that deliberately
    falls the whole group back to full validation passes ``estimand=False``.
    """
    recs = list(recs)
    declared = {r.get('ppl_estimation_rows') == 'nested_validation_holdout'
                for r in recs if isinstance(r, dict)}
    if estimand is None:
        if len(declared) > 1:
            return None, None, None
        estimand = next(iter(declared)) if declared else False
    nlls, nts = [], []
    # Fallback components retain per-row values where a record only carries
    # row arrays, so an overflowing record-level sum does not erase the
    # representable pooled ratio.
    components = []
    for r in recs:
        if not isinstance(r, dict):
            # The declared-estimand scan above skips non-objects, so a list
            # containing one used to reach ``r.get`` and abort with
            # AttributeError.  A record set with a non-record member has no
            # well-defined pooled estimator.
            return None, None, None
        # When the primary estimate is the nested holdout, the matching
        # accounting is nll_sum_holdout/n_tokens_holdout.  A missing halfword
        # cannot fall back to the full-validation totals: the pooled value and
        # the record's primary PPL would then describe different samples.
        _from_rows = None
        if bool(estimand) and r.get('ppl_estimation_rows') == 'nested_validation_holdout':
            nll = _finite_scalar(r.get('nll_sum_holdout'))
            nt = _exact_count(r.get('n_tokens_holdout'), minimum=1)
            if nll is None or nt is None:
                # Some writers persist only the holdout per-row arrays.  Their
                # totals describe the same sample and can recover the missing
                # accounting; the full-validation totals never can.
                _rn = r.get('nll_per_row_holdout')
                _rt = r.get('n_tokens_per_row_holdout')
                if (isinstance(_rn, list) and isinstance(_rt, list)
                        and _rn and len(_rn) == len(_rt)):
                    _vals = [_finite_scalar(v) for v in _rn]
                    _toks = [_exact_count(v, minimum=1) for v in _rt]
                    if (all(v is not None for v in _vals)
                            and all(v is not None for v in _toks)):
                        nll = float(sum(_vals))
                        nt = int(sum(_toks))
                        if not math.isfinite(nll):
                            # Keep the individual rows only when the record
                            # total itself overflowed; otherwise the totals
                            # are enough for any later aggregate fallback and
                            # avoid a second O(rows) copy.
                            _from_rows = list(zip(_vals, _toks))
        else:
            nll = _finite_scalar(r.get('nll_sum'))
            nt = _exact_count(r.get('n_tokens'), minimum=1)
        if nll is None or nt is None:
            return None, None, None
        nlls.append(nll)
        nts.append(nt)
        if _from_rows is None:
            components.append((nll, nt))
        else:
            components.extend(_from_rows)
    # Keep the token total as an exact Python integer.  Converting it through
    # float first both rounds sums above 2**53 and raises OverflowError for a
    # scale-up count beyond float64.
    exact_total_t = int(sum(nts))
    if exact_total_t <= 0:
        return None, None, None
    try:
        total_t = float(exact_total_t)
    except OverflowError:
        total_t = None
    try:
        total_n = float(sum(nlls))
    except OverflowError:
        total_n = None
    if (total_n is None or not math.isfinite(total_n)
            or total_t is None or not math.isfinite(total_t)):
        # The true pooled ratio can still be representable when one of the
        # raw totals is not (e.g. three records at 1e308 NLL and 1e308
        # tokens).  The NLL total is unavailable in that case, but the pooled
        # PPL and the exact token total are still reportable.  Row-array
        # records contribute their individual rows here, so a record sum that
        # overflowed does not hide a representable pooled ratio.
        ratio = _pooled_nll_ratio([c[0] for c in components],
                                  [c[1] for c in components])
        total_n_out = None
    else:
        ratio = total_n / total_t
        total_n_out = total_n
    if ratio is None or not math.isfinite(ratio):
        return None, None, None
    _pooled = _safe_exp_ppl(ratio)
    if _pooled is None or not math.isfinite(_pooled) or _pooled < 1.0:
        return None, None, None
    return _pooled, total_n_out, exact_total_t

def _record_primary_ppl(rec):
    """The per-seed PPL on the same rows as the record's primary aggregate.

    `_pooled_ppl_from_records` switches to the nested holdout accounting when
    `ppl_estimation_rows` says the estimate is a held-out-row one.  Any
    per-seed contrast must use the same number; reading `ppl` (full
    validation) next to a pooled holdout aggregate would mix two samples.
    Legacy records without a holdout marker keep `ppl`.

    A record that DECLARES the nested estimand but lacks every holdout
    accounting field is unverifiable and returns None.  Falling back to the
    full-validation `ppl` here would silently restore the selection rows that
    the nested split exists to remove.
    """
    if not isinstance(rec, dict):
        return None
    if rec.get('ppl_estimation_rows') == 'nested_validation_holdout':
        _h = _finite_scalar(rec.get('ppl_holdout'))
        if _h is not None:
            return _h if ppl_is_usable(_h) else None
        _nll = _finite_scalar(rec.get('nll_sum_holdout'))
        _nt = _exact_count(rec.get('n_tokens_holdout'), minimum=1)
        if _nll is not None and _nt is not None:
            _ratio = _nll_per_token_ratio(_nll, _nt)
            _v = _safe_exp_ppl(_ratio) if _ratio is not None else None
            return _v if ppl_is_usable(_v) else None
        _rn = rec.get('nll_per_row_holdout')
        _rt = rec.get('n_tokens_per_row_holdout')
        if (isinstance(_rn, list) and isinstance(_rt, list)
                and _rn and len(_rn) == len(_rt)):
            _vals = [_finite_scalar(v) for v in _rn]
            _toks = [_exact_count(v, minimum=1) for v in _rt]
            if (all(v is not None for v in _vals)
                    and all(v is not None for v in _toks)):
                _tot_t = int(sum(_toks))
                if _tot_t > 0:
                    _ratio = _nll_per_token_ratio(sum(_vals), _tot_t)
                    if _ratio is None or not math.isfinite(_ratio):
                        # The row-level sum overflowed although the per-row
                        # values and their weighted ratio are finite.
                        _ratio = _pooled_nll_ratio(_vals, _toks)
                    _v = (_safe_exp_ppl(_ratio)
                          if _ratio is not None else None)
                    return _v if ppl_is_usable(_v) else None
        return None
    _v = _finite_scalar(rec.get('ppl'))
    return _v if ppl_is_usable(_v) else None


def _paired_primary_delta(rec_a, rec_b):
    """(a − b) on the PRIMARY estimand, or None when either side is unmeasured.

    Every paired display/test in this module must use this helper rather than
    raw ``ppl``: a nested record's ``ppl`` is the full-validation value, which
    includes the rows that selected its hyper-parameters.

    The two records must also declare the SAME estimand.  Subtracting a
    full-validation PPL from a held-out-row PPL produces a number on neither
    sample, even though both individual values are finite.
    """
    _a_nested = rec_a.get('ppl_estimation_rows') == 'nested_validation_holdout'
    _b_nested = rec_b.get('ppl_estimation_rows') == 'nested_validation_holdout'
    if _a_nested != _b_nested:
        return None
    _a, _b = _record_primary_ppl(rec_a), _record_primary_ppl(rec_b)
    if _a is None or _b is None:
        return None
    return float(_a) - float(_b)


def _scaled_std_or_raw(values):
    """Sample SD (ddof=1), or ``None`` when the true SD exceeds float64.

    A finite panel can have a sample SD beyond the representable range even
    though each measurement is finite (e.g. ``[1.7e308, -1.7e308]``).  The
    historical return of ``inf`` is not a measured spread and is not strict-
    JSON safe, so callers get the same "unavailable" value they already use
    for a missing SD.
    """
    try:
        if len(values) < 2:
            return 0.0
    except TypeError:
        return 0.0
    _ms = _scaled_mean_std_if_extreme(values, ddof=1)
    if _ms is not None:
        sd = float(_ms[1])
    else:
        sd = float(np.std(values, ddof=1))
    return sd if math.isfinite(sd) else None


def _scaled_mean_or_raw(values):
    """Arithmetic mean that survives extreme-but-finite PPL magnitudes.

    ``np.mean`` sums the panel first, so a finite vector whose sum overflows
    reports ``inf`` even when the mean itself is representable (e.g. three
    seeds at ``1e308``).  Reuse the guarded mean/SD helper, which scales by the
    largest magnitude only for an extreme panel and therefore leaves ordinary
    panels bit-identical to the historical ``np.mean``.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float(np.mean(arr))
    _ms = _scaled_mean_std_if_extreme(arr, ddof=1)
    if _ms is not None:
        return float(_ms[0])
    return float(np.mean(arr))


def _scaled_median_or_raw(values):
    """Median that survives extreme-but-finite magnitudes.

    NumPy's even-count median computes ``(a + b) / 2``, which overflows for two
    finite PPLs near the float limit even though the median is representable.
    The ordinary path keeps ``np.median`` exactly (``np.percentile(..., 50)``
    can differ in the last bit), so persisted non-extreme aggregates do not
    change.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float(np.median(arr))
    scale = float(np.max(np.abs(arr)))
    if (math.isfinite(scale) and scale > 0.0
            and (scale < 1e-100 or scale > 1e100)):
        med = float(np.median(arr / scale)) * scale
        if math.isfinite(med):
            return med
    return float(np.median(arr))


def _scaled_percentile_or_raw(values, qs):
    """Percentiles with a scaled interpolation path for extreme magnitudes.

    ``np.percentile`` interpolates as ``a + (b - a) * t``; near the float limit
    that subtraction can overflow although every input and the requested
    quantile are finite.  Returns the requested quantiles in input order.
    """
    arr = np.asarray(values, dtype=float)
    _qs = tuple(float(q) for q in qs)
    if arr.size == 0:
        return tuple(float('nan') for _q in _qs)
    scale = float(np.max(np.abs(arr)))
    if (math.isfinite(scale) and scale > 0.0
            and (scale < 1e-100 or scale > 1e100)):
        with np.errstate(over='ignore', invalid='ignore'):
            out = np.asarray(np.percentile(arr / scale, _qs), dtype=float) * scale
        vals = tuple(float(v) for v in out.reshape(-1))
        if all(math.isfinite(v) for v in vals):
            return vals
    with np.errstate(over='ignore', invalid='ignore'):
        out = np.percentile(arr, _qs)
    return tuple(float(v) for v in np.asarray(out, dtype=float).reshape(-1))


def _scaled_std_or_none(values):
    """Sample SD (ddof=1) or None when fewer than two values are available."""
    try:
        if len(values) < 2:
            return None
    except TypeError:
        return None
    return _scaled_std_or_raw(values)


def _seed_t_ci(ppls, alpha=0.05):
    alpha = _validated_alpha(alpha, context='_seed_t_ci alpha')
    n = len(ppls)
    if n < 2:
        return None
    # PPL is exp(CE): a finite but very large value (or a tiny positive
    # difference) can overflow/underflow the squared deviations in np.std even
    # though the interval endpoints are representable.  Use the same guarded
    # mean/SD helper as the paired-effect statistics.
    _ms = _scaled_mean_std_if_extreme(ppls, ddof=1)
    if _ms is not None:
        mu, sd = _ms
    else:
        mu = float(np.mean(ppls))
        sd = float(np.std(ppls, ddof=1))
    try:
        tc = _t_critical(1.0 - alpha, n - 1)
    except Exception:
        return None
    half = tc * sd / math.sqrt(n)
    lo, hi = mu - half, mu + half
    if not (math.isfinite(lo) and math.isfinite(hi)):
        # A true endpoint beyond float64 is not a measured bound; the
        # interval is unavailable rather than an infinite-width JSON null.
        return None
    return [lo, hi]

def _primary_row_arrays(rec):
    """Per-row NLL/token arrays for the record's PRIMARY estimand.

    A nested record's reported PPL is the held-out-row estimate, so its
    intervals must bootstrap `nll_per_row_holdout`/`n_tokens_per_row_holdout`.
    Returning None when those are absent keeps a missing holdout accounting
    from silently falling back to the full-validation rows.
    """
    if not isinstance(rec, dict):
        return None
    if rec.get('ppl_estimation_rows') == 'nested_validation_holdout':
        nll = rec.get('nll_per_row_holdout')
        ntok = rec.get('n_tokens_per_row_holdout')
        if nll is None or ntok is None:
            return None
        return nll, ntok
    return rec.get('nll_per_row'), rec.get('n_tokens_per_row')


def _token_matrix(tok):
    """Per-row token counts as a float matrix, or None when unrepresentable.

    The token accounting is deliberately exact (arbitrary-precision Python
    ints), but the bootstrap arithmetic is floating point.  Narrowing to
    ``np.int64`` raised OverflowError above 2**63 and could also overflow on a
    row sum; building the float matrix directly is the same arithmetic for
    ordinary counts and makes an unrepresentable count an unavailable interval
    instead of an exception.
    """
    try:
        ints = [int(v) for v in tok]
    except (TypeError, ValueError, OverflowError):
        ints = None
    if ints is not None:
        _max = np.iinfo(np.int64).max
        if (all(0 <= v <= _max for v in ints)
                and int(sum(ints)) <= int(_max)):
            # Keep the historical exact-int64 path whenever the matrix and
            # every row sum are representable; only fall back to float for the
            # scale cases that used to overflow.
            return np.asarray(ints, dtype=np.int64)
    try:
        arr = np.asarray(tok, dtype=float)
    except (OverflowError, ValueError, TypeError):
        return None
    if not np.isfinite(arr).all() or (arr < 0).any():
        return None
    return arr


def _doc_bootstrap_ppl_ci(recs, *, n_boot=2000, seed=20261018):
    """Row-level bootstrap CI for the pooled PPL (legacy name kept).

    The historical name says "document", but the records it consumes expose
    only per-ROW NLL.  A fixed eval row is a window that can split or merge
    documents, so resampling rows preserves within-row correlation but is not
    a document bootstrap.  Use ``eval_length_gen(doc_ids=...)`` (which records
    `nll_per_doc` / `n_tokens_per_doc`) when the document-level estimand is
    required.  If any record lacks per-row NLL, no CI is invented — the caller
    must see the missing-evidence state rather than a narrow interval computed
    from the
    wrong unit.
    """
    # Draws are held in one float vector; an untrusted n_boot must not turn
    # this readout into an unbounded allocation.
    _n_draws = _as_seed_int(n_boot)
    if (_n_draws is None or _n_draws < 2
            or _n_draws > BOOTSTRAP_HARD_MAX_DRAWS):
        return None
    recs = list(recs)
    _nested_flags = {r.get('ppl_estimation_rows') == 'nested_validation_holdout'
                     for r in recs if isinstance(r, dict)}
    if len(_nested_flags) > 1:
        # Bootstrap rows from two different evaluation samples are not one
        # document panel, even when their lengths happen to match.
        return None
    rows = []
    for r in recs:
        _arrays = _primary_row_arrays(r)
        if _arrays is None:
            return None
        nll, ntok = _arrays
        if not isinstance(nll, list) or not isinstance(ntok, list) or len(nll) != len(ntok) or not nll:
            return None
        _tok = [_exact_count(v) for v in ntok]
        if any(v is None for v in _tok):
            return None
        _nll = np.asarray(nll, dtype=float)
        if not np.isfinite(_nll).all():
            return None
        _tok_arr = _token_matrix(_tok)
        if _tok_arr is None:
            return None
        rows.append((_nll, _tok_arr))
    if len({len(a) for a, _ in rows}) != 1:
        return None
    R = len(rows[0][0])
    if R == 0:
        return None
    nll_mat = np.stack([a for a, _ in rows], axis=0)          # [S, R]
    ntok_mat = np.stack([b for _, b in rows], axis=0)         # [S, R]
    rng = np.random.default_rng(int(seed))
    nll_col = nll_mat.sum(axis=0)
    ntok_col = ntok_mat.sum(axis=0)
    vals = np.empty(_n_draws, dtype=float)
    # Multinomial bootstrap counts: a single [chunk, R] count matrix is reused
    # across draws.  Size the chunk so a very large document panel does not
    # allocate 256 * R counts at once.
    _chunk = _bootstrap_draw_chunk(R)
    _b = 0
    while _b < _n_draws:
        _nb = min(_chunk, _n_draws - _b)
        counts = rng.multinomial(R, np.full(R, 1.0 / R, dtype=float), size=_nb)
        nt = counts @ ntok_col
        nn = counts @ nll_col
        ok = nt > 0
        out = np.full(_nb, np.nan, dtype=float)
        with np.errstate(over='ignore', invalid='ignore'):
            out[ok] = np.exp(nn[ok] / nt[ok])
        vals[_b:_b + _nb] = out
        _b += _nb
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size < max(20, int(0.5 * _n_draws)):
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]

def _combined_seed_doc_ci(recs, *, n_boot=2000, seed=20261018, conf=0.95):
    """Two-level (seed x document) uncertainty disclosure for the pooled PPL.

    The seed-level interval (`ppl_ci95_seed_t`) answers "how much does
    TRAINING noise move the number"; the document bootstrap
    (`ppl_ci95_doc_bootstrap`) answers "how much does the EVALUATION SUBSET
    move it".  Neither alone bounds the reported pooled PPL, because both
    sources act at once — a small seed CI next to a wide doc CI is not a
    precise measurement.  This helper estimates both components on the
    log-PPL scale (PPL is multiplicative; the interval is exponentiated back
    at the end) and combines them as independent variance sources:

      se_seed  = std_s(log pooled_ppl_s, ddof=1) / sqrt(S)
      se_doc   = bootstrap SD of the log pooled PPL over document resamples
      se_combined = sqrt(se_seed^2 + se_doc^2)
      ci95     = exp(mean_s log pooled_ppl_s  +-  t_{S-1, conf} * se_combined)

    The component ratio is reported too: it says whether precision is bought
    by more seeds or by more evaluation documents, which is the actionable
    question a small-n panel faces.  This is a disclosed two-source
    approximation, not a fitted hierarchical random-effects model; when a
    component is unmeasurable (S<2, records without per-row NLL, or fewer
    than two scored documents) the status says so explicitly and the missing
    component is never silently treated as zero."""
    recs = list(recs)
    _nested_flags = {r.get('ppl_estimation_rows') == 'nested_validation_holdout'
                     for r in recs if isinstance(r, dict)}
    rows = []
    _mixed_estimands = len(_nested_flags) > 1
    _zero_token_seed = False
    for r in recs:
        _arrays = _primary_row_arrays(r)
        if _arrays is None:
            rows = None
            break
        nll, ntok = _arrays
        if not isinstance(nll, list) or not isinstance(ntok, list) or len(nll) != len(ntok) or not nll:
            rows = None
            break
        _tok = [_exact_count(v) for v in ntok]
        _nll = np.asarray(nll, dtype=float)
        if any(v is None for v in _tok) or not np.isfinite(_nll).all():
            rows = None
            break
        _tok_arr = _token_matrix(_tok)
        if _tok_arr is None:
            rows = None
            break
        rows.append((_nll, _tok_arr))
    if isinstance(rows, list):
        if len({len(a) for a, _ in rows}) != 1:
            rows = None
    if isinstance(rows, list):
        nll_mat = np.stack([a for a, _ in rows], axis=0)          # [S, R]
        ntok_mat = np.stack([b for _, b in rows], axis=0)          # [S, R]
        keep = ntok_mat.sum(axis=0) > 0
        n_dropped = int((~keep).sum())
        nll_mat = nll_mat[:, keep]
        ntok_mat = ntok_mat[:, keep]
        if not np.isfinite(nll_mat).all() or not np.isfinite(ntok_mat).all() or (ntok_mat < 0).any():
            rows = None
        elif (ntok_mat.sum(axis=1) <= 0).any():
            # A record with no scored token anywhere is not a PPL measurement
            # (0 tokens would otherwise divide by 1e-30 and enter the seed
            # component as a fake CE=0 seed).  Its NLLs also cannot be pooled
            # into the document component.  Refuse the interval instead of
            # inventing a zero-variance seed.
            rows = None
            _zero_token_seed = True
    out = {'status': 'unverifiable', 'n_seeds': len(recs), 'n_docs': None,
           'n_docs_dropped_zero_token': None,
           'se_seed_logppl': None, 'se_doc_logppl': None,
           'se_combined_logppl': None, 'ci95_ppl': None,
           'component_ratio_seed_over_doc': None, 'conf': None,
           'method': 'two-source variance combination on log-PPL (seed training noise x evaluation-document bootstrap); approximation, not a fitted hierarchical model'}
    if _mixed_estimands:
        out['reason'] = ('records mix nested-holdout and full-validation row '
                         'samples; the seed and document components would come '
                         'from two different evaluation sets')
        return out
    if _zero_token_seed:
        out['reason'] = ('at least one record has no scored tokens after '
                         'dropping all-zero documents; its pooled PPL is '
                         'undefined, so no seed or document interval is '
                         'invented from it')
        return out
    # Coverage, draw count and RNG seed identify the interval.  Never build a
    # 95% interval while echoing conf=True/1.0, and never let a fractional
    # n_boot silently become a different draw count.
    try:
        _conf = _validated_alpha(conf, context='_combined_seed_doc_ci conf')
    except ValueError as e:
        out['reason'] = str(e)
        return out
    _nb = _as_seed_int(n_boot)
    if (_nb is None or _nb < 20
            or _nb > BOOTSTRAP_HARD_MAX_DRAWS):
        out['reason'] = (
            f'n_boot={n_boot!r} must be an integer in '
            f'[20, {BOOTSTRAP_HARD_MAX_DRAWS}] for a percentile bootstrap')
        return out
    _seed = _as_seed_int(seed)
    if _seed is None:
        out['reason'] = f'seed={seed!r} is not an integer; draws are not reproducible'
        return out
    out['conf'] = float(_conf)
    S = len(recs)
    if S < 1:
        return out
    # Seed-level component: per-seed pooled PPL over the SAME doc set.
    seed_logppl = None
    if rows is not None:
        per_tok = nll_mat.sum(axis=1) / np.maximum(ntok_mat.sum(axis=1), 1e-30)
        # Cross-entropy is zero for a perfectly predicted token, so a zero
        # log-PPL is a valid measurement (PPL=1), not a missing component.
        ok = np.isfinite(per_tok) & (per_tok >= 0)
        if ok.sum() == S and S >= 2:
            # nll/tokens is already cross-entropy = log(PPL); do not log it a
            # second time or the reported interval is not a PPL interval.
            lp = per_tok
            seed_logppl = lp
            out['se_seed_logppl'] = float(lp.std(ddof=1) / math.sqrt(S))
            out['n_docs'] = int(nll_mat.shape[1])
            out['n_docs_dropped_zero_token'] = n_dropped
        elif ok.sum() == S:
            out['n_docs'] = int(nll_mat.shape[1])
            out['n_docs_dropped_zero_token'] = n_dropped
    # Doc-level component: bootstrap SD of log pooled PPL over doc resamples
    # (seed-pooled NLL/ntok columns, the same pooled estimator the doc CI
    # uses, taken on the log scale).
    if rows is not None and nll_mat.shape[1] >= 2:
        rng = np.random.default_rng(_seed)
        nll_col = nll_mat.sum(axis=0)
        ntok_col = ntok_mat.sum(axis=0)
        R = int(nll_mat.shape[1])
        draws = np.empty(int(_nb), dtype=float)
        _chunk = _bootstrap_draw_chunk(R)
        _b = 0
        while _b < int(_nb):
            _nb_draw = min(_chunk, int(_nb) - _b)
            counts = rng.multinomial(R, np.full(R, 1.0 / R, dtype=float), size=_nb_draw)
            nt = counts @ ntok_col
            nn = counts @ nll_col
            with np.errstate(divide='ignore', invalid='ignore'):
                # nn/nt is already the pooled log-PPL (CE) of the resample.
                lp_draw = np.where((nt > 0) & np.isfinite(nn), nn / np.maximum(nt, 1e-30), np.nan)
                lp_draw = np.where(np.isfinite(lp_draw) & (lp_draw >= 0), lp_draw, np.nan)
            draws[_b:_b + _nb_draw] = lp_draw
            _b += _nb_draw
        draws = draws[np.isfinite(draws)]
        if draws.size >= max(20, int(0.5 * _nb)):
            sd = float(draws.std(ddof=1))
            if sd < 1e-12:
                # Identical documents: the bootstrap SD is pure float noise.
                # Report the component as exactly zero so the ratio field is
                # not a 1e16 number that reads like a measurement.
                sd = 0.0
            out['se_doc_logppl'] = sd
            if out['n_docs'] is None:
                out['n_docs'] = R
                out['n_docs_dropped_zero_token'] = n_dropped
    if out['se_seed_logppl'] is None and out['se_doc_logppl'] is None:
        out['status'] = 'unverifiable'
        return out
    if out['se_seed_logppl'] is None:
        out['status'] = 'seed_component_unverifiable'
        return out
    if out['se_doc_logppl'] is None:
        out['status'] = 'doc_component_unverifiable'
        return out
    se_s = out['se_seed_logppl']
    se_d = out['se_doc_logppl']
    out['se_combined_logppl'] = float(math.sqrt(se_s * se_s + se_d * se_d))
    if se_d > 0:
        out['component_ratio_seed_over_doc'] = float(se_s / se_d)
    try:
        tc = _t_critical(float(_conf), S - 1)
    except Exception:
        tc = None
    if tc is not None and math.isfinite(tc):
        m = float(seed_logppl.mean())
        _lo = float(_safe_exp_ppl(m - tc * out['se_combined_logppl']))
        _hi = float(_safe_exp_ppl(m + tc * out['se_combined_logppl']))
        if math.isfinite(_lo) and math.isfinite(_hi):
            out['ci95_ppl'] = [_lo, _hi]
            out['status'] = 'ok'
        else:
            # The endpoint exceeds float64 even though the log-scale interval
            # is finite; an infinite PPL bound would break strict JSON and is
            # not a measured interval.
            out['ci95_ppl'] = None
            out['ci95_ppl_unavailable_reason'] = (
                'an exponentiated confidence endpoint exceeds the '
                'representable float64 range')
            out['status'] = 'ci_unavailable'
    else:
        out['status'] = 'ci_unavailable'
    return out

def paired_row_bootstrap_delta(pairs, *, n_boot=1000, seed=20261018, conf=0.95):
    """Evaluation-row bootstrap CI for a PAIRED token-weighted PPL difference.

    Seed-level tests answer "is the sign stable across training draws"; they do
    not quantify how much the fixed evaluation rows move the effect.  This
    helper resamples the SAME rows for both arms and every shared seed (paired
    by row), recomputes each arm's token-pooled PPL, and returns the percentile
    CI of mean_s(PPL_a - PPL_b).  It is an evaluation-sample interval only: the
    training-seed component is deliberately NOT bootstrapped here, so it must
    not be quoted as a total uncertainty interval and does not replace the
    exact sign-flip test.

    ``pairs`` is [(record_a, record_b), ...] for shared seeds.  Every record
    must expose per-row NLL and token counts for the SAME rows; nested holdout
    records must carry the holdout row arrays.  Missing/mismatched accounting
    returns an explicit ``unverifiable`` status rather than a narrow interval
    computed from a different sample.
    """
    pairs = list(pairs or [])
    out = {'status': 'unverifiable', 'n_pairs': len(pairs), 'n_rows': None,
           'n_boot': None, 'n_boot_valid': 0, 'mean_delta': None,
           'ci95_delta_ppl': None, 'conf': None,
           'method': 'paired evaluation-row bootstrap of mean_s(PPL_a - PPL_b); '
                     'row-level uncertainty only, not training-seed uncertainty'}
    # The interval's coverage and draw count are part of the measurement.
    # Silently substituting 95% when conf=True/1.5 was passed (while echoing
    # conf=1.0 in the output) would describe an interval that was never built.
    try:
        _conf = _validated_alpha(conf, context='paired_row_bootstrap_delta conf')
    except ValueError as e:
        out['reason'] = str(e)
        return out
    _nb = _as_seed_int(n_boot)
    if (_nb is None or _nb < 20
            or _nb > BOOTSTRAP_HARD_MAX_DRAWS):
        out['reason'] = (
            f'n_boot={n_boot!r} must be an integer in '
            f'[20, {BOOTSTRAP_HARD_MAX_DRAWS}] for a percentile bootstrap')
        return out
    _seed = _as_seed_int(seed)
    if _seed is None:
        out['reason'] = f'seed={seed!r} is not an integer; draws are not reproducible'
        return out
    out['n_boot'] = int(_nb)
    out['conf'] = float(_conf)

    def _arrays(rec):
        if not isinstance(rec, dict):
            return None
        nested = rec.get('ppl_estimation_rows') == 'nested_validation_holdout'
        nll = rec.get('nll_per_row_holdout') if nested else rec.get('nll_per_row')
        ntok = rec.get('n_tokens_per_row_holdout') if nested else rec.get('n_tokens_per_row')
        if nested and (nll is None or ntok is None):
            return None
        if not isinstance(nll, list) or not isinstance(ntok, list) or len(nll) != len(ntok) or not nll:
            return None
        _tok = [_exact_count(v) for v in ntok]
        if any(v is None for v in _tok):
            return None
        try:
            _nll = np.asarray(nll, dtype=float)
        except (TypeError, ValueError, OverflowError):
            return None
        if not np.isfinite(_nll).all():
            return None
        _tok_arr = _token_matrix(_tok)
        if _tok_arr is None:
            return None
        return _nll, _tok_arr

    if not pairs:
        out['reason'] = 'no shared paired seeds'
        return out
    A, B = [], []
    _estimands = set()
    for a, b in pairs:
        _a, _b = _arrays(a), _arrays(b)
        if _a is None or _b is None:
            out['reason'] = 'a paired record lacks per-row NLL/token accounting for the estimand'
            return out
        _a_nested = (isinstance(a, dict)
                     and a.get('ppl_estimation_rows') == 'nested_validation_holdout')
        _b_nested = (isinstance(b, dict)
                     and b.get('ppl_estimation_rows') == 'nested_validation_holdout')
        if _a_nested != _b_nested:
            out['reason'] = ('the two arms of a pair declare different estimands '
                             '(nested holdout vs full validation); their row '
                             'intervals are not paired')
            return out
        _estimands.add(_a_nested)
        A.append(_a)
        B.append(_b)
    if len(_estimands) > 1:
        out['reason'] = ('the shared pairs mix nested-holdout and full-validation '
                         'estimands; the interval would average two different samples')
        return out
    R = len(A[0][0])
    if R < 2:
        out['reason'] = f'only {R} evaluation row(s); a row bootstrap needs at least 2'
        return out
    if any(len(x[0]) != R or len(y[0]) != R for x, y in zip(A, B)):
        out['reason'] = 'paired records score different row counts'
        return out
    nll_a = np.stack([x[0] for x in A], axis=0)       # [S, R]
    tok_a = np.stack([x[1] for x in A], axis=0)
    nll_b = np.stack([x[0] for x in B], axis=0)
    tok_b = np.stack([x[1] for x in B], axis=0)
    if (tok_a < 0).any() or (tok_b < 0).any():
        out['reason'] = 'negative per-row token count'
        return out
    # Point estimate on the full rows, equal weight per seed (the same
    # estimand the paired contrast uses).
    def _seed_ppls(nll, tok):
        _den = tok.sum(axis=1).astype(float)
        _ok = _den > 0
        _ppl = np.full(_den.shape, np.nan, dtype=float)
        _ppl[_ok] = np.exp(nll.sum(axis=1)[_ok] / _den[_ok])
        return _ppl
    _pa, _pb = _seed_ppls(nll_a, tok_a), _seed_ppls(nll_b, tok_b)
    _ok = np.isfinite(_pa) & np.isfinite(_pb)
    if not _ok.any():
        out['reason'] = 'no seed has a positive-token pooled PPL in both arms'
        return out
    out['n_rows'] = int(R)
    out['n_seeds_used'] = int(_ok.sum())
    out['mean_delta'] = _scaled_mean_or_raw(_pa[_ok] - _pb[_ok])
    rng = np.random.default_rng(_seed)
    vals = np.full(int(_nb), np.nan, dtype=float)
    _chunk = _bootstrap_draw_chunk(R)
    _b = 0
    while _b < int(_nb):
        _nb_draw = min(_chunk, int(_nb) - _b)
        counts = rng.multinomial(R, np.full(R, 1.0 / R, dtype=float), size=_nb_draw)
        def _draw(nll, tok):
            nt = counts @ tok.T.astype(float)          # [nb, S]
            nn = counts @ nll.T                         # [nb, S]
            with np.errstate(divide='ignore', invalid='ignore'):
                ppl = np.exp(nn / np.where(nt > 0, nt, np.nan))
            return np.where(np.isfinite(ppl), ppl, np.nan)
        da = _draw(nll_a, tok_a)
        db = _draw(nll_b, tok_b)
        diff = da - db
        _valid = np.isfinite(diff)
        _cnt = _valid.sum(axis=1)
        _sum = np.where(_valid, diff, 0.0).sum(axis=1)
        vals[_b:_b + _nb_draw] = np.where(_cnt > 0, _sum / np.maximum(_cnt, 1), np.nan)
        _b += _nb_draw
    vals = vals[np.isfinite(vals)]
    out['n_boot_valid'] = int(vals.size)
    if vals.size < max(20, int(0.5 * int(_nb))):
        out['reason'] = 'too few finite bootstrap draws'
        return out
    _a = 1.0 - _conf
    lo, hi = np.percentile(vals, [100.0 * _a / 2.0, 100.0 * (1.0 - _a / 2.0)])
    out['ci95_delta_ppl'] = [float(lo), float(hi)]
    out['status'] = 'ok'
    out['note'] = ('paired rows keep the two arms on the same evaluation documents; '
                   'the interval covers row-sampling uncertainty only.  It is not a '
                   'substitute for the seed-level exact sign-flip p-value.')
    return out


_TWO_WAY_BOOTSTRAP_ASSUMPTION = (
    'the paired seeds and the evaluation units are each exchangeable samples '
    'from their respective populations, and the two resampling dimensions are '
    'independent; this is a design-based uncertainty readout, not a fitted '
    'hierarchical/random-effects model'
)


def paired_nll_two_way_bootstrap(nll_a, nll_b, n_tokens, *, n_boot=2000,
                                 seed=20261018, conf=0.95):
    """Two-level (seed x evaluation-unit) bootstrap CI for a paired ΔPPL.

    The seed-level exact sign-flip test answers "is the direction stable across
    training draws"; the cluster-robust document interval answers "how much do
    the frozen evaluation units move the token-pooled Δ, conditional on these
    seeds".  Neither is the total uncertainty of the reported number: with 3-12
    seeds and a few dozen documents, both sources act at once.  Reporting the
    two intervals side by side lets a reader under-state the uncertainty by
    picking the narrower one.

    This helper resamples BOTH levels of the crossed design, independently and
    with replacement: a draw selects ``S`` seed indices and ``D`` unit indices,
    recomputes the paired token-pooled PPL difference on the resampled units
    for each resampled seed, and averages over the resampled seeds.  The
    resulting percentile interval covers seed-to-seed training noise and
    evaluation-unit sampling together, on the SAME estimand as
    :func:`_paired_nll_doc_ci` (point estimate unchanged).

    Inputs are ``[n_seeds, n_units]`` per-unit summed NLLs for the two arms and
    the shared per-unit token counts.  Malformed accounting raises ``ValueError``
    (a wrong interval is worse than an explicit refusal); too few seeds/units or
    too few finite draws returns ``status='unverifiable'`` with a reason.

    The interval is a **descriptive total-uncertainty readout**, not a new
    test: it never changes the exact sign-flip p, the Holm family, a SESOI
    decision or a frozen verdict.  It also does not model within-unit token
    correlation beyond the unit cluster, and a percentile interval with only a
    dozen seed clusters can under-cover; the seed/unit counts are reported so
    the reader can see that limitation.
    """
    out = {'kind': 'paired_two_way_bootstrap', 'status': 'unverifiable',
           'n_seeds': None, 'n_units': None,
           'n_boot': None, 'n_boot_valid': 0, 'conf': None,
           'delta_ppl': None, 'delta_ppl_ci95': None,
           'se_ppl': None,
           'delta_log_ratio': None, 'delta_log_ratio_ci95': None,
           'ratio_ci95': None, 'se_log_ratio': None,
           'warnings': [],
           'method': ('two-way (seed x evaluation-unit) percentile bootstrap '
                      'of the paired token-pooled ΔPPL; both levels resampled '
                      'independently'),
           'assumption': _TWO_WAY_BOOTSTRAP_ASSUMPTION,
           'role': ('descriptive total-uncertainty interval; it does not '
                    'change any p-value, Holm family, SESOI decision or '
                    'frozen confirmation verdict'),
           'note': ('point estimate is the same mean-over-seeds token-pooled '
                    'ΔPPL as the document cluster interval; the bootstrap only '
                    'widens the uncertainty to include training-seed draws'),
           'reason': None}

    A = _as_nll_matrix(nll_a, 'paired_nll_two_way_bootstrap(nll_a)')
    B = _as_nll_matrix(nll_b, 'paired_nll_two_way_bootstrap(nll_b)')
    if A.shape != B.shape:
        raise ValueError(
            'paired_nll_two_way_bootstrap: arm shapes differ '
            f'({A.shape} vs {B.shape}); a paired interval needs the same seeds '
            'and units')
    try:
        raw_t = np.asarray(n_tokens).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('paired_nll_two_way_bootstrap: n_tokens must be numeric') from None
    if _contains_bool_value(n_tokens) or raw_t.dtype.kind == 'b':
        raise ValueError(
            'paired_nll_two_way_bootstrap: token counts must be integers, not booleans')
    counts = [_exact_count(v, minimum=1) for v in raw_t.tolist()]
    if any(c is None for c in counts):
        raise ValueError(
            'paired_nll_two_way_bootstrap: every unit needs a finite positive '
            'integer token count')
    T = np.asarray(counts, dtype=float)
    if T.size != A.shape[1]:
        raise ValueError(
            f'paired_nll_two_way_bootstrap: {T.size} token counts for '
            f'{A.shape[1]} units')
    if isinstance(conf, (bool, np.bool_)):
        raise ValueError(
            'paired_nll_two_way_bootstrap: conf must be a probability in (0, 1)')
    try:
        conf = float(conf)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            'paired_nll_two_way_bootstrap: conf must be a probability in (0, 1)') from None
    if not (0.0 < conf < 1.0):
        raise ValueError(
            'paired_nll_two_way_bootstrap: conf must be a probability in (0, 1)')
    _nb = _as_seed_int(n_boot)
    if (_nb is None or _nb < 20
            or _nb > BOOTSTRAP_HARD_MAX_DRAWS):
        raise ValueError(
            f'paired_nll_two_way_bootstrap: n_boot={n_boot!r} must be an '
            f'integer in [20, {BOOTSTRAP_HARD_MAX_DRAWS}] for a percentile '
            'bootstrap')
    _seed = _as_seed_int(seed)
    if _seed is None:
        raise ValueError(
            f'paired_nll_two_way_bootstrap: seed={seed!r} is not an integer; '
            'draws are not reproducible')
    S, D = A.shape
    out.update({'n_seeds': int(S), 'n_units': int(D),
                'n_boot': int(_nb), 'conf': float(conf)})
    if S < 2:
        out['reason'] = ('a one-seed panel gives no seed-to-seed variation; '
                         'the two-level interval is undefined')
        return out
    if D < 2:
        out['reason'] = ('a one-unit panel gives no evaluation-unit variation; '
                         'the two-level interval is undefined')
        return out
    if S < 5:
        out['warnings'].append(
            f'only {S} seed clusters: the percentile two-level interval may be '
            'too narrow')
    if D < 20:
        out['warnings'].append(
            f'only {D} evaluation units: the percentile two-level interval may '
            'be coarse')
    total_tok = float(T.sum())
    if not math.isfinite(total_tok) or total_tok <= 0.0:
        out['reason'] = 'the total token count must be positive'
        return out
    ce_a = A.sum(axis=1) / total_tok
    ce_b = B.sum(axis=1) / total_tok
    ea = np.asarray([_safe_exp_ppl(x) for x in ce_a], dtype=float)
    eb = np.asarray([_safe_exp_ppl(x) for x in ce_b], dtype=float)
    if not (np.all(np.isfinite(ea)) and np.all(np.isfinite(eb))):
        out['reason'] = ('the full-panel pooled PPL overflows the representable '
                         'float range; the paired Δ is not a finite measurement')
        return out
    delta_ppl = _scaled_mean_or_raw(ea - eb)
    delta_lr = float(np.mean(ce_a - ce_b))
    out['delta_ppl'] = delta_ppl
    out['delta_log_ratio'] = delta_lr

    rng = np.random.default_rng(_seed)
    draw_ppl = np.full(int(_nb), np.nan, dtype=float)
    draw_lr = np.full(int(_nb), np.nan, dtype=float)
    # Each draw materialises [draws, n_seeds] arrays plus the [draws, n_units]
    # count matrix, so size the chunk from the crossed cell count.
    chunk = _bootstrap_draw_chunk(D * S)
    filled = 0
    while filled < int(_nb):
        m = min(chunk, int(_nb) - filled)
        # Independent resampling of the two crossed sampling levels.  Counts
        # (rather than index tables) keep duplicated units explicit and make
        # the token-weighted recomputation one matrix product.
        doc_counts = rng.multinomial(D, np.full(D, 1.0 / D), size=m).astype(float)
        seed_counts = rng.multinomial(S, np.full(S, 1.0 / S), size=m).astype(float)
        tot = doc_counts @ T
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            nll_a_draw = doc_counts @ A.T
            nll_b_draw = doc_counts @ B.T
            denom = tot[:, None]
            ppl_a = np.exp(nll_a_draw / denom)
            ppl_b = np.exp(nll_b_draw / denom)
            per_seed = ppl_a - ppl_b
            per_seed_lr = (nll_a_draw - nll_b_draw) / denom
            weights = seed_counts / float(S)
            ppl_mean = np.sum(per_seed * weights, axis=1)
            lr_mean = np.sum(per_seed_lr * weights, axis=1)
        draw_ppl[filled:filled + m] = ppl_mean
        draw_lr[filled:filled + m] = lr_mean
        filled += m
    keep = np.isfinite(draw_ppl) & np.isfinite(draw_lr)
    ppl_valid = draw_ppl[keep]
    lr_valid = draw_lr[keep]
    out['n_boot_valid'] = int(ppl_valid.size)
    if ppl_valid.size < max(20, int(0.5 * int(_nb))):
        out['reason'] = ('too few finite bootstrap draws; the resampled PPL '
                         'overflowed or produced non-finite values')
        return out
    lo_q = 100.0 * (1.0 - conf) / 2.0
    hi_q = 100.0 - lo_q
    ci = _percentile_ci(ppl_valid.tolist(), lo_q, hi_q)
    ci_lr = _percentile_ci(lr_valid.tolist(), lo_q, hi_q)
    if ci is None or ci_lr is None:
        out['reason'] = 'the percentile interval has fewer than two valid draws'
        return out
    ratio_ci = None
    try:
        _rlo, _rhi = _safe_exp_ppl(ci_lr[0]), _safe_exp_ppl(ci_lr[1])
        if (math.isfinite(_rlo) and math.isfinite(_rhi)):
            ratio_ci = [float(_rlo), float(_rhi)]
    except (TypeError, ValueError, OverflowError):
        ratio_ci = None
    out.update({
        'status': 'ok',
        'delta_ppl_ci95': [float(ci[0]), float(ci[1])],
        'delta_log_ratio_ci95': [float(ci_lr[0]), float(ci_lr[1])],
        'ratio_ci95': ratio_ci,
        'se_ppl': _scaled_std_or_none(ppl_valid),
        'se_log_ratio': _scaled_std_or_none(lr_valid),
    })
    if ppl_valid.size < int(_nb):
        out['warnings'].append(
            f'{int(_nb) - int(ppl_valid.size)} of {int(_nb)} bootstrap draws were '
            'non-finite and were excluded before the percentile interval')
    return out

def measurement_semantics_key(rec):
    """Identity of the code semantics that produced a measurement.

    `run_cfg` normally carries the code-stamp suffix, but legacy records may
    omit it (or predate the suffix).  Two records measured under different
    `_code` stamps are not two seeds of one experiment: preprocessing, the
    collectives' reduction order or a kernel path may differ.  Aggregation and
    arm summaries therefore key on this explicit value *in addition to*
    `run_cfg`; a missing stamp is its own `<unverified>` class rather than a
    wildcard that merges with a verified one.
    """
    if not isinstance(rec, dict):
        return '<unverified>'
    c = rec.get('_code')
    if c is None or str(c) == '':
        return '<unverified>'
    return str(c)


def aggregate(summary):

    def _protocol_tag(rec, key=None):
        if key is None:
            return _tag_of(rec)
        return record_tag(rec, key, _legacy_warm_ok)
    _legacy_warm_ok = legacy_warm_tags(summary)
    per_variant = {}
    for key, rec in summary.items():
        if not isinstance(rec, dict) or 'ppl' not in rec or 'seed' not in rec:
            continue
        if not ppl_is_usable(rec.get('ppl')) or rec.get('synthesized'):
            continue
        # A non-integer seed has no draw identity.  Counting it as an extra
        # seed would fabricate n (and pair it with no other reader).
        if _as_seed_int(rec.get('seed')) is None:
            print(f'[aggregate] WARNING: record {key!r} carries a non-integer seed ({rec.get("seed")!r}) — excluded from aggregation')
            continue
        v = rec.get('variant')
        if v is None:
            v = str(key).split('::', 1)[0]
        # A variant is the arm identity used as a dict key.  A malformed
        # non-string arm id (e.g. a JSON object) is unhashable and would crash
        # the whole aggregation; an invalid identity is no more usable than an
        # invalid seed, so exclude it explicitly instead of coercing it.
        if not isinstance(v, str) or not v:
            print(f'[aggregate] WARNING: record {key!r} carries a non-string/empty variant ({v!r}) — excluded from aggregation')
            continue
        per_variant.setdefault((v, _protocol_tag(rec, key)), []).append(rec)
    def _seed_key(s):
        # JSON round-trips can leave the same seed as 0 or "0"; the rest of the
        # readers normalise through `_as_seed_int`.  Keying dedup/pairing on the
        # raw value would count one seed twice and fabricate an extra n.
        n = _as_seed_int(s)
        return s if n is None else n
    _dups = {}
    for k, recs in per_variant.items():
        cnt = {}
        for r in recs:
            s = _seed_key(r['seed'])
            cnt[s] = cnt.get(s, 0) + 1
        bad = sorted((s for s, c in cnt.items() if c > 1))
        if bad:
            _dups[k] = bad
    if _dups:
        raise ValueError('aggregate: these (variant, protocol) groups hold more than one record for the same seed, so their PPL is ambiguous: ' + '; '.join((f'{v}/{t or '-'}->seeds {s}' for (v, t), s in sorted(_dups.items()))) + '. Remove the stale duplicates (or give the records distinct `protocol` tags) before aggregating.  A record whose value was never MEASURED (an `error` record, or a `synthesized` reconstruction) can carry no PPL and is IGNORED here — drop it from the summary instead of re-labelling it, because a protocol tag only silences this guard, it does not make the number real.')
    # A single (variant, protocol) group must be scored on ONE corpus and ONE
    # validation slice.  A mixed group would make ppl_pooled/dPPL meaningless;
    # refuse it here rather than letting a report average two different
    # evaluation sets that happen to share a run_cfg.
    _fp_conflicts = {}
    for (v, tag), recs in per_variant.items():
        # `eval_fp` is part of the paired identity too: same train/val but a
        # different fixed evaluation slice makes the pooled PPL a mixture of
        # two target sets (pair_reason already refuses such pairs).
        for _k in ('data_fp', 'val_fp', 'eval_fp', 'eval_subset_fp',
                   'holdout_row_fp'):
            _vals = sorted({str(r.get(_k)) for r in recs
                            if r.get(_k) is not None and str(r.get(_k)) != ''})
            if len(_vals) > 1:
                _fp_conflicts.setdefault((v, tag), {})[_k] = _vals
    if _fp_conflicts:
        raise ValueError('aggregate: these (variant, protocol) groups mix different corpus/evaluation fingerprints, so their PPL values are not comparable: '
                         + '; '.join(f'{v}/{t or '-'}: ' + ', '.join(f'{k}={[x[:12] + "…" for x in vals]}' for k, vals in sorted(d.items()))
                                     for (v, t), d in sorted(_fp_conflicts.items()))
                         + '. Report the fingerprint classes separately (they are different data/slices), do not pool them.')
    _multi = {}
    for (v, tag), recs in per_variant.items():
        _multi.setdefault(v, []).append(tag)
    _multi = {v: t for v, t in _multi.items() if len(t) > 1}
    if _multi:
        print('[aggregate] WARNING: these variants hold MULTIPLE protocol groups and are reported per-group, NOT pooled across groups (pooling would fake extra seeds): ' + ', '.join((f'{v}->{t}' for v, t in sorted(_multi.items()))))

    def _cfg_fp(rec):
        # Exact integer identity: int(1000.5) would collapse a different
        # capacity into the 1000-parameter group and pool two models as seeds.
        _p = _as_seed_int(rec.get('params'))
        if _p is not None and _p <= 0:
            _p = None
        return json.dumps([rec.get('run_cfg'), _p,
                           [rec.get(k) for k in PAIR_STEP_KEYS],
                           measurement_semantics_key(rec)],
                          sort_keys=True, default=str)

    def _steps_ok(recs):
        if len(recs) < 2:
            return True
        return len({_cfg_fp(r) for r in recs}) == 1

    def _by_seed(recs):
        out, dup = ({}, [])
        for r in recs:
            s = _seed_key(r['seed'])
            if s in out:
                dup.append(s)
            out[s] = r
        return (out, dup)

    def _paired(recs, base_recs):
        self_map, dup_a = _by_seed(recs)
        base_map, dup_b = _by_seed(base_recs)
        assert not dup_a and (not dup_b), (sorted(set(dup_a)), sorted(set(dup_b)))
        out, skipped = ([], [])
        for s, r in self_map.items():
            b = base_map.get(s)
            if b is None:
                continue
            reason = pair_reason(r, b)
            if reason is not None:
                skipped.append((s, reason))
                continue
            _delta = _paired_primary_delta(r, b)
            if _delta is None:
                skipped.append((s, 'primary estimand missing (nested record without holdout PPL/NLL accounting)'))
                continue
            out.append(_delta)
        return (out, skipped)
    agg = {}
    _split_groups = []
    _group_records = {}

    def _agg_key(v, tag, fp_idx=None):
        key = v if not tag else f'{v}#{tag}'
        return key if fp_idx is None else f'{key}@cfg{fp_idx}'
    for (v, tag), recs in per_variant.items():
        if _steps_ok(recs):
            _split_groups.append(((v, tag, None), recs))
            _group_records[v, tag, None] = recs
            continue
        _subs = {}
        for r in recs:
            _subs.setdefault(_cfg_fp(r), []).append(r)
        # the `@cfg` suffix is a digest of the config fingerprint itself, so a
        # resume pass cannot reshuffle which group a published key refers to
        _ordered = sorted(_subs.items(), key=lambda kv: kv[0])
        print(f'[aggregate] WARNING: `{_agg_key(v, tag)}` holds records from {len(_ordered)} different run_cfg/load/parameter-count/code-semantics configurations — pooling them would average incompatible runs (or two different MODEL WIDTHS or CODE STAMPS, which are not two seeds of one model) and fake extra seeds.  Reported per configuration instead.')
        for _fp, _rs in _ordered:
            _i = hashlib.sha1(_fp.encode('utf-8')).hexdigest()[:8]
            _split_groups.append(((v, tag, _i), _rs))
            _group_records[v, tag, _i] = _rs
    for (_v, _tag, _fp_idx), recs in _split_groups:
        v, tag = (_v, _tag)
        key = _agg_key(v, tag, _fp_idx)
        def _seed_sort_key(r):
            s = r.get('seed')
            if isinstance(s, (bool, np.bool_)):
                return (1, 0.0, str(s))
            if isinstance(s, (int, np.integer)):
                # Exact integer labels can exceed float64 (the seed planner
                # accepts arbitrary IDs); converting them to float raised
                # OverflowError while sorting.  Python compares int/float
                # tuples exactly, so keep the large integer as an integer.
                return (0, int(s), '')
            if isinstance(s, (float, np.floating)) and math.isfinite(float(s)):
                return (0, float(s), '')
            return (1, 0.0, str(s))
        _recs_sorted = sorted(recs, key=_seed_sort_key)
        recs = _recs_sorted
        # Every per-seed statistic must target the record's PRIMARY estimand.
        # For a nested record `ppl` is the full-validation value, which still
        # contains the rows used to select its hyper-parameters; the seed
        # mean/std/interval would then describe a different (leaky) sample
        # than `ppl_primary` and the paired dPPL column.
        _primary_vals = [_record_primary_ppl(r) for r in recs]
        _n_primary_missing = sum(v is None for v in _primary_vals)
        _nested_flags = {r.get('ppl_estimation_rows') == 'nested_validation_holdout'
                         for r in recs}
        _mixed_estimands = len(_nested_flags) > 1
        if _mixed_estimands:
            print(f'[aggregate] WARNING: {key}: the group mixes nested-holdout and '
                  'full-validation records — per-seed values and pooled NLL would come '
                  'from two different samples; the whole group falls back to '
                  'full-validation PPL and nested claims are NOT supported for it')
        if _n_primary_missing or _mixed_estimands:
            if _n_primary_missing and not _mixed_estimands:
                print(f'[aggregate] WARNING: {key}: {_n_primary_missing}/{len(recs)} record(s) declare the nested estimand but lack holdout accounting — the whole group falls back to full-validation PPL so the numbers stay on one sample; nested claims are NOT supported for this group')
            ppls = np.array([r['ppl'] for r in recs], dtype=float)
            _primary_estimand = 'full_validation_fallback'
            _pooled, _nll_total, _ntok_total = _pooled_ppl_from_records(
                recs, estimand=False)
        else:
            ppls = np.array(_primary_vals, dtype=float)
            _primary_estimand = (
                'nested_validation_holdout'
                if _nested_flags == {True}
                else 'full_validation')
            _pooled, _nll_total, _ntok_total = _pooled_ppl_from_records(recs)
        _sorted = np.sort(ppls)
        _med = _scaled_median_or_raw(_sorted)
        _q1, _q3 = (_scaled_percentile_or_raw(_sorted, (25, 75))
                    if len(_sorted) >= 2 else (_med, _med))
        _ppl_mean = _scaled_mean_or_raw(ppls)
        entry = {'variant': v, 'protocol': tag, 'n_seeds': len(recs),
                 'ppl_mean': _ppl_mean,
                 'ppl_mean_arithmetic': _ppl_mean,
                 'ppl_std': _scaled_std_or_raw(ppls),
                 'ppls': [float(p) for p in ppls], 'seeds': [r.get('seed') for r in recs],
                 'ppl_primary_estimand': _primary_estimand,
                 'ppl_primary_missing': int(_n_primary_missing),
                 'params': _group_params(recs), 'tokens_seen': recs[0].get('tokens_seen'),
                 'ppl_median': _med, 'ppl_iqr': [_q1, _q3],
                 'ppl_ci95_seed_t': _seed_t_ci(ppls),
                 'nll_sum': _nll_total, 'n_tokens': _ntok_total,
                 'ppl_pooled': (_pooled if _pooled is not None else None),
                 'ppl_ci95_doc_bootstrap': _doc_bootstrap_ppl_ci(recs),
                 'ppl_ci_seed_doc_combined': _combined_seed_doc_ci(recs),
                 'ppl_aggregation': ('pooled_token_nll' if _pooled is not None else 'arithmetic_seed_mean_fallback (records lack nll_sum/n_tokens)'),
                 'ppl_primary': (float(_pooled) if _pooled is not None
                                 else _ppl_mean)}
        _pb = next((r.get('param_breakdown') for r in recs if isinstance(r.get('param_breakdown'), dict)), None)
        if _pb:
            entry['param_breakdown'] = _pb
            entry['params_total'] = _pb.get('total')
            entry['params_attention'] = _pb.get('attention')
            entry['params_mlp'] = _pb.get('mlp')
            entry['params_embedding'] = _pb.get('embedding')
            entry['active_params'] = _pb.get('active')
            entry['attention_param_share'] = _pb.get('attention_share')
        _kv = next((r.get('kv_cache_elements_per_token') for r in recs if isinstance(r.get('kv_cache_elements_per_token'), dict)), None)
        if _kv:
            entry['kv_cache_elements_per_token'] = _kv
        if _fp_idx is not None:
            entry['run_cfg_class'] = _fp_idx
            entry['run_cfg'] = recs[0].get('run_cfg')
        dyn_lens, all_lens, deltas = ([], [], [])
        bf1, bex = ([], [])
        # `bnd_rand` has two sampling definitions; their excesses are different
        # estimands and must not be averaged into one series.
        _bnd_rand_kinds = set()

        def _cell(layer, k):
            return _finite_scalar(layer.get(k))
        for r in recs:
            r_f1, r_ex, r_dl = ([], [], [])
            _stats = r.get('stats')
            if not isinstance(_stats, dict):
                _stats = {}
            for lk, lv in _stats.items():
                if not isinstance(lv, dict):
                    continue
                lm = _cell(lv, 'len_mean')
                if lm is None:
                    continue
                all_lens.append(lm)
                if '_dyn' in lk:
                    dyn_lens.append(lm)
                    if 'bnd_f1' in lv:
                        _f1 = _cell(lv, 'bnd_f1')
                        _ex = _cell(lv, 'bnd_excess') if lv.get('bnd_rand_exact') is not False else None
                        if _f1 is not None:
                            r_f1.append(_f1)
                        if _ex is not None or lv.get('bnd_rand') is not None:
                            _kind = lv.get('bnd_rand_kind')
                            _kind = (normalize_bnd_rand_kind(_kind)
                                     if _kind is not None else 'unspecified_legacy')
                            _bnd_rand_kinds.add(_kind)
                            if _ex is not None:
                                r_ex.append(_ex)
                dl = _cell(lv, 'delta')
                if dl is not None:
                    r_dl.append(dl)
            if r_f1:
                bf1.append(float(np.mean(r_f1)))
            if r_ex:
                bex.append(float(np.mean(r_ex)))
            if r_dl:
                # Average per record FIRST (as bf1/bex do), so records with more
                # dynamic layers do not get extra weight in the cross-seed mean.
                deltas.append(float(np.mean(r_dl)))
        if len(_bnd_rand_kinds) > 1:
            entry['boundary_rand_kind_conflict'] = sorted(_bnd_rand_kinds)
            print(f'[aggregate] WARNING: {key}: `bnd_rand` cells mix the '
                  f'{sorted(_bnd_rand_kinds)} definitions — their bnd_excess '
                  'values are different estimands and are NOT averaged; '
                  'report each kind separately')
        elif _bnd_rand_kinds:
            entry['boundary_rand_kind'] = next(iter(_bnd_rand_kinds))
        if dyn_lens:
            entry['avg_dyn_block_len'] = float(np.mean(dyn_lens))
        elif all_lens:
            entry['avg_block_len'] = float(np.mean(all_lens))
        if deltas:
            entry['delta_logit_mean'] = float(np.mean(deltas))
        if bf1:
            entry['boundary_f1_dyn'] = float(np.mean(bf1))
            entry['boundary_f1_dyn_std'] = float(np.std(bf1, ddof=1)) if len(bf1) > 1 else 0.0
        if bex and 'boundary_rand_kind_conflict' not in entry:
            entry['boundary_excess_dyn'] = float(np.mean(bex))
        hists = [r['ppl_history'] for r in recs
                 if isinstance(r.get('ppl_history'), (list, tuple)) and r.get('ppl_history')]
        if hists:
            per_step = {}
            for _h in hists:
                for _pt in _h:
                    try:
                        _s = int(_pt[0])
                        _v = float(_pt[1])
                    except (TypeError, ValueError, IndexError, KeyError):
                        continue
                    per_step.setdefault(_s, []).append(_v)
            def _grid(_hist):
                _out = []
                for _p in _hist:
                    try:
                        _out.append(int(_p[0]))
                    except (TypeError, ValueError, IndexError, KeyError):
                        continue
                return _out
            _ref_steps = _grid(hists[0])
            _off_grid = []
            for _i, _h in enumerate(hists[1:], start=1):
                _hs = _grid(_h)
                if _hs != _ref_steps:
                    _off_grid.append(_i)
            if _off_grid:
                print(f'[aggregate] {key}: seed(s) {_off_grid} of {len(hists)} evaluate at a DIFFERENT step grid than seed 0 — the seed-averaged curve is taken over the steps they SHARE (aligned on the step VALUE, not on list position), so a differently-sampled seed contributes only to the common steps.')
            steps_h = sorted(per_step)
            cols = {s: i for i, s in enumerate(steps_h)}
            mat = np.full((len(hists), len(steps_h)), float('nan'))
            for _i, _h in enumerate(hists):
                _own = {}
                for _pt in _h:
                    try:
                        _own[int(_pt[0])] = float(_pt[1])
                    except (TypeError, ValueError, IndexError, KeyError):
                        continue
                for _s, _v in _own.items():
                    mat[_i, cols[_s]] = _v
            # dtype=bool is essential: an all-malformed history yields a
            # zero-column matrix whose inferred dtype is float64, and ``~good``
            # then raises instead of reporting an empty curve.
            good = np.array([[ppl_is_usable(v) for v in row] for row in mat], dtype=bool)
            n_good = good.sum(axis=0)
            # A naive column sum over two finite 1.5e308 seed points overflows
            # to inf although their mean is representable.  Average each step
            # with the guarded mean (ordinary panels keep the NumPy value).
            mean_ppl = np.full(len(steps_h), np.nan, dtype=float)
            for _j in range(len(steps_h)):
                _col = mat[good[:, _j], _j]
                if _col.size:
                    mean_ppl[_j] = float(_scaled_mean_or_raw(_col))
            n_dropped = int((~good).sum())
            if n_dropped:
                print(f'[aggregate] {key}: {n_dropped} non-finite PPL curve point(s) across {len(hists)} seed(s) excluded from the seed-averaged curve ({int((n_good == 0).sum())} step(s) had no usable seed and are omitted)')
            _keep = n_good > 0
            entry['ppl_curve'] = [[int(s), float(m)] for s, m, k in zip(steps_h, mean_ppl, _keep) if k]
        agg[key] = entry
    for key, entry in agg.items():
        recs = _group_records.get((entry['variant'], entry['protocol'], entry.get('run_cfg_class')), [])
        entry['source'] = key
        if recs:
            entry['_group_records'] = recs
        rates = []
        for r in recs:
            _ts = _finite_scalar(r.get('tokens_seen'))
            _st = _finite_scalar(r.get('steps'))
            if _ts is not None and _st is not None and _st > 0.0:
                rates.append(_ts / _st)
        if rates:
            entry['tokens_per_step'] = float(np.mean(rates))
    _full_groups = {(tag, fp_idx): recs for (v, tag, fp_idx), recs in _split_groups if v == 'full'}

    def _cfg_fp_noparam(rec):
        # Baseline-matching fingerprint WITHOUT params: a `full` baseline
        # legitimately differs from the sparse arm in parameter count, so a
        # params-including fingerprint could never match it (dead branch).
        # run_cfg + steps/tokens + code semantics is exactly the gate
        # `pair_reason` applies (the code stamp must not be mixed either).
        return json.dumps([rec.get('run_cfg'),
                           [rec.get(k) for k in PAIR_STEP_KEYS],
                           measurement_semantics_key(rec)],
                          sort_keys=True, default=str)

    def _baseline_for(tag, fp_idx, base_groups, recs):
        # Pairing is only legal inside one (variant, protocol) group.  A
        # tagged arm must never borrow an untagged (or differently tagged)
        # baseline: that changes the warmup/eval protocol while pretending the
        # contrast is controlled.  Legacy panel readers may still report the
        # arm's own marginal aggregate; they simply get no dPPL.
        _want = _cfg_fp_noparam(recs[0]) if recs else None
        if _want is not None:
            _cands = sorted(base_groups.items(), key=lambda kv: (kv[0][0] != tag, kv[0][1] is None))
            for (_t_c, _i_c), _grp in _cands:
                if _t_c != tag:
                    continue
                if any((_cfg_fp_noparam(_r) == _want for _r in _grp)):
                    return (_grp, _t_c)
        for _cand in ((tag, fp_idx), (tag, None)):
            if _cand in base_groups:
                return (base_groups[_cand], _cand[0])
        _want_cfg = recs[0].get('run_cfg') if recs else None
        if _want_cfg is not None:
            _rc = [(_t_c, _grp) for (_t_c, _i_c), _grp in base_groups.items() if _t_c == tag and _grp and (_grp[0].get('run_cfg') == _want_cfg)]
            if len(_rc) == 1:
                return (_rc[0][1], _rc[0][0])
        return (None, None)
    _base_amb = []
    for (v, tag, fp_idx), recs in _split_groups:
        if v == 'full':
            continue
        key = _agg_key(v, tag, fp_idx)
        base_recs, base_tag = _baseline_for(tag, fp_idx, _full_groups, recs)
        if base_recs is None:
            print(f'[aggregate] {key}: NO `full` baseline in this panel — dPPL_vs_full not computed')
            continue
        if base_tag != tag:
            # _baseline_for is strict, so this is a programming error rather
            # than a fallback: refuse instead of silently changing protocol.
            _base_amb.append(f'{key}: baseline protocol {base_tag!r} != arm protocol {tag!r}')
            continue
        ds, skipped = _paired(recs, base_recs)
        if skipped:
            _why = sorted({reason for _s, reason in skipped})
            print(f'[aggregate] {key}: {len(skipped)} seed(s) NOT paired with `full` — {_why}; excluded from dPPL_vs_full')
        if ds:
            d = np.array(ds, dtype=float)
            agg[key]['dPPL_vs_full_mean'] = _scaled_mean_or_raw(d)
            agg[key]['dPPL_vs_full_std'] = _scaled_std_or_raw(d)
            agg[key]['n_paired'] = len(d)
            agg[key]['n_paired_skipped'] = len(skipped)
            agg[key]['dPPL_vs_full_base'] = base_tag or '(untagged)'
    _sw_groups = {(tag, fp_idx): recs for (v, tag, fp_idx), recs in _split_groups if v == 'full_sw128_matched'}
    for (v, tag, fp_idx), recs in _split_groups:
        if v in ('full', 'full_sw128_matched'):
            continue
        key = _agg_key(v, tag, fp_idx)
        base_recs, base_tag = _baseline_for(tag, fp_idx, _sw_groups, recs)
        if base_recs is None:
            continue
        if base_tag != tag:
            _base_amb.append(f'{key}: `full_sw128_matched` baseline protocol {base_tag!r} != arm protocol {tag!r}')
            continue
        ds, skipped = _paired(recs, base_recs)
        if skipped:
            _why = sorted({reason for _s, reason in skipped})
            print(f'[aggregate] {key}: {len(skipped)} seed(s) NOT paired with `full_sw128_matched` — {_why}; excluded')
        if ds:
            d = np.array(ds, dtype=float)
            agg[key]['dPPL_vs_sw128m_mean'] = _scaled_mean_or_raw(d)
            agg[key]['dPPL_vs_sw128m_std'] = _scaled_std_or_raw(d)
            agg[key]['n_paired_sw128m'] = len(d)
            agg[key]['n_paired_sw128m_skipped'] = len(skipped)
    if _base_amb:
        raise ValueError('aggregate: unsafe/cross-protocol baseline pairing for ' + '; '.join(sorted(_base_amb)) + '. Pair only a baseline in the SAME (variant, protocol) group; remove or re-tag the mismatched baseline before aggregating.')
    return agg


# Fingerprint fields that define the sample/measurement a PPL describes.
# Records in one aggregate group must agree on all of them; otherwise the group
# is split (not silently averaged) and the split is reported.  `_code` is the
# implementation-semantics stamp: it is not a data fingerprint, but a PPL
# produced by different code semantics is still not one measurement.
_AGG_FINGERPRINT_FIELDS = ('data_fp', 'val_fp', 'eval_fp', 'eval_subset_fp',
                           'holdout_row_fp', '_code')


def _aggregation_fp_class(rec):
    return tuple((k, str(rec.get(k))) for k in _AGG_FINGERPRINT_FIELDS
                 if rec.get(k) is not None and str(rec.get(k)) != '')


def aggregate_with_exclusions(summary):
    """Aggregate a panel while quarantining records that cannot be pooled.

    ``aggregate`` deliberately fails closed: a malformed record, an ambiguous
    same-seed duplicate, or two different evaluation slices in one group are
    data-identity faults, and silently averaging them would manufacture n.
    A finished training panel, however, must still produce the aggregate for
    its valid records and report exactly what was set aside.  This wrapper:

    * validates each record (object / measured PPL / integer seed / arm name),
      names every excluded key and the reason;
    * keeps the FIRST measured record for a duplicated (arm, protocol, seed)
      and excludes later ones as ambiguous duplicates;
    * splits a group whose records carry different corpus/evaluation
      fingerprints or different code-semantics stamps into separate output
      keys instead of averaging them;
    * retries a group that still trips the strict cross-protocol pairing gate
      per arm (so the valid marginal readings survive without a dPPL).

    Returns ``(agg, report, admitted_records)``.  ``report`` is JSON-safe and
    states the tolerance boundary: excluded records are never folded into a
    number, and no "missing" record is invented.
    """
    report = {
        'kind': 'aggregation_exclusions',
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'code': code_stamp(),
        'n_input': 0, 'n_admitted': 0, 'n_excluded': 0,
        'reason_counts': {}, 'excluded': [], 'excluded_truncated': 0,
        'fingerprint_split_groups': [], 'pairing_split_groups': [],
        'n_aggregate_keys': 0,
        'tolerance': ('only malformed/unmeasured records and ambiguous '
                      'same-seed duplicates are dropped; groups that disagree '
                      'on a corpus/evaluation fingerprint or code-semantics '
                      'stamp are SPLIT, not averaged; every exclusion is named '
                      'and counted here'),
    }
    if not isinstance(summary, dict):
        report['n_input'] = 1
        report['reason_counts'] = {'summary is not an object': 1}
        report['n_excluded'] = 1
        return {}, report, {}
    report['n_input'] = len(summary)
    legacy_ok = legacy_warm_tags(summary)
    excluded = []
    reasons = collections.Counter()

    def _drop(key, why):
        excluded.append({'key': str(key), 'reason': str(why)})
        reasons[str(why)] += 1

    candidates = {}
    for key, rec in summary.items():
        if not isinstance(rec, dict):
            _drop(key, 'record is not a JSON object')
            continue
        if rec.get('synthesized'):
            _drop(key, 'synthesized reconstruction (not an independent run)')
            continue
        if not ppl_is_usable(rec.get('ppl')):
            _drop(key, 'missing/non-finite final PPL (failed or unmeasured run)')
            continue
        seed = _as_seed_int(rec.get('seed'))
        if seed is None:
            _drop(key, f'non-integer seed {rec.get("seed")!r}')
            continue
        v = rec.get('variant')
        if v is None:
            v = str(key).split('::', 1)[0]
        if not isinstance(v, str) or not v:
            _drop(key, f'non-string/empty variant {v!r}')
            continue
        # Canonicalise the fields aggregate/readers key on.  The original
        # record is kept untouched for the per-seed readers.
        canon = dict(rec, seed=seed, variant=v)
        tag = record_tag(canon, key, legacy_ok)
        ident = (v, tag, seed)
        if ident in candidates:
            _drop(key, f'duplicate (variant={v!r}, protocol={tag!r}, '
                       f'seed={seed}) — kept {candidates[ident][0]!r}')
            continue
        candidates[ident] = (str(key), canon)

    # Group by (protocol, sample/measurement class) ACROSS variants.  Splitting
    # inside each variant first (the previous shape) silently removed the
    # baseline from every subgroup, so `aggregate` could never form a paired
    # dPPL: the wrapper that every panel dump uses produced marginal means but
    # no paired contrast.  A class is the tuple of data/evaluation/code
    # fingerprints; all arms of one controlled comparison must share it, and
    # only the shared class is paired.
    groups = {}
    by_variant_tag = {}
    tag_classes = {}
    for (v, tag, _seed), (key, rec) in candidates.items():
        _cls = _aggregation_fp_class(rec)
        groups.setdefault((tag, _cls), {})[key] = rec
        by_variant_tag.setdefault((v, tag), []).append(rec)
        tag_classes.setdefault(tag, set()).add(_cls)
    for (v, tag), recs in sorted(by_variant_tag.items(),
                                 key=lambda kv: (kv[0][0], kv[0][1] or '')):
        _classes = {_aggregation_fp_class(r) for r in recs}
        if len(_classes) > 1:
            report['fingerprint_split_groups'].append({
                'variant': v, 'protocol': tag,
                'n_fingerprint_classes': len(_classes),
                'n_records': len(recs),
                'note': ('records from different corpora/evaluation slices or '
                         'code-semantics stamps were reported separately, never '
                         'pooled'),
            })
    agg = {}
    used = set()
    for (tag, _fp_class), sub in sorted(
            groups.items(), key=lambda kv: ((kv[0][0] or ''), repr(kv[0][1]))):
        suffix = ''
        if len(tag_classes.get(tag) or ()) > 1:
            _digest = hashlib.sha1(
                repr(_fp_class).encode('utf-8')).hexdigest()[:8]
            suffix = f'@fp{_digest}'
        try:
            sub_agg = aggregate(sub)
            used.update(sub.keys())
        except Exception as exc:
            # Most likely the strict cross-protocol baseline-pairing gate.
            # Re-run each arm separately so its marginal aggregate (and
            # its measured records) survive; no dPPL is then claimed.
            sub_agg = {}
            _pairing_split = False
            # Group once.  Re-filtering the whole sub-panel for every variant
            # made the fallback O(n_variants * n_records) exactly when a large
            # panel needed it; a single pass keeps the same variant order and
            # the same per-variant key order.
            _by_variant = {}
            for _k, _r in sub.items():
                _by_variant.setdefault(str(_r.get('variant')), {})[_k] = _r
            for vv in sorted(_by_variant):
                _sub_v = _by_variant[vv]
                try:
                    _one = aggregate(_sub_v)
                except Exception as exc2:
                    for k in _sub_v:
                        _drop(k, f'aggregate failed for arm {vv!r}: '
                                  f'{type(exc2).__name__}: {exc2}')
                    continue
                _pairing_split = True
                used.update(_sub_v.keys())
                for k, entry in _one.items():
                    _key = f'{k}@arm{vv}'
                    while _key in sub_agg:
                        _key += "'"
                    sub_agg[_key] = entry
            if _pairing_split:
                report['pairing_split_groups'].append({
                    'protocol': tag,
                    'variants': sorted({str(r.get('variant')) for r in sub.values()}),
                    'error': f'{type(exc).__name__}: {exc}',
                    'note': ('reported per arm; a cross-protocol dPPL is '
                             'not defined and was not produced'),
                })
            if not sub_agg:
                continue
        for key, entry in sub_agg.items():
            out_key = key + suffix
            while out_key in agg:
                out_key += "'"
            agg[out_key] = entry

    _unused = set(k for k, _r in candidates.values()) - used
    _excluded_keys = {e['key'] for e in excluded}
    for key in sorted(_unused):
        if str(key) not in _excluded_keys:
            _drop(key, 'no aggregate entry could be produced for this record')

    report['n_admitted'] = len(used)
    report['n_excluded'] = max(0, len(summary) - len(used))
    report['reason_counts'] = dict(sorted(reasons.items()))
    report['excluded'] = excluded[:200]
    report['excluded_truncated'] = max(0, len(excluded) - 200)
    report['n_aggregate_keys'] = len(agg)
    admitted_records = {key: r for (key, r) in candidates.values()
                        if key in used}
    if report['n_excluded']:
        print(f'[aggregate] {report["n_excluded"]}/{report["n_input"]} '
              f'record(s) quarantined: '
              + ', '.join(f'{k}={v}' for k, v in
                          report['reason_counts'].items()))
    for _g in report['fingerprint_split_groups']:
        print(f'[aggregate] {_g["variant"]}/{_g["protocol"] or "-"}: '
              f'{_g["n_fingerprint_classes"]} fingerprint classes reported '
              'separately (different data/evaluation slices were not pooled)')
    return agg, report, admitted_records


_TW = {'variant': 18, 'ppl': 16, 'd': 15, 'd2': 15, 'params': 10, 'seeds': 6, 'blklen': 7, 'bnd': 13}

def _fmt_cell(value, width, fmt, absent='—'):
    s = absent if value is None else f'{value:{fmt}}'
    return s if len(s) >= width else f'{s:>{width}s}'

def _fmt_pair(main, spread, width, main_fmt, spread_fmt, absent='—', main_cap=None, never_wider=None):
    if main is None:
        s = absent
    else:
        if never_wider is not None:
            fmt = never_wider
        elif main_cap is None:
            fmt = main_fmt
        else:
            fmt = main_cap if abs(main) >= 100000000.0 else main_fmt
        try:
            _spread_ok = (spread is not None
                          and math.isfinite(float(spread)))
        except (TypeError, ValueError, OverflowError):
            _spread_ok = False
        s = (f'{main:{fmt}}±{spread:{spread_fmt}}' if _spread_ok
             else f'{main:{fmt}}±{absent}')
    return s if len(s) >= width else f'{s:>{width}s}'

def format_mean_std(cell, std=None):
    """Render a report cell as ``mean ±std`` with two decimals.

    ``cell`` is either a scalar mean or a mapping with ``mean`` / ``std``
    keys; ``std`` supplies the spread for the scalar form.  A missing mean
    renders as an em dash so a table never turns "no measurement" into zero.
    """
    if isinstance(cell, dict):
        if cell.get('mean') is None:
            return '—'
        mean, std = cell['mean'], cell.get('std', 0.0)
    else:
        mean = cell
    return f'{mean:.2f} ±{std or 0.0:.2f}'

def _print_table(agg):
    print('\n============== validation perplexity (corpus per run cfg) ==============')
    print(f'{'variant':{_TW['variant']}s} {'PPL mean±std':>{_TW['ppl']}s} {'Δvs full':>{_TW['d']}s} {'Δvs sw128m':>{_TW['d2']}s} {'params(M)':>{_TW['params']}s} {'seeds':>{_TW['seeds']}s} {'blklen':>{_TW['blklen']}s} {'bndF1':>{_TW['bnd']}s}')
    order = ['full', 'full_matched', 'full_cos', 'full_sink', 'full_sw128', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic', 'csa_dyn_fuse', 'csa_dyn_nocomp', 'csa_fix_randidx', 'csa_fix_zerocont', 'csa_fix_shiftcont', 'csa_fix_nosink', 'csa_fix_topk8', 'csa_fix_topk64', 'hybrid_fixed', 'hybrid_dynamic', 'hybrid_csa_dyn', 'hybrid_hca_dyn', 'hybrid_csa_dyn_fuse']
    rows = [v for v in order if v in agg] + [v for v in agg if v not in order]
    for v in rows:
        e = agg[v]
        _n = e['n_seeds']
        ppl = _fmt_pair(e.get('ppl_primary', e['ppl_mean']), e.get('ppl_std', 0.0), _TW['ppl'], '8.2f', '5.2f', main_cap='.2f') if _n > 1 else _fmt_cell(e.get('ppl_primary', e['ppl_mean']), _TW['ppl'], '.2f')
        d = e.get('dPPL_vs_full_mean')
        dd = _fmt_pair(d, e.get('dPPL_vs_full_std', 0.0), _TW['d'], '+8.2f', '5.2f', main_cap='+.2f')
        d2 = e.get('dPPL_vs_sw128m_mean')
        dd2 = _fmt_pair(d2, e.get('dPPL_vs_sw128m_std', 0.0), _TW['d2'], '+8.2f', '5.2f', main_cap='+.2f')
        bl = e.get('avg_dyn_block_len', e.get('avg_block_len'))
        bls = _fmt_cell(bl, _TW['blklen'], '.1f')
        _pm = e.get('params')
        pms = _fmt_cell(_pm / 1000000.0 if _pm is not None else None, _TW['params'], '.2f')
        bf = e.get('boundary_f1_dyn')
        bfs = _fmt_pair(bf, e.get('boundary_f1_dyn_std', 0.0), _TW['bnd'], '6.3f', '5.3f', never_wider='.3f')
        try:
            _ns = _fmt_cell(int(_n), _TW['seeds'], 'd')
        except (TypeError, ValueError, OverflowError):
            _ns = _fmt_cell(None, _TW['seeds'], 'd')
        _cells = [(v, _TW['variant']), (ppl, _TW['ppl']), (dd, _TW['d']), (dd2, _TW['d2']), (pms, _TW['params']), (_ns, _TW['seeds']), (bls, _TW['blklen']), (bfs, _TW['bnd'])]
        print(' '.join((f'{s:>{w}s}' for s, w in _cells)))
    print('Δ vs full   = paired per-seed difference vs dense (positive = worse).')
    print('Δ vs sw128m = paired difference vs the windowed param-matched dense')
    print('              baseline = what compression adds beyond pure locality.')
    print('blklen = mean learned block length (dynamic layers); bndF1 = boundary alignment F1 mean±std over seeds (v4: n_sample=64);')
    print('        see summary.json for bnd_rand / bnd_excess: excess > 0 means the')
    print('        learned boundaries are genuinely semantic).')
    print('================================================================')

def _print_pairs(summary):
    recs, _amb = _variant_seed_map(summary)
    _pair_amb = sorted(_amb)
    _pair_skip = []
    # Read the declared leave-one-out design instead of maintaining a second
    # hard-coded table that can silently omit a declared contrast.  The top-k
    # budget sweep is descriptive (not an innovation), so its pairs are added
    # explicitly after the declared ablations.
    _declared = declared_ablation_pairs() + [
        ('csa_fix_topk8', 'csa_fixed'), ('csa_fix_topk64', 'csa_fixed')]
    PAIRS, _seen_pairs = [], set()
    for _pair in _declared:
        if _pair not in _seen_pairs:
            _seen_pairs.add(_pair)
            PAIRS.append(_pair)
    print('\n========== paired per-seed deltas (a − b, positive = a worse) ==========')
    print('  (p = exact two-sided sign-flip over the paired per-seed deltas; a noise-level Δ is flagged even if p is small)')
    if _pair_amb:
        print('  [skip] a (variant, protocol, seed) triple holds more than one measured record, so its PPL is ambiguous — no delta printed for : ' + ', '.join(sorted((f'{v}/{t or '-'}/s{s}' for v, t, s in set(_pair_amb)))[:4]))
    for a, b in PAIRS:

        def _sel(v):
            return {(t, s): p for (vv, t, s), p in recs.items() if vv == v}
        ra, rb = (_sel(a), _sel(b))
        _a_t = {t for t, _s in ra}
        _b_t = {t for t, _s in rb}
        # Same (variant, protocol) group only.  Pairing a tagged arm with an
        # untagged baseline changes the protocol (warmup/eval condition) and
        # would otherwise manufacture a controlled-looking delta.
        common = []
        for tag in sorted(_a_t & _b_t):
            seeds = sorted({s for t, s in ra if t == tag} & {s for t, s in rb if t == tag})
            common += [(tag, tag, s) for s in seeds]
        _only_a = sorted(_a_t - _b_t)
        _only_b = sorted(_b_t - _a_t)
        if _only_a or _only_b:
            print(f'  [skip] {a} vs {b}: protocol group(s) only on one side '
                  f'(a-only={_only_a}, b-only={_only_b}) — no cross-protocol pairing')
        _by_cond = {}
        _est_by_cond = {}
        _sk = []
        for ts, bs, s in common:
            _reason = pair_reason(ra[ts, s], rb[bs, s])
            if _reason is not None:
                _sk.append((s, _reason))
                continue
            _delta = _paired_primary_delta(ra[ts, s], rb[bs, s])
            if _delta is None:
                _sk.append((s, 'primary estimand missing/mismatched (nested vs '
                               'full-validation rows, or no holdout accounting)'))
                continue
            _by_cond.setdefault((ts, bs), []).append(_delta)
            _est_by_cond.setdefault((ts, bs), set()).add(
                ra[ts, s].get('ppl_estimation_rows') == 'nested_validation_holdout')
        if _sk:
            _pair_skip.append((a, b, len(_sk), len(common), sorted({r for _s, r in _sk})))
        for (ts, bs), dl in sorted(_by_cond.items()):
            if len(_est_by_cond.get((ts, bs), ())) > 1:
                _pair_skip.append((a, b, len(dl), len(dl),
                                   ['mixed estimands across seeds (nested holdout '
                                    'vs full validation)']))
                continue
            st = paired_effect_stats(dl)
            _p = ('%.4f' % st['p_exact_signflip']) if math.isfinite(st['p_exact_signflip']) else 'n/a'
            _lab = '' if ts == bs == '' else f'   [{ts or '(untagged)'} vs {bs or '(untagged)'}]'
            print(f'  {a:22s} − {b:22s} = {st["mean"]:+7.2f} ± {st["std"]:5.2f}   (n={st["n"]}, p={_p}, {st["effect_vs_noise"]}){_lab}')
    if _pair_skip:
        print('  [skip] seed(s) rejected by the run_cfg/load gate (see `pair_reason`); the printed n counts only the pairs that cleared it —')
        for _a, _b, _nsk, _ncom, _why in _pair_skip:
            print(f'    {_a} − {_b}: {_nsk}/{_ncom} rejected — {_why}')
    print('=' * 78)

def _save_csv(agg, outdir):
    cols = ['variant', 'ppl_primary', 'ppl_mean', 'ppl_pooled', 'ppl_std', 'dPPL_vs_full_mean', 'dPPL_vs_full_std', 'dPPL_vs_sw128m_mean', 'dPPL_vs_sw128m_std', 'n_seeds', 'params', 'avg_dyn_block_len', 'boundary_f1_dyn', 'boundary_excess_dyn']
    rows = [{'variant': v, **{c: e.get(c, '') for c in cols[1:]}} for v, e in agg.items()]
    atomic_write_csv(os.path.join(outdir, 'aggregate.csv'), cols, rows)
