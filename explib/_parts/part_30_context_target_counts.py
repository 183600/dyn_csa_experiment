

def _context_target_counts(vb, ctx_len, *, max_positions=None):
    """Map each observed prefix to the next-token counts seen in this split.

    Used ONLY to flag that an apparently wrong target is not uniquely
    determined by its prefix: the same prefix occurs with a different next
    token elsewhere in the evaluated rows (label inconsistency or genuine
    ambiguity).  It does not decide which label is correct.

    ``max_positions`` bounds the scan for scale-up runs.  When the evaluated
    split is larger, equally spaced rows are scanned and the returned metadata
    says the scan was incomplete; an unbounded prefix dictionary over a
    multi-million-token validation split is not an acceptable hidden cost.
    """
    n_rows, seq = int(vb.shape[0]), int(vb.shape[1])
    span = seq - 1
    if span < 1:
        return {}, 0, {'n_rows_total': n_rows, 'n_rows_scanned': 0,
                       'n_positions_per_row': 0,
                       'n_positions_scanned': 0, 'max_positions': max_positions,
                       'complete': True, 'rows': []}
    ctx_len = max(1, min(int(ctx_len), span))
    # Keep the unbounded/no-subsample case as a range object: `list(range(...))`
    # materialised one Python integer per validation row *before* the
    # max_positions branch below could replace it, so a bounded scan over a
    # multi-million-row split still paid an O(n_rows) host allocation.
    scan_rows = range(n_rows)
    positions_per_row = span
    complete = True
    if max_positions is not None:
        _max = _exact_count(max_positions, minimum=1)
        if _max is None:
            raise ValueError(f'_context_target_counts needs a positive integer max_positions, got {max_positions!r}')
        total = n_rows * span
        if total > int(_max):
            complete = False
            # Scan complete rows when the budget covers at least one; when the
            # budget is smaller than one row, scan the prefix that fits instead
            # of silently exceeding `max_positions` (the bound must hold for a
            # single long sequence too, not only for many short rows).
            full_rows = int(_max) // max(1, span)
            if full_rows >= 1:
                n_scan = min(max(1, n_rows), full_rows)
                positions_per_row = span
            else:
                n_scan = min(max(1, n_rows), 1)
                positions_per_row = int(_max)
            if n_rows <= 0:
                scan_rows = []
            else:
                scan_rows = [int(r) for r in
                             np.unique(np.linspace(0, n_rows - 1, num=n_scan, dtype=np.int64))]
    out = {}
    for r in scan_rows:
        row = vb[r]
        for pos in range(positions_per_row):
            start = max(0, pos + 1 - ctx_len)
            key = row[start:pos + 1].tobytes()
            tgt = int(row[pos + 1])
            d = out.get(key)
            if d is None:
                out[key] = {tgt: 1}
            else:
                d[tgt] = d.get(tgt, 0) + 1
    meta = {'n_rows_total': n_rows, 'n_rows_scanned': len(scan_rows),
            'n_positions_per_row': positions_per_row,
            'n_positions_scanned': len(scan_rows) * positions_per_row,
            'max_positions': max_positions, 'complete': complete}
    if not complete:
        meta['rows'] = scan_rows
    return out, ctx_len, meta


def _token_frequency_table(flat_ids, vocab, *, chunk=1 << 24):
    """Exact token counts accumulated in bounded chunks.

    ``np.bincount(np.asarray(ids, dtype=np.int64))`` materialises an 8-byte
    copy of the ENTIRE corpus first: 40 GB on a 5B-token scale-up before the
    count table itself is allocated.  Integer chunk additions return the
    identical counts while keeping the temporary at O(chunk), so the rare-
    target diagnostic remains usable at the corpus sizes this project targets.
    """
    arr = np.asarray(flat_ids).reshape(-1)
    V = int(vocab)
    if V < 1:
        raise ValueError(f'_token_frequency_table needs vocab >= 1, got {vocab!r}')
    step = _exact_count(chunk, minimum=1)
    if step is None:
        raise ValueError(f'_token_frequency_table needs a positive chunk, got {chunk!r}')
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f'_token_frequency_table needs integer ids, got dtype {arr.dtype}')
    out = np.zeros(V, dtype=np.int64)
    for s in range(0, int(arr.size), int(step)):
        _part = arr[s:s + int(step)]
        if _part.size:
            # bincount accepts every integer width here; only the small chunk
            # is converted internally, never the whole corpus.
            out += np.bincount(_part, minlength=V)
    return out


@torch.inference_mode()
def analyze_errors(model, val_batch, vocab, device, *, tokenizer=None, top_n=50,
                   top_tokens=100, chunk=8, val_bnd=None, train_ids=None,
                   context_tokens=16, topk=5, context_scan_max_positions=None,
                   reference='bigram', reference_alpha=0.1,
                   reference_interpolation=0.5,
                   reference_max_fit_tokens=20_000_000):
    """Per-token / per-sequence error analysis for one checkpoint.

    Returns the worst sequences, worst tokens (with decoded context when a
    tokenizer is available), position buckets, rare-token flags and entropy
    statistics.  This is the input to `error_analysis.py`'s JSONL/Markdown
    artifacts; it deliberately runs on validation data only."""
    import heapq as _heapq
    was_training = model.training
    model.eval()
    vb = np.asarray(val_batch)
    if vb.ndim != 2 or vb.shape[1] < 2 or vb.shape[0] < 1:
        # An empty split has no token-id range (and no measurement).  Reject it
        # with this documented error before ``vb.min()`` raises a raw NumPy
        # "zero-size array" exception that looks like an infrastructure bug.
        raise ValueError(
            'analyze_errors needs a non-empty 2-D val_batch with >=2 tokens, '
            f'got {vb.shape}')
    n_rows, seq = int(vb.shape[0]), int(vb.shape[1])
    span = seq - 1
    V = int(vocab)
    # The artifact claims the next-token labels were range-checked.  Actually
    # enforce it here: a float array or an out-of-vocabulary id would either
    # crash cross_entropy half-way through a long scoring pass or, worse,
    # produce a partial artifact that looks measured.
    if not np.issubdtype(vb.dtype, np.integer):
        raise TypeError(f'analyze_errors needs integer token ids, got dtype {vb.dtype}')
    try:
        _min_id, _max_id = int(vb.min()), int(vb.max())
    except (TypeError, ValueError, OverflowError):
        raise ValueError('analyze_errors could not read the token-id range') from None
    if _min_id < 0 or _max_id >= V:
        raise ValueError(f'analyze_errors token ids out of range [{_min_id}, {_max_id}] for vocab {V}')
    # A negative token budget asks torch.topk for a negative k; zero is a
    # valid "inventory only the whole-split tally" request.
    _top_tokens = _exact_count(top_tokens, minimum=0)
    if _top_tokens is None:
        raise ValueError(f'analyze_errors needs a non-negative integer top_tokens, got {top_tokens!r}')
    top_tokens = int(_top_tokens)
    # A zero budget is a valid "do not inventory sequences" request; the old
    # max(1, int(top_n)) silently returned one row and accepted negatives.
    _top_n = _exact_count(top_n, minimum=0)
    if _top_n is None:
        raise ValueError(f'analyze_errors needs a non-negative integer top_n, got {top_n!r}')
    top_n = int(_top_n)
    _chunk = _exact_count(chunk, minimum=1)
    if _chunk is None:
        raise ValueError(f'analyze_errors needs a positive integer chunk, got {chunk!r}')
    chunk = int(_chunk)
    _ctx_tokens = _exact_count(context_tokens, minimum=0)
    if _ctx_tokens is None:
        raise ValueError(f'analyze_errors needs a non-negative integer context_tokens, got {context_tokens!r}')
    context_tokens = int(_ctx_tokens)
    # Prefix -> next-token evidence for the bad-case review.  Only built when
    # an inventory was requested, so a zero-budget call stays cheap.
    _ctx_counts = None
    _ctx_len_used = None
    _ctx_scan = None
    if top_tokens > 0:
        _ctx_counts, _ctx_len_used, _ctx_scan = _context_target_counts(
            vb, min(16, max(1, span)), max_positions=context_scan_max_positions)
    # "Rare target" is a claim about the TRAINING frequency.  An empty or
    # malformed training sample is not a frequency table of all-zero counts:
    # reading it that way would label every token `annotation_or_rare_target`
    # and manufacture a cause that has no evidence behind it.
    freq = None
    freq_problem = None
    rare_basis = 'unavailable: no train_ids'
    if train_ids is not None:
        try:
            _tr = np.asarray(train_ids)
        except Exception as exc:
            _tr = None
            freq_problem = f'unreadable train_ids ({type(exc).__name__})'
        if _tr is not None:
            if _tr.size == 0:
                freq_problem = 'empty train_ids: token frequencies are unmeasured'
            elif not np.issubdtype(_tr.dtype, np.integer) or np.issubdtype(_tr.dtype, np.bool_):
                freq_problem = f'train_ids dtype {_tr.dtype} is not an integer token type'
            else:
                _flat_tr = _tr.reshape(-1)
                _lo, _hi = int(_flat_tr.min()), int(_flat_tr.max())
                if _lo < 0 or _hi >= V:
                    freq_problem = f'train_ids out of range [{_lo}, {_hi}] for vocab {V}'
                else:
                    # Chunked integer accumulation: no whole-corpus int64
                    # copy at scale (see _token_frequency_table).
                    try:
                        freq = _token_frequency_table(_flat_tr, V)
                    except (MemoryError, ValueError, TypeError, OverflowError) as _fe:
                        freq = None
                        freq_problem = (f'train-frequency table could not be '
                                        f'accumulated ({type(_fe).__name__}: {_fe})')
                    if freq is not None and freq.shape[0] != V:
                        freq_problem = f'frequency table has {freq.shape[0]} rows, expected {V}'
                        freq = None
        if freq is not None:
            rare_basis = 'train_frequency'
        elif freq_problem:
            print(f'[analyze_errors] WARNING: rare-target flag disabled: {freq_problem}; '
                  'targets will not be labelled rare/annotation candidates')
    else:
        freq_problem = None
    # ------------------------------------------------------------------
    # Empirical reference floor for the bad-case review.
    #
    # A high CE alone cannot distinguish "the model missed a predictable
    # continuation" from "the label is intrinsically hard/noisy".  A smoothed
    # count model fitted on the TRAINING stream is a cheap control: when the
    # reference gets a token right and the neural model does not, that is
    # direct evidence of a model blind spot rather than annotation noise;
    # when both miss, the target needs a manual label check before it is
    # blamed on the model.  The reference is fixed (not tuned on validation)
    # so it cannot flatter itself on the reported rows.
    _ref_mode = 'none' if reference is None else str(reference).lower()
    if _ref_mode not in ('none', 'unigram', 'bigram'):
        raise ValueError(f'reference must be one of none/unigram/bigram, got {reference!r}')
    _ref_order = 1 if _ref_mode == 'unigram' else 2
    _ref_counts = None
    _ref_problem = None
    _ref_meta = None
    if _ref_mode != 'none':
        _ref_cap = None if reference_max_fit_tokens is None else _exact_count(reference_max_fit_tokens, minimum=1)
        if reference_max_fit_tokens is not None and _ref_cap is None:
            raise ValueError(f'reference_max_fit_tokens must be a positive integer or None, got {reference_max_fit_tokens!r}')
        if train_ids is None:
            _ref_problem = 'no train_ids: the reference floor cannot be fitted'
        else:
            try:
                _tr_raw = np.asarray(train_ids)
                if _tr_raw.size == 0:
                    raise ValueError('empty train_ids')
                if (_tr_raw.dtype == np.bool_ or _tr_raw.dtype == object
                        or not np.issubdtype(_tr_raw.dtype, np.integer)):
                    raise TypeError(f'train_ids dtype {_tr_raw.dtype} is not an integer token type')
                _tr_flat = _tr_raw.reshape(-1)
                _lo, _hi = int(_tr_flat.min()), int(_tr_flat.max())
                if _lo < 0 or _hi >= V:
                    raise ValueError(f'train_ids out of range [{_lo}, {_hi}] for vocab {V}')
                _n_source = int(_tr_flat.size)
                _fit_ids = _tr_flat if _ref_cap is None else _tr_flat[:_ref_cap]
                if _ref_order == 1:
                    _uni = np.bincount(np.asarray(_fit_ids, dtype=np.int64),
                                       minlength=V).astype(np.float64)
                    _ref_counts = (_uni, np.empty(0, dtype=np.int64),
                                   np.zeros(V, dtype=np.int64),
                                   np.empty(0, dtype=np.int64))
                else:
                    _ref_counts = _ngram_train_counts(np.asarray(_fit_ids, dtype=np.int64), V)
                _ref_meta = {
                    'mode': _ref_mode, 'order': _ref_order,
                    'alpha': float(reference_alpha),
                    'interpolation': (0.0 if _ref_order == 1 else float(reference_interpolation)),
                    'fit_token_policy': ('full' if _ref_cap is None else 'deterministic_prefix'),
                    'max_fit_tokens': _ref_cap,
                    'n_fit_tokens': int(_fit_ids.size),
                    'n_source_tokens': _n_source,
                    'fit_truncated': bool(_fit_ids.size < _n_source),
                }
            except Exception as _re:
                _ref_problem = f'{type(_re).__name__}: {_re}'
                _ref_counts = None
        if _ref_counts is None:
            print(f'[analyze_errors] WARNING: reference floor disabled: {_ref_problem}; '
                  'bad cases will not report model-vs-reference evidence')
    bnd_np = np.asarray(val_bnd) if val_bnd is not None else None
    if bnd_np is not None and bnd_np.shape != vb.shape:
        print(f'[analyze_errors] WARNING: boundary mask shape {bnd_np.shape} != val_batch shape {vb.shape} — boundary-token statistics are OMITTED rather than computed from a misaligned/truncated mask')
        bnd_np = None
    tok = tokenizer
    tok_warned = False

    def _decode(ids):
        nonlocal tok_warned
        if tok is None:
            return None
        try:
            return tok.decode([int(x) for x in ids])
        except Exception as e:
            if not tok_warned:
                print(f'[errors] WARNING: tokenizer decode failed ({type(e).__name__}: {e}); ids will be reported without text')
                tok_warned = True
            return None

    seq_rows = []
    tok_heap = []
    buckets = collections.defaultdict(lambda: {'n': 0, 'ce_sum': 0.0})
    n_targets = 0
    n_nonfinite = 0
    # Whole-split reference-floor accumulators.  They are only reported when
    # a fitted reference exists; an unavailable reference stays `None`, never
    # a fabricated zero.
    _ref_stats = {
        'n_paired': 0, 'model_ce_sum': 0.0, 'reference_ce_sum': 0.0,
        'n_model_correct': 0, 'n_reference_correct': 0,
        'n_both_correct': 0, 'n_both_wrong': 0,
        'n_model_only_correct': 0, 'n_reference_only_correct': 0,
        'n_both_wrong_unexplained': 0,
    }
    _ref_row_delta_sum = np.zeros(n_rows, dtype=np.float64)
    _ref_row_n = np.zeros(n_rows, dtype=np.int64)

    def _bucket(pos, tgt):
        frac = pos / max(1, span)
        if frac < 0.1:
            b = 'position:first10%'
        elif frac < 0.5:
            b = 'position:10-50%'
        elif frac < 0.9:
            b = 'position:50-90%'
        else:
            b = 'position:last10%'
        return b

    # Vectorised bucket accumulation: one bincount per chunk.
    _bucket_names = ['position:first10%', 'position:10-50%', 'position:50-90%',
                     'position:last10%', 'target:rare(<=5 train occurrences)',
                     'target:common', 'target:boundary-token']
    _nb = len(_bucket_names)
    _bsum = torch.zeros(_nb, dtype=torch.float64, device=device)
    _bcnt = torch.zeros(_nb, dtype=torch.long, device=device)
    _cause_names_all = ['annotation_or_rare_target', 'ambiguous_context',
                        'confident_model_blind_spot', 'target_outside_topk',
                        'mixed_unclear']
    _all_sum = torch.zeros(len(_cause_names_all), dtype=torch.float64, device=device)
    _all_cnt = torch.zeros(len(_cause_names_all), dtype=torch.long, device=device)
    _freq_t = None
    if freq is not None:
        try:
            _freq_t = torch.as_tensor(freq, dtype=torch.long, device=device)
        except Exception:
            _freq_t = None
    _kk = min(max(1, int(topk)), int(V))
    for r0 in range(0, n_rows, max(1, int(chunk))):
        r1 = min(n_rows, r0 + max(1, int(chunk)))
        ids = torch.as_tensor(np.ascontiguousarray(vb[r0:r1]), dtype=torch.long, device=device)
        h = model.trunk(ids)
        # Per-chunk reference scores keep the memory profile bounded: a full
        # (rows x length) NLL/top-1 pair for a large validation split would be
        # a hidden multi-hundred-MB allocation.  The count tables are built
        # once (capped), only the scoring is streamed.
        _ref_chunk = None
        if _ref_counts is not None:
            try:
                _ref_chunk = ngram_reference_score(
                    None, np.ascontiguousarray(vb[r0:r1, :seq]), V,
                    order=_ref_order, alpha=reference_alpha,
                    interpolation=reference_interpolation,
                    _counts=_ref_counts)
            except Exception as _rce:
                _ref_counts = None
                _ref_problem = f'reference scoring failed: {type(_rce).__name__}: {_rce}'
                print(f'[analyze_errors] WARNING: reference floor disabled: {_ref_problem}')
        row_nll = np.zeros(r1 - r0, dtype=np.float64)
        row_n = np.zeros(r1 - r0, dtype=np.int64)
        for s0 in range(0, span, 64):
            s1 = min(s0 + 64, span)
            lg = model.head(h[:, s0:s1, :])
            tg = ids[:, s0 + 1:s1 + 1]
            ce = F.cross_entropy(lg.transpose(1, 2), tg, reduction='none')
            _b, _s = int(ce.shape[0]), int(ce.shape[1])
            ce_f = ce.detach().float()
            finite = torch.isfinite(ce_f)
            if not bool(finite.all()):
                n_nonfinite += int((~finite).sum())
            ce_safe = torch.where(finite, ce_f, torch.zeros_like(ce_f))
            row_nll += ce_safe.double().sum(dim=1).cpu().numpy()
            row_n += finite.sum(dim=1).cpu().numpy()
            # Entropy from one log_softmax table (no separate softmax/log
            # copies); top-k works directly on the fp32 logits.
            lg_work = lg.float()
            logp = torch.log_softmax(lg_work, dim=-1)
            ent = -(logp * logp.exp()).sum(-1)
            del logp
            topv, topi = torch.topk(lg_work, _kk, dim=-1)
            pred = topi[..., 0]
            # With topk=1 there is no second logit, so the top-1/top-2 margin
            # is unmeasured.  A zero here is not a measured margin: the
            # `margin <= 0.5` / `>= 2.0` rules would then label every token
            # ambiguous (and make the blind-spot cause unreachable).  A NaN
            # is the explicit "not measured" value; the classifier already
            # skips non-finite margins and keeps the entropy rules.
            margin = ((topv[..., 0] - topv[..., 1]) if _kk > 1
                      else torch.full_like(topv[..., 0], float('nan')))
            del lg_work, topv
            # Bucket ids for every token at once.
            pos = torch.arange(s0, s1, device=device, dtype=torch.float64)[None, :]
            frac = pos / max(1, span)
            bidx = ((frac >= 0.1).to(torch.long) + (frac >= 0.5).to(torch.long)
                    + (frac >= 0.9).to(torch.long)).expand(_b, _s)
            if _freq_t is not None:
                _ft = _freq_t[tg.clamp(0, _freq_t.numel() - 1)]
                rare = _ft <= 5
            else:
                rare = torch.zeros((_b, _s), dtype=torch.bool, device=device)
            if bnd_np is not None and bnd_np.shape == vb.shape:
                _bnd = torch.as_tensor(np.ascontiguousarray(bnd_np[r0:r1, s0 + 1:s1 + 1]),
                                       dtype=torch.bool, device=device)
            else:
                _bnd = torch.zeros((_b, _s), dtype=torch.bool, device=device)
            _flat_b = bidx.reshape(-1)
            _flat_ce = ce_safe.reshape(-1).double()
            _flat_fin = finite.reshape(-1)
            _flat_rare = rare.reshape(-1)
            # A non-finite token is an unmeasured target: exclude it from the
            # bucket denominators instead of counting it as CE=0, which would
            # make a diverged evaluation look artificially good.
            _bsum[:4] += torch.bincount(_flat_b[_flat_fin], weights=_flat_ce[_flat_fin], minlength=4)
            _bcnt[:4] += torch.bincount(_flat_b[_flat_fin], minlength=4)
            _sel_rare = _flat_rare & _flat_fin
            _bsum[4] += _flat_ce[_sel_rare].sum()
            _bcnt[4] += int(_sel_rare.sum())
            _sel_common = (~_flat_rare) & _flat_fin
            _bsum[5] += _flat_ce[_sel_common].sum()
            _bcnt[5] += int(_sel_common.sum())
            _sel_bnd = _bnd.reshape(-1) & _flat_fin
            _bsum[6] += _flat_ce[_sel_bnd].sum()
            _bcnt[6] += int(_sel_bnd.sum())
            # Whole-split cause tally (not only the worst tokens).  The
            # priority order matches classify_token_error so the worst-token
            # inventory and the population shares use one definition.
            _uniform = math.log(max(V, 2))
            _flat_ent = ent.reshape(-1)
            _flat_margin = margin.reshape(-1)
            _flat_top = topi.reshape(-1, _kk)
            _flat_tg = tg.reshape(-1)
            _flat_in_topk = (_flat_top == _flat_tg[:, None]).any(dim=1)
            _amb = ((_flat_ent >= 0.8 * _uniform) | (_flat_margin <= 0.5)) & _flat_fin
            _blind = ((_flat_ent <= 0.5 * _uniform) & (_flat_margin >= 2.0)) & _flat_fin
            _m_rare = _flat_rare & _flat_fin
            _m_amb = (~_flat_rare) & _amb
            _m_blind = (~_flat_rare) & (~_amb) & _blind
            _m_out = (~_flat_rare) & (~_amb) & (~_blind) & (~_flat_in_topk) & _flat_fin
            _m_mixed = (~_flat_rare) & (~_amb) & (~_blind) & _flat_in_topk & _flat_fin
            _ref_nll = _ref_top = None
            if _ref_chunk is not None:
                _ref_nll = torch.as_tensor(
                    np.ascontiguousarray(_ref_chunk['nll'][:, s0:s1].reshape(-1)),
                    dtype=torch.float64, device=device)
                _ref_top = torch.as_tensor(
                    np.ascontiguousarray(_ref_chunk['top1_ids'][:, s0:s1].reshape(-1)),
                    dtype=torch.long, device=device)
                _ref_pair = torch.isfinite(_ref_nll) & _flat_fin
                _n_ref_pair = int(_ref_pair.sum())
                if _n_ref_pair:
                    _model_ok = (pred.reshape(-1) == _flat_tg) & _ref_pair
                    _ref_ok = (_ref_top == _flat_tg) & _ref_pair
                    _both_ok = _model_ok & _ref_ok
                    _both_bad = (~_model_ok) & (~_ref_ok) & _ref_pair
                    _model_only = _model_ok & (~_ref_ok)
                    _ref_only = (~_model_ok) & _ref_ok
                    _unexplained = _both_bad & (~_m_rare) & (~_m_amb)
                    _ref_stats['n_paired'] += _n_ref_pair
                    _ref_stats['model_ce_sum'] += float(_flat_ce[_ref_pair].sum())
                    _ref_stats['reference_ce_sum'] += float(_ref_nll[_ref_pair].sum())
                    _ref_stats['n_model_correct'] += int(_model_ok.sum())
                    _ref_stats['n_reference_correct'] += int(_ref_ok.sum())
                    _ref_stats['n_both_correct'] += int(_both_ok.sum())
                    _ref_stats['n_both_wrong'] += int(_both_bad.sum())
                    _ref_stats['n_model_only_correct'] += int(_model_only.sum())
                    _ref_stats['n_reference_only_correct'] += int(_ref_only.sum())
                    _ref_stats['n_both_wrong_unexplained'] += int(_unexplained.sum())
                    _delta = _flat_ce - _ref_nll
                    _row_delta = torch.where(_ref_pair.reshape(_b, _s),
                                             _delta.reshape(_b, _s),
                                             torch.zeros((_b, _s), dtype=torch.float64,
                                                         device=device))
                    _ref_row_delta_sum[r0:r1] += _row_delta.sum(dim=1).cpu().numpy()
                    _ref_row_n[r0:r1] += _ref_pair.reshape(_b, _s).sum(dim=1).cpu().numpy()
                    del _model_ok, _ref_ok, _both_ok, _both_bad, _model_only, _ref_only
                    del _unexplained, _delta, _row_delta, _ref_pair
            for _ci, _mask in enumerate((_m_rare, _m_amb, _m_blind, _m_out, _m_mixed)):
                _all_sum[_ci] += _flat_ce[_mask].sum()
                _all_cnt[_ci] += int(_mask.sum())
            del _flat_ent, _flat_margin, _flat_top, _flat_tg, _flat_in_topk
            del _amb, _blind, _m_rare, _m_amb, _m_blind, _m_out, _m_mixed
            n_targets += int(_flat_fin.sum())
            # Only the chunk's top candidates enter the Python heap; the
            # per-token loop is bounded by top_tokens, not by B*T.
            _sel_n = min(int(top_tokens), _b * _s)
            _pick_val, _pick_idx = torch.topk(torch.where(finite.reshape(-1), ce_f.reshape(-1),
                                                          torch.full_like(ce_f.reshape(-1), float('inf'))),
                                              _sel_n)
            _finite_pick = torch.isfinite(_pick_val)
            if bool(_finite_pick.any()):
                _pick_idx = _pick_idx[_finite_pick]
                _pi = _pick_idx.cpu().numpy()
                _pv = _pick_val[_finite_pick].cpu().numpy()
                _pr = (pred.reshape(-1).index_select(0, _pick_idx)).cpu().numpy()
                _pe = (ent.reshape(-1).index_select(0, _pick_idx)).cpu().numpy()
                _pm = (margin.reshape(-1).index_select(0, _pick_idx)).cpu().numpy()
                _pt = (tg.reshape(-1).index_select(0, _pick_idx)).cpu().numpy()
                _prr = (rare.reshape(-1).index_select(0, _pick_idx)).cpu().numpy()
                _ptop = (topi.reshape(-1, _kk).index_select(0, _pick_idx)).cpu().numpy()
                _pref_ce = (_ref_nll.index_select(0, _pick_idx).cpu().numpy()
                            if _ref_nll is not None else None)
                _pref_pred = (_ref_top.index_select(0, _pick_idx).cpu().numpy()
                              if _ref_top is not None else None)
                for _k in range(len(_pi)):
                    _lin = int(_pi[_k])
                    _pos = s0 + (_lin % _s)
                    rec = (float(_pv[_k]), r0 + (_lin // _s), _pos, int(_pt[_k]),
                           int(_pr[_k]), float(_pe[_k]), float(_pm[_k]),
                           [int(x) for x in _ptop[_k]], bool(_prr[_k]),
                           (float(_pref_ce[_k]) if _pref_ce is not None else None),
                           (int(_pref_pred[_k]) if _pref_pred is not None else None))
                    if len(tok_heap) < int(top_tokens):
                        _heapq.heappush(tok_heap, rec)
                    elif rec[0] > tok_heap[0][0]:
                        _heapq.heapreplace(tok_heap, rec)
            del lg, tg, ce, ce_f, ce_safe, finite, pred, margin, topi, ent
            del bidx, rare, _bnd, _flat_b, _flat_ce, _flat_fin
        for i in range(r1 - r0):
            n = int(row_n[i])
            if n <= 0:
                # Every target in this row was unmeasured; it contributes no
                # mean and must not enter the worst-sequence ranking as 0.
                continue
            ce_mean = float(row_nll[i] / n)
            seq_rows.append((ce_mean, r0 + i, float(row_nll[i]), n))
        del h, ids
    _bsum_cpu = _bsum.detach().cpu().tolist()
    _bcnt_cpu = _bcnt.detach().cpu().tolist()
    for _name, _n, _sm in zip(_bucket_names, _bcnt_cpu, _bsum_cpu):
        buckets[_name] = {'n': int(_n), 'ce_sum': float(_sm)}
    seq_rows.sort(key=lambda t: t[0], reverse=True)
    worst_seqs = []
    for ce_mean, row, nll, n in seq_rows[:top_n]:
        recs = [t for t in tok_heap if t[1] == row]
        worst_seqs.append({'row': int(row), 'mean_ce': ce_mean, 'nll': nll, 'n_tokens': int(n),
                           'top_worst_token_from_sample': (max(recs)[0] if recs else None),
                           'context': _decode(vb[row].tolist()),
                           'ids': [int(x) for x in vb[row].tolist()]})
    worst_tokens = []
    cause_counts = collections.Counter()
    n_ctx_ambiguous = 0
    for rec in sorted(tok_heap, key=lambda t: t[0], reverse=True)[:max(1, int(top_tokens))]:
        c, row, pos, tgt, pred, ent, margin, tids, rare, ref_ce, ref_pred = rec
        lo = max(0, pos - int(context_tokens))
        hi = min(span, pos + int(context_tokens) + 1)
        ctx_ids = [int(x) for x in vb[row, lo:hi + 1].tolist()]
        ctgt = int(vb[row, pos + 1])
        _triage = classify_token_error(ce=c, entropy=ent, margin=margin,
                                       rare_target=rare, target_id=tgt,
                                       topk_ids=tids, vocab=V)
        _ref_verdict = None
        _ref_correct = None
        _model_correct = None
        if ref_ce is not None and ref_pred is not None:
            _model_correct = bool(int(pred) == int(tgt))
            _ref_correct = bool(int(ref_pred) == int(tgt))
            if _model_correct and _ref_correct:
                _ref_verdict = 'both_correct'
            elif _model_correct:
                _ref_verdict = 'model_surpasses_reference'
            elif _ref_correct:
                _ref_verdict = 'reference_catches_model_miss'
                _triage['causes'].append('reference_catches_model_miss')
            else:
                _ref_verdict = 'both_miss_check_label'
                _triage['causes'].append('reference_also_miss')
        cause_counts[_triage['primary']] += 1
        _ctx_variants = None
        _ctx_support = None
        if _ctx_counts is not None:
            _cs = max(0, pos + 1 - int(_ctx_len_used))
            _cd = _ctx_counts.get(vb[row, _cs:pos + 1].tobytes())
            if _cd:
                _ctx_variants = len(_cd)
                _ctx_support = int(sum(_cd.values()))
        if _ctx_variants is not None and _ctx_variants > 1:
            n_ctx_ambiguous += 1
        _tr_count = None
        _tr_frac = None
        if freq is not None and 0 <= ctgt < int(len(freq)):
            _tr_count = int(freq[ctgt])
            _tr_frac = float(_tr_count / max(1, int(freq.sum())))
        worst_tokens.append({
            'row': int(row), 'position': int(pos), 'ce': c, 'target_id': tgt,
            'pred_id': pred, 'entropy': ent, 'margin': margin,
            'topk_ids': tids, 'topk_text': [_decode([x]) for x in tids],
            'rare_target': bool(rare),
            'target_train_count': _tr_count,
            'target_train_fraction': _tr_frac,
            'target_text': _decode([ctgt]),
            'pred_text': _decode([pred]),
            'context_text': _decode(ctx_ids),
            'context_ids': ctx_ids,
            'target_in_topk': _triage['target_in_topk'],
            'cause_primary': _triage['primary'],
            'causes': _triage['causes'],
            'reference_order': (_ref_order if ref_ce is not None else None),
            'reference_pred_id': ref_pred,
            'reference_pred_text': (_decode([ref_pred]) if ref_pred is not None else None),
            'reference_ce': ref_ce,
            'reference_correct': _ref_correct,
            'model_correct_vs_reference': _model_correct,
            'reference_verdict': _ref_verdict,
            'model_minus_reference_ce': ((c - ref_ce)
                                         if (ref_ce is not None and math.isfinite(ref_ce)) else None),
            'same_context_target_variants': _ctx_variants,
            'same_context_support': _ctx_support,
            'same_context_ambiguous': (bool(_ctx_variants > 1) if _ctx_variants is not None else None),
            'is_boundary_token': (bool(bnd_np[row, pos + 1]) if (bnd_np is not None and bnd_np.shape == vb.shape) else None),
        })
    _all_sum_cpu = _all_sum.detach().cpu().tolist()
    _all_cnt_cpu = _all_cnt.detach().cpu().tolist()
    _all_total = sum(_all_cnt_cpu)
    cause_counts_all = {}
    for _name, _n, _sm in zip(_cause_names_all, _all_cnt_cpu, _all_sum_cpu):
        cause_counts_all[_name] = {
            'n': int(_n), 'share': (float(_n) / _all_total if _all_total else None),
            'mean_ce': (float(_sm) / _n if _n else None)}
    out_buckets = {}
    for k, v in buckets.items():
        out_buckets[k] = {'n': int(v['n']),
                          'mean_ce': (v['ce_sum'] / v['n']) if v['n'] else None,
                          'perplexity': (math.exp(min(v['ce_sum'] / v['n'], 700.0)) if v['n'] else None)}
    ent_vals = [t[5] for t in tok_heap if math.isfinite(t[5])]
    model.train(was_training)
    _pooled_ce = (sum(x[2] for x in seq_rows) / n_targets) if n_targets else None
    _ref_comparison = None
    if _ref_mode != 'none':
        _n_paired = int(_ref_stats['n_paired'])
        if _n_paired <= 0:
            _ref_comparison = {
                'available': False,
                'reason': _ref_problem or 'no finite paired target positions',
                'fit': _ref_meta,
                'complete': False,
            }
        else:
            _model_ce = _ref_stats['model_ce_sum'] / _n_paired
            _ref_ce = _ref_stats['reference_ce_sum'] / _n_paired
            _row_ok = _ref_row_n > 0
            _row_means = (_ref_row_delta_sum[_row_ok] / _ref_row_n[_row_ok]
                          if bool(_row_ok.any()) else np.asarray([], dtype=np.float64))
            _row_se = None
            _row_ci = None
            if _row_means.size >= 2:
                _row_se = float(np.std(_row_means, ddof=1) / math.sqrt(_row_means.size))
                _row_ci = [float(np.mean(_row_means) - 1.96 * _row_se),
                           float(np.mean(_row_means) + 1.96 * _row_se)]
            _n_model_only = int(_ref_stats['n_model_only_correct'])
            _n_ref_only = int(_ref_stats['n_reference_only_correct'])
            _n_both_wrong = int(_ref_stats['n_both_wrong'])
            _n_unexplained = int(_ref_stats['n_both_wrong_unexplained'])
            _ref_comparison = {
                'available': True,
                'complete': bool(_ref_problem is None),
                'reason': _ref_problem,
                'fit': _ref_meta,
                'n_paired_targets': _n_paired,
                'model_ce': float(_model_ce),
                'reference_ce': float(_ref_ce),
                'excess_ce_model_minus_reference': float(_model_ce - _ref_ce),
                'n_model_correct': int(_ref_stats['n_model_correct']),
                'n_reference_correct': int(_ref_stats['n_reference_correct']),
                'n_both_correct': int(_ref_stats['n_both_correct']),
                'n_both_wrong': _n_both_wrong,
                'n_model_only_correct': _n_model_only,
                'n_reference_only_correct': _n_ref_only,
                'n_both_wrong_unexplained': _n_unexplained,
                'share_model_only_correct': (float(_n_model_only) / _n_paired),
                'share_reference_only_correct': (float(_n_ref_only) / _n_paired),
                'share_both_wrong': (float(_n_both_wrong) / _n_paired),
                'share_both_wrong_unexplained': (float(_n_unexplained) / _n_paired),
                'row_mean_excess_ce': (float(np.mean(_row_means)) if _row_means.size else None),
                'row_excess_ce_se': _row_se,
                'row_excess_ce_ci95': _row_ci,
                'n_rows_with_paired_targets': int(_row_means.size),
                'note': ('The reference is a fixed smoothed count model fitted on the '
                         'training stream (not tuned on these validation rows). A token that '
                         'the reference gets right and the model misses is a model blind-spot '
                         'candidate; one that BOTH miss needs a manual label check before it '
                         'is called annotation noise. The row-level CI groups by document and '
                         'is the uncertainty to quote, not the token-level count.'),
            }
    return {'kind': 'token_error_analysis', 'rows': n_rows, 'seq_len': seq,
            'n_targets': int(n_targets), 'n_nonfinite_tokens': int(n_nonfinite),
            'mean_ce': _pooled_ce,
            'pooled_ppl': (math.exp(min(_pooled_ce, 700.0)) if _pooled_ce is not None else None),
            'buckets': out_buckets,
            'worst_sequences': worst_seqs,
            'worst_tokens': worst_tokens,
            'cause_counts': dict(cause_counts),
            'cause_counts_all': cause_counts_all,
            'cause_counts_all_note': 'whole-split heuristic tally over finite targets; cause thresholds are triage heuristics, not labels',
            'reference_comparison': _ref_comparison,
            'reference_basis': (_ref_meta.get('mode') if _ref_meta else None),
            'reference_problem': _ref_problem,
            'cause_guide': {
                'annotation_or_rare_target': 'check the target annotation and the training frequency; a rare token can be correct but under-trained',
                'ambiguous_context': 'the distribution is near-uniform or the top-2 margin is tiny; inspect the context and decide whether the task is genuinely ambiguous',
                'confident_model_blind_spot': 'the model is confidently wrong; look for a systematic context feature it has not learned',
                'target_outside_topk': 'the target is outside the sampled top-k; check whether this is a tokenization/label-shift issue or a real model miss',
                'mixed_unclear': 'no single heuristic explains this case; read the full context before changing the model',
                'reference_catches_model_miss': 'a training-count reference predicts this target correctly while the model does not; treat it as a model blind spot (feature/context coverage), not annotation noise',
                'reference_also_miss': 'the model and the training-count reference both miss this target; check the label/prefix before blaming the model — this is the annotation-error or intrinsically hard population',
            },
            'entropy': {'max': (max(ent_vals) if ent_vals else None),
                        'mean': (sum(ent_vals) / len(ent_vals) if ent_vals else None)},
            'context_ambiguity': {
                'context_len': _ctx_len_used,
                'n_worst_reviewed': len(worst_tokens),
                'n_ambiguous_worst': int(n_ctx_ambiguous),
                'n_contexts_with_multiple_targets': (sum(1 for _d in _ctx_counts.values() if len(_d) > 1)
                                                     if _ctx_counts is not None else None),
                'n_positions_with_ambiguous_context': (sum(sum(_d.values()) for _d in _ctx_counts.values() if len(_d) > 1)
                                                       if _ctx_counts is not None else None),
                'scan': _ctx_scan,
                'scan_complete': (None if _ctx_scan is None else bool(_ctx_scan.get('complete'))),
                'note': ('same prefix in the evaluated validation rows has more than one observed next token; '
                         'this is label inconsistency or genuine ambiguity, not proof of an annotation error'),
                'scan_note': (None if (_ctx_scan is None or _ctx_scan.get('complete'))
                              else 'prefix-ambiguity counts are from a bounded row/position sample; '
                                   'they are not whole-split totals (see scan metadata)')},
            'tokenizer_available': tok is not None,
            'rare_target_basis': rare_basis,
            'rare_target_problem': freq_problem,
            'label_check': {'next_token_shift': 'target = ids[t+1]; checked range [0, vocab) before scoring',
                            'range_checked': True, 'min_id': _min_id, 'max_id': _max_id,
                            'n_targets': int(n_targets), 'n_out_of_range': 0}}


ERROR_BUCKET_POLICY = {
    'version': 'error-bucket-1',
    'split_seed': 20261027,
    'discovery_frac': 0.5,
    'min_clusters': 3,
    'min_tokens': 10,
    'direction': 'positive',
    'alpha': 0.05,
    'note': ('内部稳定性检查，不是外部复现：验证行是固定文本，不是从总体抽取的随机样本；'
             '全样本桶 p 值只在声明族内做 Holm 校正，仍是事后描述。'
             '发现/复现两半按文本行（cluster）整块切分，选中桶只在未参与选择的半边上报告效应。'),
}


def _error_bucket_key(value, *, where='bucket_ids'):
    """Canonical JSON-safe id for a bucket or a cluster (validation row)."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{where} must not contain boolean labels, got {value!r}')
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        _f = float(value)
        if not math.isfinite(_f) or _f != math.floor(_f):
            raise ValueError(f'{where} labels must be integer-like or non-empty strings, got {value!r}')
        return str(int(_f))
    if isinstance(value, str):
        _s = value.strip()
        if not _s:
            raise ValueError(f'{where} must not contain empty labels')
        return _s
    raise ValueError(f'{where} labels must be integer-like or strings, got {type(value).__name__}')


def _error_numeric_vector(values, *, where='deltas'):
    """1-D finite-aware float vector without silently accepting flags/complex."""
    try:
        raw = np.asarray(values)
    except Exception as exc:
        raise ValueError(f'{where} is not array-like ({type(exc).__name__}: {exc})') from None
    if raw.ndim != 1:
        raise ValueError(f'{where} must be 1-D, got shape {raw.shape}')
    if raw.dtype.kind in ('b', 'c', 'S', 'U', 'V'):
        raise ValueError(f'{where} must be real numeric measurements, got dtype {raw.dtype}')
    if raw.dtype.kind == 'O':
        out = []
        for i, x in enumerate(raw):
            if isinstance(x, (bool, np.bool_)):
                raise ValueError(f'{where}[{i}] is a boolean flag, not a measurement')
            try:
                f = float(x)
            except (TypeError, ValueError, OverflowError):
                out.append(float('nan'))
                continue
            out.append(f)
        return np.asarray(out, dtype=np.float64)
    try:
        return raw.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{where} cannot be converted to float ({type(exc).__name__})') from None


def _error_label_vector(values, n, *, where):
    if isinstance(values, (str, bytes)):
        raise ValueError(f'{where} must be a sequence of labels, not a single string')
    try:
        raw = list(values)
    except TypeError:
        raise ValueError(f'{where} must be a sequence of labels') from None
    if len(raw) != n:
        raise ValueError(f'{where} length {len(raw)} does not match deltas length {n}')
    return [_error_bucket_key(x, where=where) for x in raw]


def _error_cluster_stats(values, token_counts=None):
    """Equal-cluster mean with t CI and exact sign-flip p (two-sided)."""
    vals = np.asarray(values, dtype=np.float64)
    n = int(vals.size)
    if n == 0:
        return {'n_clusters': 0, 'n_tokens': 0, 'mean': None, 'sd': None,
                'se': None, 'ci95_low': None, 'ci95_high': None,
                'p_exact_signflip': None, 'p_floor': None,
                'ci_excludes_zero': False, 'direction': 0}
    # Finite per-cluster deltas near the float64 limit overflow a naive
    # sum/squared-deviation even when the mean and SD are representable.  The
    # guarded helper leaves ordinary panels bit-identical and rescales only an
    # extreme one.  If the SD is genuinely beyond float64, withhold the
    # interval rather than emitting inf (a strict JSON writer would turn it
    # into null after downstream consumers already accepted it).
    mean = _scaled_mean_or_raw(vals)
    if n >= 2:
        _ms = _scaled_mean_std_if_extreme(vals, ddof=1)
        sd = float(_ms[1]) if _ms is not None else float(vals.std(ddof=1))
        if not math.isfinite(sd):
            # The true SD is beyond float64; `None` is "not representable",
            # not a measured infinite spread.
            sd = None
        if sd is not None and math.isfinite(mean):
            se = sd / math.sqrt(n)
            half = float(_t_critical(0.95, n - 1)) * se
            lo, hi = mean - half, mean + half
            if not (math.isfinite(lo) and math.isfinite(hi)):
                lo = hi = None
        else:
            se = lo = hi = None
        ci_excludes_zero = bool(lo is not None and hi is not None
                                and (lo > 0.0 or hi < 0.0))
    else:
        sd = se = lo = hi = None
        ci_excludes_zero = False
    sf = exact_signflip_p(vals.tolist())
    p = sf.get('p')
    p = float(p) if isinstance(p, (int, float)) and math.isfinite(float(p)) else None
    n_tokens = (int(np.sum(token_counts)) if token_counts is not None and len(token_counts) else n)
    return {'n_clusters': n, 'n_tokens': n_tokens, 'mean': mean,
            'sd': sd, 'se': se, 'ci95_low': lo, 'ci95_high': hi,
            'p_exact_signflip': p, 'p_floor': sf.get('p_floor'),
            'ci_excludes_zero': ci_excludes_zero,
            'direction': (1 if mean > 0 else (-1 if mean < 0 else 0))}


def _error_stat_rank(values):
    """Average ranks for a small vector of bucket means (ties share a rank)."""
    vals = np.asarray(values, dtype=np.float64)
    order = np.argsort(vals, kind='mergesort')
    ranks = np.empty(vals.size, dtype=np.float64)
    ranks[order] = np.arange(1, vals.size + 1, dtype=np.float64)
    pos = 0
    while pos < vals.size:
        end = pos + 1
        while end < vals.size and vals[order[end]] == vals[order[pos]]:
            end += 1
        if end - pos > 1:
            ranks[order[pos:end]] = 0.5 * (pos + 1 + end)
        pos = end
    return ranks


def _error_rank_correlation(a, b):
    if len(a) < 3 or len(a) != len(b):
        return None
    ra = _error_stat_rank(a)
    rb = _error_stat_rank(b)
    sa, sb = float(ra.std(ddof=1)), float(rb.std(ddof=1))
    if not (math.isfinite(sa) and math.isfinite(sb)) or sa <= 0.0 or sb <= 0.0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _error_bucket_split_order(clusters, split_seed):
    """Deterministic whole-cluster order from SHA-256(split_seed, cluster).

    Cluster labels are validation-row ids (or arbitrary canonical strings).
    Hashing rather than slicing the first-occurrence order prevents a corpus
    whose rows are grouped by document from systematically putting one part of
    the document order into discovery and the other into replication.  The
    seed is part of the digest, so a different declared seed reshuffles the
    split while the same seed always reproduces it exactly.
    """
    seed = int(split_seed)
    prefix = f'{seed}\x00'.encode('utf-8')

    def _key(cluster):
        text = str(cluster)
        return (hashlib.sha256(prefix + text.encode('utf-8')).digest(), text)

    return sorted({str(c) for c in clusters}, key=_key)


def paired_error_bucket_inference(deltas, bucket_ids, cluster_ids, *, alpha=0.05,
                                  direction='positive', split_seed=None,
                                  discovery_frac=None, min_clusters=None,
                                  min_tokens=None, include_units=True):
    """Split-half, cluster-level inference for paired bad-case buckets.

    ``delta = CE_B - CE_A``.  The function first aggregates tokens to the
    cluster (validation-row) level, so a long row cannot outvote many short
    rows merely by contributing more tokens.  Bucket p-values on the full
    sample are Holm-adjusted within the declared bucket family, but they are
    post-hoc descriptions: the bucket family was inspected before selection.

    To separate discovery from confirmation, whole clusters are split into
    two disjoint blocks by a deterministic SHA-256 order keyed by the declared
    seed.  Hashing avoids turning a corpus whose rows are grouped by document
    into a systematic "early documents discover, late documents replicate"
    split.  The worst bucket is chosen on the discovery block only; its mean,
    t interval and exact sign-flip p are then reported on the untouched
    replication block.  ``replicated`` requires the same sign and a replication
    CI excluding zero.  This is an internal stability check (fixed validation
    text, not a random population sample), not independent external
    replication.
    """
    alpha = _validated_alpha(alpha)
    if direction not in ('positive', 'negative', 'two-sided'):
        raise ValueError(f"direction must be positive/negative/two-sided, got {direction!r}")
    split_seed = (ERROR_BUCKET_POLICY['split_seed'] if split_seed is None else split_seed)
    if isinstance(split_seed, (bool, np.bool_)):
        raise ValueError(f'split_seed must be an integer, got {split_seed!r}')
    try:
        split_seed = int(split_seed)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'split_seed must be an integer, got {split_seed!r}') from None
    discovery_frac = (ERROR_BUCKET_POLICY['discovery_frac']
                      if discovery_frac is None else discovery_frac)
    if isinstance(discovery_frac, (bool, np.bool_)):
        raise ValueError(f'discovery_frac must be in (0,1), got {discovery_frac!r}')
    try:
        discovery_frac = float(discovery_frac)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'discovery_frac must be in (0,1), got {discovery_frac!r}') from None
    if not (0.0 < discovery_frac < 1.0) or not math.isfinite(discovery_frac):
        raise ValueError(f'discovery_frac must be in (0,1), got {discovery_frac!r}')
    min_clusters = (ERROR_BUCKET_POLICY['min_clusters'] if min_clusters is None else min_clusters)
    _mc = _exact_count(min_clusters, minimum=2)
    if _mc is None:
        raise ValueError(f'min_clusters must be an integer >= 2, got {min_clusters!r}')
    min_clusters = int(_mc)
    min_tokens = (ERROR_BUCKET_POLICY['min_tokens'] if min_tokens is None else min_tokens)
    _mt = _exact_count(min_tokens, minimum=1)
    if _mt is None:
        raise ValueError(f'min_tokens must be a positive integer, got {min_tokens!r}')
    min_tokens = int(_mt)
    include_units = bool(include_units)

    d = _error_numeric_vector(deltas, where='deltas')
    n_input = int(d.size)
    bucket_labels = _error_label_vector(bucket_ids, n_input, where='bucket_ids')
    cluster_labels = _error_label_vector(cluster_ids, n_input, where='cluster_ids')
    finite = np.isfinite(d)
    n_finite = int(finite.sum())
    base = {
        'kind': 'paired_error_bucket_inference',
        'policy_version': ERROR_BUCKET_POLICY['version'],
        'alpha': alpha,
        'direction': direction,
        'min_clusters': min_clusters,
        'min_tokens': min_tokens,
        'discovery_frac': discovery_frac,
        'split_seed': split_seed,
        'n_tokens_input': n_input,
        'n_tokens': n_finite,
        'n_tokens_nonfinite': int(n_input - n_finite),
        'status': 'no_data',
        'reason': 'no finite paired deltas',
        'policy_note': ERROR_BUCKET_POLICY['note'],
        'buckets': {},
        'selected': None,
        'split': None,
        'rank_correlation': None,
    }
    if n_finite == 0:
        return base
    keep = np.flatnonzero(finite)
    d = d[keep]
    bucket_labels = [bucket_labels[i] for i in keep]
    cluster_labels = [cluster_labels[i] for i in keep]
    clusters = sorted(set(cluster_labels))
    bucket_names = sorted(set(bucket_labels))
    nc = len(clusters)
    cmap = {c: i for i, c in enumerate(clusters)}
    cidx = np.fromiter((cmap[c] for c in cluster_labels), dtype=np.int64, count=d.size)
    bucket_reports = {}
    units_by_bucket = {}
    for b in bucket_names:
        mask = np.fromiter((x == b for x in bucket_labels), dtype=bool, count=d.size)
        n_tok = int(mask.sum())
        sums = np.bincount(cidx[mask], weights=d[mask], minlength=nc)
        cnts = np.bincount(cidx[mask], minlength=nc).astype(np.int64)
        nz = np.flatnonzero(cnts > 0)
        means = sums[nz] / cnts[nz]
        units = [{'cluster': clusters[int(i)], 'mean_delta': float(sums[i] / cnts[i]),
                  'n_tokens': int(cnts[i])} for i in nz]
        units_by_bucket[b] = units
        st = _error_cluster_stats(means, cnts[nz])
        st['n_tokens'] = n_tok
        st['mean_delta_token_weighted'] = float(d[mask].mean()) if n_tok else None
        st['underpowered_tokens'] = bool(n_tok < min_tokens)
        st['cluster_units'] = units if include_units else None
        bucket_reports[b] = st
    holm = holm_adjust([(b, bucket_reports[b].get('p_exact_signflip')) for b in bucket_names],
                       alpha=alpha)
    for b in bucket_names:
        rec = holm.get(b) or {}
        bucket_reports[b]['p_holm'] = rec.get('p_holm')
        bucket_reports[b]['holm_rank'] = rec.get('rank')
        bucket_reports[b]['holm_reject'] = bool(rec.get('reject'))
        bucket_reports[b]['role'] = 'post_hoc_full_sample_descriptive'
    base['buckets'] = bucket_reports
    base['family'] = {'m': len(bucket_names), 'method': 'holm',
                      'declared_buckets': bucket_names,
                      'note': '全样本桶 p 为事后描述；选择校正来自发现/复现切分。'}
    base['n_clusters'] = nc
    if nc < 2 * min_clusters:
        base['status'] = 'insufficient_clusters'
        base['reason'] = (f'{nc} cluster(s) cannot be split into two halves of >= '
                          f'{min_clusters} clusters each')
        return base

    ordered = _error_bucket_split_order(clusters, split_seed)
    n_disc = int(round(discovery_frac * nc))
    n_disc = max(min_clusters, min(nc - min_clusters, n_disc))
    discovery = set(ordered[:n_disc])
    replication = set(ordered[n_disc:])
    base['split'] = {
        'seed': split_seed,
        'discovery_frac_requested': discovery_frac,
        'strategy': 'sha256(seed, cluster) whole-cluster order; discovery then replication',
        'cluster_order': ordered,
        'discovery_clusters': [c for c in ordered if c in discovery],
        'replication_clusters': [c for c in ordered if c in replication],
        'n_discovery': len(discovery),
        'n_replication': len(replication),
        'disjoint': not bool(discovery & replication),
    }

    for b in bucket_names:
        units = units_by_bucket[b]
        for half, keep_c in (('discovery', discovery), ('replication', replication)):
            sub = [u for u in units if u['cluster'] in keep_c]
            vals = [u['mean_delta'] for u in sub]
            ntok = [u['n_tokens'] for u in sub]
            st = _error_cluster_stats(vals, ntok)
            if sub:
                st['mean_delta_token_weighted'] = float(np.average(vals, weights=ntok))
            else:
                st['mean_delta_token_weighted'] = None
            st['underpowered_tokens'] = bool(st['n_tokens'] < min_tokens)
            bucket_reports[b][half] = st

    both = [b for b in bucket_names
            if bucket_reports[b]['discovery'].get('n_clusters', 0) >= min_clusters
            and bucket_reports[b]['replication'].get('n_clusters', 0) >= min_clusters]
    rank_corr = None
    if len(both) >= 3:
        rank_corr = _error_rank_correlation(
            [bucket_reports[b]['discovery']['mean'] for b in both],
            [bucket_reports[b]['replication']['mean'] for b in both])
    base['rank_correlation'] = {'spearman': rank_corr, 'n_buckets': len(both),
                                'note': '桶均值的两半秩相关；仅来自同一条验证文本。'}

    def _candidate_ok(st):
        return (st.get('n_clusters', 0) >= min_clusters
                and st.get('n_tokens', 0) >= min_tokens
                and st.get('mean') is not None
                and ((direction == 'positive' and st['mean'] > 0.0)
                     or (direction == 'negative' and st['mean'] < 0.0)
                     or (direction == 'two-sided' and st['mean'] != 0.0)))

    def _rank_key(b):
        st = bucket_reports[b]['discovery']
        m = float(st['mean'])
        if direction == 'positive':
            return (-m, b)
        if direction == 'negative':
            return (m, b)
        return (-abs(m), b)

    cand = [b for b in bucket_names if _candidate_ok(bucket_reports[b]['discovery'])]
    if not cand:
        base['status'] = 'no_candidate'
        base['reason'] = ('no bucket has a discovery-half mean in the requested '
                          f'direction {direction!r} with >= {min_clusters} clusters '
                          f'and >= {min_tokens} tokens')
        return base
    selected = sorted(cand, key=_rank_key)[0]
    ds = bucket_reports[selected]['discovery']
    rs = bucket_reports[selected]['replication']
    same_sign = bool(ds['mean'] is not None and rs['mean'] is not None
                     and ds['mean'] * rs['mean'] > 0.0)
    rep_ok = (rs.get('n_clusters', 0) >= min_clusters
              and rs.get('n_tokens', 0) >= min_tokens)
    replicated = bool(rep_ok and same_sign and rs.get('ci_excludes_zero'))
    if not rep_ok:
        rep_reason = 'insufficient_replication_clusters_or_tokens'
    elif not same_sign:
        rep_reason = 'replication_sign_flip'
    elif not rs.get('ci_excludes_zero'):
        rep_reason = 'replication_ci_includes_zero'
    else:
        rep_reason = 'same_sign_and_replication_ci_excludes_zero'

    # rank by the replication half, not by the half used for selection
    def _rank_key_rep(b):
        st = bucket_reports[b]['replication']
        m = float(st['mean'])
        if direction == 'positive':
            return (-m, b)
        if direction == 'negative':
            return (m, b)
        return (-abs(m), b)
    ranked_rep = [b for b in bucket_names if _candidate_ok(bucket_reports[b]['replication'])]
    ranked_rep = sorted(ranked_rep, key=_rank_key_rep)
    rank_rep = (ranked_rep.index(selected) + 1) if selected in ranked_rep else None
    ratio = (float(rs['mean']) / float(ds['mean'])
             if (ds['mean'] not in (None, 0.0) and rs['mean'] is not None) else None)

    base['selected'] = {
        'bucket': selected,
        'selection_role': 'discovery_half_only',
        'discovery_mean': ds['mean'],
        'discovery_ci95': [ds['ci95_low'], ds['ci95_high']],
        'discovery_n_clusters': ds['n_clusters'],
        'discovery_n_tokens': ds['n_tokens'],
        'replication_mean': rs['mean'],
        'replication_ci95': [rs['ci95_low'], rs['ci95_high']],
        'replication_n_clusters': rs['n_clusters'],
        'replication_n_tokens': rs['n_tokens'],
        'replication_p_exact_signflip': rs['p_exact_signflip'],
        'same_sign': same_sign,
        'replicated': replicated,
        'replication_reason': rep_reason,
        'shrinkage_replication_over_discovery': ratio,
        'rank_in_replication': rank_rep,
        'n_ranked_replication_buckets': len(ranked_rep),
        'full_sample_mean': bucket_reports[selected]['mean'],
        'full_sample_ci95': [bucket_reports[selected]['ci95_low'],
                             bucket_reports[selected]['ci95_high']],
        'full_sample_p_holm': bucket_reports[selected].get('p_holm'),
        'full_sample_role': 'post_hoc_full_sample_descriptive',
        'claim_boundary': ('桶是在发现半边上选出的；复现半边的区间/符号是选择校正后的读数。'
                           '两半都来自同一条固定验证文本，因此这是内部稳定性而非外部复现。'),
    }
    base['status'] = 'ok'
    base['reason'] = None
    return base


@torch.inference_mode()
def compare_token_errors(model_a, model_b, val_batch, vocab, device, *, tokenizer=None,
                         top_n=50, chunk=8, val_bnd=None, train_ids=None,
                         bucket_audit=True):
    """Paired per-token CE comparison between two checkpoints.

    Answers "on which positions does arm B lose to arm A" instead of only
    comparing aggregate PPL.  Positive ``delta = ce_b - ce_a`` means B is
    worse at that exact token."""
    vb = np.asarray(val_batch)
    if vb.ndim != 2 or vb.shape[1] < 2 or vb.shape[0] < 1:
        raise ValueError(
            'compare_token_errors needs a non-empty 2-D val_batch with '
            f'>=2 tokens, got {vb.shape}')
    if not np.issubdtype(vb.dtype, np.integer):
        raise TypeError(f'compare_token_errors needs integer token ids, got dtype {vb.dtype}')
    _V = int(vocab)
    _lo, _hi = int(vb.min()), int(vb.max())
    if _lo < 0 or _hi >= _V:
        raise ValueError(f'compare_token_errors token ids out of range [{_lo}, {_hi}] for vocab {_V}')
    n_rows, seq = int(vb.shape[0]), int(vb.shape[1])
    span = seq - 1
    _top_n = _exact_count(top_n, minimum=0)
    if _top_n is None:
        raise ValueError(f'compare_token_errors needs a non-negative integer top_n, got {top_n!r}')
    top_n = int(_top_n)
    _chunk = _exact_count(chunk, minimum=1)
    if _chunk is None:
        raise ValueError(f'compare_token_errors needs a positive integer chunk, got {chunk!r}')
    chunk = int(_chunk)

    def _matrix(model):
        was = model.training
        model.eval()
        out = np.empty((n_rows, span), dtype=np.float32)
        try:
            for r0 in range(0, n_rows, max(1, int(chunk))):
                r1 = min(n_rows, r0 + max(1, int(chunk)))
                ids = torch.as_tensor(np.ascontiguousarray(vb[r0:r1]), dtype=torch.long, device=device)
                h = model.trunk(ids)
                for s0 in range(0, span, 64):
                    s1 = min(s0 + 64, span)
                    lg = model.head(h[:, s0:s1, :])
                    tg = ids[:, s0 + 1:s1 + 1]
                    ce = F.cross_entropy(lg.transpose(1, 2), tg, reduction='none')
                    out[r0:r1, s0:s1] = ce.detach().float().cpu().numpy()
                del h, ids
        finally:
            model.train(was)
        return out

    ca = _matrix(model_a)
    cb = _matrix(model_b)
    # A paired comparison must use the SAME token positions in both arms.
    # Independent nanmeans silently compare different supports (e.g. arm B
    # looks better because its hardest tokens evaluated to NaN and were
    # dropped).  Report the common finite support and compute the headline
    # means/delta on it.
    fa = np.isfinite(ca)
    fb = np.isfinite(cb)
    paired = fa & fb
    n_finite_a = int(fa.sum())
    n_finite_b = int(fb.sum())
    n_paired = int(paired.sum())
    n_total = int(ca.size)
    mean_a_paired = float(np.mean(ca[paired])) if n_paired else None
    mean_b_paired = float(np.mean(cb[paired])) if n_paired else None
    diff = cb - ca
    bucket_inference = None
    bucket_family_reasons = {}
    if bucket_audit:
        # One family = one pre-declared set of buckets.  Token rows are the
        # clustering unit: a long row must not outvote several short rows just
        # because it contributes more next-token positions.
        _flat_delta = diff.reshape(-1)
        _row_ids = np.repeat(np.arange(n_rows, dtype=np.int64), span).tolist()
        _families = []
        _frac = np.arange(span, dtype=np.float64) / max(1, span)
        _pbin = ((_frac >= 0.1).astype(np.int64) + (_frac >= 0.5).astype(np.int64)
                 + (_frac >= 0.9).astype(np.int64))
        _pos_names = np.array(['position:first10%', 'position:10-50%',
                               'position:50-90%', 'position:last10%'], dtype=object)
        _families.append(('position', _pos_names[_pbin].tolist() * n_rows))
        if train_ids is None:
            bucket_family_reasons['target_rarity'] = 'unavailable: no train_ids'
        else:
            try:
                _tr = np.asarray(train_ids)
                if _tr.size == 0:
                    raise ValueError('empty train_ids: token frequencies are unmeasured')
                if _tr.dtype == np.bool_ or not np.issubdtype(_tr.dtype, np.integer):
                    raise TypeError(f'train_ids dtype {_tr.dtype} is not an integer token type')
                _tlo, _thi = int(_tr.min()), int(_tr.max())
                if _tlo < 0 or _thi >= _V:
                    raise ValueError(f'train_ids out of range [{_tlo}, {_thi}] for vocab {_V}')
                _freq = _token_frequency_table(_tr.reshape(-1), _V)
                _targets = vb[:, 1:].reshape(-1)
                _rare = _freq[_targets] <= 5
                _families.append(('target_rarity', np.where(
                    _rare, 'target:rare(<=5 train occurrences)', 'target:common').tolist()))
            except Exception as _exc:
                bucket_family_reasons['target_rarity'] = f'{type(_exc).__name__}: {_exc}'
        if val_bnd is None:
            bucket_family_reasons['boundary'] = 'unavailable: no val_bnd'
        else:
            try:
                _bnd = np.asarray(val_bnd)
                if _bnd.shape != vb.shape:
                    raise ValueError(f'boundary mask shape {_bnd.shape} != val_batch shape {vb.shape}')
                _bnd_flat = _bnd[:, 1:].reshape(-1).astype(bool)
                _families.append(('boundary', np.where(
                    _bnd_flat, 'target:boundary-token', 'target:non-boundary').tolist()))
            except Exception as _exc:
                bucket_family_reasons['boundary'] = f'{type(_exc).__name__}: {_exc}'
        try:
            bucket_inference = []
            for _fam, _labels in _families:
                _rep = paired_error_bucket_inference(
                    _flat_delta, _labels, _row_ids,
                    alpha=ERROR_BUCKET_POLICY['alpha'],
                    direction=ERROR_BUCKET_POLICY['direction'],
                    split_seed=ERROR_BUCKET_POLICY['split_seed'],
                    discovery_frac=ERROR_BUCKET_POLICY['discovery_frac'],
                    min_clusters=ERROR_BUCKET_POLICY['min_clusters'],
                    min_tokens=ERROR_BUCKET_POLICY['min_tokens'])
                _rep['family'] = _fam
                bucket_inference.append(_rep)
        except Exception as _exc:
            bucket_inference = None
            bucket_family_reasons['_all'] = f'{type(_exc).__name__}: {_exc}'
            print(f'[errors] WARNING: paired bucket inference failed ({type(_exc).__name__}: {_exc})')
    flat = np.where(paired, diff, -np.inf).reshape(-1)
    k = min(top_n, int(flat.size))
    if k <= 0:
        idx = np.array([], dtype=np.int64)
    else:
        idx = np.argpartition(flat, -k)[-k:]
        idx = idx[np.argsort(flat[idx])[::-1]]
    tok = tokenizer
    tok_warned = False

    def _decode(ids):
        nonlocal tok_warned
        if tok is None:
            return None
        try:
            return tok.decode([int(x) for x in ids])
        except Exception as e:
            if not tok_warned:
                print(f'[errors] WARNING: tokenizer decode failed ({type(e).__name__}: {e})')
                tok_warned = True
            return None

    out = []
    for i in idx:
        row, pos = int(i // span), int(i % span)
        if not (math.isfinite(float(ca[row, pos])) and math.isfinite(float(cb[row, pos]))):
            continue
        lo, hi = max(0, pos - 16), min(span, pos + 17)
        out.append({'row': row, 'position': pos,
                    'target_id': int(vb[row, pos + 1]),
                    'ce_a': float(ca[row, pos]), 'ce_b': float(cb[row, pos]),
                    'delta_b_minus_a': float(diff[row, pos]),
                    'target_text': _decode([int(vb[row, pos + 1])]),
                    'context_text': _decode([int(x) for x in vb[row, lo:hi + 1]])})
    return {'kind': 'paired_token_error_comparison', 'rows': n_rows, 'seq_len': seq,
            'n_tokens': n_total,
            'n_finite_a': n_finite_a, 'n_finite_b': n_finite_b,
            'n_paired_finite': n_paired,
            'paired_finite_frac': (float(n_paired) / n_total if n_total else None),
            # Headline means use the common finite support; the per-arm
            # all-finite means are kept separately for coverage inspection.
            'mean_ce_a': mean_a_paired, 'mean_ce_b': mean_b_paired,
            'mean_ce_a_all_finite': (float(np.mean(ca[fa])) if n_finite_a else None),
            'mean_ce_b_all_finite': (float(np.mean(cb[fb])) if n_finite_b else None),
            'mean_delta_b_minus_a': ((mean_b_paired - mean_a_paired)
                                     if n_paired else None),
            'n_nonfinite_a': int(n_total - n_finite_a),
            'n_nonfinite_b': int(n_total - n_finite_b),
            'b_worse_at_n': int(np.sum(paired & (diff > 0))),
            'a_worse_at_n': int(np.sum(paired & (diff < 0))),
            'worst_tokens_for_b': out,
            'bucket_inference': bucket_inference,
            'bucket_family_reasons': bucket_family_reasons,
            'bucket_inference_note': ('桶级推断以验证行为 cluster；发现/复现切分是内部稳定性检查，'
                                      '不是外部复现，也不把固定验证文本当作总体样本。')}


def _batch_arrays(batch, vocab=None):
    """Validate/convert an explicit ``(inputs, targets)`` probe batch.

    Task-specific training loops (e.g. NIAH) build their own next-token
    tensors, so the generic overfit probe must accept that exact batch rather
    than only a 1-D token stream.  Returns int64 numpy arrays; both are 2-D,
    same row count, inputs at least as long as targets (a next-token target
    table is often one column shorter)."""
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise ValueError('overfit batch must be a (inputs, targets) pair')
    x = np.asarray(batch[0])
    y = np.asarray(batch[1])
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f'overfit batch inputs/targets must be 2-D, got {x.shape} and {y.shape}')
    if x.shape[0] != y.shape[0]:
        raise ValueError(f'overfit batch row mismatch: inputs {x.shape[0]} vs targets {y.shape[0]}')
    if x.shape[1] < y.shape[1]:
        raise ValueError(f'overfit batch targets are longer than inputs ({y.shape[1]} > {x.shape[1]})')
    if not np.issubdtype(x.dtype, np.integer) or not np.issubdtype(y.dtype, np.integer):
        raise TypeError('overfit batch inputs/targets must be integer token ids')
    x = np.ascontiguousarray(x, dtype=np.int64)
    y = np.ascontiguousarray(y, dtype=np.int64)
    if vocab is not None:
        # -100 / other ignore sentinels are allowed in targets; every other id
        # must address a real embedding row, or the probe would silently test a
        # different (wrapped-index) model than the training path.
        xv = x.reshape(-1)
        if xv.size and (int(xv.min()) < 0 or int(xv.max()) >= int(vocab)):
            raise ValueError(f'overfit batch inputs span [{int(xv.min())}, {int(xv.max())}], outside vocab [0, {int(vocab)})')
    return x, y

def _param_change_sample(model, max_per_tensor=4096):
    """Small, deterministic per-tensor sample for the overfit probe.

    Comparing a fixed small sample avoids duplicating the whole parameter
    vector (and its grad) just to answer "did the optimizer move anything?".
    The sample spans the tensor uniformly, including its first and last
    element."""
    parts = []
    for p in model.parameters():
        flat = p.detach().reshape(-1)
        n = int(flat.numel())
        if n == 0:
            continue
        if n <= max_per_tensor:
            parts.append(flat)
        else:
            idx = torch.linspace(0, n - 1, steps=int(max_per_tensor),
                                 device=flat.device).round().long()
            parts.append(flat.index_select(0, idx))
    return torch.cat(parts) if parts else torch.empty(0)
