
def validation_row_split(n_rows, *, seed=20261020, n_folds=2,
                         selection_fold=0, evaluation_fold=1, groups=None):
    """Deterministically partition validation ROWS into disjoint chronological folds.

    Seeds make the training draws independent, but hyper-parameter selection
    and the final tuned comparison were historically measured on the same
    validation documents.  Disjoint seeds do not fix that: the selection is a
    function of the evaluation rows, so re-using them still yields an
    optimistically selected estimate.  This split makes the two stages use
    disjoint rows without changing any training run.

    Validation rows are consecutive chunks of a token stream, so their order is
    time order.  The folds are therefore **contiguous blocks of rows**, not a
    random row permutation: selection uses an earlier block and evaluation a
    later block (``selection_fold`` must precede ``evaluation_fold`` for a
    forward-in-time estimate).  Randomly interleaving rows would let the
    selection stage see chunks from later in the stream than the rows used to
    estimate the final contrast, and could put the same document on both sides
    of a row boundary many times over.

    ``seed`` deterministically chooses which end receives the remainder rows
    when ``n_rows`` is not divisible by ``n_folds``; it never shuffles rows
    across a fold boundary.  Both returned index lists are sorted and
    contiguous and never intersect.  For the default two-fold split the union
    is exactly ``range(n_rows)``; with more folds only the two selected fold
    blocks are returned (the unselected blocks are neither selection nor
    evaluation rows).  The metadata stores only counts/ranges/fingerprints
    (not the full index lists) so it stays small when the validation panel is
    scaled up.

    ``groups`` (one group/document id per row) makes the fold boundary
    group-aware: a balanced boundary that falls inside a group is snapped to
    the nearest legal boundary between groups, so no document/group can
    contribute rows to both the selection and evaluation fold.  When the
    requested number of folds cannot be formed without cutting a group, the
    split is refused instead of returning a leaking one.  Without ``groups``
    the historical row-level split is unchanged."""
    _parsed = {name: _as_seed_int(value) for name, value in (
        ('n_rows', n_rows), ('seed', seed), ('n_folds', n_folds),
        ('selection_fold', selection_fold), ('evaluation_fold', evaluation_fold))}
    _bad = [name for name, value in _parsed.items() if value is None]
    if _bad:
        raise ValueError(
            'validation_row_split: n_rows/seed/folds must be integers, got '
            + ', '.join(f'{name}={value!r}' for name, value in (
                ('n_rows', n_rows), ('seed', seed), ('n_folds', n_folds),
                ('selection_fold', selection_fold),
                ('evaluation_fold', evaluation_fold)) if name in _bad))
    n_rows = _parsed['n_rows']
    seed = _parsed['seed']
    n_folds = _parsed['n_folds']
    selection_fold = _parsed['selection_fold']
    evaluation_fold = _parsed['evaluation_fold']
    if n_rows < 2:
        raise ValueError(f'validation_row_split: need >=2 rows for disjoint folds, got {n_rows}')
    if n_folds < 2:
        raise ValueError(f'validation_row_split: n_folds must be >=2, got {n_folds}')
    if selection_fold == evaluation_fold:
        raise ValueError('validation_row_split: selection and evaluation folds must differ')
    if not (0 <= selection_fold < n_folds and 0 <= evaluation_fold < n_folds):
        raise ValueError(f'validation_row_split: folds must be in [0, {n_folds}), got '
                         f'selection={selection_fold}, evaluation={evaluation_fold}')
    # Only the earliest block selects and the later block estimates.  A swapped
    # request would estimate the past from a model tuned on the future; refuse
    # it instead of returning a reverse-time nested split.
    if selection_fold > evaluation_fold:
        raise ValueError(
            'validation_row_split: selection_fold must precede evaluation_fold; '
            f'got selection={selection_fold}, evaluation={evaluation_fold}')
    if n_folds > n_rows:
        raise ValueError(
            f'validation_row_split: n_folds={n_folds} exceeds n_rows={n_rows}; '
            'every fold must contain at least one row')
    # Balanced contiguous folds.  The remainder rows go to the beginning or
    # the end (chosen deterministically by the seed), never interleaved, so
    # each fold stays a contiguous chronological block.
    _base, _rem = divmod(n_rows, n_folds)
    _sizes = [_base] * n_folds
    if _rem:
        _hi_first = bool(_splitmix64_u64(seed + n_rows) & 1)
        _indices = range(_rem) if _hi_first else range(n_folds - 1, n_folds - 1 - _rem, -1)
        for i in _indices:
            _sizes[i] += 1
    _bounds = [0]
    for _size in _sizes:
        _bounds.append(_bounds[-1] + _size)
    _groups_aware = groups is not None
    _n_groups = None
    if _groups_aware:
        from data_guard import _group_key
        _g = np.asarray(groups)
        if _g.ndim != 1:
            _g = _g.reshape(-1)
        if _g.shape[0] != n_rows:
            raise ValueError(
                f'validation_row_split: groups has length {_g.shape[0]}, '
                f'expected {n_rows} (one group/document id per row)')
        _gk = [_group_key(v) for v in _g.tolist()]
        _n_groups = len(set(_gk))
        # A fold boundary is legal only where the group id changes.  Snap
        # each balanced boundary to the nearest legal one; leaving room for
        # every later fold keeps all folds non-empty.
        _legal = [i for i in range(1, n_rows) if _gk[i] != _gk[i - 1]]
        _desired = list(_bounds[1:-1])
        if len(_legal) < len(_desired):
            raise ValueError(
                'validation_row_split: cannot form the requested folds '
                'without cutting a group/document; the groups span too much '
                'of the validation stream')
        _chosen = []
        _prev = -1
        for _k, _target in enumerate(_desired):
            _remain = len(_desired) - 1 - _k
            _lo = _prev + 1
            _hi = len(_legal) - _remain - 1
            if _lo > _hi:
                raise ValueError(
                    'validation_row_split: cannot form the requested folds '
                    'without cutting a group/document')
            _seg = np.asarray(_legal[_lo:_hi + 1], dtype=np.int64)
            _j = _lo + int(np.argmin(np.abs(_seg - int(_target))))
            _chosen.append(int(_legal[_j]))
            _prev = _j
        _bounds = [0] + _chosen + [int(n_rows)]
        _sizes = [_bounds[_i + 1] - _bounds[_i] for _i in range(n_folds)]
    _sel_lo, _sel_hi = _bounds[selection_fold], _bounds[selection_fold + 1]
    _ev_lo, _ev_hi = _bounds[evaluation_fold], _bounds[evaluation_fold + 1]
    if _sel_hi <= _sel_lo or _ev_hi <= _ev_lo:
        raise ValueError(
            'validation_row_split: a selected fold is empty; choose folds '
            f'with at least one row (sizes={_sizes})')
    selection = list(range(_sel_lo, _sel_hi))
    evaluation = list(range(_ev_lo, _ev_hi))
    _sel_arr = np.asarray(selection, dtype=np.int64)
    _ev_arr = np.asarray(evaluation, dtype=np.int64)
    if _groups_aware:
        # A legal FOLD BOUNDARY only proves that no group is cut between two
        # adjacent rows.  It does not prove the two returned folds are
        # group-disjoint: a non-contiguous group id can reappear later in the
        # stream, putting the same document on both sides of the split.  The
        # contract is fail-closed, so refuse that split instead of returning
        # a leaking one.
        _sel_groups = {_gk[i] for i in selection}
        _ev_groups = {_gk[i] for i in evaluation}
        _shared_groups = _sel_groups & _ev_groups
        if _shared_groups:
            raise ValueError(
                'validation_row_split: a group/document contributes rows to '
                'both the selection and evaluation folds although every fold '
                'boundary is between group ids; group ids must be contiguous '
                'in the validation stream. Refusing a leaking split '
                f'({len(_shared_groups)} shared group(s), e.g. '
                f'{sorted(map(repr, _shared_groups))[:3]})')
    meta = {
        'policy': 'chronological_contiguous_row_folds',
        'seed': int(seed),
        'n_folds': int(n_folds),
        'selection_fold': int(selection_fold),
        'evaluation_fold': int(evaluation_fold),
        'n_rows': int(n_rows),
        'n_selection_rows': int(_sel_arr.size),
        'n_evaluation_rows': int(_ev_arr.size),
        'selection_row_range': [int(_sel_lo), int(_sel_hi) - 1],
        'evaluation_row_range': [int(_ev_lo), int(_ev_hi) - 1],
        'fold_sizes': [int(x) for x in _sizes],
        'fold_ranges': [[int(_bounds[i]), int(_bounds[i + 1]) - 1]
                        for i in range(n_folds)],
        'chronological': True,
        'selection_precedes_evaluation': True,
        'groups_aware': bool(_groups_aware),
        'n_groups': (int(_n_groups) if _n_groups is not None else None),
        'selection_row_fp': _ids_fp(_sel_arr),
        'evaluation_row_fp': _ids_fp(_ev_arr),
        'disjoint': bool(np.intersect1d(_sel_arr, _ev_arr).size == 0),
        'covers_all_rows': bool(_sel_arr.size + _ev_arr.size == n_rows),
        'note': 'selection rows choose the hyper-parameters; later evaluation rows are held out from that selection. Row fingerprints let the audit verify both stages used the declared folds.',
    }
    return selection, evaluation, meta

def cached_validation_row_groups(seq_len, n_train_tokens, vocab_size=8192,
                                val_seqs=512, n_rows=None, cache_dir=None):
    """Per-row document groups from the cached validation document-id sidecar.

    The tokenised validation stream is a concatenation of whole documents; a
    fixed row window can therefore straddle a document boundary.  A row-only
    chronological split can put two rows of the SAME document on opposite
    sides of the selection/evaluation cut.  The cache stores a per-token
    document id (``val_token_doc_ids_*.npy``; the document-level sidecar is
    kept for the leakage audit); this helper collapses it to one group per row
    and merges rows connected by any shared document.  The result can be
    passed to :func:`validation_row_split` as ``groups``.  A legacy cache with
    only the document-level sidecar cannot be expanded to tokens, so it is an
    explicit ``groups_aware=False`` fallback rather than a crash or a fake
    split.

    Returns ``(groups, meta)``.  ``groups`` is ``None`` when the sidecar is
    absent or cannot describe this slice; the caller must then fall back
    explicitly rather than silently claiming group-aware folds.  A present but
    inconsistent sidecar is an error, not a fallback: its ids would otherwise
    produce a leaking split.

    This covers the WikiText path (the frozen study).  FineWeb has its own
    cache layout and callers should pass an explicit group vector if they want
    group-aware folds there.
    """
    cache_dir = _wt_cache_dir(cache_dir)
    tag = f'v5_sl{seq_len}_cap{n_train_tokens}_v{vocab_size}_vs{val_seqs}'
    token_path = os.path.join(cache_dir, f'val_token_doc_ids_{tag}.npy')
    legacy_path = os.path.join(cache_dir, f'val_doc_ids_{tag}.npy')
    meta_path = os.path.join(cache_dir, f'meta_{tag}.json')
    _token_exists = os.path.exists(token_path)
    # Metadata that declares a token-level sidecar makes its absence corrupt
    # evidence, not an invitation to fall back to the document-level file.
    # The two files have different identities, so a missing token sidecar
    # must not silently become a degraded (groups_aware=False) split.
    if not _token_exists and os.path.exists(meta_path):
        try:
            _m0 = json.load(open(meta_path, encoding='utf-8'))
        except Exception:
            _m0 = None
        if isinstance(_m0, dict):
            _d0 = _m0.get('doc_ids')
            if (isinstance(_d0, dict)
                    and (_d0.get('validation_token_fp') is not None
                         or _d0.get('validation_token_count') is not None)):
                raise ValueError(
                    'cached_validation_row_groups: cache metadata declares a '
                    f'token-level document-id sidecar, but {token_path} is '
                    'missing; restore/rebuild it instead of falling back to a '
                    'different identity')
    path = token_path if _token_exists else legacy_path
    if n_rows is None:
        n_rows = int(val_seqs)
    _nr = _as_seed_int(n_rows)
    _sl = _as_seed_int(seq_len)
    _vs = _as_seed_int(val_seqs)
    if _nr is None or _nr < 1 or _sl is None or _sl < 1 or _vs is None or _vs < 1:
        raise ValueError('cached_validation_row_groups: n_rows/seq_len/val_seqs must be positive integers')
    if not os.path.exists(path):
        return None, {'groups_aware': False,
                      'reason': f'validation token document-id sidecar {token_path} is absent'}
    _legacy_sidecar = (path == legacy_path)
    if _legacy_sidecar:
        # The old file has one id per document.  Without per-document token
        # lengths it cannot be mapped onto rows; accepting it would either
        # crash (ids < tokens) or fabricate groups.
        return None, {'groups_aware': False,
                      'reason': ('the cache has only the legacy document-level '
                                 'val_doc_ids sidecar, which cannot be expanded to '
                                 'token-level row groups; rebuild the cache to get '
                                 'val_token_doc_ids')}
    try:
        doc_ids = np.load(path)
    except Exception as e:
        raise ValueError(f'cached_validation_row_groups: cannot read {path} '
                         f'({type(e).__name__}: {e})') from None
    doc_ids = np.asarray(doc_ids)
    if doc_ids.ndim != 1:
        raise ValueError(
            'cached_validation_row_groups: token document-id sidecar must be '
            f'one-dimensional, got shape {doc_ids.shape}')
    # Document ids are written as unsigned integers.  A float/object sidecar
    # cannot faithfully represent them (int(1.5) silently becomes 1, merging
    # two documents), and a negative signed id is not a document identity at
    # all.  Refuse both instead of manufacturing a group table.
    if doc_ids.dtype.kind not in 'iu':
        raise ValueError(
            'cached_validation_row_groups: token document-id sidecar must '
            f'contain integers, got dtype {doc_ids.dtype}')
    if doc_ids.dtype.kind == 'i' and doc_ids.size and int(doc_ids.min()) < 0:
        raise ValueError(
            'cached_validation_row_groups: token document-id sidecar contains '
            'negative ids; document ids must be non-negative')
    need = int(_nr) * int(_sl)
    if doc_ids.size < need:
        raise ValueError(
            f'cached_validation_row_groups: sidecar has {doc_ids.size} token ids, '
            f'but {_nr} rows x {_sl} tokens need {need}')
    # Bind the sidecar to the cache metadata when that metadata is available.
    # A swapped sidecar can be internally plausible while describing a
    # different token stream, which would make every fold guarantee false.
    _meta = {}
    if os.path.exists(meta_path):
        try:
            _meta = json.load(open(meta_path, encoding='utf-8'))
        except Exception as e:
            raise ValueError(f'cached_validation_row_groups: cannot parse {meta_path} '
                             f'({type(e).__name__}: {e})') from None
        if not isinstance(_meta, dict):
            raise ValueError(
                f'cached_validation_row_groups: {meta_path} must contain a JSON '
                f'object, got {type(_meta).__name__}')
        _mdoc = _meta.get('doc_ids') or {}
        if not isinstance(_mdoc, dict):
            raise ValueError(
                f'cached_validation_row_groups: {meta_path} doc_ids must be a '
                f'JSON object, got {type(_mdoc).__name__}')
        _want_fp = _mdoc.get('validation_token_fp')
        if _want_fp is None:
            # A new sidecar without a token-level fingerprint cannot be bound
            # to the stream; do not fall back to the document-level fp (it
            # describes a different array).
            raise ValueError('cached_validation_row_groups: cache metadata has no '
                             'validation_token_fp for the token-level sidecar')
        if _want_fp != _ids_fp(doc_ids):
            raise ValueError('cached_validation_row_groups: validation token '
                             'document-id sidecar does not match the cache metadata '
                             'fingerprint')
        _want_n = _as_seed_int(_mdoc.get('validation_token_count'))
        if _want_n is None or _want_n != int(doc_ids.size):
            raise ValueError('cached_validation_row_groups: sidecar length disagrees '
                             'with the cache metadata token-id count')
    # Union rows connected by any shared document.  A document can legitimately
    # span many rows; all of them become one group.  If a document id reappears
    # non-contiguously (a corrupt or misaligned sidecar), the merged group also
    # spans the intervening rows; validation_row_split's contiguity check then
    # refuses the split instead of treating it as two independent groups.
    parent = list(range(int(_nr)))

    def _find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    first_row = {}
    for r in range(int(_nr)):
        block = doc_ids[r * int(_sl):(r + 1) * int(_sl)]
        for d in np.unique(block).tolist():
            key = int(d)
            if key in first_row:
                _union(r, first_row[key])
            else:
                first_row[key] = r
    groups = [int(_find(r)) for r in range(int(_nr))]
    return groups, {
        'groups_aware': True,
        'group_source': 'cached_validation_token_doc_ids',
        'group_policy': 'rows connected by any shared whole document are one group',
        'n_groups': int(len(set(groups))),
        'n_rows': int(_nr),
        'seq_len': int(_sl),
        'val_seqs': int(_vs),
        'doc_ids_fp': _ids_fp(doc_ids[:need]),
    }


def run_corpus_row_groups(cfg, n_rows):
    """Row groups for a run config, or ``(None, reason)`` when unavailable."""
    name = cfg.get('dataset') or 'wikitext'
    if name != 'wikitext':
        return None, {'groups_aware': False,
                      'reason': f'no cached document-id groups for dataset {name!r}'}
    return cached_validation_row_groups(
        cfg['seq_len'], cfg['n_train_tokens'],
        vocab_size=cfg.get('vocab_size', 8192),
        val_seqs=cfg.get('val_seqs', 512), n_rows=n_rows,
        cache_dir=cfg.get('cache_dir', './wt103_cache'))


def tuning_row_split_spec(protocol):
    """Return the normalized row-split spec declared by a tuning protocol.

    ``None`` means the legacy full-validation selection path.  Artifacts that
    use the split record ``selection_row_split.applied=True`` so the audit can
    distinguish an actually nested run from a protocol that merely mentioned
    the idea."""
    spec = (protocol or {}).get('selection_row_split')
    if spec is None:
        return None
    if spec is False:
        return None
    if spec is True:
        spec = {}
    if not isinstance(spec, dict):
        raise ValueError(f'selection_row_split must be a mapping or false, got {spec!r}')
    if spec.get('enabled') is False:
        return None
    out = {
        'enabled': True,
        'seed': spec.get('seed', 20261020),
        'n_folds': spec.get('n_folds', 2),
        'selection_fold': spec.get('selection_fold', 0),
        'evaluation_fold': spec.get('evaluation_fold', 1),
    }
    for _key in ('seed', 'n_folds', 'selection_fold', 'evaluation_fold'):
        _value = _as_seed_int(out[_key])
        if _value is None:
            raise ValueError(
                f'selection_row_split.{_key} must be an integer, got {out[_key]!r}')
        out[_key] = _value
    return out

def record_subset_ppl(rec, rows):
    """Token-weighted pooled PPL of one record on a supplied subset of rows.

    The LM primary endpoint is the token-pooled NLL/PPL, not the arithmetic
    mean of per-seed PPLs.  Recomputing it on the held-out row fold is therefore
    the same estimand as the panel-level aggregation.  Returns ``None`` when
    the record lacks per-row accounting, the lengths disagree, or the row
    indices are out of range — never a silently different sample."""
    if not isinstance(rec, dict):
        return None
    nll = rec.get('nll_per_row')
    ntok = rec.get('n_tokens_per_row')
    if not isinstance(nll, (list, tuple)) or not isinstance(ntok, (list, tuple)):
        return None
    if len(nll) != len(ntok) or not nll:
        return None
    try:
        raw_rows = list(rows)
    except TypeError:
        return None
    if not raw_rows:
        return None
    # Row indices identify the scored documents.  int(1.5) would silently
    # select row 1 (a different sample), a boolean is a flag rather than an
    # index, and a repeated index would count the same document twice and
    # inflate the token denominator.  Parse exactly and require uniqueness.
    parsed = [_as_seed_int(v) for v in raw_rows]
    if any(i is None for i in parsed):
        return None
    if any(i < 0 or i >= len(nll) for i in parsed):
        return None
    if len(set(parsed)) != len(parsed):
        return None
    try:
        idx = np.asarray(parsed, dtype=np.int64)
    except (TypeError, ValueError, OverflowError):
        return None
    if idx.ndim != 1 or idx.size == 0:
        return None
    try:
        _nll = np.asarray([nll[int(i)] for i in idx], dtype=float)
    except (TypeError, ValueError, OverflowError, IndexError):
        return None
    _parsed_tok = [_exact_count(ntok[int(i)]) for i in idx]
    if any(v is None for v in _parsed_tok):
        return None
    if not np.isfinite(_nll).all():
        return None
    # Token counts are exact denominators.  Summing them through a float (or
    # narrowing them to int64) rounds sums above 2**53, so the reported
    # n_tokens would no longer describe the rows this estimate was computed
    # on.  Keep the Python integer total and only convert for the division.
    total_tok = int(sum(_parsed_tok))
    if total_tok <= 0:
        return None
    _nll_vals = [float(x) for x in _nll]
    try:
        nll_sum = float(math.fsum(_nll_vals))
    except (OverflowError, ValueError):
        nll_sum = None
    if nll_sum is None or not math.isfinite(nll_sum):
        # The individual NLLs are finite, but their sum can overflow while the
        # pooled rate is still representable; fall back to the scaled ratio.
        nll_sum_out = None
        nll_mean = _pooled_nll_ratio(_nll_vals, _parsed_tok)
    else:
        nll_sum_out = nll_sum
        try:
            nll_mean = nll_sum / total_tok
        except (OverflowError, ValueError, ZeroDivisionError):
            nll_mean = _pooled_nll_ratio(_nll_vals, _parsed_tok)
        if nll_mean is None or not math.isfinite(nll_mean):
            nll_mean = _pooled_nll_ratio(_nll_vals, _parsed_tok)
    if nll_mean is None or not math.isfinite(nll_mean):
        return None
    return {'ppl': _safe_exp_ppl(nll_mean), 'nll': nll_mean,
            'nll_sum': nll_sum_out, 'n_tokens': total_tok,
            'n_rows': int(idx.size)}

# 基线调参协议：主方法与基线必须获得相同的搜索范围、在同一训练集上拟合、
# 在验证集上按同一指标选择。`baseline_tune.py` 读取本表；审计读取落盘的
# `baseline_tuning.json`，没有该产物时只报「不可核验」，不会把未调参的
# 弱基线当成已验证的对照。
BASELINE_TUNING_PROTOCOL = {
    'selection_metric': 'ppl',
    'selection_split': 'validation',
    'direction': 'lower_is_better',
    # Seed-disjoint selection is not enough: hyper-parameters and the final
    # estimate were still measured on the SAME validation documents.  Tune on
    # one deterministic row fold and report the tuned contrast on the disjoint
    # fold; the records keep the full-validation PPL for context only.
    'selection_row_split': {'enabled': True, 'seed': 20261020, 'n_folds': 2,
                            'selection_fold': 0, 'evaluation_fold': 1},
    'search_seed': 100,
    # Hyper-parameter search is a controlled design choice, not an implicit
    # pile of trials.  Declare the design so the audit can distinguish an
    # intentionally marginal/OFAT search from an accidentally incomplete
    # factorial one.  This grid is hand-designed (bracketing lr on both
    # sides and varying wd/warmup), NOT a full cartesian product.
    'search_design': 'hand_designed_grid',
    'search_design_note': ('8 hand-designed trials bracket lr (5e-5..6e-4), weight decay '
                           '(0.01..0.2) and warmup (50/100); interaction cells are not '
                           'exhaustive, so a marginal winner is not a jointly tuned optimum.'),
    # Selection is on the MEAN validation PPL over this declared seed set, not
    # on one seed's draw.  A single-seed selection can pick a config because
    # of seed noise; every arm and every trial uses the same set.  The search
    # partition is deliberately DISJOINT from `final_seeds`: evaluating the
    # selected config on the same draws that selected it is selection-biased.
    # The final partition has 8 seeds because the tuned family has m=4
    # declared contrasts, whose smallest attainable Holm p first drops below
    # alpha at n=8 (with 3 seeds the final panel can never confirm anything).
    'search_seeds': [100, 101, 102],
    'final_seeds': [0, 1, 2, 3, 4, 5, 6, 7],
    'arms': ['full_matched', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic',
             'csa_fixed_rope', 'full_rope'],
    # The v6 arms and the frozen v12 confirmatory arms have DIFFERENT
    # parameter-matching references (csa_dynamic vs csa_fixed_rope), so they
    # cannot share one panel config.  `arm_groups` runs each group under its
    # own matched/mlp_match_ref while keeping the SAME declared trials, the
    # SAME search-seed set and the SAME selection split, so the equal-search
    # contract holds across the union of arms.
    'arm_groups': [
        {'arms': ['full_matched', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic'],
         'matched': ['full_matched', 'full_sw128_matched'], 'mlp_match_ref': 'csa_dynamic',
         'method_arm': 'csa_dynamic'},
        {'arms': ['csa_fixed_rope', 'full_rope'],
         'matched': ['full_rope'], 'mlp_match_ref': 'csa_fixed_rope',
         'method_arm': 'csa_fixed_rope'},
    ],
    # Budget-matched intervention controls share the csa_fixed_rope recipe and
    # training budget, so their equal-search evidence is inherited from the arm
    # they intervene on.  The audit resolves coverage through this map.
    'inherits_tuning': {'csa_fixed_rope_randidx': 'csa_fixed_rope',
                        'csa_fixed_rope_allblocks': 'csa_fixed_rope'},
    # A coarse one-factor-at-a-time grid can crown a config that only looks
    # best because the true optimum lies outside the searched range.  Keep the
    # original four trials in the same order (historical artifacts remain a
    # declared subset) but bracket the learning rate on both sides and widen
    # the weight-decay/warmup coverage.  The grid-coverage audit below warns
    # when a selected value sits on the boundary.
    'trials': [
        {'lr': 0.0001, 'weight_decay': 0.1, 'warmup': 50},
        {'lr': 0.0003, 'weight_decay': 0.1, 'warmup': 50},
        {'lr': 0.0003, 'weight_decay': 0.05, 'warmup': 50},
        {'lr': 0.0003, 'weight_decay': 0.1, 'warmup': 100},
        {'lr': 0.00005, 'weight_decay': 0.1, 'warmup': 50},
        {'lr': 0.0006, 'weight_decay': 0.05, 'warmup': 50},
        {'lr': 0.0002, 'weight_decay': 0.2, 'warmup': 100},
        {'lr': 0.0003, 'weight_decay': 0.01, 'warmup': 100},
    ],
    'note': 'The declared grid brackets the learning rate on both sides of the historical best (5e-5 to 6e-4) and varies weight decay from 0.01 to 0.2; grid-coverage auditing flags a selected config that sits on a boundary, which means the search range is too narrow. Every arm gets the same declared trials, the same search-seed set and the same selection split. Selection uses TOKEN-POOLED validation PPL over a deterministic selection row fold, averaged across search_seeds (std recorded); a single-seed draw can win on noise. The final tuned contrast is measured on the DISJOINT evaluation row fold, so neither the hyper-parameters nor their reported estimate re-use the same validation documents. The search and final SEED partitions are also disjoint (search 100-102, final 0-7); 8 final seeds are the smallest n whose minimum attainable Holm p for the m=4 tuned family is below alpha. Final records keep the full-validation PPL for context but the contrast uses the held-out row fold. A tuning trial may never read the held-out test split; test is used once after selection. Contrasts are declared per parameter-matching group (each group\'s method_arm vs the other arms in that group), and unmeasurable declared pairs stay in the Holm family with p=1 instead of shrinking it. The rope group covers the frozen v12 confirmatory arms: without equal-search evidence for csa_fixed_rope/full_rope, the confirmatory comparison cannot be quoted as "beats a tuned baseline" — it is only a comparison at the frozen default hyperparameters.',
}

# 最简单基线协议（先立地板，再谈复杂模型）：语言模型的 trivial floor 是
# 在**训练集**上数出来的 unigram / bigram 计数模型，平滑与插值系数只在
# **验证集**上选择。主方法如果不能显著优于这个地板，就不值得进入与稠密
# Transformer 的机制对比；反过来，复杂模型只战胜未调参的弱基线也不算
# 证据。`baseline_tune.py simple` 生成 `simple_baseline.json`，审计函数
# 核验搜索范围、选择集与选中项；`methodology_audit` 会把它与面板的主方法
# 读数并排报告（只有 val 指纹一致时才比较）。
SIMPLE_BASELINE_POLICY = {
    'selection_metric': 'ppl',
    'selection_split': 'validation',
    'direction': 'lower_is_better',
    # Mirror the neural panel's nested validation: select smoothing on an
    # earlier chronological row fold and report the floor on a disjoint later
    # fold.  Tuning and reporting on the same rows would let the count model
    # see the panel's held-out rows and make the "beats floor" test unfair.
    'selection_row_split': {'enabled': True, 'seed': 20261020, 'n_folds': 2,
                            'selection_fold': 0, 'evaluation_fold': 1},
    'arms': {
        'unigram': {'order': 1, 'description': 'training-set unigram counts with add-alpha smoothing (a linear/count model)'},
        'bigram': {'order': 2, 'description': 'training-set bigram counts interpolated with the unigram model'},
    },
    'grids': {
        # Wide but cheap: the count tables are shared across the whole grid,
        # so extra smoothing levels cost only another scoring pass.  A narrow
        # grid can crown a floor that is merely the least-bad tested value.
        'unigram': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]},
        'bigram': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                   'interpolation': [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]},
    },
    # The bigram count table is O(unique pairs): materialising every pair of a
    # multi-billion-token B-tier corpus would need tens of GB before the
    # baseline even runs.  Fit on a deterministic PREFIX instead and disclose
    # the truncation, so the floor stays feasible and comparable when the
    # project scales.  The default WikiText-103 budget (8M tokens) is below
    # this cap, so its behavior is unchanged.
    'max_fit_tokens': 20_000_000,
    'fit_token_policy': 'deterministic_prefix',
    # The count-model grid is small and cheap, so it is declared as a complete
    # cartesian search rather than an OFAT shortcut.  The audit enforces this
    # claim: every declared (alpha, interpolation) cell must be present.
    'search_design': 'full_factorial',
    'search_design_note': ('unigram: every alpha; bigram: every alpha x interpolation cell. '
                           'The count tables are shared, so full coverage is cheap.'),
    'note': 'Counts are fitted on the training token stream only (a deterministic prefix when max_fit_tokens is exceeded, with the truncation recorded); smoothing/interpolation is selected on validation PPL. The held-out test split is never read. This is a sanity floor, not a tuned neural baseline.',
}

def _ngram_token_ids(values, label):
    """Exact integer token ids for the count baseline.

    Casting a float/boolean/object array through int64 would silently score
    different tokens (1.7 -> 1, True -> 1).  The count model is a data-path
    baseline, so it must reject such input rather than manufacture a floor
    from truncated ids.
    """
    arr = np.asarray(values)
    if (arr.dtype == np.bool_ or arr.dtype == object
            or not np.issubdtype(arr.dtype, np.integer)):
        raise TypeError(f'ngram baseline: {label} ids must be integers, '
                        f'got dtype {arr.dtype}')
    return arr.reshape(-1)

def _ngram_vocab_size(value):
    v = _as_seed_int(value)
    if v is None or v <= 0:
        raise ValueError(f'ngram baseline: vocab_size must be a positive integer, got {value!r}')
    return v

def _ngram_train_counts(train_ids, vocab_size):
    """Return (unigram counts, bigram key counts, bigram context counts)."""
    ids = np.asarray(_ngram_token_ids(train_ids, 'train'), dtype=np.int64)
    vocab_size = _ngram_vocab_size(vocab_size)
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= vocab_size):
        raise ValueError('ngram baseline: train ids outside [0, vocab_size)')
    uni = np.bincount(ids, minlength=vocab_size).astype(np.float64)
    if ids.size < 2:
        return uni, np.empty(0, dtype=np.int64), np.zeros(vocab_size, dtype=np.int64), np.empty(0, dtype=np.int64)
    prev, nxt = ids[:-1], ids[1:]
    keys = prev * vocab_size + nxt
    uniq, counts = np.unique(keys, return_counts=True)
    ctx = np.bincount(prev, minlength=vocab_size).astype(np.int64)
    return uni, uniq, ctx, counts

def _ngram_eval_parts(eval_ids):
    arr = np.asarray(eval_ids)
    if (arr.dtype == np.bool_ or arr.dtype == object
            or not np.issubdtype(arr.dtype, np.integer)):
        raise TypeError(f'ngram baseline: eval ids must be integers, got dtype {arr.dtype}')
    arr = np.asarray(arr, dtype=np.int64)
    if arr.ndim == 1:
        return arr, arr[:-1], arr[1:]
    if arr.ndim == 2:
        return arr.ravel(), arr[:, :-1].ravel(), arr[:, 1:].ravel()
    raise ValueError('eval ids must be 1-D or 2-D (sequences x length)')

def ngram_lm_score(train_ids, eval_ids, vocab_size, order=1, alpha=0.1, interpolation=1.0, _counts=None):
    """Token-weighted NLL/PPL for a tuned count model.

    order=1 is the unigram model (a context-free linear/count baseline);
    order=2 conditions each token on the previous token and interpolates the
    smoothed conditional with the unigram distribution.  Row boundaries are
    respected: the first token of every evaluation row is scored by the
    unigram model, never by the previous row's last token.  The effect size is
    token-weighted, matching the LM metric policy."""
    # When the caller hands in precomputed count tables, `train_ids` is not
    # needed.  Converting the whole fit prefix to int64 on every call made a
    # per-evaluation-row loop O(n_rows * n_fit_tokens) at scale (the FineWeb
    # cache is uint16/int32), even though the tables were already built once.
    ids = (np.asarray(_ngram_token_ids(train_ids, 'train'), dtype=np.int64)
           if _counts is None else None)
    vocab_size = _ngram_vocab_size(vocab_size)
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError('alpha must be a positive finite number, got a boolean')
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be a positive finite number')
    if isinstance(order, (bool, np.bool_)) or order not in (1, 2):
        raise ValueError('order must be 1 (unigram) or 2 (bigram)')
    all_eval, prev, nxt = _ngram_eval_parts(eval_ids)
    if all_eval.size == 0:
        return {'order': int(order), 'alpha': alpha, 'interpolation': float(interpolation),
                'n_tokens': 0, 'nll': float('nan'), 'ppl': float('nan')}
    if all_eval.size and (int(all_eval.min()) < 0 or int(all_eval.max()) >= vocab_size):
        raise ValueError('ngram baseline: eval ids outside [0, vocab_size)')
    # The count tables are a pure function of (train_ids, vocab_size) and do
    # not depend on the smoothing hyper-parameters, so a tuning grid can build
    # them once and hand the same tables to every trial (identical values).
    counts = _ngram_train_counts(ids, vocab_size) if _counts is None else _counts
    uni_counts = counts[0]
    denom = float(uni_counts.sum()) + alpha * vocab_size
    uni_prob = (uni_counts + alpha) / denom
    log_uni = np.log(np.maximum(uni_prob, np.finfo(float).tiny))
    if order == 1:
        nll = float(-log_uni[all_eval].sum() / all_eval.size)
        return {'order': 1, 'alpha': alpha, 'interpolation': 0.0,
                'n_tokens': int(all_eval.size), 'nll': nll, 'ppl': float(_safe_exp_ppl(nll))}
    _uni, uniq, ctx, pair_counts = counts
    if isinstance(interpolation, (bool, np.bool_)):
        raise ValueError('interpolation must be a finite number in [0, 1]')
    lam = float(interpolation)
    if not (math.isfinite(lam) and 0.0 <= lam <= 1.0):
        raise ValueError('interpolation must be in [0, 1]')
    # First token of each row: unigram only (no cross-row conditioning).  A
    # 1-D evaluation stream is ONE row, so only its first token is
    # unconditioned; the remaining tokens are scored by the conditional term.
    first = np.asarray(eval_ids, dtype=np.int64)
    first = first[:1] if first.ndim == 1 else first[:, 0]
    first = first.ravel()
    nll_sum = float(-log_uni[first].sum())
    if prev.size:
        if lam > 0.0:
            keys = prev * vocab_size + nxt
            if uniq.size:
                idx = np.searchsorted(uniq, keys)
                idx_clip = np.minimum(idx, uniq.size - 1)
                hit = (idx < uniq.size) & (uniq[idx_clip] == keys)
                c_pair = np.where(hit, pair_counts[idx_clip], 0).astype(np.float64)
            else:
                # A training stream with fewer than two tokens has no bigram
                # counts.  Indexing an empty pair_counts table (even through
                # np.where) raises; every transition is unseen instead.
                c_pair = np.zeros(keys.shape, dtype=np.float64)
            c_ctx = ctx[prev].astype(np.float64)
            # Add-alpha smoothing of the conditional is defined even when the
            # context was never seen: 0/0 becomes alpha/(alpha*V) = 1/V.  The
            # old fallback to the unigram probability silently skipped the
            # interpolation term for those contexts, making the bigram floor
            # score unseen transitions like a pure unigram model (and lower
            # its PPL).  Interpolate the smoothed conditional in every case.
            p_bg = (c_pair + alpha) / (c_ctx + alpha * vocab_size)
            p = lam * p_bg + (1.0 - lam) * uni_prob[nxt]
        else:
            # interpolation=0 is the pure unigram model: the conditional
            # term contributes the unigram probability of every other token.
            p = uni_prob[nxt]
        nll_sum += float(-np.log(np.maximum(p, np.finfo(float).tiny)).sum())
    n_tokens = int(all_eval.size)
    nll = float(nll_sum / n_tokens) if n_tokens else float('nan')
    return {'order': 2, 'alpha': alpha, 'interpolation': lam,
            'n_tokens': n_tokens, 'nll': nll, 'ppl': float(_safe_exp_ppl(nll)) if math.isfinite(nll) else float('nan')}

def ngram_reference_score(train_ids, eval_ids, vocab_size, *, order=2,
                          alpha=0.1, interpolation=0.5,
                          max_fit_tokens=None, _counts=None):
    """Per-target reference NLL and exact top-1 for a count-model floor.

    This is the per-token counterpart of :func:`ngram_lm_score`, used by the
    bad-case analysis to separate a model miss from an intrinsically hard or
    mislabelled token.  The returned arrays align with ``eval_ids[:, 1:]``
    (the next-token targets scored by ``analyze_errors``); the first token of
    every row is never a target, so no cross-row transition is introduced.

    The top-1 prediction is the exact argmax of the interpolated model, not a
    backoff guess: for a seen context the maximum is either the best observed
    successor or the best *unseen* token (which can differ from the global
    unigram argmax when that token is already a successor).  Ties are broken
    by the lowest token id, matching ``np.argmax``.  The unigram ranking is
    sorted once per call and each context only scans successors plus the
    first unseen candidate, so a large vocabulary does not turn this into an
    ``O(contexts * vocab)`` scan.
    """
    raw_order = order
    if isinstance(raw_order, (bool, np.bool_)):
        raise ValueError(f'reference order must be 1 or 2, got {raw_order!r}')
    order = _as_seed_int(raw_order)
    if order not in (1, 2):
        raise ValueError(f'reference order must be 1 (unigram) or 2 (bigram), got {raw_order!r}')
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError('reference alpha must be a positive finite number, got a boolean')
    try:
        alpha = float(alpha)
        interpolation = float(interpolation)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('reference alpha/interpolation must be finite numbers') from None
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError('reference alpha must be a positive finite number')
    if not (math.isfinite(interpolation) and 0.0 <= interpolation <= 1.0):
        raise ValueError('reference interpolation must be in [0, 1]')
    if order == 1:
        interpolation = 0.0
    vocab_size = _ngram_vocab_size(vocab_size)
    arr = np.asarray(eval_ids)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError('reference eval ids must be 1-D or 2-D (rows x length)')
    if arr.dtype == np.bool_ or arr.dtype == object or not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f'reference eval ids must be integers, got dtype {arr.dtype}')
    arr = np.asarray(arr, dtype=np.int64)
    if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= vocab_size):
        raise ValueError('reference eval ids outside [0, vocab_size)')
    fit_meta = {'max_fit_tokens': max_fit_tokens, 'n_fit_tokens': None,
                'n_source_tokens': None, 'fit_truncated': None,
                'fit_token_policy': 'deterministic_prefix' if max_fit_tokens is not None else 'full'}
    if _counts is None:
        ids = np.asarray(_ngram_token_ids(train_ids, 'train'), dtype=np.int64).reshape(-1)
        fit_meta['n_source_tokens'] = int(ids.size)
        if max_fit_tokens is not None:
            cap = _exact_count(max_fit_tokens, minimum=1)
            if cap is None:
                raise ValueError(f'max_fit_tokens must be a positive integer or None, got {max_fit_tokens!r}')
            if cap < ids.size:
                ids = ids[:cap]
                fit_meta['fit_truncated'] = True
            else:
                fit_meta['fit_truncated'] = False
            fit_meta['max_fit_tokens'] = int(cap)
        counts = _ngram_train_counts(ids, vocab_size)
        fit_meta['n_fit_tokens'] = int(ids.size)
    else:
        counts = _counts
    out = {'kind': 'ngram_reference_scores', 'order': int(order),
           'alpha': float(alpha), 'interpolation': float(interpolation),
           'vocab_size': int(vocab_size), 'n_rows': int(arr.shape[0]),
           'n_targets': int(max(0, arr.shape[1] - 1)),
           'fit': fit_meta,
           'nll': np.empty((arr.shape[0], 0), dtype=np.float64),
           'top1_ids': np.empty((arr.shape[0], 0), dtype=np.int64)}
    if arr.shape[1] < 2:
        out['nll'] = np.empty((arr.shape[0], 0), dtype=np.float64)
        out['top1_ids'] = np.empty((arr.shape[0], 0), dtype=np.int64)
        out['mean_nll'] = None
        return out
    targets = arr[:, 1:]
    prev = arr[:, :-1]
    uni_counts = counts[0]
    denom = float(uni_counts.sum()) + alpha * vocab_size
    uni_prob = (uni_counts + alpha) / denom
    log_uni = np.log(np.maximum(uni_prob, np.finfo(float).tiny))
    uni_top = int(np.argmax(uni_prob))
    if order == 1:
        nll = -log_uni[targets]
        top1 = np.full(targets.shape, uni_top, dtype=np.int64)
    else:
        _uni, uniq, ctx, pair_counts = counts
        keys = (prev.reshape(-1) * vocab_size + targets.reshape(-1)).astype(np.int64)
        if uniq.size:
            idx = np.searchsorted(uniq, keys)
            idx_clip = np.minimum(idx, uniq.size - 1)
            hit = (idx < uniq.size) & (uniq[idx_clip] == keys)
            c_pair = np.where(hit, pair_counts[idx_clip], 0).astype(np.float64)
        else:
            c_pair = np.zeros(keys.shape, dtype=np.float64)
        c_ctx = ctx[prev.reshape(-1)].astype(np.float64)
        p_bg = (c_pair + alpha) / (c_ctx + alpha * vocab_size)
        p = interpolation * p_bg + (1.0 - interpolation) * uni_prob[targets.reshape(-1)]
        nll = -np.log(np.maximum(p, np.finfo(float).tiny)).reshape(targets.shape)
        # Rank the unigram distribution once: unseen tokens are ordered by
        # unigram probability (ties by token id), and a context only needs
        # the first ranked token it did not observe as a successor.  When
        # interpolation=1 the conditional term is identical for every unseen
        # token, so the winner is simply the smallest unseen id.
        if interpolation >= 1.0:
            uni_order = None
        else:
            uni_order = np.lexsort((np.arange(vocab_size, dtype=np.int64),
                                    -uni_prob))
        uniq_ctx, inv = np.unique(prev, return_inverse=True)
        tops = np.empty(uniq_ctx.size, dtype=np.int64)
        for j, c in enumerate(uniq_ctx):
            c = int(c)
            lo = int(np.searchsorted(uniq, c * vocab_size))
            hi = int(np.searchsorted(uniq, (c + 1) * vocab_size))
            succ = uniq[lo:hi] - c * vocab_size
            cnts = pair_counts[lo:hi].astype(np.float64)
            den = float(ctx[c]) + alpha * vocab_size
            best_id = None
            best_score = -np.inf
            if succ.size:
                seen_scores = (interpolation * ((cnts + alpha) / den)
                               + (1.0 - interpolation) * uni_prob[succ])
                best_seen = int(np.argmax(seen_scores))
                best_id = int(succ[best_seen])
                best_score = float(seen_scores[best_seen])
            if succ.size < vocab_size:
                if interpolation >= 1.0:
                    # Smallest token id that is not a successor.
                    cand = 0
                    for s in succ.tolist():
                        if s == cand:
                            cand += 1
                        elif s > cand:
                            break
                    unseen_id = cand
                else:
                    assert uni_order is not None
                    pos = 0
                    while pos < uni_order.size:
                        cand = int(uni_order[pos])
                        k = int(np.searchsorted(succ, cand))
                        if k >= succ.size or int(succ[k]) != cand:
                            break
                        pos += 1
                    unseen_id = int(uni_order[pos])
                unseen_score = (interpolation * (alpha / den)
                                + (1.0 - interpolation) * uni_prob[unseen_id])
                if (unseen_score > best_score
                        or (unseen_score == best_score
                            and (best_id is None or unseen_id < best_id))):
                    best_id, best_score = unseen_id, unseen_score
            if best_id is None:
                # Every token is observed after this context; the seen argmax
                # above always exists in that case.
                best_id = uni_top
            tops[j] = best_id
        top1 = tops[inv].reshape(targets.shape)
    out['nll'] = nll
    out['top1_ids'] = top1
    out['mean_nll'] = float(nll.mean()) if nll.size else None
    return out


def _ngram_row_stats(train_ids, eval_ids, vocab_size, *, order=1, alpha=0.1,
                     interpolation=1.0, _counts=None):
    """Per-row NLL/token accounting for the selected count model.

    A floor contrast that treats the count model as a fixed number ignores the
    evaluation-row sampling uncertainty of that number.  These arrays let the
    paired row bootstrap compare the neural arm and the floor on the SAME
    rows.  Each row is scored separately with its own boundary (the first
    token is never conditioned on the previous row)."""
    arr = np.asarray(eval_ids)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError('ngram row stats need a 1-D or 2-D evaluation array')
    nll_rows, tok_rows = [], []
    for i in range(arr.shape[0]):
        sc = ngram_lm_score(train_ids, arr[i:i + 1], vocab_size, order=order,
                            alpha=alpha, interpolation=interpolation,
                            _counts=_counts)
        _nt = int(sc['n_tokens'])
        _nll = sc['nll']
        tok_rows.append(_nt)
        nll_rows.append(float(_nll) * _nt if _nt and math.isfinite(float(_nll))
                        else float('nan'))
    return nll_rows, tok_rows

def _simple_baseline_row_indices(rows, n_rows, label):
    """Exact, unique, in-range row indices for the nested floor evaluation."""
    if rows is None:
        return None
    try:
        raw = list(rows)
    except TypeError:
        raise ValueError(f'simple baseline {label} rows must be an iterable, got {rows!r}') from None
    parsed = [_as_seed_int(v) for v in raw]
    if any(v is None for v in parsed):
        raise ValueError(f'simple baseline {label} rows must be integers, got {raw!r}')
    if any(v < 0 or v >= n_rows for v in parsed):
        raise ValueError(f'simple baseline {label} rows outside [0, {n_rows})')
    if len(set(parsed)) != len(parsed):
        raise ValueError(f'simple baseline {label} rows contain duplicates')
    return np.asarray(sorted(parsed), dtype=np.int64)

def simple_baseline_fit(train_ids, eval_ids, vocab_size, protocol=None,
                        selection_rows=None, evaluation_rows=None,
                        row_split=None):
    """Run the declared count-model grid and select per baseline on validation.

    When ``selection_rows`` / ``evaluation_rows`` are supplied, the smoothing
    constants are selected on the earlier chronological row fold and the
    selected model is reported on the disjoint later fold.  This mirrors the
    neural panel's nested validation split: selecting and reporting on the
    same documents gives the count-model floor an unfair look at the
    evaluation rows.  ``row_split`` may carry the reproducible fold metadata
    (from ``validation_row_split``); its row fingerprints are checked against
    the supplied indices.
    """
    protocol = dict(SIMPLE_BASELINE_POLICY if protocol is None else protocol)
    arms = protocol.get('arms') or {}
    grids = protocol.get('grids') or {}
    trials_by_arm = {}
    _vocab_n = int(_ngram_vocab_size(vocab_size))
    # Fit on a deterministic training prefix when the declared cap is smaller
    # than the corpus.  This bounds the bigram table (O(unique pairs)) so the
    # floor remains runnable on a B-tier token budget; the truncation is
    # recorded, never silent.
    _train_raw = np.asarray(train_ids)
    if not np.issubdtype(_train_raw.dtype, np.integer):
        raise TypeError(f'ngram baseline: train ids must be integers, got dtype {_train_raw.dtype}')
    _train_all = _train_raw.reshape(-1)
    _cap = protocol.get('max_fit_tokens')
    if _cap is None:
        _fit_ids = _train_all
    else:
        _cap_i = _as_seed_int(_cap)
        if _cap_i is None:
            raise ValueError(f'max_fit_tokens must be an integer or None, got {_cap!r}')
        _cap = _cap_i
        if _cap < 1:
            raise ValueError(f'max_fit_tokens must be >= 1, got {_cap}')
        _fit_ids = _train_all[:_cap]
    _fit_policy = protocol.get('fit_token_policy', 'deterministic_prefix')
    if _fit_policy != 'deterministic_prefix':
        # The implementation below always uses a deterministic prefix when the
        # budget is exceeded.  Recording a different policy name would make
        # the artifact describe a data path that never ran.
        raise ValueError(
            f'fit_token_policy={_fit_policy!r} is not implemented; the count '
            'baseline only supports deterministic_prefix')
    _fit_meta = {
        'fit_token_policy': _fit_policy,
        'max_fit_tokens': _cap,
        'n_fit_tokens': int(_fit_ids.size),
        'n_source_tokens': int(_train_all.size),
        'fit_truncated': bool(_fit_ids.size < _train_all.size),
    }
    eval_arr = np.asarray(eval_ids)
    if eval_arr.ndim not in (1, 2):
        raise ValueError('eval ids must be 1-D or 2-D (sequences x length)')
    nested = selection_rows is not None or evaluation_rows is not None
    if nested:
        if selection_rows is None or evaluation_rows is None:
            raise ValueError('nested simple-baseline selection requires BOTH '
                             'selection_rows and evaluation_rows')
        if eval_arr.ndim != 2:
            raise ValueError('nested simple-baseline selection needs a 2-D evaluation tensor (rows x length)')
        _n_rows = int(eval_arr.shape[0])
        _sel_idx = _simple_baseline_row_indices(selection_rows, _n_rows, 'selection')
        _eval_idx = _simple_baseline_row_indices(evaluation_rows, _n_rows, 'evaluation')
        if _sel_idx.size == 0 or _eval_idx.size == 0:
            raise ValueError('nested simple-baseline selection needs non-empty selection and evaluation folds')
        _overlap = np.intersect1d(_sel_idx, _eval_idx)
        if _overlap.size:
            raise ValueError('simple-baseline selection and evaluation rows overlap')
        if int(_sel_idx.max()) >= int(_eval_idx.min()):
            # The neural path requires selection rows to precede evaluation
            # rows; a floor selected on later documents would not match it.
            raise ValueError('simple-baseline selection rows must precede the evaluation rows in stream order')
        if not isinstance(row_split, dict):
            row_split = {
                'policy': 'chronological_contiguous_row_folds',
                'n_rows': _n_rows,
                'n_selection_rows': int(_sel_idx.size),
                'n_evaluation_rows': int(_eval_idx.size),
                'selection_row_range': [int(_sel_idx.min()), int(_sel_idx.max())],
                'evaluation_row_range': [int(_eval_idx.min()), int(_eval_idx.max())],
                'selection_row_fp': _ids_fp(_sel_idx),
                'evaluation_row_fp': _ids_fp(_eval_idx),
                'chronological': True,
                'selection_precedes_evaluation': True,
                'disjoint': True,
            }
        else:
            row_split = dict(row_split)
            # A PRESENT but malformed count/range is not an absent legacy
            # field: ignoring it and then setdefault-ing a valid value would
            # bless contradictory split evidence.
            if 'n_rows' in row_split:
                _rn = _exact_count(row_split.get('n_rows'), minimum=1)
                if _rn is None:
                    raise ValueError('row_split n_rows is not a positive integer')
                if _rn != _n_rows:
                    raise ValueError('row_split n_rows does not match the evaluation tensor')
            for _key, _want in (('n_selection_rows', int(_sel_idx.size)),
                                ('n_evaluation_rows', int(_eval_idx.size))):
                if _key not in row_split:
                    continue
                _got = _exact_count(row_split.get(_key), minimum=1)
                if _got is None:
                    raise ValueError(f'row_split {_key} is not a positive integer')
                if _got != _want:
                    raise ValueError(f'row_split {_key} does not match the supplied rows')
            _want_sel = row_split.get('selection_row_fp')
            _want_eval = row_split.get('evaluation_row_fp')
            if _want_sel is not None and _want_sel != _ids_fp(_sel_idx):
                raise ValueError('row_split selection_row_fp does not describe the supplied selection rows')
            if _want_eval is not None and _want_eval != _ids_fp(_eval_idx):
                raise ValueError('row_split evaluation_row_fp does not describe the supplied evaluation rows')
            for _key, _want_range in (
                    ('selection_row_range', [int(_sel_idx.min()), int(_sel_idx.max())]),
                    ('evaluation_row_range', [int(_eval_idx.min()), int(_eval_idx.max())])):
                if _key not in row_split:
                    continue
                _rr = row_split.get(_key)
                if (not isinstance(_rr, (list, tuple)) or len(_rr) != 2):
                    raise ValueError(f'row_split {_key} must be a two-element range')
                _vals = [_exact_count(v, minimum=0) for v in _rr]
                if any(v is None for v in _vals):
                    raise ValueError(f'row_split {_key} must contain exact row indices')
                if [int(_vals[0]), int(_vals[1])] != _want_range:
                    raise ValueError(f'row_split {_key} does not describe the supplied rows')
            row_split.setdefault('policy', 'chronological_contiguous_row_folds')
            row_split.setdefault('n_rows', _n_rows)
            row_split.setdefault('n_selection_rows', int(_sel_idx.size))
            row_split.setdefault('n_evaluation_rows', int(_eval_idx.size))
            row_split.setdefault('selection_row_fp', _ids_fp(_sel_idx))
            row_split.setdefault('evaluation_row_fp', _ids_fp(_eval_idx))
        selection_eval = eval_arr[_sel_idx]
        scoring_eval = eval_arr[_eval_idx]
    else:
        selection_eval = eval_arr
        scoring_eval = eval_arr
    # Build the corpus count tables once for the whole grid (lazily, so an
    # empty evaluation set still short-circuits without touching the corpus):
    # every trial below consumes the identical arrays — the trial
    # hyper-parameters only affect the smoothing on top of them.
    _shared_counts = None
    for arm, meta in arms.items():
        # The model order is a discrete identity: int(1.9) or int(True) would
        # silently run a different (unigram) baseline than the declared one.
        order = _as_seed_int((meta or {}).get('order', 1))
        if order is None:
            raise ValueError(f'simple baseline {arm}: order must be an integer, '
                             f'got {(meta or {}).get("order", 1)!r}')
        grid = grids.get(arm) or {}
        alphas = list(grid.get('alpha') or [0.1])
        interps = list(grid.get('interpolation') or [1.0]) if order == 2 else [0.0]
        rows = []
        for alpha in alphas:
            for lam in interps:
                if _shared_counts is None:
                    _shared_counts = _ngram_train_counts(np.asarray(_fit_ids, dtype=np.int64), vocab_size)
                # Pass the declared grid values through unchanged so
                # ngram_lm_score's boolean/non-finite validation runs BEFORE
                # any float() coercion (float(True) == 1.0 used to bypass it).
                sc = ngram_lm_score(_fit_ids, selection_eval, vocab_size, order=order,
                                    alpha=alpha, interpolation=lam,
                                    _counts=_shared_counts)
                cfg = {'order': order, 'alpha': float(sc['alpha'])}
                if order == 2:
                    cfg['interpolation'] = float(sc['interpolation'])
                rows.append({'config': cfg, 'ppl': sc['ppl'], 'nll': sc['nll'],
                             'n_tokens': sc['n_tokens'], 'steps': 0, 'tokens_seen': int(sc['n_tokens'])})
        if not rows:
            raise RuntimeError(f'simple baseline {arm}: empty declared grid')
        best = min(rows, key=lambda r: float(r['ppl']))
        entry = {'selected': dict(best['config']),
                 'selected_metric_value': float(best['ppl']),
                 'trials': rows,
                 'order': order}
        if nested:
            # Score the SELECTED config once on the held-out rows.  The
            # selection fold chose the smoothing; this fold is the estimate
            # that may be compared with the neural panel's holdout PPL.
            sc_hold = ngram_lm_score(_fit_ids, scoring_eval, vocab_size, order=order,
                                     alpha=best['config']['alpha'],
                                     interpolation=best['config'].get('interpolation', 1.0),
                                     _counts=_shared_counts)
            for _r in rows:
                if _r is best:
                    _r['holdout_ppl'] = sc_hold['ppl']
                    _r['holdout_nll'] = sc_hold['nll']
                    _r['holdout_n_tokens'] = sc_hold['n_tokens']
                    break
            entry['holdout_metric_value'] = float(sc_hold['ppl'])
            entry['holdout_nll'] = float(sc_hold['nll'])
            entry['holdout_n_tokens'] = int(sc_hold['n_tokens'])
        # Per-row accounting for the reported sample.  It is optional evidence
        # (legacy artifacts lack it), but when present the audit and the
        # floor contrast no longer have to treat the floor as a fixed number.
        try:
            _row_nll, _row_tok = _ngram_row_stats(
                _fit_ids, scoring_eval, vocab_size, order=order,
                alpha=best['config']['alpha'],
                interpolation=best['config'].get('interpolation', 1.0),
                _counts=_shared_counts)
        except (TypeError, ValueError, OverflowError):
            _row_nll = _row_tok = None
        if _row_nll is not None:
            entry['reported_nll_per_row'] = _row_nll
            entry['reported_n_tokens_per_row'] = _row_tok
            entry['reported_estimation_rows'] = ('nested_validation_holdout'
                                                 if nested else 'full_validation')
        trials_by_arm[arm] = entry
    best_arm = min(trials_by_arm, key=lambda a: float(trials_by_arm[a]['selected_metric_value']))
    if nested:
        best_ppl = float(trials_by_arm[best_arm]['holdout_metric_value'])
    else:
        best_ppl = float(trials_by_arm[best_arm]['selected_metric_value'])
    # A uniform-over-vocabulary model is the zero-information floor: the count
    # model must at least be better than random tokens.  A floor worse than
    # uniform points at a broken tokenizer/smoothing path, not a weak model.
    uniform_ppl = float(_vocab_n)
    best_vs_uniform = (best_ppl / uniform_ppl
                       if ppl_is_usable(best_ppl) and uniform_ppl > 0 else None)
    # `_ids_fp` hashes the canonical chunked form of any integer width and
    # accepts the training stream (possibly a memmap) directly.
    _data = {'vocab_size': _vocab_n,
             'train_fp': _ids_fp(_fit_ids),
             'source_train_fp': _ids_fp(_train_all),
             'val_fp': _ids_fp(np.ascontiguousarray(eval_arr).reshape(-1)),
             'eval_fp': _ids_fp(np.ascontiguousarray(scoring_eval).reshape(-1)),
             'n_train_tokens': int(_train_all.size),
             'n_eval_tokens': int(scoring_eval.size),
             'nested_selection': bool(nested)}
    if nested:
        _data.update({'selection_fp': _ids_fp(np.ascontiguousarray(selection_eval).reshape(-1)),
                      'n_selection_tokens': int(selection_eval.size),
                      'selection_rows': int(_sel_idx.size),
                      'eval_rows': int(_eval_idx.size),
                      'row_split': row_split})
    _data.update(_fit_meta)
    _data['data_version'] = _data_version(_data)
    _prov = code_provenance()
    _prov['data'] = _data
    return {
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'protocol': protocol,
        'selection_metric': protocol.get('selection_metric', 'ppl'),
        'selection_split': protocol.get('selection_split', 'validation'),
        'direction': protocol.get('direction', 'lower_is_better'),
        'data': _data,
        'data_version': _data['data_version'],
        'provenance': _prov,
        'baselines': trials_by_arm,
        'best_baseline': best_arm,
        'best_ppl': best_ppl,
        'uniform_ppl': uniform_ppl,
        'best_ppl_vs_uniform': best_vs_uniform,
        'best_ppl_estimation_rows': ('nested_validation_holdout' if nested else 'full_validation'),
        'code': code_stamp(),
    }
