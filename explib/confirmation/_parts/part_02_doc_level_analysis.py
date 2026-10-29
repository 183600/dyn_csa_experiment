

def _doc_level_analysis(recs_a, recs_b, common, Ln, primary_deltas=None):
    """Document-sampling readout for one comparison.

    逐文档配对差（对每个文档取「种子内臂差」的均值）与两层方差分解：
    between-doc（文档抽样不确定性）与 within-doc-across-seed（训练随机性）。
    48 条评测行仍不足以充分覆盖文档分布——本读数的目的就是把这个不确定性
    显式化，而不是宣称已消除。

    等权逐文档均值回答「平均文档是否更好」，主终点回答「token 池化 PPL
    是否更低」。文档长度不同时这是两个估计量；因此额外给出与主终点同口径
    的 token 池化 Δ 的文档 cluster-robust 区间（条件在已训练种子上），等权
    bootstrap 只保留为描述量。"""
    if not common:
        return None
    rows_a = [recs_a[s]['by_len'].get(Ln, recs_a[s]['by_len'].get(str(Ln))) or {} for s in common]
    rows_b = [recs_b[s]['by_len'].get(Ln, recs_b[s]['by_len'].get(str(Ln))) or {} for s in common]
    # A fixed eval ROW is an arbitrary window, not a document: it can split
    # one document over two rows (double-counting it) or merge two documents
    # into one row.  A WikiText "document" is itself one raw LINE, so even the
    # document bootstrap resamples paragraphs of the same article.  Prefer the
    # per-TOKEN ARTICLE aggregation when every record carries it (the
    # independent corpus unit); otherwise fall back to the line-level document
    # map, and only then to fixed rows.  Every fallback is named in the output.
    _art_a = [r.get('ppl_per_article') for r in rows_a]
    _art_b = [r.get('ppl_per_article') for r in rows_b]
    _has_art = [isinstance(x, list) and bool(x) for x in _art_a + _art_b]
    _doc_a = [r.get('ppl_per_doc') for r in rows_a]
    _doc_b = [r.get('ppl_per_doc') for r in rows_b]
    _has_doc = [isinstance(x, list) and bool(x) for x in _doc_a + _doc_b]
    _article_fallback = None
    _ref_ids = None
    if any(_has_art) and all(_has_art):
        _granularity = 'article'
        _ppl_a, _ppl_b = _art_a, _art_b
        _tok_key, _ids_key = 'n_tokens_per_article', 'article_ids'
    elif any(_has_art):
        if not (all(_has_doc) and any(_has_doc)):
            return {'n_docs': 0, 'n_seeds': len(common), 'comparable': False,
                    'reason': ('some records carry article-level cells and '
                               'others do not; the article and line units '
                               'cannot be pooled'),
                    'note': 'refused rather than mixing two estimands'}
        _article_fallback = ('article cells are missing for some arm/seed '
                             'records; reported at the line-level document '
                             'unit')
        _granularity = 'document'
        _ppl_a, _ppl_b = _doc_a, _doc_b
        _tok_key, _ids_key = 'n_tokens_per_doc', 'doc_ids'
    elif all(_has_doc) and any(_has_doc):
        _granularity = 'document'
        _ppl_a, _ppl_b = _doc_a, _doc_b
        _tok_key, _ids_key = 'n_tokens_per_doc', 'doc_ids'
    else:
        if any(_has_doc):
            return {'n_docs': 0, 'n_seeds': len(common), 'comparable': False,
                    'reason': ('some records carry document-level cells and '
                               'others do not; document- and row-level '
                               'uncertainty cannot be pooled'),
                    'note': 'refused rather than mixing two estimands'}
        _granularity = 'row'
        _ppl_a = [r.get('ppl_per_seq') for r in rows_a]
        _ppl_b = [r.get('ppl_per_seq') for r in rows_b]
        if not all(isinstance(x, list) and x for x in _ppl_a + _ppl_b):
            return None
    if _granularity in ('document', 'article'):
        _tok_a = [r.get(_tok_key) for r in rows_a]
        _tok_b = [r.get(_tok_key) for r in rows_b]
        if not all(isinstance(x, list) and x for x in _tok_a + _tok_b):
            return {'n_docs': 0, 'n_seeds': len(common), 'comparable': False,
                    'reason': (f'{_granularity}-level cells lack per-unit '
                               'token counts'),
                    'note': 'refused rather than treating unequal units as equal'}
        _lens = {len(x) for x in _ppl_a + _ppl_b + _tok_a + _tok_b}
        if len(_lens) != 1:
            return {'n_docs': int(min(_lens)), 'n_seeds': len(common),
                    'comparable': False,
                    'granularity': _granularity,
                    'reason': (f'{_granularity} counts differ across arm/seed '
                               'records'),
                    'row_counts': sorted(_lens),
                    'note': 'refused rather than silently truncating'}
        _ref_tok = list(_tok_a[0])
        if any(list(x) != _ref_tok for x in _tok_a + _tok_b):
            return {'n_docs': int(min(_lens)), 'n_seeds': len(common),
                    'comparable': False,
                    'granularity': _granularity,
                    'reason': (f'per-{_granularity} token counts differ across '
                               'arm/seed records; the units are not the same'),
                    'note': 'refused rather than pairing different units'}
        _ref_ids = rows_a[0].get(_ids_key)
        if _ref_ids is not None:
            if any(r.get(_ids_key) != _ref_ids for r in rows_a + rows_b):
                return {'n_docs': int(min(_lens)), 'n_seeds': len(common),
                        'comparable': False,
                        'granularity': _granularity,
                        'reason': (f'{_granularity} id order differs across '
                                   'arm/seed records; paired deltas would pair '
                                   'different units'),
                        'note': 'refused rather than pairing by position'}
        per_doc_a, per_doc_b = _ppl_a, _ppl_b
    else:
        per_doc_a, per_doc_b = _ppl_a, _ppl_b
    # Pair unit i across arms only when every record evaluated the SAME number
    # of units.  Silently taking the minimum would pair a truncated panel
    # with a full one and claim a larger sample than either scored.
    _lens = {len(x) for x in per_doc_a + per_doc_b}
    if len(_lens) != 1:
        return {'n_docs': int(min(_lens)), 'n_seeds': len(common), 'comparable': False,
                'granularity': _granularity,
                'reason': 'per-document row counts differ across arm/seed records; document-paired deltas are not defined',
                'row_counts': sorted(_lens), 'note': 'refused rather than silently truncating to the shortest record'}
    n_docs = min(_lens)
    if n_docs < 2:
        return None
    # Unit labels for the leave-one-unit-out influence readout.  The order was
    # already checked across arm/seed records above; only a unique, complete id
    # vector may label the folds, otherwise the readout falls back to indices
    # (and the report says so) rather than guessing which unit was dropped.
    _unit_ids = None
    if isinstance(_ref_ids, (list, tuple)) and len(_ref_ids) == n_docs:
        try:
            _ids_cand = [str(x) for x in _ref_ids]
            if len(set(_ids_cand)) == n_docs:
                _unit_ids = _ids_cand
        except (TypeError, ValueError):
            _unit_ids = None
    doc_deltas = []
    doc_log_ratios = []
    within = []
    incomplete = 0
    for i in range(n_docs):
        cell = []
        logs = []
        for k in range(len(common)):
            a, b = per_doc_a[k][i], per_doc_b[k][i]
            try:
                fa, fb = float(a), float(b)
            except (TypeError, ValueError, OverflowError):
                fa = fb = float('nan')
            if (not math.isfinite(fa)) or (not math.isfinite(fb)) or fa <= 0.0 or fb <= 0.0:
                incomplete += 1
                continue
            cell.append(fa - fb)
            logs.append(math.log(fa) - math.log(fb))
        if not cell:
            continue
        doc_deltas.append(float(np.mean(cell)))
        doc_log_ratios.append(float(np.mean(logs)))
        within.append(float(np.std(cell, ddof=1)) if len(cell) > 1 else 0.0)
    if len(doc_deltas) < 2:
        return None
    # The equal-weighted per-document mean above answers "is the average
    # document better?"; the primary endpoint asks "is the token-pooled PPL
    # lower?".  When document lengths differ these are different estimands, so
    # the interval shown next to the pooled Δ must resample/weight DOCUMENTS
    # with their token counts, not give every document equal weight.
    _pooled_ci = None
    _pooled_ci_reason = None
    _unit_influence = None
    _unit_influence_reason = None
    _two_level_ci = None
    _two_level_ci_reason = None
    _variance_components = None
    _variance_components_reason = None
    if _granularity in ('document', 'article'):
        _nll_a, _why_a = _doc_nll_matrix(rows_a, n_docs, _ref_tok,
                                         unit=_granularity)
        _nll_b, _why_b = _doc_nll_matrix(rows_b, n_docs, _ref_tok,
                                         unit=_granularity)
        if _nll_a is None or _nll_b is None:
            _pooled_ci_reason = (
                'token-pooled cluster CI not available: '
                + (_why_a or _why_b
                   or f'{_granularity} NLL vectors are unusable'))
        else:
            try:
                _variance_components = L.paired_delta_variance_components(
                    _nll_a, _nll_b, _ref_tok)
            except (ValueError, TypeError, FloatingPointError,
                    OverflowError) as _vce:
                _variance_components = None
                _variance_components_reason = (
                    f'{type(_vce).__name__}: {_vce}')
            _pc = L._paired_nll_doc_ci(_nll_a, _nll_b, _ref_tok)
            if _pc is None:
                _pooled_ci_reason = (
                    'token-pooled cluster CI not available: the per-seed PPL '
                    'difference overflowed')
            else:
                # The interval is only meaningful if its point estimate is the
                # SAME per-seed pooled Δ the frozen comparison reports.
                _cell_a = [r.get('ppl') for r in rows_a]
                _cell_b = [r.get('ppl') for r in rows_b]
                _pt = []
                for _pa, _pb in zip(_cell_a, _cell_b):
                    try:
                        _fa, _fb = float(_pa), float(_pb)
                    except (TypeError, ValueError, OverflowError):
                        _pt = []
                        break
                    if (not math.isfinite(_fa)) or (not math.isfinite(_fb)):
                        _pt = []
                        break
                    _pt.append(_fa - _fb)
                if _pt:
                    _want = float(np.mean(_pt))
                    if abs(_pc['delta_ppl'] - _want) > 1e-5 * max(1.0, abs(_want)):
                        _pooled_ci_reason = (
                            f'token-pooled cluster CI refused: the '
                            f'{_granularity} vectors do not reproduce the '
                            'recorded per-seed pooled PPL difference')
                    else:
                        _pooled_ci = _pc
                        try:
                            _unit_influence = L.paired_nll_unit_influence(
                                _nll_a, _nll_b, _ref_tok, unit_ids=_unit_ids)
                        except (ValueError, TypeError, FloatingPointError,
                                OverflowError) as _ie:
                            _unit_influence = None
                            _unit_influence_reason = (
                                f'{type(_ie).__name__}: {_ie}')
                        # Total uncertainty: the cluster CI above conditions on
                        # the trained seeds, so it can be much narrower than the
                        # uncertainty of a result that also samples seeds.  The
                        # two-way bootstrap uses the SAME point estimate and
                        # resamples both levels; it is a descriptive companion,
                        # never a replacement for the frozen sign-flip test.
                        try:
                            _two = L.paired_nll_two_way_bootstrap(
                                _nll_a, _nll_b, _ref_tok)
                        except (ValueError, TypeError, FloatingPointError,
                                OverflowError) as _te:
                            _two = None
                            _two_level_ci_reason = (
                                f'{type(_te).__name__}: {_te}')
                        if _two is not None and _two.get('status') == 'ok':
                            _tp = L._finite_scalar(_two.get('delta_ppl'))
                            _pp = L._finite_scalar(_pooled_ci.get('delta_ppl'))
                            if (_tp is None or _pp is None
                                    or abs(_tp - _pp) > 1e-6 * max(1.0, abs(_pp))):
                                _two_level_ci_reason = (
                                    'two-level bootstrap refused: its point '
                                    'estimate does not reproduce the pooled Δ')
                            else:
                                _two_level_ci = _two
                        elif _two is not None:
                            _two_level_ci_reason = (
                                _two.get('reason')
                                or 'two-level bootstrap unavailable')
                else:
                    # The frozen comparison has no finite per-seed pooled PPL
                    # for this cell, so the document vectors cannot be tied to
                    # the primary estimand.  Do not publish an interval whose
                    # point estimate could not be checked: `_pt` is empty for
                    # a missing/NaN `ppl`, not because the vectors are trusted.
                    _pooled_ci_reason = (
                        'token-pooled cluster CI refused: the recorded per-seed '
                        'pooled PPL is missing or non-finite, so the '
                        f'{_granularity} vectors cannot be tied to the primary '
                        'comparison')
    arr = np.asarray(doc_deltas, dtype=float)
    lr = np.asarray(doc_log_ratios, dtype=float)
    # Seed-level sign-flip remains the confirmatory test.  These intervals
    # condition on the trained seeds and quantify only DOCUMENT sampling:
    # with n_seqs=48 they are a property of the frozen slice, not of the
    # language population.  Resampling documents (not tokens) preserves the
    # within-document correlation.
    _ci_ppl = L.bootstrap_mean_ci(arr, seed=20261018)
    _ci_log = L.bootstrap_mean_ci(lr, seed=20261018)
    # Breadth diagnostics: a mean gain can be broad-based or carried by one
    # document.  Counts/fractions are descriptive (documents are not an
    # independent randomisation of the arm), but they expose an outlier-driven
    # "win" that a mean and a bootstrap CI can hide.
    _neg = int((arr < 0).sum())
    _pos = int((arr > 0).sum())
    _zero = int((arr == 0).sum())
    _abs_sum = float(np.abs(arr).sum())
    _top_share = (float(np.abs(arr).max() / _abs_sum) if _abs_sum > 0.0 else None)
    _unit_notes = {
        'article': ('article-sampling uncertainty only; seed-level exact '
                    'sign-flip remains the confirmatory test.  WikiText rows '
                    'are lines, so the article (top-level heading group) is '
                    'the independent corpus unit; article count is typically '
                    'far below the 64-unit population floor, so '
                    'population-level claims stay refused'),
        'document': ('line-level sampling uncertainty only; seed-level exact '
                     'sign-flip remains the confirmatory test.  A WikiText '
                     '"document" is one raw line, so an article spanning '
                     'several lines is pseudoreplicated here; report the '
                     'article unit when the cache/tokenizer can supply it '
                     '(design_notes section 54)'),
        'row': ('ROW-level uncertainty only: fixed eval rows can split or '
                'merge lines/articles, so this bootstrap is not a document '
                'bootstrap; rebuild/copy the test cache with per-token line '
                'ids and a byte-level tokenizer for the article-level readout'),
    }
    _ci_method = {
        'article': ('percentile bootstrap over ARTICLES (chunked, '
                    'deterministic); conditions on the trained seeds'),
        'document': ('percentile bootstrap over LINES (the cache calls them '
                     'documents; chunked, deterministic); conditions on the '
                     'trained seeds'),
        'row': ('percentile bootstrap over fixed eval ROWS (chunked, '
                'deterministic); rows are not documents'),
    }[_granularity]
    out = {'n_docs': int(len(doc_deltas)), 'n_units': int(len(doc_deltas)),
           'n_seeds': len(common),
           'granularity': _granularity,
           'unit': _granularity,
           'per_doc_delta_mean': float(arr.mean()),
           'per_doc_delta_median': float(np.median(arr)),
           'per_doc_log_ratio_mean': float(lr.mean()),
           'doc_delta_negative': _neg, 'doc_delta_positive': _pos,
           'doc_delta_zero': _zero,
           'doc_a_better_fraction': (float(_neg / arr.size) if arr.size else None),
           'top_doc_abs_share': _top_share,
           'between_doc_sd': float(arr.std(ddof=1)) if len(doc_deltas) > 1 else 0.0,
           'between_doc_log_ratio_sd': float(lr.std(ddof=1)) if len(doc_log_ratios) > 1 else 0.0,
           'mean_within_doc_seed_sd': float(np.mean(within)),
           'per_doc_delta_ppl_ci95': ([_ci_ppl['ci_low'], _ci_ppl['ci_high']] if _ci_ppl else None),
           'per_doc_log_ratio_ci95': ([_ci_log['ci_low'], _ci_log['ci_high']] if _ci_log else None),
           'incomplete_doc_seed_cells': int(incomplete),
           'ci_method': _ci_method,
           'breadth_note': ('doc_a_better_fraction counts units whose seed-mean Δ (a-b) is negative, i.e. lower PPL for arm a; '
                            'top_doc_abs_share is max|Δ_unit| / sum|Δ_unit| (a single unit carrying a large share means the '
                            'mean is outlier-sensitive).  Both are descriptive: units are not an arm randomisation.'),
           'article_fallback_reason': _article_fallback,
           'note': _unit_notes[_granularity]}
    if _variance_components is not None:
        out['variance_components'] = _variance_components
    else:
        out['variance_components_unavailable_reason'] = (
            _variance_components_reason
            or ('variance decomposition needs per-unit NLL vectors; this '
                'row-level readout has no unit NLL matrix'))
    if _granularity == 'article':
        out['n_articles'] = int(len(doc_deltas))
        _line_units = rows_a[0].get('ppl_per_doc') if rows_a else None
        out['n_line_docs'] = (len(_line_units)
                              if isinstance(_line_units, list) and _line_units
                              else None)
        if out['n_line_docs']:
            out['pseudoreplication_factor'] = (
                float(out['n_line_docs']) / float(out['n_articles']))
        if out['n_articles'] < 20:
            out['small_cluster_warning'] = (
                'fewer than 20 article clusters: the percentile interval is '
                'coarse and must be reported with its n; it is not a '
                'population interval')
    if _pooled_ci is not None:
        out.update({
            'pooled_delta_ppl': _pooled_ci['delta_ppl'],
            'pooled_delta_ppl_ci95': _pooled_ci['delta_ppl_ci95'],
            'pooled_delta_log_ratio': _pooled_ci['delta_log_ratio'],
            'pooled_delta_log_ratio_ci95': _pooled_ci['delta_log_ratio_ci95'],
            'pooled_ratio_ci95': _pooled_ci['ratio_ci95'],
            'pooled_ci_n_docs': _pooled_ci['n_docs'],
            'pooled_ci_n_units': _pooled_ci['n_docs'],
            'pooled_ci_unit': _granularity,
            'pooled_ci_df': _pooled_ci['df'],
            'pooled_ci_method': _pooled_ci['method'],
            'pooled_ci_note': ('interval for the token-pooled Δ, the same '
                               'estimand as the primary comparison; the '
                               f'{_granularity} units are clusters and the '
                               'trained seeds are conditioned on.  '
                               '`pooled_delta_log_ratio` is the seed-averaged '
                               'log PPL ratio, not the log of the ΔPPL.  This '
                               'is not a seed-level significance test.'),
        })
        if _unit_influence is not None:
            out['unit_influence'] = _unit_influence
        else:
            out['unit_influence_unavailable_reason'] = (
                _unit_influence_reason or
                'leave-one-unit-out influence not available for this cell')
        if _two_level_ci is not None:
            out.update({
                'pooled_two_level_delta_ppl': _two_level_ci['delta_ppl'],
                'pooled_two_level_ci95': _two_level_ci['delta_ppl_ci95'],
                'pooled_two_level_ratio_ci95': _two_level_ci['ratio_ci95'],
                'pooled_two_level_n_boot': _two_level_ci['n_boot'],
                'pooled_two_level_n_boot_valid': _two_level_ci['n_boot_valid'],
                'pooled_two_level_n_seeds': _two_level_ci['n_seeds'],
                'pooled_two_level_n_units': _two_level_ci['n_units'],
                'pooled_two_level_method': _two_level_ci['method'],
                'pooled_two_level_note': (
                    'seed x unit percentile bootstrap: both training-seed and '
                    'evaluation-unit sampling are resampled.  It is a '
                    'descriptive total-uncertainty interval, not a new test; '
                    'the exact sign-flip p/Holm/confirmed verdict is unchanged. '
                    'A dozen seed clusters can make a percentile interval '
                    'under-cover, so read it with n.'),
            })
            if _two_level_ci.get('warnings'):
                out['pooled_two_level_warnings'] = list(
                    _two_level_ci['warnings'])
        else:
            out['pooled_two_level_ci_unavailable_reason'] = (
                _two_level_ci_reason
                or 'two-level bootstrap not available for this cell')
    elif _granularity in ('document', 'article'):
        out['pooled_ci_unavailable_reason'] = (
            _pooled_ci_reason or
            'token-pooled cluster CI not available for this cell')
    _seed_deltas = primary_deltas
    if _seed_deltas is None:
        _seed_deltas = []
        for _ra, _rb in zip(rows_a, rows_b):
            _pa = L._finite_scalar((_ra or {}).get('ppl'))
            _pb = L._finite_scalar((_rb or {}).get('ppl'))
            if _pa is not None and _pb is not None:
                _seed_deltas.append(_pa - _pb)
    else:
        _seed_deltas = list(_seed_deltas)
    if _seed_deltas:
        _choice = _analysis_choice_block(out, _seed_deltas)
        if _choice is not None:
            out['analysis_choice_concordance'] = _choice
    return out


def _length_interaction_analysis(recs_a, recs_b, common, lengths, *, primary_len=None,
                                 alpha=0.05, min_n=6):
    """Paired effect modification of the arm contrast across evaluation lengths.

    PPL at different context lengths is not on one additive scale: the baseline
    PPL itself grows with length, so an absolute ΔPPL at L512 and at L4096 are
    not directly comparable.  The paired effect is therefore the per-seed
    ``log(PPL_a) - log(PPL_b)`` at each length (positive = arm a is worse),
    and the interaction is the within-seed change of that effect relative to a
    reference length (the training length 512 when present, otherwise the
    shortest evaluated length).

    This is deliberately **not** "compare the p-value at 512 with the p-value
    at 4096": a difference in significance is not a significant difference.
    The same seeds must be observed at every length; a seed missing one cell is
    dropped from the whole panel and named, never paired positionally.  The
    result is post-hoc/descriptive and never enters the frozen Holm family.
    """
    lens = sorted({int(x) for x in (lengths or [])})
    common = list(common or [])
    if len(lens) < 2 or len(common) < 2:
        return None
    effects = {}
    for Ln in lens:
        vals = []
        for s in common:
            pa = _ppl_at_len(recs_a.get(s), Ln)
            pb = _ppl_at_len(recs_b.get(s), Ln)
            if pa is None or pb is None:
                vals.append(float('nan'))
                continue
            try:
                vals.append(float(math.log(pa) - math.log(pb)))
            except (ValueError, OverflowError):
                vals.append(float('nan'))
        effects[str(Ln)] = vals
    reference = 512 if 512 in lens else lens[0]
    res = L.effect_modification(
        effects, reference=str(reference),
        condition_values={str(Ln): float(math.log2(Ln)) for Ln in lens},
        seeds=[int(s) for s in common], alpha=alpha, min_n=min_n)
    res.update({
        'lengths': [int(x) for x in lens],
        'reference_length': int(reference),
        'primary_length': (int(primary_len) if primary_len is not None else None),
        'effect_definition': ('log(PPL_a) - log(PPL_b) at each length; positive = arm a '
                              'has higher PPL (worse)'),
        'condition_axis': 'log2(eval length)',
        'role': ('post-hoc length effect-modification diagnostic; not in the frozen '
                 'Holm family and never changes `confirmed`'),
        'simple_effect_note': ('per-length simple effects are descriptive; compare the '
                               'interaction, not the significance of separate lengths'),
    })
    return res


def _test_probe_stats(probe_path, manifest, slice_fp):
    """Descriptive summary of the P5T test-split distractor probe.

    The probe is measured on the SAME held-out slice as the primary by-length
    panel (records carry `val_fp`), so admission here applies the same
    identity checks: frozen arm/seed whitelist, matching test fingerprint,
    matching code semantics and training recipe.  This is explicitly
    secondary/descriptive and never enters the Holm family or the
    confirmation decision."""
    out = {'path': probe_path, 'slice_fp': slice_fp, 'available': False,
           'n_records': 0, 'n_admitted': 0, 'n_excluded': 0,
           'cells': [], 'contrasts': {},
           'note': 'test-split descriptive secondary evidence; not part of the frozen primary family or the Holm correction'}
    if not slice_fp or not probe_path or not os.path.exists(probe_path):
        return out
    try:
        raw = json.load(open(probe_path, encoding='utf-8'))
    except Exception as e:
        out['error'] = f'{type(e).__name__}: {e}'
        return out
    if not isinstance(raw, dict):
        out['error'] = 'probe summary is not a JSON object'
        return out
    out['available'] = True
    out['n_records'] = len(raw)
    arms = set(manifest['arms'])
    seeds = {int(s) for s in manifest['seeds']}
    exp_code = L.code_stamp()
    impl = _frozen_implementation(manifest)
    want_train_fp = _frozen_train_fp(manifest)
    _require_train_fp = _train_fp_required(manifest) or any(
        isinstance(_r, dict) and bool(_r.get('train_fp')) for _r in raw.values())
    groups = {}
    excluded = []
    for key, r in raw.items():
        if not isinstance(r, dict):
            excluded.append(f'{key}: not a dict')
            continue
        why = None
        _seed_i = _record_int(r.get('seed'))
        _variant = r.get('variant')
        if r.get('synthesized'):
            # A reconstructed probe trajectory is provenance, not an
            # independent measurement.  The primary admission gate and the
            # v10 probe analysis already reject it; the test-probe path must
            # not become a back door into a descriptive contrast.
            why = 'synthesized reconstruction is not an independent measurement'
        elif not isinstance(_variant, str) or _variant not in arms:
            why = f"variant {_variant!r} is not one of the frozen arms"
        elif _seed_i is None or _seed_i not in seeds:
            why = f"seed {r.get('seed')!r} is outside the frozen whitelist"
        elif r.get('val_fp') != slice_fp:
            why = f"val_fp {str(r.get('val_fp'))[:12]}… does not match the frozen test slice {str(slice_fp)[:12]}…"
        elif exp_code is not None and r.get('_code') != exp_code:
            why = f"_code {r.get('_code')!r} does not match the frozen code semantics {exp_code!r}"
        elif _require_train_fp and not r.get('train_fp'):
            why = 'missing training-corpus fingerprint (train_fp) for the probe record'
        elif want_train_fp is not None and r.get('train_fp') != want_train_fp:
            why = f"train_fp {r.get('train_fp')} does not match the frozen canonical training fingerprint {want_train_fp}"
        elif _recipe_signature(manifest, r.get('variant')) is not None and not _recipe_matches_signature(r.get('recipe'), _recipe_signature(manifest, r.get('variant'))):
            why = 'recipe does not match the frozen training_recipe signature (including the per-arm mr)'
        elif _implementation_reason(r, impl) is not None:
            why = _implementation_reason(r, impl)
        elif not L.ppl_is_usable(r.get('ppl_mean')):
            why = 'ppl_mean is not a usable measurement'
        elif not isinstance(r.get('arm'), str) or not r.get('arm'):
            why = f"arm {r.get('arm')!r} is not a non-empty string"
        elif r.get('eval_len') is None or r.get('rho') is None:
            why = 'missing arm/eval_len/rho'
        elif _record_int(r.get('eval_len')) is None:
            why = f"eval_len {r.get('eval_len')!r} is not an integer"
        else:
            try:
                _rho_i = float(r.get('rho'))
            except (TypeError, ValueError, OverflowError):
                _rho_i = float('nan')
            if not math.isfinite(_rho_i):
                why = f"rho {r.get('rho')!r} is not numeric"
        if why:
            excluded.append(f'{key}: {why}')
            continue
        mode = str(r.get('mode', 'uniform'))
        ident = (r['variant'], r['arm'], _record_int(r['eval_len']), float(r['rho']), mode,
                 json.dumps(V10._fp_norm(r.get('probe_params')), sort_keys=True),
                 str(r.get('recipe')), str(r.get('_code')),
                 str(V10._recipe_norm(r.get('recipe'))))
        grp = groups.setdefault(ident, {})
        if _seed_i in grp:
            excluded.append(f'{key}: duplicate (cell, seed) identity — the pooled mean would be ambiguous')
            continue
        grp[_seed_i] = float(r['ppl_mean'])
    out['n_excluded'] = len(excluded)
    out['n_admitted'] = sum(len(g) for g in groups.values())
    by_visible = {}
    for ident, by_seed in groups.items():
        v, arm, Ln, rho, mode = ident[:5]
        by_visible.setdefault((v, arm, Ln, rho, mode), []).append((ident, by_seed))
    for (v, arm, Ln, rho, mode), entries in sorted(by_visible.items()):
        if len(entries) > 1:
            # multiple probe parameterisations/recipes for one readable cell:
            # report each separately, but never pool them.
            for ident, by_seed in entries:
                out['cells'].append({'variant': v, 'arm': arm, 'eval_len': Ln, 'rho': rho,
                                     'mode': mode, 'n': len(by_seed),
                                     'mean': float(np.mean(list(by_seed.values()))),
                                     'ppl_by_seed': {str(sd): p for sd, p in sorted(by_seed.items())},
                                     'ambiguous_cell': True})
            continue
        _ident, by_seed = entries[0]
        out['cells'].append({'variant': v, 'arm': arm, 'eval_len': Ln, 'rho': rho,
                             'mode': mode, 'n': len(by_seed),
                             'mean': float(np.mean(list(by_seed.values()))),
                             'ppl_by_seed': {str(sd): p for sd, p in sorted(by_seed.items())}})
    # Descriptive paired contrasts against the learned arm on the same slice.
    # A visible cell key deliberately omits probe_params/recipe/code so the
    # display can show one row per cell; a CONTRAST must not do the same.
    # Keep every full identity for a visible key: if it is ambiguous, or if
    # the two arms were probed under different parameters/recipes/code, their
    # difference would compare two different measurements.
    _idx = {}
    _contrast_exclusions = []
    for ident, by_seed in groups.items():
        _idx.setdefault(ident[:5], []).append((ident, by_seed))
    for (v, arm, Ln, rho, mode), learned_entries in sorted(_idx.items()):
        if arm != 'learned' or len(learned_entries) != 1:
            continue
        _learned_id, learned = learned_entries[0]
        for (_v, other_arm, _Ln, _rho, _mode), other_entries in sorted(_idx.items()):
            if (_v != v or _Ln != Ln or _rho != rho or _mode != mode
                    or other_arm == 'learned' or len(other_entries) != 1):
                continue
            _other_id, other = other_entries[0]
            # ident = (variant, arm, eval_len, rho, mode, probe_params,
            #          recipe, code, normalised recipe); compare the
            # measurement identity.  The normalised recipe ignores only the
            # parameter-matching `mr` field, exactly like v10's probe pairs.
            if _learned_id[5] != _other_id[5]:
                _contrast_exclusions.append(
                    f'contrast L{Ln} r{rho} [{mode}] {other_arm} - learned: '
                    'the two arms carry different probe_params — pairing them '
                    'would difference two different measurements')
                continue
            if _learned_id[7] != _other_id[7] or _learned_id[8] != _other_id[8]:
                _contrast_exclusions.append(
                    f'contrast L{Ln} r{rho} [{mode}] {other_arm} - learned: '
                    'the two arms carry different recipe/code stamps — '
                    'pairing them would mix two measurement semantics')
                continue
            common = sorted(set(learned) & set(other), key=float)
            if len(common) < 2:
                continue
            dl = [other[s] - learned[s] for s in common]
            res = V.exact_sign_permutation(dl)
            res['seeds'] = [int(s) for s in common]
            res['delta_definition'] = f'{other_arm} - learned (paired per seed, test split, descriptive)'
            out['contrasts'][f'L{Ln} r{rho} [{mode}] {other_arm} - learned'] = res
    out['contrast_exclusions'] = _contrast_exclusions
    if excluded:
        print('[p5t probe] EXCLUDED test-probe records (named, never silent):\n  - ' + '\n  - '.join(excluded[:20]) + ('' if len(excluded) <= 20 else f'\n  … {len(excluded) - 20} more'))
    if _contrast_exclusions:
        print('[p5t probe] EXCLUDED descriptive contrasts (named, never silent):\n  - ' + '\n  - '.join(_contrast_exclusions[:20]))
    return out


def _confirmation_robustness_block(stats, pairs, *, alpha, family_size, param_required,
                                   family_names=None):
    """Attach the post-hoc fragility/seed-drop readout to a stats dict.

    ``pairs`` maps each comparison name to ``{'deltas': [...], 'seeds': [...]}``
    — the same per-seed deltas that produced the primary p-values.  The
    frozen decision in ``stats['confirmed']`` is never recomputed or changed;
    this only adds a diagnosis of how much that decision depends on one seed.

    ``family_names`` is the DECLARED comparison list.  An omitted comparison
    has no deltas but still occupies a family slot, so the seed-drop jackknife
    must keep it at p=1 instead of shrinking m.
    """
    if not pairs:
        return None
    rob = L.confirmation_robustness(pairs, alpha=alpha,
                                    family_size=family_size,
                                    confirmed=stats.get('confirmed'),
                                    family_names=family_names)
    comps = stats.get('comparisons') or {}
    holm = stats.get('holm') or {}
    _rej = sorted(k for k, v in holm.items()
                  if v.get('p_holm') is not None
                  and float(v['p_holm']) < alpha)

    def _param_gate(k):
        if not param_required:
            return True
        return comps.get(k, {}).get('param_match_complete') is True

    # Replay the direction/parameter/minimum-n part of the frozen rule on each
    # seed-drop fold.  The complete-coverage gate is deliberately NOT
    # replayed: dropping a seed necessarily makes coverage false, so
    # including it would fail every fold by construction and say nothing
    # about sampling stability.
    full_nocov = sorted(
        k for k in _rej
        if comps.get(k, {}).get('direction_ok') is True
        and comps.get(k, {}).get('param_matched') is not False
        and _param_gate(k)
        and int(comps.get(k, {}).get('n_effective') or 0)
        >= int(comps.get(k, {}).get('min_seeds_for_confirmation')
               or len(comps.get(k, {}).get('seeds') or [])))
    fold_nocov = []
    for f in rob['family_jackknife']['folds']:
        keep = set(f['rejected'])
        row = []
        for k in full_nocov:
            if k not in keep:
                continue
            _comp = comps.get(k, {})
            _floor = int(_comp.get('min_seeds_for_confirmation')
                         or len(_comp.get('seeds') or []))
            if int(f.get('n_pairs', {}).get(k) or 0) < _floor:
                continue
            row.append(k)
        fold_nocov.append(sorted(row))
    rob['confirmed_without_coverage_full'] = full_nocov
    rob['confirmed_without_coverage_by_fold'] = {
        str(f['seed']): s for f, s in
        zip(rob['family_jackknife']['folds'], fold_nocov)}
    rob['confirmed_without_coverage_stable'] = bool(
        all(s == full_nocov for s in fold_nocov))
    rob['coverage_gate_note'] = (
        'the complete-coverage rule is not replayed inside the jackknife: '
        'dropping one seed always makes coverage_ok false, which would fail '
        'every fold by construction.  The jackknife asks only whether the '
        'statistical/direction/minimum-n verdict survives; the frozen decision still '
        'additionally requires complete coverage.')
    stats['robustness'] = rob
    return rob


def _doc_ci_markdown_cell(dsm):
    """Render the document/row sampling interval without mixing estimands.

    Prefer the token-pooled cluster-robust interval (same estimand as the
    primary Δ); fall back to the labelled equal-weighted unit bootstrap only
    when the pooled interval is unavailable.  Kept pure so the report wording
    is testable without a panel.
    """
    dsm = dsm if isinstance(dsm, dict) else {}
    _u = dsm.get('unit') or dsm.get('granularity')
    unit = {'article': '文章', 'document': '文档'}.get(_u, '行')

    def _fmt(ci):
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            try:
                lo, hi = float(ci[0]), float(ci[1])
            except (TypeError, ValueError, OverflowError):
                return None
            if math.isfinite(lo) and math.isfinite(hi):
                return f'[{lo:+.2f}, {hi:+.2f}]'
        return None

    pooled = _fmt(dsm.get('pooled_delta_ppl_ci95'))
    if pooled is not None:
        joint = _fmt(dsm.get('pooled_two_level_ci95'))
        if joint is not None:
            return (f'{pooled}（{unit} cluster，token 池化；'
                    f'seed×{unit} 两层 bootstrap {joint}）')
        return f'{pooled}（{unit} cluster，token 池化）'
    unit_ci = _fmt(dsm.get('per_doc_delta_ppl_ci95'))
    if unit_ci is not None:
        return f'{unit_ci}（{unit} bootstrap，等权单元）'
    return '—'


def _doc_influence_markdown(ft):
    """Render the post-hoc leave-one-unit-out influence table.

    Pure function of the final-test stats dict, so the wording and the
    "does not change the frozen decision" boundary are testable without a
    panel.  It deliberately uses the scale-free log-ratio influence for the
    fragility flag: a PPL difference is expressed on the baseline's scale,
    while the log ratio is the multiplicative effect the arms are compared on.
    """
    comps = (ft or {}).get('comparisons') if isinstance(ft, dict) else None
    if not isinstance(comps, dict):
        return []
    rows = []
    missing = []
    for name, comp in comps.items():
        if not isinstance(comp, dict):
            continue
        dsm = comp.get('doc_sampling') or {}
        inf = dsm.get('unit_influence') if isinstance(dsm, dict) else None
        if not isinstance(inf, dict) or not inf.get('available'):
            if (isinstance(dsm, dict)
                    and dsm.get('pooled_delta_ppl_ci95') is not None):
                missing.append(
                    f'{name}（{dsm.get("unit_influence_unavailable_reason") or "缺少影响块"}）')
            continue
        unit = {'article': '文章', 'document': '文档'}.get(
            dsm.get('unit') or dsm.get('granularity'), '单元')
        full = inf.get('full_delta_ppl')
        idx = inf.get('most_influential_index')
        loo = inf.get('loo_delta_ppl') or []
        loo_delta = (loo[idx] if isinstance(idx, int) and 0 <= idx < len(loo)
                     else None)
        rel_lr = inf.get('max_abs_relative_change_log_ratio')
        rel_ppl = inf.get('max_abs_relative_change_ppl')
        rel_txt = '—'
        if isinstance(rel_lr, (int, float)) and math.isfinite(rel_lr):
            rel_txt = f'log {100.0 * rel_lr:.0f}%'
        if isinstance(rel_ppl, (int, float)) and math.isfinite(rel_ppl):
            rel_txt += (f' / PPL {100.0 * rel_ppl:.0f}%'
                        if rel_txt != '—' else f'PPL {100.0 * rel_ppl:.0f}%')
        same = inf.get('sign_stable_log_ratio')
        same_txt = ('—' if same is None else ('是' if same else '否（翻转）'))
        ci = inf.get('jackknife_ci95_ppl')
        ci_txt = '—'
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            try:
                lo, hi = float(ci[0]), float(ci[1])
                if math.isfinite(lo) and math.isfinite(hi):
                    ci_txt = f'[{lo:+.2f}, {hi:+.2f}]'
            except (TypeError, ValueError, OverflowError):
                pass
        _frag = inf.get('fragile_to_one_unit')
        if _frag is None:
            flag = '—（无方向）'
        elif _frag:
            flag = '⚠ 单单元主导'
        else:
            flag = '未见单单元翻转'
        uid = inf.get('most_influential_unit')
        uid_txt = (str(uid)[:16] if uid is not None else '—')
        if inf.get('label_source') == 'index_fallback':
            uid_txt += '（索引）'
        tied = inf.get('most_influential_tied_count')
        if isinstance(tied, int) and tied > 1:
            uid_txt += f'（并列 {tied}）'
        full_txt = f'{full:+.3f}' if isinstance(full, (int, float)) and math.isfinite(full) else '—'
        loo_txt = (f'{loo_delta:+.3f}'
                   if isinstance(loo_delta, (int, float)) and math.isfinite(loo_delta)
                   else '—')
        rows.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |'
                    % (name, unit, inf.get('n_units', '—'), full_txt, uid_txt,
                       loo_txt, rel_txt, same_txt, ci_txt, flag))
    if not rows and not missing:
        return []
    out = ['### 单元影响诊断（post-hoc；不改变上面的冻结判定）', '']
    if rows:
        out += ['| 比较 | 单位 | n | 全 panel ΔPPL | 最敏感单元 | 删除后 ΔPPL | 最大单单元变化 | 删除后同向 | jackknife 95% CI | 标记 |',
                '|---|---|---|---|---|---|---|---|---|---|']
        out.extend(rows)
        out += ['',
                '> 定义：对每个评测单位 i 删除它，再用**同一 token 池化估计量**重算 ΔPPL；'
                '`最敏感单元` 是使 log 效应变化最大的单位，`jackknife 95% CI` 是 delete-one-unit '
                'jackknife 区间（条件在已训练种子上）。`单单元主导` 指删除一个单位使 log 效应变化 ≥100%；'
                '方向翻转单独标记。该表是**对已发生数据的 post-hoc 描述**：不进入冻结 p/Holm/confirmed，'
                '不把评测单位当成随机化，也不能据此做总体推断；删除单位后描述的是「冻结切片减一个单位」。'
                '逐单元完整 LOO 向量与 delta-method influence 见 `analysis_v12/test_stats.json` 的 '
                '`doc_sampling.unit_influence`。', '']
    if missing:
        out += ['> 以下比较有 pooled CI 但没有可用的单元影响块，'
                '单单元敏感性**未验证**：' + '；'.join(missing) + '。', '']
    frag = [r.split('|')[1].strip() for r in rows if '单单元主导' in r]
    if frag:
        out += ['**解读提示**：' + '、'.join(frag) +
                ' 的方向/幅度依赖单个评测单位；引用时必须同时报告该敏感度，'
                '不能把结论描述为「对评测文档稳健」。', '']
    return out


def _variance_components_markdown(ft):
    """Render the crossed seed x unit variance decomposition.

    Pure function of the final-test stats dict: the table is a descriptive
    companion to the two-level bootstrap, not a new test.  The shares answer
    "where is the noise?" and the allocation column answers "which extra cell
    would reduce the mean variance most?".
    """
    comps = (ft or {}).get('comparisons') if isinstance(ft, dict) else None
    if not isinstance(comps, dict):
        return []

    def _pct(value):
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return f'{100.0 * float(value):.0f}%'
        return '—'

    def _num(value, fmt='{:.4g}'):
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return fmt.format(float(value))
        return '—'

    rows = []
    missing = []
    for name, comp in comps.items():
        if not isinstance(comp, dict):
            continue
        dsm = comp.get('doc_sampling')
        dsm = dsm if isinstance(dsm, dict) else {}
        vc = dsm.get('variance_components')
        if not isinstance(vc, dict):
            if (dsm.get('variance_components_unavailable_reason')
                    or dsm.get('pooled_delta_ppl_ci95') is not None):
                missing.append(
                    f'{name}（{dsm.get("variance_components_unavailable_reason") or "缺少方差分解块"}）')
            continue
        unit = {'article': '文章', 'document': '文档'}.get(
            dsm.get('unit') or dsm.get('granularity'), '单元')
        status = vc.get('status')
        if status != 'ok':
            rows.append('| %s | %s | %s×%s | — | — | — | — | %s | — |'
                        % (name, unit, vc.get('n_seeds', '—'),
                           vc.get('n_units', '—'),
                           '；'.join(vc.get('warnings') or []) or status or '—'))
            continue
        c = vc.get('components') or {}
        eff = vc.get('effective_n') or {}
        ci = vc.get('mean_ci') or {}
        alloc = vc.get('allocation') or {}
        nxt = alloc.get('next_cell')
        nxt_txt = {'seed': '加种子', 'unit': '加评测单元'}.get(nxt, '—')
        ci_txt = '—'
        if isinstance(ci.get('low'), (int, float)) and isinstance(ci.get('high'), (int, float)):
            ci_txt = f'[{ci["low"]:+.4f}, {ci["high"]:+.4f}]'
        rows.append('| %s | %s | %s×%s | %s | %s | %s | %s / %s× | %s | %s |'
                    % (name, unit, vc.get('n_seeds', '—'),
                       vc.get('n_units', '—'),
                       _pct((c.get('seed') or {}).get('share')),
                       _pct((c.get('unit') or {}).get('share')),
                       _pct((c.get('residual') or {}).get('share')),
                       _num(eff.get('equivalent'), '{:.1f}'),
                       _num(eff.get('design_effect'), '{:.1f}'),
                       nxt_txt, ci_txt))
    if not rows and not missing:
        return []
    out = ['### 种子 × 评测单元方差分解（描述性；不改变冻结判定）', '']
    if rows:
        out += ['| 比较 | 单元 | S×D | seed 方差占比 | unit 方差占比 | 残差占比 | 等效独立单元 / 设计效应 | 下一格建议 | 均值 95% CI（NLL/token） |',
                '|---|---|---|---|---|---|---|---|---|']
        out.extend(rows)
        out += ['',
                '> 模型：`y[s,u] = μ + seed[s] + unit[u] + residual[s,u]`，矩估计'
                '（two-way ANOVA）。`seed` 占比高表示主要不确定性来自训练随机性，'
                '加种子的边际收益更大；`unit` 占比高表示结论依赖评测单元组成，'
                '应扩大/更换评测单元；残差占比高表示单元内 seed×unit 交互大，'
                '单靠加种子或加单元都不够。`等效独立单元` = 总方差 / 面板均值的方差，'
                '`设计效应` = 实际格数 / 等效独立单元；二者越接近 1，格子越接近独立。'
                '均值 CI 对**超总体**（seed 与 unit 都视为随机）而言；'
                '`mean_ci_conditional_on_units` 固定当前评测单元，只覆盖 seed/残差抽样。'
                '这些是矩估计与 t 近似，不是拟合的 hierarchical/Bayes 模型；'
                '种子/单元很少时方差分量本身很不精确，不能据此宣称总体方差。', '']
    if missing:
        out += ['> 以下比较没有可用的方差分解块，**未验证**：' + '；'.join(missing) + '。', '']
    return out


def _analysis_choice_markdown(ft):
    """Render the post-hoc analysis-choice concordance table.

    Pure function of the final-test stats dict so the wording and the
    "does not change the frozen decision" boundary can be tested without a
    panel.  The block is descriptive: it neither adds a p-value nor selects a
    preferred specification.
    """
    comps = (ft or {}).get('comparisons') if isinstance(ft, dict) else None
    if not isinstance(comps, dict):
        return []
    labels = {
        'stable': '稳定',
        'precision_sensitive': '⚠ 精度口径敏感',
        'direction_unstable': '⚠ 方向翻转',
        'zero_effect': '零效应',
        'insufficient_specs': '口径不足',
    }
    rows = []
    details = []
    missing = []
    fragile = []
    corrupt = []

    def _mark_corrupt(name):
        if name not in corrupt:
            corrupt.append(name)

    for name, comp in comps.items():
        if not isinstance(comp, dict):
            continue
        dsm = comp.get('doc_sampling') or {}
        _has_cc = isinstance(dsm, dict) and 'analysis_choice_concordance' in dsm
        cc = dsm.get('analysis_choice_concordance') if _has_cc else None
        if not isinstance(cc, dict):
            if _has_cc:
                _mark_corrupt(name)
            elif (isinstance(dsm, dict)
                    and (dsm.get('per_doc_delta_mean') is not None
                         or dsm.get('pooled_delta_ppl') is not None)):
                missing.append(name)
            continue
        status = cc.get('status')
        if not isinstance(status, str) or status not in labels:
            _mark_corrupt(name)
            continue
        n_specs = cc.get('n_specs')
        n_avail = cc.get('n_available')
        n_same = cc.get('n_same_as_primary')
        n_opp = cc.get('n_opposite_to_primary')
        n_excl = cc.get('n_excluding_zero')
        n_ci = cc.get('n_with_ci')
        psign = cc.get('primary_sign')
        if isinstance(psign, (bool, np.bool_)):
            psign = None
        else:
            try:
                psign = float(psign) if psign is not None else None
            except (TypeError, ValueError, OverflowError):
                psign = None
        if psign is None and cc.get('primary_sign') is not None:
            _mark_corrupt(name)
        if psign is None or psign == 0:
            sign_txt = '零/无方向'
        else:
            sign_txt = '负（a 更低）' if psign < 0 else '正（a 更高）'
        same_txt = f'{n_same}/{n_opp}' if n_same is not None else '—'
        ci_txt = (f'{n_excl}/{n_ci}' if n_ci is not None else '—')
        _pc = cc.get('precision_concordant')
        if isinstance(_pc, (bool, np.bool_)):
            if not bool(_pc):
                ci_txt += '（不一致）'
        elif _pc is not None:
            _mark_corrupt(name)
        er = cc.get('effect_range_ppl')
        if isinstance(er, dict) and all(
                isinstance(er.get(k), (int, float))
                and not isinstance(er.get(k), (bool, np.bool_))
                for k in ('min', 'max')):
            rng = f'[{er["min"]:+.3f}, {er["max"]:+.3f}]'
            if (isinstance(er.get('span'), (int, float))
                    and not isinstance(er.get('span'), (bool, np.bool_))):
                rng += f'（跨度 {er["span"]:.3f}）'
        else:
            rng = '—'
            if er is not None:
                _mark_corrupt(name)
        flag = labels[status]
        _frag = cc.get('fragile')
        if isinstance(_frag, (bool, np.bool_)):
            if bool(_frag):
                fragile.append(name)
        elif _frag is not None:
            _mark_corrupt(name)
        rows.append('| %s | %s/%s | %s | %s | %s | %s | %s |'
                    % (name, n_avail, n_specs, sign_txt, same_txt,
                       ci_txt, rng, flag))
        specs = cc.get('specifications')
        if not isinstance(specs, list):
            specs = []
            _mark_corrupt(name)
        if status == 'direction_unstable':
            opp = [str(s.get('spec')) for s in specs
                   if isinstance(s, dict) and s.get('available')
                   and s.get('sign') not in (None, 0, psign)]
            if opp:
                details.append(f'{name}: 与主口径反向的规格为 {opp}')
        elif status == 'precision_sensitive':
            cross = cc.get('crossing_zero_specs') or []
            excl = cc.get('excluding_zero_specs') or []
            if cross or excl:
                details.append(
                    f'{name}: 排除零的规格 {excl or "无"}；跨零的规格 '
                    f'{cross or "无"}')
    if not rows and not missing and not corrupt:
        return []
    out = ['### 分析口径一致性（post-hoc；不改变上面的冻结判定）', '']
    if rows:
        out += ['| 比较 | 可用/总口径 | 主方向 | 同向/反向 | 区间排除零 | PPL Δ 范围 | 标记 |',
                '|---|---|---|---|---|---|---|']
        out.extend(rows)
        out += ['',
                '> 定义：同一 panel 的多种口径（逐种子 mean/median/trimmed mean、'
                'token 池化、逐评测单元 mean/median/log-ratio/win-balance）在方向与'
                '「区间是否排除零」上是否一致。`同向/反向` 相对主口径。'
                '方向一致不等于估计量相同：逐种子、逐单元与 token 池化回答不同问题，'
                '区间宽度也依赖单位。`精度口径敏感` 只说明带区间口径的零排除结论不一致，'
                '不是方向矛盾。该表是**对已发生数据的 post-hoc 描述**：不新增检验/多重校正，'
                '不进入冻结 p/Holm/confirmed，也不能用来挑选最有利口径；'
                '引用时不得只报单一聚合而隐去本表。', '']
    if details:
        out += ['**口径细节**：' + '；'.join(details) + '。', '']
    if missing:
        out += ['> 以下比较有 doc_sampling 但缺少分析口径一致性块'
                '（旧产物或未重新生成分析），该敏感性**未验证**：'
                + '、'.join(missing) + '。', '']
    if corrupt:
        out += ['> 以下比较的分析口径一致性块损坏或状态未知，该敏感性'
                '**未验证**：' + '、'.join(corrupt) + '。', '']
    if fragile:
        out += ['**解读提示**：' + '、'.join(fragile) +
                ' 的方向或精度结论依赖分析口径；引用时必须同时报告本表，'
                '不能把结论描述为「换一种聚合仍然稳健」。', '']
    return out


def _robustness_markdown(rob):
    """Render the post-hoc fragility/seed-drop table for the v12 report.

    Kept as a pure function of the JSON block so the wording and the
    "does not change the frozen decision" boundary can be unit-tested without
    a GPU panel or a full report build.
    """
    out = []
    if not (isinstance(rob, dict) and rob.get('per_comparison')):
        return out
    A = out.append
    A('### 结论稳健性（post-hoc 诊断；不改变上面的冻结判定）')
    A('')
    A('| 比较 | Δ mean [95% CI]（seed） | 同向种子 | 丢弃最弱种子仍显著 k | 翻转最小同向种子后失效 k | 逐种子删除的 Holm p 范围 | 最敏感种子 |')
    A('|---|---|---|---|---|---|---|')
    fjk = rob.get('family_jackknife') or {}
    fjc = fjk.get('per_comparison') or {}
    for rk, rr in rob['per_comparison'].items():
        n = rr.get('n')
        same = rr.get('same_sign_count')
        req = rr.get('min_same_sign_for_family_gate')
        sames = f'{same}/{n}' if same is not None and n else '—'
        if req is not None:
            sames += f'（族门等幅投影 ≥{req}）'
        _lo, _hi = rr.get('ci95_low'), rr.get('ci95_high')
        if (isinstance(rr.get('mean'), (int, float))
                and isinstance(_lo, (int, float)) and isinstance(_hi, (int, float))):
            citxt = f'{rr["mean"]:+.3f} [{_lo:+.3f}, {_hi:+.3f}]'
        else:
            citxt = '—'
        drop = rr.get('drop_fragility')
        flip = rr.get('flip_fragility')
        reason = {'p>=alpha': 'p≥α', 'direction reversed': '方向反转',
                  'not significant at the full panel': '整组已不显著',
                  'no mean direction': '无方向'}.get(
                      rr.get('flip_stop_reason'), rr.get('flip_stop_reason') or '—')
        fliptxt = f'{flip}（{reason}）' if isinstance(flip, int) else '—'
        fj = fjc.get(rk) or {}
        lo, hi = fj.get('p_holm_min'), fj.get('p_holm_max')
        rng = (f'{lo:.3g}–{hi:.3g}'
               if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) else '—')
        ms = rr.get('most_influential_seed')
        mp = rr.get('most_influential_p_shift')
        mstxt = ((f'种子 {ms}' + (f'（|Δp|={mp:.3g}）' if isinstance(mp, (int, float)) else ''))
                 if ms is not None else '—')
        _tied = rr.get('most_influential_tied_count')
        if isinstance(_tied, int) and _tied > 1 and mstxt != '—':
            mstxt += f'；并列 {_tied} 个'
        A(f'| {rk} | {citxt} | {sames} | {drop if isinstance(drop, int) else "—"} | '
          f'{fliptxt} | {rng} | {mstxt} |')
    A('')
    rejf = fjk.get('rejected_full') or []
    always = fjk.get('always_rejected') or []
    unstable = fjk.get('unstable_folds') or []
    A(f"- 整组 Holm 拒绝（全 panel）：{sorted(rejf) if rejf else '无'}。")
    if fjk.get('all_rejections_stable'):
        A('- 逐种子删除检验：**每一个**单种子删除后，Holm 拒绝集合都保持不变。')
    else:
        A(f"- 逐种子删除检验：**不稳定**。删除这些种子会改变 Holm 拒绝集合：{sorted(unstable)}；"
          f"始终被拒绝的比较：{sorted(always) if always else '无'}。")
    nocov_stable = rob.get('confirmed_without_coverage_stable')
    nocov_full = rob.get('confirmed_without_coverage_full') or []
    A(f"- 方向/参数/最小 n 判定（不含完整覆盖门）：全 panel = {sorted(nocov_full) if nocov_full else '无'}；"
      f"逐种子删除后保持不变 = {nocov_stable}。")
    A('')
    A('> 定义：`丢弃最弱种子仍显著 k` = 反复删除当前 |Δ| 最小的种子、精确符号翻转 p 仍 < α 的最大次数；'
      '`翻转最小同向种子后失效 k` = 按 |Δ| 从小到大翻转同向种子，直到 p ≥ α 或均值方向反转的次数（贪心，不是精确最小集）。'
      '两者都是**对已发生数据的 post-hoc 敏感度读数**，不进入冻结决策，也不改变上面的 p/Holm/confirmed。'
      '完整覆盖门不在 jackknife 内重放（删任何一个种子都会使覆盖失败），因此上一条只重放方向/参数/最小 n 判定。')
    A('')
    if fjk.get('all_rejections_stable') is False:
        A('**解读提示**：至少一个冻结判定在删除单个种子后不再成立。这不改变预注册的确认结论，'
          '但该结论**不能被描述为「对种子稳健」**；引用时必须同时报告这一敏感度。')
        A('')
    return out


def _length_interaction_markdown(name, li):
    """Render the post-hoc length effect-modification table.

    Pure formatting so the "interaction, not two significance verdicts" caveat
    and the table can be unit-tested without a GPU panel or a report build.
    """
    out = []
    if not (isinstance(li, dict) and li.get('conditions')):
        return out
    A = out.append

    def _fmt_ci(cell):
        lo, hi = cell.get('ci95_low'), cell.get('ci95_high')
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            if math.isfinite(float(lo)) and math.isfinite(float(hi)):
                return f'[{float(lo):+.4f}, {float(hi):+.4f}]'
        return '—'

    def _fmt_p(v):
        return f'{float(v):.4g}' if isinstance(v, (int, float)) and math.isfinite(float(v)) else '—'

    ref = str(li.get('reference'))
    A(f'**`{name}`** — 参考条件 L{li.get("reference_length", ref)}；'
      f'效应 = log(PPL_a/PPL_b)，正 = a 更差；交互 = 该长度的效应减参考长度的效应。')
    A('')
    A('| 条件 | n | 简单效应 mean [95% CI] | 与参考的交互 Δ [95% CI] | 效应比值变化 | 交互 p（原始） | 交互 p（Holm） |')
    A('|---|---|---|---|---|---|---|')
    _simples = li.get('simple_effects') or {}
    _inters = li.get('interactions') or {}
    for lab in li.get('conditions') or []:
        cell = _simples.get(lab) or {}
        mean = cell.get('mean')
        mtxt = f'{float(mean):+.4f} {_fmt_ci(cell)}' if isinstance(mean, (int, float)) else '—'
        if lab == ref:
            itxt, rtxt, praw, pholm = '—（参考）', '—', '—', '—'
        else:
            inter = _inters.get(lab) or {}
            dm = inter.get('delta_mean')
            itxt = (f'{float(dm):+.4f} {_fmt_ci(inter)}'
                    if isinstance(dm, (int, float)) else '—')
            ratio = inter.get('ratio')
            rtxt = (f'×{float(ratio):.3f}' if isinstance(ratio, (int, float))
                    and math.isfinite(float(ratio)) else '—')
            praw = _fmt_p(inter.get('p_exact_signflip'))
            pholm = _fmt_p(inter.get('p_holm'))
        A(f'| L{lab} | {cell.get("n", 0)} | {mtxt} | {itxt} | {rtxt} | {praw} | {pholm} |')
    A('')
    tr = li.get('trend') or {}
    if isinstance(tr, dict) and tr.get('n'):
        A(f"- 趋势（逐种子斜率 / {li.get('condition_axis', 'condition axis')}）："
          f"{float(tr.get('mean')):+.4f} {_fmt_ci(tr)}，p={_fmt_p(tr.get('p_exact_signflip'))}，"
          f"n={tr.get('n')}；这是 post-hoc 趋势读数，不进冻结族。")
    else:
        A(f"- 趋势：不可估（{li.get('trend_reason') or '—'}）。")
    _n, _min = li.get('n_complete'), li.get('min_n')
    if isinstance(_n, int) and isinstance(_min, int) and _n < _min:
        A(f"- 样本：只有 {_n} 个完整种子（< min_n={_min}），交互 p 仅作描述，不能称为已确认。")
    _dropped = li.get('dropped_seeds') or []
    if _dropped:
        A(f"- 完整种子集合：缺少任一长度可用单元的种子 {_dropped} 已从整个交互面板剔除（不是按长度各自配对）。")
    A('')
    A('> 这是 **post-hoc 效应修饰（interaction）诊断**，不属于冻结 Holm 族，也不改变 `confirmed`。'
      '不显著 ≠ 各长度效应相同；「L512 显著、L4096 不显著」也不能当作长度依赖证据——'
      '必须看本表的交互检验。简单效应与趋势仅作描述。')
    A('')
    return out
