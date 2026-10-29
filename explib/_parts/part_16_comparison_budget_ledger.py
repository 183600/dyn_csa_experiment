

def _int_ratio_within(numerator, denominator, tolerance):
    """Exact ``numerator / denominator <= tolerance`` for integer counts.

    The ratio itself may be too small to float (e.g. two 401-digit counts
    differing by one), but the comparison against a gate tolerance is still
    well defined.  Return None only when the inputs cannot form a ratio.
    """
    try:
        numerator, denominator = int(numerator), int(denominator)
        tolerance = float(tolerance)
    except (TypeError, ValueError, OverflowError):
        return None
    if denominator <= 0 or numerator < 0 or not math.isfinite(tolerance):
        return None
    try:
        from fractions import Fraction
        return bool(Fraction(numerator, denominator)
                    <= Fraction.from_float(tolerance))
    except (OverflowError, ValueError):
        return None


def comparison_budget_ledger(manifest, *, factory=None, seq_lens=(512, 4096),
                             read_tolerance=0.10, param_tolerance=0.01,
                             build_params=True):
    """Zero-GPU token/parameter/read/compute budget ledger per comparison.

    This is the design-time answer to "is the dense baseline and each CSA/HCA
    arm controlled on the same token and compute/read budget?".  For every arm
    it records the training-token budget, parameter total and analytic read
    budget; for every declared comparison it reports which budgets match and
    which dedicated control it actually is.  A `mechanism_claim_allowed=true`
    comparison fails the audit when any of the following is not matched:

      * training tokens (same steps x batch x sequence, same declared corpus
        exposure);
      * parameters (total within `param_tolerance`, plus no explicit
        per-arm mismatch);
      * read budget (analytic token reads within `read_tolerance`, or a
        calibration artifact for a dynamic-vs-fixed pair);
      * an identifiable dedicated control family (selection, segmentation,
        task-policy, sink, or parameter-proxy) instead of only "vs dense".

    The audit does not decide significance and never reads a result file.
    """
    for _name, _val in (('read_tolerance', read_tolerance),
                        ('param_tolerance', param_tolerance)):
        if isinstance(_val, (bool, np.bool_)):
            raise ValueError(f'{_name} must be a finite non-negative number, got {_val!r}')
        try:
            _f = float(_val)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'{_name} must be a finite non-negative number, got {_val!r}') from None
        if not math.isfinite(_f) or _f < 0.0:
            raise ValueError(f'{_name} must be a finite non-negative number, got {_val!r}')
    _seq_lens = []
    if not isinstance(seq_lens, (str, bytes)):
        try:
            for _x in seq_lens:
                _v = _design_int(_x)
                if _v is None or _v <= 0:
                    _seq_lens = []
                    break
                _seq_lens.append(_v)
        except TypeError:
            _seq_lens = []
    out = {'status': 'ok', 'manifest': None, 'factory': None,
           'seq_lens': (_seq_lens or [512]), 'arms': {}, 'comparisons': {},
           'dedicated_controls': {'token_matched_arms': [],
                                  'dense_token_matched_arms': [],
                                  'parameter_proxy_arms': [],
                                  'read_matched_arms': [],
                                  'fixed_read_matched_arms': [],
                                  'selection_control_pairs': [],
                                  'segmentation_control_pairs': [],
                                  'task_policy_pairs': [],
                                  'sink_control_pairs': [],
                                  'information_rich_allblocks': []},
           'violations': [], 'warnings': [], 'problems': [],
           'policy': {'read_tolerance': float(read_tolerance),
                      'param_tolerance': float(param_tolerance),
                      'token_budget_rule': ('steps*bs*seq_len is the CONSUMED '
                                            'token budget and must be identical '
                                            'for every arm; n_train_tokens is the '
                                            'served cache and must also be shared'),
                      'claim_rule': ('a mechanism claim requires matched tokens, '
                                     'matched parameters and a matched read '
                                     'budget plus a dedicated control family'),
                      'compute_rule': ('compute_6nd_matched uses params x consumed '
                                       'tokens only; compute_proxy_matched also '
                                       'requires the selected-read budget to match. '
                                       'Neither is a measured FLOPs number'),
                      'scale_note': ('every budget is parsed from the manifest '
                                     'signature and normalized per token/query; '
                                     'rerun the ledger for a new d/L/seq scale '
                                     'instead of assuming the small-scale result '
                                     'transfers')}}
    if not isinstance(manifest, dict):
        out['status'] = 'fail'
        out['problems'].append(f'manifest must be an object, got {type(manifest).__name__}')
        return out
    raw_arms = manifest.get('arms')
    if not isinstance(raw_arms, list) or not raw_arms:
        out['status'] = 'fail'
        out['problems'].append('manifest.arms must be a non-empty list')
        return out
    arms = []
    for i, a in enumerate(raw_arms):
        if not isinstance(a, str) or not a.strip():
            out['problems'].append(
                f'manifest.arms[{i}]={a!r} is not a non-empty string')
            continue
        arms.append(a.strip())
    if len(set(arms)) != len(arms):
        out['problems'].append('manifest.arms contains duplicate names')
    if not arms:
        out['status'] = 'fail'
        out['problems'].append('manifest.arms has no usable arm names')
        return out
    binding = manifest.get('binding') if isinstance(manifest.get('binding'), dict) else {}
    sig = binding.get('training_recipe_signature')
    if not isinstance(sig, dict):
        sig = binding.get('training_recipe')
    if not isinstance(sig, dict):
        out['problems'].append('no training_recipe_signature in binding; the token budget is unverifiable')
        sig = {}
    def _sig_int(key):
        raw = sig.get(key, 0)
        v = _design_int(0 if raw is None else raw)
        return v if v is not None else 0

    steps, bs, sl, nt = (_sig_int('steps'), _sig_int('bs'),
                         _sig_int('sl'), _sig_int('nt'))
    effective_tokens = steps * bs * sl
    if steps <= 0 or bs <= 0 or sl <= 0:
        out['problems'].append(
            f'training recipe lacks a positive steps/bs/sl ({steps}, {bs}, {sl}); '
            'the consumed token budget is unverifiable')
    if nt <= 0:
        out['problems'].append('training recipe lacks a positive n_train_tokens')
    # `nt` in the frozen signature is the TRAIN-CACHE size (how many tokens
    # the loader serves), not the number of tokens consumed by training.  The
    # run consumes steps*bs*sl tokens over that cache (possibly several
    # epochs).  Conflating the two would falsely reject the frozen plan, so the
    # ledger records both and matches the CONSUMED budget across arms.
    out['token_budget'] = {
        'steps': steps, 'batch_size': bs, 'seq_len': sl,
        'train_cache_tokens': nt,
        'consumed_tokens': effective_tokens,
        'shared_by_all_arms': bool(effective_tokens > 0),
        'definitions': {
            'train_cache_tokens': 'n_train_tokens in the recipe signature: the served train cache size',
            'consumed_tokens': 'steps * batch_size * seq_len: tokens actually consumed by the run',
            # A design-time plan may declare a budget far beyond float64's
            # range.  The ratio is then unverifiable, not a crash: use the
            # exact-integer helper shared with the scale-ladder audit.
            'epochs_over_cache': (_finite_int_ratio(effective_tokens, nt)
                                  if nt > 0 else None),
        },
    }

    def _arm_token_budget(arm):
        per_arm = (binding.get('per_arm_recipe_signature')
                   if isinstance(binding.get('per_arm_recipe_signature'), dict) else {})
        merged = dict(sig)
        ent = per_arm.get(arm)
        if isinstance(ent, dict):
            for _k in ('steps', 'bs', 'sl', 'nt'):
                if _k in ent:
                    merged[_k] = ent[_k]
        def _m(key):
            raw = merged.get(key, 0)
            v = _design_int(0 if raw is None else raw)
            return v if v is not None else 0
        return (_m('steps') * _m('bs') * _m('sl'), _m('nt'))
    _fac, _fac_name = (factory, getattr(factory, '__name__', repr(factory))) if factory is not None else _ledger_factory()
    out['factory'] = _fac_name
    cfgs_by_arm = {}
    for arm in arms:
        _consumed, _cache = _arm_token_budget(arm)
        rec = {'variant': arm, 'mr': None, 'mr_source': None, 'params': None,
               'token_budget': _consumed, 'train_cache_tokens': _cache,
               'params_per_token': None, 'read_budget': {}, 'errors': []}
        try:
            cfgs = _fac(int(sig.get('L', 6)), arm)
            cfgs_by_arm[arm] = cfgs
        except Exception as _e:
            rec['errors'].append(f'arm construction failed ({type(_e).__name__}: {_e})')
            out['arms'][arm] = rec
            continue
        mr, mr_src = _ledger_arm_mr(manifest, arm, sig)
        rec['mr'], rec['mr_source'] = mr, mr_src
        if build_params:
            if mr is None:
                rec['errors'].append('MLP ratio is unpinned/uncomputable; parameter matching is unverifiable')
            else:
                try:
                    _m = SmallGPT(int(sig.get('v', 8192)), int(sig.get('d', 256)),
                                  int(sig.get('L', 6)), int(sig.get('H', 8)),
                                  int(sig.get('Dh', 32)), int(sig.get('sl', 512)),
                                  cfgs, mlp_ratio=mr)
                    rec['params'] = int(count_params(_m))
                    del _m
                except Exception as _e:
                    rec['errors'].append(
                        f'parameter count failed ({type(_e).__name__}: {_e})')
        if rec['params'] and rec['token_budget']:
            rec['params_per_token'] = _finite_int_ratio(rec['params'],
                                                        rec['token_budget'])
        for Ln in out['seq_lens']:
            rec['read_budget'][str(Ln)] = _ledger_read_budget(cfgs, Ln)
        out['arms'][arm] = rec
        if rec['errors']:
            out['warnings'].extend(f'{arm}: {e}' for e in rec['errors'])
    _consumed = {rec.get('token_budget') for rec in out['arms'].values()
                 if rec.get('token_budget')}
    _caches = {rec.get('train_cache_tokens') for rec in out['arms'].values()
               if rec.get('train_cache_tokens')}
    out['token_budget']['arm_consumed_tokens'] = {
        a: rec.get('token_budget') for a, rec in out['arms'].items()}
    out['token_budget']['arm_train_cache_tokens'] = {
        a: rec.get('train_cache_tokens') for a, rec in out['arms'].items()}
    out['token_budget']['shared_by_all_arms'] = bool(
        len(_consumed) == 1 and next(iter(_consumed), 0) > 0)
    out['token_budget']['cache_shared_by_all_arms'] = bool(
        len(_caches) == 1 and next(iter(_caches), 0) > 0)
    if out['token_budget']['shared_by_all_arms'] is False:
        out['problems'].append(
            'arms do not share one consumed token budget '
            f'({sorted(_consumed)}); token counts are a treatment, not a nuisance')
    if out['token_budget']['cache_shared_by_all_arms'] is False:
        out['problems'].append(
            'arms do not share one served train-cache budget '
            f'({sorted(_caches)}); a different corpus/cache is a data treatment, '
            'not a nuisance')
    if out['token_budget']['shared_by_all_arms']:
        out['token_budget']['consumed_tokens'] = next(iter(_consumed))
    if out['token_budget']['cache_shared_by_all_arms']:
        out['token_budget']['train_cache_tokens'] = next(iter(_caches))
    # Design-cost ledger for forward planning.  It is deliberately a raw
    # count/token product, not a measured FLOPs number; at a larger scale the
    # caller needs to see the number of runs and total consumed tokens BEFORE
    # scheduling the family.
    _n_runs = len(arms) * len(manifest.get('seeds') or []) if isinstance(
        manifest.get('seeds'), list) else None
    out['design_cost'] = {
        'n_arms': len(arms),
        'n_seeds': (len(manifest.get('seeds')) if isinstance(manifest.get('seeds'), list) else None),
        'n_runs': _n_runs,
        'consumed_tokens_per_run': (next(iter(_consumed)) if out['token_budget']['shared_by_all_arms'] else None),
        'consumed_tokens_total': ((_n_runs * next(iter(_consumed)))
                                  if (_n_runs and out['token_budget']['shared_by_all_arms']) else None),
        'note': ('parameter count and tokens are normalized per run; this is a '
                 'scheduling aid, not a measured FLOPs or wall-clock budget'),
    }
    # Dedicated-control inventory from the arm names/configs themselves.
    for arm, cfgs in cfgs_by_arm.items():
        _kinds = {str(getattr(c, 'kind', 'full') or 'full') for c in cfgs}
        _dyn = any(bool(getattr(c, 'dynamic', False)) for c in cfgs)
        _modes = {str(getattr(c, 'indexer_mode', 'learned')) for c in cfgs}
        _allb = any(int(getattr(c, 'index_topk', 0) or 0) >= 1000000 for c in cfgs)
        if _allb:
            out['dedicated_controls']['information_rich_allblocks'].append(arm)
        if 'full' in _kinds:
            out['dedicated_controls']['parameter_proxy_arms'].append(arm)
            if out['token_budget']['shared_by_all_arms']:
                out['dedicated_controls']['dense_token_matched_arms'].append(arm)
        if _dyn and _modes == {'random'}:
            out['dedicated_controls']['read_matched_arms'].append(arm)
        if (not _dyn) and (not _allb) and ('csa' in _kinds or 'hca' in _kinds):
            out['dedicated_controls']['fixed_read_matched_arms'].append(arm)
    if out['token_budget']['shared_by_all_arms']:
        out['dedicated_controls']['token_matched_arms'] = list(arms)
    comps = manifest.get('primary_comparisons')
    if not isinstance(comps, list) or not comps:
        out['problems'].append('primary_comparisons must be a non-empty list')
        comps = []
    _ptol = float(param_tolerance)
    _rtol = float(read_tolerance)
    # A mechanism claim is a claim about evidence, not only about arithmetic:
    # only a manifest whose effective evidence level is confirmatory may carry
    # one.  Evaluating the level here (panel_paths=[] keeps this a pure
    # manifest check) prevents an exploratory/draft plan from passing the
    # budget ledger merely because its arms happen to be well matched.
    _declared_level = str(manifest.get('evidence_level') or '').strip().lower()
    _manifest_status = str(manifest.get('status') or '').strip().lower()
    _manifest_frozen = bool(str(manifest.get('frozen_at') or '').strip())
    _manifest_scope = manifest.get('scope') if isinstance(manifest.get('scope'), dict) else {}
    if _declared_level in EVIDENCE_LEVELS:
        if (_declared_level == 'confirmatory'
                and (_manifest_status == 'draft' or not _manifest_frozen)):
            _effective_level = 'draft'
        else:
            _effective_level = _declared_level
    elif _manifest_status == 'draft' or not _manifest_frozen:
        _effective_level = 'draft'
    elif str(_manifest_scope.get('confirmatory_claims') or '').strip():
        _effective_level = 'confirmatory'
    else:
        _effective_level = 'exploratory'
    out['evidence_level'] = _effective_level
    _pre = manifest.get('precondition') if isinstance(manifest.get('precondition'), dict) else {}

    def _calibration_state(comp):
        cal = comp.get('read_budget_calibration')
        if not isinstance(cal, dict) or not cal:
            cal = _pre.get('read_budget_calibration')
        if not isinstance(cal, dict) or not cal:
            return False, None, None
        status = str(cal.get('status') or '').strip().lower()
        ok = status in ('measured', 'passed', 'frozen')
        return ok, status, cal

    for i, comp in enumerate(comps):
        if not isinstance(comp, dict):
            out['problems'].append(f'primary comparison #{i} is not an object')
            continue
        name = str(comp.get('name') or f'#{i}')
        a, b = str(comp.get('a') or ''), str(comp.get('b') or '')
        _cal_ok, _cal_status, _cal_obj = _calibration_state(comp)
        entry = {'name': name, 'a': a, 'b': b, 'endpoint': comp.get('endpoint'),
                 'mechanism_claim_allowed': comp.get('mechanism_claim_allowed'),
                 'token_matched': None, 'param_matched': None,
                 'param_rel_diff': None, 'read_matched': None,
                 'read_ratio_max_min': None,
                 'read_including_window_matched': None,
                 'read_including_window_ratio': None,
                 'compute_6nd_rel_diff': None,
                 'compute_proxy_matched': None,
                 'cfg_diff_keys': None, 'single_factor_cfg': None,
                 'dedicated_control': None, 'dynamic_vs_fixed': False,
                 'read_budget_calibration_required': False,
                 'read_budget_calibration_present': bool(_cal_obj),
                 'read_budget_calibration_ok': bool(_cal_ok),
                 'read_budget_calibration_status': _cal_status,
                 'problems': [], 'warnings': []}
        if a not in arms or b not in arms:
            entry['problems'].append('comparison names arms outside manifest.arms')
            out['comparisons'][name] = entry
            continue
        if a not in cfgs_by_arm or b not in cfgs_by_arm:
            entry['problems'].append('at least one arm has no constructible config')
            out['comparisons'][name] = entry
            continue
        # Token budget: compare the CONSUMED tokens of the two arms (and the
        # served cache they draw from), not the cache size against the product.
        _ta = out['arms'][a].get('token_budget') or 0
        _tb = out['arms'][b].get('token_budget') or 0
        _ca = out['arms'][a].get('train_cache_tokens') or 0
        _cb = out['arms'][b].get('train_cache_tokens') or 0
        entry['token_matched'] = bool(_ta > 0 and _ta == _tb and _ca == _cb)
        entry['consumed_tokens'] = {a: _ta, b: _tb}
        entry['train_cache_tokens'] = {a: _ca, b: _cb}
        pa, pb = out['arms'][a].get('params'), out['arms'][b].get('params')
        if pa and pb:
            rel = _finite_int_ratio(abs(pa - pb), max(pa, pb, 1))
            entry['param_rel_diff'] = rel
            # Compare exactly against the gate tolerance even when the ratio
            # itself underflows/overflows as a float.
            entry['param_matched'] = _int_ratio_within(
                abs(pa - pb), max(pa, pb, 1), _ptol)
        else:
            entry['param_matched'] = None
            entry['problems'].append('parameter totals are unverifiable for this pair')
        # Read budget at the evaluation length (default 4096): the length at
        # which sparse-vs-dense read fractions diverge most.
        _L = 4096 if 4096 in out['seq_lens'] else out['seq_lens'][-1]
        ra = out['arms'][a]['read_budget'].get(str(_L)) or {}
        rb = out['arms'][b]['read_budget'].get(str(_L)) or {}
        # The SELECTED block-token budget is the variable read budget (the
        # manifest's 32/1024 statement).  The local window is a separate,
        # additive term present in the CSA arms; report it but do not let it
        # hide the selected-budget mismatch (or manufacture a match).
        ta = float(ra.get('selected_block_tokens_per_query_mean') or 0.0)
        tb = float(rb.get('selected_block_tokens_per_query_mean') or 0.0)
        if ta > 0 and tb > 0:
            ratio = max(ta, tb) / min(ta, tb)
            entry['read_ratio_max_min'] = float(ratio)
            entry['read_matched'] = bool(abs(ratio - 1.0) <= _rtol)
        else:
            entry['read_matched'] = None
        entry['selected_read_tokens_per_query'] = {a: ta, b: tb}
        _wa = float(ra.get('tokens_read_per_query_mean') or 0.0)
        _wb = float(rb.get('tokens_read_per_query_mean') or 0.0)
        entry['read_tokens_including_local_window'] = {a: _wa, b: _wb}
        if _wa > 0 and _wb > 0:
            _wratio = max(_wa, _wb) / min(_wa, _wb)
            entry['read_including_window_ratio'] = float(_wratio)
            entry['read_including_window_matched'] = bool(
                abs(_wratio - 1.0) <= _rtol)
        entry['local_window_tokens'] = {
            a: [x.get('local_window_tokens') for x in (ra.get('layers') or [])],
            b: [x.get('local_window_tokens') for x in (rb.get('layers') or [])]}
        # 6ND (params x tokens) and the attention read budget are separate
        # axes: two arms can match parameters and still read 32x differently.
        if pa and pb and _ta and _tb:
            # params x consumed tokens is an exact integer product; converting
            # each factor to float first can overflow on a declared scale-plan
            # budget.  Compare the products with exact integers and report the
            # relative difference only when it is representable.
            _fa = int(pa) * int(_ta)
            _fb = int(pb) * int(_tb)
            _frel = _finite_int_ratio(abs(_fa - _fb), max(_fa, _fb, 1))
            entry['compute_6nd_rel_diff'] = _frel
            _fmatched = _int_ratio_within(abs(_fa - _fb),
                                          max(_fa, _fb, 1), _ptol)
            entry['compute_6nd_matched'] = bool(
                _fmatched is True and entry['token_matched'] is True)
        else:
            entry['compute_6nd_matched'] = False
        entry['compute_proxy_matched'] = bool(
            entry['param_matched'] is True and entry['read_matched'] is True
            and entry['token_matched'] is True)
        # Config diff and the dedicated-control family.
        try:
            keys, _per_layer = _design_pair_cfg_keys(a, b, _fac, int(sig.get('L', 6)))
        except Exception as _e:
            keys = _e
        if isinstance(keys, Exception):
            entry['problems'].append(f'cfg diff failed ({type(keys).__name__}: {keys})')
        else:
            keys = set(keys)
            entry['cfg_diff_keys'] = sorted(keys)
            entry['single_factor_cfg'] = bool(cfg_keys_single_factor(keys))
            _ca = cfgs_by_arm[a]
            _cb = cfgs_by_arm[b]
            _da = any(bool(getattr(c, 'dynamic', False)) for c in _ca)
            _db = any(bool(getattr(c, 'dynamic', False)) for c in _cb)
            entry['dynamic_vs_fixed'] = bool(_da != _db)
            _mode_a = {str(getattr(c, 'indexer_mode', 'learned')) for c in _ca}
            _mode_b = {str(getattr(c, 'indexer_mode', 'learned')) for c in _cb}
            _same_mode = _mode_a == _mode_b
            _allblocks = (a in out['dedicated_controls']['information_rich_allblocks']
                          or b in out['dedicated_controls']['information_rich_allblocks'])
            if _allblocks:
                entry['dedicated_control'] = 'information_rich_allblocks_control'
            elif keys <= {'indexer_mode', 'random_salt'}:
                entry['dedicated_control'] = 'selection_control'
                out['dedicated_controls']['selection_control_pairs'].append(name)
            elif keys <= {'boundary_policy'}:
                entry['dedicated_control'] = 'task_policy_control'
                out['dedicated_controls']['task_policy_pairs'].append(name)
            elif keys <= {'use_sink', 'full_sink'}:
                entry['dedicated_control'] = 'sink_control'
                out['dedicated_controls']['sink_control_pairs'].append(name)
            elif (keys <= {'dynamic', 'chunking', 'target_block_tokens'}
                  and _same_mode):
                entry['dedicated_control'] = 'segmentation_control'
                out['dedicated_controls']['segmentation_control_pairs'].append(name)
            elif keys <= {'index_topk'}:
                entry['dedicated_control'] = 'read_dose_control'
            elif 'kind' in keys:
                entry['dedicated_control'] = 'architecture_level_parameter_proxy'
            else:
                entry['dedicated_control'] = 'multi_factor/other'
            if entry['dynamic_vs_fixed']:
                # A dynamic-vs-fixed read budget depends on the realized mean
                # block length.  A confirmatory mechanism claim needs that
                # calibration, not an assumed target.
                entry['read_budget_calibration_required'] = True
                if (entry['mechanism_claim_allowed'] is True
                        and not entry['read_budget_calibration_ok']):
                    entry['problems'].append(
                        'dynamic-vs-fixed mechanism claim requires a MEASURED '
                        'read-budget calibration (realized mean block length -> '
                        'matched fixed top-k/block size); the analytic '
                        'target_block_tokens proxy or a pending status is not enough')
        claim = entry['mechanism_claim_allowed']
        if claim is True:
            if entry['token_matched'] is not True:
                entry['problems'].append('mechanism claim but token budget is not matched')
            if entry['param_matched'] is not True:
                entry['problems'].append('mechanism claim but parameters are not matched (or unverifiable)')
            _read_ok = bool(entry['read_matched'] is True or (
                entry['dynamic_vs_fixed'] and entry['read_budget_calibration_ok']))
            if not _read_ok:
                entry['problems'].append('mechanism claim but read budget is not matched within tolerance')
            if entry['single_factor_cfg'] is not True:
                entry['problems'].append('mechanism claim but the cfg diff is not a declared single-factor family')
            _allowed_families = {
                'selection_control', 'segmentation_control', 'task_policy_control',
                'sink_control', 'architecture_level_parameter_proxy'}
            if entry['dedicated_control'] not in _allowed_families:
                entry['problems'].append(
                    'mechanism claim without an identifiable dedicated control '
                    'family (selection/segmentation/task-policy/sink/parameter-'
                    'proxy); a read-dose/all-block/multi-factor contrast is not '
                    'a mechanism control')
            if out.get('evidence_level') != 'confirmatory':
                entry['problems'].append(
                    f'mechanism claim on a manifest with evidence_level='
                    f'{out.get("evidence_level")!r}; exploratory/draft results '
                    'may not be promoted by a matched budget alone')
        if entry['problems']:
            out['status'] = 'fail'
            out['violations'].extend(f'{name}: {p}' for p in entry['problems'])
        if (entry['read_budget_calibration_required']
                and not entry['read_budget_calibration_ok']):
            entry['warnings'].append(
                'dynamic-vs-fixed read budget is an analytic proxy until a '
                'MEASURED calibration on realized block lengths is recorded; '
                'do not describe it as read-budget matched')
        out['comparisons'][name] = entry
    if out['problems']:
        out['status'] = 'fail'
        out['violations'].extend(out['problems'])
    return out


def dynamic_design_audit(manifest, *, manifest_path=None):
    """Zero-GPU hypothesis/control contract for the DYNAMIC chunking study.

    A dynamic-vs-fixed PPL contrast alone cannot tell us *which* claim is true:
    better boundaries (segmentation quality), better retrieval (selection
    precision at a fixed read budget), or better long-range coverage (less
    dilution at L4096).  This audit refuses a dynamic design that has not
    declared those hypotheses separately and has not scheduled the dedicated
    controls needed to attribute a gain:

      * fixed vs dynamic at the SAME selection mode (random-vs-random isolates
        segmentation from learned selection);
      * a read-budget-matched fixed arm (calibrated top-k), not only vs dense;
      * a parameter-proxy dense arm and a sink-matched dense arm;
      * a selection control inside the dynamic arm;
      * a task-learned-boundary arm, reported separately from the cosine
        heuristic;
      * a HCA comparison explicitly marked as system-level/unmatched-read.

    It also enforces the evidence level: a draft can never be confirmatory;
    promotion requires a disjoint test source, a measured read-budget
    calibration and a pre-test power gate.
    """
    out = {'status': 'ok', 'manifest': _evidence_manifest_path(manifest_path),
           'evidence_level': None, 'hypotheses': {}, 'controls': {},
           'comparisons': {}, 'violations': [], 'warnings': [],
           'power': {}, 'calibration': {},
           'runnable': False, 'promotion_ready': False}
    if not isinstance(manifest, dict):
        out['status'] = 'fail'
        out['violations'].append(
            f'manifest must be an object, got {type(manifest).__name__}')
        return out
    status = str(manifest.get('status') or '').strip().lower()
    frozen_at = manifest.get('frozen_at')
    level = str(manifest.get('evidence_level') or '').strip().lower()
    if not level:
        level = 'draft' if (status == 'draft' or frozen_at in (None, '')) else 'confirmatory'
    out['evidence_level'] = level
    if level == 'confirmatory' and (status == 'draft' or frozen_at in (None, '')):
        out['violations'].append(
            'dynamic design is draft/unfrozen but declares confirmatory; a '
            'draft is not a pre-registration')
    if level not in EVIDENCE_LEVELS:
        out['violations'].append(f'unknown evidence_level {level!r}')
    if status != 'draft' and level != 'confirmatory':
        out['violations'].append(
            f'a non-draft dynamic plan must declare evidence_level=confirmatory '
            f'(got {level!r}); a frozen plan labelled exploratory cannot be used '
            'to promote a claim')
    raw_arms = manifest.get('arms')
    if not isinstance(raw_arms, list) or not raw_arms:
        out['violations'].append('arms must be a non-empty list')
        raw_arms = []
    arms = []
    for i, a in enumerate(raw_arms):
        if not isinstance(a, str) or not a.strip():
            out['violations'].append(
                f'arms[{i}]={a!r} is not a non-empty string; arm names are part '
                'of the frozen design record')
            continue
        arms.append(a.strip())
    if len(set(arms)) != len(arms):
        out['violations'].append('arms contain duplicates')
    arm_set = set(arms)
    raw_seeds = manifest.get('seeds')
    seeds = []
    if not isinstance(raw_seeds, list) or not raw_seeds:
        out['violations'].append('seeds must be a non-empty frozen list')
    else:
        for i, s in enumerate(raw_seeds):
            v = _design_int(s)
            if v is None or v < 0:
                out['violations'].append(
                    f'seeds[{i}]={s!r} is not a non-negative integer seed')
            else:
                seeds.append(v)
        if len(set(seeds)) != len(seeds):
            out['violations'].append('seeds contain duplicates')
    # ---- hypotheses -----------------------------------------------------
    hyps = manifest.get('hypotheses')
    if not isinstance(hyps, dict) or not hyps:
        out['violations'].append(
            'hypotheses must be a non-empty object: name the dynamic-vs-fixed '
            'claims (segmentation quality / retrieval precision / long-range '
            'coverage) before adding arms')
        hyps = {}
    required_hyp = ('H1_segmentation_quality', 'H2_retrieval_precision',
                    'H3_long_range_coverage')
    _all_hyp = list(required_hyp) + [h for h in hyps if h not in required_hyp]
    _hyp_pairs = {}
    for hid in _all_hyp:
        h = hyps.get(hid)
        if not isinstance(h, dict):
            if hid in required_hyp:
                out['violations'].append(
                    f'missing required dynamic hypothesis {hid!r}; a dynamic arm '
                    'must state which mechanism it is meant to test')
            else:
                out['violations'].append(
                    f'hypothesis {hid!r} is not an object; a malformed extra '
                    'claim cannot be silently ignored')
            continue
        _bad = []
        for fld in ('statement', 'mechanism', 'falsifier', 'primary_metric',
                    'metric_policy', 'endpoint'):
            if not str(h.get(fld) or '').strip():
                _bad.append(fld)
        if h.get('predicted_sign') not in (-1, 1):
            _bad.append('predicted_sign must be -1 or +1')
        if h.get('metric_policy') not in METRIC_POLICY:
            _bad.append(f'metric_policy {h.get("metric_policy")!r} is not in METRIC_POLICY')
        if h.get('requires_param_match') is not True:
            _bad.append('requires_param_match must be true for a budget-controlled claim')
        if 'evidence_level' in h:
            _hl = str(h.get('evidence_level') or '').strip().lower()
            if _hl not in EVIDENCE_LEVELS:
                _bad.append(f'evidence_level {h.get("evidence_level")!r} is unknown')
            elif _hl == 'confirmatory' and level != 'confirmatory':
                _bad.append('a hypothesis cannot be confirmatory inside a non-confirmatory manifest')
        if 'mechanism_claim_eligible' in h and not isinstance(
                h.get('mechanism_claim_eligible'), bool):
            _bad.append('mechanism_claim_eligible must be a boolean')
        _pairs = h.get('contrast_arms')
        if _pairs is not None:
            if not isinstance(_pairs, list) or not _pairs:
                _bad.append('contrast_arms must be a non-empty list of [a, b] pairs')
            else:
                _clean = []
                for _p in _pairs:
                    if (not isinstance(_p, (list, tuple)) or len(_p) != 2
                            or not all(isinstance(x, str) and x.strip() for x in _p)):
                        _bad.append('contrast_arms entries must be two non-empty arm names')
                        continue
                    _pa, _pb = _p[0].strip(), _p[1].strip()
                    if _pa == _pb:
                        _bad.append(f'contrast_arms pair {_p!r} compares an arm with itself')
                    elif _pa not in arm_set or _pb not in arm_set:
                        _bad.append(
                            f'contrast_arms pair {_p!r} names an arm outside manifest.arms')
                    else:
                        _clean.append((_pa, _pb))
                _hyp_pairs[hid] = _clean
        if _bad:
            out['violations'].append(f'{hid}: {", ".join(_bad)}')
        out['hypotheses'][hid] = {k: h.get(k) for k in
                                  ('statement', 'mechanism', 'predicted_sign',
                                   'primary_metric', 'metric_policy', 'endpoint',
                                   'requires_param_match', 'falsifier',
                                   'contrast_arms', 'mechanism_claim_eligible',
                                   'evidence_level')}
    # ---- dedicated control inventory -----------------------------------
    controls = manifest.get('budget_controls')
    if not isinstance(controls, dict) or not controls:
        out['violations'].append(
            'budget_controls must declare token/parameter/read/selection/'
            'segmentation controls; "vs dense" alone cannot attribute a gain')
        controls = {}
    out['controls'] = controls
    _need = {
        'token_budget': ('token_budget',),
        'token_matched_arm': ('token_matched_arm', 'token_matched'),
        'dense_token_matched': ('dense_token_matched', 'dense_token_matched_arm'),
        'parameter_proxy': ('parameter_proxy',),
        'read_budget': ('read_budget',),
        'selection_control': ('selection_control',),
        'segmentation_control': ('segmentation_control',),
        'hca_control': ('hca_control',),
    }
    for label, keys in _need.items():
        if not any(isinstance(controls.get(k), dict) and controls.get(k) for k in keys):
            out['violations'].append(
                f'budget_controls.{label} is missing; every mechanism claim '
                'needs a declared dedicated control, not only a dense baseline')
    # Declared control roles must name real arms.  A prose control block that
    # points at a deleted/renamed arm is exactly the drift this audit exists to
    # catch before GPU spend.
    def _check_named_arm(value, where):
        if isinstance(value, str):
            if value not in arm_set:
                out['violations'].append(f'{where} names arm {value!r} outside manifest.arms')
        elif isinstance(value, (list, tuple)):
            for v in value:
                _check_named_arm(v, where)
        elif isinstance(value, dict):
            for k, v in value.items():
                if k in ('arm', 'arms', 'pair', 'pairs') or k.endswith('_arm'):
                    _check_named_arm(v, f'{where}.{k}')

    _check_named_arm(controls.get('token_matched_arm'), 'budget_controls.token_matched_arm')
    _check_named_arm(controls.get('dense_token_matched'), 'budget_controls.dense_token_matched')
    _pp = controls.get('parameter_proxy') if isinstance(controls.get('parameter_proxy'), dict) else {}
    _check_named_arm(_pp.get('arms'), 'budget_controls.parameter_proxy.arms')
    _sel = controls.get('selection_control') if isinstance(controls.get('selection_control'), dict) else {}
    _check_named_arm(_sel.get('pairs'), 'budget_controls.selection_control.pairs')
    _seg = controls.get('segmentation_control') if isinstance(controls.get('segmentation_control'), dict) else {}
    _check_named_arm(_seg.get('pair'), 'budget_controls.segmentation_control.pair')
    _hca = controls.get('hca_control') if isinstance(controls.get('hca_control'), dict) else {}
    _check_named_arm(_hca.get('pair'), 'budget_controls.hca_control.pair')
    # The parameter-proxy mr pins are part of the budget control; they must
    # match the frozen per-arm pins when both are present.
    _per_arm = ((manifest.get('binding') or {}).get('per_arm_recipe_signature') or {}) \
        if isinstance(manifest.get('binding'), dict) else {}
    _mr = _pp.get('mr') if isinstance(_pp.get('mr'), dict) else {}
    for _arm, _val in _mr.items():
        _pin = (_per_arm.get(_arm) or {}).get('mr') if isinstance(_per_arm.get(_arm), dict) else None
        if _pin is not None and not _design_close(_val, _pin, tol=1e-9):
            out['violations'].append(
                f'budget_controls.parameter_proxy.mr[{_arm!r}]={_val!r} != '
                f'binding.per_arm_recipe_signature.mr={_pin!r}')
    # Required arm roles (names are part of the design record).
    _required_roles = {
        'fixed reference (learned selection)': 'csa_fixed_rope',
        'dynamic reference (learned selection)': 'csa_dynamic_rope',
        'fixed random-selection control': 'csa_fixed_rope_randidx_seeded',
        'dynamic random-selection control': 'csa_dynamic_rope_randidx_seeded',
        'parameter-proxy dense reference': 'full_rope',
        'sink-matched dense reference': 'full_rope_sink',
        'read-rich all-block control': 'csa_fixed_rope_allblocks',
        'read-budget-matched fixed candidate': 'csa_fixed_rope_topk32',
        'task-learned boundary arm': 'csa_dynamic_rope_task',
        'HCA dynamic arm': 'hybrid_dynamic_rope',
    }
    _topk_prefix = 'csa_fixed_rope_topk'
    _topk_ks = []
    for a in arm_set:
        if a.startswith(_topk_prefix) and a[len(_topk_prefix):].isdigit():
            _topk_ks.append(int(a[len(_topk_prefix):]))
    _has_topk_arm = bool(_topk_ks)
    if any(k < 1 for k in _topk_ks):
        out['violations'].append(
            f'{_topk_prefix}{{k}} arms must have k >= 1; got {sorted(_topk_ks)}')
    for role, arm in _required_roles.items():
        if role == 'read-budget-matched fixed candidate':
            # Any concrete calibrated top-k is acceptable; freezing 32 before
            # measuring the realized dynamic block length is exactly what the
            # calibration precondition forbids.
            if not _has_topk_arm:
                out['violations'].append(
                    f'missing {role} arm {_topk_prefix}{{k}}; a fixed arm whose '
                    'top-k is calibrated from the measured dynamic block length '
                    'is required, not only a dense baseline')
            continue
        if arm not in arm_set:
            out['violations'].append(
                f'missing {role} arm {arm!r}; the dynamic study needs this arm '
                'to separate the claimed mechanism from budget/capacity effects')
    # ---- comparisons ----------------------------------------------------
    comps = manifest.get('primary_comparisons')
    if not isinstance(comps, list) or not comps:
        out['violations'].append('primary_comparisons must be a non-empty list')
        comps = []
    seen_pairs = set()
    seen_names = set()
    _tested_by_hyp = {}
    # The hypotheses that declare `mechanism_claim_eligible=true` are the ones
    # that NEED to be single-factor.  Verify that at the definition site: the
    # budget ledger checks what the arms share, but only the cfg diff can show
    # whether a "H1 segmentation-only" or "H2 selection-only" contrast also
    # moved a second knob.  The v7 factory owns the rope arms used by this
    # study; a missing factory is reported as unverifiable rather than silently
    # passing.
    _cfg_factory = None
    try:
        import explib.rope as _v7
        _cfg_factory = _v7.make_layer_cfgs_v7
    except Exception as _e:  # pragma: no cover - environment/import defect
        out['warnings'].append(
            f'dynamic single-factor check cannot build arms '
            f'({type(_e).__name__}: {_e}); mechanism-eligible contrasts are '
            'unverifiable')
    for i, comp in enumerate(comps):
        if not isinstance(comp, dict):
            out['violations'].append(f'primary comparison #{i} is not an object')
            continue
        name = str(comp.get('name') or f'#{i}')
        if name in seen_names:
            out['violations'].append(
                f'duplicate primary comparison name {name!r}; the Holm family '
                'cannot contain two indistinguishable slots')
        seen_names.add(name)
        a, b = comp.get('a'), comp.get('b')
        # Arm names are identities, not arbitrary JSON values.  Testing an
        # unhashable value (a list/dict) against the arm set used to raise
        # ``TypeError: unhashable type`` from the middle of the audit; name the
        # malformed entry and keep auditing the rest of the family instead.
        _a_ok = isinstance(a, str) and bool(a.strip())
        _b_ok = isinstance(b, str) and bool(b.strip())
        _lvl_raw = comp.get('evidence_level')
        _lvl_explicit = isinstance(_lvl_raw, str) and bool(_lvl_raw.strip())
        _lvl = str(_lvl_raw or level).strip().lower()
        _sign_raw = comp.get('expected_sign')
        _sign = _design_int(_sign_raw)
        ent = {'a': a, 'b': b, 'endpoint': comp.get('endpoint'),
               'hypothesis_id': comp.get('hypothesis_id'),
               'evidence_level': _lvl,
               'evidence_level_declared': bool(_lvl_explicit),
               'expected_sign': _sign,
               'mechanism_claim_allowed': comp.get('mechanism_claim_allowed'),
               'problems': []}
        if not (_a_ok and _b_ok):
            if not _a_ok:
                ent['problems'].append(
                    f'a={a!r} is not a non-empty arm name')
            if not _b_ok:
                ent['problems'].append(
                    f'b={b!r} is not a non-empty arm name')
        elif a not in arm_set or b not in arm_set:
            ent['problems'].append('names an arm outside manifest.arms')
        if not _lvl_explicit:
            ent['problems'].append(
                'no explicit evidence_level; every comparison added to a plan '
                'must state whether it is draft/exploratory/confirmatory before '
                'it runs')
        elif _lvl not in EVIDENCE_LEVELS:
            ent['problems'].append(f'unknown evidence_level {_lvl!r}')
        if _lvl == 'confirmatory' and level != 'confirmatory':
            ent['problems'].append(
                'comparison is confirmatory but the manifest is not; a draft '
                'cannot promote a single comparison by label')
        if level == 'confirmatory' and _lvl != 'confirmatory':
            ent['problems'].append(
                f'a confirmatory dynamic manifest still labels this comparison '
                f'{_lvl!r}; freeze must promote every family slot explicitly '
                '(no inherited draft label)')
        if _sign not in (-1, 1):
            ent['problems'].append(
                f'expected_sign={_sign_raw!r} must be -1 or +1; a non-directional '
                'primary contrast cannot be tested with a one-sided hypothesis')
        _hid = comp.get('hypothesis_id')
        _h = hyps.get(_hid) if isinstance(_hid, str) else None
        if not isinstance(_h, dict) or _hid not in out['hypotheses']:
            ent['problems'].append(
                'no declared hypothesis_id; every new comparison must map to a '
                'pre-registered claim (H1/H2/H3) or be labelled descriptive')
            _h = None
        if not str(comp.get('endpoint') or '').strip():
            ent['problems'].append('no endpoint')
        elif _h is not None and str(_h.get('endpoint') or '').strip():
            if str(comp.get('endpoint')).strip() != str(_h.get('endpoint')).strip():
                ent['problems'].append(
                    f'endpoint {comp.get("endpoint")!r} does not equal the '
                    f'hypothesis endpoint {_h.get("endpoint")!r}; selecting a '
                    'different endpoint after seeing data is not confirmation')
        if _h is not None:
            _pred = _design_int(_h.get('predicted_sign'))
            _pairs = _hyp_pairs.get(_hid) or []
            _orient = None
            if _pairs:
                for _ca, _cb in _pairs:
                    if (a, b) == (_ca, _cb):
                        _orient = 1
                        break
                    if (a, b) == (_cb, _ca):
                        _orient = -1
                        break
                if _orient is None:
                    ent['problems'].append(
                        f'comparison arms ({a!r}, {b!r}) are not a declared '
                        f'contrast_arms pair for {_hid}; a hypothesis must name '
                        'the arm pair it is allowed to test')
                elif _sign in (-1, 1) and _sign != _pred * _orient:
                    ent['problems'].append(
                        f'expected_sign={_sign} is inconsistent with the '
                        f'hypothesis predicted_sign={_pred} for orientation '
                        f'{_orient:+d}; a reversed contrast needs the reversed sign')
            else:
                if _sign in (-1, 1) and _sign not in (_pred, -_pred):
                    ent['problems'].append(
                        f'expected_sign={_sign} is not +/- the hypothesis '
                        f'predicted_sign={_pred}')
                out['warnings'].append(
                    f'{name}: hypothesis {_hid} has no contrast_arms; the sign '
                    'orientation is not machine-pinned')
            if _hid not in _tested_by_hyp:
                _tested_by_hyp[_hid] = 0
            _tested_by_hyp[_hid] += 1
        # A mechanism claim must not ride on an all-block/dense arm only, and
        # the hypothesis must be explicitly eligible for a mechanism claim.
        if comp.get('mechanism_claim_allowed') is True:
            if str(a).endswith('allblocks') or str(b).endswith('allblocks'):
                ent['problems'].append(
                    'all-block arm is not read-budget matched; it cannot serve '
                    'as the mechanism control')
            if 'hybrid' in str(a) or 'hybrid' in str(b) or str(a).startswith('hca') or str(b).startswith('hca'):
                ent['problems'].append(
                    'an HCA/hybrid arm is a system-level hierarchy readout, not '
                    'a boundary/selection mechanism control')
            if _h is not None and _h.get('mechanism_claim_eligible') is not True:
                ent['problems'].append(
                    f'hypothesis {_hid} is not declared mechanism_claim_eligible; '
                    'a system-level (H3/H5) contrast cannot be quoted as mechanism evidence')
            if _lvl == 'draft':
                ent['problems'].append(
                    'draft cannot carry an enabled mechanism claim; mark it as '
                    'a design target and clear it after calibration/freeze')
        # A mechanism-eligible hypothesis must be tested by a contrast that
        # isolates ONE declared cfg family.  Enumerating a multi-family bundle
        # somewhere else in the design does not make this pair single-factor.
        _needs_sf = bool(isinstance(_h, dict)
                         and _h.get('mechanism_claim_eligible') is True)
        ent['single_factor_required'] = _needs_sf
        ent['cfg_diff_keys'] = None
        ent['single_factor_cfg'] = None
        if _cfg_factory is not None and _a_ok and _b_ok and a in arm_set and b in arm_set:
            _keys, _per_layer = _design_pair_cfg_keys(a, b, _cfg_factory, 6)
            if isinstance(_keys, Exception):
                ent['cfg_diff_error'] = f'{type(_keys).__name__}: {_keys}'
                if _needs_sf:
                    ent['problems'].append(
                        f'hypothesis {_hid} is mechanism_claim_eligible but the '
                        f'contrast arms cannot be constructed to verify a '
                        f'single-factor diff ({ent["cfg_diff_error"]})')
            else:
                _keys = set(_keys)
                ent['cfg_diff_keys'] = sorted(_keys)
                ent['single_factor_cfg'] = bool(cfg_keys_single_factor(_keys))
                if _needs_sf and ent['single_factor_cfg'] is not True:
                    ent['problems'].append(
                        f'hypothesis {_hid} is mechanism_claim_eligible but this '
                        f'contrast moves cfg keys {sorted(_keys)}, which are not '
                        'one declared single-factor family; a mechanism claim '
                        'needs a single-factor arm pair')
                if (comp.get('mechanism_claim_allowed') is True
                        and ent['single_factor_cfg'] is not True):
                    ent['problems'].append(
                        'mechanism claim on a non-single-factor cfg diff '
                        f'{sorted(_keys)}')
        elif _needs_sf and _cfg_factory is None:
            ent['problems'].append(
                f'hypothesis {_hid} is mechanism_claim_eligible but the arm '
                'factory is unavailable; the single-factor property is '
                'unverifiable')
        if ent['problems']:
            out['violations'].extend(f'{name}: {p}' for p in ent['problems'])
        if _a_ok and _b_ok:
            seen_pairs.add(frozenset((a, b)))
        out['comparisons'][name] = ent
    for _hid in required_hyp:
        if not _tested_by_hyp.get(_hid):
            out['violations'].append(
                f'hypothesis {_hid} has no primary comparison; a declared claim '
                'without a scheduled contrast is not hypothesis-driven design')
    # Required paired contrasts: segmentation-only and selection-only.
    required_pairs = {
        'segmentation-only (both random selection)':
            frozenset(('csa_fixed_rope_randidx_seeded', 'csa_dynamic_rope_randidx_seeded')),
        'selection-only inside dynamic blocking':
            frozenset(('csa_dynamic_rope', 'csa_dynamic_rope_randidx_seeded')),
        'parameter proxy vs dynamic':
            frozenset(('full_rope', 'csa_dynamic_rope')),
    }
    for label, pair in required_pairs.items():
        if pair not in seen_pairs:
            out['violations'].append(
                f'missing required contrast {label} ({sorted(pair)}); without '
                'it the dynamic gain cannot be attributed to the declared mechanism')
    # The long-range-coverage claim must be tested on PPL against a CONCRETE
    # read-matched fixed arm, not only against the dense reference.  A
    # top-k candidate arm is acceptable while calibrating; the concrete k is
    # checked by `dynamic_read_budget_calibration` before freeze.
    _read_matched_ppl = False
    for _c in comps:
        if not isinstance(_c, dict):
            continue
        _a, _b = _c.get('a'), _c.get('b')
        _arms = {str(_a), str(_b)}
        if (str(_c.get('endpoint') or '').startswith('by_len.ppl')
                and 'csa_dynamic_rope' in _arms
                and any(x.startswith('csa_fixed_rope_topk') for x in _arms)):
            _read_matched_ppl = True
            break
    if not _read_matched_ppl:
        out['violations'].append(
            'missing a dynamic-vs-read-matched-fixed by_len.ppl contrast; the '
            'long-range-coverage claim cannot be attributed to the boundary '
            'mechanism by comparing only against a dense/fixed reference')
    # ---- statistics / power -------------------------------------------
    stat = manifest.get('statistics') if isinstance(manifest.get('statistics'), dict) else {}
    _power = planned_power_audit(manifest, require_projection=True)
    out['power'] = _power
    out['violations'].extend(_power.get('violations') or [])
    out['warnings'].extend(_power.get('warnings') or [])
    out['min_attainable_p_holm'] = _power.get('min_attainable_p_holm')
    mfam = len(comps)
    nseeds = _power.get('n_seeds') or 0
    if stat.get('require_power_gate') is not True:
        out['violations'].append(
            'statistics.require_power_gate must be true; a dynamic panel cannot '
            'be confirmed without a pre-test power simulation')
    mp = _power.get('min_power_for_test')
    if mp is None:
        out['violations'].append(
            f'statistics.min_power_for_test={stat.get("min_power_for_test")!r} is '
            'not a finite power in [0,1]')
    elif mp < 0.8:
        out['violations'].append(
            f'statistics.min_power_for_test={mp} < 0.8; an underpowered dynamic '
            'contrast cannot confirm a mechanism claim')
    if stat.get('power_simulation_required') is not True:
        out['violations'].append(
            'statistics.power_simulation_required must be true (simulate the '
            'per-seed delta vector before touching the disjoint test source)')
    if _power.get('projected_power') is not None:
        out['projected_power'] = _power.get('projected_power')
    # ---- preconditions / promotion ------------------------------------
    pre = manifest.get('precondition') if isinstance(manifest.get('precondition'), dict) else {}
    out['runnable'] = bool(pre.get('runnable') is True)
    if status == 'draft' and out['runnable']:
        out['violations'].append(
            'status=draft is inconsistent with precondition.runnable=true')
    _cal_audit = dynamic_read_budget_calibration(manifest)
    out['calibration'] = _cal_audit
    out['violations'].extend(_cal_audit.get('problems') or [])
    out['warnings'].extend(_cal_audit.get('warnings') or [])
    _cal_status = _cal_audit.get('status')
    _pwr = pre.get('power_gate') if isinstance(pre.get('power_gate'), dict) else {}
    _disjoint = bool(str(pre.get('disjoint_test_source') or pre.get('test_source') or '').strip())
    if not _disjoint:
        out['violations'].append(
            'precondition must name a disjoint test source; reusing the v5 '
            'test rows would make a second study non-independent')
    if not str(pre.get('must_not_reuse') or '').strip():
        out['violations'].append(
            'precondition.must_not_reuse must name the reserved v5 rows/slice')
    if level == 'confirmatory':
        _scope = manifest.get('scope') if isinstance(manifest.get('scope'), dict) else {}
        for _fld in ('confirmatory_claims', 'not_covered', 'mechanism_claim_policy'):
            _val = _scope.get(_fld)
            if not isinstance(_val, str) or not _val.strip():
                out['violations'].append(
                    f'scope.{_fld} must be non-empty text for a confirmatory '
                    'dynamic plan; the claim boundary is part of the freeze')
    # Freezing is not a label change: a non-draft, runnable design must carry
    # the measured read-budget calibration and the passed power gate.  A
    # frozen plan that is not runnable is either a stale draft or a plan that
    # cannot actually be executed; neither may enter confirmation.
    if status != 'draft':
        if not out['runnable']:
            out['violations'].append(
                'a non-draft dynamic plan must be runnable (precondition.runnable=true)')
        if _cal_status != 'ok':
            out['violations'].append(
                'a non-draft dynamic plan must carry a MEASURED, formula-consistent '
                f'read-budget calibration with its concrete calibrated arm (got '
                f'{_cal_status!r}); topk32 is only a candidate')
        if _pwr.get('status') != 'passed':
            out['violations'].append(
                'a non-draft dynamic plan must pass the pre-test power gate '
                f'(got {_pwr.get("status")!r}); freezing before the gate is a '
                'design defect, not a scheduling choice')
    _projected = _power.get('projected_power')
    _power_ok = bool(
        _pwr.get('status') == 'passed'
        and mp is not None
        and _projected is not None
        and float(_projected) + 1e-12 >= float(mp))
    out['promotion_ready'] = bool(
        _cal_status == 'ok'
        and _power_ok
        and out['runnable']
        and not out['violations'])
    if status == 'draft' and not out['promotion_ready']:
        out['warnings'].append(
            'DRAFT ONLY: this design is not runnable and cannot produce a '
            'confirmatory claim. Promote it only after (1) a disjoint test '
            'source is reserved, (2) the realized dynamic block length is '
            'measured and the matched fixed top-k is trained, and (3) the '
            'pre-test power gate passes on the frozen delta vector.')
    if out['violations']:
        out['status'] = 'fail'
    return out


HYPOTHESIS_DESIGN = {
    'dynamic_blocking': {
        'statement': 'Content-adaptive dynamic blocking lowers validation PPL versus fixed blocking at the same parameter count, data order and training length.',
        'mechanism': 'Learned cosine boundaries place cuts at low-similarity token transitions, so fewer cross-boundary tokens are compressed together.',
        'predicted_sign': -1,
        'primary_metric': 'ppl',
        'metric_policy': 'lm_perplexity',
        'requires_param_match': True,
        'falsifier': 'No family-Holm-corrected PPL reduction, a reduction within seed noise, or a reduction that reverses when fixed/dynamic blocking is swapped.',
    },
    'learned_selection': {
        'statement': 'Learned top-k block selection lowers validation PPL versus random selection with the same number of attended blocks.',
        'mechanism': 'Query-conditioned indexer scores retrieve blocks that carry token-relevant content instead of an arbitrary subset.',
        'predicted_sign': -1,
        'primary_metric': 'ppl',
        'metric_policy': 'lm_perplexity',
        'requires_param_match': False,
        'falsifier': 'Random selection matches or beats learned selection after Holm correction, or the gap is within seed noise.',
    },
    'content_pooling': {
        'statement': 'Pooling block content (rather than zero content) lowers validation PPL at the same block geometry.',
        'mechanism': 'Compressed content retains token identity so the indexer and attention see informative block keys/values.',
        'predicted_sign': -1,
        'primary_metric': 'ppl',
        'metric_policy': 'lm_perplexity',
        'requires_param_match': False,
        'falsifier': 'Zeroing block content does not increase PPL beyond seed noise.',
    },
    'content_alignment': {
        'statement': 'Block-aligned pooled content lowers validation PPL versus the same pooled content shifted one block back, at identical shape, norm, parameter count and read budget.',
        'mechanism': 'A compressed block key/value is informative only when it summarises the tokens that block actually contains; content from another block is a matched-magnitude placebo.',
        'predicted_sign': -1,
        'primary_metric': 'ppl',
        'metric_policy': 'lm_perplexity',
        'requires_param_match': False,
        'falsifier': 'The shifted-content control matches or beats aligned content after Holm correction, or the difference is within the pre-registered SESOI; then the zero-content contrast alone cannot support a content-information claim.',
    },
    'attention_sink': {
        'statement': 'An attention sink lowers validation PPL by giving off-context probability mass a stable destination.',
        'mechanism': 'The sink absorbs attention that would otherwise be forced onto unrelated blocks, reducing normalization pressure.',
        'predicted_sign': -1,
        'primary_metric': 'ppl',
        'metric_policy': 'lm_perplexity',
        'requires_param_match': False,
        'falsifier': 'Removing the sink does not increase PPL beyond seed noise, or improves it.',
    },
    'attention_sink_learning': {
        'statement': 'A learned attention sink lowers validation PPL versus an identical-shape frozen random sink at the same parameter count, read budget and initialization stream.',
        'mechanism': 'The learned sink adapts its per-head off-context logit to the training distribution; a frozen random sink of the same capacity cannot adapt, so a gain isolates the learned component from mere extra parameters.',
        'predicted_sign': -1,
        'primary_metric': 'ppl',
        'metric_policy': 'lm_perplexity',
        'requires_param_match': True,
        'falsifier': 'The frozen random sink matches or beats the learned sink after Holm correction, or the difference is within the pre-registered SESOI; then the sink on/off contrast alone can only support a capacity/regularisation claim, not a learned-sink mechanism.',
    },
    'compression_regularizer': {
        'statement': 'The compression regularizer lowers validation PPL at the same architecture and optimization setting.',
        'mechanism': 'The loss term discourages degenerate block boundaries and keeps compression from collapsing to fixed chunks.',
        'predicted_sign': -1,
        'primary_metric': 'ppl',
        'metric_policy': 'lm_perplexity',
        'requires_param_match': True,
        'falsifier': 'Zeroing the regularizer does not increase PPL beyond seed noise, or improves it.',
    },
    'task_learned_boundary': {
        'statement': 'Training the sampled cut policy with the LM loss (score-function term) lowers validation PPL versus the hard-threshold + comp_reg-only gate.',
        'mechanism': 'The LM objective can trade compression against language-modelling loss directly instead of optimizing only a compression-rate surrogate.',
        'predicted_sign': -1,
        'primary_metric': 'ppl',
        'metric_policy': 'lm_perplexity',
        'requires_param_match': False,
        'falsifier': 'The task-trained policy matches or loses to the threshold gate after Holm correction, or its variance makes the panel unusable.',
    },
    'local_fusion': {
        'statement': 'The learned fusion convolution lowers validation PPL versus the fixed identity fusion rule.',
        'mechanism': 'The fusion convolution reshapes the cosine segmentation signal; it does NOT gate the local window, so this is a boundary-signal smoothing contrast, not a learned sparse/local trade-off.',
        'predicted_sign': -1,
        'primary_metric': 'ppl',
        'metric_policy': 'lm_perplexity',
        'requires_param_match': False,
        'falsifier': 'The learned fusion convolution matches or loses to the identity rule after Holm correction.',
    },
    'hybrid_hierarchy': {
        'statement': 'The CSA+HCA hybrid lowers validation PPL versus CSA-only at matched parameters and training length.',
        'mechanism': 'Alternating local HCA layers and sparse CSA layers cover complementary short- and long-range structure.',
        'predicted_sign': -1,
        'primary_metric': 'ppl',
        'metric_policy': 'lm_perplexity',
        'requires_param_match': True,
        'falsifier': 'The hybrid matches or loses to CSA-only after Holm correction.',
    },
}

INTERACTION_HYPOTHESIS_DESIGN = {
    'dynamic_blocking_x_learned_selection': {
        'statement': ('The log-PPL benefit of learned top-k selection is smaller '
                      'when dynamic blocking is active than when blocking is '
                      'fixed (positive sub-additive interaction).'),
        'mechanism': ('Adaptive boundaries and learned selection both determine '
                      'which blocks are informative; improving one reduces the '
                      'marginal value of the other.'),
        'predicted_sign': 1,
        'primary_metric': 'ppl',
        'metric_policy': 'lm_perplexity',
        'requires_param_match': True,
        'falsifier': ('The interaction is not distinguishable from zero after the '
                      'declared interaction-family Holm correction, its sign is '
                      'negative (synergy), or the four cells fail the capacity/'
                      'recipe controls.'),
    },
    'dynamic_blocking_x_content_pooling': {
        'statement': ('The log-PPL benefit of pooled block content is smaller '
                      'under dynamic blocking than under fixed blocking '
                      '(positive sub-additive interaction).'),
        'mechanism': ('Adaptive cuts make blocks more internally homogeneous, '
                      'so the information lost by collapsing a block into one '
                      'summary is smaller than under fixed blocks.'),
        'predicted_sign': 1,
        'primary_metric': 'ppl',
        'metric_policy': 'lm_perplexity',
        'requires_param_match': True,
        'falsifier': ('The interaction is not distinguishable from zero after the '
                      'declared interaction-family Holm correction, its sign is '
                      'negative (synergy), or the four cells fail the capacity/'
                      'recipe controls.'),
    },
    'dynamic_blocking_x_content_alignment': {
        'statement': ('The log-PPL benefit of block-aligned over shifted content '
                      'is smaller under dynamic blocking than under fixed '
                      'blocking (positive sub-additive interaction).'),
        'mechanism': ('Adaptive cuts make a block more internally homogeneous, '
                      'so replacing its content with the previous block\'s '
                      'content destroys less of the block-specific signal than '
                      'it does under fixed blocks.'),
        'predicted_sign': 1,
        'primary_metric': 'ppl',
        'metric_policy': 'lm_perplexity',
        'requires_param_match': True,
        'falsifier': ('The interaction is not distinguishable from zero after the '
                      'declared interaction-family Holm correction, its sign is '
                      'negative (synergy), or the four cells fail the capacity/'
                      'recipe controls.'),
    },
    'dynamic_blocking_x_attention_sink': {
        'statement': ('The log-PPL benefit of the attention sink is smaller under '
                      'dynamic blocking than under fixed blocking (positive '
                      'sub-additive interaction).'),
        'mechanism': ('A sink and adaptive block reads both provide an outlet '
                      'for off-context probability mass; once one is present the '
                      'other has less unattributed mass to absorb.'),
        'predicted_sign': 1,
        'primary_metric': 'ppl',
        'metric_policy': 'lm_perplexity',
        'requires_param_match': True,
        'falsifier': ('The interaction is not distinguishable from zero after the '
                      'declared interaction-family Holm correction, its sign is '
                      'negative (synergy), or the four cells fail the capacity/'
                      'recipe controls.'),
    },
    'dynamic_blocking_x_attention_sink_learning': {
        'statement': ('The log-PPL benefit of the learned sink over an '
                      'identical-capacity frozen random sink is smaller under '
                      'dynamic blocking than under fixed blocking (positive '
                      'sub-additive interaction).'),
        'mechanism': ('Adaptive boundaries already absorb part of the '
                      'off-context mass that the learned sink would otherwise '
                      'redirect; the learned component therefore adds less.'),
        'predicted_sign': 1,
        'primary_metric': 'ppl',
        'metric_policy': 'lm_perplexity',
        'requires_param_match': True,
        'falsifier': ('The interaction is not distinguishable from zero after the '
                      'declared interaction-family Holm correction, its sign is '
                      'negative (synergy), or the four cells fail the capacity/'
                      'recipe controls; a null interaction does not prove '
                      'additivity without the declared SESOI/TOST decision.'),
    },
}

BASELINE_HYPOTHESIS_DESIGN = {
    'compression_vs_matched_dense': {
        'statement': 'Dynamic CSA lowers validation PPL versus parameter-matched dense baselines under identical data, optimizer, schedule and training length.',
        'mechanism': 'Sparse compressed reads can preserve long-range access while the parameter-matched MLP keeps capacity equal.',
        'predicted_sign': -1,
        'primary_metric': 'ppl',
        'metric_policy': 'lm_perplexity',
        'requires_param_match': True,
        'falsifier': 'A matched dense baseline matches or beats CSA after Holm correction, or the parameter matching fails.',
    },
}

# Equal-search tuning is a prerequisite, not a result.  The tuned comparison
# itself needs its own pre-registered directional hypothesis: a method arm that
# beats an *untuned* baseline has shown only that the baseline was neglected.
# This family is audited alongside the component/baseline/floor families and is
# bound to the protocol's DECLARED contrasts by `baseline_tuning_plan`.
TUNED_BASELINE_HYPOTHESIS_DESIGN = {
    'method_beats_tuned_baseline': {
        'statement': ('After the same declared hyper-parameter search on the same '
                      'selection rows, every declared method arm beats each '
                      'parameter-matched baseline in its tuning group on the '
                      'disjoint held-out validation rows.'),
        'mechanism': ('The claimed attention/compression mechanism lowers held-out '
                      'token NLL at equal capacity, data order and training budget; '
                      'equal-search tuning removes the "tuned method vs untuned '
                      'baseline" explanation.'),
        'predicted_sign': -1,
        'primary_metric': 'ppl',
        'metric_policy': 'lm_perplexity',
        'requires_param_match': True,
        'falsifier': ('Any declared method-vs-baseline contrast fails to beat that '
                      'baseline after Holm correction at the declared final seed '
                      'floor, or the selected configuration does not transfer to '
                      'the final training budget/rows.'),
    },
}

# The first, weakest gate: every declared method arm must beat the tuned
# count-model floor before any mechanism claim is interpretable.  This is a
# separate pre-registered hypothesis because a comparison can satisfy the
# matched-dense gate while failing the trivial floor (or vice versa), and the
# direction convention differs: delta = method - floor, so predicted_sign=-1
# means the method has the lower PPL.  The floor contrast is Holm-corrected
# across declared method arms by `methodology_audit`.
FLOOR_HYPOTHESIS_DESIGN = {
    'method_beats_tuned_count_floor': {
        'statement': ('Every declared method arm has a lower token-pooled validation '
                      'PPL/NLL than the tuned unigram/bigram count floor on the same '
                      'held-out rows and token denominator.'),
        'mechanism': ('A neural language model can exploit multi-token context and '
                      'learned representations that a context-free/bigram count model '
                      'cannot; without this basic gain the mechanism claim has no '
                      'demonstrated language-modelling advantage at all.'),
        'predicted_sign': -1,
        'primary_metric': 'ppl',
        'metric_policy': 'lm_perplexity',
        'requires_param_match': False,
        'falsifier': ('Any declared method arm has a mean PPL at or above the tuned '
                      'floor, fails the Holm-corrected seed-level contrast, or is not '
                      'comparable on validation rows/token denominators.'),
    },
}

# Open-block panel hypotheses (driver: python -m explib.open_block).  These are the
# declared primary contrasts; the remaining maximum-block x reading-rule cells
# are descriptive.  Delta convention is a - b on token NLL, so predicted_sign
# -1 means the first arm is expected to have lower NLL.
OPEN_BLOCK_HYPOTHESIS_DESIGN = {
    'summary - closed @128': {
        'statement': 'Reading the still-open block as a causal prefix summary lowers held-out token NLL versus ignoring the open block, at a fixed top-k budget.',
        'mechanism': 'The open block contains recent content that is otherwise invisible until the block closes; a pooled summary makes part of it addressable without spending a full block slot.',
        'predicted_sign': -1,
        'primary_metric': 'ppl',
        'metric_policy': 'lm_perplexity',
        'requires_param_match': False,
        'falsifier': 'The summary arm is not lower-NLL than closed after Holm correction, or the gap is within seed noise.',
    },
    'singletons - summary @128': {
        'statement': 'Representing the open prefix as length-1 singleton blocks preserves token identity better than one pooled summary and lowers held-out token NLL at the same top-k budget.',
        'mechanism': 'A pooled summary collapses distinct recent tokens; singletons keep their identity through the same pooling/projection parameters and let the indexer select among them.',
        'predicted_sign': -1,
        'primary_metric': 'ppl',
        'metric_policy': 'lm_perplexity',
        'requires_param_match': False,
        'falsifier': 'Singletons match or lose to the pooled summary after Holm correction, or the difference is within seed noise.',
    },
    'interaction: summary effect (none - 128)': {
        'statement': 'The NLL benefit of reading the open block is larger when no maximum block cap is imposed than under a 128-token cap.',
        'mechanism': 'Without a cap, the open block can grow, so the amount of otherwise-unread context that a prefix representation can recover increases.',
        'predicted_sign': -1,
        'primary_metric': 'ppl',
        'metric_policy': 'lm_perplexity',
        'requires_param_match': False,
        'falsifier': 'The no-cap summary benefit is not larger than the capped benefit after Holm correction, or the interaction is within seed noise.',
    },
}

NIAH_HYPOTHESIS_DESIGN = {
    'niah_learned_vs_dense_accuracy': {
        'statement': ('On the far-necessity NIAH task, learned block selection retrieves '
                      'planted answers more often than the parameter-matched dense baseline '
                      'at the same training budget. The reported contrast is control minus '
                      'learned top-1 accuracy, so a negative sign means learned is better.'),
        'mechanism': ('Query-conditioned block selection retrieves the far key/value block '
                      'instead of diluting it over the full dense context.'),
        'predicted_sign': -1,
        'primary_metric': 'top1_accuracy',
        'metric_policy': 'niah_retrieval',
        'requires_param_match': True,
        'falsifier': ('Dense matches or beats learned selection accuracy after Holm '
                      'correction, or the difference is within seed noise.'),
    },
    'niah_learned_vs_randidx_accuracy': {
        'statement': ('Learned block selection retrieves planted NIAH answers more often '
                      'than a trained random indexer with the same number of attended '
                      'blocks. The reported contrast is control minus learned accuracy, so '
                      'a negative sign means learned is better.'),
        'mechanism': ('The retrieval gain comes from the learned relevance score, not from '
                      'the extra trained parameters or the block budget itself.'),
        'predicted_sign': -1,
        'primary_metric': 'top1_accuracy',
        'metric_policy': 'niah_retrieval',
        'requires_param_match': True,
        'falsifier': ('The random indexer matches or beats learned selection accuracy after '
                      'Holm correction, or the difference is within seed noise.'),
    },
    'niah_learned_vs_allblocks_accuracy': {
        'statement': ('Against a read-rich trained all-block control, learned selection '
                      'at a fixed top-k retrieves planted NIAH answers more often. '
                      'The reported contrast is control minus learned accuracy, so a '
                      'negative sign means learned is better. This is a read-budget '
                      'intervention, not an equal-read comparison.'),
        'mechanism': ('Selecting the relevant blocks at a fixed top-k is better than '
                      'spending the full read budget on all blocks, which can dilute '
                      'attention to the far key.'),
        'predicted_sign': -1,
        'primary_metric': 'top1_accuracy',
        'metric_policy': 'niah_retrieval',
        'requires_param_match': True,
        'falsifier': ('The all-block control matches or beats learned selection accuracy '
                      'after Holm correction, or the difference is within seed noise.'),
    },
}

NIAH_INFERENCE_ARM_OVERRIDES = {
    'learned': {},
    'randidx': {'indexer_mode': 'random'},
    'allblocks': {'index_topk': 10 ** 6},
}

NIAH_CONTRAST_FACTORS = {
    'dense': {
        'a': 'csa_fixed_rope', 'b': 'full_rope',
        'single_factor': False,
        'factors': ['kind'],
        'factors_note': ('parameter-matched dense baseline: attention-operator swap, '
                         'architecture-level rather than a single knob'),
    },
    'randidx': {
        'override': 'randidx',
        'single_factor': True,
        'factors': ['indexer_mode'],
        'factors_note': ('inference-time random block-selection control on the same '
                         'checkpoint and top-k'),
    },
    'allblocks': {
        'override': 'allblocks',
        'single_factor': True,
        'factors': ['index_topk'],
        'factors_note': ('read-rich all-block intervention on the same checkpoint; '
                         'single-factor but NOT read-budget matched'),
    },
    'csa_randtrain': {
        'a': 'csa_fixed_rope_randidx', 'b': 'csa_fixed_rope',
        'single_factor': True,
        'factors': ['indexer_mode'],
        'factors_note': 'from-scratch random-selection control at the same top-k',
    },
    'csa_alltrain': {
        'a': 'csa_fixed_rope_allblocks', 'b': 'csa_fixed_rope',
        'single_factor': True,
        'factors': ['index_topk'],
        'factors_note': ('from-scratch read-rich all-block control; single-factor '
                         'but not read-budget matched'),
    },
}

def apply_niah_inference_arm(cfg, base_cfg, arm, overrides=None):
    overrides = NIAH_INFERENCE_ARM_OVERRIDES if overrides is None else overrides
    if not isinstance(overrides, dict):
        raise ValueError('NIAH arm overrides must be a mapping')
    all_keys = sorted({str(k) for spec in overrides.values()
                       if isinstance(spec, dict) for k in spec})
    for key in all_keys:
        if hasattr(base_cfg, key):
            setattr(cfg, key, getattr(base_cfg, key))
    arm_spec = overrides.get(arm) or {}
    if not isinstance(arm_spec, dict):
        raise ValueError(f'NIAH arm overrides for {arm!r} must be a mapping')
    for key, value in arm_spec.items():
        setattr(cfg, key, value)
    return cfg

def _splitmix64_u64(x):
    """Deterministic 64-bit mixing (SplitMix64) used for row folds.

    Fold assignment must not depend on NumPy's RNG implementation or on the
    process/OS: a tuning artifact has to reproduce the exact same selection
    rows on any machine.  SplitMix64 is a pure integer function, so the split
    is stable across versions and platforms."""
    mask = (1 << 64) - 1
    z = (int(x) + 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return (z ^ (z >> 31)) & mask
