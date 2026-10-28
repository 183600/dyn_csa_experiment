

def _plan_row_split(protocol, problems, warnings):
    """Validate the declared nested selection/evaluation row fold."""
    try:
        spec = tuning_row_split_spec(protocol)
    except ValueError as exc:
        problems.append(f'malformed selection_row_split: {exc}')
        return None
    if spec is None:
        warnings.append('no nested selection_row_split: hyper-parameters and the '
                        'final estimate may share validation documents')
        return None
    n_folds = spec['n_folds']
    sel = spec['selection_fold']
    ev = spec['evaluation_fold']
    if n_folds < 2:
        problems.append(f'selection_row_split.n_folds={n_folds} cannot form disjoint folds')
    if not (0 <= sel < n_folds) or not (0 <= ev < n_folds):
        problems.append('selection_row_split fold indices are outside [0, n_folds)')
    if sel == ev:
        problems.append('selection and evaluation folds are identical — the final '
                        'estimate would reuse the selection rows')
    elif sel > ev:
        problems.append('selection fold follows the evaluation fold; the nested '
                        'split must select on earlier rows')
    return spec


def _plan_trial_grid(trials, problems, warnings):
    """Check that the declared tuning grid is explicit, unique and brackets lr."""
    if not isinstance(trials, (list, tuple)):
        problems.append(f'trials must be a list, got {type(trials).__name__}')
        return {}, 0
    levels = {'lr': set(), 'weight_decay': set(), 'warmup': set()}
    signatures = set()
    for idx, trial in enumerate(trials):
        if not isinstance(trial, dict):
            problems.append(f'trial {idx} is a {type(trial).__name__}, not an object')
            continue
        cfg = {}
        for knob in ('lr', 'weight_decay', 'warmup'):
            val = _finite_scalar(trial.get(knob))
            if val is None:
                problems.append(f'trial {idx} has no finite {knob}')
                continue
            if knob == 'lr' and val <= 0:
                problems.append(f'trial {idx} lr must be positive, got {val}')
            if knob == 'weight_decay' and val < 0:
                problems.append(f'trial {idx} weight_decay must be non-negative, got {val}')
            if knob == 'warmup' and val < 0:
                problems.append(f'trial {idx} warmup must be non-negative, got {val}')
            cfg[knob] = val
            levels[knob].add(val)
        sig = json.dumps(cfg, sort_keys=True)
        if sig in signatures:
            problems.append(f'trial {idx} duplicates another declared cell {cfg}')
        signatures.add(sig)
    if len(trials) < 2:
        warnings.append('the declared tuning grid has fewer than 2 trials; a single '
                        'cell cannot show that a hyper-parameter optimum was bracketed')
    for knob, vals in levels.items():
        if vals and len(vals) < 2:
            warnings.append(f'{knob} has only one declared level ({sorted(vals)}); '
                            'the search cannot show an interior optimum for it')
    if levels['lr'] and min(levels['lr']) <= 0:
        problems.append('the declared lr levels must all be positive')
    return {k: sorted(v) for k, v in levels.items()}, len(signatures)


def _plan_arm_groups(protocol, arms, problems, warnings):
    """Every declared arm must belong to exactly one parameter-matching group.

    A tuning group also has to contain an actual comparison partner for its
    method arm.  Without that check a "baseline tuning" design can declare a
    method arm alone in its group, launch the expensive equal search, and only
    discover from the finished artifact that there was never a tuned baseline
    to beat.  Returns a machine-readable ladder summary for the plan record.
    """
    raw_groups = protocol.get('arm_groups')
    groups_out = []
    if raw_groups is None:
        warnings.append('no arm_groups declared; the legacy all-arms contrast rule '
                        'will be used and parameter-matching groups are implicit')
        return groups_out
    if not isinstance(raw_groups, (list, tuple)):
        problems.append(f'arm_groups must be a list, got {type(raw_groups).__name__}')
        return groups_out
    declared = set(arms)
    seen = set()
    for idx, group in enumerate(raw_groups):
        if not isinstance(group, dict):
            problems.append(f'arm_groups[{idx}] is a {type(group).__name__}, not an object')
            continue
        group_arms = group.get('arms')
        if not isinstance(group_arms, (list, tuple, set)):
            problems.append(f'arm_groups[{idx}].arms must be a list')
            continue
        group_arms = list(group_arms)
        if len(set(group_arms)) != len(group_arms):
            problems.append(f'arm_groups[{idx}].arms contains duplicates')
        method = group.get('method_arm')
        if method not in declared:
            problems.append(f'arm_groups[{idx}].method_arm={method!r} is not a declared arm')
        if method not in group_arms:
            problems.append(f'arm_groups[{idx}] method_arm={method!r} is not in its own arm list')
        partners = [a for a in group_arms if a != method]
        if not partners:
            problems.append(
                f'arm_groups[{idx}] declares method_arm={method!r} with no '
                'baseline/control arm in the same group; there is no tuned '
                'contrast to estimate')
        matched_raw = group.get('matched')
        matched = []
        if matched_raw is not None:
            if not isinstance(matched_raw, (list, tuple, set)):
                problems.append(f'arm_groups[{idx}].matched must be a list')
            else:
                matched = list(matched_raw)
                if method in matched:
                    problems.append(
                        f'arm_groups[{idx}] lists its own method_arm={method!r} '
                        'in the parameter-matched baseline set')
                for arm in matched:
                    if arm not in group_arms:
                        problems.append(
                            f'arm_groups[{idx}].matched contains {arm!r}, which '
                            'is not in this group')
                if not matched:
                    warnings.append(
                        f'arm_groups[{idx}] declares an empty matched set; the '
                        'method has no explicitly parameter-matched baseline')
        elif partners:
            warnings.append(
                f'arm_groups[{idx}] does not declare its parameter-matched '
                'baseline subset; matching is implicit')
        for arm in group_arms:
            if arm not in declared:
                problems.append(f'arm_groups[{idx}] contains undeclared arm {arm!r}')
            if arm in seen:
                problems.append(f'arm {arm!r} appears in more than one tuning group')
            seen.add(arm)
        groups_out.append({
            'method_arm': method,
            'arms': group_arms,
            'baseline_arms': partners,
            'matched': matched,
            'has_baseline': bool(partners),
            'has_matched_baseline': bool(matched),
            'declared_contrasts': [[method, a] for a in partners if a in declared],
        })
    missing = sorted(declared - seen)
    if missing:
        problems.append(f'declared arm(s) {missing} are not covered by any tuning group')
    inherits = protocol.get('inherits_tuning') or {}
    if inherits:
        if not isinstance(inherits, dict):
            problems.append('inherits_tuning must be a mapping')
        else:
            for arm, ref in inherits.items():
                if ref not in declared:
                    problems.append(f'inherits_tuning[{arm!r}]={ref!r} is not a declared arm')
                if arm == ref:
                    problems.append(f'inherits_tuning[{arm!r}] points at itself')
    return groups_out


def baseline_tuning_plan(protocol=None, *, alpha=0.05):
    """Zero-GPU preflight for the equal-search tuning *design*.

    Run this BEFORE spending GPU time.  It validates the declared arm groups,
    trial grid, seed partitions, nested row split and family power floor, and
    audits the tuned-baseline hypothesis.  Structural defects make the plan
    fail; an underpowered final seed partition is reported as a warning with
    ``confirmatory_capable=False`` rather than silently launching a panel whose
    p-value floor cannot reach alpha.
    """
    alpha = _validated_alpha(alpha, context='baseline_tuning_plan alpha')
    protocol = BASELINE_TUNING_PROTOCOL if protocol is None else protocol
    problems, warnings = [], []
    out = {'kind': 'baseline_tuning_plan', 'status': 'fail', 'may_launch': False,
           'confirmatory_capable': False, 'protocol': protocol,
           'arms': [], 'declared_contrasts': [], 'family_size': None,
           'family_basis': None, 'search_seeds': [], 'final_seeds': [],
           'search_final_seed_disjoint': None,
           'min_final_seeds_for_family_significance': None,
           'final_seeds_sufficient': None, 'trial_count': 0,
           'grid_levels': {}, 'tuning_groups': [], 'selection_row_split': None,
           'hypothesis_audit': {}, 'problems': problems, 'warnings': warnings}
    if not isinstance(protocol, dict):
        problems.append(f'protocol must be an object, got {type(protocol).__name__}')
        return out
    raw_arms = protocol.get('arms')
    if not isinstance(raw_arms, (list, tuple, set)):
        problems.append(f'protocol arms must be a list, got {type(raw_arms).__name__}')
        arms = []
    else:
        arms = [str(a) for a in raw_arms]
        if not arms:
            problems.append('protocol declares no arms')
        if len(set(arms)) != len(arms):
            problems.append('protocol arms list contains duplicates')
        arms = sorted(set(arms))
    out['arms'] = arms
    # The tuning target is fixed by the project's metric contract: validation
    # PPL, lower is better.  A protocol that silently tunes on test or another
    # endpoint is a design defect, not a variant.
    metric = protocol.get('selection_metric')
    split = protocol.get('selection_split')
    direction = protocol.get('direction')
    if metric != 'ppl':
        problems.append(f'selection_metric={metric!r} is not the declared LM primary `ppl`')
    if split != 'validation':
        problems.append(f'selection_split={split!r}; tuning must select on validation, never test')
    if direction != 'lower_is_better':
        problems.append(f'selection direction={direction!r}; lower PPL is better')
    # Grid and groups.
    grid_levels, n_trials = _plan_trial_grid(protocol.get('trials'), problems, warnings)
    out['grid_levels'] = grid_levels
    out['trial_count'] = n_trials
    if not isinstance(protocol.get('search_design'), str) or not protocol.get('search_design').strip():
        problems.append('search_design must name the declared search design')
    _note = protocol.get('search_design_note')
    if _note is not None and (not isinstance(_note, str) or not _note.strip()):
        problems.append('search_design_note must be non-empty when present')
    out['tuning_groups'] = _plan_arm_groups(protocol, arms, problems, warnings)
    # Seed partitions: search and final draws must be disjoint, and the final
    # partition must be large enough for the declared Holm family.
    search = _plan_seed_list(protocol.get('search_seeds'), 'search_seeds', problems)
    final = _plan_seed_list(protocol.get('final_seeds'), 'final_seeds', problems)
    out['search_seeds'] = search
    out['final_seeds'] = final
    overlap = sorted(set(search) & set(final))
    out['search_final_seed_disjoint'] = (not overlap) if (search or final) else None
    if overlap:
        problems.append(f'search_seeds and final_seeds overlap on {overlap}; the final '
                        'estimate would reuse the draws that selected the config')
    try:
        plan, basis = _tuning_declared_contrasts(protocol, arms)
    except (TypeError, ValueError) as exc:
        plan, basis = [], f'undeclarable ({type(exc).__name__}: {exc})'
        problems.append(f'cannot derive the declared tuned contrasts: {exc}')
    out['declared_contrasts'] = [[a, b] for a, b in plan]
    out['family_size'] = len(plan) if plan else None
    out['family_basis'] = basis
    if not plan:
        problems.append('no declared tuned contrast; there is no family to power or audit')
    min_n = signflip_min_n_for_holm(len(plan), alpha) if plan else None
    out['min_final_seeds_for_family_significance'] = min_n
    out['final_seeds_sufficient'] = (bool(min_n is not None and len(final) >= min_n)
                                     if (plan and final) else None)
    if final and len(final) < 3:
        warnings.append(f'only {len(final)} final seed(s) declared; the project reporting '
                        'floor for a key experiment is 3')
    if out['final_seeds_sufficient'] is False:
        warnings.append(f'{len(final)} final seed(s) cannot reach a Holm-corrected p<alpha '
                        f'for the {len(plan)} declared contrast(s) (needs n>={min_n}); '
                        'the panel is descriptive only')
    out['selection_row_split'] = _plan_row_split(protocol, problems, warnings)
    # Pre-registered tuned-baseline hypothesis.
    try:
        ha = hypothesis_audit(TUNED_BASELINE_HYPOTHESIS_DESIGN, 'tuned_baseline')
    except Exception as exc:  # audit must never mask the design report
        ha = {'status': 'fail', 'warnings': [f'audit failed: {type(exc).__name__}: {exc}']}
    out['hypothesis_audit'] = ha
    for warning in ha.get('warnings') or []:
        warnings.append(f'hypothesis audit: {warning}')
    if ha.get('status') not in (None, 'ok'):
        problems.append('the tuned-baseline hypothesis is incomplete or metric-misaligned')
    out['status'] = 'fail' if problems else ('warn' if warnings else 'ok')
    out['may_launch'] = not problems
    out['confirmatory_capable'] = bool(not problems and out['final_seeds_sufficient'])
    return out


def simple_baseline_plan(protocol=None):
    """Zero-GPU preflight for the tuned count-model floor design.

    Verifies that every declared count model has a complete, non-empty grid
    (the policy claims a full factorial search), that selection is validation
    PPL, and that the fit-token cap is explicit.  This does not read data; it
    only checks that the floor's search can be called "tuned" before it runs.
    """
    protocol = SIMPLE_BASELINE_POLICY if protocol is None else protocol
    problems, warnings = [], []
    out = {'kind': 'simple_baseline_plan', 'status': 'fail', 'may_launch': False,
           'protocol': protocol, 'arms': {}, 'expected_cells': {},
           'selection_row_split': None, 'problems': problems, 'warnings': warnings}
    if not isinstance(protocol, dict):
        problems.append(f'protocol must be an object, got {type(protocol).__name__}')
        return out
    metric = protocol.get('selection_metric')
    split = protocol.get('selection_split')
    direction = protocol.get('direction')
    if metric != 'ppl':
        problems.append(f'selection_metric={metric!r} is not `ppl`')
    if split != 'validation':
        problems.append(f'selection_split={split!r}; the floor must select on validation')
    if direction != 'lower_is_better':
        problems.append(f'selection direction={direction!r}; lower PPL is better')
    arms = protocol.get('arms')
    grids = protocol.get('grids')
    if not isinstance(arms, dict) or not arms:
        problems.append('arms must be a non-empty object')
        arms = {}
    if not isinstance(grids, dict) or not grids:
        problems.append('grids must be a non-empty object')
        grids = {}
    expected_keys = {1: {'alpha'}, 2: {'alpha', 'interpolation'}}
    for name, spec in arms.items():
        if not isinstance(spec, dict):
            problems.append(f'arm {name!r} is not an object')
            continue
        order = _as_seed_int(spec.get('order'))
        if order not in expected_keys:
            problems.append(f'arm {name!r} has unsupported order={spec.get("order")!r}')
            continue
        grid = grids.get(name)
        if not isinstance(grid, dict):
            problems.append(f'arm {name!r} has no declared grid')
            continue
        missing = sorted(expected_keys[order] - set(grid))
        extra = sorted(set(grid) - expected_keys[order])
        if missing:
            problems.append(f'arm {name!r} grid is missing {missing}')
        if extra:
            problems.append(f'arm {name!r} grid has undeclared key(s) {extra}')
        cells = 0
        for knob, values in grid.items():
            if not isinstance(values, (list, tuple)) or not values:
                problems.append(f'arm {name!r} grid[{knob!r}] must be a non-empty list')
                continue
            parsed = []
            for value in values:
                val = _finite_scalar(value)
                if val is None:
                    problems.append(f'arm {name!r} grid[{knob!r}] has a non-finite value {value!r}')
                    continue
                if knob == 'alpha' and val <= 0:
                    problems.append(f'arm {name!r} grid alpha must be positive, got {val}')
                if knob == 'interpolation' and not (0.0 <= val <= 1.0):
                    problems.append(f'arm {name!r} interpolation must be in [0,1], got {val}')
                parsed.append(val)
            if len(set(parsed)) != len(parsed):
                problems.append(f'arm {name!r} grid[{knob!r}] contains duplicate levels')
            if len(parsed) < 2:
                warnings.append(f'arm {name!r} {knob} has fewer than 2 levels; the '
                                'search cannot show an interior optimum')
            cells = len(parsed) if cells == 0 else cells * len(parsed)
        out['arms'][name] = {'order': order, 'grid': grid}
        out['expected_cells'][name] = cells
    if protocol.get('search_design') != 'full_factorial':
        warnings.append(f"search_design={protocol.get('search_design')!r}; the policy "
                        'claims a full factorial count-model search')
    cap = protocol.get('max_fit_tokens')
    if cap is not None:
        cap_i = _as_seed_int(cap)
        if cap_i is None or cap_i < 1:
            problems.append(f'max_fit_tokens={cap!r} must be a positive integer or null')
    policy = protocol.get('fit_token_policy', 'deterministic_prefix')
    if policy != 'deterministic_prefix':
        problems.append(f'fit_token_policy={policy!r} is not the implemented '
                        '`deterministic_prefix`')
    out['selection_row_split'] = _plan_row_split(protocol, problems, warnings)
    out['status'] = 'fail' if problems else ('warn' if warnings else 'ok')
    out['may_launch'] = not problems
    return out


def baseline_plan(protocol=None, simple_protocol=None, *, alpha=0.05):
    """Combined preflight for the tuned neural baselines and the count floor."""
    tuned = baseline_tuning_plan(protocol, alpha=alpha)
    simple = simple_baseline_plan(simple_protocol)
    status = ('fail' if 'fail' in (tuned.get('status'), simple.get('status'))
              else ('warn' if 'warn' in (tuned.get('status'), simple.get('status')) else 'ok'))
    return {'kind': 'baseline_plan', 'status': status,
            'may_launch': bool(tuned.get('may_launch') and simple.get('may_launch')),
            'confirmatory_capable': bool(tuned.get('confirmatory_capable')),
            'tuned': tuned, 'simple': simple,
            'problems': (list(tuned.get('problems') or [])
                         + [f'simple: {p}' for p in (simple.get('problems') or [])]),
            'warnings': (list(tuned.get('warnings') or [])
                         + [f'simple: {w}' for w in (simple.get('warnings') or [])])}
