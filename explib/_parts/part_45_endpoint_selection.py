# ---- endpoint provenance: which step produced the quoted measurement? ----
# A run evaluates several times.  The quoted PPL can come from the fixed final
# step, from the best validation step, or from an early-stopped step.  Those
# are different estimands.  Choosing the best validation point per arm and then
# comparing the selected values is a max-selection (winner's curse); comparing
# one arm's final step with another arm's best step is not a controlled
# comparison at all; and evaluating a selected point on the same rows that
# chose it is optimistically biased.
#
# This fragment adds a zero-GPU audit of the *selection rule* behind a reported
# endpoint.  It is provenance/reporting, not a new test: it never recomputes a
# p-value, changes a Holm family, or rewrites a frozen decision.  A clean
# `final_step` panel is the default; a data-dependent rule is allowed only when
# its selection split is explicit and independent of the evaluation rows.

ENDPOINT_SELECTION_POLICY = {
    'version': 'endpoint-selection-1',
    'rules': ('final_step', 'best_validation_step', 'early_stopped_best',
              'early_stopped_stop', 'intermediate_step', 'truncated_step',
              'unknown'),
    'data_dependent_rules': ('best_validation_step', 'early_stopped_best',
                             'early_stopped_stop', 'intermediate_step'),
    'splits': ('validation', 'holdout', 'nested_validation', 'test', 'other'),
    'default_min_n': 3,
    'note': ('reported-endpoint provenance only: a data-dependent checkpoint '
             'rule is a selection mechanism, not a free accuracy gain.  This '
             'audit adds no p-value and changes no Holm/verified/frozen '
             'decision.'),
}

_ENDPOINT_RULES = frozenset(ENDPOINT_SELECTION_POLICY['rules'])
_ENDPOINT_DATA_DEPENDENT = frozenset(
    ENDPOINT_SELECTION_POLICY['data_dependent_rules'])
_ENDPOINT_SPLITS = frozenset(ENDPOINT_SELECTION_POLICY['splits'])
_ENDPOINT_CKPT_MODES = frozenset({'final', 'best', 'both', 'none', 'unknown'})

_ENDPOINT_RULE_ALIASES = {
    'final': 'final_step',
    'final_step': 'final_step',
    'last_step': 'final_step',
    'best': 'best_validation_step',
    'best_step': 'best_validation_step',
    'best_validation': 'best_validation_step',
    'best_validation_step': 'best_validation_step',
    'early_stop': 'early_stopped_best',
    'early_stopped': 'early_stopped_best',
    'early_stopped_best': 'early_stopped_best',
    'early_stopped_stop': 'early_stopped_stop',
    'intermediate': 'intermediate_step',
    'intermediate_step': 'intermediate_step',
    'truncated': 'truncated_step',
    'truncated_step': 'truncated_step',
    'unknown': 'unknown',
}

_ENDPOINT_SPLIT_ALIASES = {
    'val': 'validation',
    'valid': 'validation',
    'validation': 'validation',
    'holdout': 'holdout',
    'heldout': 'holdout',
    'nested': 'nested_validation',
    'nested_validation': 'nested_validation',
    'test': 'test',
    'other': 'other',
}


def _endpoint_rule(value):
    """Normalise a declared rule name; return None for an unknown spelling."""
    if not isinstance(value, str):
        return None
    key = value.strip().lower().replace('-', '_').replace(' ', '_')
    if key in _ENDPOINT_RULE_ALIASES:
        return _ENDPOINT_RULE_ALIASES[key]
    return key if key in _ENDPOINT_RULES else None


def _endpoint_split(value):
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    key = value.strip().lower().replace('-', '_').replace(' ', '_')
    return _ENDPOINT_SPLIT_ALIASES.get(key)


def _endpoint_int(value):
    """Non-negative integer field; booleans and fractional floats are rejected."""
    value = _as_seed_int(value)
    if value is None or value < 0:
        return None
    return int(value)


def _endpoint_first(obj, *keys):
    """First non-None value among ``keys`` (a present-but-None alias must
    not mask the next field)."""
    if not isinstance(obj, dict):
        return None
    for key in keys:
        value = obj.get(key)
        if value is not None:
            return value
    return None


def _endpoint_split_from_rows(value):
    """Best-effort read of the evaluation split from an estimation-rows label."""
    if not isinstance(value, str):
        return None
    label = value.strip().lower()
    if 'test' in label:
        return 'test'
    if 'holdout' in label or 'held' in label:
        return 'holdout'
    if 'nested' in label:
        return 'nested_validation'
    if 'validation' in label or label.startswith('val') or '_val' in label:
        return 'validation'
    return None


def _endpoint_record(raw, *, where):
    """Normalise one per-(arm, tag, seed) record and infer its endpoint rule."""
    violations = []
    warnings = []
    if not isinstance(raw, dict):
        return None, [f'{where}: record is a {type(raw).__name__}, not an object'], []
    arm = raw.get('arm')
    if arm is None:
        arm = raw.get('variant')
    if not isinstance(arm, str) or not arm.strip():
        violations.append(f'{where}: arm must be a non-empty string')
        arm = None
    else:
        arm = arm.strip()
    tag = raw.get('tag', '')
    if tag is None:
        tag = ''
    if not isinstance(tag, str):
        violations.append(f'{where}: tag must be a string or None')
        tag = ''
    seed = _as_seed_int(raw.get('seed'))
    if seed is None:
        violations.append(f'{where}: seed must be an integer')
    # Steps.  `steps` is the planned budget in the training records;
    # `steps_completed` is how far the run actually got.
    planned_raw = _endpoint_first(raw, 'steps_planned', 'steps')
    completed_raw = _endpoint_first(raw, 'steps_completed', 'steps_done')
    # Legacy/rope records carry only `steps` (the run length); the training
    # record carries both the planned `steps` and the actual
    # `steps_completed`.  Only that legacy shape may be read as a completed
    # run.  A record that declares `steps_planned` but omits
    # `steps_completed`/`steps_done` is *not* evidence that the planned budget
    # was reached: treating the plan as the actual count would turn an
    # unrecorded truncation into a clean `final_step`.
    completed_inferred = False
    if (completed_raw is None and raw.get('steps_planned') is None
            and raw.get('steps') is not None):
        completed_raw = raw.get('steps')
        completed_inferred = True
    planned = None
    completed = None
    if planned_raw is not None:
        planned = _endpoint_int(planned_raw)
        if planned is None or planned < 1:
            violations.append(
                f'{where}: steps_planned must be a positive integer '
                f'({planned_raw!r})')
    if completed_raw is not None:
        completed = _endpoint_int(completed_raw)
        if completed is None:
            violations.append(
                f'{where}: steps_completed must be a non-negative integer '
                f'({completed_raw!r})')
        elif completed == 0:
            violations.append(
                f'{where}: steps_completed=0 is not a measured endpoint')
    if planned is not None and completed is not None and completed > planned:
        violations.append(
            f'{where}: steps_completed={completed} exceeds steps_planned={planned}')
    # `steps_completed` and `steps_done` are aliases for the same actual
    # count; two different values are a record-integrity failure.
    _done_vals = [_endpoint_int(raw.get(_k))
                  for _k in ('steps_completed', 'steps_done')
                  if raw.get(_k) is not None]
    _done_vals = [v for v in _done_vals if v is not None]
    if len(set(_done_vals)) > 1:
        violations.append(
            f'{where}: conflicting steps_completed/steps_done {_done_vals!r}')
    cap_truncated = raw.get('cap_truncated')
    if cap_truncated is not None and not isinstance(cap_truncated, (bool, np.bool_)):
        violations.append(f'{where}: cap_truncated must be a boolean')
        cap_truncated = None
    cap_truncated = bool(cap_truncated) if cap_truncated is not None else False
    reported_raw = _endpoint_first(raw, 'reported_step', 'eval_step',
                                   'endpoint_step')
    reported = None
    if reported_raw is not None:
        reported = _endpoint_int(reported_raw)
        if reported is None or reported < 1:
            violations.append(
                f'{where}: reported_step must be a positive integer '
                f'({reported_raw!r})')
    if reported is not None and completed is not None and reported > completed:
        violations.append(
            f'{where}: reported_step={reported} exceeds steps_completed='
            f'{completed}')
    # PPL is optional: endpoint-only records are useful for the rule audit, but
    # a present value must be a real PPL (>=1) before it can enter a pair.
    ppl = raw.get('ppl')
    ppl_usable = False
    if ppl is not None:
        if not ppl_is_usable(ppl):
            violations.append(f'{where}: ppl must be finite and >= 1 ({ppl!r})')
        else:
            ppl_usable = True
    # Checkpoint metadata: a selected checkpoint must name the step it holds.
    ckpt = raw.get('reported_checkpoint')
    if ckpt is None:
        ckpt = raw.get('checkpoint')
    ckpt_mode = None
    ckpt_step = None
    ckpt_sha = None
    if ckpt is not None:
        if not isinstance(ckpt, dict):
            violations.append(
                f'{where}: checkpoint must be an object, not '
                f'{type(ckpt).__name__}')
        else:
            mode_raw = ckpt.get('mode')
            if mode_raw is not None:
                mode = str(mode_raw).strip().lower()
                if mode not in _ENDPOINT_CKPT_MODES:
                    violations.append(
                        f'{where}: unknown checkpoint mode {mode_raw!r}; '
                        f'expected one of {sorted(_ENDPOINT_CKPT_MODES)}')
                else:
                    ckpt_mode = mode
                    if mode == 'both':
                        violations.append(
                            f'{where}: checkpoint mode "both" does not say '
                            'which weight produced the quoted number')
            step_raw = ckpt.get('step')
            if step_raw is not None:
                ckpt_step = _endpoint_int(step_raw)
                if ckpt_step is None or ckpt_step < 1:
                    violations.append(
                        f'{where}: checkpoint.step must be a positive integer '
                        f'({step_raw!r})')
            sha_raw = ckpt.get('sha256', ckpt.get('digest'))
            if sha_raw is not None:
                if not isinstance(sha_raw, str) or not sha_raw.strip():
                    violations.append(
                        f'{where}: checkpoint.sha256 must be a non-empty string')
                else:
                    ckpt_sha = sha_raw.strip()
    # Early stopping.
    early = raw.get('early_stop')
    early_stopped = None
    early_best_step = None
    if early is not None:
        if not isinstance(early, dict):
            violations.append(
                f'{where}: early_stop must be an object, not '
                f'{type(early).__name__}')
        else:
            stopped_raw = early.get('stopped')
            if stopped_raw is not None:
                if not isinstance(stopped_raw, (bool, np.bool_)):
                    violations.append(
                        f'{where}: early_stop.stopped must be a boolean')
                else:
                    early_stopped = bool(stopped_raw)
            best_raw = early.get('best_step')
            if best_raw is not None:
                early_best_step = _endpoint_int(best_raw)
                if early_best_step is None or early_best_step < 1:
                    violations.append(
                        f'{where}: early_stop.best_step must be a positive '
                        f'integer ({best_raw!r})')
    # Curve health / top-level best-step diagnostics.  These are diagnostic
    # fields recorded by train_variant; they do not by themselves prove that
    # the quoted PPL came from the best checkpoint, but they let the audit
    # recognise the common "reported at best_step" record.
    health = raw.get('curve_health')
    if health is None:
        health = {}
    elif not isinstance(health, dict):
        violations.append(
            f'{where}: curve_health must be an object, not '
            f'{type(health).__name__}')
        health = {}
    best_step_raw = _endpoint_first(health, 'best_step')
    if best_step_raw is None:
        best_step_raw = _endpoint_first(raw, 'best_step')
    if best_step_raw is None and early_best_step is not None:
        best_step_raw = early_best_step
    best_val_raw = _endpoint_first(health, 'best_val_ppl')
    if best_val_raw is None:
        best_val_raw = _endpoint_first(raw, 'best_val_ppl')
    final_minus_best_raw = _endpoint_first(health, 'final_minus_best')
    if final_minus_best_raw is None:
        final_minus_best_raw = _endpoint_first(raw, 'final_minus_best')
    best_step = None
    if best_step_raw is not None:
        best_step = _endpoint_int(best_step_raw)
        if best_step is None or best_step < 1:
            violations.append(
                f'{where}: best_step must be a positive integer '
                f'({best_step_raw!r})')
    best_val = None
    if best_val_raw is not None:
        if not ppl_is_usable(best_val_raw):
            violations.append(
                f'{where}: best_val_ppl must be finite and >= 1 '
                f'({best_val_raw!r})')
        else:
            best_val = float(best_val_raw)
    final_minus_best = None
    if final_minus_best_raw is not None:
        final_minus_best = _finite_scalar(final_minus_best_raw)
        if final_minus_best is None:
            violations.append(
                f'{where}: final_minus_best must be finite '
                f'({final_minus_best_raw!r})')
    if best_step is not None and completed is not None and best_step > completed:
        violations.append(
            f'{where}: best_step={best_step} exceeds steps_completed='
            f'{completed}')
    if best_step is not None and early_best_step is not None and best_step != early_best_step:
        warnings.append(
            f'{where}: curve_health.best_step={best_step} disagrees with '
            f'early_stop.best_step={early_best_step}')
    # Declared split identities.
    selection_split = _endpoint_split(raw.get('selection_split'))
    if raw.get('selection_split') is not None and selection_split is None:
        violations.append(
            f'{where}: unknown selection_split {raw.get("selection_split")!r}')
    eval_split = _endpoint_split(raw.get('eval_split'))
    if raw.get('eval_split') is not None and eval_split is None:
        violations.append(
            f'{where}: unknown eval_split {raw.get("eval_split")!r}')
    if eval_split is None:
        eval_split = _endpoint_split_from_rows(raw.get('ppl_estimation_rows'))
    # The reported step defaults to the last completed step: train_variant's
    # `ppl` is the evaluation at the end of the run, not an arbitrary history
    # point.  An explicit reported_step always wins.
    if reported is None:
        if completed is not None:
            reported = completed
        elif ckpt_step is not None:
            reported = ckpt_step
    if ckpt_step is not None and reported is not None and ckpt_step != reported:
        violations.append(
            f'{where}: checkpoint.step={ckpt_step} disagrees with '
            f'reported_step={reported}')
    # Declared rule (if any) wins over inference, but it must agree with the
    # step fields; otherwise the declaration is a claim the record contradicts.
    rule_raw = _endpoint_first(raw, 'endpoint_selection_rule',
                               'selection_rule')
    declared = _endpoint_rule(rule_raw)
    if rule_raw is not None and declared is None:
        violations.append(
            f'{where}: unknown selection_rule {rule_raw!r}')
    rule = None
    source = None
    if declared is not None:
        rule = declared
        source = 'declared'
        if rule == 'final_step':
            if completed is not None and reported is not None and reported != completed:
                violations.append(
                    f'{where}: declared final_step but reported_step='
                    f'{reported} != steps_completed={completed}')
            if planned is not None and completed is not None and completed < planned:
                violations.append(
                    f'{where}: declared final_step but only {completed} of '
                    f'{planned} planned steps completed')
            if cap_truncated:
                violations.append(
                    f'{where}: declared final_step but cap_truncated is true')
        elif rule == 'best_validation_step':
            if best_step is not None and reported is not None and best_step != reported:
                violations.append(
                    f'{where}: declared best_validation_step but reported_step='
                    f'{reported} != best_step={best_step}')
        elif rule in ('early_stopped_best', 'early_stopped_stop'):
            if early_stopped is False:
                violations.append(
                    f'{where}: declared {rule} but early_stop.stopped is false')
            if rule == 'early_stopped_best' and best_step is not None and reported is not None and best_step != reported:
                violations.append(
                    f'{where}: declared early_stopped_best but reported_step='
                    f'{reported} != best_step={best_step}')
        elif rule == 'intermediate_step':
            if completed is not None and reported is not None and reported >= completed:
                violations.append(
                    f'{where}: declared intermediate_step but reported_step='
                    f'{reported} is not before steps_completed={completed}')
        elif rule == 'truncated_step':
            if planned is not None and completed is not None and completed >= planned:
                violations.append(
                    f'{where}: declared truncated_step but steps_completed >= '
                    'steps_planned')
    elif cap_truncated:
        # The runtime cap is an explicit budget statement.  It takes priority
        # over an early-stop flag: the run did not reach its planned budget,
        # so the quoted value is a truncated endpoint regardless of what the
        # validation curve was doing when the cap fired.
        rule = 'truncated_step'
        source = 'inferred'
    elif early_stopped is True:
        if (best_step is not None and reported is not None
                and best_step == reported):
            rule = 'early_stopped_best'
        else:
            rule = 'early_stopped_stop'
        source = 'inferred'
    elif planned is not None and completed is not None and completed < planned:
        rule = 'truncated_step'
        source = 'inferred'
    elif (best_step is not None and reported is not None
          and best_step == reported and completed is not None
          and reported < completed):
        rule = 'best_validation_step'
        source = 'inferred'
    elif completed is not None and reported is not None and reported < completed:
        rule = 'intermediate_step'
        source = 'inferred'
    elif completed is not None and reported == completed:
        rule = 'final_step'
        source = 'inferred'
    else:
        rule = 'unknown'
        source = 'unresolved'
        warnings.append(
            f'{where}: endpoint selection rule could not be inferred '
            '(missing steps/reported_step)')
    data_dependent = rule in _ENDPOINT_DATA_DEPENDENT
    if selection_split == 'test':
        violations.append(
            f'{where}: selection_split=test means the endpoint was chosen on '
            'the test split; a confirmatory measurement cannot be selected there')
    if (data_dependent and selection_split is not None
            and eval_split is not None and selection_split == eval_split):
        violations.append(
            f'{where}: selection and evaluation both use '
            f'{selection_split} rows; a selected endpoint evaluated on the '
            'same rows is optimistically biased')
    if data_dependent and selection_split is None:
        warnings.append(
            f'{where}: {rule} is a data-dependent selection but no '
            'selection_split is recorded; independence from the evaluation '
            'rows is unverifiable')
    out = {
        'arm': arm, 'tag': tag, 'seed': seed,
        'rule': rule, 'source': source,
        'data_dependent': bool(data_dependent),
        'reported_step': reported,
        'steps_planned': planned, 'steps_completed': completed,
        'steps_completed_inferred': bool(completed_inferred),
        'selection_split': selection_split, 'eval_split': eval_split,
        'checkpoint_mode': ckpt_mode,
        'checkpoint_step': ckpt_step,
        'checkpoint_sha256': ckpt_sha,
        'best_step': best_step, 'best_val_ppl': best_val,
        'final_minus_best': final_minus_best,
        'ppl_usable': ppl_usable,
        'warnings': warnings,
    }
    return out, violations, warnings


def _endpoint_policy_entries(manifest):
    """Normalise manifest `endpoint_policy` declarations.

    A declaration is either one object or a list of objects with `kind` (or
    `rule`), optional `selection_split`, `comparison` / `name`, `arms`, and
    `require_checkpoint_identity`.  A manifest without the key declares
    nothing; that is not silently read as `final_step`.
    """
    if manifest is None:
        return [], []
    if not isinstance(manifest, dict):
        return [], [f'manifest must be an object, not {type(manifest).__name__}']
    raw = manifest.get('endpoint_policy')
    if raw is None and isinstance(manifest.get('statistics'), dict):
        raw = manifest['statistics'].get('endpoint_policy')
    if raw is None:
        return [], []
    items = [raw] if isinstance(raw, dict) else list(raw) if isinstance(raw, (list, tuple)) else None
    if items is None:
        return [], [f'endpoint_policy must be an object or a list, not {type(raw).__name__}']
    out = []
    problems = []
    for i, item in enumerate(items):
        where = f'endpoint_policy[{i}]'
        if not isinstance(item, dict):
            problems.append(f'{where}: declaration is not an object')
            continue
        kind = _endpoint_rule(item.get('kind', item.get('rule')))
        if kind is None:
            problems.append(f'{where}: missing/unknown kind {item.get("kind", item.get("rule"))!r}')
            continue
        split_raw = item.get('selection_split')
        split = _endpoint_split(split_raw)
        if split_raw is not None and split is None:
            problems.append(f'{where}: unknown selection_split {split_raw!r}')
            continue
        comparison = item.get('comparison', item.get('name'))
        if comparison is not None and not isinstance(comparison, str):
            problems.append(f'{where}: comparison must be a string or None')
            continue
        arms = item.get('arms')
        if arms is not None:
            if (not isinstance(arms, (list, tuple)) or len(arms) != 2
                    or not all(isinstance(x, str) and x.strip() for x in arms)):
                problems.append(f'{where}: arms must be a pair of arm names')
                continue
            arms = tuple(x.strip() for x in arms)
        require_ckpt = item.get('require_checkpoint_identity', False)
        if not isinstance(require_ckpt, (bool, np.bool_)):
            problems.append(f'{where}: require_checkpoint_identity must be boolean')
            continue
        out.append({'kind': kind, 'selection_split': split,
                    'comparison': comparison.strip() if isinstance(comparison, str) else None,
                    'arms': arms, 'require_checkpoint_identity': bool(require_ckpt)})
    return out, problems


def _endpoint_policy_matches(policy, comp):
    if policy.get('comparison') is not None:
        return policy['comparison'] == comp['name']
    if policy.get('arms') is not None:
        return set(policy['arms']) == {comp['a'], comp['b']}
    return True


def _endpoint_comp_key(a, tag_a, b, tag_b):
    if tag_a == tag_b:
        base = f'{a} - {b}'
        return f'{base} [{tag_a}]' if tag_a else base
    return f'{a}[{tag_a}] - {b}[{tag_b}]'


def endpoint_selection_audit(records, *, comparisons=None, manifest=None,
                             require_preregistered=False, min_n=None,
                             require_checkpoint_identity=False):
    """Audit which checkpoint/step produced each quoted endpoint.

    The unit of analysis is one measured record per ``(arm, tag, seed)``.  A
    record may carry ``steps``/``steps_planned``, ``steps_completed``,
    ``reported_step``/``eval_step``, ``selection_rule``, ``selection_split``,
    ``eval_split``, ``checkpoint`` (with ``mode``/``step``/``sha256``),
    ``early_stop`` and the diagnostic ``best_step``/``best_val_ppl``/
    ``final_minus_best`` fields written by :func:`train_variant`.

    The audit infers a rule only when the record itself supports it:

    * ``final_step`` -- the last completed step is the quoted endpoint;
    * ``best_validation_step`` -- the quoted endpoint is the best validation
      step (a data-dependent max-selection);
    * ``early_stopped_best`` / ``early_stopped_stop`` -- early stopping chose
      the endpoint (best step or the stop step);
    * ``truncated_step`` -- the run stopped before its planned budget without
      an early-stop declaration;
    * ``intermediate_step`` -- some earlier step is quoted;
    * ``unknown`` -- the fields do not identify a rule.

    Paired comparisons fail closed when the two arms use different rules, when
    one arm mixes rules across seeds, when the selected checkpoint's step
    disagrees with the reported step, or when selection and evaluation use the
    same split.  A data-dependent rule on an independent split is a warning
    (its selection split should still be reported); selection on the test
    split, evaluation on the same rows that selected the endpoint, or a
    missing required manifest declaration is a failure.

    ``comparisons`` accepts dicts with ``a``/``b`` and optional
    ``tag_a``/``tag_b``/``name``, two-tuples ``(a, b)`` (empty tags), or
    four-tuples ``(a, tag_a, b, tag_b)``.  When omitted, every pair of arms
    sharing a tag is compared.

    This function computes **no p-value**, performs no multiplicity
    correction, and never changes a frozen verdict.  It returns a report dict
    with ``status`` in ``ok`` / ``warn`` / ``fail`` / ``not_verifiable``.
    """
    mn = ENDPOINT_SELECTION_POLICY['default_min_n'] if min_n is None else min_n
    if isinstance(mn, (bool, np.bool_)) or _as_seed_int(mn) is None:
        raise ValueError(f'min_n must be an integer >= 2, got {min_n!r}')
    mn = int(_as_seed_int(mn))
    if mn < 2:
        raise ValueError(f'min_n must be >= 2, got {min_n!r}')
    if not isinstance(require_preregistered, (bool, np.bool_)):
        raise ValueError(
            f'require_preregistered must be a boolean, got '
            f'{require_preregistered!r}')
    if not isinstance(require_checkpoint_identity, (bool, np.bool_)):
        raise ValueError(
            f'require_checkpoint_identity must be a boolean, got '
            f'{require_checkpoint_identity!r}')
    require_preregistered = bool(require_preregistered)
    require_checkpoint_identity = bool(require_checkpoint_identity)
    # A records mapping is accepted for artifact readers; the key is only a
    # label, so the record itself must carry (arm, tag, seed).
    if isinstance(records, dict):
        raw_items = []
        for key, value in records.items():
            if not isinstance(value, dict):
                raw_items.append(value)
                continue
            rec = dict(value)
            if rec.get('arm') is None or rec.get('seed') is None:
                if isinstance(key, tuple) and len(key) >= 2:
                    rec.setdefault('arm', key[0])
                    rec.setdefault('seed', key[-1])
                    if len(key) >= 3:
                        rec.setdefault('tag', key[-2] or '')
                elif isinstance(key, str) and '::' in key:
                    parts = key.split('::')
                    rec.setdefault('arm', parts[0])
                    last = parts[-1]
                    # Parse (do not scavenge) the seed suffix.  Taking every
                    # digit from the label turned `arm::seed-1` into seed 1
                    # and `arm::seed12x` into seed 12, silently inventing an
                    # identity that the key does not encode.
                    seed_text = last[4:] if last.startswith('seed') else None
                    try:
                        seed_val = (int(seed_text) if seed_text is not None
                                    else None)
                    except (TypeError, ValueError, OverflowError):
                        seed_val = None
                    if (seed_val is not None and seed_val >= 0
                            and str(seed_val) == seed_text):
                        rec.setdefault('seed', seed_val)
                    if len(parts) >= 3:
                        rec.setdefault('tag', '::'.join(parts[1:-1]))
            raw_items.append(rec)
    elif isinstance(records, (list, tuple)):
        raw_items = list(records)
    else:
        raise TypeError(
            f'records must be a list/tuple or an object mapping, got '
            f'{type(records).__name__}')
    normalized = {}
    violations = []
    warnings = []
    for i, raw in enumerate(raw_items):
        where = f'record[{i}]'
        rec, rec_viol, rec_warn = _endpoint_record(raw, where=where)
        violations.extend(rec_viol)
        warnings.extend(rec_warn)
        if rec is None:
            continue
        # A record without a valid arm or seed has no comparison identity.
        # Keep its violations (so the audit fails closed) but do not insert a
        # ``(None, ...)`` key into the panel: the default comparison builder and
        # the paired-seed sort would otherwise raise TypeError on mixed
        # malformed/valid records instead of reporting a failed audit.
        if rec['arm'] is None or rec['seed'] is None:
            continue
        key = (rec['arm'], rec['tag'], rec['seed'])
        label = (f'{rec["arm"]}::s{rec["seed"]}' if not rec['tag']
                 else f'{rec["arm"]}::{rec["tag"]}::s{rec["seed"]}')
        if key in normalized:
            violations.append(
                f'{label}: duplicate record for the same (arm, tag, seed)')
            continue
        normalized[key] = rec
    warnings = sorted(set(warnings))
    # Normalise the comparison list.
    if comparisons is None:
        tags = sorted({tag for (_arm, tag, _seed) in normalized})
        comps = []
        for tag in tags:
            arms = sorted({arm for (arm, t, _seed) in normalized if t == tag})
            for i, a in enumerate(arms):
                for b in arms[i + 1:]:
                    comps.append({'a': a, 'tag_a': tag, 'b': b, 'tag_b': tag,
                                  'name': _endpoint_comp_key(a, tag, b, tag)})
    else:
        comps = []
        for i, item in enumerate(list(comparisons)):
            where = f'comparison[{i}]'
            if isinstance(item, dict):
                a, b = item.get('a'), item.get('b')
                tag_a = item.get('tag_a', item.get('tag', '')) or ''
                tag_b = item.get('tag_b', item.get('tag', '')) or ''
                name = item.get('name')
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                a, b = item
                tag_a = tag_b = ''
                name = None
            elif isinstance(item, (list, tuple)) and len(item) == 4:
                a, tag_a, b, tag_b = item
                tag_a = tag_a or ''
                tag_b = tag_b or ''
                name = None
            else:
                violations.append(
                    f'{where}: expected an object or 2/4-tuple, got {item!r}')
                continue
            if not all(isinstance(x, str) and x.strip() for x in (a, b)):
                violations.append(f'{where}: a and b must be arm names')
                continue
            if not isinstance(tag_a, str) or not isinstance(tag_b, str):
                violations.append(f'{where}: tags must be strings')
                continue
            a, b, tag_a, tag_b = a.strip(), b.strip(), tag_a.strip(), tag_b.strip()
            if a == b and tag_a == tag_b:
                violations.append(f'{where}: an arm cannot be compared to itself')
                continue
            if name is not None and not isinstance(name, str):
                violations.append(f'{where}: name must be a string or None')
                continue
            comps.append({'a': a, 'tag_a': tag_a, 'b': b, 'tag_b': tag_b,
                          'name': name or _endpoint_comp_key(a, tag_a, b, tag_b)})
    # Manifest declarations.
    policies, policy_problems = _endpoint_policy_entries(manifest)
    if policy_problems:
        violations.extend(policy_problems)
    by_arm_tag = {}
    for rec in normalized.values():
        by_arm_tag.setdefault((rec['arm'], rec['tag']), {})[rec['seed']] = rec
    comp_reports = {}
    for comp in comps:
        name = comp['name']
        if name in comp_reports:
            violations.append(f'{name}: duplicate comparison name')
            continue
        map_a = by_arm_tag.get((comp['a'], comp['tag_a']))
        map_b = by_arm_tag.get((comp['b'], comp['tag_b']))
        comp_viol = []
        comp_warn = []
        if map_a is None:
            comp_viol.append(
                f'{name}: arm {comp["a"]!r} with tag {comp["tag_a"]!r} has no records')
        if map_b is None:
            comp_viol.append(
                f'{name}: arm {comp["b"]!r} with tag {comp["tag_b"]!r} has no records')
        if comp_viol:
            comp_reports[name] = {
                'a': comp['a'], 'tag_a': comp['tag_a'],
                'b': comp['b'], 'tag_b': comp['tag_b'],
                'status': 'fail', 'n_pairs': 0, 'seeds': [],
                'rule_a': None, 'rule_b': None, 'rule': None,
                'data_dependent': None, 'selection_split': None,
                'eval_split': None, 'warnings': [], 'violations': comp_viol,
            }
            violations.extend(comp_viol)
            continue
        shared = sorted(set(map_a) & set(map_b))
        measured = [s for s in shared
                    if map_a[s].get('ppl_usable') and map_b[s].get('ppl_usable')]
        # A comparison with no measured record is unverifiable, not a pass.
        rules_a = {map_a[s]['rule'] for s in measured}
        rules_b = {map_b[s]['rule'] for s in measured}
        rules = rules_a | rules_b
        if len(measured) < mn:
            comp_reports[name] = {
                'a': comp['a'], 'tag_a': comp['tag_a'],
                'b': comp['b'], 'tag_b': comp['tag_b'],
                'status': 'not_verifiable', 'n_pairs': len(measured),
                'seeds': measured, 'rule_a': None, 'rule_b': None,
                'rule': None, 'data_dependent': None,
                'selection_split': None, 'eval_split': None,
                'warnings': [], 'violations': [],
                'reason': (f'{len(measured)} measurable paired seed(s) < '
                           f'min_n={mn}'),
            }
            continue
        if len(rules_a) > 1 or len(rules_b) > 1:
            comp_viol.append(
                f'{name}: endpoint rule changes across seeds within an arm '
                f'(a={sorted(rules_a)}, b={sorted(rules_b)}); the panel is a '
                'mixture of estimands')
        if len(rules) > 1:
            comp_viol.append(
                f'{name}: the two arms use different endpoint rules '
                f'(a={sorted(rules_a)}, b={sorted(rules_b)}); a final step '
                'versus a selected best step is not a controlled comparison')
        rule = next(iter(rules)) if len(rules) == 1 else None
        data_dependent = (None if rule is None
                          else rule in _ENDPOINT_DATA_DEPENDENT)
        splits_sel = {map_a[s]['selection_split'] for s in measured} | {
            map_b[s]['selection_split'] for s in measured}
        splits_sel.discard(None)
        split_sel = next(iter(splits_sel)) if len(splits_sel) == 1 else None
        splits_eval = {map_a[s]['eval_split'] for s in measured} | {
            map_b[s]['eval_split'] for s in measured}
        splits_eval.discard(None)
        split_eval = next(iter(splits_eval)) if len(splits_eval) == 1 else None
        if len(splits_eval) > 1:
            comp_viol.append(
                f'{name}: the two arms do not share one evaluation split '
                f'({sorted(splits_eval)})')
        if len(splits_sel) > 1:
            comp_viol.append(
                f'{name}: the selection split changes within the comparison '
                f'({sorted(splits_sel)})')
        if rule is not None:
            if rule == 'unknown':
                comp_reports[name] = {
                    'a': comp['a'], 'tag_a': comp['tag_a'],
                    'b': comp['b'], 'tag_b': comp['tag_b'],
                    'status': 'not_verifiable', 'n_pairs': len(measured),
                    'seeds': measured, 'rule_a': None, 'rule_b': None,
                    'rule': 'unknown', 'data_dependent': None,
                    'selection_split': split_sel, 'eval_split': split_eval,
                    'warnings': [], 'violations': comp_viol,
                    'reason': ('the endpoint selection rule could not be '
                               'inferred from the recorded steps'),
                }
                violations.extend(comp_viol)
                continue
            if rule == 'truncated_step':
                comp_warn.append(
                    f'{name}: the run stopped before its planned budget '
                    'without an early-stop record; the endpoint is a '
                    'truncated budget, not the planned final step')
            if data_dependent:
                comp_warn.append(
                    f'{name}: {rule} is a data-dependent selection; report '
                    'the selection rule and the independent selection split, '
                    'and do not quote the value as a final-step measurement')
                if split_sel is None:
                    comp_warn.append(
                        f'{name}: data-dependent selection has no recorded '
                        'selection split, so independence from the evaluation '
                        'rows is unverifiable')
                if split_sel == 'test':
                    comp_viol.append(
                        f'{name}: the endpoint was selected on the test split')
                if (split_sel is not None and split_eval is not None
                        and split_sel == split_eval):
                    comp_viol.append(
                        f'{name}: selection and evaluation both use '
                        f'{split_sel} rows')
            if split_sel is None and split_eval is not None:
                comp_warn.append(
                    f'{name}: selection split is unrecorded while evaluation '
                    f'uses {split_eval} rows')
        # Equal-budget check on the paired seeds: a different number of
        # completed steps is a different training budget, even when the rule
        # name agrees.
        step_pairs = []
        for s in measured:
            ca = map_a[s].get('steps_completed')
            cb = map_b[s].get('steps_completed')
            if ca is not None and cb is not None:
                step_pairs.append((s, ca, cb))
        if step_pairs and any(ca != cb for _s, ca, cb in step_pairs):
            diffs = sorted({abs(ca - cb) for _s, ca, cb in step_pairs})
            comp_warn.append(
                f'{name}: paired seeds completed different step counts '
                f'(max difference {max(diffs)}); the comparison does not use '
                'one training budget')
        # Diagnostic: a negative final_minus_best means the final subset PPL
        # was better than the recorded best, which contradicts the label.
        fmbs = []
        for s in measured:
            for rec in (map_a[s], map_b[s]):
                v = rec.get('final_minus_best')
                if v is not None:
                    fmbs.append(float(v))
        if any(v < 0.0 for v in fmbs):
            comp_warn.append(
                f'{name}: a negative final_minus_best was recorded; the '
                '"best" label is inconsistent with the stored curve')
        ckpt_missing = sum(
            1 for s in measured for rec in (map_a[s], map_b[s])
            if not rec.get('checkpoint_sha256'))
        ckpt_any = any(
            rec.get('checkpoint_mode') is not None
            or rec.get('checkpoint_step') is not None
            or rec.get('checkpoint_sha256') is not None
            for s in measured for rec in (map_a[s], map_b[s]))
        if require_checkpoint_identity and ckpt_missing:
            comp_viol.append(
                f'{name}: {ckpt_missing} of {2 * len(measured)} records do '
                'not carry a checkpoint identity')
        elif ckpt_any and ckpt_missing:
            comp_warn.append(
                f'{name}: {ckpt_missing} of {2 * len(measured)} records do '
                'not carry a checkpoint identity while the rest do; the saved '
                'weight behind the number is only partially attested')
        # Apply a matching manifest declaration.
        matched = [p for p in policies if _endpoint_policy_matches(p, comp)]
        if len(matched) > 1:
            comp_viol.append(
                f'{name}: {len(matched)} manifest endpoint policies match '
                'this comparison; the declaration is ambiguous')
        elif matched:
            policy = matched[0]
            if policy['kind'] != rule:
                comp_viol.append(
                    f'{name}: manifest declares endpoint kind '
                    f'{policy["kind"]!r} but the records implement {rule!r}')
            if policy['selection_split'] is not None:
                if split_sel is None:
                    comp_viol.append(
                        f'{name}: manifest declares selection_split='
                        f'{policy["selection_split"]!r} but the records do '
                        'not record one')
                elif policy['selection_split'] != split_sel:
                    comp_viol.append(
                        f'{name}: manifest selection_split='
                        f'{policy["selection_split"]!r} disagrees with the '
                        f'recorded {split_sel!r}')
            if (require_preregistered
                    and policy['kind'] in _ENDPOINT_DATA_DEPENDENT
                    and policy['selection_split'] is None):
                comp_viol.append(
                    f'{name}: manifest declares the data-dependent kind '
                    f'{policy["kind"]!r} without a selection_split')
            if policy['require_checkpoint_identity'] and ckpt_missing:
                comp_viol.append(
                    f'{name}: manifest requires checkpoint identity but '
                    f'{ckpt_missing} record(s) lack it')
        elif require_preregistered:
            comp_viol.append(
                f'{name}: require_preregistered=True but no manifest endpoint '
                'policy covers this comparison')
        if comp_viol:
            status = 'fail'
        elif comp_warn:
            status = 'warn'
        else:
            status = 'ok'
        comp_reports[name] = {
            'a': comp['a'], 'tag_a': comp['tag_a'],
            'b': comp['b'], 'tag_b': comp['tag_b'],
            'status': status, 'n_pairs': len(measured), 'seeds': measured,
            'rule_a': next(iter(rules_a)) if len(rules_a) == 1 else None,
            'rule_b': next(iter(rules_b)) if len(rules_b) == 1 else None,
            'rule': rule,
            'data_dependent': data_dependent,
            'selection_split': split_sel, 'eval_split': split_eval,
            'n_checkpoint_identity_missing': ckpt_missing,
            'warnings': sorted(set(comp_warn)),
            'violations': sorted(set(comp_viol)),
        }
        violations.extend(comp_viol)
        warnings.extend(comp_warn)
    warnings = sorted(set(warnings))
    violations = sorted(set(violations))
    statuses = [c.get('status') for c in comp_reports.values()]
    if violations or 'fail' in statuses:
        status = 'fail'
    elif 'warn' in statuses:
        status = 'warn'
    elif not comp_reports or all(s == 'not_verifiable' for s in statuses):
        status = 'not_verifiable'
    else:
        status = 'ok'
    rule_counts = {}
    for rec in normalized.values():
        rule_counts[rec['rule']] = rule_counts.get(rec['rule'], 0) + 1
    out = {
        'kind': 'endpoint_selection_audit',
        'version': ENDPOINT_SELECTION_POLICY['version'],
        'status': status,
        'policy': dict(ENDPOINT_SELECTION_POLICY),
        'min_n': mn,
        'require_preregistered': require_preregistered,
        'require_checkpoint_identity': require_checkpoint_identity,
        'manifest_declarations': policies,
        'records': {
            (f'{r["arm"]}::s{r["seed"]}' if not r['tag']
             else f'{r["arm"]}::{r["tag"]}::s{r["seed"]}'): r
            for r in normalized.values()
        },
        'comparisons': comp_reports,
        'counts': {
            'n_records': len(normalized),
            'n_comparisons': len(comp_reports),
            'n_ok': sum(1 for s in statuses if s == 'ok'),
            'n_warn': sum(1 for s in statuses if s == 'warn'),
            'n_fail': sum(1 for s in statuses if s == 'fail'),
            'n_not_verifiable': sum(1 for s in statuses if s == 'not_verifiable'),
            'rule_counts': dict(sorted(rule_counts.items())),
            'n_data_dependent_records': sum(
                1 for r in normalized.values() if r['data_dependent']),
            'n_checkpoint_identity_missing': sum(
                1 for r in normalized.values() if not r['checkpoint_sha256']),
        },
        'warnings': warnings,
        'violations': violations,
        'role': ('endpoint/checkpoint selection provenance only; it adds no '
                 'p-value, no Holm family and no frozen decision, and it does '
                 'not prove that a final-step run trained correctly'),
    }
    return out
