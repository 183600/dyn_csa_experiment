"""Paired evaluation-trajectory stability for a reported endpoint contrast.

A final-step PPL contrast is a single snapshot of a training run.  Two arms can
cross near the end of training, or one arm can still be improving faster when
the run stops, so the final sign can describe a transient regime rather than a
persistent ordering.  The frozen endpoint is still the fixed final evaluation;
this readout only asks whether the *evaluation history* tells the same story,
and it never changes a p-value, Holm family or verdict.

Design choices
--------------
* **Balanced common-step grid.**  Only evaluation steps present for every
  included seed in BOTH arms are used.  Mixing per-seed grids would let a seed
  that evaluates more often dominate a step mean; dropped steps are reported.
* **Log-PPL differences.**  ``log(a) - log(b)`` is scale-free and equals the
  paired NLL difference; a multiplicative rescaling of both arms cannot change
  the sign, crossing count or slope.  Raw PPL deltas are reported alongside
  for readers who think in PPL points.
* **The tail is declared, not tuned.**  ``tail_points`` (default: the last 3
  common evaluations, final included) is fixed in the policy.  A final sign
  that disagrees with an earlier tail point is a late crossing; it is not
  relabelled as "stable" by choosing a shorter window after looking.
* **History is not the primary endpoint.**  Per-eval-step history is measured
  on the evaluation subset, while the final record uses the full fixed
  validation batch.  The function therefore compares the history's final sign
  with the primary delta and reports agreement/disagreement instead of
  assuming the two numbers are the same sample.

This is a descriptive stability diagnostic.  It adds no test, no confidence
interval and no decision rule; ``status`` is a summary of the trajectory shape.
"""


TRAJECTORY_POLICY = {
    'version': 'trajectory-stability-1',
    'estimand': ('paired log-PPL difference along the per-eval-step validation '
                 'history, aligned on the common step grid of every included '
                 'seed in both arms'),
    'tail_points': 3,
    'min_shared_steps': 2,
    'zero_tol_log': 1e-12,
    'slope_unit': 'log PPL per 1000 training steps',
    'role': ('descriptive endpoint/trajectory consistency readout; it does '
             'not add a test, change p/Holm/verified or replace the frozen '
             'final-step endpoint'),
}


def _traj_step_int(value):
    """An exact non-negative integer step, or ``None``."""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, str) or isinstance(value, (bytes, bytearray)):
        return None
    try:
        as_int = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        if float(value) != float(as_int):
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return as_int if as_int >= 0 else None


def _traj_history(raw, *, seed, label):
    """Parse one ``[(step, ppl), ...]`` history into a sorted step->PPL map."""
    if raw is None:
        return {}
    if isinstance(raw, (str, bytes, bytearray)) or isinstance(raw, dict):
        raise ValueError(f'{label} history for seed {seed} must be a list of '
                         f'[step, ppl] pairs')
    try:
        rows = list(raw)
    except TypeError:
        raise ValueError(f'{label} history for seed {seed} is not iterable') from None
    out = {}
    for i, point in enumerate(rows):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f'{label} history for seed {seed} row #{i} is not '
                             f'a [step, ppl] pair: {point!r}')
        step = _traj_step_int(point[0])
        if step is None:
            raise ValueError(f'{label} history for seed {seed} row #{i} has a '
                             f'non-integer/non-negative step: {point[0]!r}')
        if isinstance(point[1], (bool, np.bool_)):
            raise ValueError(f'{label} history for seed {seed} row #{i} has a '
                             'boolean PPL')
        if isinstance(point[1], (str, bytes, bytearray)):
            # A numeric-looking string is still a serialisation artefact, not
            # a measurement; the rest of the statistics layer rejects strings
            # through `_finite_scalar`, and the step parser above does too.
            raise ValueError(f'{label} history for seed {seed} row #{i} has a '
                             'non-numeric PPL')
        try:
            ppl = float(point[1])
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'{label} history for seed {seed} row #{i} has a '
                             'non-numeric PPL') from None
        if not math.isfinite(ppl) or ppl <= 0.0:
            raise ValueError(f'{label} history for seed {seed} row #{i} has a '
                             f'non-finite/non-positive PPL: {point[1]!r}')
        if step in out:
            raise ValueError(f'{label} history for seed {seed} repeats step '
                             f'{step}; a duplicated evaluation is ambiguous')
        out[step] = ppl
    return out


def _traj_slope(points, *, unit=1000.0):
    """OLS slope of y on step, per ``unit`` steps; ``None`` when undefined."""
    if len(points) < 2:
        return None
    xs = np.asarray([float(p[0]) for p in points], dtype=float)
    ys = np.asarray([float(p[1]) for p in points], dtype=float)
    xm = float(np.mean(xs))
    ym = float(np.mean(ys))
    dx = xs - xm
    den = float(np.sum(dx * dx))
    if not math.isfinite(den) or den <= 0.0:
        return None
    slope = float(np.sum(dx * (ys - ym)) / den) * float(unit)
    return slope if math.isfinite(slope) else None


def _traj_sign(value, zero_tol):
    value = _finite_scalar(value)
    if value is None:
        return None
    if abs(value) <= abs(float(zero_tol)):
        return 0
    return 1 if value > 0.0 else -1


def paired_trajectory_stability(history_a, history_b, *, seeds=None,
                                primary_delta=None, endpoint_step=None,
                                tail_points=None, min_shared_steps=None,
                                zero_tol=None):
    """Summarise whether a paired evaluation history supports the final sign.

    Parameters
    ----------
    history_a, history_b : mapping
        ``{seed: [(step, ppl), ...]}`` for the treatment / reference arm.
        Keys must be integer seeds.  A seed absent from either side is named
        and skipped (the remaining seeds must still form a common grid); an
        empty/missing history for every seed returns ``insufficient_history``.
    seeds : iterable of int, optional
        Restrict the analysis to this seed list (default: the union of the
        two mappings, so a seed present on only one side is reported as
        skipped instead of silently ignored).  A requested seed without a
        history on both sides is reported as skipped, not dropped.
    primary_delta : float, optional
        The reported paired endpoint difference (``with - without``) on the
        primary estimand.  Used only to check whether the history's final sign
        agrees with the number the report actually quotes.
    endpoint_step : int, optional
        The training step at which the primary endpoint was measured.  When
        the balanced history ends before it, the trajectory does not cover the
        endpoint; this is reported as ``final_step_is_endpoint=False`` with a
        warning, never silently treated as the endpoint history.
    tail_points : int, optional
        Size of the declared tail window, final point included.  Default 3.
    min_shared_steps : int, optional
        Minimum common evaluation steps needed to say anything.  Default 2.
    zero_tol : float, optional
        Absolute log-PPL tolerance for calling a step "zero".  Default 1e-12.

    Returns
    -------
    dict
        ``status`` is ``stable`` when every non-zero tail sign equals the
        final sign, ``late_crossing`` when the final sign differs from an
        earlier tail point, ``final_point_only`` when the pre-final tail is
        exactly zero, ``zero_final`` when the final log difference is zero,
        and ``insufficient_history`` when the balanced grid is too short.
        Malformed histories raise ``ValueError`` (the caller may record an
        ``invalid`` block); this function never guesses a missing point.
    """
    if not isinstance(history_a, dict) or not isinstance(history_b, dict):
        raise ValueError('history_a and history_b must be seed -> history mappings')
    if tail_points is None:
        tail_points = TRAJECTORY_POLICY['tail_points']
    if min_shared_steps is None:
        min_shared_steps = TRAJECTORY_POLICY['min_shared_steps']
    if zero_tol is None:
        zero_tol = TRAJECTORY_POLICY['zero_tol_log']
    _tail = _as_seed_int(tail_points)
    _min_steps = _as_seed_int(min_shared_steps)
    if _tail is None or _tail < 2:
        raise ValueError(f'tail_points must be an integer >= 2, got {tail_points!r}')
    if _min_steps is None or _min_steps < 2:
        raise ValueError(f'min_shared_steps must be an integer >= 2, got {min_shared_steps!r}')
    _zero = _finite_scalar(zero_tol)
    if _zero is None or _zero < 0.0:
        raise ValueError(f'zero_tol must be a finite non-negative number, got {zero_tol!r}')
    _primary = None
    if primary_delta is not None:
        _primary = _finite_scalar(primary_delta)
        if _primary is None:
            raise ValueError(f'primary_delta must be a finite number or None, got {primary_delta!r}')
    _endpoint = None
    if endpoint_step is not None:
        _endpoint = _traj_step_int(endpoint_step)
        if _endpoint is None:
            raise ValueError(f'endpoint_step must be a non-negative integer or '
                             f'None, got {endpoint_step!r}')

    # Record the EFFECTIVE policy, not only the module defaults.  The audit
    # gate replays the stored series with `policy.tail_points` /
    # `policy.zero_tol_log`; leaving the defaults here made every caller that
    # legitimately overrode tail_points or zero_tol look like a tampered
    # artifact (and mis-stated the convention in the report).
    _policy = dict(TRAJECTORY_POLICY)
    _policy['tail_points'] = int(_tail)
    _policy['min_shared_steps'] = int(_min_steps)
    _policy['zero_tol_log'] = float(_zero)

    def _normalise(mapping, label):
        out = {}
        for key, value in mapping.items():
            seed = _as_seed_int(key)
            if seed is None:
                raise ValueError(f'{label} history keys must be integer seeds, '
                                 f'got {key!r}')
            if seed in out:
                raise ValueError(f'{label} history has two keys for seed {seed}')
            out[seed] = value
        return out

    hist_a = _normalise(history_a, 'with')
    hist_b = _normalise(history_b, 'without')
    keys_a, keys_b = set(hist_a), set(hist_b)
    if seeds is not None:
        if isinstance(seeds, (str, bytes, bytearray)):
            raise ValueError('seeds must be an iterable of integer seeds')
        requested = []
        for item in seeds:
            seed = _as_seed_int(item)
            if seed is None:
                raise ValueError(f'seeds contains a non-integer seed: {item!r}')
            requested.append(seed)
        if len(set(requested)) != len(requested):
            raise ValueError(f'seeds contains duplicates: {requested!r}')
        candidates = sorted(requested)
    else:
        candidates = sorted(keys_a | keys_b)

    included, skipped, raw_a, raw_b = [], [], {}, {}
    for seed in candidates:
        if seed not in keys_a or seed not in keys_b:
            skipped.append({'seed': seed, 'reason': 'no history on both arms'})
            continue
        ha = _traj_history(hist_a[seed], seed=seed, label='with')
        hb = _traj_history(hist_b[seed], seed=seed, label='without')
        common = sorted(set(ha) & set(hb))
        if not common:
            skipped.append({'seed': seed, 'reason': 'no common evaluation step'})
            continue
        raw_a[seed], raw_b[seed] = ha, hb
        included.append(seed)

    out = {
        'kind': 'paired_trajectory_stability',
        'policy': _policy,
        'status': 'insufficient_history',
        'seeds': included,
        'skipped_seeds': skipped,
        'n_seeds_requested': len(candidates),
        'n_seeds_included': len(included),
        'primary_delta_ppl': _primary,
        'primary_sign': _traj_sign(_primary, _zero) if _primary is not None else None,
        'primary_vs_history_sign': None,
        'primary_vs_history_log_sign': None,
        'final_delta_ppl_sign': None,
        'scale_sensitive_final': None,
        'endpoint_step': _endpoint,
        'final_step_is_endpoint': None,
        'shared_steps': [],
        'n_shared_steps': 0,
        'steps_dropped_unbalanced': [],
        'series': [],
        'final_step': None,
        'final_mean_log_ratio': None,
        'final_mean_delta_ppl': None,
        'final_sign': None,
        'tail_points': int(_tail),
        'tail_steps': [],
        'tail_sign_stable': None,
        'n_sign_changes': None,
        'last_sign_change_step': None,
        'n_seeds_final_sign_matches': None,
        'n_seeds_tail_stable': None,
        'per_seed_tail': [],
        'tail_slope_log_per_1k_with': None,
        'tail_slope_log_per_1k_without': None,
        'tail_gap_slope_log_per_1k': None,
        'gap_trend': None,
        'warnings': [],
    }
    if not included:
        out['warnings'].append(
            'no seed has a paired evaluation history on both arms; the '
            'endpoint trajectory cannot be checked (this is a missing '
            'diagnostic, not evidence that the final endpoint is stable)')
        return out

    # Balanced grid across every included seed in both arms.  A step present
    # for only some seeds is dropped rather than averaged on an unequal n.
    per_seed_common = {s: sorted(set(raw_a[s]) & set(raw_b[s])) for s in included}
    common = sorted(set.intersection(*(set(v) for v in per_seed_common.values())))
    all_steps = sorted(set().union(*(set(v) for v in per_seed_common.values())))
    out['shared_steps'] = common
    out['n_shared_steps'] = len(common)
    out['steps_dropped_unbalanced'] = [s for s in all_steps if s not in set(common)]
    if out['steps_dropped_unbalanced']:
        out['warnings'].append(
            '%d evaluation step(s) are not present for every included seed in '
            'both arms and were dropped from the balanced trajectory grid'
            % len(out['steps_dropped_unbalanced']))
    if skipped:
        out['warnings'].append(
            '%d requested seed(s) were skipped for lack of a paired history '
            '(%s); the trajectory readout is conditional on the remaining %d'
            % (len(skipped), ', '.join(str(x['seed']) for x in skipped),
               len(included)))
    if len(common) < _min_steps:
        out['warnings'].append(
            'only %d common evaluation step(s) (minimum %d); the endpoint '
            'trajectory is not identifiable' % (len(common), _min_steps))
        return out

    series = []
    for step in common:
        deltas = [raw_a[s][step] - raw_b[s][step] for s in included]
        log_ratios = [math.log(raw_a[s][step]) - math.log(raw_b[s][step])
                      for s in included]
        mean_log = _scaled_mean_or_raw(log_ratios)
        sign = _traj_sign(mean_log, _zero)
        series.append({
            'step': int(step),
            'n_seeds': len(included),
            'mean_log_ratio': mean_log,
            'mean_delta_ppl': _scaled_mean_or_raw(deltas),
            'sign': sign,
            'frac_same_sign': (sum(1 for x in log_ratios
                                   if _traj_sign(x, _zero) == sign) / len(included)
                               if sign not in (None, 0) else None),
        })
    out['series'] = series
    final = series[-1]
    out['final_step'] = final['step']
    out['final_mean_log_ratio'] = final['mean_log_ratio']
    out['final_mean_delta_ppl'] = final['mean_delta_ppl']
    out['final_sign'] = final['sign']
    out['final_delta_ppl_sign'] = _traj_sign(final['mean_delta_ppl'], _zero)
    out['scale_sensitive_final'] = bool(
        out['final_sign'] not in (None, 0)
        and out['final_delta_ppl_sign'] not in (None, 0)
        and out['final_sign'] != out['final_delta_ppl_sign'])
    if out['scale_sensitive_final']:
        out['warnings'].append(
            'the final mean log-PPL ratio (%+d) and the final mean raw PPL '
            'difference (%+d) have opposite signs; the effect is scale-'
            'sensitive across seeds, so a sign claim must name the estimand'
            % (out['final_sign'], out['final_delta_ppl_sign']))
    if _endpoint is not None:
        out['final_step_is_endpoint'] = bool(final['step'] == _endpoint)
        if not out['final_step_is_endpoint']:
            out['warnings'].append(
                'the balanced history ends at step %s, but the primary '
                'endpoint was measured at step %s; the trajectory does not '
                'cover the reported endpoint'
                % (final['step'], _endpoint))

    # All non-zero sign changes over the full shared grid.
    changes = []
    last_sign = None
    for row in series:
        if row['sign'] in (None, 0):
            continue
        if last_sign is not None and row['sign'] != last_sign:
            changes.append(row['step'])
        last_sign = row['sign']
    out['n_sign_changes'] = len(changes)
    out['last_sign_change_step'] = changes[-1] if changes else None

    tail = series[-min(int(_tail), len(series)):]
    out['tail_steps'] = [row['step'] for row in tail]
    final_sign = final['sign']
    if final_sign == 0:
        out['status'] = 'zero_final'
        out['tail_sign_stable'] = False
        out['warnings'].append(
            'the final history step has a zero log-PPL difference (within '
            'zero_tol); no trajectory direction is defined')
    else:
        opposite = [row['step'] for row in tail[:-1]
                    if row['sign'] not in (None, 0, final_sign)]
        nonzero_earlier = [row['step'] for row in tail[:-1]
                           if row['sign'] not in (None, 0)]
        if opposite:
            out['status'] = 'late_crossing'
            out['tail_sign_stable'] = False
            out['warnings'].append(
                'the final history sign (%+d) differs from an earlier point '
                'in the declared %d-point tail (steps %s); the final ordering '
                'is a late crossing, not a persistent regime'
                % (final_sign, len(tail), ', '.join(map(str, opposite))))
        elif not nonzero_earlier:
            out['status'] = 'final_point_only'
            out['tail_sign_stable'] = False
            out['warnings'].append(
                'every pre-final point in the declared tail is zero; only the '
                'final point carries a sign, so the trajectory gives no '
                'independent support')
        else:
            out['status'] = 'stable'
            out['tail_sign_stable'] = True

    # Per-seed consistency: does each seed's own final sign agree with the
    # panel sign, and is each seed's tail internally stable?
    matches = 0
    tail_stable = 0
    for seed in included:
        own = [_traj_sign(math.log(raw_a[seed][s]) - math.log(raw_b[seed][s]), _zero)
               for s in out['tail_steps']]
        out['per_seed_tail'].append({
            'seed': int(seed),
            'tail_signs': list(own),
            'final_sign': own[-1],
        })
        own_final = own[-1]
        if final_sign not in (None, 0) and own_final == final_sign:
            matches += 1
        nz = [x for x in own if x not in (None, 0)]
        if nz and all(x == nz[-1] for x in nz):
            tail_stable += 1
    out['n_seeds_final_sign_matches'] = matches
    out['n_seeds_tail_stable'] = tail_stable
    if final_sign not in (None, 0) and matches * 2 <= len(included):
        out['warnings'].append(
            'the panel final sign is carried by at most half of the included '
            'seeds (%d/%d); report the per-seed trajectory, not only the panel '
            'mean' % (matches, len(included)))

    # Tail slopes (log PPL per 1000 steps) and the gap trend.  A shrinking
    # final gap is a warning about persistence, not a new hypothesis test.
    tail_with = [(row['step'],
                  _scaled_mean_or_raw([math.log(raw_a[s][row['step']])
                                       for s in included]))
                 for row in tail]
    tail_without = [(row['step'],
                     _scaled_mean_or_raw([math.log(raw_b[s][row['step']])
                                          for s in included]))
                    for row in tail]
    out['tail_slope_log_per_1k_with'] = _traj_slope(tail_with)
    out['tail_slope_log_per_1k_without'] = _traj_slope(tail_without)
    gap_slope = _traj_slope([(row['step'], row['mean_log_ratio']) for row in tail])
    out['tail_gap_slope_log_per_1k'] = gap_slope
    if gap_slope is None or final_sign in (None, 0):
        out['gap_trend'] = None
    elif _traj_sign(gap_slope, _zero) == 0:
        out['gap_trend'] = 'flat'
    elif _traj_sign(gap_slope, _zero) == final_sign:
        out['gap_trend'] = 'widening'
    else:
        out['gap_trend'] = 'narrowing'
        out['warnings'].append(
            'the history-series gap is narrowing over the declared tail '
            '(%.6g log-PPL per 1000 steps against a final gap of %.6g); the '
            'final gap may not persist, so do not describe it as a settled '
            'ordering' % (gap_slope, final['mean_log_ratio']))

    if out['primary_sign'] is not None:
        if out['final_delta_ppl_sign'] not in (None, 0):
            out['primary_vs_history_sign'] = bool(
                out['primary_sign'] == out['final_delta_ppl_sign'])
        if final_sign not in (None, 0):
            out['primary_vs_history_log_sign'] = bool(
                out['primary_sign'] == final_sign)
        if out['primary_vs_history_sign'] is False:
            out['warnings'].append(
                'the history-series final raw-PPL sign (%+d) disagrees with '
                'the primary endpoint sign (%+d); per-eval-step history is '
                'measured on the evaluation subset while the primary estimate '
                'uses the full fixed validation rows, so inspect which sample '
                'drives the difference before treating the trajectory as '
                'corroboration'
                % (out['final_delta_ppl_sign'], out['primary_sign']))
    return out
