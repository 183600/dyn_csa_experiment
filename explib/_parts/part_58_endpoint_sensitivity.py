"""Design-stage endpoint sensitivity: headroom, resolution and minimum effect.

A pre-registered endpoint can fail to answer the research question for two
independent reasons:

1. **It is underpowered** -- the planned number of seeds/units cannot separate
   the effect from noise.  ``paired_design_sensitivity`` and
   ``unit_precision_design`` already cover that axis.
2. **It has no dynamic range** -- the endpoint is bounded (accuracy, F1,
   recall, ...) and the control arm already sits at (or next to) the bound in
   the beneficial direction.  Then even an infinite-sample experiment cannot
   show an improvement, and a non-significant result is uninformative rather
   than evidence of "no benefit".  For an unbounded endpoint (PPL, NLL) the
   analogous binding limit is the *measurement resolution*: the run-to-run
   spread of repeated evaluations of the identical checkpoint.

This module makes the second axis a design-time, machine-checkable property.
It never invents a margin: a ``minimum_meaningful_effect`` (MEE, the smallest
effect the study would care about) must be declared by the caller/manifest.
Without one, only the structural facts (bounds, direction, exhausted headroom,
measurement resolution) are reported.

The functions are pure/zero-GPU.  They read a *pilot/validation* level and
optional repeated measurements, never ``WikiText-103`` test rows, and they do
not change any p-value, Holm family, endpoint or frozen verdict.
"""

# Metric registry.  ``bounds`` are ``[lower, upper]`` with ``None`` for an
# open side; ``direction`` is the beneficial direction for the study.  Only
# metrics whose scale is unambiguous in this project are listed.  An unknown
# endpoint can still be audited by supplying ``metric``/``bounds``/
# ``direction`` explicitly -- the registry is a default, not a hard gate.
ENDPOINT_SENSITIVITY_POLICY = {
    'version': 1,
    'near_bound_tolerance': 0.02,
    'min_relative_headroom': 0.05,
    'metrics': {
        # token-level language-modelling units: positive and unbounded above
        'ppl': {'family': 'positive', 'bounds': [0.0, None],
                'direction': 'lower_is_better'},
        'perplexity': {'family': 'positive', 'bounds': [0.0, None],
                       'direction': 'lower_is_better'},
        'nll': {'family': 'positive', 'bounds': [0.0, None],
                'direction': 'lower_is_better'},
        'answer_nll': {'family': 'positive', 'bounds': [0.0, None],
                       'direction': 'lower_is_better'},
        'cross_entropy': {'family': 'positive', 'bounds': [0.0, None],
                          'direction': 'lower_is_better'},
        # bounded 0..1 quality metrics
        'accuracy': {'family': 'proportion', 'bounds': [0.0, 1.0],
                     'direction': 'higher_is_better'},
        'top1_accuracy': {'family': 'proportion', 'bounds': [0.0, 1.0],
                          'direction': 'higher_is_better'},
        'balanced_accuracy': {'family': 'proportion', 'bounds': [0.0, 1.0],
                              'direction': 'higher_is_better'},
        'macro_recall': {'family': 'proportion', 'bounds': [0.0, 1.0],
                         'direction': 'higher_is_better'},
        'recall': {'family': 'proportion', 'bounds': [0.0, 1.0],
                   'direction': 'higher_is_better'},
        'precision': {'family': 'proportion', 'bounds': [0.0, 1.0],
                      'direction': 'higher_is_better'},
        'f1': {'family': 'proportion', 'bounds': [0.0, 1.0],
               'direction': 'higher_is_better'},
        'f1_macro': {'family': 'proportion', 'bounds': [0.0, 1.0],
                     'direction': 'higher_is_better'},
        'f1_micro': {'family': 'proportion', 'bounds': [0.0, 1.0],
                     'direction': 'higher_is_better'},
        'f1_weighted': {'family': 'proportion', 'bounds': [0.0, 1.0],
                        'direction': 'higher_is_better'},
        'boundary_f1': {'family': 'proportion', 'bounds': [0.0, 1.0],
                        'direction': 'higher_is_better'},
        'auroc': {'family': 'proportion', 'bounds': [0.0, 1.0],
                  'direction': 'higher_is_better'},
        'roc_auc': {'family': 'proportion', 'bounds': [0.0, 1.0],
                    'direction': 'higher_is_better'},
        'pr_auc': {'family': 'proportion', 'bounds': [0.0, 1.0],
                   'direction': 'higher_is_better'},
    },
    'status_values': ['ok', 'low_headroom', 'saturated', 'uninformative',
                      'not_verifiable', 'not_estimable', 'invalid'],
    'rule': ('A beneficial-direction claim needs (i) non-zero headroom to the '
             'bound, (ii) headroom >= the declared minimum meaningful effect, '
             '(iii) a measurement resolution smaller than that effect and '
             '(iv) an MDE no larger than it.  Missing pilot data or a missing '
             'margin is not a pass; it is reported as not_verifiable.'),
    'note': ('bounds/direction are scale conventions, not estimated from the '
             'data; the minimum meaningful effect is always caller-declared'),
}

_DEFAULT_NEAR_BOUND_TOLERANCE = 0.02
_DEFAULT_MIN_RELATIVE_HEADROOM = 0.05


def _es_float(value):
    """Finite float or ``None`` (booleans are not numbers here)."""
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _es_int(value, *, minimum=None):
    """Exact integer or ``None``; rejects booleans/non-integral floats."""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, numbers.Integral):
        out = int(value)
    else:
        f = _es_float(value)
        if f is None or not f.is_integer():
            return None
        out = int(f)
    if minimum is not None and out < minimum:
        return None
    return out


def _es_probability(value):
    f = _es_float(value)
    return f if (f is not None and 0.0 < f < 1.0) else None


def _es_metric_spec(metric):
    """Copy of one registry entry, or ``None`` for an unknown metric."""
    if not isinstance(metric, str):
        return None
    spec = ENDPOINT_SENSITIVITY_POLICY['metrics'].get(metric)
    if not isinstance(spec, dict):
        return None
    return {'family': spec.get('family'),
            'bounds': list(spec.get('bounds') or [None, None]),
            'direction': spec.get('direction')}


def _es_normalise_bounds(bounds):
    """``[lower, upper]`` with finite-or-None sides, or ``(None, error)``."""
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return None, 'bounds must be a two-element [lower, upper] list'
    lo = None if bounds[0] is None else _es_float(bounds[0])
    hi = None if bounds[1] is None else _es_float(bounds[1])
    if bounds[0] is not None and lo is None:
        return None, f'bounds[0]={bounds[0]!r} is not finite'
    if bounds[1] is not None and hi is None:
        return None, f'bounds[1]={bounds[1]!r} is not finite'
    if lo is not None and hi is not None and not (hi > lo):
        return None, f'bounds must satisfy lower < upper, got {bounds!r}'
    if lo is None and hi is None:
        return [None, None], None
    return [lo, hi], None


def _es_normalise_direction(direction):
    if direction is None:
        return None, None
    if isinstance(direction, (bool, np.bool_)):
        return None, f'direction={direction!r} must not be a boolean'
    text = str(direction).strip().lower()
    if text in ('higher_is_better', 'higher', 'up', '+', '+1', '1'):
        return 'higher_is_better', None
    if text in ('lower_is_better', 'lower', 'down', '-', '-1'):
        return 'lower_is_better', None
    return None, (f'direction={direction!r} must be '
                  "'higher_is_better' or 'lower_is_better'")


def endpoint_metric(endpoint):
    """Resolve an endpoint string such as ``by_len.ppl@4096`` to metric metadata.

    The metric token must be delimited by a non-identifier character (or the
    string boundary), and the longest registry key wins.  That keeps
    ``boundary_f1`` from being read as ``f1`` and
    ``niah.top1_accuracy`` from being read as ``accuracy``.  An unknown
    endpoint is ``not_estimable`` -- it is never silently treated as
    unbounded.
    """
    out = {'kind': 'endpoint_metric', 'endpoint': None, 'metric': None,
           'family': None, 'bounds': None, 'direction': None,
           'bounded_beneficial': None, 'status': 'not_estimable',
           'reason': None}
    if not isinstance(endpoint, str) or not endpoint.strip():
        out['reason'] = 'endpoint must be a non-empty string'
        return out
    text = endpoint.strip()
    out['endpoint'] = text
    stem = text.split('@', 1)[0].strip().lower()
    for key in sorted(ENDPOINT_SENSITIVITY_POLICY['metrics'],
                      key=len, reverse=True):
        if re.search(r'(?<![a-z0-9_])' + re.escape(key) + r'(?![a-z0-9_])',
                     stem):
            spec = _es_metric_spec(key)
            out.update({'metric': key, 'family': spec['family'],
                        'bounds': list(spec['bounds']),
                        'direction': spec['direction'],
                        'bounded_beneficial': bool(
                            spec['bounds'][1] is not None
                            if spec['direction'] == 'higher_is_better'
                            else spec['bounds'][0] is not None),
                        'status': 'ok'})
            return out
    out['reason'] = (
        f'endpoint {text!r} does not name a metric in '
        'ENDPOINT_SENSITIVITY_POLICY; supply metric/bounds/direction '
        'explicitly')
    return out


def _es_direction_from(meta, direction):
    if direction is not None:
        return _es_normalise_direction(direction)
    d = meta.get('direction') if isinstance(meta, dict) else None
    return _es_normalise_direction(d)


def _es_bounds_from(meta, bounds):
    if bounds is not None:
        return _es_normalise_bounds(bounds)
    b = meta.get('bounds') if isinstance(meta, dict) else None
    if b is None:
        return [None, None], None
    return _es_normalise_bounds(b)


def endpoint_headroom(endpoint=None, control_level=None, *, arm_levels=None,
                      metric=None, direction=None, bounds=None,
                      minimum_meaningful_effect=None,
                      near_bound_tolerance=None,
                      min_relative_headroom=None):
    """Distance from the best observed arm to the beneficial endpoint bound.

    ``control_level`` is one operating level (usually the baseline arm's
    validation value) and, when supplied, is the reference.  ``arm_levels``
    can instead supply both arms; then the *best* level is used as the
    reference, because the maximum additional improvement the comparison could
    possibly measure is the distance from the already-better arm to the bound.
    Using the better arm is the conservative choice when the baseline is not
    named explicitly.

    ``near_bound_tolerance`` and ``min_relative_headroom`` are fractions of the
    bounded span (absolute when the span is open-ended).  A zero headroom is
    ``saturated`` regardless of any declared margin; a non-zero headroom below
    the declared ``minimum_meaningful_effect`` is ``uninformative``.  Without
    a declared margin, a small relative headroom is only ``low_headroom``.
    """
    meta = endpoint_metric(endpoint) if endpoint is not None else {}
    if not isinstance(meta, dict):
        meta = {}
    chosen_metric = metric if metric is not None else meta.get('metric')
    spec = _es_metric_spec(chosen_metric) or {}
    dir_text, dir_err = _es_direction_from(
        {'direction': spec.get('direction') or meta.get('direction')},
        direction)
    bounds_pair, bounds_err = _es_bounds_from(
        {'bounds': spec.get('bounds') or meta.get('bounds')}, bounds)
    near_tol = (_DEFAULT_NEAR_BOUND_TOLERANCE
                if near_bound_tolerance is None else _es_float(near_bound_tolerance))
    min_rel = (_DEFAULT_MIN_RELATIVE_HEADROOM
               if min_relative_headroom is None else _es_float(min_relative_headroom))
    mee = (None if minimum_meaningful_effect is None
           else _es_float(minimum_meaningful_effect))

    problems, warnings = [], []
    if dir_err:
        problems.append(dir_err)
    if bounds_err:
        problems.append(bounds_err)
    if near_tol is None or not (0.0 <= near_tol < 1.0):
        problems.append(
            f'near_bound_tolerance={near_bound_tolerance!r} must be a '
            'finite fraction in [0,1)')
    if min_rel is None or not (0.0 < min_rel <= 1.0):
        problems.append(
            f'min_relative_headroom={min_relative_headroom!r} must be a '
            'finite fraction in (0,1]')
    if minimum_meaningful_effect is not None and (mee is None or mee <= 0.0):
        problems.append(
            f'minimum_meaningful_effect={minimum_meaningful_effect!r} must be '
            'a finite positive number')

    levels = {}
    if arm_levels is not None:
        if not isinstance(arm_levels, dict):
            problems.append(
                f'arm_levels must be an object, got {type(arm_levels).__name__}')
        else:
            for name, value in arm_levels.items():
                f = _es_float(value)
                if f is None:
                    warnings.append(
                        f'arm_levels[{name!r}]={value!r} is not a finite '
                        'number and was dropped')
                else:
                    levels[str(name)] = f
    if control_level is not None:
        f = _es_float(control_level)
        if f is None:
            problems.append(
                f'control_level={control_level!r} is not a finite number')
        else:
            levels['control'] = f

    out = {'kind': 'endpoint_headroom', 'status': 'not_verifiable',
           'endpoint': (endpoint if isinstance(endpoint, str) else None),
           'metric': chosen_metric if isinstance(chosen_metric, str) else None,
           'family': spec.get('family') or meta.get('family'),
           'direction': dir_text,
           'bounds': (list(bounds_pair) if bounds_pair is not None else None),
           'control_level': (levels.get('control') if 'control' in levels else None),
           'arm_levels': dict(levels),
           'reference_level': None, 'reference_arm': None,
           'headroom': None, 'headroom_relative': None,
           'opposite_headroom': None, 'span': None,
           'bounded_beneficial': None, 'exhausted': None,
           'near_bound': None, 'structurally_possible': None,
           'minimum_meaningful_effect': mee,
           'headroom_covers_mee': None,
           'problems': problems, 'warnings': warnings,
           'policy': {'near_bound_tolerance': near_tol,
                      'min_relative_headroom': min_rel,
                      'rule': ENDPOINT_SENSITIVITY_POLICY['rule']}}

    if problems:
        out['status'] = 'invalid'
        return out
    if chosen_metric is None or dir_text is None or bounds_pair is None:
        out['status'] = 'not_estimable'
        out['warnings'].append(
            'metric/direction/bounds are incomplete; endpoint sensitivity '
            'cannot be classified')
        return out
    lo, hi = bounds_pair
    out['span'] = (float(hi - lo) if (lo is not None and hi is not None)
                   else None)
    if not levels:
        out['status'] = 'not_verifiable'
        out['warnings'].append(
            'no finite control_level/arm_levels was supplied; headroom is '
            'not verifiable from the pilot')
        return out

    higher = (dir_text == 'higher_is_better')
    check_tol = 1e-9 * max(1.0, abs(lo or 0.0), abs(hi or 0.0))
    for name, value in levels.items():
        if lo is not None and value < lo - check_tol:
            problems.append(
                f'level {name}={value!r} lies below the declared lower bound {lo}')
        if hi is not None and value > hi + check_tol:
            problems.append(
                f'level {name}={value!r} lies above the declared upper bound {hi}')
    if problems:
        out['status'] = 'invalid'
        return out
    # Clamp only the numerical epsilon, never a real out-of-range value.
    for name in list(levels):
        if lo is not None and levels[name] < lo:
            levels[name] = float(lo)
        if hi is not None and levels[name] > hi:
            levels[name] = float(hi)

    if control_level is not None and 'control' in levels:
        # An explicitly named control/baseline is the reference: the ceiling
        # question is "can the treatment exceed THIS arm?", not "which arm is
        # already best?".
        ref_name = 'control'
    else:
        ref_name = (max(levels, key=lambda k: (levels[k], k)) if higher
                    else min(levels, key=lambda k: (levels[k], k)))
    ref = float(levels[ref_name])
    out['reference_level'] = ref
    out['reference_arm'] = ref_name

    beneficial = hi if higher else lo
    opposite = lo if higher else hi
    out['bounded_beneficial'] = beneficial is not None
    if opposite is not None:
        out['opposite_headroom'] = float((ref - opposite) if higher
                                         else (opposite - ref))
    if beneficial is None:
        out['status'] = 'ok'
        out['structurally_possible'] = True
        out['exhausted'] = False
        out['near_bound'] = False
        out['warnings'].append(
            'the beneficial direction is open-ended; the bound does not '
            'limit the measurable improvement (measurement resolution still '
            'does)')
        return out

    headroom = float(max(0.0, (beneficial - ref) if higher else (ref - beneficial)))
    out['headroom'] = headroom
    if out['span'] and out['span'] > 0:
        out['headroom_relative'] = float(headroom / out['span'])
    out['exhausted'] = bool(headroom <= 1e-12)
    out['near_bound'] = bool(
        out['exhausted'] or (out['headroom_relative'] is not None
                             and out['headroom_relative'] <= near_tol))
    low_headroom = bool(
        out['headroom_relative'] is not None
        and out['headroom_relative'] <= min_rel)
    out['structurally_possible'] = (not out['exhausted'])
    if mee is not None:
        out['headroom_covers_mee'] = bool(headroom + 1e-12 >= mee)
    if out['exhausted']:
        out['status'] = 'saturated'
        out['warnings'].append(
            f'the best observed level {ref!r} is at the beneficial bound '
            f'{beneficial!r}; no improvement can be measured on this endpoint')
    elif mee is not None and not out['headroom_covers_mee']:
        out['status'] = 'uninformative'
        out['warnings'].append(
            f'available headroom {headroom:.6g} is smaller than the declared '
            f'minimum meaningful effect {mee:.6g}; the endpoint cannot show a '
            'worthwhile improvement even with unlimited seeds')
    elif low_headroom:
        out['status'] = 'low_headroom'
        out['warnings'].append(
            f'headroom {headroom:.6g} is at most {min_rel:.3g} of the bounded '
            'span; a ceiling/floor effect may hide a real difference')
    else:
        out['status'] = 'ok'
    if (mee is not None and out['headroom_covers_mee'] and low_headroom):
        out['warnings'].append(
            'headroom covers the declared margin but is a small fraction of '
            'the bounded span; report the distance to the bound with the '
            'effect')
    return out


def endpoint_measurement_resolution(repeat_values=None, *, endpoint=None,
                                    minimum_meaningful_effect=None,
                                    max_relative_resolution=None):
    """Run-to-run resolution from repeated evaluations of ONE frozen checkpoint.

    The caller must pass values produced by the *same* checkpoint, data and
    decoding configuration; a "repeat" that changes any of those is a new
    measurement, not a resolution probe.  With fewer than two finite repeats
    the resolution is ``not_verifiable``.  The reported ``resolution`` is the
    larger of the observed range and the sample SD -- a conservative envelope
    of what a re-run can move by chance.
    """
    problems, warnings = [], []
    mee = (None if minimum_meaningful_effect is None
           else _es_float(minimum_meaningful_effect))
    if minimum_meaningful_effect is not None and (mee is None or mee <= 0.0):
        problems.append(
            f'minimum_meaningful_effect={minimum_meaningful_effect!r} must be '
            'a finite positive number')
    max_rel = (None if max_relative_resolution is None
               else _es_float(max_relative_resolution))
    if max_relative_resolution is not None and (max_rel is None or max_rel <= 0.0):
        problems.append(
            f'max_relative_resolution={max_relative_resolution!r} must be a '
            'finite positive number')
    values, dropped = [], 0
    if repeat_values is not None:
        if isinstance(repeat_values, (str, bytes, bytearray, dict)):
            problems.append(
                'repeat_values must be a sequence of numbers, not a '
                f'{type(repeat_values).__name__}')
        else:
            try:
                raw = list(repeat_values)
            except TypeError:
                problems.append('repeat_values must be an iterable of numbers')
                raw = []
            for value in raw:
                f = _es_float(value)
                if f is None:
                    dropped += 1
                else:
                    values.append(f)
    out = {'kind': 'endpoint_measurement_resolution',
           'status': 'not_verifiable', 'endpoint': endpoint,
           'n_repeats': len(values), 'dropped_nonfinite': dropped,
           'mean': None, 'sd': None, 'range': None, 'resolution': None,
           'relative_resolution': None,
           'minimum_meaningful_effect': mee, 'resolves_mee': None,
           'max_relative_resolution': max_rel,
           'problems': problems, 'warnings': warnings,
           'note': ('repeats must reuse the same checkpoint, data and decode '
                    'configuration; the resolution is a measurement envelope, '
                    'not a confidence interval for the arm effect')}
    if problems:
        out['status'] = 'invalid'
        return out
    if len(values) < 2:
        out['warnings'].append(
            'fewer than two finite repeat values; the measurement resolution '
            'cannot be estimated')
        return out
    mean = float(np.mean(np.asarray(values, dtype=float)))
    sd = float(np.std(np.asarray(values, dtype=float), ddof=1))
    value_range = float(max(values) - min(values))
    resolution = float(max(value_range, sd))
    out.update({'mean': mean, 'sd': sd, 'range': value_range,
                'resolution': resolution,
                'relative_resolution': (float(resolution / abs(mean))
                                        if mean != 0.0 else None)})
    if mee is not None:
        out['resolves_mee'] = bool(resolution + 1e-12 < mee)
    if mee is not None and not out['resolves_mee']:
        out['status'] = 'resolution_limited'
        out['warnings'].append(
            f'repeated identical evaluations move by {resolution:.6g}, which '
            f'is not smaller than the declared minimum meaningful effect '
            f'{mee:.6g}; the endpoint cannot resolve a worthwhile difference')
    elif (mee is None and max_rel is not None
          and out['relative_resolution'] is not None
          and out['relative_resolution'] > max_rel):
        out['status'] = 'coarse_resolution'
        out['warnings'].append(
            f'relative measurement resolution {out["relative_resolution"]:.6g} '
            f'exceeds the declared maximum {max_rel:.6g}')
    else:
        out['status'] = 'ok'
    return out


def endpoint_sensitivity(endpoint=None, *, control_level=None, arm_levels=None,
                         metric=None, direction=None, bounds=None,
                         minimum_meaningful_effect=None,
                         repeat_values=None, paired_sd=None, n_units=None,
                         alpha=0.05, target_power=0.80,
                         near_bound_tolerance=None,
                         min_relative_headroom=None,
                         max_relative_resolution=None):
    """Combine headroom, measurement resolution and MDE into one design verdict.

    The three checks are deliberately separate: a bounded endpoint can have
    plenty of headroom but too coarse a measurement, and an unbounded PPL
    endpoint has no ceiling but can still be unresolved.  ``claim_allowed`` is
    ``True`` only when all three checks pass and a MEE was declared; missing
    resolution/MDE evidence yields ``None`` (unverified), never an implicit
    pass, while any failed check yields ``False``.
    """
    head = endpoint_headroom(
        endpoint, control_level, arm_levels=arm_levels, metric=metric,
        direction=direction, bounds=bounds,
        minimum_meaningful_effect=minimum_meaningful_effect,
        near_bound_tolerance=near_bound_tolerance,
        min_relative_headroom=min_relative_headroom)
    res = endpoint_measurement_resolution(
        repeat_values, endpoint=endpoint,
        minimum_meaningful_effect=minimum_meaningful_effect,
        max_relative_resolution=max_relative_resolution)
    mee = head.get('minimum_meaningful_effect')

    mde = {'kind': 'endpoint_mde', 'status': 'not_verifiable', 'mde': None,
           'paired_sd': None, 'n_units': None, 'alpha': None,
           'target_power': None, 'mde_covers_mee': None,
           'problems': [], 'warnings': [],
           'method': ('two-sided t MDE at the unit level: '
                      '(t_{1-alpha/2,n-1} + z_{power}) * paired_sd / sqrt(n);'
                      ' a normal/t approximation, not an exact sign-flip MDE')}
    a = _es_probability(alpha)
    tp = _es_probability(target_power)
    if alpha is not None and a is None:
        mde['problems'].append(f'alpha={alpha!r} must be in (0,1)')
    if target_power is not None and tp is None:
        mde['problems'].append(
            f'target_power={target_power!r} must be in (0,1)')
    sd = _es_float(paired_sd) if paired_sd is not None else None
    nu = _es_int(n_units, minimum=2) if n_units is not None else None
    if paired_sd is not None and (sd is None or sd <= 0.0):
        mde['problems'].append(
            f'paired_sd={paired_sd!r} must be a finite positive number')
    if n_units is not None and nu is None:
        mde['problems'].append(
            f'n_units={n_units!r} must be an integer >= 2')
    if paired_sd is None and n_units is None:
        mde['warnings'].append(
            'no paired_sd/n_units supplied; the MDE is not verifiable')
    elif paired_sd is None or n_units is None:
        mde['warnings'].append(
            'both paired_sd and n_units are required to compute an MDE')
    elif not mde['problems'] and a is not None and tp is not None:
        tc = _t_critical(1.0 - a, float(nu - 1))
        z = _normal_quantile(tp)
        if (math.isfinite(tc) and math.isfinite(z) and z > 0.0):
            value = float((tc + z) * sd / math.sqrt(float(nu)))
            mde.update({'status': 'ok', 'mde': value, 'paired_sd': sd,
                        'n_units': int(nu), 'alpha': a,
                        'target_power': tp})
            if mee is not None:
                mde['mde_covers_mee'] = bool(value + 1e-12 <= mee)
                if not mde['mde_covers_mee']:
                    mde['status'] = 'underpowered'
                    mde['warnings'].append(
                        f'MDE {value:.6g} exceeds the declared minimum '
                        f'meaningful effect {mee:.6g}; the planned n cannot '
                        'resolve a worthwhile difference')
        else:
            mde['warnings'].append(
                'the t critical value or target-power quantile is not finite '
                'at these inputs; the MDE is not verifiable')
    if mde['problems']:
        mde['status'] = 'invalid'

    blockers = []
    if head.get('status') in ('invalid', 'not_estimable', 'not_verifiable'):
        blockers.append(f"headroom: {head.get('status')}")
    if head.get('exhausted') is True:
        blockers.append('headroom exhausted at the beneficial bound')
    if head.get('headroom_covers_mee') is False:
        blockers.append('headroom is smaller than the declared MEE')
    if res.get('status') == 'invalid':
        blockers.append('measurement resolution: invalid')
    if res.get('resolves_mee') is False:
        blockers.append('measurement resolution is not smaller than the MEE')
    if mde.get('status') == 'invalid':
        blockers.append('MDE: invalid')
    if mde.get('mde_covers_mee') is False:
        blockers.append('MDE exceeds the declared MEE')

    # A declared MEE turns the three checks into a joint claim.  Missing
    # evidence is not a pass: ``informative`` stays None and the status is
    # ``not_verifiable`` even when the headroom check itself is green.
    head_ok = None
    if head.get('status') not in ('invalid', 'not_estimable', 'not_verifiable'):
        head_ok = not (head.get('exhausted') is True
                       or head.get('headroom_covers_mee') is False)
    res_ok = res.get('resolves_mee')
    mde_ok = mde.get('mde_covers_mee')
    if mee is None:
        informative = None
    elif any(v is False for v in (head_ok, res_ok, mde_ok)):
        informative = False
    elif all(v is True for v in (head_ok, res_ok, mde_ok)):
        informative = True
    else:
        informative = None
    if mee is not None:
        if head_ok is None:
            blockers.append('headroom is not verifiable')
        if res_ok is None:
            blockers.append('measurement resolution is not verifiable')
        if mde_ok is None:
            blockers.append('MDE is not verifiable')
    claim_allowed = (None if (mee is None or informative is None)
                     else bool(informative))
    status = 'ok'
    if head.get('status') == 'invalid' or res.get('status') == 'invalid':
        status = 'invalid'
    elif head.get('status') == 'not_estimable':
        status = 'not_estimable'
    elif head.get('status') == 'not_verifiable':
        status = 'not_verifiable'
    elif head.get('exhausted') is True:
        status = 'saturated'
    elif mee is not None and informative is False:
        status = 'uninformative'
    elif mee is not None and informative is None:
        status = 'not_verifiable'
    elif head.get('status') == 'low_headroom':
        status = 'low_headroom'
    return {'kind': 'endpoint_sensitivity', 'status': status,
            'endpoint': endpoint, 'metric': head.get('metric'),
            'direction': head.get('direction'), 'bounds': head.get('bounds'),
            'minimum_meaningful_effect': mee,
            'informative': informative, 'claim_allowed': claim_allowed,
            'blockers': blockers, 'headroom': head,
            'measurement_resolution': res, 'mde': mde,
            'policy': {'near_bound_tolerance':
                       (head.get('policy') or {}).get('near_bound_tolerance'),
                       'min_relative_headroom':
                       (head.get('policy') or {}).get('min_relative_headroom'),
                       'rule': ENDPOINT_SENSITIVITY_POLICY['rule']}}


def _es_load_mapping(obj, *, label):
    """(mapping, status, detail) for an inline mapping or a JSON path."""
    if obj is None:
        return None, 'not_supplied', f'no {label} supplied'
    if isinstance(obj, dict):
        return obj, 'inline', None
    if isinstance(obj, (str, os.PathLike)):
        path = os.fspath(obj)
        if not os.path.exists(path):
            return None, 'missing', f'{label} {path!r} does not exist'
        try:
            with open(path, encoding='utf-8') as fh:
                data = json.load(fh)
        except Exception as exc:  # noqa: BLE001 - report, never crash the gate
            return None, 'unreadable', (
                f'{label} {path!r} is not readable JSON '
                f'({type(exc).__name__}: {exc})')
        if not isinstance(data, dict):
            return None, 'not_an_object', (
                f'{label} {path!r} is a {type(data).__name__}, not an object')
        return data, 'loaded', None
    return None, 'invalid', (
        f'{label} must be an object or a JSON path, got {type(obj).__name__}')


def _es_bool(value, *, name, problems):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    problems.append(f'{name}={value!r} must be a boolean')
    return False


def _es_merge_mee(declared, observed, *, endpoint, problems):
    """One MEE per endpoint; a conflicting declaration is a hard error."""
    d = None if declared is None else _es_float(declared)
    o = None if observed is None else _es_float(observed)
    if declared is not None and (d is None or d <= 0.0):
        problems.append(
            f'{endpoint}: declared minimum_meaningful_effect={declared!r} must '
            'be a finite positive number')
    if observed is not None and (o is None or o <= 0.0):
        problems.append(
            f'{endpoint}: pilot minimum_meaningful_effect={observed!r} must '
            'be a finite positive number')
    if d is not None and o is not None and abs(d - o) > 1e-12 * max(1.0, abs(d), abs(o)):
        problems.append(
            f'{endpoint}: the manifest and the pilot declare different '
            f'minimum meaningful effects ({d!r} vs {o!r}); a margin must be '
            'fixed before the run')
        return d
    return d if d is not None else o


def endpoint_sensitivity_audit(manifest=None, pilot=None, *, config=None,
                               verbose=True):
    """Audit pre-registered endpoint sensitivity before any run/test access.

    ``manifest`` is the frozen/draft design record (a mapping or a JSON path);
    ``pilot`` is an optional mapping/JSON with the observed validation level,
    repeated-evaluation values and paired SD per endpoint.  ``config`` can
    override the manifest's ``endpoint_sensitivity`` block when a design is
    being explored.

    The audit distinguishes three outcomes that are easy to conflate:

    * ``ok`` / ``warnings``: every *declared* check passes / nothing was
      declared that turns a missing pilot into a failure;
    * ``fail``: a declared requirement (``require_headroom``,
      ``require_measurement_resolution``, ``require_mde_within_headroom``,
      ``require_declared_mee``) is not met, or the declaration is malformed;
    * ``not_verifiable_locally``: the manifest itself is unavailable, so there
      is no design to audit.

    A missing pilot is never a pass: when the manifest requires headroom, the
    missing pilot is a failure; otherwise it is an explicit warning.
    """
    problems, warnings = [], []
    default_manifest = None
    m, mstat, mdetail = _es_load_mapping(manifest, label='manifest')
    if m is None and manifest is None:
        default = 'confirmation_frozen.json'
        if os.path.exists(default):
            default_manifest = default
            m, mstat, mdetail = _es_load_mapping(default, label='manifest')
        else:
            mstat, mdetail = 'missing', (
                'no manifest supplied and no ./confirmation_frozen.json exists')
    out = {'kind': 'endpoint_sensitivity_audit', 'status': 'not_verifiable_locally',
           'manifest': (os.fspath(manifest) if isinstance(manifest, (str, os.PathLike))
                        else (manifest.get('_path') if isinstance(manifest, dict) else None)),
           'manifest_status': mstat, 'pilot': (os.fspath(pilot)
                                               if isinstance(pilot, (str, os.PathLike))
                                               else None),
           'pilot_status': 'not_supplied',
           'n_endpoints': 0, 'n_comparisons': 0,
           'endpoint_sensitivity': {}, 'comparisons': {},
           'requirements': {}, 'problems': problems, 'warnings': warnings,
           'note': ('design-stage only: reads the manifest and a validation '
                    'pilot; never reads wiki.test.raw, never changes a '
                    'p-value/Holm/frozen verdict')}
    if out['manifest'] is None:
        out['manifest'] = default_manifest
    if m is None:
        if mstat in ('missing', 'not_supplied'):
            out['status'] = 'not_verifiable_locally'
            out['warnings'].append(mdetail or 'manifest is unavailable')
        else:
            out['status'] = 'fail'
            problems.append(mdetail or 'manifest is unavailable/unreadable')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    out['manifest'] = m.get('_path') or out['manifest']

    # ---- pilot ---------------------------------------------------------
    default_pilot = None
    p, pstat, pdetail = _es_load_mapping(pilot, label='pilot')
    if p is None and pilot is None:
        default = os.path.join('analysis_v12', 'endpoint_sensitivity_pilot.json')
        if os.path.exists(default):
            default_pilot = default
            p, pstat, pdetail = _es_load_mapping(default, label='pilot')
        else:
            pstat = 'not_available'
            pdetail = ('no endpoint-sensitivity pilot on disk; run the '
                       'validation pilot before a confirmatory/headroom gate')
    if out['pilot'] is None:
        out['pilot'] = default_pilot
    out['pilot_status'] = pstat
    out['pilot_split'] = (p.get('split') if isinstance(p, dict) else None)
    if p is not None:
        psplit = p.get('split')
        if psplit is None:
            warnings.append(
                'pilot.split is not declared; a design pilot must come from '
                'the validation split, never the held-out test split')
        elif str(psplit).strip().lower() in ('test', 'heldout', 'held-out',
                                             'holdout'):
            problems.append(
                f'pilot.split={psplit!r} is the held-out test split; design '
                'calibration must not use the one-shot test data')
        elif str(psplit).strip().lower() not in ('validation', 'valid', 'val'):
            problems.append(
                f'pilot.split={psplit!r} is not a recognized validation '
                'split')
    if p is None:
        if pstat in ('unreadable', 'not_an_object', 'invalid'):
            out['status'] = 'fail'
            problems.append(pdetail or 'pilot is unusable')
        else:
            warnings.append(pdetail or 'no endpoint-sensitivity pilot')

    decl = config
    if decl is None:
        decl = m.get('endpoint_sensitivity')
        if decl is None:
            _stats = m.get('statistics')
            if isinstance(_stats, dict):
                decl = _stats.get('endpoint_sensitivity')
    if decl is not None and not isinstance(decl, dict):
        out['status'] = 'fail'
        problems.append(
            f'endpoint_sensitivity must be an object, got '
            f'{type(decl).__name__}')
        decl = {}
    decl = decl or {}
    decl_eps = decl.get('endpoints')
    if decl_eps is not None and not isinstance(decl_eps, dict):
        out['status'] = 'fail'
        problems.append(
            f'endpoint_sensitivity.endpoints must be an object, got '
            f'{type(decl_eps).__name__}')
        decl_eps = {}
    decl_eps = decl_eps or {}
    pilot_eps = {}
    if p is not None:
        pilot_eps = p.get('endpoints')
        if pilot_eps is not None and not isinstance(pilot_eps, dict):
            out['status'] = 'fail'
            problems.append(
                f'pilot.endpoints must be an object, got '
                f'{type(pilot_eps).__name__}')
            pilot_eps = {}
        pilot_eps = pilot_eps or {}

    comps = m.get('primary_comparisons')
    comps = comps if isinstance(comps, list) else []
    comp_eps = []
    for comp in comps:
        if isinstance(comp, dict) and isinstance(comp.get('endpoint'), str):
            comp_eps.append(comp['endpoint'])
    endpoints = sorted(set(comp_eps) | set(decl_eps) | set(pilot_eps))
    out['n_endpoints'] = len(endpoints)

    # ---- per-endpoint design ------------------------------------------
    for ep in endpoints:
        cfg = decl_eps.get(ep)
        pcfg = pilot_eps.get(ep)
        if cfg is None:
            cfg = {}
        if pcfg is None:
            pcfg = {}
        if not isinstance(cfg, dict):
            problems.append(
                f'{ep}: endpoint_sensitivity.endpoints entry is a '
                f'{type(cfg).__name__}, not an object')
            continue
        if not isinstance(pcfg, dict):
            problems.append(
                f'{ep}: pilot.endpoints entry is a {type(pcfg).__name__}, '
                'not an object')
            continue
        mee = _es_merge_mee(
            cfg.get('minimum_meaningful_effect'),
            pcfg.get('minimum_meaningful_effect'),
            endpoint=ep, problems=problems)
        control = (pcfg.get('control_level')
                   if pcfg.get('control_level') is not None
                   else pcfg.get('level'))
        arm_levels = pcfg.get('arm_levels')
        if arm_levels is None and isinstance(pcfg.get('arm_level'), dict):
            arm_levels = pcfg.get('arm_level')
        repeat_values = pcfg.get('repeat_values')
        if repeat_values is None:
            repeat_values = pcfg.get('repeats')
        paired_sd = (pcfg.get('paired_sd')
                     if pcfg.get('paired_sd') is not None
                     else pcfg.get('pilot_sd'))
        n_units = (pcfg.get('n_units')
                   if pcfg.get('n_units') is not None
                   else pcfg.get('units'))
        block = endpoint_sensitivity(
            ep, control_level=control, arm_levels=arm_levels,
            metric=cfg.get('metric'), direction=cfg.get('direction'),
            bounds=cfg.get('bounds'),
            minimum_meaningful_effect=mee,
            repeat_values=repeat_values, paired_sd=paired_sd,
            n_units=n_units,
            alpha=(cfg.get('alpha')
                   if cfg.get('alpha') is not None else 0.05),
            target_power=(cfg.get('target_power')
                          if cfg.get('target_power') is not None else 0.80),
            near_bound_tolerance=(cfg.get('near_bound_tolerance')
                                  if cfg.get('near_bound_tolerance') is not None
                                  else decl.get('near_bound_tolerance')),
            min_relative_headroom=(cfg.get('min_relative_headroom')
                                   if cfg.get('min_relative_headroom') is not None
                                   else decl.get('min_relative_headroom')),
            max_relative_resolution=(
                cfg.get('max_relative_resolution')
                if cfg.get('max_relative_resolution') is not None
                else (decl.get('max_relative_resolution')
                      if decl.get('max_relative_resolution') is not None
                      else pcfg.get('max_relative_resolution'))))
        out['endpoint_sensitivity'][ep] = block

    # ---- declared requirements ----------------------------------------
    req = {'require_headroom': False,
           'require_measurement_resolution': False,
           'require_mde_within_headroom': False,
           'require_declared_mee': False}
    for key in list(req):
        if key in decl:
            req[key] = _es_bool(decl[key], name=f'endpoint_sensitivity.{key}',
                                problems=problems)
    out['requirements'] = dict(req)
    if decl and not endpoints:
        problems.append(
            'endpoint_sensitivity is declared but no endpoint is listed or '
            'used by a primary comparison; the requirements have no subject')

    for ep, block in out['endpoint_sensitivity'].items():
        if block.get('status') == 'invalid':
            problems.extend(f'{ep}: {p}' for p in
                            (block.get('headroom') or {}).get('problems') or [])
            problems.extend(f'{ep}: {p}' for p in
                            (block.get('measurement_resolution') or {}).get('problems') or [])
            problems.extend(f'{ep}: {p}' for p in
                            (block.get('mde') or {}).get('problems') or [])
        if block.get('status') in ('not_estimable', 'not_verifiable'):
            warnings.append(
                f'{ep}: endpoint sensitivity is {block.get("status")}; '
                'supply a pilot level (and metric/bounds if the endpoint is '
                'not in the policy) before relying on this endpoint')
        if block.get('status') == 'low_headroom':
            warnings.append(
                f'{ep}: low headroom to the beneficial bound; the design '
                'should state how a ceiling/floor effect is ruled out')
        if block.get('status') == 'saturated':
            warnings.append(
                f'{ep}: the best observed level is already at the beneficial '
                'bound; a positive improvement cannot be measured')
        if block.get('status') == 'uninformative':
            warnings.append(
                f'{ep}: the declared minimum meaningful effect is not '
                'resolvable from the observed headroom/measurement '
                'resolution/MDE; the endpoint cannot support that claim')
        head = block.get('headroom') or {}
        res = block.get('measurement_resolution') or {}
        mde = block.get('mde') or {}
        if req['require_declared_mee'] and block.get('minimum_meaningful_effect') is None:
            problems.append(
                f'{ep}: require_declared_mee is set but no '
                'minimum_meaningful_effect is declared')
        if req['require_headroom']:
            if head.get('status') in ('not_verifiable', 'not_estimable', 'invalid'):
                problems.append(
                    f'{ep}: require_headroom is set but headroom is '
                    f'{head.get("status")}')
            elif head.get('exhausted') is True:
                problems.append(
                    f'{ep}: require_headroom is set but the best observed '
                    'level is already at the beneficial bound (no headroom)')
            elif head.get('headroom_covers_mee') is False:
                problems.append(
                    f'{ep}: require_headroom is set but the available '
                    'headroom is smaller than the declared minimum '
                    'meaningful effect')
        if req['require_measurement_resolution']:
            if (block.get('minimum_meaningful_effect') is None
                    and decl.get('max_relative_resolution') is None
                    and cfg.get('max_relative_resolution') is None
                    and pcfg.get('max_relative_resolution') is None):
                problems.append(
                    f'{ep}: require_measurement_resolution is set but no '
                    'MEE or max_relative_resolution is declared; the '
                    'resolution cannot be judged against a target')
            elif res.get('status') in ('not_verifiable', 'invalid',
                                       'coarse_resolution'):
                problems.append(
                    f'{ep}: require_measurement_resolution is set but the '
                    'measurement resolution is not verifiable or is coarser '
                    'than the declared maximum; supply repeat_values from '
                    'identical evaluations')
            elif res.get('resolves_mee') is False:
                problems.append(
                    f'{ep}: require_measurement_resolution is set but the '
                    'measurement resolution is not smaller than the declared '
                    'minimum meaningful effect')
        if req['require_mde_within_headroom']:
            if block.get('minimum_meaningful_effect') is None:
                problems.append(
                    f'{ep}: require_mde_within_headroom is set but no '
                    'minimum_meaningful_effect is declared; an MDE without a '
                    'target cannot be "within" it')
            elif mde.get('status') in ('not_verifiable', 'invalid'):
                problems.append(
                    f'{ep}: require_mde_within_headroom is set but the MDE is '
                    'not verifiable; supply paired_sd and n_units')
            elif mde.get('mde_covers_mee') is False:
                problems.append(
                    f'{ep}: require_mde_within_headroom is set but the '
                    'planned MDE exceeds the declared minimum meaningful '
                    'effect')

    if not decl:
        warnings.append(
            'the manifest has no endpoint_sensitivity block: metric bounds '
            'and direction can be classified, but headroom, measurement '
            'resolution and the minimum meaningful effect are not '
            'pre-registered')
    if p is None and (decl_eps or decl):
        warnings.append(
            'the endpoint_sensitivity design is declared but no validation '
            'pilot is available; headroom/resolution/MDE stay unverified')

    # ---- comparison-level claims --------------------------------------
    for comp in comps:
        if not isinstance(comp, dict):
            continue
        name = comp.get('name') or comp.get('id') or '?'
        ep = comp.get('endpoint')
        block = out['endpoint_sensitivity'].get(ep) if isinstance(ep, str) else None
        out['n_comparisons'] += 1
        if block is None:
            out['comparisons'][str(name)] = {
                'endpoint': ep, 'status': 'not_estimable',
                'claim_allowed': None,
                'blockers': ['the comparison endpoint is not declared']}
            continue
        out['comparisons'][str(name)] = {
            'endpoint': ep, 'status': block.get('status'),
            'claim_allowed': block.get('claim_allowed'),
            'structural_possible': (block.get('headroom') or {}).get(
                'structurally_possible'),
            'headroom': (block.get('headroom') or {}).get('headroom'),
            'minimum_meaningful_effect': block.get(
                'minimum_meaningful_effect'),
            'blockers': list(block.get('blockers') or [])}

    if problems:
        out['status'] = 'fail'
    elif any(b.get('status') == 'low_headroom'
             for b in out['endpoint_sensitivity'].values()):
        out['status'] = 'warnings'
    elif warnings:
        out['status'] = 'warnings'
    else:
        out['status'] = 'ok'
    if verbose:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
