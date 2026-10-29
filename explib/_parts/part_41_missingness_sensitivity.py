"""Missing-data sensitivity for complete-case paired panels.

The panel audit already refuses to promote a contrast whose paired seed set is
incomplete (``paired_effect_stats.significant`` requires ``complete_pairs``).
That is the right decision rule, but it leaves the reader with only two states:
"complete" and "not confirmatory".  It does not say whether the observed effect
is *robust* to the cells the run-time cap / failures / resume logic removed, or
how extreme the unobserved seeds would have to be to overturn it.

This fragment supplies the missing quantitative readout, with no new
assumption about the missing mechanism:

* **Partial-identification bounds.**  For a declared range ``[lo, hi]`` of
  plausible per-seed differences, the overall mean over all planned seeds is
  exactly

      mean_obs + (u / N) * (x - mean_obs),      x in [lo, hi],

  where ``u`` is the number of unobserved seeds (missing cells plus cells
  excluded by the same-run_cfg/control gate) and ``N = n + u``.  The two
  endpoints are the sharp bounds for *any* configuration of the unobserved
  seeds inside that range; no MAR/MCAR assumption is involved.

* **Tipping point.**  The average unobserved difference needed to put the
  overall mean exactly on zero (or exactly on a declared SESOI margin) is a
  closed-form number.  If it lies outside the declared range, no arrangement
  of unobserved seeds inside the range can reverse the sign / erase the
  margin.  The ratio to the observed mean says how many "observed effects"
  the unseen seeds would have to contribute.

* **Coverage accounting.**  Which planned cells are absent, which are present
  but unusable, and whether the missing sets differ across arms.  Unequal
  coverage is the operational warning that a complete-case contrast may be
  selected by which arm happened to lose a seed.

Boundaries
----------
* This is a **sensitivity analysis, not a new test**.  It does not produce a
  p-value for the missing cells and never changes ``significant``, a Holm
  family, ``verified`` or a frozen confirmation verdict.
* The default range is the observed delta min/max.  It is explicitly labelled
  as descriptive; a missing seed can of course be more extreme, and then the
  bound no longer covers it.  A pre-registered range can be passed instead.
* Unobserved *excluded* seeds (present but rejected by the control gate) are
  counted with missing cells because their delta is unknown.  They are kept as
  a separate count so a gate failure is not misreported as an unrun cell.
* p-value scenario readouts are limited to ``N <= 16`` planned seeds.  Above
  that cap the exact computation is skipped (not approximated) and the block
  says why.
"""

import math
import numbers
import re

import numpy as np


MISSINGNESS_POLICY = {
    'version': 'missing-data-1',
    'default_range': 'observed_delta_min_max',
    'role': ('partial-identification bounds and tipping points for a '
             'complete-case paired panel; descriptive sensitivity only, it '
             'never changes p/Holm/verified or a frozen verdict'),
    'assumption': ('the unobserved per-seed differences lie inside the '
                   'declared range; the sharp bounds need no MAR/MCAR model, '
                   'but they also cannot cover a value outside the range'),
}

_MISSING_SCENARIO_MAX_N = 16
_MISSING_CELL_RE = re.compile(r'^(?P<arm>.+)::seed(?P<seed>[+-]?\d+)$')


def _miss_alpha(alpha):
    return _validated_alpha(alpha, context='missingness_sensitivity alpha')


def _miss_nonneg(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{label} must be a real number, not a boolean')
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{label} must be a finite real number, got {value!r}') from None
    if not math.isfinite(f) or f < 0.0:
        raise ValueError(f'{label} must be finite and >= 0, got {value!r}')
    return f


def _miss_seed_set(values, label, *, allow_empty=True):
    if values is None:
        values = []
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f'{label} must be an iterable of seed integers, not a string')
    try:
        raw = list(values)
    except TypeError:
        raise ValueError(f'{label} must be an iterable of seed integers') from None
    out = []
    for item in raw:
        seed = _as_seed_int(item)
        if seed is None:
            raise ValueError(f'{label} contains a non-integer seed: {item!r}')
        out.append(seed)
    if len(set(out)) != len(out):
        raise ValueError(f'{label} contains duplicate seeds: {out!r}')
    if not out and not allow_empty:
        raise ValueError(f'{label} must not be empty')
    return out


def _miss_delta_list(deltas):
    if deltas is None:
        return []
    if isinstance(deltas, (str, bytes, bytearray)):
        raise ValueError('deltas must be an iterable of finite per-seed differences')
    try:
        raw = list(deltas)
    except TypeError:
        raise ValueError('deltas must be an iterable of finite per-seed differences') from None
    out = []
    for item in raw:
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(f'a boolean is a flag, not a paired difference: {item!r}')
        try:
            value = float(item)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'a paired difference is not numeric: {item!r}') from None
        if not math.isfinite(value):
            raise ValueError(f'a paired difference is not finite: {item!r}')
        out.append(value)
    return out


def _miss_range(delta_range, deltas):
    if delta_range is None:
        if not deltas:
            return None, 'unavailable_no_observed_delta'
        return [min(deltas), max(deltas)], 'observed_delta_min_max'
    if isinstance(delta_range, (str, bytes, bytearray)):
        raise ValueError('delta_range must be a (low, high) pair')
    try:
        pair = list(delta_range)
    except TypeError:
        raise ValueError('delta_range must be a (low, high) pair') from None
    if len(pair) != 2:
        raise ValueError(f'delta_range must have exactly two endpoints, got {pair!r}')
    endpoints = []
    for item in pair:
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(f'delta_range endpoints must be finite reals, got {item!r}')
        try:
            value = float(item)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'delta_range endpoint is not numeric: {item!r}') from None
        if not math.isfinite(value):
            raise ValueError(f'delta_range endpoint is not finite: {item!r}')
        endpoints.append(value)
    if endpoints[0] > endpoints[1]:
        raise ValueError(f'delta_range low {endpoints[0]!r} exceeds high {endpoints[1]!r}')
    return endpoints, 'declared_range'


def _miss_lerp(mean, weight, value):
    """``mean + weight*(value-mean)`` without an intermediate overflow."""
    try:
        return float(math.fsum((mean, weight * (value - mean))))
    except (OverflowError, ValueError):
        return None


def missingness_sensitivity(deltas, *, seeds, planned_seeds, excluded_seeds=(),
                            margin=0.0, delta_range=None, alpha=0.05, min_n=3):
    """Partial-identification sensitivity for complete-case paired deltas.

    ``deltas`` are the observed per-seed ``with - without`` differences and
    ``seeds`` their identities (same order).  ``planned_seeds`` is the seed
    frame the panel was scheduled to cover.  ``excluded_seeds`` are planned
    seeds whose records exist but whose delta is unusable (the control gate
    rejected the pair); they are counted with missing cells because their
    effect is unknown, but kept as a separate count.

    ``margin`` is a non-negative SESOI in delta units (default 0 = sign).  It
    is applied on the observed side: a positive observed effect is compared
    with ``+margin`` and a negative one with ``-margin``.  A zero observed
    mean has no side, so the margin readout is not applicable.

    The return value is JSON-friendly.  ``status`` is one of ``complete``,
    ``robust_beyond_margin``, ``sign_robust_margin_not``,
    ``not_robust_within_range``, ``zero_observed``, ``no_observed_pairs`` or
    ``bounds_unavailable``.  None of these is a significance verdict.
    """
    alpha = _miss_alpha(alpha)
    _min_n = _as_seed_int(min_n)
    if _min_n is None or _min_n < 0:
        raise ValueError(f'min_n must be a non-negative integer, got {min_n!r}')
    min_n = _min_n
    margin = _miss_nonneg(margin, 'margin')
    d = _miss_delta_list(deltas)
    obs = _miss_seed_set(seeds, 'seeds')
    if len(obs) != len(d):
        raise ValueError(
            f'missingness_sensitivity: {len(d)} delta(s) but {len(obs)} seed(s); '
            'the paired accounting must match')
    planned = _miss_seed_set(planned_seeds, 'planned_seeds', allow_empty=False)
    excluded = _miss_seed_set(excluded_seeds, 'excluded_seeds')
    planned_set = set(planned)
    obs_set = set(obs)
    ex_set = set(excluded)
    if not obs_set <= planned_set:
        extra = sorted(obs_set - planned_set)
        raise ValueError(
            f'missingness_sensitivity: observed seed(s) {extra} are outside the '
            'planned frame; fix the plan or the record identity instead of '
            'silently widening the frame')
    if not ex_set <= planned_set:
        extra = sorted(ex_set - planned_set)
        raise ValueError(
            f'missingness_sensitivity: excluded seed(s) {extra} are outside the '
            'planned frame')
    if obs_set & ex_set:
        raise ValueError('missingness_sensitivity: a seed cannot be both observed and excluded')
    missing = sorted(planned_set - obs_set - ex_set)
    n = len(obs)
    n_missing = len(missing)
    n_excluded = len(excluded)
    u = n_missing + n_excluded
    N = n + u
    if N != len(planned):
        # Defensive: the three accounting buckets must partition the planned
        # frame exactly.  A mismatch would make every bound below describe a
        # different panel than the one the caller thinks it is auditing.
        raise ValueError('missingness_sensitivity: observed/excluded/missing do not partition the planned frame')
    st = paired_effect_stats(d, alpha=alpha, min_n=min_n)
    mean = float(st.get('mean')) if st.get('n') else float('nan')
    std = float(st.get('std')) if st.get('std_defined') else None
    direction = _sign_of(mean) if n else None
    rng, range_basis = _miss_range(delta_range, d)
    out = {
        'kind': 'missingness_sensitivity',
        'policy': MISSINGNESS_POLICY,
        'alpha': float(alpha),
        'min_n': int(min_n),
        'n_planned': int(N),
        'n_observed': int(n),
        'n_missing': int(n_missing),
        'n_excluded': int(n_excluded),
        'n_unobserved': int(u),
        'observed_seeds': sorted(obs),
        'observed_deltas': ([float(x) for _s, x in sorted(zip(obs, d))]
                            if n else []),
        'missing_seeds': missing,
        'excluded_seeds': sorted(excluded),
        'planned_seeds': sorted(planned),
        'observed_mean': (float(mean) if n else None),
        'observed_std': (float(std) if std is not None else None),
        'observed_min': (min(d) if d else None),
        'observed_max': (max(d) if d else None),
        'direction': direction,
        'margin': float(margin),
        'threshold': None,
        'range': ([float(rng[0]), float(rng[1])] if rng else None),
        'range_basis': range_basis,
        'unobserved_mean_for_zero': None,
        'unobserved_mean_for_threshold': None,
        'tipping_ratio_for_zero': None,
        'tipping_in_range_zero': None,
        'tipping_in_range_threshold': None,
        'overall_mean_bounds': None,
        'bounds_direction': None,
        'robust_zero': None,
        'robust_margin': None,
        'status': None,
        'verdict': None,
        'scenarios': None,
        'scenario_note': None,
        'warnings': [],
    }
    if n == 0:
        out['status'] = 'no_observed_pairs'
        out['verdict'] = ('no controlled paired seed is observed; the planned '
                          'frame cannot be summarized')
        out['warnings'].append('no observed paired delta — missingness bounds need at least one measured pair')
        return out
    if direction == 0:
        out['status'] = 'zero_observed'
        out['verdict'] = 'the observed mean is exactly zero; there is no side for a sign or margin claim'
    threshold = (margin if direction > 0 else -margin) if direction else None
    out['threshold'] = threshold
    if u == 0:
        out['status'] = 'complete'
        out['overall_mean_bounds'] = [float(mean), float(mean)]
        out['bounds_direction'] = ('positive' if mean > 0 else 'negative' if mean < 0 else 'zero')
        out['robust_zero'] = bool(mean != 0.0)
        out['robust_margin'] = bool(threshold is not None and
                                    ((direction > 0 and mean > threshold) or
                                     (direction < 0 and mean < threshold)))
        out['tipping_in_range_zero'] = False
        out['tipping_in_range_threshold'] = False
        out['verdict'] = 'complete planned frame: no unobserved cell, so no missing-data sensitivity is needed'
        out['scenarios'] = []
        out['scenario_note'] = 'complete frame; no imputed scenario'
        return out
    if mean != 0.0:
        try:
            x0 = float(-(mean * n) / u)
        except (OverflowError, ValueError):
            x0 = None
        out['unobserved_mean_for_zero'] = x0
        out['tipping_ratio_for_zero'] = (float(n) / float(u))
        if x0 is not None and rng is not None:
            out['tipping_in_range_zero'] = bool(rng[0] <= x0 <= rng[1])
    if threshold is not None:
        try:
            xt = float(math.fsum((threshold, (float(n) / float(u)) * (threshold - mean))))
        except (OverflowError, ValueError):
            xt = None
        out['unobserved_mean_for_threshold'] = xt
        if xt is not None and rng is not None:
            out['tipping_in_range_threshold'] = bool(rng[0] <= xt <= rng[1])
    if rng is None:
        out['status'] = 'bounds_unavailable'
        out['verdict'] = ('no plausible delta range is available; the tipping '
                          'point is reported but the overall mean cannot be bounded')
        out['warnings'].append('no delta range: supply delta_range or at least one observed delta')
        return out
    weight = float(u) / float(N)
    low = _miss_lerp(mean, weight, rng[0])
    high = _miss_lerp(mean, weight, rng[1])
    if low is None or high is None:
        out['status'] = 'bounds_unavailable'
        out['verdict'] = 'the overall-mean bounds exceed the representable float64 range'
        out['warnings'].append('overall-mean bounds are unrepresentable; no robustness verdict is given')
        return out
    out['overall_mean_bounds'] = [float(low), float(high)]
    if low > 0.0 and high > 0.0:
        out['bounds_direction'] = 'positive'
    elif low < 0.0 and high < 0.0:
        out['bounds_direction'] = 'negative'
    elif low == 0.0 and high == 0.0:
        out['bounds_direction'] = 'zero'
    else:
        out['bounds_direction'] = 'straddles_zero'
    # The observed side is the only side a claim could be on.  Robustness is
    # read from the sharp bounds, not from the point estimate.
    out['robust_zero'] = bool(direction > 0 and low > 0.0 or
                              direction < 0 and high < 0.0)
    if threshold is None:
        out['robust_margin'] = None
    elif direction > 0:
        out['robust_margin'] = bool(low > threshold)
    else:
        out['robust_margin'] = bool(high < threshold)
    if direction == 0:
        out['status'] = 'zero_observed'
    elif out['robust_margin']:
        out['status'] = 'robust_beyond_margin'
    elif out['robust_zero']:
        out['status'] = 'sign_robust_margin_not'
    else:
        out['status'] = 'not_robust_within_range'
    if out['status'] == 'robust_beyond_margin':
        out['verdict'] = (
            'the sign and the declared margin survive every unobserved value '
            f'in [{rng[0]:.6g}, {rng[1]:.6g}]')
    elif out['status'] == 'sign_robust_margin_not':
        out['verdict'] = (
            'the sign survives the declared range but the SESOI margin does '
            'not; do not quote the margin-clearing claim as missing-robust')
    elif out['status'] == 'not_robust_within_range':
        out['verdict'] = (
            f'the {u} unobserved seed(s) could reverse the sign for some '
            f'values inside [{rng[0]:.6g}, {rng[1]:.6g}]; the complete-case '
            'direction is not identified by the declared range')
    else:
        out['verdict'] = out.get('verdict') or 'descriptive missingness readout'
    if out['tipping_in_range_zero']:
        out['warnings'].append(
            'the zero tipping value lies inside the declared range: there '
            'exists a permitted assignment of the unobserved seeds that '
            'erases the observed sign')
    if out['tipping_in_range_threshold'] is True and margin > 0.0:
        out['warnings'].append(
            'the SESOI-margin tipping value lies inside the declared range: '
            'the margin-clearing conclusion is not robust to the unobserved cells')
    if range_basis == 'observed_delta_min_max':
        out['warnings'].append(
            'the range is the observed per-seed min/max, not a pre-registered '
            'plausible range; a more extreme unobserved seed lies outside this '
            'sensitivity readout')
    # p-value scenarios.  These use the same exact sign-flip statistic on the
    # completed vector, but the imputed seeds are scenarios, not measurements.
    if N <= _MISSING_SCENARIO_MAX_N:
        candidates = [('range_low', rng[0]), ('range_high', rng[1]),
                      ('zero', 0.0), ('observed_mean', mean)]
        if out['unobserved_mean_for_zero'] is not None:
            candidates.append(('zero_tipping', out['unobserved_mean_for_zero']))
        if out['unobserved_mean_for_threshold'] is not None:
            candidates.append(('margin_tipping', out['unobserved_mean_for_threshold']))
        seen = []
        rows = []
        scale = max([abs(x) for x in d] + [abs(rng[0]), abs(rng[1]), 1.0])
        for label, value in candidates:
            if value is None or not math.isfinite(value):
                continue
            duplicate = False
            for prev in seen:
                if abs(value - prev[1]) <= 1e-12 * scale:
                    duplicate = True
                    break
            if duplicate:
                continue
            seen.append((label, value))
            full = list(d) + [float(value)] * u
            sf = exact_signflip_p(full)
            p = sf.get('p')
            overall = _miss_lerp(mean, weight, float(value))
            rows.append({
                'label': label,
                'unobserved_value': float(value),
                'overall_mean': overall,
                'direction': _sign_of(overall),
                'direction_consistent': bool(overall is not None and
                                             _sign_of(overall) == direction),
                'p_exact_signflip': (float(p) if p is not None and math.isfinite(float(p)) else None),
                'nominal_significant': bool(
                    p is not None and math.isfinite(float(p)) and float(p) < alpha
                    and overall is not None and _sign_of(overall) == direction
                    and N >= min_n),
                'imputed': True,
            })
        out['scenarios'] = rows
        out['scenario_note'] = (
            'imputed scenarios only: every unobserved seed is set to the same '
            'value and the exact sign-flip p is recomputed. These are not '
            'measurements and are never used as a significance verdict.')
        _obs_p = st.get('p_exact_signflip')
        _obs_nominal = bool(_obs_p is not None and math.isfinite(float(_obs_p))
                            and float(_obs_p) < alpha)
        if _obs_nominal and all(r['nominal_significant'] is False for r in rows):
            out['warnings'].append(
                'the observed contrast is nominally significant, but no '
                'scenario in the declared range stays nominally significant '
                'in the observed direction; the result is not missing-data '
                'robust even at the favourable endpoint')
    else:
        out['scenarios'] = None
        out['scenario_note'] = (
            f'N={N} planned seeds exceeds the exact scenario cap '
            f'({_MISSING_SCENARIO_MAX_N}); p scenarios are skipped rather '
            'than approximated')
    return out


def panel_missingness_audit(summary, *, planned_cells=None, planned_seeds=None):
    """Coverage audit for one panel: which planned cells are absent/unusable.

    ``planned_cells`` is the run manifest's flat list (``arm::seedN``).  When
    it is absent, a caller-supplied ``planned_seeds`` list is applied to every
    observed arm; with neither, the audit reports ``no_plan`` and only counts
    what is present (it never pretends that the observed cells were the plan).

    The output separates three states that complete-case analysis collapses:
    a cell with a usable PPL, a cell present but unusable (no primary PPL), and
    a cell with no record at all.  It also reports whether the absent seeds are
    the same across arms; unequal missing sets are the concrete warning that a
    contrast's complete-case subset may be selected by arm.
    """
    if not isinstance(summary, dict):
        return {
            'kind': 'panel_missingness_audit',
            'status': 'not_verifiable',
            'reason': f'summary is a {type(summary).__name__}, not a record mapping',
            'policy': MISSINGNESS_POLICY,
            'by_arm': {}, 'warnings': [],
        }
    recs, ambiguous = _variant_seed_map(summary)
    present, usable = {}, {}
    _legacy_ok = legacy_warm_tags(summary)
    for _key, _rec in summary.items():
        if not isinstance(_rec, dict) or 'seed' not in _rec or _rec.get('synthesized'):
            continue
        _rk = _record_identity(_rec, _key, _legacy_ok)
        if _rk is None:
            continue
        _variant, _tag, _seed = _rk
        present.setdefault(_variant, set()).add(_seed)
        if (ppl_is_usable(_rec.get('ppl'))
                and _record_primary_ppl(_rec) is not None):
            usable.setdefault(_variant, set()).add(_seed)
    observed_seeds = sorted({s for (_v, _t, s) in recs})
    usable_seeds = sorted({s for v, ss in usable.items() for s in ss})
    plan = None
    plan_basis = None
    if planned_cells is not None:
        if isinstance(planned_cells, (str, bytes, bytearray)):
            raise ValueError('planned_cells must be an iterable of arm::seedN strings')
        try:
            cells = list(planned_cells)
        except TypeError:
            raise ValueError('planned_cells must be an iterable of arm::seedN strings') from None
        plan = {}
        seen = set()
        for cell in cells:
            if not isinstance(cell, str):
                raise ValueError(f'planned cell is not a string: {cell!r}')
            match = _MISSING_CELL_RE.match(cell)
            if match is None:
                raise ValueError(
                    f'planned cell {cell!r} is not in arm::seedN form; a malformed '
                    'plan cannot be used to infer what is missing')
            arm = match.group('arm')
            seed = int(match.group('seed'))
            key = (arm, seed)
            if key in seen:
                raise ValueError(f'planned cell {cell!r} is duplicated')
            seen.add(key)
            plan.setdefault(arm, set()).add(seed)
        plan_basis = 'run_manifest_planned_cells'
    elif planned_seeds is not None:
        frame = _miss_seed_set(planned_seeds, 'planned_seeds')
        plan = {arm: set(frame) for arm in sorted(set(present) | set(usable))}
        plan_basis = 'declared_planned_seeds'
    out = {
        'kind': 'panel_missingness_audit',
        'policy': MISSINGNESS_POLICY,
        'plan_basis': plan_basis,
        'ambiguous_records': sorted(str(x) for x in ambiguous),
        'n_observed_cells': int(sum(len(v) for v in present.values())),
        'n_usable_cells': int(sum(len(v) for v in usable.values())),
        'n_arms_observed': int(len(present)),
        'observed_seeds_union': observed_seeds,
        'usable_seeds_union': usable_seeds,
        'by_arm': {},
        'n_planned_cells': None,
        'n_missing_cells': None,
        'n_unusable_cells': None,
        'coverage_rate': None,
        'unequal_arm_coverage': None,
        'status': None,
        'warnings': [],
    }
    if plan is None:
        out['status'] = 'no_plan'
        out['warnings'].append(
            'no planned-cell list was supplied; missing cells cannot be '
            'distinguished from seeds that were never scheduled. Coverage is '
            'reported only over the observed cells, and contrasts use the '
            'observed seed union as a relative frame.')
        for arm in sorted(set(present) | set(usable)):
            out['by_arm'][arm] = {
                'planned_seeds': None,
                'observed_seeds': sorted(present.get(arm, set())),
                'usable_seeds': sorted(usable.get(arm, set())),
                'unusable_seeds': sorted(present.get(arm, set()) - usable.get(arm, set())),
                'missing_seeds': [],
                'n_missing': None,
                'missing_rate': None,
            }
        return out
    arms = sorted(set(plan) | set(present) | set(usable))
    total_planned = 0
    total_missing = 0
    total_unusable = 0
    missing_sets = []
    for arm in arms:
        planned = set(plan.get(arm, set()))
        pres = present.get(arm, set())
        use = usable.get(arm, set())
        missing = sorted(planned - pres)
        unusable = sorted((pres & planned) - use)
        total_planned += len(planned)
        total_missing += len(missing)
        total_unusable += len(unusable)
        if planned:
            missing_sets.append(tuple(missing))
        out['by_arm'][arm] = {
            'planned_seeds': sorted(planned),
            'observed_seeds': sorted(pres & planned),
            'usable_seeds': sorted(use & planned),
            'unusable_seeds': unusable,
            'missing_seeds': missing,
            'n_missing': len(missing),
            'missing_rate': (float(len(missing)) / float(len(planned)) if planned else None),
        }
        if planned and pres - planned:
            out['warnings'].append(
                f'{arm}: measured seed(s) {sorted(pres - planned)} are outside '
                'the planned frame; they are not counted as coverage')
    out['n_planned_cells'] = int(total_planned)
    out['n_missing_cells'] = int(total_missing)
    out['n_unusable_cells'] = int(total_unusable)
    out['coverage_rate'] = (float(total_planned - total_missing - total_unusable) /
                            float(total_planned) if total_planned else None)
    out['unequal_arm_coverage'] = bool(len(set(missing_sets)) > 1)
    if total_missing or total_unusable:
        out['status'] = 'incomplete'
        out['warnings'].append(
            f'{total_missing} planned cell(s) have no record and '
            f'{total_unusable} present cell(s) have no usable primary PPL; '
            'a complete-case contrast over this panel is a selected subset')
    else:
        out['status'] = 'complete'
    if out['unequal_arm_coverage']:
        out['warnings'].append(
            'the missing seed sets differ across arms; a complete-case '
            'contrast can be selected by which arm lost a seed')
    return out
