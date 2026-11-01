# Design-stage Type S / Type M error diagnostics.
#
# Power and the minimum detectable effect answer "will this panel reject?".
# They do not answer the two questions a reader asks after a rejection:
#
#   * Type S: conditional on a statistically significant result, how often is
#     the estimated effect on the wrong side of zero?
#   * Type M: conditional on a statistically significant result, how much is
#     the magnitude exaggerated relative to the assumed true effect?
#
# Gelman & Carlin (2014) name these readouts and show that they are worst for
# small true effects and small n -- exactly the region a 12-seed panel occupies
# when it is asked about a SESOI-sized effect.  This block computes them under
# a two-sided normal location model with a family-adjusted critical value.
#
# SCOPE: design/preparation only.  It reads no split, produces no p-value,
# enters no Holm family, and never changes a frozen `verified`/`confirmed`
# verdict or `CODE_SEMANTICS`.  The output is a planning statement about an
# ASSUMED true effect, not a posterior or a claim about the observed data.

import math
import numbers
import time

from statistics import NormalDist


TYPE_SM_POLICY = {
    'schema': 1,
    'model': ('two-sided normal location model with a known paired-delta SD; '
              'the critical value is the Bonferroni-adjusted two-sided '
              'normal quantile unless an explicit critical value is given'),
    'type_s_definition': (
        'P(sign(estimate) != sign(assumed true effect) | reject); '
        'conditional on a significant result, not an unconditional error rate'),
    'type_m_definition': (
        'E[|estimate| | reject] / |assumed true effect|; the expected '
        'magnitude exaggeration conditional on a significant result'),
    'family_adjustment': (
        'alpha_adjusted = alpha / family_size (Bonferroni); this is '
        'conservative under positive dependence and is not the exact '
        'Holm/sign-flip threshold of the frozen decision rule'),
    'anchor_rule': (
        'the caller must state the assumed true effect.  A pilot mean is a '
        'design anchor only: it is itself an estimate and may be optimistic. '
        'The curve below the anchor shows how the risk grows for smaller '
        'effects, including the SESOI'),
    'thresholds': {
        'target_power': 0.80,
        'max_type_s': 0.05,
        'max_type_m': 2.0,
        'role': ('display/planning thresholds.  They do not become a frozen '
                 'decision rule unless the manifest declares '
                 'statistics.type_sm.require=true with explicit thresholds'),
    },
    'boundary': (
        'normal location model; known sigma; one assumed true effect; not a '
        'substitute for exact sign-flip power, the minimum detectable effect, '
        'the SESOI/TOST analysis, or the final test'),
    'role': ('design-stage sign/magnitude risk readout; no p-value, no Holm '
             'member, no frozen verdict, no test-split read'),
}


_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------

def _tsm_number(value, name):
    """Return a finite float or ``None`` (bool/str/NaN/inf are not numbers)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _tsm_int(value, name, *, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        return None
    out = int(value)
    if out < minimum:
        return None
    if maximum is not None and out > maximum:
        return None
    return out


def _tsm_pdf(x):
    return math.exp(-0.5 * x * x) / _SQRT2PI


def _tsm_cdf(x):
    # sqrt-free stable form; for very negative x erf returns -1 and this is 0.
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _tsm_quantile(p):
    if not (0.0 < p < 1.0):
        raise ValueError(f'probability must be in (0,1), got {p!r}')
    return float(NormalDist().inv_cdf(p))


def _tsm_z(alpha, family_size, critical_value):
    """Return (critical_value, alpha_adjusted, rule, problem)."""
    if critical_value is not None:
        z = _tsm_number(critical_value, 'critical_value')
        if z is None or z <= 0.0:
            return None, None, 'explicit', (
                f'critical_value={critical_value!r} must be finite and > 0')
        return z, None, 'explicit', None
    a = _tsm_number(alpha, 'alpha')
    if a is None or not (0.0 < a < 1.0):
        return None, None, 'bonferroni', (
            f'alpha={alpha!r} must be a finite probability in (0,1)')
    m = _tsm_int(family_size, 'family_size', minimum=1)
    if m is None:
        return None, None, 'bonferroni', (
            f'family_size={family_size!r} must be an integer >= 1')
    adj = a / float(m)
    if not (0.0 < adj < 1.0):
        return None, None, 'bonferroni', (
            f'alpha/family_size={adj!r} is not a probability in (0,1)')
    try:
        z = _tsm_quantile(1.0 - adj / 2.0)
    except (ValueError, OverflowError):
        return None, None, 'bonferroni', (
            'the family-adjusted two-sided critical value is not representable')
    return z, adj, 'bonferroni', None


def _tsm_metrics(n, lam, z):
    """Return the conditional readouts for standardized effect ``lam``."""
    if not (isinstance(lam, (int, float)) and math.isfinite(lam) and lam > 0.0):
        return None
    if not (isinstance(z, (int, float)) and math.isfinite(z) and z > 0.0):
        return None
    upper = 1.0 - _tsm_cdf(z - lam)
    lower = _tsm_cdf(-z - lam)
    power = upper + lower
    if power <= 0.0:
        return None
    type_s = lower / power
    # E[|Z| 1{reject}] = lam*(upper-lower) + phi(z-lam) + phi(-z-lam)
    numerator = (lam * (upper - lower)
                 + _tsm_pdf(z - lam) + _tsm_pdf(-z - lam))
    type_m = numerator / (lam * power)
    if not math.isfinite(type_m):
        return None
    # P(|Z| > 2*lam | reject).  Rejection already requires |Z| > z, so the
    # conditional event is |Z| > max(z, 2*lam); a naive P(|Z| > 2*lam) can
    # exceed the rejection probability when 2*lam < z.
    _cut = max(z, 2.0 * lam)
    p_over_2 = ((1.0 - _tsm_cdf(_cut - lam))
                + _tsm_cdf(-_cut - lam)) / power
    # P(|Z| < lam/2 | reject).  This is empty unless the understatement band
    # overlaps the rejection region; otherwise the earlier formula divided the
    # unconditional band mass by a much smaller rejection probability and
    # clamped a spurious 1.0.
    if 0.5 * lam <= z:
        p_under_half = 0.0
    else:
        p_under_half = ((_tsm_cdf(0.5 * lam - lam)
                         - _tsm_cdf(z - lam))
                        + (_tsm_cdf(-z - lam)
                           - _tsm_cdf(-0.5 * lam - lam))) / power
    return {
        'power': float(power),
        'power_correct_direction': float(upper),
        'power_wrong_direction': float(lower),
        'type_s': float(type_s),
        'type_m': float(type_m),
        'p_overstate_2x': float(min(1.0, max(0.0, p_over_2))),
        'p_understate_half': float(min(1.0, max(0.0, p_under_half))),
    }


def _tsm_first_n(n_max, predicate):
    for k in range(2, n_max + 1):
        if predicate(k):
            return k
    return None


def _tsm_search_requirements(n, sigma, abs_effect, z, *, target_power,
                             max_type_s, max_type_m, n_max):
    """Find the smallest n meeting each planning requirement."""
    def metrics(k):
        try:
            lam = abs_effect * math.sqrt(k) / sigma
        except (OverflowError, ZeroDivisionError):
            return None
        return _tsm_metrics(k, lam, z)

    def _power_ok(k):
        met = metrics(k)
        return met is not None and met['power'] >= target_power

    def _type_s_ok(k):
        met = metrics(k)
        return (met is not None and met['type_s'] <= max_type_s
                and met['power'] >= target_power)

    def _type_m_ok(k):
        met = metrics(k)
        return (met is not None and met['type_m'] <= max_type_m
                and met['power'] >= target_power)

    min_power = _tsm_first_n(n_max, _power_ok)
    min_type_s = _tsm_first_n(n_max, _type_s_ok)
    min_type_m = _tsm_first_n(n_max, _type_m_ok)
    return {
        'min_n_for_power': min_power,
        'min_n_for_type_s': min_type_s,
        'min_n_for_type_m': min_type_m,
        'n_shortfall_for_power': (None if min_power is None
                                  else max(0, min_power - n)),
        'n_shortfall_for_type_s': (None if min_type_s is None
                                   else max(0, min_type_s - n)),
        'n_shortfall_for_type_m': (None if min_type_m is None
                                   else max(0, min_type_m - n)),
    }


# ---------------------------------------------------------------------------
# core design diagnostic
# ---------------------------------------------------------------------------

def type_sm_design(n, sigma, effect, *, alpha=0.05, family_size=1,
                   critical_value=None, n_max=512, target_power=0.80,
                   max_type_s=0.05, max_type_m=2.0, include_curve=False,
                   curve_factors=(1.0, 0.75, 0.50, 0.25)):
    """Design-stage Type S / Type M readout for one assumed true effect.

    Parameters are in the SAME unit as the paired effect (e.g. raw PPL
    difference, or a log ratio).  Only the standardized effect
    ``|effect| * sqrt(n) / sigma`` matters, so the result is scale-invariant
    when both ``effect`` and ``sigma`` are multiplied by the same factor.

    ``effect`` is the ASSUMED true effect; it is not estimated from the pilot
    here.  Use :func:`type_sm_design_from_deltas` when the pilot mean should be
    the anchor, and read the returned curve for smaller effects.

    Returns a dict with ``status='ok'`` and ``available=True`` when the
    diagnostic is computable.  ``design_ok`` is the conjunction of the three
    planning thresholds; ``status='warnings'`` means computable but not
    meeting them.  Invalid inputs return ``status='invalid'`` with named
    problems instead of raising.
    """
    out = {
        'kind': 'type_sm_design',
        'schema': TYPE_SM_POLICY['schema'],
        'status': 'invalid',
        'available': False,
        'n': None, 'sigma': None, 'effect': None, 'abs_effect': None,
        'alpha': None, 'family_size': None, 'alpha_adjusted': None,
        'critical_value': None, 'critical_rule': None,
        'se': None, 'noncentrality': None,
        'power': None, 'power_correct_direction': None,
        'power_wrong_direction': None,
        'type_s': None, 'type_m': None,
        'expected_signed_estimate': None,
        'p_overstate_2x': None, 'p_understate_half': None,
        'target_power': None, 'max_type_s': None, 'max_type_m': None,
        'meets': {'power': None, 'type_s': None, 'type_m': None},
        'design_ok': False,
        'min_n_for_power': None, 'min_n_for_type_s': None,
        'min_n_for_type_m': None,
        'n_shortfall_for_power': None, 'n_shortfall_for_type_s': None,
        'n_shortfall_for_type_m': None,
        'problems': [], 'warnings': [],
        'interpretation': None,
        'policy': TYPE_SM_POLICY,
    }
    problems = out['problems']

    nn = _tsm_int(n, 'n', minimum=2, maximum=1000000)
    if nn is None:
        problems.append(f'n={n!r} must be an integer >= 2')
    sd = _tsm_number(sigma, 'sigma')
    if sd is None or sd <= 0.0:
        problems.append(f'sigma={sigma!r} must be a finite number > 0')
    eff = _tsm_number(effect, 'effect')
    if eff is None or eff == 0.0:
        problems.append(f'effect={effect!r} must be a finite non-zero number')

    a = _tsm_number(alpha, 'alpha')
    m = _tsm_int(family_size, 'family_size', minimum=1)
    if critical_value is not None:
        # An explicit critical value makes alpha/family_size optional, but a
        # supplied value must still be usable rather than recorded as 0/None.
        if a is not None and not (0.0 < a < 1.0):
            problems.append(
                f'alpha={alpha!r} must be a finite probability in (0,1)')
        if m is not None and m < 1:
            problems.append(
                f'family_size={family_size!r} must be an integer >= 1')
    z, adj, rule, z_problem = _tsm_z(a, m, critical_value)
    if z_problem:
        problems.append(z_problem)
    tp = _tsm_number(target_power, 'target_power')
    if tp is None or not (0.0 < tp < 1.0):
        problems.append(
            f'target_power={target_power!r} must be a finite probability in (0,1)')
    mts = _tsm_number(max_type_s, 'max_type_s')
    if mts is None or not (0.0 < mts <= 0.5):
        problems.append(
            f'max_type_s={max_type_s!r} must be in (0, 0.5]')
    mtm = _tsm_number(max_type_m, 'max_type_m')
    if mtm is None or mtm < 1.0:
        problems.append(
            f'max_type_m={max_type_m!r} must be a finite number >= 1')
    nm = _tsm_int(n_max, 'n_max', minimum=2, maximum=100000)
    if nm is None:
        problems.append(f'n_max={n_max!r} must be an integer in [2, 100000]')

    out.update({
        'n': nn, 'sigma': sd, 'effect': eff,
        'abs_effect': (None if eff is None else abs(eff)),
        'alpha': a, 'family_size': m,
        'alpha_adjusted': adj, 'critical_value': z, 'critical_rule': rule,
        'target_power': tp, 'max_type_s': mts, 'max_type_m': mtm,
    })
    if problems:
        return out

    try:
        se = sd / math.sqrt(nn)
        lam = abs(eff) / se
    except (OverflowError, ZeroDivisionError):
        problems.append(
            'the standardized effect is not representable at this n/sigma/effect')
        return out
    if not (math.isfinite(se) and se > 0.0):
        problems.append('the paired standard error is zero/non-finite')
        return out
    if not (math.isfinite(lam) and lam > 0.0):
        problems.append(
            'the standardized effect is zero/non-finite; Type S/M are not '
            'computable')
        return out
    met = _tsm_metrics(nn, lam, z)
    if met is None:
        problems.append('the conditional readout is undefined at this effect')
        return out
    searches = _tsm_search_requirements(
        nn, sd, abs(eff), z, target_power=tp, max_type_s=mts,
        max_type_m=mtm, n_max=nm)
    meets = {
        'power': bool(met['power'] >= tp),
        'type_s': bool(met['type_s'] <= mts),
        'type_m': bool(met['type_m'] <= mtm),
    }
    design_ok = all(meets.values())
    signed = (1.0 if eff > 0.0 else -1.0)
    out.update({
        'status': 'ok' if design_ok else 'warnings',
        'available': True,
        'se': float(se),
        'noncentrality': float(lam),
        'power': met['power'],
        'power_correct_direction': met['power_correct_direction'],
        'power_wrong_direction': met['power_wrong_direction'],
        'type_s': met['type_s'],
        'type_m': met['type_m'],
        'expected_signed_estimate': float(signed * met['type_m'] * abs(eff)),
        'p_overstate_2x': met['p_overstate_2x'],
        'p_understate_half': met['p_understate_half'],
        'meets': meets,
        'design_ok': bool(design_ok),
    })
    out.update(searches)
    if design_ok:
        out['interpretation'] = (
            f'At the assumed effect, power={met["power"]:.3f} and a '
            f'significant result would be expected to exaggerate the '
            f'magnitude by {met["type_m"]:.2f}x with a wrong-sign probability '
            f'of {met["type_s"]:.3f}; the planning thresholds are met.')
    else:
        _need = []
        if not meets['power']:
            _need.append(f'power >= {tp:.2f}')
        if not meets['type_s']:
            _need.append(f'Type S <= {mts:.3f}')
        if not meets['type_m']:
            _need.append(f'Type M <= {mtm:.2f}')
        _smallest = [x for x in (out['min_n_for_power'],
                                 out['min_n_for_type_s'],
                                 out['min_n_for_type_m']) if x is not None]
        _n_hint = (f' The smallest n meeting all thresholds in the search is '
                   f'{max(_smallest)}.' if _smallest else
                   ' No n <= n_max meets all thresholds for this effect.')
        out['interpretation'] = (
            f'At the assumed effect, power={met["power"]:.3f}, Type S='
            f'{met["type_s"]:.3f}, Type M={met["type_m"]:.2f}x; expected '
            f'overstatement is {met["type_m"]:.2f}x and '
            f'{met["p_overstate_2x"]:.1%} of significant results would exceed '
            f'twice the assumed effect. The plan misses: {", ".join(_need)}.'
            f'{_n_hint}')
    if include_curve:
        out['curve'] = type_sm_design_curve(
            nn, sd, eff, factors=curve_factors, alpha=a, family_size=m,
            critical_value=critical_value, n_max=nm, target_power=tp,
            max_type_s=mts, max_type_m=mtm)
    return out


def type_sm_design_curve(n, sigma, effect, *, factors=(1.0, 0.75, 0.5, 0.25),
                         alpha=0.05, family_size=1, critical_value=None,
                         n_max=512, target_power=0.80, max_type_s=0.05,
                         max_type_m=2.0):
    """Evaluate Type S/M over a grid of assumed effects (multiples of anchor).

    Smaller true effects are the dangerous direction: Type M grows and the
    wrong-sign probability eventually approaches 0.5.  The default grid spans
    the anchor down to a quarter of it, matching the common situation where a
    pilot mean is an optimistic estimate of the SESOI-sized effect.
    """
    out = {'kind': 'type_sm_design_curve', 'schema': 1,
           'status': 'invalid', 'available': False,
           'n': None, 'sigma': None, 'anchor_effect': None,
           'factors': [], 'points': [], 'problems': [], 'warnings': [],
           'policy': TYPE_SM_POLICY}
    nn = _tsm_int(n, 'n', minimum=2, maximum=1000000)
    if nn is None:
        out['problems'].append(f'n={n!r} must be an integer >= 2')
    sd = _tsm_number(sigma, 'sigma')
    if sd is None or sd <= 0.0:
        out['problems'].append(f'sigma={sigma!r} must be a finite number > 0')
    eff = _tsm_number(effect, 'effect')
    if eff is None or eff == 0.0:
        out['problems'].append(f'effect={effect!r} must be finite and non-zero')
    if factors is None:
        factors = (1.0, 0.75, 0.5, 0.25)
    if isinstance(factors, (str, bytes)):
        out['problems'].append('factors must be an iterable of positive numbers')
        factors = ()
    try:
        factors = list(factors)
    except TypeError:
        out['problems'].append('factors must be an iterable of positive numbers')
        factors = ()
    clean = []
    for i, value in enumerate(factors):
        f = _tsm_number(value, f'factors[{i}]')
        if f is None or f <= 0.0:
            out['problems'].append(
                f'factors[{i}]={value!r} must be a finite number > 0')
        else:
            clean.append(f)
    if not clean:
        out['problems'].append('at least one positive factor is required')
    out.update({'n': nn, 'sigma': sd, 'anchor_effect': eff,
                'factors': list(clean)})
    if out['problems']:
        return out
    points = []
    for f in clean:
        one = type_sm_design(
            nn, sd, eff * f, alpha=alpha, family_size=family_size,
            critical_value=critical_value, n_max=n_max,
            target_power=target_power, max_type_s=max_type_s,
            max_type_m=max_type_m)
        points.append({'factor': f, 'effect': eff * f,
                       'status': one.get('status'),
                       'available': one.get('available'),
                       'power': one.get('power'),
                       'type_s': one.get('type_s'),
                       'type_m': one.get('type_m'),
                       'p_overstate_2x': one.get('p_overstate_2x'),
                       'design_ok': one.get('design_ok')})
        out['warnings'].extend(one.get('warnings') or [])
        out['problems'].extend(one.get('problems') or [])
    out['points'] = points
    out['status'] = ('ok' if points and not out['problems']
                     and all(p['available'] for p in points)
                     else 'invalid')
    out['available'] = out['status'] == 'ok'
    # The curve is monotone in the assumed effect; flag gross violations so a
    # malformed caller cannot present a "larger effect is riskier" table.
    for left, right in zip(points, points[1:]):
        if (left['power'] is not None and right['power'] is not None
                and right['power'] > left['power'] + 1e-12):
            out['warnings'].append(
                'curve is not monotone in effect: a larger assumed effect has '
                'higher power')
    return out


def _tsm_mean_sd(values):
    """Stable mean and sample SD (ddof=1), avoiding overflow at extreme scales."""
    finite = [float(x) for x in values]
    n = len(finite)
    if n < 2:
        return (finite[0] if finite else None), None
    scale = max(abs(x) for x in finite)
    if scale == 0.0:
        return 0.0, 0.0
    scaled = [x / scale for x in finite]
    mean = math.fsum(scaled) / n
    var = math.fsum((x - mean) ** 2 for x in scaled) / (n - 1)
    return mean * scale, math.sqrt(var) * scale


def type_sm_design_from_deltas(n, deltas=None, *, pilot_sd=None, sesoi=None,
                               alpha=0.05, family_size=1, critical_value=None,
                               n_max=512, target_power=0.80,
                               max_type_s=0.05, max_type_m=2.0,
                               include_curve=True,
                               curve_factors=(1.0, 0.75, 0.5, 0.25)):
    """Use a pilot/validation paired-delta vector as the design anchor.

    ``deltas`` are paired effects (for example per-seed PPL differences) on
    the validation split.  The sample SD is the design ``sigma`` and, unless
    ``sesoi`` supplies a smaller/other assumed true effect, the absolute pilot
    mean is the anchor.  Non-finite values are dropped by position and named;
    they are never silently replaced.

    A pilot mean of exactly zero is NOT a zero-variance pass: Type M is
    undefined at a true zero effect, so the result is ``not_applicable`` with
    an explicit reason.  Use ``sesoi=`` if the design target is a non-zero
    smallest effect of interest.
    """
    out = {'kind': 'type_sm_design_from_deltas', 'schema': 1,
           'status': 'invalid', 'available': False,
           'n': None, 'pilot': {}, 'anchor': None, 'problems': [],
           'warnings': [], 'policy': TYPE_SM_POLICY}
    nn = _tsm_int(n, 'n', minimum=2, maximum=1000000)
    if nn is None:
        out['problems'].append(f'n={n!r} must be an integer >= 2')
    raw = []
    dropped = []
    if deltas is not None:
        if isinstance(deltas, (str, bytes, bytearray)):
            out['problems'].append(
                'deltas must be an iterable of numbers, not a string')
            deltas = None
        else:
            try:
                seq = list(deltas)
            except TypeError:
                out['problems'].append(
                    'deltas must be an iterable of numbers, got '
                    f'{type(deltas).__name__}')
                seq = []
            for i, value in enumerate(seq):
                f = _tsm_number(value, f'deltas[{i}]')
                if f is None:
                    dropped.append(i)
                else:
                    raw.append(f)
    explicit_sd = None
    if pilot_sd is not None:
        explicit_sd = _tsm_number(pilot_sd, 'pilot_sd')
        if explicit_sd is None or explicit_sd <= 0.0:
            out['problems'].append(
                f'pilot_sd={pilot_sd!r} must be a finite number > 0')
    if not raw and explicit_sd is None:
        out['problems'].append(
            'at least two finite deltas or an explicit pilot_sd are required')
    if raw and len(raw) < 2 and explicit_sd is None:
        out['problems'].append(
            'a sample SD needs at least two finite deltas; supply pilot_sd '
            'for a single pilot value')
    if not raw and explicit_sd is not None and sesoi is None:
        out['problems'].append(
            'pilot_sd without deltas needs an explicit sesoi to anchor the '
            'assumed true effect')
    sd = explicit_sd
    mean = None
    if raw:
        mean, calc_sd = _tsm_mean_sd(raw)
        if sd is None:
            sd = calc_sd
        elif calc_sd is not None and not math.isclose(
                calc_sd, sd, rel_tol=1e-12, abs_tol=0.0):
            out['warnings'].append(
                'pilot_sd overrides the SD recomputed from deltas; report '
                'which sample each value came from')
    if sd is not None and (not math.isfinite(sd) or sd <= 0.0):
        out['problems'].append(
            'the pilot SD is zero or non-finite; Type S/M are not computable')
        sd = None
    margin = None
    if sesoi is not None:
        margin = _tsm_number(sesoi, 'sesoi')
        if margin is None or margin <= 0.0:
            out['problems'].append(
                f'sesoi={sesoi!r} must be a finite number > 0')
    out['pilot'] = {
        'n': len(raw) + len(dropped),
        'n_finite': len(raw),
        'n_dropped': len(dropped),
        'dropped_positions': dropped,
        'mean': mean,
        'sd': sd,
        'source': ('deltas' if raw else ('pilot_sd' if explicit_sd else None)),
        'sesoi': margin,
    }
    out['n'] = nn
    if out['problems']:
        return out
    anchor = margin if margin is not None else (
        None if mean is None else abs(mean))
    if anchor is None or anchor == 0.0:
        out.update({
            'status': 'not_applicable',
            'available': False,
            'anchor': 'sesoi' if margin is not None else 'pilot_mean',
            'not_applicable_reason': (
                'a true zero effect has no finite Type M; declare a non-zero '
                'sesoi or supply a non-zero pilot mean'),
        })
        return out
    core = type_sm_design(
        nn, sd, anchor, alpha=alpha, family_size=family_size,
        critical_value=critical_value, n_max=n_max,
        target_power=target_power, max_type_s=max_type_s,
        max_type_m=max_type_m, include_curve=include_curve,
        curve_factors=curve_factors)
    out.update({k: v for k, v in core.items()
                if k not in ('problems', 'warnings', 'policy')})
    out['problems'] = list(core.get('problems') or [])
    out['warnings'] = list(out.get('warnings') or []) + list(
        core.get('warnings') or [])
    out['anchor'] = 'declared_sesoi' if margin is not None else 'pilot_mean'
    out['anchor_effect'] = float(anchor)
    if core.get('status') == 'invalid':
        out['status'] = 'invalid'
        out['available'] = False
    return out


def _tsm_close(a, b, rel=1e-9, abs_=1e-12):
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if a is None or b is None:
        return a is b
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError, OverflowError):
        return False
    if not (math.isfinite(fa) and math.isfinite(fb)):
        return False
    return abs(fa - fb) <= max(abs_, rel * max(1.0, abs(fa), abs(fb)))


# ---------------------------------------------------------------------------
# manifest/pilot audit
# ---------------------------------------------------------------------------

def _tsm_manifest_sesoi(manifest, name):
    """Return an absolute-ppl declared SESOI for ``name``, or ``None``.

    A relative/log-ratio margin cannot be converted to an absolute paired
    delta without the baseline PPL, so it is deliberately not guessed.  The
    caller then falls back to the pilot mean and labels that anchor.
    """
    try:
        pol = confirmation_sesoi_for_manifest(manifest)  # noqa: F821
    except Exception:
        return None
    margins = pol.get('margins') if isinstance(pol, dict) else None
    if not isinstance(margins, dict):
        return None
    one = margins.get(name)
    if not isinstance(one, dict):
        return None
    if one.get('unit') != 'absolute_ppl':
        return None
    margin = _tsm_number(one.get('margin'), 'sesoi')
    if margin is None or margin <= 0.0:
        return None
    return margin


def type_sm_from_manifest(manifest, pilot=None, *, max_type_s=None,
                          max_type_m=None, target_power=None, n_max=512):
    """Design audit of every primary comparison against a pilot artifact.

    ``pilot`` is the parsed ``analysis_v12/power_validation.json`` object (or
    the same comparison blocks embedded elsewhere).  A missing pilot is
    ``not_verifiable``, not a pass.  No test split is read.
    """
    out = {'kind': 'type_sm_audit', 'schema': 1, 'status': 'ok',
           'available': True, 'manifest': None, 'pilot': None,
           'pilot_status': 'not_available',
           'n_seeds': None, 'm_family': None, 'alpha': None,
           'thresholds': {'target_power': 0.80, 'max_type_s': 0.05,
                          'max_type_m': 2.0},
           'required': False, 'comparisons': {},
           'violations': [], 'warnings': [], 'problems': [],
           'policy': TYPE_SM_POLICY,
           'note': ('design-stage only; no p-value, no Holm member, no '
                    'frozen verdict, no test-split read')}
    if not isinstance(manifest, dict):
        out['status'] = 'fail'
        out['problems'].append('manifest must be a JSON object')
        return out
    stats = manifest.get('statistics')
    if not isinstance(stats, dict):
        stats = {}
    seeds = manifest.get('seeds')
    comps = manifest.get('primary_comparisons')
    if not isinstance(seeds, list) or not seeds:
        out['problems'].append('manifest.seeds must be a non-empty list')
        seeds = []
    if not isinstance(comps, list) or not comps:
        out['problems'].append(
            'manifest.primary_comparisons must be a non-empty list')
        comps = []
    n = len(seeds)
    m = len(comps)
    out['n_seeds'], out['m_family'] = n, m
    a = _tsm_number(stats.get('alpha', 0.05), 'alpha')
    if a is None or not (0.0 < a < 1.0):
        out['problems'].append(
            f'statistics.alpha={stats.get("alpha")!r} must be in (0,1)')
    out['alpha'] = a
    block = stats.get('type_sm')
    if block is None:
        block = {}
    if not isinstance(block, dict):
        out['problems'].append('statistics.type_sm must be an object')
        block = {}
    required = block.get('require', False)
    if not isinstance(required, bool):
        out['problems'].append('statistics.type_sm.require must be a boolean')
        required = False
    out['required'] = required
    tp = target_power
    if tp is None:
        tp = block.get('target_power', stats.get('min_power_for_test', 0.80))
    ts = max_type_s if max_type_s is not None else block.get(
        'max_type_s', 0.05)
    tm = max_type_m if max_type_m is not None else block.get(
        'max_type_m', 2.0)
    out['thresholds'] = {'target_power': tp, 'max_type_s': ts,
                         'max_type_m': tm}
    # Validate the thresholds once; a malformed declaration is a design
    # failure, not an invitation to use a default.
    _probe = type_sm_design(
        2, 1.0, 0.1, alpha=0.05, family_size=1, n_max=2,
        target_power=tp, max_type_s=ts, max_type_m=tm)
    for p in (_probe.get('problems') or []):
        if ('n=' in p or 'n_max=' in p):
            continue
        out['problems'].append(f'statistics.type_sm: {p}')
    pilot_status = 'not_available'
    if pilot is not None:
        if not isinstance(pilot, dict):
            out['problems'].append('pilot must be a JSON object')
        else:
            pilot_status = 'available'
            if pilot.get('alpha') is not None and a is not None:
                pv_a = _tsm_number(pilot.get('alpha'), 'pilot.alpha')
                if pv_a is None or not _tsm_close(pv_a, a, rel=1e-12):
                    out['problems'].append(
                        f'pilot alpha {pilot.get("alpha")!r} does not match '
                        f'manifest alpha {a!r}')
            if pilot.get('m_family') is not None and m:
                pv_m = _tsm_int(pilot.get('m_family'), 'pilot.m_family',
                                minimum=1)
                if pv_m != m:
                    out['problems'].append(
                        f'pilot family size {pilot.get("m_family")!r} does not '
                        f'match manifest family size {m}')
            _m_sha = manifest.get('manifest_sha256')
            _p_sha = pilot.get('manifest_sha256')
            if (isinstance(_m_sha, str) and _m_sha
                    and isinstance(_p_sha, str) and _p_sha
                    and _m_sha != _p_sha):
                out['problems'].append(
                    'pilot manifest_sha256 does not match the current '
                    'manifest; the pilot was produced for another freeze')
            if not isinstance(pilot.get('comparisons'), dict):
                out['problems'].append(
                    'pilot.comparisons must be an object keyed by comparison '
                    'name')
                pilot = None
    out['pilot_status'] = pilot_status
    out['manifest_sha256'] = manifest.get('manifest_sha256')
    if isinstance(pilot, dict) and pilot.get('status') not in (None, 'pass'):
        out['warnings'].append(
            'the validation power pilot reports status='
            f'{pilot.get("status")!r}; Type S/M is computed, but the pilot '
            'itself failed its power/coverage gate and this is not a clean '
            'pre-launch state')
    out['pilot'] = (None if pilot is None else {
        'status': pilot.get('status'),
        'alpha': pilot.get('alpha'),
        'm_family': pilot.get('m_family'),
        'manifest_sha256': pilot.get('manifest_sha256'),
    })
    pilot_comps = (pilot.get('comparisons')
                   if isinstance(pilot, dict) else None)
    for comp in comps:
        if not isinstance(comp, dict):
            out['problems'].append('a primary_comparisons entry is not an object')
            continue
        name = comp.get('name')
        if not isinstance(name, str) or not name:
            out['problems'].append(
                'a primary_comparisons entry has no non-empty name')
            continue
        cell = (pilot_comps.get(name)
                if isinstance(pilot_comps, dict) else None)
        if not isinstance(cell, dict):
            out['comparisons'][name] = {
                'status': 'not_verifiable',
                'available': False,
                'reason': ('no pilot/validation comparison block with this '
                           'name; Type S/M cannot be anchored'),
            }
            out['warnings'].append(
                f'{name}: no pilot deltas; Type S/M is not verifiable')
            if required:
                out['violations'].append(
                    f'{name}: type_sm.require=true but no pilot deltas are '
                    'available')
            continue
        sesoi = _tsm_manifest_sesoi(manifest, name)
        deltas = cell.get('deltas')
        sd = cell.get('sd')
        mean = cell.get('mean')
        if isinstance(deltas, (list, tuple)) and len(deltas) >= 2:
            res = type_sm_design_from_deltas(
                n, deltas, sesoi=sesoi, alpha=(a if a is not None else 0.05),
                family_size=max(1, m), n_max=n_max, target_power=tp,
                max_type_s=ts, max_type_m=tm, include_curve=True)
        else:
            effect = sesoi if sesoi is not None else (
                None if mean is None else abs(mean))
            if effect is None or sd is None:
                out['comparisons'][name] = {
                    'status': 'not_verifiable',
                    'available': False,
                    'reason': ('pilot block has neither >=2 deltas nor a '
                               'mean+sd pair'),
                    'pilot_cell_keys': sorted(cell),
                }
                out['warnings'].append(
                    f'{name}: pilot block is not reconstructible; Type S/M is '
                    'not verifiable')
                if required:
                    out['violations'].append(
                        f'{name}: type_sm.require=true but the pilot block '
                        'cannot be reconstructed')
                continue
            res = type_sm_design(
                n, sd, effect, alpha=(a if a is not None else 0.05),
                family_size=max(1, m), n_max=n_max, target_power=tp,
                max_type_s=ts, max_type_m=tm, include_curve=True)
            res['anchor'] = ('declared_sesoi' if sesoi is not None
                             else 'pilot_mean')
            res['anchor_effect'] = float(effect)
        rec = dict(res)
        rec['comparison'] = name
        comp_status = rec.get('status')
        if comp_status == 'ok':
            pass
        elif comp_status == 'warnings':
            out['warnings'].append(
                f'{name}: {rec.get("interpretation") or "planning thresholds not met"}')
            if required:
                out['violations'].append(
                    f'{name}: type_sm thresholds not met at the assumed effect')
        elif comp_status in ('invalid', 'not_applicable'):
            out['warnings'].append(
                f'{name}: Type S/M {comp_status}: '
                + '; '.join(rec.get('problems') or [])[:400])
            if required:
                out['violations'].append(
                    f'{name}: type_sm.require=true but Type S/M is '
                    f'{comp_status}')
        out['comparisons'][name] = rec
    if out['violations'] or out['problems']:
        out['status'] = 'fail'
    elif not comps or not seeds:
        out['status'] = 'not_verifiable'
    elif pilot is None:
        out['status'] = ('fail' if required else 'not_verifiable')
    elif out['warnings']:
        out['status'] = 'warnings'
    else:
        out['status'] = 'ok'
    return out


# ---------------------------------------------------------------------------
# replayable artifact
# ---------------------------------------------------------------------------

_TSM_INPUT_KEYS = (
    'n', 'sigma', 'effect', 'alpha', 'family_size', 'critical_value',
    'n_max', 'target_power', 'max_type_s', 'max_type_m',
)

_TSM_SCALAR_FIELDS = (
    'n', 'sigma', 'effect', 'abs_effect', 'alpha', 'family_size',
    'alpha_adjusted', 'critical_value', 'se', 'noncentrality',
    'power', 'power_correct_direction', 'power_wrong_direction',
    'type_s', 'type_m', 'expected_signed_estimate',
    'p_overstate_2x', 'p_understate_half', 'target_power',
    'max_type_s', 'max_type_m',
)

_TSM_EXACT_FIELDS = (
    'status', 'available', 'design_ok', 'critical_rule', 'meets',
    'min_n_for_power', 'min_n_for_type_s', 'min_n_for_type_m',
    'n_shortfall_for_power', 'n_shortfall_for_type_s',
    'n_shortfall_for_type_m',
)

_TSM_FORBIDDEN_KEYS = frozenset({
    'confirmed', 'p_holm', 'p_value', 'p_exact_signflip', 'verified',
    'verdict', 'decision', 'significant', 'significance',
})


def _tsm_frozen_payload(obj):
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str) and key in _TSM_FORBIDDEN_KEYS:
                found.append(key)
            found.extend(_tsm_frozen_payload(value))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found.extend(_tsm_frozen_payload(value))
    return found


def build_type_sm_design(n, sigma, effect, *, alpha=0.05, family_size=1,
                         critical_value=None, n_max=512, target_power=0.80,
                         max_type_s=0.05, max_type_m=2.0,
                         created_at=None):
    """Build a small replayable artifact around :func:`type_sm_design`.

    Every input is finite and JSON-representable; a non-finite value is
    rejected here rather than silently dropped, because the artifact's
    ``content_sha256`` must be a function of exactly the recorded inputs.
    """
    for label, value in (('n', n), ('sigma', sigma), ('effect', effect),
                         ('alpha', alpha), ('family_size', family_size),
                         ('critical_value', critical_value), ('n_max', n_max),
                         ('target_power', target_power),
                         ('max_type_s', max_type_s),
                         ('max_type_m', max_type_m)):
        if value is None:
            continue
        if isinstance(value, bool) or _tsm_number(value, label) is None:
            raise ValueError(f'{label}={value!r} must be a finite number')
    if isinstance(n, bool) or _tsm_int(n, 'n', minimum=2) is None:
        raise ValueError(f'n={n!r} must be an integer >= 2')
    if isinstance(family_size, bool) or _tsm_int(
            family_size, 'family_size', minimum=1) is None:
        raise ValueError(f'family_size={family_size!r} must be an integer >= 1')
    if isinstance(n_max, bool) or _tsm_int(n_max, 'n_max', minimum=2) is None:
        raise ValueError(f'n_max={n_max!r} must be an integer >= 2')
    result = type_sm_design(
        n, sigma, effect, alpha=alpha, family_size=family_size,
        critical_value=critical_value, n_max=n_max,
        target_power=target_power, max_type_s=max_type_s,
        max_type_m=max_type_m)
    inputs = {
        'n': int(n), 'sigma': float(sigma), 'effect': float(effect),
        'alpha': float(alpha), 'family_size': int(family_size),
        'critical_value': (None if critical_value is None
                           else float(critical_value)),
        'n_max': int(n_max), 'target_power': float(target_power),
        'max_type_s': float(max_type_s), 'max_type_m': float(max_type_m),
    }
    artifact = {
        'kind': 'type_sm_design_artifact',
        'schema': 1,
        'created_at': (created_at if created_at is not None
                       else time.strftime('%Y-%m-%dT%H:%M:%S')),
        'inputs': inputs,
        'result': result,
    }
    artifact['content_sha256'] = _sha256_json(  # noqa: F821
        {k: v for k, v in artifact.items()
         if k not in ('created_at', 'content_sha256')})
    return artifact


def validate_type_sm_design(artifact):
    """Replay a stored Type S/M artifact and compare every derived field.

    A hash that reproduces but a result that does not is a ``fail``.  A
    missing hash is not a pass: the artifact then has no identity to replay.
    Smuggled frozen-decision fields are rejected even when the numeric body
    happens to be self-consistent.
    """
    out = {'kind': 'type_sm_design_validation', 'schema': 1,
           'status': 'fail', 'problems': [], 'warnings': [],
           'content_sha256': None, 'result': None}
    if not isinstance(artifact, dict):
        out['problems'].append('artifact must be a JSON object')
        return out
    if artifact.get('kind') != 'type_sm_design_artifact':
        out['problems'].append(
            f'artifact.kind must be "type_sm_design_artifact", got '
            f'{artifact.get("kind")!r}')
    if artifact.get('schema') != 1:
        out['problems'].append(
            f'artifact.schema must be 1, got {artifact.get("schema")!r}')
    stolen = sorted(set(_tsm_frozen_payload(artifact)))
    if stolen:
        out['problems'].append(
            'artifact carries frozen-decision field(s) that a design readout '
            f'must not contain: {stolen}')
    inputs = artifact.get('inputs')
    if not isinstance(inputs, dict):
        out['problems'].append('artifact.inputs must be an object')
        inputs = {}
    missing = [k for k in _TSM_INPUT_KEYS if k not in inputs]
    if missing:
        out['problems'].append(
            f'artifact.inputs is missing required key(s): {missing}')
    recorded = artifact.get('content_sha256')
    out['content_sha256'] = recorded
    body = {k: v for k, v in artifact.items()
            if k not in ('created_at', 'content_sha256')}
    try:
        replay_sha = _sha256_json(body)  # noqa: F821
    except Exception as e:
        out['problems'].append(f'artifact body is not canonical JSON: {e}')
        replay_sha = None
    if not isinstance(recorded, str) or len(recorded) != 64:
        out['problems'].append(
            'artifact.content_sha256 is missing or not a 64-character digest')
    elif replay_sha is not None and replay_sha != recorded:
        out['problems'].append(
            'artifact.content_sha256 does not reproduce from the artifact body')
    if missing:
        return out
    try:
        replay = type_sm_design(
            inputs.get('n'), inputs.get('sigma'), inputs.get('effect'),
            alpha=inputs.get('alpha'), family_size=inputs.get('family_size'),
            critical_value=inputs.get('critical_value'),
            n_max=inputs.get('n_max'),
            target_power=inputs.get('target_power'),
            max_type_s=inputs.get('max_type_s'),
            max_type_m=inputs.get('max_type_m'))
    except Exception as e:
        out['problems'].append(f'replay raised {type(e).__name__}: {e}')
        return out
    out['result'] = replay
    stored = artifact.get('result')
    if not isinstance(stored, dict):
        out['problems'].append('artifact.result must be an object')
        return out
    for key in _TSM_SCALAR_FIELDS:
        if key not in stored:
            out['problems'].append(f'result.{key} is missing')
            continue
        if not _tsm_close(stored.get(key), replay.get(key)):
            out['problems'].append(
                f'result.{key}: stored={stored.get(key)!r}, '
                f'replay={replay.get(key)!r}')
    for key in _TSM_EXACT_FIELDS:
        if key not in stored:
            out['problems'].append(f'result.{key} is missing')
            continue
        if stored.get(key) != replay.get(key):
            out['problems'].append(
                f'result.{key}: stored={stored.get(key)!r}, '
                f'replay={replay.get(key)!r}')
    if not out['problems']:
        out['status'] = 'ok'
    return out


def write_type_sm_design(path, artifact, *, indent=2):
    """Atomically write a Type S/M artifact; returns the path."""
    atomic_write_json(path, artifact, indent=indent)  # noqa: F821
    return path
