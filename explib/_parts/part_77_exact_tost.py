# Distribution-free equivalence testing: exact sign-flip TOST.
#
# The repository's paired-t TOST answers "is the effect inside the declared
# SESOI?" with a Student-t approximation.  The confirmatory superiority test,
# by contrast, is an exact sign-flip test.  Mixing the two means that the
# equivalence half of a practical-significance claim can be much more
# optimistic than the superiority half -- most visibly when a tiny panel has
# zero within-panel variance: the t limit returns p=0 for any constant vector
# below the margin, while a sign-flip test cannot go below 1/2**n.
#
# This block adds the exact companion.  For a paired-difference vector d and a
# positive margin M, TOST is the intersection-union test of
#
#     H0_low : mu <= -M   vs   H1_low : mu > -M
#     H0_high: mu >=  M   vs   H1_high: mu <  M
#
# Under the same "paired differences are symmetric about the null boundary"
# assumption as `exact_signflip_p`, each one-sided test is a sign-flip test on
# the shifted vector d+M (low) or M-d (high).  The p-value is the fraction of
# sign assignments whose shifted sum is at least the observed shifted sum.
# The TOST p is the maximum of the two one-sided p-values.
#
# SCOPE: descriptive/robustness readout only.  It produces no Holm member,
# never changes the pre-registered t-TOST decision, `verified`/`confirmed`,
# any PPL, or `CODE_SEMANTICS`.  It reads no split and trains nothing.

import math
import numbers
import time

import numpy as np


EXACT_TOST_POLICY = {
    'schema': 1,
    'method': ('two one-sided exact sign-flip tests on the shifted vectors '
               'd+M and M-d; TOST p is the maximum of the two one-sided p'),
    'assumption': (
        'each shifted paired difference is symmetric about zero under its '
        'one-sided null; the same exchangeability/symmetry assumption as the '
        'confirmatory exact sign-flip test'),
    'resolution': (
        'each one-sided sign-flip p has floor 1/2**n_nonzero, so a TOST '
        'equivalence claim at level alpha is impossible when that floor is '
        'at least alpha, regardless of the observed effect'),
    'role': ('distribution-free companion to the paired-t TOST; descriptive '
             'only, no p_holm, no frozen verdict, no split read'),
    'boundary': (
        'usually no more powerful than the parametric t-TOST when normality holds and '
        'does not replace a pre-registered t-based decision; the exact p is '
        'discrete, so a non-rejection can be a resolution limit rather than '
        'evidence against equivalence'),
}

_ETOST_ENUM_MAX_N = 20
_ETOST_MITM_MAX_N = 40
_ETOST_TOL = 1e-12
_ETOST_INPUT_KEYS = ('deltas', 'margin', 'conf')
_ETOST_FROZEN_FIELDS = ('confirmed', 'p_holm', 'verified', 'verdict',
                        'decision')


def _etost_number(value, name, *, positive=False):
    """Finite float, or ``None`` for bool/str/NaN/inf (and non-positive)."""
    if isinstance(value, (bool, np.bool_)):
        return None
    if not isinstance(value, numbers.Real):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(out):
        return None
    if positive and out <= 0.0:
        return None
    return out


def _etost_inputs(deltas):
    """Return ``(finite ndarray, n_dropped)`` or raise on a non-iterable.

    The scalar contract mirrors `tost_paired`: booleans, strings, NaN and
    infinities are not measurements and are dropped by the direct readout.
    The artifact builder is stricter and refuses them entirely.
    """
    if isinstance(deltas, (str, bytes, bytearray, dict)):
        raise ValueError('exact_signflip_tost: deltas must be an iterable of '
                         f'numbers, not a {type(deltas).__name__}')
    try:
        raw = list(deltas if deltas is not None else [])
    except TypeError:
        raise ValueError('exact_signflip_tost: deltas must be an iterable') from None
    values, dropped = [], 0
    for x in raw:
        f = _finite_scalar(x)  # noqa: F821
        if f is None:
            dropped += 1
        else:
            values.append(float(f))
    return np.asarray(values, dtype=float), dropped


def _etost_floor(n_nonzero):
    """Smallest attainable one-sided sign-flip p with this many non-zeros."""
    n = int(n_nonzero)
    if n <= 0:
        return 1.0
    return math.ldexp(1.0, -n)


def _etost_signed_sums(values):
    """All 2**k signed sums of ``values`` (deterministic doubling table)."""
    out = np.zeros(1, dtype=float)
    for v in np.asarray(values, dtype=float):
        out = np.concatenate((out + v, out - v))
    return out


def _etost_scaled(values):
    """Scale by the max absolute value so sums stay representable."""
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return vals, 0.0
    maxabs = float(np.max(np.abs(vals)))
    if not math.isfinite(maxabs):
        return None, None
    if maxabs == 0.0:
        return vals, 0.0
    return vals / maxabs, maxabs


def _etost_mitm_upper(values, threshold, tol):
    """Exact count of sign assignments with sum >= threshold, by MITM."""
    n = int(values.size)
    half = n // 2
    left = _etost_signed_sums(values[:half])
    right = _etost_signed_sums(values[half:])
    right.sort()
    # For each left sum a, the admissible right sums are b >= T - a (with
    # the same absolute tie tolerance as the n<=20 enumeration).
    idx = np.searchsorted(right, threshold - tol - left, side='left')
    count = int((right.size - idx).sum())
    return count / float(1 << n)


def _etost_saddle_upper(values, threshold, floor):
    """One-sided Lugannani-Rice saddlepoint tail for n > 40.

    The null distribution of ``S = sum(s_i v_i)`` is symmetric, so a negative
    threshold is handled as ``1 - P(S >= -threshold)``.  This is an
    approximation, never labelled exact.  Zeros are removed because they
    multiply sign assignments without changing the distribution; the reported
    floor is the exact floor for the non-zero count.
    """
    vals = [float(x) for x in values if float(x) != 0.0]
    if not vals:
        return 1.0
    sabs = math.fsum(abs(x) for x in vals)
    if threshold <= -sabs:
        return 1.0
    if threshold >= sabs:
        return float(floor)
    if threshold == 0.0:
        return 0.5
    if threshold < 0.0:
        p = 1.0 - _etost_saddle_upper(values, -threshold, floor)
        return float(min(1.0, max(0.0, p)))
    T = float(threshold)
    if not math.isfinite(T) or T <= 0.0:
        return 0.5
    if T >= sabs * (1.0 - 1e-13):
        return float(floor)
    inv = 1.0 / max(abs(x) for x in vals)

    def _kp(t):
        return math.fsum(x * math.tanh(t * x) for x in vals)

    def _k(t):
        # log cosh(z) = |z| + log1p(exp(-2|z|)) - log(2), stable for large |z|
        return math.fsum(
            abs(t * x) + math.log1p(math.exp(-2.0 * abs(t * x))) - math.log(2.0)
            for x in vals)

    def _kpp(t):
        return math.fsum(x * x * (1.0 - math.tanh(t * x) ** 2) for x in vals)

    hi = inv
    for _ in range(60):
        if _kp(hi) >= T:
            break
        hi *= 2.0
    else:
        sd = math.sqrt(math.fsum(x * x for x in vals))
        p = math.erfc(T / (sd * math.sqrt(2.0))) if sd > 0.0 else floor
        return float(min(1.0, max(floor, p)))
    lo = 0.0
    for _ in range(160):
        mid = 0.5 * (lo + hi)
        if _kp(mid) < T:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    Kt = _k(t)
    K2 = _kpp(t)
    w = math.sqrt(max(0.0, 2.0 * (t * T - Kt)))
    u = t * math.sqrt(K2) if K2 > 0.0 else 0.0
    if w < 1e-8 or u <= 0.0:
        sd = math.sqrt(math.fsum(x * x for x in vals))
        p = math.erfc(T / (sd * math.sqrt(2.0))) if sd > 0.0 else floor
    else:
        phi = math.exp(-0.5 * w * w) / math.sqrt(2.0 * math.pi)
        p = 0.5 * math.erfc(w / math.sqrt(2.0)) + phi * (1.0 / u - 1.0 / w)
    return float(min(1.0, max(floor, p)))


def _etost_upper_tail(values):
    """P(sum(s_i * values_i) >= sum(values)) under random signs s_i.

    Returns a small record with the p-value, the exact resolution floor and
    the method actually used.  This is the one-sided primitive behind both
    arms of the exact TOST.
    """
    vals = np.asarray(values, dtype=float)
    n = int(vals.size)
    n_nonzero = int(np.count_nonzero(vals))
    floor = _etost_floor(n_nonzero)
    out = {'n': n, 'n_nonzero': n_nonzero, 'p': float('nan'),
           'p_floor': floor, 'method': 'none'}
    if n == 0:
        return out
    if n_nonzero == 0:
        out.update({'p': 1.0, 'method': 'exact'})
        return out
    scaled, _maxabs = _etost_scaled(vals)
    if scaled is None:
        return out
    try:
        threshold = float(math.fsum(float(x) for x in scaled))
        sabs = float(math.fsum(abs(float(x)) for x in scaled))
    except (TypeError, ValueError, OverflowError):
        return out
    tol = _ETOST_TOL * n
    if threshold <= -sabs + tol:
        out.update({'p': 1.0, 'method': 'exact'})
        return out
    if threshold >= sabs - tol:
        out.update({'p': floor, 'method': 'exact'})
        return out
    if n <= _ETOST_ENUM_MAX_N:
        sums = _etost_signed_sums(scaled)
        count = int(np.count_nonzero(sums >= threshold - tol))
        p = count / float(sums.size)
        method = 'exact'
    elif n <= _ETOST_MITM_MAX_N:
        p = _etost_mitm_upper(scaled, threshold, tol)
        method = 'exact-mitm'
    else:
        p = _etost_saddle_upper(scaled, threshold, floor)
        method = 'saddlepoint'
    if not math.isfinite(p):
        return out
    out.update({'p': float(min(1.0, max(floor, p))), 'method': method})
    return out


def _etost_method(low, high):
    methods = {low.get('method'), high.get('method')}
    if 'saddlepoint' in methods:
        return 'saddlepoint'
    if 'exact-mitm' in methods:
        return 'exact-mitm'
    if methods == {'exact'}:
        return 'exact'
    return 'mixed'


def exact_signflip_tost(deltas, margin, conf=0.90):
    """Distribution-free two-one-sided sign-flip equivalence test.

    ``deltas`` are paired per-seed effects on one scale.  ``margin`` is the
    pre-declared SESOI on the same scale.  ``conf`` is the two-sided
    confidence level used by the paired-t TOST, so the one-sided decision
    level is ``(1-conf)/2`` (0.05 at the default 0.90).

    The result is always descriptive.  It never changes a frozen p-value,
    Holm family, `verified`/`confirmed` verdict, or any measurement.
    """
    out = {
        'kind': 'exact_signflip_tost',
        'schema': 1,
        'status': 'not_estimable',
        'role': ('distribution-free TOST companion; descriptive only, no '
                 'p_holm, no frozen verdict'),
        'n': 0,
        'n_dropped': 0,
        'n_nonzero_low': 0,
        'n_nonzero_high': 0,
        'margin': None,
        'conf': None,
        'alpha_one_sided': None,
        'mean': None,
        'p_low': None,
        'p_high': None,
        'p_tost': None,
        'p_floor_low': None,
        'p_floor_high': None,
        'p_floor': None,
        'p_floor_by_n': None,
        'resolution_min_n': None,
        'min_n_for_alpha': None,
        'resolution_limited': None,
        'equivalent': None,
        'method': 'none',
        'p_low_method': 'none',
        'p_high_method': 'none',
        'reason': None,
        'note': ('two one-sided sign-flip tests; exact only under symmetry of '
                 'the shifted paired differences; the p-value is discrete'),
    }
    d, dropped = _etost_inputs(deltas)
    out['n'] = int(d.size)
    out['n_dropped'] = int(dropped)
    m = _etost_number(margin, 'margin', positive=True)
    c = _etost_number(conf, 'conf')
    out['margin'] = m
    out['conf'] = c
    if d.size < 2:
        out['reason'] = f'fewer than two finite paired seeds (n={d.size})'
        return out
    if m is None:
        out['reason'] = f'margin={margin!r} must be finite and > 0'
        return out
    if c is None or not (0.0 < c < 1.0):
        out['reason'] = f'conf={conf!r} must be a finite probability in (0,1)'
        return out
    alpha = (1.0 - c) / 2.0
    out['alpha_one_sided'] = float(alpha)
    _ms = _scaled_mean_std_if_extreme(d, ddof=1)  # noqa: F821
    try:
        mean = float(_ms[0]) if _ms is not None else float(np.mean(d))
    except (TypeError, ValueError, OverflowError):
        mean = None
    out['mean'] = float(mean) if mean is not None and math.isfinite(mean) else None
    # The test is invariant to a common positive rescaling.  Form the shifted
    # vectors on a max-scaled copy so a finite panel like
    # ``d=[1e308, 1.5e308], margin=1e308`` cannot overflow before the exact
    # tail is taken; the original units are still what the result records.
    try:
        _scale = max(float(np.max(np.abs(d))) if d.size else 0.0, m)
    except (TypeError, ValueError, OverflowError):
        _scale = 0.0
    if not (math.isfinite(_scale) and _scale > 0.0):
        out['reason'] = 'the panel/margin scale is not representable'
        return out
    d_work = d / _scale
    m_work = m / _scale
    try:
        low = _etost_upper_tail(d_work + m_work)
        high = _etost_upper_tail(m_work - d_work)
    except Exception as e:  # defensive: a non-finite input must not crash a report
        out['reason'] = f'exact sign-flip TOST failed ({type(e).__name__}: {e})'
        return out
    if not (math.isfinite(low['p']) and math.isfinite(high['p'])):
        out['reason'] = 'one-sided sign-flip p is not representable'
        return out
    out.update({
        'status': 'ok',
        'n_nonzero_low': int(low['n_nonzero']),
        'n_nonzero_high': int(high['n_nonzero']),
        'p_low': float(low['p']),
        'p_high': float(high['p']),
        'p_tost': float(max(low['p'], high['p'])),
        'p_floor_low': float(low['p_floor']),
        'p_floor_high': float(high['p_floor']),
        'p_floor': float(max(low['p_floor'], high['p_floor'])),
        'p_floor_by_n': float(_etost_floor(d.size)),
        'method': _etost_method(low, high),
        'p_low_method': low['method'],
        'p_high_method': high['method'],
    })
    try:
        # Strict rejection (p < alpha) needs 2**-n < alpha, so an exact power
        # of two (e.g. alpha=0.25 at conf=0.5) needs the next n, not log2.
        min_n = int(math.floor(math.log2(1.0 / alpha) + 1e-12)) + 1
    except (ValueError, OverflowError):
        min_n = None
    out['resolution_min_n'] = min_n
    out['min_n_for_alpha'] = min_n
    out['resolution_limited'] = bool(out['p_floor_by_n'] >= alpha)
    out['equivalent'] = bool((not out['resolution_limited'])
                             and out['p_tost'] < alpha)
    out['note'] = (
        'two one-sided exact sign-flip tests on d+margin (low) and margin-d '
        '(high); p_floor is this panel\'s smallest attainable one-sided p and '
        'p_floor_by_n=2**-n is the resolution limit for n nonzero seeds, so '
        'p_floor_by_n >= alpha means equivalence is impossible at this n even '
        'if every observed difference is zero.  It does not replace the '
        'pre-registered paired-t TOST decision.')
    return out


def _etost_json_equal(a, b):
    """Structural equality with a 1e-12 float tolerance."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, numbers.Real) and isinstance(b, numbers.Real):
        try:
            x, y = float(a), float(b)
        except (TypeError, ValueError, OverflowError):
            return False
        if not (math.isfinite(x) and math.isfinite(y)):
            return x == y
        return math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return (len(a) == len(b)
                and all(_etost_json_equal(x, y) for x, y in zip(a, b)))
    if isinstance(a, dict) and isinstance(b, dict):
        return (set(a) == set(b)
                and all(_etost_json_equal(a[k], b[k]) for k in a))
    return a == b


def _etost_frozen_fields(obj, *, top=True):
    """Names of frozen-decision fields smuggled into an artifact."""
    found = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in _ETOST_FROZEN_FIELDS:
                found.add(key)
            found.update(_etost_frozen_fields(value, top=False))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found.update(_etost_frozen_fields(value, top=False))
    return sorted(found)


def build_exact_tost(deltas, margin, conf=0.90, created_at=None):
    """Build a replayable ``exact_tost_artifact`` around one test.

    Non-finite/boolean/string deltas are rejected here (rather than dropped),
    because an artifact's content digest must be a function of exactly the
    recorded inputs.  A genuinely computable test with fewer than two finite
    seeds is also refused.
    """
    if isinstance(deltas, (str, bytes, bytearray, dict)):
        raise ValueError('deltas must be an iterable of finite numbers')
    try:
        raw = list(deltas if deltas is not None else [])
    except TypeError:
        raise ValueError('deltas must be an iterable of finite numbers') from None
    vals = []
    for x in raw:
        f = _finite_scalar(x)  # noqa: F821
        if f is None:
            raise ValueError(f'deltas contain a non-finite/non-numeric value: {x!r}')
        vals.append(float(f))
    if len(vals) < 2:
        raise ValueError(f'exact TOST needs at least two finite deltas, got {len(vals)}')
    m = _etost_number(margin, 'margin', positive=True)
    c = _etost_number(conf, 'conf')
    if m is None:
        raise ValueError(f'margin={margin!r} must be finite and > 0')
    if c is None or not (0.0 < c < 1.0):
        raise ValueError(f'conf={conf!r} must be a finite probability in (0,1)')
    result = exact_signflip_tost(vals, margin=m, conf=c)
    if result.get('status') != 'ok':
        raise ValueError('exact TOST inputs are not computable: '
                         + (result.get('reason') or 'unknown reason'))
    artifact = {
        'kind': 'exact_tost_artifact',
        'schema': 1,
        'created_at': (created_at if created_at is not None
                       else time.strftime('%Y-%m-%dT%H:%M:%S')),
        'inputs': {'deltas': vals, 'margin': float(m), 'conf': float(c)},
        'result': result,
    }
    artifact['content_sha256'] = _sha256_json(  # noqa: F821
        {k: v for k, v in artifact.items()
         if k not in ('created_at', 'content_sha256')})
    return artifact


def validate_exact_tost(artifact):
    """Replay a stored exact-TOST artifact and compare every derived field."""
    out = {'kind': 'exact_tost_validation', 'schema': 1,
           'status': 'fail', 'problems': [], 'warnings': [],
           'content_sha256': None, 'result': None}
    if not isinstance(artifact, dict):
        out['problems'].append('artifact must be a JSON object')
        return out
    if artifact.get('kind') != 'exact_tost_artifact':
        out['problems'].append(
            f'artifact.kind must be "exact_tost_artifact", got '
            f'{artifact.get("kind")!r}')
    schema = artifact.get('schema')
    if isinstance(schema, bool) or not isinstance(schema, int) or schema != 1:
        out['problems'].append(f'artifact.schema must be 1, got {schema!r}')
    stolen = _etost_frozen_fields(artifact)
    if stolen:
        out['problems'].append(
            'artifact carries frozen-decision field(s) that a descriptive '
            f'readout must not contain: {stolen}')
    inputs = artifact.get('inputs')
    if not isinstance(inputs, dict):
        out['problems'].append('artifact.inputs must be an object')
        inputs = {}
    missing = [k for k in _ETOST_INPUT_KEYS if k not in inputs]
    if missing:
        out['problems'].append(
            f'artifact.inputs is missing required key(s): {missing}')
    extra = sorted(set(inputs) - set(_ETOST_INPUT_KEYS))
    if extra:
        out['problems'].append(
            'artifact.inputs carries field(s) outside the replay schema: '
            + ', '.join(repr(k) for k in extra))
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
        replay = exact_signflip_tost(
            inputs.get('deltas'), margin=inputs.get('margin'),
            conf=inputs.get('conf'))
    except Exception as e:
        out['problems'].append(f'replay raised {type(e).__name__}: {e}')
        return out
    out['result'] = replay
    if replay.get('status') != 'ok':
        out['problems'].append(
            'replayed inputs describe a non-estimable exact TOST: '
            + (replay.get('reason') or 'no readout'))
        return out
    stored = artifact.get('result')
    if not isinstance(stored, dict):
        out['problems'].append('artifact.result must be an object')
        return out
    extra_result = sorted(set(stored) - set(replay))
    if extra_result:
        out['problems'].append(
            'result carries field(s) outside the replay schema: '
            + ', '.join(repr(k) for k in extra_result))
    missing_result = sorted(set(replay) - set(stored))
    if missing_result:
        out['problems'].append(
            f'result is missing replay field(s): {missing_result}')
    for key in sorted(set(stored) & set(replay)):
        if not _etost_json_equal(stored.get(key), replay.get(key)):
            out['problems'].append(
                f'result.{key}: stored={stored.get(key)!r}, '
                f'replay={replay.get(key)!r}')
    if not out['problems']:
        out['status'] = 'ok'
    return out


def write_exact_tost(path, artifact, *, indent=2):
    """Atomically write an exact-TOST artifact; returns the path."""
    atomic_write_json(path, artifact, indent=indent)  # noqa: F821
    return path


def _exact_tost_companion(deltas, *, margin, conf):
    """Best-effort companion block for an existing TOST result.

    The caller (``tost_paired`` / ``confirmation_practical_significance``)
    owns the t-based decision.  This helper never raises: a malformed margin
    or panel becomes an explicit ``not_estimable`` block instead of breaking
    the historical result.
    """
    try:
        return exact_signflip_tost(deltas, margin=margin, conf=conf)
    except Exception as e:
        return {
            'kind': 'exact_signflip_tost', 'schema': 1,
            'status': 'not_estimable',
            'reason': f'{type(e).__name__}: {e}',
            'equivalent': None,
            'note': ('the distribution-free companion could not be computed; '
                     'this does not change the t-TOST decision'),
        }
