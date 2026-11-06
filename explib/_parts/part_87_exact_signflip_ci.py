"""Exact sign-flip confidence set for a paired mean effect.

The frozen confirmation decision is an exact two-sided sign-flip test on the
per-seed paired deltas.  The interval reported next to that p-value is, by
contrast, a Student-t interval (``mean +/- t * sd / sqrt(n)``).  Those two
objects answer the same question about the same estimand under different
assumptions, and they can disagree about whether zero is in the interval:

    mean = +0.4639, sd = 0.7266, n = 12, t 95% CI = [+0.0022, +0.9256]
    exact sign-flip p = 0.0518 > 0.05

The t interval excludes zero while the exact test does not reject.  Reporting
the t interval beside the exact p therefore invites a reader to "see" a
significant result that the pre-registered test did not produce.

This fragment closes that gap by inverting the SAME exact sign-flip test.  For
a candidate location ``mu`` the two-sided p-value is the fraction of the
``2**n`` sign assignments whose flipped sum is at least as large as the
observed shifted sum.  The confidence set at level ``1 - alpha`` is

    A = { mu : p(mu) >= alpha }.

Because the p-value is piecewise constant and each sign pattern contributes a
closed interval of accepted ``mu`` (the inequality is a downward-opening
quadratic in ``mu``), ``A`` is computed exactly with integer/Fraction
arithmetic and a boundary sweep.  Zero is in the set iff the exact rational
test does not reject.  ``exact_signflip_p`` is the same test implemented with
a documented ``1e-12`` tie tolerance on unit-scaled floats; only at an exact
floating-point tie can the two counts differ by one pattern, and
``p_consistency`` names that instead of hiding it.  The interval is
distribution-free under the same symmetry assumption as the sign-flip test,
and is conservative (never narrower than the test's resolution).

SCOPE: descriptive interval for the paired mean effect.  It reads no split,
trains nothing, produces no new p-value for the frozen Holm family, and never
changes ``p_exact_signflip`` / ``p_holm`` / ``verified`` / ``confirmed``.  The
t interval is reported next to it, not replaced, and a disagreement is named
rather than hidden.
"""

import hashlib as _esci_hashlib
import itertools as _esci_itertools
import json as _esci_json
import math as _esci_math
import time as _esci_time
from fractions import Fraction as _EsciFraction

import numpy as np


EXACT_SIGNFLIP_CI_KIND = 'exact_signflip_ci'
EXACT_SIGNFLIP_CI_BLOCK_KIND = 'exact_signflip_ci_block'
EXACT_SIGNFLIP_CI_RESULT_KIND = 'exact_signflip_ci_result'
EXACT_SIGNFLIP_CI_SCHEMA = 1
EXACT_SIGNFLIP_CI_MAX_N = 16
EXACT_SIGNFLIP_CI_HARD_MAX_N = 18
# Seed labels are identities, not measurements, but the artifact is strict
# JSON.  Keep them inside the range CPython can spell (the same bound the
# other scale-safe artifacts use) so a hash/replay diagnostic is possible.
EXACT_SIGNFLIP_CI_MAX_REPORTABLE_INT = 10 ** 4000

EXACT_SIGNFLIP_CI_INPUT_KEYS = ('deltas', 'seeds', 'alpha', 'max_n')
EXACT_SIGNFLIP_CI_ARTIFACT_KEYS = (
    'kind', 'schema', 'created_at', 'source', 'inputs', 'result',
    'content_sha256',
)
EXACT_SIGNFLIP_CI_BLOCK_KEYS = ('kind', 'inputs', 'result')
EXACT_SIGNFLIP_CI_FORBIDDEN_KEYS = (
    'confirmed', 'p_holm', 'verified', 'verdict', 'decision',
)

EXACT_SIGNFLIP_CI_POLICY = {
    'version': 'exact-signflip-ci-1',
    'method': ('two-sided sign-flip test inversion over all 2**n sign '
               'patterns, exact integer/Fraction arithmetic and a boundary '
               'sweep'),
    'assumption': (
        'paired differences are symmetric about the true paired mean; this is '
        'the same assumption the exact sign-flip p-value already relies on.  '
        'The interval is not distribution-free without symmetry: skewed or '
        'heavy-tailed errors can undercover.'),
    'coverage': (
        'test inversion gives finite-sample coverage at least 1 - alpha under '
        'the symmetry assumption; discreteness makes it conservative, so the '
        'attained coverage can be strictly larger than the nominal level'),
    'role': (
        'descriptive confidence set consistent with the exact sign-flip test; '
        'it adds no p-value to the frozen Holm family and never changes '
        'p_exact_signflip, p_holm, verified or confirmed'),
    'boundary': (
        'the inversion is exact only for n <= max_n (default 16, hard cap 18); '
        'above that the exact p itself is a different algorithm and this block '
        'returns not_estimable rather than silently substituting a t interval. '
        'A zero-width set on a constant panel is a discreteness artifact, not '
        'infinite precision.  The set is a confidence set for the mean, not a '
        'prediction interval for a new seed, not a posterior probability, and '
        'not a repair for selection, missing seeds or a non-exchangeable '
        'panel.  The t interval is still shown for transparency; when the two '
        'disagree about zero the exact set is the one consistent with the '
        'exact rational test, and the disagreement is named.  Only at an '
        'exact floating-point tie can the float enumeration used by '
        'exact_signflip_p differ by one pattern; p_consistency names it.'),
}
EXACT_SIGNFLIP_CI_STATUSES = ('ok', 'warnings', 'not_estimable')


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------

def _esci_repr(value, limit=200):
    try:
        text = repr(value)
    except Exception:
        return f'<{type(value).__name__}>'
    return text if len(text) <= limit else text[:limit] + '...'


def _esci_key_text(value, limit=120):
    """Bounded text for a dict key without formatting a huge integer."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit - 3] + '...'
    try:
        text = str(value)
    except Exception:
        return _esci_repr(value, limit=limit)
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _esci_json_key(value):
    """JSON-object key: preserve strings, safely render anything else."""
    return value if isinstance(value, str) else _esci_key_text(value)


def _esci_strict_json_problem(label, body):
    """Return a bounded diagnostic when ``body`` is not strict JSON."""
    try:
        _esci_json.dumps(body, allow_nan=False)
    except Exception as exc:
        detail = str(exc)
        if ('Exceeds the limit' in detail
                or 'integer string conversion' in detail):
            return (f'{label} carries an unrepresentable int (beyond '
                    "CPython's decimal-conversion limit) and cannot be "
                    'strict-JSON serialised')
        return (f'{label} is not strict-JSON serialisable: '
                f'{type(exc).__name__}: {_esci_repr(exc)}')
    return None


def _esci_number(value, name, *, positive=False):
    """Finite float, or ``None`` for bool/str/NaN/inf (and non-positive)."""
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not _esci_math.isfinite(out):
        return None
    if positive and out <= 0.0:
        return None
    return out


def _esci_alpha(value):
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray)):
        raise ValueError(f'alpha must be a real number, got {_esci_repr(value)}')
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'alpha must be a real number, got {_esci_repr(value)}') from None
    if not _esci_math.isfinite(out) or not (0.0 < out < 1.0):
        raise ValueError(f'alpha must be finite and in (0,1), got {out!r}')
    return out


def _esci_max_n(value):
    if value is None:
        return EXACT_SIGNFLIP_CI_MAX_N
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'max_n must be an integer, got {_esci_repr(value)}')
    out = int(value)
    if out < 2:
        raise ValueError(f'max_n must be >= 2, got {_esci_repr(out)}')
    if out > EXACT_SIGNFLIP_CI_HARD_MAX_N:
        raise ValueError(
            f'max_n must be <= {EXACT_SIGNFLIP_CI_HARD_MAX_N} '
            f'(the 2**n boundary sweep), got {_esci_repr(out)}')
    return out


def _esci_seed(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        return None
    return int(value)


def _esci_inputs(deltas, seeds=None, *, strict=False, where='deltas'):
    """Normalise ``deltas`` and optional seed labels.

    ``strict=False`` drops non-finite entries by position (the direct readout
    names them); ``strict=True`` refuses them, which is what the replayable
    artifact builder uses so the input cannot contain a value that strict JSON
    cannot represent.
    """
    if isinstance(deltas, (str, bytes, bytearray, dict)):
        raise ValueError(
            f'{where} must be an iterable of real numbers, not a '
            f'{type(deltas).__name__}')
    try:
        raw = list(deltas if deltas is not None else [])
    except TypeError:
        raise ValueError(f'{where} must be an iterable') from None
    n_input = len(raw)
    vals = []
    kept = []
    dropped = []
    for i, x in enumerate(raw):
        f = _esci_number(x, f'{where}[{i}]')
        if f is None:
            if strict:
                raise ValueError(
                    f'{where}[{i}] is not a finite real number '
                    f'({_esci_repr(x)}); a replayable artifact must be strict JSON')
            dropped.append(i)
            continue
        vals.append(float(f))
        kept.append(i)
    seeds_out = None
    dropped_seeds = []
    if seeds is not None:
        if isinstance(seeds, (str, bytes, bytearray, dict)):
            raise ValueError(
                'seeds must be an iterable of integers, not a '
                f'{type(seeds).__name__}')
        try:
            seed_in = list(seeds)
        except TypeError:
            raise ValueError('seeds must be an iterable') from None
        if len(seed_in) != n_input:
            raise ValueError(
                f'seeds length {len(seed_in)} does not match deltas length '
                f'{n_input}')
        norm = [_esci_seed(s) for s in seed_in]
        if any(s is None for s in norm):
            raise ValueError(f'seeds must be integers, got {_esci_repr(seed_in)}')
        for index, seed_value in enumerate(norm):
            if abs(seed_value) > EXACT_SIGNFLIP_CI_MAX_REPORTABLE_INT:
                raise ValueError(
                    f'seeds[{index}] is outside the strict-JSON representable '
                    f'integer range +/-{_esci_repr(EXACT_SIGNFLIP_CI_MAX_REPORTABLE_INT)}; '
                    f'got {_esci_repr(seed_value)}')
        if len(set(norm)) != len(norm):
            raise ValueError(f'seeds contain duplicates: {_esci_repr(norm)}')
        seeds_out = [norm[i] for i in kept]
        dropped_seeds = [norm[i] for i in dropped]
    return vals, seeds_out, dropped, dropped_seeds, n_input


def _esci_p_floor(n_nonzero):
    if n_nonzero <= 0:
        return 1.0
    return min(1.0, _esci_math.ldexp(2.0, -int(n_nonzero)))


def _esci_min_n_for_alpha(alpha):
    """Smallest n whose all-nonzero sign-flip p floor is <= alpha."""
    a = _EsciFraction(alpha)
    n = 1
    while n < 20000:
        if _EsciFraction(2, 1 << n) <= a:
            return n
        n += 1
    return None


def _esci_attainable_count(alpha, n):
    """Smallest integer count k with k / 2**n >= alpha (exact)."""
    a = _EsciFraction(alpha)
    total = 1 << int(n)
    return max(1, (a.numerator * total + a.denominator - 1) // a.denominator)


def _esci_scale_to_integers(vals):
    """Common-denominator integer representation of the float deltas."""
    fr = [_EsciFraction(float(x)) for x in vals]
    denom = 1
    for f in fr:
        denom = denom * f.denominator // _esci_math.gcd(denom, f.denominator)
    xi = [int(f * denom) for f in fr]
    return xi, denom


def _esci_intervals(xi, denom, alpha):
    """Return (intervals, meta) for A = {mu: p(mu) >= alpha}.

    ``intervals`` is a list of closed ``[Fraction, Fraction]`` pairs, or
    ``None`` when the acceptance set is the whole real line (which happens
    exactly when the test's p floor already exceeds alpha).  ``meta`` records
    the enumeration size and the exact count threshold.
    """
    n = len(xi)
    total = 1 << n
    k = _esci_attainable_count(alpha, n)
    meta = {'n_patterns': int(total), 'min_count': int(k),
            'all_real': False, 'n_intervals': 0, 'n_boundaries': 0}
    # The all-plus and all-minus patterns satisfy the inequality for every mu
    # (their null sum is the observed sum).  If k <= 2 the whole line is
    # accepted, so there is no finite set to report.
    if k <= 2:
        meta['all_real'] = True
        return None, meta
    S = sum(xi)
    intervals = []
    for signs in _esci_itertools.product((-1, 1), repeat=n):
        m = sum(signs)
        if m == n or m == -n:
            # all-plus / all-minus: Q(mu) == 0 for every mu
            continue
        ss = 0
        for s, x in zip(signs, xi):
            ss += s * x
        r1 = _EsciFraction(S - ss, (n - m) * denom)
        r2 = _EsciFraction(S + ss, (n + m) * denom)
        if r1 <= r2:
            intervals.append((r1, r2))
        else:
            intervals.append((r2, r1))
    meta['n_intervals'] = int(len(intervals))
    starts = {}
    ends = {}
    for lo, hi in intervals:
        starts[lo] = starts.get(lo, 0) + 1
        ends[hi] = ends.get(hi, 0) + 1
    boundaries = sorted(set(starts) | set(ends))
    meta['n_boundaries'] = int(len(boundaries))
    active = 2  # all-plus and all-minus cover every mu
    accepted = []
    for i, v in enumerate(boundaries):
        left = starts.get(v, 0)
        right = ends.get(v, 0)
        # ``active`` counts intervals with lo < v and hi >= v at this point.
        if active + left >= k:
            accepted.append((v, v))
        if (active + left - right) >= k and i + 1 < len(boundaries):
            accepted.append((v, boundaries[i + 1]))
        active = active + left - right
    if not accepted:
        # p(mean) == 1 always, so this is defensive only; return a point at the
        # observed mean rather than an empty set that would be read as a
        # zero-width certainty.
        mean = _EsciFraction(sum(xi), n * denom)
        accepted = [(mean, mean)]
    accepted.sort()
    merged = []
    for lo, hi in accepted:
        if not merged or lo > merged[-1][1]:
            merged.append([lo, hi])
        elif hi > merged[-1][1]:
            merged[-1][1] = hi
    return merged, meta


def _esci_p_at(xi, denom, value):
    """Exact two-sided sign-flip p for H0: mean == value."""
    n = len(xi)
    mu = _EsciFraction(value)
    shifted = [_EsciFraction(x, denom) - mu for x in xi]
    observed = abs(sum(shifted))
    count = 0
    for signs in _esci_itertools.product((-1, 1), repeat=n):
        total = 0
        for s, e in zip(signs, shifted):
            total += s * e
        if abs(total) >= observed:
            count += 1
    return _EsciFraction(count, 1 << n)


def _esci_t_interval(vals, alpha):
    n = len(vals)
    if n < 2:
        return None
    try:
        mean = float(_scaled_mean_or_raw(vals))  # noqa: F821 (part 32)
    except Exception:
        return None
    if not _esci_math.isfinite(mean):
        # A centre beyond float64 cannot have representable endpoints.
        return None
    try:
        scaled = _scaled_mean_std_if_extreme(vals, ddof=1)  # noqa: F821 (part 00)
        if scaled is not None:
            sd = float(scaled[1])
        else:
            var = _esci_math.fsum((float(x) - mean) ** 2 for x in vals) / (n - 1)
            sd = _esci_math.sqrt(var)
    except (OverflowError, ValueError):
        return None
    if not _esci_math.isfinite(sd):
        return None
    try:
        tc = _t_critical(1.0 - alpha, n - 1)  # noqa: F821 (part 10)
    except Exception:
        tc = None
    if tc is None or not _esci_math.isfinite(float(tc)):
        return None
    half = float(tc) * sd / _esci_math.sqrt(n)
    if not _esci_math.isfinite(half):
        return None
    lo, hi = mean - half, mean + half
    # ``half`` can be finite while ``mean + half`` (or the lower endpoint)
    # overflows: an extreme mean near float64's limit plus a wide interval.
    # Returning [..., inf] used to leak a non-strict-JSON value into the
    # unit-split artifact; withhold the whole interval, exactly as the other
    # t-interval helpers in this repository do.
    if not (_esci_math.isfinite(lo) and _esci_math.isfinite(hi)):
        return None
    return {
        'method': 'student-t (df=n-1); normality approximation',
        'conf': float(1.0 - alpha),
        'df': int(n - 1),
        'sd': float(sd),
        'ci': [float(lo), float(hi)],
    }


def _esci_contains_zero(intervals):
    if intervals is None:
        return True
    zero = _EsciFraction(0)
    for lo, hi in intervals:
        if lo <= zero <= hi:
            return True
    return False


def _esci_relation(exact_contains, t_ci):
    if t_ci is None:
        return 't_unavailable'
    t_lo, t_hi = t_ci
    t_contains = bool(t_lo <= 0.0 <= t_hi)
    if exact_contains and t_contains:
        return 'both_include_zero'
    if (not exact_contains) and (not t_contains):
        return 'both_exclude_zero'
    if exact_contains and (not t_contains):
        return 't_excludes_exact_includes'
    return 't_includes_exact_excludes'


def _esci_mean_median(vals):
    n = len(vals)
    if n == 0:
        return None, None
    try:
        mean = float(_scaled_mean_or_raw(vals))  # noqa: F821 (part 32)
    except Exception:
        mean = None
    try:
        median = float(_scaled_median_or_raw(vals))  # noqa: F821 (part 32)
    except Exception:
        median = None
    return mean, median


def _esci_jsonable(value):
    if isinstance(value, dict):
        return {_esci_json_key(k): _esci_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_esci_jsonable(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        out = int(value)
        if not isinstance(value, (bool, np.bool_)):
            try:
                str(out)
            except (ValueError, OverflowError):
                # Keep a tampered/legacy artifact hashable so the validator
                # can name the unrepresentable int instead of crashing while
                # recomputing the content digest.
                return {'__unprintable_int_hex__': hex(out)}
        return out
    if isinstance(value, (float, np.floating)):
        out = float(value)
        return out if _esci_math.isfinite(out) else None
    return value


def _esci_content_sha(artifact):
    body = {k: v for k, v in artifact.items()
            if k not in ('created_at', 'content_sha256')}
    text = _esci_json.dumps(_esci_jsonable(body), sort_keys=True,
                            separators=(',', ':'), ensure_ascii=False,
                            allow_nan=False)
    return _esci_hashlib.sha256(text.encode('utf-8')).hexdigest()


def _esci_forbidden_paths(obj, path=''):
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = _esci_key_text(key)
            here = f'{path}.{name}' if path else name
            if name in EXACT_SIGNFLIP_CI_FORBIDDEN_KEYS:
                found.append(here)
            found.extend(_esci_forbidden_paths(value, here))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            found.extend(_esci_forbidden_paths(value, f'{path}[{index}]'))
    return found


def _esci_compare(a, b, path='result', *, rtol=1e-9, atol=1e-12, diffs=None):
    if diffs is None:
        diffs = []
    if isinstance(a, bool) or isinstance(b, bool):
        # A boolean is a JSON value of its own; ``bool(np.array([...]))`` is
        # ambiguous and must not replace the comparison with an exception.
        if not (isinstance(a, bool) and isinstance(b, bool) and a is b):
            diffs.append(f'{path}: {_esci_repr(a)} != {_esci_repr(b)}')
        return diffs
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        try:
            same = _esci_math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol)
        except (TypeError, ValueError, OverflowError):
            same = (a == b)
        if not same:
            diffs.append(f'{path}: {_esci_repr(a)} != {_esci_repr(b)}')
        return diffs
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            diffs.append(
                f'{path}: keys {sorted(map(_esci_key_text, a))} != '
                f'{sorted(map(_esci_key_text, b))}')
            return diffs
        for k in a:
            _esci_compare(a[k], b[k], f'{path}.{_esci_key_text(k)}',
                          rtol=rtol, atol=atol, diffs=diffs)
        return diffs
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diffs.append(f'{path}: length {len(a)} != {len(b)}')
            return diffs
        for i, (x, y) in enumerate(zip(a, b)):
            _esci_compare(x, y, f'{path}[{i}]', rtol=rtol, atol=atol,
                          diffs=diffs)
        return diffs
    # A tampered artifact can put an array/object where replay produced a
    # scalar (or vice versa).  ``a != b`` then yields an array whose truth
    # value is ambiguous, so collapse any non-scalar comparison to "different".
    try:
        same = bool(a == b)
    except (TypeError, ValueError, OverflowError):
        same = False
    if not same:
        diffs.append(f'{path}: {_esci_repr(a)} != {_esci_repr(b)}')
    return diffs


def _esci_equal(a, b):
    """True when two JSON values match, without raising on an array scalar.

    The full comparator already collapses an array/scalar mismatch to
    "different"; reuse it so the schema checks cannot be the one place where
    ``bool(a == b)`` escapes as an ambiguous-truth ``ValueError``.
    """
    return not _esci_compare(a, b, 'value')


def _esci_inputs_problems(inputs):
    problems = []
    if not isinstance(inputs, dict):
        return ['inputs must be an object']
    extra = sorted(set(inputs) - set(EXACT_SIGNFLIP_CI_INPUT_KEYS),
                   key=_esci_key_text)
    missing = [k for k in EXACT_SIGNFLIP_CI_INPUT_KEYS if k not in inputs]
    if extra or missing:
        problems.append(
            f'inputs keys differ: missing={missing}, '
            f'extra={[_esci_key_text(k) for k in extra]}')
        return problems
    try:
        alpha = _esci_alpha(inputs.get('alpha'))
    except ValueError as exc:
        problems.append(f'inputs.alpha is malformed: {exc}')
        alpha = None
    try:
        max_n = _esci_max_n(inputs.get('max_n'))
    except ValueError as exc:
        problems.append(f'inputs.max_n is malformed: {exc}')
        max_n = None
    try:
        _esci_inputs(inputs.get('deltas'), inputs.get('seeds'), strict=True,
                     where='inputs.deltas')
    except ValueError as exc:
        problems.append(f'inputs.deltas/seeds are malformed: {exc}')
    return problems


def _esci_result_problems(result):
    problems = []
    if not isinstance(result, dict):
        return ['result must be an object']
    if not _esci_equal(result.get('kind'), EXACT_SIGNFLIP_CI_RESULT_KIND):
        problems.append(
            f"result.kind must be {EXACT_SIGNFLIP_CI_RESULT_KIND!r}, got "
            f'{_esci_repr(result.get("kind"))}')
    if not _esci_equal(result.get('version'),
                       EXACT_SIGNFLIP_CI_POLICY['version']):
        problems.append(
            f"result.version must be {EXACT_SIGNFLIP_CI_POLICY['version']!r}, "
            f'got {_esci_repr(result.get("version"))}')
    status = result.get('status')
    if not any(_esci_equal(status, value)
               for value in EXACT_SIGNFLIP_CI_STATUSES):
        problems.append(
            f'result.status must be one of {EXACT_SIGNFLIP_CI_STATUSES}, '
            f'got {_esci_repr(status)}')
        return problems
    if status == 'not_estimable':
        reason = result.get('reason')
        if not isinstance(reason, str) or not reason.strip():
            problems.append('a not_estimable result needs a non-empty reason')
        return problems
    if result.get('unbounded') is True:
        if result.get('ci') is not None:
            problems.append('an unbounded result must have ci=null')
        if result.get('intervals') not in (None, []):
            problems.append('an unbounded result must have no finite intervals')
        return problems
    ci = result.get('ci')
    intervals = result.get('intervals')
    if not isinstance(intervals, list) or not intervals:
        problems.append('a finite result needs a non-empty intervals list')
        return problems
    for i, item in enumerate(intervals):
        if (not isinstance(item, (list, tuple)) or len(item) != 2
                or any(_esci_number(x, f'intervals[{i}]') is None for x in item)):
            problems.append(f'intervals[{i}] must be a finite [lo, hi] pair')
            return problems
        if float(item[0]) > float(item[1]):
            problems.append(f'intervals[{i}] must be ordered [lo <= hi]')
            return problems
    if ci is not None:
        if (not isinstance(ci, (list, tuple)) or len(ci) != 2
                or any(_esci_number(x, 'ci') is None for x in ci)):
            problems.append('ci must be null or a finite [lo, hi] pair')
            return problems
        if float(ci[0]) > float(ci[1]):
            problems.append('ci must be ordered [lo <= hi]')
            return problems
    p0 = result.get('p_at_zero')
    if _esci_number(p0, 'p_at_zero') is None or not (0.0 <= float(p0) <= 1.0):
        problems.append('p_at_zero must be a finite probability')
    return problems


# ---------------------------------------------------------------------------
# public estimator
# ---------------------------------------------------------------------------

def exact_signflip_location_p(deltas, value=0.0):
    """Exact two-sided sign-flip p for H0: paired mean == ``value``.

    This is the test that :func:`exact_signflip_ci` inverts.  At ``value=0``
    it is the same statistic as :func:`exact_signflip_p` (up to the float
    enumeration's tie tolerance), which lets the CI report its own p at zero
    and lets an auditor query any candidate endpoint.
    """
    vals, _seeds, dropped, _dropped_seeds, n_input = _esci_inputs(
        deltas, strict=False, where='deltas')
    n = len(vals)
    val = _esci_number(value, 'value')
    if val is None:
        raise ValueError(f'value must be a finite real number, got {_esci_repr(value)}')
    if n == 0:
        return {'p': None, 'n': 0, 'n_nonzero': 0, 'n_effective': 0,
                'value': float(val),
                'n_input': n_input, 'n_dropped': len(dropped),
                'dropped_positions': dropped, 'method': 'none'}
    mu = _EsciFraction(val)
    # At the zero null an exactly-zero delta is sign-neutral: both of its
    # signs contribute the same shifted sum, so removing those columns leaves
    # the rational p unchanged.  Use that effective width for the enumeration
    # cap so a panel padded with unused seed slots is not refused.  For
    # ``value != 0`` a zero delta is no longer neutral after centring, so the
    # full declared width is required.
    n_nonzero = int(sum(1 for x in vals if x != 0.0))
    if mu == 0:
        calc_vals = [x for x in vals if x != 0.0]
        n_effective = n_nonzero
    else:
        calc_vals = vals
        n_effective = n
    if n_effective > EXACT_SIGNFLIP_CI_HARD_MAX_N:
        raise ValueError(
            f'n_effective={n_effective} (of n={n} declared) exceeds the exact '
            f'location-p inversion hard cap {EXACT_SIGNFLIP_CI_HARD_MAX_N}; '
            'enumerating 2**n sign patterns would be unbounded')
    xi, denom = _esci_scale_to_integers(calc_vals)
    p = _esci_p_at(xi, denom, mu)
    return {
        'p': float(p),
        'p_exact': f'{p.numerator}/{p.denominator}',
        'n': int(n),
        'n_nonzero': n_nonzero,
        'n_effective': int(n_effective),
        'value': float(val),
        'n_input': int(n_input),
        'n_dropped': int(len(dropped)),
        'dropped_positions': list(dropped),
        'method': ('exact sign-flip test inversion '
                   '(2**n_effective enumeration)'),
    }


def exact_signflip_ci(deltas, *, alpha=0.05, max_n=None, seeds=None):
    """Exact sign-flip confidence set for the paired mean effect.

    Returns a strict-JSON dict.  ``ci`` is the single finite interval when the
    acceptance set is connected, else ``None`` and ``intervals`` carries the
    union.  ``unbounded`` means the p floor already exceeds ``alpha`` at this
    n, so no finite interval can be claimed.  ``status='warnings'`` is a
    successful computation with a named caveat (dropped values, a degenerate
    zero-width panel, a t-vs-exact disagreement, or a disconnected set).
    """
    alpha = _esci_alpha(alpha)
    max_n = _esci_max_n(max_n)
    vals, seeds_out, dropped, dropped_seeds, n_input = _esci_inputs(
        deltas, seeds, strict=False, where='deltas')
    n = len(vals)
    base = {
        'kind': EXACT_SIGNFLIP_CI_RESULT_KIND,
        'version': EXACT_SIGNFLIP_CI_POLICY['version'],
        'method': EXACT_SIGNFLIP_CI_POLICY['method'],
        'assumption': EXACT_SIGNFLIP_CI_POLICY['assumption'],
        'coverage': EXACT_SIGNFLIP_CI_POLICY['coverage'],
        'role': EXACT_SIGNFLIP_CI_POLICY['role'],
        'boundary': EXACT_SIGNFLIP_CI_POLICY['boundary'],
        'alpha': float(alpha),
        'conf': float(1.0 - alpha),
        'max_n': int(max_n),
        'n_input': int(n_input),
        'n': int(n),
        'n_dropped': int(len(dropped)),
        'dropped_positions': list(dropped),
        'dropped_seeds': list(dropped_seeds),
        'seeds': seeds_out,
        'warnings': [],
    }
    if n == 0:
        base.update({
            'status': 'not_estimable',
            'reason': 'no finite paired deltas remain',
            'ci': None, 'intervals': [], 'unbounded': False,
            'connected': None, 'degenerate': None, 'width': None,
            'contains_zero': None, 'p_at_zero': None,
            'mean': None, 'median': None, 'std': None,
            't_interval': None, 'zero_relation': None,
            'p_consistency': None, 'resolution': None,
            'enumeration': None,
        })
        return _esci_jsonable(base)
    if n > max_n:
        base.update({
            'status': 'not_estimable',
            'reason': (f'n={n} exceeds the exact inversion cap max_n={max_n}; '
                       'the exact sign-flip p itself changes algorithm above '
                       'the cap, so no exact CI is substituted'),
            'ci': None, 'intervals': [], 'unbounded': False,
            'connected': None, 'degenerate': None, 'width': None,
            'contains_zero': None, 'p_at_zero': None,
            'mean': None, 'median': None, 'std': None,
            't_interval': None, 'zero_relation': None,
            'p_consistency': None, 'resolution': None,
            'enumeration': None,
        })
        return _esci_jsonable(base)
    n_nonzero = int(sum(1 for x in vals if x != 0.0))
    # Two different floors matter here and must not be conflated:
    #   * p_floor is the resolution of the test AT ZERO.  A zero delta is
    #     sign-invariant there, so it multiplies every attainable count by
    #     the same 2**(number of zeros) and the relative floor is 2/2**n_nonzero.
    #   * tail_floor is the value of p(mu) as |mu| -> infinity.  A zero delta
    #     is *not* sign-invariant after centering at mu != 0, so only the
    #     all-plus and all-minus patterns remain and the floor is 2/2**n.
    # The acceptance set {mu: p(mu) >= alpha} is the whole real line exactly
    # when tail_floor >= alpha.  Deciding unboundedness from p_floor would
    # wrongly return "unbounded" for a panel with some zero deltas even though
    # values far from the observed mean are rejected.
    p_floor = _esci_p_floor(n_nonzero)
    tail_floor = _esci_p_floor(n)
    # An all-zero panel carries no non-zero paired difference.  Keep the
    # documented conservative policy (no finite interval) even though the
    # exact acceptance set degenerates to a single point; a zero-width set
    # would otherwise be read as infinite precision.
    unbounded = bool(n_nonzero == 0 or tail_floor >= alpha)
    min_n = _esci_min_n_for_alpha(alpha)
    mean, median = _esci_mean_median(vals)
    sd = None
    if n > 1:
        try:
            scaled = _scaled_mean_std_if_extreme(vals, ddof=1)  # noqa: F821 (part 00)
            if scaled is not None:
                sd = float(scaled[1])
            elif mean is not None:
                sd = float(_esci_math.sqrt(
                    _esci_math.fsum((float(x) - float(mean)) ** 2 for x in vals)
                    / (n - 1)))
        except (OverflowError, ValueError):
            sd = None
    if sd is not None and not _esci_math.isfinite(sd):
        sd = None
    base.update({
        'n_nonzero': n_nonzero,
        'mean': float(mean) if mean is not None else None,
        'median': float(median) if median is not None else None,
        'std': sd,
        'resolution': {
            'p_floor': float(p_floor),
            'tail_p_floor': float(tail_floor),
            'alpha': float(alpha),
            # ``attainable`` keeps its historical meaning: can the test
            # reject at its zero null?  ``ci_attainable`` says whether the
            # inverted acceptance set is finite (equivalently, whether the
            # far-tail floor is below alpha).  They differ only when some
            # deltas are exactly zero.
            'attainable': bool(p_floor <= alpha),
            'tail_attainable': bool(tail_floor < alpha),
            'ci_attainable': bool(not unbounded),
            'min_n_for_alpha': (int(min_n) if min_n is not None else None),
            'n_nonzero': n_nonzero,
            'note': ('p_floor is the exact-test resolution at zero and uses '
                     'n_nonzero because a zero delta is sign-invariant there; '
                     'unboundedness uses tail_p_floor=2/2**n, the p(mu) limit '
                     'as |mu| grows, where zero deltas are no longer '
                     'sign-invariant'),
        },
    })
    if unbounded:
        p0 = _esci_p_at_zero_for(vals)
        t_int = _esci_t_interval(vals, alpha)
        base.update({
            'status': 'warnings',
            'ci': None,
            'intervals': [],
            'unbounded': True,
            'connected': None,
            'degenerate': None,
            'width': None,
            'bounding_interval': None,
            'p_at_zero': float(p0),
            'p_at_zero_exact': f'{p0.numerator}/{p0.denominator}',
            'contains_zero': True,
            't_interval': t_int,
            'zero_relation': {
                'exact_contains_zero': True,
                't_contains_zero': (None if t_int is None else bool(
                    t_int['ci'][0] <= 0.0 <= t_int['ci'][1])),
                'relation': _esci_relation(True, None if t_int is None else t_int['ci']),
                'consistent_with_p': True,
            },
            'p_consistency': _esci_p_consistency(vals, p0),
            'enumeration': {'n_patterns': int(1 << n),
                            'min_count': int(_esci_attainable_count(alpha, n)),
                            'all_real': True, 'n_intervals': 0,
                            'n_boundaries': 0},
            'warnings': [
                ((f'every paired delta is exactly zero at n={n}; the exact '
                  'acceptance set collapses to a single point, and a '
                  'zero-width confidence set is not claimed as infinite '
                  'precision (resolution limit)')
                 if n_nonzero == 0 else
                 (f'the exact test cannot reject any value at n={n} '
                  f'(far-tail p floor {tail_floor:.4g} >= alpha={alpha:.4g}); '
                  'no finite confidence set exists, which is a resolution '
                  'limit rather than evidence of no effect'))
            ],
        })
        return _esci_jsonable(base)
    xi, denom = _esci_scale_to_integers(vals)
    intervals, meta = _esci_intervals(xi, denom, alpha)
    p0 = _esci_p_at(xi, denom, 0.0)
    t_int = _esci_t_interval(vals, alpha)
    warnings = list(base['warnings'])
    if dropped:
        warnings.append(
            f'{len(dropped)} non-finite/boolean delta(s) dropped at positions '
            f'{dropped}; the interval is over the remaining n={n}')
    if intervals is None:
        # Defensive: _esci_intervals already handles the unbounded case via
        # p_floor, but keep a named branch rather than emitting a null set.
        base.update({
            'status': 'warnings',
            'ci': None, 'intervals': [], 'unbounded': True,
            'connected': None, 'degenerate': None, 'width': None,
            'bounding_interval': None,
            'p_at_zero': float(p0),
            'p_at_zero_exact': f'{p0.numerator}/{p0.denominator}',
            'contains_zero': True, 't_interval': t_int,
            'zero_relation': {
                'exact_contains_zero': True,
                't_contains_zero': (None if t_int is None else bool(
                    t_int['ci'][0] <= 0.0 <= t_int['ci'][1])),
                'relation': _esci_relation(True, None if t_int is None else t_int['ci']),
                'consistent_with_p': bool(p0 >= _EsciFraction(alpha)),
            },
            'p_consistency': _esci_p_consistency(vals, p0),
            'enumeration': meta,
            'warnings': warnings + [
                'the exact acceptance set is the whole real line; no finite '
                'confidence set exists at this n and alpha'],
        })
        return _esci_jsonable(base)
    float_intervals = [[float(lo), float(hi)] for lo, hi in intervals]
    bounding = [min(lo for lo, _ in float_intervals),
                max(hi for _, hi in float_intervals)]
    exact_contains = _esci_contains_zero(intervals)
    t_ci = None if t_int is None else t_int['ci']
    relation = _esci_relation(exact_contains, t_ci)
    connected = len(float_intervals) == 1
    degenerate = bool(connected and intervals[0][0] == intervals[0][1])
    if not connected:
        warnings.append(
            f'the exact acceptance set is disconnected ({len(float_intervals)} '
            'intervals); report the union, not only its convex hull')
    if degenerate:
        warnings.append(
            'the confidence set is a single point; a zero-width set (most '
            'often a constant/zero-variance panel) is a discreteness artifact, '
            'not infinite precision')
    if relation == 't_excludes_exact_includes':
        warnings.append(
            'the Student-t interval excludes zero while the exact sign-flip p '
            'does not reject; the t interval is anti-conservative on this '
            'panel, and the exact set is the interval consistent with the '
            'reported exact p')
    elif relation == 't_includes_exact_excludes':
        warnings.append(
            'the Student-t interval includes zero while the exact sign-flip p '
            'rejects; the t interval is conservative here, and the exact set '
            'is the interval consistent with the reported exact p')
    consistency = _esci_p_consistency(vals, p0)
    if not consistency.get('matches', True):
        warnings.append(
            'the exact p-at-zero does not equal exact_signflip_p within 1e-12; '
            'this can happen at an exact tie under the float enumeration\'s '
            'tie tolerance, and the exact inversion is authoritative')
    if not bool(exact_contains == (p0 >= _EsciFraction(alpha))):
        warnings.append(
            'internal inconsistency: zero membership and p-at-zero disagree; '
            'do not quote this block')
    base.update({
        'status': 'warnings' if warnings else 'ok',
        'ci': (float_intervals[0] if connected else None),
        'intervals': float_intervals,
        'unbounded': False,
        'connected': bool(connected),
        'degenerate': bool(degenerate),
        'width': float(bounding[1] - bounding[0]),
        'bounding_interval': [float(bounding[0]), float(bounding[1])],
        'p_at_zero': float(p0),
        'p_at_zero_exact': f'{p0.numerator}/{p0.denominator}',
        'contains_zero': bool(exact_contains),
        't_interval': t_int,
        'zero_relation': {
            'exact_contains_zero': bool(exact_contains),
            't_contains_zero': (None if t_ci is None else bool(t_ci[0] <= 0.0 <= t_ci[1])),
            'relation': relation,
            'consistent_with_p': bool(exact_contains == (p0 >= _EsciFraction(alpha))),
        },
        'p_consistency': consistency,
        'enumeration': meta,
        'warnings': warnings,
    })
    return _esci_jsonable(base)


def _esci_p_at_zero_for(vals):
    xi, denom = _esci_scale_to_integers(vals)
    return _esci_p_at(xi, denom, 0.0)


def _esci_p_consistency(vals, p0):
    try:
        ref = exact_signflip_p(vals)  # noqa: F821 (part 09)
    except Exception:
        return {'p_exact_signflip': None, 'matches': None,
                'reason': 'exact_signflip_p unavailable'}
    p_ref = ref.get('p')
    try:
        matches = bool(p_ref is not None and _esci_math.isfinite(float(p_ref))
                       and abs(float(p0) - float(p_ref)) <= 1e-12)
    except (TypeError, ValueError, OverflowError):
        matches = None
    return {
        'p_exact_signflip': (None if p_ref is None else float(p_ref)),
        'p_method': ref.get('method'),
        'p_at_zero': float(p0),
        'matches': matches,
        'note': ('p_at_zero is the exact rational count; exact_signflip_p is '
                 'the float enumeration with its documented tie tolerance, so '
                 'a disagreement can only occur at an exact tie'),
    }


# ---------------------------------------------------------------------------
# timestamp-free block and replayable artifact
# ---------------------------------------------------------------------------

def exact_signflip_ci_block(deltas, *, alpha=0.05, max_n=None, seeds=None):
    """Return the timestamp-free ``inputs``/``result`` block for a report."""
    alpha = _esci_alpha(alpha)
    max_n = _esci_max_n(max_n)
    vals, seeds_out, _dropped, _dropped_seeds, _n_input = _esci_inputs(
        deltas, seeds, strict=True, where='deltas')
    inputs = {
        'deltas': [float(x) for x in vals],
        'seeds': (None if seeds_out is None else [int(s) for s in seeds_out]),
        'alpha': float(alpha),
        'max_n': int(max_n),
    }
    result = exact_signflip_ci(vals, alpha=alpha, max_n=max_n, seeds=seeds_out)
    return {'kind': EXACT_SIGNFLIP_CI_BLOCK_KIND, 'inputs': inputs,
            'result': result}


def build_exact_signflip_ci(deltas, *, alpha=0.05, max_n=None, seeds=None,
                            source=None, created_at=None):
    """Build a strict-JSON, content-addressed exact-CI artifact."""
    block = exact_signflip_ci_block(deltas, alpha=alpha, max_n=max_n,
                                    seeds=seeds)
    artifact = {
        'kind': EXACT_SIGNFLIP_CI_KIND,
        'schema': EXACT_SIGNFLIP_CI_SCHEMA,
        'created_at': (_esci_time.strftime('%Y-%m-%dT%H:%M:%S')
                       if created_at is None else _esci_key_text(created_at, limit=80)),
        'source': (None if source is None else _esci_key_text(source, limit=500)),
        'inputs': block['inputs'],
        'result': block['result'],
    }
    artifact['content_sha256'] = _esci_content_sha(artifact)
    return artifact


def compare_exact_signflip_ci_results(stored, replay, *, rtol=1e-9,
                                      atol=1e-12):
    diffs = _esci_compare(stored, replay, 'result', rtol=rtol, atol=atol)
    return {'match': not diffs, 'differences': diffs}


def _esci_validate_common(artifact, *, block=False):
    out = {
        'kind': ('exact_signflip_ci_block_validation' if block
                 else 'exact_signflip_ci_validation'),
        'status': 'fail',
        'content_sha256': None,
        'result_status': None,
        'replayed': False,
        'problems': [],
        'warnings': [],
        'result': None,
    }
    if not isinstance(artifact, dict):
        out['problems'].append('artifact must be a JSON object')
        return out
    expected = (EXACT_SIGNFLIP_CI_BLOCK_KEYS if block
                else EXACT_SIGNFLIP_CI_ARTIFACT_KEYS)
    strict = _esci_strict_json_problem(
        'block' if block else 'artifact', artifact)
    if strict:
        out['problems'].append(strict)
    extra = sorted(set(artifact) - set(expected), key=_esci_key_text)
    missing = [k for k in expected if k not in artifact]
    if extra or missing:
        out['problems'].append(
            f'artifact keys differ from the schema: missing={missing}, '
            f'extra={[_esci_key_text(k) for k in extra]}')
    want_kind = (EXACT_SIGNFLIP_CI_BLOCK_KIND if block
                 else EXACT_SIGNFLIP_CI_KIND)
    if not _esci_equal(artifact.get('kind'), want_kind):
        out['problems'].append(
            f'artifact.kind must be {want_kind!r}, got '
            f'{_esci_repr(artifact.get("kind"))}')
    if (not block
            and not _esci_equal(artifact.get('schema'),
                                EXACT_SIGNFLIP_CI_SCHEMA)):
        out['problems'].append(
            f'artifact.schema must be {EXACT_SIGNFLIP_CI_SCHEMA}, got '
            f'{_esci_repr(artifact.get("schema"))}')
    try:
        forbidden = _esci_forbidden_paths(artifact)
    except Exception as exc:
        out['problems'].append(
            'artifact cannot be inspected for frozen-decision fields: '
            f'{type(exc).__name__}: {_esci_repr(exc)}')
        return out
    if forbidden:
        out['problems'].append(
            'artifact carries frozen-decision key(s): ' + ', '.join(forbidden[:8]))
    if not block:
        recorded = artifact.get('content_sha256')
        if not isinstance(recorded, str) or len(recorded) != 64:
            out['problems'].append(
                'artifact.content_sha256 must be a 64-hex string')
        else:
            out['content_sha256'] = recorded
            try:
                actual = _esci_content_sha(artifact)
            except Exception as exc:
                out['problems'].append(
                    'content hash is not computable: '
                    f'{type(exc).__name__}: {_esci_repr(exc)}')
                actual = None
            if actual is not None and actual != recorded:
                out['problems'].append(
                    'artifact.content_sha256 does not match the recorded body')
    inputs = artifact.get('inputs')
    out['problems'].extend(_esci_inputs_problems(inputs))
    result = artifact.get('result')
    if not isinstance(result, dict):
        out['problems'].append('artifact.result must be an object')
    else:
        out['result_status'] = (result.get('status')
                                if isinstance(result.get('status'), str)
                                else _esci_repr(result.get('status')))
        out['problems'].extend(_esci_result_problems(result))
    if out['problems']:
        return out
    if not block:
        if not isinstance(artifact.get('created_at'), str):
            out['problems'].append('created_at must be a string')
            return out
        source = artifact.get('source')
        if source is not None and not isinstance(source, str):
            out['problems'].append('source must be a string or null')
            return out
    return out


def validate_exact_signflip_ci(artifact, *, replay=True):
    out = _esci_validate_common(artifact, block=False)
    if out['problems']:
        return out
    inputs = artifact['inputs']
    result = artifact['result']
    if not replay:
        out['status'] = ('warnings' if result.get('status') == 'warnings'
                         else ('not_estimable'
                               if result.get('status') == 'not_estimable'
                               else 'ok'))
        out['result'] = _esci_jsonable(result)
        return out
    try:
        replayed = exact_signflip_ci(
            inputs['deltas'], alpha=inputs['alpha'], max_n=inputs['max_n'],
            seeds=inputs['seeds'])
    except Exception as exc:
        out['problems'].append(
            f'replay failed: {type(exc).__name__}: {_esci_repr(exc)}')
        return out
    out['replayed'] = True
    cmp = compare_exact_signflip_ci_results(result, replayed)
    if not cmp['match']:
        out['problems'].append('replayed result differs from the stored result')
        out['warnings'].extend(cmp['differences'][:8])
        return out
    out['result'] = _esci_jsonable(replayed)
    if replayed.get('status') == 'not_estimable':
        out['status'] = 'not_estimable'
    elif replayed.get('status') == 'warnings':
        out['status'] = 'warnings'
        out['warnings'].extend(list(replayed.get('warnings') or []))
    else:
        out['status'] = 'ok'
    return out


def validate_exact_signflip_ci_block(block, *, replay=True):
    out = _esci_validate_common(block, block=True)
    if out['problems']:
        return out
    inputs = block['inputs']
    result = block['result']
    if not replay:
        out['status'] = ('warnings' if result.get('status') == 'warnings'
                         else ('not_estimable'
                               if result.get('status') == 'not_estimable'
                               else 'ok'))
        out['result'] = _esci_jsonable(result)
        return out
    try:
        replayed = exact_signflip_ci(
            inputs['deltas'], alpha=inputs['alpha'], max_n=inputs['max_n'],
            seeds=inputs['seeds'])
    except Exception as exc:
        out['problems'].append(
            f'replay failed: {type(exc).__name__}: {_esci_repr(exc)}')
        return out
    out['replayed'] = True
    cmp = compare_exact_signflip_ci_results(result, replayed)
    if not cmp['match']:
        out['problems'].append('replayed result differs from the stored result')
        out['warnings'].extend(cmp['differences'][:8])
        return out
    out['result'] = _esci_jsonable(replayed)
    if replayed.get('status') == 'not_estimable':
        out['status'] = 'not_estimable'
    elif replayed.get('status') == 'warnings':
        out['status'] = 'warnings'
        out['warnings'].extend(list(replayed.get('warnings') or []))
    else:
        out['status'] = 'ok'
    return out


def write_exact_signflip_ci(path_or_dir, artifact, *, overwrite=False,
                            name='exact_signflip_ci.json'):
    """Write a validated artifact; same content is a no-op, different conflicts."""
    return _write_stamped_artifact(
        path_or_dir, artifact, validator=validate_exact_signflip_ci,
        name=name, overwrite=overwrite)


def exact_signflip_ci_markdown(result, *, title='Exact sign-flip confidence set'):
    title = str(title)[:240]
    if not isinstance(result, dict):
        return f'**{title}**: not estimable (not an object).'
    if result.get('status') == 'not_estimable':
        return (f'**{title}**: not estimable — '
                f'{_esci_repr(result.get("reason") or "no reason", limit=500)}.')
    alpha = result.get('alpha')
    n = result.get('n')
    try:
        pct = int(round(100.0 * float(result.get('conf'))))
    except (TypeError, ValueError, OverflowError):
        pct = 95
    lines = [
        f'**{title}** (n={n}; alpha={alpha}; method=`{result.get("method")}`)',
        '',
        '| quantity | value |',
        '|---|---|',
    ]

    def _fmt(x, fmt='{:+.4f}'):
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return '—'
        x = float(x)
        return fmt.format(x) if _esci_math.isfinite(x) else '—'

    def _ci_text(ci):
        if not isinstance(ci, (list, tuple)) or len(ci) != 2:
            return '—'
        return f'[{_fmt(ci[0])}, {_fmt(ci[1])}]'

    lines.append(f'| mean paired effect | {_fmt(result.get("mean"))} |')
    if result.get('unbounded'):
        lines.append('| exact set | whole real line (resolution limit) |')
    else:
        lines.append(f'| exact {pct}% set | {_ci_text(result.get("ci"))} |')
        if result.get('connected') is False:
            lines.append(
                f'| exact set (union) | {"; ".join(_ci_text(x) for x in (result.get("intervals") or []))} |')
    t_int = result.get('t_interval') or {}
    lines.append(f'| Student-t interval ({pct}%) | {_ci_text(t_int.get("ci"))} |')
    lines.append(f'| p at zero (rational) | {_fmt(result.get("p_at_zero"), "{:.4g}")} |')
    lines.append(f'| contains zero | {result.get("contains_zero")} |')
    rel = (result.get('zero_relation') or {}).get('relation')
    lines.append(f'| zero relation | `{rel}` |')
    lines.append('')
    lines.append('> Boundary: the exact set is the inversion of the same '
                 'sign-flip test (symmetry assumption), so it cannot disagree '
                 'with the exact p about zero; it is conservative and finite '
                 'only when the far-tail p floor 2/2**n is below alpha. The t interval is '
                 'shown for transparency and is not a replacement for the '
                 'pre-registered test. It is not a prediction interval and '
                 'not a posterior probability.')
    return '\n'.join(lines)
