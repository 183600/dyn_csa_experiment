"""Data-blind split-half consistency of a paired effect across evaluation units.

The frozen confirmation test is a seed-level exact sign-flip test on the
token-pooled PPL difference at one length.  It answers "is the mean per-seed
effect distinguishable from zero?".  It does **not** answer "is the effect
present across the evaluation units, or is it carried by one half of the
documents/articles?".  A significant mean can coexist with a large positive
effect on a few evaluation units and a near-zero or opposite effect on the
rest, because the test treats each seed as one observation and pools all
tokens.

This fragment adds an internal, outcome-blind split-half check:

* the evaluation units (document ids, or article ids when they are the
  declared independent unit) are partitioned deterministically from a fixed
  salt and a SHA-256 of each unit id -- never from the observed outcome;
* on each half the per-seed token-weighted log-PPL effect is recomputed and a
  one-sided exact sign-flip test is run over the same seeds;
* the claim "the effect has the declared direction in BOTH halves" is an
  intersection-union claim, so its p-value is ``max(p_half0, p_half1)``.  No
  independence between halves is assumed and no Bonferroni correction is
  applied: IUT is valid exactly because it requires every component to reject;
* the split assignment is a specification choice, so a deterministic
  sensitivity curve over many distinct salts reports how often the both-halves
  claim survives.  That curve is descriptive and is never used to pick the
  reported split.

This is an **internal consistency** readout, not an independent replication.
The two halves are disjoint documents from the same fixed evaluation slice,
so they cannot establish external validity, and they share the same trained
seeds.  The block changes no frozen p-value, no Holm family and never writes
``confirmed``/``verified``/``verdict``/``decision``.
"""

import hashlib as _usr_hashlib
import json as _usr_json
import math as _usr_math
import os as _usr_os
import time as _usr_time

import numpy as np


UNIT_SPLIT_REPLICATION_KIND = 'unit_split_replication'
UNIT_SPLIT_REPLICATION_BLOCK_KIND = 'unit_split_replication_block'
UNIT_SPLIT_REPLICATION_RESULT_KIND = 'unit_split_replication_result'
UNIT_SPLIT_REPLICATION_SCHEMA = 1
UNIT_SPLIT_REPLICATION_MAX_N = 20
UNIT_SPLIT_REPLICATION_HARD_MAX_N = 24
UNIT_SPLIT_REPLICATION_DEFAULT_SALT = 'csas-unit-split-v1'
UNIT_SPLIT_REPLICATION_MIN_UNITS = 4
UNIT_SPLIT_REPLICATION_MIN_UNITS_PER_HALF = 2
UNIT_SPLIT_REPLICATION_MIN_TOKEN_SHARE = 0.25
UNIT_SPLIT_REPLICATION_DEFAULT_SALTS = 64
UNIT_SPLIT_REPLICATION_MAX_SALTS = 512
# Deterministic work cap for the split-sensitivity curve.  Each salt costs
# two exact sign-flip enumerations of 2**n patterns; beyond this many pattern
# evaluations the curve is truncated (and says so) instead of turning a cheap
# robustness readout into an accidental Monte-Carlo job.
UNIT_SPLIT_REPLICATION_SENSITIVITY_BUDGET = 5_000_000
UNIT_SPLIT_REPLICATION_MAX_REPORTABLE_INT = 10 ** 4000

UNIT_SPLIT_REPLICATION_INPUT_KEYS = (
    'log_ratios', 'tokens', 'unit_ids', 'seeds', 'direction', 'alpha',
    'min_n', 'max_n', 'min_units', 'min_units_per_half', 'min_token_share',
    'salt', 'n_salts',
)
UNIT_SPLIT_REPLICATION_ARTIFACT_KEYS = (
    'kind', 'schema', 'created_at', 'source', 'inputs', 'result',
    'content_sha256',
)
UNIT_SPLIT_REPLICATION_BLOCK_KEYS = ('kind', 'inputs', 'result')
UNIT_SPLIT_REPLICATION_RESULT_KEYS = (
    'kind', 'version', 'status', 'reason', 'alpha', 'direction', 'min_n',
    'max_n', 'min_units', 'min_units_per_half', 'min_token_share', 'method',
    'role', 'boundary', 'n_units', 'n_seeds_input', 'n_seeds', 'seeds',
    'dropped_seeds', 'unit_id_source', 'token_total', 'assignment', 'halves',
    'iut', 'sign_agreement', 'half_difference', 'composition', 'warnings',
    'split_sensitivity',
)
UNIT_SPLIT_REPLICATION_FORBIDDEN_KEYS = (
    'confirmed', 'p_holm', 'verified', 'verdict', 'decision',
)
UNIT_SPLIT_REPLICATION_DIRECTIONS = ('positive', 'negative')
UNIT_SPLIT_REPLICATION_STATUSES = ('ok', 'warnings', 'not_estimable',
                                   'not_applicable')

UNIT_SPLIT_REPLICATION_POLICY = {
    'version': 'unit-split-replication-1',
    'estimand': (
        'per-half mean of the per-seed token-weighted paired log-PPL effect '
        '(log PPL_arm_a - log PPL_arm_b); positive means arm a is worse'),
    'assignment': (
        'deterministic: every unit id is SHA-256 hashed with a fixed salt, '
        'units are visited in hash order, and each is greedily assigned to the '
        'half with the smaller running token total (ties: smaller unit count, '
        'then half 0).  The outcome is never read by the assignment.'),
    'test': (
        'one-sided exact sign-flip test over the paired per-seed effects in '
        'each half; the both-halves claim is an intersection-union test with '
        'p = max(p_half0, p_half1), which needs no independence assumption'),
    'role': (
        'internal split-half consistency companion; it adds no p-value to the '
        'frozen Holm family and never changes p_exact_signflip, p_holm, '
        'verified or confirmed'),
    'boundary': (
        'the two halves are disjoint units of the SAME fixed evaluation slice '
        'and share the same trained seeds, so this is internal consistency, '
        'not independent replication and not external validity.  It does not '
        'repair a sample that is unrepresentative of the corpus, a '
        'non-exchangeable seed panel, or selection on the validation split.  '
        'The split-sensitivity curve is descriptive: reporting the best salt '
        'is exactly the garden of forking paths this block is meant to make '
        'visible.  Exact only for n <= max_n (default 20).'),
}


# ---------------------------------------------------------------------------
# small validation helpers
# ---------------------------------------------------------------------------

def _usr_repr(value, limit=200):
    try:
        text = repr(value)
    except Exception:
        return f'<unrepresentable {type(value).__name__}>'
    return text if len(text) <= limit else text[:limit] + '...'


def _usr_safe_text(value, limit=240):
    """Exception-safe bounded ``str`` for provenance labels.

    ``created_at``/``source`` accept caller data (and tampered archived
    artifacts), so a huge integer or an object whose ``__str__`` raises must
    not be able to crash artifact construction.  ``repr`` is no safer: CPython
    also refuses to spell an integer beyond its decimal-conversion limit.
    """
    if value is None:
        return None
    try:
        text = str(value)
    except Exception:
        try:
            text = f'<unrepresentable {type(value).__name__}>'
        except Exception:
            text = '<unrepresentable>'
    return text if len(text) <= limit else text[:limit]


def _usr_finite_sum(values):
    """Left-to-right float sum, or ``None`` when it overflows/degrades."""
    total = 0.0
    try:
        for value in values:
            total += float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return total if _usr_math.isfinite(total) else None


def _usr_safe_diff(left, right):
    """``left - right`` without overflowing for opposite extreme signs."""
    try:
        diff = float(left) - float(right)
    except (TypeError, ValueError, OverflowError):
        return None
    if _usr_math.isfinite(diff):
        return diff
    # The true difference can lie outside float64 (e.g. 1.7e308 - -1.7e308).
    # Scale first so same-sign / near-equal values still produce a finite
    # result; return None only when the mathematical difference really is
    # unrepresentable.
    scale = max(abs(float(left)), abs(float(right)))
    if not _usr_math.isfinite(scale) or scale == 0.0:
        return None
    try:
        diff = (float(left) / scale - float(right) / scale) * scale
    except (TypeError, ValueError, OverflowError):
        return None
    return diff if _usr_math.isfinite(diff) else None



def _usr_number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not _usr_math.isfinite(out):
        return None
    if positive and out <= 0.0:
        return None
    return out


def _usr_alpha(value):
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray)):
        raise ValueError(f'alpha must be a real number, got {_usr_repr(value)}')
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'alpha must be a real number, got {_usr_repr(value)}') from None
    if not _usr_math.isfinite(out) or not (0.0 < out < 1.0):
        raise ValueError(f'alpha must be finite and in (0,1), got {out!r}')
    return out


def _usr_int(name, value, *, minimum, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer, got {_usr_repr(value)}')
    out = int(value)
    if out < int(minimum) or out > int(maximum):
        # ``out`` can be an attacker/scale-supplied integer above CPython's
        # decimal-conversion limit; an f-string would then raise the
        # interpreter's ValueError instead of this named diagnostic.
        raise ValueError(
            f'{name} must be in [{_usr_repr(minimum)}, {_usr_repr(maximum)}], '
            f'got {_usr_repr(out)}')
    return out


def _usr_salt(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'salt must be a non-empty string, got {_usr_repr(value)}')
    if len(value) > 240:
        raise ValueError(f'salt must be at most 240 characters, got {len(value)}')
    return value


def _usr_direction(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f'direction must be a string or None, got {_usr_repr(value)}')
    out = value.strip().lower()
    if out not in UNIT_SPLIT_REPLICATION_DIRECTIONS:
        raise ValueError(
            f'direction must be one of {UNIT_SPLIT_REPLICATION_DIRECTIONS} or '
            f'None, got {_usr_repr(value)}')
    return out


# ---------------------------------------------------------------------------
# deterministic, outcome-blind assignment
# ---------------------------------------------------------------------------

def _usr_unit_digest(salt, unit_id):
    return _usr_hashlib.sha256(
        (salt + '\x00' + unit_id).encode('utf-8')).hexdigest()


def unit_split_assignment(unit_ids, tokens, *, salt=UNIT_SPLIT_REPLICATION_DEFAULT_SALT,
                          min_units_per_half=UNIT_SPLIT_REPLICATION_MIN_UNITS_PER_HALF):
    """Return the deterministic token-balanced two-half assignment.

    Pure function of the unit ids, the token counts and the salt; the outcome
    (the per-seed effects) is not an input.  The digest binds the salt, the
    ids and the token counts so the exact partition can be replayed.
    """
    if isinstance(unit_ids, (str, bytes)) or not isinstance(unit_ids, (list, tuple)):
        raise ValueError(f'unit_ids must be a sequence, got {_usr_repr(unit_ids)}')
    ids = []
    for value in unit_ids:
        if not isinstance(value, str):
            raise ValueError(f'unit_ids must be strings, got {_usr_repr(value)}')
        text = value.strip()
        if not text:
            raise ValueError('unit_ids must be non-empty strings')
        if len(text) > 500:
            raise ValueError('unit_ids must be at most 500 characters')
        ids.append(text)
    if len(set(ids)) != len(ids):
        raise ValueError('unit_ids contain duplicates; the partition would be ambiguous')
    if isinstance(tokens, (str, bytes)) or not isinstance(tokens, (list, tuple)):
        raise ValueError(f'tokens must be a sequence, got {_usr_repr(tokens)}')
    weights = []
    for value in tokens:
        number = _usr_number(value, 'tokens', positive=True)
        if number is None:
            raise ValueError(f'tokens must be finite and positive, got {_usr_repr(value)}')
        weights.append(number)
    if len(weights) != len(ids):
        raise ValueError(
            f'tokens length {len(weights)} does not match unit_ids length {len(ids)}')
    salt = _usr_salt(salt)
    _min_half = _usr_int('min_units_per_half', min_units_per_half, minimum=1,
                         maximum=max(1, len(ids)))
    if len(ids) < 2 * _min_half:
        raise ValueError(
            f'{len(ids)} units cannot give two halves of >= {_min_half} units')
    order = sorted(range(len(ids)),
                   key=lambda i: (_usr_unit_digest(salt, ids[i]), ids[i]))

    def _balance(used_weights):
        halves_out = ([], [])
        totals_out = [0.0, 0.0]
        counts_out = [0, 0]
        for index in order:
            # Token balance first; ties fall back to the smaller unit count and
            # then to half 0, so the assignment is a total order with no
            # dependence on iteration order.
            left = (totals_out[0], counts_out[0], 0)
            right = (totals_out[1], counts_out[1], 1)
            target = 0 if left <= right else 1
            halves_out[target].append(index)
            totals_out[target] += used_weights[index]
            counts_out[target] += 1
        return ([sorted(h) for h in halves_out], totals_out, counts_out)

    halves, totals, counts = _balance(weights)
    total_tokens = _usr_finite_sum(weights)
    scaled_tokens = False
    if (total_tokens is None
            or not all(_usr_math.isfinite(float(t)) for t in totals)):
        # A future panel can declare per-unit token counts whose sum exceeds
        # float64 even though every individual count is finite.  Balancing on
        # raw totals would then compare inf == inf (and report nan shares);
        # normalise by the largest count, which is a monotone rescaling of the
        # same greedy rule, and keep the unrepresentable absolute totals null.
        token_scale = max(weights)
        if (not _usr_math.isfinite(token_scale) or token_scale <= 0.0):
            raise ValueError('tokens cannot be normalised for token balance')
        scaled = [float(w) / float(token_scale) for w in weights]
        halves, totals, counts = _balance(scaled)
        total_tokens = _usr_finite_sum(scaled)
        scaled_tokens = True
    body = {
        'salt': salt,
        'halves': [[ids[i] for i in halves[0]], [ids[i] for i in halves[1]]],
        'tokens': [weights[i] for i in range(len(ids))],
    }
    digest = _usr_hashlib.sha256(
        _usr_json.dumps(body, sort_keys=True, separators=(',', ':'),
                        ensure_ascii=False, allow_nan=False).encode('utf-8')
    ).hexdigest()
    if scaled_tokens:
        _share_total = float(sum(totals))
        shares = [(float(totals[i]) / _share_total
                   if _share_total > 0.0 else None) for i in (0, 1)]
    else:
        shares = [(float(sum(weights[i] for i in h)) / total_tokens
                   if total_tokens is not None and total_tokens > 0.0 else None)
                  for h in halves]
    return {
        'salt': salt,
        'algorithm': ('sha256(salt\\0unit_id) order + greedy token balance '
                      '(ties: unit count, then half 0)'),
        'digest': digest,
        'n_units': int(len(ids)),
        'halves': [
            {
                'name': 'half_0',
                'indices': [int(i) for i in halves[0]],
                'unit_ids': [ids[i] for i in halves[0]],
                'n_units': int(len(halves[0])),
                'n_tokens': (None if scaled_tokens else float(totals[0])),
                'token_share': shares[0],
            },
            {
                'name': 'half_1',
                'indices': [int(i) for i in halves[1]],
                'unit_ids': [ids[i] for i in halves[1]],
                'n_units': int(len(halves[1])),
                'n_tokens': (None if scaled_tokens else float(totals[1])),
                'token_share': shares[1],
            },
        ],
    }


# ---------------------------------------------------------------------------
# the exact one-sided primitive
# ---------------------------------------------------------------------------

def _usr_one_sided(deltas, direction):
    """Exact one-sided sign-flip p for the declared direction.

    ``direction='negative'`` tests H1: mean < 0, ``'positive'`` tests H1:
    mean > 0.  The one-sided upper-tail primitive is the same one used by the
    exact TOST; passing the negated vector turns it into the lower tail.
    """
    try:
        values = [float(x) for x in deltas]
    except (TypeError, ValueError, OverflowError):
        return None
    if any(not _usr_math.isfinite(x) for x in values):
        return None
    probe = values if direction == 'positive' else [-x for x in values]
    tail = _etost_upper_tail(probe)  # noqa: F821 (part 77)
    p = tail.get('p')
    if not isinstance(p, (int, float)) or isinstance(p, (bool, np.bool_)):
        return None
    if not _usr_math.isfinite(float(p)):
        return None
    return {
        'p_one_sided': float(p),
        'p_floor': tail.get('p_floor'),
        'n_nonzero': int(tail.get('n_nonzero') or 0),
        'method': tail.get('method'),
    }


def _usr_stats(values):
    n = len(values)
    if n == 0:
        return None
    # NumPy's mean/std sum and square the raw panel.  Finite per-unit effects
    # near float64's limit then overflow to inf (or underflow to a fake zero
    # spread) even though the mean and SD are representable; reuse the
    # repository's scale-normalised helpers so the ordinary path stays
    # bit-identical and the extreme path stays finite.
    try:
        mean = float(_scaled_mean_or_raw(values))  # noqa: F821 (part 32)
    except Exception:
        mean = None
    if mean is None or not _usr_math.isfinite(mean):
        mean = None
    sd = None
    if n > 1:
        try:
            sd = _scaled_std_or_raw(values)  # noqa: F821 (part 32)
        except Exception:
            sd = None
        if sd is not None and not _usr_math.isfinite(float(sd)):
            sd = None
    sem = (float(sd) / _usr_math.sqrt(n) if sd is not None else None)
    if sem is not None and not _usr_math.isfinite(sem):
        sem = None
    return {
        'n': int(n),
        'mean': mean,
        'sd': (None if sd is None else float(sd)),
        'sem': sem,
        'min': float(min(values)),
        'max': float(max(values)),
        'n_positive': int(sum(1 for x in values if x > 0.0)),
        'n_negative': int(sum(1 for x in values if x < 0.0)),
        'n_zero': int(sum(1 for x in values if x == 0.0)),
    }


def _usr_weighted_mean(values, weights):
    """Token-weighted mean that survives extreme-but-finite magnitudes.

    The historical expression ``fsum(x*w) / sum(w)`` overflows once either the
    product or the weight total leaves float64, even though the weighted mean
    is bounded by the (finite) input values.  Try the exact historical path
    first so normal panels do not change by a single bit, then fall back to a
    doubly scaled computation (values by max |value|, weights by max weight).
    """
    try:
        vals = [float(v) for v in values]
        ws = [float(w) for w in weights]
    except (TypeError, ValueError, OverflowError):
        return None
    if not vals or len(vals) != len(ws):
        return None
    try:
        # Keep the historical denominator expression for the ordinary path
        # (builtin left-to-right sum); the scaled fallback below is what
        # handles an overflowing total.
        numer = _usr_math.fsum(v * w for v, w in zip(vals, ws))
        denom = float(sum(ws))
    except (TypeError, ValueError, OverflowError):
        numer = None
        denom = None
    if (numer is not None and denom is not None
            and _usr_math.isfinite(numer) and _usr_math.isfinite(denom)
            and denom > 0.0):
        out = numer / denom
        if _usr_math.isfinite(out):
            return float(out)
    if any(not _usr_math.isfinite(v) for v in vals + ws):
        return None
    vscale = max((abs(v) for v in vals), default=0.0)
    wscale = max(ws, default=0.0)
    if (not _usr_math.isfinite(vscale) or not _usr_math.isfinite(wscale)
            or wscale <= 0.0):
        return None
    if vscale == 0.0:
        return 0.0
    try:
        denom_n = _usr_math.fsum(w / wscale for w in ws)
        if not (_usr_math.isfinite(denom_n) and denom_n > 0.0):
            return None
        numer_n = _usr_math.fsum((v / vscale) * (w / wscale)
                                 for v, w in zip(vals, ws))
        if not _usr_math.isfinite(numer_n):
            return None
        out = (numer_n / denom_n) * vscale
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None
    return float(out) if _usr_math.isfinite(out) else None


def _usr_half_effects(rows, indices, tokens):
    weights = [float(tokens[i]) for i in indices]
    effects = []
    for row in rows:
        effect = _usr_weighted_mean([row[i] for i in indices], weights)
        if effect is None:
            return None
        effects.append(effect)
    return effects


# ---------------------------------------------------------------------------
# core: one salt
# ---------------------------------------------------------------------------

def _usr_core(rows, unit_ids, tokens, *, direction, salt, alpha, min_n, max_n,
              min_units_per_half):
    assignment = unit_split_assignment(
        unit_ids, tokens, salt=salt,
        min_units_per_half=min_units_per_half)
    halves = []
    effects_by_half = []
    for half in assignment['halves']:
        effects = _usr_half_effects(rows, half['indices'], tokens)
        stats = _usr_stats(effects) if effects is not None else None
        test = (_usr_one_sided(effects, direction)
                if effects is not None else None)
        halves.append({
            'name': half['name'],
            'n_units': half['n_units'],
            'n_seeds': int(len(effects)) if effects is not None else 0,
            'n_tokens': half['n_tokens'],
            'token_share': half['token_share'],
            'stats': stats,
            'test': test,
        })
        effects_by_half.append(effects)
    p0 = (halves[0]['test'] or {}).get('p_one_sided')
    p1 = (halves[1]['test'] or {}).get('p_one_sided')
    if p0 is None or p1 is None:
        return {'assignment': assignment, 'halves': halves, 'iut': None,
                'sign_agreement': 'unavailable', 'half_difference': None,
                'effects_by_half': effects_by_half}
    iut_p = max(float(p0), float(p1))
    mean0 = (halves[0]['stats'] or {}).get('mean')
    mean1 = (halves[1]['stats'] or {}).get('mean')
    if mean0 is None or mean1 is None:
        return {'assignment': assignment, 'halves': halves, 'iut': None,
                'sign_agreement': 'unavailable', 'half_difference': None,
                'effects_by_half': effects_by_half}
    if mean0 < 0.0 and mean1 < 0.0:
        sign = 'both_negative'
    elif mean0 > 0.0 and mean1 > 0.0:
        sign = 'both_positive'
    elif mean0 * mean1 < 0.0:
        sign = 'discordant'
    else:
        sign = 'zero_in_one_half'
    expected = (mean0 < 0.0 and mean1 < 0.0) if direction == 'negative' else (
        mean0 > 0.0 and mean1 > 0.0)
    if expected and iut_p <= alpha:
        claim = 'both_halves_support_direction'
    elif sign == 'discordant':
        claim = 'halves_discordant'
    elif expected:
        claim = 'direction_consistent_but_not_jointly_significant'
    elif (p0 <= alpha or p1 <= alpha) and sign in ('both_negative',
                                                   'both_positive'):
        claim = 'one_half_significant'
    else:
        claim = 'direction_not_supported'
    diff = [_usr_safe_diff(a, b)
            for a, b in zip(effects_by_half[0], effects_by_half[1])]
    diff_available = (bool(diff) and all(d is not None for d in diff))
    diff_stats = _usr_stats(diff) if diff_available else None
    t_ci = (_esci_t_interval(diff, alpha)  # noqa: F821 (part 87)
            if diff_available else None)
    return {
        'assignment': assignment,
        'halves': halves,
        'iut': {
            'p_iut': iut_p,
            'p_half_0': float(p0),
            'p_half_1': float(p1),
            'alpha': float(alpha),
            'method': ('intersection-union: max of the two one-sided exact '
                       'sign-flip p-values; no independence assumption'),
            'claim': claim,
            'direction': direction,
            'reject': bool(iut_p <= alpha),
        },
        'sign_agreement': sign,
        'half_difference': {
            'mean': (diff_stats['mean'] if diff_stats is not None else None),
            'sd': (diff_stats['sd'] if diff_stats is not None else None),
            'sem': (diff_stats['sem'] if diff_stats is not None else None),
            'ci95': (t_ci.get('ci') if isinstance(t_ci, dict) else None),
            'ci_method': ('student-t (df=n-1); descriptive only, no p-value'
                          if isinstance(t_ci, dict) else None),
            'definition': 'mean(half_0 per-seed effect - half_1 per-seed effect)',
        },
        'effects_by_half': effects_by_half,
    }


# ---------------------------------------------------------------------------
# public result
# ---------------------------------------------------------------------------

def _usr_clean_rows(log_ratios, tokens, unit_ids, seeds):
    if isinstance(log_ratios, (str, bytes)) or not isinstance(log_ratios, (list, tuple)):
        raise ValueError('log_ratios must be a sequence of per-seed rows')
    rows_in = list(log_ratios)
    if not rows_in:
        raise ValueError('log_ratios must contain at least one seed row')
    if isinstance(tokens, (str, bytes)) or not isinstance(tokens, (list, tuple)):
        raise ValueError('tokens must be a sequence')
    weights = []
    for value in tokens:
        number = _usr_number(value, 'tokens', positive=True)
        if number is None:
            raise ValueError(f'tokens must be finite and positive, got {_usr_repr(value)}')
        weights.append(number)
    n_units = len(weights)
    if unit_ids is not None:
        if isinstance(unit_ids, (str, bytes)) or not isinstance(unit_ids, (list, tuple)):
            raise ValueError('unit_ids must be a sequence or None')
        ids = []
        for value in unit_ids:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'unit_ids must be non-empty strings, got {_usr_repr(value)}')
            text = value.strip()
            if len(text) > 500:
                raise ValueError('unit_ids must be at most 500 characters')
            ids.append(text)
        if len(ids) != n_units:
            raise ValueError(
                f'unit_ids length {len(ids)} does not match tokens length {n_units}')
        if len(set(ids)) != len(ids):
            raise ValueError('unit_ids contain duplicates')
        id_source = 'supplied'
    else:
        ids = [f'u{index:04d}' for index in range(n_units)]
        id_source = 'position'
    if seeds is not None:
        if isinstance(seeds, (str, bytes)) or not isinstance(seeds, (list, tuple)):
            raise ValueError('seeds must be a sequence or None')
        seed_out = []
        for index, value in enumerate(seeds):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f'seeds must be integers, got {_usr_repr(value)}')
            number = int(value)
            if abs(number) > UNIT_SPLIT_REPLICATION_MAX_REPORTABLE_INT:
                raise ValueError(
                    f'seeds[{index}] is outside the strict-JSON '
                    'representable integer range')
            seed_out.append(number)
        if len(seed_out) != len(rows_in):
            raise ValueError(
                f'seeds length {len(seed_out)} does not match log_ratios length '
                f'{len(rows_in)}')
        if len(set(seed_out)) != len(seed_out):
            raise ValueError('seeds contain duplicates')
    else:
        seed_out = list(range(len(rows_in)))
    rows = []
    kept_seeds = []
    dropped = []
    for index, row in enumerate(rows_in):
        if isinstance(row, (str, bytes)) or not isinstance(row, (list, tuple)):
            raise ValueError(
                f'log_ratios[{index}] must be a per-unit sequence, got '
                f'{type(row).__name__}')
        if len(row) != n_units:
            raise ValueError(
                f'log_ratios[{index}] has length {len(row)}, expected {n_units}')
        values = []
        bad = None
        for j, value in enumerate(row):
            number = _usr_number(value, f'log_ratios[{index}][{j}]')
            if number is None:
                bad = f'log_ratios[{index}][{j}] is not a finite number'
                break
            values.append(number)
        if bad is not None:
            dropped.append({'seed': int(seed_out[index]), 'reason': bad})
            continue
        rows.append(values)
        kept_seeds.append(int(seed_out[index]))
    return rows, kept_seeds, dropped, weights, ids, id_source


def _usr_sensitivity(rows, unit_ids, tokens, *, direction, base_salt, alpha,
                     min_n, max_n, min_units_per_half, n_salts,
                     budget=UNIT_SPLIT_REPLICATION_SENSITIVITY_BUDGET):
    n = len(rows)
    # The exact test costs two 2**n enumerations per salt, and rebuilding the
    # two half effects costs n*units float operations.  Count both so a future
    # panel with many evaluation units does not turn the default sensitivity
    # curve into an accidental O(salts * n * units) job.
    per_salt_work = 2 * (1 << int(n)) + int(n) * max(1, len(unit_ids))
    cap = max(1, int(budget // max(1, per_salt_work)))
    requested = int(n_salts)
    used = min(requested, cap)
    ps = []
    passed = []
    digests = set()
    skipped = 0
    for k in range(used):
        if k == 0:
            salt = base_salt
        else:
            # ``_usr_salt`` caps a salt at 240 characters; a caller-supplied
            # base at the cap would otherwise make every derived salt invalid
            # and silently turn a multi-salt sensitivity curve into one split.
            suffix = f'#{k}'
            keep = max(0, 240 - len(suffix))
            salt = f'{base_salt[:keep]}{suffix}'
        try:
            core = _usr_core(rows, unit_ids, tokens, direction=direction,
                             salt=salt, alpha=alpha, min_n=min_n, max_n=max_n,
                             min_units_per_half=min_units_per_half)
        except (ValueError, TypeError):
            skipped += 1
            continue
        iut = core.get('iut') or {}
        p = iut.get('p_iut')
        if not isinstance(p, (int, float)) or isinstance(p, (bool, np.bool_)):
            skipped += 1
            continue
        ps.append(float(p))
        digests.add((core.get('assignment') or {}).get('digest'))
        if float(p) <= alpha and iut.get('claim') == 'both_halves_support_direction':
            passed.append({'salt': salt,
                           'p_iut': float(p),
                           'assignment_digest': (core.get('assignment') or {}).get('digest')})
    out = {
        'n_salts_requested': requested,
        'n_salts_used': int(used),
        'n_salts_skipped': int(skipped),
        'n_estimable': int(len(ps)),
        'n_pass': int(len(passed)),
        'pass_fraction': (float(len(passed)) / float(len(ps)) if ps else None),
        'p_min': (float(min(ps)) if ps else None),
        'p_median': (float(np.median(ps)) if ps else None),
        'p_max': (float(max(ps)) if ps else None),
        'n_distinct_assignments': int(len(digests)),
        'passing_salts_sample': passed[:5],
        'truncated': bool(used < requested),
        'role': ('descriptive specification curve over deterministic splits; '
                 'it is NEVER used to choose the reported split and its '
                 'fraction is not a p-value'),
    }
    if used < requested:
        out['reason'] = (
            f'requested {requested} salts but the deterministic work budget '
            f'({budget} pattern/unit operations) allows {used} at n={n}')
    return out


def unit_split_replication(log_ratios, tokens, *, unit_ids=None, seeds=None,
                           direction='negative', alpha=0.05, min_n=6,
                           max_n=UNIT_SPLIT_REPLICATION_MAX_N,
                           min_units=UNIT_SPLIT_REPLICATION_MIN_UNITS,
                           min_units_per_half=UNIT_SPLIT_REPLICATION_MIN_UNITS_PER_HALF,
                           min_token_share=UNIT_SPLIT_REPLICATION_MIN_TOKEN_SHARE,
                           salt=UNIT_SPLIT_REPLICATION_DEFAULT_SALT,
                           n_salts=UNIT_SPLIT_REPLICATION_DEFAULT_SALTS):
    """Split-half consistency of a paired log-PPL effect across units.

    ``log_ratios[s][u]`` is ``log PPL_a - log PPL_b`` for seed ``s`` on
    evaluation unit ``u``; ``tokens[u]`` is the unit's scored-token count and
    defines the within-half pooling weights.  The function never reads an arm
    label beyond the sign convention of ``log_ratios``.
    """
    alpha = _usr_alpha(alpha)
    direction = _usr_direction(direction)
    max_n = _usr_int('max_n', max_n, minimum=2,
                     maximum=UNIT_SPLIT_REPLICATION_HARD_MAX_N)
    min_n = _usr_int('min_n', min_n, minimum=2, maximum=100000)
    min_units = _usr_int('min_units', min_units, minimum=4, maximum=100000)
    min_units_per_half = _usr_int(
        'min_units_per_half', min_units_per_half, minimum=1,
        maximum=max(1, min_units))
    try:
        share = float(min_token_share)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f'min_token_share must be a real number, got {_usr_repr(min_token_share)}') from None
    if not _usr_math.isfinite(share) or not (0.0 < share < 0.5):
        raise ValueError(
            f'min_token_share must be finite and in (0,0.5), got {share!r}')
    salt = _usr_salt(salt)
    n_salts = _usr_int('n_salts', n_salts, minimum=0,
                       maximum=UNIT_SPLIT_REPLICATION_MAX_SALTS)
    base = {
        'kind': UNIT_SPLIT_REPLICATION_RESULT_KIND,
        'version': UNIT_SPLIT_REPLICATION_POLICY['version'],
        'status': 'ok',
        'reason': None,
        'alpha': float(alpha),
        'direction': direction,
        'min_n': int(min_n),
        'max_n': int(max_n),
        'min_units': int(min_units),
        'min_units_per_half': int(min_units_per_half),
        'min_token_share': float(share),
        'method': UNIT_SPLIT_REPLICATION_POLICY['test'],
        'role': UNIT_SPLIT_REPLICATION_POLICY['role'],
        'boundary': UNIT_SPLIT_REPLICATION_POLICY['boundary'],
    }
    if direction is None:
        base.update({
            'status': 'not_applicable',
            'reason': ('no pre-registered direction; a both-halves directional '
                       'claim is undefined without one'),
        })
        return base
    rows, kept_seeds, dropped, weights, ids, id_source = _usr_clean_rows(
        log_ratios, tokens, unit_ids, seeds)
    token_total = _usr_finite_sum(weights)
    base.update({
        'n_units': int(len(weights)),
        'n_seeds_input': int(len(list(log_ratios))),
        'n_seeds': int(len(rows)),
        'seeds': kept_seeds,
        'dropped_seeds': dropped,
        'unit_id_source': id_source,
        # A future panel can declare unit token counts whose total is beyond
        # float64 even though every count is finite.  Reporting the sum as
        # ``inf`` would break the strict-JSON artifact contract; report the
        # unrepresentable aggregate as null instead.
        'token_total': token_total,
    })
    if len(weights) < min_units:
        base.update({
            'status': 'not_estimable',
            'reason': (f'{len(weights)} evaluation units is below '
                       f'min_units={min_units}'),
        })
        return base
    if len(rows) < min_n:
        base.update({
            'status': 'not_estimable',
            'reason': (f'only {len(rows)} complete seed row(s) remain after '
                       f'dropping non-finite entries; min_n={min_n}'),
        })
        return base
    if len(rows) > max_n:
        base.update({
            'status': 'not_estimable',
            'reason': (f'{len(rows)} seeds exceeds the exact sign-flip cap '
                       f'max_n={max_n}; no approximate fallback is substituted'),
        })
        return base
    try:
        core = _usr_core(rows, ids, weights, direction=direction, salt=salt,
                         alpha=alpha, min_n=min_n, max_n=max_n,
                         min_units_per_half=min_units_per_half)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
        base.update({'status': 'not_estimable',
                     'reason': _usr_safe_text(exc, 500) or 'build failed'})
        return base
    if core.get('iut') is None:
        base.update({
            'status': 'not_estimable',
            'reason': 'the one-sided exact p is not estimable in at least one half',
        })
        return base
    base['assignment'] = core['assignment']
    base['halves'] = core['halves']
    base['iut'] = core['iut']
    base['sign_agreement'] = core['sign_agreement']
    base['half_difference'] = core['half_difference']
    # Composition diagnostics: a half carried by one oversized unit is not an
    # independent half-panel even when its token share looks balanced.
    per_half = []
    for half in core['assignment']['halves']:
        idx = half['indices']
        largest = (max(weights[i] for i in idx) if idx else None)
        half_tokens = half['n_tokens']
        largest_share = None
        if idx and largest is not None:
            if half_tokens is not None and half_tokens > 0.0:
                largest_share = float(largest) / float(half_tokens)
            else:
                # Absolute half totals can be unrepresentable at scale.  The
                # share is a ratio, so recompute it from max-normalised
                # weights instead of dividing by inf.
                local_scale = max(weights[i] for i in idx)
                try:
                    denom_n = _usr_math.fsum(
                        weights[i] / local_scale for i in idx)
                    if denom_n > 0.0:
                        largest_share = (largest / local_scale) / denom_n
                except (TypeError, ValueError, OverflowError, ZeroDivisionError):
                    largest_share = None
        if largest_share is not None and not _usr_math.isfinite(largest_share):
            largest_share = None
        per_half.append({
            'name': half['name'],
            'largest_unit_token_share': largest_share,
        })
    base['composition'] = per_half
    warnings = []
    if token_total is None:
        warnings.append(
            'the total scored-token count is beyond float64; absolute token '
            'totals are null and the split uses max-normalised weights')
    shares = [h.get('token_share') for h in core['assignment']['halves']]
    if any(s is None for s in shares):
        warnings.append('token balance is undefined')
    else:
        if min(shares) < share:
            warnings.append(
                f'token shares {shares} are below min_token_share={share}; the '
                'two halves are not comparable in evaluation mass')
        if max(per_half[0]['largest_unit_token_share'] or 0.0,
               per_half[1]['largest_unit_token_share'] or 0.0) > 0.5:
            warnings.append(
                'one evaluation unit contributes more than half of a half\'s '
                'tokens; the half is effectively one unit')
    if id_source != 'supplied':
        warnings.append(
            'unit ids were not supplied; the split uses array positions, which '
            'are only as stable as the recorded unit order')
    if len(rows) < 8:
        warnings.append(
            f'only {len(rows)} seeds; a both-halves IUT at alpha={alpha} is '
            'resolution-limited')
    base['warnings'] = warnings
    if warnings:
        base['status'] = 'warnings'
    base['split_sensitivity'] = (
        _usr_sensitivity(rows, ids, weights, direction=direction, base_salt=salt,
                         alpha=alpha, min_n=min_n, max_n=max_n,
                         min_units_per_half=min_units_per_half,
                         n_salts=n_salts)
        if n_salts > 0 else None)
    return base


# ---------------------------------------------------------------------------
# timestamp-free block and replayable artifact
# ---------------------------------------------------------------------------

def unit_split_replication_block(log_ratios, tokens, *, unit_ids=None,
                                 seeds=None, direction='negative', alpha=0.05,
                                 min_n=6, max_n=UNIT_SPLIT_REPLICATION_MAX_N,
                                 min_units=UNIT_SPLIT_REPLICATION_MIN_UNITS,
                                 min_units_per_half=UNIT_SPLIT_REPLICATION_MIN_UNITS_PER_HALF,
                                 min_token_share=UNIT_SPLIT_REPLICATION_MIN_TOKEN_SHARE,
                                 salt=UNIT_SPLIT_REPLICATION_DEFAULT_SALT,
                                 n_salts=UNIT_SPLIT_REPLICATION_DEFAULT_SALTS):
    """Return the timestamp-free ``inputs``/``result`` block for a report."""
    alpha = _usr_alpha(alpha)
    direction = _usr_direction(direction)
    max_n = _usr_int('max_n', max_n, minimum=2,
                     maximum=UNIT_SPLIT_REPLICATION_HARD_MAX_N)
    min_n = _usr_int('min_n', min_n, minimum=2, maximum=100000)
    min_units = _usr_int('min_units', min_units, minimum=4, maximum=100000)
    min_units_per_half = _usr_int(
        'min_units_per_half', min_units_per_half, minimum=1,
        maximum=max(1, min_units))
    try:
        share = float(min_token_share)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f'min_token_share must be a real number, got {_usr_repr(min_token_share)}') from None
    if not _usr_math.isfinite(share) or not (0.0 < share < 0.5):
        raise ValueError(
            f'min_token_share must be finite and in (0,0.5), got {share!r}')
    salt = _usr_salt(salt)
    n_salts = _usr_int('n_salts', n_salts, minimum=0,
                       maximum=UNIT_SPLIT_REPLICATION_MAX_SALTS)
    if direction is None:
        raise ValueError('a replayable split-half block needs a direction')
    if isinstance(log_ratios, (str, bytes)) or not isinstance(log_ratios, (list, tuple)):
        raise ValueError('log_ratios must be a sequence of per-seed rows')
    raw_rows = list(log_ratios)
    if isinstance(tokens, (str, bytes)) or not isinstance(tokens, (list, tuple)):
        raise ValueError('tokens must be a sequence')
    weights = []
    for value in tokens:
        number = _usr_number(value, 'tokens', positive=True)
        if number is None:
            raise ValueError(f'tokens must be finite and positive, got {_usr_repr(value)}')
        weights.append(number)
    n_units = len(weights)
    if unit_ids is not None:
        if isinstance(unit_ids, (str, bytes)) or not isinstance(unit_ids, (list, tuple)):
            raise ValueError('unit_ids must be a sequence or None')
        ids = []
        for value in unit_ids:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'unit_ids must be non-empty strings, got {_usr_repr(value)}')
            text = value.strip()
            if len(text) > 500:
                raise ValueError('unit_ids must be at most 500 characters')
            ids.append(text)
        if len(ids) != n_units:
            raise ValueError(
                f'unit_ids length {len(ids)} does not match tokens length {n_units}')
        if len(set(ids)) != len(ids):
            raise ValueError('unit_ids contain duplicates')
    else:
        ids = [f'u{index:04d}' for index in range(n_units)]
    if seeds is not None:
        if isinstance(seeds, (str, bytes)) or not isinstance(seeds, (list, tuple)):
            raise ValueError('seeds must be a sequence or None')
        seed_out = []
        for index, value in enumerate(seeds):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f'seeds must be integers, got {_usr_repr(value)}')
            number = int(value)
            if abs(number) > UNIT_SPLIT_REPLICATION_MAX_REPORTABLE_INT:
                raise ValueError(
                    f'seeds[{index}] is outside the strict-JSON '
                    'representable integer range')
            seed_out.append(number)
        if len(seed_out) != len(raw_rows):
            raise ValueError(
                f'seeds length {len(seed_out)} does not match log_ratios length '
                f'{len(raw_rows)}')
        if len(set(seed_out)) != len(seed_out):
            raise ValueError('seeds contain duplicates')
    else:
        seed_out = list(range(len(raw_rows)))
    # Keep a null placeholder for a non-finite entry instead of dropping the
    # row here: replay must see the same input and reproduce the same named
    # seed drop.  ``null`` is strict JSON; ``NaN``/``inf`` are not.
    stored_rows = []
    for index, row in enumerate(raw_rows):
        if isinstance(row, (str, bytes)) or not isinstance(row, (list, tuple)):
            raise ValueError(
                f'log_ratios[{index}] must be a per-unit sequence, got '
                f'{type(row).__name__}')
        if len(row) != n_units:
            raise ValueError(
                f'log_ratios[{index}] has length {len(row)}, expected {n_units}')
        stored = []
        for j, value in enumerate(row):
            number = _usr_number(value, f'log_ratios[{index}][{j}]')
            stored.append(None if number is None else float(number))
        stored_rows.append(stored)
    inputs = {
        'log_ratios': stored_rows,
        'tokens': [float(x) for x in weights],
        'unit_ids': list(ids),
        'seeds': [int(s) for s in seed_out],
        'direction': direction,
        'alpha': float(alpha),
        'min_n': int(min_n),
        'max_n': int(max_n),
        'min_units': int(min_units),
        'min_units_per_half': int(min_units_per_half),
        'min_token_share': float(share),
        'salt': salt,
        'n_salts': int(n_salts),
    }
    result = unit_split_replication(
        inputs['log_ratios'], inputs['tokens'], unit_ids=inputs['unit_ids'],
        seeds=inputs['seeds'], direction=direction, alpha=alpha, min_n=min_n,
        max_n=max_n, min_units=min_units,
        min_units_per_half=min_units_per_half, min_token_share=share,
        salt=salt, n_salts=n_salts)
    return {'kind': UNIT_SPLIT_REPLICATION_BLOCK_KIND, 'inputs': inputs,
            'result': result}


def build_unit_split_replication(log_ratios, tokens, *, unit_ids=None,
                                 seeds=None, direction='negative', alpha=0.05,
                                 min_n=6, max_n=UNIT_SPLIT_REPLICATION_MAX_N,
                                 min_units=UNIT_SPLIT_REPLICATION_MIN_UNITS,
                                 min_units_per_half=UNIT_SPLIT_REPLICATION_MIN_UNITS_PER_HALF,
                                 min_token_share=UNIT_SPLIT_REPLICATION_MIN_TOKEN_SHARE,
                                 salt=UNIT_SPLIT_REPLICATION_DEFAULT_SALT,
                                 n_salts=UNIT_SPLIT_REPLICATION_DEFAULT_SALTS,
                                 source=None, created_at=None):
    """Build a strict-JSON, content-addressed split-half artifact."""
    block = unit_split_replication_block(
        log_ratios, tokens, unit_ids=unit_ids, seeds=seeds,
        direction=direction, alpha=alpha, min_n=min_n, max_n=max_n,
        min_units=min_units, min_units_per_half=min_units_per_half,
        min_token_share=min_token_share, salt=salt, n_salts=n_salts)
    artifact = {
        'kind': UNIT_SPLIT_REPLICATION_KIND,
        'schema': UNIT_SPLIT_REPLICATION_SCHEMA,
        'created_at': (_usr_time.strftime('%Y-%m-%dT%H:%M:%S')
                       if created_at is None else _usr_safe_text(created_at, 80)),
        'source': _usr_safe_text(source, 500),
        'inputs': block['inputs'],
        'result': block['result'],
    }
    artifact['content_sha256'] = _esci_content_sha(artifact)  # noqa: F821
    return artifact


def compare_unit_split_replication_results(stored, replay, *, rtol=1e-9,
                                           atol=1e-12):
    diffs = _esci_compare(stored, replay, 'result', rtol=rtol, atol=atol)  # noqa: F821
    return {'match': not diffs, 'differences': diffs}


def _usr_is_int(value):
    return (isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_)))


def _usr_inputs_problems(inputs):
    """Validate every field a replay consumes, without replaying.

    ``replay=False`` is used by archive/tamper tooling, so a structurally
    plausible but edited input object must still fail closed here.  In
    particular, huge integer labels are rendered with ``_usr_repr`` (CPython
    refuses to format more than ~4300 decimal digits) and malformed nested
    values are named rather than raising.
    """
    problems = []
    if not isinstance(inputs, dict):
        return ['inputs must be an object']
    extra = sorted(set(inputs) - set(UNIT_SPLIT_REPLICATION_INPUT_KEYS),
                   key=_esci_key_text)  # noqa: F821 (part 87)
    missing = [k for k in UNIT_SPLIT_REPLICATION_INPUT_KEYS if k not in inputs]
    if extra or missing:
        problems.append(
            f'inputs keys differ: missing={missing}, '
            f'extra={[_esci_key_text(k) for k in extra]}')  # noqa: F821
        return problems
    try:
        _usr_alpha(inputs.get('alpha'))
    except ValueError as exc:
        problems.append(f'inputs.alpha is malformed: {exc}')
    try:
        _usr_int('inputs.max_n', inputs.get('max_n'), minimum=2,
                 maximum=UNIT_SPLIT_REPLICATION_HARD_MAX_N)
    except ValueError as exc:
        problems.append(f'inputs.max_n is malformed: {exc}')
    try:
        _usr_int('inputs.min_n', inputs.get('min_n'), minimum=2,
                 maximum=100000)
    except ValueError as exc:
        problems.append(f'inputs.min_n is malformed: {exc}')
    try:
        _usr_int('inputs.min_units', inputs.get('min_units'), minimum=4,
                 maximum=100000)
    except ValueError as exc:
        problems.append(f'inputs.min_units is malformed: {exc}')
    try:
        _usr_int('inputs.min_units_per_half',
                 inputs.get('min_units_per_half'), minimum=1,
                 maximum=100000)
    except ValueError as exc:
        problems.append(f'inputs.min_units_per_half is malformed: {exc}')
    try:
        _usr_int('inputs.n_salts', inputs.get('n_salts'), minimum=0,
                 maximum=UNIT_SPLIT_REPLICATION_MAX_SALTS)
    except ValueError as exc:
        problems.append(f'inputs.n_salts is malformed: {exc}')
    try:
        _usr_salt(inputs.get('salt'))
    except ValueError as exc:
        problems.append(f'inputs.salt is malformed: {exc}')
    try:
        direction = _usr_direction(inputs.get('direction'))
        if direction is None:
            problems.append('inputs.direction must be positive or negative '
                            'for a replayable block')
    except ValueError as exc:
        problems.append(f'inputs.direction is malformed: {exc}')
    try:
        share = float(inputs.get('min_token_share'))
        if not _usr_math.isfinite(share) or not (0.0 < share < 0.5):
            raise ValueError(f'must be finite and in (0,0.5), got {share!r}')
    except (TypeError, ValueError, OverflowError) as exc:
        problems.append(f'inputs.min_token_share is malformed: {exc}')

    tokens = inputs.get('tokens')
    weights = None
    if isinstance(tokens, (str, bytes)) or not isinstance(tokens, (list, tuple)):
        problems.append('inputs.tokens must be a sequence')
    else:
        weights = []
        for index, value in enumerate(tokens):
            number = _usr_number(value, f'inputs.tokens[{index}]', positive=True)
            if number is None:
                problems.append(
                    f'inputs.tokens[{index}] must be finite and positive, got '
                    f'{_usr_repr(value)}')
                break
            weights.append(number)

    ids = None
    unit_ids = inputs.get('unit_ids')
    if isinstance(unit_ids, (str, bytes)) or not isinstance(unit_ids, (list, tuple)):
        problems.append('inputs.unit_ids must be a sequence')
    else:
        ids = []
        for index, value in enumerate(unit_ids):
            if not isinstance(value, str) or not value.strip():
                problems.append(
                    f'inputs.unit_ids[{index}] must be a non-empty string, got '
                    f'{_usr_repr(value)}')
                break
            text = value.strip()
            if len(text) > 500:
                problems.append(
                    f'inputs.unit_ids[{index}] is longer than 500 characters')
                break
            ids.append(text)
        if ids is not None and len(set(ids)) != len(ids):
            problems.append('inputs.unit_ids contain duplicates')
        if (ids is not None and weights is not None
                and len(ids) != len(weights)):
            problems.append('inputs.unit_ids and inputs.tokens lengths differ')

    seed_values = inputs.get('seeds')
    seeds = None
    if isinstance(seed_values, (str, bytes)) or not isinstance(seed_values, (list, tuple)):
        problems.append('inputs.seeds must be a sequence')
    else:
        seeds = []
        for index, value in enumerate(seed_values):
            if not _usr_is_int(value):
                problems.append(
                    f'inputs.seeds[{index}] must be an integer, got '
                    f'{_usr_repr(value)}')
                break
            number = int(value)
            if abs(number) > UNIT_SPLIT_REPLICATION_MAX_REPORTABLE_INT:
                problems.append(
                    f'inputs.seeds[{index}] is outside the strict-JSON '
                    'representable integer range')
                break
            seeds.append(number)
        if seeds is not None and len(set(seeds)) != len(seeds):
            problems.append('inputs.seeds contain duplicates')

    rows = inputs.get('log_ratios')
    if isinstance(rows, (str, bytes)) or not isinstance(rows, (list, tuple)):
        problems.append('inputs.log_ratios must be a sequence')
    elif not rows:
        problems.append('inputs.log_ratios must not be empty')
    elif seeds is not None and len(seeds) != len(rows):
        problems.append('inputs.log_ratios and inputs.seeds lengths differ')
    else:
        n_units = len(weights) if weights is not None else None
        for rindex, row in enumerate(rows):
            if isinstance(row, (str, bytes)) or not isinstance(row, (list, tuple)):
                problems.append(
                    f'inputs.log_ratios[{rindex}] must be a sequence')
                break
            if n_units is not None and len(row) != n_units:
                problems.append(
                    f'inputs.log_ratios[{rindex}] has length {len(row)}, '
                    f'expected {n_units}')
                break
            for cindex, value in enumerate(row):
                if value is None:
                    continue
                if _usr_number(
                        value,
                        f'inputs.log_ratios[{rindex}][{cindex}]') is None:
                    problems.append(
                        f'inputs.log_ratios[{rindex}][{cindex}] is not a '
                        f'finite number or null, got {_usr_repr(value)}')
                    break
    return problems


def _usr_result_problems(result, inputs=None):
    problems = []
    if not isinstance(result, dict):
        return ['result must be an object']
    # An untrusted result may carry an extra (possibly huge-int) key.  The
    # strict-JSON diagnostic is produced by the caller; this schema check makes
    # the same tamper fail even after the hash is refreshed.
    extra = sorted(set(result) - set(UNIT_SPLIT_REPLICATION_RESULT_KEYS),
                   key=_esci_key_text)  # noqa: F821 (part 87)
    if extra:
        problems.append(
            f'result keys differ from the schema: '
            f'extra={[_esci_key_text(k) for k in extra]}')  # noqa: F821
    if result.get('kind') != UNIT_SPLIT_REPLICATION_RESULT_KIND:
        problems.append(
            f"result.kind must be {UNIT_SPLIT_REPLICATION_RESULT_KIND!r}, got "
            f'{_usr_repr(result.get("kind"))}')
    if result.get('version') != UNIT_SPLIT_REPLICATION_POLICY['version']:
        problems.append(
            f"result.version must be {UNIT_SPLIT_REPLICATION_POLICY['version']!r}, "
            f'got {_usr_repr(result.get("version"))}')
    status = result.get('status')
    if status not in UNIT_SPLIT_REPLICATION_STATUSES:
        problems.append(
            f'result.status must be one of {UNIT_SPLIT_REPLICATION_STATUSES}, '
            f'got {_usr_repr(status)}')
        return problems
    if status in ('not_estimable', 'not_applicable'):
        reason = result.get('reason')
        if not isinstance(reason, str) or not reason.strip():
            problems.append(f'a {status} result needs a non-empty reason')
        return problems

    # Knobs copied from the inputs must agree even when replay is skipped.
    def _same_number(left, right):
        try:
            a, b = float(left), float(right)
        except (TypeError, ValueError, OverflowError):
            return False
        return _usr_math.isfinite(a) and _usr_math.isfinite(b) and a == b

    declared = inputs if isinstance(inputs, dict) else {}
    if result.get('direction') not in UNIT_SPLIT_REPLICATION_DIRECTIONS:
        problems.append('result.direction must be positive or negative')
    elif ('direction' in declared
          and result.get('direction') != declared.get('direction')):
        problems.append('result.direction does not match inputs.direction')
    alpha = _usr_number(result.get('alpha'), 'result.alpha')
    if alpha is None or not (0.0 < alpha < 1.0):
        problems.append('result.alpha must be a finite probability')
    elif ('alpha' in declared
          and not _same_number(alpha, declared.get('alpha'))):
        problems.append('result.alpha does not match inputs.alpha')
    for key, decoded_key in (('min_n', 'min_n'), ('max_n', 'max_n'),
                             ('min_units', 'min_units'),
                             ('min_units_per_half', 'min_units_per_half')):
        try:
            value = _usr_int(f'result.{key}', result.get(key), minimum=0,
                             maximum=100000)
        except ValueError as exc:
            problems.append(f'result.{key} is malformed: {exc}')
            continue
        if (decoded_key in declared
                and _usr_is_int(declared.get(decoded_key))
                and value != int(declared.get(decoded_key))):
            problems.append(f'result.{key} does not match inputs.{decoded_key}')
    share = _usr_number(result.get('min_token_share'),
                        'result.min_token_share')
    if share is None or not (0.0 < share < 0.5):
        problems.append('result.min_token_share must be finite and in (0,0.5)')
    elif ('min_token_share' in declared
          and not _same_number(share, declared.get('min_token_share'))):
        problems.append(
            'result.min_token_share does not match inputs.min_token_share')

    n_units = None
    try:
        n_units = _usr_int('result.n_units', result.get('n_units'), minimum=0,
                           maximum=1000000)
    except ValueError as exc:
        problems.append(f'result.n_units is malformed: {exc}')
    if (n_units is not None and isinstance(declared.get('tokens'), (list, tuple))
            and n_units != len(declared['tokens'])):
        problems.append('result.n_units does not match inputs.tokens length')

    n_input = None
    try:
        n_input = _usr_int('result.n_seeds_input',
                           result.get('n_seeds_input'), minimum=0,
                           maximum=1000000)
    except ValueError as exc:
        problems.append(f'result.n_seeds_input is malformed: {exc}')
    if (n_input is not None
            and isinstance(declared.get('log_ratios'), (list, tuple))
            and n_input != len(declared['log_ratios'])):
        problems.append(
            'result.n_seeds_input does not match inputs.log_ratios length')

    n_seeds = None
    try:
        n_seeds = _usr_int('result.n_seeds', result.get('n_seeds'), minimum=0,
                           maximum=1000000)
    except ValueError as exc:
        problems.append(f'result.n_seeds is malformed: {exc}')
    seed_values = result.get('seeds')
    parsed_seeds = None
    if not isinstance(seed_values, list):
        problems.append('result.seeds must be a list')
    else:
        parsed_seeds = []
        for index, value in enumerate(seed_values):
            if not _usr_is_int(value):
                problems.append(
                    f'result.seeds[{index}] must be an integer, got '
                    f'{_usr_repr(value)}')
                parsed_seeds = None
                break
            number = int(value)
            if abs(number) > UNIT_SPLIT_REPLICATION_MAX_REPORTABLE_INT:
                problems.append(
                    f'result.seeds[{index}] is outside the strict-JSON '
                    'representable integer range')
                parsed_seeds = None
                break
            parsed_seeds.append(number)
        if parsed_seeds is not None:
            if len(set(parsed_seeds)) != len(parsed_seeds):
                problems.append('result.seeds contains a duplicate seed')
            if n_seeds is not None and len(parsed_seeds) != n_seeds:
                problems.append('result.n_seeds does not match result.seeds length')
            if (isinstance(declared.get('seeds'), (list, tuple))
                    and not isinstance(declared.get('log_ratios'), (list, tuple))):
                problems.append('declared seeds require declared log_ratios')
    dropped = result.get('dropped_seeds')
    if not isinstance(dropped, list):
        problems.append('result.dropped_seeds must be a list')
    else:
        for index, item in enumerate(dropped):
            if not isinstance(item, dict):
                problems.append(f'result.dropped_seeds[{index}] must be an object')
                break
            if not _usr_is_int(item.get('seed')):
                problems.append(
                    f'result.dropped_seeds[{index}].seed must be an integer')
                break
            if not isinstance(item.get('reason'), str) or not item['reason'].strip():
                problems.append(
                    f'result.dropped_seeds[{index}].reason must be a non-empty string')
                break
        if (isinstance(declared.get('seeds'), (list, tuple))
                and parsed_seeds is not None):
            declared_seeds = [int(s) for s in declared['seeds']
                              if _usr_is_int(s)]
            dropped_seeds = [int(item.get('seed')) for item in dropped
                             if isinstance(item, dict)
                             and _usr_is_int(item.get('seed'))]
            if (set(parsed_seeds) | set(dropped_seeds)
                    != set(declared_seeds)):
                problems.append(
                    'result.seeds plus dropped_seeds do not cover inputs.seeds')
            if (isinstance(declared.get('log_ratios'), (list, tuple))
                    and len(declared['log_ratios']) == len(declared_seeds)):
                expected_dropped = []
                for seed, row in zip(declared_seeds, declared['log_ratios']):
                    if not isinstance(row, (list, tuple)):
                        continue
                    if any(value is None for value in row):
                        expected_dropped.append(seed)
                if sorted(expected_dropped) != sorted(dropped_seeds):
                    problems.append(
                        'result.dropped_seeds does not match the rows with a '
                        'null cell in inputs.log_ratios')
                expected_kept = [seed for seed, row in zip(
                    declared_seeds, declared['log_ratios'])
                    if isinstance(row, (list, tuple))
                    and not any(value is None for value in row)]
                if expected_kept != parsed_seeds:
                    problems.append(
                        'result.seeds is not the ordered list of complete rows '
                        'in inputs.log_ratios')

    if result.get('unit_id_source') not in ('supplied', 'position'):
        problems.append("result.unit_id_source must be 'supplied' or 'position'")
    token_total = result.get('token_total')
    if token_total is not None:
        total = _usr_number(token_total, 'result.token_total')
        if total is None or total < 0.0:
            problems.append('result.token_total must be null or a finite '
                            'non-negative number')

    assignment = result.get('assignment')
    assign_halves = None
    if not isinstance(assignment, dict):
        problems.append('result.assignment must be an object')
    else:
        if assignment.get('salt') != declared.get('salt', assignment.get('salt')):
            if 'salt' in declared:
                problems.append('result.assignment.salt does not match inputs.salt')
        digest = assignment.get('digest')
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            problems.append('result.assignment.digest must be a 64-hex string')
        try:
            a_units = _usr_int('result.assignment.n_units',
                               assignment.get('n_units'), minimum=0,
                               maximum=1000000)
        except ValueError as exc:
            problems.append(f'result.assignment.n_units is malformed: {exc}')
            a_units = None
        assign_halves = assignment.get('halves')
        if not isinstance(assign_halves, list) or len(assign_halves) != 2:
            problems.append('result.assignment.halves must have exactly two entries')
            assign_halves = None
        else:
            seen = []
            for i, half in enumerate(assign_halves):
                if not isinstance(half, dict):
                    problems.append(f'result.assignment.halves[{i}] must be an object')
                    assign_halves = None
                    break
                if half.get('name') != f'half_{i}':
                    problems.append(
                        f"result.assignment.halves[{i}].name must be 'half_{i}'")
                indices = half.get('indices')
                unit_ids = half.get('unit_ids')
                if not isinstance(indices, list) or not all(
                        _usr_is_int(x) for x in indices):
                    problems.append(
                        f'result.assignment.halves[{i}].indices must be integers')
                    continue
                if not isinstance(unit_ids, list) or not all(
                        isinstance(x, str) for x in unit_ids):
                    problems.append(
                        f'result.assignment.halves[{i}].unit_ids must be strings')
                    continue
                if len(unit_ids) != len(indices):
                    problems.append(
                        f'result.assignment.halves[{i}] id/index lengths differ')
                try:
                    h_units = _usr_int(
                        f'result.assignment.halves[{i}].n_units',
                        half.get('n_units'), minimum=0, maximum=1000000)
                except ValueError as exc:
                    problems.append(
                        f'result.assignment.halves[{i}].n_units is malformed: {exc}')
                    h_units = None
                if h_units is not None and h_units != len(indices):
                    problems.append(
                        f'result.assignment.halves[{i}].n_units does not match '
                        'its indices')
                h_tokens = half.get('n_tokens')
                if h_tokens is not None:
                    h_tok = _usr_number(
                        h_tokens, f'result.assignment.halves[{i}].n_tokens')
                    if h_tok is None or h_tok < 0.0:
                        problems.append(
                            f'result.assignment.halves[{i}].n_tokens must be '
                            'null or finite non-negative')
                h_share = half.get('token_share')
                if h_share is not None:
                    h_share_n = _usr_number(
                        h_share, f'result.assignment.halves[{i}].token_share')
                    if h_share_n is None or not (0.0 <= h_share_n <= 1.0):
                        problems.append(
                            f'result.assignment.halves[{i}].token_share must be '
                            'null or in [0,1]')
                seen.extend(indices)
            if (a_units is not None and len(seen) != a_units):
                problems.append(
                    'result.assignment.n_units does not match its two halves')
            if seen and sorted(seen) != list(range(len(seen))):
                problems.append(
                    'result.assignment halves are not a partition of the unit '
                    'indices')
            if (isinstance(declared.get('unit_ids'), (list, tuple))
                    and assign_halves is not None):
                declared_ids = declared['unit_ids']
                for i, half in enumerate(assign_halves):
                    if not isinstance(half, dict):
                        continue
                    idx = half.get('indices')
                    ids = half.get('unit_ids')
                    if not isinstance(idx, list) or not isinstance(ids, list):
                        continue
                    if any(not _usr_is_int(j) or j < 0 or j >= len(declared_ids)
                           for j in idx):
                        problems.append(
                            f'result.assignment.halves[{i}].indices are out of '
                            'range for inputs.unit_ids')
                        continue
                    if [declared_ids[j] for j in idx] != ids:
                        problems.append(
                            f'result.assignment.halves[{i}].unit_ids do not '
                            'match inputs.unit_ids at its indices')

    halves = result.get('halves')
    if not isinstance(halves, list) or len(halves) != 2:
        problems.append('an estimable result needs exactly two halves')
        return problems
    component_ps = []
    for i, half in enumerate(halves):
        if not isinstance(half, dict):
            problems.append(f'halves[{i}] must be an object')
            return problems
        if half.get('name') != f'half_{i}':
            problems.append(f"halves[{i}].name must be 'half_{i}'")
        stats_obj = half.get('stats')
        if not isinstance(stats_obj, dict):
            problems.append(f'halves[{i}].stats must be an object')
        else:
            for key in ('mean', 'sd', 'sem'):
                value = stats_obj.get(key)
                if value is None:
                    if key == 'mean':
                        problems.append(f'halves[{i}].stats.mean must be finite')
                    continue
                parsed = _usr_number(value, f'halves[{i}].stats.{key}')
                if parsed is None:
                    problems.append(
                        f'halves[{i}].stats.{key} must be null or finite')
            for key in ('n', 'n_positive', 'n_negative', 'n_zero'):
                try:
                    _usr_int(f'halves[{i}].stats.{key}', stats_obj.get(key),
                             minimum=0, maximum=1000000)
                except ValueError as exc:
                    problems.append(f'halves[{i}].stats.{key} is malformed: {exc}')
        test = half.get('test')
        if not isinstance(test, dict):
            problems.append(f'halves[{i}].test must be an object')
            return problems
        p = _usr_number(test.get('p_one_sided'), f'halves[{i}].p')
        if p is None or not (0.0 <= p <= 1.0):
            problems.append(f'halves[{i}].test.p_one_sided must be a probability')
            return problems
        component_ps.append(float(p))
        if (n_seeds is not None and stats_obj is not None
                and _usr_is_int(stats_obj.get('n'))
                and int(stats_obj.get('n')) != n_seeds):
            problems.append(f'halves[{i}].stats.n does not match result.n_seeds')
        if (assign_halves is not None
                and isinstance(assign_halves[i], dict)
                and half.get('n_units') != assign_halves[i].get('n_units')):
            problems.append(
                f'halves[{i}].n_units does not match result.assignment')
        if (assign_halves is not None
                and isinstance(assign_halves[i], dict)
                and half.get('token_share') != assign_halves[i].get('token_share')):
            problems.append(
                f'halves[{i}].token_share does not match result.assignment')
    iut = result.get('iut')
    if not isinstance(iut, dict):
        problems.append('iut must be an object')
        return problems
    p = _usr_number(iut.get('p_iut'), 'iut.p_iut')
    if p is None or not (0.0 <= p <= 1.0):
        problems.append('iut.p_iut must be a probability')
    elif component_ps and p + 1e-12 < max(component_ps):
        problems.append('iut.p_iut is smaller than a component one-sided p; '
                        'the intersection-union maximum was not applied')
    elif component_ps and abs(p - max(component_ps)) > 1e-12:
        problems.append('iut.p_iut is not the maximum of the two component '
                        'one-sided p-values')
    for key, component in (('p_half_0', 0), ('p_half_1', 1)):
        value = _usr_number(iut.get(key), f'iut.{key}')
        if value is None or not (0.0 <= value <= 1.0):
            problems.append(f'iut.{key} must be a probability')
        elif component_ps and abs(value - component_ps[component]) > 1e-12:
            problems.append(f'iut.{key} does not match the corresponding half')
    if not isinstance(iut.get('method'), str) or not iut['method'].strip():
        problems.append('iut.method must be a non-empty string')
    iut_alpha = _usr_number(iut.get('alpha'), 'iut.alpha')
    if iut_alpha is None or not (0.0 < iut_alpha < 1.0):
        problems.append('iut.alpha must be a finite probability')
    elif alpha is not None and abs(iut_alpha - alpha) > 1e-12:
        problems.append('iut.alpha does not match result.alpha')
    iut_direction = iut.get('direction')
    if iut_direction not in UNIT_SPLIT_REPLICATION_DIRECTIONS:
        problems.append('iut.direction must be positive or negative')
    elif iut_direction != result.get('direction'):
        problems.append('iut.direction does not match result.direction')
    if iut.get('claim') not in (
            'both_halves_support_direction',
            'direction_consistent_but_not_jointly_significant',
            'one_half_significant', 'halves_discordant',
            'direction_not_supported'):
        problems.append(
            f"iut.claim is not one of the documented claims, got "
            f'{_usr_repr(iut.get("claim"))}')
    if not isinstance(iut.get('reject'), (bool, np.bool_)):
        problems.append('iut.reject must be a boolean')
    elif (p is not None and alpha is not None
          and bool(iut.get('reject')) != bool(p <= alpha)):
        problems.append('iut.reject does not agree with p_iut <= alpha')
    if (result.get('sign_agreement') not in
            ('both_negative', 'both_positive', 'discordant',
             'zero_in_one_half', 'unavailable')):
        problems.append('result.sign_agreement is not a documented value')
    diff = result.get('half_difference')
    if not isinstance(diff, dict):
        problems.append('result.half_difference must be an object')
    else:
        for key in ('mean', 'sd', 'sem'):
            value = diff.get(key)
            if value is not None and _usr_number(
                    value, f'half_difference.{key}') is None:
                problems.append(f'half_difference.{key} must be null or finite')
        ci = diff.get('ci95')
        if ci is not None:
            if (not isinstance(ci, (list, tuple)) or len(ci) != 2
                    or any(_usr_number(x, 'half_difference.ci95') is None
                           for x in ci)):
                problems.append('half_difference.ci95 must be null or a finite pair')
            elif float(ci[0]) > float(ci[1]):
                problems.append('half_difference.ci95 must be ordered')
    composition = result.get('composition')
    if not isinstance(composition, list) or len(composition) != 2:
        problems.append('result.composition must have exactly two entries')
    else:
        for i, item in enumerate(composition):
            if not isinstance(item, dict):
                problems.append(f'result.composition[{i}] must be an object')
                continue
            value = item.get('largest_unit_token_share')
            if value is not None:
                parsed = _usr_number(
                    value, f'composition[{i}].largest_unit_token_share')
                if parsed is None or not (0.0 <= parsed <= 1.0):
                    problems.append(
                        f'composition[{i}].largest_unit_token_share must be '
                        'null or in [0,1]')
    warnings = result.get('warnings')
    if not isinstance(warnings, list) or not all(
            isinstance(w, str) for w in warnings):
        problems.append('result.warnings must be a list of strings')
    sens = result.get('split_sensitivity')
    declared_n_salts = declared.get('n_salts')
    if sens is None:
        if declared_n_salts:
            problems.append('split_sensitivity is null but inputs.n_salts > 0')
    elif not isinstance(sens, dict):
        problems.append('result.split_sensitivity must be an object or null')
    elif not declared_n_salts:
        problems.append('split_sensitivity is present but inputs.n_salts is 0')
    else:
        parsed_sens = {}
        for key in ('n_salts_requested', 'n_salts_used', 'n_salts_skipped',
                    'n_estimable', 'n_pass', 'n_distinct_assignments'):
            try:
                parsed_sens[key] = _usr_int(
                    f'split_sensitivity.{key}', sens.get(key), minimum=0,
                    maximum=1000000)
            except ValueError as exc:
                problems.append(f'split_sensitivity.{key} is malformed: {exc}')
        for key in ('pass_fraction', 'p_min', 'p_median', 'p_max'):
            value = sens.get(key)
            if value is None:
                continue
            parsed = _usr_number(value, f'split_sensitivity.{key}')
            if parsed is None or not (0.0 <= parsed <= 1.0):
                problems.append(
                    f'split_sensitivity.{key} must be null or in [0,1]')
        if not isinstance(sens.get('truncated'), (bool, np.bool_)):
            problems.append('split_sensitivity.truncated must be a boolean')
        if ('n_pass' in parsed_sens and 'n_estimable' in parsed_sens
                and parsed_sens['n_pass'] > parsed_sens['n_estimable']):
            problems.append(
                'split_sensitivity.n_pass exceeds n_estimable')
        if ('n_estimable' in parsed_sens and 'n_salts_used' in parsed_sens
                and parsed_sens['n_estimable'] > parsed_sens['n_salts_used']):
            problems.append(
                'split_sensitivity.n_estimable exceeds n_salts_used')
        if ('n_salts_used' in parsed_sens
                and 'n_salts_requested' in parsed_sens
                and parsed_sens['n_salts_used']
                > parsed_sens['n_salts_requested']):
            problems.append(
                'split_sensitivity.n_salts_used exceeds n_salts_requested')
        if (_usr_is_int(declared_n_salts)
                and 'n_salts_requested' in parsed_sens
                and parsed_sens['n_salts_requested'] != int(declared_n_salts)):
            problems.append(
                'split_sensitivity.n_salts_requested does not match '
                'inputs.n_salts')
    return problems


def _usr_validate_common(artifact, *, block=False):
    out = {
        'kind': ('unit_split_replication_block_validation' if block
                 else 'unit_split_replication_validation'),
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
    expected = (UNIT_SPLIT_REPLICATION_BLOCK_KEYS if block
                else UNIT_SPLIT_REPLICATION_ARTIFACT_KEYS)
    try:
        strict = _esci_strict_json_problem(  # noqa: F821
            'block' if block else 'artifact', artifact)
    except Exception:
        strict = None
    if strict:
        out['problems'].append(strict)
    extra = sorted(set(artifact) - set(expected), key=_esci_key_text)  # noqa: F821
    missing = [k for k in expected if k not in artifact]
    if extra or missing:
        out['problems'].append(
            f'artifact keys differ from the schema: missing={missing}, '
            f'extra={[_esci_key_text(k) for k in extra]}')  # noqa: F821
    want_kind = (UNIT_SPLIT_REPLICATION_BLOCK_KIND if block
                 else UNIT_SPLIT_REPLICATION_KIND)
    if artifact.get('kind') != want_kind:
        out['problems'].append(
            f'artifact.kind must be {want_kind!r}, got '
            f'{_usr_repr(artifact.get("kind"))}')
    if not block and artifact.get('schema') != UNIT_SPLIT_REPLICATION_SCHEMA:
        out['problems'].append(
            f'artifact.schema must be {UNIT_SPLIT_REPLICATION_SCHEMA}, got '
            f'{_usr_repr(artifact.get("schema"))}')
    try:
        forbidden = _esci_forbidden_paths(artifact)  # noqa: F821
    except Exception as exc:
        out['problems'].append(
            'artifact cannot be inspected for frozen-decision fields: '
            f'{type(exc).__name__}: {_usr_repr(exc)}')
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
                actual = _esci_content_sha(artifact)  # noqa: F821
            except Exception as exc:
                out['problems'].append(
                    'content hash is not computable: '
                    f'{type(exc).__name__}: {_usr_repr(exc)}')
                actual = None
            if actual is not None and actual != recorded:
                out['problems'].append(
                    'artifact.content_sha256 does not match the recorded body')
    inputs = artifact.get('inputs')
    input_problems = []
    if not isinstance(inputs, dict):
        out['problems'].append('artifact.inputs must be an object')
    else:
        input_problems = _usr_inputs_problems(inputs)
        out['problems'].extend(input_problems)
    result = artifact.get('result')
    if not isinstance(result, dict):
        out['problems'].append('artifact.result must be an object')
    else:
        out['result_status'] = (result.get('status')
                                if isinstance(result.get('status'), str)
                                else _usr_repr(result.get('status')))
        out['problems'].extend(_usr_result_problems(result, inputs))
        # Re-derive the deterministic assignment from the declared inputs.
        # This is not a full replay, but it means replay=False (archive/tamper
        # tooling) still rejects an edited salt, token vector or unit id list;
        # those fields are supposed to explain the recorded partition.
        if (not input_problems
                and result.get('status') in ('ok', 'warnings')):
            try:
                expected_assignment = unit_split_assignment(
                    inputs['unit_ids'], inputs['tokens'],
                    salt=inputs['salt'],
                    min_units_per_half=inputs['min_units_per_half'])
            except (ValueError, TypeError, OverflowError) as exc:
                out['problems'].append(
                    'assignment is not re-derivable from inputs: '
                    f'{_usr_repr(exc)}')
            else:
                recorded = result.get('assignment')
                if (not isinstance(recorded, dict)
                        or recorded.get('digest')
                        != expected_assignment.get('digest')):
                    out['problems'].append(
                        'result.assignment.digest does not match the partition '
                        're-derived from inputs.unit_ids/tokens/salt')
                elif recorded != expected_assignment:
                    out['problems'].append(
                        'result.assignment body does not match the partition '
                        're-derived from inputs.unit_ids/tokens/salt')
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


def _usr_replay(inputs):
    return unit_split_replication(
        inputs['log_ratios'], inputs['tokens'],
        unit_ids=inputs['unit_ids'], seeds=inputs['seeds'],
        direction=inputs['direction'], alpha=inputs['alpha'],
        min_n=inputs['min_n'], max_n=inputs['max_n'],
        min_units=inputs['min_units'],
        min_units_per_half=inputs['min_units_per_half'],
        min_token_share=inputs['min_token_share'], salt=inputs['salt'],
        n_salts=inputs['n_salts'])


def _usr_finish(out, stored, replay):
    out['replayed'] = True
    cmp = compare_unit_split_replication_results(stored, replay)
    if not cmp['match']:
        out['problems'].append('replayed result differs from the stored result')
        out['warnings'].extend(cmp['differences'][:8])
        return out
    out['result'] = _esci_jsonable(replay)  # noqa: F821
    if replay.get('status') == 'not_estimable':
        out['status'] = 'not_estimable'
    elif replay.get('status') == 'not_applicable':
        out['status'] = 'not_applicable'
    elif replay.get('status') == 'warnings':
        out['status'] = 'warnings'
        out['warnings'].extend(list(replay.get('warnings') or []))
    else:
        out['status'] = 'ok'
    return out


def validate_unit_split_replication(artifact, *, replay=True):
    out = _usr_validate_common(artifact, block=False)
    if out['problems']:
        return out
    if not replay:
        status = artifact['result'].get('status')
        out['status'] = status if status in UNIT_SPLIT_REPLICATION_STATUSES else 'warnings'
        out['result'] = _esci_jsonable(artifact['result'])  # noqa: F821
        return out
    try:
        replayed = _usr_replay(artifact['inputs'])
    except Exception as exc:
        out['problems'].append(
            f'replay failed: {type(exc).__name__}: {_usr_repr(exc)}')
        return out
    return _usr_finish(out, artifact['result'], replayed)


def validate_unit_split_replication_block(block, *, replay=True):
    out = _usr_validate_common(block, block=True)
    if out['problems']:
        return out
    if not replay:
        status = block['result'].get('status')
        out['status'] = status if status in UNIT_SPLIT_REPLICATION_STATUSES else 'warnings'
        out['result'] = _esci_jsonable(block['result'])  # noqa: F821
        return out
    try:
        replayed = _usr_replay(block['inputs'])
    except Exception as exc:
        out['problems'].append(
            f'replay failed: {type(exc).__name__}: {_usr_repr(exc)}')
        return out
    return _usr_finish(out, block['result'], replayed)


def write_unit_split_replication(path_or_dir, artifact, *, overwrite=False,
                                 name='unit_split_replication.json'):
    verdict = validate_unit_split_replication(artifact)
    if verdict['status'] == 'fail':
        raise ValueError('refusing to write an invalid artifact: '
                         + '; '.join(verdict['problems'][:4]))
    if (isinstance(path_or_dir, (str, _usr_os.PathLike))
            and str(path_or_dir).endswith('.json')):
        path = _usr_os.fspath(path_or_dir)
    else:
        path = _usr_os.path.join(_usr_os.fspath(path_or_dir), name)
    _usr_os.makedirs(_usr_os.path.dirname(_usr_os.path.abspath(path)),
                     exist_ok=True)
    existing = None
    if _usr_os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as fh:
                existing = _usr_json.load(fh)
        except Exception:
            existing = None
    if isinstance(existing, dict):
        if existing.get('content_sha256') == artifact.get('content_sha256'):
            return {'status': 'noop', 'path': _usr_os.path.abspath(path)}
        if not overwrite:
            return {'status': 'conflict',
                    'path': _usr_os.path.abspath(path),
                    'note': ('an artifact with different content already '
                             'exists; pass overwrite=True to replace it')}
    atomic_write_json(path, artifact)  # noqa: F821 (part 08)
    return {'status': 'written', 'path': _usr_os.path.abspath(path)}


def unit_split_replication_markdown(result, *, title='Split-half consistency across evaluation units'):
    title = str(title)[:240]
    if not isinstance(result, dict):
        return f'**{title}**: not estimable (not an object).'
    if result.get('status') in ('not_estimable', 'not_applicable'):
        return (f'**{title}**: {result.get("status")} — '
                f'{_usr_repr(result.get("reason") or "no reason", limit=500)}.')

    def _fmt(x, fmt='{:+.4f}'):
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return '—'
        x = float(x)
        return fmt.format(x) if _usr_math.isfinite(x) else '—'

    def _p(x):
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return '—'
        x = float(x)
        return '{:.4g}'.format(x) if _usr_math.isfinite(x) else '—'

    iut = result.get('iut') or {}
    lines = [
        f'**{title}** (direction `{result.get("direction")}`, '
        f'n_seeds={result.get("n_seeds")}, alpha={result.get("alpha")}, '
        f'split=`{result.get("assignment", {}).get("salt")}`)',
        '',
        '| half | units | tokens | token share | mean ΔlogPPL | p (one-sided exact) |',
        '|---|---|---|---|---|---|',
    ]
    for half in result.get('halves') or []:
        stats = half.get('stats') or {}
        test = half.get('test') or {}
        lines.append(
            f'| {half.get("name")} | {half.get("n_units")} | '
            f'{_fmt(half.get("n_tokens"), "{:.0f}")} | '
            f'{_fmt(half.get("token_share"), "{:.1%}")} | '
            f'{_fmt(stats.get("mean"))} | {_p(test.get("p_one_sided"))} |')
    lines.append('')
    lines.append(f'| both-halves IUT | value |')
    lines.append('|---|---|')
    lines.append(f'| p (max of halves) | {_p(iut.get("p_iut"))} |')
    lines.append(f'| claim | `{iut.get("claim")}` |')
    lines.append(f'| sign agreement | `{result.get("sign_agreement")}` |')
    sens = result.get('split_sensitivity') or {}
    if sens:
        lines.append(
            f'| split sensitivity | {sens.get("n_pass")}/{sens.get("n_estimable")} '
            f'salts pass; p median {_p(sens.get("p_median"))} |')
    lines.append('')
    lines.append('> Boundary: this is an internal split-half consistency readout, '
                 'not independent replication: the halves are disjoint units of '
                 'the same fixed evaluation slice and share the same seeds. '
                 'It adds no p-value to the frozen Holm family and never changes '
                 'the confirmatory verdict. The split is chosen by a fixed hash '
                 'salt, not by the outcome; the sensitivity curve is descriptive '
                 'and the best salt is not the reported split.')
    return '\n'.join(lines)
