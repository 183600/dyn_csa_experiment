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
        return f'<{type(value).__name__}>'
    return text if len(text) <= limit else text[:limit] + '...'


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
        raise ValueError(f'{name} must be in [{minimum}, {maximum}], got {out}')
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
    halves = ([], [])
    totals = [0.0, 0.0]
    counts = [0, 0]
    for index in order:
        # Token balance first; ties fall back to the smaller unit count and
        # then to half 0, so the assignment is a total order with no
        # dependence on iteration order.
        left = (totals[0], counts[0], 0)
        right = (totals[1], counts[1], 1)
        target = 0 if left <= right else 1
        halves[target].append(index)
        totals[target] += weights[index]
        counts[target] += 1
    halves = [sorted(h) for h in halves]
    body = {
        'salt': salt,
        'halves': [[ids[i] for i in halves[0]], [ids[i] for i in halves[1]]],
        'tokens': [weights[i] for i in range(len(ids))],
    }
    digest = _usr_hashlib.sha256(
        _usr_json.dumps(body, sort_keys=True, separators=(',', ':'),
                        ensure_ascii=False, allow_nan=False).encode('utf-8')
    ).hexdigest()
    total_tokens = float(sum(weights))
    shares = [(float(sum(weights[i] for i in h)) / total_tokens
               if total_tokens > 0.0 else None) for h in halves]
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
                'n_tokens': float(totals[0]),
                'token_share': shares[0],
            },
            {
                'name': 'half_1',
                'indices': [int(i) for i in halves[1]],
                'unit_ids': [ids[i] for i in halves[1]],
                'n_units': int(len(halves[1])),
                'n_tokens': float(totals[1]),
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
    values = [float(x) for x in deltas]
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
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else None
    return {
        'n': int(n),
        'mean': mean,
        'sd': sd,
        'sem': (float(sd / _usr_math.sqrt(n)) if sd is not None else None),
        'min': float(min(values)),
        'max': float(max(values)),
        'n_positive': int(sum(1 for x in values if x > 0.0)),
        'n_negative': int(sum(1 for x in values if x < 0.0)),
        'n_zero': int(sum(1 for x in values if x == 0.0)),
    }


def _usr_half_effects(rows, indices, tokens):
    weights = [float(tokens[i]) for i in indices]
    denom = float(sum(weights))
    effects = []
    for row in rows:
        numer = _usr_math.fsum(float(row[i]) * w for i, w in zip(indices, weights))
        effects.append(numer / denom)
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
        stats = _usr_stats(effects)
        test = _usr_one_sided(effects, direction)
        halves.append({
            'name': half['name'],
            'n_units': half['n_units'],
            'n_seeds': int(len(effects)),
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
    mean0 = halves[0]['stats']['mean']
    mean1 = halves[1]['stats']['mean']
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
    diff = [a - b for a, b in zip(effects_by_half[0], effects_by_half[1])]
    diff_stats = _usr_stats(diff)
    t_ci = _esci_t_interval(diff, alpha)  # noqa: F821 (part 87)
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
            'mean': diff_stats['mean'],
            'sd': diff_stats['sd'],
            'sem': diff_stats['sem'],
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
            ids.append(value.strip())
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
        for value in seeds:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f'seeds must be integers, got {_usr_repr(value)}')
            number = int(value)
            if abs(number) > UNIT_SPLIT_REPLICATION_MAX_REPORTABLE_INT:
                raise ValueError(
                    f'seed {_usr_repr(value)} is outside the strict-JSON '
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
    per_salt_patterns = 2 * (1 << int(n))
    cap = max(1, int(budget // max(1, per_salt_patterns)))
    requested = int(n_salts)
    used = min(requested, cap)
    ps = []
    passed = []
    digests = set()
    skipped = 0
    for k in range(used):
        salt = base_salt if k == 0 else f'{base_salt}#{k}'
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
            f'requested {requested} salts but the deterministic work cap '
            f'({budget} pattern evaluations) allows {used} at n={n}')
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
    base.update({
        'n_units': int(len(weights)),
        'n_seeds_input': int(len(list(log_ratios))),
        'n_seeds': int(len(rows)),
        'seeds': kept_seeds,
        'dropped_seeds': dropped,
        'unit_id_source': id_source,
        'token_total': float(sum(weights)),
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
    except ValueError as exc:
        base.update({'status': 'not_estimable', 'reason': str(exc)})
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
        per_half.append({
            'name': half['name'],
            'largest_unit_token_share': (float(largest) / float(half['n_tokens'])
                                         if idx and half['n_tokens'] > 0.0 else None),
        })
    base['composition'] = per_half
    warnings = []
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
            ids.append(value.strip())
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
        for value in seeds:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f'seeds must be integers, got {_usr_repr(value)}')
            number = int(value)
            if abs(number) > UNIT_SPLIT_REPLICATION_MAX_REPORTABLE_INT:
                raise ValueError(
                    f'seed {_usr_repr(value)} is outside the strict-JSON '
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
                       if created_at is None else str(created_at)[:80]),
        'source': (None if source is None else str(source)[:500]),
        'inputs': block['inputs'],
        'result': block['result'],
    }
    artifact['content_sha256'] = _esci_content_sha(artifact)  # noqa: F821
    return artifact


def compare_unit_split_replication_results(stored, replay, *, rtol=1e-9,
                                           atol=1e-12):
    diffs = _esci_compare(stored, replay, 'result', rtol=rtol, atol=atol)  # noqa: F821
    return {'match': not diffs, 'differences': diffs}


def _usr_result_problems(result):
    problems = []
    if not isinstance(result, dict):
        return ['result must be an object']
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
        if not isinstance(result.get('reason'), str) or not result['reason'].strip():
            problems.append(f'a {status} result needs a non-empty reason')
        return problems
    halves = result.get('halves')
    if not isinstance(halves, list) or len(halves) != 2:
        problems.append('an estimable result needs exactly two halves')
        return problems
    for i, half in enumerate(halves):
        if not isinstance(half, dict):
            problems.append(f'halves[{i}] must be an object')
            return problems
        test = half.get('test')
        if not isinstance(test, dict):
            problems.append(f'halves[{i}].test must be an object')
            return problems
        p = _usr_number(test.get('p_one_sided'), f'halves[{i}].p')
        if p is None or not (0.0 <= p <= 1.0):
            problems.append(f'halves[{i}].test.p_one_sided must be a probability')
            return problems
    iut = result.get('iut')
    if not isinstance(iut, dict):
        problems.append('iut must be an object')
        return problems
    p = _usr_number(iut.get('p_iut'), 'iut.p_iut')
    if p is None or not (0.0 <= p <= 1.0):
        problems.append('iut.p_iut must be a probability')
    elif p + 1e-12 < max(float(halves[0]['test']['p_one_sided']),
                         float(halves[1]['test']['p_one_sided'])):
        problems.append('iut.p_iut is smaller than a component one-sided p; '
                        'the intersection-union maximum was not applied')
    assignment = result.get('assignment')
    if not isinstance(assignment, dict) or assignment.get('n_units') != sum(
            int(h.get('n_units') or 0) for h in assignment.get('halves') or []):
        problems.append('assignment.n_units does not match its two halves')
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
    extra = sorted(set(artifact) - set(expected))
    missing = [k for k in expected if k not in artifact]
    if extra or missing:
        out['problems'].append(
            f'artifact keys differ from the schema: missing={missing}, extra={extra}')
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
    if not isinstance(inputs, dict):
        out['problems'].append('artifact.inputs must be an object')
    else:
        extra_in = sorted(set(inputs) - set(UNIT_SPLIT_REPLICATION_INPUT_KEYS))
        missing_in = [k for k in UNIT_SPLIT_REPLICATION_INPUT_KEYS if k not in inputs]
        if extra_in or missing_in:
            out['problems'].append(
                f'inputs keys differ: missing={missing_in}, extra={extra_in}')
        else:
            try:
                _usr_alpha(inputs.get('alpha'))
                _usr_int('max_n', inputs.get('max_n'), minimum=2,
                         maximum=UNIT_SPLIT_REPLICATION_HARD_MAX_N)
                _usr_int('min_n', inputs.get('min_n'), minimum=2,
                         maximum=100000)
                _usr_int('min_units', inputs.get('min_units'), minimum=4,
                         maximum=100000)
                _usr_int('min_units_per_half', inputs.get('min_units_per_half'),
                         minimum=1, maximum=100000)
                _usr_salt(inputs.get('salt'))
                _usr_int('n_salts', inputs.get('n_salts'), minimum=0,
                         maximum=UNIT_SPLIT_REPLICATION_MAX_SALTS)
                _usr_direction(inputs.get('direction'))
            except ValueError as exc:
                out['problems'].append(f'inputs are malformed: {exc}')
    result = artifact.get('result')
    if not isinstance(result, dict):
        out['problems'].append('artifact.result must be an object')
    else:
        out['result_status'] = (result.get('status')
                                if isinstance(result.get('status'), str)
                                else _usr_repr(result.get('status')))
        out['problems'].extend(_usr_result_problems(result))
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
