# ---- a curve is not a row of pointwise p-values: simultaneous inference and
#      intersection-union equivalence over ordered conditions ----
# The panel reports one effect per evaluation length (or per budget / dose).
# Each row has its own exact sign-flip p and t interval, and the report also
# tests effect modification against a reference.  That still leaves three
# claims the row-wise table cannot support:
#
#   * "the effect has the same sign at EVERY condition" needs a simultaneous
#     statement over the whole curve, not the conjunction of individually
#     significant rows;
#   * "the effect is practically constant" needs an equivalence statement over
#     ALL conditions (and all pairwise differences), not a non-significant
#     interaction;
#   * "the curve is flat" needs a single family test whose null is equality of
#     all condition means, not one p-value per adjacent pair.
#
# This fragment adds those three readouts on the same per-seed matrix:
#
#   * exact sign-flip max-T (Westfall-Young) over the condition family gives
#     per-condition adjusted p-values and simultaneous intervals, reusing the
#     repository's `maxt_signflip_adjust` (score-studentised, sign-invariant);
#   * a single exact sign-flip RANGE test of H0: all condition means equal
#     (the range is invariant to a common location shift, so the sign-flip
#     null is the flatness null);
#   * intersection-union (IUT) one-sided tests for "all conditions improve"
#     and IUT TOST for "all conditions are within the declared margin" and
#     "all pairwise differences are within the flatness margin".  An IUT needs
#     no multiplicity correction: the claim is the intersection, so the
#     maximum of the component p-values is a valid p-value.
#
# Every condition uses the same complete seed set; a seed missing one condition
# is dropped from the whole curve and named, never paired positionally.  The
# block is exploratory/descriptive: no frozen p/Holm/confirmed change, no
# training, no split read, no CODE_SEMANTICS bump.

import json
import math
import os
import statistics
import time

import numpy as np

CURVE_INFERENCE_KIND = 'curve_inference'
CURVE_INFERENCE_BLOCK_KIND = 'curve_inference_block'
CURVE_INFERENCE_SCHEMA = 1
CURVE_INFERENCE_MAX_N = 20
CURVE_INFERENCE_HARD_MAX_N = 20
CURVE_INFERENCE_MAX_CONDITIONS = 64
CURVE_INFERENCE_MAX_SEEDS = 4096
# Seeds are labels, not measurements, but an artifact is strict JSON: keep the
# declared value inside the range CPython can spell (the same bound the other
# scale-safe artifacts use) so a hash/validation diagnostic is possible.
CURVE_INFERENCE_MAX_REPORTABLE_INT = 10 ** 4000
CURVE_INFERENCE_MAX_PAIRS = 200
CURVE_INFERENCE_TOL = 1e-12

CURVE_INFERENCE_POLICY = {
    'version': 'curve-inference-1',
    'effect_definition': ('one paired per-seed effect per ordered condition '
                          '(the caller supplies the scale; for PPL curves the '
                          'natural choice is log(PPL_a)-log(PPL_b))'),
    'family': ('exact sign-flip Westfall-Young max-T over all declared '
               'conditions; per-condition adjusted p-values and studentised '
               'simultaneous intervals'),
    'flatness': ('one exact sign-flip test of the range of condition means; '
                 'the range is invariant to a common location shift, so the '
                 'sign-flip null is the all-means-equal null'),
    'uniform_direction': ('intersection-union test: the maximum one-sided '
                          'exact sign-flip p over conditions is a valid p for '
                          '"all means improve"; no multiplicity correction is '
                          'needed for an intersection claim'),
    'uniform_equivalence': ('intersection-union TOST: the maximum exact '
                            'sign-flip TOST p over conditions is a valid p for '
                            '"all means are within the margin"'),
    'flatness_equivalence': ('intersection-union TOST over all pairwise '
                             'differences; the maximum pairwise TOST p is a '
                             'valid p for "the whole curve is within the '
                             'flatness margin"'),
    'completeness': ('a condition is estimated only when every declared seed '
                     'has a finite effect at every condition; incomplete '
                     'seeds are dropped from the whole curve and named'),
    'role': ('post-hoc curve-level inference; it creates no confirmatory '
             'p-value, enters no frozen Holm family and does not change '
             'p/Holm/verified/confirmed or any PPL'),
    'boundary': ('seed-level sign symmetry is the same assumption as the '
                 'repository primary test.  The max-T interval is t-style, '
                 'not an exact inversion of the multivariate sign-flip test; '
                 'the exact objects are the adjusted p-values, the range test '
                 'and the IUT p-values.  A non-significant flatness test is '
                 'not equivalence unless the flatness IUT rejects.'),
}

CURVE_INFERENCE_ARTIFACT_KEYS = (
    'kind', 'schema', 'created_at', 'source', 'inputs', 'result',
    'content_sha256',
)
CURVE_INFERENCE_INPUT_KEYS = (
    'conditions', 'effects', 'seeds', 'alpha', 'min_n', 'margin',
    'flatness_margin', 'max_n',
)
CURVE_INFERENCE_FORBIDDEN_KEYS = frozenset({
    'confirmed', 'confirmed_effective', 'p_holm', 'verified', 'verdict',
    'decision', 'frozen_decision', 'final_decision', 'holm', 'confirmed_effect',
})


def _cv_repr(value, limit=160):
    try:
        text = repr(value)
    except Exception:
        return f'<unrepresentable {type(value).__name__}>'
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _cv_text(value, name, *, limit=500):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string, got '
                         f'{_cv_repr(value)}')
    text = value.strip()
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _cv_key_text(value, limit=120):
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit - 3] + '...'
    try:
        text = str(value)
    except Exception:
        return _cv_repr(value, limit=limit)
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _cv_int(value, name, *, minimum=None, maximum=None):
    out = _as_seed_int(value)  # noqa: F821 (part_13)
    if out is None:
        raise ValueError(f'{name} must be an exact integer, got '
                         f'{_cv_repr(value)}')
    if minimum is not None and out < minimum:
        raise ValueError(f'{name} must be >= {minimum}, got {_cv_repr(out)}')
    if maximum is not None and out > maximum:
        raise ValueError(f'{name} must be <= {maximum}, got {_cv_repr(out)}')
    return out


def _cv_number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a finite real number, got '
                         f'{_cv_repr(value)}')
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must be a finite real number, got '
                         f'{_cv_repr(value)}') from None
    if not math.isfinite(out):
        raise ValueError(f'{name} must be finite, got {_cv_repr(value)}')
    if positive and not (out > 0.0):
        raise ValueError(f'{name} must be > 0, got {_cv_repr(out)}')
    if nonnegative and not (out >= 0.0):
        raise ValueError(f'{name} must be >= 0, got {_cv_repr(out)}')
    return out


def _cv_jsonable(value):
    return _json_clean(_ba_jsonable_safe(value))  # noqa: F821 (parts 08/85)


def _cv_content_sha(artifact):
    body = {k: v for k, v in artifact.items()
            if k not in ('created_at', 'content_sha256')}
    return _sha256_json(_cv_jsonable(body))  # noqa: F821 (part_00)


def _cv_forbidden_paths(obj, path=''):
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = _cv_key_text(key)
            here = f'{path}.{name}' if path else name
            if name in CURVE_INFERENCE_FORBIDDEN_KEYS:
                found.append(here)
            found.extend(_cv_forbidden_paths(value, here))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            found.extend(_cv_forbidden_paths(value, f'{path}[{index}]'))
    return found


def _cv_strict_json_problem(label, body):
    try:
        json.dumps(body, allow_nan=False)
    except Exception as exc:
        detail = str(exc)
        if ('Exceeds the limit' in detail
                or 'integer string conversion' in detail):
            return (f'{label} carries an unrepresentable int (beyond '
                    "CPython's decimal-conversion limit) and cannot be "
                    'strict-JSON serialised')
        return (f'{label} is not strict-JSON serialisable: '
                f'{type(exc).__name__}: {_cv_repr(exc)}')
    return None


def _cv_normalise_conditions(conditions):
    if isinstance(conditions, (str, bytes, bytearray, dict)):
        raise ValueError('conditions must be an ordered iterable of labels, '
                         f'not a {type(conditions).__name__}')
    # A sized over-large declaration can be rejected without materialising it
    # (a future panel may pass a range or a huge list of candidate lengths).
    try:
        declared_len = len(conditions)
    except (TypeError, OverflowError):
        declared_len = None
    if (declared_len is not None
            and declared_len > CURVE_INFERENCE_MAX_CONDITIONS):
        raise ValueError(f'conditions has {declared_len} labels; the supported '
                         f'maximum is {CURVE_INFERENCE_MAX_CONDITIONS}')
    try:
        raw = list(conditions)
    except TypeError:
        raise ValueError('conditions must be an ordered iterable of labels') from None
    if not raw:
        raise ValueError('conditions must contain at least one label')
    if len(raw) > CURVE_INFERENCE_MAX_CONDITIONS:
        raise ValueError(f'conditions has {len(raw)} labels; the supported '
                         f'maximum is {CURVE_INFERENCE_MAX_CONDITIONS}')
    out = []
    for item in raw:
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(f'condition labels must be strings, got '
                             f'{_cv_repr(item)}')
        label = str(item).strip()
        if not label:
            raise ValueError('condition labels must be non-empty')
        if label in out:
            raise ValueError(f'duplicate condition label {label!r}')
        out.append(label)
    return out


def _cv_normalise_effects(conditions, effects):
    if isinstance(effects, dict):
        by_label = {}
        for key, value in effects.items():
            label = str(key).strip()
            if not label:
                raise ValueError('effects keys must be non-empty')
            if label in by_label:
                raise ValueError(f'duplicate effects key {label!r}')
            by_label[label] = value
        if set(by_label) != set(conditions):
            missing = sorted(set(conditions) - set(by_label))
            extra = sorted(set(by_label) - set(conditions))
            raise ValueError('effects keys must match conditions exactly; '
                             f'missing={missing}, extra={extra}')
        rows = [by_label[label] for label in conditions]
    elif isinstance(effects, (list, tuple)):
        if len(effects) != len(conditions):
            raise ValueError(f'effects has {len(effects)} rows but '
                             f'{len(conditions)} conditions were declared')
        rows = list(effects)
    else:
        raise ValueError('effects must be a mapping condition->values or an '
                         f'ordered sequence, got {type(effects).__name__}')
    out = []
    for label, row in zip(conditions, rows):
        if isinstance(row, (str, bytes, bytearray, dict)):
            raise ValueError(f'effects[{label!r}] must be an iterable of '
                             'numbers, not a string/mapping')
        try:
            row_len = len(row)
        except (TypeError, OverflowError):
            row_len = None
        if row_len is not None and row_len > CURVE_INFERENCE_MAX_SEEDS:
            raise ValueError(
                f'effects[{label!r}] has {row_len} values; the supported '
                f'maximum is {CURVE_INFERENCE_MAX_SEEDS} seeds')
        try:
            values = list(row)
        except TypeError:
            raise ValueError(f'effects[{label!r}] is not an iterable') from None
        # Check the cheap length cap before validating every element.  A
        # malformed multi-million-entry row must fail immediately instead of
        # consuming memory/time proportional to its size.
        if len(values) > CURVE_INFERENCE_MAX_SEEDS:
            raise ValueError(
                f'effects[{label!r}] has {len(values)} values; the supported '
                f'maximum is {CURVE_INFERENCE_MAX_SEEDS} seeds')
        clean = []
        for index, value in enumerate(values):
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f'effects[{label!r}][{index}] is a boolean, '
                                 'not a measurement')
            number = _finite_scalar(value)  # noqa: F821 (part_00)
            clean.append(None if number is None else float(number))
        out.append(clean)
    return out


def _cv_normalise_inputs(conditions, effects, seeds, *, alpha, min_n,
                         margin, flatness_margin, max_n):
    conditions = _cv_normalise_conditions(conditions)
    rows = _cv_normalise_effects(conditions, effects)
    if not rows:
        raise ValueError('effects is empty')
    width = len(rows[0])
    if width > CURVE_INFERENCE_MAX_SEEDS:
        raise ValueError(
            f'effects rows have {width} values; the supported maximum is '
            f'{CURVE_INFERENCE_MAX_SEEDS} seeds')
    for label, row in zip(conditions, rows):
        if len(row) != width:
            raise ValueError(f'effects[{label!r}] has {len(row)} values but '
                             f'{conditions[0]!r} has {width}; every condition '
                             'must be aligned to the same seed list')
    if seeds is None:
        if width == 0:
            raise ValueError('cannot infer seeds from an empty effect row')
        seed_labels = list(range(width))
        declared_seeds = False
    else:
        if isinstance(seeds, (str, bytes, bytearray, dict)):
            raise ValueError('seeds must be an iterable of integers, not a '
                             f'{type(seeds).__name__}')
        try:
            declared_seed_len = len(seeds)
        except (TypeError, OverflowError):
            declared_seed_len = None
        if (declared_seed_len is not None
                and declared_seed_len > CURVE_INFERENCE_MAX_SEEDS):
            raise ValueError(
                f'seeds has {declared_seed_len} entries; the supported '
                f'maximum is {CURVE_INFERENCE_MAX_SEEDS}')
        try:
            raw_seeds = list(seeds)
        except TypeError:
            raise ValueError('seeds must be an iterable of integers') from None
        # Check the declared-set cap before the cheaper alignment message:
        # an over-large seed vector is a scale error regardless of how many
        # effect values a malformed caller supplied.
        if len(raw_seeds) > CURVE_INFERENCE_MAX_SEEDS:
            raise ValueError(
                f'seeds has {len(raw_seeds)} entries; the supported maximum '
                f'is {CURVE_INFERENCE_MAX_SEEDS}')
        if len(raw_seeds) != width:
            raise ValueError(f'seeds has {len(raw_seeds)} entries but the '
                             f'effect rows have {width}')
        seed_labels = []
        seen_seeds = set()
        for index, value in enumerate(raw_seeds):
            parsed = _as_seed_int(value)  # noqa: F821 (part_13)
            if parsed is None:
                raise ValueError(f'seeds[{index}] is not an exact integer: '
                                 f'{_cv_repr(value)}')
            if not (-CURVE_INFERENCE_MAX_REPORTABLE_INT <= parsed
                    <= CURVE_INFERENCE_MAX_REPORTABLE_INT):
                # Beyond this bound str()/json.dumps of the label itself may
                # raise, so the artifact could not be hashed or replayed.
                raise ValueError(
                    f'seeds[{index}] is outside the strict-JSON representable '
                    f'integer range +/-{CURVE_INFERENCE_MAX_REPORTABLE_INT}')
            if parsed in seen_seeds:
                raise ValueError(f'duplicate seed {parsed!r}')
            seen_seeds.add(parsed)
            seed_labels.append(parsed)
        declared_seeds = True
    alpha = _validated_alpha(alpha, context='curve inference alpha')  # noqa: F821
    min_n = _cv_int(min_n, 'min_n', minimum=1,
                    maximum=CURVE_INFERENCE_MAX_SEEDS)
    margin = (None if margin is None
              else _cv_number(margin, 'margin', positive=True))
    flatness_margin = (None if flatness_margin is None
                       else _cv_number(flatness_margin, 'flatness_margin',
                                      positive=True))
    max_n = _cv_int(max_n, 'max_n', minimum=2,
                    maximum=CURVE_INFERENCE_HARD_MAX_N)
    inputs = {
        'conditions': list(conditions),
        'effects': {label: list(row) for label, row in zip(conditions, rows)},
        'seeds': [int(s) for s in seed_labels],
        'alpha': float(alpha),
        'min_n': int(min_n),
        'margin': margin,
        'flatness_margin': flatness_margin,
        'max_n': int(max_n),
    }
    # Keep the declared-seed distinction out of the artifact schema; an
    # inferred 0..n-1 list is a valid explicit list for replay.
    return inputs, declared_seeds


def _cv_complete_matrix(inputs):
    conditions = list(inputs['conditions'])
    seeds = list(inputs['seeds'])
    effects = inputs['effects']
    n_input = len(seeds)
    complete = []
    dropped = []
    for index, seed in enumerate(seeds):
        values = []
        missing = []
        for label in conditions:
            row = effects[label]
            value = row[index] if index < len(row) else None
            if value is None:
                missing.append(label)
            values.append(value)
        if missing:
            dropped.append({'seed': int(seed),
                            'missing_conditions': missing})
            continue
        complete.append((int(seed), values))
    n = len(complete)
    if n == 0:
        return conditions, [], [], dropped, n_input
    D = np.asarray([values for _seed, values in complete], dtype=float).T
    return conditions, [seed for seed, _values in complete], D, dropped, n_input


def _cv_not_estimable(reason, *, conditions=None, seeds=None,
                      dropped=None, n_input=None, warnings=None):
    return {
        'kind': 'curve_simultaneous_inference',
        'schema': CURVE_INFERENCE_SCHEMA,
        'status': 'not_estimable',
        'reason': str(reason),
        'conditions': list(conditions or []),
        'n': 0,
        'n_input': (int(n_input) if n_input is not None else 0),
        'seeds': [int(s) for s in (seeds or [])],
        'dropped_seeds': list(dropped or []),
        'family': None,
        'per_condition': {},
        'flatness': None,
        'uniform': None,
        'warnings': list(warnings or []),
        'role': CURVE_INFERENCE_POLICY['role'],
        'boundary': CURVE_INFERENCE_POLICY['boundary'],
    }


def _cv_maxt(D, conditions, *, alpha, max_n):
    named = {label: [float(x) for x in D[k]]
             for k, label in enumerate(conditions)}
    try:
        return maxt_signflip_adjust(named, alpha=alpha, max_n=max_n)  # noqa: F821 (part_40)
    except ValueError as exc:
        raise ValueError(f'exact sign-flip max-T is not estimable: {exc}') from None


def _cv_exact_enumeration(D, *, alpha, margin, flatness_margin):
    """One exact sign-flip pass for one-sided, range and TOST readouts.

    The max-T p-values/interval come from ``maxt_signflip_adjust``; this pass
    adds the objects that a family of two-sided p-values cannot express: the
    lower/upper one-sided p for each condition, the range null, and the TOST
    p for each condition and each pair.
    """
    K, n = D.shape
    scale = float(np.max(np.abs(D)))
    if not (scale > 0.0):
        raise ValueError('the complete effect matrix is exactly zero')
    # Normalise each condition by its own max-abs, not by one global scale:
    # with rows that differ by many orders of magnitude, a global tolerance
    # swallowed every signed sum of the small row, so its one-sided p became
    # 1.0 and an "all conditions positive" IUT flipped to unsupported.
    row_scale = np.max(np.abs(D), axis=1)
    row_div = np.where(row_scale > 0.0, row_scale, 1.0)
    Dn = D / row_div[:, None]
    obs_sums = Dn.sum(axis=1)
    tol = CURVE_INFERENCE_TOL * np.maximum(1.0, np.abs(obs_sums))
    # The range/flatness test is a location-invariant comparison across
    # conditions, so it keeps one common scale (otherwise each row's
    # normalisation would change the range's meaning).
    Dg = D / scale
    obs_sums_g = Dg.sum(axis=1)
    range_tol = CURVE_INFERENCE_TOL * max(
        1.0, float(np.max(np.abs(obs_sums_g))))
    # When neither the location margin nor the flatness margin is declared,
    # a seed whose effect is exactly zero in EVERY condition is sign-neutral:
    # its sign changes no signed sum and no condition range.  Enumerating that
    # column only duplicates every pattern.  Drop such columns from the sign
    # vector so a zero-padded panel is not pushed past the exact cap.  A
    # non-zero margin makes even a zero effect informative after shifting, so
    # the full vector is kept in that case.
    if margin is None and flatness_margin is None:
        active = np.any(D != 0.0, axis=0)
        n_effective = int(np.count_nonzero(active))
    else:
        active = None
        n_effective = n
    Dn_eff = Dn if active is None else Dn[:, active]
    Dg_eff = Dg if active is None else Dg[:, active]
    total = 1 << n_effective
    lower_counts = np.zeros(K, dtype=np.int64)
    upper_counts = np.zeros(K, dtype=np.int64)
    range_values = np.empty(total, dtype=float)
    low_counts = high_counts = None
    obs_low = obs_high = None
    tol_low = tol_high = None
    Dt = Mt = None
    if margin is not None:
        _margin = float(margin)
        # Scale a condition by max(|effects|, margin) so both the effect and
        # the equivalence shift stay in an O(1) range for the comparison.
        m_scale_row = np.maximum(row_div, _margin)
        Dt = D / m_scale_row[:, None]
        Mt = _margin / m_scale_row
        obs_low = (Dt + Mt[:, None]).sum(axis=1)
        obs_high = (Mt[:, None] - Dt).sum(axis=1)
        tol_low = CURVE_INFERENCE_TOL * np.maximum(1.0, np.abs(obs_low))
        tol_high = CURVE_INFERENCE_TOL * np.maximum(1.0, np.abs(obs_high))
        low_counts = np.zeros(K, dtype=np.int64)
        high_counts = np.zeros(K, dtype=np.int64)
    pairs = []
    pair_low_counts = pair_high_counts = None
    pair_obs_low = pair_obs_high = None
    pair_tol_low = pair_tol_high = None
    pair_effects = pair_M = None
    if flatness_margin is not None and K >= 2:
        pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
        if len(pairs) <= CURVE_INFERENCE_MAX_PAIRS:
            _flat = float(flatness_margin)
            pair_raw = np.asarray([D[i] - D[j] for i, j in pairs])
            pair_scale = np.maximum(
                np.max(np.abs(pair_raw), axis=1), _flat)
            pair_effects = pair_raw / pair_scale[:, None]
            pair_M = _flat / pair_scale
            pair_obs_low = (pair_effects + pair_M[:, None]).sum(axis=1)
            pair_obs_high = (pair_M[:, None] - pair_effects).sum(axis=1)
            pair_tol_low = CURVE_INFERENCE_TOL * np.maximum(
                1.0, np.abs(pair_obs_low))
            pair_tol_high = CURVE_INFERENCE_TOL * np.maximum(
                1.0, np.abs(pair_obs_high))
            pair_low_counts = np.zeros(len(pairs), dtype=np.int64)
            pair_high_counts = np.zeros(len(pairs), dtype=np.int64)
        else:
            pairs = []
    cell_count = max(1, K, len(pairs))
    chunk = max(1, min(total, max(1, (1 << 18) // cell_count)))
    bitpos = np.arange(n_effective, dtype=np.int64)
    for off in range(0, total, chunk):
        size = min(chunk, total - off)
        masks = np.arange(off, off + size, dtype=np.int64)[:, None]
        signs = np.where((masks & (1 << bitpos)) != 0, 1.0, -1.0)
        sums = signs @ Dn_eff.T
        lower_counts += (sums <= obs_sums + tol).sum(axis=0)
        upper_counts += (sums >= obs_sums - tol).sum(axis=0)
        range_sums = signs @ Dg_eff.T
        range_values[off:off + size] = (
            (range_sums.max(axis=1) - range_sums.min(axis=1)) / float(n))
        if margin is not None:
            low = signs @ (Dt + Mt[:, None]).T
            high = signs @ (Mt[:, None] - Dt).T
            low_counts += (low >= obs_low - tol_low).sum(axis=0)
            high_counts += (high >= obs_high - tol_high).sum(axis=0)
        if pair_effects is not None and len(pairs):
            plow = signs @ (pair_effects + pair_M[:, None]).T
            phigh = signs @ (pair_M[:, None] - pair_effects).T
            pair_low_counts += (plow >= pair_obs_low - pair_tol_low).sum(axis=0)
            pair_high_counts += (phigh >= pair_obs_high - pair_tol_high).sum(axis=0)
    p_lower = lower_counts / float(total)
    p_upper = upper_counts / float(total)
    obs_range = float((obs_sums_g.max() - obs_sums_g.min()) / float(n))
    p_range = float(np.count_nonzero(
        range_values >= obs_range - range_tol) / float(total))
    idx = min(total - 1,
              max(0, int(math.ceil((1.0 - alpha) * total)) - 1))
    range_critical = float(np.partition(range_values, idx)[idx])
    # Exact zero effects are sign-invariant: a condition with m non-zero
    # entries has a one-sided floor 1/2**m, not 1/2**n.  The family floor is
    # the smallest attainable p, i.e. the one from the widest-support row
    # (the same convention ``maxt_signflip_adjust`` reports as
    # p_floor_single).  Using 2**n overstated resolution when any condition
    # carried an exact zero.
    _nz_per_row = np.count_nonzero(D, axis=1) if int(D.size) else np.empty(0)
    _max_nz = int(_nz_per_row.max()) if _nz_per_row.size else 0
    _floor_one = (1.0 if _max_nz <= 0
                  else min(1.0, math.ldexp(1.0, -_max_nz)))
    _floor_two = (1.0 if _max_nz <= 0
                  else min(1.0, math.ldexp(2.0, -_max_nz)))
    out = {
        'total': int(total),
        'p_lower': p_lower,
        'p_upper': p_upper,
        'p_floor_one_sided': _floor_one,
        'p_floor_two_sided': _floor_two,
        'range_observed': obs_range * scale,
        'range_critical': range_critical * scale,
        'range_upper_bound': (obs_range + range_critical) * scale,
        'p_range': p_range,
        'range_method': 'exact sign-flip range of condition means',
        'equivalence': None,
        'flatness_equivalence': None,
        'n_pairs': len(pairs),
    }
    if margin is not None:
        p_low = low_counts / float(total)
        p_high = high_counts / float(total)
        p_tost = np.maximum(p_low, p_high)
        out['p_tost'] = p_tost
        out['p_low_margin'] = p_low
        out['p_high_margin'] = p_high
        out['uniform_equivalence_p'] = float(np.max(p_tost))
        out['equivalent_conditions'] = [
            bool(p_tost[k] <= alpha) for k in range(K)]
    if pair_effects is not None and len(pairs):
        pair_low = pair_low_counts / float(total)
        pair_high = pair_high_counts / float(total)
        pair_tost = np.maximum(pair_low, pair_high)
        out['pair_tost'] = pair_tost
        out['flatness_equivalence_p'] = float(np.max(pair_tost))
        out['pairs'] = pairs
    return out


def _cv_normal_enumeration(D, *, alpha, margin, flatness_margin):
    """Normal-approximation fallback for n > the exact-enumeration cap.

    The fallback is used exactly where a future panel can have hundreds of
    seeds, so it must not square an unscaled effect: ``std(1e200)`` overflows
    even though the mean, standard error and p-value are representable.  Work
    on each condition's max-abs scale (the same row-scaling used by the exact
    max-T path) and restore units only for the reported summaries.
    """
    K, n = D.shape
    normal = statistics.NormalDist()  # noqa: F821 (stdlib import added below)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        row_scale = np.max(np.abs(D), axis=1) if K else np.empty(0)
        row_div = np.where(row_scale > 0.0, row_scale, 1.0)
        Dn = D / row_div[:, None]
        mean_n = Dn.mean(axis=1) if K else np.empty(0)
        sd_n = (Dn.std(axis=1, ddof=1) if (K and n > 1)
                else np.zeros(K))
        sem_n = sd_n / math.sqrt(n) if n > 0 else np.zeros(K)
        means = mean_n * row_scale
        sds = sd_n * row_scale
        sems = sem_n * row_scale
    # z = mean/se.  The lower-tail p (H1: mean < 0) is Phi(z); the upper-tail
    # p (H1: mean > 0) is the survival function.  Taking the ratio in
    # normalised units keeps it finite when both the mean and the SE are near
    # float64's limit.
    z = np.zeros(K, dtype=float)
    pos = sem_n > 0.0
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        z[pos] = mean_n[pos] / sem_n[pos]
    p_lower = np.asarray([_cv_norm_cdf(float(v)) for v in z])
    p_upper = np.asarray([_cv_norm_sf(float(v)) for v in z])
    # A measured zero spread with a non-zero mean is degenerate; the exact
    # branch refuses it.  The old fallback synthesised p_lower=1/p_upper=0
    # (or the reverse), which then produced p_two=0 and a "reject" at n
    # above the exact cap — a discrete sign-flip p can never be zero.  Mark
    # the p-values unavailable instead and let the report call the condition
    # unresolved.
    spread_zero = ~pos
    p_lower = np.asarray(p_lower, dtype=float)
    p_upper = np.asarray(p_upper, dtype=float)
    p_lower[spread_zero] = np.nan
    p_upper[spread_zero] = np.nan
    p_two = np.minimum(1.0, 2.0 * np.minimum(p_lower, p_upper))
    p_two[spread_zero] = np.nan
    # Solve the two-sided critical value in the LOWER tail.  ``1 - alpha/2``
    # rounds to exactly 1.0 for alpha below ~2e-16 (a large Bonferroni family
    # reaches that scale), and clamping to 1-1e-15 silently understates the
    # interval.  Quantile symmetry keeps the exact tiny tail representable.
    _tail = float(alpha) / 2.0
    if _tail <= 0.0:
        # alpha/2 underflowed for a subnormal alpha; the one-sided tail is a
        # conservative (wider) interval rather than a fake fixed level.
        _tail = float(alpha)
    zcrit = -float(normal.inv_cdf(_tail))
    with np.errstate(over='ignore', invalid='ignore'):
        ci_n = np.column_stack([mean_n - zcrit * sem_n,
                                mean_n + zcrit * sem_n])
        ci = ci_n * row_scale[:, None]
    # The range is a location-invariant summary; compute it on the globally
    # normalised matrix so opposite-signed effects cannot overflow the mean.
    data_scale = float(np.max(np.abs(D))) if D.size else 0.0
    if data_scale > 0.0:
        with np.errstate(over='ignore', invalid='ignore'):
            mean_g = (D / data_scale).mean(axis=1)
            range_observed = float((mean_g.max() - mean_g.min())
                                   * data_scale)
    else:
        range_observed = 0.0
    out = {
        'total': None,
        'p_lower': p_lower,
        'p_upper': p_upper,
        'p_floor_one_sided': None,
        'p_floor_two_sided': None,
        'range_observed': range_observed,
        'range_critical': None,
        'range_upper_bound': None,
        'p_range': None,
        'range_method': 'normal approximation does not provide a range null',
        'equivalence': None,
        'flatness_equivalence': None,
        'n_pairs': 0,
        'means': means,
        'sds': sds,
        'sems': sems,
        'p_two': p_two,
        'ci': ci,
        'zcrit': zcrit,
    }
    if margin is not None:
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            margin_n = margin / row_div
            # A zero SEM is the same degenerate case as above: the exact
            # branch has no TOST here, so do not turn no measured spread into
            # a zero equivalence p-value.
            p_low = np.asarray([
                (_cv_norm_sf(float((m + gm) / s)) if s > 0.0
                 else float('nan'))
                for m, s, gm in zip(mean_n, sem_n, margin_n)])
            p_high = np.asarray([
                (_cv_norm_sf(float((gm - m) / s)) if s > 0.0
                 else float('nan'))
                for m, s, gm in zip(mean_n, sem_n, margin_n)])
        p_tost = np.maximum(p_low, p_high)
        out['p_tost'] = p_tost
        out['p_low_margin'] = p_low
        out['p_high_margin'] = p_high
        out['uniform_equivalence_p'] = float(np.max(p_tost))
        out['equivalent_conditions'] = [
            bool(p_tost[k] <= alpha) for k in range(K)]
    if flatness_margin is not None and K >= 2:
        pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
        if len(pairs) <= CURVE_INFERENCE_MAX_PAIRS:
            pair_tost = []
            if data_scale > 0.0:
                Dg = D / data_scale
                with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                    margin_n = flatness_margin / data_scale
                for i, j in pairs:
                    diff = Dg[i] - Dg[j]
                    sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
                    se = sd / math.sqrt(n) if sd > 0.0 else 0.0
                    md = float(np.mean(diff))
                    if se > 0.0:
                        pl = _cv_norm_sf((md + margin_n) / se)
                        ph = _cv_norm_sf((margin_n - md) / se)
                    else:
                        # A zero spread in the pairwise difference is the same
                        # degenerate case as a zero-SEM condition: the normal
                        # approximation has no measured variance, so withhold
                        # the equivalence p instead of manufacturing p=0 and
                        # a false "equivalent" claim.
                        pl = ph = float('nan')
                    pair_tost.append(max(pl, ph))
            else:
                # All effects are exactly zero; without a location shift the
                # pairwise difference is degenerate at zero.
                for _i, _j in pairs:
                    pair_tost.append(0.0 if flatness_margin > 0.0 else 1.0)
            out['pair_tost'] = np.asarray(pair_tost)
            out['flatness_equivalence_p'] = float(np.max(pair_tost))
            out['pairs'] = pairs
            out['n_pairs'] = len(pairs)
    return out


def _cv_family_summary(maxt, enum, *, alpha):
    contrasts = maxt.get('contrasts') or {}
    return {
        'm': int(maxt.get('m') or 0),
        'n_patterns': maxt.get('n_patterns'),
        'p_floor_single': _cv_num(maxt.get('p_floor_single')),
        'p_floor_one_sided': _cv_num(enum.get('p_floor_one_sided')),
        'p_floor_two_sided': _cv_num(enum.get('p_floor_two_sided')),
        'critical_value': _cv_num(maxt.get('critical_value')),
        'ci_level_requested': _cv_num(maxt.get('ci_level_requested')),
        'ci_tail_level': _cv_num(maxt.get('ci_tail_level')),
        'ci_method': maxt.get('ci_method'),
        'method': maxt.get('method'),
        'reject': sorted(
            name for name, row in contrasts.items()
            if isinstance(row, dict) and row.get('reject_maxt')),
        'deltas_fp': maxt.get('deltas_fp'),
        'note': maxt.get('note'),
    }


def _cv_num(value):
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _cv_norm_cdf(x):
    """Standard-normal CDF through ``erfc`` (accurate in the lower tail)."""
    return 0.5 * math.erfc(-float(x) / math.sqrt(2.0))


def _cv_norm_sf(x):
    """Standard-normal upper tail ``P(Z > x)`` without ``1 - cdf`` cancellation.

    ``statistics.NormalDist().cdf`` rounds to exactly 1.0 for ``z`` above about
    8.3, so ``1 - cdf(z)`` becomes a fake exact zero.  At the scale this
    fallback exists for (hundreds of seeds), a standard error well below the
    mean easily produces such a z; ``erfc`` keeps the real tail value and
    prevents ``p=0`` / a certainty-valued TOST decision.
    """
    return 0.5 * math.erfc(float(x) / math.sqrt(2.0))


def _cv_t_interval(mean, sem, n, alpha):
    if mean is None or sem is None or n < 2:
        return None
    if not (math.isfinite(mean) and math.isfinite(sem)):
        return None
    if sem == 0.0:
        return [float(mean), float(mean)]
    try:
        crit = _t_critical_alpha(alpha, n - 1)  # noqa: F821 (part_11)
    except Exception:
        return None
    if crit is None or not math.isfinite(float(crit)):
        return None
    lo = float(mean) - float(crit) * float(sem)
    hi = float(mean) + float(crit) * float(sem)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return [lo, hi]


def _cv_direction(mean, lower, upper, *, alpha, zero_tol=1e-12):
    if mean is None:
        return 'unavailable'
    if lower is not None and lower > zero_tol:
        return 'positive'
    if upper is not None and upper < -zero_tol:
        return 'negative'
    if mean > zero_tol:
        return 'positive_unresolved'
    if mean < -zero_tol:
        return 'negative_unresolved'
    return 'zero_unresolved'


def curve_simultaneous_inference(conditions, effects, *, seeds=None,
                                 alpha=0.05, min_n=3, margin=None,
                                 flatness_margin=None, max_n=20):
    """Simultaneous inference for an ordered curve of paired effects.

    ``conditions`` is an ordered list of labels and ``effects`` maps each
    label to one per-seed paired effect (or is an ordered sequence of rows).
    The same seed set is required at every condition; a seed missing any
    condition is dropped from the whole curve and named in ``dropped_seeds``.

    The returned object is strict-JSON safe.  Data-level incompleteness is
    reported as a ``not_estimable`` result; malformed declarations raise
    ``ValueError`` so a typo cannot silently change the curve.
    """
    inputs, _declared = _cv_normalise_inputs(
        conditions, effects, seeds, alpha=alpha, min_n=min_n,
        margin=margin, flatness_margin=flatness_margin, max_n=max_n)
    conds, complete_seeds, D, dropped, n_input = _cv_complete_matrix(inputs)
    if n_input < 2:
        return _cv_not_estimable(
            'fewer than two declared seeds; a paired curve cannot be '
            'estimated', conditions=conds, seeds=complete_seeds,
            dropped=dropped, n_input=n_input)
    if len(complete_seeds) < 2:
        return _cv_not_estimable(
            'fewer than two complete seeds across every condition; the '
            'curve is not estimable', conditions=conds,
            seeds=complete_seeds, dropped=dropped, n_input=n_input)
    n = len(complete_seeds)
    if float(np.max(np.abs(D))) <= 0.0:
        return _cv_not_estimable(
            'every complete paired effect is exactly zero; the sign-flip '
            'distribution is degenerate and the curve carries no '
            'information', conditions=conds, seeds=complete_seeds,
            dropped=dropped, n_input=n_input)
    # The exact cap applies to sign columns that can move a statistic.  With
    # no declared margin, a seed that is exactly zero at every condition is
    # sign-neutral and must not trigger the approximation just because the
    # panel was padded with zero rows.
    if inputs['margin'] is None and inputs['flatness_margin'] is None:
        n_exact = int(np.count_nonzero(np.any(D != 0.0, axis=0)))
    else:
        n_exact = n
    n_exact = max(1, n_exact)
    if n_exact > int(inputs['max_n']):
        enum = _cv_normal_enumeration(
            D, alpha=float(inputs['alpha']), margin=inputs['margin'],
            flatness_margin=inputs['flatness_margin'])
        method = 'normal_approximation_fallback'
        fallback = True
        maxt = None
        family = {
            'm': len(conds),
            'n_patterns': None,
            'p_floor_single': None,
            'p_floor_one_sided': None,
            'p_floor_two_sided': None,
            'critical_value': _cv_num(enum.get('zcrit')),
            'ci_level_requested': 1.0 - float(inputs['alpha']),
            'ci_tail_level': None,
            'ci_method': 'normal-approximation interval',
            'method': 'normal approximation (exact cap exceeded)',
            'reject': [],
            'deltas_fp': None,
            'note': ('n exceeds the exact sign-flip enumeration cap; this '
                     'is an approximate fallback, not an exact result'),
        }
    else:
        try:
            maxt = _cv_maxt(D, conds, alpha=float(inputs['alpha']),
                            max_n=int(inputs['max_n']))
        except ValueError as exc:
            return _cv_not_estimable(str(exc), conditions=conds,
                                     seeds=complete_seeds, dropped=dropped,
                                     n_input=n_input)
        try:
            enum = _cv_exact_enumeration(
                D, alpha=float(inputs['alpha']), margin=inputs['margin'],
                flatness_margin=inputs['flatness_margin'])
        except ValueError as exc:
            return _cv_not_estimable(str(exc), conditions=conds,
                                     seeds=complete_seeds, dropped=dropped,
                                     n_input=n_input)
        method = 'exact_signflip_maxt_iut'
        fallback = False
        family = _cv_family_summary(maxt, enum,
                                    alpha=float(inputs['alpha']))
    alpha = float(inputs['alpha'])
    min_n = int(inputs['min_n'])
    margin = inputs['margin']
    flatness_margin = inputs['flatness_margin']
    # In the fallback branch use the already scale-normalised summaries from
    # ``_cv_normal_enumeration``.  Recomputing ``np.std(D[k])`` here would
    # square an unscaled 1e200 effect and turn a representable SE into None.
    means = (np.asarray(enum['means'], dtype=float)
             if maxt is None else
             np.asarray([maxt['contrasts'][label]['mean'] for label in conds]))
    sds = (np.asarray(enum['sds'], dtype=float)
           if maxt is None else
           np.asarray([maxt['contrasts'][label]['sd'] for label in conds]))
    sems = (np.asarray(enum['sems'], dtype=float)
            if maxt is None else
            np.asarray([maxt['contrasts'][label]['sem'] for label in conds]))
    per_condition = {}
    for k, label in enumerate(conds):
        mean = _cv_num(means[k])
        sd = _cv_num(sds[k])
        sem = _cv_num(sems[k])
        if maxt is None:
            p_two = _cv_num(enum['p_two'][k])
            p_lower = _cv_num(enum['p_lower'][k])
            p_upper = _cv_num(enum['p_upper'][k])
            ci_sim = [_cv_num(x) for x in enum['ci'][k]]
            if any(x is None for x in ci_sim):
                # An endpoint can overflow even when the centre and SEM are
                # representable; a [null, null] pair reads as a reported
                # interval.  Withhold the whole simultaneous interval instead.
                ci_sim = None
            ci_point = _cv_t_interval(mean, sem, n, alpha)
            # The normal fallback has no exact max-T null; use an explicitly
            # approximate Bonferroni family p so the direction flag is not
            # silently based on an unadjusted per-condition p.
            p_adj_single = (None if p_two is None
                            else min(1.0, len(conds) * p_two))
            p_adj_step = p_adj_single
            reject_maxt = bool(p_adj_single is not None
                               and p_adj_single <= alpha)
        else:
            row = maxt['contrasts'][label]
            p_two = _cv_num(row.get('p_raw_signflip'))
            p_lower = _cv_num(enum['p_lower'][k])
            p_upper = _cv_num(enum['p_upper'][k])
            ci_point = _cv_t_interval(mean, sem, n, alpha)
            ci_sim = row.get('ci_simultaneous')
            if not (isinstance(ci_sim, (list, tuple)) and len(ci_sim) == 2):
                ci_sim = None
            else:
                ci_sim = [_cv_num(ci_sim[0]), _cv_num(ci_sim[1])]
                if any(x is None for x in ci_sim):
                    ci_sim = None
            p_adj_single = _cv_num(row.get('p_maxt_single'))
            p_adj_step = _cv_num(row.get('p_maxt'))
            reject_maxt = bool(row.get('reject_maxt'))
        tost = None
        if margin is not None:
            if maxt is None:
                p_tost = _cv_num(enum['p_tost'][k])
            else:
                p_tost = _cv_num(enum['p_tost'][k])
            tost = {
                'margin': float(margin),
                'p_tost': p_tost,
                'equivalent': bool(p_tost is not None and p_tost <= alpha),
            }
        per_condition[label] = {
            'condition': label,
            'n': int(n),
            'mean': mean,
            'sd': sd,
            'sem': sem,
            'ci95_pointwise': ci_point,
            'ci95_simultaneous': ci_sim,
            'p_pointwise_two_sided': p_two,
            'p_pointwise_lower': p_lower,
            'p_pointwise_upper': p_upper,
            'p_adj_single': p_adj_single,
            'p_adj_stepdown': p_adj_step,
            'reject_simultaneous': reject_maxt,
            # The exact family-adjusted p decides "resolved"; the t-style
            # simultaneous interval is a descriptive companion and must not
            # promote a condition whose adjusted p did not clear alpha.
            'direction': (
                ('positive' if mean > 0.0 else 'negative')
                if (reject_maxt and mean is not None) else
                _cv_direction(mean, None, None, alpha=alpha)),
            'ci_direction': _cv_direction(
                mean,
                (ci_sim[0] if ci_sim else None),
                (ci_sim[1] if ci_sim else None),
                alpha=alpha),
            'uniform_lower_supported': bool(
                p_lower is not None and p_lower <= alpha),
            'uniform_upper_supported': bool(
                p_upper is not None and p_upper <= alpha),
            'equivalence': tost,
        }
    if maxt is None:
        family['reject'] = sorted(
            label for label in conds
            if per_condition[label]['reject_simultaneous'])
    # A single range test of H0: all condition means are equal.  The range of
    # the summed vector is unchanged by a common location shift, so sign-flip
    # is the correct null for flatness as well as for the zero-effect null.
    if len(conds) < 2:
        flatness = {
            'status': 'not_applicable',
            'reason': 'a single condition has no flatness contrast',
            'n_pairs': 0,
        }
    else:
        flatness = {
            'status': 'ok',
            'observed_range': enum.get('range_observed'),
            'p_range': enum.get('p_range'),
            'range_critical': enum.get('range_critical'),
            'range_upper_bound': enum.get('range_upper_bound'),
            'range_method': enum.get('range_method'),
            'n_pairs': int(enum.get('n_pairs') or 0),
        }
        if flatness_margin is not None:
            p_eq = _cv_num(enum.get('flatness_equivalence_p'))
            if p_eq is None:
                flatness['equivalence'] = {
                    'margin': float(flatness_margin),
                    'p_tost': None,
                    'supported': False,
                    'reason': ('the pairwise IUT is not available for this '
                               'panel (too many pairs or a degenerate '
                               'zero-spread pairwise difference)'),
                }
            else:
                pairs = enum.get('pairs') or []
                pt = enum.get('pair_tost')
                worst = None
                worst_p = None
                if pairs and pt is not None:
                    idx = int(np.argmax(pt))
                    worst = [int(pairs[idx][0]), int(pairs[idx][1])]
                    worst_p = _cv_num(pt[idx])
                flatness['equivalence'] = {
                    'margin': float(flatness_margin),
                    'p_tost': p_eq,
                    'supported': bool(p_eq <= alpha),
                    'n_pairs': len(pairs),
                    'worst_pair': (None if worst is None else
                                   [conds[worst[0]], conds[worst[1]]]),
                    'worst_pair_p': worst_p,
                    'method': ('IUT over all pairwise exact sign-flip TOST; '
                               'the maximum pairwise p is the intersection '
                               'p-value'),
                }
        else:
            flatness['equivalence'] = None
    # Intersection-union claims.  The maximum component p is the p-value for
    # the all-components claim; no multiplicity correction is applied because
    # the claim is the intersection, not the union, of the component nulls.
    negative_p = _cv_num(np.max(enum['p_lower'])) if len(conds) else None
    positive_p = _cv_num(np.max(enum['p_upper'])) if len(conds) else None
    not_negative = [label for k, label in enumerate(conds)
                    if not (enum['p_lower'][k] <= alpha)]
    not_positive = [label for k, label in enumerate(conds)
                    if not (enum['p_upper'][k] <= alpha)]
    uniform = {
        'negative': {
            'p_iut': negative_p,
            'supported': bool(negative_p is not None and negative_p <= alpha),
            'conditions_not_supported': not_negative,
        },
        'positive': {
            'p_iut': positive_p,
            'supported': bool(positive_p is not None and positive_p <= alpha),
            'conditions_not_supported': not_positive,
        },
        'equivalence': None,
        'all_conditions_reject': bool(
            all(per_condition[label]['reject_simultaneous'] for label in conds)
            if maxt is not None else False),
        'note': ('the IUT p is the maximum one-sided sign-flip p over the '
                 'conditions; no multiplicity correction is applied because '
                 'the claim is an intersection ("all"), not a union'),
    }
    if margin is not None:
        p_eq = _cv_num(enum.get('uniform_equivalence_p'))
        eq_not = [label for k, label in enumerate(conds)
                  if not (enum['p_tost'][k] <= alpha)]
        uniform['equivalence'] = {
            'margin': float(margin),
            'p_tost': p_eq,
            'supported': bool(p_eq is not None and p_eq <= alpha),
            'conditions_not_equivalent': eq_not,
            'method': ('IUT over the per-condition exact sign-flip TOST; '
                       'the maximum component TOST p is the intersection '
                       'p-value'),
        }
    resolved = [label for label in conds
                if per_condition[label]['direction'] in
                ('positive', 'negative')]
    mixed = (any(per_condition[label]['direction'] == 'positive'
                 for label in conds)
             and any(per_condition[label]['direction'] == 'negative'
                     for label in conds))
    warnings = []
    if n < min_n:
        warnings.append(
            f'underpowered: n={n} complete seeds is below min_n={min_n}')
    floor = _cv_num(family.get('p_floor_two_sided')
                    if family.get('p_floor_two_sided') is not None
                    else family.get('p_floor_single'))
    if floor is not None and floor > alpha:
        warnings.append(
            f'resolution-limited: the exact family floor {floor:.6g} is '
            f'above alpha={alpha:.6g}; no effect, however large, can clear '
            'the bar at this seed count')
    if dropped:
        warnings.append(
            f'{len(dropped)} declared seed(s) are incomplete across the '
            'conditions and were dropped from the whole curve: '
            + ', '.join(str(row.get('seed')) for row in dropped[:8])
            + ('' if len(dropped) <= 8 else
               f' (+{len(dropped) - 8} more)'))
    if family.get('critical_value') is None:
        warnings.append(
            'the exact family null cannot reach the requested level at this '
            'n; no finite simultaneous interval is reported')
    if fallback:
        warnings.append(
            'n exceeds the exact sign-flip enumeration cap; this result uses '
            'a normal-approximation fallback and is not exact')
        if any(_cv_num(x) == 0.0 for x in sems):
            warnings.append(
                'at least one condition has zero seed variance; its normal '
                'p-values and equivalence readouts are withheld rather than '
                'reported as a zero-variance rejection')
    if len(conds) < 2:
        warnings.append('a one-condition curve cannot test flatness')
    if all(abs(float(means[k])) <= CURVE_INFERENCE_TOL for k in range(len(conds))):
        warnings.append(
            'all condition means are numerically zero; the IUT/equivalence '
            'readings are degenerate and are not a scientific conclusion')
    status = 'warnings' if warnings else 'ok'
    result = {
        'kind': 'curve_simultaneous_inference',
        'schema': CURVE_INFERENCE_SCHEMA,
        'status': status,
        'method': method,
        'exact': not fallback,
        'alpha': float(alpha),
        'min_n': int(min_n),
        'margin': (None if margin is None else float(margin)),
        'flatness_margin': (None if flatness_margin is None else
                            float(flatness_margin)),
        'n': int(n),
        'n_input': int(n_input),
        'seeds': [int(s) for s in complete_seeds],
        'dropped_seeds': dropped,
        'conditions': list(conds),
        'family': family,
        'per_condition': per_condition,
        'flatness': flatness,
        'uniform': uniform,
        'resolved_conditions': resolved,
        'mixed_direction': bool(mixed),
        'warnings': warnings,
        'role': CURVE_INFERENCE_POLICY['role'],
        'boundary': CURVE_INFERENCE_POLICY['boundary'],
        'policy': dict(CURVE_INFERENCE_POLICY),
    }
    return _cv_jsonable(result)


def build_curve_inference(conditions, effects, *, seeds=None, alpha=0.05,
                          min_n=3, margin=None, flatness_margin=None,
                          max_n=20, source=None, created_at=None):
    inputs, _declared = _cv_normalise_inputs(
        conditions, effects, seeds, alpha=alpha, min_n=min_n,
        margin=margin, flatness_margin=flatness_margin, max_n=max_n)
    result = curve_simultaneous_inference(**inputs)
    artifact = {
        'kind': CURVE_INFERENCE_KIND,
        'schema': CURVE_INFERENCE_SCHEMA,
        'created_at': (time.strftime('%Y-%m-%dT%H:%M:%S') if not created_at
                       else _cv_key_text(created_at, limit=80)),
        'source': (None if source is None
                   else _cv_key_text(source, limit=500)),
        'inputs': inputs,
        'result': result,
    }
    artifact['content_sha256'] = _cv_content_sha(artifact)
    return artifact


def curve_inference_block(conditions, effects, *, seeds=None, alpha=0.05,
                          min_n=3, margin=None, flatness_margin=None,
                          max_n=20):
    """Return the timestamp-free ``inputs``/``result`` block for a report."""
    inputs, _declared = _cv_normalise_inputs(
        conditions, effects, seeds, alpha=alpha, min_n=min_n,
        margin=margin, flatness_margin=flatness_margin, max_n=max_n)
    result = curve_simultaneous_inference(**inputs)
    return {
        'kind': CURVE_INFERENCE_BLOCK_KIND,
        'inputs': inputs,
        'result': result,
    }


def compare_curve_inference_results(stored, replay, *, rtol=1e-9,
                                    atol=1e-12):
    diffs = []
    _ba_same(stored, replay, 'result', rtol=rtol, atol=atol, diffs=diffs)
    return {'match': not diffs, 'differences': diffs}


def _cv_inputs_problems(inputs):
    problems = []
    if not isinstance(inputs, dict):
        return ['inputs must be an object']
    extra = sorted(set(inputs) - set(CURVE_INFERENCE_INPUT_KEYS),
                   key=_cv_key_text)
    missing = [k for k in CURVE_INFERENCE_INPUT_KEYS if k not in inputs]
    if extra or missing:
        problems.append(f'inputs keys differ: missing={missing}, extra=['
                        + ', '.join(_cv_key_text(k) for k in extra) + ']')
        return problems
    try:
        conditions = _cv_normalise_conditions(inputs.get('conditions'))
        _cv_normalise_effects(conditions, inputs.get('effects'))
        _cv_normalise_inputs(
            conditions, inputs.get('effects'), inputs.get('seeds'),
            alpha=inputs.get('alpha'), min_n=inputs.get('min_n'),
            margin=inputs.get('margin'),
            flatness_margin=inputs.get('flatness_margin'),
            max_n=inputs.get('max_n'))
    except (TypeError, ValueError) as exc:
        problems.append(f'inputs are malformed: {exc}')
    return problems


def _cv_shape_problems(inputs, result):
    problems = []
    if not isinstance(result, dict):
        return ['result must be an object']
    if (not isinstance(result.get('kind'), str)
            or result.get('kind') != 'curve_simultaneous_inference'):
        problems.append(
            "result.kind must be 'curve_simultaneous_inference', got "
            f'{_cv_repr(result.get("kind"))}')
    _schema = result.get('schema')
    # ``True == 1`` in Python, so a boolean schema must be rejected explicitly.
    if (isinstance(_schema, (bool, np.bool_))
            or not isinstance(_schema, (int, np.integer))
            or int(_schema) != CURVE_INFERENCE_SCHEMA):
        problems.append(
            f'result.schema must be {CURVE_INFERENCE_SCHEMA}, got '
            f'{_cv_repr(result.get("schema"))}')
    status = result.get('status')
    if (not isinstance(status, str)
            or status not in ('ok', 'warnings', 'not_estimable')):
        problems.append('result.status must be ok/warnings/not_estimable, '
                        f'got {_cv_repr(status)}')
        return problems
    if status == 'not_estimable':
        reason = result.get('reason')
        if not isinstance(reason, str) or not reason.strip():
            problems.append('result.reason must be a non-empty string for a '
                            'not_estimable result')
    if not isinstance(inputs, dict):
        problems.append('inputs must be an object')
        return problems

    # Validate the knobs replay would consume even when replay is skipped by
    # tamper/archive tooling.  Without this, rewriting inputs.alpha (or the
    # result copy) and refreshing the hash would look structurally valid.
    try:
        alpha = _cv_number(inputs.get('alpha'), 'inputs.alpha')
        if not (0.0 < alpha < 1.0):
            raise ValueError(f'inputs.alpha must be in (0,1), got {alpha!r}')
    except (TypeError, ValueError) as exc:
        problems.append(f'inputs.alpha is malformed: {exc}')
        alpha = None
    try:
        min_n = _cv_int(inputs.get('min_n'), 'inputs.min_n', minimum=1,
                        maximum=CURVE_INFERENCE_MAX_SEEDS)
    except (TypeError, ValueError) as exc:
        problems.append(f'inputs.min_n is malformed: {exc}')
        min_n = None
    input_margins = {}
    for key in ('margin', 'flatness_margin'):
        value = inputs.get(key)
        if value is None:
            input_margins[key] = None
            continue
        try:
            input_margins[key] = _cv_number(value, f'inputs.{key}',
                                            positive=True)
        except (TypeError, ValueError) as exc:
            problems.append(f'inputs.{key} is malformed: {exc}')
            input_margins[key] = None

    conditions = inputs.get('conditions')
    got = result.get('conditions')
    if not isinstance(got, list) or got != conditions:
        problems.append('result.conditions does not match the declared order')

    if status != 'not_estimable':
        try:
            result_alpha = _cv_number(result.get('alpha'), 'result.alpha')
        except (TypeError, ValueError) as exc:
            problems.append(f'result.alpha is malformed: {exc}')
        else:
            if (alpha is not None
                    and not math.isclose(result_alpha, alpha,
                                         rel_tol=1e-12, abs_tol=1e-15)):
                problems.append('result.alpha does not match inputs.alpha')
        try:
            result_min_n = _cv_int(result.get('min_n'), 'result.min_n',
                                   minimum=1,
                                   maximum=CURVE_INFERENCE_MAX_SEEDS)
        except (TypeError, ValueError) as exc:
            problems.append(f'result.min_n is malformed: {exc}')
        else:
            if min_n is not None and result_min_n != min_n:
                problems.append('result.min_n does not match inputs.min_n')
        for key in ('margin', 'flatness_margin'):
            value = result.get(key)
            wanted = input_margins.get(key)
            if value is None:
                got_margin = None
            else:
                try:
                    got_margin = _cv_number(value, f'result.{key}',
                                            positive=True)
                except (TypeError, ValueError) as exc:
                    problems.append(f'result.{key} is malformed: {exc}')
                    got_margin = None
            if wanted is None:
                if got_margin is not None:
                    problems.append(
                        f'result.{key} is present but inputs.{key} is null')
            elif (got_margin is None
                  or not math.isclose(got_margin, wanted,
                                      rel_tol=1e-12, abs_tol=1e-15)):
                problems.append(
                    f'result.{key} does not match inputs.{key}')

    per = result.get('per_condition')
    if not isinstance(per, dict):
        problems.append('result.per_condition must be an object')
    elif (status != 'not_estimable' and isinstance(conditions, list)
          and set(per) != set(conditions)):
        problems.append('result.per_condition keys do not match conditions')
    family = result.get('family')
    if status == 'not_estimable':
        # The not-estimable result intentionally carries no family/interval
        # object (there is no estimated curve); only its reason and declared
        # shape are checkable.
        pass
    elif not isinstance(family, dict):
        problems.append('result.family must be an object')
    elif isinstance(conditions, list) and family.get('m') != len(conditions):
        problems.append('result.family.m does not match the condition count')
    try:
        n = _cv_int(result.get('n'), 'result.n', minimum=0,
                    maximum=CURVE_INFERENCE_MAX_SEEDS)
        n_input = _cv_int(result.get('n_input'), 'result.n_input', minimum=0,
                          maximum=CURVE_INFERENCE_MAX_SEEDS)
        if n > n_input:
            problems.append('result.n exceeds result.n_input')
        if isinstance(inputs.get('seeds'), list):
            if n_input != len(inputs['seeds']):
                problems.append('result.n_input does not match inputs.seeds')
    except (TypeError, ValueError) as exc:
        problems.append(f'result counts are malformed: {exc}')
        return problems
    result_seeds = result.get('seeds')
    if not isinstance(result_seeds, list):
        problems.append('result.seeds must be a list')
    else:
        try:
            parsed_seeds = [_cv_int(value, 'result.seeds[]', minimum=None)
                            for value in result_seeds]
        except (TypeError, ValueError) as exc:
            problems.append(f'result.seeds is malformed: {exc}')
        else:
            if status != 'not_estimable' and len(parsed_seeds) != n:
                problems.append('result.n does not match result.seeds length')
            if len(set(parsed_seeds)) != len(parsed_seeds):
                problems.append('result.seeds contains a duplicate seed')
    dropped = result.get('dropped_seeds')
    if not isinstance(dropped, list):
        problems.append('result.dropped_seeds must be a list')
    elif status != 'not_estimable':
        if len(dropped) != n_input - n:
            problems.append(
                'result.dropped_seeds length does not match n_input - n')
    if not isinstance(result.get('warnings'), list):
        problems.append('result.warnings must be a list')
    if status != 'not_estimable':
        if not isinstance(result.get('flatness'), dict):
            problems.append('result.flatness must be an object')
        if not isinstance(result.get('uniform'), dict):
            problems.append('result.uniform must be an object')
        if not isinstance(result.get('resolved_conditions'), list):
            problems.append('result.resolved_conditions must be a list')
    if (status != 'not_estimable' and isinstance(per, dict)
            and isinstance(conditions, list)):
        for label in conditions:
            row = per.get(label)
            if not isinstance(row, dict):
                problems.append(f'result.per_condition[{label!r}] must be '
                                'an object')
                continue
            if row.get('condition') != label:
                problems.append(
                    f'result.per_condition[{label!r}].condition does not '
                    'match its key')
            if row.get('n') != n:
                problems.append(
                    f'result.per_condition[{label!r}].n does not match '
                    'result.n')
            for key in ('p_pointwise_two_sided', 'p_pointwise_lower',
                        'p_pointwise_upper', 'p_adj_single',
                        'p_adj_stepdown'):
                value = row.get(key)
                if value is None:
                    continue
                try:
                    parsed_p = _cv_number(
                        value, f'result.per_condition[{label!r}].{key}')
                except (TypeError, ValueError) as exc:
                    problems.append(str(exc))
                    continue
                if not (0.0 <= parsed_p <= 1.0):
                    problems.append(
                        f'result.per_condition[{label!r}].{key} is not a '
                        'probability')
    return problems


def validate_curve_inference(artifact, *, replay=True):
    out = {
        'kind': 'curve_inference_validation',
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
    extra = sorted(set(artifact) - set(CURVE_INFERENCE_ARTIFACT_KEYS),
                   key=_cv_key_text)
    missing = [k for k in CURVE_INFERENCE_ARTIFACT_KEYS if k not in artifact]
    if extra or missing:
        out['problems'].append(
            f'artifact keys differ from the schema: missing={missing}, '
            'extra=[' + ', '.join(_cv_key_text(k) for k in extra) + ']')
    if (not isinstance(artifact.get('kind'), str)
            or artifact.get('kind') != CURVE_INFERENCE_KIND):
        out['problems'].append(
            f'artifact.kind must be {CURVE_INFERENCE_KIND!r}, got '
            f'{_cv_repr(artifact.get("kind"))}')
    _schema = artifact.get('schema')
    if (isinstance(_schema, (bool, np.bool_))
            or not isinstance(_schema, (int, np.integer))
            or int(_schema) != CURVE_INFERENCE_SCHEMA):
        out['problems'].append(
            f'artifact.schema must be {CURVE_INFERENCE_SCHEMA}, got '
            f'{_cv_repr(artifact.get("schema"))}')
    try:
        forbidden = _cv_forbidden_paths(artifact)
    except Exception as exc:
        out['problems'].append(
            'artifact cannot be inspected for frozen-decision fields: '
            f'{type(exc).__name__}: {_cv_repr(exc)}')
        return out
    if forbidden:
        out['problems'].append('artifact carries frozen-decision key(s): '
                               + ', '.join(forbidden[:8]))
    recorded = artifact.get('content_sha256')
    if not isinstance(recorded, str) or len(recorded) != 64:
        out['problems'].append('artifact.content_sha256 must be a 64-hex string')
    else:
        out['content_sha256'] = recorded
        try:
            actual = _cv_content_sha(artifact)
        except Exception as exc:
            out['problems'].append(
                'content hash is not computable: '
                f'{type(exc).__name__}: {_cv_repr(exc)}')
            actual = None
        if actual is not None and actual != recorded:
            out['problems'].append(
                'artifact.content_sha256 does not match the recorded body')
    inputs = artifact.get('inputs')
    out['problems'].extend(_cv_inputs_problems(inputs))
    result = artifact.get('result')
    if not isinstance(result, dict):
        out['problems'].append('artifact.result must be an object')
    else:
        out['result_status'] = (result.get('status')
                                if isinstance(result.get('status'), str)
                                else _cv_repr(result.get('status')))
    if out['problems']:
        return out
    if not isinstance(artifact.get('created_at'), str):
        out['problems'].append('created_at must be a string')
        return out
    source = artifact.get('source')
    if source is not None and not isinstance(source, str):
        out['problems'].append('source must be a string or null')
        return out
    for label, body in (('inputs', inputs), ('result', result)):
        problem = _cv_strict_json_problem(label, body)
        if problem is not None:
            out['problems'].append(problem)
            return out
    problems = _cv_shape_problems(inputs, result)
    if problems:
        out['problems'].extend(problems)
        return out
    if not replay:
        out['status'] = ('warnings' if result.get('status') == 'warnings'
                         else ('not_estimable'
                               if result.get('status') == 'not_estimable'
                               else 'ok'))
        out['result'] = _cv_jsonable(result)
        return out
    try:
        replayed = curve_simultaneous_inference(**inputs)
    except Exception as exc:
        out['problems'].append(
            f'replay failed: {type(exc).__name__}: {_cv_repr(exc)}')
        return out
    out['replayed'] = True
    cmp = compare_curve_inference_results(result, replayed)
    if not cmp['match']:
        out['problems'].append('replayed result differs from the stored result')
        out['warnings'].extend(cmp['differences'][:8])
        return out
    out['result'] = _cv_jsonable(replayed)
    if replayed.get('status') == 'not_estimable':
        out['status'] = 'not_estimable'
    elif replayed.get('status') == 'warnings':
        out['status'] = 'warnings'
        out['warnings'].extend(list(replayed.get('warnings') or []))
    else:
        out['status'] = 'ok'
    return out


def validate_curve_inference_block(block, *, replay=True):
    """Validate a timestamp-free ``curve_inference_block``."""
    out = {
        'kind': 'curve_inference_block_validation',
        'status': 'fail',
        'result_status': None,
        'replayed': False,
        'problems': [],
        'warnings': [],
        'result': None,
    }
    if not isinstance(block, dict):
        out['problems'].append('block must be a JSON object')
        return out
    if (not isinstance(block.get('kind'), str)
            or block.get('kind') != CURVE_INFERENCE_BLOCK_KIND):
        out['problems'].append(
            f'block.kind must be {CURVE_INFERENCE_BLOCK_KIND!r}, got '
            f'{_cv_repr(block.get("kind"))}')
    extra = sorted(set(block) - {'kind', 'inputs', 'result'},
                   key=_cv_key_text)
    if extra:
        out['problems'].append('block keys differ from the schema: extra=['
                               + ', '.join(_cv_key_text(k) for k in extra)
                               + ']')
    try:
        forbidden = _cv_forbidden_paths(block)
    except Exception as exc:
        out['problems'].append(
            'block cannot be inspected for frozen-decision fields: '
            f'{type(exc).__name__}: {_cv_repr(exc)}')
        return out
    if forbidden:
        out['problems'].append('block carries frozen-decision key(s): '
                               + ', '.join(forbidden[:8]))
    inputs = block.get('inputs')
    out['problems'].extend(_cv_inputs_problems(inputs))
    result = block.get('result')
    if not isinstance(result, dict):
        out['problems'].append('block.result must be an object')
    else:
        out['result_status'] = (result.get('status')
                                if isinstance(result.get('status'), str)
                                else _cv_repr(result.get('status')))
    if out['problems']:
        return out
    for label, body in (('inputs', inputs), ('result', result)):
        problem = _cv_strict_json_problem(label, body)
        if problem is not None:
            out['problems'].append(problem)
            return out
    problems = _cv_shape_problems(inputs, result)
    if problems:
        out['problems'].extend(problems)
        return out
    if not replay:
        out['status'] = ('warnings' if result.get('status') == 'warnings'
                         else ('not_estimable'
                               if result.get('status') == 'not_estimable'
                               else 'ok'))
        out['result'] = _cv_jsonable(result)
        return out
    try:
        replayed = curve_simultaneous_inference(**inputs)
    except Exception as exc:
        out['problems'].append(
            f'replay failed: {type(exc).__name__}: {_cv_repr(exc)}')
        return out
    out['replayed'] = True
    cmp = compare_curve_inference_results(result, replayed)
    if not cmp['match']:
        out['problems'].append('replayed result differs from the stored result')
        out['warnings'].extend(cmp['differences'][:8])
        return out
    out['result'] = _cv_jsonable(replayed)
    if replayed.get('status') == 'not_estimable':
        out['status'] = 'not_estimable'
    elif replayed.get('status') == 'warnings':
        out['status'] = 'warnings'
        out['warnings'].extend(list(replayed.get('warnings') or []))
    else:
        out['status'] = 'ok'
    return out


def write_curve_inference(path_or_dir, artifact, *, overwrite=False,
                          name='curve_inference.json'):
    """Write a validated artifact; same content is a no-op, different conflicts."""
    return _write_stamped_artifact(
        path_or_dir, artifact, validator=validate_curve_inference,
        name=name, overwrite=overwrite)


def curve_inference_markdown(result, *, title='Curve-level simultaneous inference'):
    title = _cv_text(title, 'title', limit=240)
    if not isinstance(result, dict):
        return f'**{title}**: not estimable (not an object).'
    if result.get('status') == 'not_estimable':
        return (f'**{title}**: not estimable — '
                f'{_cv_repr(result.get("reason") or "no reason", limit=500)}.')
    conditions = result.get('conditions')
    conditions = conditions if isinstance(conditions, list) else []
    per = result.get('per_condition')
    per = per if isinstance(per, dict) else {}
    lines = [
        f'**{title}** (n={result.get("n")}; m={len(conditions)}; '
        f'method=`{result.get("method")}`)', '',
        '| condition | n | mean | pointwise 95% | adjusted p | simultaneous 95% | direction |',
        '|---|---|---|---|---|---|---|',
    ]

    def _fmt(value, digits=4):
        if value is None:
            return '—'
        try:
            return f'{float(value):.{digits}g}'
        except (TypeError, ValueError, OverflowError):
            return _cv_repr(value)
    for label in conditions:
        row = per.get(label)
        if not isinstance(row, dict):
            continue
        pt = row.get('ci95_pointwise') or [None, None]
        sim = row.get('ci95_simultaneous') or [None, None]
        lines.append(
            f'| `{_cv_key_text(label)}` | {row.get("n")} | '
            f'{_fmt(row.get("mean"))} | [{_fmt(pt[0])}, {_fmt(pt[1])}] | '
            f'{_fmt(row.get("p_adj_single"))} | '
            f'[{_fmt(sim[0])}, {_fmt(sim[1])}] | '
            f'{_cv_key_text(row.get("direction") or "—")} |')
    uniform = result.get('uniform') if isinstance(result.get('uniform'), dict) else {}
    neg = uniform.get('negative') if isinstance(uniform.get('negative'), dict) else {}
    pos = uniform.get('positive') if isinstance(uniform.get('positive'), dict) else {}
    eq = uniform.get('equivalence') if isinstance(uniform.get('equivalence'), dict) else {}
    lines.extend([
        '',
        f'- all conditions improve: p_IUT={_fmt(neg.get("p_iut"))}; '
        f'supported={neg.get("supported")}',
        f'- all conditions worsen: p_IUT={_fmt(pos.get("p_iut"))}; '
        f'supported={pos.get("supported")}',
    ])
    if eq:
        lines.append(
            f'- all conditions within ±{_fmt(eq.get("margin"))}: '
            f'p_IUT={_fmt(eq.get("p_tost"))}; supported={eq.get("supported")}')
    flat = result.get('flatness') if isinstance(result.get('flatness'), dict) else {}
    if flat.get('status') == 'ok':
        lines.append(
            f'- flatness (H0: all condition means equal): '
            f'range={_fmt(flat.get("observed_range"))}; '
            f'p={_fmt(flat.get("p_range"))}')
        feq = flat.get('equivalence')
        if isinstance(feq, dict):
            lines.append(
                f'- flatness equivalence within ±{_fmt(feq.get("margin"))}: '
                f'p_IUT={_fmt(feq.get("p_tost"))}; '
                f'supported={feq.get("supported")}')
    warnings = result.get('warnings') or []
    if warnings:
        lines.extend(['', '> ' + '; '.join(_cv_repr(w, limit=300)
                                           for w in warnings[:6])])
    return '\n'.join(lines)
