# ---- ablation effects are not budget-invariant: paired budget ladder + exact interaction ----
# A leave-one-out / on-off ablation is almost always summarised at ONE training
# budget.  "Removing component X costs 0.8 PPL" is then read as a property of X.
# It is not: it is a property of X *at that stopping point*.  A component that
# helps a 1.5k-step model can be neutral at 8k, or an effect can grow with
# compute (the component is a capacity substitute rather than a mechanism).
# Reporting only the final budget therefore hides a budget x ablation
# interaction and licenses an extrapolation the experiment never tested.
#
# This fragment adds the paired budget-ladder readout:
#
#   * declare ordered budgets B_1 < ... < B_K and, for each component, a
#     with/without arm pair.  Every declared (arm, seed, budget) cell must be
#     present; a missing cell is not interpolated and an extra seed is not
#     used to fill it.
#   * per budget and seed, the ablation benefit is
#         d_{b,s} = log(PPL_without[b,s]) - log(PPL_with[b,s])
#     (positive = removing the component hurts).  The raw-PPL difference is a
#     declared sensitivity scale, not a replacement.
#   * the budget interaction is the per-seed OLS slope of d on log(budget),
#     g_s.  Under the same seed-exchangeability/symmetry assumption as the
#     repository's exact sign-flip test, sign(g_s) is a valid randomisation
#     statistic.  For K=2 this is exactly the endpoint difference in
#     difference-in-differences (up to the positive log-budget constant).
#   * the component slopes form ONE declared exploratory family and are
#     Holm-adjusted.  Per-budget p-values are descriptive and are not family
#     members: they answer "where does the effect appear", not "does the
#     effect change".
#   * a declared interaction SESOI (margin) adds an exact sign-flip TOST on
#     the slopes.  Without a margin, "no interaction detected" is explicitly
#     NOT "stable": absence of evidence is not equivalence.
#   * `single_budget_extrapolation_allowed` requires equivalence AND no sign
#     reversal.  It is never `confirmed` and never enters a frozen family.
#
# SCOPE: exploratory secondary readout over existing per-budget records.  No
# training, no split read, no PPL recomputation, no frozen p/Holm/confirmed
# change, no CODE_SEMANTICS bump.

import json

BUDGET_ABLATION_KIND = 'budget_ablation'
BUDGET_ABLATION_SCHEMA = 1
BUDGET_ABLATION_MAX_BUDGETS = 64
BUDGET_ABLATION_MAX_COMPONENTS = 64
BUDGET_ABLATION_MAX_SEEDS = 4096
BUDGET_ABLATION_MAX_REPORTABLE_INT = 10 ** 4000

BUDGET_ABLATION_POLICY = {
    'version': 'budget-ablation-1',
    'primary_scale': 'log_ppl',
    'sensitivity_scale': 'ppl',
    'effect_definition': ('log(PPL_without) - log(PPL_with); positive means '
                          'removing the component hurts (the component helps)'),
    'interaction_definition': ('per-seed OLS slope of the paired log-PPL '
                               'effect on log(budget); for two budgets this is '
                               'the difference-in-differences endpoint change '
                               'up to the positive log-budget scale'),
    'family': ('one exploratory Holm family over all declared component '
               'slopes; per-budget p-values are descriptive, not family '
               'members'),
    'extrapolation_rule': ('a single-budget conclusion is extrapolable only '
                           'when a declared interaction SESOI is rejected by '
                           'the exact sign-flip TOST AND no budget shows the '
                           'opposite effect sign.  No margin -> no stability '
                           'claim.'),
    'sign_reversal_rule': ('a budget contributes a sign only when '
                           '|mean| > max(zero_tol, 1.96*sem); a budget whose '
                           'effect is indistinguishable from zero is not '
                           'evidence of a reversal'),
    'role': ('exploratory budget x ablation interaction readout; it creates '
             'no confirmatory p-value, enters no frozen Holm family and does '
             'not change p/Holm/verified/confirmed or any PPL.'),
    'boundary': ('the slope is a seed-level randomisation statistic, not a '
                 'within-seed measurement-error model.  Budgets are treated as '
                 'declared ordered points; unequal optimisation/token budgets '
                 'remain a design problem.  A missing cell is never imputed '
                 'and a non-estimable component still occupies its family slot.'),
}

BUDGET_ABLATION_ARTIFACT_KEYS = (
    'kind', 'schema', 'created_at', 'source', 'inputs', 'result',
    'content_sha256',
)
BUDGET_ABLATION_INPUT_KEYS = (
    'design', 'cells', 'alpha', 'min_n', 'margin', 'conf', 'zero_tol',
)
BUDGET_ABLATION_FORBIDDEN_KEYS = frozenset({
    'confirmed', 'confirmed_effective', 'p_holm', 'verified', 'verdict',
    'decision', 'frozen_decision', 'final_decision', 'holm', 'confirmed_effect',
})


def _ba_repr(value, limit=160):
    """Bounded, exception-safe repr for diagnostics."""
    try:
        text = repr(value)
    except Exception:
        return f'<unrepresentable {type(value).__name__}>'
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _ba_key_text(value, limit=120):
    """Bounded text for an untrusted mapping key/path component."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit - 3] + '...'
    try:
        text = str(value)
    except Exception:
        return _ba_repr(value, limit=limit)
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _ba_key_json(value):
    """JSON object key without an int-to-decimal conversion limit.

    Ordinary keys keep their historical ``str(key)`` spelling.  A huge integer
    key falls back to exact hexadecimal (which CPython can always format), so
    hashing a tampered artifact reports a mismatch instead of raising.
    """
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:
        if isinstance(value, int) and not isinstance(value, bool):
            return hex(value)
        return _ba_key_text(value)


def _ba_text_or_repr(value, *, limit=240):
    """Display text for untrusted metadata (a huge int must not abort)."""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = str(value)
        except Exception:
            text = _ba_repr(value, limit=limit)
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _ba_jsonable_safe(value):
    """Rewrite untrusted keys/int before :func:`_json_clean`.

    An artifact is untrusted input: a hand-edited mapping key or integer can be
    beyond CPython 3.12's integer-to-decimal conversion limit.  Hashing and
    reporting such a body must not raise before the validator can name it.
    Ordinary values keep their historical JSON spelling.
    """
    if isinstance(value, dict):
        return {_ba_key_json(key): _ba_jsonable_safe(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ba_jsonable_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        if value.dtype.kind in 'Mm':
            return value
        return _ba_jsonable_safe(value.tolist())
    if isinstance(value, np.generic):
        if isinstance(value, (np.datetime64, np.timedelta64)):
            return value
        if isinstance(value, np.integer):
            return _ba_jsonable_safe(int(value))
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        try:
            str(value)
        except (ValueError, OverflowError):
            return {'__unprintable_int_hex__': hex(value)}
    return value


def _ba_strict_json_problem(label, body):
    """A fail-closed description when ``body`` is not strict JSON, or None."""
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
                f'{type(exc).__name__}: {_ba_repr(exc)}')
    return None


def _ba_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string, got '
                         f'{_ba_repr(value)}')
    return value.strip()


def _ba_text_or_none(value, name, *, limit=240):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f'{name} must be a string or None, got '
                         f'{_ba_repr(value)}')
    text = value.strip()
    return text[:limit] if text else None


def _ba_int(value, name, *, minimum=None, maximum=None):
    out = _as_seed_int(value)  # noqa: F821 (part_13)
    if out is None:
        raise ValueError(f'{name} must be an exact integer, got '
                         f'{_ba_repr(value)}')
    if minimum is not None and out < minimum:
        raise ValueError(f'{name} must be >= {minimum}, got {_ba_repr(out)}')
    if maximum is not None and out > maximum:
        raise ValueError(f'{name} must be <= {maximum}, got {_ba_repr(out)}')
    return out


def _ba_finite(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a finite real number, got '
                         f'{_ba_repr(value)}')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must be a finite real number, got '
                         f'{_ba_repr(value)}') from None
    if not math.isfinite(number):
        raise ValueError(f'{name} must be finite, got {_ba_repr(value)}')
    if positive and number <= 0.0:
        raise ValueError(f'{name} must be > 0, got {_ba_repr(value)}')
    if nonnegative and number < 0.0:
        raise ValueError(f'{name} must be >= 0, got {_ba_repr(value)}')
    return number


def _ba_num(value):
    """Finite float or None (booleans/strings/NaN/inf are not measurements)."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _ba_jsonable(value):
    return _json_clean(_ba_jsonable_safe(value))  # noqa: F821 (part_08)


def _ba_content_sha(artifact):
    body = {k: v for k, v in artifact.items()
            if k not in ('created_at', 'content_sha256')}
    return _sha256_json(_ba_jsonable(body))  # noqa: F821 (part_00)


def _ba_forbidden_paths(obj, path=''):
    """Paths of frozen-decision keys anywhere in an artifact."""
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = _ba_key_text(key)
            here = f'{path}.{name}' if path else name
            if name in BUDGET_ABLATION_FORBIDDEN_KEYS:
                found.append(here)
            found.extend(_ba_forbidden_paths(value, here))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            found.extend(_ba_forbidden_paths(value, f'{path}[{index}]'))
    return found


def _ba_normalise_design(design):
    """Validate and canonicalise the declared budget ladder and components."""
    if not isinstance(design, dict):
        raise ValueError('design must be a JSON object')
    allowed = {'budget_key', 'budgets', 'components', 'seeds', 'label', 'note'}
    extra = sorted(set(design) - allowed, key=_ba_key_text)
    if extra:
        raise ValueError('design has undeclared key(s): '
                         + ', '.join(_ba_key_text(k) for k in extra))
    budget_key = _ba_text(design.get('budget_key', 'steps'), 'budget_key')
    budgets_raw = design.get('budgets')
    if not isinstance(budgets_raw, (list, tuple)):
        raise ValueError('design.budgets must be a list of at least two '
                         'strictly increasing positive integers')
    # Check the cheap length cap before validating every element, so a
    # malformed multi-million-entry ladder fails immediately instead of
    # consuming time proportional to its size.
    if len(budgets_raw) > BUDGET_ABLATION_MAX_BUDGETS:
        raise ValueError(f'design.budgets is capped at '
                         f'{BUDGET_ABLATION_MAX_BUDGETS} points, got '
                         f'{len(budgets_raw)}')
    budgets = []
    for index, raw in enumerate(budgets_raw):
        budgets.append(_ba_int(raw, f'design.budgets[{index}]', minimum=1,
                               maximum=BUDGET_ABLATION_MAX_REPORTABLE_INT))
    if len(budgets) < 2:
        raise ValueError('design.budgets needs at least two budgets, got '
                         f'{_ba_repr(budgets)}')
    if any(b <= a for a, b in zip(budgets, budgets[1:])):
        raise ValueError('design.budgets must be strictly increasing, got '
                         f'{_ba_repr(budgets)}')
    components_raw = design.get('components')
    if not isinstance(components_raw, (list, tuple)):
        raise ValueError('design.components must be a list of component rows')
    if not (1 <= len(components_raw) <= BUDGET_ABLATION_MAX_COMPONENTS):
        raise ValueError('design.components needs 1..'
                         f'{BUDGET_ABLATION_MAX_COMPONENTS} rows, got '
                         f'{len(components_raw)}')
    components = []
    names = set()
    for index, raw in enumerate(components_raw):
        if not isinstance(raw, dict):
            raise ValueError(f'design.components[{index}] must be an object')
        allowed_c = {'name', 'with_arm', 'without_arm', 'note'}
        extra_c = sorted(set(raw) - allowed_c, key=_ba_key_text)
        if extra_c:
            raise ValueError(
                f'design.components[{index}] has undeclared key(s): '
                + ', '.join(_ba_key_text(k) for k in extra_c))
        name = _ba_text(raw.get('name'), f'design.components[{index}].name')
        if name in names:
            raise ValueError(f'design.components has duplicate name {name!r}')
        names.add(name)
        with_arm = _ba_text(raw.get('with_arm'),
                            f'design.components[{index}].with_arm')
        without_arm = _ba_text(raw.get('without_arm'),
                               f'design.components[{index}].without_arm')
        if with_arm == without_arm:
            raise ValueError(
                f'design.components[{index}] uses the same arm for with and '
                'without; an ablation needs two distinct arms')
        components.append({
            'name': name,
            'with_arm': with_arm,
            'without_arm': without_arm,
            'note': _ba_text_or_none(raw.get('note'),
                                     f'design.components[{index}].note'),
        })
    seeds = design.get('seeds')
    if seeds is not None:
        if not isinstance(seeds, (list, tuple)):
            raise ValueError('design.seeds must be a list of exact integers or None')
        if not (1 <= len(seeds) <= BUDGET_ABLATION_MAX_SEEDS):
            raise ValueError('design.seeds needs 1..'
                             f'{BUDGET_ABLATION_MAX_SEEDS} entries, got '
                             f'{len(seeds)}')
        normalised_seeds = []
        seen = set()
        for index, raw in enumerate(seeds):
            seed = _ba_int(raw, f'design.seeds[{index}]',
                           minimum=-BUDGET_ABLATION_MAX_REPORTABLE_INT,
                           maximum=BUDGET_ABLATION_MAX_REPORTABLE_INT)
            if seed in seen:
                raise ValueError(f'design.seeds has duplicate seed {seed}')
            seen.add(seed)
            normalised_seeds.append(seed)
        seeds = sorted(normalised_seeds)
    return {
        'label': _ba_text_or_none(design.get('label'), 'design.label'),
        'note': _ba_text_or_none(design.get('note'), 'design.note'),
        'budget_key': budget_key,
        'budgets': budgets,
        'components': components,
        'seeds': seeds,
    }


def _ba_normalise_cells(cells):
    """Canonicalise the (arm, seed, budget, ppl) cells; structural errors raise.

    A non-positive PPL is kept so the analysis can report a named
    ``not_estimable`` result instead of raising in the middle of an artifact
    build; a non-finite or boolean value is not a measurement and is refused.
    """
    if isinstance(cells, dict):
        cells = cells.get('cells')
    if not isinstance(cells, (list, tuple)):
        raise ValueError('cells must be a list of {arm, seed, budget, ppl} rows')
    rows = []
    for index, raw in enumerate(cells):
        if not isinstance(raw, dict):
            raise ValueError(f'cells[{index}] must be an object')
        arm = _ba_text(raw.get('arm'), f'cells[{index}].arm')
        seed = _ba_int(raw.get('seed'), f'cells[{index}].seed',
                       minimum=-BUDGET_ABLATION_MAX_REPORTABLE_INT,
                       maximum=BUDGET_ABLATION_MAX_REPORTABLE_INT)
        budget = _ba_int(raw.get('budget'), f'cells[{index}].budget',
                         minimum=1,
                         maximum=BUDGET_ABLATION_MAX_REPORTABLE_INT)
        if isinstance(raw.get('ppl'), (bool, np.bool_)):
            raise ValueError(f'cells[{index}].ppl must be a finite number, not '
                             f'a boolean')
        try:
            ppl = float(raw.get('ppl'))
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'cells[{index}].ppl must be a finite number, got '
                             f'{_ba_repr(raw.get("ppl"))}') from None
        if not math.isfinite(ppl):
            raise ValueError(f'cells[{index}].ppl must be finite, got '
                             f'{_ba_repr(raw.get("ppl"))}')
        rows.append({'arm': arm, 'seed': seed, 'budget': budget, 'ppl': ppl})
    rows.sort(key=lambda r: (r['arm'], r['budget'], r['seed']))
    return rows


def _ba_cell_table(cells):
    """Return ({(arm, seed, budget): ppl}, [problem, ...])."""
    rows = _ba_normalise_cells(cells)
    table = {}
    problems = []
    for row in rows:
        key = (row['arm'], row['seed'], row['budget'])
        if key in table:
            problems.append(
                f'duplicate cell for arm={row["arm"]!r}, seed={row["seed"]}, '
                f'budget={row["budget"]}')
            continue
        table[key] = row['ppl']
    return table, problems


def _ba_log_budget_ratio(budget, base):
    """``log(budget / base)`` without overflowing an intermediate float.

    ``budget`` may be an exact integer far above float64's range (the design
    caps it at ``10**4000`` for strict-JSON identity).  Dividing two such
    integers or converting one to float raises ``OverflowError`` even though
    the log-ratio is small and representable.  Prefer the overflow-free
    integer quotient, and fall back to the difference of logs.
    """
    budget = int(budget)
    base = int(base)
    if budget <= 0 or base <= 0:
        return None
    try:
        ratio = budget / base
    except (OverflowError, ZeroDivisionError):
        ratio = None
    if ratio is not None and math.isfinite(ratio) and ratio > 0.0:
        try:
            value = math.log(ratio)
        except (ValueError, OverflowError):
            value = None
        if value is not None and math.isfinite(value):
            return value
    try:
        value = math.log(budget) - math.log(base)
    except (ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _ba_slope(values, budgets):
    """OLS slope of values on log(budget / first budget).

    The raw-PPL sensitivity readout can hold finite effects near float64's
    limit.  Summing them over the (up to 64-point) ladder then overflows even
    when the slope itself is representable -- a constant 1e308 effect has a
    true slope of exactly zero.  Ordinary panels keep the historical
    expression bit-for-bit; only an extreme magnitude is centred on a common
    scale, and the result is restored once at the end.
    """
    if len(values) != len(budgets) or len(values) < 2:
        return None
    x = [_ba_log_budget_ratio(b, budgets[0]) for b in budgets]
    if any(value is None for value in x):
        return None
    xbar = sum(x) / len(x)
    denom = sum((xi - xbar) ** 2 for xi in x)
    if not math.isfinite(denom) or denom <= 0.0:
        return None
    try:
        yscale = max(abs(float(y)) for y in values)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(yscale):
        return None
    if yscale <= 1e100:
        ybar = sum(values) / len(values)
        num = sum((xi - xbar) * (yi - ybar) for xi, yi in zip(x, values))
        out = num / denom
        return out if math.isfinite(out) else None
    ys = [float(y) / yscale for y in values]
    ybar = sum(ys) / len(ys)
    num = sum((xi - xbar) * (yi - ybar) for xi, yi in zip(x, ys))
    try:
        out = (num / denom) * yscale
    except (OverflowError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _ba_mean_or_raw(values):
    """Per-budget mean PPL without an overflowing intermediate sum.

    ``sum(values) / len(values)`` overflows to ``inf`` when a scale-up panel
    holds several finite PPLs near the float64 limit, even though their mean
    is representable.  Keep the historical expression for ordinary panels
    (bit-identical output) and use the repository's scaled mean only on the
    overflow path; ``_ba_num`` then withholds a genuinely unrepresentable
    mean instead of leaking a non-strict-JSON ``inf``.
    """
    if not values:
        return None
    raw = sum(values) / len(values)
    if math.isfinite(raw):
        return raw
    return _scaled_mean_or_raw(values)  # noqa: F821 (part_32)


def _ba_stats(values, seeds, *, alpha, min_n):
    """Paired exact-sign-flip summary plus explicit per-seed values."""
    stats = paired_effect_stats(values, alpha=alpha, min_n=min_n, seeds=seeds)
    values = list(values)
    per_seed = []
    for index, seed in enumerate(seeds):
        per_seed.append({'seed': int(seed),
                         'value': _ba_num(values[index]) if index < len(values)
                         else None})
    mean = _ba_num(stats.get('mean'))
    return {
        'n': int(stats.get('n') or 0),
        'n_nonzero': int(stats.get('n_nonzero') or 0),
        'mean': mean,
        'std': _ba_num(stats.get('std')),
        'sem': _ba_num(stats.get('sem')),
        'cohens_dz': _ba_num(stats.get('cohens_dz')),
        'ci95_low': _ba_num(stats.get('ci95_low')),
        'ci95_high': _ba_num(stats.get('ci95_high')),
        'p_exact_signflip': _ba_num(stats.get('p_exact_signflip')),
        'p_floor': _ba_num(stats.get('p_floor')),
        'min_reachable_p': _ba_num(stats.get('min_reachable_p')),
        'underpowered': bool(stats.get('underpowered')),
        'resolution_limited': bool(stats.get('resolution_limited')),
        'within_seed_noise': bool(stats.get('within_seed_noise')),
        'significant': bool(stats.get('significant')),
        'seed_driven': stats.get('seed_driven'),
        'seeds': [int(s) for s in seeds],
        'per_seed': per_seed,
        'direction': _ba_direction(mean),
        'family_member': False,
    }


def _ba_direction(mean, zero_tol=1e-12):
    if mean is None:
        return 'unavailable'
    if mean > zero_tol:
        return 'positive'
    if mean < -zero_tol:
        return 'negative'
    return 'zero'


def _ba_clear_direction(stats, zero_tol):
    """Sign of a budget effect only when it is clearly away from zero.

    A per-budget mean of 1e-9 in the opposite direction is not a sign
    reversal.  The threshold is the larger of the declared zero tolerance and
    the normal 95% half-width; this is a descriptive gate for the reversal
    warning, not a test and not a family member.
    """
    mean = stats.get('mean') if isinstance(stats, dict) else None
    if mean is None:
        return 'unavailable'
    sem = stats.get('sem') if isinstance(stats, dict) else None
    threshold = float(zero_tol)
    if sem is not None:
        try:
            sem = float(sem)
        except (TypeError, ValueError, OverflowError):
            sem = None
        if sem is not None and math.isfinite(sem) and sem > 0.0:
            threshold = max(threshold, 1.96 * sem)
    if mean > threshold:
        return 'positive'
    if mean < -threshold:
        return 'negative'
    return 'zero'


def _ba_not_estimable(reason, *, extra=None):
    out = {
        'kind': BUDGET_ABLATION_KIND,
        'status': 'not_estimable',
        'reason': str(reason),
        'primary_scale': BUDGET_ABLATION_POLICY['primary_scale'],
        'components': {},
        'family': None,
        'warnings': [],
        'note': BUDGET_ABLATION_POLICY['role'],
    }
    if isinstance(extra, dict):
        out.update(extra)
    return out


def _ba_component_not_estimable(component, reason):
    return {
        'name': component['name'],
        'with_arm': component['with_arm'],
        'without_arm': component['without_arm'],
        'status': 'not_estimable',
        'reason': str(reason),
        'n': 0,
        'seeds': [],
        'budgets': [],
        'per_budget': [],
        'log_slope': None,
        'raw_slope': None,
        'endpoint_change': None,
        'interaction_detected': False,
        'budget_sensitive': False,
        'sign_reversal': False,
        'scale_sensitive': False,
        'p_family': None,
        'rank': None,
        'equivalence': None,
        'classification': 'not_estimable',
        'single_budget_extrapolation_allowed': False,
        'warnings': [str(reason)],
        'note': component.get('note'),
    }


def budget_ablation_analysis(design, cells, *, alpha=0.05, min_n=3,
                             margin=None, conf=None, zero_tol=1e-12):
    """Paired budget-ladder interaction readout for declared components.

    The result is JSON-safe (non-finite numbers become ``null``) and has
    ``status`` in ``ok`` / ``warnings`` / ``not_estimable``.  It never raises
    for a data-level problem (missing/duplicate/negative-PPL cells); malformed
    *declarations* raise ``ValueError`` so a typo cannot silently change the
    analysis.
    """
    alpha = _validated_alpha(alpha, context='budget_ablation alpha')  # noqa: F821
    min_n = _ba_int(min_n, 'min_n', minimum=1,
                    maximum=BUDGET_ABLATION_MAX_SEEDS)
    zero_tol = _ba_finite(zero_tol, 'zero_tol', nonnegative=True)
    if conf is None:
        conf = 1.0 - 2.0 * alpha
    conf = _ba_finite(conf, 'conf', positive=True)
    if not (0.0 < conf < 1.0):
        raise ValueError(f'conf must be in (0,1), got {_ba_repr(conf)}')
    margin = (None if margin is None
              else _ba_finite(margin, 'margin', positive=True))
    norm = _ba_normalise_design(design)
    budgets = list(norm['budgets'])
    components = list(norm['components'])
    table, cell_problems = _ba_cell_table(cells)
    if cell_problems:
        return _ba_not_estimable('; '.join(cell_problems[:4]),
                                 extra={'cell_problems': cell_problems})
    declared_seeds = norm['seeds']
    # Index once; inferring the common seed set by rescanning the whole table
    # for every component x budget would be cubic in the declared scale.
    seeds_by_arm_budget = {}
    for (arm, seed, budget) in table:
        seeds_by_arm_budget.setdefault((arm, budget), set()).add(seed)

    rows = {}
    component_problems = {}
    for component in components:
        name = component['name']
        with_arm = component['with_arm']
        without_arm = component['without_arm']
        if declared_seeds is not None:
            seeds = list(declared_seeds)
        else:
            per_budget = []
            for budget in budgets:
                with_seeds = seeds_by_arm_budget.get((with_arm, budget))
                without_seeds = seeds_by_arm_budget.get((without_arm, budget))
                if with_seeds is None or without_seeds is None:
                    per_budget.append(set())
                else:
                    per_budget.append(with_seeds & without_seeds)
            if not per_budget:
                seeds = []
            else:
                common = set(per_budget[0])
                for seed_set in per_budget[1:]:
                    common &= seed_set
                seeds = sorted(common)
        missing = []
        for budget in budgets:
            for seed in seeds:
                for arm in (with_arm, without_arm):
                    if (arm, seed, budget) not in table:
                        missing.append({'arm': arm, 'seed': int(seed),
                                        'budget': int(budget)})
        # A seed that appears for one arm/budget but not all is named; it is
        # not silently dropped unless the caller declared a seed list.
        if missing:
            component_problems[name] = (
                'missing declared cell(s): '
                + ', '.join(
                    f'{row["arm"]}@seed={row["seed"]},budget={row["budget"]}'
                    for row in missing[:8])
                + ('' if len(missing) <= 8 else f' (+{len(missing) - 8} more)'))
            continue
        if len(seeds) < 2:
            component_problems[name] = (
                f'fewer than two complete paired seeds (n={len(seeds)}); a '
                'budget interaction cannot be estimated')
            continue
        invalid = []
        for budget in budgets:
            for seed in seeds:
                for arm in (with_arm, without_arm):
                    ppl = table[(arm, seed, budget)]
                    if not (math.isfinite(ppl) and ppl > 0.0):
                        invalid.append(
                            f'{arm}@seed={seed},budget={budget}: PPL={ppl!r}')
        if invalid:
            component_problems[name] = (
                'non-positive/non-finite PPL cell(s): '
                + ', '.join(invalid[:6])
                + ('' if len(invalid) <= 6 else f' (+{len(invalid) - 6} more)'))
            continue
        # Complete paired seeds: build per-seed effects.
        log_effects = {seed: [] for seed in seeds}
        raw_effects = {seed: [] for seed in seeds}
        with_ppl = {seed: [] for seed in seeds}
        without_ppl = {seed: [] for seed in seeds}
        per_budget_rows = []
        for budget in budgets:
            log_values, raw_values = [], []
            with_values, without_values = [], []
            for seed in seeds:
                w = table[(with_arm, seed, budget)]
                wo = table[(without_arm, seed, budget)]
                log_values.append(math.log(wo) - math.log(w))
                raw_values.append(wo - w)
                with_values.append(w)
                without_values.append(wo)
                log_effects[seed].append(math.log(wo) - math.log(w))
                raw_effects[seed].append(wo - w)
                with_ppl[seed].append(w)
                without_ppl[seed].append(wo)
            log_stats = _ba_stats(log_values, seeds, alpha=alpha, min_n=min_n)
            raw_stats = _ba_stats(raw_values, seeds, alpha=alpha, min_n=min_n)
            per_budget_rows.append({
                'budget': int(budget),
                'log_effect': log_stats,
                'raw_effect': raw_stats,
                'with_mean_ppl': _ba_num(_ba_mean_or_raw(with_values)),
                'without_mean_ppl': _ba_num(_ba_mean_or_raw(without_values)),
                'with_geometric_mean_ppl': math.exp(
                    sum(math.log(x) for x in with_values) / len(with_values)),
                'without_geometric_mean_ppl': math.exp(
                    sum(math.log(x) for x in without_values) / len(without_values)),
            })
        log_slopes = []
        raw_slopes = []
        for seed in seeds:
            log_slopes.append(_ba_slope(log_effects[seed], budgets))
            raw_slopes.append(_ba_slope(raw_effects[seed], budgets))
        if any(value is None for value in log_slopes):
            component_problems[name] = (
                'a per-seed log-budget slope is not representable for this '
                'budget ladder')
            continue
        log_slope_stats = _ba_stats(log_slopes, seeds, alpha=alpha,
                                    min_n=min_n)
        raw_slope_stats = _ba_stats(raw_slopes, seeds, alpha=alpha,
                                    min_n=min_n)
        endpoint_values = [log_effects[seed][-1] - log_effects[seed][0]
                           for seed in seeds]
        endpoint_stats = _ba_stats(endpoint_values, seeds, alpha=alpha,
                                   min_n=min_n)
        endpoint_stats['family_member'] = False
        # A sign reversal is descriptive: it is not a p-value and does not
        # replace the interaction test, but it closes the extrapolation gate.
        signs = set()
        for row in per_budget_rows:
            direction = _ba_clear_direction(row['log_effect'], zero_tol)
            if direction in ('positive', 'negative'):
                signs.add(direction)
        sign_reversal = len(signs) > 1
        # Scale sensitivity: the sign or the significance decision differs
        # between the declared primary (log) and sensitivity (raw) scale.
        log_mean = log_slope_stats['mean']
        raw_mean = raw_slope_stats['mean']
        log_sign = _ba_direction(log_mean, zero_tol)
        raw_sign = _ba_direction(raw_mean, zero_tol)
        sign_agree = (log_sign == raw_sign
                      or 'zero' in (log_sign, raw_sign)
                      or 'unavailable' in (log_sign, raw_sign))
        log_p = log_slope_stats['p_exact_signflip']
        raw_p = raw_slope_stats['p_exact_signflip']
        log_reject = bool(log_p is not None and log_p < alpha
                          and not log_slope_stats['underpowered']
                          and not log_slope_stats['resolution_limited'])
        raw_reject = bool(raw_p is not None and raw_p < alpha
                          and not raw_slope_stats['underpowered']
                          and not raw_slope_stats['resolution_limited'])
        scale_sensitive = bool((not sign_agree) or (log_reject != raw_reject))
        equivalence = None
        if margin is not None:
            try:
                equivalence = exact_signflip_tost(  # noqa: F821 (part_77)
                    log_slopes, margin, conf=conf)
            except Exception as exc:  # a TOST failure is a caveat, not a crash
                equivalence = {
                    'status': 'not_estimable',
                    'reason': f'{type(exc).__name__}: {exc}',
                    'equivalent': False,
                }
        row = {
            'name': name,
            'with_arm': with_arm,
            'without_arm': without_arm,
            'status': 'ok',
            'reason': None,
            'n': len(seeds),
            'seeds': [int(s) for s in seeds],
            'budgets': [int(b) for b in budgets],
            'per_budget': per_budget_rows,
            'log_slope': log_slope_stats,
            'raw_slope': raw_slope_stats,
            'endpoint_change': endpoint_stats,
            'interaction_detected': False,
            'budget_sensitive': False,
            'sign_reversal': sign_reversal,
            'scale_sensitive': scale_sensitive,
            'p_family': None,
            'rank': None,
            'equivalence': equivalence,
            'classification': None,
            'single_budget_extrapolation_allowed': False,
            'warnings': [],
            'note': component.get('note'),
        }
        if log_slope_stats['underpowered']:
            row['warnings'].append(
                f'n={len(seeds)} is below min_n={min_n}; the interaction is '
                'underpowered and cannot support a stability claim')
        if sign_reversal:
            row['warnings'].append(
                'the ablation effect changes sign across the declared budget '
                'ladder; a single-budget ranking is not the component property')
        if scale_sensitive:
            row['warnings'].append(
                'the log-PPL and raw-PPL interaction readouts disagree on '
                'sign or rejection; do not choose the convenient scale')
        rows[name] = row
    if not rows and not component_problems:
        return _ba_not_estimable('no component could be analysed')
    # Holm over ALL declared components; a not-estimable component occupies
    # its slot with p=1 so a failed measurement cannot shrink the family.
    named = []
    for component in components:
        name = component['name']
        row = rows.get(name)
        p = None
        if isinstance(row, dict):
            p = row['log_slope']['p_exact_signflip']
        named.append((name, 1.0 if p is None else float(p)))
    holm = holm_adjust(named, alpha=alpha)  # noqa: F821 (part_09)
    for component in components:
        name = component['name']
        row = rows.get(name)
        if row is None:
            row = _ba_component_not_estimable(
                component, component_problems.get(
                    name, 'component could not be analysed'))
            rows[name] = row
        holm_row = holm.get(name) or {}
        row['p_family'] = _ba_num(holm_row.get('p_holm'))
        row['rank'] = holm_row.get('rank')
        p_family = row['p_family']
        if row['status'] == 'ok':
            p_raw = row['log_slope']['p_exact_signflip']
            resolvable = bool(
                not row['log_slope']['underpowered']
                and (not row['log_slope']['resolution_limited']
                     or row['log_slope']['n_nonzero'] == 0))
            detected = bool(
                p_family is not None and p_family < alpha and resolvable)
            raw_detected = bool(
                p_raw is not None and p_raw < alpha and resolvable)
            row['interaction_detected'] = detected
            row['interaction_detected_raw'] = raw_detected
            row['budget_sensitive'] = detected
            equivalence = row['equivalence']
            equivalent = bool(
                isinstance(equivalence, dict)
                and equivalence.get('equivalent') is True
                and not row['sign_reversal'])
            if detected:
                row['classification'] = 'interaction_detected'
            elif raw_detected:
                row['classification'] = 'interaction_detected_raw_only'
            elif row['sign_reversal']:
                row['classification'] = 'direction_unstable'
            elif equivalent:
                row['classification'] = 'equivalence_supported'
            elif (p_family is not None and resolvable):
                row['classification'] = 'no_interaction_detected'
            else:
                row['classification'] = 'inconclusive'
            if (row['log_slope']['resolution_limited']
                    and row['log_slope']['n_nonzero'] > 0
                    and not equivalent):
                row['warnings'].append(
                    'the exact sign-flip interaction test is '
                    'resolution-limited at this n '
                    f'(p_floor={row["log_slope"]["p_floor"]!r}); a '
                    'non-rejection is a ceiling, not stability')
            if row['log_slope']['n_nonzero'] == 0:
                row['warnings'].append(
                    'every per-seed budget slope is exactly zero; no '
                    'interaction is observed, but a zero observed slope is '
                    'not an equivalence test')
            row['single_budget_extrapolation_allowed'] = bool(
                row['classification'] == 'equivalence_supported'
                and not row['sign_reversal'])
            if detected:
                row['warnings'].append(
                    'a budget x ablation interaction is detected; the '
                    'final-budget effect is not a budget-invariant property')
            elif raw_detected:
                row['warnings'].append(
                    'the nominal interaction p is below alpha but the '
                    'declared Holm family does not reject; report it as a '
                    'nominal signal, not as a family-level interaction')
            if margin is None:
                row['stability_test'] = {
                    'status': 'not_tested',
                    'reason': ('no interaction SESOI (margin) was declared; '
                               'absence of a detected interaction is not '
                               'equivalence/stability'),
                }
            else:
                row['stability_test'] = {
                    'status': ('equivalence_supported' if equivalent
                               else 'not_equivalent_or_resolution_limited'),
                    'margin': float(margin),
                    'conf': float(conf),
                }
    estimable = [row for row in rows.values() if row['status'] == 'ok']
    all_warnings = []
    for component in components:
        all_warnings.extend(rows[component['name']]['warnings'])
    if margin is None:
        all_warnings.append(
            'no interaction SESOI was declared: a non-significant interaction '
            'is reported as no interaction detected, never as stability')
    status = 'ok'
    if not estimable:
        status = 'not_estimable'
    elif any(row['status'] != 'ok' for row in rows.values()) or all_warnings:
        status = 'warnings'
    result = {
        'kind': BUDGET_ABLATION_KIND,
        'status': status,
        'reason': (None if status != 'not_estimable'
                   else 'no declared component has complete paired cells'),
        'primary_scale': BUDGET_ABLATION_POLICY['primary_scale'],
        'sensitivity_scale': BUDGET_ABLATION_POLICY['sensitivity_scale'],
        'budget_key': norm['budget_key'],
        'budgets': [int(b) for b in budgets],
        'label': norm['label'],
        'design_note': norm['note'],
        'seeds_declared': (declared_seeds is not None),
        'seeds': (None if declared_seeds is None
                  else [int(s) for s in declared_seeds]),
        'components': {name: rows[name] for name in
                       (c['name'] for c in components)},
        'family': {
            'name': 'budget_ablation_component_slopes',
            'm': len(components),
            'alpha': float(alpha),
            'method': ('Holm step-down over the declared component budget '
                       'slopes (one exploratory family)'),
            'per_budget_p_values_are_family_members': False,
            'note': ('a not-estimable component occupies its slot with p=1; '
                     'the family is not shrunk by missing measurements'),
        },
        'n_estimable': len(estimable),
        'n_not_estimable': len(components) - len(estimable),
        'warnings': all_warnings,
        'note': BUDGET_ABLATION_POLICY['role'],
    }
    return _ba_jsonable(result)


def _ba_same(a, b, path='result', *, rtol=1e-9, atol=1e-12, diffs=None):
    if diffs is None:
        diffs = []
    if isinstance(a, bool) or isinstance(b, bool):
        if a is not b:
            diffs.append(f'{path}: {_ba_repr(a)} != {_ba_repr(b)}')
        return diffs
    if a is None or b is None:
        if a is not b:
            diffs.append(f'{path}: {_ba_repr(a)} != {_ba_repr(b)}')
        return diffs
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            # Compare exact integers directly: a valid budget above float64's
            # range must not become "uncomparable" during replay.
            if int(a) != int(b):
                diffs.append(f'{path}: {_ba_repr(a)} != {_ba_repr(b)}')
            return diffs
        try:
            if not math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol):
                diffs.append(f'{path}: {_ba_repr(a)} != {_ba_repr(b)}')
        except (TypeError, ValueError, OverflowError):
            diffs.append(f'{path}: uncomparable numbers '
                         f'{_ba_repr(a)}, {_ba_repr(b)}')
        return diffs
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            missing = sorted(set(b) - set(a), key=_ba_key_text)
            extra = sorted(set(a) - set(b), key=_ba_key_text)
            if missing:
                diffs.append(f'{path}: missing keys '
                             + ', '.join(_ba_key_text(k)
                                         for k in missing[:8]))
            if extra:
                diffs.append(f'{path}: extra keys '
                             + ', '.join(_ba_key_text(k)
                                         for k in extra[:8]))
        for key in sorted(set(a) & set(b), key=_ba_key_text):
            _ba_same(a[key], b[key],
                     f'{path}.{_ba_key_text(key)}', rtol=rtol, atol=atol,
                     diffs=diffs)
        return diffs
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diffs.append(f'{path}: length {len(a)} != {len(b)}')
        for index, (left, right) in enumerate(zip(a, b)):
            _ba_same(left, right, f'{path}[{index}]', rtol=rtol, atol=atol,
                     diffs=diffs)
        return diffs
    try:
        same = bool(a == b)
    except Exception:
        same = False
    if not same:
        diffs.append(f'{path}: {_ba_repr(a)} != {_ba_repr(b)}')
    return diffs


def compare_budget_ablation_results(stored, replay, *, rtol=1e-9,
                                    atol=1e-12):
    if not isinstance(stored, dict) or not isinstance(replay, dict):
        return {'match': False,
                'differences': ['both results must be JSON objects']}
    diffs = _ba_same(stored, replay, 'result', rtol=rtol, atol=atol)
    return {'match': not diffs, 'differences': diffs}


def build_budget_ablation(design, cells, *, alpha=0.05, min_n=3,
                          margin=None, conf=None, zero_tol=1e-12,
                          source=None, created_at=None):
    """Build a content-addressed budget-ablation artifact.

    The declared design is validated strictly; a data-level problem produces a
    valid ``not_estimable`` artifact rather than an exception, so a report can
    record why the ladder could not answer the question.
    """
    alpha = _validated_alpha(alpha, context='budget_ablation alpha')  # noqa: F821
    min_n = _ba_int(min_n, 'min_n', minimum=1,
                    maximum=BUDGET_ABLATION_MAX_SEEDS)
    zero_tol = _ba_finite(zero_tol, 'zero_tol', nonnegative=True)
    if conf is None:
        conf = 1.0 - 2.0 * alpha
    conf = _ba_finite(conf, 'conf', positive=True)
    if not (0.0 < conf < 1.0):
        raise ValueError(f'conf must be in (0,1), got {_ba_repr(conf)}')
    margin = (None if margin is None
              else _ba_finite(margin, 'margin', positive=True))
    norm_design = _ba_normalise_design(design)
    norm_cells = _ba_normalise_cells(cells)
    inputs = {
        'design': norm_design,
        'cells': norm_cells,
        'alpha': float(alpha),
        'min_n': int(min_n),
        'margin': margin,
        'conf': float(conf),
        'zero_tol': float(zero_tol),
    }
    result = budget_ablation_analysis(
        norm_design, norm_cells, alpha=alpha, min_n=min_n, margin=margin,
        conf=conf, zero_tol=zero_tol)
    artifact = {
        'kind': BUDGET_ABLATION_KIND,
        'schema': BUDGET_ABLATION_SCHEMA,
        'created_at': (time.strftime('%Y-%m-%dT%H:%M:%S') if not created_at
                       else str(created_at)[:80]),
        'source': (None if source is None else str(source)[:500]),
        'inputs': inputs,
        'result': result,
    }
    artifact['content_sha256'] = _ba_content_sha(artifact)
    return artifact


def _ba_shape_problems(inputs, result):
    problems = []
    if not isinstance(result, dict):
        return ['result must be an object']
    status = result.get('status')
    if status not in ('ok', 'warnings', 'not_estimable'):
        problems.append('result.status must be ok/warnings/not_estimable, got '
                        f'{_ba_repr(status)}')
        return problems
    if status == 'not_estimable':
        return problems
    components = result.get('components')
    if not isinstance(components, dict):
        problems.append('result.components must be an object')
        return problems
    declared = inputs.get('design')
    if not isinstance(declared, dict):
        problems.append('inputs.design must be an object')
        return problems
    declared_components = declared.get('components')
    if not isinstance(declared_components, (list, tuple)):
        problems.append('inputs.design.components must be a list')
        return problems
    names = [row.get('name') for row in declared_components
             if isinstance(row, dict)]
    if any(not isinstance(name, str) for name in names):
        # A tampered declaration can put a list/dict where a component name
        # belongs; hashing it into a set of names would raise TypeError
        # instead of producing the fail-closed diagnostic.
        problems.append('inputs.design.components names must be strings')
    elif set(components) != set(names):
        problems.append('result.components does not match the declared '
                        'component names')
    # Validate the analysis knobs that replay would consume even when the
    # expensive replay is skipped by archive/tamper tooling.
    try:
        alpha = _ba_finite(inputs.get('alpha'), 'inputs.alpha')
        if not (0.0 < alpha < 1.0):
            raise ValueError(f'inputs.alpha must be in (0,1), got {alpha!r}')
        _ba_int(inputs.get('min_n'), 'inputs.min_n', minimum=1)
        _ba_finite(inputs.get('zero_tol'), 'inputs.zero_tol', nonnegative=True)
        conf = _ba_finite(inputs.get('conf'), 'inputs.conf')
        if not (0.0 < conf < 1.0):
            raise ValueError(f'inputs.conf must be in (0,1), got {conf!r}')
        margin = inputs.get('margin')
        if margin is not None:
            _ba_finite(margin, 'inputs.margin', positive=True)
    except (TypeError, ValueError) as exc:
        problems.append(f'inputs analysis parameters are malformed: {exc}')
        alpha = None
    # Cheap dimension checks: a hand-edited result must be rejected even when
    # the expensive replay is skipped.  They also keep ``replay=False`` from
    # copying an unrepresentable/tampered scalar into the verdict.
    declared_budgets = declared.get('budgets')
    result_budgets = result.get('budgets')
    wanted_budgets = None
    if isinstance(declared_budgets, (list, tuple)):
        try:
            wanted_budgets = [_ba_int(value, 'inputs.design.budgets[]',
                                      minimum=1) for value in declared_budgets]
        except (TypeError, ValueError):
            wanted_budgets = None
        if wanted_budgets is not None:
            if not isinstance(result_budgets, (list, tuple)):
                problems.append('result.budgets must be a list')
            else:
                try:
                    got_budgets = [_ba_int(value, 'result.budgets[]',
                                           minimum=1)
                                   for value in result_budgets]
                except (TypeError, ValueError):
                    got_budgets = None
                if got_budgets is not None and got_budgets != wanted_budgets:
                    problems.append(
                        'result.budgets does not match the declared ladder')
    try:
        n_estimable = _ba_int(result.get('n_estimable'),
                              'result.n_estimable', minimum=0)
        n_not_estimable = _ba_int(result.get('n_not_estimable'),
                                  'result.n_not_estimable', minimum=0)
    except (TypeError, ValueError) as exc:
        problems.append(f'result component counts are malformed: {exc}')
        return problems
    if n_estimable + n_not_estimable != len(components):
        problems.append(
            'result.n_estimable + result.n_not_estimable must equal the '
            'number of declared components')
    family = result.get('family')
    if not isinstance(family, dict):
        problems.append('result.family must be an object')
    else:
        try:
            family_m = _ba_int(family.get('m'), 'result.family.m', minimum=0)
        except (TypeError, ValueError) as exc:
            problems.append(f'result.family.m is malformed: {exc}')
        else:
            if family_m != len(declared_components):
                problems.append(
                    'result.family.m does not match the declared component '
                    'count')
        family_alpha = _ba_num(family.get('alpha'))
        if alpha is not None:
            if (family_alpha is None
                    or not math.isclose(family_alpha, alpha,
                                        rel_tol=1e-12, abs_tol=1e-15)):
                problems.append(
                    'result.family.alpha does not match inputs.alpha')
    for name, row in components.items():
        label = _ba_key_text(name)
        if not isinstance(row, dict):
            problems.append(f'result.components[{label!r}] is not an object')
            continue
        if row.get('status') not in ('ok', 'not_estimable'):
            problems.append(
                f'result.components[{label!r}].status is invalid')
            continue
        if row.get('status') == 'not_estimable':
            continue
        for key in ('n', 'seeds', 'budgets', 'per_budget', 'log_slope',
                    'classification', 'single_budget_extrapolation_allowed'):
            if key not in row:
                problems.append(
                    f'result.components[{label!r}] is missing {key!r}')
        try:
            row_n = _ba_int(row.get('n'), f'component {label} n', minimum=0)
        except (TypeError, ValueError) as exc:
            row_n = None
            problems.append(f'result.components[{label!r}].n is malformed: '
                            f'{exc}')
        row_seeds = row.get('seeds')
        if not isinstance(row_seeds, (list, tuple)):
            problems.append(
                f'result.components[{label!r}].seeds must be a list')
        elif row_n is not None and len(row_seeds) != row_n:
            problems.append(
                f'result.components[{label!r}].n does not match its seeds '
                'list length')
        row_budgets = row.get('budgets')
        if not isinstance(row_budgets, (list, tuple)):
            problems.append(
                f'result.components[{label!r}].budgets must be a list')
        elif wanted_budgets is not None:
            try:
                got_row_budgets = [_ba_int(value,
                                           f'component {label} budgets[]',
                                           minimum=1)
                                   for value in row_budgets]
            except (TypeError, ValueError):
                got_row_budgets = None
            if (got_row_budgets is not None
                    and got_row_budgets != wanted_budgets):
                problems.append(
                    f'result.components[{label!r}].budgets does not match '
                    'the declared ladder')
        if not isinstance(row.get('per_budget'), list):
            problems.append(
                f'result.components[{label!r}].per_budget must be a list')
        if not isinstance(row.get('log_slope'), dict):
            problems.append(
                f'result.components[{label!r}].log_slope must be an object')
    return problems


def validate_budget_ablation(artifact, *, replay=True):
    """Validate structure/hash and replay the result from ``inputs``."""
    out = {
        'kind': 'budget_ablation_validation',
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
    extra = sorted(set(artifact) - set(BUDGET_ABLATION_ARTIFACT_KEYS),
                   key=_ba_key_text)
    missing = [k for k in BUDGET_ABLATION_ARTIFACT_KEYS if k not in artifact]
    if extra or missing:
        out['problems'].append(
            f'artifact keys differ from the schema: missing={missing}, '
            'extra=[' + ', '.join(_ba_key_text(k) for k in extra) + ']')
    if (not isinstance(artifact.get('kind'), str)
            or artifact.get('kind') != BUDGET_ABLATION_KIND):
        out['problems'].append(
            f'artifact.kind must be {BUDGET_ABLATION_KIND!r}, got '
            f'{_ba_repr(artifact.get("kind"))}')
    if (not isinstance(artifact.get('schema'), (int, np.integer))
            or isinstance(artifact.get('schema'), (bool, np.bool_))
            or int(artifact.get('schema')) != BUDGET_ABLATION_SCHEMA):
        out['problems'].append(
            f'artifact.schema must be {BUDGET_ABLATION_SCHEMA}, got '
            f'{_ba_repr(artifact.get("schema"))}')
    try:
        forbidden = _ba_forbidden_paths(artifact)
    except Exception as exc:
        out['problems'].append(
            'artifact cannot be inspected for frozen-decision fields: '
            f'{type(exc).__name__}: {_ba_repr(exc)}')
        return out
    if forbidden:
        out['problems'].append(
            'artifact carries frozen-decision key(s): '
            + ', '.join(forbidden[:8]))
    recorded = artifact.get('content_sha256')
    if not isinstance(recorded, str) or len(recorded) != 64:
        out['problems'].append(
            'artifact.content_sha256 must be a 64-hex string')
    else:
        out['content_sha256'] = recorded
        try:
            actual = _ba_content_sha(artifact)
        except Exception as exc:
            out['problems'].append(
                'content hash is not computable: '
                f'{type(exc).__name__}: {_ba_repr(exc)}')
            actual = None
        if actual is not None and actual != recorded:
            out['problems'].append(
                'artifact.content_sha256 does not match the recorded body')
    inputs = artifact.get('inputs')
    if not isinstance(inputs, dict):
        out['problems'].append('artifact.inputs must be an object')
    else:
        extra_i = sorted(set(inputs) - set(BUDGET_ABLATION_INPUT_KEYS),
                         key=_ba_key_text)
        missing_i = [k for k in BUDGET_ABLATION_INPUT_KEYS if k not in inputs]
        if extra_i or missing_i:
            out['problems'].append(
                f'inputs keys differ: missing={missing_i}, extra=['
                + ', '.join(_ba_key_text(k) for k in extra_i) + ']')
    result = artifact.get('result')
    if not isinstance(result, dict):
        out['problems'].append('artifact.result must be an object')
    else:
        result_status = result.get('status')
        out['result_status'] = (result_status if isinstance(result_status, str)
                                else (None if result_status is None
                                      else _ba_repr(result_status)))
    if out['problems']:
        return out
    # Metadata excluded from the content hash is still untrusted input.  Keep
    # the diagnostic record strictly JSON-serialisable and fail closed on a
    # hand-edited unrepresentable/typed field.
    if not isinstance(artifact.get('created_at'), str):
        out['problems'].append(
            'created_at must be a string, got '
            f'{_ba_repr(artifact.get("created_at"))}')
        return out
    source = artifact.get('source')
    if source is not None and not isinstance(source, str):
        out['problems'].append(
            f'source must be a string or null, got {_ba_repr(source)}')
        return out
    # Reject values that a strict JSON writer cannot represent (a huge
    # integer, NaN/Infinity, or an unprintable mapping key) before copying
    # them into the verdict, even when the expensive replay is skipped.
    for label, body in (('inputs', inputs), ('result', result)):
        problem = _ba_strict_json_problem(label, body)
        if problem is not None:
            out['problems'].append(problem)
            return out
    problems = _ba_shape_problems(inputs, result)
    if problems:
        out['problems'].extend(problems)
        return out
    if not replay:
        out['status'] = ('warnings' if result.get('status') == 'warnings'
                         else ('not_estimable'
                               if result.get('status') == 'not_estimable'
                               else 'ok'))
        out['result'] = _ba_jsonable(result)
        return out
    try:
        replayed = budget_ablation_analysis(
            inputs['design'], inputs['cells'],
            alpha=inputs['alpha'], min_n=inputs['min_n'],
            margin=inputs['margin'], conf=inputs['conf'],
            zero_tol=inputs['zero_tol'])
    except Exception as exc:
        out['problems'].append(
            f'replay failed: {type(exc).__name__}: {_ba_repr(exc)}')
        return out
    out['replayed'] = True
    cmp = compare_budget_ablation_results(result, replayed)
    if not cmp['match']:
        out['problems'].append(
            'replayed result differs from the stored result')
        out['warnings'].extend(cmp['differences'][:8])
        return out
    out['result'] = _ba_jsonable(replayed)
    if replayed.get('status') == 'not_estimable':
        out['status'] = 'not_estimable'
    elif replayed.get('status') == 'warnings':
        out['status'] = 'warnings'
        out['warnings'].extend(list(replayed.get('warnings') or []))
    else:
        out['status'] = 'ok'
    return out


def write_budget_ablation(path_or_dir, artifact, *, overwrite=False,
                          name='budget_ablation.json'):
    """Write a validated artifact; same content is a no-op, different conflicts."""
    return _write_stamped_artifact(
        path_or_dir, artifact, validator=validate_budget_ablation,
        name=name, overwrite=overwrite)


def budget_ablation_markdown(result, *, title='Budget x ablation interaction'):
    """Render a compact Markdown table for one result body."""
    title = _ba_text_or_repr(title, limit=240)
    if not isinstance(result, dict):
        return f'**{title}**: not estimable (not an object).'
    if result.get('status') == 'not_estimable':
        reason = _ba_text_or_repr(result.get('reason') or 'no reason',
                                  limit=500)
        return f'**{title}**: not estimable — {reason}.'
    components = result.get('components')
    components = components if isinstance(components, dict) else {}
    family = result.get('family')
    family = family if isinstance(family, dict) else {}
    lines = [
        f'**{title}** (`{_ba_text_or_repr(result.get("budget_key", "budget"))}`; '
        f'n_estimable={_ba_text_or_repr(result.get("n_estimable"))}; '
        f'family m={_ba_text_or_repr(family.get("m"))})', '',
        '| component | n | log-PPL slope / e-fold | 95% CI | p | p_family | '
        'classification | extrapolable |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for name, row in components.items():
        label = _ba_key_text(name)
        if not isinstance(row, dict):
            continue
        if row.get('status') != 'ok':
            lines.append(f'| `{label}` | — | — | — | — | — | not_estimable | '
                         'no |')
            continue
        slope = row.get('log_slope') or {}
        mean = slope.get('mean')
        lo = slope.get('ci95_low')
        hi = slope.get('ci95_high')
        p = slope.get('p_exact_signflip')
        def _fmt(value, digits=4):
            if value is None:
                return '—'
            try:
                return f'{float(value):.{digits}g}'
            except (TypeError, ValueError, OverflowError):
                return _ba_repr(value)
        lines.append(
            f'| `{label}` | {_ba_text_or_repr(row.get("n"))} | {_fmt(mean)} | '
            f'[{_fmt(lo)}, {_fmt(hi)}] | {_fmt(p)} | '
            f'{_fmt(row.get("p_family"))} | '
            f'{_ba_text_or_repr(row.get("classification"))} | '
            f'{"yes" if row.get("single_budget_extrapolation_allowed") else "no"} |')
    warnings = result.get('warnings') or []
    if warnings:
        lines.extend(['', '> ' + '; '.join(
            _ba_text_or_repr(w, limit=300) for w in warnings[:6])])
    return '\n'.join(lines)


def budget_ablation_cells_from_records(records, design):
    """Project per-run records into the cell list this analysis consumes.

    A record is matched by its ``variant`` (or ``arm``) label, its budget field
    (the declared ``budget_key``, or an explicit ``budget``), and its ``seed``.
    Records outside the declared arms/budgets are ignored; a duplicate
    (arm, seed, budget) is refused because the two PPLs would be ambiguous.
    This is a transport adapter, not a second analysis: it does not impute a
    missing cell and does not choose a seed.
    """
    norm = _ba_normalise_design(design)
    arms = set()
    for component in norm['components']:
        arms.add(component['with_arm'])
        arms.add(component['without_arm'])
    budgets = set(norm['budgets'])
    key = norm['budget_key']
    rows = []
    seen = set()
    for index, raw in enumerate(records or []):
        if not isinstance(raw, dict):
            raise ValueError(f'records[{index}] must be an object')
        arm = raw.get('variant', raw.get('arm'))
        if arm not in arms:
            continue
        budget = raw.get('budget', raw.get(key))
        try:
            budget_int = _as_seed_int(budget)  # noqa: F821
            seed_int = _as_seed_int(raw.get('seed'))  # noqa: F821
        except Exception:
            continue
        if budget_int is None or budget_int not in budgets or seed_int is None:
            continue
        ppl = raw.get('ppl')
        if isinstance(ppl, (bool, np.bool_)):
            continue
        try:
            ppl = float(ppl)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(ppl):
            continue
        cell_key = (str(arm), int(seed_int), int(budget_int))
        if cell_key in seen:
            raise ValueError(
                f'records[{index}]: duplicate (arm={cell_key[0]!r}, '
                f'seed={cell_key[1]}, budget={cell_key[2]}) cell; the two '
                'PPLs would be ambiguous')
        seen.add(cell_key)
        rows.append({'arm': cell_key[0], 'seed': cell_key[1],
                     'budget': cell_key[2], 'ppl': ppl})
    return _ba_normalise_cells(rows)
