# ---- 2x2 交互不是「一个显著、一个不显著」：差中差、族内 max-T 与尺度敏感性 ----
#
# 冻结确认清单里的 sink-matched / no-sink 两个对照构成一个 2x2 敏感度设计
# （architecture × sink）。报告想回答的是「主对照的优势是不是 sink 解释的」，
# 这是一个**交互**（difference-in-differences）估计量：
#
#     interaction_s = (csa_sink - dense_sink)_s - (csa_nosink - dense_nosink)_s
#
# 它不能由「sink-matched 显著 / no-sink 不显著」或反过来推出。两个边际检验
# 的显著性差异本身不是交互的显著性（Gelman & Stern 2006）；两个边际共享
# 种子，且交互是它们的线性组合，逐条 CI 的目测重叠也不是交互的 CI。
#
# 本片段给这个估计量一个可重放的读数：
#
#   * 声明式 2x2：factors / levels / cells / contrast_factor，必须恰好四个
#     level 组合，每个组合一个 cell；不是「按臂名猜」。
#   * 逐种子差中差：只用四个 cell 的共同种子；缺 cell/缺种子不插补，
#     直接 not_estimable。
#   * 主尺度是对数 PPL（乘法效应/比值），另并列原始 PPL 尺度作敏感性；
#     两种尺度的符号或拒绝结论不一致时显式 warning，不挑一个好看的。
#   * 单交互用 exact sign-flip；两个边际 + 交互再用 exact max-T 给族内
#     p 与同时区间；枚举上限/退化时具名退 Holm，不伪装成 max-T。
#   * 若声明了交互 SESOI，再给 paired TOST；未声明就只报描述，不把
#     「未拒绝零」写成「无交互」。
#   * 产物 builder/validator/writer + audit gate；`confirmed`/`verified`/
#     `p_holm` 等冻结字段不出现、也不被本块改变。
#
# SCOPE: 探索性次级读数，不新增训练、不读任何 split、不改变冻结判定。

CONTRAST_INTERACTION_POLICY = {
    'version': 'contrast-interaction-1',
    'primary_scale': 'log_ppl',
    'scales': ('log_ppl', 'ppl'),
    'default_alpha': 0.05,
    'default_min_n': 3,
    'family': ('marginal_at_level_0', 'marginal_at_level_1', 'interaction'),
    'family_size': 3,
    'max_n_exact_family': JOINT_SIGNFLIP_MAX_N,
    'role': ('exploratory 2x2 difference-in-differences readout: exact '
             'sign-flip interaction, family max-T, paired TOST when a margin '
             'is declared.  It does not change a frozen p/Holm/confirmed '
             'verdict, does not add a training run, and never reads a split.'),
    'fallacy_note': ('a difference in significance is not a significant '
                     'difference: one marginal rejecting and the other not '
                     'does not by itself establish an interaction.'),
}

CONTRAST_INTERACTION_ARTIFACT_KIND = 'contrast_interaction'
CONTRAST_INTERACTION_ARTIFACT_SCHEMA = 1
# CPython 3.12 refuses to decimalise an integer above ~4300 digits.  Exact seed
# labels and min_n are copied into the strict-JSON result, so keep them inside
# a bound that can still be serialised (the same 10**4000 convention used by
# the scale-ladder audit).
CONTRAST_INTERACTION_MAX_REPORTABLE_INT = 10 ** 4000
CONTRAST_INTERACTION_ARTIFACT_KEYS = (
    'kind', 'schema', 'created_at', 'source', 'inputs', 'result',
    'content_sha256',
)
CONTRAST_INTERACTION_FORBIDDEN_KEYS = frozenset({
    'confirmed', 'confirmed_effective', 'p_holm', 'verified', 'verdict',
    'decision', 'frozen_decision', 'final_decision', 'holm',
})
CONTRAST_INTERACTION_INPUT_KEYS = (
    'design', 'cells', 'alpha', 'min_n', 'margin', 'margin_relative_pct',
)
CONTRAST_INTERACTION_SCALE_KEYS = (
    'scale', 'marginals', 'interaction', 'family',
)
CONTRAST_INTERACTION_STAT_FIELDS = (
    'n', 'n_nonzero', 'mean', 'ci95', 'ci95_ppl_ratio', 'p_exact_signflip',
    'p_floor', 'method', 'per_seed', 'seeds', 'resolution_limited',
    'underpowered', 'direction', 'definition', 'name',
)


def _ci_repr(value, limit=160):
    """Exception-safe short repr (integers can be too large to stringify)."""
    try:
        text = repr(value)
    except Exception:
        return f'<unrepresentable {type(value).__name__}>'
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _ci_json_equal(a, b):
    """Structural JSON equality that never raises on a tampered value.

    A hand-edited artifact can carry an array where the replay has a list;
    plain ``a != b`` then returns an array and its truth value is ambiguous.
    Only JSON scalar types take part in equality, so a non-JSON object is
    simply different.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return (isinstance(a, bool) and isinstance(b, bool) and a is b)
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return False
        if set(a) != set(b):
            return False
        return all(_ci_json_equal(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple))
                and isinstance(b, (list, tuple))):
            return False
        return (len(a) == len(b)
                and all(_ci_json_equal(x, y) for x, y in zip(a, b)))
    if (isinstance(a, (int, float)) and not isinstance(a, bool)
            and isinstance(b, (int, float)) and not isinstance(b, bool)):
        try:
            return bool(a == b)
        except (TypeError, ValueError, OverflowError):
            return False
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    return False


def _ci_key_text(value, limit=120):
    """Bounded text for an untrusted mapping key/path component.

    ``str(10**10000)`` raises in CPython 3.12, and a mixed int/str key set
    cannot be ``sorted()``; both occur in tamper tests.  Ordinary keys keep
    their historical ``str(key)`` spelling (including NumPy scalars/Decimal),
    so diagnostics do not silently change for valid inputs.
    """
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit - 3] + '...'
    try:
        text = str(value)
    except Exception:
        return _ci_repr(value, limit=limit)
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _ci_key_json(value):
    """JSON object key without an int->decimal conversion limit.

    Ordinary keys keep their historical ``str(key)`` spelling.  A huge integer
    key falls back to exact hexadecimal (which CPython can always format), so
    hashing a tampered artifact reports a mismatch instead of raising the
    interpreter's digit-limit error.
    """
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:
        if isinstance(value, int) and not isinstance(value, bool):
            return hex(value)
        return _ci_key_text(value)


def _ci_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string, got '
                         f'{_ci_repr(value)}')
    return value.strip()


def _ci_int(value, name, *, minimum=None, maximum=None):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be an integer, not a boolean')
    # Numeric strings are the legacy spelling used by the endpoint parser
    # (``by_len.ppl@4096``), so keep accepting an exact integer string.  Every
    # other value goes through the repository-wide exact-integer reader:
    # plain ``int()`` silently truncates a fractional Fraction/Decimal or a
    # NumPy scalar (``min_n=2.5`` became 2), which would accept a malformed
    # analysis contract instead of failing closed.
    if isinstance(value, (str, bytes, bytearray)):
        try:
            out = int(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'{name} must be an integer, got '
                             f'{_ci_repr(value)}') from None
    else:
        out = _as_seed_int(value)  # noqa: F821 (part_13)
        if out is None:
            raise ValueError(f'{name} must be an exact integer, got '
                             f'{_ci_repr(value)}')
    if minimum is not None and out < minimum:
        raise ValueError(f'{name} must be >= {minimum}, got {_ci_repr(out)}')
    if maximum is not None and out > maximum:
        raise ValueError(f'{name} must be <= {maximum}, got {_ci_repr(out)}')
    return out


def _ci_alpha(value):
    return _validated_alpha(
        value, context='contrast_interaction alpha')


def _ci_number(value, name):
    number = _finite_scalar(value)
    if number is None:
        raise ValueError(f'{name} must be a finite real number, got '
                         f'{_ci_repr(value)}')
    return float(number)


def _ci_float_or_none(value):
    """Best-effort finite float for an error/not-estimable record."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _ci_int_or_none(value):
    """Best-effort integer for an error/not-estimable record."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _ci_endpoint_text(value, limit=240):
    """JSON-safe, bounded endpoint label for a not-estimable diagnostic."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit - 3] + '...'
    try:
        text = str(value)
    except Exception:
        return _ci_key_text(value, limit=limit)
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _ci_md_text(value, limit=500):
    """Display text for a Markdown field (a huge int must not abort rendering)."""
    if isinstance(value, str):
        text = value
    else:
        text = _ci_repr(value, limit=limit)
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _ci_md_num(value, fmt):
    """Format a numeric Markdown field, withholding unrepresentable values."""
    number = _ci_float_or_none(value)
    if number is None:
        return '—'
    try:
        return fmt.format(number)
    except (TypeError, ValueError, OverflowError):
        return '—'


def _ci_ppl(value, name):
    number = _ci_number(value, name)
    if number <= 0.0:
        raise ValueError(f'{name} must be a positive PPL, got {number!r}')
    return number


def _ci_seed(value, name):
    seed = _as_seed_int(value)
    if (seed is None or seed < 0
            or seed > CONTRAST_INTERACTION_MAX_REPORTABLE_INT):
        raise ValueError(f'{name} must be an exact non-negative integer seed '
                         f'within the reportable range, got {_ci_repr(value)}')
    return seed


def _ci_margin(value, name):
    if value is None:
        return None
    margin = _ci_number(value, name)
    if margin <= 0.0:
        raise ValueError(f'{name} must be positive, got {margin!r}')
    return margin


def _ci_margin_pair(margin, margin_relative_pct):
    """Normalise a declared SESOI, accepting either log units or a percentage.

    Declaring both is allowed only when they describe the same margin; a
    silent disagreement would make the stored artifact replay under a
    different equivalence claim than the one the caller intended.
    """
    margin = _ci_margin(margin, 'margin')
    rel = None
    if margin_relative_pct is not None:
        rel = _ci_number(margin_relative_pct, 'margin_relative_pct')
        if rel <= -100.0:
            raise ValueError('margin_relative_pct must be > -100')
        derived = math.log1p(rel / 100.0)
        if derived <= 0.0:
            raise ValueError('margin_relative_pct yields a non-positive log margin')
        if margin is None:
            margin = derived
        elif not math.isclose(float(margin), float(derived),
                              rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                'margin and margin_relative_pct describe different SESOIs: '
                f'{margin!r} log units vs {derived!r} log units from '
                f'{rel!r}%')
    return margin, rel


def _ci_jsonable(value):
    """Convert a nested mapping/list to strict-JSON safe Python objects."""
    if isinstance(value, dict):
        return {_ci_key_json(k): _ci_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ci_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _ci_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _ci_jsonable(value.item())
    if isinstance(value, int) and not isinstance(value, bool):
        try:
            str(value)
        except (ValueError, OverflowError):
            # A tampered artifact can carry an integer beyond CPython's
            # decimal-conversion limit.  Encode it exactly in hex so a
            # validation/hash diagnostic can still be produced and written.
            return {'__unprintable_int_hex__': hex(value)}
    return value


def _ci_content_sha(artifact):
    body = {k: v for k, v in artifact.items()
            if k not in ('created_at', 'content_sha256')}
    blob = json.dumps(_ci_jsonable(body), sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def _ci_strict_json_problem(label, body):
    """Return a bounded diagnostic when ``body`` is not strict JSON."""
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
                f'{type(exc).__name__}: {_ci_repr(exc)}')
    return None


def _ci_shape_problems(inputs, result):
    """Cheap result checks that must hold even when replay is skipped.

    ``replay=False`` is used by archive/tamper tooling, so a hand-edited body
    whose hash has been refreshed must still be rejected.  Only structural and
    knob-identity fields are checked here; the full replay remains the
    authoritative comparison.
    """
    problems = []
    if not isinstance(result, dict):
        return ['result must be an object']
    status = result.get('status')
    if (not isinstance(status, str)
            or status not in ('ok', 'warnings', 'not_estimable')):
        problems.append('result.status must be ok/warnings/not_estimable, got '
                        f'{_ci_repr(status)}')
        return problems
    if not _ci_json_equal(result.get('kind'),
                          CONTRAST_INTERACTION_ARTIFACT_KIND):
        problems.append(
            f'result.kind must be {CONTRAST_INTERACTION_ARTIFACT_KIND!r}, got '
            f'{_ci_repr(result.get("kind"))}')
    if not _ci_json_equal(
            result.get('primary_scale'),
            CONTRAST_INTERACTION_POLICY['primary_scale']):
        problems.append('result.primary_scale does not match the declared policy')
    if status == 'not_estimable':
        reason = result.get('reason')
        if not isinstance(reason, str) or not reason.strip():
            problems.append('a not_estimable result needs a non-empty reason')
        return problems

    knobs = (
        ('alpha', _ci_alpha, 'result.alpha'),
        ('min_n', lambda v: _ci_int(v, 'result.min_n', minimum=2,
                                    maximum=CONTRAST_INTERACTION_MAX_REPORTABLE_INT),
         'result.min_n'),
    )
    for key, decode, label in knobs:
        try:
            value = decode(result.get(key))
        except (TypeError, ValueError) as exc:
            problems.append(f'{label} is malformed: {exc}')
            continue
        if not isinstance(inputs, dict) or key not in inputs:
            continue
        try:
            declared = decode(inputs.get(key))
        except (TypeError, ValueError):
            continue
        if value != declared:
            problems.append(f'{label} does not match inputs.{key}')

    if isinstance(inputs, dict) and 'margin' in inputs:
        try:
            result_margin = (None if result.get('margin') is None
                             else _ci_margin(result.get('margin'),
                                             'result.margin'))
        except (TypeError, ValueError) as exc:
            problems.append(f'result.margin is malformed: {exc}')
        else:
            declared_margin = inputs.get('margin')
            if declared_margin is None:
                if result_margin is not None:
                    problems.append('result.margin does not match inputs.margin')
            else:
                try:
                    declared_margin = _ci_margin(
                        declared_margin, 'inputs.margin')
                except (TypeError, ValueError):
                    declared_margin = None
                if (declared_margin is None or result_margin is None
                        or not math.isclose(result_margin, declared_margin,
                                            rel_tol=1e-12, abs_tol=1e-12)):
                    problems.append('result.margin does not match inputs.margin')

    if isinstance(inputs, dict) and isinstance(inputs.get('design'), dict):
        if not _ci_json_equal(result.get('design'), inputs.get('design')):
            problems.append('result.design does not match inputs.design')

    scales = result.get('scales')
    if not isinstance(scales, dict):
        problems.append('result.scales must be an object')
        return problems
    for scale in CONTRAST_INTERACTION_POLICY['scales']:
        block = scales.get(scale)
        if not isinstance(block, dict):
            problems.append(f'result.scales[{scale!r}] must be an object')
            continue
        if not isinstance(block.get('interaction'), dict):
            problems.append(
                f'result.scales[{scale!r}].interaction must be an object')
        if not isinstance(block.get('family'), dict):
            problems.append(
                f'result.scales[{scale!r}].family must be an object')
        if not isinstance(block.get('marginals'), dict):
            problems.append(
                f'result.scales[{scale!r}].marginals must be an object')
    return problems


def _ci_not_estimable(reason, *, extra=None):
    out = {
        'kind': 'contrast_interaction',
        'status': 'not_estimable',
        'reason': str(reason),
        'primary_scale': CONTRAST_INTERACTION_POLICY['primary_scale'],
        'interaction_claim_allowed': False,
        'warnings': [],
        'note': CONTRAST_INTERACTION_POLICY['role'],
    }
    if isinstance(extra, dict):
        out.update(extra)
    return out


# ---- design algebra --------------------------------------------------------

def _ci_normalise_design(design):
    """Validate a complete 2x2 declaration and derive cell roles.

    ``cells`` maps a cell name to ``{factor: level}``.  Exactly four cells are
    required, one for every combination of the two declared levels per factor.
    The same level spelled differently in two cells is a different level; no
    silent stripping/case-folding is performed.
    """
    if not isinstance(design, dict):
        raise ValueError('design must be a mapping with factors/levels/cells')
    factors = design.get('factors')
    if isinstance(factors, (str, bytes)) or not isinstance(factors, (list, tuple)):
        raise ValueError('design.factors must be a sequence of exactly two names')
    factors = [_ci_text(f, 'design.factors[]') for f in factors]
    if len(factors) != 2 or factors[0] == factors[1]:
        raise ValueError('design.factors must contain two distinct names, got '
                         f'{_ci_repr(factors)}')
    levels = design.get('levels')
    if not isinstance(levels, dict):
        raise ValueError('design.levels must be a mapping factor -> [level0, level1]')
    norm_levels = {}
    for factor in factors:
        raw = levels.get(factor)
        if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
            raise ValueError(f'design.levels[{factor!r}] must be a sequence of two levels')
        vals = [_ci_text(v, f'design.levels[{factor!r}][]') for v in raw]
        if len(vals) != 2 or vals[0] == vals[1]:
            raise ValueError(f'design.levels[{factor!r}] must contain two distinct '
                             f'levels, got {_ci_repr(vals)}')
        norm_levels[factor] = vals
    cells = design.get('cells')
    if not isinstance(cells, dict) or not cells:
        raise ValueError('design.cells must be a non-empty mapping cell -> {factor: level}')
    norm_cells = {}
    seen = {}
    for raw_name, assignment in cells.items():
        name = _ci_text(raw_name, 'design.cells key')
        if name in norm_cells:
            raise ValueError(f'design.cells has a duplicate name {name!r}')
        if not isinstance(assignment, dict):
            raise ValueError(f'design.cells[{name!r}] must be a mapping factor -> level')
        extra = set(assignment) - set(factors)
        missing = set(factors) - set(assignment)
        if extra or missing:
            raise ValueError(
                f'design.cells[{name!r}] must assign exactly {factors}; '
                f'missing={_ci_repr(sorted(missing, key=_ci_repr))}, '
                f'extra={_ci_repr(sorted(extra, key=_ci_repr))}')
        combo = []
        for factor in factors:
            level = assignment[factor]
            if not isinstance(level, str) or level not in norm_levels[factor]:
                raise ValueError(
                    f'design.cells[{name!r}][{factor!r}]={_ci_repr(level)} is '
                    f'not one of {norm_levels[factor]}')
            combo.append(level)
        key = tuple(combo)
        if key in seen:
            raise ValueError(
                f'design.cells[{name!r}] repeats the level combination already '
                f'used by {seen[key]!r}: {key!r}')
        seen[key] = name
        norm_cells[name] = {factors[0]: combo[0], factors[1]: combo[1]}
    expected = {(a, b) for a in norm_levels[factors[0]]
                for b in norm_levels[factors[1]]}
    if set(seen) != expected:
        missing = sorted(expected - set(seen))
        raise ValueError(f'design.cells must cover the full 2x2; missing '
                         f'combination(s) {missing}')
    contrast = design.get('contrast_factor', factors[0])
    contrast = _ci_text(contrast, 'design.contrast_factor')
    if contrast not in factors:
        raise ValueError(f'design.contrast_factor={contrast!r} is not one of '
                         f'{factors}')
    moderator = factors[0] if factors[1] == contrast else factors[1]
    roles = {}
    for cl in norm_levels[contrast]:
        roles[cl] = {}
        for ml in norm_levels[moderator]:
            match = [name for name, assignment in norm_cells.items()
                     if assignment[contrast] == cl
                     and assignment[moderator] == ml]
            if len(match) != 1:
                raise ValueError(
                    f'design.cells is not a bijection for {contrast}={cl!r}, '
                    f'{moderator}={ml!r}: matches={match!r}')
            roles[cl][ml] = match[0]
    label = design.get('label')
    label = (_ci_text(label, 'design.label') if label is not None
             else f'{contrast} x {moderator}')
    return {
        'factors': factors,
        'levels': norm_levels,
        'cells': norm_cells,
        'contrast_factor': contrast,
        'moderator_factor': moderator,
        'contrast_levels': list(norm_levels[contrast]),
        'moderator_levels': list(norm_levels[moderator]),
        'cell_roles': roles,
        'label': label,
        'effect': f'{contrast}: {norm_levels[contrast][0]} - '
                  f'{norm_levels[contrast][1]}',
    }


def _ci_normalise_cells(cells, design):
    """Validate cell measurements and return the common-seed panel.

    Missing/extra cells and malformed measurements are returned as a named
    ``not_estimable`` record rather than interpolated.  Seeds present in only
    some cells are dropped and reported; the interaction is only defined on
    the common set.
    """
    out = {
        'ok': False, 'reason': None, 'per_cell': {},
        'missing_cells': [], 'extra_cells': [], 'missing_seeds': {},
        'n_by_cell': {}, 'seeds': [],
    }
    if not isinstance(cells, dict):
        out['reason'] = 'cells must be a mapping cell -> {seed: PPL}'
        return out
    expected = list(design['cells'])
    out['missing_cells'] = [name for name in expected if name not in cells]
    out['extra_cells'] = sorted(set(cells) - set(expected), key=_ci_repr)
    if out['missing_cells']:
        out['reason'] = f'missing cell(s): {out["missing_cells"]}'
        return out
    if out['extra_cells']:
        out['reason'] = (f'undeclared cell(s): '
                         f'{_ci_repr(out["extra_cells"])}')
        return out
    per_cell = {}
    for name in expected:
        raw = cells[name]
        if not isinstance(raw, dict):
            out['reason'] = f'cells[{name!r}] must be a mapping seed -> PPL'
            return out
        seed_map = {}
        for raw_seed, raw_ppl in raw.items():
            try:
                seed = _ci_seed(raw_seed, f'cells[{name!r}] seed')
            except ValueError as exc:
                out['reason'] = str(exc)
                return out
            try:
                ppl = _ci_ppl(raw_ppl, f'cells[{name!r}][seed {seed}]')
            except ValueError as exc:
                out['reason'] = str(exc)
                return out
            if seed in seed_map:
                out['reason'] = f'cells[{name!r}] repeats seed {seed}'
                return out
            seed_map[seed] = ppl
        if not seed_map:
            out['reason'] = f'cells[{name!r}] has no measurements'
            return out
        per_cell[name] = seed_map
        out['n_by_cell'][name] = len(seed_map)
    all_seeds = set.intersection(*[set(v) for v in per_cell.values()])
    union = set().union(*[set(v) for v in per_cell.values()])
    out['seeds'] = sorted(all_seeds)
    out['missing_seeds'] = {
        name: sorted(union - set(per_cell[name])) for name in expected
        if union - set(per_cell[name])
    }
    out['per_cell'] = per_cell
    out['ok'] = True
    return out


# ---- one scale: marginals, interaction, family -----------------------------

def _ci_stat_block(values, seeds, *, alpha, min_n, scale, name, definition):
    """Exact sign-flip + t CI for one paired contrast on one scale."""
    st = paired_effect_stats(values, alpha=alpha, min_n=min_n)
    ex = exact_signflip_p(values)
    n = int(st.get('n') or 0)
    mean = _finite_scalar(st.get('mean'))
    lo = _finite_scalar(st.get('ci95_low'))
    hi = _finite_scalar(st.get('ci95_high'))
    p = _finite_scalar(st.get('p_exact_signflip'))
    floor = _finite_scalar(st.get('p_floor'))
    ci = ([float(lo), float(hi)] if lo is not None and hi is not None else None)
    ratio_ci = None
    ratio = None
    if scale == 'log_ppl':
        if mean is not None:
            ratio = _safe_exp_ppl(mean)
            if not math.isfinite(ratio):
                ratio = None
        if ci is not None:
            a, b = _safe_exp_ppl(ci[0]), _safe_exp_ppl(ci[1])
            ratio_ci = ([a, b] if math.isfinite(a) and math.isfinite(b) else None)
    direction = None
    if mean is not None and math.isfinite(mean):
        direction = ('zero' if mean == 0.0 else
                     ('positive' if mean > 0.0 else 'negative'))
    return {
        'name': name,
        'definition': definition,
        'scale': scale,
        'n': n,
        'n_nonzero': int(ex.get('n_nonzero') or 0),
        'mean': mean,
        'ci95': ci,
        'ci95_ppl_ratio': ratio_ci,
        'p_exact_signflip': p,
        'p_floor': floor,
        'method': ex.get('method'),
        'per_seed': [float(x) for x in values],
        'seeds': [int(s) for s in seeds],
        'resolution_limited': bool(st.get('resolution_limited')),
        'underpowered': bool(st.get('underpowered')),
        'direction': direction,
        'mean_ppl_ratio': ratio,
    }


def _ci_family(records, *, alpha):
    """Exact max-T over the three declared contrasts, with a named Holm fallback."""
    names = list(records)
    raw = {}
    for name in names:
        p = _finite_scalar(records[name].get('p_exact_signflip'))
        raw[name] = (1.0 if p is None else float(p))
    try:
        fam = maxt_joint_correction(records, alpha=alpha,
                                    family_size=len(names))
    except Exception as exc:
        fam = {'kind': 'maxt_joint_correction', 'status': 'not_estimable',
               'reason': f'{type(exc).__name__}: {exc}', 'by_key': {}}
    if fam.get('status') == 'ok' and fam.get('family_ready') is True:
        p_family = {}
        p_raw_studentized = {}
        ci_sim = {}
        for name in names:
            rec = (fam.get('by_key') or {}).get(name) or {}
            p = _finite_scalar(rec.get('p_maxt'))
            p_family[name] = (None if p is None else float(p))
            pr = _finite_scalar(rec.get('p_raw_signflip'))
            p_raw_studentized[name] = (None if pr is None else float(pr))
            ci = rec.get('ci_simultaneous')
            if (isinstance(ci, (list, tuple)) and len(ci) == 2
                    and all(_finite_scalar(x) is not None for x in ci)):
                ci_sim[name] = [float(ci[0]), float(ci[1])]
        if all(v is not None for v in p_family.values()):
            _groups = (fam.get('groups')
                       if isinstance(fam.get('groups'), dict) else {})
            _group = _groups.get(str(list(records[names[0]]['seeds'])), {})
            return {
                'method': 'exact sign-flip max-T',
                'status': 'ok',
                'family_size': len(names),
                'family_ready': True,
                'p_family': p_family,
                'p_raw': raw,
                'p_raw_studentized': p_raw_studentized,
                'p_family_statistic': 'studentized t max-T',
                'ci_simultaneous': ci_sim,
                'critical_value': (_finite_scalar(_group.get('critical_value'))
                                   if isinstance(_group, dict) else None),
                'warnings': list(fam.get('warnings') or []),
                'note': ('simultaneous intervals are the family max-T '
                         'intervals; they are not the per-contrast t CIs'),
            }
    # Fallback: Holm on the SAME three raw p-values.  This is still a valid
    # family-wise readout, but it is not exact max-T and has no simultaneous
    # interval; say so instead of silently presenting a different method.
    try:
        holm = holm_adjust([(name, raw[name]) for name in names], alpha=alpha)
        p_family = {name: (float(holm[name]['p_holm'])
                           if holm.get(name, {}).get('p_holm') is not None
                           else None) for name in names}
        method = 'holm_fallback'
        warnings = [f'max-T unavailable: {fam.get("reason") or "family not ready"}']
    except Exception as exc:
        p_family = {name: None for name in names}
        method = 'not_estimable'
        warnings = [f'Holm fallback failed: {type(exc).__name__}: {exc}']
    return {
        'method': method,
        'status': ('not_estimable' if all(v is None for v in p_family.values())
                   else 'fallback'),
        'family_size': len(names),
        'family_ready': False,
        'p_family': p_family,
        'p_raw': raw,
        'p_raw_studentized': {name: None for name in names},
        'p_family_statistic': 'holm of the mean sign-flip p',
        'ci_simultaneous': {},
        'critical_value': None,
        'warnings': warnings,
        'note': ('named fallback: Holm on the same three p-values.  It is '
                 'family-wise valid but is not the exact max-T statistic and '
                 'does not supply simultaneous intervals.'),
    }


def _ci_tost(values, margin, alpha):
    if margin is None:
        return None
    try:
        out = tost_paired(values, margin, conf=1.0 - float(alpha))
    except Exception as exc:
        return {'status': 'not_estimable',
                'reason': f'{type(exc).__name__}: {exc}',
                'margin': float(margin)}
    return {
        'status': 'ok',
        'margin': float(margin),
        'conf': _finite_scalar(out.get('conf')),
        'equivalent': out.get('equivalent'),
        'p_lo': _finite_scalar(out.get('p_lo')),
        'p_hi': _finite_scalar(out.get('p_hi')),
        'p_tost': _finite_scalar(out.get('p_tost')),
        'ci95': ([float(out['ci'][0]), float(out['ci'][1])]
                 if isinstance(out.get('ci'), (list, tuple))
                 and len(out['ci']) == 2
                 and all(_finite_scalar(x) is not None for x in out['ci'])
                 else None),
        'method': out.get('method'),
        'note': ('TOST is an equivalence claim and needs a pre-registered '
                 'margin; a non-significant interaction alone is not '
                 'equivalence.'),
    }


def _ci_scale_analysis(per_cell, seeds, design, *, alpha, min_n, scale,
                       margin):
    contrast = design['contrast_factor']
    moderator = design['moderator_factor']
    cl0, cl1 = design['contrast_levels']
    ml0, ml1 = design['moderator_levels']
    transform = (lambda x: math.log(x)) if scale == 'log_ppl' else (lambda x: x)
    values = {
        cell: {seed: transform(per_cell[cell][seed]) for seed in seeds}
        for cell in design['cells']
    }

    def _contrast(ml):
        left = design['cell_roles'][cl0][ml]
        right = design['cell_roles'][cl1][ml]
        return {s: values[left][s] - values[right][s] for s in seeds}

    marginal_vals = {ml: _contrast(ml) for ml in (ml0, ml1)}
    interaction_vals = {
        s: marginal_vals[ml0][s] - marginal_vals[ml1][s] for s in seeds}
    # Every input PPL is finite, but a raw-PPL difference of two finite
    # values near the float limit can still overflow to inf, and the family
    # is only paired when all three contrasts are defined on the same seed.
    # Drop such a seed from THIS scale (recorded explicitly) instead of
    # emitting an inf/NaN into a strict-JSON artifact or a bogus statistic.
    all_contrasts = (marginal_vals[ml0], marginal_vals[ml1], interaction_vals)
    kept_seeds, dropped_seeds = [], []
    for seed in seeds:
        if all(math.isfinite(float(c[seed])) for c in all_contrasts):
            kept_seeds.append(seed)
        else:
            dropped_seeds.append(seed)
    marginals = {}
    for ml in (ml0, ml1):
        vals = [marginal_vals[ml][s] for s in kept_seeds]
        name = f'{contrast}: {cl0} - {cl1} @ {moderator}={ml}'
        marginals[ml] = _ci_stat_block(
            vals, kept_seeds, alpha=alpha, min_n=min_n, scale=scale, name=name,
            definition=f'({cl0} - {cl1}) at {moderator}={ml}')
    inter_vals = [interaction_vals[s] for s in kept_seeds]
    inter_name = (f'interaction: ({contrast}: {cl0} - {cl1}) @ '
                  f'{moderator}={ml0} minus @ {moderator}={ml1}')
    interaction = _ci_stat_block(
        inter_vals, kept_seeds, alpha=alpha, min_n=min_n, scale=scale,
        name=inter_name,
        definition=(f'({marginals[ml0]["name"]}) - '
                    f'({marginals[ml1]["name"]})'))
    records = {
        marginals[ml0]['name']: {
            'deltas': marginals[ml0]['per_seed'], 'seeds': list(kept_seeds),
            'eligible': True},
        marginals[ml1]['name']: {
            'deltas': marginals[ml1]['per_seed'], 'seeds': list(kept_seeds),
            'eligible': True},
        interaction['name']: {
            'deltas': interaction['per_seed'], 'seeds': list(kept_seeds),
            'eligible': True},
    }
    family = _ci_family(records, alpha=alpha)
    interaction['p_family'] = (family.get('p_family') or {}).get(
        interaction['name'])
    interaction['p_family_statistic'] = family.get('p_family_statistic')
    interaction['ci95_simultaneous'] = (
        (family.get('ci_simultaneous') or {}).get(interaction['name']))
    for ml in (ml0, ml1):
        marginals[ml]['p_family'] = (family.get('p_family') or {}).get(
            marginals[ml]['name'])
    # A SESOI declared in log-PPL units is not a raw-PPL-point margin.  The
    # raw scale is a sensitivity companion, so it must not silently reuse the
    # log-scale equivalence claim; TOST is reported on the primary scale only.
    tost = _ci_tost(inter_vals, margin, alpha) if scale == 'log_ppl' else None
    return {
        'scale': scale,
        'marginals': {ml0: marginals[ml0], ml1: marginals[ml1]},
        'interaction': interaction,
        'family': family,
        'tost': tost,
        'seeds': list(kept_seeds),
        'n_input_seeds': len(seeds),
        'dropped_seeds': list(dropped_seeds),
    }


# ---- interpretation --------------------------------------------------------

def _ci_sign(value):
    x = _finite_scalar(value)
    if x is None:
        return None
    return 0 if x == 0.0 else (1 if x > 0.0 else -1)


def _ci_interpret(analysis, *, alpha, min_n):
    interaction = analysis['interaction']
    family = analysis['family']
    p_raw = _finite_scalar(interaction.get('p_exact_signflip'))
    p_fam = _finite_scalar(interaction.get('p_family'))
    raw_marg = {}
    fam_marg = {}
    for level, block in analysis['marginals'].items():
        raw_marg[level] = (_finite_scalar(block.get('p_exact_signflip')) is not None
                           and float(block['p_exact_signflip']) <= alpha)
        pf = _finite_scalar(block.get('p_family'))
        fam_marg[level] = (pf is not None and float(pf) <= alpha)
    raw_reject = p_raw is not None and float(p_raw) <= alpha
    fam_reject = p_fam is not None and float(p_fam) <= alpha
    n = int(interaction.get('n') or 0)
    resolution_limited = bool(interaction.get('resolution_limited'))
    naive_marg = [k for k, v in raw_marg.items() if v]
    adj_marg = [k for k, v in fam_marg.items() if v]
    fallacy = bool(len(naive_marg) == 1 and not raw_reject)
    if fam_reject and n >= min_n and not resolution_limited:
        verdict = 'effect_modification'
    elif len(adj_marg) == 1:
        verdict = 'difference_in_significance_unresolved'
    elif len(adj_marg) == 2:
        verdict = 'parallel_effects'
    elif len(naive_marg) == 1 and not raw_reject:
        verdict = 'difference_in_significance_only'
    else:
        verdict = 'no_evidence_of_interaction'
    claim_allowed = bool(fam_reject and n >= min_n and not resolution_limited)
    direction = _ci_sign(interaction.get('mean'))
    contrast = analysis['interaction']['definition']
    if direction is None:
        direction_text = '交互的逐种子均值不可估计'
    elif direction > 0:
        direction_text = ('交互为正：log 尺度上后一个 moderator 水平的对照效应更小／'
                          '更负，两个水平的简单效应不同；正负含义取决于对照方向')
    elif direction < 0:
        direction_text = ('交互为负：log 尺度上前一个 moderator 水平的对照效应更小／'
                          '更负，两个水平的简单效应不同；正负含义取决于对照方向')
    else:
        direction_text = '交互的逐种子均值恰为 0'
    return {
        'interaction_detected': fam_reject,
        'interaction_detected_raw': raw_reject,
        'interaction_claim_allowed': claim_allowed,
        'claim_level': 'exploratory',
        'marginal_raw_rejections': sorted(naive_marg),
        'marginal_family_rejections': sorted(adj_marg),
        'difference_in_significance_fallacy': fallacy,
        'classification': verdict,
        'direction': direction,
        'direction_text': direction_text,
        'definition': contrast,
        'note': ('if the interaction is not rejected, a significant marginal '
                 'and a non-significant marginal do not establish that the '
                 'effects differ; report the interaction, not the pattern of '
                 'marginal p-values'),
    }


# ---- public analysis -------------------------------------------------------

def difference_in_differences(design, cells, *, alpha=0.05, min_n=3,
                              margin=None, margin_relative_pct=None):
    """Estimate a declared 2x2 interaction from per-seed PPL cells.

    ``design`` is the declaration (see :func:`_ci_normalise_design`);
    ``cells`` maps each declared cell to ``{seed: PPL}``.  The interaction is
    ``(cell_0_at_moderator_0 - cell_1_at_moderator_0) -
    (cell_0_at_moderator_1 - cell_1_at_moderator_1)`` on the primary log-PPL
    scale, with the raw-PPL scale reported as a sensitivity companion.

    Insufficient data (missing cell, fewer than ``min_n`` common seeds,
    non-positive PPL) returns ``status='not_estimable'``; invalid API arguments
    raise ``ValueError``.
    """
    a = _ci_alpha(alpha)
    mn = _ci_int(min_n, 'min_n', minimum=2,
                  maximum=CONTRAST_INTERACTION_MAX_REPORTABLE_INT)
    margin, rel = _ci_margin_pair(margin, margin_relative_pct)
    norm_design = _ci_normalise_design(design)
    norm_cells = _ci_normalise_cells(cells, norm_design)
    if not norm_cells['ok']:
        return _ci_not_estimable(norm_cells['reason'], extra={
            'design': norm_design,
            'alpha': a, 'min_n': mn, 'margin': margin,
            'margin_relative_pct': rel,
            'missing_cells': norm_cells['missing_cells'],
            'extra_cells': _ci_jsonable(norm_cells['extra_cells']),
            'n_by_cell': norm_cells['n_by_cell'],
        })
    seeds = list(norm_cells['seeds'])
    n = len(seeds)
    if n < mn:
        return _ci_not_estimable(
            f'only {n} common seed(s) across all four cells; need at least '
            f'{mn}', extra={
                'design': norm_design,
                'alpha': a, 'min_n': mn, 'margin': margin,
                'margin_relative_pct': rel,
                'seeds': seeds, 'n': n,
                'n_by_cell': norm_cells['n_by_cell'],
                'missing_seeds': norm_cells['missing_seeds'],
            })
    cell_means = {}
    for cell in norm_design['cells']:
        values = [norm_cells['per_cell'][cell][seed] for seed in seeds]
        try:
            mean_ppl = math.fsum(values) / float(len(values))
        except OverflowError:
            # Every PPL is a finite positive float, but their sum can exceed
            # the float range even when the mean is representable (e.g. three
            # seeds at 1e308).  Use the guarded scaled mean instead of
            # aborting the whole readout on a units-only overflow.
            mean_ppl = _scaled_mean_or_raw(values)  # noqa: F821 (part_32)
        mean_log = math.fsum(math.log(v) for v in values) / float(len(values))
        cell_means[cell] = {
            'n': len(values),
            'mean_ppl': float(mean_ppl),
            'mean_log_ppl': float(mean_log),
            'assignment': dict(norm_design['cells'][cell]),
        }
    analyses = {}
    for scale in CONTRAST_INTERACTION_POLICY['scales']:
        analyses[scale] = _ci_scale_analysis(
            norm_cells['per_cell'], seeds, norm_design, alpha=a, min_n=mn,
            scale=scale, margin=margin)
    primary = analyses[CONTRAST_INTERACTION_POLICY['primary_scale']]
    raw = analyses['ppl']
    log_inter = primary['interaction']
    raw_inter = raw['interaction']
    log_sign = _ci_sign(log_inter.get('mean'))
    raw_sign = _ci_sign(raw_inter.get('mean'))
    sign_agree = (None if log_sign is None or raw_sign is None
                  else bool(log_sign == raw_sign))
    log_p = _finite_scalar(log_inter.get('p_family'))
    raw_p = _finite_scalar(raw_inter.get('p_family'))
    reject_agree = (None if log_p is None or raw_p is None
                    else bool((float(log_p) <= a) == (float(raw_p) <= a)))
    scale_sensitive = bool(sign_agree is False or reject_agree is False)
    interpretation = _ci_interpret(primary, alpha=a, min_n=mn)
    warnings = []
    _dropped_by_scale = {
        name: block.get('dropped_seeds')
        for name, block in analyses.items() if block.get('dropped_seeds')}
    if _dropped_by_scale:
        warnings.append(
            'some paired seeds are not representable on a reported scale '
            '(e.g. a raw-PPL difference beyond the float range) and were '
            'dropped only from that scale: '
            + ', '.join(f'{name}: {len(seeds_)}' for name, seeds_ in
                        sorted(_dropped_by_scale.items())))
    if scale_sensitive:
        warnings.append('the log-PPL and raw-PPL interactions disagree in '
                        'sign or in the family-level reject decision; do not '
                        'pick the scale after seeing the result')
    if interpretation['difference_in_significance_fallacy']:
        warnings.append(CONTRAST_INTERACTION_POLICY['fallacy_note'])
    if primary['family'].get('method') != 'exact sign-flip max-T':
        warnings.append('family correction used the named fallback, not '
                        'exact max-T')
    if (log_inter.get('resolution_limited')
            or log_inter.get('underpowered')):
        warnings.append('the interaction is resolution-limited at this n; a '
                        'non-rejection is not evidence of no interaction')
    status = 'warnings' if warnings else 'ok'
    return {
        'kind': 'contrast_interaction',
        'status': status,
        'design': norm_design,
        'alpha': a,
        'min_n': mn,
        'margin': margin,
        'margin_relative_pct': rel,
        'primary_scale': CONTRAST_INTERACTION_POLICY['primary_scale'],
        'seeds': seeds,
        'n': n,
        'n_by_cell': norm_cells['n_by_cell'],
        'missing_seeds': norm_cells['missing_seeds'],
        'cell_means': cell_means,
        'scales': analyses,
        'scale_agreement': {
            'sign_agree': sign_agree,
            'reject_agree': reject_agree,
            'scale_sensitive': scale_sensitive,
            'log_interaction_mean': log_inter.get('mean'),
            'raw_interaction_mean': raw_inter.get('mean'),
        },
        'interpretation': interpretation,
        'interaction_claim_allowed': interpretation['interaction_claim_allowed'],
        'claim_level': 'exploratory',
        'warnings': warnings,
        'note': ('exploratory 2x2 difference-in-differences readout; it is '
                 'not in the frozen Holm family and never changes a frozen '
                 'p/Holm/verified/confirmed verdict.'),
    }


# ---- artifact build / validate / write ------------------------------------

def build_contrast_interaction(design, cells, *, alpha=0.05, min_n=3,
                               margin=None, margin_relative_pct=None,
                               source=None):
    """Build a content-addressed 2x2 interaction artifact.

    ``created_at`` is excluded from ``content_sha256``.  The stored ``inputs``
    are exactly the arguments needed to replay the result, so an auditor does
    not have to trust the stored statistics.
    """
    a = _ci_alpha(alpha)
    mn = _ci_int(min_n, 'min_n', minimum=2,
                  maximum=CONTRAST_INTERACTION_MAX_REPORTABLE_INT)
    margin, rel = _ci_margin_pair(margin, margin_relative_pct)
    norm_design = _ci_normalise_design(design)
    if not isinstance(cells, dict):
        raise ValueError('cells must be a mapping cell -> {seed: PPL}')
    # Serialise the caller's cells verbatim (after JSON normalisation) so the
    # replay is over the same measurements, not over a silently cleaned copy.
    encoded_cells = _ci_jsonable(cells)
    result = difference_in_differences(
        norm_design, cells, alpha=a, min_n=mn, margin=margin,
        margin_relative_pct=rel)
    # A JSON object cannot carry an integer key beyond CPython's decimal
    # limit, and a huge integer value cannot be written at all.  If the
    # serialised cells no longer replay to the same result, storing this
    # artifact would create one that its own validator must reject; fail the
    # builder instead so malformed input cannot masquerade as a valid record.
    try:
        encoded_result = difference_in_differences(
            norm_design, encoded_cells, alpha=a, min_n=mn, margin=margin,
            margin_relative_pct=rel)
    except Exception as exc:
        raise ValueError(
            'cells cannot be serialised without changing their identity: '
            f'{type(exc).__name__}: {exc}') from None
    if not compare_contrast_interaction_results(result, encoded_result).get(
            'match'):
        raise ValueError(
            'cells cannot be serialised without changing their identity '
            '(unrepresentable seed/PPL value or non-JSON cell key)')
    inputs = {
        'design': norm_design,
        'cells': encoded_cells,
        'alpha': a,
        'min_n': mn,
        'margin': margin,
        'margin_relative_pct': rel,
    }
    artifact = {
        'kind': CONTRAST_INTERACTION_ARTIFACT_KIND,
        'schema': CONTRAST_INTERACTION_ARTIFACT_SCHEMA,
        'created_at': reproducible_timestamp(),
        'source': (None if source is None
                   else _ci_endpoint_text(source)),
        'inputs': inputs,
        'result': _ci_jsonable(result),
    }
    artifact['content_sha256'] = _ci_content_sha(artifact)
    return artifact


def _ci_forbidden_paths(obj, path='artifact'):
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f'{path}.{_ci_key_text(key)}'
            if key in CONTRAST_INTERACTION_FORBIDDEN_KEYS:
                found.append(here)
            found.extend(_ci_forbidden_paths(value, here))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            found.extend(_ci_forbidden_paths(value, f'{path}[{i}]'))
    return found


def _ci_same(a, b, path='result', *, rtol=1e-9, atol=1e-12, diffs=None):
    """Recursive comparison of a stored and replayed JSON body."""
    if diffs is None:
        diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            only_a = sorted(set(a) - set(b), key=_ci_key_text)
            only_b = sorted(set(b) - set(a), key=_ci_key_text)
            diffs.append(
                f'{path}: keys differ (stored=['
                + ', '.join(_ci_key_text(k) for k in only_a) + '], replay=['
                + ', '.join(_ci_key_text(k) for k in only_b) + '])')
            return diffs
        for key in sorted(a, key=_ci_key_text):
            _ci_same(a[key], b[key], f'{path}.{_ci_key_text(key)}',
                     rtol=rtol, atol=atol, diffs=diffs)
        return diffs
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f'{path}: length {len(a)} != {len(b)}')
            return diffs
        for i, (x, y) in enumerate(zip(a, b)):
            _ci_same(x, y, f'{path}[{i}]', rtol=rtol, atol=atol, diffs=diffs)
        return diffs
    if isinstance(a, bool) or isinstance(b, bool):
        if a is not b:
            diffs.append(f'{path}: {_ci_repr(a)} != {_ci_repr(b)}')
        return diffs
    if a is None or b is None:
        if a is not b:
            diffs.append(f'{path}: {_ci_repr(a)} != {_ci_repr(b)}')
        return diffs
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        try:
            if not math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol):
                diffs.append(f'{path}: {_ci_repr(a)} != {_ci_repr(b)}')
        except (TypeError, ValueError, OverflowError):
            diffs.append(f'{path}: uncomparable numbers '
                         f'{_ci_repr(a)}, {_ci_repr(b)}')
        return diffs
    try:
        same = bool(a == b)
    except (TypeError, ValueError, OverflowError):
        same = False
    if not same:
        diffs.append(f'{path}: {_ci_repr(a)} != {_ci_repr(b)}')
    return diffs


def compare_contrast_interaction_results(stored, replay, *,
                                         rtol=1e-9, atol=1e-12):
    """Compare two result bodies field by field; used by validation and gates."""
    if not isinstance(stored, dict) or not isinstance(replay, dict):
        return {'match': False,
                'differences': ['both results must be JSON objects']}
    diffs = _ci_same(stored, replay, 'result', rtol=rtol, atol=atol)
    return {'match': not diffs, 'differences': diffs}


def validate_contrast_interaction(artifact, *, replay=True):
    """Validate structure/hash and replay the result from ``inputs``."""
    out = {
        'kind': 'contrast_interaction_validation',
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
    strict = _ci_strict_json_problem('artifact', artifact)
    if strict:
        out['problems'].append(strict)
    extra = sorted(set(artifact) - set(CONTRAST_INTERACTION_ARTIFACT_KEYS),
                   key=_ci_key_text)
    missing = [k for k in CONTRAST_INTERACTION_ARTIFACT_KEYS
               if k not in artifact]
    if extra or missing:
        out['problems'].append(
            f'artifact keys differ from the schema: missing={missing}, '
            'extra=[' + ', '.join(_ci_key_text(k) for k in extra) + ']')
    if not _ci_json_equal(artifact.get('kind'),
                          CONTRAST_INTERACTION_ARTIFACT_KIND):
        out['problems'].append(
            f'artifact.kind must be {CONTRAST_INTERACTION_ARTIFACT_KIND!r}, '
            f'got {_ci_repr(artifact.get("kind"))}')
    if not _ci_json_equal(artifact.get('schema'),
                          CONTRAST_INTERACTION_ARTIFACT_SCHEMA):
        out['problems'].append(
            f'artifact.schema must be {CONTRAST_INTERACTION_ARTIFACT_SCHEMA}, '
            f'got {_ci_repr(artifact.get("schema"))}')
    forbidden = _ci_forbidden_paths(artifact)
    if forbidden:
        out['problems'].append(
            'artifact carries frozen-decision key(s): ' + ', '.join(forbidden))
    recorded = artifact.get('content_sha256')
    if not isinstance(recorded, str) or len(recorded) != 64:
        out['problems'].append('artifact.content_sha256 must be a 64-hex string')
    else:
        out['content_sha256'] = recorded
        try:
            actual = _ci_content_sha(artifact)
        except Exception as exc:
            out['problems'].append(
                f'content hash is not computable: {type(exc).__name__}: {exc}')
            actual = None
        if actual is not None and actual != recorded:
            out['problems'].append('artifact.content_sha256 does not match the '
                                   'recorded body')
    inputs = artifact.get('inputs')
    if not isinstance(inputs, dict):
        out['problems'].append('artifact.inputs must be an object')
    else:
        extra_i = sorted(set(inputs) - set(CONTRAST_INTERACTION_INPUT_KEYS),
                         key=_ci_key_text)
        missing_i = [k for k in CONTRAST_INTERACTION_INPUT_KEYS
                     if k not in inputs]
        if extra_i or missing_i:
            out['problems'].append(
                f'inputs keys differ: missing={missing_i}, extra=['
                + ', '.join(_ci_key_text(k) for k in extra_i) + ']')
    result = artifact.get('result')
    if not isinstance(result, dict):
        out['problems'].append('artifact.result must be an object')
    else:
        status = result.get('status')
        out['result_status'] = (status if isinstance(status, str)
                                else _ci_repr(status))
        out['problems'].extend(_ci_shape_problems(inputs, result))
    if out['problems']:
        return out
    if not replay:
        out['status'] = ('warnings' if result.get('status') == 'warnings'
                         else 'ok')
        out['result'] = _ci_jsonable(result)
        return out
    try:
        replayed = difference_in_differences(
            inputs['design'], inputs['cells'],
            alpha=inputs['alpha'], min_n=inputs['min_n'],
            margin=inputs['margin'],
            margin_relative_pct=inputs['margin_relative_pct'])
    except Exception as exc:
        out['problems'].append(
            f'replay failed: {type(exc).__name__}: {exc}')
        return out
    out['replayed'] = True
    cmp = compare_contrast_interaction_results(result, replayed)
    if not cmp['match']:
        out['problems'].append('replayed result differs from the stored result')
        out['warnings'].extend(cmp['differences'][:8])
        return out
    out['result'] = replayed
    if replayed.get('status') == 'not_estimable':
        out['status'] = 'not_estimable'
    elif replayed.get('status') == 'warnings':
        out['status'] = 'warnings'
        out['warnings'].extend(list(replayed.get('warnings') or []))
    else:
        out['status'] = 'ok'
    return out


def write_contrast_interaction(outdir, artifact, *, overwrite=False,
                               name='contrast_interaction.json'):
    """Write a validated artifact; same content is a no-op, different conflicts."""
    return _write_stamped_artifact(
        outdir, artifact, validator=validate_contrast_interaction,
        name=name, overwrite=overwrite)


def contrast_interaction_markdown(result, *, title='2x2 interaction'):
    """Render a compact Markdown block for a result body."""
    title = _ci_md_text(title, 240)
    if not isinstance(result, dict):
        return f'**{title}**: not estimable (not an object).'
    if result.get('status') == 'not_estimable':
        reason = _ci_md_text(result.get('reason', 'no reason'))
        return f'**{title}**: not estimable — {reason}.'
    design = result.get('design')
    design = design if isinstance(design, dict) else {}
    lines = [f'**{title}** (`{_ci_md_text(design.get("label", "2x2"))}`; '
             f'n={_ci_md_text(result.get("n"))})', '']
    scales = result.get('scales')
    scales = scales if isinstance(scales, dict) else {}
    primary = scales.get(result.get('primary_scale', 'log_ppl'))
    primary = primary if isinstance(primary, dict) else {}
    lines.append('| scale | contrast | mean | 95% CI | p (sign-flip) | '
                 'p (family) |')
    lines.append('|---|---|---|---|---|---|')
    for scale_name in (result.get('primary_scale'), 'ppl'):
        block = scales.get(scale_name)
        if not isinstance(block, dict):
            continue
        rows = list((block.get('marginals') or {}).values()) + [
            block.get('interaction') or {}]
        for row in rows:
            if not isinstance(row, dict):
                continue
            ci = row.get('ci95')
            ci_txt = ('[%s, %s]' % (_ci_md_num(ci[0], '{:+.4f}'),
                                    _ci_md_num(ci[1], '{:+.4f}'))
                      if isinstance(ci, list) and len(ci) == 2 else '—')
            lines.append('| %s | %s | %s | %s | %s | %s |' % (
                _ci_md_text(scale_name), _ci_md_text(row.get('name', '—')),
                _ci_md_num(row.get('mean'), '{:+.4f}'), ci_txt,
                _ci_md_num(row.get('p_exact_signflip'), '{:.4g}'),
                _ci_md_num(row.get('p_family'), '{:.4g}')))
    lines.append('')
    inter = primary.get('interaction') or {}
    lines.append('- primary interaction: `%s`; direction `%s`.' % (
        _ci_md_text(inter.get('definition', '—')),
        _ci_md_text(inter.get('direction', '—'))))
    fam = primary.get('family') or {}
    lines.append('- family: `%s`; simultaneous 95%% CI %s.' % (
        _ci_md_text(fam.get('method', '—')),
        _ci_md_text(inter.get('ci95_simultaneous'))
        if inter.get('ci95_simultaneous') else '—'))
    interpretation = result.get('interpretation')
    interpretation = interpretation if isinstance(interpretation, dict) else {}
    lines.append('- verdict: `%s`; claim allowed: `%s` (exploratory).' % (
        _ci_md_text(interpretation.get('classification')),
        _ci_md_text(interpretation.get('interaction_claim_allowed'))))
    if interpretation.get('difference_in_significance_fallacy'):
        lines.append('- **difference-in-significance warning**: one marginal is '
                     'significant and the other is not, but the interaction is '
                     'not rejected; do not read this as a significant '
                     'difference between the marginal effects.')
    agreement = result.get('scale_agreement')
    agreement = agreement if isinstance(agreement, dict) else {}
    if agreement.get('scale_sensitive'):
        lines.append('- **scale sensitivity**: the log-PPL and raw-PPL '
                     'interactions disagree; the choice of scale changes the '
                     'readout.')
    warnings = result.get('warnings')
    if isinstance(warnings, (list, tuple)):
        for warning in warnings:
            lines.append(f'- warning: {_ci_md_text(warning)}')
    lines.append('')
    lines.append('> exploratory only: this block is not in the frozen Holm '
                 'family and does not change any confirmed/verified verdict.')
    return '\n'.join(lines)


# ---- v12-specific wiring ---------------------------------------------------

CONFIRMATION_SINK_ARMS = {
    'csa_sink': 'csa_fixed_rope',
    'dense_sink': 'full_rope_sink',
    'csa_nosink': 'csa_fixed_rope_nosink',
    'dense_nosink': 'full_rope',
}

CONFIRMATION_SINK_2X2 = {
    'factors': ['architecture', 'sink'],
    'levels': {'architecture': ['csa', 'dense'], 'sink': ['sink', 'no_sink']},
    'cells': {
        'csa_sink': {'architecture': 'csa', 'sink': 'sink'},
        'dense_sink': {'architecture': 'dense', 'sink': 'sink'},
        'csa_nosink': {'architecture': 'csa', 'sink': 'no_sink'},
        'dense_nosink': {'architecture': 'dense', 'sink': 'no_sink'},
    },
    'contrast_factor': 'architecture',
    'label': 'sink x architecture (frozen v12 sensitivity design)',
}


def confirmation_sink_interaction(by_length, *, endpoint='by_len.ppl@4096',
                                  alpha=0.05, min_n=3, margin=None,
                                  margin_relative_pct=None):
    """Replay the frozen sink x architecture 2x2 from ``by_length`` cells.

    ``by_length`` is the ``test_stats.json`` mapping
    ``arm -> length -> {ppl_by_seed: {seed: PPL}}``.  It does not read the
    manifest or a split; it only re-describes cells that were already admitted
    to the frozen analysis.
    """
    try:
        if not isinstance(by_length, dict):
            raise ValueError('by_length must be a mapping')
        text = str(endpoint)
        if '@' not in text:
            raise ValueError(f'endpoint {_ci_endpoint_text(endpoint)!r} has '
                             'no @length suffix')
        length = _ci_int(text.rsplit('@', 1)[1], 'endpoint length', minimum=1)
        cells = {}
        reasons = {}
        for cell, arm in CONFIRMATION_SINK_ARMS.items():
            block = by_length.get(arm)
            if not isinstance(block, dict):
                reasons[cell] = f'arm {arm!r} has no by-length cells'
                continue
            cell_block = block.get(str(length), block.get(length))
            if not isinstance(cell_block, dict):
                reasons[cell] = f'arm {arm!r} has no L{length} cell'
                continue
            values = cell_block.get('ppl_by_seed')
            if not isinstance(values, dict) or not values:
                reasons[cell] = f'arm {arm!r} L{length} cell has no ppl_by_seed'
                continue
            cells[cell] = values
        if reasons:
            return _ci_not_estimable(
                'frozen 2x2 cells are incomplete: '
                + '; '.join(f'{k}: {v}' for k, v in sorted(reasons.items())),
                extra={'endpoint': _ci_endpoint_text(endpoint),
                       'alpha': _ci_float_or_none(alpha),
                       'min_n': _ci_int_or_none(min_n),
                       'margin': _ci_float_or_none(margin),
                       'margin_relative_pct': _ci_float_or_none(
                           margin_relative_pct),
                       'missing': reasons})
        result = difference_in_differences(
            CONFIRMATION_SINK_2X2, cells, alpha=alpha, min_n=min_n,
            margin=margin, margin_relative_pct=margin_relative_pct)
        result['endpoint'] = _ci_endpoint_text(endpoint)
        result['upstream'] = ('finaltest_stats: frozen by-length cells for '
                              'csa_fixed_rope / full_rope_sink / '
                              'csa_fixed_rope_nosink / full_rope')
        return result
    except Exception as exc:
        return _ci_not_estimable(
            f'{type(exc).__name__}: {exc}',
            extra={'endpoint': _ci_endpoint_text(endpoint),
                   'alpha': _ci_float_or_none(alpha),
                   'min_n': _ci_int_or_none(min_n),
                   'margin': _ci_float_or_none(margin),
                   'margin_relative_pct': _ci_float_or_none(
                       margin_relative_pct)})
