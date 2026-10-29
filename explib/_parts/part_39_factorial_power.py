# ---------------------------------------------------------------------------
# 2x2 interaction power / minimum-detectable-effect design
#
# The frozen confirmation family has a power plan (section 68) and the
# exploratory ablation panels have a SESOI + TOST decision rule (section 67),
# but neither answers a design question that must be settled BEFORE a
# factorial panel is launched: can the planned seed count detect an
# interaction of the size the study cares about?
#
# A main-effect power calculation is not an interaction power calculation.
# The interaction is a difference of differences on the log-PPL scale, so its
# seed-to-seed SD is generally larger than either main effect's, and the
# family-Holm correction makes the resolution floor stricter.  With a
# two-sided exact sign-flip test the smallest attainable p at n seeds is
# 2/2**n; with m interactions in the family the smallest attainable Holm p is
# m*2/2**n.  At n=6 and m=5 that floor is 0.15625, so no observed effect,
# however large, can reject at alpha=0.05.  A "non-significant interaction"
# from such a panel is a resolution failure, not evidence of additivity.
#
# These helpers are design-stage only.  They take pilot/validation cell values
# or an explicit seed count, never the held-out test split, and they do not
# change any measured PPL, p-value, Holm family or frozen decision.
# ---------------------------------------------------------------------------

FACTORIAL_POWER_CELL_KEYS = ('11', '10', '01', '00')

# Keys a metadata-only (or legacy bare) pilot may carry at the top level.
# They are not factorial names when their value is a scalar.
_FACTORIAL_PILOT_META_KEYS = frozenset({
    'm_family', 'n_seeds', 'kind', 'schema', 'source', 'generated_at', 'note',
})

FACTORIAL_POWER_POLICY = {
    'alpha': 0.05,
    'target_power': 0.80,
    'min_n': 3,
    'relative_pct': 1.0,
    'n_max': 512,
    'note': ('interaction power is a planning projection, not a post-hoc '
             'observed power; pilot deltas come from validation and are '
             'selection-prone'),
}


def _fp_int(name, value, *, minimum=None, maximum=None):
    """Strict non-boolean integer for a design parameter."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f'{name} must be an integer, got {type(value).__name__}')
    out = int(value)
    if minimum is not None and out < int(minimum):
        raise ValueError(f'{name} must be >= {int(minimum)}, got {out}')
    if maximum is not None and out > int(maximum):
        raise ValueError(f'{name} must be <= {int(maximum)}, got {out}')
    return out


def _fp_positive(name, value):
    """Strict finite positive PPL value (strings/bools are not measurements)."""
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray)):
        raise TypeError(f'{name} must be a finite positive number, got {type(value).__name__}')
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise TypeError(f'{name} must be a finite positive number, got {type(value).__name__}') from None
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f'{name} must be a finite positive number, got {value!r}')
    return out


def _fp_bool(name, value):
    """Strict boolean for a fail-closed design flag."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise TypeError(f'{name} must be a boolean, got {type(value).__name__}')


def _fp_probability(name, value, *, lo_open=False, hi_open=False):
    """Strict probability for a design parameter.

    ``_design_probability`` also accepts numeric strings because it is reused
    by legacy text parsers.  A JSON design object should carry real numbers;
    a quoted probability is a label, not a declared threshold, so the design
    API rejects it before it can silently become a valid plan.
    """
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray)):
        raise TypeError(
            f'{name} must be a real probability, got {type(value).__name__}')
    out = _design_probability(value, name=name,
                              lo_open=lo_open, hi_open=hi_open)
    if out is None:
        raise ValueError(f'{name}={value!r} must be a finite probability')
    return out


def _fp_cell_vectors(cells):
    """Validate the four 2x2 cell sequences and return aligned arrays.

    The keys are the usual ``11/10/01/00`` coding: first digit = first factor
    active, second digit = second factor active.  Every cell must have the
    same number of seed-aligned observations; a missing cell is not silently
    dropped into a complete-case interaction.
    """
    if not isinstance(cells, dict):
        raise TypeError(f'cells must be an object keyed by {FACTORIAL_POWER_CELL_KEYS}, got {type(cells).__name__}')
    normalised = {}
    for key, value in cells.items():
        if not isinstance(key, str):
            raise TypeError(f'cell key {key!r} is not a string')
        if key not in FACTORIAL_POWER_CELL_KEYS:
            raise ValueError(f'unknown cell key {key!r}; expected {FACTORIAL_POWER_CELL_KEYS}')
        if key in normalised:
            raise ValueError(f'duplicate cell key {key!r}')
        if (isinstance(value, (str, bytes, bytearray, dict, set, frozenset))
                or not hasattr(value, '__iter__')):
            raise TypeError(f'cells[{key!r}] must be an ordered sequence of PPL values')
        vals = []
        for i, x in enumerate(value):
            vals.append(_fp_positive(f'cells[{key!r}][{i}]', x))
        if len(vals) < 2:
            raise ValueError(f'cells[{key!r}] has {len(vals)} value(s); at least two paired seeds are required')
        normalised[key] = np.asarray(vals, dtype=float)
    missing = [k for k in FACTORIAL_POWER_CELL_KEYS if k not in normalised]
    if missing:
        raise ValueError(f'cells is missing {missing}; a 2x2 interaction needs all four cells')
    n = int(normalised[FACTORIAL_POWER_CELL_KEYS[0]].size)
    bad = {k: int(v.size) for k, v in normalised.items() if int(v.size) != n}
    if bad:
        raise ValueError(f'cell lengths differ (expected {n} seed-aligned values): {bad}')
    return normalised, n


def _min_n_for_holm_family(m_family, alpha, n_max=10000):
    """Smallest seed count whose two-sided sign-flip floor can clear Holm.

    This is the design resolution limit: below it no effect size can produce
    ``p_holm < alpha``, so adding a larger effect or a better selector cannot
    rescue the panel.  Only the seed count and the declared family size matter.
    """
    m = _fp_int('m_family', m_family, minimum=1)
    a = _design_probability(alpha, name='alpha', lo_open=True, hi_open=True)
    if a is None:
        raise ValueError(f'alpha={alpha!r} must be a finite probability in (0,1)')
    n_max = _fp_int('n_max', n_max, minimum=2, maximum=10000)
    for n in range(2, n_max + 1):
        if holm_min_attainable_p(n, m, a) <= a:
            return n
    return None


def _factorial_resolution(n, m_family, alpha):
    """Resolution-floor block shared by the single- and multi-factorial APIs."""
    floor = float(signflip_p_floor(n))
    min_holm = float(holm_min_attainable_p(n, m_family, alpha))
    # The resolution limit is a property of the design, not of the particular
    # n being audited.  Scan the full supported range (the same 10,000-seed
    # cap the rest of the planning helpers use) so a scale-up panel with
    # n > 512 and a small alpha still gets a real ``min_n_for_alpha`` instead
    # of a spurious ``None``.
    min_n = _min_n_for_holm_family(m_family, alpha, n_max=10000)
    if min_n is None:
        note = ('two-sided exact sign-flip floor with a Holm family of m; '
                'no seed count up to the 10,000-seed planning cap can make '
                'this family significant')
    else:
        note = ('two-sided exact sign-flip floor with a Holm family of m; '
                'below min_n_for_alpha no effect can be significant')
    return {
        'n_seeds': int(n),
        'm_family': int(m_family),
        'alpha': float(alpha),
        'p_floor': floor,
        'min_attainable_p_holm': min_holm,
        'resolution_ok': bool(min_holm <= float(alpha)),
        'min_n_for_alpha': min_n,
        'note': note,
    }


def factorial_interaction_power(cells, *, n_seeds=None, m_family=1,
                                alpha=0.05, target_power=0.80,
                                relative_pct=1.0, expected_sign=None,
                                require_sesoi_power=False, min_n=3,
                                n_max=512, pair_retention=None,
                                require_attrition_power=False):
    """Design-stage power for ONE 2x2 interaction from pilot cell PPLs.

    ``cells`` maps ``'11'/'10'/'01'/'00'`` to seed-aligned PPL sequences.
    The per-seed interaction is the same log ratio-of-ratios used by
    ``factorial_interaction_audit``::

        d_i = log(PPL_11 / PPL_10) - log(PPL_01 / PPL_00)

    The function first checks the exact sign-flip resolution floor for the
    declared family size.  If the floor cannot clear alpha it returns
    ``resolution_limited`` instead of a meaningless power curve.  Otherwise it
    calls ``paired_design_sensitivity`` with the SESOI margin
    ``log(1 + relative_pct/100)`` and reports power at that margin, the
    approximate MDE, the paired CI half-width and the required seed count.
    """
    out = {
        'kind': 'factorial_interaction_power', 'status': 'ok',
        'cells': list(FACTORIAL_POWER_CELL_KEYS), 'n_seeds': None,
        'm_family': None, 'alpha': None, 'target_power': None,
        'expected_sign': None, 'relative_pct': None, 'sesoi_margin': None,
        'interaction': {}, 'resolution': {}, 'sensitivity': {},
        'mde_relative_pct': None, 'attrition': {},
        'violations': [], 'warnings': [],
        'note': ('interaction power is a design projection on pilot '
                 'validation cells; it is not observed power and it does not '
                 'read the held-out test split'),
    }
    try:
        vectors, n_cells = _fp_cell_vectors(cells)
    except (TypeError, ValueError) as exc:
        out['status'] = 'fail'
        out['violations'].append(str(exc))
        return out
    out['n_seeds'] = n_cells
    # Validate every design parameter BEFORE the resolution early return.  A
    # malformed n/margin/flag must fail closed even when the panel happens to
    # be resolution-limited; otherwise the "no effect can be significant"
    # branch would silently accept an invalid plan.
    try:
        if n_seeds is not None:
            declared_n = _fp_int('n_seeds', n_seeds, minimum=2)
            if declared_n != n_cells:
                out['violations'].append(
                    f'n_seeds={declared_n} does not equal the {n_cells} aligned '
                    'pilot cells; the planned and pilot panels must be the same '
                    'seed-complete design')
        m = _fp_int('m_family', m_family, minimum=1)
        a = _fp_probability('alpha', alpha, lo_open=True, hi_open=True)
        tp = _fp_probability('target_power', target_power,
                             lo_open=True, hi_open=True)
        min_n = _fp_int('min_n', min_n, minimum=0)
        n_max = _fp_int('n_max', n_max, minimum=2, maximum=10000)
        pct = None
        margin = None
        if relative_pct is not None:
            pct = _fp_positive('relative_pct', relative_pct)
            margin = float(math.log1p(pct / 100.0))
        direction, direction_error = _design_direction(expected_sign)
        if direction_error:
            out['violations'].append(direction_error)
        retention = None
        if pair_retention is not None:
            # Match ``attrition_power_audit``: a retention probability is
            # defined on [0, 1].  A value of exactly 1.0 is the no-attrition
            # plan and is legitimate; values outside the interval are not.
            retention = _fp_probability('pair_retention', pair_retention)
        req_sesoi_power = _fp_bool(
            'require_sesoi_power', require_sesoi_power)
        req_attrition_power = _fp_bool(
            'require_attrition_power', require_attrition_power)
        if req_sesoi_power and relative_pct is None:
            out['violations'].append(
                'require_sesoi_power=true but no SESOI/relative_pct was '
                'declared')
        if req_attrition_power and pair_retention is None:
            out['violations'].append(
                'require_attrition_power=true but no pair_retention was '
                'declared')
    except (TypeError, ValueError) as exc:
        out['status'] = 'fail'
        out['violations'].append(str(exc))
        return out
    out['m_family'], out['alpha'], out['target_power'] = m, a, tp
    out['relative_pct'] = pct
    out['sesoi_margin'] = margin
    out['expected_sign'] = direction
    if n_cells < min_n:
        out['warnings'].append(
            f'n_seeds={n_cells} is below the declared min_n={min_n}; the '
            'interaction panel is too small to interpret even before power')
    v11 = vectors['11']
    v10 = vectors['10']
    v01 = vectors['01']
    v00 = vectors['00']
    deltas = (np.log(v11) - np.log(v10)) - (np.log(v01) - np.log(v00))
    out['interaction'] = {
        'deltas': [float(x) for x in deltas],
        'mean': float(np.mean(deltas)),
        'sd': float(np.std(deltas, ddof=1)),
        'n': int(deltas.size),
        'unit': 'natural-log ratio-of-ratios',
        'formula': 'log(PPL11/PPL10) - log(PPL01/PPL00)',
    }
    resolution = _factorial_resolution(n_cells, m, a)
    out['resolution'] = resolution
    if not resolution['resolution_ok']:
        # Input violations outrank the resolution readout: an invalid plan is
        # not merely under-resolved.
        out['status'] = 'fail' if out['violations'] else 'resolution_limited'
        need = resolution['min_n_for_alpha']
        need_txt = (f'at least {need} seeds' if need is not None
                    else 'more seeds than the 10,000-seed planning cap')
        out['note'] = (
            f'with n={n_cells} seeds and m={m} interactions the smallest '
            f'attainable Holm p is {resolution["min_attainable_p_holm"]:.6g} '
            f'> alpha={a}; no effect size can make a significant interaction. '
            f'This family needs {need_txt} '
            '(or a smaller pre-registered family).')
        out['warnings'] = list(dict.fromkeys(out['warnings']))
        return out
    if direction is None:
        out['warnings'].append(
            'no directional interaction hypothesis; a sign-flip SESOI power '
            'is undefined and only the resolution/MDE/CI readout is usable')
    base = paired_design_sensitivity(
        n_cells, m, alpha=a, target_power=tp, pilot_deltas=deltas,
        expected_sign=direction, sesoi=margin,
        require_sesoi_power=req_sesoi_power, n_max=n_max)
    out['sensitivity'] = base
    out['warnings'].extend(base.get('warnings') or [])
    if base.get('status') == 'fail':
        out['status'] = 'fail'
        out['violations'].extend(base.get('violations') or [])
    else:
        ses = base.get('sesoi') or {}
        if ses.get('status') == 'ok':
            out['status'] = 'ok'
        elif ses.get('status') == 'underpowered':
            out['status'] = 'underpowered'
            out['warnings'].append(
                f'power at the pre-registered {out["relative_pct"]}% '
                f'interaction SESOI is {ses.get("power_at_sesoi")} < '
                f'target_power={tp}; the planned n cannot resolve the margin')
        elif ses.get('status') == 'no_direction':
            out['status'] = 'not_verifiable'
        elif ses.get('status') == 'not_declared':
            out['status'] = 'not_verifiable'
            out['warnings'].append(
                'no interaction SESOI supplied; MDE/CI are reported but '
                'practical detectability is unverified')
        elif ses.get('status') == 'not_verifiable':
            out['status'] = 'not_verifiable'
        else:
            # A directional, resolvable panel with a properly declared margin
            # and enough pilot SD should always land in ok/underpowered; keep
            # an unknown branch visible rather than silently passing it.
            out['status'] = 'not_verifiable'
            out['warnings'].append(
                f'unknown SESOI status {ses.get("status")!r}')
    mde = (base.get('mde') or {}).get('absolute')
    if mde is not None and math.isfinite(float(mde)):
        try:
            out['mde_relative_pct'] = float(100.0 * math.expm1(abs(float(mde))))
        except OverflowError:
            out['mde_relative_pct'] = None
    if retention is not None:
        q = (base.get('pilot') or {}).get('q_sign_rate')
        if q is None:
            q = base.get('min_q_for_target_power')
        if q is None:
            out['attrition'] = {
                'status': 'not_verifiable',
                'reason': 'no directional sign rate is available for an attrition projection',
            }
            if req_attrition_power:
                out['violations'].append(
                    'require_attrition_power=true but no directional sign '
                    'rate is available for an attrition projection')
        else:
            att = attrition_power_audit(
                n_cells, m, q=q, alpha=a, target_power=tp,
                pair_retention=retention,
                require_complete_coverage=True,
                require_attrition_power=req_attrition_power)
            out['attrition'] = att
            if att.get('status') == 'fail':
                out['status'] = 'fail'
                out['violations'].extend(
                    f'attrition: {v}' for v in (att.get('violations') or []))
            elif att.get('status') not in ('ok', 'not_declared'):
                out['warnings'].append(
                    f'attrition status={att.get("status")!r}; the joint '
                    'run-and-reject probability is below the declared target')
    out['warnings'] = list(dict.fromkeys(out['warnings']))
    out['violations'] = list(dict.fromkeys(out['violations']))
    if out['violations']:
        out['status'] = 'fail'
    return out


def factorial_power_audit(pilot=None, *, design=None, sesoi=None,
                          n_seeds=None, m_family=None, alpha=0.05,
                          target_power=0.80, min_n=3, require_power=False,
                          require_sesoi_power=False, n_max=512,
                          pair_retention=None, require_attrition_power=False):
    """Audit the DESIGN-TIME interaction power of every declared 2x2.

    ``pilot`` is an optional artifact with a ``factorials`` mapping::

        {"factorials": {
            "dynamic_blocking_x_learned_selection": {
                "source": "validation L4096",
                "cells": {"11": [...], "10": [...], "01": [...], "00": [...]}}}}

    ``n_seeds`` may be a scalar or a factorial-name -> planned-n mapping.  It
    is used even without a pilot, so the resolution floor (the minimum n that
    can clear Holm) is checkable before any GPU run.  Missing pilot data is
    ``not_verifiable``, never a pass; an underpowered or resolution-limited
    panel is a warning unless ``require_power=True``.  The audit never invents
    an SD or an effect size.
    """
    out = {
        'kind': 'factorial_power_audit', 'status': 'ok',
        'm_family': None, 'alpha': None, 'target_power': None,
        'min_n': None, 'require_power': bool(require_power),
        'family_resolution': {}, 'factorials': {},
        'powered': [], 'underpowered': [], 'resolution_limited': [],
        'not_measured': [], 'violations': [], 'warnings': [],
        'note': ('design-stage interaction power; a missing pilot is '
                 'not_verifiable, not a pass. n below family_resolution.'
                 'min_n_for_alpha can never reject regardless of effect size'),
    }
    design = FACTORIAL_DESIGN if design is None else design
    sesoi = INTERACTION_SESOI if sesoi is None else sesoi
    if not isinstance(design, dict) or not design:
        out['status'] = 'fail'
        out['violations'].append('design must be a non-empty factorial declaration')
        return out
    if not isinstance(sesoi, dict):
        out['status'] = 'fail'
        out['violations'].append('sesoi must be an object mapping factorial -> margin')
        sesoi = {}
    # A pilot artifact may declare its own family size / planned n; explicit
    # arguments still win so a caller can audit an alternative design.
    if m_family is None and isinstance(pilot, dict) and pilot.get('m_family') is not None:
        m_family = pilot.get('m_family')
    if n_seeds is None and isinstance(pilot, dict) and pilot.get('n_seeds') is not None:
        n_seeds = pilot.get('n_seeds')
    try:
        m = (_fp_int('m_family', m_family, minimum=1)
             if m_family is not None else len(design))
        a = _fp_probability('alpha', alpha, lo_open=True, hi_open=True)
        tp = _fp_probability('target_power', target_power,
                             lo_open=True, hi_open=True)
        min_n = _fp_int('min_n', min_n, minimum=0)
        n_max = _fp_int('n_max', n_max, minimum=2, maximum=10000)
        req_power = _fp_bool('require_power', require_power)
        req_sesoi = _fp_bool('require_sesoi_power', require_sesoi_power)
        req_attr = _fp_bool('require_attrition_power',
                            require_attrition_power)
        retention = None
        if pair_retention is not None:
            retention = _fp_probability('pair_retention', pair_retention)
        if req_attr and retention is None:
            out['violations'].append(
                'require_attrition_power=true but no pair_retention was '
                'declared')
    except (TypeError, ValueError) as exc:
        out['status'] = 'fail'
        out['violations'].append(str(exc))
        return out
    out['m_family'], out['alpha'], out['target_power'] = m, a, tp
    out['min_n'] = min_n
    out['require_power'] = req_power
    out['require_sesoi_power'] = req_sesoi
    out['require_attrition_power'] = req_attr
    out['pair_retention'] = retention
    # ``n_max`` bounds the required-n search, not the resolution ceiling: the
    # ceiling is what makes a design impossible, so scan the full supported
    # range even when the caller only searches for a small required n.
    min_n_family = _min_n_for_holm_family(m, a, n_max=10000)
    if min_n_family is None:
        _fam_note = ('no seed count up to the 10,000-seed planning cap can '
                     'make this family significant, regardless of effect size')
    else:
        _fam_note = ('no interaction in this family can be significant below '
                     'min_n_for_alpha, regardless of effect size')
    out['family_resolution'] = {
        'm_family': m, 'alpha': a, 'min_n_for_alpha': min_n_family,
        'p_floor_rule': '2/2**n', 'holm_rule': 'm * 2/2**n',
        'note': _fam_note,
    }
    # ---- planned n (scalar or per-factorial) ---------------------------
    n_scalar, n_map = None, {}
    if n_seeds is not None:
        if isinstance(n_seeds, dict):
            for name, value in n_seeds.items():
                if not isinstance(name, str):
                    out['violations'].append(
                        f'n_seeds key {name!r} is not a string')
                    continue
                try:
                    n_map[name] = _fp_int(
                        f'n_seeds[{name!r}]', value, minimum=2)
                except (TypeError, ValueError) as exc:
                    out['violations'].append(str(exc))
            extra = sorted(set(n_map) - set(design), key=str)
            if extra:
                out['warnings'].append(
                    'n_seeds names factorial(s) not present in the design: '
                    + ', '.join(str(x) for x in extra))
        else:
            try:
                n_scalar = _fp_int('n_seeds', n_seeds, minimum=2)
            except (TypeError, ValueError) as exc:
                out['violations'].append(str(exc))
    # ---- pilot mapping --------------------------------------------------
    if pilot is None:
        p_map = {}
    elif not isinstance(pilot, dict):
        out['violations'].append(
            f'pilot must be an object, got {type(pilot).__name__}')
        p_map = {}
    elif 'factorials' in pilot:
        p_map = pilot.get('factorials')
        if not isinstance(p_map, dict):
            out['violations'].append('pilot.factorials must be an object')
            p_map = {}
    else:
        # A bare mapping is the legacy direct factorial map, but top-level
        # metadata such as {"n_seeds": 8, "m_family": 5} (or a planned-n
        # mapping) is not a factorial.  The declared factorial names are the
        # design keys, so metadata keys can be removed unconditionally here.
        p_map = {_k: _v for _k, _v in pilot.items()
                 if _k not in _FACTORIAL_PILOT_META_KEYS}
    # A direct Python caller can pass non-string keys even though JSON object
    # keys are strings.  Refuse them explicitly instead of letting a mixed-type
    # ``sorted()`` raise after some records have already been processed.
    p_map_clean = {}
    for _key, _value in p_map.items():
        if not isinstance(_key, str):
            out['violations'].append(
                f'pilot factorial key {_key!r} is not a string')
            continue
        p_map_clean[_key] = _value
    p_map = p_map_clean
    extra_p = sorted(set(p_map) - set(design), key=str)
    if extra_p:
        out['warnings'].append(
            'pilot contains factorial(s) not present in the design: '
            + ', '.join(str(x) for x in extra_p))
    for name, spec in design.items():
        rec = {'name': name, 'status': 'not_verifiable', 'n_seeds': None,
               'resolution': {}, 'power': None, 'warnings': []}
        # planned n: explicit per-factorial > pilot cell length > scalar
        planned = n_map.get(name, n_scalar)
        p_spec = p_map.get(name)
        if p_spec is not None and not isinstance(p_spec, dict):
            out['violations'].append(f'{name}: pilot entry is not an object')
            rec['status'] = 'fail'
            out['factorials'][name] = rec
            continue
        pilot_n = None
        if isinstance(p_spec, dict) and isinstance(p_spec.get('cells'), dict):
            cells = p_spec.get('cells')
            # Infer n from any cell length without trusting malformed values;
            # factorial_interaction_power re-validates all four lengths.
            for _v in cells.values():
                if isinstance(_v, (list, tuple, np.ndarray)):
                    pilot_n = len(_v)
                    break
        n_for_rec = pilot_n if pilot_n is not None else planned
        if planned is not None and pilot_n is not None and int(planned) != int(pilot_n):
            out['violations'].append(
                f'{name}: planned n_seeds={planned} != pilot cell n={pilot_n}')
        rec['n_seeds'] = int(n_for_rec) if isinstance(n_for_rec, int) else None
        if isinstance(n_for_rec, int) and n_for_rec >= 2:
            rec['resolution'] = _factorial_resolution(n_for_rec, m, a)
            if not rec['resolution']['resolution_ok']:
                rec['status'] = 'resolution_limited'
                _need = rec['resolution']['min_n_for_alpha']
                _need_txt = (f'at least {_need} seeds' if _need is not None
                             else 'more seeds than the 10,000-seed cap')
                rec['warnings'].append(
                    f'n={n_for_rec} with m={m} has minimum Holm p '
                    f'{rec["resolution"]["min_attainable_p_holm"]:.6g} > '
                    f'alpha={a}; this factorial family needs {_need_txt}')
            if n_for_rec < min_n:
                rec['warnings'].append(
                    f'n={n_for_rec} is below min_n={min_n}')
        elif n_for_rec is not None:
            out['violations'].append(
                f'{name}: planned n_seeds={n_for_rec!r} must be an integer >= 2')
            rec['status'] = 'fail'
        if p_spec is not None:
            if rec['status'] == 'fail':
                out['factorials'][name] = rec
                continue
            if not isinstance(p_spec.get('cells'), dict):
                out['violations'].append(
                    f'{name}: pilot entry has no cells object; cannot compute '
                    'an interaction power')
                rec['status'] = 'fail'
                out['factorials'][name] = rec
                continue
            rel = p_spec.get('relative_pct')
            rel_source = 'pilot'
            if rel is None:
                rel = (sesoi.get(name) or {}).get('relative_pct') if isinstance(
                    sesoi.get(name), dict) else None
                rel_source = 'declared_sesoi'
            if rel is None:
                rel = FACTORIAL_POWER_POLICY['relative_pct']
                rel_source = 'policy_default'
                rec['warnings'].append(
                    'no pre-registered interaction SESOI for this factorial; '
                    f'the policy default {rel}% is reported as a design aid, '
                    'not as a declared margin')
            rec['relative_pct_source'] = rel_source
            spec_req_sesoi = p_spec.get('require_sesoi_power', req_sesoi)
            spec_req_attr = p_spec.get('require_attrition_power', req_attr)
            if spec_req_sesoi and rel_source == 'policy_default':
                out['violations'].append(
                    f'{name}: require_sesoi_power=true but no pre-registered '
                    'SESOI is declared; the policy default is only a design aid')
            sign = p_spec.get('expected_sign')
            if sign is None:
                h = INTERACTION_HYPOTHESIS_DESIGN.get(name)
                sign = h.get('predicted_sign') if isinstance(h, dict) else None
            pw = factorial_interaction_power(
                p_spec['cells'],
                n_seeds=(n_for_rec if isinstance(n_for_rec, int) else None),
                m_family=m, alpha=a, target_power=tp, relative_pct=rel,
                expected_sign=sign,
                # Per-factorial flags are passed through unchanged: the
                # single-factorial API validates their type fail-closed
                # instead of coercing a string such as "no" to true.
                require_sesoi_power=spec_req_sesoi,
                min_n=min_n, n_max=n_max, pair_retention=retention,
                require_attrition_power=spec_req_attr)
            rec['power'] = pw
            _pw_n = pw.get('n_seeds')
            if (isinstance(planned, int) and isinstance(_pw_n, int)
                    and int(planned) != int(_pw_n)):
                _msg = (f'{name}: planned n_seeds={planned} != pilot cell '
                        f'n={_pw_n}')
                if _msg not in out['violations']:
                    out['violations'].append(_msg)
            rec['n_seeds'] = _pw_n if isinstance(_pw_n, int) else rec['n_seeds']
            rec['resolution'] = pw.get('resolution', rec['resolution'])
            rec['status'] = pw.get('status', 'not_verifiable')
            rec['warnings'].extend(pw.get('warnings') or [])
            out['violations'].extend(
                f'{name}: {v}' for v in (pw.get('violations') or []))
            if rec['status'] == 'ok':
                out['powered'].append(name)
            elif rec['status'] == 'underpowered':
                out['underpowered'].append(name)
            elif rec['status'] == 'resolution_limited':
                if name not in out['resolution_limited']:
                    out['resolution_limited'].append(name)
            elif rec['status'] == 'not_verifiable':
                out['not_measured'].append(name)
        else:
            if req_sesoi:
                out['violations'].append(
                    f'{name}: require_sesoi_power=true but no pilot cells are '
                    'available to compute SESOI power')
            if req_attr:
                out['violations'].append(
                    f'{name}: require_attrition_power=true but no pilot cells '
                    'are available to compute attrition power')
            if rec['status'] == 'resolution_limited':
                # A resolution-limited panel has no power curve to measure;
                # do not also report it as merely "power_not_measured".
                out['resolution_limited'].append(name)
            elif rec['resolution']:
                rec['status'] = 'power_not_measured'
                rec['warnings'].append(
                    'the panel is resolution-capable but no pilot cells are '
                    'available; interaction power/SESOI remain unverified')
                out['not_measured'].append(name)
            else:
                out['not_measured'].append(name)
        for w in rec['warnings']:
            out['warnings'].append(f'{name}: {w}')
        out['factorials'][name] = rec
    # ---- aggregate ------------------------------------------------------
    out['powered'] = list(dict.fromkeys(out['powered']))
    out['underpowered'] = list(dict.fromkeys(out['underpowered']))
    out['resolution_limited'] = list(dict.fromkeys(out['resolution_limited']))
    out['not_measured'] = list(dict.fromkeys(out['not_measured']))
    if out['violations']:
        out['status'] = 'fail'
    elif out['require_power']:
        # A design gate that requires power must not accept "no pilot" or an
        # underpowered/resolution-limited family as a pass.
        if out['underpowered'] or out['resolution_limited'] or out['not_measured']:
            out['status'] = 'fail'
            out['violations'].extend(
                ['require_power=true but factorial(s) %s are not powered' %
                 sorted(set(out['underpowered']) | set(out['resolution_limited'])
                        | set(out['not_measured']))])
    elif (not out['powered'] and not out['underpowered']
          and not out['resolution_limited']
          and all((r.get('status') == 'not_verifiable')
                  for r in out['factorials'].values())):
        # No pilot and no declared n is not a warning about a design; it is
        # the absence of a verifiable design readout.
        out['status'] = 'not_verifiable_locally'
    elif out['underpowered'] or out['resolution_limited'] or out['not_measured']:
        out['status'] = 'warnings'
    else:
        out['status'] = 'ok'
    if out['status'] == 'warnings':
        out['warnings'].append(
            'the interaction design is not yet fully powered; a '
            'non-significant result from such a panel cannot distinguish '
            'additivity from an undetected interaction of the declared SESOI')
    out['warnings'] = list(dict.fromkeys(out['warnings']))
    out['violations'] = list(dict.fromkeys(out['violations']))
    return out
