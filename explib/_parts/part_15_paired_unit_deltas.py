

def paired_unit_deltas(rows_a, rows_b, *, unit='auto'):
    """Per-unit paired PPL differences averaged over paired seeds.

    ``rows_a``/``rows_b`` are equal-length lists of by-length cells (for
    example ``record['by_len'][4096]``) in the same seed order.  Each cell may
    carry ``ppl_per_article``, ``ppl_per_doc`` or ``ppl_per_seq``.  The
    function selects the most independent available unit (article before
    document before row), checks that all seeds share the same unit count and
    identities, and returns the seed-averaged per-unit difference
    ``mean_seed(ppl_a[u] - ppl_b[u])``.

    The result is the pilot input for :func:`unit_precision_design`.  It is a
    paired design quantity, not a test statistic: averaging over seeds removes
    the seed-level pairing noise and leaves the evaluation-unit sampling
    distribution that the test-time document bootstrap must confront.
    """
    if unit not in ('auto', 'article', 'document', 'row'):
        raise ValueError(
            f"unit must be 'auto', 'article', 'document' or 'row', got {unit!r}")
    try:
        rows_a = list(rows_a)
        rows_b = list(rows_b)
    except TypeError:
        raise ValueError('rows_a and rows_b must be sequences of by-length cells') from None
    if len(rows_a) != len(rows_b):
        return {'status': 'not_available', 'reason': (
            f'seed panels differ in length: {len(rows_a)} vs {len(rows_b)}')}
    if len(rows_a) < 1:
        return {'status': 'not_available', 'reason': 'no paired seed cells'}
    if not all(isinstance(r, dict) for r in rows_a + rows_b):
        raise ValueError('rows_a and rows_b must contain by-length cell objects')
    candidates = [c for c in _UNIT_GRANULARITIES if unit == 'auto' or c[0] == unit]
    reasons = {}
    for label, key, id_key in candidates:
        vectors = [r.get(key) for r in rows_a + rows_b]
        if not all(_unit_sequence(v) for v in vectors):
            reasons[label] = f'{key} is absent, empty or not a list in every seed cell'
            continue
        lengths = {len(v) for v in vectors}
        if len(lengths) != 1:
            reasons[label] = f'{key} has inconsistent unit counts across seeds: {sorted(lengths)}'
            continue
        n_units = lengths.pop()
        if n_units < 2:
            reasons[label] = f'{key} has fewer than two units ({n_units})'
            continue
        ids = None
        if id_key is not None:
            id_vectors = [r.get(id_key) for r in rows_a + rows_b]
            if all(_unit_sequence(v) and len(v) == n_units
                   for v in id_vectors):
                first = (id_vectors[0].tolist()
                         if isinstance(id_vectors[0], np.ndarray)
                         else list(id_vectors[0]))
                if any(list(v) != first for v in id_vectors[1:]):
                    reasons[label] = f'{id_key} order/identities differ across seeds'
                    continue
                ids = first
            else:
                reasons[label] = (
                    f'{id_key} is absent or has the wrong length; unit order '
                    'cannot be verified across seeds')
                continue
        values = []
        usable = True
        for u in range(n_units):
            diffs = []
            for ra, rb in zip(rows_a, rows_b):
                va, vb = _finite_scalar(ra[key][u]), _finite_scalar(rb[key][u])
                if (va is None or vb is None
                        or not ppl_is_usable(va) or not ppl_is_usable(vb)):
                    usable = False
                    break
                diff = va - vb
                if not math.isfinite(diff):
                    usable = False
                    break
                diffs.append(diff)
            if not usable:
                break
            values.append(float(np.mean(diffs)))
        mean = float(np.mean(values)) if values else None
        sd = float(np.std(values, ddof=1)) if len(values) >= 2 else None
        if (not usable or len(values) < 2 or mean is None or sd is None
                or not (math.isfinite(mean) and math.isfinite(sd))):
            reasons[label] = f'{key} contains non-finite/sub-1 PPL values'
            continue
        return {
            'status': 'ok', 'unit': label, 'unit_label': label,
            'key': key, 'n_units': int(len(values)), 'n_seeds': len(rows_a),
            'deltas': values, 'ids': ids,
            'id_key': id_key,
            'mean': mean,
            'sd': sd,
            'source': (f'{key} averaged within unit over {len(rows_a)} paired seeds'),
        }
    return {'status': 'not_available', 'unit': None,
            'reason': '; '.join(f'{k}: {v}' for k, v in reasons.items()),
            'candidates': reasons}


def _unit_power_at(n, sd, margin, alpha):
    """Two-sided normal-approximation power at a unit-level mean margin."""
    try:
        n = float(n)
        sd = float(sd)
        margin = float(margin)
        alpha = float(alpha)
    except (TypeError, ValueError, OverflowError):
        return None
    if not (math.isfinite(n) and n > 1.0):
        return None
    if not (math.isfinite(sd) and sd > 0.0):
        return None
    if not (math.isfinite(margin) and 0.0 < alpha < 1.0):
        return None
    se = sd / math.sqrt(n)
    if not (math.isfinite(se) and se > 0.0):
        return None
    tc = _t_critical(1.0 - alpha, n - 1.0)
    if not math.isfinite(tc):
        return None
    z = margin / se
    return float(min(1.0, max(0.0, _normal_cdf(z - tc) + _normal_cdf(-z - tc))))


def _unit_needed_n(n0, *, sd, alpha, target, kind, n_max,
                   target_power=None):
    """Smallest n in ``[2, n_max]`` meeting the t-based criterion.

    ``n0`` is the normal-approximation hint retained for compatibility.  The
    hint can be above ``n_max`` even when a smaller valid n exists, or below
    the true minimum; scanning only a short window around it therefore missed
    feasible designs and could report a non-minimal n.  Both criteria are
    monotone in n, so a binary search over the whole feasible range returns
    the minimal integer without ever enumerating up to ``n_max``.
    """
    _ = n0
    try:
        hi = int(n_max)
        sd = float(sd)
        alpha = float(alpha)
        target = float(target)
    except (TypeError, ValueError, OverflowError):
        return None
    if hi < 2 or not (math.isfinite(sd) and sd > 0.0):
        return None
    if not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
        return None
    if not (math.isfinite(target) and target > 0.0):
        return None
    if kind == 'power':
        try:
            tp = float(target_power)
        except (TypeError, ValueError, OverflowError):
            return None
        if not (math.isfinite(tp) and 0.0 < tp < 1.0):
            return None

        def _ok(n):
            power = _unit_power_at(n, sd, target, alpha)
            return power is not None and power + 1e-15 >= tp
    else:
        def _ok(n):
            tc = _t_critical(1.0 - alpha, float(n) - 1.0)
            return (math.isfinite(tc)
                    and tc * sd / math.sqrt(float(n)) <= target)
    if _ok(2):
        return 2
    if not _ok(hi):
        return None
    lo = 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _ok(mid):
            hi = mid
        else:
            lo = mid
    return int(hi)


def unit_precision_design(n_units, *, pilot_unit_deltas=None,
                          pilot_unit_sd=None, unit_label='document',
                          sesoi=None, target_power=0.80, alpha=0.05,
                          target_halfwidth=None, require_sesoi_power=False,
                          min_units_for_population=MIN_DOCS_FOR_POPULATION_CLAIM,
                          n_max=100000):
    """Design-stage precision for an evaluation-unit paired contrast.

    ``paired_design_sensitivity`` plans the SEED panel: it asks how often the
    paired sign pattern would reject.  This function plans the separate
    evaluation-unit panel at test time: given the frozen number of documents
    (or articles), what mean PPL difference can the design resolve, and what
    population claim is justified?

    The pilot vector is one paired difference per evaluation unit, usually
    the seed-averaged ``ppl_a - ppl_b`` obtained from
    :func:`paired_unit_deltas`.  The pilot units come from validation; the
    planned ``n_units`` is the frozen test-unit count, so the two counts need
    not match.  The output is an approximate two-sided
    one-sample-t precision statement (normal power approximation with a t
    critical value), not a substitute for the exact sign-flip test or the
    document bootstrap.  A population claim additionally requires the frozen
    unit count to reach ``min_units_for_population`` and a non-row unit.
    """
    out = {'kind': 'unit_precision_design', 'status': 'ok',
           'n_units': None, 'unit_label': None, 'alpha': None,
           'target_power': None, 'pilot': {}, 'precision': {},
           'mde': {}, 'sesoi': {}, 'population': {}, 'violations': [],
           'warnings': [],
           'policy': {
               'estimand': ('paired mean PPL difference over independent '
                            'evaluation units, conditional on the frozen '
                            'trained seeds'),
               'unit': ('article is the corpus sampling unit; document is the '
                        'real document boundary; row is a fixed evaluation '
                        'window and cannot support population inference'),
               'power_rule': ('two-sided one-sample-t normal approximation '
                              'with the actual t critical value; the exact '
                              'sign-flip test and unit bootstrap remain the '
                              'analysis-time procedures'),
               'population_rule': ('population claims require at least '
                                   f'{MIN_DOCS_FOR_POPULATION_CLAIM} independent '
                                   'units and a non-row unit label; otherwise '
                                   'claims are limited to the frozen slice'),
               'sesoi_rule': ('a declared SESOI is failed only when '
                              'require_sesoi_power is true, matching the '
                              'seed-level sensitivity policy'),
           }}
    n = _design_int(n_units)
    out['n_units'] = n
    if n is None or n < 2:
        out['violations'].append(
            f'n_units={n_units!r} must be an integer >= 2')
        n = None
    label = str(unit_label).strip().lower() if isinstance(
        unit_label, str) else unit_label
    if label not in ('article', 'document', 'row'):
        out['violations'].append(
            f"unit_label={unit_label!r} must be 'article', 'document' or 'row'")
        label = None
    out['unit_label'] = label
    a = _design_probability(alpha, name='alpha', lo_open=True, hi_open=True)
    if a is None:
        out['violations'].append(
            f'alpha={alpha!r} must be a finite probability in (0,1)')
    out['alpha'] = a
    tp = _design_probability(target_power, name='target_power',
                             lo_open=True, hi_open=True)
    if tp is None:
        out['violations'].append(
            f'target_power={target_power!r} must be a finite probability in (0,1)')
    out['target_power'] = tp
    if not isinstance(require_sesoi_power, (bool, np.bool_)):
        out['violations'].append(
            f'require_sesoi_power={require_sesoi_power!r} must be a boolean')
        require_sesoi_power = False
    require_sesoi_power = bool(require_sesoi_power)
    out['require_sesoi_power'] = require_sesoi_power
    mu = _design_int(min_units_for_population)
    if mu is None or mu < 2 or mu > 1000000:
        out['violations'].append(
            f'min_units_for_population={min_units_for_population!r} must be '
            'an integer in [2, 1000000]')
        mu = MIN_DOCS_FOR_POPULATION_CLAIM
    n_max_i = _design_int(n_max)
    if n_max_i is None or n_max_i < 2 or n_max_i > 1000000:
        out['violations'].append(
            f'n_max={n_max!r} must be an integer in [2, 1000000]')
        n_max_i = 100000
    out['policy']['population_rule'] = (
        f'population claims require at least {mu} independent units and a '
        'non-row unit label; otherwise claims are limited to the frozen slice')
    raw = []
    dropped = 0
    if isinstance(pilot_unit_deltas, (str, bytes)):
        out['violations'].append(
            'pilot_unit_deltas must be a sequence of numbers, not a string')
        pilot_unit_deltas = None
    if pilot_unit_deltas is not None:
        try:
            for value in pilot_unit_deltas:
                f = _finite_scalar(value)
                if f is None:
                    dropped += 1
                else:
                    raw.append(f)
        except TypeError:
            out['violations'].append(
                'pilot_unit_deltas must be an iterable of finite numbers, '
                f'got {type(pilot_unit_deltas).__name__}')
            raw = []
    d = np.asarray(raw, dtype=float)
    explicit_sd = None
    if pilot_unit_sd is not None:
        explicit_sd = _finite_scalar(pilot_unit_sd)
        if explicit_sd is None or explicit_sd <= 0.0:
            out['violations'].append(
                f'pilot_unit_sd={pilot_unit_sd!r} must be a finite positive number')
            explicit_sd = None
    sd = None
    if explicit_sd is not None:
        sd = float(explicit_sd)
        if d.size >= 2:
            calculated = float(d.std(ddof=1))
            if abs(calculated - sd) > 1e-12 * max(1.0, abs(sd)):
                out['warnings'].append(
                    'pilot_unit_sd overrides the SD recomputed from '
                    'pilot_unit_deltas; report which sample each value came from')
    elif d.size >= 2:
        sd = float(d.std(ddof=1))
        if not (math.isfinite(sd) and sd > 0.0):
            out['warnings'].append(
                'pilot unit deltas have zero/non-finite SD; MDE and SESOI '
                'power are not computable')
            sd = None
    elif d.size == 1:
        out['warnings'].append(
            'only one finite pilot unit delta; a paired SD needs at least two')
    if dropped:
        out['warnings'].append(
            f'{dropped} non-finite/non-numeric pilot unit delta(s) were dropped')
    out['pilot'] = {
        'status': ('not_provided' if (not d.size and explicit_sd is None)
                   else 'available'),
        'n': int(d.size), 'n_dropped': int(dropped),
        'mean': float(d.mean()) if d.size else None, 'sd': sd,
        'source': ('deltas' if d.size else ('pilot_unit_sd' if sd is not None else None)),
    }
    target_halfwidth_value = None
    if target_halfwidth is not None:
        target_halfwidth_value = _finite_scalar(target_halfwidth)
        if target_halfwidth_value is None or target_halfwidth_value <= 0.0:
            out['violations'].append(
                f'target_halfwidth={target_halfwidth!r} must be a finite '
                'positive number')
            target_halfwidth_value = None
    out['target_halfwidth'] = target_halfwidth_value
    half = None
    if sd is not None and a is not None and n is not None:
        tc = _t_critical(1.0 - a, float(n) - 1.0)
        if math.isfinite(tc):
            half = float(tc * sd / math.sqrt(n))
        out['precision'] = {
            'confidence': 1.0 - a,
            't_critical': float(tc) if math.isfinite(tc) else None,
            'ci_halfwidth': half, 'ci95_halfwidth': half,
            'target_halfwidth': target_halfwidth_value,
            'meets_target': (None if target_halfwidth_value is None or half is None
                             else bool(half <= target_halfwidth_value)),
            'n_required_for_target_halfwidth': None,
            'method': 'paired t interval at the frozen unit count',
        }
        if target_halfwidth_value is not None:
            out['precision']['n_required_for_target_halfwidth'] = _unit_needed_n(
                2, sd=sd, alpha=a, target=target_halfwidth_value,
                kind='halfwidth', n_max=n_max_i)
            if out['precision']['meets_target'] is False:
                out['warnings'].append(
                    f'the expected paired CI half-width at n={n} '
                    f'({half:.6g}) exceeds target_halfwidth='
                    f'{target_halfwidth_value!r}')
    if sd is not None and tp is not None and a is not None and n is not None:
        ze = float(_normal_quantile(tp))
        tc = _t_critical(1.0 - a, float(n) - 1.0)
        mde = ((tc + ze) * sd / math.sqrt(n)) if math.isfinite(tc) else None
        out['mde'] = {
            'absolute': float(mde) if mde is not None else None,
            'relative_to_pilot_sd': (float(mde) / sd if mde is not None else None),
            'method': ('normal power approximation with the t critical value: '
                       '(t_crit + Phi^-1(target_power)) * sd / sqrt(n)'),
            'multiple_of_sesoi': None,
        }
    sesoi_value = None
    if sesoi is not None:
        sesoi_value = _finite_scalar(sesoi)
        if sesoi_value is None or sesoi_value <= 0.0:
            out['violations'].append(
                f'sesoi={sesoi!r} must be a finite positive margin in the same '
                'units as the paired unit deltas')
            sesoi_value = None
    out['sesoi'] = {
        'margin': sesoi_value, 'status': 'not_declared',
        'power_at_sesoi': None, 'required_n_for_target_power': None,
        'power_ok': None,
        'note': ('no pre-registered SESOI supplied; statistical detectability '
                 'does not establish practical relevance'),
    }
    if require_sesoi_power and sesoi is None:
        out['violations'].append(
            'require_sesoi_power=true but no SESOI margin was declared')
    if sesoi_value is not None:
        if sd is None:
            out['sesoi'].update({
                'status': 'not_verifiable',
                'note': 'SESOI declared but no pilot unit SD is available'})
            if require_sesoi_power:
                out['violations'].append(
                    'require_sesoi_power=true but the pilot unit SD is missing')
        elif tp is not None and a is not None and n is not None:
            power = _unit_power_at(n, sd, sesoi_value, a)
            ok = bool(power is not None and power + 1e-15 >= tp)
            needed = _unit_needed_n(
                2, sd=sd, alpha=a, target=sesoi_value, kind='power',
                n_max=n_max_i, target_power=tp)
            out['sesoi'].update({
                'status': 'ok' if ok else 'underpowered',
                'power_at_sesoi': power,
                'required_n_for_target_power': needed,
                'power_ok': ok,
                'note': ('the planned unit count meets the target power at the '
                         'pre-registered margin' if ok else
                         'the planned unit count cannot reach target power at '
                         'the pre-registered margin'),
            })
            if not ok:
                out['warnings'].append(
                    f'the pre-registered SESOI has approximate unit power '
                    f'{power if power is not None else "n/a"} < '
                    f'target_power={tp}')
                if require_sesoi_power:
                    out['violations'].append(
                        'require_sesoi_power=true and the planned unit count is '
                        'underpowered at the pre-registered SESOI')
    if out['mde'].get('absolute') is not None and sesoi_value is not None:
        out['mde']['multiple_of_sesoi'] = float(
            out['mde']['absolute'] / sesoi_value)
    claim_allowed = False
    claim_note = None
    if n is not None and label is not None:
        if label == 'row':
            claim_note = ('row is an evaluation window, not an independent '
                          'corpus unit; population claims are refused')
        elif n < mu:
            claim_note = (f'n_units={n} < min_units_for_population={mu}; '
                          'claims are limited to the frozen evaluation slice')
        else:
            claim_allowed = True
            claim_note = (f'n_units={n} >= {mu} and unit={label}; a '
                          'population claim is admissible only with a '
                          'declared probability sampling design')
        if not claim_allowed:
            out['warnings'].append(claim_note)
    out['population'] = {
        'n_units': n, 'min_units_for_population': mu,
        'unit_label': label, 'claim_allowed': claim_allowed,
        'note': claim_note,
    }
    if not out['pilot'].get('n') and pilot_unit_sd is None:
        out['warnings'].append(
            'no pilot unit deltas or SD supplied; unit MDE/SESOI power cannot '
            'be computed')
    if out['violations']:
        out['status'] = 'fail'
    elif sd is None:
        out['status'] = 'not_verifiable'
    return out


def evaluation_design_units(manifest):
    """Frozen test-slice unit count and label declared by a manifest.

    ``evaluation_design.n_units`` is authoritative; legacy manifests fall back
    to ``eval.n_seqs_clean`` and then ``binding.test_slice.n_seqs``.  The raw
    value is returned alongside the coerced integer so callers can let
    :func:`unit_precision_design` reject a malformed declaration with the
    original value in the violation message.
    """
    out = {'n_units': None, 'raw_n_units': None, 'unit_label': 'document',
           'label_declared': False}
    if not isinstance(manifest, dict):
        return out
    design = manifest.get('evaluation_design')
    if not isinstance(design, dict):
        design = manifest.get('evaluation')
    if not isinstance(design, dict):
        design = {}
    raw_n = design.get('n_units')
    if raw_n is None:
        raw_n = design.get('unit_count')
    if raw_n is None:
        binding = manifest.get('binding') if isinstance(
            manifest.get('binding'), dict) else {}
        test_slice = binding.get('test_slice') if isinstance(
            binding.get('test_slice'), dict) else {}
        eval_blk = manifest.get('eval') if isinstance(
            manifest.get('eval'), dict) else {}
        raw_n = eval_blk.get('n_seqs_clean')
        if raw_n is None:
            raw_n = test_slice.get('n_seqs')
    label = design.get('unit')
    if label is None:
        label = design.get('unit_label')
    if label is None:
        label = 'document'
    out['unit_label'] = label.strip().lower() if isinstance(
        label, str) else label
    out['label_declared'] = ('unit' in design) or ('unit_label' in design)
    out['raw_n_units'] = raw_n
    out['n_units'] = _design_int(raw_n)
    return out


def evaluation_design_audit(manifest, pilot=None):
    """Audit the evaluation-unit panel declared by a frozen manifest.

    Seed-level power answers whether the training replication can reject.
    This audit asks the orthogonal preparation questions: how many independent
    test units exist, whether that count can support a population claim, and
    (when ``python -m explib.confirmation power`` wrote per-unit deltas) what mean PPL difference
    the frozen slice can resolve.

    ``pilot`` is the parsed ``power_validation.json`` object.  A missing pilot
    is reported as ``not_verifiable`` with a warning, never as a pass.  No
    test data is read.
    """
    out = {'kind': 'evaluation_design_audit', 'status': 'ok',
           'n_units': None, 'unit_label': None, 'population': {},
           'sampling': {}, 'comparisons': {}, 'violations': [],
           'warnings': []}
    if not isinstance(manifest, dict):
        out['status'] = 'fail'
        out['violations'].append(
            f'manifest must be an object, got {type(manifest).__name__}')
        return out
    binding = manifest.get('binding') if isinstance(
        manifest.get('binding'), dict) else {}
    test_slice = binding.get('test_slice') if isinstance(
        binding.get('test_slice'), dict) else {}
    design = manifest.get('evaluation_design')
    if not isinstance(design, dict):
        design = manifest.get('evaluation') if isinstance(
            manifest.get('evaluation'), dict) else {}
    if not isinstance(design, dict):
        design = {}
    stat = manifest.get('statistics') if isinstance(
        manifest.get('statistics'), dict) else {}
    _units = evaluation_design_units(manifest)
    raw_n = _units['raw_n_units']
    unit_label = _units['unit_label']
    out['n_units'] = _units['n_units']
    out['unit_label'] = unit_label
    alpha = stat.get('alpha', 0.05)
    target_power = stat.get('min_power_for_test', 0.80)
    min_units = design.get('min_units_for_population')
    if min_units is None:
        min_units = stat.get(
            'min_units_for_population', MIN_DOCS_FOR_POPULATION_CLAIM)
    sampling = design.get('test_slice_sampling')
    if sampling is None:
        sampling = test_slice.get('sampling')
    method = None
    if isinstance(sampling, dict):
        method = sampling.get('method', sampling.get('design'))
    elif isinstance(sampling, str):
        method = sampling
    method = str(method).strip().lower() if isinstance(method, str) else None
    out['sampling'] = {
        'declared': sampling is not None, 'method': method,
        'probability_sample': None,
        'note': ('no machine-readable sampling declaration in the manifest; '
                 'the prefix/first-n loader cannot establish a probability '
                 'sample for population inference' if sampling is None else
                 'declared sampling design is not independently replayed here'),
    }
    if method in ('prefix', 'first_n', 'first-n', 'source_order'):
        out['sampling']['probability_sample'] = False
        out['warnings'].append(
            f'test-slice sampling method {method!r} is a deterministic '
            'prefix/first-n rule, not a probability sample; population claims '
            'are refused')
    elif method in ('probability', 'simple_random', 'srs', 'random',
                    'stratified_document', 'document_stratified',
                    'stratified_article', 'article_stratified'):
        out['sampling']['probability_sample'] = True
    supports_population = design.get('supports_population_claim')
    if supports_population is None:
        supports_population = design.get('population_claim')
    if supports_population is None:
        supports_population = stat.get('supports_population_claim', False)
    if not isinstance(supports_population, (bool, np.bool_)):
        out['violations'].append(
            'supports_population_claim must be a boolean when declared')
        supports_population = False
    supports_population = bool(supports_population)
    comps = manifest.get('primary_comparisons')
    comps = comps if isinstance(comps, list) else []
    pilot_comps = pilot.get('comparisons') if isinstance(
        pilot, dict) and isinstance(pilot.get('comparisons'), dict) else {}
    def _pilot_deltas(vals):
        if isinstance(vals, np.ndarray):
            return vals.tolist() if vals.ndim == 1 else None
        if isinstance(vals, (list, tuple)):
            return list(vals)
        return None

    def _has_pilot_delta(vals):
        return isinstance(vals, list) and any(
            _finite_scalar(v) is not None for v in vals)

    pilot_cells = 0
    if comps:
        for comp in comps:
            if not isinstance(comp, dict):
                out['violations'].append(
                    f'primary comparison is not an object: {comp!r}')
                continue
            name = str(comp.get('name') or f'comparison_{len(out["comparisons"])}')
            if name in out['comparisons']:
                out['violations'].append(
                    f'duplicate primary-comparison name {name!r}')
                continue
            cell = pilot_comps.get(name) if isinstance(pilot_comps, dict) else None
            unit_deltas = None
            if isinstance(cell, dict):
                unit_deltas = _pilot_deltas(cell.get('unit_deltas'))
                if _has_pilot_delta(unit_deltas):
                    pilot_cells += 1
            sesoi = comp.get('sesoi')
            if sesoi is None:
                sesoi = stat.get('sesoi')
            require_sesoi = (comp['require_sesoi_power']
                             if 'require_sesoi_power' in comp
                             else stat.get('require_sesoi_power', False))
            rep = unit_precision_design(
                raw_n, pilot_unit_deltas=unit_deltas, unit_label=unit_label,
                sesoi=sesoi, target_power=target_power, alpha=alpha,
                require_sesoi_power=require_sesoi,
                min_units_for_population=min_units)
            out['comparisons'][name] = rep
            out['violations'].extend(
                f'{name}: {v}' for v in (rep.get('violations') or []))
            out['warnings'].extend(
                f'{name}: {w}' for w in (rep.get('warnings') or []))
            if rep.get('status') == 'fail':
                out['status'] = 'fail'
            elif (rep.get('status') == 'not_verifiable'
                  and out['status'] == 'ok'):
                out['status'] = 'not_verifiable'
    else:
        unit_deltas = _pilot_deltas(design.get('pilot_unit_deltas'))
        if unit_deltas is None and isinstance(pilot, dict):
            unit_deltas = _pilot_deltas(pilot.get('unit_deltas'))
        if _has_pilot_delta(unit_deltas):
            pilot_cells = 1
        rep = unit_precision_design(
            raw_n, pilot_unit_deltas=unit_deltas, unit_label=unit_label,
            sesoi=stat.get('sesoi'), target_power=target_power, alpha=alpha,
            require_sesoi_power=stat.get('require_sesoi_power', False),
            min_units_for_population=min_units)
        out['comparisons']['all'] = rep
        out['violations'].extend(rep.get('violations') or [])
        out['warnings'].extend(rep.get('warnings') or [])
        if rep.get('status') == 'fail':
            out['status'] = 'fail'
        elif rep.get('status') == 'not_verifiable' and out['status'] == 'ok':
            out['status'] = 'not_verifiable'
    out['population'] = {
        'n_units': out['n_units'], 'unit_label': out['unit_label'],
        'min_units_for_population': min_units,
        'supports_population_claim_declared': supports_population,
        'claim_allowed': None,
        'note': None,
    }
    reps = [r for r in out['comparisons'].values() if isinstance(r, dict)]
    if reps:
        pop = reps[0].get('population') or {}
        allowed = bool(pop.get('claim_allowed'))
        out['population']['claim_allowed'] = allowed
        out['population']['note'] = pop.get('note')
        if supports_population and not allowed:
            out['violations'].append(
                'manifest declares supports_population_claim=true but the '
                f'frozen design cannot support it ({pop.get("note")})')
            out['status'] = 'fail'
        if supports_population and out['sampling'].get('probability_sample') is False:
            out['violations'].append(
                'manifest declares a population claim but the test slice is a '
                'deterministic prefix/first-n sample')
            out['status'] = 'fail'
        elif supports_population and out['sampling'].get('probability_sample') is None:
            out['violations'].append(
                'manifest declares a population claim but no machine-readable '
                'probability sampling design is declared')
            out['status'] = 'fail'
    elif supports_population:
        _n = out['n_units']
        _label = out['unit_label']
        _mu_check = _design_int(min_units)
        if (_n is None or _label == 'row' or _mu_check is None
                or _mu_check < 2 or _mu_check > 1000000
                or _n < _mu_check):
            out['violations'].append(
                'manifest declares supports_population_claim=true but the '
                f'frozen design (n_units={_n!r}, unit={_label!r}, floor='
                f'{min_units}) cannot support it')
            out['status'] = 'fail'
        if out['sampling'].get('probability_sample') is False:
            out['violations'].append(
                'manifest declares a population claim but the test slice is a '
                'deterministic prefix/first-n sample')
            out['status'] = 'fail'
        elif out['sampling'].get('probability_sample') is None:
            out['violations'].append(
                'manifest declares a population claim but no machine-readable '
                'probability sampling design is declared')
            out['status'] = 'fail'
    if pilot_cells == 0:
        if out['status'] == 'ok':
            out['status'] = 'not_verifiable'
        out['warnings'].append(
            'no per-unit pilot deltas found in the power artifact; unit MDE and '
            'SESOI precision are not verifiable (run `python -m explib.confirmation power`)')
    if out['violations']:
        out['status'] = 'fail'
    return out


def dynamic_read_budget_calibration(manifest):
    """Validate the measured read-budget calibration and its matched fixed arm.

    Dynamic block lengths are data-dependent.  An analytic
    ``target_block_tokens`` proxy is not a read-budget match, so a frozen
    dynamic-vs-fixed mechanism claim must carry a MEASURED mean block length
    and the concrete fixed top-k derived from it:

        k_fixed = round(topk_dynamic * mean_dynamic_block_tokens / fixed_block_size)

    The matching ``csa_fixed_rope_topk{k_fixed}`` arm must exist in the
    manifest.  This is the precondition ``dynamic_design_audit`` promotes on;
    a pending calibration is allowed only while the manifest is a draft.
    """
    out = {'status': 'not_declared', 'declared_status': None,
           'formula': ('k_fixed = round(topk_dynamic * '
                       'mean_dynamic_block_tokens / fixed_block_size)'),
           'expected_topk': None, 'arm': None, 'problems': [], 'warnings': []}
    if not isinstance(manifest, dict):
        out['status'] = 'fail'
        out['problems'].append(
            f'manifest must be an object, got {type(manifest).__name__}')
        return out
    binding = manifest.get('binding') if isinstance(manifest.get('binding'), dict) else {}
    rb = binding.get('read_budget_binding') if isinstance(
        binding.get('read_budget_binding'), dict) else {}
    pre = manifest.get('precondition') if isinstance(manifest.get('precondition'), dict) else {}
    cal = pre.get('read_budget_calibration')
    if not isinstance(cal, dict):
        out['warnings'].append(
            'precondition.read_budget_calibration is absent; a dynamic-vs-fixed '
            'read budget is only an analytic proxy')
        return out
    status = str(cal.get('status') or '').strip().lower()
    out['declared_status'] = status
    allowed = {'pending', 'required_before_freeze', 'measured', 'passed', 'frozen'}
    if status not in allowed:
        out['problems'].append(
            f'read_budget_calibration.status={status!r} is not one of {sorted(allowed)}')
    fixed_bs = _design_int(cal.get('fixed_block_size', rb.get('fixed_block_size')))
    topk = _design_int(cal.get('topk_dynamic', cal.get('dynamic_topk', rb.get('topk'))))
    mean_raw = cal.get('mean_dynamic_block_tokens',
                       cal.get('realized_mean_block_tokens',
                               cal.get('measured_mean_block_tokens')))
    mean = _design_probability(mean_raw, name='mean_dynamic_block_tokens',
                               lo=0.0, hi=float('inf'), lo_open=True) if mean_raw is not None else None
    if fixed_bs is not None and fixed_bs <= 0:
        out['problems'].append('fixed_block_size must be a positive integer')
        fixed_bs = None
    if topk is not None and topk <= 0:
        out['problems'].append('topk_dynamic must be a positive integer')
        topk = None
    expected = None
    if mean is not None and fixed_bs and topk:
        try:
            expected = int(math.floor(topk * mean / float(fixed_bs) + 0.5))
        except OverflowError:
            # A declared top-k beyond float64's range still has an exact
            # formula result.  Keep the historical float expression for the
            # ordinary path (so existing expected_topk values do not move) and
            # fall back to exact rational arithmetic only when it overflows.
            from fractions import Fraction as _Fraction
            _q = (_Fraction(int(topk)) * _Fraction.from_float(float(mean))
                  / int(fixed_bs))
            expected = int(math.floor(_q + _Fraction(1, 2)))
        out['expected_topk'] = expected
    declared_k = _design_int(cal.get('matched_fixed_topk', cal.get('k_fixed')))
    if status in ('measured', 'passed', 'frozen'):
        if mean is None:
            out['problems'].append(
                'a measured read-budget calibration must record a positive '
                'mean_dynamic_block_tokens (the realized measurement, not the target)')
        if fixed_bs is None:
            out['problems'].append(
                'a measured read-budget calibration must record fixed_block_size')
        if topk is None:
            out['problems'].append(
                'a measured read-budget calibration must record topk_dynamic')
        if declared_k is None:
            out['problems'].append(
                'a measured read-budget calibration must record matched_fixed_topk')
        elif expected is not None and declared_k != expected:
            out['problems'].append(
                f'matched_fixed_topk={declared_k} does not equal the calibration '
                f'formula result {expected}')
        if declared_k is not None:
            arm = f'csa_fixed_rope_topk{declared_k}'
            out['arm'] = arm
            arms = [str(a) for a in (manifest.get('arms') or [])]
            if arm not in arms:
                out['problems'].append(
                    f'the calibrated fixed arm {arm!r} is not in manifest.arms; '
                    'freeze the concrete arm that actually matches the measured read budget')
            declared_arm = cal.get('arm')
            if declared_arm is not None and str(declared_arm) != arm:
                out['problems'].append(
                    f'read_budget_calibration.arm={declared_arm!r} does not name '
                    f'the calibrated {arm!r}')
    elif declared_k is not None and expected is not None and declared_k != expected:
        out['problems'].append(
            f'an un-frozen calibration records matched_fixed_topk={declared_k} '
            f'but the formula gives {expected}; a guess must not be frozen later')
    if out['problems']:
        out['status'] = 'fail'
    elif status in ('measured', 'passed', 'frozen'):
        out['status'] = 'ok'
    else:
        out['status'] = 'pending'
    return out


def evidence_level_audit(manifest, *, manifest_path=None, panel_paths=None):
    """Zero-GPU evidence-level discipline audit.

    The project's claim ladder is: `confirmatory` > `exploratory` >
    `descriptive`/`draft`.  A result that was selected on the same validation
    split as its reported comparison is exploratory by construction, no matter
    how small its p-value is.  This audit checks:

      * a manifest/comparison declares a known evidence level;
      * `status=draft` (or `frozen_at=null`) can never carry a `confirmatory`
        comparison;
      * a confirmatory manifest must actually declare its protection:
        pre-test power gate, complete coverage, frozen scope and endpoint;
      * a summary inside `EXPLORATORY_PANEL_REGISTRY` cannot claim
        confirmatory status;
      * every comparison carries a level or inherits a defensible one.

    Returns a report with `status` ('ok'/'fail'/'not_verifiable_locally'),
    `violations`, `warnings`, per-comparison levels and the effective manifest
    level.  It never mutates the manifest.
    """
    out = {'status': 'ok', 'manifest': _evidence_manifest_path(manifest_path),
           'manifest_level': None, 'comparisons': {}, 'panels': {},
           'violations': [], 'warnings': [], 'levels': dict(EVIDENCE_LEVELS)}
    if not isinstance(manifest, dict):
        out['status'] = 'fail'
        out['violations'].append(
            f'manifest must be an object, got {type(manifest).__name__}')
        return out
    _status = str(manifest.get('status') or '').strip().lower()
    _frozen_at = manifest.get('frozen_at')
    _declared = str(manifest.get('evidence_level') or '').strip().lower()
    if _declared and _declared not in EVIDENCE_LEVELS:
        out['violations'].append(
            f'manifest evidence_level={_declared!r} is not one of {sorted(EVIDENCE_LEVELS)}')
        _declared = ''
    if not _declared:
        if _status == 'draft' or _frozen_at in (None, ''):
            _declared = 'draft'
        elif os.path.basename(out['manifest'] or '') == 'confirmation_frozen_dynamic.json':
            _declared = 'draft'
        elif os.path.basename(out['manifest'] or '') == 'confirmation_frozen.json':
            _declared = 'confirmatory'
        else:
            _declared = 'exploratory'
    # A draft manifest is a design record, not a pre-registration.
    if (_status == 'draft' or _frozen_at in (None, '')) and _declared == 'confirmatory':
        out['violations'].append(
            'a manifest with status=draft / frozen_at=null cannot declare '
            'evidence_level=confirmatory; a draft is a design record, not a '
            'pre-registration')
        _declared = 'draft'
    out['manifest_level'] = _declared
    _stat = manifest.get('statistics') if isinstance(manifest.get('statistics'), dict) else {}
    _scope = manifest.get('scope') if isinstance(manifest.get('scope'), dict) else {}
    # A confirmatory manifest's power plan is part of the frozen contract: the
    # declared p-floor / Holm floor must equal the exact-test numbers, and the
    # minimum-seed floor must still be able to clear alpha.  This is a zero-GPU
    # arithmetic check, not a claim that the effect is large enough.
    if _declared == 'confirmatory':
        _power = planned_power_audit(manifest)
        out['power_plan'] = _power
        out['violations'].extend(_power.get('violations') or [])
        out['warnings'].extend(_power.get('warnings') or [])
    _confirm_ready = True
    _why = []
    if _declared == 'confirmatory':
        if _stat.get('require_power_gate') is not True:
            _confirm_ready = False
            _why.append('statistics.require_power_gate is not true')
        if _stat.get('require_complete_coverage') is not True:
            _confirm_ready = False
            _why.append('statistics.require_complete_coverage is not true')
        # A claims statement is text, not an arbitrary truthy object.  Calling
        # ``.strip()`` on a list/dict/number raised an unrelated AttributeError
        # from the middle of the audit, so a malformed manifest could crash
        # instead of failing closed; merely stringifying it would also let a
        # non-empty container masquerade as a declared claim.
        _claims = _scope.get('confirmatory_claims')
        if not isinstance(_claims, str) or not _claims.strip():
            _confirm_ready = False
            _why.append('scope.confirmatory_claims is empty or not text')
        if not str(manifest.get('frozen_at') or '').strip():
            _confirm_ready = False
            _why.append('frozen_at is empty')
        if _confirm_ready is False:
            out['violations'].append(
                'manifest declares confirmatory but lacks the protection fields: '
                + '; '.join(_why))
    # Comparison-level declarations.
    comps = manifest.get('primary_comparisons')
    if not isinstance(comps, list) or not comps:
        out['violations'].append('primary_comparisons must be a non-empty list')
        comps = []
    _seen_comp_names = set()
    for i, comp in enumerate(comps):
        if not isinstance(comp, dict):
            out['violations'].append(f'primary comparison #{i} is not an object')
            continue
        _name = str(comp.get('name') or f'#{i}')
        # A dict keyed by comparison name cannot represent a duplicate family
        # slot; silently overwriting would shrink or corrupt the Holm family.
        if _name in _seen_comp_names:
            out['violations'].append(
                f'duplicate primary comparison name {_name!r}; each family slot '
                'must have a unique name')
        _seen_comp_names.add(_name)
        _sign = comp.get('expected_sign')
        if _sign is not None:
            if isinstance(_sign, str):
                _sign_ok = (_sign.strip().lower() in
                            ('negative', 'decrease', 'lower', 'less', '<',
                             'positive', 'increase', 'higher', 'greater', '>'))
            else:
                _sign_ok = (not isinstance(_sign, (bool, np.bool_))
                            and isinstance(_sign, numbers.Integral)
                            and int(_sign) in (-1, 1))
            if not _sign_ok:
                out['violations'].append(
                    f'{_name}: expected_sign={_sign!r} must be -1/+1 or a '
                    'negative/positive direction word (or null)')
        _lvl = str(comp.get('evidence_level') or _declared).strip().lower()
        if _lvl not in EVIDENCE_LEVELS:
            out['violations'].append(
                f'{_name}: evidence_level={_lvl!r} is not one of {sorted(EVIDENCE_LEVELS)}')
            _lvl = _declared
        if _lvl == 'confirmatory' and _declared != 'confirmatory':
            out['violations'].append(
                f'{_name}: comparison is labelled confirmatory but the manifest '
                f'level is {_declared!r}; confirmation cannot be established by a '
                'single comparison label')
        if _lvl == 'confirmatory' and _status == 'draft':
            out['violations'].append(
                f'{_name}: draft manifest cannot contain a confirmatory comparison')
        _mech = comp.get('mechanism_claim_allowed')
        if _mech is True and _lvl in ('draft', 'exploratory', 'descriptive'):
            out['warnings'].append(
                f'{_name}: mechanism_claim_allowed=true is a DESIGN target at '
                f'evidence_level={_lvl}; it may not be quoted as established '
                'evidence until the level is promoted by a frozen confirmation')
        out['comparisons'][_name] = {
            'a': comp.get('a'), 'b': comp.get('b'),
            'endpoint': comp.get('endpoint'), 'evidence_level': _lvl,
            'mechanism_claim_allowed': _mech,
        }
    # Existing result panels: registry membership is decisive.
    _panels = list(panel_paths) if panel_paths is not None else list(EXPLORATORY_PANEL_REGISTRY)
    for _p in _panels:
        _p = str(_p)
        _base = os.path.basename(os.path.normpath(_p))
        _full = _p if os.path.isabs(_p) else os.path.join(os.path.dirname(os.path.abspath(__file__)), _p)
        _summ = os.path.join(_full, 'summary.json')
        _rec = {'registry_label': EXPLORATORY_PANEL_REGISTRY.get(_base),
                'exists': os.path.isdir(_full), 'summary': _summ,
                'declared_level': None, 'ok': None, 'reason': None}
        if os.path.exists(_summ):
            try:
                with open(_summ, encoding='utf-8') as _f:
                    _s = json.load(_f)
            except Exception as _e:
                _s = None
                _rec['reason'] = f'summary is not readable JSON ({type(_e).__name__})'
            if isinstance(_s, dict):
                _lvl_raw = _s.get('evidence_level', _s.get('claim_level'))
                if _lvl_raw is None:
                    # `level` is a generic field in some artifacts; only use it
                    # when its value is actually an evidence-level word, never
                    # when it is a numeric training level or an unrelated label.
                    _cand = _s.get('level')
                    if isinstance(_cand, str) and _cand.strip().lower() in (
                            set(EVIDENCE_LEVELS) | {'confirmed', 'confirmation'}):
                        _lvl_raw = _cand
                _rec['declared_level'] = _lvl_raw
                _lvl = str(_rec['declared_level'] or '').strip().lower()
                if _lvl in ('confirmatory', 'confirmed', 'confirmation'):
                    _rec['ok'] = False
                    _rec['reason'] = ('a panel selected on the same validation '
                                      'split cannot declare confirmatory evidence')
                    out['violations'].append(f'{_base}: {_rec["reason"]}')
                elif _lvl and _lvl not in EVIDENCE_LEVELS:
                    _rec['ok'] = False
                    _rec['reason'] = f'unknown evidence level {_lvl!r}'
                    out['violations'].append(f'{_base}: {_rec["reason"]}')
                else:
                    _rec['ok'] = True
                    _rec['declared_level'] = _lvl or 'exploratory (registry default)'
            elif _rec['reason'] is None:
                _rec['ok'] = True
                _rec['declared_level'] = 'exploratory (registry default)'
        elif EXPLORATORY_PANEL_REGISTRY.get(_base):
            _rec['ok'] = True
            _rec['declared_level'] = 'exploratory (registry; no local summary)'
        out['panels'][_base or _p] = _rec
    if out['violations']:
        out['status'] = 'fail'
    return out


def _ledger_factory():
    """Arm factory for the ledger: v7's rope factory when available.

    The dynamic study is rope-only (mixing rope and non-rope arms would add an
    undeclared positional-encoding factor).  Importing v7 also installs the
    matching SmallGPT factory, so the parameter counts below are the ones the
    trainer actually builds.
    """
    try:
        import explib.rope as _V7
        return _V7.make_layer_cfgs_v7, 'v7_rope'
    except Exception:
        return make_layer_cfgs, 'base'


def _ledger_read_budget(cfgs, seq_len):
    """Analytic read-budget proxy for one arm at one sequence length.

    The unit is a token read by a query.  For full attention every query reads
    T tokens; HCA reads every closed block (T tokens up to the current
    position); CSA reads `min(index_topk, n_blocks)` blocks.  Dynamic block
    lengths are data-dependent, so a dynamic arm's token read is an ESTIMATE
    from `target_block_tokens` and is flagged as such.
    """
    T = max(int(seq_len), 1)

    def _read_ratio(numerator, denominator, *, cap=None):
        """Finite exact-integer ratio, with an explicit capped fallback.

        A declared sequence length or block count can exceed float64's range;
        converting it to float would raise ``OverflowError``.  The audit must
        still report a usable read fraction (0 or the cap) in that case, not
        crash on a syntactically valid scale plan.
        """
        value = _finite_int_ratio(numerator, denominator)
        if value is None:
            if cap is None:
                return None
            return float(cap) if int(numerator) >= int(denominator) else 0.0
        return min(float(cap), value) if cap is not None else value

    layers = []
    for cfg in cfgs:
        kind = str(getattr(cfg, 'kind', 'full') or 'full')
        dyn = bool(getattr(cfg, 'dynamic', False))
        if kind == 'full':
            unit = T
            n_blocks = 1
            k = 1
            # A windowed dense arm (`full_sw128_matched`) reads only its
            # window; treating it as full attention would overstate its read
            # budget in a future scale/baseline ledger.
            _win = int(getattr(cfg, 'window', 0) or 0)
            selected = min(T, _win) if _win > 0 else T
            overlap = 0
            estimate = False
        elif kind == 'hca':
            unit = max(int(getattr(cfg, 'block_size', 64) or 64), 1)
            # Exact integer ceiling: converting T to float first overflows for
            # a declared length beyond float64's range.
            n_blocks = (T + unit - 1) // unit
            k = n_blocks
            selected = min(T, n_blocks * unit)
            overlap = int(getattr(cfg, 'overlap', 0) or 0)
            estimate = dyn
        else:
            unit = (max(int(getattr(cfg, 'target_block_tokens', 1) or 1), 1)
                    if dyn else max(int(getattr(cfg, 'block_size', 4) or 4), 1))
            n_blocks = (T + unit - 1) // unit
            _topk = int(getattr(cfg, 'index_topk', 0) or 0)
            k = min(_topk, n_blocks) if _topk > 0 else n_blocks
            selected = k * unit
            # The local window is additive, not part of the selected blocks.
            overlap = int(getattr(cfg, 'sliding_window', 0) or getattr(cfg, 'overlap', 0) or 0)
            estimate = dyn
        _read_lanes = selected + (0 if kind == 'full' else overlap)
        layers.append({
            'kind': kind, 'dynamic': dyn, 'block_unit_tokens': unit,
            'n_blocks_available': n_blocks, 'blocks_read': k,
            'selected_block_tokens': selected,
            'local_window_tokens': overlap,
            'tokens_read_per_query': min(T, _read_lanes),
            'read_fraction': _read_ratio(_read_lanes, T, cap=1.0),
            'read_budget_is_estimate': estimate,
        })
    n = max(len(layers), 1)
    _sum_selected = sum(x['selected_block_tokens'] for x in layers)
    _sum_tokens = sum(x['tokens_read_per_query'] for x in layers)
    _sum_blocks = sum(x['blocks_read'] for x in layers)
    mean_selected = _read_ratio(_sum_selected, n)
    mean_tokens = _read_ratio(_sum_tokens, n)
    mean_fraction = sum(x['read_fraction'] for x in layers) / float(n)
    return {
        'seq_len': T,
        'n_layers': len(layers),
        'blocks_read_per_query_mean': _read_ratio(_sum_blocks, n),
        'selected_block_tokens_per_query_mean': mean_selected,
        'selected_read_fraction_mean': _read_ratio(_sum_selected, n * T),
        'tokens_read_per_query_mean': mean_tokens,
        'read_fraction_mean': mean_fraction,
        'read_budget_is_estimate': any(x['read_budget_is_estimate'] for x in layers),
        'layers': layers,
        'note': ('analytic proxy; dynamic block lengths are data-dependent, so '
                 'a dynamic read budget must be calibrated on the realized '
                 'segmentation before it can match a fixed arm'),
    }


def _ledger_arm_mr(manifest, arm, sig):
    """Per-arm MLP ratio: explicit pin first, then the project matcher.

    A missing pin is reported as unverifiable rather than silently assumed,
    because a dense parameter-proxy arm that silently reverted to mr=4 would
    stop being a parameter proxy.
    """
    binding = manifest.get('binding') if isinstance(manifest.get('binding'), dict) else {}
    per_arm = binding.get('per_arm_recipe_signature') if isinstance(
        binding.get('per_arm_recipe_signature'), dict) else {}
    ent = per_arm.get(arm)
    if isinstance(ent, dict) and ent.get('mr') is not None:
        try:
            return float(ent['mr']), 'frozen per-arm pin'
        except (TypeError, ValueError, OverflowError):
            return None, 'malformed per-arm mr'
    ref = 'csa_fixed_rope' if 'csa_fixed_rope' in (manifest.get('arms') or []) else 'csa_dynamic'
    try:
        mr = variant_mlp_ratio(
            arm, int(sig.get('v', 8192)), d=int(sig.get('d', 256)),
            n_layers=int(sig.get('L', 6)), n_heads=int(sig.get('H', 8)),
            d_head=int(sig.get('Dh', 32)), seq_len=int(sig.get('sl', 512)),
            matched={arm}, ref_variant=ref)
        return float(mr), 'computed by variant_mlp_ratio'
    except Exception as _e:
        return None, f'not computable ({type(_e).__name__}: {_e})'
