
# ---- 复现是两次运行之间的比较，不是一句声明 -------------------------------
#
# 同一 seed 的 in-process `reproducibility_check` 只证明「这段代码连续跑两遍
# 得到同一串 loss」，不证明**另一份产物**是这次运行的复现。跨目录、跨进程、
# 跨机器的「复现」需要逐维比较身份：代码字节、语料/评测切片、配置、执行
# 环境、初始化，以及最终指标。把这一比较写成一个可执行的只读检查，才不会
# 把「PPL 看起来差不多」当成复现，也不会把「环境不同但数值接近」误报成
# 代码/数据复现。

REPRODUCTION_SCHEMA = 'reproduction_diagnosis_v1'

_REPRODUCTION_REQUIRED_DIMENSIONS = ('code', 'data', 'config', 'environment',
                                     'initialization')
_REPRODUCTION_DIMENSION_VERDICTS = {
    'code': 'code_changed',
    'data': 'data_changed',
    'config': 'config_changed',
    'environment': 'environment_changed',
    'initialization': 'initialization_changed',
}


def _repro_text(value):
    """A non-empty textual identity value, or None.

    Booleans are not identities: ``True`` must not become the string ``'True'``
    and compare equal to a malformed record that stored the same typo.
    """
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        as_float = float(value)
        if not math.isfinite(as_float):
            # NaN/inf are not identities: str() would turn them into the
            # tokens 'nan'/'inf' and let two corrupt records compare equal.
            return None
        return str(as_float)
    return None


def _repro_int(value):
    """An exact positive integer count, or None.

    Token counts are denominators, not seeds.  ``int(1000.9)`` would silently
    turn a malformed count into a different valid denominator, and a boolean
    would become 0/1, so fractional, boolean, zero and negative values are
    refused here instead of being truncated into a comparable measurement.
    """
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, numbers.Integral):
        number = int(value)
    elif isinstance(value, numbers.Real):
        try:
            as_float = float(value)
        except (OverflowError, ValueError):
            # An exact Rational (Fraction) can be integral yet beyond
            # float64's exponent range; fall through to the exact check.
            as_float = None
        if as_float is not None:
            if not math.isfinite(as_float) or not as_float.is_integer():
                return None
            number = int(as_float)
        else:
            try:
                number = int(value)
            except (TypeError, ValueError, OverflowError):
                return None
            try:
                if value != number:
                    return None
            except (TypeError, ValueError, OverflowError):
                return None
    elif isinstance(value, (str, bytes)):
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
    else:
        # Objects outside numbers.Real can define __int__ and would be
        # truncated silently by int(value); require an exact numeric reading.
        # Decimal (not a numbers.Real) may be integral far beyond float64, so
        # check its value in the integer domain before falling back to float.
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        try:
            if value != number:
                return None
        except (TypeError, ValueError, OverflowError):
            try:
                as_float = float(value)
            except (TypeError, ValueError, OverflowError):
                return None
            if not math.isfinite(as_float) or float(number) != as_float:
                return None
    return number if number > 0 else None


def _repro_config_identity(value):
    """Type-aware identity token for one configuration value.

    The text normaliser used for fingerprints deliberately rejects booleans,
    because ``True`` must not compare equal to a malformed string ``'True'``.
    Configuration values are different: ``eager_batch=False`` and
    ``eager_batch=True`` are both valid, and collapsing either to "absent"
    made two different runs look identical.  Keep the JSON type in the token
    so ``False``/``True``, ``1``/``1.0`` and missing values stay distinct.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return 'bool:true' if bool(value) else 'bool:false'
    if isinstance(value, numbers.Integral):
        return 'int:' + str(int(value))
    if isinstance(value, numbers.Real):
        as_float = float(value)
        if not math.isfinite(as_float):
            return None
        return 'float:' + repr(as_float)
    if isinstance(value, str):
        text = value.strip()
        return 'str:' + text if text else None
    # A structured config value (a policy list/object) is still an identity;
    # canonical JSON keeps it comparable and JSON-safe.
    return 'json:' + _repro_canonical(value)


def _repro_canonical(value):
    """Stable text for an arbitrary descriptor; used only for equality."""
    try:
        return json.dumps(_jsonable(value), sort_keys=True,
                          separators=(',', ':'), ensure_ascii=True,
                          allow_nan=False)
    except Exception:
        return repr(value)


def _repro_render(value):
    """Human/JSON-safe rendering of a raw identity value for a report field."""
    text = _repro_text(value)
    return text if text is not None else _repro_canonical(value)


def _repro_validate_tolerance(name, value):
    if isinstance(value, bool):
        raise ValueError(f'reproduction: {name} must be a finite non-negative number')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'reproduction: {name} must be a finite non-negative number') from None
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f'reproduction: {name} must be a finite non-negative number')
    return number


def _repro_field(name, a, b, *, required=False, normalize=None):
    """Compare one optional/required identity field.

    ``required`` means the field must be present on both sides for a match;
    otherwise a both-absent field is treated as equal (legacy optional).
    ``normalize`` selects the identity representation; the default is the
    textual fingerprint normaliser.  A value that is present but cannot be
    normalised is ``unverified`` either way -- it is never silently equal to
    an absent value.
    """
    _norm = _repro_text if normalize is None else normalize
    va, vb = _norm(a), _norm(b)
    if va is None and vb is None:
        if a is not None or b is not None:
            return {'field': name, 'original': _repro_canonical(a),
                    'rerun': _repro_canonical(b), 'status': 'unverified',
                    'detail': ('one or both records carry a value that is not '
                               'a usable identity')}
        return {'field': name, 'original': None, 'rerun': None,
                'status': 'unverified' if required else 'match',
                'detail': ('both records omit this required identity' if required
                           else 'both records omit this optional identity')}
    if va is None or vb is None:
        return {'field': name, 'original': va, 'rerun': vb,
                'status': 'unverified',
                'detail': 'only one record carries a usable identity'}
    if va == vb:
        return {'field': name, 'original': va, 'rerun': vb, 'status': 'match',
                'detail': 'equal'}
    return {'field': name, 'original': va, 'rerun': vb, 'status': 'mismatch',
            'detail': 'values differ'}


def _repro_dimension(name, fields, *, notes=None):
    statuses = [f['status'] for f in fields]
    if 'mismatch' in statuses:
        status = 'mismatch'
    elif 'unverified' in statuses:
        status = 'unverified'
    else:
        status = 'match'
    out = {'name': name, 'status': status, 'fields': fields}
    if notes:
        out['notes'] = list(notes)
    return out


def _repro_registry_identity(value):
    """Comparable identity for a recorded split-registry block.

    The digest is the scientific identity; the status is included so a
    present-but-invalid registry (or a malformed object) cannot compare equal
    to a healthy one with an accidentally equal digest.  A bare string is
    accepted for forward compatibility, while every other shape becomes a
    ``malformed:`` token that ``_repro_field`` reports as unverified.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        digest = value.get('sha256')
        if isinstance(digest, str) and digest.strip():
            status = value.get('status')
            status = status.strip().lower() if isinstance(status, str) else ''
            if status != 'ok':
                return f'invalid:{status or "missing"}:{digest.strip()}'
            return f'ok:{digest.strip()}'
    return 'malformed:' + _repro_canonical(value)


def _repro_code_dimension(original, rerun):
    def _ident(rec):
        prov = rec.get('provenance') if isinstance(rec.get('provenance'), dict) else {}
        code = prov.get('code') if isinstance(prov.get('code'), dict) else {}
        tree = code.get('source_tree') if isinstance(code.get('source_tree'), dict) else {}
        # Keep raw values so a present-but-malformed optional stamp becomes
        # unverified instead of silently disappearing from the field list.
        _stamp = rec.get('_code')
        if _repro_text(_stamp) is None:
            _fallback = code.get('code_stamp')
            if _repro_text(_fallback) is not None:
                _stamp = _fallback
        return {
            'source_sha256': tree.get('sha256'),
            'code_semantics': code.get('code_semantics'),
            'code_stamp': _stamp,
            'split_registry': tree.get('split_registry'),
        }

    a, b = _ident(original), _ident(rerun)
    fields = [_repro_field('source_sha256', a['source_sha256'], b['source_sha256'],
                           required=True)]
    # The semantic stamp is a second, independent identity: two different code
    # bytes that happen to carry the same source digest would be a contradiction,
    # not a reproduction.
    if a['code_semantics'] is not None or b['code_semantics'] is not None:
        fields.append(_repro_field('code_semantics', a['code_semantics'],
                                   b['code_semantics']))
    if a['code_stamp'] is not None or b['code_stamp'] is not None:
        fields.append(_repro_field('code_stamp', a['code_stamp'],
                                   b['code_stamp']))
    # The source digest binds the bytes present, not the ordered fragment
    # registration that the split loader actually executed.  When either side
    # carries the derived registry identity, compare it: it names order and
    # membership explicitly (and rejects entries outside the tree, whose
    # bytes the source digest never saw).  Both absent is a legacy record,
    # named as such rather than silently promoted to a match.
    registry_a, registry_b = a['split_registry'], b['split_registry']
    if registry_a is not None or registry_b is not None:
        _ra = _repro_registry_identity(registry_a)
        _rb = _repro_registry_identity(registry_b)
        _registry_field = _repro_field('split_registry', _ra, _rb,
                                       required=True)
        # A present but malformed/invalid registry is not a comparable
        # identity even when both sides carry the same bad value.
        if (str(_ra or '').startswith(('malformed:', 'invalid:'))
                or str(_rb or '').startswith(('malformed:', 'invalid:'))):
            _registry_field['status'] = 'unverified'
            _registry_field['detail'] = (
                'a present split registry is malformed or not status=ok; '
                'the executed registration cannot be attested')
        fields.append(_registry_field)
    notes = []
    if registry_a is None and registry_b is None:
        notes.append('both records predate split_registry; the executed '
                     'fragment order/membership is not bound by this '
                     'comparison (the plain source digest is still checked)')
    if a['source_sha256'] is None and b['source_sha256'] is None:
        notes.append('only the CODE_SEMANTICS stamp is recorded; the git commit '
                     'and stamp cannot prove which source bytes produced the '
                     'record, so this dimension stays unverified')
    return _repro_dimension('code', fields, notes=notes)


def _repro_data_dimension(original, rerun):
    keys = ('data_fp', 'val_fp', 'eval_fp', 'data_version', 'corpus_fp')

    def _ident(rec):
        prov = rec.get('provenance') if isinstance(rec.get('provenance'), dict) else {}
        data = prov.get('data') if isinstance(prov.get('data'), dict) else {}
        out = {}
        for key in keys:
            value = rec.get(key)
            if value is None:
                value = data.get(key)
            # Keep the raw value: presence (including a malformed non-string)
            # is itself part of the identity, so normalisation happens in
            # _repro_field rather than erasing it here.
            out[key] = value
        return out

    a, b = _ident(original), _ident(rerun)
    fields, common, one_sided = [], [], []
    for key in keys:
        va, vb = a[key], b[key]
        if va is None and vb is None:
            continue
        if va is None or vb is None:
            # A one-sided fingerprint cannot attest the same data slice.  The
            # field is reported as unverified (not merely omitted from the
            # evidence set) so a matching sibling field cannot hide it.
            one_sided.append(key)
            fields.append({
                'field': key, 'original': _repro_render(va),
                'rerun': _repro_render(vb), 'status': 'unverified',
                'detail': 'only one record carries this data fingerprint'})
            continue
        common.append(key)
        fields.append(_repro_field(key, va, vb))
    if not common:
        fields.append({'field': 'data_identity', 'original': None, 'rerun': None,
                       'status': 'unverified',
                       'detail': 'the two records share no data fingerprint'})
    notes = []
    if one_sided:
        notes.append('one-sided data fields are not used as equality evidence: '
                     + ', '.join(one_sided))
    return _repro_dimension('data', fields, notes=notes)


def _repro_config_dimension(original, rerun):
    fields = [_repro_field('run_cfg', original.get('run_cfg'),
                           rerun.get('run_cfg'), required=True)]
    for key in ('protocol', 'warm_steps', 'amp_dtype', 'eager_batch'):
        fields.append(_repro_field(key, original.get(key), rerun.get(key),
                                   normalize=_repro_config_identity))
    # The compact run_cfg does not carry the optimizer/schedule descriptor or
    # the objective terms; compare those with the same strict helpers the
    # arm-pairing gates use, so a silent MLP/optimizer/loss-term change cannot
    # be read as a reproduction.
    for label, reason in (
            ('train_knobs', training_knobs_reason(original.get('train_knobs'),
                                                  rerun.get('train_knobs'))),
            ('train_objective', objective_reason(original, rerun))):
        if reason is None:
            fields.append({'field': label, 'original': None, 'rerun': None,
                           'status': 'match', 'detail': 'equal'})
            continue
        # Only a definite "these two valid descriptors differ" is a
        # mismatch.  A malformed/partial/one-sided descriptor is unverified:
        # we cannot claim the configurations differ, only that they cannot be
        # shown to match.
        _reason = str(reason)
        _mismatch = (_reason.startswith(('different ', 'objective differs'))
                     or ' differ' in _reason)
        fields.append({'field': label,
                       'original': _repro_canonical(original.get(label)),
                       'rerun': _repro_canonical(rerun.get(label)),
                       'status': 'mismatch' if _mismatch else 'unverified',
                       'detail': _reason})
    return _repro_dimension('config', fields)


def _repro_environment_dimension(original, rerun):
    def _fp(rec):
        value = rec.get('exec_fp')
        if value is None and isinstance(rec.get('hyperparams'), dict):
            value = rec['hyperparams'].get('exec_fp')
        return value

    def _fp_v2(rec):
        value = rec.get('env_fp_v2')
        if value is None and isinstance(rec.get('hyperparams'), dict):
            value = rec['hyperparams'].get('env_fp_v2')
        return value

    def _fp_v3(rec):
        value = rec.get('env_fp_v3')
        if value is None and isinstance(rec.get('hyperparams'), dict):
            value = rec['hyperparams'].get('env_fp_v3')
        return value

    fields = [_repro_field('exec_fp', _fp(original), _fp(rerun), required=True)]
    a2, b2 = _fp_v2(original), _fp_v2(rerun)
    notes = []
    if a2 is not None or b2 is not None:
        # Present on at least one side: compare it.  A one-sided value is
        # unverified, never an inferred match (the digests were computed by
        # different provenance generations).
        fields.append(_repro_field('env_fp_v2', a2, b2))
        if a2 is None or b2 is None:
            notes.append('one record predates env_fp_v2; the strict '
                         'lock-pinned environment identity cannot be compared '
                         'and this dimension stays unverified')
    else:
        notes.append('both records predate env_fp_v2; only the legacy '
                     'six-package exec_fp is compared (transitive pins, the '
                     'lock bytes and the OS/driver block are unverified)')
    a3, b3 = _fp_v3(original), _fp_v3(rerun)
    if a3 is not None or b3 is not None:
        # v3 adds the BLAS/LAPACK/SIMD backend and unpinned runtime
        # distributions.  It is only meaningful when both sides carry it;
        # a one-sided v3 is a generation difference and is reported as
        # unverified rather than silently downgraded to the v2 match.
        fields.append(_repro_field('env_fp_v3', a3, b3))
        if a3 is None or b3 is None:
            notes.append('one record predates env_fp_v3; the numerical '
                         'backend (BLAS/LAPACK/SIMD) and unpinned runtime '
                         'distributions cannot be compared and this dimension '
                         'stays unverified')
        else:
            notes.append('env_fp_v3 additionally binds the NumPy/PyTorch '
                         'BLAS, LAPACK and SIMD build/runtime identity and '
                         'every unpinned non-tooling distribution')
    return _repro_dimension('environment', fields, notes=notes)


def _repro_initialization_dimension(original, rerun):
    def _fp(rec):
        value = rec.get('init_core_fp')
        if value is None and isinstance(rec.get('hyperparams'), dict):
            value = rec['hyperparams'].get('init_core_fp')
        return value

    return _repro_dimension('initialization', [
        _repro_field('init_core_fp', _fp(original), _fp(rerun), required=True)])


def _repro_rng_dimension(original, rerun):
    """Diagnostic only, mirroring the library's rng_fp contract.

    ``rng_fp`` is recorded immediately after ``set_seed`` and is not part of
    the arm-pairing gate: a different architecture consumes a different amount
    of RNG.  For two executions of the SAME cell it is still a useful warning,
    so it is reported but does not by itself make the runs non-reproducible.
    """
    va = original.get('rng_fp')
    vb = rerun.get('rng_fp')
    return _repro_field('rng_fp', va, vb)


def _repro_metric(original, rerun, *, rtol, atol):
    def _ppl(rec):
        value = rec.get('ppl')
        return float(value) if ppl_is_usable(value) else None

    a, b = _ppl(original), _ppl(rerun)
    na, nb = _repro_int(original.get('n_tokens')), _repro_int(rerun.get('n_tokens'))
    out = {'name': 'ppl', 'original': a, 'rerun': b, 'rtol': rtol, 'atol': atol,
           'n_tokens': {'original': na, 'rerun': nb},
           'abs_diff': None, 'rel_diff': None}
    if a is None or b is None:
        out['status'] = 'missing'
        out['detail'] = 'one or both records carry no usable PPL (>=1, finite)'
        return out
    if na is None or nb is None:
        # A PPL is exp(mean CE) over a token denominator.  Without an exact
        # count on both sides the two numbers cannot be shown to average the
        # same measurement, so equality of the scalar alone is not a
        # reproduction.  Malformed counts (fractional/boolean/negative) are
        # reported the same way rather than truncated into a match.
        missing_side = ('both records' if na is None and nb is None
                        else 'one record')
        out['status'] = 'denominator_unverified'
        out['detail'] = (f'{missing_side} carry no exact positive token '
                         'count; the PPL denominators cannot be shown to match')
        return out
    if na != nb:
        out['status'] = 'denominator_mismatch'
        out['detail'] = (f'the PPLs are averaged over different token counts '
                         f'({na} vs {nb}); they are not the same measurement')
        return out
    diff = abs(b - a)
    out['abs_diff'] = diff
    out['rel_diff'] = (diff / abs(a)) if a != 0 else None
    if a == b:
        out['status'] = 'exact'
        out['detail'] = 'bit-identical PPL'
    elif diff <= atol + rtol * abs(a):
        out['status'] = 'within_tolerance'
        out['detail'] = (f'|{b} - {a}| = {diff:.6g} <= atol + rtol*|a| = '
                         f'{atol + rtol * abs(a):.6g}')
    else:
        out['status'] = 'drift'
        out['detail'] = (f'|{b} - {a}| = {diff:.6g} > atol + rtol*|a| = '
                         f'{atol + rtol * abs(a):.6g}')
    return out


def reproduction_diagnosis(original, rerun, *, rtol=1e-6, atol=0.0):
    """Compare one recorded cell against an independent re-execution.

    Returns a dimension-by-dimension verdict.  ``reproduced`` is True only
    when every required identity dimension is a definite match AND the final
    metric is exact or within the declared tolerance.  A missing fingerprint
    is ``provenance_incomplete`` (never a pass), and a different environment
    or data slice is reported as its own failure mode rather than as generic
    numerical drift.

    This is a descriptive identity comparison, not a statistical test: it has
    no p-value, no confidence interval and no tolerance for selection.  The
    tolerance must be declared by the caller and is echoed in the output.
    """
    if not isinstance(original, dict) or not isinstance(rerun, dict):
        raise ValueError('reproduction_diagnosis: both records must be mappings')
    rtol = _repro_validate_tolerance('rtol', rtol)
    atol = _repro_validate_tolerance('atol', atol)

    problems = []
    warnings = []
    if original.get('synthesized') or rerun.get('synthesized'):
        return {
            'kind': 'reproduction_diagnosis', 'schema': REPRODUCTION_SCHEMA,
            'verdict': 'not_comparable', 'reproduced': False,
            'bit_identical': False, 'identity_ok': False,
            'rtol': rtol, 'atol': atol,
            'dimensions': {}, 'metric': None,
            'problems': ['a synthesized record was reconstructed from an '
                         'aggregate and has no weights/run to reproduce'],
            'note': 'synthesized records are display-only evidence',
        }

    dimensions = {
        'code': _repro_code_dimension(original, rerun),
        'data': _repro_data_dimension(original, rerun),
        'config': _repro_config_dimension(original, rerun),
        'environment': _repro_environment_dimension(original, rerun),
        'initialization': _repro_initialization_dimension(original, rerun),
    }
    rng_field = _repro_rng_dimension(original, rerun)
    dimensions['rng_state'] = {
        'name': 'rng_state', 'diagnostic': True,
        'fields': [rng_field], 'status': rng_field['status'],
    }
    metric = _repro_metric(original, rerun, rtol=rtol, atol=atol)

    mismatch = [d for d in _REPRODUCTION_REQUIRED_DIMENSIONS
                if dimensions[d]['status'] == 'mismatch']
    unverified = [d for d in _REPRODUCTION_REQUIRED_DIMENSIONS
                  if dimensions[d]['status'] == 'unverified']
    for dim in _REPRODUCTION_REQUIRED_DIMENSIONS:
        for field in dimensions[dim]['fields']:
            if field['status'] != 'match':
                problems.append(f'{dim}.{field["field"]}: {field["detail"]}')
    if dimensions['rng_state']['status'] == 'mismatch':
        warnings.append('rng_state.rng_fp: the initial RNG stream differs; '
                        'a same-seed replay did not start from the recorded '
                        'state (diagnostic only, not an arm-pairing failure)')
    if metric and metric['status'] not in ('exact', 'within_tolerance'):
        problems.append(f'metric.ppl: {metric.get("detail")}')

    if mismatch:
        verdict = _REPRODUCTION_DIMENSION_VERDICTS[mismatch[0]]
        reproduced = False
    elif unverified:
        verdict = 'provenance_incomplete'
        reproduced = False
    elif metric['status'] == 'missing':
        verdict = 'metric_missing'
        reproduced = False
    elif metric['status'] in ('denominator_mismatch',
                              'denominator_unverified'):
        verdict = 'metric_mismatch'
        reproduced = False
    elif metric['status'] == 'exact':
        verdict = 'identical'
        reproduced = True
    elif metric['status'] == 'within_tolerance':
        verdict = 'within_tolerance'
        reproduced = True
    else:
        verdict = 'numerical_drift'
        reproduced = False

    identity_ok = bool(not mismatch and not unverified)
    return {
        'kind': 'reproduction_diagnosis', 'schema': REPRODUCTION_SCHEMA,
        'verdict': verdict, 'reproduced': bool(reproduced),
        'bit_identical': bool(identity_ok and metric['status'] == 'exact'),
        'identity_ok': identity_ok,
        'rtol': rtol, 'atol': atol,
        'dimensions': dimensions, 'metric': metric,
        'problems': problems,
        'warnings': warnings,
        'note': ('identity comparison of two independent executions; absence '
                 'of a fingerprint is not evidence of a match'),
    }


def _repro_load_summary(source, label):
    if isinstance(source, dict):
        return source, label
    try:
        path = os.fspath(source)
    except TypeError:
        raise ValueError(f'reproduction: {label} must be a summary mapping, '
                         'a directory or a summary.json path') from None
    # ``os.fspath`` may yield bytes; joining that with the literal
    # ``'summary.json'`` raises TypeError, so decode before path arithmetic.
    path = os.path.abspath(os.fsdecode(path))
    if os.path.isdir(path):
        path = os.path.join(path, 'summary.json')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'reproduction: {label} summary not found: {path}')
    with open(path, encoding='utf-8') as fh:
        obj = json.load(fh)
    if not isinstance(obj, dict):
        raise ValueError(f'reproduction: {label} summary is a '
                         f'{type(obj).__name__}, not a JSON object')
    return obj, path


def _repro_cells(summary):
    """Split a summary into comparable cells and malformed declared cells.

    Non-cell aggregate/metadata entries (keys without ``::seed``) are allowed
    and ignored.  A key that declares itself a cell but does not carry an
    object record is corrupt: silently dropping it would let the remaining
    cells produce a green "reproduced" verdict for an incomplete panel.
    """
    out = {}
    malformed = []
    for key, rec in summary.items():
        key = str(key)
        if '::seed' not in key:
            continue
        if not isinstance(rec, dict):
            malformed.append(key)
            continue
        out[key] = rec
    return out, sorted(malformed)


def compare_reproductions(original, rerun, *, rtol=1e-6, atol=0.0):
    """Compare two panel summaries cell by cell.

    ``original`` / ``rerun`` may be summary mappings, result directories or
    paths to ``summary.json``.  Cells are matched by their canonical
    ``variant::seedN`` key; a key present on only one side is not silently
    dropped.  The aggregate verdict is ``reproduced`` only when every matched
    cell reproduces and no planned cell is missing.
    """
    rtol = _repro_validate_tolerance('rtol', rtol)
    atol = _repro_validate_tolerance('atol', atol)
    a_summary, a_label = _repro_load_summary(original, 'original')
    b_summary, b_label = _repro_load_summary(rerun, 'rerun')
    a_cells, a_malformed = _repro_cells(a_summary)
    b_cells, b_malformed = _repro_cells(b_summary)
    common = sorted(set(a_cells) & set(b_cells))
    only_original = sorted(set(a_cells) - set(b_cells))
    only_rerun = sorted(set(b_cells) - set(a_cells))

    cells, counts, problems, warnings = {}, {}, [], []
    for key in common:
        diag = reproduction_diagnosis(a_cells[key], b_cells[key],
                                      rtol=rtol, atol=atol)
        diag['key'] = key
        cells[key] = diag
        counts[diag['verdict']] = counts.get(diag['verdict'], 0) + 1
        for problem in diag.get('problems') or []:
            problems.append(f'{key}: {problem}')
        for warning in diag.get('warnings') or []:
            warnings.append(f'{key}: {warning}')
    for key in only_original:
        problems.append(f'{key}: recorded only in the original summary')
    for key in only_rerun:
        problems.append(f'{key}: recorded only in the rerun summary')
    for key in a_malformed:
        problems.append(f'{key}: original cell is not an object record')
    for key in b_malformed:
        problems.append(f'{key}: rerun cell is not an object record')

    if not common:
        verdict = 'not_comparable'
    elif any(v in ('code_changed', 'data_changed', 'config_changed',
                   'environment_changed', 'initialization_changed')
             for v in counts):
        verdict = 'not_reproduced'
    elif only_original or only_rerun or a_malformed or b_malformed:
        verdict = 'incomplete'
    elif any(v in ('provenance_incomplete', 'metric_missing',
                   'metric_mismatch', 'not_comparable') for v in counts):
        verdict = 'incomplete'
    elif any(v == 'numerical_drift' for v in counts):
        verdict = 'drift'
    else:
        verdict = 'reproduced'
    reproduced = verdict == 'reproduced'
    bit_identical = bool(reproduced and all(
        d['verdict'] == 'identical' for d in cells.values()))
    for dim in ('code', 'data', 'config', 'environment', 'initialization'):
        n_bad = sum(1 for d in cells.values()
                    if (d.get('dimensions') or {}).get(dim, {}).get('status')
                    in ('mismatch', 'unverified'))
        if n_bad:
            problems.append(f'{n_bad} cells have a {dim} identity that is '
                            'mismatched or unverified')
    n_rng = sum(1 for d in cells.values()
                if (d.get('dimensions') or {}).get('rng_state', {}).get('status')
                == 'mismatch')
    if n_rng:
        warnings.append(f'{n_rng} cells started from a different initial RNG '
                        'stream (diagnostic only; not an arm-pairing failure)')
    return {
        'kind': 'reproduction_report', 'schema': REPRODUCTION_SCHEMA,
        'verdict': verdict, 'reproduced': bool(reproduced),
        'bit_identical': bit_identical,
        'rtol': rtol, 'atol': atol,
        'original': a_label, 'rerun': b_label,
        'n_compared': len(common),
        'n_only_original': len(only_original),
        'n_only_rerun': len(only_rerun),
        'n_malformed_original': len(a_malformed),
        'n_malformed_rerun': len(b_malformed),
        'only_original': only_original[:50],
        'only_rerun': only_rerun[:50],
        'malformed_original': a_malformed[:50],
        'malformed_rerun': b_malformed[:50],
        'counts': dict(sorted(counts.items())),
        'cells': cells,
        'problems': problems,
        'warnings': warnings,
        'note': ('a reproduced verdict requires matching source bytes, data '
                 'slice, configuration, execution environment, matched '
                 'initialization and a metric inside the declared tolerance; '
                 'missing provenance is reported, not assumed equal'),
    }


_REPRODUCTION_USAGE = """usage:
  python -m explib.reproduce compare ORIGINAL RERUN [options]
  python -m explib.reproduce report ORIGINAL.json RERUN.json [options]

compare: two independent executions of the same cells (result directories or
summary.json paths).  Exit status: 0 reproduced, 1 not reproduced, 2 usage or
input error.  --require exact removes the tolerance and demands identical PPL.

report: two stamped analysis/report JSON files (for example an audit_report.json
and its rebuild).  Separates a changed result body from changed evidence inputs,
changed code and a changed execution environment.

compare options:
  --rtol FLOAT   relative PPL tolerance (default 1e-6)
  --atol FLOAT   absolute PPL tolerance (default 0)
  --json PATH    also write the machine-readable report
  --require {tolerance,exact}

report options:
  --ignore POINTER       ignore a JSON pointer (repeatable), e.g.
                         /experiment_ledger for an append-only audit trace
  --ignore-ledger        shorthand for --ignore /experiment_ledger
  --max-differences N    stop after N reported pointers (default 200)
  --json PATH            also write the machine-readable report
"""


def _reproduction_cli(argv=None):
    import argparse
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(_REPRODUCTION_USAGE)
        return 2
    if args[0] in ('-h', '--help', 'help'):
        print(_REPRODUCTION_USAGE)
        return 0
    command = args.pop(0)
    if command in ('report', 'artifact', 'analysis_report', 'analysis'):
        return _report_diff_cli(args)
    if command not in ('compare', 'check', 'reproduce'):
        print(f'reproduction: unknown subcommand {command!r}\n\n'
              + _REPRODUCTION_USAGE)
        return 2
    parser = argparse.ArgumentParser(prog='python -m explib.reproduce',
                                     add_help=False)
    parser.add_argument('original')
    parser.add_argument('rerun')
    parser.add_argument('--rtol', type=float, default=1e-6)
    parser.add_argument('--atol', type=float, default=0.0)
    parser.add_argument('--json', dest='json_path', default=None)
    parser.add_argument('--require', choices=('tolerance', 'exact'),
                        default='tolerance')
    try:
        ns = parser.parse_args(args)
    except SystemExit:
        print(_REPRODUCTION_USAGE)
        return 2
    rtol = 0.0 if ns.require == 'exact' else ns.rtol
    atol = 0.0 if ns.require == 'exact' else ns.atol
    try:
        report = compare_reproductions(ns.original, ns.rerun,
                                       rtol=rtol, atol=atol)
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({'kind': 'reproduction_report', 'verdict': 'error',
                          'reproduced': False,
                          'error': f'{type(exc).__name__}: {exc}'},
                         ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True,
                     allow_nan=False, default=float))
    if ns.json_path:
        try:
            out_path = os.path.abspath(ns.json_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            atomic_write_json(out_path, report, indent=2)
        except OSError as exc:
            print(f'reproduction: could not write {ns.json_path}: {exc}')
            return 2
    ok = bool(report['reproduced'])
    if ns.require == 'exact':
        ok = ok and bool(report['bit_identical'])
    return 0 if ok else 1


def _report_diff_cli(args):
    """CLI for ``python -m explib.reproduce report A B`` (see usage above)."""
    import argparse
    parser = argparse.ArgumentParser(prog='python -m explib.reproduce report',
                                     add_help=False)
    parser.add_argument('original')
    parser.add_argument('rerun')
    parser.add_argument('--ignore', action='append', default=[])
    parser.add_argument('--ignore-ledger', action='store_true')
    parser.add_argument('--max-differences', type=int, default=200)
    parser.add_argument('--json', dest='json_path', default=None)
    try:
        ns = parser.parse_args(args)
    except SystemExit:
        print(_REPRODUCTION_USAGE)
        return 2
    if ns.max_differences < 1:
        print('reproduction: --max-differences must be a positive integer')
        return 2
    ignore = list(ns.ignore)
    if ns.ignore_ledger and '/experiment_ledger' not in ignore:
        ignore.append('/experiment_ledger')
    try:
        report = analysis_report_diff(
            ns.original, ns.rerun, ignore=ignore,
            max_differences=ns.max_differences)
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({'kind': 'analysis_report_diff', 'verdict': 'error',
                          'reproduced': False,
                          'error': f'{type(exc).__name__}: {exc}'},
                         ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True,
                     allow_nan=False, default=float))
    if ns.json_path:
        try:
            out_path = os.path.abspath(ns.json_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            atomic_write_json(out_path, report, indent=2)
        except OSError as exc:
            print(f'reproduction: could not write {ns.json_path}: {exc}')
            return 2
    return 0 if report.get('reproduced') else 1
