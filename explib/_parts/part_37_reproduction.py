
# ---- 复现是两次运行之间的比较，不是一句声明 -------------------------------
#
# 同一 seed 的 in-process `reproducibility_check` 只证明「这段代码连续跑两遍
# 得到同一串 loss」，不证明**另一份产物**是这次运行的复现。跨目录、跨进程、
# 跨机器的「复现」需要逐维比较身份：代码字节、语料/评测切片、配置、执行
# 环境、初始化，以及最终指标。把这一比较写成一个可执行的只读检查，才不会
# 把「PPL 看起来差不多」当成复现，也不会把「环境不同但数值接近」误报成
# 代码/数据复现。

# v2 adds the recorded-curve dimension and makes ``bit_identical`` require an
# identical trajectory.  A v1 report could call two runs "bit-identical" from
# the final PPL alone; that verdict is not comparable with v2.
REPRODUCTION_SCHEMA = 'reproduction_diagnosis_v2'

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
    # ``np.bool_`` is not a Python ``bool`` subclass, but it is still a flag,
    # not a tolerance: ``float(np.bool_(True))`` used to become 1.0 and make a
    # replay comparison arbitrarily loose.
    if isinstance(value, (bool, np.bool_)):
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


_REPRO_CURVE_STATS = (
    'curve_n_points', 'curve_first', 'curve_last', 'curve_best',
    'curve_n_nonfinite', 'curve_first_nonfinite_index',
)


def _repro_curve_sha(value):
    """A recorded SHA-256 hex digest, or None for a missing/malformed value."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, str):
        return None
    text = value.strip().lower()
    if len(text) != 64 or any(c not in '0123456789abcdef' for c in text):
        return None
    return text


def _repro_curve_stats(rec):
    """The scalar curve summary a summary.json record may carry."""
    out = {}
    for key in _REPRO_CURVE_STATS:
        raw = rec.get(key)
        if raw is None or isinstance(raw, (bool, np.bool_, str, bytes, bytearray)):
            out[key] = None
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError, OverflowError):
            out[key] = None
            continue
        out[key] = number if math.isfinite(number) else None
    return out


def _repro_curve_payload(rec, base):
    """Load one cell's recorded trajectory and its identity fields.

    The panel writer moves per-cell arrays to a ``curves/<cell>.json`` sidecar
    when they are large and keeps only ``curve_ref`` / ``curve_sha256`` in the
    summary.  Older summaries may still carry ``losses`` / ``ppl_history``
    inline.  Both shapes, and the scalar summary-only legacy shape, are read
    here so a comparison is not silently reduced to "final PPL only".

    Nothing here writes or trains; a missing/unreadable sidecar is recorded in
    ``problems`` and leaves the curve unverified rather than raising.
    """
    out = {
        'values': None, 'kind': None, 'source': None,
        'sha256': _repro_curve_sha(rec.get('curve_sha256')),
        'ppl_sha256': _repro_curve_sha(rec.get('ppl_history_sha256')),
        'stats': _repro_curve_stats(rec), 'problems': [], 'path': None,
        'digest_invalid': False,
    }
    for kind in ('losses', 'ppl_history'):
        raw = rec.get(kind)
        if isinstance(raw, list) and raw:
            out.update(values=raw, kind=kind, source='inline')
            return out
    ref_key = 'curve_ref'
    ref = rec.get('curve_ref')
    if not isinstance(ref, str) or not ref.strip():
        ref_key = 'ppl_history_ref'
        ref = rec.get('ppl_history_ref')
    if isinstance(ref, str) and ref.strip():
        ref = ref.strip()
        # A sidecar reference is an artifact-relative path.  Without the
        # summary's directory there is no honest way to resolve it, so leave
        # values absent and let the scalar summary or the digest decide.
        # Resolve through realpath and refuse a target outside that directory:
        # a hand-edited summary must not turn a comparison into a read of an
        # unrelated file elsewhere on the audit machine.
        path = None
        if base:
            base_real = os.path.realpath(base)
            candidate = os.path.realpath(
                ref if os.path.isabs(ref) else os.path.join(base, ref))
            try:
                inside = os.path.commonpath([base_real, candidate]) == base_real
            except (OSError, ValueError):
                inside = False
            if inside:
                path = candidate
            else:
                out['problems'].append(
                    f'curve sidecar {ref!r} resolves outside the summary '
                    'directory; refusing to read it')
        else:
            out['problems'].append(
                'curve_ref is present but there is no summary directory to '
                'resolve it against')
        out['path'] = path
        if path is not None:
            try:
                with open(path, 'rb') as fh:
                    raw_bytes = fh.read()
                file_sha = hashlib.sha256(raw_bytes).hexdigest()
                doc = json.loads(raw_bytes.decode('utf-8'))
            except (OSError, ValueError, UnicodeDecodeError) as exc:
                out['problems'].append(
                    f'curve sidecar {ref!r} is unreadable '
                    f'({type(exc).__name__}: {exc})')
            else:
                if not isinstance(doc, dict):
                    out['problems'].append(
                        f'curve sidecar {ref!r} is a {type(doc).__name__}, '
                        'not a JSON object')
                else:
                    # A recorded digest is only identity evidence if it still
                    # describes the bytes on disk.  A stale/tampered sidecar
                    # (or a summary copied without its curve file) must not
                    # let two different arrays claim the same digest.
                    claimed = []
                    if ref_key == 'curve_ref':
                        claimed.append(('sha256', 'curve_sha256'))
                        ph_ref = rec.get('ppl_history_ref')
                        if (not isinstance(ph_ref, str) or not ph_ref.strip()
                                or ph_ref.strip() == ref):
                            claimed.append(('ppl_sha256',
                                            'ppl_history_sha256'))
                    else:
                        claimed.append(('ppl_sha256', 'ppl_history_sha256'))
                        c_ref = rec.get('curve_ref')
                        if (not isinstance(c_ref, str) or not c_ref.strip()
                                or c_ref.strip() == ref):
                            claimed.append(('sha256', 'curve_sha256'))
                    for slot, label in claimed:
                        recorded = out[slot]
                        if recorded is not None and recorded != file_sha:
                            out[slot] = None
                            out['digest_invalid'] = True
                            out['problems'].append(
                                f'{label} does not match the sidecar bytes '
                                f'for {ref!r}')
                    for kind in ('losses', 'ppl_history'):
                        raw = doc.get(kind)
                        if isinstance(raw, list) and raw:
                            out.update(values=raw, kind=kind,
                                       source='curve_sidecar')
                            return out
    return out


def _repro_curve_array(values):
    """Convert one recorded series to float, refusing type coercion.

    ``np.asarray(['3.0'], dtype=float)`` would turn a malformed string into a
    measurement and ``bool`` into 0/1.  The writer emits numbers or ``null``;
    anything else is not a reproducible measurement and must stay unverified
    instead of being silently converted.  ``None`` becomes NaN (the same
    missing/non-finite marker used for a diverged step).
    """
    out = np.empty(len(values), dtype=float)
    for i, value in enumerate(values):
        if value is None:
            out[i] = np.nan
            continue
        if isinstance(value, (bool, np.bool_, str, bytes, bytearray)):
            return None, f'element {i} is a {type(value).__name__}, not a number'
        try:
            out[i] = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            return None, (f'element {i} is not numeric '
                          f'({type(exc).__name__}: {exc})')
    return out, None


def _repro_curve_max_abs_diff(aa, bb):
    """Largest finite pairwise difference, or None if it is unrepresentable.

    ``aa`` and ``bb`` can each be near float64's limit with opposite signs;
    ``a - b`` then overflows to inf even though both inputs are finite.  The
    report is strict JSON, so an unrepresentable magnitude must be withheld
    (None) rather than serialised as ``Infinity``.  Non-finite entries are
    missing/diverged markers and are not part of a numeric magnitude.
    """
    finite = np.isfinite(aa) & np.isfinite(bb)
    if not np.any(finite):
        return None
    with np.errstate(over='ignore', invalid='ignore'):
        diffs = np.abs(aa[finite] - bb[finite])
    if not diffs.size:
        return None
    value = float(np.max(diffs))
    return value if math.isfinite(value) else None


def _repro_curve_values_compare(a_values, b_values, *, rtol, atol):
    """Compare two recorded numeric series with an explicit tolerance.

    ``None`` and non-finite entries are treated as the same missing marker
    (the writer uses ``null`` for a non-finite loss), so two runs that both
    diverged at the same step are not falsely reported as a reproduction.
    """
    if a_values is None or b_values is None:
        return {'status': 'unverified', 'method': 'none',
                'detail': 'one or both records carry no curve values'}
    if len(a_values) != len(b_values):
        return {'status': 'mismatch', 'method': 'curve_values',
                'n_points_original': len(a_values),
                'n_points_rerun': len(b_values),
                'detail': (f'different curve lengths ({len(a_values)} vs '
                           f'{len(b_values)})')}
    if not a_values:
        return {'status': 'unverified', 'method': 'none',
                'detail': 'an empty curve is not a trajectory'}
    aa, a_problem = _repro_curve_array(a_values)
    bb, b_problem = _repro_curve_array(b_values)
    if a_problem or b_problem:
        return {'status': 'unverified', 'method': 'none',
                'detail': '; '.join(p for p in (a_problem, b_problem) if p)}
    if aa.shape != bb.shape:
        return {'status': 'mismatch', 'method': 'curve_values',
                'detail': f'curve shapes differ ({aa.shape} vs {bb.shape})'}
    n_a = int(np.count_nonzero(~np.isfinite(aa)))
    n_b = int(np.count_nonzero(~np.isfinite(bb)))
    if np.array_equal(aa, bb, equal_nan=True):
        return {'status': 'exact', 'method': 'curve_values',
                'n_points_original': int(aa.size),
                'n_points_rerun': int(bb.size),
                'n_nonfinite_original': n_a, 'n_nonfinite_rerun': n_b,
                'first_difference_index': None, 'max_abs_diff': 0.0,
                'byte_identical': True,
                'detail': 'every recorded curve value is identical'}
    # Tolerance must be symmetric: comparing A->B and B->A cannot disagree
    # about whether the same two trajectories are close.  ``np.isclose``
    # scales by |bb| only, while the declared contract (and the scalar-summary
    # comparator) uses max(|a|,|b|).  Keep non-finite entries as missing
    # markers instead of letting inf satisfy an inf tolerance.
    close = np.isnan(aa) & np.isnan(bb)
    finite = np.isfinite(aa) & np.isfinite(bb)
    if np.any(finite):
        with np.errstate(invalid='ignore', over='ignore'):
            scale = np.maximum(np.abs(aa[finite]), np.abs(bb[finite]))
            tolerance = atol + rtol * scale
            difference = np.abs(aa[finite] - bb[finite])
            close[finite] = difference <= tolerance
    if bool(np.all(close)):
        return {'status': 'within_tolerance', 'method': 'curve_values',
                'n_points_original': int(aa.size),
                'n_points_rerun': int(bb.size),
                'n_nonfinite_original': n_a, 'n_nonfinite_rerun': n_b,
                'first_difference_index': None,
                'max_abs_diff': _repro_curve_max_abs_diff(aa, bb),
                'byte_identical': False,
                'detail': ('every curve value is within atol + rtol*'
                           'max(|original|,|rerun|); the trajectories are '
                           'close, not identical')}
    bad = np.flatnonzero(~close)
    first = int(bad[0]) if bad.size else None
    return {'status': 'mismatch', 'method': 'curve_values',
            'n_points_original': int(aa.size),
            'n_points_rerun': int(bb.size),
            'n_nonfinite_original': n_a, 'n_nonfinite_rerun': n_b,
            'first_difference_index': first,
            'max_abs_diff': _repro_curve_max_abs_diff(aa, bb),
            'byte_identical': False,
            'detail': (f'curve differs at index {first}'
                       if first is not None else 'curve differs')}


def _repro_curve_stats_compare(a_stats, b_stats, *, rtol, atol):
    """Compare the scalar legacy summary when the arrays are unavailable."""
    common = [k for k in _REPRO_CURVE_STATS
              if a_stats.get(k) is not None and b_stats.get(k) is not None]
    if not common:
        return {'status': 'unverified', 'method': 'none',
                'detail': 'no comparable curve values or scalar summary'}
    problems = []
    for key in common:
        a, b = a_stats[key], b_stats[key]
        if key in ('curve_n_points', 'curve_n_nonfinite',
                   'curve_first_nonfinite_index'):
            # Counts and indexes are exact integers.  ``int(1.5)`` would
            # silently truncate a malformed summary into a match.
            if not (float(a).is_integer() and float(b).is_integer()):
                problems.append(f'{key} is not an integer ({a!r} vs {b!r})')
            elif int(a) != int(b):
                problems.append(f'{key} {int(a)} != {int(b)}')
            continue
        if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            problems.append(f'{key} {a!r} != {b!r}')
    if problems:
        return {'status': 'mismatch', 'method': 'summary_stats',
                'coarse': True, 'byte_identical': False,
                'detail': 'scalar curve summary differs: ' + '; '.join(problems)}
    return {'status': 'exact', 'method': 'summary_stats', 'coarse': True,
            'byte_identical': False,
            'detail': ('only the scalar curve summary is available; it '
                       'matches, but a pointwise trajectory comparison and '
                       'bit-identical claim require the array/sidecar')}


def _repro_curve_dimension(original, rerun, *, rtol, atol,
                           original_base=None, rerun_base=None):
    """Compare the recorded loss/eval trajectory, not only the final PPL.

    The final PPL is one scalar; two runs with different data order,
    nondeterministic kernels or a different resume point can coincidentally
    end at the same value.  A reproduction claim is therefore checked against
    the full recorded curve where available.  The dimension is deliberately
    optional for legacy records: missing curves make ``bit_identical``
    unverifiable, but they do not turn an otherwise matching endpoint into a
    failure.
    """
    a = _repro_curve_payload(original, original_base)
    b = _repro_curve_payload(rerun, rerun_base)
    out = {
        'name': 'curve', 'status': 'unverified', 'method': 'none',
        'kind': a['kind'] or b['kind'],
        'coarse': False, 'byte_identical': False,
        'n_points_original': (len(a['values']) if a['values'] is not None
                              else a['stats'].get('curve_n_points')),
        'n_points_rerun': (len(b['values']) if b['values'] is not None
                           else b['stats'].get('curve_n_points')),
        'sha256_original': a['sha256'] or a['ppl_sha256'],
        'sha256_rerun': b['sha256'] or b['ppl_sha256'],
        'n_nonfinite_original': a['stats'].get('curve_n_nonfinite'),
        'n_nonfinite_rerun': b['stats'].get('curve_n_nonfinite'),
        'digest_invalid': bool(a.get('digest_invalid')
                               or b.get('digest_invalid')),
        'first_difference_index': None, 'max_abs_diff': None,
        'detail': None,
    }
    problems = list(a['problems']) + list(b['problems'])

    def _detail(base):
        parts = [str(p) for p in problems if p]
        if base:
            parts.append(str(base))
        return '; '.join(parts)

    if problems:
        out['detail'] = _detail(None)

    # The writer records the same sidecar bytes under both digest fields.  A
    # record claiming one equal and one unequal digest is internally
    # contradictory; do not let the first field's "equal" hide that.
    if (a['sha256'] is not None and b['sha256'] is not None
            and a['ppl_sha256'] is not None and b['ppl_sha256'] is not None):
        if ((a['sha256'] == b['sha256'])
                != (a['ppl_sha256'] == b['ppl_sha256'])):
            out.update(status='mismatch', method='curve_sha256',
                       byte_identical=False,
                       detail=_detail(
                           'curve_sha256 and ppl_history_sha256 disagree '
                           'about whether the recorded sidecar is equal'))
            return out

    # A recorded sidecar digest is the strongest identity: equal digests mean
    # the exact serialized array (and its auxiliary diagnostics) is the same.
    for label, ha, hb in (
            ('curve_sha256', a['sha256'], b['sha256']),
            ('ppl_history_sha256', a['ppl_sha256'], b['ppl_sha256'])):
        if ha is None or hb is None:
            continue
        if ha == hb:
            out.update(status='exact', method=label, byte_identical=True,
                       detail=_detail(f'{label} is identical'))
            return out
        # Digests differ.  If both arrays are available a value comparison
        # can distinguish a formatting-only difference from a real drift;
        # otherwise the recorded bytes are the only evidence and they differ.
        if a['values'] is None or b['values'] is None:
            out.update(status='mismatch', method=label, byte_identical=False,
                       detail=_detail(
                           f'{label} differs and the curve payload is '
                           'not available for a numeric comparison'))
            return out
        break

    if a['values'] is not None and b['values'] is not None:
        if a['kind'] != b['kind']:
            out.update(status='unverified', method='none',
                       detail=_detail(
                           f'different trajectory kinds '
                           f'({a["kind"]!r} vs {b["kind"]!r})'))
            return out
        cmp = _repro_curve_values_compare(a['values'], b['values'],
                                          rtol=rtol, atol=atol)
        out.update(cmp)
        # A digest that no longer matches the bytes on disk is not identity
        # evidence.  Keep the numeric verdict, but never call the run
        # byte-identical on a stale/tampered sidecar.
        if out['digest_invalid']:
            out['byte_identical'] = False
            if out['status'] == 'exact':
                out['detail'] = _detail(
                    'numeric curve values are identical but a recorded '
                    'curve digest does not match the sidecar bytes')
            else:
                out['detail'] = _detail(out.get('detail'))
            return out
        # The sidecar digest covers more than the loss array (delta_trace /
        # ppl_history / opt_diag).  Equal numbers with unequal bytes are not
        # a bit-identical run, so keep the digest verdict separate.
        if (a['sha256'] is not None and b['sha256'] is not None
                and a['sha256'] != b['sha256']):
            out['byte_identical'] = False
            if out['status'] == 'exact':
                out['detail'] = _detail(
                    'numeric curve values are identical but the '
                    'serialized sidecar bytes differ')
        elif (a['ppl_sha256'] is not None and b['ppl_sha256'] is not None
                and a['ppl_sha256'] != b['ppl_sha256']):
            out['byte_identical'] = False
        elif ((a['sha256'] is None) != (b['sha256'] is None)
              or (a['ppl_sha256'] is None) != (b['ppl_sha256'] is None)):
            # One side carries a sidecar digest and the other does not (an
            # inline legacy curve or a summary-only record): numeric values
            # can still match, but full byte identity is unverified.
            out['byte_identical'] = False
        return out

    if (a['sha256'] is not None or b['sha256'] is not None
            or a['ppl_sha256'] is not None or b['ppl_sha256'] is not None):
        out.update(status='unverified', method='none',
                   detail=_detail(
                       'a recorded curve digest is present but the '
                       'curve payload is unavailable on at least one side'))
        return out

    if problems:
        # A reference or sidecar that cannot be read must not be silently
        # downgraded to a scalar-summary match: the caller asked to compare
        # the recorded trajectory, and that trajectory is unavailable.
        out.update(status='unverified', method='none',
                   detail=_detail('a curve reference or sidecar is unusable; '
                                  'the scalar summary is not substituted'))
        return out

    cmp = _repro_curve_stats_compare(a['stats'], b['stats'],
                                     rtol=rtol, atol=atol)
    out.update(cmp)
    out['detail'] = _detail(out.get('detail'))
    return out


def reproduction_diagnosis(original, rerun, *, rtol=1e-6, atol=0.0,
                           original_base=None, rerun_base=None):
    """Compare one recorded cell against an independent re-execution.

    Returns a dimension-by-dimension verdict.  ``reproduced`` is True only
    when every required identity dimension is a definite match AND the final
    metric is exact or within the declared tolerance.  A missing fingerprint
    is ``provenance_incomplete`` (never a pass), and a different environment
    or data slice is reported as its own failure mode rather than as generic
    numerical drift.

    The recorded loss/eval curve is compared too when available.  Equal final
    PPL is not evidence that the same trajectory was executed; a curve
    mismatch fails the reproduction, and ``bit_identical`` is only claimed
    when the curve (not just the endpoint) is identical.  Legacy records
    without curves remain comparable, but their ``bit_identical`` claim stays
    false because it cannot be verified.

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
            'curve_compared': False, 'curve_identical': False,
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
    curve = _repro_curve_dimension(
        original, rerun, rtol=rtol, atol=atol,
        original_base=original_base, rerun_base=rerun_base)
    curve['fields'] = [{
        'field': 'recorded_curve',
        'original': curve.get('sha256_original'),
        'rerun': curve.get('sha256_rerun'),
        'status': ('mismatch' if curve['status'] == 'mismatch'
                   else 'unverified' if curve['status'] == 'unverified'
                   else 'match'),
        'detail': curve.get('detail'),
    }]
    dimensions['curve'] = curve
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
    curve_mismatch = curve['status'] == 'mismatch'
    curve_unverified = curve['status'] == 'unverified'
    if curve_mismatch:
        problems.append(f'curve.recorded_curve: {curve.get("detail")}')
    elif curve_unverified:
        warnings.append(
            'curve.recorded_curve: no comparable loss/eval trajectory is '
            'recorded on both sides; the endpoint can still match, but '
            'bit_identical cannot be claimed')
    elif curve.get('coarse'):
        warnings.append(
            'curve.recorded_curve: only the scalar curve summary is available '
            '(no array/sidecar); it matches, but pointwise reproduction and '
            'bit_identical remain unverified')
    elif curve['status'] == 'within_tolerance':
        warnings.append(
            'curve.recorded_curve: the trajectories match only within the '
            'declared tolerance, so the run is not bit-identical')
    if metric and metric['status'] not in ('exact', 'within_tolerance'):
        problems.append(f'metric.ppl: {metric.get("detail")}')

    if mismatch:
        verdict = _REPRODUCTION_DIMENSION_VERDICTS[mismatch[0]]
        reproduced = False
    elif curve_mismatch:
        verdict = 'curve_changed'
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
    elif (metric['status'] == 'exact'
          and curve['status'] == 'exact' and not curve.get('coarse')
          and curve.get('byte_identical')):
        verdict = 'identical'
        reproduced = True
    elif (metric['status'] in ('exact', 'within_tolerance')
          and curve['status'] in ('exact', 'within_tolerance')
          and not curve.get('coarse')):
        # The endpoint and the full trajectory both reproduced, but at least
        # one needed the declared tolerance.
        verdict = 'within_tolerance'
        reproduced = True
    elif metric['status'] in ('exact', 'within_tolerance'):
        # The endpoint reproduced, but the trajectory is missing or only a
        # scalar summary is available.  That is a reproduced endpoint, not a
        # reproduced run.
        verdict = 'endpoint_only'
        reproduced = True
    else:
        verdict = 'numerical_drift'
        reproduced = False

    identity_ok = bool(not mismatch and not unverified)
    curve_identical = bool(
        curve['status'] == 'exact' and not curve.get('coarse')
        and curve.get('byte_identical'))
    return {
        'kind': 'reproduction_diagnosis', 'schema': REPRODUCTION_SCHEMA,
        'verdict': verdict, 'reproduced': bool(reproduced),
        'bit_identical': bool(identity_ok and metric['status'] == 'exact'
                              and curve_identical),
        'identity_ok': identity_ok,
        'curve_compared': bool(curve['status'] != 'unverified'),
        'curve_identical': curve_identical,
        'rtol': rtol, 'atol': atol,
        'dimensions': dimensions, 'metric': metric,
        'problems': problems,
        'warnings': warnings,
        'note': ('identity and recorded-trajectory comparison of two '
                 'independent executions; absence of a fingerprint or curve '
                 'is not evidence of a match'),
    }


def _repro_curve_base(label):
    """Directory a summary's ``curve_ref`` paths are relative to.

    ``_repro_load_summary`` returns the resolved summary path for a directory
    or file input and the caller's label for an in-memory mapping.  Only the
    former can locate sidecars; a mapping has no directory and must not borrow
    the current working directory (that would make the comparison depend on
    where the auditor happened to run it).
    """
    if not isinstance(label, str):
        return None
    path = os.path.abspath(label)
    if os.path.isfile(path):
        return os.path.dirname(path)
    if os.path.isdir(path):
        return path
    return None


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

    For directory/file inputs the cell's ``curve_ref`` sidecar is resolved
    against its own summary directory, so a recorded trajectory is compared
    even though the arrays were moved out of ``summary.json``.  A differing
    trajectory makes the cell (and therefore the panel) ``curve_changed``;
    ``bit_identical`` is reserved for panels whose recorded curves are also
    identical.
    """
    rtol = _repro_validate_tolerance('rtol', rtol)
    atol = _repro_validate_tolerance('atol', atol)
    a_summary, a_label = _repro_load_summary(original, 'original')
    b_summary, b_label = _repro_load_summary(rerun, 'rerun')
    a_base = _repro_curve_base(a_label)
    b_base = _repro_curve_base(b_label)
    a_cells, a_malformed = _repro_cells(a_summary)
    b_cells, b_malformed = _repro_cells(b_summary)
    common = sorted(set(a_cells) & set(b_cells))
    only_original = sorted(set(a_cells) - set(b_cells))
    only_rerun = sorted(set(b_cells) - set(a_cells))

    cells, counts, problems, warnings = {}, {}, [], []
    for key in common:
        diag = reproduction_diagnosis(a_cells[key], b_cells[key],
                                      rtol=rtol, atol=atol,
                                      original_base=a_base, rerun_base=b_base)
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
                   'environment_changed', 'initialization_changed',
                   'curve_changed')
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
        d['verdict'] == 'identical' and d.get('curve_identical')
        for d in cells.values()))
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
    n_curve_compared = sum(1 for d in cells.values() if d.get('curve_compared'))
    n_curve_identical = sum(1 for d in cells.values() if d.get('curve_identical'))
    n_curve_unverified = len(cells) - n_curve_compared
    n_curve_mismatch = sum(
        1 for d in cells.values()
        if (d.get('dimensions') or {}).get('curve', {}).get('status')
        == 'mismatch')
    if n_curve_mismatch:
        problems.append(f'{n_curve_mismatch} cells have a recorded curve that '
                        'differs from the original run')
    if n_curve_unverified:
        warnings.append(f'{n_curve_unverified} cells have no comparable '
                        'recorded curve; their endpoint may match, but the '
                        'trajectory (and bit_identical) is unverified')
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
        'n_curve_compared': n_curve_compared,
        'n_curve_identical': n_curve_identical,
        'n_curve_unverified': n_curve_unverified,
        'n_curve_mismatch': n_curve_mismatch,
        'cells': cells,
        'problems': problems,
        'warnings': warnings,
        'note': ('a reproduced verdict requires matching source bytes, data '
                 'slice, configuration, execution environment, matched '
                 'initialization, a metric inside the declared tolerance and a '
                 'recorded trajectory that does not differ; bit_identical '
                 'additionally requires the recorded curve to be identical. '
                 'Missing provenance or a missing curve is reported, not '
                 'assumed equal'),
    }


_REPRODUCTION_USAGE = """usage:
  python -m explib.reproduce compare ORIGINAL RERUN [options]
  python -m explib.reproduce report ORIGINAL.json RERUN.json [options]
  python -m explib.reproduce canary [--write] [--verify PATH] [options]

compare: two independent executions of the same cells (result directories or
summary.json paths).  Exit status: 0 reproduced, 1 not reproduced, 2 usage or
input error.  Each cell compares its recorded loss/eval curve as well as its
final PPL: an equal endpoint with a different trajectory is not a reproduced
run.  --require exact removes the tolerance and demands an identical PPL and an
identical recorded curve (``bit_identical``).

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

canary: record or replay the observed arithmetic identity of this process.
  The probe hashes fixed operator outputs; it is not a statistical test and
  does not prove that training is bitwise deterministic.
  --write                write analysis_v12/numerical_canary.json
  --out DIR              artifact directory for --write (default analysis_v12)
  --overwrite            replace a different existing artifact after review
  --verify PATH          replay a stored artifact (0 ok, 1 drift/fail, 2 n/a)
  --seed/--size/--dtype  probe parameters (defaults 0/8/float32)
  --device               cpu or cuda[:N] (default cpu; never guessed)
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
    if command in ('canary', 'numerical_canary', 'numeric_canary',
                   'arithmetic_canary'):
        return _nc_cli(args)
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
