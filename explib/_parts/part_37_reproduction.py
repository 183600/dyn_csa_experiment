
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
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (str, int, float)):
        text = str(value).strip()
        return text or None
    return None


def _repro_int(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number


def _repro_canonical(value):
    """Stable text for an arbitrary descriptor; used only for equality."""
    try:
        return json.dumps(_jsonable(value), sort_keys=True,
                          separators=(',', ':'), ensure_ascii=True,
                          allow_nan=False)
    except Exception:
        return repr(value)


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


def _repro_field(name, a, b, *, required=False):
    """Compare one optional/required identity field.

    ``required`` means the field must be present on both sides for a match;
    otherwise a both-absent field is treated as equal (legacy optional).
    """
    va, vb = _repro_text(a), _repro_text(b)
    if va is None and vb is None:
        return {'field': name, 'original': None, 'rerun': None,
                'status': 'unverified' if required else 'match',
                'detail': ('both records omit this required identity' if required
                           else 'both records omit this optional identity')}
    if va is None or vb is None:
        return {'field': name, 'original': va, 'rerun': vb,
                'status': 'unverified',
                'detail': 'only one record carries this identity'}
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


def _repro_code_dimension(original, rerun):
    def _ident(rec):
        prov = rec.get('provenance') if isinstance(rec.get('provenance'), dict) else {}
        code = prov.get('code') if isinstance(prov.get('code'), dict) else {}
        tree = code.get('source_tree') if isinstance(code.get('source_tree'), dict) else {}
        return {
            'source_sha256': _repro_text(tree.get('sha256')),
            'code_semantics': _repro_text(code.get('code_semantics')),
            'code_stamp': (_repro_text(rec.get('_code'))
                           or _repro_text(code.get('code_stamp'))),
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
    notes = []
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
            out[key] = _repro_text(value)
        return out

    a, b = _ident(original), _ident(rerun)
    fields, common, one_sided = [], [], []
    for key in keys:
        va, vb = a[key], b[key]
        if va is None and vb is None:
            continue
        if va is None or vb is None:
            one_sided.append(key)
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
        fields.append(_repro_field(key, original.get(key), rerun.get(key)))
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
        status = ('unverified'
                  if str(reason).startswith(('unverifiable', 'a record'))
                  or 'only one' in str(reason)
                  else 'mismatch')
        fields.append({'field': label,
                       'original': _repro_canonical(original.get(label)),
                       'rerun': _repro_canonical(rerun.get(label)),
                       'status': status, 'detail': str(reason)})
    return _repro_dimension('config', fields)


def _repro_environment_dimension(original, rerun):
    def _fp(rec):
        value = rec.get('exec_fp')
        if value is None and isinstance(rec.get('hyperparams'), dict):
            value = rec['hyperparams'].get('exec_fp')
        return _repro_text(value)

    return _repro_dimension('environment', [
        _repro_field('exec_fp', _fp(original), _fp(rerun), required=True)])


def _repro_initialization_dimension(original, rerun):
    def _fp(rec):
        value = rec.get('init_core_fp')
        if value is None and isinstance(rec.get('hyperparams'), dict):
            value = rec['hyperparams'].get('init_core_fp')
        return _repro_text(value)

    return _repro_dimension('initialization', [
        _repro_field('init_core_fp', _fp(original), _fp(rerun), required=True)])


def _repro_rng_dimension(original, rerun):
    """Diagnostic only, mirroring the library's rng_fp contract.

    ``rng_fp`` is recorded immediately after ``set_seed`` and is not part of
    the arm-pairing gate: a different architecture consumes a different amount
    of RNG.  For two executions of the SAME cell it is still a useful warning,
    so it is reported but does not by itself make the runs non-reproducible.
    """
    va = _repro_text(original.get('rng_fp'))
    vb = _repro_text(rerun.get('rng_fp'))
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
    if na is not None and nb is not None and na != nb:
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
    elif metric['status'] == 'denominator_mismatch':
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
    path = os.path.abspath(path)
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
    out = {}
    for key, rec in summary.items():
        if not isinstance(rec, dict):
            continue
        key = str(key)
        if '::seed' not in key:
            continue
        out[key] = rec
    return out


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
    a_cells, b_cells = _repro_cells(a_summary), _repro_cells(b_summary)
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

    if not common:
        verdict = 'not_comparable'
    elif any(v in ('code_changed', 'data_changed', 'config_changed',
                   'environment_changed', 'initialization_changed')
             for v in counts):
        verdict = 'not_reproduced'
    elif only_original or only_rerun:
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
        'only_original': only_original[:50],
        'only_rerun': only_rerun[:50],
        'counts': dict(sorted(counts.items())),
        'cells': cells,
        'problems': problems,
        'warnings': warnings,
        'note': ('a reproduced verdict requires matching source bytes, data '
                 'slice, configuration, execution environment, matched '
                 'initialization and a metric inside the declared tolerance; '
                 'missing provenance is reported, not assumed equal'),
    }


_REPRODUCTION_USAGE = """usage: python -m explib.reproduce compare ORIGINAL RERUN [options]

Compare two independent executions of the same cells (result directories or
summary.json paths).  Exit status: 0 reproduced, 1 not reproduced, 2 usage or
input error.  --require exact removes the tolerance and demands identical PPL.

options:
  --rtol FLOAT   relative PPL tolerance (default 1e-6)
  --atol FLOAT   absolute PPL tolerance (default 0)
  --json PATH    also write the machine-readable report
  --require {tolerance,exact}
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
