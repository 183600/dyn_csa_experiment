# Independent-run reproduction audit (zero GPU).
#
# The implementation is intentionally a thin wrapper around
# exp_lib.compare_reproductions: the dimension decisions live next to the
# statistical helpers, while this file only maps the verdict onto the audit
# gate's ok/fail/not_verifiable vocabulary.


def audit_reproduction(original=None, rerun=None, *, rtol=1e-6, atol=0.0):
    """Compare two independent executions of the same panel cells.

    A "reproduced" claim is only quotable when the two runs share source
    bytes, data slice, configuration, execution environment and matched
    initialization, and the final PPL is inside the declared tolerance.
    Missing provenance is reported as ``not_verifiable_locally``, never as a
    pass.  This gate is read-only and never touches the test split.
    """
    L = _import_exp_lib()
    out = {
        'kind': 'reproduction_audit',
        'status': 'not_verifiable_locally',
        'original': original,
        'rerun': rerun,
        'rtol': rtol,
        'atol': atol,
        'verdict': None,
        'report': None,
        'problems': [],
        'warnings': [],
        'note': '',
    }
    if not original or not rerun:
        out['note'] = ('two result directories (or summary.json paths) are '
                       'required; this audit never guesses which pair of runs '
                       'is a replication')
        print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out
    try:
        report = L.compare_reproductions(original, rerun, rtol=rtol, atol=atol)
    except (OSError, ValueError, TypeError) as exc:
        out['status'] = 'fail'
        out['problems'].append(f'{type(exc).__name__}: {exc}')
        out['note'] = 'the comparison could not be performed'
        print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out
    out['report'] = report
    out['verdict'] = report.get('verdict')
    out['reproduced'] = bool(report.get('reproduced'))
    out['bit_identical'] = bool(report.get('bit_identical'))
    out['n_compared'] = report.get('n_compared')
    out['counts'] = report.get('counts') or {}
    out['problems'] = list(report.get('problems') or [])
    out['warnings'] = list(report.get('warnings') or [])
    if report.get('verdict') == 'reproduced':
        out['status'] = 'ok'
        out['note'] = ('every matched cell matches on source/data/config/'
                       'environment/initialization and the final metric is '
                       'inside the declared tolerance')
    elif report.get('verdict') in ('not_reproduced', 'drift', 'not_comparable'):
        out['status'] = 'fail'
        out['note'] = ('the rerun is not a reproduction under the declared '
                       'tolerance; see the dimension-level report')
    else:
        out['status'] = 'not_verifiable_locally'
        out['note'] = ('the comparison is incomplete: missing cells or '
                       'provenance mean the runs cannot be called a '
                       'reproduction either way')
    print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
    return out
