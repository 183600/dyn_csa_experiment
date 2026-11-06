# Audit gate: replay the curve-level simultaneous inference block/artifact.
#
# A by-length (or by-budget) table of pointwise p-values does not say whether
# the sign is uniform over the whole curve, whether the curve is flat, or
# whether every condition is practically within a declared margin.  That is a
# curve-level intersection claim, not the conjunction of row-wise tests.  This
# gate replays the timestamp-free block embedded in
# `analysis_v12/test_stats.json` (preferred) or a standalone
# `curve_inference.json` artifact.  It never trains, reads a split, or changes
# a frozen p/Holm/confirmed verdict.

import json
import os


_CV_GATE_DEFAULTS = (
    os.path.join('analysis_v12', 'curve_inference.json'),
    os.path.join('analysis_audit', 'curve_inference.json'),
    'curve_inference.json',
)

_cv_gate_load = _gate_load  # shared loader: unreadable/malformed == no evidence


def _cv_gate_repr(value, limit=160):
    try:
        text = repr(value)
    except Exception:
        return f'<unrepresentable {type(value).__name__}>'
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _cv_gate_path_text(value):
    try:
        return os.fsdecode(os.fspath(value))
    except Exception:
        return None


def _cv_gate_resolve(path):
    if path is not None:
        if isinstance(path, (str, os.PathLike)):
            try:
                return os.fspath(path)
            except Exception:
                return None
        return None
    for candidate in _CV_GATE_DEFAULTS:
        full = os.path.join(REPO, candidate)  # noqa: F821
        if os.path.exists(full):
            return full
    return None


def _cv_gate_result_summary(result):
    result = result if isinstance(result, dict) else {}
    flat = result.get('flatness') if isinstance(result.get('flatness'), dict) else {}
    uniform = result.get('uniform') if isinstance(result.get('uniform'), dict) else {}
    neg = uniform.get('negative') if isinstance(uniform.get('negative'), dict) else {}
    eq = uniform.get('equivalence') if isinstance(uniform.get('equivalence'), dict) else {}
    feq = flat.get('equivalence') if isinstance(flat.get('equivalence'), dict) else {}
    return {
        'result_status': result.get('status'),
        'method': result.get('method'),
        'n': result.get('n'),
        'm': (result.get('family') or {}).get('m')
        if isinstance(result.get('family'), dict) else None,
        'uniform_negative': neg.get('supported'),
        'uniform_negative_p': neg.get('p_iut'),
        'uniform_equivalence': eq.get('supported'),
        'uniform_equivalence_p': eq.get('p_tost'),
        'flatness_p': flat.get('p_range'),
        'flatness_equivalence': feq.get('supported'),
        'flatness_equivalence_p': feq.get('p_tost'),
        'resolved_conditions': result.get('resolved_conditions') or [],
        'warnings': list(result.get('warnings') or []),
    }


def _cv_gate_add_block(out, name, verdict):
    """Merge one validated block/artifact verdict into the gate result."""
    status = verdict.get('status')
    result = verdict.get('result') if isinstance(verdict.get('result'), dict) else None
    row = {'name': name, 'status': status}
    row.update(_cv_gate_result_summary(result))
    out['blocks'].append(row)
    for problem in (verdict.get('problems') or []):
        text = f'{name}: {problem}' if name else str(problem)
        if text not in out['problems']:
            out['problems'].append(text)
    for warning in (verdict.get('warnings') or []):
        text = f'{name}: {warning}' if name else str(warning)
        if text not in out['warnings']:
            out['warnings'].append(text)
    if status == 'fail':
        out['_fail'] = True
    elif status == 'not_estimable':
        out['_not_estimable'] = out.get('_not_estimable', 0) + 1
    elif status == 'warnings':
        out['_warnings'] = True
    elif status == 'ok':
        out['_ok'] = out.get('_ok', 0) + 1
    return row


def _cv_gate_embedded(test_stats, L, out):
    comps = test_stats.get('comparisons')
    if not isinstance(comps, dict):
        out['problems'].append(
            'test_stats.json has no comparisons object; the curve block '
            'cannot be located')
        return 0
    seen = 0
    for name in sorted(comps):
        comp = comps.get(name)
        if not isinstance(comp, dict):
            continue
        block = comp.get('length_curve_inference')
        if block is None:
            continue
        seen += 1
        verdict = L.validate_curve_inference_block(block)
        _cv_gate_add_block(out, name, verdict)
    return seen


def audit_curve_inference(path=None, *, verbose=True):
    """Replay a curve-level simultaneous inference block or artifact.

    Status semantics:
      * ``ok`` — every block/artifact hash/replay agrees and names no caveat;
      * ``warnings`` — it replays, but a block is underpowered,
        resolution-limited, an approximation, or carries another caveat;
      * ``not_applicable`` — a test_stats.json exists but contains no curve
        block (the optional analysis was not run);
      * ``not_verifiable_locally`` — no local block/artifact, or the block is
        not estimable;
      * ``fail`` — malformed/tampered block/artifact or a replay mismatch.
    """
    L = _import_exp_lib()  # noqa: F821
    out = {
        'kind': 'curve_inference_audit',
        'status': 'not_verifiable_locally',
        'path': None,
        'mode': None,
        'blocks': [],
        'n_blocks': 0,
        'problems': [],
        'warnings': [],
        'note': ('curve-level simultaneous inference: exact sign-flip max-T '
                 'over conditions, a range test of flatness, and IUT '
                 'one-sided/TOST claims.  It does not change a frozen '
                 'p/Holm/verified/confirmed verdict.'),
    }
    resolved = None
    explicit = path is not None
    if explicit:
        resolved = _cv_gate_resolve(path)
        if resolved is None:
            out['status'] = 'fail'
            out['problems'].append(
                'path is not a usable filesystem path: '
                f'{_cv_gate_repr(path)}')
            if verbose:
                print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
            return out
    else:
        # Prefer the embedded confirmation block: it is the object a reader
        # of REPORT_v12 can actually see.  Fall back to a standalone artifact.
        ts = os.path.join(REPO, 'analysis_v12', 'test_stats.json')  # noqa: F821
        if os.path.exists(ts):
            resolved = ts
        else:
            resolved = _cv_gate_resolve(None)
    if resolved is None:
        # This is an optional curve-level analysis, not a required gate.  A
        # source-only checkout that never ran the analysis is `not_applicable`
        # (like the other optional design blocks), not a failed verification.
        out['status'] = 'not_applicable'
        out['note'] = ('no analysis_v12/test_stats.json curve block and no '
                       'curve_inference.json artifact were found; the optional '
                       'curve-level analysis was not run')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
        return out
    text = _cv_gate_path_text(resolved)
    if text is None:
        out['status'] = 'fail'
        out['problems'].append(
            'resolved path is not usable: ' + _cv_gate_repr(resolved))
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
        return out
    resolved = text
    out['path'] = resolved
    if not os.path.exists(resolved):
        out['problems'].append(f'{resolved}: artifact does not exist')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
        return out
    data = _cv_gate_load(resolved)  # noqa: F821
    if not isinstance(data, dict):
        out['status'] = 'fail'
        out['problems'].append(f'{resolved}: not a readable JSON object')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
        return out
    if data.get('kind') == 'curve_inference':
        out['mode'] = 'artifact'
        verdict = L.validate_curve_inference(data)
        _cv_gate_add_block(out, None, verdict)
        out['n_blocks'] = 1
    elif data.get('kind') == 'curve_inference_block':
        out['mode'] = 'block'
        verdict = L.validate_curve_inference_block(data)
        _cv_gate_add_block(out, None, verdict)
        out['n_blocks'] = 1
    elif isinstance(data.get('comparisons'), dict):
        out['mode'] = 'test_stats'
        seen = _cv_gate_embedded(data, L, out)
        out['n_blocks'] = seen
        if seen == 0 and not explicit:
            # A newer standalone artifact may exist next to an older
            # test_stats.json that predates the embedded block.  Verify the
            # artifact instead of reporting the optional analysis as absent.
            fallback = _cv_gate_resolve(None)
            if fallback is not None and os.path.abspath(fallback) != os.path.abspath(resolved):
                return audit_curve_inference(fallback, verbose=verbose)
        if seen == 0:
            out['status'] = 'not_applicable'
            out['problems'].append(
                f'{resolved}: no comparison carries a '
                'length_curve_inference block')
            if verbose:
                print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
            return out
    else:
        out['status'] = 'fail'
        out['problems'].append(
            f'{resolved}: not a curve_inference artifact, block, or '
            'test_stats report')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
        return out
    # A not_estimable block is not a pass, but it is also not a tamper
    # failure: report it explicitly and keep the remaining blocks visible.
    if out.get('_fail'):
        out['status'] = 'fail'
    elif out.get('_ok', 0) == 0 and out.get('_not_estimable', 0):
        out['status'] = 'not_verifiable_locally'
        out['problems'].append(
            'every curve block/artifact is not estimable; there is no '
            'verifiable curve-level reading')
    elif out.get('_warnings'):
        out['status'] = 'warnings'
    else:
        out['status'] = 'ok'
    out.pop('_fail', None)
    out.pop('_not_estimable', None)
    out.pop('_warnings', None)
    out.pop('_ok', None)
    if verbose:
        print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
    return out
