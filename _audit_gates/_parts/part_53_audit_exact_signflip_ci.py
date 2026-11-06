# Audit gate: exact sign-flip confidence set replay.
#
# The confirmation test is an exact sign-flip test, but the interval shown
# beside it has historically been a Student-t interval.  The two can disagree
# about zero.  ``exact_signflip_ci`` adds a confidence set obtained by
# inverting the SAME sign-flip test; this gate replays either a standalone
# artifact or the nested blocks in a final-test stats file.  It never changes
# a p/Holm/confirmed verdict, reads no split, and trains nothing.

import json
import os


_ESCI_ARTIFACT_DEFAULTS = (
    os.path.join('analysis_v12', 'exact_signflip_ci.json'),
    os.path.join('analysis_audit', 'exact_signflip_ci.json'),
    'exact_signflip_ci.json',
)
_ESCI_STATS_DEFAULTS = (
    os.path.join('analysis_v12', 'test_stats.json'),
    os.path.join('analysis_v12', 'stats.json'),
)
_ESCI_BLOCK_FIELDS = ('exact_signflip_ci', 'exact_signflip_ci_log_ratio')


def _esci_gate_key_text(value, limit=160):
    """Bounded key text; huge integer dict keys are diagnostics, not crashes."""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = str(value)
        except Exception:
            return _esci_gate_repr(value, limit=limit)
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _esci_gate_path_text(value):
    """Filesystem path as text, or ``None`` when the value is not usable."""
    try:
        return os.fsdecode(os.fspath(value))
    except Exception:
        return None


def _esci_gate_resolve(path, defaults):
    if path is not None:
        if isinstance(path, (str, os.PathLike)):
            return _esci_gate_path_text(path)
        return None
    for rel in defaults:
        try:
            candidate = os.path.join(REPO, rel)  # noqa: F821
            if os.path.exists(candidate):
                return candidate
        except (OSError, ValueError, TypeError):
            continue
    return None


def _esci_gate_repr(value, limit=200):
    try:
        text = repr(value)
    except Exception:
        return f'<{type(value).__name__}>'
    return text if len(text) <= limit else text[:limit] + '...'


def _esci_gate_row(value):
    """Compact, bounded summary of one result for the gate output."""
    if not isinstance(value, dict):
        return {'status': None}
    zero = value.get('zero_relation')
    zero = zero if isinstance(zero, dict) else {}
    t_int = value.get('t_interval')
    t_int = t_int if isinstance(t_int, dict) else {}
    return {
        'status': value.get('status'),
        'n': value.get('n'),
        'ci': value.get('ci'),
        'intervals': value.get('intervals'),
        'unbounded': value.get('unbounded'),
        'connected': value.get('connected'),
        'degenerate': value.get('degenerate'),
        'p_at_zero': value.get('p_at_zero'),
        'contains_zero': value.get('contains_zero'),
        'zero_relation': zero.get('relation'),
        'consistent_with_p': zero.get('consistent_with_p'),
        't_ci': t_int.get('ci'),
        'warnings': list(value.get('warnings') or []),
    }


def _esci_gate_add_row(out, name, verdict):
    status = verdict.get('status')
    result = verdict.get('result') if isinstance(verdict.get('result'), dict) else None
    row = {'name': name, 'status': status}
    row.update(_esci_gate_row(result))
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
    if isinstance(result, dict):
        rel = (result.get('zero_relation') or {}).get('relation')
        if rel in ('t_excludes_exact_includes', 't_includes_exact_excludes'):
            out['n_t_exact_disagreements'] = out.get('n_t_exact_disagreements', 0) + 1
    return row


def _esci_gate_finish(out):
    if out.get('_fail'):
        out['status'] = 'fail'
    elif out.get('_warnings'):
        out['status'] = 'warnings'
    elif out.get('_not_estimable'):
        # The aggregate audit-report status vocabulary treats
        # ``not_estimable`` as a local verification limitation, not as a clean
        # pass.  Keep the child gate in that vocabulary so the aggregate does
        # not silently ignore it.
        out['status'] = 'not_verifiable_locally'
        if not any('not estimable' in str(w) for w in out.get('warnings') or []):
            out.setdefault('warnings', []).append(
                'one or more exact sign-flip confidence blocks are not '
                'estimable (resolution limit, n above the exact cap, or a '
                'build error); no finite interval was replayed')
    elif out.get('_ok'):
        out['status'] = 'ok'
    else:
        out['status'] = 'not_applicable'
    for key in ('_fail', '_warnings', '_not_estimable', '_ok'):
        out.pop(key, None)
    return out


def _esci_gate_scan_stats(path, out):
    data = _gate_load(path)  # noqa: F821
    if data is None:
        out['status'] = 'fail'
        out['problems'].append(f'{path} exists but is not a readable JSON object')
        return out
    if not isinstance(data, dict):
        out['status'] = 'fail'
        out['problems'].append(f'{path} is not a JSON object')
        return out
    comps = data.get('comparisons')
    if not isinstance(comps, dict):
        # A final-test artifact may nest the comparisons under `finaltest`.
        ft = data.get('finaltest')
        comps = (ft.get('comparisons') if isinstance(ft, dict) else None)
    if not isinstance(comps, dict):
        out['status'] = 'not_applicable'
        out['warnings'].append(
            f'{path} has no comparison table; no exact sign-flip CI block to replay')
        return out
    L = _import_exp_lib()  # noqa: F821
    found = 0
    for name in sorted(comps, key=_esci_gate_key_text):
        rec = comps.get(name)
        if not isinstance(rec, dict):
            continue
        for field in _ESCI_BLOCK_FIELDS:
            if field not in rec:
                continue
            block = rec.get(field)
            found += 1
            label = f'{_esci_gate_key_text(name)}.{field}'
            if not isinstance(block, dict):
                out['blocks'].append({
                    'name': label,
                    'status': 'fail',
                    'reason': 'the stored block is not an object',
                })
                out['_fail'] = True
                out['problems'].append(
                    f'{label}: the stored block is not an object')
                continue
            if block.get('kind') == 'exact_signflip_ci_error':
                out['blocks'].append({
                    'name': label,
                    'status': 'not_estimable',
                    'reason': block.get('reason'),
                })
                out['_not_estimable'] = out.get('_not_estimable', 0) + 1
                continue
            if block.get('kind') == 'exact_signflip_ci_block':
                verdict = L.validate_exact_signflip_ci_block(block)
            elif block.get('kind') == 'exact_signflip_ci':
                verdict = L.validate_exact_signflip_ci(block)
            else:
                verdict = {
                    'status': 'fail',
                    'problems': [f'unexpected block kind {_esci_gate_repr(block.get("kind"))}'],
                    'warnings': [],
                    'result': None,
                }
            _esci_gate_add_row(out, label, verdict)
    if not found:
        out['status'] = 'not_applicable'
        out['warnings'].append(
            f'{path} has no exact_signflip_ci block; the artifact predates the '
            'exact confidence set (a legacy artifact is not a failed replay)')
        return out
    out['mode'] = 'stats'
    out['path'] = path
    out['n_blocks'] = found
    return out


def audit_exact_signflip_ci(path=None, *, verbose=True):
    """Replay a standalone exact-CI artifact or the nested v12 blocks.

    Resolution order: an explicit path; a standalone
    ``analysis_v12/exact_signflip_ci.json``; the final-test stats artifact's
    nested ``exact_signflip_ci`` / ``exact_signflip_ci_log_ratio`` blocks.
    Missing/legacy artifacts are ``not_applicable``; a tampered or
    non-recomputable block is ``fail``.
    """
    out = {
        'kind': 'exact_signflip_ci_audit',
        'schema': 1,
        'status': 'ok',
        'mode': None,
        'path': None,
        'n_blocks': 0,
        'n_t_exact_disagreements': 0,
        'blocks': [],
        'problems': [],
        'warnings': [],
        'note': ('exact sign-flip test inversion replay; it reads the same '
                 'per-seed deltas and never changes p_exact_signflip, p_holm, '
                 'verified or confirmed'),
    }
    if path is not None:
        resolved = _esci_gate_resolve(path, ())
        if resolved is None:
            # A caller-supplied path is untrusted too: ``os.path.exists``
            # raises for a huge integer and an arbitrary object is not
            # JSON-serialisable in the report.  Fail closed with a bounded
            # diagnostic before touching the filesystem.
            out['status'] = 'fail'
            out['path'] = None
            out['problems'].append(
                'path is not a usable filesystem path: '
                f'{_esci_gate_repr(path)}')
            if verbose:
                print(json.dumps(out, ensure_ascii=False, indent=2)[:5000])
            return out
        out['path'] = resolved
        try:
            exists = os.path.exists(resolved)
        except (OSError, ValueError, TypeError) as exc:
            out['status'] = 'fail'
            out['problems'].append(
                f'path is not usable for filesystem access: '
                f'{type(exc).__name__}: {_esci_gate_repr(exc)}')
            if verbose:
                print(json.dumps(out, ensure_ascii=False, indent=2)[:5000])
            return out
        if not exists:
            out['status'] = 'not_verifiable_locally'
            out['reason'] = 'the explicit exact-CI path does not exist'
            if verbose:
                print(json.dumps(out, ensure_ascii=False, indent=2)[:5000])
            return out
        data = _gate_load(resolved)  # noqa: F821
        if data is None:
            out['status'] = 'fail'
            out['problems'].append(
                f'{resolved} exists but is not a readable JSON object')
            if verbose:
                print(json.dumps(out, ensure_ascii=False, indent=2)[:5000])
            return out
        if isinstance(data, dict) and data.get('kind') in (
                'exact_signflip_ci', 'exact_signflip_ci_block'):
            L = _import_exp_lib()  # noqa: F821
            if data.get('kind') == 'exact_signflip_ci':
                verdict = L.validate_exact_signflip_ci(data)
            else:
                verdict = L.validate_exact_signflip_ci_block(data)
            out['mode'] = 'artifact'
            out['path'] = resolved
            out['n_blocks'] = 1
            _esci_gate_add_row(out, os.path.basename(resolved), verdict)
            _esci_gate_finish(out)
            if verbose:
                print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
            return out
        if isinstance(data, dict) and (
                'comparisons' in data or 'finaltest' in data):
            _esci_gate_scan_stats(resolved, out)
            _esci_gate_finish(out)
            if verbose:
                print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
            return out
        out['status'] = 'fail'
        out['problems'].append(
            f'{resolved} is not an exact_signflip_ci artifact, block, or '
            'final-test stats file')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:5000])
        return out
    # Prefer the embedded confirmation blocks: they are the objects a reader
    # of REPORT_v12 can actually see.  Fall back to a standalone artifact.
    stats_path = _esci_gate_resolve(None, _ESCI_STATS_DEFAULTS)
    if stats_path is not None:
        data = _gate_load(stats_path)  # noqa: F821
        if isinstance(data, dict):
            comps = data.get('comparisons')
            if not isinstance(comps, dict):
                ft = data.get('finaltest')
                comps = (ft.get('comparisons') if isinstance(ft, dict) else None)
            has_block = (isinstance(comps, dict) and any(
                isinstance(rec, dict)
                and any(field in rec for field in _ESCI_BLOCK_FIELDS)
                for rec in comps.values()))
            if has_block:
                _esci_gate_scan_stats(stats_path, out)
                _esci_gate_finish(out)
                if verbose:
                    print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
                return out
    artifact_path = _esci_gate_resolve(None, _ESCI_ARTIFACT_DEFAULTS)
    if artifact_path is not None:
        data = _gate_load(artifact_path)  # noqa: F821
        if data is not None:
            L = _import_exp_lib()  # noqa: F821
            verdict = (L.validate_exact_signflip_ci_block(data)
                       if data.get('kind') == 'exact_signflip_ci_block'
                       else L.validate_exact_signflip_ci(data))
            out['mode'] = 'artifact'
            out['path'] = artifact_path
            out['n_blocks'] = 1
            _esci_gate_add_row(out, os.path.basename(artifact_path), verdict)
            _esci_gate_finish(out)
            if verbose:
                print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
            return out
    if stats_path is not None:
        _esci_gate_scan_stats(stats_path, out)
        _esci_gate_finish(out)
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
        return out
    out['status'] = 'not_verifiable_locally'
    out['reason'] = ('no standalone exact_signflip_ci artifact and no final-test '
                     'stats artifact with an exact-CI block')
    if verbose:
        print(json.dumps(out, ensure_ascii=False, indent=2)[:5000])
    return out


def audit_exact_signflip_ci_artifact(path=None, *, verbose=True):
    """Narrow companion: replay one standalone exact-CI artifact."""
    p = _esci_gate_resolve(path, _ESCI_ARTIFACT_DEFAULTS)
    if p is None or not os.path.exists(p):
        out = {'kind': 'exact_signflip_ci_validation',
               'status': 'not_verifiable_locally', 'path': p,
               'problems': ['no stored exact sign-flip CI artifact'],
               'warnings': []}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    data = _gate_load(p)  # noqa: F821
    L = _import_exp_lib()  # noqa: F821
    out = (L.validate_exact_signflip_ci_block(data)
           if isinstance(data, dict)
           and data.get('kind') == 'exact_signflip_ci_block'
           else L.validate_exact_signflip_ci(data))
    out['path'] = p
    print(json.dumps(out, ensure_ascii=False, indent=2)[:5000])
    return out
