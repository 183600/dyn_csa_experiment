# Audit gate: data-blind split-half consistency across evaluation units.
#
# The seed-level exact test can be significant while the effect is carried by
# one half of the evaluation units.  ``unit_split_replication`` recomputes the
# per-seed token-weighted log-PPL effect in two deterministic disjoint halves
# and forms the both-halves intersection-union p.  This gate replays either a
# standalone artifact or the nested blocks in a final-test stats file.  It
# never changes a p/Holm/confirmed verdict, reads no split, and trains nothing.

import json
import os


_USR_ARTIFACT_DEFAULTS = (
    os.path.join('analysis_v12', 'unit_split_replication.json'),
    os.path.join('analysis_audit', 'unit_split_replication.json'),
    'unit_split_replication.json',
)
_USR_STATS_DEFAULTS = (
    os.path.join('analysis_v12', 'test_stats.json'),
    os.path.join('analysis_v12', 'stats.json'),
)
_USR_BLOCK_FIELD = 'unit_split_replication'


def _usr_gate_key_text(value, limit=160):
    """Bounded key text; huge integer dict keys are diagnostics, not crashes."""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = str(value)
        except Exception:
            return _usr_gate_repr(value, limit=limit)
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _usr_gate_path_text(value):
    """Path text or ``None``; shared rule in :func:`_gate_path_text`."""
    return _gate_path_text(value)


def _usr_gate_resolve(path, defaults):
    if path is not None:
        if isinstance(path, (str, os.PathLike)):
            return _usr_gate_path_text(path)
        return None
    for rel in defaults:
        try:
            candidate = os.path.join(REPO, rel)  # noqa: F821
            if os.path.exists(candidate):
                return candidate
        except (OSError, ValueError, TypeError):
            continue
    return None


def _usr_gate_repr(value, limit=200):
    try:
        text = repr(value)
    except Exception:
        return f'<unrepresentable {type(value).__name__}>'
    return text if len(text) <= limit else text[:limit] + '...'


def _usr_gate_row(result):
    if not isinstance(result, dict):
        return {'status': None}
    halves = result.get('halves')
    halves = halves if isinstance(halves, list) else []
    test = [((h.get('test') if isinstance(h, dict) else None) or {})
            for h in halves]
    stats = [((h.get('stats') if isinstance(h, dict) else None) or {})
             for h in halves]
    iut = result.get('iut')
    iut = iut if isinstance(iut, dict) else {}
    sens = result.get('split_sensitivity')
    sens = sens if isinstance(sens, dict) else {}
    warnings = result.get('warnings')
    warnings = warnings if isinstance(warnings, list) else []
    return {
        'status': result.get('status'),
        'direction': result.get('direction'),
        'n_units': result.get('n_units'),
        'n_seeds': result.get('n_seeds'),
        'p_half_0': (test[0].get('p_one_sided') if test else None),
        'p_half_1': (test[1].get('p_one_sided') if len(test) > 1 else None),
        'mean_half_0': (stats[0].get('mean') if stats else None),
        'mean_half_1': (stats[1].get('mean') if len(stats) > 1 else None),
        'p_iut': iut.get('p_iut'),
        'claim': iut.get('claim'),
        'sign_agreement': result.get('sign_agreement'),
        'split_sensitivity_pass': sens.get('n_pass'),
        'split_sensitivity_estimable': sens.get('n_estimable'),
        'warnings': [w for w in warnings if isinstance(w, str)],
    }


def _usr_gate_add_row(out, name, verdict):
    status = verdict.get('status')
    result = verdict.get('result') if isinstance(verdict.get('result'), dict) else None
    row = {'name': name, 'status': status}
    row.update(_usr_gate_row(result))
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
    elif status == 'not_applicable':
        out['_not_applicable'] = out.get('_not_applicable', 0) + 1
    elif status == 'warnings':
        out['_warnings'] = True
    elif status == 'ok':
        out['_ok'] = out.get('_ok', 0) + 1
    if isinstance(result, dict):
        sens = result.get('split_sensitivity') or {}
        n_pass = sens.get('n_pass')
        n_est = sens.get('n_estimable')
        if (isinstance(n_pass, int) and isinstance(n_est, int) and n_est > 0
                and n_pass < n_est):
            out['n_split_unstable'] = out.get('n_split_unstable', 0) + 1
    return row


def _usr_gate_finish(out):
    if out.get('_fail'):
        out['status'] = 'fail'
    elif out.get('_warnings'):
        out['status'] = 'warnings'
    elif out.get('_not_estimable'):
        out['status'] = 'not_verifiable_locally'
        if not any('not estimable' in str(w) for w in out.get('warnings') or []):
            out.setdefault('warnings', []).append(
                'one or more split-half blocks are not estimable (too few '
                'units/seeds, no direction, or a build error); no both-halves '
                'claim was replayed')
    elif out.get('_ok'):
        out['status'] = 'ok'
    else:
        out['status'] = 'not_applicable'
    for key in ('_fail', '_warnings', '_not_estimable', '_ok', '_not_applicable'):
        out.pop(key, None)
    return out


def _usr_gate_scan_stats(path, out):
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
        ft = data.get('finaltest')
        comps = (ft.get('comparisons') if isinstance(ft, dict) else None)
    if not isinstance(comps, dict):
        out['status'] = 'not_applicable'
        out['warnings'].append(
            f'{path} has no comparison table; no split-half block to replay')
        return out
    L = _import_exp_lib()  # noqa: F821
    found = 0
    for name in sorted(comps, key=_usr_gate_key_text):
        rec = comps.get(name)
        if not isinstance(rec, dict) or _USR_BLOCK_FIELD not in rec:
            continue
        block = rec.get(_USR_BLOCK_FIELD)
        found += 1
        label = f'{_usr_gate_key_text(name)}.{_USR_BLOCK_FIELD}'
        if not isinstance(block, dict):
            out['blocks'].append({
                'name': label, 'status': 'fail',
                'reason': 'the stored block is not an object',
            })
            out['_fail'] = True
            out['problems'].append(f'{label}: the stored block is not an object')
            continue
        if block.get('kind') == 'unit_split_replication_error':
            status = block.get('status')
            reason = block.get('reason')
            out['blocks'].append({
                'name': label,
                'status': (status if status in ('not_estimable', 'not_applicable')
                           else 'not_estimable'),
                'reason': (reason if isinstance(reason, str)
                           else _usr_gate_repr(reason)),
            })
            if block.get('status') == 'not_applicable':
                out['_not_applicable'] = out.get('_not_applicable', 0) + 1
            else:
                out['_not_estimable'] = out.get('_not_estimable', 0) + 1
            continue
        if block.get('kind') == 'unit_split_replication_block':
            verdict = L.validate_unit_split_replication_block(block)
        elif block.get('kind') == 'unit_split_replication':
            verdict = L.validate_unit_split_replication(block)
        else:
            verdict = {
                'status': 'fail',
                'problems': [f'unexpected block kind {_usr_gate_repr(block.get("kind"))}'],
                'warnings': [],
                'result': None,
            }
        _usr_gate_add_row(out, label, verdict)
    if not found:
        out['status'] = 'not_applicable'
        out['warnings'].append(
            f'{path} has no unit_split_replication block; the artifact '
            'predates the split-half readout (a legacy artifact is not a '
            'failed replay)')
        return out
    out['mode'] = 'stats'
    out['path'] = path
    out['n_blocks'] = found
    return out


def audit_unit_split_replication(path=None, *, verbose=True):
    """Replay a standalone split-half artifact or the nested v12 blocks.

    Resolution order: an explicit path; a standalone
    ``analysis_v12/unit_split_replication.json``; the final-test stats
    artifact's nested ``unit_split_replication`` blocks.  Missing/legacy
    artifacts are ``not_applicable``; a tampered or non-recomputable block is
    ``fail``.
    """
    out = {
        'kind': 'unit_split_replication_audit',
        'schema': 1,
        'status': 'ok',
        'mode': None,
        'path': None,
        'n_blocks': 0,
        'n_split_unstable': 0,
        'blocks': [],
        'problems': [],
        'warnings': [],
        'note': ('data-blind split-half intersection-union replay; it reads '
                 'the same per-seed per-unit log ratios and never changes '
                 'p_exact_signflip, p_holm, verified or confirmed'),
    }
    if path is not None:
        resolved = _usr_gate_resolve(path, ())
        if resolved is None:
            # A caller-supplied path is untrusted too: ``os.path.exists``
            # raises for a huge integer and an arbitrary object is not
            # JSON-serialisable in the report.  Fail closed with a bounded
            # diagnostic before touching the filesystem.
            out['status'] = 'fail'
            out['path'] = None
            out['problems'].append(
                'path is not a usable filesystem path: '
                f'{_usr_gate_repr(path)}')
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
                f'{type(exc).__name__}: {_usr_gate_repr(exc)}')
            if verbose:
                print(json.dumps(out, ensure_ascii=False, indent=2)[:5000])
            return out
        if not exists:
            out['status'] = 'not_verifiable_locally'
            out['reason'] = 'the explicit split-half path does not exist'
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
                'unit_split_replication', 'unit_split_replication_block'):
            L = _import_exp_lib()  # noqa: F821
            if data.get('kind') == 'unit_split_replication':
                verdict = L.validate_unit_split_replication(data)
            else:
                verdict = L.validate_unit_split_replication_block(data)
            out['mode'] = 'artifact'
            out['path'] = resolved
            out['n_blocks'] = 1
            _usr_gate_add_row(out, os.path.basename(resolved), verdict)
            _usr_gate_finish(out)
            if verbose:
                print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
            return out
        if isinstance(data, dict) and (
                'comparisons' in data or 'finaltest' in data):
            _usr_gate_scan_stats(resolved, out)
            _usr_gate_finish(out)
            if verbose:
                print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
            return out
        out['status'] = 'fail'
        out['problems'].append(
            f'{resolved} is not a unit_split_replication artifact, block, or '
            'final-test stats file')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:5000])
        return out
    stats_path = _usr_gate_resolve(None, _USR_STATS_DEFAULTS)
    if stats_path is not None:
        data = _gate_load(stats_path)  # noqa: F821
        if isinstance(data, dict):
            comps = data.get('comparisons')
            if not isinstance(comps, dict):
                ft = data.get('finaltest')
                comps = (ft.get('comparisons') if isinstance(ft, dict) else None)
            has_block = (isinstance(comps, dict) and any(
                isinstance(rec, dict) and _USR_BLOCK_FIELD in rec
                for rec in comps.values()))
            if has_block:
                _usr_gate_scan_stats(stats_path, out)
                _usr_gate_finish(out)
                if verbose:
                    print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
                return out
    artifact_path = _usr_gate_resolve(None, _USR_ARTIFACT_DEFAULTS)
    if artifact_path is not None:
        data = _gate_load(artifact_path)  # noqa: F821
        if data is not None:
            L = _import_exp_lib()  # noqa: F821
            verdict = (L.validate_unit_split_replication_block(data)
                       if data.get('kind') == 'unit_split_replication_block'
                       else L.validate_unit_split_replication(data))
            out['mode'] = 'artifact'
            out['path'] = artifact_path
            out['n_blocks'] = 1
            _usr_gate_add_row(out, os.path.basename(artifact_path), verdict)
            _usr_gate_finish(out)
            if verbose:
                print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
            return out
    out['status'] = 'not_applicable'
    out['reason'] = ('no test_stats split-half block and no standalone '
                     'unit_split_replication.json artifact')
    if verbose:
        print(json.dumps(out, ensure_ascii=False, indent=2)[:5000])
    return out


def audit_unit_split_replication_artifact(path, *, verbose=True):
    """Replay an explicit standalone split-half artifact path."""
    return audit_unit_split_replication(path, verbose=verbose)
