# Audit gate: known-truth rehearsal of the frozen decision path.
#
# The rehearsal itself lives in ``explib`` so it can be called from
# ``confirmation plan`` before any training.  This gate either validates a
# stored ``analysis_rehearsal.json`` (or the copy embedded in a
# ``plan_snapshot.json``) or runs the zero-GPU rehearsal live against the current
# frozen manifest.  Missing evidence is ``not_verifiable``, never green.

import json
import os


_ANALYSIS_REHEARSAL_DEFAULTS = (
    os.path.join('analysis_v12', 'analysis_rehearsal.json'),
    'analysis_rehearsal.json',
)


def _ar_gate_read_json(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def _ar_gate_embedded(obj):
    """Return a rehearsal object from a raw artifact or a plan snapshot."""
    if not isinstance(obj, dict):
        return None
    if obj.get('kind') == 'analysis_rehearsal':
        return obj
    plan = obj.get('plan') if isinstance(obj.get('plan'), dict) else None
    if isinstance(plan, dict) and plan.get('kind') == 'analysis_rehearsal':
        return plan
    if isinstance(plan, dict) and isinstance(plan.get('analysis_rehearsal'), dict):
        return plan['analysis_rehearsal']
    if isinstance(obj.get('analysis_rehearsal'), dict):
        return obj['analysis_rehearsal']
    return None


def _ar_gate_validate(path, manifest=None):
    L = _import_exp_lib()  # noqa: F821
    item = {'path': path, 'status': 'not_verifiable', 'problems': [],
            'warnings': [], 'content_sha256': None}
    if not os.path.exists(path):
        item['problems'].append(f'{path}: artifact does not exist')
        return item
    try:
        raw = _ar_gate_read_json(path)
    except Exception as exc:
        item['status'] = 'fail'
        item['problems'].append(
            f'{path}: not readable JSON ({type(exc).__name__}: {exc})')
        return item
    rehearsal = _ar_gate_embedded(raw)
    if rehearsal is None:
        item['status'] = 'not_verifiable'
        item['problems'].append(
            f'{path}: no analysis_rehearsal object found')
        return item
    verdict = L.validate_analysis_rehearsal(rehearsal, manifest=manifest)
    item['status'] = verdict.get('status')
    item['problems'].extend(verdict.get('problems') or [])
    item['content_sha256'] = rehearsal.get('content_sha256')
    item['manifest_sha256'] = rehearsal.get('manifest_sha256')
    item['n_scenarios'] = rehearsal.get('n_scenarios')
    item['n_failures'] = rehearsal.get('n_failures')
    return item


def audit_analysis_rehearsal(paths=None):
    """Validate stored rehearsal evidence or run the live known-truth control.

    No training, no data loader, no test split.  The gate is green only when
    every scenario in every inspected artifact passes and its body hash
    reproduces.
    """
    L = _import_exp_lib()  # noqa: F821
    out = {'kind': 'analysis_rehearsal_audit', 'status': 'not_verifiable',
           'problems': [], 'warnings': [], 'artifacts': {}, 'live': None,
           'synthetic': True, 'test_split_read': False}
    manifest_path = os.path.join(REPO, 'confirmation_frozen.json')  # noqa: F821
    manifest = None
    if os.path.exists(manifest_path):
        try:
            manifest = _ar_gate_read_json(manifest_path)
        except Exception as exc:
            out['warnings'].append(
                f'frozen manifest is not readable ({type(exc).__name__}: {exc})')
    if paths is None:
        found = []
        for rel in _ANALYSIS_REHEARSAL_DEFAULTS:
            candidate = os.path.join(REPO, rel)  # noqa: F821
            if os.path.exists(candidate):
                found.append(candidate)
        if found:
            paths = found
        else:
            # No stored rehearsal: run it live.  This is what `audit_gates.py
            # report` does on a source-only checkout; it must not be reported
            # as missing evidence just because no local run directory exists.
            try:
                live = L.build_analysis_rehearsal(
                    manifest_path=manifest_path)
            except Exception as exc:
                out['status'] = 'fail'
                out['problems'].append(
                    'live rehearsal could not run '
                    f'({type(exc).__name__}: {exc})')
                return out
            out['live'] = {
                'status': live.get('status'),
                'synthetic': live.get('synthetic'),
                'test_split_read': live.get('test_split_read'),
                'manifest_sha256': live.get('manifest_sha256'),
                'content_sha256': live.get('content_sha256'),
                'n_scenarios': live.get('n_scenarios'),
                'n_checks': live.get('n_checks'),
                'family_size': live.get('family_size'),
                'n_failures': live.get('n_failures'),
                'failures': list(live.get('failures') or []),
            }
            out['status'] = live.get('status')
            if out['status'] == 'fail':
                out['problems'].extend(live.get('failures') or [])
            return out
    if isinstance(paths, str):
        paths = [paths]
    statuses = []
    for raw_path in paths:
        path = str(raw_path)
        if os.path.isdir(path):
            path = os.path.join(path, 'analysis_rehearsal.json')
        item = _ar_gate_validate(path, manifest=manifest)
        out['artifacts'][str(raw_path)] = item
        statuses.append(item.get('status'))
        out['problems'].extend(item.get('problems') or [])
    if any(s == 'fail' for s in statuses):
        out['status'] = 'fail'
    elif any(s == 'ok' for s in statuses):
        out['status'] = 'ok'
    else:
        out['status'] = 'not_verifiable'
    return out
