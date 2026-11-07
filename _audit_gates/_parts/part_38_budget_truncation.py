# Audit gate: budget-truncation feasibility of the registered cell panel.
#
# The analysis itself lives in ``explib`` (part_72) so ``confirmation plan``
# can compile it before any training.  This gate either validates a stored
# ``budget_truncation.json`` (including the copy embedded in a
# ``plan_snapshot.json``) or runs the zero-GPU analysis live against the
# current frozen manifest.  A stored artifact is replayed from its recorded
# inputs, so a tampered frontier, cost block or verdict is a failure.  A
# missing budget declaration is ``not_verifiable``, never green.

import json
import os


_BUDGET_TRUNCATION_DEFAULTS = (
    os.path.join('analysis_v12', 'budget_truncation.json'),
    'budget_truncation.json',
)


# Shared replay loader (part_00): unreadable evidence is "no evidence".
_bt_gate_load = _gate_load


def _bt_gate_embedded(obj):
    """Return a budget-truncation object from a raw artifact or a snapshot."""
    if not isinstance(obj, dict):
        return None
    if obj.get('kind') == 'budget_truncation':
        return obj
    plan = obj.get('plan') if isinstance(obj.get('plan'), dict) else None
    if isinstance(plan, dict) and plan.get('kind') == 'budget_truncation':
        return plan
    if isinstance(plan, dict) and isinstance(plan.get('budget_truncation'),
                                              dict):
        return plan['budget_truncation']
    if isinstance(obj.get('budget_truncation'), dict):
        return obj['budget_truncation']
    return None


def _bt_gate_dict(value):
    return value if isinstance(value, dict) else {}


def _bt_gate_list(value):
    """String list or ``[]``; shared rule in :func:`_gate_str_list`."""
    return _gate_str_list(value)


def _bt_gate_manifest(manifest_path):
    if manifest_path is None:
        path = os.path.join(REPO, 'confirmation_frozen.json')  # noqa: F821
    elif isinstance(manifest_path, (str, os.PathLike)):
        path = os.fspath(manifest_path)
    else:
        return None, f'<invalid manifest_path {type(manifest_path).__name__}>'
    if not os.path.exists(path):
        return None, path
    data = _bt_gate_load(path)
    return (data if isinstance(data, dict) else None), path


def _bt_gate_validate(L, path, manifest):
    item = {'path': path, 'status': 'not_verifiable', 'problems': [],
            'warnings': [], 'content_sha256': None,
            'required_prefix_cells': None,
            'budget_attempts': None,
            'required_attempts_for_risk': None}
    if not os.path.exists(path):
        item['problems'].append(f'{path}: artifact does not exist')
        return item
    raw = _bt_gate_load(path)
    if raw is None:
        item['status'] = 'fail'
        item['problems'].append(f'{path}: not readable JSON')
        return item
    artifact = _bt_gate_embedded(raw)
    if artifact is None:
        item['status'] = 'not_verifiable'
        item['problems'].append(
            f'{path}: no budget_truncation object found')
        return item
    verdict = L.validate_budget_truncation(artifact, manifest=manifest)
    item['warnings'].extend(_bt_gate_list(verdict.get('warnings')))
    item['content_sha256'] = artifact.get('content_sha256')
    artifact_status = artifact.get('status')
    item['artifact_status'] = artifact_status
    if verdict.get('status') == 'fail' or artifact_status == 'fail':
        item['status'] = 'fail'
        item['problems'].extend(_bt_gate_list(verdict.get('problems')))
    elif artifact_status == 'not_verifiable':
        # Internally consistent, but the budget question was not answered:
        # replaying it must not turn an undeclared budget into a green gate.
        item['status'] = 'not_verifiable'
        item['problems'].extend(_bt_gate_list(verdict.get('problems')))
        item['reason'] = ('the stored analysis carries no declared budget; '
                          'its internal replay is consistent but the '
                          'feasibility question is unanswered')
    elif artifact_status == 'warnings':
        # Feasible only under a caveat (e.g. no failure model): the artifact
        # is internally valid, but the gate must not label it green.
        item['status'] = 'warnings'
        item['problems'].extend(_bt_gate_list(verdict.get('problems')))
    else:
        item['status'] = 'ok'
        item['problems'].extend(_bt_gate_list(verdict.get('problems')))
    cost = _bt_gate_dict(artifact.get('cost'))
    budget = _bt_gate_dict(artifact.get('budget'))
    floor = _bt_gate_dict(artifact.get('floor'))
    item['required_prefix_cells'] = cost.get('required_prefix_cells')
    item['budget_attempts'] = budget.get('declared_attempts')
    item['required_attempts_for_risk'] = cost.get(
        'required_attempts_for_risk')
    item['effective_required_seeds'] = floor.get('effective_required_seeds')
    item['declared_min_seeds'] = floor.get('declared_min_seeds')
    return item


def audit_budget_truncation(paths=None, manifest_path=None):
    """Validate stored budget feasibility evidence or run it live.

    Zero GPU, no data loader, no split read.  The gate is green only when an
    inspected artifact validates *and* replays; a missing budget declaration
    is reported as ``not_verifiable`` with the deterministic floor so the
    launch decision sees the number instead of a silent pass.
    """
    L = _import_exp_lib()  # noqa: F821
    out = {'kind': 'budget_truncation_audit', 'status': 'not_verifiable',
           'problems': [], 'warnings': [], 'artifacts': {}, 'live': None,
           'budget_unit': 'cell_attempt'}
    manifest, mpath = _bt_gate_manifest(manifest_path)
    out['manifest'] = mpath
    if manifest is None:
        out['status'] = 'not_verifiable'
        out['reason'] = f'no readable frozen manifest at {mpath}'
        out['warnings'].append(out['reason'])
        return out
    if paths is None:
        found = []
        for rel in _BUDGET_TRUNCATION_DEFAULTS:
            candidate = os.path.join(REPO, rel)  # noqa: F821
            if os.path.exists(candidate):
                found.append(candidate)
        if not found:
            snapshot = os.path.join(REPO, 'analysis_v12',  # noqa: F821
                                    'plan_snapshot.json')
            if os.path.exists(snapshot):
                found.append(snapshot)
        if not found:
            # No stored evidence: run the zero-GPU analysis live.  This is
            # what `audit_gates.py report` does on a source-only checkout.
            try:
                live = L.budget_truncation_from_manifest(manifest)
            except Exception as exc:
                out['status'] = 'fail'
                out['problems'].append(
                    'live budget-truncation analysis could not run '
                    f'({type(exc).__name__}: {exc})')
                return out
            out['live'] = {
                'status': live.get('status'),
                'content_sha256': live.get('content_sha256'),
                'n_cells': _bt_gate_dict(live.get('design')).get('n_cells'),
                'required_prefix_cells': _bt_gate_dict(
                    live.get('cost')).get('required_prefix_cells'),
                'effective_required_seeds': _bt_gate_dict(
                    live.get('floor')).get('effective_required_seeds'),
                'declared_min_seeds': _bt_gate_dict(
                    live.get('floor')).get('declared_min_seeds'),
                'budget_attempts': _bt_gate_dict(
                    live.get('budget')).get('declared_attempts'),
                'required_attempts_for_risk': _bt_gate_dict(
                    live.get('cost')).get('required_attempts_for_risk'),
                'warnings': _bt_gate_list(live.get('warnings')),
                'problems': _bt_gate_list(live.get('problems')),
            }
            out['status'] = live.get('status')
            out['warnings'].extend(_bt_gate_list(live.get('warnings')))
            if out['status'] == 'fail':
                out['problems'].extend(_bt_gate_list(live.get('problems')))
            return out
        paths = found
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    else:
        try:
            paths = list(paths)
        except TypeError:
            out['status'] = 'fail'
            out['problems'].append(
                'paths must be a path or an iterable of paths, got '
                f'{type(paths).__name__}')
            return out
    if not paths:
        out['status'] = 'not_verifiable'
        out['reason'] = 'no candidate artifact path was supplied or found'
        out['warnings'].append(out['reason'])
        return out
    statuses = []
    for raw_path in paths:
        path = os.fspath(raw_path) if isinstance(
            raw_path, os.PathLike) else str(raw_path)
        if os.path.isdir(path):
            path = os.path.join(path, 'budget_truncation.json')
        item = _bt_gate_validate(L, path, manifest)
        out['artifacts'][str(raw_path)] = item
        statuses.append(item.get('status'))
        out['problems'].extend(_bt_gate_list(item.get('problems')))
        out['warnings'].extend(_bt_gate_list(item.get('warnings')))
    if any(s == 'fail' for s in statuses):
        out['status'] = 'fail'
    elif any(s == 'ok' for s in statuses):
        out['status'] = 'warnings' if out['warnings'] else 'ok'
    elif any(s == 'warnings' for s in statuses):
        out['status'] = 'warnings'
    else:
        out['status'] = 'not_verifiable'
    return out
