# Audit gate: contrast estimability of the registered cell design.
#
# The analysis lives in ``explib`` (part_74) so it is also reachable from
# zero-GPU preparation tooling.  This gate either replays a stored
# ``design_estimability.json`` (including a copy embedded in
# ``plan_snapshot.json``) or runs the analysis live from the frozen manifest
# and an optional run order.  A non-estimable declared comparison is a design
# failure, not a null result; a missing design artifact is reported as
# ``not_verifiable`` and never as green.

import json
import os


_DESIGN_ESTIMABILITY_DEFAULTS = (
    os.path.join('analysis_v12', 'design_estimability.json'),
    'design_estimability.json',
)
_RUN_ORDER_DEFAULTS = (
    os.path.join('analysis_v12', 'run_order.json'),
    'run_order.json',
    os.path.join('results_lm_test', 'run_order.json'),
)


# Shared replay loader (part_00): unreadable evidence is "no evidence".
_de_gate_load = _gate_load


def _de_gate_embedded(obj):
    if not isinstance(obj, dict):
        return None
    if obj.get('kind') == 'design_estimability':
        return obj
    plan = obj.get('plan') if isinstance(obj.get('plan'), dict) else None
    if isinstance(plan, dict):
        if plan.get('kind') == 'design_estimability':
            return plan
        if isinstance(plan.get('design_estimability'), dict):
            return plan['design_estimability']
    if isinstance(obj.get('design_estimability'), dict):
        return obj['design_estimability']
    return None


def _de_gate_list(value):
    """String list or ``[]``; shared rule in :func:`_gate_str_list`."""
    return _gate_str_list(value)


def _de_gate_manifest(manifest_path):
    if manifest_path is None:
        path = os.path.join(REPO, 'confirmation_frozen.json')  # noqa: F821
    elif isinstance(manifest_path, (str, os.PathLike)):
        path = os.fspath(manifest_path)
    else:
        return None, f'<invalid manifest_path {type(manifest_path).__name__}>'
    if not os.path.exists(path):
        return None, path
    data = _de_gate_load(path)
    return (data if isinstance(data, dict) else None), path


def _de_gate_resolve_run_order(run_order_path):
    if run_order_path is None:
        for rel in _RUN_ORDER_DEFAULTS:
            candidate = os.path.join(REPO, rel)  # noqa: F821
            if os.path.exists(candidate):
                return candidate
        return None
    if isinstance(run_order_path, (str, os.PathLike)):
        return os.fspath(run_order_path)
    return None


def _de_gate_summary(artifact):
    design = artifact.get('design') if isinstance(
        artifact.get('design'), dict) else {}
    summary = artifact.get('summary') if isinstance(
        artifact.get('summary'), dict) else {}
    return {
        'status': artifact.get('status'),
        'n_cells': design.get('n_cells'),
        'n_arms': design.get('n_arms'),
        'n_blocks': design.get('n_blocks'),
        'rank': design.get('rank'),
        'n_columns': design.get('n_columns'),
        'deficiency': design.get('deficiency'),
        'complete_grid': design.get('complete_grid'),
        'components': design.get('components'),
        'n_components': summary.get('n_components'),
        'n_comparisons': summary.get('n_comparisons'),
        'n_estimable': summary.get('n_estimable'),
        'n_not_estimable': summary.get('n_not_estimable'),
        'n_terms_not_estimable': summary.get('n_terms_not_estimable'),
        'n_terms_not_separately_estimable': summary.get(
            'n_terms_not_separately_estimable'),
        'n_aliased_pairs': summary.get('n_aliased_pairs'),
        'n_redundant_pairs': summary.get('n_redundant_pairs'),
        'content_sha256': artifact.get('content_sha256'),
    }


def audit_design_estimability(path=None, manifest_path=None,
                              run_order_path=None, *, verbose=True):
    """Replay stored estimability evidence or run it live.

    Zero GPU, no data loader, no split read.  The gate is green only when the
    inspected design is estimable *and* (for a stored artifact) replays from
    its recorded inputs.  A design with an unanswerable comparison is
    ``fail``; no artifact and no manifest is ``not_verifiable``.
    """
    L = _import_exp_lib()  # noqa: F821
    def _finish(result):
        if verbose:
            print(json.dumps(result, ensure_ascii=False, indent=2)[:4000])
        return result

    out = {
        'kind': 'design_estimability_audit',
        'status': 'not_verifiable',
        'problems': [],
        'warnings': [],
        'artifact_path': None,
        'artifact_status': None,
        'content_sha256': None,
        'summary': None,
        'live': None,
        'manifest': None,
        'run_order': None,
        'note': (
            'a non-estimable declared contrast is a design failure; it is '
            'not a null result and cannot be repaired by adding seeds inside '
            'the same disconnected block'),
    }
    manifest, mpath = _de_gate_manifest(manifest_path)
    out['manifest'] = mpath
    explicit_run_order = run_order_path is not None
    run_order_path = _de_gate_resolve_run_order(run_order_path)
    out['run_order'] = run_order_path
    run_order = None
    if run_order_path is not None:
        if not os.path.exists(run_order_path):
            if explicit_run_order:
                out['status'] = 'fail'
                out['problems'].append(
                    f'{run_order_path}: run order does not exist')
                return _finish(out)
        else:
            run_order = _de_gate_load(run_order_path)
            if run_order is None:
                out['status'] = 'fail'
                out['problems'].append(
                    f'{run_order_path}: run order is not readable JSON')
                return _finish(out)

    # 1) Prefer an explicitly supplied / discovered artifact.  A stored
    # artifact is the complete evidence: replay it and do not silently
    # substitute a live design if replay fails.
    candidates = []
    explicit_artifact = path is not None
    if explicit_artifact:
        candidates = [os.fspath(path)]
    else:
        for rel in _DESIGN_ESTIMABILITY_DEFAULTS:
            candidate = os.path.join(REPO, rel)  # noqa: F821
            if os.path.exists(candidate):
                candidates.append(candidate)
        snapshot = os.path.join(REPO, 'analysis_v12',  # noqa: F821
                                'plan_snapshot.json')
        if os.path.exists(snapshot):
            candidates.append(snapshot)
    for candidate in candidates:
        if not os.path.exists(candidate):
            out['problems'].append(f'{candidate}: artifact does not exist')
            out['status'] = 'fail'
            return _finish(out)
        raw = _de_gate_load(candidate)
        if raw is None:
            out['problems'].append(f'{candidate}: not readable JSON')
            out['status'] = 'fail'
            return _finish(out)
        artifact = _de_gate_embedded(raw)
        if artifact is None:
            if explicit_artifact:
                out['problems'].append(
                    f'{candidate}: no design_estimability object found')
                out['status'] = 'fail'
                return _finish(out)
            if os.path.basename(candidate) == 'plan_snapshot.json':
                # A plan snapshot without the block is not a failed design;
                # it just has no estimability evidence.  Keep looking for a
                # dedicated artifact, then fall through to the live path.
                continue
            out['problems'].append(
                f'{candidate}: no design_estimability object found')
            out['status'] = 'fail'
            return _finish(out)
        out['artifact_path'] = candidate
        if manifest is not None:
            inputs = artifact.get('inputs')
            inputs = inputs if isinstance(inputs, dict) else {}
            recorded = inputs.get('manifest_canonical_sha256')
            if recorded:
                current = L._de_manifest_canonical_sha(manifest)
                if current is not None and current != recorded:
                    msg = (f'{candidate}: artifact was built from a different '
                           'manifest body (manifest_canonical_sha256 mismatch)')
                    if explicit_artifact:
                        out['status'] = 'fail'
                        out['problems'].append(msg)
                        return _finish(out)
                    # A discovered artifact from another freeze is stale
                    # evidence for the manifest being audited, not a design
                    # defect: ignore it and analyse the current design live.
                    out['warnings'].append(
                        msg + '; ignoring the stale artifact and running the '
                        'analysis live')
                    out['artifact_status'] = 'stale_manifest'
                    out['stale_artifact'] = True
                    break
        verdict = L.validate_design_estimability(artifact)
        out['warnings'].extend(_de_gate_list(verdict.get('warnings')))
        out['artifact_status'] = artifact.get('status')
        out['content_sha256'] = artifact.get('content_sha256')
        out['summary'] = _de_gate_summary(artifact)
        if verdict.get('status') == 'fail':
            out['status'] = 'fail'
            out['problems'].extend(_de_gate_list(verdict.get('problems')))
            return _finish(out)
        artifact_status = artifact.get('status')
        if artifact_status == 'fail':
            out['status'] = 'fail'
            out['problems'].extend(_de_gate_list(artifact.get('problems')))
        elif artifact_status == 'not_estimable':
            out['status'] = 'not_verifiable'
            out['problems'].extend(_de_gate_list(artifact.get('problems')))
            out['note'] = (
                'the stored report is internally valid but declares no '
                'estimable comparison; the design question is unanswered')
        elif artifact_status == 'warnings':
            out['status'] = 'warnings'
            out['warnings'].extend(_de_gate_list(artifact.get('warnings')))
        else:
            out['status'] = 'ok'
        return _finish(out)

    # 2) No stored artifact: run the same analysis live from the frozen
    # manifest and the registered run order (if one exists).
    if manifest is None:
        out['reason'] = f'no readable frozen manifest at {mpath}'
        out['warnings'].append(out['reason'])
        return _finish(out)
    try:
        live = L.design_estimability_from_manifest(
            manifest, run_order=run_order)
    except Exception as exc:
        out['status'] = 'fail'
        out['problems'].append(
            'live estimability analysis could not run '
            f'({type(exc).__name__}: {exc})')
        return _finish(out)
    out['live'] = _de_gate_summary(live)
    out['summary'] = out['live']
    out['content_sha256'] = live.get('content_sha256')
    live_status = live.get('status')
    out['warnings'].extend(_de_gate_list(live.get('warnings')))
    if live_status == 'fail':
        out['status'] = 'fail'
        out['problems'].extend(_de_gate_list(live.get('problems')))
    elif live_status == 'not_estimable':
        out['status'] = 'not_verifiable'
        out['problems'].extend(_de_gate_list(live.get('problems')))
    elif live_status == 'warnings':
        out['status'] = 'warnings'
    else:
        out['status'] = 'ok'
    if out.get('stale_artifact') and out['status'] == 'ok':
        out['status'] = 'warnings'
    return _finish(out)
