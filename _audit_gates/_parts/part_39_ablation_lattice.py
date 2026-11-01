# Audit gate: ablation lattice attribution and design licensing.
#
# The analysis lives in ``explib`` (part_73).  This gate does two independent
# things:
#   1. audits the declared design: each 2x2 factorial is a complete
#      two-component lattice; the leave-one-out family is a star and therefore
#      cannot license a global Shapley attribution;
#   2. optionally replays a stored ``ablation_lattice.json`` from its recorded
#      cells.  A tampered Shapley/LOO/interaction/axiom/status is a failure.
#
# A missing artifact is not a failed measurement: the design audit still runs
# and the gate says whether a measured lattice exists.  It never reads a data
# split and never changes a frozen p/Holm/verified/confirmed value.

import json
import os


_AL_GATE_DEFAULTS = (
    os.path.join('analysis_v12', 'ablation_lattice.json'),
    'ablation_lattice.json',
)


# Shared replay loader (part_00): unreadable evidence is "no evidence".
_al_gate_load = _gate_load


def _al_gate_resolve(path):
    if path is not None:
        return os.fspath(path)
    for candidate in _AL_GATE_DEFAULTS:
        full = os.path.join(REPO, candidate)  # noqa: F821
        if os.path.exists(full):
            return full
    return None


def audit_ablation_lattice(path=None):
    """Design licensing plus an optional replay of a measured lattice."""
    L = _import_exp_lib()  # noqa: F821
    design = L.ablation_lattice_design()
    out = {
        'kind': 'ablation_lattice_audit',
        'status': design.get('status', 'not_verifiable_locally'),
        'mode': design.get('mode'),
        'n_scopes': len(design.get('scopes') or {}),
        'scopes': {
            name: {
                'components': scope.get('components'),
                'complete': scope.get('complete'),
                'shapley_identifiable': scope.get('shapley_identifiable'),
                'missing_arms_from_run_panel': scope.get(
                    'missing_arms_from_run_panel'),
            }
            for name, scope in (design.get('scopes') or {}).items()
            if isinstance(scope, dict)
        },
        'global': design.get('global') or {},
        'star_family': design.get('star_family') or {},
        'licensed_claims': list(design.get('licensed_claims') or []),
        'unlicensed_claims': list(design.get('unlicensed_claims') or []),
        'measured': False,
        'artifact_path': None,
        'artifact_status': None,
        'content_sha256': None,
        'n_components': (design.get('global') or {}).get('n_components'),
        'global_attribution_licensed': bool(
            (design.get('global') or {}).get('shapley_identifiable')),
        'problems': list(design.get('problems') or []),
        'warnings': list(design.get('warnings') or []),
        'note': ('design licensing only; a leave-one-out star is not a subset '
                 'lattice and cannot license a global Shapley attribution'),
    }
    if out['problems']:
        out['status'] = 'fail'
    p = _al_gate_resolve(path)
    if p is None:
        return out
    out['artifact_path'] = p
    if not os.path.exists(p):
        out['status'] = 'not_verifiable_locally'
        out['problems'].append(f'{p}: artifact does not exist')
        return out
    data = _al_gate_load(p)
    if not isinstance(data, dict):
        out['status'] = 'fail'
        out['problems'].append(f'{p}: not readable JSON object')
        return out
    out['measured'] = True
    out['artifact_status'] = data.get('status')
    out['content_sha256'] = data.get('content_sha256')
    verdict = L.validate_ablation_lattice(data)
    out['warnings'].extend(verdict.get('warnings') or [])
    if verdict.get('status') == 'fail':
        out['status'] = 'fail'
        out['problems'].extend(verdict.get('problems') or [])
        return out
    out['replayed_status'] = verdict.get('replayed_status')
    artifact_status = data.get('status')
    if artifact_status == 'not_estimable':
        # A structurally valid artifact that could not answer the question is
        # not green: the design is incomplete, and the gate must say so.
        out['status'] = 'not_verifiable_locally'
        out['problems'].extend(data.get('problems') or [])
        out['note'] = ('the stored lattice is structurally valid but '
                       'not_estimable from its recorded cells')
    elif artifact_status == 'warnings':
        out['status'] = 'warnings'
        out['warnings'].extend(data.get('warnings') or [])
    elif artifact_status == 'ok':
        if out['status'] == 'ok':
            out['status'] = 'ok'
        out['warnings'].extend(data.get('warnings') or [])
        summary = L.ablation_lattice_summary(data)
        out['summary'] = summary
    else:
        out['status'] = 'fail'
        out['problems'].append(
            f'artifact.status={artifact_status!r} is not recognised')
    return out
