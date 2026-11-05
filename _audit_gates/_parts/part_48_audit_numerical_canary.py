# Audit gate: replay the numerical-stack canary.
#
# A result record can carry the same lock, the same source digest and the same
# `env_fp_v3`, yet be produced by a process whose BLAS/kernel path rounds
# differently.  This gate reads the stored operator-level arithmetic
# fingerprint and recomputes it in the current process.  A missing artifact is
# not a pass; an intact artifact whose arithmetic no longer replays is a
# failure.  It never trains, reads a split, or changes a frozen verdict.

import json
import os


_NC_GATE_DEFAULTS = (
    os.path.join('analysis_v12', 'numerical_canary.json'),
    os.path.join('analysis_audit', 'numerical_canary.json'),
    'numerical_canary.json',
)

_nc_gate_load = _gate_load  # shared loader: unreadable/malformed == no evidence


def _nc_gate_resolve(path):
    if path is not None:
        if isinstance(path, (str, os.PathLike)):
            return os.fspath(path)
        return None
    for candidate in _NC_GATE_DEFAULTS:
        full = os.path.join(REPO, candidate)  # noqa: F821
        if os.path.exists(full):
            return full
    return None


def audit_numerical_canary(path=None, *, verbose=True):
    """Replay a recorded numerical canary against the current process.

    Returns ``ok`` when the artifact is intact and its operator digests
    replay; ``warnings`` when the arithmetic still replays but the recorded
    environment/checkout metadata has changed; ``not_verifiable_locally``
    when no artifact exists or the recorded device is unavailable on this
    machine; and ``fail`` for a malformed/tampered artifact or observed
    arithmetic drift.
    """
    L = _import_exp_lib()  # noqa: F821
    out = {
        'kind': 'numerical_canary_audit',
        'status': 'not_verifiable_locally',
        'path': None,
        'content_sha256': None,
        'result_status': None,
        'arithmetic_fingerprint': None,
        'replay_fingerprint': None,
        'arithmetic_match': None,
        'device': None,
        'dtype': None,
        'seed': None,
        'size': None,
        'n_ops': None,
        'drifted_ops': [],
        'environment_changes': [],
        'provenance_changes': [],
        'source_tree_changed': None,
        'environment_changed': None,
        'problems': [],
        'warnings': [],
        'note': ('observed arithmetic identity: it is not a new statistical '
                 'test and does not prove that training is bitwise '
                 'deterministic; it detects same-declared-stack arithmetic '
                 'drift.  Re-record only after reviewing the drift.'),
    }
    resolved = _nc_gate_resolve(path)
    if resolved is None:
        out['problems'].append('no numerical_canary.json artifact found')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out
    out['path'] = resolved
    if not os.path.exists(resolved):
        out['problems'].append(f'{resolved}: artifact does not exist')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out
    data = _nc_gate_load(resolved)  # noqa: F821
    if not isinstance(data, dict):
        out['status'] = 'fail'
        out['problems'].append(f'{resolved}: not a readable JSON object')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out
    if data.get('kind') != 'numerical_canary' or 'result' not in data:
        out['status'] = 'fail'
        out['problems'].append(
            f'{resolved}: not a numerical_canary artifact; run '
            '`make canary` to write one')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out
    verdict = L.validate_numerical_canary_artifact(data)
    status = verdict.get('status', 'fail')
    if status == 'drift':
        out['status'] = 'fail'
    elif status == 'fail':
        out['status'] = 'fail'
    elif status == 'not_verifiable_locally':
        out['status'] = 'not_verifiable_locally'
    elif status == 'ok':
        out['status'] = 'warnings' if verdict.get('warnings') else 'ok'
    else:
        out['status'] = 'fail'
    out['content_sha256'] = verdict.get('content_sha256')
    out['problems'] = list(verdict.get('problems') or [])
    out['warnings'] = list(verdict.get('warnings') or [])
    result = verdict.get('result') if isinstance(verdict.get('result'), dict) else {}
    params = result.get('params') if isinstance(result.get('params'), dict) else {}
    arithmetic = (result.get('arithmetic')
                  if isinstance(result.get('arithmetic'), dict) else {})
    out['result_status'] = result.get('status')
    out['arithmetic_fingerprint'] = arithmetic.get('fingerprint')
    out['arithmetic_match'] = verdict.get('arithmetic_match')
    out['device'] = params.get('device')
    out['dtype'] = params.get('dtype')
    out['seed'] = params.get('seed')
    out['size'] = params.get('size')
    ops = arithmetic.get('ops') if isinstance(arithmetic.get('ops'), dict) else {}
    out['n_ops'] = len(ops)
    out['drifted_ops'] = list(verdict.get('drifted_ops') or [])
    out['environment_changes'] = list(verdict.get('environment_changes') or [])
    out['provenance_changes'] = list(verdict.get('provenance_changes') or [])
    out['source_tree_changed'] = ('source_tree' in out['provenance_changes'])
    out['environment_changed'] = bool(out['environment_changes'])
    replay = verdict.get('replay') if isinstance(verdict.get('replay'), dict) else {}
    replay_arith = (replay.get('arithmetic')
                    if isinstance(replay.get('arithmetic'), dict) else {})
    out['replay_fingerprint'] = replay_arith.get('fingerprint')
    if verbose:
        print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
    return out


def audit_numerical_canary_artifact(path=None, *, verbose=True):
    """Single-artifact replay alias for :func:`audit_numerical_canary`."""
    return audit_numerical_canary(path, verbose=verbose)
