# Audit gate: global (cross-panel) exploratory false-discovery-rate readout.
#
# The analysis lives in ``explib`` (part_78).  A stored artifact records the
# exact declared exploratory claim family in ``inputs`` and the BH/BY
# q-values in ``result``.  This gate replays every derived field from those
# inputs.  It never reads a split, never creates a confirmatory p-value,
# never enters the frozen Holm family, and a missing artifact is not a pass.

import json
import os


_GLOBAL_FDR_GATE_DEFAULTS = (
    os.path.join('analysis_v12', 'global_fdr.json'),
    'global_fdr.json',
)

_gfdr_gate_load = _gate_load  # shared loader (part_00): unreadable == no evidence


def _gfdr_gate_resolve(path):
    if path is not None:
        if isinstance(path, (str, os.PathLike)):
            return os.fspath(path)
        return None
    for candidate in _GLOBAL_FDR_GATE_DEFAULTS:
        full = os.path.join(REPO, candidate)  # noqa: F821
        if os.path.exists(full):
            return full
    return None


def audit_global_fdr(path=None, *, verbose=True):
    """Replay a stored ``global_fdr.json`` exploratory FDR artifact.

    ``not_verifiable_locally`` when no artifact exists (a source-only
    checkout has no claim family to correct); ``fail`` for malformed inputs,
    a non-reproducing digest, tampered q-values, an undeclared/extra claim or
    a smuggled frozen-decision field; ``not_applicable`` for a structurally
    valid artifact whose declared family has no valid p-value.
    """
    L = _import_exp_lib()  # noqa: F821
    out = {
        'kind': 'global_fdr_audit',
        'status': 'not_verifiable_locally',
        'path': None,
        'm_declared': None,
        'm_valid': None,
        'm_invalid': None,
        'method': None,
        'alpha': None,
        'n_reject': None,
        'result_status': None,
        'problems': [],
        'warnings': [],
        'note': ('replays the exploratory BH/BY global FDR readout from its '
                 'declared claim family; the frozen confirmatory Holm family '
                 'is not changed or re-derived'),
    }
    p = _gfdr_gate_resolve(path)
    if p is None:
        out['problems'].append('no global_fdr.json artifact found')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out
    out['path'] = p
    if not os.path.exists(p):
        out['problems'].append(f'{p}: artifact does not exist')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out
    data = _gfdr_gate_load(p)  # noqa: F821
    if not isinstance(data, dict):
        out['status'] = 'fail'
        out['problems'].append(f'{p}: not a readable JSON object')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out
    verdict = L.validate_global_fdr(data)
    out['status'] = verdict.get('status', 'fail')
    out['problems'] = list(verdict.get('problems') or [])
    out['warnings'] = list(verdict.get('warnings') or [])
    out['result_status'] = verdict.get('result_status')
    result = verdict.get('result') if isinstance(verdict.get('result'), dict) else {}
    out['m_declared'] = result.get('m_declared')
    out['m_valid'] = result.get('m_valid')
    out['m_invalid'] = result.get('m_invalid')
    out['method'] = result.get('method')
    out['alpha'] = result.get('alpha')
    out['n_reject'] = result.get('n_reject')
    if out['status'] == 'ok' and out['result_status'] not in (None, 'ok'):
        # A self-consistent artifact can describe an unusable family.  That is
        # an honest "not applicable", not a failed replay and not a green pass.
        out['status'] = 'not_applicable'
    if verbose:
        print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
    return out


def audit_global_fdr_artifact(path=None, *, verbose=True):
    """Single-artifact replay alias for :func:`audit_global_fdr`."""
    return audit_global_fdr(path, verbose=verbose)
