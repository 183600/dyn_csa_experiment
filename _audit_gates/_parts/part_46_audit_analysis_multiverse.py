# Audit gate: declared analysis-multiverse / specification-curve replay.
#
# The analysis lives in ``explib`` (part_80).  A stored
# ``analysis_multiverse.json`` records the exact declared specifications in
# ``inputs`` and the IUT / per-family max-T readout in ``result``.  This gate
# replays every derived field from those specs.  A missing artifact is not a
# pass; a self-consistent but non-robust result is a warning, not a tamper;
# a changed estimate, p-value, family, hash or a smuggled frozen-decision
# field is a failure.  It never reads a split and never changes a frozen
# p-value, Holm family or confirmed verdict.

import json
import os


_AM_GATE_DEFAULTS = (
    os.path.join('analysis_v12', 'analysis_multiverse.json'),
    'analysis_multiverse.json',
)

_am_gate_load = _gate_load  # shared loader: unreadable/malformed == no evidence


def _am_gate_resolve(path):
    if path is not None:
        if isinstance(path, (str, os.PathLike)):
            return os.fspath(path)
        return None
    for candidate in _AM_GATE_DEFAULTS:
        full = os.path.join(REPO, candidate)  # noqa: F821
        if os.path.exists(full):
            return full
    return None


def audit_analysis_multiverse(path=None, *, verbose=True):
    """Replay a stored ``analysis_multiverse.json`` robustness artifact.

    ``not_verifiable_locally`` when no artifact exists or when the declared
    multiverse is structurally too small to support any all-specifications
    claim; ``warnings`` when the artifact is internally consistent but the
    measured pattern is direction-unstable, precision-unstable, resolution
    limited or a zero primary effect; ``fail`` for malformed inputs, a
    non-reproducing digest, tampered fields or a smuggled frozen-decision key.
    """
    L = _import_exp_lib()  # noqa: F821
    out = {
        'kind': 'analysis_multiverse_audit',
        'status': 'not_verifiable_locally',
        'path': None,
        'artifact_status': None,
        'result_status': None,
        'pattern': None,
        'primary': None,
        'primary_sign': None,
        'n_specs': None,
        'n_families': None,
        'p_iut': None,
        'iut_p_floor': None,
        'resolution_limited': None,
        'sign_agreement': None,
        'n_sign_flips': None,
        'n_reject_raw': None,
        'robust_across_specs': False,
        'not_comparable_families': [],
        'not_estimable_families': [],
        'problems': [],
        'warnings': [],
        'note': ('replays the exploratory analysis multiverse from its '
                 'declared specifications; it does not change the frozen '
                 'confirmatory Holm family or a confirmed verdict'),
    }
    resolved = _am_gate_resolve(path)
    if resolved is None:
        out['problems'].append('no analysis_multiverse.json artifact found')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out
    out['path'] = resolved
    if not os.path.exists(resolved):
        out['problems'].append(f'{resolved}: artifact does not exist')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out
    data = _am_gate_load(resolved)  # noqa: F821
    if not isinstance(data, dict):
        out['status'] = 'fail'
        out['problems'].append(f'{resolved}: not a readable JSON object')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out
    verdict = L.validate_analysis_multiverse(data)
    out['status'] = verdict.get('status', 'fail')
    out['problems'] = list(verdict.get('problems') or [])
    out['warnings'] = list(verdict.get('warnings') or [])
    out['artifact_status'] = verdict.get('result_status')
    result = (verdict.get('result')
              if isinstance(verdict.get('result'), dict) else {})
    out['result_status'] = result.get('status')
    out['pattern'] = result.get('pattern')
    out['primary'] = result.get('primary')
    out['primary_sign'] = result.get('primary_sign')
    out['n_specs'] = result.get('n_specs')
    out['n_families'] = result.get('n_families')
    out['p_iut'] = result.get('p_iut')
    out['iut_p_floor'] = result.get('iut_p_floor')
    out['resolution_limited'] = result.get('resolution_limited')
    summary = (result.get('summary')
               if isinstance(result.get('summary'), dict) else {})
    out['sign_agreement'] = summary.get('sign_agreement')
    out['n_sign_flips'] = summary.get('n_sign_flips')
    out['n_reject_raw'] = summary.get('n_reject_raw')
    claim = result.get('claim') if isinstance(result.get('claim'), dict) else {}
    out['robust_across_specs'] = bool(claim.get('robust_across_specs'))
    families = result.get('families')
    if isinstance(families, dict):
        for name, block in sorted(families.items()):
            if not isinstance(block, dict):
                continue
            if block.get('joint_status') == 'not_comparable':
                out['not_comparable_families'].append(name)
            elif block.get('joint_status') == 'not_estimable':
                out['not_estimable_families'].append(name)
    if out['status'] == 'ok':
        # A self-consistent artifact can honestly measure a non-robust
        # pattern.  That is not a tamper failure, but it is not a green
        # robustness claim either.
        if out['result_status'] == 'not_estimable':
            out['status'] = 'not_verifiable_locally'
        elif out['result_status'] == 'fail':
            out['status'] = 'fail'
        elif out['result_status'] == 'warnings':
            out['status'] = 'warnings'
    if verbose:
        print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
    return out


def audit_analysis_multiverse_artifact(path=None, *, verbose=True):
    """Single-artifact replay alias for :func:`audit_analysis_multiverse`."""
    return audit_analysis_multiverse(path, verbose=verbose)
