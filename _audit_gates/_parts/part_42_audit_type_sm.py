# Audit gate: design-stage Type S / Type M sign and magnitude error.
#
# The analysis lives in ``explib`` (part_76).  The gate reads the frozen
# manifest and, when available, the validation ``power_validation.json``
# written before the one-shot test access.  It reports, per primary
# comparison, the conditional wrong-sign probability and expected magnitude
# exaggeration at the pilot-anchored effect and at smaller effects.  A missing
# pilot is ``not_verifiable``, never green; the gate never reads a split and
# never changes a p-value/Holm/frozen verdict.

import json
import hashlib
import os


_TSM_GATE_PILOT_DEFAULTS = (
    os.path.join('analysis_v12', 'power_validation.json'),
    'power_validation.json',
)
_TSM_GATE_ARTIFACT_DEFAULTS = (
    os.path.join('analysis_v12', 'type_sm_design.json'),
    'type_sm_design.json',
)


def _tsm_gate_resolve(path, defaults):
    """Resolve a path; shared rule in :func:`_gate_resolve_first`."""
    return _gate_resolve_first(path, defaults)


def audit_type_sm(manifest_path=None, pilot_path=None, *, verbose=True):
    """Type S/M design audit for every frozen primary comparison.

    Zero GPU.  Reads ``confirmation_frozen.json`` and the validation power
    artifact; no test split, no training, no data loader.  The gate status is
    ``not_verifiable`` when the pilot is missing (a plan cannot be checked
    without a variance anchor), ``warnings`` when a comparison misses the
    planning thresholds, and ``fail`` only for malformed evidence or an
    explicitly required threshold that is not met.
    """
    L = _import_exp_lib()  # noqa: F821
    mp = manifest_path
    if mp is None:
        mp = os.path.join(REPO, 'confirmation_frozen.json')  # noqa: F821
    elif isinstance(mp, (str, os.PathLike)):
        mp = os.fspath(mp)
    else:
        return {'kind': 'type_sm_audit', 'status': 'fail',
                'problems': ['manifest_path must be a path or None'],
                'violations': [], 'warnings': [], 'comparisons': {}}
    manifest = _gate_load(mp)  # noqa: F821
    if not isinstance(manifest, dict):
        out = {'kind': 'type_sm_audit', 'status': 'fail',
               'manifest': mp, 'pilot': None, 'pilot_status': 'not_available',
               'problems': [f'frozen manifest is missing/unreadable: {mp}'],
               'violations': [], 'warnings': [], 'comparisons': {}}
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out
    pp = _tsm_gate_resolve(pilot_path, _TSM_GATE_PILOT_DEFAULTS)
    pilot = _gate_load(pp) if pp and os.path.exists(pp) else None  # noqa: F821
    # The driver loader stamps the file SHA-256 on the manifest; a raw
    # json.load does not.  Bind the raw manifest here so a pilot produced for
    # another freeze cannot anchor this one.
    if isinstance(manifest, dict) and not manifest.get('manifest_sha256'):
        try:
            with open(mp, 'rb') as _mf:
                manifest['manifest_sha256'] = hashlib.sha256(
                    _mf.read()).hexdigest()
        except OSError:
            pass
    report = L.type_sm_from_manifest(manifest, pilot)
    report['manifest'] = mp
    report['pilot'] = pp
    if pilot is None:
        reason = ('no validation power artifact (analysis_v12/'
                  'power_validation.json); Type S/M is a design readout and '
                  'cannot be computed without a paired-delta SD')
        if reason not in report['warnings']:
            report['warnings'].append(reason)
        if report.get('status') == 'ok':
            report['status'] = 'not_verifiable'
    if pp and pilot is not None:
        report['pilot_status'] = 'available'
    if verbose:
        print(json.dumps(report, ensure_ascii=False, indent=2)[:6000])
    return report


def audit_type_sm_artifact(path=None):
    """Replay one stored ``type_sm_design.json`` artifact.

    This is the narrow companion to :func:`audit_type_sm`: a single assumed
    effect can be published and rechecked without re-reading the manifest.
    """
    L = _import_exp_lib()  # noqa: F821
    p = _tsm_gate_resolve(path, _TSM_GATE_ARTIFACT_DEFAULTS)
    if p is None or not os.path.exists(p):
        out = {'kind': 'type_sm_design_validation', 'status':
               'not_verifiable_locally', 'path': p,
               'problems': ['no stored Type S/M artifact']}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    artifact = _gate_load(p)  # noqa: F821
    out = L.validate_type_sm_design(artifact)
    out['path'] = p
    print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
    return out
