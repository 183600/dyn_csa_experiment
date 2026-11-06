# Audit gate: replay the budget x ablation interaction readout.
#
# A single-budget ablation table cannot establish that a component's effect is
# stable as training compute grows.  This gate replays the standalone
# `analysis_v12/budget_ablation.json` artifact (there is no embedded v12 block:
# the frozen confirmation panel is a fixed-budget design).  It never trains,
# reads a split, or changes a frozen p/Holm/confirmed verdict.

import json
import os


_BA_GATE_DEFAULTS = (
    os.path.join('analysis_v12', 'budget_ablation.json'),
    os.path.join('analysis_audit', 'budget_ablation.json'),
    'budget_ablation.json',
)

_ba_gate_load = _gate_load  # shared loader: unreadable/malformed == no evidence


def _ba_gate_repr(value, limit=160):
    """Bounded, exception-safe repr for an untrusted path/diagnostic value."""
    try:
        text = repr(value)
    except Exception:
        return f'<unrepresentable {type(value).__name__}>'
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _ba_gate_path_text(value):
    """Filesystem path as text, or None when the value is not usable."""
    try:
        return os.fsdecode(os.fspath(value))
    except Exception:
        return None


def _ba_gate_resolve(path):
    if path is not None:
        if isinstance(path, (str, os.PathLike)):
            try:
                return os.fspath(path)
            except Exception:
                return None
        return None
    for candidate in _BA_GATE_DEFAULTS:
        full = os.path.join(REPO, candidate)  # noqa: F821
        if os.path.exists(full):
            return full
    return None


def audit_budget_ablation(path=None, *, verbose=True):
    """Replay a budget-ablation artifact.

    Status semantics:
      * ``ok`` — the artifact/hash/replay agrees and the result names no caveat;
      * ``warnings`` — it replays, but the result is not estimable, is
        resolution-limited/scale-sensitive, or contains a sign reversal;
      * ``not_verifiable_locally`` — no artifact, or the declared ladder is
        incomplete/not estimable (a missing artifact is not a pass);
      * ``fail`` — malformed/tampered artifact or a replay mismatch.
    """
    L = _import_exp_lib()  # noqa: F821
    out = {
        'kind': 'budget_ablation_audit',
        'status': 'not_verifiable_locally',
        'path': None,
        'artifact_status': None,
        'result_status': None,
        'n_estimable': None,
        'family_m': None,
        'interaction_detected_components': [],
        'interaction_detected_raw_only_components': [],
        'scale_sensitive_components': [],
        'sign_reversal_components': [],
        'underpowered_components': [],
        'inconclusive_components': [],
        'extrapolable_components': [],
        'problems': [],
        'warnings': [],
        'note': ('exploratory budget x ablation interaction: paired log-PPL '
                 'slope on log(budget), exact sign-flip, one Holm family and '
                 'an optional exact TOST stability companion.  It does not '
                 'change a frozen p/Holm/verified/confirmed verdict.'),
    }
    resolved = _ba_gate_resolve(path)
    if path is not None and resolved is None:
        # A caller-supplied path is untrusted too: ``os.path.exists(10**10000)``
        # raises, and an arbitrary object is not JSON-serialisable in the
        # report.  Fail closed before touching the filesystem.
        out['status'] = 'fail'
        out['problems'].append(
            'path is not a usable filesystem path: '
            f'{_ba_gate_repr(path)}')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
        return out
    if resolved is None:
        out['problems'].append(
            'no budget_ablation.json artifact was found in the default '
            'locations')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
        return out
    text = _ba_gate_path_text(resolved)
    if text is None:
        out['status'] = 'fail'
        out['problems'].append(
            'resolved artifact path is not usable: '
            f'{_ba_gate_repr(resolved)}')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
        return out
    resolved = text
    out['path'] = resolved
    if not os.path.exists(resolved):
        out['problems'].append(f'{resolved}: artifact does not exist')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
        return out
    data = _ba_gate_load(resolved)  # noqa: F821
    if not isinstance(data, dict):
        out['status'] = 'fail'
        out['problems'].append(f'{resolved}: not a readable JSON object')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
        return out
    if data.get('kind') != 'budget_ablation':
        out['status'] = 'fail'
        out['problems'].append(
            f'{resolved}: not a budget_ablation artifact')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
        return out
    verdict = L.validate_budget_ablation(data)
    out['artifact_status'] = verdict.get('status')
    out['result_status'] = verdict.get('result_status')
    out['problems'] = list(verdict.get('problems') or [])
    if verdict.get('status') == 'fail':
        out['status'] = 'fail'
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
        return out
    result = verdict.get('result') if isinstance(verdict.get('result'), dict) else {}
    if not result:
        result = data.get('result') if isinstance(data.get('result'), dict) else {}
    out['n_estimable'] = result.get('n_estimable')
    family = result.get('family') if isinstance(result.get('family'), dict) else {}
    out['family_m'] = family.get('m')
    components = result.get('components')
    components = components if isinstance(components, dict) else {}
    seen_warnings = []
    for name, row in components.items():
        if not isinstance(row, dict):
            continue
        if row.get('status') != 'ok':
            out['inconclusive_components'].append(name)
            continue
        if row.get('interaction_detected'):
            out['interaction_detected_components'].append(name)
        elif row.get('interaction_detected_raw'):
            out['interaction_detected_raw_only_components'].append(name)
        if row.get('scale_sensitive'):
            out['scale_sensitive_components'].append(name)
        if row.get('sign_reversal'):
            out['sign_reversal_components'].append(name)
        slope = row.get('log_slope') if isinstance(row.get('log_slope'), dict) else {}
        if slope.get('underpowered'):
            out['underpowered_components'].append(name)
        if row.get('single_budget_extrapolation_allowed'):
            out['extrapolable_components'].append(name)
        if row.get('classification') == 'inconclusive':
            out['inconclusive_components'].append(name)
        for warning in (row.get('warnings') or []):
            text = str(warning)
            if text not in seen_warnings:
                seen_warnings.append(text)
    for warning in (result.get('warnings') or []):
        text = str(warning)
        if text not in seen_warnings:
            seen_warnings.append(text)
    out['warnings'] = seen_warnings
    if result.get('status') == 'not_estimable':
        out['status'] = 'not_verifiable_locally'
        out['problems'].append(
            'the budget-ablation result is not estimable: '
            + str(result.get('reason') or 'no reason'))
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
        return out
    # A sign reversal or scale disagreement is a substantive caveat in the
    # result, not a replay failure.  A detected interaction is the intended
    # finding and is reported without turning the gate red.
    out['status'] = ('warnings'
                     if (verdict.get('status') == 'warnings'
                         or result.get('status') == 'warnings'
                         or out['sign_reversal_components']
                         or out['scale_sensitive_components'])
                     else 'ok')
    if verbose:
        print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
    return out
