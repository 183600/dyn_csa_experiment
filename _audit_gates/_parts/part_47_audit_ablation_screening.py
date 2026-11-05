# Audit gate: budget-constrained ablation screening replay.
#
# The analysis lives in ``explib`` (part_81).  A stored
# ``ablation_screening.json`` records the declared components, defining
# relations, exact measured cells, model terms and the fitted alias table in
# ``inputs``/``result``.  This gate replays every derived field from those
# cells.  A missing artifact is not a pass; a self-consistent but
# ``not_estimable`` screening is reported as not-verifiable locally; a changed
# design, alias class, effect, family p or a smuggled frozen-decision field is
# a failure.  It never reads a split and never changes a frozen p-value, Holm
# family or confirmed verdict.

import json
import os


_AS_GATE_DEFAULTS = (
    os.path.join('analysis_v12', 'ablation_screening.json'),
    'ablation_screening.json',
)

_as_gate_load = _gate_load  # shared loader: unreadable/malformed == no evidence


def _as_gate_resolve(path):
    if path is not None:
        if isinstance(path, (str, os.PathLike)):
            return os.fspath(path)
        return None
    for candidate in _AS_GATE_DEFAULTS:
        full = os.path.join(REPO, candidate)  # noqa: F821
        if os.path.exists(full):
            return full
    return None


def audit_ablation_screening(path=None, *, verbose=True):
    """Replay a stored ``ablation_screening.json`` artifact.

    ``not_verifiable_locally`` when no artifact exists or when the design is
    structurally incomplete/rank deficient; ``warnings`` when the artifact is
    internally consistent but the screening is resolution limited or no main
    effect survives the family correction; ``fail`` for malformed inputs, a
    non-reproducing digest, tampered fields or a smuggled frozen-decision key.
    """
    L = _import_exp_lib()  # noqa: F821
    out = {
        'kind': 'ablation_screening_audit',
        'status': 'not_verifiable_locally',
        'path': None,
        'artifact_status': None,
        'result_status': None,
        'resolution': None,
        'resolution_label': None,
        'n_cells': None,
        'n_cells_full': None,
        'saved_cells': None,
        'n_seeds': None,
        'n_terms': None,
        'n_significant_main': None,
        'family_method': None,
        'p_floor_single': None,
        'resolution_limited': None,
        'main_effects_licensed': False,
        'interactions_licensed': False,
        'aliased_terms': [],
        'problems': [],
        'warnings': [],
        'note': ('replays an exploratory fractional-factorial screening readout '
                 'from its measured cells; it does not change the frozen '
                 'confirmatory Holm family or a confirmed verdict'),
    }
    resolved = _as_gate_resolve(path)
    if resolved is None:
        out['problems'].append('no ablation_screening.json artifact found')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out
    out['path'] = resolved
    if not os.path.exists(resolved):
        out['problems'].append(f'{resolved}: artifact does not exist')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out
    data = _as_gate_load(resolved)  # noqa: F821
    if not isinstance(data, dict):
        out['status'] = 'fail'
        out['problems'].append(f'{resolved}: not a readable JSON object')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out
    verdict = L.validate_ablation_screening(data)
    out['status'] = verdict.get('status', 'fail')
    out['problems'] = list(verdict.get('problems') or [])
    out['warnings'] = list(verdict.get('warnings') or [])
    result = (verdict.get('result')
              if isinstance(verdict.get('result'), dict) else {})
    out['artifact_status'] = verdict.get('result_status')
    out['result_status'] = result.get('status')
    out['resolution'] = result.get('resolution')
    out['resolution_label'] = result.get('resolution_label')
    out['n_cells'] = result.get('n_cells')
    out['n_cells_full'] = result.get('n_cells_full')
    out['saved_cells'] = result.get('saved_cells')
    out['n_seeds'] = result.get('n_seeds')
    terms = result.get('terms') if isinstance(result.get('terms'), dict) else {}
    out['n_terms'] = len(terms)
    out['n_significant_main'] = result.get('n_significant_main')
    family = result.get('family') if isinstance(result.get('family'), dict) else {}
    out['family_method'] = family.get('method')
    out['p_floor_single'] = family.get('p_floor_single')
    out['resolution_limited'] = family.get('resolution_limited')
    out['main_effects_licensed'] = bool(result.get('main_effects_licensed'))
    out['interactions_licensed'] = bool(result.get('interactions_licensed'))
    aliased = sorted(
        name for name, row in terms.items()
        if isinstance(row, dict) and row.get('aliased_with'))
    out['aliased_terms'] = aliased
    out['warnings'] = list(result.get('warnings') or []) + out['warnings']
    if out['status'] == 'ok':
        # A self-consistent artifact can honestly describe a design that is
        # incomplete or a screening null.  That is not a tamper failure, but
        # it is not a green measured effect either.
        if out['result_status'] in ('not_estimable', 'not_applicable'):
            out['status'] = 'not_verifiable_locally'
        elif out['result_status'] == 'fail':
            out['status'] = 'fail'
        elif out['result_status'] == 'warnings':
            out['status'] = 'warnings'
        elif out['resolution_limited'] is True:
            out['status'] = 'warnings'
    if verbose:
        print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
    return out


def audit_ablation_screening_artifact(path=None, *, verbose=True):
    """Single-artifact replay alias for :func:`audit_ablation_screening`."""
    return audit_ablation_screening(path, verbose=verbose)
