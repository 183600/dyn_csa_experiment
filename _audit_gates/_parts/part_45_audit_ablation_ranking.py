# Audit gate: joint inference for Shapley component rankings (ablation layer).
#
# The analysis lives in ``explib`` (part_79).  A stored
# ``ablation_ranking.json`` records the exact PPL cells in ``inputs`` and the
# joint component/pair ranking in ``result``.  This gate replays every derived
# field from those cells.  A missing artifact is not a pass; a structurally
# valid but not-estimable ranking is reported as not-verifiable locally; a
# changed Shapley value, pair order, rank, hash or a smuggled frozen-decision
# field is a failure.  It never reads a split and never changes a frozen
# p-value, Holm family or confirmed verdict.

import json
import os


_AR_GATE_DEFAULTS = (
    os.path.join('analysis_v12', 'ablation_ranking.json'),
    'ablation_ranking.json',
)

_ar_gate_load = _gate_load  # shared loader: unreadable/malformed == no evidence


def _ar_gate_resolve(path):
    if path is not None:
        if isinstance(path, (str, os.PathLike)):
            return os.fspath(path)
        return None
    for candidate in _AR_GATE_DEFAULTS:
        full = os.path.join(REPO, candidate)  # noqa: F821
        if os.path.exists(full):
            return full
    return None


def audit_ablation_ranking(path=None, *, verbose=True):
    """Replay a stored ``ablation_ranking.json`` joint attribution artifact.

    ``not_verifiable_locally`` when no artifact exists or when the artifact's
    own result is ``not_estimable``/``not_applicable``; ``fail`` for malformed
    inputs, a non-reproducing digest, tampered fields, an undeclared component
    or a smuggled frozen-decision field.
    """
    L = _import_exp_lib()  # noqa: F821
    out = {
        'kind': 'ablation_ranking_audit',
        'status': 'not_verifiable_locally',
        'path': None,
        'artifact_status': None,
        'result_status': None,
        'primary_scale': None,
        'n_components': None,
        'n_seeds': None,
        'p_floor': None,
        'resolution_limited': None,
        'top1_licensed': False,
        'bottom1_licensed': False,
        'full_ranking_licensed': False,
        'ranking_consistency': None,
        'scale_sensitivity_status': None,
        'scale_flips': [],
        'method': None,
        'problems': [],
        'warnings': [],
        'note': ('replays the exploratory joint Shapley ranking from its '
                 'recorded lattice cells; it does not change the frozen '
                 'confirmatory Holm family or a confirmed verdict'),
    }
    p = _ar_gate_resolve(path)
    if p is None:
        out['problems'].append('no ablation_ranking.json artifact found')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out
    out['path'] = p
    if not os.path.exists(p):
        out['problems'].append(f'{p}: artifact does not exist')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out
    data = _ar_gate_load(p)  # noqa: F821
    if not isinstance(data, dict):
        out['status'] = 'fail'
        out['problems'].append(f'{p}: not a readable JSON object')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out
    verdict = L.validate_ablation_ranking(data)
    out['status'] = verdict.get('status', 'fail')
    out['problems'] = list(verdict.get('problems') or [])
    out['artifact_status'] = verdict.get('result_status')
    result = verdict.get('result') if isinstance(verdict.get('result'), dict) else {}
    out['result_status'] = result.get('status')
    out['primary_scale'] = result.get('primary_scale')
    out['n_components'] = result.get('n_components')
    out['n_seeds'] = result.get('n_seeds')
    out['p_floor'] = result.get('p_floor')
    out['resolution_limited'] = result.get('resolution_limited')
    out['top1_licensed'] = bool(result.get('top1_licensed'))
    out['bottom1_licensed'] = bool(result.get('bottom1_licensed'))
    out['full_ranking_licensed'] = bool(result.get('full_ranking_licensed'))
    out['ranking_consistency'] = result.get('ranking_consistency')
    out['method'] = result.get('method')
    sens = result.get('scale_sensitivity')
    if isinstance(sens, dict):
        out['scale_sensitivity_status'] = sens.get('status')
        out['scale_flips'] = list(sens.get('pair_order_flips') or [])
    if out['status'] == 'ok':
        # A self-consistent artifact can honestly describe an unusable
        # lattice (incomplete cells / <2 components).  That is not a green
        # measured ranking, and it is not a tamper failure either.
        if out['result_status'] in ('not_estimable', 'not_applicable'):
            out['status'] = 'not_verifiable_locally'
        elif out['result_status'] == 'fail':
            out['status'] = 'fail'
        elif out['result_status'] == 'warnings':
            out['status'] = 'warnings'
    if verbose:
        print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
    return out


def audit_ablation_ranking_artifact(path=None, *, verbose=True):
    """Single-artifact replay alias for :func:`audit_ablation_ranking`."""
    return audit_ablation_ranking(path, verbose=verbose)
