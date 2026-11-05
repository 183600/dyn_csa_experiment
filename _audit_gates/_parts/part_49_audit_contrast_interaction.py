# Audit gate: replay the 2x2 difference-in-differences readout.
#
# The frozen v12 family contains both a sink-matched contrast and a no-sink
# contrast.  Reading "one significant, one not" as evidence that the sink
# explains the main result is the difference-in-significance fallacy; the
# estimator for that sentence is the interaction.  This gate replays either a
# standalone `analysis_v12/contrast_interaction.json` artifact or the
# `sink_factorial` block embedded in `test_stats.json`.  It never trains,
# reads a split, or changes a frozen p/Holm/confirmed verdict.

import json
import os


_CI_GATE_DEFAULTS = (
    os.path.join('analysis_v12', 'contrast_interaction.json'),
    os.path.join('analysis_audit', 'contrast_interaction.json'),
    'contrast_interaction.json',
)

_ci_gate_load = _gate_load  # shared loader: unreadable/malformed == no evidence


def _ci_gate_resolve(path):
    if path is not None:
        if isinstance(path, (str, os.PathLike)):
            return os.fspath(path)
        return None
    for candidate in _CI_GATE_DEFAULTS:
        full = os.path.join(REPO, candidate)  # noqa: F821
        if os.path.exists(full):
            return full
    return None


def _ci_gate_from_test_stats(L, stored, path):
    """Replay the embedded sink 2x2 from the same test_stats file."""
    out = {
        'kind': 'contrast_interaction_audit',
        'status': 'not_verifiable_locally',
        'path': path,
        'source': 'test_stats.sink_factorial',
        'artifact_status': None,
        'result_status': stored.get('status'),
        'classification': None,
        'n': None,
        'scale_sensitive': None,
        'difference_in_significance_fallacy': None,
        'problems': [],
        'warnings': [],
        'note': ('embedded replay of the frozen sink x architecture 2x2; it '
                 'is exploratory and does not change confirmed/verified'),
    }
    if not isinstance(stored, dict):
        out['problems'].append('sink_factorial is not an object')
        return out
    if stored.get('status') == 'not_estimable':
        out['problems'].append(
            'sink_factorial is not estimable: '
            + str(stored.get('reason') or 'no reason'))
        return out
    try:
        with open(path, encoding='utf-8') as fh:
            ts = json.load(fh)
    except Exception as exc:
        out['status'] = 'fail'
        out['problems'].append(
            f'{path}: cannot read test_stats ({type(exc).__name__}: {exc})')
        return out
    by_length = ts.get('by_length') if isinstance(ts, dict) else None
    if not isinstance(by_length, dict):
        out['problems'].append(
            'test_stats has a sink_factorial block but no by_length cells to '
            'replay it from')
        return out
    try:
        replayed = L.confirmation_sink_interaction(
            by_length,
            endpoint=stored.get('endpoint', 'by_len.ppl@4096'),
            alpha=stored.get('alpha', 0.05),
            min_n=stored.get('min_n', 3),
            margin=stored.get('margin'),
            margin_relative_pct=stored.get('margin_relative_pct'))
    except Exception as exc:
        out['status'] = 'fail'
        out['problems'].append(
            f'replay failed: {type(exc).__name__}: {exc}')
        return out
    if replayed.get('status') == 'not_estimable':
        out['problems'].append(
            'replay is not estimable: '
            + str(replayed.get('reason') or 'no reason'))
        return out
    cmp = L.compare_contrast_interaction_results(stored, replayed)
    if not cmp.get('match'):
        out['status'] = 'fail'
        out['problems'].append(
            'stored sink_factorial differs from the replay of the same '
            'by-length cells')
        out['warnings'].extend(list(cmp.get('differences') or [])[:8])
        return out
    out['result_status'] = replayed.get('status')
    out['classification'] = (replayed.get('interpretation') or {}).get(
        'classification')
    out['n'] = replayed.get('n')
    out['scale_sensitive'] = (replayed.get('scale_agreement') or {}).get(
        'scale_sensitive')
    out['difference_in_significance_fallacy'] = (
        replayed.get('interpretation') or {}).get(
            'difference_in_significance_fallacy')
    out['warnings'].extend(list(replayed.get('warnings') or []))
    out['status'] = ('warnings' if (replayed.get('status') == 'warnings'
                                    or out['warnings'])
                     else 'ok')
    return out


def audit_contrast_interaction(path=None, *, verbose=True):
    """Replay a standalone 2x2 artifact or the embedded test_stats block.

    Status semantics:
      * ``ok`` — the artifact/hash/replay agrees and the block is clean;
      * ``warnings`` — it replays, but the result itself names a caveat
        (scale sensitivity, difference-in-significance risk, resolution);
      * ``not_verifiable_locally`` — no artifact, or the frozen 2x2 cells are
        incomplete/not estimable (a missing artifact is not a pass);
      * ``fail`` — malformed/tampered artifact or a replay mismatch.
    """
    L = _import_exp_lib()  # noqa: F821
    out = {
        'kind': 'contrast_interaction_audit',
        'status': 'not_verifiable_locally',
        'path': None,
        'source': None,
        'artifact_status': None,
        'result_status': None,
        'classification': None,
        'n': None,
        'scale_sensitive': None,
        'difference_in_significance_fallacy': None,
        'problems': [],
        'warnings': [],
        'note': ('exploratory 2x2 difference-in-differences: log-PPL '
                 'interaction with raw-PPL sensitivity, exact sign-flip and '
                 'family max-T/Holm fallback.  It does not change a frozen '
                 'p/Holm/verified/confirmed verdict.'),
    }
    resolved = _ci_gate_resolve(path)
    if resolved is None:
        ts_path = os.path.join(REPO, 'analysis_v12', 'test_stats.json')  # noqa: F821
        ts = _ci_gate_load(ts_path) if os.path.exists(ts_path) else None  # noqa: F821
        if isinstance(ts, dict) and isinstance(ts.get('sink_factorial'), dict):
            out = _ci_gate_from_test_stats(L, ts['sink_factorial'], ts_path)
        else:
            out['problems'].append(
                'no contrast_interaction.json artifact and no '
                'test_stats.json sink_factorial block')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
        return out
    out['path'] = resolved
    out['source'] = 'artifact'
    if not os.path.exists(resolved):
        out['problems'].append(f'{resolved}: artifact does not exist')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
        return out
    data = _ci_gate_load(resolved)  # noqa: F821
    if not isinstance(data, dict):
        out['status'] = 'fail'
        out['problems'].append(f'{resolved}: not a readable JSON object')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
        return out
    if data.get('kind') != 'contrast_interaction':
        out['status'] = 'fail'
        out['problems'].append(
            f'{resolved}: not a contrast_interaction artifact')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
        return out
    if not isinstance(data.get('inputs'), dict):
        # A bare result (for example the embedded block written to its own
        # file) has no stored inputs; replay it from test_stats instead of
        # trusting its numbers.
        ts_path = os.path.join(REPO, 'analysis_v12', 'test_stats.json')  # noqa: F821
        ts = _ci_gate_load(ts_path) if os.path.exists(ts_path) else None  # noqa: F821
        if isinstance(ts, dict) and isinstance(ts.get('sink_factorial'), dict):
            out = _ci_gate_from_test_stats(L, data, ts_path)
            out['path'] = resolved
            out['source'] = 'bare_result+test_stats'
        else:
            out['problems'].append(
                f'{resolved}: a bare result has no inputs to replay from, and '
                'no test_stats sink_factorial is available')
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
        return out
    verdict = L.validate_contrast_interaction(data)
    out['artifact_status'] = verdict.get('status')
    out['result_status'] = verdict.get('result_status')
    out['problems'] = list(verdict.get('problems') or [])
    out['warnings'] = list(verdict.get('warnings') or [])
    status = verdict.get('status')
    result = verdict.get('result') if isinstance(verdict.get('result'), dict) else {}
    out['classification'] = (result.get('interpretation') or {}).get(
        'classification')
    out['n'] = result.get('n')
    out['scale_sensitive'] = (result.get('scale_agreement') or {}).get(
        'scale_sensitive')
    out['difference_in_significance_fallacy'] = (
        result.get('interpretation') or {}).get(
            'difference_in_significance_fallacy')
    if status == 'fail':
        out['status'] = 'fail'
    elif status == 'not_estimable':
        out['status'] = 'not_verifiable_locally'
    elif status == 'warnings' or result.get('status') == 'warnings':
        out['status'] = 'warnings'
        out['warnings'].extend(list(result.get('warnings') or []))
    elif status == 'ok':
        out['status'] = 'ok'
    else:
        out['status'] = 'fail'
        out['problems'].append(f'unknown validation status {status!r}')
    if verbose:
        print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
    return out


def audit_contrast_interaction_artifact(path=None, *, verbose=True):
    """Single-artifact replay alias for :func:`audit_contrast_interaction`."""
    return audit_contrast_interaction(path, verbose=verbose)
