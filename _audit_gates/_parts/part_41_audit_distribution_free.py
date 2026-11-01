# Audit gate: distribution-free (conformal) prediction interval.
#
# The analysis lives in ``explib`` (part_10).  A stored artifact is a small,
# replayable claim: ``inputs`` carries the finite calibration effects and the
# nominal level, ``result`` is the order-statistic interval.  This gate
# rebuilds the interval from ``inputs`` and compares every derived field.  It
# never reads a split, never changes a p-value/Holm decision, and a missing
# artifact is not a pass.

import json
import math
import os


_DF_GATE_DEFAULTS = (
    os.path.join('analysis_v12', 'distribution_free_prediction.json'),
    'distribution_free_prediction.json',
)

_df_gate_load = _gate_load  # shared loader (part_00): unreadable == no evidence


def _df_gate_resolve(path):
    if path is not None:
        return os.fspath(path)
    for candidate in _DF_GATE_DEFAULTS:
        full = os.path.join(REPO, candidate)  # noqa: F821
        if os.path.exists(full):
            return full
    return None


def _df_gate_finite_number(value):
    """Whether ``value`` is a finite JSON number representable as float64.

    ``json`` can carry integers with more digits than a float64 can hold.
    Converting one raises ``OverflowError``; that is a malformed calibration
    value for this replay (the interval helper would drop it), not a reason to
    abort the audit.  Booleans are flags and are never measurements.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _df_gate_close(a, b, rel=1e-9, abs_=1e-12):
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError, OverflowError):
        return False
    if not (math.isfinite(fa) and math.isfinite(fb)):
        return False
    return abs(fa - fb) <= max(abs_, rel * max(1.0, abs(fa), abs(fb)))


def _df_gate_result_diff(stored, replay):
    """Field-by-field comparison of a stored result against a replay."""
    problems = []
    scalar_fields = (
        'kind', 'version', 'method', 'assumption', 'role',
        'available', 'n', 'n_input', 'n_dropped', 'conf', 'direction',
        'attained_conf', 'conf_achievable', 'coverage_shortfall',
        'n_needed_for_conf', 'n_shortfall', 'order_statistic_j',
        'interval_width', 'interval_side', 'favourable_side',
        'n_positive', 'n_negative', 'n_zero', 'reason',
    )
    for key in scalar_fields:
        a, b = stored.get(key), replay.get(key)
        if isinstance(a, bool) or isinstance(b, bool):
            if a is not b:
                problems.append(f'{key}: stored={a!r}, replay={b!r}')
            continue
        if a is None or b is None:
            if a is not b:
                problems.append(f'{key}: stored={a!r}, replay={b!r}')
            continue
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not _df_gate_close(a, b):
                problems.append(f'{key}: stored={a!r}, replay={b!r}')
            continue
        if a != b:
            problems.append(f'{key}: stored={a!r}, replay={b!r}')
    for key in ('interval', 'dropped_positions'):
        a, b = stored.get(key), replay.get(key)
        if a is None and b is None:
            continue
        if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
            problems.append(f'{key}: stored={a!r}, replay={b!r}')
            continue
        for i, (x, y) in enumerate(zip(a, b)):
            if isinstance(x, bool) or isinstance(y, bool):
                if x is not y:
                    problems.append(f'{key}[{i}]: stored={x!r}, replay={y!r}')
            elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
                if not _df_gate_close(x, y):
                    problems.append(f'{key}[{i}]: stored={x!r}, replay={y!r}')
            elif x != y:
                problems.append(f'{key}[{i}]: stored={x!r}, replay={y!r}')
    if stored.get('policy') != replay.get('policy'):
        problems.append('policy block differs from the replay')
    for key in ('confirmed', 'p_holm', 'verified', 'verdict', 'decision',
                'p_exact_signflip', 'p_value', 'p', 'alpha', 'holm',
                'significant'):
        if key in stored:
            problems.append(f'result must not carry frozen-decision field {key!r}')
    # This artifact's result schema is exactly the replay result.  Ignoring an
    # extra key would let a hand-edited file smuggle an unchecked claim (for
    # example ``significant: true``) next to a valid interval.
    extra = sorted(set(stored) - set(replay))
    if extra:
        problems.append(
            'result carries field(s) outside the replay schema: '
            + ', '.join(repr(key) for key in extra))
    return problems


def audit_distribution_free_prediction(path=None):
    """Replay a stored distribution-free prediction artifact.

    Artifact shape::

        {"kind": "distribution_free_prediction_artifact", "schema": 1,
         "inputs": {"deltas": [...], "conf": 0.95, "direction": null},
         "result": {...}}

    A missing artifact is ``not_verifiable_locally``; malformed inputs,
    tampered derived values or a frozen-decision field are ``fail``; a
    structurally valid artifact whose interval is unavailable replays as
    ``not_applicable``.
    """
    L = _import_exp_lib()  # noqa: F821
    out = {
        'kind': 'distribution_free_prediction_audit',
        'status': 'not_verifiable_locally',
        'path': None,
        'n_input': 0,
        'available': None,
        'replayed_available': None,
        'interval': None,
        'attained_conf': None,
        'n_needed_for_conf': None,
        'problems': [],
        'warnings': [],
        'note': ('replays the distribution-free order-statistic interval from '
                 'stored finite calibration effects; no p-value, Holm family '
                 'or frozen verdict is changed'),
    }
    p = _df_gate_resolve(path)
    if p is None:
        out['problems'].append('no distribution_free_prediction.json artifact found')
        return out
    out['path'] = p
    if not os.path.exists(p):
        out['problems'].append(f'{p}: artifact does not exist')
        return out
    data = _df_gate_load(p)
    if not isinstance(data, dict):
        out['status'] = 'fail'
        out['problems'].append(f'{p}: not readable JSON object')
        return out
    # The artifact type/schema is part of the evidence contract.  Without this
    # check any JSON object that happens to carry ``inputs``/``result`` would
    # replay as this artifact, so a renamed or version-mismatched file could
    # be accepted as a conformal-prediction record.
    expected_kind = getattr(
        L, 'DISTRIBUTION_FREE_PREDICTION_ARTIFACT_KIND',
        'distribution_free_prediction_artifact')
    expected_schema = getattr(
        L, 'DISTRIBUTION_FREE_PREDICTION_ARTIFACT_SCHEMA', 1)
    if data.get('kind') != expected_kind:
        out['status'] = 'fail'
        out['problems'].append(
            f'{p}: kind is {data.get("kind")!r}, not {expected_kind!r}')
        return out
    schema = data.get('schema')
    if (isinstance(schema, bool) or not isinstance(schema, int)
            or schema != expected_schema):
        out['status'] = 'fail'
        out['problems'].append(
            f'{p}: schema is {schema!r}, not {expected_schema!r}')
        return out
    inputs = data.get('inputs')
    stored = data.get('result')
    if not isinstance(inputs, dict):
        out['status'] = 'fail'
        out['problems'].append('inputs is not an object')
        return out
    if not isinstance(stored, dict):
        out['status'] = 'fail'
        out['problems'].append('result is not an object')
        return out
    deltas = inputs.get('deltas')
    if not isinstance(deltas, list):
        out['status'] = 'fail'
        out['problems'].append('inputs.deltas is not a list')
        return out
    if not all(_df_gate_finite_number(x) for x in deltas):
        out['status'] = 'fail'
        out['problems'].append(
            'inputs.deltas must contain finite JSON numbers representable as '
            'float64 only')
        return out
    try:
        replay = L.distribution_free_prediction_interval(
            deltas, conf=inputs.get('conf', 0.95),
            direction=inputs.get('direction'))
    except (TypeError, ValueError) as e:
        out['status'] = 'fail'
        out['problems'].append(f'inputs do not replay: {type(e).__name__}: {e}')
        return out
    out['n_input'] = len(deltas)
    out['available'] = stored.get('available')
    out['replayed_available'] = replay.get('available')
    out['interval'] = replay.get('interval')
    out['attained_conf'] = replay.get('attained_conf')
    out['n_needed_for_conf'] = replay.get('n_needed_for_conf')
    problems = _df_gate_result_diff(stored, replay)
    out['problems'].extend(problems)
    if problems:
        out['status'] = 'fail'
        return out
    if replay.get('available') is False:
        out['status'] = 'not_applicable'
        out['warnings'].append(
            'the artifact replays consistently but declares the prediction '
            'interval unavailable; it is not a measured interval')
    else:
        out['status'] = 'ok'
    return out
