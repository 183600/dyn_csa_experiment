# Audit gate: distribution-free exact sign-flip TOST companion.
#
# ``tost_paired`` and ``confirmation_practical_significance`` attach a nested
# ``distribution_free`` block next to the historical paired-t TOST.  That
# block is a one-sided exact sign-flip test on each shifted vector, with a
# discrete p-value floor.  This gate replays either a standalone
# ``exact_tost_artifact`` or the nested blocks in a final-test stats file.
#
# It never changes the t-TOST decision, a frozen p/Holm/confirmed verdict, or
# any measurement.  A missing companion on a legacy artifact is a warning;
# a tampered or non-recomputable companion is a failure.

import json
import math
import os


_ETOST_ARTIFACT_DEFAULTS = (
    os.path.join('analysis_v12', 'exact_tost.json'),
    'exact_tost.json',
)
_ETOST_STATS_DEFAULTS = (
    os.path.join('analysis_v12', 'test_stats.json'),
    os.path.join('analysis_v12', 'stats.json'),
)
# Fields the enclosing practical-significance row adds to the companion.  They
# are checked separately because they are not part of the standalone test
# result, but they may not be used to smuggle an arbitrary extra field.
_ETOST_OPTIONAL_FIELDS = ('t_equivalent', 'agreement')
_ETOST_FROZEN_FIELDS = (
    'confirmed', 'p_holm', 'p_exact_signflip', 'p_value', 'verified',
    'verdict', 'decision', 'significant', 'significance',
)


def _etost_gate_resolve(path, defaults):
    if path is not None:
        if isinstance(path, (str, os.PathLike)):
            return os.fspath(path)
        return None
    for rel in defaults:
        candidate = os.path.join(REPO, rel)  # noqa: F821
        if os.path.exists(candidate):
            return candidate
    return None


def _etost_gate_repr(value, limit=200):
    """Bounded repr for a malformed companion value in a diagnostic."""
    try:
        text = repr(value)
    except Exception:
        return f'<{type(value).__name__}>'
    return text if len(text) <= limit else text[:limit] + '...'


def _etost_gate_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _etost_gate_equal(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    an, bn = _etost_gate_number(a), _etost_gate_number(b)
    if an is not None and bn is not None:
        return math.isclose(an, bn, rel_tol=1e-12, abs_tol=1e-12)
    return a == b


def _etost_gate_deltas(row):
    """Per-seed analysis-scale deltas from a practical-significance row."""
    if not isinstance(row, dict):
        return None
    raw = row.get('per_seed_log_ratio')
    if isinstance(raw, list) and raw:
        vals = [_etost_gate_number(x) for x in raw]
        if all(v is not None for v in vals):
            return vals
        return None
    pa, pb = row.get('per_seed_ppl_a'), row.get('per_seed_ppl_b')
    if (isinstance(pa, list) and isinstance(pb, list) and pa and len(pa) == len(pb)):
        vals = []
        for a, b in zip(pa, pb):
            fa, fb = _etost_gate_number(a), _etost_gate_number(b)
            if fa is None or fb is None or fa <= 0.0 or fb <= 0.0:
                return None
            vals.append(math.log(fa) - math.log(fb))
        return vals
    return None


def _etost_gate_compare(name, stored, replay, t_equivalent=None):
    """Return ``(problems, warnings)`` for one nested companion block."""
    problems, warnings = [], []
    if not isinstance(stored, dict):
        return problems, [f'{name}: no distribution_free companion block']
    if stored.get('status') not in (None, 'ok'):
        return problems, [f'{name}: companion status is {_etost_gate_repr(stored.get("status"))}']
    # Replay every derived field, not just the headline p-values.  The
    # companion also carries the shifted-vector floors, drop counts, methods and
    # prose; leaving any of them unchecked lets a hand-edited file keep a valid
    # p_tost while claiming a different resolution or dropping a seed.
    for key in sorted(replay):
        if key not in stored:
            problems.append(f'{name}.distribution_free.{key} is missing')
        elif not _etost_gate_equal(stored.get(key), replay.get(key)):
            problems.append(
                f'{name}.distribution_free.{key}: stored={_etost_gate_repr(stored.get(key))}, '
                f'replay={_etost_gate_repr(replay.get(key))}')
    extra = sorted(set(stored) - set(replay) - set(_ETOST_OPTIONAL_FIELDS))
    if extra:
        problems.append(
            f'{name}.distribution_free carries field(s) outside the replay '
            'schema: ' + ', '.join(repr(key) for key in extra))
    stolen = sorted(set(stored) & set(_ETOST_FROZEN_FIELDS))
    if stolen:
        problems.append(
            f'{name}.distribution_free carries frozen-decision field(s) that '
            'a descriptive companion must not contain: ' + repr(stolen))
    if 't_equivalent' in stored and not _etost_gate_equal(
            stored.get('t_equivalent'), t_equivalent):
        problems.append(
            f'{name}.distribution_free.t_equivalent: '
            f'stored={_etost_gate_repr(stored.get("t_equivalent"))}, t-block={_etost_gate_repr(t_equivalent)}')
    if 'agreement' in stored:
        expected = None
        if stored.get('equivalent') is not None and t_equivalent is not None:
            expected = bool(stored.get('equivalent') == t_equivalent)
        if stored.get('agreement') is not expected:
            problems.append(
                f'{name}.distribution_free.agreement: '
                f'stored={_etost_gate_repr(stored.get("agreement"))}, expected={_etost_gate_repr(expected)}')
    if stored.get('agreement') is False:
        warnings.append(
            f'{name}: t-TOST concludes equivalence but the exact sign-flip '
            'companion does not; do not quote the t-only result as a '
            'distribution-free equivalence claim')
    if stored.get('resolution_limited') is True:
        warnings.append(
            f'{name}: the exact TOST resolution floor is >= alpha at this n; '
            'equivalence is not attainable regardless of the observed effect')
    if (t_equivalent is True and stored.get('equivalent') is False
            and stored.get('agreement') is not False):
        warnings.append(
            f'{name}: t-TOST says equivalent while the exact companion says '
            'no; report both readings and the edge of the SESOI')
    return problems, warnings


def _etost_gate_scan_stats(path, out):
    """Replay the nested companions in a final-test stats artifact."""
    L = _import_exp_lib()  # noqa: F821
    ft = _load_finaltest_stats(out, path)  # noqa: F821
    if ft is None:
        return out
    block = ft.get('practical_significance') if isinstance(ft, dict) else None
    rows = block.get('comparisons') if isinstance(block, dict) else None
    comps = ft.get('comparisons') if isinstance(ft, dict) else None
    names = set()
    if isinstance(rows, dict):
        names.update(str(k) for k in rows)
    if isinstance(comps, dict):
        for name, comp in comps.items():
            if isinstance(comp, dict) and isinstance(
                    comp.get('practical_significance'), dict):
                names.add(str(name))
    out['path'] = path
    out['mode'] = 'stats'
    out['comparisons'] = {}
    out['n_replayed'] = 0
    out['n_missing'] = 0
    out['n_skipped'] = 0
    for name in sorted(names):
        comp = comps.get(name) if isinstance(comps, dict) else None
        inner = (comp.get('practical_significance')
                 if isinstance(comp, dict) else None)
        row = (inner if isinstance(inner, dict)
               else (rows.get(name) if isinstance(rows, dict) else None))
        if not isinstance(row, dict):
            out['n_skipped'] += 1
            continue
        if row.get('status') == 'not_declared':
            out['n_skipped'] += 1
            continue
        margin = _etost_gate_number(row.get('margin'))
        conf = _etost_gate_number(
            row.get('tost_conf',
                    block.get('tost_conf') if isinstance(block, dict) else None))
        deltas = _etost_gate_deltas(row)
        if margin is None or margin <= 0.0 or conf is None or deltas is None:
            out['n_skipped'] += 1
            continue
        replay = L.exact_signflip_tost(deltas, margin=margin, conf=conf)
        if replay.get('status') != 'ok':
            out['problems'].append(
                f'{name}: exact TOST replay is not estimable '
                f'({replay.get("reason")})')
            continue
        stored = row.get('distribution_free')
        if not isinstance(stored, dict):
            out['n_missing'] += 1
            out['warnings'].append(
                f'{name}: legacy practical-significance block has no '
                'distribution_free companion')
            continue
        t_equivalent = row.get('equivalent')
        if t_equivalent is None:
            # Some stored practical blocks keep the t decision only in the
            # companion's provenance field; do not turn that into a failure.
            t_equivalent = stored.get('t_equivalent')
        probs, warns = _etost_gate_compare(
            name, stored, replay, t_equivalent)
        out['problems'].extend(probs)
        out['warnings'].extend(warns)
        out['comparisons'][name] = {
            'p_tost': replay.get('p_tost'),
            'p_floor': replay.get('p_floor'),
            'p_floor_by_n': replay.get('p_floor_by_n'),
            'equivalent': replay.get('equivalent'),
            'method': replay.get('method'),
        }
        out['n_replayed'] += 1
    if out['problems']:
        out['status'] = 'fail'
    elif out['n_replayed'] == 0:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('the stats artifact carries no replayable '
                         'practical-significance TOST row')
    elif out['n_missing'] or out['warnings']:
        out['status'] = 'warnings'
    else:
        out['status'] = 'ok'
    return out


def audit_exact_tost(path=None, *, verbose=True):
    """Audit a stored exact-TOST artifact or the nested stats companions.

    Resolution order: an explicit path; ``analysis_v12/exact_tost.json``;
    the final-test stats artifact's nested ``distribution_free`` blocks.
    A standalone artifact is replayed and hash-checked.  A legacy stats
    artifact without the companion is a warning, never a clean pass.
    """
    L = _import_exp_lib()  # noqa: F821
    out = {
        'kind': 'exact_tost_audit', 'schema': 1, 'status': 'ok',
        'mode': None, 'path': None, 'problems': [], 'warnings': [],
        'comparisons': {}, 'n_replayed': 0, 'n_missing': 0, 'n_skipped': 0,
        'note': ('distribution-free exact sign-flip TOST companion replay; '
                 'it never changes the paired-t decision, p/Holm, or the '
                 'frozen `confirmed` set'),
    }
    artifact_path = _etost_gate_resolve(path, _ETOST_ARTIFACT_DEFAULTS)
    if path is not None and (artifact_path is None
                             or not os.path.exists(artifact_path)):
        out['status'] = 'not_verifiable_locally'
        out['path'] = artifact_path
        out['reason'] = 'the explicit exact-TOST path does not exist'
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out
    if artifact_path is not None and os.path.exists(artifact_path):
        artifact = _gate_load(artifact_path)  # noqa: F821
        if artifact is None:
            out['status'] = 'fail'
            out['problems'].append(
                f'{artifact_path} exists but is not a readable JSON object')
        elif isinstance(artifact, dict) and (
                'practical_significance' in artifact
                or 'comparisons' in artifact
                or 'finaltest' in artifact):
            # An explicit path may point at the final-test stats file rather
            # than a standalone artifact; replay its nested companions.
            out = _etost_gate_scan_stats(artifact_path, out)
        else:
            out = L.validate_exact_tost(artifact)
            out['kind'] = 'exact_tost_audit'
            out['mode'] = 'artifact'
            out['path'] = artifact_path
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:5000])
        return out
    stats_path = _etost_gate_resolve(None, _ETOST_STATS_DEFAULTS)
    if stats_path is not None and os.path.exists(stats_path):
        out = _etost_gate_scan_stats(stats_path, out)
        if verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2)[:6000])
        return out
    out['status'] = 'not_verifiable_locally'
    out['reason'] = ('no exact_tost.json artifact and no final-test stats '
                     'artifact with a practical-significance block')
    if verbose:
        print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
    return out


def audit_exact_tost_artifact(path=None):
    """Narrow companion: replay one standalone exact-TOST artifact."""
    L = _import_exp_lib()  # noqa: F821
    p = _etost_gate_resolve(path, _ETOST_ARTIFACT_DEFAULTS)
    if p is None or not os.path.exists(p):
        out = {'kind': 'exact_tost_validation',
               'status': 'not_verifiable_locally', 'path': p,
               'problems': ['no stored exact-TOST artifact'],
               'warnings': []}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    artifact = _gate_load(p)  # noqa: F821
    out = L.validate_exact_tost(artifact)
    out['path'] = p
    print(json.dumps(out, ensure_ascii=False, indent=2)[:5000])
    return out
