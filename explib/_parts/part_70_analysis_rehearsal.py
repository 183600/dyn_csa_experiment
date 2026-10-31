"""Known-truth rehearsal of the frozen decision path (preparation, no test read).

The repository already validates its *statistical primitives* (type-I error,
interval coverage, sign-flip power) and its *plan admission* (arms, seeds,
recipes, parameter matching).  Neither check exercises the path that turns a
set of stored per-seed records into a frozen verdict: exact sign-flip p ->
declared-family Holm -> three-way directional verdict -> coverage/n-floor
gates.  A mis-wired family size, a dropped expected sign, or an omission that
silently shrinks the correction family can therefore survive every existing
gate and still produce a confident-looking result after the one-shot test
read.

This fragment adds a zero-GPU, no-test-read **known-truth rehearsal**.  It
plants synthetic per-seed deltas into the *actual frozen comparison family*
and runs the same public statistics used by ``finaltest_stats``:

* null (balanced signs) must yield no supported/refuted comparison;
* a uniform effect in the pre-registered direction must be supported;
* the opposite effect must be refuted, not supported;
* a panel below the declared n floor / coverage requirement must be
  inconclusive even when its raw p is tiny;
* one strong effect plus one raw-significant-but-Holm-null effect must leave
  only the strong comparison supported (declared family, no family shrinkage);
* an omitted comparison must stay not_testable and keep its family slot.

It is a preparation control, not a new test: it adds no run, reads no split,
changes no frozen p/alpha/Holm/verdict, and its synthetic p-values are labelled
``synthetic`` and ``test_split_read=False`` so they can never be quoted as
evidence.  The rehearsal is bound into ``confirmation plan`` and exposed as a
standalone audit gate.
"""

import hashlib
import json
import math
import os
import time


ANALYSIS_REHEARSAL_KIND = 'analysis_rehearsal'
ANALYSIS_REHEARSAL_SCHEMA = 1
ANALYSIS_REHEARSAL_POLICY = {
    'version': 1,
    'role': (
        'pre-test known-truth rehearsal of the frozen decision path: exact '
        'sign-flip p, declared-family Holm, directional verdicts and '
        'coverage/n-floor gates on synthetic per-seed deltas from the real '
        'comparison family.'),
    'synthetic': True,
    'test_split_read': False,
    'rule': (
        'a scenario passes only when every comparison returns the verdict its '
        'planted truth predicts; the rehearsal is green only when every '
        'scenario passes.'),
    'boundary': (
        'the rehearsal validates the decision wiring on synthetic data.  It '
        'is not a power calculation, not a replication, not evidence about a '
        'model, and not a substitute for plan admission, the pre-test power '
        'gate or the result-level audits.  A green rehearsal cannot prove the '
        'analysis is correct; a red one means the decision path must not be '
        'used for a confirmatory read.'),
}


def _ar_text(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _ar_int(value, *, minimum=None):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        out = int(value)
    elif isinstance(value, float) and value.is_integer():
        out = int(value)
    else:
        return None
    if minimum is not None and out < minimum:
        return None
    return out


def _ar_canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False)


def _ar_sha_obj(obj):
    return hashlib.sha256(_ar_canonical(obj).encode('utf-8')).hexdigest()


def _ar_sha256_text(value):
    text = _ar_text(value)
    if text is None:
        return None
    text = text.lower()
    if len(text) != 64 or any(ch not in '0123456789abcdef' for ch in text):
        return None
    return text


def _ar_sign(value):
    text = _ar_text(value)
    if text is None:
        return None
    text = text.lower()
    return text if text in ('positive', 'negative') else None


def _ar_manifest_sha(manifest, manifest_path=None):
    """Return the recorded file hash, or hash the parsed object if absent."""
    recorded = _ar_sha256_text(manifest.get('manifest_sha256'))
    if recorded is not None:
        return recorded
    if isinstance(manifest_path, str) and manifest_path and os.path.isfile(manifest_path):
        try:
            with open(manifest_path, 'rb') as fh:
                return hashlib.sha256(fh.read()).hexdigest()
        except OSError:
            return None
    try:
        return _ar_sha_obj(manifest)
    except (TypeError, ValueError, RecursionError):
        return None


def _ar_manifest_canonical(manifest):
    """Canonical manifest identity, ignoring path/hash fields added at load."""
    if not isinstance(manifest, dict):
        return None
    body = {k: v for k, v in manifest.items()
            if k not in ('_path', 'manifest_sha256')}
    return _ar_sha_obj(body)


def _ar_load_manifest(path):
    if not isinstance(path, str) or not path:
        return None, 'manifest path is missing'
    if not os.path.exists(path):
        return None, f'manifest {path!r} does not exist'
    try:
        with open(path, encoding='utf-8') as fh:
            manifest = json.load(fh)
    except Exception as exc:
        return None, f'manifest {path!r} is not readable JSON ({type(exc).__name__}: {exc})'
    if not isinstance(manifest, dict):
        return None, f'manifest {path!r} is not a JSON object'
    return manifest, None


def _ar_specs(manifest):
    """Normalise the frozen comparison family into rehearsal specifications."""
    specs = []
    raw = manifest.get('primary_comparisons')
    if not isinstance(raw, list) or not raw:
        raise ValueError('manifest has no primary_comparisons list')
    for i, comp in enumerate(raw):
        if not isinstance(comp, dict):
            raise ValueError(f'primary_comparisons[{i}] is not an object')
        name = _ar_text(comp.get('name')) or f'comparison_{i}'
        if any(s['name'] == name for s in specs):
            raise ValueError(f'primary_comparisons contains duplicate name {name!r}')
        specs.append({
            'name': name,
            'a': _ar_text(comp.get('a')),
            'b': _ar_text(comp.get('b')),
            'endpoint': _ar_text(comp.get('endpoint')),
            'expected_sign': _ar_sign(comp.get('expected_sign')),
            'raw_expected_sign': comp.get('expected_sign'),
            'mechanism_claim_allowed': comp.get('mechanism_claim_allowed'),
        })
    return specs


def _ar_directional(specs):
    return [s for s in specs if s.get('expected_sign') is not None]


def _ar_deltas(specs, pattern, *, n_seeds, omitted=(), strong=None,
               marginal=None, marginal_bad=None):
    """Construct one synthetic per-seed delta vector per comparison.

    ``pattern`` is ``zero``, ``balanced``, ``forward``, ``reverse``,
    ``strong`` or ``marginal``.  Magnitudes are deliberately simple (1.0) so
    the planted sign counts, not an arbitrary scale, determine the exact
    sign-flip result.
    """
    out = {}
    for spec in specs:
        name = spec['name']
        sign = 1.0 if spec['expected_sign'] != 'negative' else -1.0
        if pattern == 'zero' or name in omitted:
            vals = [0.0] * n_seeds
        elif pattern == 'balanced':
            # Exact mean zero, including odd n (the extra pair is a zero).
            vals = []
            while len(vals) + 1 < n_seeds:
                vals.extend([1.0, -1.0])
            if len(vals) < n_seeds:
                vals.append(0.0)
        elif pattern == 'forward':
            vals = [sign] * n_seeds
        elif pattern == 'reverse':
            vals = [-sign] * n_seeds
        elif pattern == 'strong':
            vals = [sign if name == strong else 0.0 for _ in range(n_seeds)]
        elif pattern == 'marginal':
            if name == strong:
                vals = [sign] * n_seeds
            elif name == marginal:
                # Same-sign majority with a precomputed number of opposite
                # pairs: the raw exact two-sided p is < alpha, but Holm over
                # the declared family is not.  This is the family-shrinkage
                # trap the rehearsal must exclude.
                n_bad = (marginal_bad if marginal_bad is not None
                         else (2 if n_seeds >= 4 else 1))
                n_bad = max(1, min(int(n_bad), max(1, n_seeds - 1)))
                vals = [sign] * (n_seeds - n_bad) + [-sign] * n_bad
            else:
                vals = [0.0] * n_seeds
        else:
            raise ValueError(f'unknown rehearsal pattern {pattern!r}')
        out[name] = vals
    return out


def _ar_direction_ok(deltas, expected_sign):
    if expected_sign is None or not deltas:
        return None
    mean = float(sum(deltas)) / float(len(deltas))
    if mean == 0.0:
        return False
    return bool((mean > 0.0) == (expected_sign == 'positive'))


def _ar_family(manifest, specs, deltas_by_name, *, n_effective=None,
               coverage_ok=True, omitted=()):
    """Run the public numeric path and build the verdict input records."""
    alpha = manifest['statistics'].get('alpha', 0.05)
    min_n = manifest['statistics'].get('min_seeds_for_confirmation')
    min_n = _ar_int(min_n, minimum=2) or len(manifest.get('seeds') or [])
    pairs = []
    raw = {}
    for spec in specs:
        dl = list(deltas_by_name.get(spec['name']) or [])
        res = exact_signflip_p(dl)  # noqa: F821
        raw[spec['name']] = {
            'p_raw': res.get('p'),
            'n': res.get('n'),
            'n_nonzero': res.get('n_nonzero'),
            'p_floor': res.get('p_floor'),
            'method': res.get('method'),
        }
        p = res.get('p')
        valid = (isinstance(p, (int, float)) and not isinstance(p, bool)
                 and math.isfinite(float(p)) and 0.0 <= float(p) <= 1.0)
        pairs.append((spec['name'], float(p) if valid else 1.0))
    holm = holm_adjust(pairs, alpha=alpha)  # noqa: F821
    verdict_in = {}
    for spec in specs:
        name = spec['name']
        rec = raw[name]
        n_used = (n_effective if n_effective is not None else rec['n'])
        rec['p_holm'] = (holm.get(name) or {}).get('p_holm')
        rec['m_effective'] = (holm.get(name) or {}).get('m_effective')
        rec['n_effective'] = n_used
        rec['min_seeds_for_confirmation'] = min_n
        rec['coverage_ok'] = bool(coverage_ok)
        rec['param_matched'] = True
        rec['param_match_complete'] = True
        rec['same_init_core_fp'] = True
        rec['expected_sign'] = spec['expected_sign']
        rec['direction_ok'] = _ar_direction_ok(
            list(deltas_by_name.get(name) or []), spec['expected_sign'])
        rec['mechanism_claim_allowed'] = spec['mechanism_claim_allowed']
        rec['n_declared_seeds'] = len(manifest.get('seeds') or [])
        if name in omitted:
            rec['omitted'] = 'rehearsal: comparison deliberately omitted'
        verdict_in[name] = rec
    verdicts = directional_claim_verdicts(  # noqa: F821
        verdict_in, names=[s['name'] for s in specs], alpha=alpha,
        require_complete_coverage=bool(
            (manifest.get('statistics') or {}).get('require_complete_coverage')),
        require_param_match=bool(
            (manifest.get('statistics') or {}).get('param_match_required')),
        default_min_seeds=min_n)
    return {
        'alpha': alpha,
        'min_seeds_for_confirmation': min_n,
        'n_declared_seeds': len(manifest.get('seeds') or []),
        'records': verdict_in,
        'holm': holm,
        'verdicts': verdicts['verdicts'],
        'verdict_summary': {k: verdicts[k] for k in
                            ('supported', 'refuted', 'inconclusive',
                             'significant_nondirectional', 'not_testable',
                             'opposite_nonsignificant',
                             'reverse_significant_unverified', 'counts')},
    }


def _ar_expected(scenario, specs, *, strong=None, marginal=None):
    exp = {}
    for spec in specs:
        name = spec['name']
        sign = spec['expected_sign']
        if scenario in ('null_balanced', 'underpowered'):
            exp[name] = 'inconclusive'
        elif scenario == 'forward_effect':
            exp[name] = ('supported' if sign is not None
                         else 'significant_nondirectional')
        elif scenario == 'reverse_effect':
            exp[name] = ('refuted' if sign is not None
                         else 'significant_nondirectional')
        elif scenario == 'family_correction':
            if name == strong:
                exp[name] = 'supported'
            else:
                exp[name] = 'inconclusive'
        elif scenario == 'omitted_comparison':
            exp[name] = ('not_testable' if name == specs[0]['name']
                         else 'inconclusive')
        else:
            raise ValueError(f'unknown rehearsal scenario {scenario!r}')
    return exp


def _ar_checks(scenario, specs, got, expected, *, family, strong=None,
               marginal=None):
    checks = []
    for spec in specs:
        name = spec['name']
        observed = (got.get('verdicts', {}).get(name) or {}).get('verdict')
        want = expected.get(name)
        checks.append({
            'check': f'verdict::{name}',
            'ok': observed == want,
            'expected': want,
            'observed': observed,
        })
    if scenario == 'null_balanced':
        for spec in specs:
            name = spec['name']
            rec = (got.get('records') or {}).get(name) or {}
            checks.append({
                'check': f'null_p_not_significant::{name}',
                'ok': (rec.get('p_holm') is None
                       or float(rec.get('p_holm')) >= float(family['alpha'])),
                'expected': f'p_holm >= {family["alpha"]}',
                'observed': rec.get('p_holm'),
            })
    if scenario == 'family_correction':
        m = len(specs)
        for spec in specs:
            name = spec['name']
            rec = (got.get('records') or {}).get(name) or {}
            checks.append({
                'check': f'family_slot_not_shrunk::{name}',
                'ok': rec.get('m_effective') == m,
                'expected': m,
                'observed': rec.get('m_effective'),
            })
        if marginal is not None:
            mrec = (got.get('records') or {}).get(marginal) or {}
            srec = (got.get('records') or {}).get(strong) or {}
            checks.append({
                'check': 'marginal_raw_significant_but_holm_null',
                'ok': (mrec.get('p_raw') is not None
                       and float(mrec.get('p_raw')) < float(family['alpha'])
                       and mrec.get('p_holm') is not None
                       and float(mrec.get('p_holm')) >= float(family['alpha'])),
                'expected': 'p_raw < alpha <= p_holm',
                'observed': {'p_raw': mrec.get('p_raw'),
                             'p_holm': mrec.get('p_holm')},
            })
            checks.append({
                'check': 'strong_effect_survives_holm',
                'ok': (srec.get('p_holm') is not None
                       and float(srec.get('p_holm')) < float(family['alpha'])),
                'expected': f'p_holm < {family["alpha"]}',
                'observed': srec.get('p_holm'),
            })
    if scenario == 'underpowered':
        for spec in specs:
            name = spec['name']
            failures = ((got.get('verdicts', {}).get(name) or {})
                        .get('gate_failures') or [])
            checks.append({
                'check': f'underpowered_gate_named::{name}',
                'ok': bool(failures),
                'expected': 'at least one named validity gate',
                'observed': list(failures),
            })
    if scenario == 'omitted_comparison':
        first = specs[0]['name']
        rec = (got.get('verdicts', {}).get(first) or {})
        checks.append({
            'check': 'omitted_is_not_testable',
            'ok': rec.get('verdict') == 'not_testable',
            'expected': 'not_testable',
            'observed': rec.get('verdict'),
        })
    return checks


def _ar_marginal_bad(n_seeds, family_size, alpha):
    """Find a same-sign majority whose raw p is significant but Holm is not.

    Holm gives the largest p multiplier 1, so a family of size m has a
    middle-rank trap only when some raw p satisfies
    ``alpha / (m - 1) <= p < alpha``.  The discrete exact sign-flip p-values
    are searched instead of assuming a fixed number of contrary pairs.
    """
    if n_seeds < 4 or family_size < 3:
        return None
    for n_bad in range(1, n_seeds // 2 + 1):
        n_same = n_seeds - n_bad
        if n_same <= n_bad:
            break
        p = exact_signflip_p([1.0] * n_same + [-1.0] * n_bad).get('p')  # noqa: F821
        if (isinstance(p, (int, float)) and not isinstance(p, bool)
                and float(p) < float(alpha)
                and float(p) * float(family_size - 1) >= float(alpha)):
            return n_bad
    return None


def _ar_scenario_plan(manifest, specs):
    seeds = [int(s) for s in (manifest.get('seeds') or [])]
    n = len(seeds)
    min_n = _ar_int((manifest.get('statistics') or {}).get(
        'min_seeds_for_confirmation'), minimum=2) or n
    under_n = (min_n - 1) if min_n > 2 else 1
    under_n = max(1, min(under_n, n))
    directional = _ar_directional(specs)
    strong = directional[0]['name'] if len(directional) >= 1 else None
    marginal = directional[1]['name'] if len(directional) >= 2 else None
    alpha = float((manifest.get('statistics') or {}).get('alpha', 0.05))
    marginal_bad = _ar_marginal_bad(n, len(specs), alpha)
    plans = [
        {'name': 'null_balanced', 'pattern': 'balanced',
         'description': 'balanced signs; no comparison may be supported',
         'n_used': n, 'coverage_ok': True, 'omitted': (),
         'strong': strong, 'marginal': marginal,
         'marginal_bad': marginal_bad},
        {'name': 'forward_effect', 'pattern': 'forward',
         'description': 'uniform effect in every pre-registered direction',
         'n_used': n, 'coverage_ok': True, 'omitted': (),
         'strong': strong, 'marginal': marginal},
        {'name': 'reverse_effect', 'pattern': 'reverse',
         'description': 'uniform effect opposite to the pre-registered direction',
         'n_used': n, 'coverage_ok': True, 'omitted': (),
         'strong': strong, 'marginal': marginal},
        {'name': 'underpowered', 'pattern': 'forward',
         'description': 'effect present but below the declared n/coverage floor',
         'n_used': under_n, 'coverage_ok': False, 'omitted': (),
         'strong': strong, 'marginal': marginal},
        {'name': 'omitted_comparison', 'pattern': 'zero',
         'description': 'a frozen comparison is omitted; its family slot stays',
         'n_used': n, 'coverage_ok': True,
         'omitted': (specs[0]['name'],), 'strong': strong,
         'marginal': marginal},
    ]
    # Holm gives the largest p in the family multiplier 1.  With m=2 the
    # second-ranked raw-significant p is therefore always Holm-significant;
    # the "raw significant, Holm null" trap needs at least three slots.
    if (len(directional) >= 2 and len(specs) >= 3
            and marginal_bad is not None):
        plans.append({
            'name': 'family_correction', 'pattern': 'marginal',
            'description': ('one strong effect plus one raw-significant/'
                            'Holm-null effect'),
            'n_used': n, 'coverage_ok': True, 'omitted': (),
            'strong': strong, 'marginal': marginal,
            'marginal_bad': marginal_bad})
    return plans


def build_analysis_rehearsal(manifest=None, *, manifest_path=None, created_at=None):
    """Run the known-truth rehearsal over the frozen comparison family.

    ``manifest`` may be a parsed object (as ``confirmation plan`` already
    has); otherwise it is read from ``manifest_path`` (default
    ``confirmation_frozen.json``).  No training, no data loader, no test split.
    """
    if manifest is None:
        manifest_path = manifest_path or 'confirmation_frozen.json'
        manifest, err = _ar_load_manifest(manifest_path)
        if err is not None:
            return {
                'kind': ANALYSIS_REHEARSAL_KIND,
                'schema': ANALYSIS_REHEARSAL_SCHEMA,
                'status': 'not_verifiable',
                'synthetic': True,
                'test_split_read': False,
                'manifest': manifest_path,
                'failures': [err],
                'problems': [err],
                'warnings': [],
                'scenarios': {},
                'checks': [],
                'role': ANALYSIS_REHEARSAL_POLICY['role'],
                'boundary': ANALYSIS_REHEARSAL_POLICY['boundary'],
            }
    if not isinstance(manifest, dict):
        raise TypeError('manifest must be an object')
    if manifest_path is None:
        manifest_path = manifest.get('_path') or 'confirmation_frozen.json'
    stats = manifest.get('statistics')
    if not isinstance(stats, dict):
        raise ValueError('manifest.statistics must be an object')
    specs = _ar_specs(manifest)
    alpha = stats.get('alpha', 0.05)
    try:
        alpha = float(alpha)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'manifest.statistics.alpha is not numeric: {alpha!r}')
    if not (0.0 < alpha < 1.0):
        raise ValueError(f'manifest.statistics.alpha must be in (0,1), got {alpha!r}')
    seeds = manifest.get('seeds')
    if not isinstance(seeds, list) or not seeds:
        raise ValueError('manifest.seeds must be a non-empty list')
    plans = _ar_scenario_plan(manifest, specs)
    scenarios = {}
    failures = []
    all_checks = []
    for plan in plans:
        name = plan['name']
        n_used = plan['n_used']
        coverage_ok = plan['coverage_ok']
        omitted = plan['omitted']
        strong = plan.get('strong')
        marginal = plan.get('marginal')
        deltas = _ar_deltas(
            specs, plan['pattern'], n_seeds=n_used, omitted=omitted,
            strong=strong, marginal=marginal,
            marginal_bad=plan.get('marginal_bad'))
        family = _ar_family(
            manifest, specs, deltas, n_effective=n_used,
            coverage_ok=coverage_ok, omitted=omitted)
        expected = _ar_expected(
            name, specs, strong=strong, marginal=marginal)
        checks = _ar_checks(
            name, specs, family, expected, family=family, strong=strong,
            marginal=marginal)
        ok = all(c['ok'] for c in checks)
        scenarios[name] = {
            'scenario': name,
            'description': plan['description'],
            'n_seeds_used': n_used,
            'coverage_ok': bool(coverage_ok),
            'synthetic': True,
            'ok': ok,
            'expected_verdicts': expected,
            'observed_verdicts': {
                s['name']: (family['verdicts'].get(s['name']) or {}).get('verdict')
                for s in specs},
            'records': family['records'],
            'verdict_summary': family['verdict_summary'],
            'checks': checks,
        }
        all_checks.extend(checks)
        if not ok:
            for c in checks:
                if not c['ok']:
                    failures.append(
                        f'{name}: {c["check"]} expected {c["expected"]!r} '
                        f'got {c["observed"]!r}')
    body = {
        'kind': ANALYSIS_REHEARSAL_KIND,
        'schema': ANALYSIS_REHEARSAL_SCHEMA,
        'status': 'ok' if not failures else 'fail',
        'synthetic': True,
        'test_split_read': False,
        'manifest': manifest_path,
        'manifest_sha256': _ar_manifest_sha(manifest, manifest_path),
        'manifest_canonical_sha256': _ar_manifest_canonical(manifest),
        'manifest_raw_canonical_sha256': (
            _ar_sha_obj(manifest)
            if isinstance(manifest, dict) else None),
        'code_semantics': CODE_SEMANTICS,  # noqa: F821
        'alpha': alpha,
        'n_seeds': len(seeds),
        'min_seeds_for_confirmation': _ar_int(
            stats.get('min_seeds_for_confirmation'), minimum=2),
        'family_size': len(specs),
        'comparisons': [s['name'] for s in specs],
        'n_scenarios': len(scenarios),
        'n_checks': len(all_checks),
        'n_failures': len(failures),
        'failures': failures,
        'checks': all_checks,
        'scenarios': scenarios,
        'policy': dict(ANALYSIS_REHEARSAL_POLICY),
        'role': ANALYSIS_REHEARSAL_POLICY['role'],
        'boundary': ANALYSIS_REHEARSAL_POLICY['boundary'],
        'created_at': created_at or time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    hash_body = {k: v for k, v in body.items()
                 if k not in ('content_sha256', 'created_at', 'revision',
                              'supersedes', 'reason', 'notes')}
    body['content_sha256'] = _ar_sha_obj(hash_body)
    return body


def validate_analysis_rehearsal(rehearsal, *, manifest=None):
    """Recompute the identity and verdict of a rehearsal artifact."""
    out = {'kind': 'analysis_rehearsal_validation', 'status': 'ok',
           'problems': [], 'warnings': [], 'content_sha256': None,
           'recomputed_sha256': None}
    if not isinstance(rehearsal, dict):
        out['status'] = 'fail'
        out['problems'].append(
            f'rehearsal must be an object, got {type(rehearsal).__name__}')
        return out
    if rehearsal.get('kind') != ANALYSIS_REHEARSAL_KIND:
        out['status'] = 'fail'
        out['problems'].append(
            f"rehearsal.kind={rehearsal.get('kind')!r} is not "
            f"{ANALYSIS_REHEARSAL_KIND!r}")
    if rehearsal.get('schema') != ANALYSIS_REHEARSAL_SCHEMA:
        out['status'] = 'fail'
        out['problems'].append(
            f"rehearsal.schema={rehearsal.get('schema')!r} != "
            f'{ANALYSIS_REHEARSAL_SCHEMA}')
    recorded = _ar_sha256_text(rehearsal.get('content_sha256'))
    body = {k: v for k, v in rehearsal.items()
            if k not in ('content_sha256', 'created_at', 'revision',
                         'supersedes', 'reason', 'notes')}
    try:
        recomputed = _ar_sha_obj(body)
    except (TypeError, ValueError, RecursionError) as exc:
        out['status'] = 'fail'
        out['problems'].append(
            f'rehearsal body is not canonical JSON ({type(exc).__name__}: {exc})')
        return out
    out['content_sha256'] = recorded
    out['recomputed_sha256'] = recomputed
    if recorded is None:
        out['status'] = 'fail'
        out['problems'].append('rehearsal.content_sha256 is missing or malformed')
    elif recorded != recomputed:
        out['status'] = 'fail'
        out['problems'].append(
            'rehearsal.content_sha256 does not reproduce the body')
    if rehearsal.get('synthetic') is not True:
        out['status'] = 'fail'
        out['problems'].append(
            'rehearsal.synthetic must be true; synthetic p-values must never '
            'be presented as a measurement')
    if rehearsal.get('test_split_read') is not False:
        out['status'] = 'fail'
        out['problems'].append(
            'rehearsal.test_split_read must be false; this control is pre-test')
    if rehearsal.get('status') != 'ok':
        out['status'] = 'fail'
        out['problems'].append(
            f"rehearsal.status={rehearsal.get('status')!r} is not 'ok'")
    if not any(_ar_sha256_text(rehearsal.get(k)) for k in (
            'manifest_sha256', 'manifest_canonical_sha256',
            'manifest_raw_canonical_sha256')):
        out['status'] = 'fail'
        out['problems'].append(
            'rehearsal has no manifest identity; it cannot be tied to the '
            'registered frozen plan')
    scenarios = rehearsal.get('scenarios')
    if not isinstance(scenarios, dict) or not scenarios:
        out['status'] = 'fail'
        out['problems'].append('rehearsal.scenarios is missing or empty')
        scenarios = {}
    for name, scen in scenarios.items():
        if not isinstance(scen, dict):
            out['status'] = 'fail'
            out['problems'].append(f'scenario {name!r} is not an object')
            continue
        if scen.get('ok') is not True:
            out['status'] = 'fail'
            out['problems'].append(f'scenario {name!r} did not pass')
        checks = scen.get('checks')
        if not isinstance(checks, list) or not checks:
            out['status'] = 'fail'
            out['problems'].append(f'scenario {name!r} has no checks')
            continue
        for chk in checks:
            if not isinstance(chk, dict) or chk.get('ok') is not True:
                out['status'] = 'fail'
                out['problems'].append(
                    f'scenario {name!r} has a failed/malformed check')
    failures = rehearsal.get('failures')
    n_fail = _ar_int(rehearsal.get('n_failures'), minimum=0)
    if n_fail != 0:
        out['status'] = 'fail'
        out['problems'].append(
            f"rehearsal.n_failures={rehearsal.get('n_failures')!r} is not zero")
    if not isinstance(failures, list):
        out['status'] = 'fail'
        out['problems'].append('rehearsal.failures must be a list')
    elif n_fail is not None and len(failures) != n_fail:
        out['status'] = 'fail'
        out['problems'].append(
            'rehearsal.n_failures does not match the failures list')
    n_checks = _ar_int(rehearsal.get('n_checks'), minimum=0)
    # ``checks`` is untrusted JSON.  Only a list has a length; an int/string/
    # mapping is a malformed scenario, already reported in the loop above, and
    # must not abort the validator with TypeError.
    computed_checks = sum(
        len(checks) for checks in
        ((s.get('checks') if isinstance(s, dict) else None)
         for s in scenarios.values())
        if isinstance(checks, list))
    if n_checks is not None and n_checks != computed_checks:
        out['status'] = 'fail'
        out['problems'].append(
            f'n_checks={n_checks} != scenario check count {computed_checks}')
    n_scen = _ar_int(rehearsal.get('n_scenarios'), minimum=0)
    if n_scen is not None and n_scen != len(scenarios):
        out['status'] = 'fail'
        out['problems'].append(
            f'n_scenarios={n_scen} != recorded scenario count {len(scenarios)}')
    family = _ar_int(rehearsal.get('family_size'), minimum=1)
    comps = rehearsal.get('comparisons')
    if family is not None and isinstance(comps, list) and family != len(comps):
        out['status'] = 'fail'
        out['problems'].append(
            f'family_size={family} != len(comparisons)={len(comps)}')
    if isinstance(manifest, dict):
        want_c = _ar_manifest_canonical(manifest)
        got_c = _ar_sha256_text(rehearsal.get('manifest_canonical_sha256'))
        if want_c is not None and got_c is not None:
            if want_c != got_c:
                out['status'] = 'fail'
                out['problems'].append(
                    'rehearsal was run against a different frozen manifest')
        else:
            # Legacy artifacts carry only the file hash.  Compare it when the
            # current manifest still names a readable file.
            want = _ar_manifest_sha(manifest, rehearsal.get('manifest'))
            got = _ar_sha256_text(rehearsal.get('manifest_sha256'))
            if want is not None and got is not None and want != got:
                out['status'] = 'fail'
                out['problems'].append(
                    'rehearsal was run against a different frozen manifest')
    return out


def write_analysis_rehearsal(outdir, rehearsal, *, overwrite=False):
    """Atomically persist a rehearsal artifact; refuse silent replacement."""
    if not isinstance(outdir, str) or not outdir:
        raise ValueError('outdir must be a non-empty path')
    verdict = validate_analysis_rehearsal(rehearsal)
    path = os.path.join(outdir, 'analysis_rehearsal.json')
    out = {'kind': 'write_analysis_rehearsal', 'path': path,
           'written': False, 'unchanged': False, 'conflict': False,
           'status': verdict['status'], 'problems': list(verdict['problems'])}
    if verdict['status'] != 'ok':
        return out
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as fh:
                old = json.load(fh)
        except Exception:
            old = None
        if isinstance(old, dict) and old.get('content_sha256') == rehearsal.get('content_sha256'):
            out['unchanged'] = True
            return out
        if not overwrite:
            out['conflict'] = True
            out['problems'].append(
                'a different analysis_rehearsal.json already exists; pass '
                'overwrite=True only after re-running the rehearsal')
            return out
    os.makedirs(outdir, exist_ok=True)
    atomic_write_json(path, rehearsal, indent=2)  # noqa: F821
    out['written'] = True
    return out


def analysis_rehearsal_summary(rehearsal):
    """Compact status block for preflight/clearance wiring."""
    if not isinstance(rehearsal, dict):
        return {'kind': 'analysis_rehearsal_summary', 'status': 'missing',
                'ok': False, 'n_failures': None}
    verdict = validate_analysis_rehearsal(rehearsal)
    return {
        'kind': 'analysis_rehearsal_summary',
        'status': verdict['status'],
        'ok': verdict['status'] == 'ok',
        'manifest_sha256': rehearsal.get('manifest_sha256'),
        'content_sha256': rehearsal.get('content_sha256'),
        'n_scenarios': rehearsal.get('n_scenarios'),
        'n_failures': rehearsal.get('n_failures'),
        'synthetic': rehearsal.get('synthetic'),
        'test_split_read': rehearsal.get('test_split_read'),
    }
