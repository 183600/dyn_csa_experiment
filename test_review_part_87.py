"""Analysis multiverse: IUT robustness and within-family exact max-T (zero GPU).

These checks exercise a declared specification curve for one paired contrast:
the all-specifications claim is an intersection-union test, the at-least-one
claim is a within-family exact sign-flip max-T, pointwise intervals are never
called simultaneous, and a self-consistent non-robust result is a warning (not
a tamper).  Artifact replay, audit/CLI/Makefile wiring and the frozen-input
boundary are covered too.

No training, no data split, no frozen p/Holm/verdict is changed.
"""
import copy
import inspect
import json
import math
import os
import statistics
import tempfile

from test_support import *  # noqa: F401,F403

REPO = os.path.dirname(os.path.abspath(__file__))


def _row(values):
    return {seed: float(value) for seed, value in enumerate(values)}


def _stable_specs():
    return [
        {'name': 'primary', 'family': 'log_ratio',
         'deltas': _row([1.00, 1.10, 0.90, 1.05, 0.95, 1.00]),
         'features': {'scale': 'log', 'estimator': 'mean'}},
        {'name': 'trimmed', 'family': 'log_ratio',
         'deltas': _row([1.02, 1.08, 0.92, 1.03, 0.97, 1.00]),
         'features': {'scale': 'log', 'estimator': 'trimmed'}},
        {'name': 'winsorized', 'family': 'log_ratio',
         'deltas': _row([0.98, 1.12, 0.88, 1.06, 0.94, 1.00]),
         'features': {'scale': 'log', 'estimator': 'winsorized'}},
    ]


def _assert_close(got, want, tol=1e-12, msg=''):
    assert abs(got - want) <= tol, (msg, got, want)


def test_stable_multiverse_is_an_iut_not_a_majority_vote():
    out = L.analysis_multiverse(_stable_specs(), primary='primary')
    assert out['status'] == 'ok', out['warnings']
    assert out['pattern'] == 'stable'
    assert out['claim']['robust_across_specs'] is True
    assert out['claim']['direction'] == 1
    assert out['summary']['sign_agreement'] == 1.0
    assert out['summary']['all_specs_reject'] is True
    # IUT p is the maximum component p; it is valid but conservative.
    want = max(row['p_exact_signflip']
               for row in out['specs'].values())
    _assert_close(out['p_iut'], want)
    assert out['p_iut'] <= 0.05
    assert out['iut_p_floor'] == 2.0 ** -5
    for name, row in out['specs'].items():
        assert row['p_exact_signflip'] == 2.0 ** -5, (name, row)
        assert row['reject_raw'] is True
        assert row['same_sign_as_primary'] is True
        assert row['ci95'] is not None and row['ci95'][0] > 0.0
        assert row['mean'] > 0.0
    family = out['families']['log_ratio']
    assert family['joint_status'] == 'ok', family
    assert family['joint']['method']
    assert family['joint']['m'] == 3
    assert family['joint']['critical_value'] is not None
    for name, rec in family['joint']['contrasts'].items():
        _assert_close(rec['mean'], out['specs'][name]['mean'])
        assert rec['p_maxt'] >= rec['p_raw_signflip'] - 1e-15
        assert 0.0 < rec['p_maxt'] <= 1.0
        ci = rec['ci_simultaneous']
        assert ci is not None and ci[0] <= rec['mean'] <= ci[1]
    assert out['summary']['scale_comparable'] is True
    json.dumps(out, allow_nan=False)


def test_one_sign_flip_blocks_the_all_specifications_claim():
    specs = _stable_specs()
    specs[2]['deltas'] = _row([-1.0, -1.1, -0.9, -1.05, -0.95, -1.0])
    out = L.analysis_multiverse(specs, primary='primary')
    assert out['pattern'] == 'direction_unstable'
    assert out['status'] == 'warnings'
    assert out['claim']['robust_across_specs'] is False
    assert out['summary']['n_sign_flips'] == 1
    assert out['summary']['n_not_reject'] == 0
    assert 'winsorized' in ' '.join(out['warnings'])
    # The union/max-T family is still computable and is NOT the robustness
    # claim; the two questions must stay separate.
    assert out['families']['log_ratio']['joint_status'] == 'ok'


def test_sign_stable_but_not_every_specification_rejects():
    specs = _stable_specs()
    specs[1]['deltas'] = _row([1.0, 1.0, 1.0, 1.0, 1.0, -0.5])
    out = L.analysis_multiverse(specs, primary='primary')
    assert out['pattern'] == 'precision_unstable'
    assert out['claim']['robust_across_specs'] is False
    spec = out['specs']['trimmed']
    assert spec['p_exact_signflip'] == 0.0625
    assert spec['reject_raw'] is False
    assert spec['same_sign_as_primary'] is True
    assert out['summary']['sign_agreement'] == 1.0
    assert out['summary']['n_reject_raw'] == 2
    assert out['summary']['n_not_reject'] == 1
    assert 'trimmed' in ' '.join(out['warnings'])


def test_resolution_floor_is_reported_not_read_as_no_effect():
    specs = _stable_specs()
    for rec in specs:
        rec['deltas'] = {seed: rec['deltas'][seed] for seed in range(4)}
    out = L.analysis_multiverse(specs, primary='primary')
    assert out['pattern'] == 'resolution_limited'
    assert out['resolution_limited'] is True
    assert out['iut_p_floor'] == 2.0 ** -3
    assert out['p_iut'] == 2.0 ** -3
    assert out['claim']['robust_across_specs'] is False
    assert 'not resolvable' in out['claim']['statement']
    assert out['summary']['n_resolution_limited'] == 3


def test_zero_primary_effect_supplies_no_direction():
    specs = _stable_specs()
    specs[0]['deltas'] = _row([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    out = L.analysis_multiverse(specs, primary='primary')
    assert out['pattern'] == 'zero_effect'
    assert out['primary_sign'] == 0
    assert out['claim']['robust_across_specs'] is False
    assert out['claim']['direction'] == 0


def test_too_few_specifications_are_not_estimable():
    out = L.analysis_multiverse(_stable_specs()[:2], primary='primary',
                                min_specs=3)
    assert out['status'] == 'not_estimable'
    assert out['pattern'] == 'insufficient_specs'
    assert out['claim']['robust_across_specs'] is False
    try:
        L.build_analysis_multiverse(_stable_specs()[:2], primary='primary',
                                    min_specs=3)
    except ValueError:
        pass
    else:
        raise AssertionError('build must refuse a structurally empty family')


def test_family_with_different_seed_sets_has_no_joint_test():
    specs = _stable_specs()
    specs[2]['deltas'] = {seed + 10: value
                          for seed, value in specs[2]['deltas'].items()}
    out = L.analysis_multiverse(specs, primary='primary')
    family = out['families']['log_ratio']
    assert family['joint_status'] == 'not_comparable'
    assert family['joint'] is None
    assert family['joint_reason']
    assert any('not_comparable' in text or 'share one exact seed set' in text
               for text in out['warnings'])
    # The all-specifications IUT does not require a common seed vector.
    assert out['pattern'] in ('stable', 'precision_unstable')


def test_multiple_estimand_families_are_never_pooled():
    specs = _stable_specs()
    specs.append({
        'name': 'pct_change', 'family': 'percent_change',
        'deltas': _row([10.0, 11.0, 9.0, 10.5, 9.5, 10.0]),
        'features': {'scale': 'percent'},
    })
    out = L.analysis_multiverse(specs, primary='primary', min_specs=3)
    assert out['n_families'] == 2
    assert out['summary']['scale_comparable'] is False
    assert out['families']['log_ratio']['effects']['primary'] is not None
    assert out['families']['percent_change']['effects']['primary'] is None
    assert any('multiple estimand' in text for text in out['warnings'])
    # The IUT is still valid: every specification rejects its own null.
    assert out['pattern'] == 'stable'


def test_choice_influence_names_the_feature_that_flips_the_sign():
    specs = _stable_specs()
    specs[1]['features']['estimator'] = 'mean'
    specs[2]['features']['estimator'] = 'mean'
    specs[2]['features']['scale'] = 'percent'
    specs[2]['deltas'] = _row([-10.0, -11.0, -9.0, -10.5, -9.5, -10.0])
    out = L.analysis_multiverse(specs, primary='primary')
    family = out['families']['log_ratio']
    most = family['most_influential_feature']
    assert most is not None
    assert most['feature'] == 'scale'
    assert most['sign_flip'] is True
    assert most['spread'] > 0.0
    influence = family['feature_influence']['scale']
    assert influence['n_levels'] == 2
    assert influence['sign_flip'] is True
    assert influence['relative_to_primary'] is not None
    # The point estimates are on different scales; the spread is descriptive
    # and does not pretend to be a variance decomposition.
    assert 'unbalanced' in influence['note']


def test_estimates_and_intervals_recompute_from_the_deltas():
    specs = _stable_specs()
    out = L.analysis_multiverse(specs, primary='primary')
    values = [specs[0]['deltas'][seed] for seed in range(6)]
    row = out['specs']['primary']
    _assert_close(row['mean'], statistics.mean(values))
    _assert_close(row['sd'], statistics.stdev(values))
    _assert_close(row['sem'], statistics.stdev(values) / math.sqrt(6))
    tc = float(L._t_critical(0.95, 5))
    half = tc * row['sem']
    _assert_close(row['ci95'][0], row['mean'] - half)
    _assert_close(row['ci95'][1], row['mean'] + half)
    exact = L.exact_signflip_p(values)
    assert row['p_exact_signflip'] == exact['p']
    assert row['p_floor'] == exact['p_floor']


def test_extreme_scales_stay_finite_and_scale_invariant():
    huge = _row([1.0e308, 1.1e308, 0.9e308, 1.05e308, 0.95e308, 1.0e308])
    tiny = _row([1.0e-200, 1.1e-200, 0.9e-200, 1.05e-200, 0.95e-200,
                 1.0e-200])
    specs = [
        {'name': 'primary', 'deltas': huge},
        {'name': 'small', 'deltas': tiny},
        {'name': 'middle', 'deltas': _row([1.0, 1.1, 0.9, 1.05, 0.95, 1.0])},
    ]
    out = L.analysis_multiverse(specs, primary='primary')
    assert out['pattern'] == 'stable', out['warnings']
    for row in out['specs'].values():
        assert math.isfinite(row['mean'])
        assert row['sd'] is None or math.isfinite(row['sd'])
        assert row['p_exact_signflip'] == 2.0 ** -5
        if row['ci95'] is not None:
            assert all(math.isfinite(x) for x in row['ci95'])
    # The exact sign-flip p is scale invariant; a 1e308 panel must not
    # overflow the sample SD or turn a small effect into a fake one.
    assert out['specs']['small']['mean'] > 0.0
    json.dumps(out, allow_nan=False)


def test_artifact_round_trip_determinism_and_created_at():
    specs = _stable_specs()
    first = L.build_analysis_multiverse(specs, primary='primary',
                                        created_at='2026-01-01T00:00:00')
    second = L.build_analysis_multiverse(specs, primary='primary',
                                         created_at='2026-01-02T00:00:00')
    assert first['content_sha256'] == second['content_sha256']
    assert first['result'] == second['result']
    verdict = L.validate_analysis_multiverse(first)
    assert verdict['status'] == 'ok', verdict['problems']
    assert verdict['result_status'] == 'ok'
    assert verdict['pattern'] == 'stable'
    round_trip = json.loads(json.dumps(first, allow_nan=False))
    assert L.validate_analysis_multiverse(round_trip)['status'] == 'ok'


def test_tampered_artifacts_fail_closed():
    art = L.build_analysis_multiverse(_stable_specs(), primary='primary')

    def _rehash(obj):
        body = {k: v for k, v in obj.items()
                if k not in ('created_at', 'content_sha256')}
        obj['content_sha256'] = L._sha256_json(body)
        return obj

    cases = []
    bad = copy.deepcopy(art)
    bad['result']['pattern'] = 'direction_unstable'
    _rehash(bad)
    cases.append(bad)
    bad = copy.deepcopy(art)
    bad['result']['specs']['primary']['mean'] += 0.25
    _rehash(bad)
    cases.append(bad)
    bad = copy.deepcopy(art)
    bad['result']['families']['log_ratio']['joint_status'] = 'not_comparable'
    _rehash(bad)
    cases.append(bad)
    bad = copy.deepcopy(art)
    bad['inputs']['specs'][0]['deltas'][0][1] += 0.5
    _rehash(bad)
    cases.append(bad)
    bad = copy.deepcopy(art)
    bad['inputs']['specs'].append(
        {'name': 'extra', 'family': 'log_ratio',
         'deltas': [[0, 1.0], [1, 1.0]], 'features': {}})
    _rehash(bad)
    cases.append(bad)
    bad = copy.deepcopy(art)
    bad['result']['claim']['confirmed'] = True
    _rehash(bad)
    cases.append(bad)
    bad = copy.deepcopy(art)
    del bad['content_sha256']
    cases.append(bad)
    bad = copy.deepcopy(art)
    bad['extra_top'] = 1
    _rehash(bad)
    cases.append(bad)
    for index, bad in enumerate(cases):
        verdict = L.validate_analysis_multiverse(bad)
        assert verdict['status'] == 'fail', (index, verdict['problems'])


def test_writer_noop_conflict_and_overwrite():
    art = L.build_analysis_multiverse(_stable_specs(), primary='primary',
                                      created_at='2026-01-01T00:00:00')
    with tempfile.TemporaryDirectory() as tmp:
        first = L.write_analysis_multiverse(tmp, art)
        assert first['written'] is True and first['status'] == 'ok'
        again = L.write_analysis_multiverse(tmp, art)
        assert again['unchanged'] is True and again['written'] is False
        other = copy.deepcopy(art)
        other['created_at'] = '2026-01-02T00:00:00'
        # Same body, different provenance: unchanged by content identity.
        same = L.write_analysis_multiverse(tmp, other)
        assert same['unchanged'] is True
        changed_specs = copy.deepcopy(_stable_specs())
        changed_specs[0]['deltas'][0] = 1.4
        changed = L.build_analysis_multiverse(
            changed_specs, primary='primary', min_specs=3,
            created_at='2026-01-03T00:00:00')
        out = L.write_analysis_multiverse(tmp, changed)
        assert out['conflict'] is True and out['written'] is False
        out = L.write_analysis_multiverse(tmp, changed, overwrite=True)
        assert out['written'] is True


def test_gate_maps_missing_stable_unstable_and_tampered():
    art = L.build_analysis_multiverse(_stable_specs(), primary='primary')
    with tempfile.TemporaryDirectory() as tmp:
        missing = A.audit_analysis_multiverse(
            os.path.join(tmp, 'none.json'), verbose=False)
        assert missing['status'] == 'not_verifiable_locally'
        path = os.path.join(tmp, 'analysis_multiverse.json')
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(art, handle)
        ok = A.audit_analysis_multiverse(path, verbose=False)
        assert ok['status'] == 'ok', ok['problems']
        assert ok['pattern'] == 'stable'
        assert ok['robust_across_specs'] is True
        assert ok['n_specs'] == 3
        unstable = _stable_specs()
        unstable[2]['deltas'] = _row([-1.0, -1.1, -0.9, -1.05, -0.95, -1.0])
        art2 = L.build_analysis_multiverse(unstable, primary='primary')
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(art2, handle)
        warned = A.audit_analysis_multiverse(path, verbose=False)
        assert warned['status'] == 'warnings'
        assert warned['pattern'] == 'direction_unstable'
        assert warned['robust_across_specs'] is False
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump({'kind': 'analysis_multiverse_artifact'}, handle)
        bad = A.audit_analysis_multiverse(path, verbose=False)
        assert bad['status'] == 'fail'


def test_gate_reports_structurally_small_artifact_as_not_verifiable():
    out = L.analysis_multiverse(_stable_specs()[:2], primary='primary',
                                min_specs=3)
    declared = L._am_input_specs(L._am_normalise_specs(_stable_specs()[:2]))
    art = {
        'kind': L.AM_ARTIFACT_KIND,
        'schema': L.AM_SCHEMA,
        'created_at': '2026-01-01T00:00:00',
        'inputs': {'specs': declared, 'primary': 'primary', 'alpha': 0.05,
                   'min_specs': 3, 'min_seeds': 2},
        'result': out,
    }
    art['content_sha256'] = L._sha256_json(
        {k: v for k, v in art.items()
         if k not in ('created_at', 'content_sha256')})
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'analysis_multiverse.json')
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(art, handle)
        audited = A.audit_analysis_multiverse(path, verbose=False)
    assert audited['status'] == 'not_verifiable_locally'
    assert audited['result_status'] == 'not_estimable'
    assert L.validate_analysis_multiverse(art)['status'] == 'ok'


def test_declaration_errors_are_fail_closed():
    specs = _stable_specs()
    bad_inputs = [
        [None],
        [{'deltas': _row([1.0] * 6)}],
        [{'name': 'a', 'deltas': _row([1.0])}],
        [{'name': 'a', 'deltas': _row([1.0] * 6)},
         {'name': 'a', 'deltas': _row([1.0] * 6)}],
        [{'name': 'a', 'deltas': _row([1.0] * 6), 'unknown': 1}],
        [{'name': 'a', 'deltas': _row([1.0] * 6),
          'features': {'x': [1, 2]}}],
        [{'name': 'a', 'deltas': {seed: 1.0 for seed in range(6)},
          'features': {'x': None}}],
    ]
    for bad in bad_inputs:
        try:
            L.analysis_multiverse(bad, primary='a')
        except ValueError:
            pass
        else:
            raise AssertionError(f'malformed declaration accepted: {bad!r}')
    for key, value in (('alpha', 0.0), ('alpha', 1.0), ('alpha', True),
                       ('min_specs', 0), ('min_seeds', 1), ('min_seeds', 7)):
        kwargs = {'primary': 'primary', key: value}
        try:
            L.analysis_multiverse(specs, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f'{key}={value!r} accepted')
    for row in (-1, True, 'x', None):
        bad = copy.deepcopy(specs)
        bad[0]['deltas'] = {row: 1.0}
        try:
            L.analysis_multiverse(bad, primary='primary')
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f'seed {row!r} accepted')
    for delta in (True, '1.0', None, float('nan'), float('inf')):
        bad = copy.deepcopy(specs)
        bad[0]['deltas'][0] = delta
        try:
            L.analysis_multiverse(bad, primary='primary')
        except ValueError:
            pass
        else:
            raise AssertionError(f'delta {delta!r} accepted')
    for bad in (
            [{'name': 'verified', 'deltas': _row([1.0] * 6)}],
            [{'name': 'a', 'family': 'holm', 'deltas': _row([1.0] * 6)}],
            [{'name': 'a', 'deltas': _row([1.0] * 6),
              'features': {'confirmed': 'x'}}]):
        try:
            L.analysis_multiverse(bad, primary=bad[0]['name'])
        except ValueError:
            pass
        else:
            raise AssertionError(f'frozen field name accepted: {bad!r}')
    try:
        L.analysis_multiverse(specs, primary='not-declared')
    except ValueError:
        pass
    else:
        raise AssertionError('an undeclared primary was accepted')


def test_markdown_and_wiring():
    out = L.analysis_multiverse(_stable_specs(), primary='primary')
    text = L.analysis_multiverse_markdown(out)
    assert 'primary' in text and 'log_ratio' in text
    assert 'stable' in text and 'IUT' in text
    assert 'analysis_multiverse' in inspect.getsource(A.run_report)
    assert 'analysis_multiverse' in inspect.getsource(A._run_cli)
    assert 'analysis_multiverse' in inspect.getsource(
        A.audit_analysis_multiverse)
    with open(os.path.join(REPO, 'Makefile'), encoding='utf-8') as handle:
        make = handle.read()
    assert 'multiverse:' in make
    assert 'audit_gates.py analysis_multiverse' in make
    with open(os.path.join(REPO, 'explib', '__init__.py'),
              encoding='utf-8') as handle:
        assert 'part_80_analysis_multiverse.py' in handle.read()
    with open(os.path.join(REPO, '_audit_gates', '__init__.py'),
              encoding='utf-8') as handle:
        assert 'part_46_audit_analysis_multiverse.py' in handle.read()
    for name in ('analysis_multiverse', 'build_analysis_multiverse',
                 'validate_analysis_multiverse', 'write_analysis_multiverse',
                 'analysis_multiverse_markdown'):
        assert callable(getattr(L, name)), name


def test_no_readme_or_github_mutation_in_this_round():
    for name in ('README.md', 'README_zh.md', '.github'):
        assert os.path.exists(os.path.join(REPO, name)), name
