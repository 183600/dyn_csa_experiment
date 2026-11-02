"""Joint inference for Shapley component rankings (zero GPU).

§136 gives exact Shapley values on a complete ablation lattice but reports a
separate interval per component.  These checks exercise the next object: a
family-corrected component test, pairwise ordering with rank bounds, the
scale sensitivity of Shapley (log PPL vs raw PPL), artifact replay and the
audit/CLI wiring.

No training, no data split, no frozen p/Holm/verdict is changed.
"""
import copy
import inspect
import itertools
import json
import math
import os
import tempfile

from test_support import *  # noqa: F401,F403

REPO = os.path.dirname(os.path.abspath(__file__))


def _cells(log_values, seeds, *, jitter=0.002):
    """Positive PPL cells ``exp(log value + deterministic seed jitter)``."""
    ordered = sorted(log_values.items(),
                     key=lambda kv: (len(kv[0]), tuple(sorted(kv[0]))))
    out = {}
    for idx, (subset, value) in enumerate(ordered):
        out[frozenset(subset)] = {
            seed: math.exp(value + jitter * math.sin((seed + 1) * (idx + 1)))
            for seed in seeds}
    return out


def _additive(comps, effects, *, base=3.0):
    vals = {}
    for size in range(len(comps) + 1):
        for combo in itertools.combinations(comps, size):
            vals[frozenset(combo)] = base + sum(effects[c] for c in combo)
    return vals


def _scale_flip_values():
    # Chosen so that the log-PPL Shapley order is a < b < c while the raw-PPL
    # order is a < c < b.  Shapley is not invariant under monotone transforms.
    return {
        frozenset(): 5.912069,
        frozenset({'a'}): 3.065977,
        frozenset({'b'}): 5.695242,
        frozenset({'c'}): 5.611696,
        frozenset({'a', 'b'}): 0.551256,
        frozenset({'a', 'c'}): 0.844866,
        frozenset({'b', 'c'}): 3.669537,
        frozenset({'a', 'b', 'c'}): -0.407747,
    }


def _assert_close(got, want, tol=1e-9, msg=''):
    assert abs(got - want) <= tol, (msg, got, want)


def test_scale_flip_is_reported_not_hidden():
    comps = ['a', 'b', 'c']
    cells = _cells(_scale_flip_values(), range(10), jitter=0.001)
    out = L.ablation_ranking(cells, components=comps)
    assert out['status'] == 'warnings', out['problems']
    sens = out['scale_sensitivity']
    assert sens['status'] == 'scale_dependent', sens
    assert sens['rank_log']['a'] == 1.0
    assert sens['rank_log']['b'] == 2.0
    assert sens['rank_log']['c'] == 3.0
    assert sens['rank_ppl']['b'] == 3.0
    assert sens['rank_ppl']['c'] == 2.0
    flipped = {tuple(sorted(pair)) for pair in sens['pair_order_flips']}
    assert ('b', 'c') in flipped, sens['pair_order_flips']
    # The primary (log) order is still a total order; the warning is about
    # scale dependence, not about an unresolved primary ranking.
    assert out['ranking']['ranking_consistency'] == 'total_order'
    assert out['full_ranking_licensed'] is True
    json.dumps(out, allow_nan=False)


def test_additive_lattice_gives_joint_intervals_and_total_order():
    comps = ['a', 'b', 'c']
    effects = {'a': -0.60, 'b': -0.30, 'c': -0.10}
    cells = _cells(_additive(comps, effects), range(10))
    out = L.ablation_ranking(cells, components=comps, scales=('log_ppl',))
    assert out['status'] == 'ok', (out['problems'], out['warnings'])
    assert out['joint']['method'] == 'exact_signflip_maxt'
    assert out['joint']['fallback'] is False
    assert out['n_seeds'] == 10
    for comp, want in effects.items():
        _assert_close(out['component_stats']['log_ppl'][comp]['mean'], want,
                      tol=1e-3)
        rec = out['component_stats']['log_ppl'][comp]
        assert rec['p_maxt_component'] is not None
        assert rec['p_maxt_component'] <= 0.05, rec
        ci = rec['ci_simultaneous']
        assert ci is not None and ci[0] <= rec['mean'] <= ci[1], rec
        assert rec['point_rank'] == (1.0 if comp == 'a' else
                                     2.0 if comp == 'b' else 3.0)
    assert out['ranking']['ranking_consistency'] == 'total_order'
    assert out['ranking']['top1'] == ['a']
    assert out['ranking']['top1_licensed'] is True
    assert out['ranking']['full_ranking_licensed'] is True
    assert out['ranking']['n_unresolved_pairs'] == 0
    assert out['scale_sensitivity'] is None
    json.dumps(out, allow_nan=False)


def test_constant_vectors_take_the_named_holm_fallback():
    comps = ['a', 'b', 'c']
    effects = {'a': -0.60, 'b': -0.30, 'c': -0.30}
    cells = _cells(_additive(comps, effects), range(10), jitter=0.0)
    out = L.ablation_ranking(cells, components=comps, scales=('log_ppl',))
    assert out['status'] == 'warnings'
    assert out['joint']['method'] == 'holm_fallback'
    assert out['joint']['fallback'] is True
    assert out['joint']['fallback_reason']
    assert 'Holm' in (out['joint']['ci_note'] or '')
    # The raw p and the Holm companion are finite even when max-T is
    # degenerate; the fallback does not turn the family into a pass.
    assert out['component_stats']['log_ppl']['a']['p_raw_signflip'] is not None
    assert out['component_stats']['log_ppl']['a']['p_holm_component'] is not None


def test_unresolved_pair_blocks_full_ranking():
    comps = ['a', 'b', 'c']
    effects = {'a': -0.60, 'b': -0.30, 'c': -0.30}
    cells = _cells(_additive(comps, effects), range(10), jitter=0.0)
    out = L.ablation_ranking(cells, components=comps, scales=('log_ppl',))
    rank = out['ranking']
    assert rank['ranking_consistency'] == 'partial_order', rank
    assert rank['full_ranking_licensed'] is False
    assert rank['unique_rank']['a'] is True
    assert rank['unique_rank']['b'] is False
    assert rank['unique_rank']['c'] is False
    assert rank['rank_lower']['b'] == 2 and rank['rank_upper']['b'] == 3
    assert set(rank['unresolved_pairs']) == {'b|c'}
    unresolved = [row for row in rank['pairs'] if row['name'] == 'b|c'][0]
    assert unresolved['separated'] is False
    assert unresolved['better'] is None
    assert unresolved['point_better'] is None
    assert rank['top1'] == ['a'] and rank['top1_licensed'] is True


def test_resolution_floor_blocks_every_separation():
    comps = ['a', 'b']
    effects = {'a': -0.60, 'b': -0.30}
    cells = _cells(_additive(comps, effects), range(4))
    out = L.ablation_ranking(cells, components=comps, scales=('log_ppl',),
                             alpha=0.05)
    assert out['status'] == 'warnings'
    assert out['resolution_limited'] is True
    _assert_close(out['p_floor'], 2.0 / (1 << 4))
    assert out['ranking']['ranking_consistency'] == 'unresolved'
    assert out['ranking']['top1_licensed'] is False
    assert out['ranking']['full_ranking_licensed'] is False
    assert out['joint']['critical_value'] is None
    assert all(not row['separated'] for row in out['ranking']['pairs'])


def test_incomplete_lattice_is_not_estimable_and_gate_is_not_a_pass():
    comps = ['a', 'b', 'c']
    effects = {'a': -0.60, 'b': -0.30, 'c': -0.10}
    cells = _cells(_additive(comps, effects), range(8))
    incomplete = {k: v for k, v in cells.items()
                  if k != frozenset({'b', 'c'})}
    out = L.ablation_ranking(incomplete, components=comps)
    assert out['status'] == 'not_estimable'
    assert out['ranking'] is None
    assert any('missing' in p.lower() for p in out['problems']), out['problems']
    art = L.build_ablation_ranking(incomplete, components=comps)
    verdict = L.validate_ablation_ranking(art)
    assert verdict['status'] == 'ok', verdict['problems']
    assert verdict['result_status'] == 'not_estimable'
    path = os.path.join(tempfile.mkdtemp(), 'ablation_ranking.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(art, fh, allow_nan=False)
    gate = A.audit_ablation_ranking(path)
    assert gate['status'] == 'not_verifiable_locally', gate
    assert gate['result_status'] == 'not_estimable'


def test_one_component_is_not_applicable():
    cells = _cells({
        frozenset(): 3.0,
        frozenset({'a'}): 2.5,
    }, range(6))
    out = L.ablation_ranking(cells, components=['a'])
    assert out['status'] == 'not_applicable'
    assert out['ranking'] is None
    art = L.build_ablation_ranking(cells, components=['a'])
    assert L.validate_ablation_ranking(art)['result_status'] == 'not_applicable'


def test_invalid_inputs_fail_closed():
    valid = _cells(_additive(['a', 'b'], {'a': -0.5, 'b': -0.2}), range(6))
    bad_cases = [
        lambda: L.ablation_ranking(None),
        lambda: L.ablation_ranking(valid, alpha=0.0),
        lambda: L.ablation_ranking(valid, alpha=1.0),
        lambda: L.ablation_ranking(valid, alpha=True),
        lambda: L.ablation_ranking(valid, alpha='0.05'),
        lambda: L.ablation_ranking(valid, min_n=-1),
        lambda: L.ablation_ranking(valid, min_n=2.5),
        lambda: L.ablation_ranking(valid, primary_scale='raw'),
        lambda: L.ablation_ranking(valid, scales=()),
        lambda: L.ablation_ranking(valid, primary_scale='ppl',
                                   scales=('log_ppl',)),
        lambda: L.ablation_ranking(valid, max_components=0),
        lambda: L.ablation_ranking(valid, tolerance=-1.0),
        lambda: L.ablation_ranking(
            {frozenset(): {0: 0.0, 1: 1.0},
             frozenset({'a'}): {0: 1.0, 1: 1.0}}, components=['a']),
        lambda: L.ablation_ranking(
            {frozenset(): {0: 1.0}, frozenset({'z'}): {0: 1.0}},
            components=['a']),
        lambda: L.ablation_ranking(
            {'a': {0: 1.0}, frozenset({'a'}): {0: 1.0}}, components=['a', 'b']),
    ]
    for fn in bad_cases:
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError('malformed ranking input must fail closed')
    # An empty declaration is a degenerate lattice, not a type error: it must
    # be reported as not_estimable rather than raising or passing.
    assert L.ablation_ranking({})['status'] == 'not_estimable'


def test_artifact_round_trip_and_tamper_rejection():
    comps = ['a', 'b']
    cells = _cells(_additive(comps, {'a': -0.5, 'b': -0.2}), range(8))
    art = L.build_ablation_ranking(cells, components=comps,
                                   scales=('log_ppl',))
    assert L.validate_ablation_ranking(art)['status'] == 'ok'
    roundtrip = json.loads(json.dumps(art, allow_nan=False))
    assert L.validate_ablation_ranking(roundtrip)['status'] == 'ok'
    # created_at is provenance, not scientific content: changing it must not
    # invalidate a content-addressed artifact.
    moved = copy.deepcopy(art)
    moved['created_at'] = '2030-01-01T00:00:00'
    assert L.validate_ablation_ranking(moved)['status'] == 'ok'
    bad = copy.deepcopy(art)
    bad['result']['component_stats']['log_ppl']['a']['mean'] += 0.01
    assert L.validate_ablation_ranking(bad)['status'] == 'fail'
    bad = copy.deepcopy(art)
    bad['result']['ranking']['top1_licensed'] = False
    assert L.validate_ablation_ranking(bad)['status'] == 'fail'
    bad = copy.deepcopy(art)
    bad['content_sha256'] = '0' * 64
    assert L.validate_ablation_ranking(bad)['status'] == 'fail'
    bad = copy.deepcopy(art)
    bad['result']['confirmed'] = True
    assert L.validate_ablation_ranking(bad)['status'] == 'fail'
    bad = copy.deepcopy(art)
    bad['inputs']['cells'] = bad['inputs']['cells'][:-1]
    assert L.validate_ablation_ranking(bad)['status'] == 'fail'
    bad = copy.deepcopy(art)
    bad['result']['extra_field'] = 1
    assert L.validate_ablation_ranking(bad)['status'] == 'fail'


def test_writer_noop_conflict_and_overwrite():
    comps = ['a', 'b']
    cells = _cells(_additive(comps, {'a': -0.5, 'b': -0.2}), range(8))
    art = L.build_ablation_ranking(cells, components=comps,
                                   scales=('log_ppl',))
    outdir = tempfile.mkdtemp()
    first = L.write_ablation_ranking(outdir, art)
    assert first['written'] is True and first['status'] == 'ok'
    again = L.write_ablation_ranking(outdir, art)
    assert again['unchanged'] is True and again['written'] is False
    changed_cells = copy.deepcopy(cells)
    for per in changed_cells.values():
        for seed in per:
            per[seed] *= 1.01
    other = L.build_ablation_ranking(changed_cells, components=comps,
                                    scales=('log_ppl',))
    conflict = L.write_ablation_ranking(outdir, other)
    assert conflict['conflict'] is True
    replaced = L.write_ablation_ranking(outdir, other, overwrite=True)
    assert replaced['written'] is True


def test_gate_valid_and_tampered_artifacts():
    comps = ['a', 'b']
    cells = _cells(_additive(comps, {'a': -0.5, 'b': -0.2}), range(8))
    art = L.build_ablation_ranking(cells, components=comps,
                                   scales=('log_ppl',))
    path = os.path.join(tempfile.mkdtemp(), 'ablation_ranking.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(art, fh, allow_nan=False)
    gate = A.audit_ablation_ranking(path)
    assert gate['status'] == 'ok', gate
    assert gate['result_status'] == 'ok'
    assert gate['n_components'] == 2
    assert gate['top1_licensed'] is True
    bad = copy.deepcopy(art)
    bad['result']['ranking']['rank_lower']['a'] = 2
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(bad, fh, allow_nan=False)
    assert A.audit_ablation_ranking(path)['status'] == 'fail'
    assert A.audit_ablation_ranking(os.path.join(
        tempfile.mkdtemp(), 'missing.json'))['status'] == 'not_verifiable_locally'
    assert A.audit_ablation_ranking_artifact(path)['status'] == 'fail'


def test_primary_scale_switch_changes_the_ranking_family():
    comps = ['a', 'b', 'c']
    cells = _cells(_scale_flip_values(), range(10), jitter=0.001)
    log_first = L.ablation_ranking(cells, components=comps,
                                   primary_scale='log_ppl')
    raw_first = L.ablation_ranking(cells, components=comps,
                                   primary_scale='ppl')
    assert log_first['joint']['method'] == 'exact_signflip_maxt'
    assert raw_first['joint']['method'] == 'exact_signflip_maxt'
    assert log_first['component_stats']['ppl']['a'][
        'p_maxt_component'] is None
    assert raw_first['component_stats']['log_ppl']['a'][
        'p_maxt_component'] is None
    # Both scales are reported in both runs; only the primary scale owns the
    # joint family and the licensed ranking.
    assert set(log_first['component_stats']) == {'log_ppl', 'ppl'}
    assert log_first['scale_sensitivity']['status'] == 'scale_dependent'
    assert raw_first['scale_sensitivity']['status'] == 'scale_dependent'
    raw_pair = [p for p in raw_first['ranking']['pairs']
                if p['name'] == 'b|c'][0]
    assert raw_pair['better'] == 'c', raw_pair
    log_pair = [p for p in log_first['ranking']['pairs']
                if p['name'] == 'b|c'][0]
    assert log_pair['better'] == 'b', log_pair
    json.dumps(raw_first, allow_nan=False)


def test_from_lattice_matches_direct_analysis():
    comps = ['a', 'b', 'c']
    effects = {'a': -0.60, 'b': -0.30, 'c': -0.10}
    log_values = _additive(comps, effects)
    cells = _cells(log_values, range(8))
    direct = L.ablation_ranking(cells, components=comps,
                                scales=('log_ppl',))
    lattice = L.build_ablation_lattice(cells, components=comps, scale='ppl')
    via_ppl = L.ablation_ranking_from_lattice(
        lattice, primary_scale='log_ppl', scales=('log_ppl',))
    assert via_ppl['status'] == direct['status']
    for comp in comps:
        _assert_close(via_ppl['component_stats']['log_ppl'][comp]['mean'],
                      direct['component_stats']['log_ppl'][comp]['mean'])
    assert via_ppl['ranking']['top1'] == direct['ranking']['top1']
    # The log-PPL lattice form stores log values; the adapter exponentates
    # before running the same analysis rather than reinterpreting the units.
    log_cells = {subset: {seed: math.log(value) for seed, value in per.items()}
                 for subset, per in cells.items()}
    log_lattice = L.build_ablation_lattice(log_cells, components=comps,
                                           scale='log_ppl')
    via_log = L.ablation_ranking_from_lattice(
        log_lattice, primary_scale='log_ppl', scales=('log_ppl',))
    for comp in comps:
        _assert_close(via_log['component_stats']['log_ppl'][comp]['mean'],
                      direct['component_stats']['log_ppl'][comp]['mean'])
    bad = copy.deepcopy(lattice)
    bad['content_sha256'] = '0' * 64
    try:
        L.ablation_ranking_from_lattice(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('a tampered lattice must not be used as input')


def test_report_cli_makefile_and_part_registration():
    assert 'ablation_ranking' in inspect.getsource(A.run_report)
    assert 'ablation_ranking' in inspect.getsource(A._run_cli)
    assert 'ablation_ranking' in inspect.getsource(A.audit_ablation_ranking)
    with open(os.path.join(REPO, 'Makefile'), encoding='utf-8') as fh:
        make = fh.read()
    assert 'ablation-rank:' in make
    assert 'audit_gates.py ablation_ranking' in make
    with open(os.path.join(REPO, 'explib', '__init__.py'),
              encoding='utf-8') as fh:
        assert 'part_79_ablation_ranking.py' in fh.read()
    with open(os.path.join(REPO, '_audit_gates', '__init__.py'),
              encoding='utf-8') as fh:
        assert 'part_45_audit_ablation_ranking.py' in fh.read()
    assert callable(L.ablation_ranking)
    assert callable(L.build_ablation_ranking)
    assert callable(L.validate_ablation_ranking)


def test_no_readme_or_github_mutation_in_this_round():
    for name in ('README.md', 'README_zh.md', '.github'):
        assert os.path.exists(os.path.join(REPO, name)), name
