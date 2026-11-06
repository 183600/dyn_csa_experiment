"""Ablation-lattice attribution and design licensing (zero GPU).

The study already reports leave-one-out contrasts and 2x2 interactions.  These
checks exercise the next object: an exact Shapley/interaction decomposition on
a *complete* subset lattice, the refusal to compute it from a star or an
incomplete/unequal-seed panel, the design audit that distinguishes complete
2x2 scopes from a global lattice, and the artifact replay/audit/CLI wiring.

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


def _ppl_cells(log_values, seeds=(0, 1, 2)):
    return {subset: {seed: math.exp(value) for seed in seeds}
            for subset, value in log_values.items()}


def _k3_additive():
    comps = ['a', 'b', 'c']
    vals = {}
    for size in range(4):
        for combo in itertools.combinations(comps, size):
            subset = frozenset(combo)
            # `a` is inert; `b` and `c` have identical, additive effects.
            vals[subset] = -0.1 * (('b' in subset) + ('c' in subset))
    return comps, vals


def _assert_close(got, want, tol=1e-12, msg=''):
    assert abs(got - want) <= tol, (msg, got, want)


def test_two_component_additive_lattice_is_exact():
    va, vb = -math.log(1.1), -math.log(1.2)
    cells = _ppl_cells({
        frozenset(): 0.0,
        frozenset({'a'}): va,
        frozenset({'b'}): vb,
        frozenset({'a', 'b'}): va + vb,
    })
    out = L.shapley_lattice_attribution(cells, components=['a', 'b'], min_n=3)
    assert out['status'] == 'ok', out['problems']
    assert out['complete'] is True
    assert out['missing_subsets'] == []
    _assert_close(out['shapley']['a']['mean'], va)
    _assert_close(out['shapley']['b']['mean'], vb)
    _assert_close(out['total_effect']['mean'], va + vb)
    _assert_close(out['interactions'][0]['at_empty']['mean'], 0.0)
    assert out['axioms']['efficiency']['pass'] is True
    assert out['axioms']['efficiency']['gap'] == 0.0
    assert out['axioms']['dummy']['a']['pass'] is True
    assert out['rank_agreement']['spearman'] == 1.0


def test_redundant_components_split_the_joint_effect():
    cells = _ppl_cells({
        frozenset(): 0.0,
        frozenset({'a'}): -1.0,
        frozenset({'b'}): -1.0,
        frozenset({'a', 'b'}): -1.0,
    })
    out = L.shapley_lattice_attribution(cells, components=['a', 'b'])
    assert out['status'] == 'ok'
    _assert_close(out['shapley']['a']['mean'], -0.5)
    _assert_close(out['shapley']['b']['mean'], -0.5)
    _assert_close(out['loo']['a']['mean'], 0.0)
    _assert_close(out['loo']['b']['mean'], 0.0)
    _assert_close(out['interactions'][0]['at_empty']['mean'], 1.0)
    _assert_close(out['total_effect']['mean'], -1.0)
    assert out['marginals']['a']['range'] == 1.0
    assert out['marginals']['b']['range'] == 1.0


def test_synergy_is_negative_and_efficiency_holds():
    cells = _ppl_cells({
        frozenset(): 0.0,
        frozenset({'a'}): -1.0,
        frozenset({'b'}): -1.0,
        frozenset({'a', 'b'}): -3.0,
    })
    out = L.shapley_lattice_attribution(cells, components=['a', 'b'])
    _assert_close(out['interactions'][0]['at_empty']['mean'], -1.0)
    _assert_close(out['shapley']['a']['mean'], -1.5)
    _assert_close(out['shapley']['b']['mean'], -1.5)
    _assert_close(out['total_effect']['mean'], -3.0)
    assert out['axioms']['efficiency']['pass'] is True


def test_three_component_dummy_and_symmetry_axioms():
    comps, vals = _k3_additive()
    # b and c are symmetric; a is a dummy (its marginal is zero everywhere).
    cells = _ppl_cells(vals)
    out = L.shapley_lattice_attribution(cells, components=comps)
    assert out['status'] == 'ok'
    _assert_close(out['shapley']['a']['mean'], 0.0)
    _assert_close(out['shapley']['b']['mean'], out['shapley']['c']['mean'])
    assert out['axioms']['dummy']['a']['all_marginals_zero'] is True
    assert out['axioms']['dummy']['a']['pass'] is True
    assert ['b', 'c'] in out['axioms']['symmetry']['symmetric_pairs']
    assert out['axioms']['symmetry']['violations'] == []
    assert out['axioms']['efficiency']['pass'] is True


def test_context_dependent_marginal_is_reported_not_hidden():
    cells = _ppl_cells({
        frozenset(): 0.0,
        frozenset({'a'}): -1.0,
        frozenset({'b'}): -1.0,
        frozenset({'a', 'b'}): 2.0,
    })
    out = L.shapley_lattice_attribution(cells, components=['a', 'b'])
    assert out['status'] == 'warnings', out['status']
    assert out['marginals']['a']['context_dependent'] is True
    assert out['marginals']['b']['context_dependent'] is True
    _assert_close(out['marginals']['a']['range'], 4.0)
    _assert_close(out['loo']['a']['mean'], 3.0)
    _assert_close(out['shapley']['a']['mean'], 1.0)
    _assert_close(out['interactions'][0]['at_empty']['mean'], 4.0)
    assert out['axioms']['efficiency']['pass'] is True
    assert any('changes sign' in w for w in out['warnings'])


def test_missing_cell_is_refused_without_imputation():
    cells = _ppl_cells({
        frozenset(): 0.0,
        frozenset({'a'}): -1.0,
        frozenset({'b'}): -1.0,
    })
    out = L.shapley_lattice_attribution(cells, components=['a', 'b'])
    assert out['status'] == 'not_estimable'
    assert out['complete'] is False
    assert out['missing_subsets'] == ['a+b']
    assert out['shapley'] == {}
    assert out['loo'] == {}
    assert out['interactions'] == []
    assert any('incomplete' in p for p in out['problems'])
    art = L.build_ablation_lattice(cells, components=['a', 'b'])
    ver = L.validate_ablation_lattice(art)
    assert ver['status'] == 'ok'
    assert ver['replayed_status'] == 'not_estimable'


def test_seed_coverage_is_complete_case_intersection():
    cells = _ppl_cells({
        frozenset(): 0.0,
        frozenset({'a'}): -1.0,
        frozenset({'b'}): -1.0,
        frozenset({'a', 'b'}): -2.0,
    })
    # One cell is missing seed 2.  With min_n=3 the lattice cannot answer;
    # with min_n=2 it answers on the 2-seed intersection and says so.
    cells[frozenset({'b'})].pop(2)
    strict = L.shapley_lattice_attribution(
        cells, components=['a', 'b'], min_n=3)
    assert strict['status'] == 'not_estimable'
    assert strict['seed_coverage']['n_common_seeds'] == 2
    assert strict['seed_coverage']['coverage_ok'] is False
    assert strict['seed_coverage']['missing_by_cell'] == {'b': [2]}
    loose = L.shapley_lattice_attribution(
        cells, components=['a', 'b'], min_n=2)
    assert loose['status'] == 'warnings'
    assert loose['shapley']['a']['n'] == 2
    assert loose['shapley']['a']['seeds'] == [0, 1]
    assert any('complete-case' in w for w in loose['warnings'])
    # Disjoint seed sets have no Shapley value at all.
    disjoint = _ppl_cells({frozenset(): 0.0, frozenset({'a'}): -1.0,
                           frozenset({'b'}): -1.0,
                           frozenset({'a', 'b'}): -2.0},
                          seeds=(0, 1, 2))
    for subset in disjoint:
        disjoint[subset] = {s: v for s, v in disjoint[subset].items()
                            if s == (0 if subset == frozenset() else 1)}
    none = L.shapley_lattice_attribution(
        disjoint, components=['a', 'b'], min_n=1)
    assert none['status'] == 'not_estimable'
    assert any('no seed' in p for p in none['problems'])


def test_malformed_lattice_inputs_fail_closed():
    good = _ppl_cells({
        frozenset(): 0.0, frozenset({'a'}): -1.0,
        frozenset({'b'}): -1.0, frozenset({'a', 'b'}): -2.0,
    })
    bad = copy.deepcopy(good)
    bad[frozenset({'a'})][0] = 0.0
    for bad_cells in (bad,):
        try:
            L.shapley_lattice_attribution(bad_cells, components=['a', 'b'])
        except ValueError:
            pass
        else:
            raise AssertionError('non-positive PPL must be rejected')
    dup = copy.deepcopy(good)
    dup[frozenset({'a'})]['0'] = 99.0
    try:
        L.shapley_lattice_attribution(dup, components=['a', 'b'])
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate seed after normalisation must fail')
    outside = copy.deepcopy(good)
    outside[frozenset({'a', 'z'})] = {0: 1.0, 1: 1.0, 2: 1.0}
    try:
        L.shapley_lattice_attribution(outside, components=['a', 'b'])
    except ValueError:
        pass
    else:
        raise AssertionError('a cell outside the declared components must fail')
    for kwargs in ({'alpha': 0.0}, {'alpha': 1.0}, {'min_n': -1},
                   {'scale': 'raw'}, {'max_components': 0}):
        try:
            L.shapley_lattice_attribution(good, components=['a', 'b'], **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f'{kwargs} must be rejected')
    too_many = L.shapley_lattice_attribution(
        good, components=['a', 'b'], max_components=1)
    assert too_many['status'] == 'not_estimable'
    assert any('max_components' in p for p in too_many['problems'])


def test_numpy_boolean_scalars_are_flags_not_ppls_or_counts():
    """``np.bool_`` is not a Python bool subclass but is still a flag.

    Accepting it as a measurement turns a boolean cell into PPL=1.0, and as
    ``min_n``/``max_components`` into an integer gate of 1.  The scalar
    boundary must fail closed just like the other numeric readers.
    """
    cells = _ppl_cells({
        frozenset(): 0.0, frozenset({'a'}): -1.0,
        frozenset({'b'}): -1.0, frozenset({'a', 'b'}): -2.0,
    })
    for flag in (np.bool_(True), np.bool_(False)):
        bad = copy.deepcopy(cells)
        bad[frozenset({'a'})][0] = flag
        with _pytest.raises(ValueError):
            L.shapley_lattice_attribution(
                bad, components=['a', 'b'], min_n=3)
        with _pytest.raises(ValueError):
            L.shapley_lattice_attribution(
                cells, components=['a', 'b'], min_n=flag)
        with _pytest.raises(ValueError):
            L.shapley_lattice_attribution(
                cells, components=['a', 'b'], max_components=flag)
    assert L.shapley_lattice_attribution(
        cells, components=['a', 'b'], min_n=3)['status'] == 'ok'


def test_one_component_degenerate_lattice():
    cells = _ppl_cells({frozenset(): 0.0, frozenset({'a'}): -1.0})
    out = L.shapley_lattice_attribution(cells, components=['a'])
    assert out['status'] == 'ok'
    _assert_close(out['shapley']['a']['mean'], -1.0)
    _assert_close(out['loo']['a']['mean'], -1.0)
    assert out['interactions'] == []
    assert out['higher_order']['mean'] == 0.0
    assert out['axioms']['efficiency']['pass'] is True


def test_ppl_and_log_ppl_scales_agree():
    log_values = {frozenset(): 0.0, frozenset({'a'}): -1.0,
                  frozenset({'b'}): -2.0, frozenset({'a', 'b'}): -2.5}
    ppl = _ppl_cells(log_values)
    logs = {k: {seed: value for seed in (0, 1, 2)}
            for k, value in log_values.items()}
    a = L.shapley_lattice_attribution(ppl, components=['a', 'b'])
    b = L.shapley_lattice_attribution(
        logs, components=['a', 'b'], scale='log_ppl')
    for comp in ('a', 'b'):
        _assert_close(a['shapley'][comp]['mean'], b['shapley'][comp]['mean'])
    _assert_close(a['total_effect']['mean'], b['total_effect']['mean'])
    _assert_close(a['interactions'][0]['at_empty']['mean'],
                  b['interactions'][0]['at_empty']['mean'])


def test_factorial_adapter_matches_direct_lattice():
    rows = []
    for seed, p00, p10, p01, p11 in (
            (0, 100.0, 90.0, 80.0, 60.0),
            (1, 100.0, 90.0, 80.0, 60.0),
            (2, 100.0, 90.0, 80.0, 60.0)):
        rows.append({'seed': seed,
                     'ppl': {'a00': p00, 'a10': p10,
                             'a01': p01, 'a11': p11}})
    out = L.factorial_lattice_attribution(
        rows, factors=['a', 'b'], arms=['a00', 'a10', 'a01', 'a11'])
    direct = L.shapley_lattice_attribution({
        frozenset(): {r['seed']: r['ppl']['a00'] for r in rows},
        frozenset({'a'}): {r['seed']: r['ppl']['a10'] for r in rows},
        frozenset({'b'}): {r['seed']: r['ppl']['a01'] for r in rows},
        frozenset({'a', 'b'}): {r['seed']: r['ppl']['a11'] for r in rows},
    }, components=['a', 'b'])
    for comp in ('a', 'b'):
        _assert_close(out['shapley'][comp]['mean'],
                      direct['shapley'][comp]['mean'])
    _assert_close(out['interactions'][0]['at_empty']['mean'],
                  math.log(60) - math.log(90) - math.log(80) + math.log(100))
    for bad_factors, bad_arms in (
            (['a'], ['a00', 'a10', 'a01', 'a11']),
            (['a', 'a'], ['a00', 'a10', 'a01', 'a11']),
            (['a', 'b'], ['a00', 'a10', 'a01']),
            (['a', 'b'], ['a00', 'a00', 'a01', 'a11'])):
        try:
            L.factorial_lattice_attribution(
                rows, factors=bad_factors, arms=bad_arms)
        except ValueError:
            pass
        else:
            raise AssertionError('malformed factorial adapter input must fail')


def test_design_audit_licenses_pair_scopes_not_global_shapley():
    out = L.ablation_lattice_design()
    assert out['status'] == 'ok', out['problems']
    assert out['mode'] == 'canonical'
    assert len(out['scopes']) == 5
    for scope in out['scopes'].values():
        assert scope['complete'] is True
        assert scope['shapley_identifiable'] is True
        assert scope['missing_subsets'] == []
    glob = out['global']
    assert glob['n_components'] == 6
    assert glob['shapley_identifiable'] is False
    assert glob['scope_overlap_counts']['dynamic_blocking'] == 5
    assert out['star_family']['complete_lattice'] is False
    assert out['star_family']['n_contrasts'] >= 20
    assert 'within_scope_two_component_shapley' in out['licensed_claims']
    assert ('global_component_importance_from_a_star_design'
            in out['unlicensed_claims'])


def test_design_audit_names_missing_arms_and_prices_completion():
    missing = L.ablation_lattice_design(present={'csa_dynamic'})
    assert missing['status'] == 'warnings'
    assert missing['scopes']['dynamic_blocking_x_learned_selection'][
        'shapley_identifiable'] is False
    assert missing['scopes']['dynamic_blocking_x_learned_selection'][
        'missing_arms_from_run_panel']
    proposed = L.ablation_lattice_design(
        components=['a', 'b', 'c'],
        arms_by_subset={
            frozenset(): 'a0', frozenset({'a'}): 'a1',
            frozenset({'b'}): 'b1', frozenset({'c'}): 'c1',
            frozenset({'a', 'b'}): 'ab', frozenset({'a', 'c'}): 'ac',
        },
        present={'a0', 'a1', 'b1', 'c1', 'ab', 'ac'}, n_seeds=4)
    assert proposed['status'] == 'incomplete'
    assert proposed['scopes']['proposed']['missing_subsets'] == [
        'b+c', 'a+b+c']
    assert proposed['scopes']['proposed']['completion']['additional_arms'] == 2
    assert proposed['scopes']['proposed']['completion'][
        'additional_cells'] == 8
    assert proposed['global']['shapley_identifiable'] is False
    complete = L.ablation_lattice_design(
        components=['a', 'b'],
        arms_by_subset={
            frozenset(): 'a0', frozenset({'a'}): 'a1',
            frozenset({'b'}): 'b1', frozenset({'a', 'b'}): 'ab'},
        present={'a0', 'a1', 'b1', 'ab'}, n_seeds=3)
    assert complete['status'] == 'ok'
    assert complete['global']['shapley_identifiable'] is True
    assert 'within_scope_shapley' in complete['licensed_claims']


def test_artifact_replay_detects_tampering_and_writer_is_safe():
    cells = _ppl_cells({
        frozenset(): 0.0, frozenset({'a'}): -1.0,
        frozenset({'b'}): -2.0, frozenset({'a', 'b'}): -2.5,
    })
    art = L.build_ablation_lattice(cells, components=['a', 'b'])
    assert art['status'] == 'ok'
    assert len(art['content_sha256']) == 64
    assert L.validate_ablation_lattice(art)['status'] == 'ok'
    # A JSON round-trip must remain replayable (no NaN/Infinity values).
    assert L.validate_ablation_lattice(json.loads(json.dumps(art)))[
        'status'] == 'ok'
    for path in (('shapley', 'a', 'mean'),
                 ('loo', 'b', 'mean'),
                 ('interactions', 0, 'at_empty', 'mean'),
                 ('total_effect', 'mean'),
                 ('status',),
                 ('inputs', 'cells')):
        bad = copy.deepcopy(art)
        target = bad
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = ([-999.0] if path[-1] == 'cells' else -999.0)
        verdict = L.validate_ablation_lattice(bad)
        assert verdict['status'] == 'fail', (path, verdict['problems'])
    assert L.validate_ablation_lattice([])['status'] == 'fail'
    d = tempfile.mkdtemp()
    first = L.write_ablation_lattice(d, art)
    assert first['written'] is True
    again = L.write_ablation_lattice(d, art)
    assert again['unchanged'] is True and again['written'] is False
    other = L.build_ablation_lattice(_ppl_cells({
        frozenset(): 0.0, frozenset({'a'}): -1.0,
        frozenset({'b'}): -1.0, frozenset({'a', 'b'}): -2.0,
    }), components=['a', 'b'])
    conflict = L.write_ablation_lattice(d, other)
    assert conflict['conflict'] is True and conflict['written'] is False
    replaced = L.write_ablation_lattice(d, other, overwrite=True)
    assert replaced['written'] is True
    with open(os.path.join(d, 'ablation_lattice.json'), encoding='utf-8') as fh:
        assert json.load(fh)['content_sha256'] == other['content_sha256']
    summary = L.ablation_lattice_summary(other)
    assert summary['ok'] is True
    assert summary['shapley']['a'] == other['shapley']['a']['mean']
    # A completely empty lattice is a valid not_estimable artifact, not a
    # crash: its recorded inputs must replay just like a populated one.
    empty = L.build_ablation_lattice({})
    assert empty['status'] == 'not_estimable'
    assert empty['inputs']['components'] == []
    assert L.validate_ablation_lattice(empty)['status'] == 'ok'
    assert L.validate_ablation_lattice(empty)['replayed_status'] == \
        'not_estimable'


def test_gate_design_and_artifact_replay():
    base = A.audit_ablation_lattice()
    assert base['status'] == 'ok', base['problems']
    assert base['measured'] is False
    assert base['n_scopes'] == 5
    assert base['global_attribution_licensed'] is False
    assert base['star_family']['complete_lattice'] is False
    missing = A.audit_ablation_lattice(os.path.join(
        tempfile.mkdtemp(), 'does_not_exist.json'))
    assert missing['status'] == 'not_verifiable_locally'
    cells = _ppl_cells({
        frozenset(): 0.0, frozenset({'a'}): -1.0,
        frozenset({'b'}): -2.0, frozenset({'a', 'b'}): -2.5,
    })
    art = L.build_ablation_lattice(cells, components=['a', 'b'])
    p = os.path.join(tempfile.mkdtemp(), 'ablation_lattice.json')
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(art, fh, allow_nan=False)
    good = A.audit_ablation_lattice(p)
    assert good['status'] == 'ok', good['problems']
    assert good['measured'] is True
    assert good['summary']['shapley']['a'] == art['shapley']['a']['mean']
    bad = copy.deepcopy(art)
    bad['shapley']['a']['mean'] = -999.0
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(bad, fh, allow_nan=False)
    assert A.audit_ablation_lattice(p)['status'] == 'fail'
    incomplete = L.build_ablation_lattice({
        frozenset(): {0: 1.0, 1: 1.0, 2: 1.0},
        frozenset({'a'}): {0: 0.9, 1: 0.9, 2: 0.9},
    }, components=['a', 'b'])
    p2 = os.path.join(tempfile.mkdtemp(), 'ablation_lattice.json')
    with open(p2, 'w', encoding='utf-8') as fh:
        json.dump(incomplete, fh, allow_nan=False)
    partial = A.audit_ablation_lattice(p2)
    assert partial['status'] == 'not_verifiable_locally'
    assert partial['artifact_status'] == 'not_estimable'


def test_report_cli_makefile_and_part_registration():
    assert 'ablation_lattice' in inspect.getsource(A.run_report)
    assert 'ablation_lattice' in inspect.getsource(A._run_cli)
    assert 'ablation_lattice' in inspect.getsource(A.audit_ablation_lattice)
    with open(os.path.join(REPO, 'Makefile'), encoding='utf-8') as fh:
        make = fh.read()
    assert 'attribution:' in make
    assert 'audit_gates.py ablation_lattice' in make
    with open(os.path.join(REPO, 'explib', '__init__.py'),
              encoding='utf-8') as fh:
        assert 'part_73_ablation_lattice.py' in fh.read()
    with open(os.path.join(REPO, '_audit_gates', '__init__.py'),
              encoding='utf-8') as fh:
        assert 'part_39_ablation_lattice.py' in fh.read()


def test_no_readme_or_github_mutation_in_this_round():
    # The project has a hard rule for this round: CI configuration is
    # off-limits and the README family may not be deleted/moved.  A source
    # check of the tracked paths is enough for this tree; it does not inspect
    # remote history.
    for name in ('README.md', 'README_zh.md', '.github'):
        assert os.path.exists(os.path.join(REPO, name)), name
