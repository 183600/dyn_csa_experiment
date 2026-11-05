"""Budget-constrained ablation screening: fractional designs and aliasing.

These checks exercise a 2^(k-p) screening design as the positive alternative
to a full subset lattice when 2^k arms are unaffordable.  The key scientific
point is that aliasing is not a nuisance to hide: a resolution III main effect
is a sum with an aliased two-factor interaction, and a resolution IV pair of
two-factor interactions is one estimable group.  The tests therefore verify
the alias algebra, per-seed inference, artifact replay and the audit/CLI/
Makefile wiring, without training or reading any split.

No frozen p/Holm/verdict is changed.
"""
import copy
import inspect
import json
import math
import os
import random
import tempfile

import pytest

from test_support import *  # noqa: F401,F403

REPO = os.path.dirname(os.path.abspath(__file__))


def _words(*names):
    return list(names)


def _make_cells(components, relations, *, main_effects=None,
                interactions=None, n_seeds=12, noise=0.01, base=3.0,
                seed0=0):
    """Synthetic log-PPL panel on exactly the declared fraction.

    A main effect is the active-minus-inactive mean difference, so it enters
    the ±1 regression as ``effect/2 * x``; a two-factor interaction contrast
    enters as ``effect/4 * x_i*x_j``.  This keeps the generated values on the
    same scale as the reported screening estimates.
    """
    design = L.fractional_factorial_design(
        components, defining_relations=relations)
    main_effects = dict(main_effects or {})
    interactions = dict(interactions or {})
    cells = {}
    for row_index, row in enumerate(design['cells']):
        signs = {name: (1.0 if name in row else -1.0)
                 for name in components}
        value = float(base)
        for name, effect in main_effects.items():
            value += 0.5 * float(effect) * signs[name]
        for pair, effect in interactions.items():
            left, right = pair
            value += 0.25 * float(effect) * signs[left] * signs[right]
        per_seed = {}
        for seed in range(int(n_seeds)):
            # Noise must be independent across cells; sharing the same draw
            # for every subset would cancel exactly in every OLS contrast.
            rng = random.Random(
                10_000 + int(seed0) * 10_000 + seed * 37 + row_index)
            per_seed[seed] = math.exp(value + rng.gauss(0.0, float(noise)))
        cells[frozenset(row)] = per_seed
    return design, cells


def _assert_close(got, want, tol=0.02, msg=''):
    assert got is not None, (msg, got, want)
    assert abs(got - want) <= tol, (msg, got, want)


def test_resolution_iii_main_effects_alias_two_factor_interactions():
    design = L.fractional_factorial_design(
        ['A', 'B', 'C'], defining_relations=[['A', 'B', 'C']])
    assert design['n_cells'] == 4
    assert design['n_cells_full'] == 8
    assert design['saved_cells'] == 4
    assert design['saved_fraction'] == 0.5
    assert design['resolution'] == 3
    assert design['status'] == 'warnings'
    assert design['main_effects_separable'] is True
    assert design['main_effects_clear'] is False
    assert design['interactions_licensed'] is False
    assert design['requires_sparsity_assumption'] is True
    assert design['model_terms'] == ['A', 'B', 'C']
    aliases = {row['name']: row for row in [design['terms'][name]
                                            for name in design['terms']]}
    assert aliases['A']['aliased_with'] == ['B+C']
    assert aliases['B']['aliased_with'] == ['A+C']
    assert aliases['C']['aliased_with'] == ['A+B']
    # The empty subset is not part of an odd defining relation; the four
    # measured cells are the odd-cardinality subsets.
    assert sorted(design['cells'], key=lambda row: (len(row), row)) == [
        ['A'], ['B'], ['C'], ['A', 'B', 'C']]
    # No two main effects share an alias class: the design is a valid
    # main-effect screening design under the sparsity assumption.
    assert design['claim']['main_effects_licensed'] is True
    assert design['claim']['requires_sparsity_assumption'] is True


def test_resolution_iv_and_v_designs_price_the_tradeoff():
    res_iv = L.fractional_factorial_design(
        ['A', 'B', 'C', 'D'], defining_relations=[['A', 'B', 'C', 'D']],
        n_seeds=8)
    assert res_iv['n_cells'] == 8
    assert res_iv['saved_cells'] == 8
    assert res_iv['resolution'] == 4
    assert res_iv['main_effects_clear'] is True
    assert res_iv['interactions_licensed'] is False
    assert res_iv['model_terms'][:4] == ['A', 'B', 'C', 'D']
    assert 'A+B' in res_iv['model_terms']
    assert res_iv['terms']['A+B']['aliased_with'] == ['C+D']
    assert res_iv['terms']['A+C']['aliased_with'] == ['B+D']
    assert res_iv['terms']['A+D']['aliased_with'] == ['B+C']

    res_v = L.fractional_factorial_design(
        ['A', 'B', 'C', 'D', 'E'],
        defining_relations=[['A', 'B', 'C', 'D', 'E']], n_seeds=12)
    assert res_v['n_cells'] == 16
    assert res_v['saved_cells'] == 16
    assert res_v['resolution'] == 5
    assert res_v['status'] == 'ok'
    assert res_v['main_effects_clear'] is True
    assert res_v['interactions_licensed'] is True
    assert len(res_v['model_terms']) == 15

    full = L.fractional_factorial_design(['A', 'B', 'C'], n_seeds=12)
    assert full['n_cells'] == 8
    assert full['n_cells_full'] == 8
    assert full['saved_cells'] == 0
    assert full['resolution_label'] == 'full'
    assert full['status'] == 'ok'
    assert full['interactions_licensed'] is True


def test_invalid_designs_fail_closed():
    for components, relations in (
            (['A', 'A'], []),
            (['A'], []),
            (['A', 'B'], [['A']]),
            (['A', 'B'], [['A', 'C']]),
            (['A', 'B'], [['A', 'B'], ['A', 'B']]),
            ([f'C{i}' for i in range(9)], [])):
        with pytest.raises(ValueError):
            L.fractional_factorial_design(components,
                                          defining_relations=relations)
    degenerate = L.fractional_factorial_design(
        ['A', 'B'], defining_relations=[['A', 'B']])
    assert degenerate['status'] == 'not_estimable'
    # A resolution-II relation is a valid declaration but not a usable
    # screening design: main effects share an alias class.
    aliased = L.fractional_factorial_design(
        ['A', 'B', 'C'], defining_relations=[['A', 'B', 'C'], ['A', 'B']])
    assert aliased['status'] == 'not_estimable'
    assert aliased['main_effects_separable'] is False
    with pytest.raises(ValueError):
        L.fractional_factorial_design(['A', 'B'], max_order=3)
    with pytest.raises(ValueError):
        L.fractional_factorial_design(['A', 'B'], n_seeds=0)


def test_not_estimable_design_cannot_license_a_claim():
    # A 2^(4-2) fraction with four cells cannot fit an intercept plus four
    # main effects.  The alias algebra alone may look separable, but the
    # design-level and embedded claim flags must both stay false.
    design = L.fractional_factorial_design(
        ['A', 'B', 'C', 'D'],
        defining_relations=[['A', 'B', 'C', 'D'], ['A', 'B', 'C']],
        max_order=2)
    assert design['status'] == 'not_estimable'
    assert design['main_effects_licensed'] is False
    assert design['interactions_licensed'] is False
    assert design['claim']['main_effects_licensed'] is False
    assert design['claim']['interactions_licensed'] is False

    # The analysis must return the named not_estimable verdict from the same
    # cells instead of raising on the design's own aliased default terms.
    cells = {frozenset(row): {0: 30.0, 1: 31.0, 2: 32.0}
             for row in design['cells']}
    out = L.ablation_screening_analysis(
        cells, components=['A', 'B', 'C', 'D'],
        defining_relations=[['A', 'B', 'C', 'D'], ['A', 'B', 'C']],
        max_order=2, min_n=2)
    assert out['status'] == 'not_estimable'
    assert out['main_effects_licensed'] is False
    assert out['interactions_licensed'] is False
    json.dumps(out, allow_nan=False)


def test_all_empty_cells_report_zero_seeds_not_a_crash():
    design = L.fractional_factorial_design(['A', 'B'])
    cells = {frozenset(row): {} for row in design['cells']}
    out = L.ablation_screening_analysis(
        cells, components=['A', 'B'], min_n=2)
    assert out['status'] == 'not_estimable'
    assert out['n_seeds'] == 0
    assert any('n=0' in problem for problem in out['problems'])
    json.dumps(out, allow_nan=False)


def test_seed_level_effects_recover_declared_main_effects():
    design, cells = _make_cells(
        ['A', 'B', 'C', 'D'], [['A', 'B', 'C', 'D']],
        main_effects={'A': -0.30, 'B': -0.20, 'C': -0.10, 'D': 0.0},
        n_seeds=12, noise=0.01)
    out = L.ablation_screening_analysis(
        cells, components=['A', 'B', 'C', 'D'],
        defining_relations=[['A', 'B', 'C', 'D']], alpha=0.05, min_n=8)
    assert out['status'] == 'ok', (out['status'], out['warnings'])
    assert out['n_seeds'] == 12
    assert out['family']['method'] == 'exact_maxt'
    assert out['family']['n_patterns'] == 2 ** 12
    assert out['n_significant_main'] >= 3
    for name, want in (('A', -0.30), ('B', -0.20), ('C', -0.10), ('D', 0.0)):
        row = out['terms'][name]
        _assert_close(row['mean'], want, tol=0.02, msg=name)
        assert row['n'] == 12
        assert row['effect_scale'] == 2
        assert row['p_family'] is not None
        assert 0.0 < row['p_family'] <= 1.0
        assert row['ci95'] is not None
        assert row['ci95'][0] <= row['mean'] <= row['ci95'][1]
        simultaneous = row['ci95_simultaneous']
        if simultaneous is not None:
            assert simultaneous[0] <= row['mean'] <= simultaneous[1]
    assert out['terms']['A']['reject_family'] is True
    assert out['terms']['B']['reject_family'] is True
    assert out['terms']['D']['reject_family'] is False
    # Resolution IV makes the two-factor columns alias in pairs; the analysis
    # labels the class and must not pretend the two effects are separate.
    assert out['terms']['A+B']['aliased_with'] == ['C+D']
    assert out['terms']['A+B']['p_family'] is not None
    json.dumps(out, allow_nan=False)


def test_resolution_iii_estimate_is_main_plus_half_the_alias():
    # On the fraction x_A = x_B*x_C, so a 2FI contrast of +0.30 shifts the
    # reported A "main effect" by +0.15 (one half, exactly the alias algebra).
    design, cells = _make_cells(
        ['A', 'B', 'C'], [['A', 'B', 'C']],
        main_effects={'A': -0.20, 'B': -0.10, 'C': -0.05},
        interactions={('B', 'C'): 0.30}, n_seeds=12, noise=0.01)
    out = L.ablation_screening_analysis(
        cells, components=['A', 'B', 'C'],
        defining_relations=[['A', 'B', 'C']], alpha=0.05, min_n=8)
    assert out['status'] == 'warnings'
    assert out['requires_sparsity_assumption'] is True
    assert out['main_effects_separable'] is True
    assert out['main_effects_clear'] is False
    assert out['terms']['A']['aliased_with'] == ['B+C']
    _assert_close(out['terms']['A']['mean'], -0.20 + 0.15, tol=0.02,
                  msg='aliased main effect')
    _assert_close(out['terms']['B']['mean'], -0.10, tol=0.02)
    _assert_close(out['terms']['C']['mean'], -0.05, tol=0.02)
    assert out['claim']['screening_only'] is True
    assert out['claim']['interactions_licensed'] is False
    assert any('sparsity-of-effects' in text for text in out['warnings'])


def test_resolution_iv_pair_is_one_estimable_group():
    _, cells = _make_cells(
        ['A', 'B', 'C', 'D'], [['A', 'B', 'C', 'D']],
        main_effects={'A': -0.2, 'B': -0.1, 'C': -0.05, 'D': 0.0},
        interactions={('A', 'B'): 0.40, ('C', 'D'): 0.20},
        n_seeds=12, noise=0.01)
    out = L.ablation_screening_analysis(
        cells, components=['A', 'B', 'C', 'D'],
        defining_relations=[['A', 'B', 'C', 'D']], alpha=0.05, min_n=8)
    # AB and CD are the same column: the reported effect is their sum.
    _assert_close(out['terms']['A+B']['mean'], 0.60, tol=0.03,
                  msg='aliased pair sum')
    assert out['terms']['A+B']['aliased_with'] == ['C+D']
    assert out['terms']['A+C']['aliased_with'] == ['B+D']
    assert out['interactions_licensed'] is False
    assert out['claim']['interactions_licensed'] is False


def test_missing_and_extra_cells_are_not_imputed():
    design = L.fractional_factorial_design(
        ['A', 'B', 'C', 'D'], defining_relations=[['A', 'B', 'C', 'D']])
    _, cells = _make_cells(
        ['A', 'B', 'C', 'D'], [['A', 'B', 'C', 'D']],
        main_effects={'A': -0.2}, n_seeds=8)
    dropped = frozenset(design['cells'][0])
    del cells[dropped]
    out = L.ablation_screening_analysis(
        cells, components=['A', 'B', 'C', 'D'],
        defining_relations=[['A', 'B', 'C', 'D']], min_n=3)
    assert out['status'] == 'not_estimable'
    assert out['missing_cells'] == [[]]  # the removed cell is the empty subset
    assert out['main_effects_licensed'] is False
    assert out['n_significant_main'] == 0
    art = L.build_ablation_screening(
        cells, components=['A', 'B', 'C', 'D'],
        defining_relations=[['A', 'B', 'C', 'D']], min_n=3)
    assert L.validate_ablation_screening(art)['status'] == 'ok'

    _, cells2 = _make_cells(
        ['A', 'B', 'C', 'D'], [['A', 'B', 'C', 'D']], n_seeds=8)
    cells2[frozenset({'A'})] = {0: 1.0, 1: 1.0, 2: 1.0}
    out2 = L.ablation_screening_analysis(
        cells2, components=['A', 'B', 'C', 'D'],
        defining_relations=[['A', 'B', 'C', 'D']], min_n=3)
    assert out2['status'] == 'not_estimable'
    assert out2['extra_cells'] == [['A']]


def test_seed_intersection_and_resolution_limit_are_reported():
    design, cells = _make_cells(
        ['A', 'B', 'C', 'D'], [['A', 'B', 'C', 'D']],
        main_effects={'A': -0.3}, n_seeds=8)
    first = frozenset(design['cells'][1])
    cells[first].pop(0)
    out = L.ablation_screening_analysis(
        cells, components=['A', 'B', 'C', 'D'],
        defining_relations=[['A', 'B', 'C', 'D']], min_n=3)
    assert out['n_seeds'] == 7
    assert out['seeds'] == list(range(1, 8))
    assert out['missing_seeds']
    assert out['status'] in ('ok', 'warnings')
    out2 = L.ablation_screening_analysis(
        cells, components=['A', 'B', 'C', 'D'],
        defining_relations=[['A', 'B', 'C', 'D']], min_n=8)
    assert out2['status'] == 'not_estimable'
    assert any('intersection' in text for text in out2['problems'])


def test_small_seed_panel_reports_resolution_limit_not_null():
    _, cells = _make_cells(
        ['A', 'B', 'C', 'D'], [['A', 'B', 'C', 'D']],
        main_effects={'A': -0.5}, n_seeds=4, noise=0.01)
    out = L.ablation_screening_analysis(
        cells, components=['A', 'B', 'C', 'D'],
        defining_relations=[['A', 'B', 'C', 'D']], min_n=2)
    assert out['family']['resolution_limited'] is True
    assert out['family']['p_floor_single'] == 2.0 ** -3
    assert out['status'] == 'warnings'
    assert any('resolution limit' in text for text in out['warnings'])
    assert out['n_significant_main'] == 0


def test_bad_cells_model_terms_and_thresholds_fail_closed():
    components = ['A', 'B', 'C']
    relations = [['A', 'B', 'C']]
    design = L.fractional_factorial_design(components,
                                           defining_relations=relations)
    base = {frozenset(row): {seed: 2.0 for seed in range(4)}
            for row in design['cells']}
    for bad in ([], 'cells'):
        with pytest.raises(ValueError):
            L.ablation_screening_analysis(
                bad, components=components, defining_relations=relations)
    empty = L.ablation_screening_analysis(
        {}, components=components, defining_relations=relations)
    assert empty['status'] == 'not_estimable'
    for bad_ppl in (0.0, -1.0, float('nan'), float('inf'), True, '2.0'):
        cells = copy.deepcopy(base)
        cells[frozenset(design['cells'][0])][0] = bad_ppl
        with pytest.raises(ValueError):
            L.ablation_screening_analysis(
                cells, components=components, defining_relations=relations)
    cells = copy.deepcopy(base)
    cells[frozenset(design['cells'][0])]['0'] = 2.0
    with pytest.raises(ValueError):
        L.ablation_screening_analysis(
            cells, components=components, defining_relations=relations)
    cells = copy.deepcopy(base)
    cells[frozenset({'A', 'Z'})] = {0: 2.0}
    with pytest.raises(ValueError):
        L.ablation_screening_analysis(
            cells, components=components, defining_relations=relations)
    for bad_alpha in (0.0, 1.0, True, 'nan'):
        with pytest.raises(ValueError):
            L.ablation_screening_analysis(
                base, components=components, defining_relations=relations,
                alpha=bad_alpha)
    with pytest.raises(ValueError):
        L.ablation_screening_analysis(
            base, components=components, defining_relations=relations,
            min_n=1)
    with pytest.raises(ValueError):
        L.ablation_screening_analysis(
            base, components=components, defining_relations=relations,
            max_order=4)
    with pytest.raises(ValueError):
        L.ablation_screening_analysis(
            base, components=components, defining_relations=relations,
            model_terms=['A', 'B'])  # missing C
    with pytest.raises(ValueError):
        L.ablation_screening_analysis(
            base, components=components, defining_relations=relations,
            model_terms=['A', 'B', 'C', 'B+C'])  # aliased with A
    with pytest.raises(ValueError):
        L.ablation_screening_analysis(
            base, components=components, defining_relations=relations,
            model_terms=['A', 'B', 'C', 'A+B', 'A+B'])


def test_artifact_round_trip_identity_and_tamper_rejection():
    _, cells = _make_cells(
        ['A', 'B', 'C', 'D'], [['A', 'B', 'C', 'D']],
        main_effects={'A': -0.3, 'B': -0.2}, n_seeds=12)
    art = L.build_ablation_screening(
        cells, components=['A', 'B', 'C', 'D'],
        defining_relations=[['A', 'B', 'C', 'D']], created_at='2026-01-01')
    art2 = L.build_ablation_screening(
        cells, components=['A', 'B', 'C', 'D'],
        defining_relations=[['A', 'B', 'C', 'D']], created_at='2027-02-02')
    assert art['content_sha256'] == art2['content_sha256']
    assert L.validate_ablation_screening(art)['status'] == 'ok'
    round_trip = json.loads(json.dumps(art, allow_nan=False))
    assert L.validate_ablation_screening(round_trip)['status'] == 'ok'
    assert art['result']['n_cells'] == 8
    assert art['result']['n_cells_full'] == 16
    assert art['result']['saved_cells'] == 8

    tamper = copy.deepcopy(art)
    tamper['result']['terms']['A']['mean'] = 0.0
    assert L.validate_ablation_screening(tamper)['status'] == 'fail'
    tamper = copy.deepcopy(art)
    tamper['result']['terms']['A+B']['aliased_with'] = []
    assert L.validate_ablation_screening(tamper)['status'] == 'fail'
    tamper = copy.deepcopy(art)
    tamper['result']['n_cells_full'] = 8
    assert L.validate_ablation_screening(tamper)['status'] == 'fail'
    tamper = copy.deepcopy(art)
    tamper['result']['confirmed'] = True
    assert L.validate_ablation_screening(tamper)['status'] == 'fail'
    tamper = copy.deepcopy(art)
    del tamper['result']['n_seeds']
    assert L.validate_ablation_screening(tamper)['status'] == 'fail'
    tamper = copy.deepcopy(art)
    tamper['content_sha256'] = '0' * 64
    assert L.validate_ablation_screening(tamper)['status'] == 'fail'
    tamper = copy.deepcopy(art)
    tamper['extra'] = 1
    assert L.validate_ablation_screening(tamper)['status'] == 'fail'
    tamper = copy.deepcopy(art)
    tamper['inputs']['extra'] = 1
    assert L.validate_ablation_screening(tamper)['status'] == 'fail'


def test_writer_is_atomic_and_refuses_silent_replacement():
    _, cells = _make_cells(
        ['A', 'B', 'C', 'D'], [['A', 'B', 'C', 'D']],
        main_effects={'A': -0.3}, n_seeds=12)
    art = L.build_ablation_screening(
        cells, components=['A', 'B', 'C', 'D'],
        defining_relations=[['A', 'B', 'C', 'D']])
    with tempfile.TemporaryDirectory() as tmp:
        first = L.write_ablation_screening(tmp, art)
        assert first['status'] == 'ok' and first['written'] is True
        second = L.write_ablation_screening(tmp, art)
        assert second['status'] == 'ok' and second['unchanged'] is True
        changed = copy.deepcopy(art)
        changed['created_at'] = '2027-01-01'
        # created_at is outside the content hash: same identity, no rewrite.
        same = L.write_ablation_screening(tmp, changed)
        assert same['unchanged'] is True
        _, cells2 = _make_cells(
            ['A', 'B', 'C', 'D'], [['A', 'B', 'C', 'D']],
            main_effects={'A': -0.1}, n_seeds=12)
        different = L.build_ablation_screening(
            cells2, components=['A', 'B', 'C', 'D'],
            defining_relations=[['A', 'B', 'C', 'D']])
        assert different['content_sha256'] != art['content_sha256']
        conflict = L.write_ablation_screening(tmp, different)
        assert conflict['conflict'] is True and conflict['written'] is False
        overwritten = L.write_ablation_screening(tmp, different, overwrite=True)
        assert overwritten['status'] == 'ok' and overwritten['written'] is True


def test_gate_statuses_and_replay(tmp_path):
    _, cells = _make_cells(
        ['A', 'B', 'C', 'D'], [['A', 'B', 'C', 'D']],
        main_effects={'A': -0.4, 'B': -0.3, 'C': -0.2, 'D': -0.1},
        n_seeds=12, noise=0.005)
    art = L.build_ablation_screening(
        cells, components=['A', 'B', 'C', 'D'],
        defining_relations=[['A', 'B', 'C', 'D']])
    path = os.path.join(str(tmp_path), 'ablation_screening.json')
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(art, handle, allow_nan=False)
    gate = A.audit_ablation_screening(path, verbose=False)
    assert gate['status'] == 'ok', gate
    assert gate['resolution'] == 4
    assert gate['n_cells'] == 8
    assert gate['saved_cells'] == 8
    assert gate['main_effects_licensed'] is True
    assert gate['interactions_licensed'] is False
    assert gate['aliased_terms'] == ['A+B', 'A+C', 'B+C']

    # A self-consistent screening null is a warning, not a tamper.
    _, null_cells = _make_cells(
        ['A', 'B', 'C', 'D'], [['A', 'B', 'C', 'D']], n_seeds=12)
    null_art = L.build_ablation_screening(
        null_cells, components=['A', 'B', 'C', 'D'],
        defining_relations=[['A', 'B', 'C', 'D']])
    null_path = os.path.join(str(tmp_path), 'null.json')
    with open(null_path, 'w', encoding='utf-8') as handle:
        json.dump(null_art, handle, allow_nan=False)
    assert A.audit_ablation_screening(null_path, verbose=False)['status'] == 'warnings'

    # An incomplete design is not locally verifiable.
    incomplete = copy.deepcopy(cells)
    incomplete.pop(frozenset([]))
    incomplete_art = L.build_ablation_screening(
        incomplete, components=['A', 'B', 'C', 'D'],
        defining_relations=[['A', 'B', 'C', 'D']])
    incomplete_path = os.path.join(str(tmp_path), 'incomplete.json')
    with open(incomplete_path, 'w', encoding='utf-8') as handle:
        json.dump(incomplete_art, handle, allow_nan=False)
    assert (A.audit_ablation_screening(incomplete_path, verbose=False)['status']
            == 'not_verifiable_locally')

    # Tampering is a failure.
    bad = copy.deepcopy(art)
    bad['result']['terms']['A']['mean'] = 0.0
    bad_path = os.path.join(str(tmp_path), 'bad.json')
    with open(bad_path, 'w', encoding='utf-8') as handle:
        json.dump(bad, handle, allow_nan=False)
    assert A.audit_ablation_screening(bad_path, verbose=False)['status'] == 'fail'

    # Missing artifact is not a pass.
    missing = A.audit_ablation_screening(
        os.path.join(str(tmp_path), 'absent.json'), verbose=False)
    assert missing['status'] == 'not_verifiable_locally'


def test_markdown_and_wiring():
    _, cells = _make_cells(
        ['A', 'B', 'C', 'D'], [['A', 'B', 'C', 'D']],
        main_effects={'A': -0.3}, n_seeds=12)
    out = L.ablation_screening_analysis(
        cells, components=['A', 'B', 'C', 'D'],
        defining_relations=[['A', 'B', 'C', 'D']])
    text = L.ablation_screening_markdown(out)
    assert 'A+B' in text and 'C+D' in text
    assert 'IV' in text and 'cells=' in text

    for name in ('fractional_factorial_design',
                 'ablation_screening_analysis',
                 'build_ablation_screening',
                 'validate_ablation_screening',
                 'write_ablation_screening',
                 'ablation_screening_markdown'):
        assert callable(getattr(L, name)), name
    assert 'ablation_screening' in inspect.getsource(A.run_report)
    assert 'ablation_screening' in inspect.getsource(A._run_cli)
    assert 'ablation_screening' in inspect.getsource(
        A.audit_ablation_screening)
    with open(os.path.join(REPO, 'Makefile'), encoding='utf-8') as handle:
        make = handle.read()
    assert 'ablation-screen:' in make
    assert 'audit_gates.py ablation_screening' in make
    with open(os.path.join(REPO, 'explib', '__init__.py'),
              encoding='utf-8') as handle:
        assert 'part_81_ablation_screening.py' in handle.read()
    with open(os.path.join(REPO, '_audit_gates', '__init__.py'),
              encoding='utf-8') as handle:
        assert 'part_47_audit_ablation_screening.py' in handle.read()


def test_no_readme_or_github_mutation_in_this_round():
    for name in ('README.md', 'README_zh.md', '.github'):
        assert os.path.exists(os.path.join(REPO, name)), name
