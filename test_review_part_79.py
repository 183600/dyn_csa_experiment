"""Contrast estimability / design-rank audit (zero GPU, no split read).

These checks exercise the design-preparation object that sits before power,
attrition, budget and rehearsal: whether the declared contrast is actually
identified by the planned cells.  They cover the exact block-arm incidence
criterion, multi-arm component cancellation, the exact rational design rank,
factor main effects/interactions, empty cells, aliased factors, artifact
replay and the gate/CLI/Makefile wiring.

No training, no data split, no frozen p/Holm/verdict is changed.
"""
import copy
import json
import os
import tempfile

import pytest

from test_support import *  # noqa: F401,F403

REPO = os.path.dirname(os.path.abspath(__file__))


def _complete_grid(arms, blocks):
    return [(arm, block) for block in blocks for arm in arms]


def _manifest():
    with open(os.path.join(REPO, 'confirmation_frozen.json'),
              encoding='utf-8') as fh:
        return json.load(fh)


def _run_order(arms, seeds):
    labels = [f'{arm}::s{seed}' for seed in seeds for arm in arms]
    return {
        'planned_cells': labels,
        'cell_variants': {f'{arm}::s{seed}': arm
                          for seed in seeds for arm in arms},
        'cell_blocks': {f'{arm}::s{seed}': str(seed)
                        for seed in seeds for arm in arms},
    }


def test_complete_block_design_is_estimable():
    cells = _complete_grid(['A', 'B'], [0, 1, 2])
    out = L.build_design_estimability(cells, [{'a': 'A', 'b': 'B'}])
    assert out['status'] == 'ok', out['problems']
    design = out['design']
    assert design['n_cells'] == 6
    assert design['n_arms'] == 2
    assert design['n_blocks'] == 3
    # intercept + 2 block dummies + 1 arm dummy
    assert design['n_columns'] == 4
    assert design['rank'] == 4
    assert design['deficiency'] == 0
    assert design['complete_grid'] is True
    assert design['components'] == [['A', 'B']]
    assert design['component_of'] == {'A': 0, 'B': 0}
    comparison = out['comparisons'][0]
    assert comparison['estimable'] is True
    assert comparison['reason'] == 'ok'
    assert comparison['n_blocks_both'] == 3
    assert out['summary']['n_estimable'] == 1
    assert out['summary']['n_not_estimable'] == 0


def test_disconnected_blocks_are_not_estimable():
    # A and B are each observed twice, but never in the same block.  The arm
    # effect is perfectly confounded with the block factor.
    cells = [('A', 0), ('A', 1), ('B', 2), ('B', 3)]
    out = L.build_design_estimability(cells, [{'a': 'A', 'b': 'B'}])
    assert out['status'] == 'fail', out
    assert out['design']['components'] == [['A'], ['B']]
    comparison = out['comparisons'][0]
    assert comparison['estimable'] is False
    assert comparison['reason'] == 'disconnected_components'
    assert comparison['n_blocks_both'] == 0
    assert any('never share a block' in p for p in out['problems'])


def test_multi_arm_component_cancellation():
    # A/B share block 0; C/D share block 1.  A - B + C - D has zero weight in
    # each component and is estimable; A - C does not cancel and is not.
    cells = [('A', 0), ('B', 0), ('C', 1), ('D', 1)]
    out = L.build_design_estimability(cells, [
        {'name': 'A - B + C - D',
         'weights': {'A': 1, 'B': -1, 'C': 1, 'D': -1}},
        {'name': 'A - C', 'weights': {'A': 1, 'C': -1}},
    ])
    assert out['status'] == 'fail'
    by_name = {c['name']: c for c in out['comparisons']}
    assert by_name['A - B + C - D']['estimable'] is True
    assert by_name['A - B + C - D']['reason'] == 'ok'
    assert by_name['A - C']['estimable'] is False
    assert by_name['A - C']['reason'] == 'unequal_component_totals'
    assert by_name['A - C']['components'] == {'0': 1.0, '1': -1.0}
    assert any('same total inside every connected component'
               in p for p in out['problems'])


def test_nonzero_sum_weights_warn_but_are_estimable():
    cells = _complete_grid(['A', 'B'], [0, 1])
    out = L.build_design_estimability(cells, [
        {'name': 'mean A', 'weights': {'A': 1}},
    ])
    assert out['status'] == 'warnings'
    comparison = out['comparisons'][0]
    assert comparison['estimable'] is True
    assert comparison['is_contrast'] is False
    assert comparison['sum_weights'] == 1.0
    assert any('not a pure contrast' in w for w in out['warnings'])


def test_missing_arm_and_duplicate_cell_fail_closed():
    cells = _complete_grid(['A', 'B'], [0, 1])
    out = L.build_design_estimability(cells, [{'a': 'A', 'b': 'Z'}])
    assert out['status'] == 'fail'
    comparison = out['comparisons'][0]
    assert comparison['estimable'] is False
    assert comparison['reason'] == 'arm_not_planned'
    assert comparison['missing_arms'] == ['Z']
    assert any('not planned' in p for p in out['problems'])

    duplicated = cells + [cells[0]]
    with pytest.raises(ValueError, match='duplicate'):
        L.build_design_estimability(duplicated, [{'a': 'A', 'b': 'B'}])


def test_factor_main_effects_and_interaction_are_estimable():
    arms = ['00', '01', '10', '11']
    cells = [(arm, block) for block in [0, 1] for arm in arms]
    factors = {
        'f1': {arm: arm[0] for arm in arms},
        'f2': {arm: arm[1] for arm in arms},
    }
    out = L.build_design_estimability(
        cells, [{'a': '00', 'b': '11'}], factors=factors)
    assert out['status'] == 'ok', out['problems']
    model = out['design']['treatment_model']
    # intercept + 1 block dummy + 3 effect-coded term columns
    assert model == {'n_columns': 5, 'rank': 5, 'deficiency': 0,
                     'base_columns': 2}
    assert set(out['terms']) == {'main:f1', 'main:f2', 'interaction:f1|f2'}
    for term in out['terms'].values():
        assert term['df'] == 1
        assert term['unique_df'] == 1
        assert term['own_df'] == 1
        assert term['estimable'] is True
        assert term['separately_estimable'] is True
        assert term['status'] == 'estimable'
    assert out['empty_combinations'] == {'f1|f2': []}
    assert out['aliased_pairs'] == []
    assert out['redundant_pairs'] == []


def test_empty_factorial_cell_makes_terms_aliased():
    arms = ['00', '01', '10']
    cells = [(arm, block) for block in [0, 1] for arm in arms]
    factors = {
        'f1': {arm: arm[0] for arm in arms},
        'f2': {arm: arm[1] for arm in arms},
    }
    out = L.build_design_estimability(
        cells, [{'a': '00', 'b': '10'}], factors=factors)
    assert out['status'] == 'warnings'
    assert out['empty_combinations']['f1|f2'] == [
        {'levels': ['1', '1']}]
    model = out['design']['treatment_model']
    assert model['deficiency'] == 1
    assert model['rank'] == model['n_columns'] - 1
    for term in out['terms'].values():
        # Each term is in the design span on its own, but no term has a
        # unique degree of freedom in the full declared model.
        assert term['estimable'] is True
        assert term['separately_estimable'] is False
        assert term['unique_df'] == 0
        assert term['status'] == 'aliased'
    assert any('unique df=0' in w for w in out['warnings'])


def test_identical_factor_partitions_are_aliased():
    arms = ['a', 'b']
    cells = [(arm, block) for block in [0, 1] for arm in arms]
    factors = {
        'f1': {'a': 'x', 'b': 'y'},
        'f2': {'a': 'x', 'b': 'y'},
    }
    out = L.build_design_estimability(
        cells, [{'a': 'a', 'b': 'b'}], factors=factors)
    assert out['status'] == 'warnings'
    assert {'terms': ['main:f1', 'main:f2'],
            'kind': 'identical_partition'} in out['aliased_pairs']
    assert out['redundant_pairs'][0]['overlap_df'] == 1
    for label in ('main:f1', 'main:f2'):
        assert out['terms'][label]['status'] == 'aliased'
        assert out['terms'][label]['separately_estimable'] is False


def test_one_level_factor_is_degenerate():
    cells = _complete_grid(['A', 'B'], [0, 1])
    out = L.build_design_estimability(
        cells, [{'a': 'A', 'b': 'B'}],
        factors={'f': {'A': 'x', 'B': 'x'}})
    assert out['status'] == 'warnings'
    term = out['terms']['main:f']
    assert term['df'] == 0
    assert term['status'] == 'degenerate'
    assert term['estimable'] is False
    assert any('only one level' in w for w in out['warnings'])


def test_exact_rank_and_fractional_weights_round_trip():
    cells = _complete_grid(['A', 'B'], [0, 1])
    out = L.build_design_estimability(cells, [
        {'name': 'half contrast',
         'weights': {'A': '1/2', 'B': '-1/2'}},
    ])
    assert out['status'] == 'ok', out['problems']
    assert out['comparisons'][0]['estimable'] is True
    # The stored inputs use exact fraction text, and replay accepts it.
    assert out['inputs']['comparisons'][0]['weights'] == {
        'A': '1/2', 'B': '-1/2'}
    verdict = L.validate_design_estimability(out)
    assert verdict['status'] == 'ok', verdict['problems']

    # A disconnected arm still makes the design deficient even though both
    # arms have the same number of cells.
    disconnected = [('A', 0), ('A', 1), ('B', 2), ('B', 3), ('C', 2)]
    bad = L.build_design_estimability(
        disconnected, [{'a': 'A', 'b': 'B'}])
    assert bad['design']['deficiency'] > 0
    assert bad['design']['rank'] < bad['design']['n_columns']


def test_replay_hash_and_writer_conflict():
    cells = _complete_grid(['A', 'B'], [0, 1])
    art = L.build_design_estimability(cells, [{'a': 'A', 'b': 'B'}])
    verdict = L.validate_design_estimability(art)
    assert verdict['status'] == 'ok'
    assert verdict['replayed'] is True
    assert art['content_sha256'] == L.design_estimability_content_sha(art)
    assert L.replay_design_estimability(art)['content_sha256'] == (
        art['content_sha256'])

    tampered = copy.deepcopy(art)
    tampered['design']['rank'] = 99
    bad = L.validate_design_estimability(tampered)
    assert bad['status'] == 'fail'
    assert any('content_sha256' in p for p in bad['problems'])
    assert any('design' in p for p in bad['problems'])

    with tempfile.TemporaryDirectory() as outdir:
        first = L.write_design_estimability(outdir, art)
        assert first['written'] is True
        again = L.write_design_estimability(outdir, art)
        assert again['written'] is False and again['noop'] is True
        different = L.build_design_estimability(
            _complete_grid(['A', 'B'], [0, 1, 2]),
            [{'a': 'A', 'b': 'B'}])
        assert different['content_sha256'] != art['content_sha256']
        conflict = L.write_design_estimability(outdir, different)
        assert conflict['written'] is False
        assert conflict['conflict'] is True
        overwrite = L.write_design_estimability(
            outdir, different, keep_existing=False)
        assert overwrite['written'] is True


def test_manifest_and_run_order_paths():
    manifest = _manifest()
    grid = L.design_estimability_from_manifest(manifest)
    assert grid['status'] == 'ok', grid['problems']
    assert grid['inputs']['cell_source'] == (
        'canonical_complete_arm_x_seed_grid')
    assert grid['design']['n_cells'] == 72
    assert grid['summary']['n_comparisons'] == 6
    assert grid['summary']['n_estimable'] == 6

    plan = _run_order(manifest['arms'], [0, 1])
    registered = L.design_estimability_from_manifest(
        manifest, run_order=plan)
    assert registered['status'] == 'ok', registered['problems']
    assert registered['inputs']['cell_source'] == 'registered_run_order'
    assert registered['design']['n_cells'] == 12
    assert registered['content_sha256'] != grid['content_sha256']

    broken = _run_order(manifest['arms'], [0])
    broken['cell_variants'] = dict(broken['cell_variants'])
    broken['cell_variants'][next(iter(broken['planned_cells']))] = 'not_an_arm'
    bad = L.design_estimability_from_manifest(manifest, run_order=broken)
    assert bad['status'] == 'fail'
    assert any('not planned' in p for p in bad['problems'])


def test_gate_statuses_and_tamper():
    live = A.audit_design_estimability(verbose=False)
    assert live['status'] == 'ok', live
    assert live['summary']['n_estimable'] == 6
    assert live['summary']['n_not_estimable'] == 0

    cells = _complete_grid(['A', 'B'], [0, 1])
    art = L.build_design_estimability(cells, [{'a': 'A', 'b': 'B'}])
    with tempfile.TemporaryDirectory() as outdir:
        written = L.write_design_estimability(outdir, art)
        gate = A.audit_design_estimability(path=written['path'],
                                           verbose=False)
        assert gate['status'] == 'ok'
        assert gate['artifact_status'] == 'ok'
        assert gate['summary']['n_estimable'] == 1

        broken = copy.deepcopy(art)
        broken['comparisons'][0]['estimable'] = False
        broken['status'] = 'fail'
        broken['content_sha256'] = L.design_estimability_content_sha(broken)
        with open(written['path'], 'w', encoding='utf-8') as fh:
            json.dump(broken, fh)
        tampered = A.audit_design_estimability(
            path=written['path'], verbose=False)
        assert tampered['status'] == 'fail'

    missing = A.audit_design_estimability(
        path=os.path.join(tempfile.gettempdir(), 'does-not-exist-de.json'),
        verbose=False)
    assert missing['status'] == 'fail'
    assert any('does not exist' in p for p in missing['problems'])

    # An artifact built from a different manifest body must not be replayed
    # against the frozen manifest.
    fake = _manifest()
    fake['seeds'] = [0, 1]
    foreign = L.design_estimability_from_manifest(fake)
    with tempfile.TemporaryDirectory() as outdir:
        written = L.write_design_estimability(outdir, foreign)
        mismatch = A.audit_design_estimability(
            path=written['path'], verbose=False)
        assert mismatch['status'] == 'fail'
        assert any('different manifest' in p for p in mismatch['problems'])


def test_cell_means_model_ignores_blocks_with_a_warning():
    cells = [('A', 0), ('A', 1), ('B', 2), ('B', 3)]
    out = L.build_design_estimability(
        cells, [{'a': 'A', 'b': 'B'}], model='cell_means')
    assert out['status'] == 'warnings'
    assert out['comparisons'][0]['estimable'] is True
    assert any('ignores block/seed effects' in w for w in out['warnings'])


def test_invalid_inputs_fail_closed():
    cells = _complete_grid(['A', 'B'], [0, 1])
    with pytest.raises(ValueError, match='model'):
        L.build_design_estimability(
            cells, [{'a': 'A', 'b': 'B'}], model='magic')
    with pytest.raises(ValueError, match='non-empty'):
        L.build_design_estimability([], [{'a': 'A', 'b': 'B'}])
    with pytest.raises(ValueError, match='non-empty'):
        L.build_design_estimability(cells, [])
    with pytest.raises(ValueError, match='duplicate name'):
        L.build_design_estimability(cells, [
            {'name': 'x', 'a': 'A', 'b': 'B'},
            {'name': 'x', 'a': 'A', 'b': 'B'},
        ])
    with pytest.raises(ValueError, match='every weight is zero'):
        L.build_design_estimability(cells, [
            {'name': 'zero', 'weights': {'A': 0, 'B': 0}},
        ])
    with pytest.raises(ValueError, match='not declared'):
        L.build_design_estimability(
            cells, [{'a': 'A', 'b': 'B'}],
            factors={'f': {'A': 'x'}})


def test_factor_shape_round_trips_both_orientations():
    cells = _complete_grid(['A', 'B'], [0, 1])
    declared = {'f': {'A': 'x', 'B': 'y'}}
    by_arm = {'A': {'f': 'x'}, 'B': {'f': 'y'}}
    for factors in (declared, by_arm):
        art = L.build_design_estimability(
            cells, [{'a': 'A', 'b': 'B'}], factors=factors)
        assert art['inputs']['factors_shape'] == 'arm_to_factor'
        verdict = L.validate_design_estimability(art)
        assert verdict['status'] == 'ok', verdict['problems']
    assert L.build_design_estimability(
        cells, [{'a': 'A', 'b': 'B'}],
        factors=declared)['inputs']['declared_factors_shape'] == (
            'factor_to_arm')


def test_wiring_and_no_inferential_fields():
    assert hasattr(A, 'audit_design_estimability')
    assert hasattr(L, 'build_design_estimability')
    assert hasattr(L, 'design_estimability_from_manifest')
    assert hasattr(L, 'validate_design_estimability')
    assert hasattr(L, 'write_design_estimability')
    import explib
    assert 'part_74_design_estimability.py' in explib._PARTS
    with open(os.path.join(REPO, 'Makefile'), encoding='utf-8') as fh:
        makefile = fh.read()
    assert 'estimability:' in makefile
    assert 'audit_gates.py design_estimability' in makefile
    # The audit is a design report, not a new inferential result.
    out = A.audit_design_estimability(verbose=False)
    for forbidden in ('p_value', 'p_holm', 'confirmed', 'holm_adjusted',
                      'verified'):
        assert forbidden not in out
    assert len(out['content_sha256']) == 64
    assert os.path.isdir(os.path.join(REPO, '.github'))
    assert os.path.exists(os.path.join(REPO, 'README.md'))
