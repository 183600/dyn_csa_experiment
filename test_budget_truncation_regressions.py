"""Regression checks for budget-truncation feasibility (part_72 / gate).

These are zero-GPU checks.  They cover the fail-open where the block set was
inferred from the cells (so a plan that dropped a frozen seed could still be
reported as complete), malformed declarations that used to be silently
defaulted, and untrusted stored artifacts that used to crash the validator,
summary formatter or audit gate instead of failing closed.
"""
import copy
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

import pytest

REPO = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(REPO, 'confirmation_frozen.json')


def _manifest():
    with open(MANIFEST, encoding='utf-8') as fh:
        return json.load(fh)


def _artifact(manifest=None, **kwargs):
    manifest = manifest or _manifest()
    return L.budget_truncation_from_manifest(manifest, **kwargs)


def _rehash(artifact):
    out = copy.deepcopy(artifact)
    out['content_sha256'] = L.budget_truncation_content_sha(out)
    return out


def test_missing_declared_seed_cannot_look_complete():
    man = _manifest()
    a = _artifact(man)
    cells = [c for c in a['inputs']['cells'] if c['block'] != '11']
    bad = L.build_budget_truncation(
        cells=cells, comparisons=a['inputs']['comparisons'],
        arms=man['arms'], seeds=man['seeds'], min_n=8,
        complete_coverage=True, max_attempts_per_cell=2, budget=72)
    assert bad['status'] == 'fail', bad['status']
    assert any('has no cell in the panel' in p for p in bad['problems']), \
        bad['problems']
    # The design still declares 12 seeds; it must not be relabelled as 11.
    assert bad['design']['n_blocks'] == 12
    assert bad['floor']['planned_seeds'] == 12
    assert bad['floor']['effective_required_seeds'] == 12
    assert bad['floor']['effective_required_cells'] is None
    assert bad['floor']['shortest_complete_prefix_cells'] is None
    assert bad['frontier']['first_complete_cells'] is None


def test_declared_grid_mismatches_fail_closed():
    man = _manifest()
    a = _artifact(man)
    base = dict(cells=a['inputs']['cells'],
                comparisons=a['inputs']['comparisons'],
                arms=man['arms'], seeds=man['seeds'], min_n=8,
                complete_coverage=True, max_attempts_per_cell=2, budget=72)
    # A declared arm outside the comparison family with no cell is a grid
    # warning (it consumes no pair); an undeclared observed arm is a problem.
    extra = dict(base)
    extra['arms'] = list(man['arms']) + ['unused_arm']
    rep = L.build_budget_truncation(**extra)
    assert rep['status'] in ('warnings', 'not_verifiable'), rep['status']
    assert any('declared arm' in w for w in rep['warnings']), rep['warnings']
    unknown_arm = dict(base)
    unknown_arm['arms'] = list(man['arms'])
    unknown_arm['cells'] = list(base['cells']) + [
        {'label': 'not_declared::seed0', 'variant': 'not_declared',
         'block': '0'}]
    rep = L.build_budget_truncation(**unknown_arm)
    assert rep['status'] == 'fail'
    assert any('undeclared arm' in p for p in rep['problems'])
    # An undeclared block / duplicate declared seed are structural too.
    unknown = dict(base)
    unknown['cells'] = list(base['cells']) + [
        {'label': 'full_rope::seed99', 'variant': 'full_rope',
         'block': '99'}]
    rep = L.build_budget_truncation(**unknown)
    assert rep['status'] == 'fail'
    assert any('undeclared seed/block' in p for p in rep['problems'])
    dup = dict(base)
    dup['seeds'] = list(man['seeds']) + [11]
    rep = L.build_budget_truncation(**dup)
    assert rep['status'] == 'fail'
    assert any('duplicate identifier' in p for p in rep['problems'])
    # A string is not an iterable of arm names.
    rep = L.build_budget_truncation(**{**base, 'arms': 'abc'})
    assert rep['status'] == 'fail'
    assert any('arms must be a list' in p for p in rep['problems'])


def test_frontier_cap_keeps_exact_thresholds_and_rejects_bad_cap():
    man = _manifest()
    a = _artifact(man)
    blocks = [str(s) for s in man['seeds']]
    capped = L.truncation_frontier(
        a['inputs']['cells'], a['inputs']['comparisons'], min_n=8,
        complete_coverage=True, max_points=5, required_blocks=blocks)
    assert capped['status'] == 'ok', capped['problems']
    assert len(capped['points']) <= 5
    assert capped['sampled'] is True
    assert capped['first_min_n_cells'] == 48
    assert capped['first_complete_cells'] == 72
    for value in ('x', 0, -1, 1.5, True, {}):
        bad = L.truncation_frontier(
            a['inputs']['cells'], a['inputs']['comparisons'], min_n=8,
            complete_coverage=True, max_points=value, required_blocks=blocks)
        assert bad['status'] == 'fail', (value, bad)
        assert any('max_points' in p for p in bad['problems']), bad['problems']


def test_frontier_point_rejects_malformed_min_n_and_prefix():
    man = _manifest()
    a = _artifact(man)
    for value in (None, 'x', 0, 1, 1.5, True):
        with pytest.raises(ValueError):
            L.truncation_frontier_point(
                a['inputs']['cells'], a['inputs']['comparisons'], 1,
                min_n=value)
    with pytest.raises(ValueError):
        L.truncation_frontier_point(
            a['inputs']['cells'], a['inputs']['comparisons'], 'x', min_n=8)


def test_validation_rejects_missing_binding_tampered_policy_and_shapes():
    man = _manifest()
    a = _artifact(man)
    assert L.validate_budget_truncation(a, manifest=man)['status'] == 'ok'

    no_binding = copy.deepcopy(a)
    no_binding['study']['manifest_canonical_sha256'] = None
    no_binding['inputs']['manifest_canonical_sha256'] = None
    verdict = L.validate_budget_truncation(_rehash(no_binding), manifest=man)
    assert verdict['status'] == 'fail'
    assert any('binding' in p for p in verdict['problems']), verdict['problems']

    bad_policy = copy.deepcopy(a)
    bad_policy['policy'] = []
    verdict = L.validate_budget_truncation(_rehash(bad_policy))
    assert verdict['status'] == 'fail'
    assert any('policy' in p for p in verdict['problems']), verdict['problems']

    # Non-list problems/warnings used to raise TypeError in the replay check.
    for field in ('problems', 'warnings'):
        bad = copy.deepcopy(a)
        bad[field] = 1
        verdict = L.validate_budget_truncation(bad)
        assert verdict['status'] == 'fail'
        assert any(field in p for p in verdict['problems']), verdict['problems']

    # A malformed nested section must not crash the replay validator either.
    for field in ('study', 'design', 'floor', 'cost', 'budget', 'frontier'):
        bad = copy.deepcopy(a)
        bad[field] = 'not-an-object'
        verdict = L.validate_budget_truncation(bad)
        assert verdict['status'] == 'fail'


def test_summary_and_gate_fail_closed_on_malformed_artifacts():
    man = _manifest()
    a = _artifact(man)
    for field in ('study', 'design', 'floor', 'cost', 'budget'):
        bad = copy.deepcopy(a)
        bad[field] = 'not-an-object'
        summary = L.budget_truncation_summary(bad)
        json.dumps(summary, allow_nan=False)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'budget_truncation.json')
        for field in ('study', 'design', 'floor', 'cost', 'budget',
                      'problems', 'warnings'):
            bad = copy.deepcopy(a)
            bad[field] = 'not-an-object'
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(bad, fh)
            gate = A.audit_budget_truncation([path])
            assert gate['status'] in ('fail', 'not_verifiable'), (field, gate)
            json.dumps(gate, allow_nan=False)
    # A malformed path argument is a reported failure, never an exception.
    gate = A.audit_budget_truncation(5)
    assert gate['status'] == 'fail'


def test_gate_does_not_turn_warning_artifact_green():
    a = _artifact(_manifest(), budget=72)  # covers the floor, no risk model
    assert a['status'] == 'warnings', (a['status'], a['problems'])
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'budget_truncation.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(a, fh)
        gate = A.audit_budget_truncation([path])
        assert gate['status'] == 'warnings', gate
        item = gate['artifacts'][path]
        assert item['artifact_status'] == 'warnings'
        assert item['status'] == 'warnings', item


def test_failure_model_attempts_are_validated_not_silently_defaulted():
    man = _manifest()
    a = _artifact(man)
    base = dict(cells=a['inputs']['cells'],
                comparisons=a['inputs']['comparisons'], arms=man['arms'],
                seeds=man['seeds'], min_n=8, complete_coverage=True,
                max_attempts_per_cell=2, budget=76)
    for bad in ('x', 0, -1, 1.5, True):
        model = {'replaceable_per_attempt': 0.02,
                 'non_replaceable_per_attempt': 0.0,
                 'max_attempts_per_cell': bad}
        rep = L.build_budget_truncation(
            **{**base, 'failure_model': model})
        assert rep['status'] == 'fail', (bad, rep['status'])
        assert any('max_attempts_per_cell' in p for p in rep['problems'])
    # An omitted/None cap means "use the registered cap", not a malformed one.
    for value in (None,):
        model = {'replaceable_per_attempt': 0.02,
                 'non_replaceable_per_attempt': 0.0,
                 'max_attempts_per_cell': value}
        rep = L.build_budget_truncation(
            **{**base, 'failure_model': model})
        assert rep['status'] == 'warnings', rep['problems']
        assert rep['inputs']['failure_model']['max_attempts_per_cell'] == 2


def test_complete_coverage_declaration_is_not_invented():
    man = _manifest()
    for value in (1, 0, 'true', None):
        bad = copy.deepcopy(man)
        bad['statistics']['require_complete_coverage'] = value
        rep = L.budget_truncation_from_manifest(bad)
        assert rep['status'] == 'fail', (value, rep['status'])
        assert any('complete_coverage' in p for p in rep['problems']), \
            (value, rep['problems'])
    missing = copy.deepcopy(man)
    del missing['statistics']['require_complete_coverage']
    rep = L.budget_truncation_from_manifest(missing)
    assert rep['status'] == 'fail'
    assert any('complete_coverage' in p for p in rep['problems']), \
        rep['problems']


def test_declared_n_seeds_must_match_the_seed_grid():
    man = _manifest()
    mismatch = copy.deepcopy(man)
    mismatch['statistics']['n_seeds'] = len(man['seeds']) - 1
    rep = L.budget_truncation_from_manifest(mismatch)
    assert rep['status'] == 'fail'
    assert any('declared n_seeds' in p for p in rep['problems']), rep['problems']
    nonint = copy.deepcopy(man)
    nonint['statistics']['n_seeds'] = 'twelve'
    rep = L.budget_truncation_from_manifest(nonint)
    assert rep['status'] == 'fail'
    assert any('n_seeds' in p for p in rep['problems']), rep['problems']
    assert L.budget_truncation_from_manifest(man)['inputs'][
        'declared_n_seeds'] == len(man['seeds'])


def test_malformed_disposition_is_not_silently_defaulted():
    man = _manifest()
    # An explicit run_disposition artifact with no policy used to fall back to
    # the default retry cap; the declaration must instead fail closed.
    rep = L.budget_truncation_from_manifest(
        man, disposition={'kind': 'run_disposition'})
    assert rep['status'] == 'fail'
    assert rep['cost']['max_attempts_source'] == 'disposition_policy_invalid'
    assert rep['cost']['max_attempts_per_cell'] is None
    rep = L.budget_truncation_from_manifest(man, disposition='not-a-policy')
    assert rep['status'] == 'fail'
    assert rep['cost']['max_attempts_source'] == 'disposition_policy_invalid'
    # A valid partial declaration is merged over the defaults, not rejected.
    rep = L.budget_truncation_from_manifest(
        man, disposition={'max_attempts_per_cell': 3})
    assert rep['status'] == 'not_verifiable', rep['problems']
    assert rep['cost']['max_attempts_source'] == 'disposition_policy'
    assert rep['cost']['max_attempts_per_cell'] == 3


def test_scaled_panel_skips_the_unbounded_exact_dp():
    # 1000 seeds x 2 arms is far below the memory cap on the table width but
    # its convolution would be ~1.4e8 Python iterations.  The work guard must
    # skip it quickly instead of hanging, and must stay not_verifiable.
    seeds = list(range(1000))
    arms = ['a', 'b']
    cells = [{'label': f'{arm}::seed{seed}', 'variant': arm,
              'block': str(seed)}
             for seed in seeds for arm in arms]
    rep = L.build_budget_truncation(
        cells=cells,
        comparisons=[{'name': 'a-b', 'a': 'a', 'b': 'b'}],
        arms=arms, seeds=seeds, min_n=8, complete_coverage=True,
        max_attempts_per_cell=2, budget=2000,
        failure_model={'replaceable_per_attempt': 0.0001,
                       'non_replaceable_per_attempt': 0.0,
                       'max_attempts_per_cell': 2})
    assert rep['cost']['completion_risk_method'] == 'skipped_panel_too_large'
    assert rep['cost']['required_attempts_for_risk'] is None
    assert rep['status'] == 'not_verifiable', rep['status']
    assert rep['cost']['p_required_prefix_unlimited'] is not None
    # The private DP refuses an unbounded convolution even when called
    # directly (the state width alone is below the cap; the work is not).
    with pytest.raises(ValueError):
        L._bt_prefix_completion_distribution(
            2_000_000, 2, 0.0001, 0.0, 4_000_000)
