"""Budget-truncation feasibility of the pre-registered cell panel (zero GPU).

The study freezes a complete 6 x 12 arm/seed grid and ``require_complete_
coverage``, but no gate relates the available cell-attempt budget and the
disposition retry cap to the shortest execution prefix that can still answer
the frozen family.  These checks exercise the exact deterministic frontier,
the exact truncated-geometric completion-risk calculation, the
resolution-floor/operational-floor distinction, the artifact replay, and the
audit/CLI wiring.

No training, no data split, no frozen p/Holm/verdict is changed.
"""
import copy
import inspect
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

REPO = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(REPO, 'confirmation_frozen.json')


def _manifest():
    with open(MANIFEST, encoding='utf-8') as fh:
        return json.load(fh)


def _toy_manifest(min_n=4, n_seeds=4):
    seeds = list(range(n_seeds))
    return {
        'manifest_version': 3,
        'arms': ['a_arm', 'b_arm'],
        'seeds': seeds,
        'primary_comparisons': [
            {'name': 'c1', 'a': 'a_arm', 'b': 'b_arm',
             'endpoint': 'by_len.ppl@4096'},
        ],
        'statistics': {
            'n_seeds': n_seeds, 'alpha': 0.05,
            'min_seeds_for_confirmation': min_n,
            'require_complete_coverage': True,
            'correction': 'holm step-down',
        },
    }


def _toy_plan(manifest):
    cells = [(arm, int(seed)) for seed in manifest['seeds']
             for arm in manifest['arms']]
    return L.build_run_order_plan(cells, block_key=lambda c: c[1], seed=0)


def _toy_artifact(manifest=None, **kwargs):
    manifest = manifest or _toy_manifest()
    kwargs.setdefault('plan', _toy_plan(manifest))
    kwargs.setdefault('plan_source', 'registered_run_order')
    kwargs.setdefault('budget', 8)
    kwargs.setdefault('failure_model', {
        'replaceable_per_attempt': 0.0,
        'non_replaceable_per_attempt': 0.0,
        'max_attempts_per_cell': 1,
    })
    return L.budget_truncation_from_manifest(manifest, **kwargs)


def _rehash(artifact):
    out = copy.deepcopy(artifact)
    out['content_sha256'] = L.budget_truncation_content_sha(out)
    return out


def _strip_cell(cells, arm, block):
    out = []
    for cell in cells:
        if cell['variant'] == arm and cell['block'] == block:
            continue
        out.append(cell)
    return out


def _cells(artifact):
    return copy.deepcopy(artifact['inputs']['cells'])


def test_frontier_is_exact_on_the_frozen_family():
    man = _manifest()
    a = L.budget_truncation_from_manifest(man)
    assert a['kind'] == 'budget_truncation'
    assert a['status'] == 'not_verifiable', a['status']
    assert a['problems'] == [], a['problems']
    assert a['design']['n_cells'] == 72
    assert a['design']['n_blocks'] == 12
    assert a['design']['n_arms'] == 6
    assert a['floor']['shortest_min_n_prefix_cells'] == 48
    assert a['floor']['shortest_complete_prefix_cells'] == 72
    assert a['cost']['planned_attempts'] == 72
    assert a['cost']['worst_case_attempts'] == 144
    assert a['cost']['max_attempts_per_cell'] == 2
    assert a['cost']['max_attempts_source'] == 'default_run_disposition_policy'
    assert a['budget']['declared_attempts'] is None
    assert a['frontier']['n_points'] == 72
    assert a['frontier']['sampled'] is False
    points = a['frontier']['points']
    assert [p['cells'] for p in points] == list(range(1, 73))
    last_complete = 0
    names = [c['name'] for c in a['design']['comparisons']]
    last_pairs = {name: 0 for name in names}
    for p in points:
        assert p['complete_seeds'] >= last_complete
        assert p['complete_seeds'] <= 12
        assert p['imbalance'] <= 1, p
        for name, n in p['pairs'].items():
            assert n >= last_pairs[name], (name, p['cells'])
            assert n <= 12
            last_pairs[name] = n
        last_complete = p['complete_seeds']
    at48 = points[47]
    assert at48['complete_seeds'] == 8
    assert at48['min_n_met'] is True
    assert at48['complete_coverage_met'] is False
    assert all(v == 8 for v in at48['pairs'].values())
    at72 = points[71]
    assert at72['complete_seeds'] == 12
    assert at72['complete_coverage_met'] is True
    assert all(v == 12 for v in at72['pairs'].values())
    assert L.validate_budget_truncation(a, manifest=man)['status'] == 'ok'


def test_resolution_floor_is_not_the_operational_floor():
    a = L.budget_truncation_from_manifest(_manifest())
    floor = a['floor']
    assert floor['declared_min_seeds'] == 8
    assert floor['planned_seeds'] == 12
    assert floor['effective_required_seeds'] == 12
    assert floor['effective_required_cells'] == 72
    assert floor['declared_floor_is_operational'] is False
    assert floor['single_cell_lethal'] is True
    assert any('resolution floor' in w for w in a['warnings']), a['warnings']
    assert any('72' in w for w in a['warnings'])


def test_missing_arm_is_structural():
    a = L.budget_truncation_from_manifest(_manifest())
    cells = [c for c in _cells(a) if c['variant'] != 'full_rope']
    bad = L.build_budget_truncation(
        cells=cells, comparisons=a['inputs']['comparisons'],
        arms=a['inputs']['arms'], seeds=a['inputs']['seeds'],
        min_n=8, complete_coverage=True, max_attempts_per_cell=2)
    assert bad['status'] == 'fail'
    assert any('never appears' in p for p in bad['problems']), bad['problems']


def test_complete_coverage_requires_every_arm_in_every_block():
    a = L.budget_truncation_from_manifest(_manifest())
    cells = _strip_cell(_cells(a), 'csa_fixed_rope', '3')
    bad = L.build_budget_truncation(
        cells=cells, comparisons=a['inputs']['comparisons'],
        arms=a['inputs']['arms'], seeds=a['inputs']['seeds'],
        min_n=8, complete_coverage=True, max_attempts_per_cell=2)
    assert bad['status'] == 'fail'
    assert any('missing block arms' in p for p in bad['problems']), bad['problems']
    assert bad['floor']['shortest_complete_prefix_cells'] is None


def test_budget_below_the_required_prefix_fails():
    a = L.budget_truncation_from_manifest(_manifest(), budget=60)
    assert a['status'] == 'fail'
    assert any('cannot complete the 72-cell required prefix' in p
               for p in a['problems']), a['problems']
    assert a['budget']['affordable_planned_cells'] == 60
    assert a['budget']['complete_seeds_at_budget'] == 10
    assert a['budget']['complete_coverage_met_at_budget'] is False


def test_budget_without_failure_model_is_not_green():
    a = L.budget_truncation_from_manifest(_manifest(), budget=72)
    assert a['status'] == 'warnings', a['status']
    assert a['problems'] == []
    assert any('retry reserve' in w for w in a['warnings']), a['warnings']
    assert a['budget']['affordable_planned_cells'] == 72
    assert a['budget']['complete_coverage_met_at_budget'] is True
    assert a['cost']['required_attempts_for_risk'] is None
    assert a['budget']['p_required_prefix_within_budget'] is None


def test_completion_risk_is_exact_and_minimal():
    fm = {'replaceable_per_attempt': 0.02,
          'non_replaceable_per_attempt': 0.0,
          'max_attempts_per_cell': 2}
    a = L.budget_truncation_from_manifest(
        _manifest(), budget=76, failure_model=fm, risk_level=0.05)
    need = a['cost']['required_attempts_for_risk']
    assert need == 76, need
    assert abs(a['cost']['expected_attempts_full_panel'] - 73.44) < 1e-9
    p_at = a['budget']['p_required_prefix_within_budget']
    assert p_at >= 0.95, p_at
    below = L.budget_truncation_from_manifest(
        _manifest(), budget=need - 1, failure_model=fm, risk_level=0.05)
    assert below['budget']['p_required_prefix_within_budget'] < 0.95
    assert below['status'] == 'fail'
    assert any('completion probability' in p for p in below['problems'])
    # The probability of an unlimited-budget complete panel is below 1:
    # non-replaceable failures are what retries cannot repair.
    assert 0.0 < a['cost']['p_full_panel_unlimited'] < 1.0
    assert a['cost']['p_required_prefix_unlimited'] == \
        a['cost']['p_full_panel_unlimited']
    assert a['status'] == 'warnings'  # feasible, with the floor caveat
    assert a['problems'] == []
    assert a['budget']['worst_case_covered'] is False


def test_non_replaceable_failures_cannot_be_repaired_by_budget():
    fm = {'replaceable_per_attempt': 0.05,
          'non_replaceable_per_attempt': 0.01,
          'max_attempts_per_cell': 2}
    a = L.budget_truncation_from_manifest(
        _manifest(), budget=144, failure_model=fm)
    assert a['status'] == 'fail'
    assert a['cost']['required_attempts_for_risk'] is None
    assert any('non-replaceable' in p for p in a['problems']), a['problems']


def test_worst_case_budget_is_green_on_a_clean_design():
    a = _toy_artifact()
    assert a['status'] == 'ok', (a['status'], a['problems'], a['warnings'])
    assert a['problems'] == [] and a['warnings'] == []
    assert a['design']['n_cells'] == 8
    assert a['cost']['required_prefix_cells'] == 8
    assert a['cost']['required_attempts_for_risk'] == 8
    assert a['budget']['p_required_prefix_within_budget'] == 1.0
    # The registered disposal cap is 2 attempts per cell even though the
    # declared model retries at most once; 8 attempts does not cover 8 * 2.
    assert a['cost']['max_attempts_per_cell'] == 2
    assert a['cost']['worst_case_attempts'] == 16
    assert a['budget']['worst_case_covered'] is False
    assert L.budget_truncation_summary(a)['ok'] is True
    assert L.budget_truncation_summary(a)['required_prefix_cells'] == 8
    assert L.validate_budget_truncation(a, manifest=_toy_manifest())['status'] \
        == 'ok'


def test_replay_rejects_every_tampered_section():
    a = _toy_artifact()
    tampered = []
    bad = copy.deepcopy(a)
    name = bad['design']['comparisons'][0]['name']
    bad['frontier']['points'][3]['pairs'][name] += 1
    tampered.append(bad)
    bad = copy.deepcopy(a)
    bad['cost']['required_attempts_for_risk'] += 1
    tampered.append(bad)
    bad = copy.deepcopy(a)
    bad['budget']['p_required_prefix_within_budget'] = 0.1
    tampered.append(bad)
    bad = copy.deepcopy(a)
    bad['status'] = 'fail'
    tampered.append(bad)
    bad = copy.deepcopy(a)
    bad['floor']['effective_required_cells'] = 1
    tampered.append(bad)
    for bad in tampered:
        verdict = L.validate_budget_truncation(_rehash(bad))
        assert verdict['status'] == 'fail', (verdict['status'], bad)
        assert verdict['problems'], bad
    raw = copy.deepcopy(a)
    raw['frontier']['points'][0]['complete_seeds'] = 99
    assert L.validate_budget_truncation(raw)['status'] == 'fail'


def test_manifest_binding_rejects_a_different_freeze():
    a = _toy_artifact()
    other = _toy_manifest()
    other['seeds'] = [0, 1, 2, 3, 4]
    verdict = L.validate_budget_truncation(a, manifest=other)
    assert verdict['status'] == 'fail'
    assert any('different manifest' in p for p in verdict['problems'])


def test_writer_is_append_only_unless_overwritten():
    a = _toy_artifact()
    with tempfile.TemporaryDirectory() as td:
        first = L.write_budget_truncation(td, a)
        assert first['written'] is True and first['status'] == 'ok'
        again = L.write_budget_truncation(td, a)
        assert again['unchanged'] is True and again['written'] is False
        other = _toy_artifact(budget=9)
        conflict = L.write_budget_truncation(td, other)
        assert conflict['conflict'] is True and conflict['written'] is False
        replaced = L.write_budget_truncation(td, other, overwrite=True)
        assert replaced['written'] is True
        with open(os.path.join(td, 'budget_truncation.json'),
                  encoding='utf-8') as fh:
            stored = json.load(fh)
        assert stored['content_sha256'] == other['content_sha256']


def test_gate_validates_stored_artifacts_and_rejects_tampering():
    a = _toy_artifact()
    with tempfile.TemporaryDirectory() as td:
        mpath = os.path.join(td, 'manifest.json')
        with open(mpath, 'w', encoding='utf-8') as fh:
            json.dump(_toy_manifest(), fh)
        path = os.path.join(td, 'budget_truncation.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(a, fh)
        good = A.audit_budget_truncation([path], manifest_path=mpath)
        assert good['status'] == 'ok', good
        assert good['artifacts'][path]['artifact_status'] == 'ok'
        # Embedded in a plan snapshot (the confirmation-plan location).
        snap = os.path.join(td, 'plan_snapshot.json')
        with open(snap, 'w', encoding='utf-8') as fh:
            json.dump({'plan': {'budget_truncation': a}}, fh)
        embedded = A.audit_budget_truncation([snap], manifest_path=mpath)
        assert embedded['status'] == 'ok', embedded
        # A tampered frontier must fail, not be reported as unverifiable.
        bad = copy.deepcopy(a)
        bad['frontier']['points'][0]['complete_seeds'] = 99
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(bad, fh)
        broken = A.audit_budget_truncation([path], manifest_path=mpath)
        assert broken['status'] == 'fail', broken


def test_gate_reports_missing_and_undeclared_budgets_as_not_verifiable():
    missing = A.audit_budget_truncation([os.path.join(tempfile.gettempdir(),
                                                      'no_such_budget.json')])
    assert missing['status'] == 'not_verifiable', missing
    live = L.budget_truncation_from_manifest(_manifest())
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'budget_truncation.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(live, fh)
        gate = A.audit_budget_truncation([path])
        assert gate['status'] == 'not_verifiable', gate
        assert gate['artifacts'][path]['artifact_status'] == 'not_verifiable'


def test_confirmation_plan_embeds_budget_truncation():
    rep = V12.plan_check(MANIFEST)
    bt = rep.get('budget_truncation')
    assert isinstance(bt, dict)
    assert bt['kind'] == 'budget_truncation'
    assert bt['status'] == 'not_verifiable'
    assert bt['floor']['effective_required_cells'] == 72
    assert any('budget truncation' in n for n in rep['notes']), rep['notes']
    assert rep['ok'] is True


def test_wiring_in_report_makefile_and_plan():
    src = inspect.getsource(A.run_report)
    assert "'budget_truncation': audit_budget_truncation()" in src
    assert "budget_truncation" in inspect.getsource(V12.plan_check)
    with open(os.path.join(REPO, 'Makefile'), encoding='utf-8') as fh:
        makefile = fh.read()
    assert 'budget_truncation' in makefile
    assert '\nbudget:' in makefile
