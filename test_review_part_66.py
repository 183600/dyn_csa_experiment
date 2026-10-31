"""CPU review checks (split group 66): multi-arm contrast composability.

A per-pair audit can pass while the table as a whole is not one experiment:
the same arm label may be measured in two different runs, or two verified
components may sit in disconnected graphs with no common reference.  These
checks cover ``contrast_network_audit`` and the replay gate
``audit_gates.contrast_network``.
"""
import copy as _copy
import json as _json
import os as _os
import sys as _sys
import tempfile as _tempfile

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def _entry(with_arm, without_arm, seeds, with_vals, without_vals, comp,
           *, kind='component', tag_with='', tag_without='', include_ppl=True):
    row = {
        'with': with_arm, 'without': without_arm,
        'controlled': True, 'fully_paired': True,
        'component': comp, 'kind': kind,
        'tag_with': tag_with, 'tag_without': tag_without,
        'seeds': list(seeds),
        'deltas': [w - wo for w, wo in zip(with_vals, without_vals)],
    }
    if include_ppl:
        row['per_seed_with'] = list(with_vals)
        row['per_seed_without'] = list(without_vals)
    return row


def _chain(*, a=45.0, b=50.0, c=52.0):
    seeds = [0, 1, 2]
    return [
        _entry('a', 'b', seeds, [a] * 3, [b] * 3, 'ab'),
        _entry('b', 'c', seeds, [b] * 3, [c] * 3, 'bc'),
    ]


def test_network_chain_is_one_connected_common_reference_graph():
    r = L.contrast_network_audit(_chain())
    assert r['status'] == 'ok', r
    assert r['n_edges'] == 2 and r['n_nodes'] == 3 and r['n_components'] == 1, r
    assert r['cross_component_ranking_allowed'] is True, r['ranking_reasons']
    assert r['reference']['arm'] in ('a', 'b', 'c'), r['reference']
    assert r['reference']['missing_declared_components'] == [], r['reference']
    assert r['reference']['seed_support'] == [0, 1, 2], r['reference']
    assert r['closure']['status'] == 'not_applicable', r['closure']
    assert set(r['edges']) == {'b -> a', 'c -> b'}, r['edges']
    assert abs(r['edges']['b -> a']['log_ratio_mean']
               - __import__('math').log(45.0 / 50.0)) < 1e-12, r['edges']
    assert r['declared_components']['ab']['connected'] is True, r['declared_components']


def test_network_names_a_split_arm_instead_of_averaging_it():
    # B is 50.0 in pair (a,b) and 50.5 in pair (b,c): the pair gates would each
    # pass in isolation, but the label refers to two measurements.
    rows = [
        _entry('a', 'b', [0, 1, 2], [45.0] * 3, [50.0] * 3, 'ab'),
        _entry('b', 'c', [0, 1, 2], [50.5] * 3, [52.0] * 3, 'bc'),
    ]
    r = L.contrast_network_audit(rows)
    assert r['status'] == 'warn', r
    assert r['n_edges'] == 0, r
    assert len(r['node_conflicts']) == 1, r['node_conflicts']
    conflict = r['node_conflicts'][0]
    assert conflict['node'] == 'b' and len(conflict['seeds']) == 3, conflict
    assert conflict['max_relative_difference'] > 0.0, conflict
    assert r['cross_component_ranking_allowed'] is False, r
    assert any('split arm node' in reason for reason in r['ranking_reasons']), r


def test_network_disconnected_components_withhold_ranking():
    rows = [
        _entry('a', 'b', [0, 1, 2], [45.0] * 3, [50.0] * 3, 'ab'),
        _entry('c', 'd', [0, 1, 2], [52.0] * 3, [55.0] * 3, 'cd'),
    ]
    r = L.contrast_network_audit(rows)
    assert r['status'] == 'warn' and r['n_components'] == 2, r
    assert r['cross_component_ranking_allowed'] is False, r
    assert any('disconnected' in reason for reason in r['ranking_reasons']), r
    # Two declared components are usually connected through a shared control;
    # a disconnected graph is therefore a design/data problem, not a pass.
    assert any('disconnected' in w for w in r['warnings']), r['warnings']


def test_network_excludes_a_stale_delta_by_name():
    rows = _chain()
    rows[0]['deltas'] = [0.0, 0.0, 0.0]  # disagrees with 45 - 50
    r = L.contrast_network_audit(rows)
    assert r['n_edges'] == 1, r
    assert r['ignored_entries'] and 'stored delta disagrees' in r['ignored_entries'][0]['reason'], r
    assert r['status'] == 'warn', r


def test_network_requires_per_seed_arm_ppls_not_just_the_delta():
    rows = _chain()
    for row in rows:
        row.pop('per_seed_with', None)
        row.pop('per_seed_without', None)
    r = L.contrast_network_audit(rows)
    assert r['status'] == 'not_estimable', r
    assert r['n_edges'] == 0 and r['indirect_contrasts'] == {}, r
    assert any('per-seed arm PPL' in e['reason'] for e in r['ignored_entries']), r


def test_network_duplicate_edge_with_different_values_is_a_conflict():
    seeds = [0, 1, 2]
    rows = [
        _entry('a', 'b', seeds, [45.0] * 3, [50.0] * 3, 'ab'),
        _entry('a', 'b', seeds, [44.0] * 3, [50.0] * 3, 'ab2'),
    ]
    r = L.contrast_network_audit(rows)
    assert r['status'] == 'warn', r
    assert len(r['conflicting_edges']) == 1, r['conflicting_edges']
    assert r['n_edges'] == 0 and r['n_nodes'] == 0, r


def test_network_cycle_closure_is_an_identity_not_an_interaction():
    seeds = [0, 1, 2]
    rows = _chain() + [
        _entry('a', 'c', seeds, [45.0] * 3, [52.0] * 3, 'ac'),
    ]
    r = L.contrast_network_audit(rows)
    assert r['status'] == 'ok', r
    assert r['closure']['status'] == 'coherent', r['closure']
    assert r['closure']['n_cycles'] == 1, r['closure']
    assert r['closure']['max_abs_residual'] <= r['closure']['tolerance'], r['closure']
    # A cycle residual is not a 2x2 interaction: the block explicitly labels
    # it as a measurement identity check, and it must not create a p-value.
    assert 'interaction' in r['policy']['closure_note'], r['policy']


def test_network_derived_path_contrast_is_descriptive_only():
    r = L.contrast_network_audit(_chain())
    assert list(r['indirect_contrasts']) == ['a -> c'], r['indirect_contrasts']
    row = r['indirect_contrasts']['a -> c']
    assert row['source'] == 'derived_path' and row['claim_allowed'] is False, row
    assert row['path'] == ['c -> b', 'b -> a'], row
    # The row is log(PPL_a / PPL_c): a is 45 and c is 52, so it must be
    # negative (lower PPL for `from`), not the reciprocal +15.6%.
    assert abs(row['relative_effect_pct'] - (100.0 * (45.0 / 52.0 - 1.0))) < 1e-12, row
    assert row['log_ratio_mean'] < 0.0, row
    assert 'p_exact_signflip' in row, row
    assert all(k not in row for k in ('p_holm', 'verified', 'confirmed')), row


def test_network_does_not_mutate_entries_or_create_holm_fields():
    rows = _chain()
    before = _copy.deepcopy(rows)
    r = L.contrast_network_audit(rows)
    assert rows == before, 'the audit must not mutate the caller entries'
    assert r['status'] == 'ok', r
    for edge in r['edges'].values():
        assert 'p_holm' not in edge and 'verified' not in edge, edge
    for row in r['indirect_contrasts'].values():
        assert 'p_holm' not in row and 'verified' not in row, row


def test_network_is_attached_to_methodology_audit_without_changing_p_holm():
    fp = ('steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_'
          'dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX')
    summary = {}
    for i in range(4):
        for arm, ppl in (('a', 45.0), ('b', 50.0), ('c', 52.0)):
            summary['%s::seed%d' % (arm, i)] = dict(
                variant=arm, seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                params=100, ppl=ppl)
    design = {'ab': [('a', 'b', 'A vs B')], 'bc': [('b', 'c', 'B vs C')]}
    report = L.methodology_audit(summary, design=design, baselines={},
                                 verbose=False)
    block = report.get('contrast_network')
    assert isinstance(block, dict) and block['status'] == 'ok', block
    assert block['cross_component_ranking_allowed'] is True, block
    assert report['components']['ab'][0]['p_holm_global'] is not None, report
    # The block is a readout, not a member of the declared Holm family.
    assert report['family']['declared_contrasts'] == 2, report['family']


def _write_report(td, report):
    p = _os.path.join(td, 'methodology_audit.json')
    with open(p, 'w', encoding='utf-8') as f:
        _json.dump(report, f, allow_nan=False)
    return p


def test_network_connected_by_an_arm_but_disjoint_seeds_cannot_rank_paired():
    # The graph is connected through B, but the two component effects live on
    # disjoint seed draws.  A point-estimate table may still show them side by
    # side; a PAIRED cross-component ranking is not identified.
    rows = [
        _entry('a', 'b', [0, 1, 2], [45.0] * 3, [50.0] * 3, 'ab'),
        _entry('b', 'c', [3, 4, 5], [50.0] * 3, [52.0] * 3, 'bc'),
    ]
    r = L.contrast_network_audit(rows)
    assert r['status'] == 'ok' and r['n_components'] == 1, r
    assert r['node_conflicts'] == [], r['node_conflicts']
    assert r['reference']['seed_support'] == [], r['reference']
    assert r['cross_component_ranking_allowed'] is False, r
    assert any('common' in reason and 'seed' in reason
               for reason in r['ranking_reasons']), r['ranking_reasons']


def test_contrast_network_gate_replays_and_rejects_tampering():
    fp = ('steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_'
          'dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX')
    summary = {}
    for i in range(4):
        for arm, ppl in (('a', 45.0), ('b', 50.0), ('c', 52.0)):
            summary['%s::seed%d' % (arm, i)] = dict(
                variant=arm, seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                params=100, ppl=ppl)
    design = {'ab': [('a', 'b', 'A vs B')], 'bc': [('b', 'c', 'B vs C')]}
    with _tempfile.TemporaryDirectory() as td:
        L.methodology_audit(summary, design=design, baselines={},
                            outdir=td, verbose=False)
        p = _os.path.join(td, 'methodology_audit.json')
        ok = A.audit_contrast_network(p)
        assert ok['status'] == 'ok', ok
        assert ok['violations'] == [], ok
        raw = _json.load(open(p, encoding='utf-8'))
        # A different stored ranking verdict must not survive replay.
        raw['contrast_network']['cross_component_ranking_allowed'] = False
        _json.dump(raw, open(p, 'w', encoding='utf-8'), allow_nan=False)
        bad = A.audit_contrast_network(p)
        assert bad['status'] == 'fail', bad
        assert any('cross_component_ranking_allowed' in v for v in bad['violations']), bad
        # A decision key cannot be smuggled into the descriptive block.
        raw['contrast_network']['cross_component_ranking_allowed'] = True
        raw['contrast_network']['verdict'] = 'confirmed'
        _json.dump(raw, open(p, 'w', encoding='utf-8'), allow_nan=False)
        bad2 = A.audit_contrast_network(p)
        assert bad2['status'] == 'fail', bad2
        assert any('decision keys' in v for v in bad2['violations']), bad2
        # Missing block is honestly unverifiable, not a pass.
        raw2 = _json.load(open(p, encoding='utf-8'))
        del raw2['contrast_network']
        _json.dump(raw2, open(p, 'w', encoding='utf-8'), allow_nan=False)
        missing = A.audit_contrast_network(p)
        assert missing['status'] == 'not_verifiable_locally', missing
