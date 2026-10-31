"""Regressions for scale/overflow and fail-closed boundary defects.

These checks were added after a focused audit of paths that only fail on
untrusted JSON, extreme finite values or a large declared family.  Each test
asserts the *reported* outcome (not merely that a function returns), because a
strict-JSON report or a launch gate is only useful when its named status is
correct.
"""
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def test_contrast_network_extreme_ppl_log_ratio_is_finite_and_json_safe():
    base = {
        'with': 'W', 'without': 'WO', 'controlled': True,
        'fully_paired': True, 'seeds': [0, 1], 'component': 'c1',
        'kind': 'component',
    }
    for with_ppl, without_ppl in ((1e-308, 1e308), (1e308, 1e-308)):
        entry = dict(base, per_seed_with=[with_ppl] * 2,
                     per_seed_without=[without_ppl] * 2)
        out = L.contrast_network_audit([entry])
        edge = out['edges']['WO -> W']
        assert math.isfinite(edge['log_ratio_mean']), (with_ppl, without_ppl, edge)
        json.dumps(out, allow_nan=False)


def test_confirmatory_access_huge_read_counter_fails_closed():
    for huge in (10 ** 400, 10 ** 5000, -(10 ** 5000)):
        lock = {'stage': 'analysis', 'test_reads': huge,
                'test_ids_fp': 'abc', 'history': []}
        stats = {'alpha': 0.05,
                 'comparisons': {'A vs B': {'p_holm': 0.01}},
                 'confirmed': []}
        out = L.confirmatory_access_audit(lock, stats=stats)
        assert out['status'] == 'fail', out
        assert out['confirmatory_allowed'] is False, out
        assert out['n_test_reads'] is None, out
        json.dumps(out, allow_nan=False)
    assert L._ta_same(10 ** 400, 10 ** 400) is True
    assert L._ta_same(10 ** 400, 10 ** 400 + 1) is False


def test_preparation_clearance_conflicting_read_evidence_is_conservative():
    protocol = {'study': {'level': 'confirmatory'}}
    with tempfile.TemporaryDirectory() as td:
        lock = os.path.join(td, '.freeze_lock.json')
        with open(lock, 'w', encoding='utf-8') as fh:
            fh.write('{}')
        # A positive flag / existing lock must not be erased by a reset
        # counter or an explicit false flag.
        for state in (
                {'path': lock, 'test_read': True, 'test_reads': 0},
                {'path': lock, 'test_read': False},
                {'test_read': True, 'test_reads': 0}):
            out = L._prep_test_access_check(protocol, state)
            assert out['status'] == 'fail', (state, out)
            assert out['read'] is True, (state, out)
        clean = L._prep_test_access_check(
            protocol, {'test_read': False, 'test_reads': 0})
        assert clean['status'] == 'ok', clean
        assert clean['read'] is False, clean


def test_analysis_rehearsal_malformed_checks_is_named_failure():
    body = {'kind': 'analysis_rehearsal',
            'scenarios': {'s': {'ok': True, 'checks': 5}}}
    out = L.validate_analysis_rehearsal(body)
    assert out['status'] == 'fail', out
    assert any('checks' in p for p in out['problems']), out
    json.dumps(out, allow_nan=False)


def test_disposition_cell_validation_stays_strict_at_scale():
    cells = [f'arm{i % 97}::seed{i}' for i in range(20000)]
    assert L._dispo_cells(cells) == cells
    import pytest
    with pytest.raises(ValueError, match='duplicate'):
        L._dispo_cells(cells + [cells[-1]])


def test_cleaning_sensitivity_missing_validation_sidecar_is_not_ok():
    import test_review_part_73 as T73
    rows = T73._rows(6)
    val = T73._rows(2, prefix='val')
    protocol = T73._protocol(shingle_words=3, max_shingles=32, bands=4,
                             near_hamming_threshold=2)
    expected_train, _expected_val = T73._reference(
        rows, val, protocol=protocol, fit_rows=1)
    out = L.cleaning_sensitivity(
        rows, val, protocol=protocol, fit_rows=1,
        expected_train_doc_ids=expected_train,
        expected_validation_doc_ids=None, n_order_permutations=0)
    assert out['baseline_match']['validation']['match'] is None, out
    assert out['status'] == 'warnings', out
    json.dumps(out, allow_nan=False)


def test_cleaning_sensitivity_rejects_unbounded_order_work():
    import pytest
    import test_review_part_73 as T73
    protocol = T73._protocol(shingle_words=3, max_shingles=32, bands=4,
                             near_hamming_threshold=2)
    with pytest.raises(ValueError, match='n_order_permutations'):
        L.cleaning_sensitivity(['A B C'], [], protocol=protocol, fit_rows=0,
                               n_order_permutations=(
                                   L.CLEANING_SENSITIVITY_MAX_ORDER_PERMUTATIONS
                                   + 1))


def test_look_number_recovers_from_corrupt_entries(monkeypatch):
    monkeypatch.setattr(L, 'panel_look_snapshot',
                        lambda *args, **kwargs: {'a vs b': {'ok': True}})
    with tempfile.TemporaryDirectory() as td:
        for bad in (-3, 0, 'not-a-number', {'nested': 1}):
            history = {'schema': L.LOOK_HISTORY_SCHEMA,
                       'looks': [{'look': bad, 'contrasts': {'old': {}}}]}
            with open(os.path.join(td, 'look_history.json'), 'w',
                      encoding='utf-8') as fh:
                json.dump(history, fh)
            hist = L.record_panel_look(td, [], now='2026-01-01T00:00:00')
            assert hist['looks'][-1]['look'] >= 1, (bad, hist)
            assert hist.get('look_number_recovered') is True, (bad, hist)
            json.dumps(hist, allow_nan=False)


def test_ablation_practical_margin_ratio_overflow_is_null():
    out = L.ablation_practical_decision(
        [1e-300] * 5, margin=1e300, baseline_ppl=1e-300)
    assert out['relative_margin_pct'] is None, out
    json.dumps(out, allow_nan=False)


def test_family_correction_missing_mean_p_is_unavailable_not_different():
    seeds = [0, 1, 2, 3, 4, 5]
    comps = {
        'a': {'per_seed': [.2, .3, .25, .22, .31, .29],
              'seeds': seeds, 'p_exact_signflip': 0.03125},
        'b': {'per_seed': [.1, .12, .11, .09, .13, .1],
              'seeds': seeds},
    }
    out = L.confirmation_family_correction(comps, family_names=['a', 'b'])
    assert out['status'] == 'ok', out
    assert out['summary']['statistic_effect'] is None, out['summary']
    assert out['summary']['mean_rejections_available'] is False, out['summary']
    assert out['sets']['score_vs_mean_changed'] == [], out['sets']
    json.dumps(out, allow_nan=False)


def test_metric_policy_huge_support_counts_do_not_overflow():
    huge = 10 ** 400
    rec = {'variant': 'm', 'task': 'multiclass_classification', 'acc': 0.5,
           'per_class': {'a': {'pr_auc': 0.5, 'roc_auc': 0.5,
                               'support': huge}},
           'n': huge}
    out = L.metric_policy_audit({'k': rec})
    assert isinstance(out, dict), out
    json.dumps(out, allow_nan=False)


def test_scale_ladder_huge_integer_is_refused_before_json():
    huge = 10 ** 4500

    def tier(name, tokens):
        return {'name': name,
                'cfg': {'steps': tokens, 'batch_size': 1, 'seq_len': 1,
                        'n_train_tokens': tokens, 'lr': 1e-3, 'warmup': 0,
                        'seeds': [0, 1]},
                'params': 1}
    out = L.scale_ladder_audit([tier('S1', huge // 2), tier('S2', huge)])
    assert out['status'] == 'fail', out
    assert any('reportable integer bound' in p for p in out['problems']), out
    json.dumps(out, allow_nan=False)


def test_meta_analysis_invalid_scale_is_null_not_nan():
    pts = [{'label': 'a', 'scale': 'bad', 'effects': [1.0, 2.0, 3.0],
            'seeds': [0, 1, 2]},
           {'label': 'b', 'scale': 10.0, 'effects': [1.0, 2.0, 3.0],
            'seeds': [0, 1, 2]},
           {'label': 'c', 'scale': 100.0, 'effects': [1.0, 2.0, 3.0],
            'seeds': [0, 1, 2]}]
    out = L.meta_analyze_effects(pts)
    assert out['invalid_scale_points'] == ['a'], out
    assert out['per_point'][0]['scale'] is None, out['per_point'][0]
    json.dumps(out, allow_nan=False)
    trend = L.scaling_trend(pts)
    assert None in trend['declared_scales'], trend['declared_scales']
    json.dumps(trend, allow_nan=False)


def test_scaling_trend_pair_detail_is_bounded_but_verdict_is_not():
    pts = [{'label': f'S{i}', 'scale': 2.0 ** i,
            'effects': [1.0, 2.0, 3.0], 'seeds': [0, 1, 2]}
           for i in range(60)]
    out = L.scaling_trend(pts, pair_detail_limit=10)
    overlap = out['seed_overlap']
    assert len(overlap['per_pair_common_seeds']) == 10, overlap
    assert overlap['n_pairs'] == 60 * 59 // 2, overlap
    assert overlap['per_pair_common_seeds_truncated'] is True, overlap
    assert overlap['any_pair_overlap'] is True, overlap
    json.dumps(out, allow_nan=False)


def test_attrition_power_huge_n_is_refused_not_overflowed():
    out = L.attrition_power_audit(
        10 ** 5000, 1, q=0.8, pair_retention=0.95)
    assert out['status'] == 'fail', out
    assert out['failure_reason'] == 'scan_cap', out
    json.dumps(out, allow_nan=False)


def test_pooled_ppl_non_dict_record_is_not_estimable():
    assert L._pooled_ppl_from_records([None]) == (None, None, None)
    assert L._pooled_ppl_from_records(
        [{'nll_sum': 1.0, 'n_tokens': 2}, None]) == (None, None, None)
