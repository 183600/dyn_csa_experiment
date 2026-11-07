"""Regressions for extreme-input paths found by the scale audit.

Every case below uses a finite Python integer that has no float64 spelling
(``10**400``) or one beyond CPython's decimal-conversion limit (``10**5000``).
None of these values is a malformed measurement: the helpers must keep their
documented "unavailable" value, decide the exact result, or skip the row --
never leak ``OverflowError`` (or an integer-conversion ``ValueError``) from a
diagnostic path.
"""
import json
import math
import os
import tempfile

import numpy as np
import pytest

import exp_lib as L


HUGE = 10 ** 400
UNPRINTABLE = 10 ** 5000


def test_nll_doc_ci_keeps_exact_integer_documents():
    got = L._nll_doc_ci([HUGE, -HUGE], [1, 1])
    # The two exact NLLs cancel; the pooled PPL is exp(0).
    assert got['n_docs'] == 2
    assert got['ppl'] == 1.0
    json.dumps(got, allow_nan=False)

    mixed = L._nll_doc_ci([HUGE, 1.0], [HUGE, 1])
    assert mixed['n_docs'] == 2
    assert mixed['ppl'] == pytest.approx(math.e)


def test_nll_row_ce_accepts_an_empty_row_set():
    assert L._nll_row_ce([], 1).tolist() == []
    # An explicit empty matrix still has two (zero-token) rows.
    assert L._nll_row_ce([[], []], 1).tolist() == [0.0, 0.0]


def test_training_health_accepts_an_exact_integer_tolerance():
    curve = [{'step': 0, 'train_ce': 1.0, 'val_ce': 1.0},
             {'step': 1, 'train_ce': 1.0, 'val_ce': 1.0}]
    got = L.classify_training_health(curve, slope_tol=HUGE)
    assert got['slope_tol_mode'] == 'fixed'
    assert got['train_slope_tol'] == float('inf')


def test_rehearsal_direction_decides_an_exact_integer_panel():
    assert L._ar_direction_ok([HUGE, HUGE + 1], 'positive') is True
    assert L._ar_direction_ok([-HUGE, -HUGE - 1], 'positive') is False
    assert L._ar_direction_ok([HUGE, -HUGE], 'positive') is False
    assert L._ar_direction_ok([1.0, 2.0], 'positive') is True


def test_length_position_helpers_withhold_unrepresentable_integers():
    stats = L._lpd_stats([HUGE, -HUGE], alpha=0.05, min_seeds=2,
                         seeds=[0, 1])
    assert stats['n'] == 2 and stats['status'] == 'not_estimable'
    assert stats['mean'] is None and stats['std'] is None
    json.dumps(stats, allow_nan=False)

    rows = [
        {'label': 'a', 'per_seed': [{'seed': 0, 'log_ratio': HUGE},
                                    {'seed': 1, 'log_ratio': 1.0}]},
        {'label': 'b', 'per_seed': [{'seed': 0, 'log_ratio': -HUGE},
                                    {'seed': 1, 'log_ratio': 2.0}]},
    ]
    got = L._lpd_max_band_signflip(rows, alpha=0.05)
    assert got['status'] == 'ok'
    json.dumps(got, allow_nan=False)


def test_length_strata_refuses_an_underflowing_reference_share():
    strata = [{'n_tokens': 1}, {'n_tokens': 1}]
    with pytest.raises(ValueError, match='positive floats'):
        L._ls_reference_weights([1e-300, 1e300], strata, 2)
    # Ordinary weights keep the historical normalisation.
    kind, weights = L._ls_reference_weights([1.0, 3.0], strata, 2)
    assert kind == 'custom' and weights == [0.25, 0.75]


def test_unit_precision_design_survives_an_exact_integer_count():
    got = L.unit_precision_design(HUGE, pilot_unit_sd=1.0)
    assert got['status'] in ('ok', 'not_estimable')
    assert got['n_units'] == HUGE
    json.dumps(got, allow_nan=False)


def test_long_aggregate_merge_skips_an_unrepresentable_secondary_metric():
    import run_gapfill as G

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'aggregate.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump({'v': {'seeds': [0, 1], 'ppls': [1.0, 1.1],
                             'avg_block_len': HUGE}}, fh)
        G.merge_long_aggregate(
            td, {'v': {'seeds': [0, 1], 'ppls': [1.0, 1.1],
                       'avg_block_len': 5.0}})
        with open(path, encoding='utf-8') as fh:
            merged = json.load(fh)
        assert merged['v']['avg_block_len'] == 5.0


def test_fuse_helpers_delegate_extreme_deltas_instead_of_precasting():
    import fuse_analysis as FA

    assert FA.signflip([HUGE, 1.0]) == 1.0
    assert FA.sample_std([HUGE, HUGE]) == 0.0
    assert math.isnan(FA.sample_std([HUGE, 1.0]))


def test_error_report_formatting_withholds_unrepresentable_numbers():
    import error_analysis as EA

    assert EA._ci_text([HUGE, 1.0]) == 'n/a'
    assert EA._num_text(HUGE) == 'n/a'
    assert EA._num_text(float('nan')) == 'n/a'
    assert EA._num_text(1.25) == '+1.250'


def test_arm_rank_validator_fails_closed_on_a_huge_probability():
    import audit_gates as A

    ok, why = A._aru_axis_coherent(
        {'p_best': {'a': HUGE}, 'rank_ci95': {'a': [1, 1]}},
        arms=['a'], key='axis')
    assert ok is False and 'not a probability' in why


def test_flops_report_skips_an_unprintable_sequence_length(monkeypatch):
    import _build_report as BR

    monkeypatch.setattr(
        BR, 'load',
        lambda path: {'rows': [{'seq_len': HUGE}],
                      'crossover_seq_len_csa_beats_dense': None})
    out = BR.sec_flops()
    assert out.startswith('## 解析 FLOPs')


def test_huge_integer_labels_do_not_abort_public_audits():
    # A finite component/arm label is a display key; it must not crash the
    # audit merely because CPython cannot decimalise it.
    ep = L.endpoint_headroom('by_len.ppl@4096',
                             arm_levels={UNPRINTABLE: 10.0, 'b': 11.0})
    assert isinstance(ep, dict)
    cps = L.cross_panel_synthesis({UNPRINTABLE: {}})
    assert isinstance(cps, dict)
    cn = L.contrast_network_audit(
        [{'with': 'a', 'without': 'b', 'controlled': True,
          'fully_paired': True, 'seeds': [0, 1, 2],
          'per_seed_with': [1.0, 1.1, 1.2],
          'per_seed_without': [1.2, 1.3, 1.4],
          'deltas': [-0.2, -0.2, -0.2], 'component': UNPRINTABLE}],
        min_n=3)
    assert isinstance(cn, dict)

    # The design compiler treats an unprintable manifest arm as a named input
    # error, not as a raw CPython integer-conversion failure.
    with pytest.raises(ValueError, match='too large'):
        L.design_estimability_from_manifest(
            {'arms': [UNPRINTABLE, 'b'], 'seeds': [0, 1]})
