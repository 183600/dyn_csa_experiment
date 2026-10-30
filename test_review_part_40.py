"""Regression checks for the review-round scale/identity fixes.

These are zero-GPU checks.  They cover bugs that only appear on larger or
less tidy inputs: exact 64-bit object integers, pandas nullable scalars,
NumPy temporal scalars in ledger artifacts, half-precision attention with an
fp32 sink, the dedup temporary budget, and an unavailable seed-plan floor.
"""
import json
import math
import os
import sys as _sys
import tempfile

import numpy as np
import torch

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

import data_guard as DG
import _csa_kernels as K

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def test_object_int64_identity_is_exact_not_rounded_through_float():
    exact = 2 ** 53
    next_int = exact + 1
    table = np.array([[np.int64(next_int)], [np.int64(exact)]], dtype=object)
    # Distinct object integers must not become one row through float64.
    assert DG.duplicate_mask(table).tolist() == [False, False]
    obj_key = DG._row_keys(np.array([[np.int64(next_int)]], dtype=object))[0]
    native_key = DG._row_keys(np.array([[next_int]], dtype=np.int64))[0]
    assert obj_key == native_key
    # A rounded neighbour must not share the same key.
    assert obj_key != DG._row_keys(np.array([[exact]], dtype=np.int64))[0]


def test_pandas_nullable_missing_scalars_are_missing():
    pd = __import__('pytest').importorskip('pandas')
    assert DG._category_is_missing(pd.NA) is True
    assert DG._label_key(pd.NA) is None
    assert DG._target_key(pd.NA) == '__missing__'
    # All non-temporal missing scalars are one row identity.
    na_key = DG._row_keys(np.array([[pd.NA, 1.0]], dtype=object))[0]
    assert na_key == DG._row_keys(np.array([[None, 1.0]], dtype=object))[0]
    assert na_key == DG._row_keys(np.array([[np.nan, 1.0]], dtype=object))[0]
    # pd.NaT is a datetime subclass; it must not reach np.datetime64(pd.NaT).
    nat_obj = DG._row_keys(np.array([[pd.NaT]], dtype=object))[0]
    nat_np = DG._row_keys(np.array([[np.datetime64('NaT')]], dtype=object))[0]
    assert nat_obj == nat_np
    assert DG._target_key(pd.NaT) == '__missing__'
    # A nullable target column no longer crashes the target encoder.
    X = np.array([['a'], ['b'], ['c']], dtype=object)
    y = np.array([1.0, pd.NA, 2.0], dtype=object)
    out = DG.LeakageSafeTargetEncoder(n_folds=2).fit_transform(X, y)
    assert np.isfinite(out).all()
    enc = DG.LeakageSafeCategoricalEncoder().fit(
        np.array([[pd.NA], ['x']], dtype=object))
    assert enc.feature_stats_[0]['categories'] == ['x']
    # Missing group ids must share one identity; otherwise pd.NA/pd.NaT rows
    # form their own group and can be split across train/val/test.
    assert DG._group_key(pd.NA) == DG._group_key(None)
    assert DG._group_key(pd.NaT) == DG._group_key(np.datetime64('NaT'))
    groups = np.array([pd.NA, pd.NA, 'g2', 'g3'], dtype=object)
    split = DG.grouped_chronological_split_indices(
        4, groups, train_size=2, val_size=1, test_size=1)
    assert len(split['train']) == 2 and len(split['val']) == 1
    assert len(split['test']) == 1


def test_object_timedelta64_ns_keeps_temporal_identity():
    a = np.array([[np.timedelta64(1, 'ns')]], dtype=object)
    b = np.array([[np.timedelta64(1, 'ns')]])
    assert DG._row_keys(a)[0] == DG._row_keys(b)[0]
    big = 2 ** 53 + 1
    obj = DG._row_keys(np.array([[np.timedelta64(big, 'ns')]], dtype=object))[0]
    raw_int = DG._row_keys(np.array([[big]], dtype=np.int64))[0]
    assert obj != raw_int


def test_json_clean_serializes_temporal_objects_and_ledger_rows():
    assert L._json_clean(np.datetime64('2020-01-01')) == '2020-01-01'
    assert L._json_clean(np.timedelta64(2, 's')) == 2.0
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'experiments.jsonl')
        L.append_jsonl(path, {'kind': 'probe',
                               'when': np.datetime64('2020-01-01')})
        with open(path, encoding='utf-8') as fh:
            row = json.loads(fh.readline())
        assert row['when'] == '2020-01-01'
        assert row['_ledger']['seq'] == 0


def test_scale_ladder_missing_seed_plan_is_strict_json_safe():
    def tier(name):
        return {'name': name,
                'cfg': {'steps': 1, 'batch_size': 1, 'seq_len': 1,
                        'n_train_tokens': 1, 'lr': 3e-4, 'warmup': 0},
                'params': 10}
    out = L.scale_ladder_audit([tier('S1'), tier('S2')])
    json.dumps(out, allow_nan=False)
    for row in out['tiers']:
        assert row['p_floor'] is None
        assert row['descriptive_only'] is True
    assert out['common_seed_paired_p_floor'] is None


def test_run_disposition_uses_earliest_start_not_append_order():
    cells = ['a::seed0', 'b::seed0']

    def event(key, rid, started):
        return {'kind': 'cell', 'key': key, 'variant': key.split('::')[0],
                'seed': 0, 'status': 'ok', 'run_id': rid, 'run_cfg': 'c1',
                'started_at': started}

    # B finishes first (appended first) but A started earlier.
    events = [event('b::seed0', 'rb', '2026-01-01T00:03:00'),
              event('a::seed0', 'ra', '2026-01-01T00:01:00')]
    retained = {k: {'run_id': v, 'run_cfg': 'c1'}
                for k, v in (('a::seed0', 'ra'), ('b::seed0', 'rb'))}
    plan = L.build_run_disposition_plan(cells,
                                        created_at='2026-01-01T00:02:00')
    rep = L.run_disposition_audit(events, plan=plan, retained=retained)
    assert rep['first_attempt_at'] == '2026-01-01T00:01:00', rep
    assert rep['plan_before_first_attempt'] is False, rep
    assert rep['status'] == 'fail', rep


def test_maxt_null_uses_a_sign_invariant_score_scale():
    row = [1.0612690943, 1.9697997908, 0.2296875127, 1.2870695428,
           -0.1732428531, -0.3497827229, 1.0867127522, 1.6399534347]
    rep = L.maxt_signflip_adjust({'c': row})['contrasts']['c']
    expected_t = abs(sum(row)) / math.sqrt(sum(x * x for x in row))
    assert abs(rep['t_stat'] - expected_t) < 1e-12, rep
    # A single hypothesis is still the ordinary exact sign-flip test.
    assert rep['p_maxt'] == L.exact_signflip_p(row)['p']


def test_interaction_coverage_all_waived_is_not_fully_measured():
    rep = L.interaction_coverage_audit(
        design={'a': [], 'b': []}, factorials={}, scope=['a', 'b'],
        waivers={('a', 'b'): 'not studied'})
    assert rep['coverage_status'] == 'complete', rep
    assert rep['coverage_complete'] is True, rep
    assert rep['tested_pairs'] == [], rep
    assert rep['full_pairwise_measured'] is False, rep


def test_attention_half_precision_masks_invalid_lanes_like_the_kernel():
    T, Bm, nh, hd = 1, 1, 1, 2
    q = torch.tensor([[[1., 1.]]], dtype=torch.float16)
    kb = torch.tensor([[[-65504., -65504.]]], dtype=torch.float16)
    vb = torch.tensor([[[1., 2.]]], dtype=torch.float16)
    ks = torch.tensor([[[-65504., -65504.]]], dtype=torch.float16)
    vs = torch.tensor([[[3., 4.]]], dtype=torch.float16)
    idx = torch.zeros(T, 1, dtype=torch.long)
    lt = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        eager = L.gathered_attention(q, kb, vb, idx, lt, ks, vs, 1, 1.0)
        kernel = K.gathered_attention_kernel(
            q[None], kb[None], vb[None], idx[None], lt[None], ks[None],
            vs[None], 1, 1.0, impl='ref')
    assert eager.dtype == torch.float16
    assert torch.allclose(eager, kernel.reshape_as(eager)), (eager, kernel)
    assert torch.equal(eager, vs.reshape_as(eager))


def test_attention_fp32_sink_with_fp16_activations_does_not_mix_dtypes():
    T, Bm, nh, hd = 3, 2, 2, 4
    g = torch.Generator().manual_seed(1)
    q = torch.randn(T, nh, hd, generator=g, dtype=torch.float16)
    kb = torch.randn(Bm, nh, hd, generator=g, dtype=torch.float16)
    vb = torch.randn(Bm, nh, hd, generator=g, dtype=torch.float16)
    ks = torch.randn(T, nh, hd, generator=g, dtype=torch.float16)
    vs = torch.randn(T, nh, hd, generator=g, dtype=torch.float16)
    idx = torch.zeros(T, 1, dtype=torch.long)
    lt = torch.zeros(1, dtype=torch.long)
    sink = torch.randn(nh, generator=g, dtype=torch.float32)
    with torch.no_grad():
        eager = L.gathered_attention(q, kb, vb, idx, lt, ks, vs, 1, 1.0,
                                     sink_logits=sink)
        kernel = K.gathered_attention_kernel(
            q[None], kb[None], vb[None], idx[None], lt[None], ks[None],
            vs[None], 1, 1.0, sink_logits=sink, impl='ref')
    assert eager.dtype == torch.float16
    assert torch.allclose(eager, kernel.reshape_as(eager), atol=2e-3,
                          rtol=2e-3), (eager, kernel)


def test_full_width_gate_accounts_for_the_dedup_comparison_table():
    # k=32 -> the dedup comparison table is 32*32 bool bytes per row; this
    # shape formerly passed the 64 MiB gate and then allocated >170 MiB.
    assert L._attn_full_width_ok(150000, 32, 0, True) is False
    assert L._attn_full_width_ok(128, 8, 16, True) is True


def _prefix_fixture(dtype=torch.float64):
    T, Fd, cap = 13, 3, 16
    cuts = np.zeros(T - 1, dtype=bool)
    cuts[[1, 4, 8, 10]] = True
    bid = L.blocks_from_cuts(T, cuts, 2, 8, torch.device('cpu'))
    g = torch.Generator().manual_seed(7)
    Xa = torch.randn(T, Fd, generator=g, dtype=dtype)
    Xb = torch.randn(T, Fd, generator=g, dtype=dtype)
    Za = torch.randn(T, Fd, generator=g, dtype=dtype)
    Zb = torch.randn(T, Fd, generator=g, dtype=dtype)
    Ba = torch.randn(cap, Fd, generator=g, dtype=dtype)
    Bb = torch.randn(cap, Fd, generator=g, dtype=dtype)
    return bid, Xa, Xb, Za, Zb, Ba, Bb


def test_causal_prefix_pool_reference_keeps_fp64_precision():
    bid, Xa, Xb, Za, Zb, Ba, Bb = _prefix_fixture(torch.float64)
    nb = int(bid.max()) + 1
    out = L.causal_prefix_pool(Xa, Xb, Za, Zb, bid, Ba, Bb, 3,
                               n_blocks=nb, scope='full_prefix', window=4)
    ref = L.causal_prefix_pool_reference(Xa, Xb, Za, Zb, bid, Ba, Bb, 3,
                                         n_blocks=nb, scope='full_prefix',
                                         window=4)
    assert ref.dtype == torch.float64
    err = float((out - ref).abs().max())
    assert err < 1e-12, err
