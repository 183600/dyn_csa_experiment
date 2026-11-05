"""Regression checks for descriptive/protocol single-factor contrasts.

Two experiments were reported but never verified at the definition site:

* the descriptive fixed-block top-k budget sweep (``csa_fix_topk8/64`` vs
  ``csa_fixed``), whose pair list was hard-coded in the report; and
* the v7 dense->sparse warmup panel, whose only factor is the protocol switch
  step (``warm_steps``) and whose pairs were rejected by ``pair_reason``
  because two protocol tags never pair.

Both now have explicit declaration registries.  The top-k registry goes
through the same arm-diff audit as every other sweep; the warmup registry goes
through the protocol branch and the report dispatches those pairs through
``protocol_contrast_reason`` (which removes exactly the declared axis before
applying the ordinary controlled-pair checks).
"""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


def test_topk_budget_registry_is_single_factor():
    decl = L.TOPK_BUDGET_CONTRAST_FACTORS
    assert set(decl) == {'topk8 vs fixed32', 'topk64 vs fixed32'}, decl
    rep = L.sweep_contrast_factor_audit(decl, verbose=False)
    assert rep['status'] == 'ok', rep['violations']
    for name, entry in rep['pairs'].items():
        assert entry['verified'] is True, (name, entry)
        assert entry['actual_keys'] == ['index_topk'], (name, entry)
        assert entry['actual_single_factor'] is True, (name, entry)
    assert L.declared_topk_budget_pairs() == [
        ('csa_fix_topk8', 'csa_fixed'),
        ('csa_fix_topk64', 'csa_fixed'),
    ], L.declared_topk_budget_pairs()


def test_topk_budget_registry_refuses_a_second_knob():
    decl = {k: dict(v) for k, v in L.TOPK_BUDGET_CONTRAST_FACTORS.items()}
    decl['topk8 vs fixed32'] = dict(
        decl['topk8 vs fixed32'], factors=['index_topk', 'block_size'])
    rep = L.sweep_contrast_factor_audit(decl, verbose=False)
    assert rep['status'] == 'fail', rep
    assert rep['pairs']['topk8 vs fixed32']['verified'] is False, rep['pairs']


def test_warmup_registry_is_protocol_single_factor():
    decl = V7.WARMUP_CONTRAST_FACTORS
    assert set(decl) == {
        'warmup w=5000 vs scratch', 'warmup w=10000 vs scratch'}, decl
    rep = L.sweep_contrast_factor_audit(decl, verbose=False)
    assert rep['status'] == 'ok', rep['violations']
    for name, entry in rep['pairs'].items():
        assert entry['mode'] == 'protocol', (name, entry)
        assert entry['protocol_factor'] == 'warm_steps', (name, entry)
        assert entry['verified'] is True, (name, entry)
        assert entry['actual_single_factor'] is True, (name, entry)


def test_warmup_registry_refuses_a_second_protocol_axis():
    decl = {k: dict(v) for k, v in V7.WARMUP_CONTRAST_FACTORS.items()}
    decl['warmup w=5000 vs scratch'] = dict(
        decl['warmup w=5000 vs scratch'], factors=['warm_steps', 'batch_size'])
    rep = L.sweep_contrast_factor_audit(decl, verbose=False)
    assert rep['status'] == 'fail', rep
    assert rep['pairs']['warmup w=5000 vs scratch']['verified'] is False, rep['pairs']


def _warmup_records():
    knobs = L.training_knobs(amp_dtype=None, eager_batch=False)
    base = {
        'variant': 'csa_fixed',
        'run_cfg': 'warm_v2_steps20000_sl512_bs12_nt8000000',
        'train_knobs': knobs,
        'steps': 20000,
        'tokens_seen': 20000 * 12 * 512,
        'ppl': 10.0,
        'data_fp': 'data-fp',
        'val_fp': 'val-fp',
        '_code': V7.CKPT_CODE,
    }
    return (dict(base, warm_steps=5000, ppl=9.5),
            dict(base, warm_steps=0, ppl=10.0))


def test_protocol_gate_accepts_only_the_declared_warmup_axis():
    decl = V7.WARMUP_CONTRAST_FACTORS['warmup w=5000 vs scratch']
    a, b = _warmup_records()
    # The ordinary gate refuses two protocol tags; the declared protocol gate
    # removes exactly `warm_steps` and then checks everything else.
    assert L.pair_reason(a, b) is not None, L.pair_reason(a, b)
    assert L.protocol_contrast_reason(a, b, decl) is None, \
        L.protocol_contrast_reason(a, b, decl)
    assert V7.contrast_pair_reason('warmup w=5000 vs scratch', a, b) is None

    wrong_level = dict(b, warm_steps=10000)
    assert V7.contrast_pair_reason(
        'warmup w=5000 vs scratch', a, wrong_level) is not None
    second_factor = dict(b, data_fp='other-fp')
    reason = V7.contrast_pair_reason(
        'warmup w=5000 vs scratch', a, second_factor)
    assert reason is not None and 'data_fp' in reason, reason
    # An ordinary arm still uses the ordinary gate, not the protocol waiver.
    assert V7.contrast_pair_reason(
        'CSA+RoPE vs CSA absPE', a, second_factor) is not None


def test_v7_report_pairs_and_declarations_stay_in_sync():
    names = [name for name, _a, _b in V7.ROPE_REPORT_PAIRS]
    assert names[:2] == ['warmup w=5000 vs scratch',
                         'warmup w=10000 vs scratch'], names
    for name in names:
        assert name in V7.ROPE_CONTRAST_FACTORS, name
    assert V7.ROPE_REPORT_PAIRS[1][1].endswith('::w10000')
    assert V7.ROPE_REPORT_PAIRS[1][2].endswith('::w0')


def test_v7_report_uses_the_protocol_gate_and_labels_the_pair():
    with tempfile.TemporaryDirectory() as td:
        od = os.path.join(td, 'results_lm_v7_warmup')
        os.makedirs(od)
        summary = {}
        for seed, (warm, ppl) in enumerate(((0, 10.0), (0, 11.0),
                                            (5000, 9.5), (5000, 10.5))):
            rec, _ = _warmup_records()
            rec = dict(rec, seed=seed % 2, warm_steps=warm, ppl=ppl)
            summary[f'csa_fixed::w{warm}::seed{seed % 2}'] = rec
        with open(os.path.join(od, 'summary.json'), 'w', encoding='utf-8') as fh:
            json.dump(summary, fh)
        rep = V7.bootstrap_report(
            [od], out=os.path.join(td, 'analysis_v7', 'stats.json'))
    entry = rep['comparisons']['warmup w=5000 vs scratch']
    assert entry.get('n') == 2, entry
    assert entry.get('single_factor') is True, entry
    assert entry.get('protocol_factor') == 'warm_steps', entry
    assert entry.get('a_level') == 5000 and entry.get('b_level') == 0, entry
    assert entry.get('factor_declaration_verified') is True, entry


def test_aggregate_single_factor_gate_includes_descriptive_blocks():
    import contextlib
    import io
    import audit_gates as A
    with contextlib.redirect_stdout(io.StringIO()):
        rep = A.audit_single_factor()
    assert rep['status'] == 'ok', rep
    assert rep['topk_budget_contrasts']['status'] == 'ok', \
        rep['topk_budget_contrasts']
    assert rep['warmup_contrasts']['status'] == 'ok', rep['warmup_contrasts']
    assert rep['v7_contrasts']['status'] == 'ok', rep['v7_contrasts']
    for name in ('warmup w=5000 vs scratch', 'warmup w=10000 vs scratch'):
        assert name in rep['warmup_contrasts']['pairs'], name
        assert name in rep['v7_contrasts']['pairs'], name
