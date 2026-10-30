"""CPU checks for length post-stratification of a paired PPL contrast.

The token-pooled ΔPPL is a token-weighted estimand.  These checks pin the new
post-stratified readout: length bins come only from unit token counts, the
overall row reproduces the frozen pooled estimand, equal-strata standardisation
makes composition sensitivity visible, and the audit gate replays every derived
number from the stored sufficient statistics.  No result panel is read.
"""
import copy
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

import pytest as _pytest


def _panel(log_ratio_by_unit, tok, *, n_seeds=8, noise=0.02, seed=7):
    """NLL matrices whose per-unit log PPL ratio is ``log_ratio_by_unit``.

    ``nll_a - nll_b = log_ratio * tokens`` because the per-unit NLLs are
    summed over tokens; a small seed-level perturbation keeps the panel from
    being exactly degenerate without changing the target effect.
    """
    tok = np.asarray(tok, dtype=np.int64)
    lr = np.asarray(log_ratio_by_unit, dtype=float)
    rng = np.random.default_rng(seed)
    base_ce = 3.0
    b = np.tile(base_ce * tok, (n_seeds, 1)).astype(float)
    a = np.tile((base_ce + lr) * tok, (n_seeds, 1)).astype(float)
    a = a + rng.normal(0.0, noise, size=a.shape)
    b = b + rng.normal(0.0, noise, size=b.shape)
    return a, b, tok


def _stats(block, name='c1', *, n_units=None, seeds=None):
    comp = {'doc_sampling': {'length_strata': block,
                             'pooled_ci_n_units': n_units}}
    if seeds is not None:
        comp['per_seed'] = [{'seed': s} for s in seeds]
    return {'finaltest': {'comparisons': {name: comp}}}


def _run_audit(obj):
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'test_stats.json')
        with open(p, 'w', encoding='utf-8') as fh:
            json.dump(obj, fh)
        return A.audit_length_strata(p)


def test_policy_and_status_are_declared():
    assert L.LENGTH_STRATA_POLICY['version'] == 'length-strata-1'
    assert L.LENGTH_STRATA_POLICY['reference'] == 'equal_strata'
    assert L.LENGTH_STRATA_STATUSES == ('ok', 'not_estimable')
    assert 'p-value' in L.LENGTH_STRATA_POLICY['boundary']
    a, b, tok = _panel([-0.01] * 12, [10] * 12)
    block = L.length_stratified_effect(a, b, tok)
    assert block['kind'] == 'length_stratification'
    assert block['status'] == 'ok'
    assert block['n_strata'] == 4
    assert block['n_units'] == 12 and block['n_seeds'] == 8
    assert block['units']['n_tokens'] == int(tok.sum())
    # The block must not carry a confirmatory field.
    assert not ({'confirmed', 'p_holm', 'verified', 'verdict'} & set(block))


def test_monotone_length_effect_and_pooled_estimand():
    tok = [50] * 10 + [200] * 10 + [800] * 10 + [3200] * 10
    lr = [-0.002] * 10 + [-0.006] * 10 + [-0.012] * 10 + [-0.024] * 10
    a, b, tok = _panel(lr, tok, noise=0.02)
    block = L.length_stratified_effect(a, b, tok, seeds=list(range(100, 108)))
    assert block['status'] == 'ok', block
    means = [s['log_ratio'] for s in block['strata']]
    assert all(means[i] > means[i + 1] for i in range(3)), means
    assert block['seeds'] == list(range(100, 108))
    assert block['sign_consistency']['sign_consistent'] is True
    assert block['sign_consistency']['opposite_strata'] == []
    # Same token-pooled estimand as the frozen comparison's cluster CI.
    pc = L._paired_nll_doc_ci(a, b, tok)
    assert abs(block['overall']['log_ratio'] - pc['delta_log_ratio']) < 1e-10
    assert abs(block['overall']['delta_ppl'] - pc['delta_ppl']) < 1e-10
    # Longest stratum dominates the token-weighted log effect.
    conc = block['concentration']
    assert conc['available'] is True
    assert conc['dominant_stratum'] == 3
    assert 0.0 <= conc['max_abs_share'] <= 1.0
    assert len(conc['contributions']) == 4
    assert abs(sum(conc['contributions']) - block['overall']['log_ratio']) < 1e-10
    het = block['heterogeneity']
    assert het['df'] == 3
    assert isinstance(het['q'], float) and math.isfinite(het['q'])
    assert isinstance(het['i2'], float) and 0.0 <= het['i2'] <= 1.0
    assert het['extreme_contrast'] is not None
    assert len(het['pairwise']) == 6


def test_equal_strata_standardisation_flips_a_mixed_effect():
    # Short units improve strongly; long units worsen slightly but carry
    # almost all tokens, so the token-pooled log effect is positive while the
    # equal-strata effect is negative.
    tok = [10] * 10 + [10000] * 10
    lr = [-0.5] * 10 + [0.05] * 10
    a, b, tok = _panel(lr, tok, noise=0.005)
    block = L.length_stratified_effect(a, b, tok, n_strata=2)
    assert block['status'] == 'ok', block
    assert block['overall']['log_ratio'] > 0.0
    assert block['standardized']['log_ratio'] < 0.0
    assert block['standardized']['reference'] == 'equal_strata'
    assert block['standardized']['log_ratio_shift_from_overall'] < 0.0
    sign = block['sign_consistency']
    assert sign['standardization_flips_sign'] is True
    assert sign['sign_consistent'] is False
    assert sign['opposite_strata'] == [0]
    assert sign['raw_delta_opposite_strata'] == [0]
    assert 'raw_delta_reversal' not in sign


def test_token_and_custom_reference_weights():
    tok = [10] * 10 + [1000] * 10
    lr = [-0.2] * 10 + [0.01] * 10
    a, b, tok = _panel(lr, tok, noise=0.004)
    block = L.length_stratified_effect(a, b, tok, n_strata=2, reference='token')
    assert block['status'] == 'ok'
    # The token reference reproduces the overall log effect exactly.
    assert abs(block['standardized']['log_ratio']
               - block['overall']['log_ratio']) < 1e-12
    assert block['standardized']['reference'] == 'token'
    custom = L.length_stratified_effect(
        a, b, tok, n_strata=2, reference=[9.0, 1.0])
    assert custom['status'] == 'ok'
    assert custom['standardized']['reference'] == 'custom'
    assert len(custom['standardized']['weights']) == 2
    assert abs(sum(custom['standardized']['weights']) - 1.0) < 1e-12
    # Zero/negative/NaN weights are refused rather than silently replaced.
    for bad in ([1.0, 0.0], [1.0, -1.0], [1.0, float('nan')], [1.0],
                {'a': 1, 'b': 1}, 'equal'):
        if bad == 'equal':
            continue
        with _pytest.raises(ValueError):
            L.length_stratified_effect(a, b, tok, n_strata=2, reference=bad)


def test_small_panels_reduce_or_refuse_strata():
    # 5 units cannot make four bins of >=2, but can make two.
    a, b, tok = _panel([-0.01] * 5, [10, 20, 30, 40, 50])
    block = L.length_stratified_effect(a, b, tok, n_strata=4,
                                        min_units_per_stratum=2)
    assert block['status'] == 'ok'
    assert block['n_strata'] == 2
    assert block['policy']['requested_n_strata'] == 4
    # With a 3-unit floor even two strata are impossible.
    refused = L.length_stratified_effect(a, b, tok, n_strata=4,
                                          min_units_per_stratum=3)
    assert refused['status'] == 'not_estimable'
    assert refused['kind'] == 'length_stratification'
    assert refused['reason']
    one = L.length_stratified_effect(a[:, :1], b[:, :1], tok[:1])
    assert one['status'] == 'not_estimable'


def test_invalid_inputs_fail_closed():
    a, b, tok = _panel([-0.01] * 8, [10] * 8)
    bad_calls = [
        lambda: L.length_stratified_effect(a, b[:, :-1], tok),
        lambda: L.length_stratified_effect(a, b, tok[:-1]),
        lambda: L.length_stratified_effect(a, b, [True] * 8),
        lambda: L.length_stratified_effect(a, b, [0] * 8),
        lambda: L.length_stratified_effect(a, b, [1.5] * 8),
        lambda: L.length_stratified_effect(a, b, tok, n_strata=1),
        lambda: L.length_stratified_effect(a, b, tok, n_strata=True),
        lambda: L.length_stratified_effect(a, b, tok, n_strata=9),
        lambda: L.length_stratified_effect(a, b, tok, conf=0.0),
        lambda: L.length_stratified_effect(a, b, tok, conf=1.0),
        lambda: L.length_stratified_effect(a, b, tok, seeds=[0] * 8),
        lambda: L.length_stratified_effect(a, b, tok, seeds=[0] * 7),
        lambda: L.length_stratified_effect(a, b, tok, seeds=['x'] * 8),
    ]
    for call in bad_calls:
        with _pytest.raises(ValueError):
            call()
    # A non-finite NLL is not a measurement.
    a2 = a.copy()
    a2[0, 0] = float('nan')
    with _pytest.raises(ValueError):
        L.length_stratified_effect(a2, b, tok)


def test_extreme_finite_values_stay_strict_json():
    tok = [1] * 8 + [100] * 8
    a, b, tok = _panel([0.0] * 16, tok, noise=0.0)
    a = a.copy()
    b = b.copy()
    # Finite but so large that exp(NLL) overflows; log ratios stay finite.
    a[0, 0] = 1e308
    b[0, 0] = 1e308 - 1e292
    block = L.length_stratified_effect(a, b, tok)
    assert block['status'] == 'ok', block
    # The affected PPL-scale cells are unavailable (None), never inf/NaN.
    assert block['strata'][0]['per_seed_delta_ppl'][0] is None
    assert block['overall']['delta_ppl'] is None
    json.dumps(block, allow_nan=False)


def test_doc_level_analysis_wires_the_block():
    tok = np.asarray([40] * 6 + [400] * 6, dtype=np.int64)
    lr = [-0.01] * 6 + [-0.03] * 6
    a, b, tok = _panel(lr, tok, n_seeds=6, noise=0.01)
    common = list(range(6))
    recs_a, recs_b = {}, {}
    for si, s in enumerate(common):
        nll_a = [float(x) for x in a[si]]
        nll_b = [float(x) for x in b[si]]
        pa = [math.exp(x / float(t)) for x, t in zip(nll_a, tok)]
        pb = [math.exp(x / float(t)) for x, t in zip(nll_b, tok)]
        cell_a = {'ppl': math.exp(sum(nll_a) / float(tok.sum())),
                  'n_tok': int(tok.sum()), 'truncated': False,
                  'nll_per_doc': nll_a, 'n_tokens_per_doc': [int(t) for t in tok],
                  'ppl_per_doc': pa, 'doc_ids': [f'd{i}' for i in range(len(tok))],
                  'nll_sum': float(sum(nll_a)), 'n_docs': int(len(tok))}
        cell_b = {'ppl': math.exp(sum(nll_b) / float(tok.sum())),
                  'n_tok': int(tok.sum()), 'truncated': False,
                  'nll_per_doc': nll_b, 'n_tokens_per_doc': [int(t) for t in tok],
                  'ppl_per_doc': pb, 'doc_ids': [f'd{i}' for i in range(len(tok))],
                  'nll_sum': float(sum(nll_b)), 'n_docs': int(len(tok))}
        recs_a[s] = {'variant': 'a', 'seed': s, 'by_len': {8: cell_a}}
        recs_b[s] = {'variant': 'b', 'seed': s, 'by_len': {8: cell_b}}
    out = V12._doc_level_analysis(recs_a, recs_b, common, 8)
    assert isinstance(out, dict)
    assert 'length_strata' in out, out
    ls = out['length_strata']
    assert ls['status'] == 'ok', ls
    assert ls['seeds'] == common
    assert ls['n_units'] == len(tok)
    assert abs(ls['overall']['delta_ppl'] - out['pooled_delta_ppl']) < 1e-9
    assert ls['strata'][1]['log_ratio'] < ls['strata'][0]['log_ratio']


def test_audit_replays_and_rejects_tampering():
    tok = [20] * 8 + [200] * 8
    lr = [-0.01] * 8 + [-0.03] * 8
    a, b, tok = _panel(lr, tok, n_seeds=6, noise=0.01)
    block = L.length_stratified_effect(a, b, tok, seeds=list(range(6)))
    obj = _stats(block, n_units=len(tok), seeds=list(range(6)))
    out = _run_audit(obj)
    assert out['status'] == 'ok', out
    assert out['n_replayed'] == 1 and out['n_ok'] == 1
    assert out['comparisons']['c1']['n_units'] == len(tok)

    bad = copy.deepcopy(obj)
    bad['finaltest']['comparisons']['c1']['doc_sampling']['length_strata'][
        'standardized']['log_ratio'] += 0.01
    assert _run_audit(bad)['status'] == 'fail'

    bad = copy.deepcopy(obj)
    bad['finaltest']['comparisons']['c1']['doc_sampling']['length_strata'][
        'strata'][0]['nll_a_sum'][0] += 1.0
    assert _run_audit(bad)['status'] == 'fail'

    bad = copy.deepcopy(obj)
    bad['finaltest']['comparisons']['c1']['doc_sampling']['length_strata'][
        'confirmed'] = True
    assert _run_audit(bad)['status'] == 'fail'


def test_audit_legacy_and_not_estimable_status():
    a, b, tok = _panel([-0.01] * 8, [10] * 8)
    block = L.length_stratified_effect(a, b, tok)
    obj = _stats(block, n_units=len(tok), seeds=list(range(8)))
    del obj['finaltest']['comparisons']['c1']['doc_sampling']['length_strata']
    assert _run_audit(obj)['status'] == 'not_verifiable_locally'
    bare = L.length_stratified_effect(a[:, :1], b[:, :1], tok[:1])
    obj = _stats(bare, n_units=1, seeds=list(range(8)))
    assert _run_audit(obj)['status'] == 'not_applicable'


def test_report_markdown_renders_table_and_boundary():
    tok = [10] * 10 + [1000] * 10
    lr = [-0.2] * 10 + [0.01] * 10
    a, b, tok = _panel(lr, tok, noise=0.004)
    block = L.length_stratified_effect(a, b, tok, n_strata=2)
    lines = V12._length_strata_markdown(
        {'comparisons': {'c1': {'doc_sampling': {'length_strata': block}}}})
    text = '\n'.join(lines)
    assert '长度分层 / 构成敏感性' in text
    assert '全体（token 池化）' in text
    assert '标准化（equal_strata）' in text
    assert 'Cochran Q=' in text
    assert '不改变' in text and '冻结' in text
    assert '不能把 token 池化结论描述为' in text

def test_report_markdown_tolerates_malformed_block():
    # A corrupt legacy block must be named as unverified, never crash the
    # report builder with an AttributeError on a missing sub-object.
    lines = V12._length_strata_markdown({'comparisons': {'c1': {
        'doc_sampling': {'length_strata': {
            'status': 'ok', 'strata': [{}], 'policy': None,
            'overall': [], 'standardized': [], 'sign_consistency': [],
            'heterogeneity': [], 'concentration': [],
        }, 'pooled_delta_ppl_ci95': [-1.0, 1.0]}}}})
    text = '\n'.join(lines)
    assert '长度分层' in text
    assert '—' in text
