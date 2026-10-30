"""CPU checks for the equal-unit distribution of a paired PPL effect.

The frozen endpoint is a token-pooled mean.  These checks pin the new
descriptive readout: per-unit quantiles with distribution-free order-statistic
intervals, the improvement fraction with an exact Clopper-Pearson interval, a
gross gain/harm split, tail asymmetry, the equal-unit vs token-pooled sign gap,
per-seed quantile spread, and a replay audit that rebuilds every derived number
from the stored sufficient statistics.  No result panel is read.
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


def _panel(unit_lr, tok, *, n_seeds=5, noise=0.0, seed=7):
    """NLL-difference matrix whose per-unit log PPL ratio is ``unit_lr``."""
    tok = np.asarray(tok, dtype=np.int64)
    lr = np.asarray(unit_lr, dtype=float)
    rng = np.random.default_rng(seed)
    base = np.tile(lr * tok, (n_seeds, 1)).astype(float)
    if noise:
        base = base + rng.normal(0.0, noise, size=base.shape)
    return base, tok


def _nll_panel(unit_lr, tok, *, n_seeds=5, noise=0.0, seed=7):
    """Two arm NLL matrices with the requested paired difference."""
    diff, tok = _panel(unit_lr, tok, n_seeds=n_seeds, noise=noise, seed=seed)
    b = np.tile(3.0 * np.asarray(tok), (n_seeds, 1)).astype(float)
    a = b + diff
    return a, b, tok


def _stats(block, name='c1', *, n_units=None, seeds=None):
    comp = {'doc_sampling': {'unit_quantile_profile': block,
                             'pooled_ci_n_units': n_units}}
    if seeds is not None:
        comp['per_seed'] = [{'seed': s} for s in seeds]
    return {'finaltest': {'comparisons': {name: comp}}}


def _run_audit(obj):
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'test_stats.json')
        with open(p, 'w', encoding='utf-8') as fh:
            json.dump(obj, fh)
        return A.audit_unit_quantiles(p)


def test_policy_status_and_boundary_are_declared():
    assert L.UNIT_QUANTILE_POLICY['version'] == 'unit-quantile-1'
    assert L.UNIT_QUANTILE_STATUSES == ('ok', 'not_estimable')
    assert 'p-value' in L.UNIT_QUANTILE_POLICY['boundary']
    diff, tok = _panel([-0.01] * 12, [10] * 12)
    block = L.unit_quantile_profile(diff, tok)
    assert block['kind'] == 'unit_quantile_profile'
    assert block['status'] == 'ok'
    assert block['n_units'] == 12 and block['n_seeds'] == 5
    assert [r['q'] for r in block['quantiles']] == [
        0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    assert not ({'confirmed', 'p_holm', 'verified', 'verdict'} & set(block))
    # A valid but too-small panel is an explicit not_estimable readout.
    small = L.unit_quantile_profile(diff[:, :4], tok[:4])
    assert small['status'] == 'not_estimable'
    assert small['kind'] == 'unit_quantile_profile'
    assert small['reason']


def test_quantiles_and_order_statistic_intervals():
    n = 48
    lr = [-0.5 + 0.02 * i for i in range(n)]
    diff, tok = _panel(lr, [10] * n, n_seeds=1)
    block = L.unit_quantile_profile(diff, tok, seeds=[11])
    assert block['status'] == 'ok', block
    vals = [r['log_ratio'] for r in block['quantiles']]
    want = np.quantile(np.asarray(lr), block['policy']['quantiles'])
    assert np.allclose(vals, want, atol=1e-12)
    rows = {r['q']: r for r in block['quantiles']}
    med = rows[0.5]
    assert med['ci_rank'][0] is not None and med['ci_rank'][1] is not None
    assert med['ci95'][0] <= med['log_ratio'] <= med['ci95'][1]
    assert med['ci_coverage'] >= 0.95
    # At n=48 the 5% and 95% quantiles cannot be bounded on both sides; the
    # unavailable side is null, never the sample min/max disguised as a bound.
    assert rows[0.05]['ci95'][0] is None
    assert rows[0.95]['ci95'][1] is None
    # At n=48 the 10% quantile can be bounded below by the sample minimum and
    # the 90% quantile above by the sample maximum; those are real order
    # statistics, not fabricated endpoints.
    assert rows[0.1]['ci95'][0] == _pytest.approx(min(lr))
    assert rows[0.9]['ci95'][1] == _pytest.approx(max(lr))
    # The point quantiles and the rank intervals are deterministic.
    again = L.unit_quantile_profile(diff, tok, seeds=[11])
    assert again == block


def test_equal_unit_and_token_pooled_weighting_can_flip():
    # Eight short units improve a lot; eight long units worsen slightly.  The
    # token-pooled mean is positive while the equal-unit median is negative.
    tok = [10] * 8 + [1000] * 8
    lr = [-0.2] * 8 + [0.05] * 8
    diff, tok = _panel(lr, tok)
    block = L.unit_quantile_profile(diff, tok)
    assert block['status'] == 'ok', block
    assert block['mean']['equal_unit_log_ratio'] < 0.0
    assert block['mean']['token_pooled_log_ratio'] > 0.0
    assert block['mean']['weighting_flips_sign'] is True
    assert block['sign']['direction_agree'] is False
    assert block['sign']['median_sign'] == -1
    assert block['sign']['token_pooled_sign'] == 1
    assert block['improvement']['fraction_better'] == 0.5
    assert block['improvement']['fraction_better_token_weighted'] < 0.02
    # The token-pooled mean must reproduce the cluster-CI point estimate.
    nll_a, nll_b, tok2 = _nll_panel(lr, tok)
    pc = L._paired_nll_doc_ci(nll_a, nll_b, tok2)
    assert abs(block['mean']['token_pooled_log_ratio']
               - pc['delta_log_ratio']) < 1e-10


def test_improvement_fraction_uses_exact_clopper_pearson():
    diff, tok = _panel([-0.01] * 20, [10] * 20)
    all_better = L.unit_quantile_profile(diff, tok)
    imp = all_better['improvement']
    assert imp['n_better'] == 20 and imp['n_worse'] == 0
    assert imp['fraction_better'] == 1.0
    lo, hi = imp['fraction_better_ci95']
    assert 0.80 < lo < 0.90 and hi == 1.0

    diff, tok = _panel([0.01] * 20, [10] * 20)
    all_worse = L.unit_quantile_profile(diff, tok)
    imp = all_worse['improvement']
    assert imp['n_better'] == 0 and imp['n_worse'] == 20
    lo, hi = imp['fraction_better_ci95']
    assert lo == 0.0 and 0.10 < hi < 0.20

    # Ties are reported separately and count as "not better" in the interval.
    lr = [-0.02] * 10 + [0.0] * 5 + [0.02] * 5
    diff, tok = _panel(lr, [10] * 20)
    tied = L.unit_quantile_profile(diff, tok)
    imp = tied['improvement']
    assert imp['n_better'] == 10 and imp['n_tied'] == 5 and imp['n_worse'] == 5
    assert imp['fraction_better'] == 0.5
    lo, hi = imp['fraction_better_ci95']
    assert lo < 0.5 < hi


def test_tail_asymmetry_and_gross_gain_harm():
    # Most units improve slightly; two units are badly harmed.  The mean can
    # still be positive, but the distribution must show the harmed tail.
    lr = [-0.01] * 18 + [0.5, 0.5]
    diff, tok = _panel(lr, [10] * 20)
    block = L.unit_quantile_profile(diff, tok)
    t = block['tails']
    assert t['tail_k'] == 2
    assert t['worst_tail_mean_log'] > 0.0
    assert t['best_tail_mean_log'] < 0.0
    assert t['tail_asymmetry'] > 0.0
    assert t['n_worst_tail_worse'] == 2
    assert t['worst_tail_harm_share'] == _pytest.approx(1.0)
    assert 0.0 < t['best_tail_gain_share'] < 1.0
    assert t['worst_unit']['log_ratio'] == _pytest.approx(0.5)
    assert t['best_unit']['log_ratio'] == _pytest.approx(-0.01)
    g = block['gross']
    assert g['gross_gain_log'] == _pytest.approx(0.18)
    assert g['gross_harm_log'] == _pytest.approx(1.0)
    assert g['net_log'] == _pytest.approx(-0.82)
    assert g['gain_harm_ratio'] == _pytest.approx(0.18)
    assert 0.0 < g['harm_share'] < 1.0


def test_seed_range_and_per_unit_seed_consistency():
    # One unit flips sign across seeds, three are always better and four are
    # always worse.  The block must name the mix without turning it into a test.
    tok = np.asarray([10] * 8, dtype=np.int64)
    d = np.zeros((2, 8), dtype=float)
    d[:, 0] = [-1.0, 1.0]
    d[:, 1:4] = -0.1
    d[:, 4:] = 0.2
    block = L.unit_quantile_profile(d * tok[None, :], tok, seeds=[0, 1])
    sc = block['seed_consistency']
    assert sc['n_units_all_seed_same_sign'] == 7
    assert sc['n_units_seed_sign_mixed'] == 1
    assert sc['n_units_all_seed_better'] == 3
    assert sc['n_units_all_seed_worse'] == 4
    assert sc['fraction_all_seed_same_sign'] == _pytest.approx(7 / 8)
    ranges = {r['q']: r for r in block['seed_range']}
    assert ranges[0.1]['min'] < ranges[0.1]['max']
    assert ranges[0.1]['span'] == _pytest.approx(
        ranges[0.1]['max'] - ranges[0.1]['min'])
    assert block['per_seed_quantiles'][0]['seed'] == 0


def test_wrapper_reproduces_paired_estimand_and_rejects_bad_inputs():
    lr = [-0.02] * 8 + [-0.05] * 8
    a, b, tok = _nll_panel(lr, [20] * 8 + [200] * 8, noise=0.001)
    block = L.paired_nll_quantile_profile(
        a, b, tok, unit_ids=[f'u{i}' for i in range(16)], seeds=list(range(5)))
    assert block['status'] == 'ok', block
    pc = L._paired_nll_doc_ci(a, b, tok)
    assert abs(block['mean']['token_pooled_log_ratio']
               - pc['delta_log_ratio']) < 1e-10
    assert block['unit_ids'][0] == 'u0'
    bad = [
        lambda: L.paired_nll_quantile_profile(a, b[:, :-1], tok),
        lambda: L.paired_nll_quantile_profile(a, b, tok[:-1]),
        lambda: L.unit_quantile_profile(a, [True] * 16),
        lambda: L.unit_quantile_profile(a, [0] * 16),
        lambda: L.unit_quantile_profile(a, [1.5] * 16),
        lambda: L.unit_quantile_profile(a, tok, seeds=[0] * 5),
        lambda: L.unit_quantile_profile(a, tok, seeds=[0] * 4),
        lambda: L.unit_quantile_profile(a, tok, seeds=['x'] * 5),
        lambda: L.unit_quantile_profile(a, tok, unit_ids=['x'] * 16),
        lambda: L.unit_quantile_profile(a, tok, quantiles=[0.5, 0.1]),
        lambda: L.unit_quantile_profile(a, tok, quantiles=[0.0]),
        lambda: L.unit_quantile_profile(a, tok, conf=0.0),
        lambda: L.unit_quantile_profile(a, tok, conf=1.0),
        lambda: L.unit_quantile_profile(a, tok, tail_fraction=0.5),
        lambda: L.unit_quantile_profile([['1.0'] * 16] * 5, tok),
        lambda: L.unit_quantile_profile(a, ['10'] * 16),
    ]
    for call in bad:
        with _pytest.raises(ValueError):
            call()
    a2 = a.copy()
    a2[0, 0] = float('nan')
    with _pytest.raises(ValueError):
        L.unit_quantile_profile(a2, tok)


def test_overflowing_pair_is_not_estimable_not_inf():
    a = np.full((2, 8), 1e308)
    b = np.full((2, 8), -1e308)
    tok = [10] * 8
    block = L.paired_nll_quantile_profile(a, b, tok)
    assert block['status'] == 'not_estimable'
    assert block['reason']
    json.dumps(block, allow_nan=False)
    # Finite but extreme per-unit values still produce a strict-JSON block.
    diff, tok = _panel([1e300] * 8, [10] * 8)
    block = L.unit_quantile_profile(diff, tok)
    assert block['status'] == 'ok', block
    for row in block['quantiles']:
        assert row['ppl_ratio'] is None
    json.dumps(block, allow_nan=False)


def test_doc_level_analysis_wires_the_block():
    tok = np.asarray([40] * 6 + [400] * 6, dtype=np.int64)
    lr = [-0.01] * 6 + [-0.03] * 6
    diff, tok = _panel(lr, tok, n_seeds=6, noise=0.01)
    common = list(range(6))
    recs_a, recs_b = {}, {}
    for si, s in enumerate(common):
        na = [float(x) for x in (3.0 * tok + diff[si])]
        nb = [float(x) for x in (3.0 * tok)]
        pa = [math.exp(x / float(t)) for x, t in zip(na, tok)]
        pb = [math.exp(x / float(t)) for x, t in zip(nb, tok)]
        cell_a = {'ppl': math.exp(sum(na) / float(tok.sum())),
                  'n_tok': int(tok.sum()), 'truncated': False,
                  'nll_per_doc': na,
                  'n_tokens_per_doc': [int(t) for t in tok],
                  'ppl_per_doc': pa,
                  'doc_ids': [f'd{i}' for i in range(len(tok))],
                  'nll_sum': float(sum(na)), 'n_docs': int(len(tok))}
        cell_b = {'ppl': math.exp(sum(nb) / float(tok.sum())),
                  'n_tok': int(tok.sum()), 'truncated': False,
                  'nll_per_doc': nb,
                  'n_tokens_per_doc': [int(t) for t in tok],
                  'ppl_per_doc': pb,
                  'doc_ids': [f'd{i}' for i in range(len(tok))],
                  'nll_sum': float(sum(nb)), 'n_docs': int(len(tok))}
        recs_a[s] = {'variant': 'a', 'seed': s, 'by_len': {8: cell_a}}
        recs_b[s] = {'variant': 'b', 'seed': s, 'by_len': {8: cell_b}}
    out = V12._doc_level_analysis(recs_a, recs_b, common, 8)
    assert isinstance(out, dict)
    assert 'unit_quantile_profile' in out, out
    uq = out['unit_quantile_profile']
    assert uq['status'] == 'ok', uq
    assert uq['n_units'] == len(tok) and uq['n_seeds'] == 6
    assert abs(uq['mean']['token_pooled_log_ratio']
               - out['pooled_delta_log_ratio']) < 1e-9
    assert uq['quantiles'][5]['log_ratio'] > uq['quantiles'][1]['log_ratio']


def test_audit_replays_and_rejects_tampering():
    lr = [-0.02] * 8 + [-0.05] * 8
    diff, tok = _panel(lr, [20] * 8 + [200] * 8, n_seeds=6)
    block = L.unit_quantile_profile(diff, tok, seeds=list(range(6)))
    obj = _stats(block, n_units=len(tok), seeds=list(range(6)))
    out = _run_audit(obj)
    assert out['status'] == 'ok', out
    assert out['n_replayed'] == 1 and out['n_ok'] == 1
    assert out['comparisons']['c1']['n_units'] == len(tok)

    bad = copy.deepcopy(obj)
    bad['finaltest']['comparisons']['c1']['doc_sampling'][
        'unit_quantile_profile']['quantiles'][3]['log_ratio'] += 0.01
    assert _run_audit(bad)['status'] == 'fail'

    bad = copy.deepcopy(obj)
    bad['finaltest']['comparisons']['c1']['doc_sampling'][
        'unit_quantile_profile']['sufficient_statistics'][
        'per_seed_nll_diff'][0][0] += 1.0
    assert _run_audit(bad)['status'] == 'fail'

    bad = copy.deepcopy(obj)
    bad['finaltest']['comparisons']['c1']['doc_sampling'][
        'unit_quantile_profile']['confirmed'] = True
    assert _run_audit(bad)['status'] == 'fail'

    bad = copy.deepcopy(obj)
    bad['finaltest']['comparisons']['c1']['doc_sampling'][
        'unit_quantile_profile']['n_units'] += 1
    assert _run_audit(bad)['status'] == 'fail'

    bad = copy.deepcopy(obj)
    bad['finaltest']['comparisons']['c1']['per_seed'][0]['seed'] = 99
    assert _run_audit(bad)['status'] == 'fail'


def test_audit_legacy_not_estimable_and_corrupt_states():
    diff, tok = _panel([-0.01] * 8, [10] * 8)
    block = L.unit_quantile_profile(diff, tok)
    obj = _stats(block, n_units=len(tok), seeds=list(range(5)))
    del obj['finaltest']['comparisons']['c1']['doc_sampling'][
        'unit_quantile_profile']
    assert _run_audit(obj)['status'] == 'not_verifiable_locally'
    bare = L.unit_quantile_profile(diff[:, :4], tok[:4])
    obj = _stats(bare, n_units=4, seeds=list(range(5)))
    assert _run_audit(obj)['status'] == 'not_applicable'
    # A top-level non-object is not a valid artifact.
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'bad.json')
        with open(p, 'w', encoding='utf-8') as fh:
            json.dump([1, 2, 3], fh)
        assert A.audit_unit_quantiles(p)['status'] == 'fail'


def test_report_markdown_renders_table_and_boundary():
    tok = [10] * 8 + [1000] * 8
    lr = [-0.2] * 8 + [0.05] * 8
    diff, tok = _panel(lr, tok)
    block = L.unit_quantile_profile(diff, tok)
    lines = V12._unit_quantile_markdown(
        {'comparisons': {'c1': {'doc_sampling': {
            'unit_quantile_profile': block,
            'pooled_delta_ppl_ci95': [-1.0, 1.0]}}}})
    text = '\n'.join(lines)
    assert '逐单位分布 / 分位敏感性' in text
    assert 'q10' in text and 'q90' in text
    assert 'Clopper' in text
    assert '⚠ 等权/token 池化反向' in text
    assert '不改变' in text and '冻结' in text
    assert '不能只用 token 池化均值概括' in text


def test_report_does_not_treat_a_skewed_uniform_gain_as_fragile():
    # Every unit improves, but the upper tail improves less; that is worth a
    # descriptive flag, not the "the conclusion is fragile" warning.
    lr = [-0.05] * 16 + [-0.001] * 4
    diff, tok = _panel(lr, [10] * 20)
    block = L.unit_quantile_profile(diff, tok)
    assert block['tails']['tail_asymmetry'] > 0.0
    assert block['tails']['worst_tail_mean_log'] < 0.0
    lines = V12._unit_quantile_markdown(
        {'comparisons': {'c1': {'doc_sampling': {
            'unit_quantile_profile': block,
            'pooled_delta_ppl_ci95': [-1.0, 1.0]}}}})
    text = '\n'.join(lines)
    assert '上尾更分散' in text
    assert '解读提示' not in text


def test_report_markdown_tolerates_malformed_block():
    lines = V12._unit_quantile_markdown({'comparisons': {'c1': {
        'doc_sampling': {
            'unit_quantile_profile': {
                'status': 'ok', 'quantiles': [{}], 'mean': None,
                'improvement': None, 'tails': None, 'sign': None,
            },
            'pooled_delta_ppl_ci95': [-1.0, 1.0]}}}})
    text = '\n'.join(lines)
    assert '逐单位分布' in text
    assert '—' in text
    missing = V12._unit_quantile_markdown({'comparisons': {'c2': {
        'doc_sampling': {'pooled_delta_ppl_ci95': [-1.0, 1.0]}}}})
    assert any('未验证' in line for line in missing)
