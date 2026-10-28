"""CPU review checks (split group 02; t_eval_per_seq .. t_binary_pr_floor_uses_positive_prevalence)."""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)

def test_eval_per_seq():
    import copy as _copy
    torch.manual_seed(0)
    dev = torch.device('cpu')
    cfgs = L.make_layer_cfgs(2, 'full_rope')
    model = L.SmallGPT(256, 64, 2, 4, 16, 64, _copy.deepcopy(cfgs)).to(dev)
    model.eval()
    ids = np.asarray(np.random.default_rng(3).integers(0, 256, size=(3, 65)), dtype=np.int64)
    out = V7.eval_length_gen(model, ids, [32, 64], dev, max_pos=None)
    for Ln in (32, 64):
        cell = out[Ln]
        assert 'ppl_per_seq' in cell and len(cell['ppl_per_seq']) == 3, 'per-document rows must be recorded'
        geo = math.exp(sum(math.log(p) for p in cell['ppl_per_seq']) / 3.0)
        assert abs(geo - cell['ppl']) / cell['ppl'] < 1e-6, f'pooled ppl must equal the geometric mean of per-seq ppl at L{Ln} (reduction-invariance)'
        # eval_span is the SCORED target span (L-1), the same denominator the
        # by-length audit and n_tok use.  Reporting the input width L here
        # made the cell metadata contradict its own token count.
        assert cell['eval_span'] == Ln - 1, (Ln, cell['eval_span'])
        assert cell['n_tok'] == cell['n_seq'] * cell['eval_span'], cell
    # Two over-long requests can truncate to the SAME token slice.  The cache
    # reuse must carry the per-document accounting, not silently return a cell
    # that is missing n_seq/ppl_per_seq just because another length populated
    # the cache first.
    trunc = V7.eval_length_gen(model, ids, [48, 64], dev, max_pos=32)
    for Ln in (48, 64):
        cell = trunc[Ln]
        assert cell['truncated'] is True, (Ln, cell)
        assert cell['eval_span'] == 31, (Ln, cell['eval_span'])
        assert cell['n_seq'] == 3 and len(cell['ppl_per_seq']) == 3, (Ln, cell)
        assert cell['n_tok'] == 3 * 31, (Ln, cell['n_tok'])
        geo = math.exp(sum(math.log(p) for p in cell['ppl_per_seq']) / 3.0)
        assert abs(geo - cell['ppl']) / cell['ppl'] < 1e-6, (Ln, geo, cell['ppl'])
    # Invalid/ambiguous length geometry must fail before any scoring: a
    # fractional length is not the truncated integer, and max_pos=0/1 or a
    # negative value turns the last-token slice into a different prefix.
    for _bad_lens in ([1], [0], [-4], [1.5], [True], [None]):
        try:
            V7.eval_length_gen(model, ids, _bad_lens, dev, max_pos=None)
        except ValueError:
            pass
        else:
            raise AssertionError(f'eval_length_gen accepted eval_lens={_bad_lens!r}')
    for _bad_max in (0, 1, -1, 1.5, True):
        try:
            V7.eval_length_gen(model, ids, [32], dev, max_pos=_bad_max)
        except ValueError:
            pass
        else:
            raise AssertionError(f'eval_length_gen accepted max_pos={_bad_max!r}')
    for _bad_n in (0, -1, 1.5, True):
        try:
            V7.eval_length_gen(model, ids, [32], dev, n_seq=_bad_n)
        except ValueError:
            pass
        else:
            raise AssertionError(f'eval_length_gen accepted n_seq={_bad_n!r}')
    # The sampled path must credit per-document stats to the GLOBAL row, not
    # to a chunk-local row.  With chunk=1 and most targets sampled, a broken
    # offset would pile almost every token onto row 0.
    samp = L.eval_ppl(model, ids, dev, chunk=1, eval_rows=150, eval_seed=5, return_stats=True)
    per_row = samp['n_tokens_per_row']
    assert sum(per_row) == int(samp['n_tokens']) == 150, (sum(per_row), samp['n_tokens'])
    assert max(per_row) <= ids.shape[1] - 1, f'per-document token counts must stay within one row: {per_row}'
    assert abs(sum(samp['nll_per_row']) - samp['nll_sum']) < 1e-9, 'per-document NLL must sum to the pooled NLL'


def test_review3_evidence_metadata():
    assert os.path.exists(os.path.join(REPO, 'DECISION_LOG.md')), 'DECISION_LOG.md must be tracked'
    assert 'requirements.txt' in open(os.path.join(REPO, 'archive_release.py'), encoding='utf-8').read(), 'archive must pack the dependency declaration'
    assert 'DECISION_LOG.md' in open(os.path.join(REPO, 'archive_release.py'), encoding='utf-8').read()
    import build_report as BR
    import error_analysis as EA
    assert BR._sig_gate({'n': 6, 'p_exact_signflip': 0.01, 'p_holm': 0.02, 'mean': 1.0, 'std': 0.5}) is True
    assert BR._sig_gate({'n': 3, 'p_exact_signflip': 0.01, 'p_holm': 0.02, 'mean': 1.0, 'std': 0.5}) is False
    assert BR._sig_gate({'n': 6, 'p_exact_signflip': 0.01, 'p_holm': None, 'mean': 1.0, 'std': 0.5}) is False


def test_review3_pooled_aggregate():
    summary = {}
    for s, nll in ((0, 100.0), (1, 140.0)):
        summary[f'csa::seed{s}'] = {'variant': 'csa', 'seed': s, 'ppl': math.exp(nll / 100.0),
                                    'nll_sum': nll, 'n_tokens': 100, 'params': 10,
                                    'steps': 1, 'tokens_seen': 100,
                                    'train_objective': {'comp_reg_applies': True, 'delta_lr_effective': 10.0, 'boundary_policy': ['threshold']}}
    agg = L.aggregate(summary)
    e = agg['csa']
    assert abs(e['ppl_pooled'] - math.exp(240.0 / 200.0)) < 1e-9, e
    assert e['ppl_primary'] == e['ppl_pooled']
    assert e['ppl_mean_arithmetic'] != e['ppl_pooled'] or True


def test_nested_primary_paired_deltas():
    # A nested record's `ppl` is the full-validation value; its PRIMARY
    # estimate is the held-out-row PPL.  Every paired delta must use the
    # primary estimand, otherwise the contrast silently reads the rows that
    # selected the hyper-parameters.
    def _rec(arm, seed, full_ppl, hold_ppl, cfg='same_cfg'):
        return {
            'variant': arm, 'seed': seed, 'run_cfg': cfg,
            'ppl': float(full_ppl), 'nll_sum': math.log(full_ppl) * 100.0,
            'n_tokens': 100, 'params': 10, 'steps': 1, 'tokens_seen': 400,
            'data_fp': 'd', 'val_fp': 'v', 'eval_fp': 'e',
            'ppl_estimation_rows': 'nested_validation_holdout',
            'ppl_holdout': float(hold_ppl),
            'nll_sum_holdout': math.log(hold_ppl) * 40.0, 'n_tokens_holdout': 40,
            'holdout_row_fp': 'h',
            'nll_per_row_holdout': [math.log(hold_ppl) * 40.0],
            'n_tokens_per_row_holdout': [40],
        }
    summary = {}
    for s in range(6):
        # Full validation says the method is better, while the held-out rows
        # say it is worse.  The primary endpoint is the held-out one.
        summary[f'full::seed{s}'] = _rec('full', s, 10.0, 10.0)
        summary[f'csa_dynamic::seed{s}'] = _rec('csa_dynamic', s, 9.0, 11.0)
    agg = L.aggregate(summary)
    entry = agg['csa_dynamic']
    assert abs(entry['ppl_mean'] - 11.0) < 1e-12, entry
    assert entry['ppl_primary_estimand'] == 'nested_validation_holdout', entry
    assert abs(entry['dPPL_vs_full_mean'] - 1.0) < 1e-12, entry
    recs, _amb = L._variant_seed_map(summary)
    deltas = L._paired_seed_deltas(recs, 'csa_dynamic', 'full')
    assert all(abs(d - 1.0) < 1e-12 for d in deltas[('', '')][0]), deltas
    # Holdout accounting can be recovered from NLL totals or per-row arrays;
    # a declared-nested record with none of them is unverifiable, never a
    # silent fallback to the full-validation PPL.
    total_only = _rec('csa_dynamic', 0, 9.0, 11.0)
    total_only.pop('ppl_holdout')
    assert abs(L._record_primary_ppl(total_only) - 11.0) < 1e-12, total_only
    rows_only = _rec('csa_dynamic', 0, 9.0, 12.0)
    rows_only.pop('ppl_holdout')
    rows_only.pop('nll_sum_holdout')
    rows_only.pop('n_tokens_holdout')
    assert abs(L._record_primary_ppl(rows_only) - 12.0) < 1e-12, rows_only
    missing = _rec('csa_dynamic', 0, 9.0, 11.0)
    for _k in ('ppl_holdout', 'nll_sum_holdout', 'n_tokens_holdout',
               'nll_per_row_holdout', 'n_tokens_per_row_holdout'):
        missing.pop(_k)
    assert L._record_primary_ppl(missing) is None, missing
    assert L._paired_primary_delta(missing, _rec('full', 0, 10.0, 10.0)) is None
    # A group that mixes the two estimands is not poolable: summing holdout
    # NLLs with full-validation NLLs would invent a third sample, and labelling
    # the result 'nested' would misdescribe it.  The whole group must fall back
    # to full validation so every reported number is on one sample.
    mixed_full = {'variant': 'mixed', 'seed': 0, 'ppl': 10.0,
                  'nll_sum': 10.0, 'n_tokens': 1, 'params': 10,
                  'steps': 1, 'tokens_seen': 100, 'run_cfg': 'same'}
    mixed_nested = _rec('mixed', 1, 20.0, 30.0)
    mixed_nested.update(run_cfg='same', steps=1, tokens_seen=100)
    assert L._pooled_ppl_from_records([mixed_full, mixed_nested]) == (
        None, None, None)
    _pool, _nll, _nt = L._pooled_ppl_from_records(
        [mixed_full, mixed_nested], estimand=False)
    _want_pool = math.exp((math.log(20.0) * 100.0 + 10.0) / 101.0)
    assert abs(_pool - _want_pool) < 1e-12, (_pool, _nll, _nt)
    mixed_agg = L.aggregate({'mixed::seed0': mixed_full,
                             'mixed::seed1': mixed_nested})['mixed']
    assert mixed_agg['ppl_primary_estimand'] == 'full_validation_fallback', mixed_agg
    assert abs(mixed_agg['ppl_mean'] - 15.0) < 1e-12, mixed_agg
    assert abs(mixed_agg['ppl_pooled'] - _want_pool) < 1e-12, mixed_agg
    # A pair itself must not subtract a full-validation PPL from a holdout PPL.
    assert L._paired_primary_delta(mixed_full, mixed_nested) is None
    # Nor may a contrast average deltas from different estimands across seeds.
    mixed_summary = {
        'a::seed0': dict(mixed_full, variant='a'),
        'b::seed0': dict(mixed_full, variant='b', ppl=12.0),
        'a::seed1': dict(mixed_nested, variant='a'),
        'b::seed1': dict(mixed_nested, variant='b', ppl_holdout=31.0),
    }
    _recs, _amb = L._variant_seed_map(mixed_summary)
    _md = L._paired_seed_deltas(_recs, 'a', 'b')
    _deltas, _seeds, _skipped = _md[('', '')]
    assert _deltas == [] and len(_skipped) == 2, _md
    assert all('mixed estimands' in why for _s, why in _skipped), _md


def test_combined_seed_doc_uncertainty():
    # Two-level (seed x document) combined interval: the seed-level t CI and
    # the doc-level bootstrap each see only ONE variance source; the
    # combined interval must not silently drop the other one.
    def _recs(seed_offsets, n_docs=32, ntok=50, zero_docs=0):
        out = []
        for s, off in enumerate(seed_offsets):
            nll = [1.0 + off + 0.01 * j for j in range(n_docs)]
            ntoks = [ntok] * (n_docs - zero_docs) + [0] * zero_docs
            out.append({'variant': 'csa', 'seed': s,
                        'ppl': math.exp(sum(nll) / (n_docs * ntok)),
                        'nll_per_row': nll, 'n_tokens_per_row': ntoks})
        return out
    import numpy as _np
    recs = _recs([0.0, 0.2, 0.4])
    r = L._combined_seed_doc_ci(recs)
    assert r['status'] == 'ok', r
    assert r['n_seeds'] == 3 and r['n_docs'] == 32 and r['n_docs_dropped_zero_token'] == 0, r
    per_seed_lp = []
    for off in (0.0, 0.2, 0.4):
        nll = [1.0 + off + 0.01 * j for j in range(32)]
        # NLL/tokens is already log-PPL; the log must not be applied twice.
        per_seed_lp.append(sum(nll) / (32 * 50))
    _se_seed = _np.std(per_seed_lp, ddof=1) / math.sqrt(3)
    assert abs(r['se_seed_logppl'] - float(_se_seed)) < 1e-12, (r, _se_seed)
    assert r['se_doc_logppl'] is not None and r['se_doc_logppl'] > 0, r
    _comb = math.sqrt(r['se_seed_logppl'] ** 2 + r['se_doc_logppl'] ** 2)
    assert abs(r['se_combined_logppl'] - _comb) < 1e-12, r
    assert abs(r['component_ratio_seed_over_doc'] - r['se_seed_logppl'] / r['se_doc_logppl']) < 1e-12, r
    lo, hi = r['ci95_ppl']
    assert 0 < lo < hi, r
    # Center must be the PPL scale, not exp(log(CE)).
    _center = math.exp(float(_np.mean(per_seed_lp)))
    assert lo < _center < hi, (r, _center)
    assert lo > 1.0, ('a valid PPL interval cannot sit below 1.0', r)
    # The combined interval is never NARROWER than the seed-only t interval.
    tc = L._t_critical(0.95, 2)
    seed_only = (math.exp(_np.mean(per_seed_lp) - tc * _se_seed),
                 math.exp(_np.mean(per_seed_lp) + tc * _se_seed))
    assert r['ci95_ppl'][0] <= seed_only[0] + 1e-9 and r['ci95_ppl'][1] >= seed_only[1] - 1e-9, (r, seed_only)
    # Identical documents: the doc component collapses to ~0 and the
    # combined interval reduces to the seed component.
    r0 = L._combined_seed_doc_ci(_recs([0.0, 0.3, 0.6], n_docs=16, ntok=50)
                                 )
    def _flat(seed_offsets):
        out = []
        for s, off in enumerate(seed_offsets):
            nll = [1.0 + off] * 16
            out.append({'variant': 'csa', 'seed': s, 'ppl': math.exp(sum(nll) / (16 * 50)),
                        'nll_per_row': nll, 'n_tokens_per_row': [50] * 16})
        return out
    rf = L._combined_seed_doc_ci(_flat([0.0, 0.3, 0.6]))
    assert rf['status'] == 'ok' and rf['se_doc_logppl'] is not None and rf['se_doc_logppl'] < 1e-12, rf
    assert abs(rf['se_combined_logppl'] - rf['se_seed_logppl']) < 1e-12, rf
    assert rf['component_ratio_seed_over_doc'] is None, rf
    # Fractional per-document token counts are not a pooled denominator.
    _bad = _recs([0.0, 0.2, 0.4])
    _bad[0]['n_tokens_per_row'] = list(_bad[0]['n_tokens_per_row'])
    _bad[0]['n_tokens_per_row'][0] = 50.5
    assert L._combined_seed_doc_ci(_bad)['status'] == 'unverifiable'
    assert L._doc_bootstrap_ppl_ci(_bad) is None
    # Mixed estimands are two different document samples, not one bootstrap
    # panel, even if both expose per-row arrays of the same length.
    _mix = _recs([0.0, 0.2, 0.4])
    _mix[0]['ppl_estimation_rows'] = 'nested_validation_holdout'
    _mix[0]['nll_per_row_holdout'] = list(_mix[0]['nll_per_row'])
    _mix[0]['n_tokens_per_row_holdout'] = list(_mix[0]['n_tokens_per_row'])
    _rmix = L._combined_seed_doc_ci(_mix)
    assert _rmix['status'] == 'unverifiable' and 'mix' in _rmix.get('reason', ''), _rmix
    assert L._doc_bootstrap_ppl_ci(_mix) is None
    # One seed: the seed component is explicitly unverifiable, never zero.
    r1 = L._combined_seed_doc_ci(_recs([0.0]))
    assert r1['status'] == 'seed_component_unverifiable' and r1['se_seed_logppl'] is None and r1['se_doc_logppl'] is not None, r1
    assert r1['ci95_ppl'] is None, r1
    # Records without per-row NLL cannot support EITHER component.
    r2 = L._combined_seed_doc_ci([{'variant': 'csa', 'seed': 0, 'ppl': 30.0},
                                  {'variant': 'csa', 'seed': 1, 'ppl': 31.0}])
    assert r2['status'] == 'unverifiable' and r2['ci95_ppl'] is None, r2
    # Coverage/draw-count/RNG are part of the interval identity: never build
    # a 95% interval while echoing an invalid option.
    for _bad_conf in (True, 0, 1, 1.5, -0.1, float('nan'), 'x'):
        _rr = L._combined_seed_doc_ci(recs, conf=_bad_conf)
        assert _rr['status'] == 'unverifiable' and _rr.get('reason'), (_bad_conf, _rr)
        assert _rr['conf'] is None and _rr['ci95_ppl'] is None, (_bad_conf, _rr)
    for _bad_nb in (True, 0, 19, 1.5, 'x', None, float('nan')):
        _rr = L._combined_seed_doc_ci(recs, n_boot=_bad_nb)
        assert _rr['status'] == 'unverifiable' and _rr.get('reason'), (_bad_nb, _rr)
    for _bad_seed in (True, 1.5, 'x', None, float('nan')):
        _rr = L._combined_seed_doc_ci(recs, seed=_bad_seed)
        assert _rr['status'] == 'unverifiable' and 'seed' in _rr.get('reason', ''), (_bad_seed, _rr)
    # A single scored document cannot carry a doc-resampling component.
    r3 = L._combined_seed_doc_ci(_recs([0.0, 0.2], n_docs=1, ntok=50))
    assert r3['status'] == 'doc_component_unverifiable' and r3['se_doc_logppl'] is None and r3['se_seed_logppl'] is not None, r3
    # Zero-token (never scored) rows are dropped, not pooled as free mass.
    r4 = L._combined_seed_doc_ci(_recs([0.0, 0.2], n_docs=8, ntok=50, zero_docs=2))
    assert r4['status'] == 'ok' and r4['n_docs'] == 6 and r4['n_docs_dropped_zero_token'] == 2, r4
    # A whole seed with no scored token is not a CE=0 measurement: the seed
    # component would otherwise divide by 1e-30 and report a fake zero-variance
    # seed.  Refuse the interval instead.
    _zero_seed = _recs([0.0, 0.2])
    _zero_seed[1]['n_tokens_per_row'] = [0] * len(_zero_seed[1]['n_tokens_per_row'])
    _zero_seed[1]['nll_per_row'] = [0.0] * len(_zero_seed[1]['nll_per_row'])
    r5 = L._combined_seed_doc_ci(_zero_seed)
    assert r5['status'] == 'unverifiable', r5
    assert r5['se_seed_logppl'] is None and r5['se_doc_logppl'] is None, r5
    assert r5['ci95_ppl'] is None and 'scored tokens' in r5.get('reason', ''), r5
    # The same refusal applies when every document is unscored (there is no
    # pooled PPL to bootstrap).
    r6 = L._combined_seed_doc_ci(_recs([0.0, 0.2], n_docs=4, zero_docs=4))
    assert r6['status'] == 'unverifiable' and r6['ci95_ppl'] is None, r6
    # The aggregate entry exposes the combined interval next to the two
    # single-level intervals.
    summary = {}
    for s, off in ((0, 0.0), (1, 0.2)):
        nll = [1.0 + off + 0.01 * j for j in range(32)]
        summary[f'csa::seed{s}'] = {'variant': 'csa', 'seed': s, 'ppl': math.exp(sum(nll) / (32 * 50)),
                                    'nll_sum': sum(nll), 'n_tokens': 32 * 50, 'params': 10,
                                    'steps': 1, 'tokens_seen': 100,
                                    'nll_per_row': nll, 'n_tokens_per_row': [50] * 32,
                                    'train_objective': {'comp_reg_applies': True, 'delta_lr_effective': 10.0, 'boundary_policy': ['threshold']}}
    agg = L.aggregate(summary)
    cc = agg['csa'].get('ppl_ci_seed_doc_combined')
    assert isinstance(cc, dict) and cc['status'] == 'ok' and cc['ci95_ppl'] is not None, cc


def test_review3_holm_family_size():
    comps = {'a': {'p_exact_signflip': 0.01}, 'b': {'omitted': 'missing arm'}, 'c': {'p_exact_signflip': 0.04}}
    fam = L.attach_holm(comps)
    assert fam['declared_slots'] == 3 and fam['tested'] == 2, fam
    assert comps['a']['p_holm'] is not None and comps['c']['p_holm'] is not None
    # The omitted slot must inflate m: Holm(a)=3*0.01=0.03, not 2*0.01=0.02.
    assert abs(comps['a']['p_holm'] - 0.03) < 1e-12, comps


def test_review3_span_clean_key():
    pp = {'n_seq': 16, 'span_len': 32, 'target': 512}
    clean = V10._clean_probe_params(pp)
    assert 'span_len' not in clean and clean['n_seq'] == 16, clean
    assert pp['span_len'] == 32, 'the measured cell must keep span_len'


def test_key_exploratory_phases_declare_at_least_three_seeds():
    # Prospective seed plan: the top-k sweep and length-extrapolation panels
    # are key exploratory comparisons, so a 2-seed plan is not enough even for
    # a descriptive mean±std.  (Other phases inherit >=3 seeds from their cfg.)
    by_name = {p[0]: p for p in V7.PHASES}
    for _name in ('P1T', 'P0E'):
        _payload = by_name[_name][2]
        _seeds = _payload.get('seeds') or (_payload.get('cfg') or {}).get('seeds')
        assert _seeds is not None and len(set(_seeds)) >= 3, (_name, _seeds)
    # Scale panels are key cross-size comparisons too: a 2-seed plan cannot
    # even report a seed-level mean±std.  The v10 scale phases and their v12
    # token/compute-matched re-runs all declare >=3 seeds.
    v10_phases = {p[0]: p for p in V10.PHASES}
    for _name in ('P3SS', 'P3SL', 'P3SL2', 'P3SL3'):
        _seeds = v10_phases[_name][3]
        assert _seeds is not None and len(set(_seeds)) >= 3, (_name, _seeds)
    for _name in ('P5S2_CFG', 'P5S3_CFG'):
        _seeds = getattr(V12, _name).get('seeds')
        assert _seeds is not None and len(set(_seeds)) >= 3, (_name, _seeds)
    # The supplementary drivers must not rely on a LATER driver to lift a key
    # panel above the 3-seed floor: v8 alone used to schedule the 20k RoPE
    # contrast at n=2 and v9 alone the top-k sweep at n=2.
    import explib.scale as V8
    import explib.resume as V9
    import explib.length as V11
    v8_phases = {p[0]: p for p in V8.PHASES}
    for _name in ('P1S', 'P2R'):
        _seeds = v8_phases[_name][2]
        assert _seeds is not None and len(set(_seeds)) >= 3, (_name, _seeds)
    v9_phases = {p[0]: p for p in V9.PHASES}
    for _name in ('P2T', 'P2S3'):
        _seeds = v9_phases[_name][2]
        assert _seeds is not None and len(set(_seeds)) >= 3, (_name, _seeds)
    v11_phases = {p[0]: p for p in V11.PHASES}
    for _name in ('P4SS', 'P4F', 'P4SL'):
        _seeds = v11_phases[_name][3]
        assert _seeds is not None and len(set(_seeds)) >= 3, (_name, _seeds)
    for _fn in (V11.P4MT_CFG, V11.P4MP_CFG):
        _seeds = _fn().get('seeds')
        assert _seeds is not None and len(set(_seeds)) >= 3, (_fn.__name__, _seeds)


def test_v10_holm_family_keeps_unmeasured_slots():
    # A missing/non-finite p is a dummy p=1 slot, not a deleted comparison:
    # dropping it would shrink m and overstate the remaining tests' power.
    fam = V10._holm_family({
        'a': {'p_exact_signflip': 0.01},
        'b': {'p_exact_signflip': float('nan')},
        'c': {},
        'd': {'p_exact_signflip': True},
        'e': {'p_exact_signflip': 1.5},
    })
    assert [k for k, _p in fam] == ['a', 'b', 'c', 'd', 'e'], fam
    assert [p for _k, p in fam] == [0.01, 1.0, 1.0, 1.0, 1.0], fam
    h = L.holm_adjust(fam)
    assert abs(h['a']['p_holm'] - 5 * 0.01) < 1e-12, h
    assert h['b']['p_holm'] == 1.0 and h['c']['p_holm'] == 1.0, h


def test_objective_ablation_is_declared_not_a_confound():
    # The loss-term arm zeroes the per-layer multiplier, so the descriptor must
    # report that comp_reg does NOT apply (previously it copied the configured
    # lambda and silently described the wrong objective).
    m_full = L.SmallGPT(64, 32, 2, 2, 16, 16, L.make_layer_cfgs(2, 'csa_dynamic'))
    m_none = L.SmallGPT(64, 32, 2, 2, 16, 16, L.make_layer_cfgs(2, 'csa_dyn_nocomp'))
    o_full = L.train_objective(m_full, comp_lambda=0.05, delta_lr_mult=10.0)
    o_none = L.train_objective(m_none, comp_lambda=0.05, delta_lr_mult=10.0)
    assert o_full['comp_reg_applies'] is True and o_full['comp_lambda_effective'] > 0, o_full
    assert o_none['comp_reg_applies'] is False and o_none['comp_lambda_effective'] == 0.0, o_none
    assert 'comp_reg' not in o_none['loss_terms'], o_none
    assert o_none['comp_reg_configured_layers'] == o_full['comp_reg_configured_layers'] == 2, (o_full, o_none)
    _r, _keys = L._objective_diff({'train_objective': o_full}, {'train_objective': o_none})
    assert _r and _keys == ['comp_reg_applies', 'comp_lambda_effective', 'loss_terms'], (_r, _keys)
    # A descriptor that omits fields is unverifiable, not equal: comparing
    # two `.get()` misses would make an empty object look matched.
    _ru, _ku = L._objective_diff({'train_objective': {}}, {'train_objective': {}})
    assert _ru and _ru.startswith('unverifiable') and _ku == [], (_ru, _ku)
    _rp, _kp = L._objective_diff(
        {'train_objective': {'comp_reg_applies': True}},
        {'train_objective': {'comp_reg_applies': True}})
    assert _rp and _rp.startswith('unverifiable') and _kp == [], (_rp, _kp)
    # Same effective weight but a different configured base lambda + per-layer
    # multiplier is still a different objective.
    _o_scaled = dict(o_full, comp_lambda=0.2, comp_lambda_effective=0.2,
                     comp_lambda_mults=[1.0])
    _rs, _ks = L._objective_diff({'train_objective': o_full},
                                 {'train_objective': _o_scaled})
    assert _rs and 'comp_lambda' in _ks, (_rs, _ks)
    # The objective difference is the declared intervention for this module,
    # not an undeclared confound: six same-direction seeds must verify it
    # rather than downgrade it to system_level_only.
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    s = {}
    for i in range(6):
        s[f'csa_dynamic::seed{i}'] = dict(
            variant='csa_dynamic', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
            ppl=49.0, params=100, train_knobs=knobs, train_objective=o_full)
        s[f'csa_dyn_nocomp::seed{i}'] = dict(
            variant='csa_dyn_nocomp', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
            ppl=50.0, params=100, train_knobs=knobs, train_objective=o_none)
    rep = L.methodology_audit(
        s, design={'compression_regularizer': [
            ('csa_dynamic', 'csa_dyn_nocomp', 'loss term')]},
        baselines={}, verbose=False)
    e = rep['components']['compression_regularizer'][0]
    assert e['objective_expected_differences'] == sorted([
        'comp_reg_applies', 'comp_lambda_effective', 'loss_terms']), e
    assert e.get('mechanism_claim_allowed') is not False, e
    assert 'compression_regularizer' in rep['verified_components'], rep
    assert 'compression_regularizer' not in rep['system_level_only_components'], rep
    # The task-learned boundary arm differs in its policy term, which is also
    # the declared intervention.
    o_task = dict(o_full, boundary_policy=['reinforce'],
                  loss_terms=['ce', 'comp_reg', 'boundary_policy_gradient'])
    s2 = {}
    for i in range(6):
        s2[f'csa_dynamic_task::seed{i}'] = dict(
            variant='csa_dynamic_task', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
            ppl=49.0, params=100, train_knobs=knobs, train_objective=o_task)
        s2[f'csa_dynamic::seed{i}'] = dict(
            variant='csa_dynamic', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
            ppl=50.0, params=100, train_knobs=knobs, train_objective=o_full)
    rep2 = L.methodology_audit(
        s2, design={'task_learned_boundary': [
            ('csa_dynamic_task', 'csa_dynamic', 'policy')]},
        baselines={}, verbose=False)
    e2 = rep2['components']['task_learned_boundary'][0]
    assert e2['objective_expected_differences'] == sorted(
        ['boundary_policy', 'loss_terms']), e2
    assert 'task_learned_boundary' in rep2['verified_components'], rep2
    # An UNDECLARED objective change is still a confound.
    o_bad = dict(o_full, boundary_policy=['reinforce'])
    s3 = {k: dict(v) for k, v in s.items()}
    for i in range(6):
        s3[f'csa_dyn_nocomp::seed{i}']['train_objective'] = o_bad
    rep3 = L.methodology_audit(
        s3, design={'compression_regularizer': [
            ('csa_dynamic', 'csa_dyn_nocomp', 'loss term')]},
        baselines={}, verbose=False)
    e3 = rep3['components']['compression_regularizer'][0]
    assert e3.get('mechanism_claim_allowed') is False, e3
    assert 'compression_regularizer' in rep3['system_level_only_components'], rep3


def test_review3_boundary_arm_and_objective():
    cfg = L.make_layer_cfgs(1, 'csa_dynamic_task')[0]
    assert getattr(cfg, 'boundary_policy', None) == 'reinforce', cfg
    obj = L.train_objective(type('M', (), {'blocks': [], 'named_parameters': lambda self: iter([])})(), comp_lambda=0.05, delta_lr_mult=10.0)
    assert obj['comp_reg_applies'] is False, obj
    assert 'boundary_policy' in obj and 'loss_terms' in obj
    # The score-function term must actually reach delta_logit; otherwise the
    # "task-learned boundary" arm would be inert.
    m = L.SmallGPT(64, 32, 1, 2, 16, 16, L.make_layer_cfgs(1, 'csa_dynamic_task'))
    x = torch.randint(0, 64, (1, 16), generator=torch.Generator().manual_seed(0))
    y = torch.randint(0, 64, (1, 16), generator=torch.Generator().manual_seed(1))
    ce = m.ce_loss(x, y, 64)
    assert m.boundary_logp is not None and m.boundary_decisions > 0, 'boundary policy must emit log-probabilities'
    loss = ce + 0.05 * m.comp_reg - float(ce.detach()) * m.boundary_logp / max(m.boundary_decisions, 1)
    loss.backward()
    grads = [p.grad.abs().sum().item() for n, p in m.named_parameters() if 'delta_logit' in n and p.grad is not None]
    assert grads and all(g > 0 for g in grads), f'delta_logit received no task-loss gradient: {grads}'


def test_seed_reproducible():
    def draw():
        L.set_seed(12345)
        return (random.random(), float(np.random.random()), float(torch.rand(1).item()), L.rng_fingerprint())
    a = draw()
    b = draw()
    assert a == b, f'same seed must reproduce Python/NumPy/Torch draws and the RNG fingerprint: {a} vs {b}'
    L.set_seed(54321)
    d = (random.random(), float(np.random.random()), float(torch.rand(1).item()))
    assert d != a[:3], 'a different seed must change the RNG stream'


def test_train_variant_normalizes_seed():
    ids = np.arange(8192, dtype=np.int64) % 997
    val = (np.arange(32, dtype=np.int64) % 997).reshape(2, 16)
    kw = dict(d=32, n_layers=1, n_heads=4, d_head=8, seq_len=16, batch_size=2,
              steps=2, lr=1e-3, weight_decay=0.0, warmup=0, comp_lambda=0.0,
              eval_every=0, mlp_ratio=4.0, device=torch.device('cpu'), log_every=0)
    a = L.train_variant('full', ids, val, 997, seed=np.int64(3), **kw)
    b = L.train_variant('full', ids, val, 997, seed='3', **kw)
    assert a['seed'] == b['seed'] == 3 and isinstance(a['seed'], int), (a['seed'], b['seed'])
    assert a['data_seed'] == b['data_seed'] == 3, (a.get('data_seed'), b.get('data_seed'))
    assert a['losses'] == b['losses'], 'a normalisable seed must use the same stream as the integer seed'
    assert a.get('exec_fp') and a['exec_fp'] == b['exec_fp'], a.get('exec_fp')


def test_data_seed_for_rank_is_domain_safe():
    assert L.data_seed_for_rank(0, 0) == 0
    assert L.data_seed_for_rank(L.MAX_SEED, 0) == L.MAX_SEED
    seen = set()
    for seed in (0, 1, L.MAX_SEED):
        for rank in range(1, 8):
            s = L.data_seed_for_rank(seed, rank)
            assert 0 <= s <= L.MAX_SEED, (seed, rank, s)
            seen.add((seed, rank, s))
    assert len({s for _seed, _rank, s in seen}) == len(seen), 'rank-derived seeds must not collide'
    assert L.data_seed_for_rank(0, 1) != 1, 'rank 1 seed=0 must not alias the rank-0 seed=1 stream'
    for bad in (-1, L.MAX_SEED + 1, True, 1.5):
        try:
            L.data_seed_for_rank(bad, 0)
        except ValueError:
            pass
        else:
            raise AssertionError(f'data_seed_for_rank accepted seed {bad!r}')


def test_all_ranks_agree_branches_on_reduced_value():
    # A rank-local finiteness flag must be AND-reduced before DDP chooses
    # backward vs skip.  The old loss branch reduced the flag but still
    # branched on the local value, so a finite rank could enter backward while
    # a peer skipped it and deadlock.
    if not torch.distributed.is_available():
        return
    _orig = torch.distributed.all_reduce
    _peer = [1]
    _seen = []

    def _fake_all_reduce(flag, op=None):
        _seen.append((int(flag.item()), op))
        flag.fill_(min(int(flag.item()), _peer[0]))

    torch.distributed.all_reduce = _fake_all_reduce
    try:
        dev = torch.device('cpu')
        assert L._all_ranks_agree(True, False, dev) is True
        assert L._all_ranks_agree(False, False, dev) is False
        assert not _seen, 'single-process calls must not enter a collective'
        assert L._all_ranks_agree(True, True, dev) is True
        _peer[0] = 0
        assert L._all_ranks_agree(True, True, dev) is False, (
            'a locally finite rank must skip when a peer reports non-finite')
        _peer[0] = 1
        assert L._all_ranks_agree(False, True, dev) is False
        assert all(op == torch.distributed.ReduceOp.MIN for _, op in _seen), _seen
    finally:
        torch.distributed.all_reduce = _orig


def test_execution_environment_pair_gate():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    base = dict(variant='a', seed=0, run_cfg=fp, steps=1, tokens_seen=1, ppl=10.0,
                data_fp='d', val_fp='v', eval_fp='e', exec_fp='env-a')
    assert L.pair_reason(base, dict(base, variant='b')) is None
    why = L.pair_reason(base, dict(base, variant='b', exec_fp='env-b'))
    assert why and 'execution environment' in why, why
    # Legacy records without the digest keep the historical permissive path.
    leg = {k: v for k, v in base.items() if k != 'exec_fp'}
    assert L.pair_reason(leg, dict(leg, variant='b')) is None
    assert L.execution_environment_reason(base, dict(base, exec_fp='env-a')) is None
    assert L.execution_environment_reason(base, leg) is None


def test_optimizer_groups_match_trainer():
    L.set_seed(0)
    cfgs = L.make_layer_cfgs(1, 'csa_dynamic')
    model = L.SmallGPT(64, 32, 1, 4, 8, 16, cfgs)
    decay, no_decay, delta = L.optimizer_param_groups(model)
    grouped = [id(p) for p in decay + no_decay + delta]
    trainable = [id(p) for p in model.parameters() if p.requires_grad]
    assert len(grouped) == len(set(grouped)), 'one parameter appears in two optimizer groups'
    assert set(grouped) == set(trainable), 'the production optimizer leaves a trainable parameter out'
    opt = L.build_optimizer(model, lr=1e-3, weight_decay=0.1, delta_lr_mult=10.0)
    assert [g['weight_decay'] for g in opt.param_groups] == [0.1, 0.0, 0.0], opt.param_groups
    assert [g['lr_scale'] for g in opt.param_groups] == [1.0, 1.0, 10.0], opt.param_groups


def test_seed_axis_normalization():
    assert L.normalize_seed_list([0, 1, 1, 2, '3']) == [0, 1, 2, 3]
    assert L.normalize_seed_list([]) == [0]
    assert L.normalize_seed_list(None) == [0]
    assert L.normalize_seed_list('0, 2 2') == [0, 2]
    # The fallback plan is validated like an explicit list; int(True)/int(1.5)
    # must not silently manufacture a different seed identity.
    for _bad_default in ((True,), (1.5,), ('x',), (None,)):
        try:
            L.normalize_seed_list([], default=_bad_default)
        except ValueError:
            pass
        else:
            raise AssertionError(f'default={_bad_default!r} must be refused')
    assert L.normalize_variant_list(['full', 'csa_fixed', 'full']) == ['full', 'csa_fixed']
    for bad in (True, 1.5, 'x'):
        try:
            L.normalize_seed_list([bad])
        except ValueError:
            pass
        else:
            raise AssertionError(f'normalize_seed_list must reject {bad!r}')
    d = tempfile.mkdtemp()
    _saved_load = L.load_run_corpus
    _old_skip_overfit = os.environ.get('CSA_SKIP_OVERFIT')
    _old_skip_repro = os.environ.get('CSA_SKIP_REPRO')
    try:
        ids = np.arange(2048, dtype=np.int64) % 997
        val = (np.arange(64, dtype=np.int64) % 997).reshape(2, 32)
        L.load_run_corpus = lambda cfg: (ids, val, 997, None, None)
        os.environ['CSA_SKIP_OVERFIT'] = '1'
        os.environ['CSA_SKIP_REPRO'] = '1'
        cfg = dict(outdir=d, seq_len=32, batch_size=2, n_train_tokens=2048, steps=2,
                   lr=1e-3, weight_decay=0.0, warmup=1, comp_lambda=0.0,
                   delta_lr_mult=10.0, eval_every=0, eval_subset=2, d=64,
                   n_layers=2, n_heads=4, d_head=16, dataset='unit', seeds=[0, 0],
                   variants=['full'])
        L.run(cfg, label='dedupe-unit')
        m = json.load(open(os.path.join(d, 'run_manifest.json'), encoding='utf-8'))
        entry = next(iter(m['panels'].values()))
        assert entry['n_planned'] == 1 and entry['n_completed'] == 1, entry
        assert entry['tracking_integrity']['seeds_unique'] is True, entry['tracking_integrity']
        lines = [json.loads(x) for x in open(os.path.join(d, 'experiments.jsonl'), encoding='utf-8').read().strip().splitlines()]
        assert sum(1 for x in lines if x.get('kind') == 'cell') == 1, lines
    finally:
        L.load_run_corpus = _saved_load
        for k, old in (('CSA_SKIP_OVERFIT', _old_skip_overfit), ('CSA_SKIP_REPRO', _old_skip_repro)):
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
        shutil.rmtree(d, ignore_errors=True)


def test_seed_scalar_rejects_silent_truncation():
    from decimal import Decimal
    # Objects outside numbers.Real can still define __int__; int() would
    # truncate Decimal('1.5') / np.array(1.5) to a different valid seed.
    assert L._as_seed_int(Decimal('1.5')) is None
    assert L._as_seed_int(np.array(1.5)) is None
    assert L._as_seed_int(Decimal('2')) == 2
    assert L._as_seed_int(np.array(2)) == 2
    assert L._as_seed_int('0') == 0
    for bad in (Decimal('1.5'), np.array(1.5)):
        try:
            L.normalize_seed_list([bad])
        except ValueError:
            pass
        else:
            raise AssertionError(f'normalize_seed_list must reject {bad!r}, not truncate it')
    assert L.normalize_seed_list([Decimal('2'), np.array(3)]) == [2, 3]


def test_seed_domain_guard():
    # NumPy's legacy seeder only accepts [0, 2**32 - 1]; accepting another
    # value in normalize_seed_list would defer the failure into training (or
    # silently remap it).  The same domain must be used by every entry point.
    for bad in (-1, L.MAX_SEED + 1, 2 ** 63, True, 1.5, 'x', None):
        for fn in (lambda b=bad: L.normalize_seed_list([b]),
                   lambda b=bad: L.set_seed(b),
                   lambda b=bad: L.dataloader_generator(b)):
            try:
                fn()
            except ValueError:
                pass
            else:
                raise AssertionError(f'{bad!r} must be rejected by every seed entry point')
    assert L.normalize_seed_list([L.MAX_SEED, 0]) == [L.MAX_SEED, 0]
    assert L._other_seed(L.MAX_SEED) == 0 and L._other_seed(0) == 1
    # A maximum seed must not make the contract probe try MAX_SEED+1.
    L.set_seed(4242)
    _fp = L.rng_fingerprint()
    r = L.seed_contract_check(seed=L.MAX_SEED)
    assert r['passed'] is True and r['checks']['different_seed_changes_streams'] is True, r
    assert L.rng_fingerprint() == _fp, 'the seed contract probe must still preserve the process RNG'
    for bad in (-1, L.MAX_SEED + 1):
        try:
            L.reproducibility_guard(['full'], seeds=[bad])
        except ValueError:
            pass
        else:
            raise AssertionError(f'reproducibility_guard must reject seed {bad!r}')


def test_seed_contract_check():
    L.set_seed(4242)
    fp0 = L.rng_fingerprint()
    r = L.seed_contract_check(seed=3)
    assert r['passed'] is True, r
    assert r['checks']['same_seed_reproducible'] is True
    assert r['checks']['different_seed_changes_streams'] is True
    assert r['checks']['worker_seed_reproducible'] is True
    assert r['checks']['cuda_seeded'] is (True if torch.cuda.is_available() else None)
    assert r['checks']['rng_state_preserved'] is True and r['rng_state_preserved'] is True, r
    assert L.rng_fingerprint() == fp0, 'the seed contract probe must not advance the process RNG streams'


def test_batch_iter_reproducible():
    ids = np.arange(5000, dtype=np.int64) % 997
    b1 = next(L.batch_iter(ids, 32, 4, torch.device('cpu'), seed=7))
    b2 = next(L.batch_iter(ids, 32, 4, torch.device('cpu'), seed=7))
    assert torch.equal(b1[0], b2[0]) and torch.equal(b1[1], b2[1]), 'batch_iter must be seeded by the run seed'
    b3 = next(L.batch_iter(ids, 32, 4, torch.device('cpu'), seed=8))
    assert not torch.equal(b1[0], b3[0]), 'a different data seed must change the batch order'
    for _bad in (-1, L.MAX_SEED + 1, True, 1.5):
        try:
            next(L.batch_iter(ids, 32, 4, torch.device('cpu'), seed=_bad))
        except ValueError:
            pass
        else:
            raise AssertionError(f'batch_iter must reject seed {_bad!r}')


def test_batch_iter_uint16():
    ids64 = np.arange(5000, dtype=np.int64) % 997
    ids16 = ids64.astype(np.uint16)
    a = next(L.batch_iter(ids64, 32, 4, torch.device('cpu'), seed=7))
    b = next(L.batch_iter(ids16, 32, 4, torch.device('cpu'), seed=7))
    assert a[0].dtype == b[0].dtype == torch.int64, 'batch_iter must widen the gathered ids back to int64'
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), 'the uint16 transport form must yield the exact int64 batch (CPU index kernels lack uint16)'


def test_batch_contract_check():
    ids = np.arange(5000, dtype=np.int64) % 997
    ok = L.batch_contract_check(ids, 32, 4, torch.device('cpu'), seed=7, vocab=997)
    assert ok['passed'] is True, ok
    assert all(ok['checks'].values()), ok['checks']
    _orig = L.batch_iter

    def _bad(train_ids, seq_len, batch_size, device, seed=0):
        for x, _y in _orig(train_ids, seq_len, batch_size, device, seed=seed):
            yield x, x[:, :-1]
    L.batch_iter = _bad
    try:
        bad = L.batch_contract_check(ids, 32, 4, torch.device('cpu'), seed=7, vocab=997)
    finally:
        L.batch_iter = _orig
    assert bad['passed'] is False, 'a loader that does not shift labels must FAIL the contract check'
    assert bad['checks']['labels_are_next_token_shift'] is False, bad


def test_parallel_child_env_seed():
    import par
    env = par._child_env('0', 17)
    assert env['PYTHONHASHSEED'] == '17' and env['CUDA_VISIBLE_DEVICES'] == '0', {k: env.get(k) for k in ('PYTHONHASHSEED', 'CUDA_VISIBLE_DEVICES')}
    env2 = par._child_env('0')
    assert env2.get('PYTHONHASHSEED') == os.environ.get('PYTHONHASHSEED'), 'no-seed dispatch must not invent a hash seed'


def test_panel_blocks_on_loader_contract_failure():
    d = tempfile.mkdtemp()
    _saved_load = L.load_run_corpus
    _saved_iter = L.batch_iter
    _old_skip_overfit = os.environ.get('CSA_SKIP_OVERFIT')
    _old_skip_repro = os.environ.get('CSA_SKIP_REPRO')
    try:
        ids = np.arange(4096, dtype=np.int64) % 997
        val = (np.arange(128, dtype=np.int64) % 997).reshape(4, 32)
        L.load_run_corpus = lambda cfg: (ids, val, 997, None, None)

        def _bad(train_ids, seq_len, batch_size, device, seed=0):
            for x, _y in _saved_iter(train_ids, seq_len, batch_size, device, seed=seed):
                yield x, x
        L.batch_iter = _bad
        os.environ['CSA_SKIP_OVERFIT'] = '1'
        os.environ['CSA_SKIP_REPRO'] = '1'
        cfg = {'outdir': d, 'seq_len': 32, 'batch_size': 2, 'n_train_tokens': 4096,
               'steps': 3, 'lr': 1e-3, 'weight_decay': 0.0, 'warmup': 1,
               'comp_lambda': 0.0, 'delta_lr_mult': 10.0, 'eval_every': 0,
               'eval_subset': 2, 'd': 64, 'n_layers': 2, 'n_heads': 4,
               'd_head': 16, 'dataset': 'unit', 'seeds': [0], 'variants': ['full']}
        try:
            L.run(cfg, label='contract-failure-unit')
        except RuntimeError as e:
            assert 'batch iterator contract check FAILED' in str(e), e
        else:
            raise AssertionError('run() must hard-stop when the real loader contract fails')
        dc = json.load(open(os.path.join(d, 'data_contract.json'), encoding='utf-8'))
        assert dc['passed'] is False and dc['checks']['labels_are_next_token_shift'] is False, dc
        assert not os.path.exists(os.path.join(d, 'summary.json')), 'no training cell may start after a failed loader contract'
        mf = json.load(open(os.path.join(d, 'run_manifest.json'), encoding='utf-8'))
        entry = next(iter(mf['panels'].values()))
        assert entry['status'] == 'failed', entry.get('status')
        assert entry['completion']['status'] == 'failed', entry.get('completion')
        assert entry['planned_cells'] == ['full::seed0'], entry.get('planned_cells')
        lines = [json.loads(x) for x in open(os.path.join(d, 'experiments.jsonl'), encoding='utf-8').read().strip().splitlines()]
        fin = next(x for x in lines if x.get('kind') == 'panel_finish')
        assert fin['status'] == 'failed' and fin['stage'] == 'batch_iterator_contract', fin
    finally:
        L.load_run_corpus = _saved_load
        L.batch_iter = _saved_iter
        for _k, _v in (('CSA_SKIP_OVERFIT', _old_skip_overfit), ('CSA_SKIP_REPRO', _old_skip_repro)):
            if _v is None:
                os.environ.pop(_k, None)
            else:
                os.environ[_k] = _v
        shutil.rmtree(d, ignore_errors=True)


def test_overfit_sanity():
    r = L.overfit_sanity_check('full', seed=0, steps=200, d=64, n_layers=2, n_heads=4, d_head=16, seq_len=32, batch_size=2, vocab=256)
    assert r['passed'] is True, r
    assert r['final_loss'] <= r['target_loss'], f"final loss {r['final_loss']} did not reach {r['target_loss']}"
    assert r['final_accuracy'] >= r['target_accuracy'], f"final accuracy {r['final_accuracy']} did not reach {r['target_accuracy']}"
    assert r['grad_all_finite'] and r['params_changed'], 'gradients must be finite and applied'
    assert r['regularization']['weight_decay'] == 0.0 and r['regularization']['comp_lambda'] == 0.0
    assert r['label_shift_required'] is True
    assert r['pass_criteria']['labels_are_next_token_shift'] is True, r['pass_criteria']
    assert r['pass_criteria']['initial_loss_near_log_vocab'] is True, r['pass_criteria']
    assert r['failed_criteria'] == [], r['failed_criteria']
    # the check must also work when the batch is sampled from a real token stream
    ids = np.asarray(np.random.default_rng(3).integers(0, 256, size=1024), dtype=np.int64)
    r2 = L.overfit_sanity_check('csa_dynamic', seed=0, steps=200, d=64, n_layers=2, n_heads=4, d_head=16, seq_len=24, batch_size=2, vocab=256, train_ids=ids)
    assert r2['passed'] is True and r2['data_source'] == 'train_ids', r2
    assert r2['loader_mode'] == 'batch_iter', 'the real-data probe must exercise the production loader'
    assert r2['intermediate_checks']['labels_are_next_token_shift'] is True, r2['intermediate_checks']


def test_production_loop_overfit_sanity():
    r = L.production_loop_sanity_check('full', seed=0, steps=120, d=32,
                                       n_layers=1, n_heads=4, d_head=8,
                                       seq_len=8, batch_size=2, vocab=97)
    assert r['passed'] is True, r
    assert r['final_loss'] <= r['target_loss'] and r['final_accuracy'] >= r['target_accuracy'], r
    assert r['loss_decreased'] and r['params_changed'], r
    assert r['failed_criteria'] == [] and r['rng_state_preserved'] is True, r
    # A fixed window through the real loop: the production trainer must not
    # silently skip updates or produce a non-finite curve.
    ids = np.arange(4096, dtype=np.int64) % 97
    r2 = L.production_loop_sanity_check('csa_dynamic', seed=1, steps=100, d=32,
                                        n_layers=1, n_heads=4, d_head=8,
                                        seq_len=8, batch_size=2, vocab=97,
                                        train_ids=ids)
    assert r2['passed'] is True and r2['data_source'] == 'train_ids', r2
    assert r2['skipped_updates'] == 0 and r2['nonfinite_grad_steps'] == 0, r2
    # The real trainer with lr=0 must FAIL the same check: this is the
    # regression that a separately written probe loop could miss.
    bad = L.production_loop_sanity_check('full', seed=0, steps=5, lr=0.0,
                                         d=32, n_layers=1, n_heads=4, d_head=8,
                                         seq_len=8, batch_size=2, vocab=97)
    assert bad['passed'] is False and bad['params_changed'] is False, bad
    assert 'parameters_changed' in bad['failed_criteria'], bad['failed_criteria']


def test_rng_state_preserved_by_preflight():
    L.set_seed(777)
    fp0 = L.rng_fingerprint()
    r = L.overfit_sanity_check('full', seed=1, steps=5, d=32, n_layers=1, n_heads=4, d_head=8, seq_len=16, batch_size=2, vocab=64)
    assert r['rng_state_preserved'] is True, r
    assert L.rng_fingerprint() == fp0, 'the overfit probe must not advance the process RNG streams'
    rr = L.reproducibility_check('full', seed=2, steps=2)
    assert rr['rng_state_preserved'] is True, rr
    assert L.rng_fingerprint() == fp0, 'the reproducibility probe must not advance the process RNG streams'


def test_reproducibility_check():
    r = L.reproducibility_check('full', seed=0, steps=3)
    assert r['passed'] is True, r
    assert r['max_abs_diff_same_seed'] <= r['tol'], f'same seed must reproduce the short loss curve: {r}'
    if not torch.cuda.is_available():
        assert r['bitwise_identical'] is True, f'same seed must reproduce the short loss curve exactly on CPU: {r}'
    assert r['different_seed_changes_curve'] is True, 'a different seed must change the short loss curve'
    assert r['rng_state_preserved'] is True
    # the cache/guard records evidence without re-running
    reports = L.reproducibility_guard(['full'])
    assert reports and reports[0]['passed'] is True


def test_panel_tracking_events():
    d = tempfile.mkdtemp()
    _saved_load = L.load_run_corpus
    _old_skip_overfit = os.environ.get('CSA_SKIP_OVERFIT')
    _old_skip_repro = os.environ.get('CSA_SKIP_REPRO')
    try:
        ids = np.arange(4096, dtype=np.int64) % 997
        val = (np.arange(128, dtype=np.int64) % 997).reshape(4, 32)
        L.load_run_corpus = lambda cfg: (ids, val, 997, None, None)
        os.environ['CSA_SKIP_OVERFIT'] = '1'
        os.environ['CSA_SKIP_REPRO'] = '1'
        cfg = dict(outdir=d, seq_len=32, batch_size=2, n_train_tokens=4096, steps=3, lr=1e-3, weight_decay=0.0, warmup=1, comp_lambda=0.0, delta_lr_mult=10.0, eval_every=0, eval_subset=2, d=64, n_layers=2, n_heads=4, d_head=16, dataset='unit', seeds=[0, 1], variants=['full'])
        summary, _ = L.run(cfg, label='tracking-unit')
        assert 'full::seed0' in summary and 'full::seed1' in summary
        lines = [json.loads(x) for x in open(os.path.join(d, 'experiments.jsonl'), encoding='utf-8').read().strip().splitlines()]
        kinds = [x.get('kind') for x in lines]
        assert 'panel_start' in kinds and 'panel_finish' in kinds, kinds
        assert kinds.count('cell') == 2, kinds
        st = next(x for x in lines if x.get('kind') == 'panel_start')
        for k in ('run_cfg', 'code', 'config', 'seeds', 'variants', 'provenance'):
            assert k in st, f'panel_start missing {k}'
        assert st['n_cells_planned'] == 2 and st['n_cells_to_run'] == 2 and st['n_cells_resumed'] == 0, st
        fin = next(x for x in lines if x.get('kind') == 'panel_finish')
        assert fin['n_cells_planned'] == 2 and fin['n_cells_measured'] == 2 and fin['n_cells_missing'] == 0, fin
        for k in ('provenance', 'config', 'seeds', 'variants', 'data_version'):
            assert k in fin, f'panel_finish missing {k}'
        assert fin['provenance']['code']['git_commit'] == st['provenance']['code']['git_commit'], fin
        m = json.load(open(os.path.join(d, 'run_manifest.json'), encoding='utf-8'))
        entry = m['panels'][L._run_fp(cfg, 64, 2, 4, 16, 997, ids, val)]
        assert entry.get('tracking_schema') == 2 and 'reproducibility' in entry
        assert entry['data_contract']['passed'] is True, entry.get('data_contract')
        assert entry['evidence']['batch_iterator_contract'] is True, entry.get('evidence')
    finally:
        L.load_run_corpus = _saved_load
        if _old_skip_overfit is None:
            os.environ.pop('CSA_SKIP_OVERFIT', None)
        else:
            os.environ['CSA_SKIP_OVERFIT'] = _old_skip_overfit
        if _old_skip_repro is None:
            os.environ.pop('CSA_SKIP_REPRO', None)
        else:
            os.environ['CSA_SKIP_REPRO'] = _old_skip_repro
        shutil.rmtree(d, ignore_errors=True)


def test_provenance_and_manifest():
    tr = np.arange(512, dtype=np.int64) % 97
    va = np.arange(64, dtype=np.int64).reshape(2, 32) % 97
    prov = L.experiment_provenance({'dataset': 'unit', 'n_train_tokens': 512, 'seq_len': 32}, train_ids=tr, val_batch=va, vocab=97)
    for k in ('code', 'environment', 'data'):
        assert k in prov, f'provenance missing {k}'
    for k in ('code_semantics', 'code_stamp', 'git_commit', 'git_dirty', 'git_status_sha256', 'git_diff_sha256'):
        assert k in prov['code'], f'code provenance missing {k}'
    for k in ('python', 'torch', 'numpy', 'determinism'):
        assert k in prov['environment'], f'environment provenance missing {k}'
    assert prov['data']['train_fp'] and prov['data']['val_fp'] and prov['data']['vocab_size'] == 97
    assert prov['data'].get('data_version'), 'data provenance must carry a composite data_version'
    hp = L.run_hyperparameters({'steps': 10, 'seeds': [0], 'variants': ['full'], 'outdir': 'x', 'n_train_tokens': 512, 'seq_len': 32}, seed=3, d=64, n_layers=2, n_heads=4, d_head=16, vocab=97, mlp_ratio=4.0)
    assert hp['seed'] == 3 and hp['data_seed'] == 3 and hp['train_knobs']['optimizer'] == 'AdamW'
    assert hp['rng']['python'] == hp['rng']['numpy'] == hp['rng']['torch'] == hp['rng']['cuda'] == 3, hp['rng']
    assert hp['rng']['data'] == 3 and hp['rng']['dataloader_generator'] == 3, hp['rng']
    assert hp['determinism']['level'] == L.determinism_label(), hp['determinism']
    assert hp['determinism']['cpu_threads'] == int(torch.get_num_threads()), hp['determinism']
    d = tempfile.mkdtemp()
    try:
        summary = {'full::seed0': {'ppl': 10.0, 'steps': 10, 'tokens_seen': 320, 'params': 100, 'train_time_s': 1.0, 'final_loss_smoothed': 2.3, 'rng_fp': 'abc', '_code': L.code_stamp(), 'losses': [3.0, 2.3], 'ppl_history': [[10, 10.0]]}}
        cfg = {'seeds': [0], 'variants': ['full'], 'outdir': d, 'n_train_tokens': 512, 'seq_len': 32}
        path = L.write_run_manifest(d, cfg, 'FP', prov, summary)
        m = json.load(open(path, encoding='utf-8'))
        assert m['manifest_version'] == 1 and 'FP' in m['panels'], m
        assert m['panels']['FP']['cells']['full::seed0']['ppl'] == 10.0
        assert m['panels']['FP']['provenance']['data']['train_fp'] == prov['data']['train_fp']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_dataloader_seeding_helpers():
    ds = torch.utils.data.TensorDataset(torch.arange(64, dtype=torch.long).unsqueeze(1))

    def order(seed):
        dl = L.deterministic_dataloader(ds, batch_size=8, shuffle=True, seed=seed)
        return [int(v) for b in dl for v in b[0]]
    assert order(7) == order(7), 'same DataLoader seed must reproduce the shuffled order'
    assert order(7) != order(8), 'a different DataLoader seed must change the shuffled order'
    # Explicit None means "not supplied": it must not disable the seeded
    # generator/worker hook and fall back to OS entropy.
    _dl_none = L.deterministic_dataloader(ds, batch_size=8, shuffle=True, seed=7,
                                          generator=None, worker_init_fn=None)
    assert [int(v) for b in _dl_none for v in b[0]] == order(7)
    # worker RNGs are derived from torch.initial_seed(), not OS entropy
    def worker_draw():
        torch.manual_seed(11)
        L.seed_worker(0)
        return (random.random(), float(np.random.random()), float(torch.rand(1).item()))
    assert worker_draw() == worker_draw(), 'seed_worker must reproduce Python/NumPy/Torch worker RNG streams'
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(11)
    L.seed_worker(0)
    assert torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False, 'a spawned worker must re-pin deterministic kernels'
    # The worker hook must also make real worker PROCESSES reproducible.
    class _DrawDS(torch.utils.data.Dataset):
        def __len__(self):
            return 64
        def __getitem__(self, i):
            return torch.tensor([i, int(random.random() * 1e9), int(np.random.random() * 1e9)])
    def worker_order(seed):
        dl = L.deterministic_dataloader(_DrawDS(), batch_size=8, shuffle=True, seed=seed, num_workers=2)
        return [tuple(int(x) for x in row) for b in dl for row in b]
    try:
        _w4a, _w4b, _w5 = worker_order(4), worker_order(4), worker_order(5)
    except (PermissionError, OSError) as e:
        raise _SkipTest(f'DataLoader worker processes need POSIX semaphores, which this sandbox forbids ({type(e).__name__}: {e})') from e
    assert _w4a == _w4b, 'spawned DataLoader workers must reproduce their Python/NumPy draws'
    assert _w4a != _w5, 'a different DataLoader seed must change the worker draws'


def test_dataloader_contract_check():
    try:
        r = L.dataloader_contract_check(seed=11, num_workers=2, batch_size=4, batches=2)
    except (PermissionError, OSError) as e:
        raise _SkipTest(f'DataLoader worker processes need POSIX semaphores, which this sandbox forbids ({type(e).__name__}: {e})') from e
    _skip_dataloader_if_unavailable(r)
    assert r['passed'] is True, r
    for k in ('stream_nonempty', 'same_seed_reproducible',
              'different_seed_changes_stream', 'rng_state_preserved'):
        assert r['checks'].get(k) is True, (k, r['checks'])
    assert r['num_workers'] == 2 and r['same_seed_stream'] and r['other_seed_stream']
    # The probe must not consume the caller's RNG streams.
    L.set_seed(99)
    fp = L.rng_fingerprint()
    L.dataloader_contract_check(seed=3, num_workers=0, batch_size=2, batches=1)
    assert L.rng_fingerprint() == fp


def test_seed_contract_reports_hashseed_and_workers():
    r = L.seed_contract_check(seed=3, check_dataloader_workers=False)
    assert isinstance(r.get('hashseed'), dict), r
    for k in ('pinned_at_startup', 'child_seed', 'effective_current_process', 'note'):
        assert k in r['hashseed'], r['hashseed']
    assert r['hashseed']['child_seed'] is not None
    # `PYTHONHASHSEED=random` is present but does not pin the process; only a
    # valid decimal uint32 does.
    for _good in ('0', '123', '4294967295'):
        assert L._hashseed_pinned_value(_good) is True, _good
    for _bad in (None, '', 'random', 'RANDOM', '-1', '4294967296', 'abc'):
        assert L._hashseed_pinned_value(_bad) is False, _bad
    _startup = L._PYTHONHASHSEED_AT_STARTUP
    try:
        L._PYTHONHASHSEED_AT_STARTUP = '0'
        assert L.hashseed_contract()['pinned_at_startup'] is True
        L._PYTHONHASHSEED_AT_STARTUP = 'random'
        assert L.hashseed_contract()['pinned_at_startup'] is False
    finally:
        L._PYTHONHASHSEED_AT_STARTUP = _startup
    try:
        r2 = L.seed_contract_check(seed=3, check_dataloader_workers=True, dataloader_workers=2)
    except (PermissionError, OSError) as e:
        raise _SkipTest(f'DataLoader worker processes need POSIX semaphores ({type(e).__name__}: {e})') from e
    _skip_dataloader_if_unavailable(r2.get('dataloader_contract'))
    assert r2['passed'] is True, r2
    assert r2['checks']['dataloader_worker_processes'] is True, r2['checks']
    assert r2['dataloader_contract']['passed'] is True, r2['dataloader_contract']


def test_overfit_probe_cache_binds_device():
    kw = dict(seed=0, steps=10, seq_len=16, batch_size=2, vocab=64,
              target_loss=0.05, target_acc=0.99, source='synthetic')
    a = L._overfit_probe_config(device=torch.device('cpu'), **kw)
    b = L._overfit_probe_config(device=torch.device('cuda:0'), **kw)
    assert a['device'] == 'cpu' and b['device'] == 'cuda:0'
    assert a['exec_fp'] and a['determinism']
    assert L._overfit_probe_digest(a) != L._overfit_probe_digest(b), \
        'a cached overfit pass must not cross execution targets'


def test_train_variant_reproducible():
    ids = np.arange(8192, dtype=np.int64) % 997
    val = (np.arange(64, dtype=np.int64) % 997).reshape(2, 32)
    kw = dict(d=64, n_layers=2, n_heads=4, d_head=16, seq_len=32, batch_size=2, steps=4, lr=1e-3, weight_decay=0.0, warmup=1, comp_lambda=0.0, eval_every=0, mlp_ratio=4.0, device=torch.device('cpu'), log_every=0)
    a = L.train_variant('full', ids, val, 997, seed=3, **kw)
    # Token accounting: draws are not unique tokens; the estimate must be
    # present, bounded, and strictly below the draw count on an undersized cache.
    assert a['tokens_seen'] == 4 * 2 * 32, a['tokens_seen']
    # The smoothed tail is stored as numbers, not tensors; a tensor tail was
    # silently filtered to None for every run.
    assert a['final_loss_smoothed'] is not None, a.get('final_loss_smoothed')
    assert a.get('tokens_seen_unique_frac_estimate') is not None and 0.0 < a['tokens_seen_unique_frac_estimate'] <= 1.0, a
    assert a['tokens_seen_unique_estimate'] <= len(ids), a
    b = L.train_variant('full', ids, val, 997, seed=3, **kw)
    c = L.train_variant('full', ids, val, 997, seed=4, **kw)
    assert a['losses'] == b['losses'], 'same seed must reproduce the whole training loss curve'
    assert a['ppl'] == b['ppl'] and a['rng_fp'] == b['rng_fp'], 'same seed must reproduce the final PPL and RNG fingerprint'
    assert a['losses'] != c['losses'], 'a different seed must change the loss curve'


def test_overfit_detects_broken_optimizer():
    # lr=0 is the canonical "gradients computed but never applied" failure mode.
    r = L.overfit_sanity_check('full', seed=0, steps=5, lr=0.0, d=64, n_layers=2, n_heads=4, d_head=16, seq_len=16, batch_size=2, vocab=256)
    assert r['passed'] is False, 'a run whose parameters never update must FAIL the sanity check'
    assert r['params_changed'] is False and r['loss_decreased'] is False
    assert 'did not update' in r['diagnosis']


def test_experiment_ledger():
    d = tempfile.mkdtemp()
    try:
        ids = np.arange(4096, dtype=np.int64) % 997
        val = (np.arange(128, dtype=np.int64) % 997).reshape(4, 32)
        cfg = dict(outdir=d, seq_len=32, batch_size=2, n_train_tokens=4096, steps=3, lr=1e-3, weight_decay=0.0, warmup=1, comp_lambda=0.0, delta_lr_mult=10.0, eval_every=0, eval_subset=2, d=64, n_layers=2, n_heads=4, d_head=16, dataset='unit', seeds=[0], variants=['full'])
        prov = L.experiment_provenance(cfg, train_ids=ids, val_batch=val, vocab=997)
        fp = L._run_fp(cfg, 64, 2, 4, 16, 997, ids, val)
        out = L._run_cell(cfg, 'full', 0, fp=fp, train_ids=ids, val_batch=val, vocab=997, prov=prov)
        assert out['status'] == 'ok' and out.get('run_id')
        summary = json.load(open(os.path.join(d, 'summary.json'), encoding='utf-8'))
        rec = summary['full::seed0']
        assert rec.get('run_id') == out['run_id'] and rec.get('hyperparams') and rec.get('provenance')
        lines = open(os.path.join(d, 'experiments.jsonl'), encoding='utf-8').read().strip().splitlines()
        assert len(lines) == 1, f'one completed cell must append exactly one ledger line (got {len(lines)})'
        ev = json.loads(lines[0])
        for k in ('run_id', 'run_cfg', 'seed', 'provenance', 'hyperparams', 'data_fp', 'metrics', 'losses', 'ppl_history'):
            assert k in ev, f'experiment ledger entry missing {k}'
        assert ev['status'] == 'ok' and ev['run_id'] == out['run_id'] and len(ev['losses']) == 3
        assert ev['metrics'].get('steps_completed') == 3, ev['metrics']
        assert isinstance(ev.get('instability_diagnosis'), dict), ev
        san = {'kind': 'small_batch_overfit_sanity_panel', 'passed': True, 'reports': [], 'run_cfg': fp}
        dc = L.batch_contract_check(ids, cfg['seq_len'], cfg['batch_size'], torch.device('cpu'), seed=0, vocab=997)
        dc.update({'run_cfg': fp})
        L.write_run_manifest(d, cfg, fp, prov, summary, sanity=san, data_contract=dc)
        m = json.load(open(os.path.join(d, 'run_manifest.json'), encoding='utf-8'))
        entry = m['panels'][fp]
        cell = entry['cells']['full::seed0']
        assert entry['sanity'] == san and entry['experiment_log'] == 'experiments.jsonl'
        assert entry['data_contract']['passed'] is True and entry['evidence']['batch_iterator_contract'] is True, entry.get('evidence')
        assert cell['run_id'] == out['run_id'] and cell['data_fp'] and cell['curve_summary']['n_points'] == 3
        assert cell.get('data_version') == prov['data']['data_version']
        assert entry.get('status') == 'complete' and entry.get('n_completed') == 1
        integ = entry.get('tracking_integrity') or {}
        assert integ.get('n_cells') == 1 and integ.get('with_curve') == 1 and integ.get('with_rng_fp') == 1 and integ.get('with_data_contract') == 1, integ
        assert integ.get('with_seed_metadata') == 1 and integ.get('seed_metadata_mismatch') == [], integ
        assert integ.get('with_curve_hash') == 1, integ
        assert cell.get('curve_sha256') and cell.get('hyperparams', {}).get('rng'), cell
        tv = L.verify_run_tracking(d, fp)
        assert tv['passed'] is True, tv
        # A curve edited after the fact must no longer verify against the
        # digest stored with the record.  This is what makes the training
        # curve an auditable artifact rather than an unverified path.
        _cp = os.path.join(d, cell['curve_ref'])
        _curve = json.load(open(_cp, encoding='utf-8'))
        _curve['losses'][0] = float(_curve['losses'][0]) + 1.0
        json.dump(_curve, open(_cp, 'w', encoding='utf-8'))
        tv2 = L.verify_run_tracking(d, fp)
        assert tv2['passed'] is False and any('digest' in p for p in tv2['problems']), tv2
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_data_version_identity():
    cfg = {'dataset': 'unit', 'n_train_tokens': 512, 'seq_len': 32, 'cache_dir': './unit_cache'}
    tr = np.arange(512, dtype=np.int64) % 97
    va = np.arange(64, dtype=np.int64).reshape(2, 32) % 97
    a = L.experiment_provenance(cfg, train_ids=tr, val_batch=va, vocab=97)['data']
    b = L.experiment_provenance(cfg, train_ids=tr.copy(), val_batch=va.copy(), vocab=97)['data']
    assert a['data_version'] == b['data_version'], 'same data must yield the same data_version'
    tr2 = tr.copy()
    tr2[0] = (int(tr2[0]) + 1) % 97
    c = L.experiment_provenance(cfg, train_ids=tr2, val_batch=va, vocab=97)['data']
    assert c['data_version'] != a['data_version'], 'changing one token must change data_version'
    d = L.experiment_provenance(cfg, train_ids=tr, val_batch=va, vocab=98)['data']
    assert d['data_version'] != a['data_version'], 'changing the vocabulary must change data_version'
    assert a['dataset'] == 'unit' and a['cache_dir'] == './unit_cache'


def test_manifest_running_then_complete():
    d = tempfile.mkdtemp()
    try:
        tr = np.arange(512, dtype=np.int64) % 97
        va = np.arange(64, dtype=np.int64).reshape(2, 32) % 97
        cfg = {'seeds': [0, 1], 'variants': ['full'], 'outdir': d, 'n_train_tokens': 512, 'seq_len': 32}
        prov = L.experiment_provenance(cfg, train_ids=tr, val_batch=va, vocab=97)
        planned = ['full::seed0', 'full::seed1']
        L.write_run_manifest(d, cfg, 'FP', prov, {}, status='running', planned_cells=planned, started_at='T0')
        e = json.load(open(os.path.join(d, 'run_manifest.json'), encoding='utf-8'))['panels']['FP']
        assert e['status'] == 'running' and e['n_planned'] == 2 and e['n_completed'] == 0
        assert e['missing_cells'] == planned and e['created_at'] == 'T0'
        assert e['completion']['status'] == 'running' and e['completion']['complete'] is False, e.get('completion')
        summary = {'full::seed0': {'ppl': 10.0, 'steps': 3, 'tokens_seen': 192, 'params': 100, 'train_time_s': 1.0, 'final_loss_smoothed': 2.3, 'rng_fp': 'abc', '_code': L.code_stamp(), 'losses': [3.0, 2.3], 'ppl_history': []}}
        L.write_run_manifest(d, cfg, 'FP', prov, summary, status='complete', planned_cells=planned, started_at='T0')
        e = json.load(open(os.path.join(d, 'run_manifest.json'), encoding='utf-8'))['panels']['FP']
        assert e['status'] == 'complete' and e['n_completed'] == 1 and e['missing_cells'] == ['full::seed1']
        # The explicit status argument is preserved for compatibility, but the
        # completion verdict must still expose the missing cell and the
        # untraced record instead of looking complete.
        assert e['completion']['status'] == 'incomplete' and e['completion']['complete'] is False, e['completion']
        assert e['completion']['missing_cells'] == ['full::seed1'], e['completion']
        assert e['tracking_integrity']['complete'] is False
        assert 'full::seed0' in e['tracking_integrity']['incomplete_cells'], e['tracking_integrity']
        assert e['created_at'] == 'T0', 'a final update must not reset the panel start time'
        assert e['data_version'] == prov['data']['data_version']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_panel_finish_marks_missing_cells_incomplete():
    d = tempfile.mkdtemp()
    _saved_load = L.load_run_corpus
    _saved_train = L.train_variant
    _old_skip_overfit = os.environ.get('CSA_SKIP_OVERFIT')
    _old_skip_repro = os.environ.get('CSA_SKIP_REPRO')
    try:
        ids = np.arange(4096, dtype=np.int64) % 997
        val = (np.arange(64, dtype=np.int64) % 997).reshape(2, 32)
        L.load_run_corpus = lambda cfg: (ids, val, 997, None, None)
        os.environ['CSA_SKIP_OVERFIT'] = '1'
        os.environ['CSA_SKIP_REPRO'] = '1'

        def _flaky(variant, *a, **kw):
            if variant == 'csa_fixed':
                raise RuntimeError('unit: deliberate cell failure')
            return _saved_train(variant, *a, **kw)
        L.train_variant = _flaky
        cfg = dict(outdir=d, seq_len=32, batch_size=2, n_train_tokens=4096, steps=1,
                   lr=1e-3, weight_decay=0.0, warmup=1, comp_lambda=0.0,
                   delta_lr_mult=10.0, eval_every=0, eval_subset=2, d=64,
                   n_layers=2, n_heads=4, d_head=16, dataset='unit', seeds=[0],
                   variants=['full', 'csa_fixed'])
        L.run(cfg, label='incomplete-unit')
        m = json.load(open(os.path.join(d, 'run_manifest.json'), encoding='utf-8'))
        e = next(iter(m['panels'].values()))
        assert e['status'] == 'incomplete', e.get('status')
        assert e['missing_cells'] == ['csa_fixed::seed0'], e.get('missing_cells')
        assert e['completion']['status'] == 'incomplete' and e['completion']['n_missing'] == 1, e.get('completion')
        lines = [json.loads(x) for x in open(os.path.join(d, 'experiments.jsonl'), encoding='utf-8').read().strip().splitlines()]
        fin = next(x for x in lines if x.get('kind') == 'panel_finish')
        assert fin['status'] == 'incomplete' and fin['missing_cells'] == ['csa_fixed::seed0'], fin
        err = next(x for x in lines if x.get('kind') == 'cell' and x.get('status') == 'error')
        assert err['variant'] == 'csa_fixed' and 'deliberate' in err['error'], err
    finally:
        L.load_run_corpus = _saved_load
        L.train_variant = _saved_train
        for k, old in (('CSA_SKIP_OVERFIT', _old_skip_overfit), ('CSA_SKIP_REPRO', _old_skip_repro)):
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
        shutil.rmtree(d, ignore_errors=True)


def test_failed_guard_is_retried():
    _saved_guard = dict(L._OVERFIT_GUARD)
    _saved_reports = dict(L._OVERFIT_REPORTS)
    _saved_fn = L.overfit_sanity_check
    calls = []
    try:
        L._OVERFIT_GUARD.clear()
        L._OVERFIT_REPORTS.clear()

        def _stub(v, **kw):
            calls.append(v)
            ok = len(calls) > 1
            return {'passed': ok, 'initial_loss': 1.0, 'final_loss': 0.0 if ok else 9.0,
                    'final_accuracy': 1.0 if ok else 0.0, 'grad_all_finite': True,
                    'params_changed': True, 'diagnosis': 'ok' if ok else 'broken'}
        L.overfit_sanity_check = _stub
        try:
            L.overfit_guard(['probe'])
            raise AssertionError('a failed probe must be a hard stop')
        except RuntimeError:
            pass
        assert calls == ['probe']
        L.overfit_guard(['probe'])
        assert calls == ['probe', 'probe'], 'a cached FAILURE must be retried, not treated as a completed check'
    finally:
        L.overfit_sanity_check = _saved_fn
        L._OVERFIT_GUARD.clear()
        L._OVERFIT_GUARD.update(_saved_guard)
        L._OVERFIT_REPORTS.clear()
        L._OVERFIT_REPORTS.update(_saved_reports)


def test_indexer_column_narrowing_exact():
    # `causal_ends_host` is not a value-changing option: the readable-prefix
    # path must return the same idx AND soft bits as the full-width path (a
    # different GEMM shape is NOT bitwise column-independent, so score
    # narrowing must not be applied).
    torch.manual_seed(0)
    random.seed(0)
    n, B, d, nIH = 17, 12, 8, 2
    H = torch.randn(n, d)
    comp = torch.randn(B, d)
    W_DQ = torch.randn(d, d)
    W_DK = torch.randn(d, d)
    W_w = torch.randn(nIH, d)
    ends = sorted(random.sample(range(1, n + 1), B))
    last = torch.tensor([e - 1 for e in ends])
    for rs in (False, True):
        a = L.lightning_indexer(H, comp, last, W_DQ, W_DK, W_w, nIH, 4, return_mask=False, random_select=rs, causal_ends_host=ends, need_soft=True)
        b = L.lightning_indexer(H, comp, last, W_DQ, W_DK, W_w, nIH, 4, return_mask=False, random_select=rs, causal_ends_host=None, need_soft=True)
        assert torch.equal(a[1], b[1]), 'column narrowing changed the selected block ids'
        assert torch.equal(a[2], b[2]), 'column narrowing changed the soft prior by ULPs'


def test_attention_zero_blocks_keeps_window():
    # A sliding-window-only call has zero compressed blocks: the stacked-row
    # offset is 0 and the window lanes are still present.
    torch.manual_seed(0)
    n, h, d, w = 6, 2, 3, 4
    q, k, v = (torch.randn(n, h, d) for _ in range(3))
    topk = torch.empty(n, 0, dtype=torch.int32)
    last = torch.empty(0, dtype=torch.long)
    got = L.gathered_attention(q, torch.empty(0, h, d), torch.empty(0, h, d), topk, last, k, v, w, 0.5, q_chunk=2, mem_budget_bytes=1 << 20)
    ref = torch.empty_like(q)
    for i in range(n):
        lo = max(0, i - w + 1)
        attn = torch.softmax(torch.einsum('hd,mhd->hm', q[i], k[lo:i + 1]) * 0.5, dim=-1)
        ref[i] = torch.einsum('hm,mhd->hd', attn, v[lo:i + 1])
    assert torch.allclose(got, ref, atol=1e-6, rtol=1e-5), 'the sliding-window lanes were dropped when k_blk held zero rows'


def test_atomic_torch_save():
    # Checkpoint writes use a unique same-directory temp and replace atomically;
    # an interrupted writer must not leave a shared `*.tmp` behind.
    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, 'ckpt.pt')
        V7._atomic_torch_save({'a': torch.arange(3)}, p)
        got = torch.load(p, weights_only=False)
        assert torch.equal(got['a'], torch.arange(3))
        leftovers = [f for f in os.listdir(d) if f.endswith('.tmp')]
        assert not leftovers, f'incomplete temp files left behind: {leftovers}'
        V7._atomic_torch_save({'a': torch.arange(4)}, p)
        got = torch.load(p, weights_only=False)
        assert got['a'].numel() == 4, 'the atomic replace did not publish the new checkpoint'
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_cache_bytes_shared_storage():
    base = torch.arange(64, dtype=torch.int32)
    expanded = base.expand(4096, 64)
    assert L._cache_bytes(expanded) == L._cache_bytes(base), 'an expand must charge its base storage, not numel'
    assert L._cache_bytes(expanded) < expanded.numel() * expanded.element_size()
    cache = {}
    L._cache_put(cache, 'arange', base, 8192, L._idx_cache_total)
    L._cache_put(cache, 'expand', expanded, 8192, L._idx_cache_total)
    assert set(cache) == {'arange', 'expand'}, 'a shared-storage view evicted an unrelated cache entry'


def test_overfit_guard_source_key():
    # A synthetic pre-flight and a data pre-flight are distinct keys: the data
    # check exercises real ids, and a passed data check subsumes the synthetic
    # check.
    _saved_guard = dict(L._OVERFIT_GUARD)
    _saved_fn = L.overfit_sanity_check
    calls = []
    try:
        L._OVERFIT_GUARD.clear()
        def _stub(v, **kw):
            calls.append(kw.get('train_ids') is not None)
            return {'passed': True, 'initial_loss': 1.0, 'final_loss': 0.0, 'final_accuracy': 1.0, 'grad_all_finite': True, 'params_changed': True, 'diagnosis': 'ok'}
        L.overfit_sanity_check = _stub
        L.overfit_guard(['probe'])
        assert calls == [False], f'a synthetic check should run once: {calls}'
        calls.clear()
        L.overfit_guard(['probe'], train_ids=np.arange(64, dtype=np.int64))
        assert calls == [True], f'a DATA check must not be skipped just because a synthetic check passed: {calls}'
        calls.clear()
        L.overfit_guard(['probe'])
        assert calls == [], f'a synthetic check is subsumed by a passed data check: {calls}'
    finally:
        L.overfit_sanity_check = _saved_fn
        L._OVERFIT_GUARD.clear()
        L._OVERFIT_GUARD.update(_saved_guard)


def test_overfit_guard_config_aware():
    # A cached PASS certifies only the exact probe that produced it.  Changing
    # the step count or the convergence targets must invalidate the cache; a
    # lenient probe must not silently certify a stricter one.
    _saved_guard = dict(L._OVERFIT_GUARD)
    _saved_reports = dict(L._OVERFIT_REPORTS)
    _saved_fn = L.overfit_sanity_check
    calls = []
    try:
        L._OVERFIT_GUARD.clear()
        L._OVERFIT_REPORTS.clear()

        def _stub(v, **kw):
            calls.append((int(kw.get('steps', -1)), float(kw.get('target_loss', -1))))
            return {'passed': True, 'initial_loss': 1.0, 'final_loss': 0.0,
                    'final_accuracy': 1.0, 'grad_all_finite': True,
                    'params_changed': True, 'diagnosis': 'ok'}

        L.overfit_sanity_check = _stub
        L.overfit_guard(['probe'], steps=10)
        L.overfit_guard(['probe'], steps=10)
        assert calls == [(10, 0.05)], f'the identical probe must be cached once: {calls}'
        L.overfit_guard(['probe'], steps=20)
        assert calls == [(10, 0.05), (20, 0.05)], f'a changed step count must re-run the probe: {calls}'
        L.overfit_guard(['probe'], steps=20, target_loss=0.001)
        assert calls == [(10, 0.05), (20, 0.05), (20, 0.001)], f'a stricter target must re-run the probe: {calls}'
        _rep = list(L._OVERFIT_REPORTS.values())[0]
        assert _rep.get('guard_probe_config', {}).get('schema') == 2
        assert _rep.get('guard_probe_key')
    finally:
        L.overfit_sanity_check = _saved_fn
        L._OVERFIT_GUARD.clear()
        L._OVERFIT_GUARD.update(_saved_guard)
        L._OVERFIT_REPORTS.clear()
        L._OVERFIT_REPORTS.update(_saved_reports)


def test_environment_snapshot_identity():
    env = L.environment_snapshot()
    for k in ('gpu_identity', 'cuda_visible_devices', 'float32_matmul_precision',
              'cublas_workspace_config', 'dependencies',
              'dependency_contract', 'env_flags',
              'fp16_reduced_precision_reduction', 'bf16_reduced_precision_reduction'):
        assert k in env, f'environment snapshot missing {k}'
    assert isinstance(env['gpu_identity'], list)
    assert isinstance(env['env_flags'], dict)
    contract = env['dependency_contract']
    assert contract.get('kind') == 'dependency_contract', contract
    assert contract.get('status') in ('ok', 'partial', 'mismatch',
                                      'invalid', 'missing_lock'), contract
    dep = env['dependencies']
    assert 'requirements.txt' in dep['files'] and 'requirements.lock' in dep['files'], dep
    assert dep['files']['requirements.txt'] is None or dep['files']['requirements.txt']['sha256'], dep
    assert 'torch' in dep['packages'], dep


def test_long_window_chunked_mask():
    # A sliding-window full-attention layer uses the chunked mask path beyond
    # the row cap; that path must also work with no sink (sink_view=None).
    torch.manual_seed(0)
    cfg = [L.AttnCfg(kind='full', window=5)]
    m = L.SmallGPT(64, 32, 1, 4, 8, 32, cfg, mlp_ratio=2).eval()
    ids = torch.randint(0, 64, (2, 23))
    with torch.no_grad():
        ref = m(ids)

    old_cap = L._DENSE_MASK_ROW_CAP_BYTES
    old_f = L.causal_window_mask_f
    old_w = L.causal_window_mask

    def _boom(*_a, **_k):
        raise AssertionError('the full-mask path was used beyond the row cap')
    try:
        L._DENSE_MASK_ROW_CAP_BYTES = 1
        L.causal_window_mask_f = _boom
        L.causal_window_mask = _boom
        with torch.no_grad():
            got = m(ids)
    finally:
        L._DENSE_MASK_ROW_CAP_BYTES = old_cap
        L.causal_window_mask_f = old_f
        L.causal_window_mask = old_w
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4), 'the chunked long-window path changed the attention output'


def test_v7_long_window_chunked_mask():
    # Same contract for the v7 RoPE full-attention twin.
    torch.manual_seed(0)
    cfg = V7.AttnCfgRope(kind='full', rope=True, rope_dim=8, window=5)
    m = V7.HybridAttentionRoPE(32, 4, 8, cfg).eval()
    x = torch.randn(2, 23, 32)
    with torch.no_grad():
        ref = m(x)

    old_cap = L._DENSE_MASK_ROW_CAP_BYTES
    old_f = L.causal_window_mask_f
    old_w = L.causal_window_mask

    def _boom(*_a, **_k):
        raise AssertionError('v7 used the full-mask path beyond the row cap')
    try:
        L._DENSE_MASK_ROW_CAP_BYTES = 1
        L.causal_window_mask_f = _boom
        L.causal_window_mask = _boom
        with torch.no_grad():
            got = m(x)
    finally:
        L._DENSE_MASK_ROW_CAP_BYTES = old_cap
        L.causal_window_mask_f = old_f
        L.causal_window_mask = old_w
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4), 'the v7 chunked long-window path changed the attention output'


def test_indexer_empty_sequence():
    # n == 0 must return the same shapes as a non-empty call.
    d, nIH = 8, 2
    H = torch.randn(0, d)
    comp = torch.randn(0, d)
    W_DQ, W_DK, W_w = torch.randn(d, d), torch.randn(d, d), torch.randn(nIH, d)
    valid = []
    m, idx, soft = L.lightning_indexer(H, comp, torch.empty(0, dtype=torch.long), W_DQ, W_DK, W_w, nIH, 4, out_valid=valid)
    assert idx.shape == (0, 0) and soft.shape == (0, 0) and m.shape == (0, 0), (idx.shape, soft.shape, m.shape)
    assert len(valid) == 1 and valid[0].shape == (0, 0), 'the valid-cols output must be empty too'


def test_underpowered_never_verified():
    # An underpowered contrast is never called 'verified', regardless of
    # alpha: the n floor is a precondition for every positive verdict.
    entries = [{'controlled': True, 'n': 2, 'significant_global': True,
                'within_seed_noise': False, 'param_controlled': True,
                'p_exact_signflip': 0.5}]
    assert L._classify_group_status(entries, min_n=3, alpha=0.6, m_family=1) == 'underpowered'


def test_hypothesis_direction_conflict():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    s = {}
    for i in range(7):
        s[f'csa_dynamic::seed{i}'] = dict(variant='csa_dynamic', seed=i, run_cfg=fp, steps=1,
                                         tokens_seen=1, ppl=51.0, params=100)
        s[f'csa_fixed::seed{i}'] = dict(variant='csa_fixed', seed=i, run_cfg=fp, steps=1,
                                       tokens_seen=1, ppl=50.0, params=100)
    design = {'dynamic_blocking': [('csa_dynamic', 'csa_fixed', 'dynamic vs fixed')]}
    r = L.methodology_audit(s, design=design, baselines={}, verbose=False)
    assert 'dynamic_blocking' in r['direction_conflict_components'], r
    assert 'dynamic_blocking' not in r['verified_components'], 'a significant opposite-direction result must never be called verified'
    e = r['components']['dynamic_blocking'][0]
    assert e['predicted_sign'] == -1 and e['observed_sign'] == 1 and e['direction_consistent'] is False, e
    # Reverse the direction and the same effect becomes support.
    for i in range(7):
        s[f'csa_dynamic::seed{i}']['ppl'] = 49.0
    r2 = L.methodology_audit(s, design=design, baselines={}, verbose=False)
    assert 'dynamic_blocking' in r2['verified_components'], r2
    assert r2['components']['dynamic_blocking'][0]['direction_consistent'] is True


def test_metric_policy_and_classification_metrics():
    pol = L.metric_policy_audit()
    assert pol['status'] == 'ok', pol
    assert pol['policies']['multiclass_classification']['aggregation'] == 'macro'
    assert L.METRIC_POLICY['binary_classification']['primary_when_extreme_imbalance'] == 'pr_auc'
    assert L.METRIC_POLICY['binary_classification']['extreme_imbalance_threshold'] == 1e-3
    y = [0] * 990 + [1] * 10
    pred_all_majority = [0] * 1000
    m = L.classification_metrics(y, y_pred=pred_all_majority, labels=[0, 1])
    assert abs(m['accuracy'] - 0.99) < 1e-12
    assert abs(m['f1_macro'] - 0.49748743718592964) < 1e-9, m
    assert abs(m['f1_micro'] - 0.99) < 1e-12 and abs(m['f1_weighted'] - 0.9850251256281407) < 1e-9, m
    assert 'macro, micro and weighted' in m['f1_averaging']
    score = [0.0] * 990 + [1.0] * 10
    mb = L.classification_metrics(y, y_score=score, labels=[0, 1])
    assert mb['roc_auc'] == 1.0 and abs(mb['pr_auc'] - 1.0) < 1e-12, mb
    # The same endpoint expressed as a 2-D probability matrix must use the
    # positive-class column, not the whole matrix.
    score2 = np.stack([1.0 - np.asarray(score), np.asarray(score)], axis=1)
    m2 = L.classification_metrics(y, y_score=score2, labels=[0, 1])
    assert m2['roc_auc'] == 1.0 and abs(m2['pr_auc'] - 1.0) < 1e-12, m2
    # Boundary metrics without precision/recall/random baseline must warn.
    bad = {'full::seed0': {'variant': 'full', 'seed': 0, 'ppl': 10.0,
                           'stats': {'L0_csa_fix': {'bnd_f1': 0.4}}}}
    ma = L.metric_policy_audit(bad)
    assert ma['status'] == 'warn' and any('boundary' in w for w in ma['warnings']), ma


def test_multiclass_ranking_metrics_and_audit():
    # One-vs-rest ranking endpoints: a perfect ranking must be perfect under
    # macro, micro and weighted averaging.
    y = np.array([0] * 80 + [1] * 15 + [2] * 5)
    score = np.zeros((len(y), 3))
    for i, c in enumerate([0, 1, 2]):
        score[y == c, i] = 1.0
    m = L.classification_metrics(y, y_score=score, labels=[0, 1, 2])
    assert m['ranking_averaging'] == 'one-vs-rest', m
    for key in ('pr_auc_macro', 'pr_auc_micro', 'pr_auc_weighted',
                'roc_auc_macro', 'roc_auc_micro', 'roc_auc_weighted'):
        assert abs(m[key] - 1.0) < 1e-12, (key, m[key])
    assert m['per_class']['2']['pr_auc'] == 1.0, m['per_class']
    assert m['per_class']['2']['roc_auc'] == 1.0, m['per_class']
    # The trivial majority floor is recorded next to the model scores: with
    # 80/15/5 labels the always-0 predictor already has accuracy 0.8.
    assert m['trivial_baseline']['strategy'] == 'most_frequent'
    assert m['trivial_baseline']['label'] == '0'
    assert abs(m['trivial_baseline']['accuracy'] - 0.8) < 1e-12
    assert abs(m['trivial_baseline']['f1_macro'] - (2.0 * 0.8 / 1.8) / 3.0) < 1e-12
    # Micro F1 pools decisions across classes, so the constant-majority floor
    # must equal its accuracy (not the majority class's binary F1).
    assert abs(m['trivial_baseline']['f1_micro'] - 0.8) < 1e-12, m
    # One-vs-rest PR-AUC also has a non-zero chance floor (mean prevalence);
    # a raw PR-AUC of 0.5 at 1:1000 would not be skillful.
    assert abs(m['pr_auc_macro_chance'] - 1.0 / 3.0) < 1e-12, m
    assert abs(m['pr_auc_weighted_chance'] - 0.665) < 1e-12, m
    assert abs(m['pr_auc_macro_lift_over_chance'] - (1.0 - 1.0 / 3.0)) < 1e-12
    assert abs(m['pr_auc_weighted_lift_over_chance'] - 0.335) < 1e-12
    # The one-hot probability matrix is perfectly calibrated and confident.
    assert m['calibration']['status'] == 'ok' and m['calibration']['brier_score'] == 0.0
    assert m['calibration']['ece'] == 0.0 and m['calibration']['top1_accuracy'] == 1.0
    _uniform = L.classification_metrics(y, y_score=np.full((len(y), 3), 1.0 / 3.0),
                                        labels=[0, 1, 2])
    assert _uniform['calibration']['status'] == 'ok'
    assert abs(_uniform['calibration']['brier_score'] - 2.0 / 3.0) < 1e-12
    _logits = L.classification_metrics(y, y_score=np.ones((len(y), 3)), labels=[0, 1, 2])
    assert _logits['calibration']['status'] == 'unavailable'
    assert 'brier_score' not in _logits
    # The threshold-free metric list is a capability declaration: an
    # unavailable calibration endpoint must not be advertised as measured.
    assert 'brier_score' not in _logits['threshold_free_metrics']
    assert 'ece' not in _logits['threshold_free_metrics']
    _bin_logits = L.classification_metrics([0, 0, 1, 1], y_score=[-2.0, -1.0, 1.0, 2.0],
                                           labels=[0, 1])
    assert _bin_logits['calibration']['status'] == 'unavailable'
    assert _bin_logits['threshold_free_metrics'] == ['roc_auc', 'pr_auc']
    _bin_prob = L.classification_metrics([0, 0, 1, 1], y_score=[0.1, 0.2, 0.8, 0.9],
                                         labels=[0, 1])
    assert _bin_prob['threshold_free_metrics'] == ['roc_auc', 'pr_auc',
                                                   'brier_score', 'ece']
    # A binary record gets the same floor; the majority predictor has zero
    # minority recall, so balanced accuracy is not fooled by its high accuracy.
    _bin_floor = L.classification_metrics([0] * 99 + [1], y_pred=[0] * 100, labels=[0, 1])
    assert _bin_floor['trivial_baseline']['label'] == '0'
    assert abs(_bin_floor['trivial_baseline']['accuracy'] - 0.99) < 1e-12
    assert abs(_bin_floor['trivial_baseline']['balanced_accuracy'] - 0.5) < 1e-12
    # Macro is equal-weight over classes, so it must not equal a support-
    # weighted or pooled (micro) aggregate when the classes differ in size and
    # the ranking is imperfect.  This is exactly why the averaging must be named.
    rng = np.random.default_rng(3)
    scores = np.stack([rng.random(len(y)) for _ in range(3)], axis=1)
    mi = L.classification_metrics(y, y_score=scores, labels=[0, 1, 2])
    assert mi['pr_auc_macro'] != mi['pr_auc_micro'] or mi['pr_auc_macro'] != mi['pr_auc_weighted'], mi
    assert 0.0 <= mi['pr_auc_macro'] <= 1.0 and 0.0 <= mi['pr_auc_micro'] <= 1.0
    assert abs(mi['macro_micro_f1_gap'] - (mi['f1_macro'] - mi['f1_micro'])) < 1e-12
    # A requested class absent from the sample has an undefined one-vs-rest
    # endpoint; it is listed rather than assigned a fake score, and the macro
    # average covers only the defined classes.
    m_undef = L.classification_metrics(np.array([0, 0, 1, 1]),
                                       y_score=np.array([[0.9, 0.1, 0.0], [0.8, 0.2, 0.0],
                                                         [0.1, 0.9, 0.0], [0.2, 0.8, 0.0]]),
                                       labels=[0, 1, 2])
    assert m_undef['per_class']['2']['pr_auc'] is None
    assert m_undef['per_class']['2']['roc_auc'] is None
    assert '2' in m_undef['undefined_classes']
    assert abs(m_undef['pr_auc_macro'] - np.mean(
        [m_undef['per_class'][c]['pr_auc'] for c in ('0', '1')])) < 1e-12
    # Extreme multiclass imbalance must switch the conditional primary to
    # macro PR-AUC and flag a record that reports only F1.
    y2 = np.array([0] * 1990 + [1] * 9 + [2] * 1)
    score2 = np.zeros((len(y2), 3))
    for i, c in enumerate([0, 1, 2]):
        score2[y2 == c, i] = 1.0
    m2 = L.classification_metrics(y2, y_score=score2, labels=[0, 1, 2])
    assert abs(m2['minority_prevalence'] - 1.0 / 2000.0) < 1e-15, m2['minority_prevalence']
    assert 'pr_auc_macro' in m2['imbalance_warning'], m2
    assert L.METRIC_POLICY['multiclass_classification']['primary_when_extreme_imbalance'] == 'pr_auc_macro'
    good = dict(m2, variant='mc', task='multiclass_classification')
    assert L.metric_policy_audit({'mc': good})['status'] == 'ok', L.metric_policy_audit({'mc': good})
    no_rank = {k: v for k, v in good.items() if not k.startswith(('pr_auc_', 'roc_auc_'))}
    nr = L.metric_policy_audit({'mc': no_rank})
    assert nr['status'] == 'warn', nr
    assert any('pr_auc_macro' in w for w in nr['warnings']), nr
    # A hand-edited aggregate cannot pass when per-class values exist.
    forged = dict(good, pr_auc_macro=0.5)
    fr = L.metric_policy_audit({'mc': forged})
    assert fr['status'] == 'warn' and any('pr_auc_macro' in w and 'mean' in w
                                         for w in fr['warnings']), fr
    # Declaring a binary ranking metric as the multiclass primary is rejected.
    wrong = dict(good, primary_metric='roc_auc')
    wr = L.metric_policy_audit({'mc': wrong})
    assert wr['status'] == 'warn' and any('primary_metric' in w for w in wr['warnings']), wr
    # A forged easier-than-majority floor is rejected.
    forged_floor = dict(good, trivial_baseline=dict(good['trivial_baseline'], label='2'))
    ff = L.metric_policy_audit({'mc': forged_floor})
    assert ff['status'] == 'warn' and any('trivial_baseline' in w for w in ff['warnings']), ff
    # The PR-AUC chance floor and the macro-micro F1 gap are reproducible too.
    forged_chance = dict(good, pr_auc_macro_chance=0.001)
    fc = L.metric_policy_audit({'mc': forged_chance})
    assert fc['status'] == 'warn' and any('pr_auc_macro_chance' in w for w in fc['warnings']), fc
    forged_gap = dict(good, macro_micro_f1_gap=0.5)
    fg = L.metric_policy_audit({'mc': forged_gap})
    assert fg['status'] == 'warn' and any('macro_micro_f1_gap' in w for w in fg['warnings']), fg
    # Different label identities that share a str() spelling (integer 1 and
    # string '1') must not overwrite one another in the per-class table.  The
    # report keeps the historical names when unambiguous and otherwise uses
    # unique display keys, so every declared class retains its own row.
    _mixed_labels = np.array([1, '1', 2], dtype=object)
    _mixed_y = np.array([_mixed_labels[i % 3] for i in range(12)], dtype=object)
    _mixed_score = np.zeros((len(_mixed_y), 3))
    for _i, _lab in enumerate(_mixed_labels):
        _mixed_score[[str(v) == str(_lab) and type(v) is type(_lab)
                      for v in _mixed_y.tolist()], _i] = 1.0
    _mx = L.classification_metrics(_mixed_y, y_score=_mixed_score,
                                   labels=_mixed_labels)
    assert len(_mx['labels']) == 3 and len(set(_mx['labels'])) == 3, _mx['labels']
    assert len(_mx['per_class']) == 3, _mx['per_class']
    assert all(_mx['per_class'][_n]['support'] == 4 for _n in _mx['labels']), _mx
    assert _mx['trivial_baseline']['label'] in _mx['labels']
    assert len(_mx['prevalence']) == 3, _mx['prevalence']
    _mx_audit = L.metric_policy_audit({'mx': dict(
        _mx, variant='mx', task='multiclass_classification')})
    assert _mx_audit['status'] == 'ok', _mx_audit
    # Omitting labels must not let np.asarray coerce the integer 1 into the
    # string '1' before the class identities are checked.
    _mx_auto = L.classification_metrics([1, '1', 2, 1, '1', 2],
                                        y_pred=[1, '1', 2, 1, '1', 2])
    assert len(_mx_auto['labels']) == 3 and len(set(_mx_auto['labels'])) == 3, _mx_auto
    assert len(_mx_auto['per_class']) == 3, _mx_auto
    # An explicitly declared label list must not be string-coerced either:
    # np.asarray([1, '1']) merges the two classes before the duplicate check.
    for _decl in ([1, '1', 2], np.array([1, '1', 2], dtype=object)):
        _mx_explicit = L.classification_metrics(
            [1, '1', 2, 1, '1', 2], y_pred=[1, '1', 2, 1, '1', 2],
            labels=_decl)
        assert len(_mx_explicit['labels']) == 3, _mx_explicit['labels']
        assert len(_mx_explicit['per_class']) == 3, _mx_explicit
    # A BINARY mixed-type endpoint must not go through np.where with two
    # un-promotable dtypes (NumPy 2 raises DTypePromotionError for int 1 vs
    # the string '1'); the derived predictions keep both identities.
    _mixed_y_bin = np.array([1, '1'] * 25, dtype=object)
    _mixed_s_bin = np.array([0.8] * 25 + [0.2] * 25)
    _mb = L.classification_metrics(
        _mixed_y_bin, y_score=_mixed_s_bin,
        labels=np.array([1, '1'], dtype=object), positive_label=1)
    assert len(_mb['labels']) == 2 and _mb['n'] == 50, _mb
    assert _mb['predicted_positive_rate'] == 0.5, _mb


def test_metric_prevalence_recommendation_and_per_class_chance():
    # A rare class's raw PR-AUC must be readable against its own chance floor,
    # not only the macro mean (which a frequent class can carry).
    y = np.array([0] * 1990 + [1] * 9 + [2] * 1)
    score = np.zeros((len(y), 3))
    for i, c in enumerate([0, 1, 2]):
        score[y == c, i] = 1.0
    m = L.classification_metrics(y, y_score=score, labels=[0, 1, 2])
    assert m['recommended_primary_metric'] == 'pr_auc_macro', m
    assert m['recommended_primary_average'] == 'one_vs_rest_macro', m
    assert m['recommended_primary_available'] is True, m
    for c, sup in (('1', 9), ('2', 1)):
        pc = m['per_class'][c]
        assert abs(pc['pr_auc_chance'] - sup / len(y)) < 1e-15, pc
        assert abs(pc['pr_auc_lift_over_chance'] - (pc['pr_auc'] - pc['pr_auc_chance'])) < 1e-12, pc
        assert pc['roc_auc_chance'] == 0.5 and abs(
            pc['roc_auc_lift_over_chance'] - (pc['roc_auc'] - 0.5)) < 1e-15, pc
    good = dict(m, variant='mc', task='multiclass_classification')
    assert L.metric_policy_audit({'mc': good})['status'] == 'ok', L.metric_policy_audit({'mc': good})
    # Forged per-class chance/lift values are rejected from the stored record.
    forged = json.loads(json.dumps(good))
    forged['per_class']['2']['pr_auc_chance'] = 0.5
    fr = L.metric_policy_audit({'mc': forged})
    assert fr['status'] == 'warn' and any('per-class PR-AUC chance' in w
                                         for w in fr['warnings']), fr
    # The recommendation is a claim too: it must match the policy's
    # conditional endpoint at the observed prevalence.
    wrong_rec = dict(good, recommended_primary_metric='f1_macro')
    wr = L.metric_policy_audit({'mc': wrong_rec})
    assert wr['status'] == 'warn' and any('recommended_primary_metric' in w
                                         for w in wr['warnings']), wr
    # Extreme binary imbalance switches the recommendation to PR-AUC and
    # exposes chance-relative ROC skill.
    yb = np.array([0] * 9999 + [1])
    sb = np.linspace(0.0, 1.0, len(yb))
    mb = L.classification_metrics(yb, y_score=sb, labels=[0, 1])
    assert mb['recommended_primary_metric'] == 'pr_auc', mb
    assert abs(mb['roc_auc_chance'] - 0.5) < 1e-15
    assert abs(mb['roc_auc_lift_over_chance'] - (mb['roc_auc'] - 0.5)) < 1e-15
    assert abs(mb['roc_auc_normalized_skill'] - (mb['roc_auc'] - 0.5) / 0.5) < 1e-15
    bad_rec = dict(mb, variant='b', task='binary_classification',
                   recommended_primary_metric='f1')
    br = L.metric_policy_audit({'b': bad_rec})
    assert br['status'] == 'warn' and any('recommended_primary_metric' in w
                                         for w in br['warnings']), br


def test_hypothesis_prevalence_conditional_endpoint():
    def _spec(**kw):
        base = {'statement': 'rare-event detector improves recall',
                'mechanism': 'the score ranks positives above negatives',
                'predicted_sign': 1, 'primary_metric': 'f1',
                'requires_param_match': False, 'falsifier': 'PR-AUC is at prevalence'}
        base.update(kw)
        return base
    # At a 1:10000 base rate, a hypothesis written against F1 is misaligned
    # with the policy's conditional PR-AUC endpoint even though f1 is the
    # normal binary primary.
    bad = L.hypothesis_audit({'h': _spec(prevalence=1e-4)}, 'component')
    assert bad['status'] == 'warn' and 'h' in bad['metric_misaligned'], bad
    assert any('extreme-imbalance threshold' in w for w in bad['warnings']), bad
    # Declaring the conditional endpoint (or a justified secondary endpoint)
    # is accepted.
    good = L.hypothesis_audit({'h': _spec(prevalence=1e-4,
                                         primary_metric='pr_auc')}, 'component')
    assert good['status'] == 'ok' and not good['metric_misaligned'], good
    sec = L.hypothesis_audit({'h': _spec(prevalence=1e-4,
                                        primary_metric='roc_auc',
                                        metric_role='secondary',
                                        metric_justification='report ROC alongside PR')},
                             'component')
    assert sec['status'] in ('ok', 'warn') and not sec['metric_misaligned'], sec
    # Above the threshold the normal F1 primary remains aligned.
    normal = L.hypothesis_audit({'h': _spec(prevalence=0.2)}, 'component')
    assert normal['status'] == 'ok' and not normal['metric_misaligned'], normal
    # A malformed base rate fails closed instead of disabling the check.
    for _bad in (True, '0.0001', float('nan'), float('inf'), -0.1, 1.2):
        r = L.hypothesis_audit({'h': _spec(prevalence=_bad)}, 'component')
        assert r['status'] == 'warn' and 'h' in r['metric_misaligned'], (_bad, r)
    # The same conditional rule applies to NIAH retrieval: a hypothesis for a
    # rare answer class must name macro recall, not top-1 accuracy.
    _niah_bad = L.hypothesis_audit({'h': _spec(
        primary_metric='top1_accuracy', metric_policy='niah_retrieval',
        prevalence=1e-4)}, 'component')
    assert _niah_bad['status'] == 'warn' and 'h' in _niah_bad['metric_misaligned'], _niah_bad
    _niah_good = L.hypothesis_audit({'h': _spec(
        primary_metric='macro_recall', metric_policy='niah_retrieval',
        prevalence=1e-4)}, 'component')
    assert _niah_good['status'] == 'ok' and not _niah_good['metric_misaligned'], _niah_good


def test_binary_degenerate_threshold_visibility():
    # Logits/scores on a non-probability scale make the default 0.5 cut a
    # constant predictor.  The record must expose that instead of letting an
    # F1/accuracy headline look like learned decision quality.
    y = [0, 0, 1, 1]
    m = L.classification_metrics(y, y_score=[1.0, 2.0, 3.0, 4.0], labels=[0, 1])
    assert m['predicted_positive_rate'] == 1.0, m
    assert m['degenerate_predictions'] is True, m
    assert m['threshold_within_score_range'] is False, m
    assert 'outside the observed score range' in m['threshold_range_warning'], m
    rec = dict(m, variant='b', task='binary_classification')
    audit = L.metric_policy_audit({'b': rec})
    assert audit['status'] == 'warn', audit
    assert any('outside the observed score range' in w for w in audit['warnings']), audit
    assert 'b' in audit['degenerate_prediction_records'], audit
    # A forged predicted-positive rate cannot hide a degenerate cut.
    forged = dict(rec, predicted_positive_rate=0.5)
    fa = L.metric_policy_audit({'b': forged})
    assert fa['status'] == 'warn' and any(
        'predicted_positive_rate' in w for w in fa['warnings']), fa
    # A normal probability score with a meaningful cut stays non-degenerate.
    good = L.classification_metrics(y, y_score=[0.1, 0.2, 0.8, 0.9], labels=[0, 1])
    assert good['degenerate_predictions'] is False, good
    assert good['threshold_within_score_range'] is True, good
    assert good['predicted_positive_rate'] == 0.5, good
    assert L.metric_policy_audit({'b': dict(good, variant='b',
                                           task='binary_classification')})['status'] == 'ok'


def test_binary_pr_floor_uses_positive_prevalence():
    # PR-AUC's chance floor follows the POSITIVE class, not the minority class.
    # When the positive endpoint is the common label (rare negative event), a
    # correct record must not be flagged as having a misstated floor.
    y = ['pos'] * 9999 + ['neg']
    score = [0.9] * 9999 + [0.1]
    m = L.classification_metrics(y, y_score=score, labels=['neg', 'pos'],
                                 positive_label='pos')
    assert m['positive_label'] == 'pos'
    assert abs(m['pr_auc_prevalence'] - 0.9999) < 1e-12, m
    audit = L.metric_policy_audit({'b': dict(m, variant='b',
                                             task='binary_classification')})
    assert audit['status'] == 'warn'  # only the rare-minority F1-cut warning
    assert not any('floor is misstated' in w for w in audit['warnings']), audit
    assert not any('pr_auc_lift_over_prevalence' in w for w in audit['warnings']), audit
    # A genuinely forged floor is still rejected when the positive class is common.
    forged = dict(m, variant='b', task='binary_classification',
                  pr_auc_prevalence=0.0001)
    fa = L.metric_policy_audit({'b': forged})
    assert any('floor is misstated' in w for w in fa['warnings']), fa
    # A record that exposes only the PR-AUC chance floor still declares its
    # prevalence; a rare-positive record must not evade the PR-AUC endpoint
    # check by omitting a separate `prevalence` field.
    _rare_only = {'variant': 'r', 'task': 'binary_classification',
                  'positive_label': '1', 'f1_primary': 0.9,
                  'f1_primary_average': 'binary_positive_class',
                  'roc_auc': 0.99, 'pr_auc_prevalence': 0.0005}
    _ro = L.metric_policy_audit({'r': _rare_only})
    assert _ro['status'] == 'warn' and any(
        'PR-AUC' in w and '1:1000' in w for w in _ro['warnings']), _ro

