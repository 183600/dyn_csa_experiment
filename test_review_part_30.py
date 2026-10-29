"""CPU checks for the correlation-aware exact max-T ablation family.

Holm is valid but pays the full family penalty even when every contrast shares
a reference arm, seeds and evaluation rows.  These checks pin the exact
Westfall-Young max-T reading: it must agree with the per-contrast exact
sign-flip p at m=1, exploit the correlation of duplicated/controlled contrasts
without ever touching p_holm/verified, refuse partial families, and be
replayable from the stored per-seed deltas.  No test reads a result panel or a
held-out split.
"""
import copy
import json
import math
import os
import sys as _sys
import tempfile

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)


_RUN_CFG = ('steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_'
            'dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX')


def _close(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def _summary(deltas, arms=('a', 'b'), base=50.0):
    """An a-b paired summary whose per-seed deltas are exactly ``deltas``."""
    out = {}
    for i, d in enumerate(deltas):
        out[f'{arms[0]}::seed{i}'] = dict(
            variant=arms[0], seed=i, run_cfg=_RUN_CFG, steps=1,
            tokens_seen=1, params=100, ppl=float(base) + float(d))
        out[f'{arms[1]}::seed{i}'] = dict(
            variant=arms[1], seed=i, run_cfg=_RUN_CFG, steps=1,
            tokens_seen=1, params=100, ppl=float(base))
    return out


def _two_contrast_summary(deltas, base=50.0):
    """a-b and a-c with identical deltas, on the same six seeds."""
    out = {}
    for i, d in enumerate(deltas):
        out[f'a::seed{i}'] = dict(variant='a', seed=i, run_cfg=_RUN_CFG,
                                  steps=1, tokens_seen=1, params=100,
                                  ppl=float(base) + float(d))
        for arm in ('b', 'c'):
            out[f'{arm}::seed{i}'] = dict(variant=arm, seed=i,
                                          run_cfg=_RUN_CFG, steps=1,
                                          tokens_seen=1, params=100,
                                          ppl=float(base))
    return out


def _write_audit(outdir, design=None, summary=None):
    design = design or {'compA': [('a', 'b', 'x')],
                        'compB': [('a', 'c', 'y')]}
    summary = summary if summary is not None else _two_contrast_summary(
        [1.0, 1.1, 0.9, 1.05, 0.95, 1.2])
    return L.methodology_audit(summary, design=design, baselines={},
                               verbose=False, outdir=outdir, write=True)


def test_maxt_single_contrast_equals_exact_signflip():
    for n in range(3, 9):
        deltas = [1.0 + 0.1 * i for i in range(n)]
        r = L.maxt_signflip_adjust({'only': deltas})
        exact = L.exact_signflip_p(deltas)
        rec = r['contrasts']['only']
        assert _close(rec['p_raw_signflip'], exact['p']), (n, rec, exact)
        # With one hypothesis the free step-down is the single test; the
        # correlation-aware reading must not manufacture a different p.
        assert _close(rec['p_maxt'], exact['p']), (n, rec, exact)
        assert rec['reject_maxt'] is (exact['p'] <= 0.05)
        assert _close(r['p_floor_single'], 2.0 / 2 ** n)
        assert r['n_patterns'] == 2 ** n and r['m'] == 1
        assert rec['p_maxt'] >= rec['p_raw_signflip'] - 1e-12
        if rec['ci_simultaneous'] is not None:
            lo, hi = rec['ci_simultaneous']
            assert lo <= rec['mean'] <= hi
            assert _close(hi - lo, 2.0 * rec['sem'] * r['critical_value'],
                          tol=1e-9)
    # n=5 cannot reach a 95% two-sided sign-flip tail (its floor is 2/32);
    # the function must say so instead of quoting a 93.75% interval as 95%.
    r5 = L.maxt_signflip_adjust({'only': [1.0, 1.1, 0.9, 1.05, 0.95]})
    assert r5['contrasts']['only']['ci_simultaneous'] is None
    assert _close(r5['ci_tail_level'], 0.9375)
    assert 'cannot reach' in r5['ci_note']


def test_maxt_best_case_design_floor_is_labelled_optimistic():
    for alpha, n in ((0.05, 6), (0.01, 8), (0.10, 5)):
        r = L.maxt_min_n_for_alpha(alpha)
        assert r['min_n'] == n, r
        assert r['p_floor'] <= alpha
        assert 2.0 / (1 << (n - 1)) > alpha
        assert 'best case' in r['scope']
    assert L.maxt_min_n_for_alpha(1e-9)['beyond_enumeration_cap'] is True
    assert L.maxt_min_n_for_alpha(0.05)['beyond_enumeration_cap'] is False
    # Exact power-of-two boundaries must not be pushed up by log2 round-off.
    assert L.maxt_min_n_for_alpha(0.25)['min_n'] == 3
    assert L.maxt_min_n_for_alpha(0.50)['min_n'] == 2
    try:
        L.maxt_min_n_for_alpha(True)
    except ValueError:
        pass
    else:
        raise AssertionError('a boolean alpha must be refused')


def test_max_t_extreme_scale_does_not_overflow_the_sum():
    base = [1.0, 1.1, 0.9, 1.05]
    normal = L.maxt_signflip_adjust({'a': base})['contrasts']['a']
    huge = [x * 1e308 for x in base]
    tiny = [x * 1e-300 for x in base]
    for deltas in (huge, tiny):
        rec = L.maxt_signflip_adjust({'a': deltas})['contrasts']['a']
        assert math.isfinite(rec['mean']) and math.isfinite(rec['sd'])
        assert _close(rec['p_maxt'], normal['p_maxt'])
        assert _close(rec['t_stat'], normal['t_stat'])


def test_max_t_stepdown_beats_holm_for_correlated_contrasts():
    deltas = [1.0, 1.1, 0.9, 1.05, 0.95, 1.2]
    r = L.maxt_signflip_adjust({'a': deltas, 'b': deltas})
    raw = L.exact_signflip_p(deltas)['p']
    assert raw <= 0.05, raw
    for name in ('a', 'b'):
        rec = r['contrasts'][name]
        assert _close(rec['p_raw_signflip'], raw)
        assert _close(rec['p_maxt'], raw), rec
        assert rec['reject_maxt'] is True
        assert rec['p_maxt'] <= rec['p_maxt_single'] + 1e-12
    holm = L.holm_adjust([('a', raw), ('b', raw)])
    # Two perfectly dependent contrasts: Holm doubles the smallest p and
    # loses the rejection; the exact joint null sees that they move together.
    assert holm['a']['p_holm'] > 0.05, holm
    assert holm['a']['reject'] is False
    assert r['contrasts']['a']['p_maxt'] < holm['a']['p_holm']
    assert r['critical_value'] is not None
    assert 0.9 < r['ci_tail_level'] <= 1.0


def test_tied_t_statistics_are_order_independent():
    deltas = [1.0, 1.1, 0.9, 1.05, 0.95, 1.2]
    a = L.maxt_signflip_adjust({'z': deltas, 'a': deltas})
    b = L.maxt_signflip_adjust({'a': deltas, 'z': deltas})
    for name in ('a', 'z'):
        assert _close(a['contrasts'][name]['p_maxt'],
                      b['contrasts'][name]['p_maxt'])
        assert a['contrasts'][name]['rank'] == b['contrasts'][name]['rank']
    # Exact ties receive one common adjusted p, not an ordering artefact.
    assert _close(a['contrasts']['a']['p_maxt'],
                  a['contrasts']['z']['p_maxt'])


def test_max_t_is_scale_invariant_and_deterministic():
    deltas = [0.7, 1.3, 0.4, 1.1, 0.9, 1.0, 0.8]
    a = L.maxt_signflip_adjust({'x': deltas, 'y': [d * 3.0 for d in deltas]})
    b = L.maxt_signflip_adjust({'x': deltas, 'y': [d * 3.0 for d in deltas]})
    assert a['contrasts']['x']['p_maxt'] == b['contrasts']['x']['p_maxt']
    assert a['contrasts']['y']['p_maxt'] == b['contrasts']['y']['p_maxt']
    # Studentising by each contrast's own SD makes a common rescaling of the
    # whole family a no-op; the interval scales with the deltas.
    scaled = L.maxt_signflip_adjust({'x': [d * 5.0 for d in deltas]})
    assert _close(scaled['contrasts']['x']['p_maxt'],
                  a['contrasts']['x']['p_maxt'])
    assert _close(scaled['contrasts']['x']['t_stat'],
                  a['contrasts']['x']['t_stat'])
    if a['contrasts']['x']['ci_simultaneous'] and scaled['contrasts']['x']['ci_simultaneous']:
        assert _close(scaled['contrasts']['x']['ci_simultaneous'][0],
                      a['contrasts']['x']['ci_simultaneous'][0] * 5.0, tol=1e-9)
        assert _close(scaled['contrasts']['x']['ci_simultaneous'][1],
                      a['contrasts']['x']['ci_simultaneous'][1] * 5.0, tol=1e-9)


def test_incomplete_family_is_not_promoted_through_methodology_audit():
    summary = _summary([1.0, 1.1, 0.9, 1.05, 0.95, 1.2])
    design = {'compA': [('a', 'b', 'x'), ('a', 'missing-arm', 'missing')]}
    rep = L.methodology_audit(summary, design=design, baselines={},
                              verbose=False)
    fam = rep['family']['maxt']
    assert fam['status'] == 'subset_only' and fam['family_ready'] is False
    assert fam['m_eligible'] == 1 and fam['family_size'] == 2, fam
    e = rep['components']['compA'][0]
    assert e.get('p_maxt') is not None and e.get('significant_maxt') is False
    assert rep['maxt_only_contrasts'] == []
    # Holm/verified are untouched by the max-T readout.
    assert e.get('significant_global') is False
    assert 'compA' not in rep['verified_components']
    assert any('not family-ready' in w for w in rep['warnings']), rep['warnings']


def test_methodology_audit_reports_ready_family_without_promoting():
    rep = _write_audit(tempfile.mkdtemp())
    fam = rep['family']['maxt']
    assert fam['status'] == 'ok' and fam['family_ready'] is True, fam
    assert fam['m_eligible'] == 2 and fam['family_size'] == 2
    assert fam['best_case']['min_n'] == 6 and fam['best_case']['p_floor'] <= 0.05
    assert fam['n_maxt_reject'] == 2 and fam['n_holm_reject'] == 0, fam
    assert rep['verified_components'] == []
    for comp in ('compA', 'compB'):
        e = rep['components'][comp][0]
        assert e.get('p_maxt') is not None
        assert e.get('p_holm_global') > e.get('p_maxt')
        assert e.get('significant_maxt') is True
        assert e.get('significant_global') is False
        assert e.get('maxt_family_ready') is True
        assert e.get('ci_simultaneous') is not None
    assert len(rep['maxt_only_contrasts']) == 2
    assert len(rep['maxt_only_pairs']) == 2
    assert any('max-T' in w and 'sensitivity readout' in w
               for w in rep['warnings']), rep['warnings']
    assert 'sensitivity readout' in fam['role']


def test_joint_correction_groups_and_excludes_fail_closed():
    r = L.maxt_joint_correction({
        'a': {'deltas': [1.0, 1.1, 0.9], 'seeds': [0, 1, 2]},
        'b': {'deltas': [1.0, 1.1, 0.9, 1.05], 'seeds': [0, 1, 2, 3]},
    }, family_size=2)
    assert r['n_groups'] == 2 and r['family_ready'] is False
    assert r['status'] == 'subset_only'
    assert any('different seed sets' in w for w in r['warnings'])
    # A declared slot that never became a complete controlled contrast keeps
    # the family from being "ready", even when the surviving group is fine.
    r2 = L.maxt_joint_correction({
        'a': {'deltas': [1.0, 1.1, 0.9], 'seeds': [0, 1, 2]},
        'missing': {'deltas': [], 'seeds': [], 'eligible': False,
                    'reason': 'arm absent'},
    }, family_size=2)
    assert r2['family_ready'] is False and r2['status'] == 'subset_only'
    assert 'missing' in r2['excluded']
    # Degenerate zero-variance non-zero mean is named, not turned into a p.
    r3 = L.maxt_joint_correction({
        'a': {'deltas': [1.0, 1.0, 1.0], 'seeds': [0, 1, 2]},
    }, family_size=1)
    assert r3['status'] == 'not_estimable' and r3['family_ready'] is False
    assert r3['by_key']['a']['status'] == 'not_estimable'


def test_maxt_fail_closed_on_malformed_inputs():
    def _raises(fn, why):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(why)

    _raises(lambda: L.maxt_signflip_adjust({}), 'empty family')
    _raises(lambda: L.maxt_signflip_adjust({'a': [1, 2], 'b': [1]}), 'ragged')
    _raises(lambda: L.maxt_signflip_adjust({'a': [1, True]}), 'boolean delta')
    _raises(lambda: L.maxt_signflip_adjust({'a': [1, float('nan')]}), 'NaN')
    _raises(lambda: L.maxt_signflip_adjust({'a': [1.0]}), 'n<2')
    _raises(lambda: L.maxt_signflip_adjust({'a': [1, 2, 3, 4, 5]}, max_n=4),
            'above the enumeration cap')
    _raises(lambda: L.maxt_signflip_adjust({'a': [1, 2, 3, 4]}, max_n=4.5),
            'a fractional cap is not an integer cap')
    _raises(lambda: L.maxt_signflip_adjust({'a': [1, 1, 1]}),
            'zero variance with non-zero mean')
    _raises(lambda: L.maxt_signflip_adjust({'a': [1, 2]}, alpha=0.0), 'alpha')
    _raises(lambda: L.maxt_signflip_adjust({'a': [1, 2]}, alpha=True), 'alpha bool')
    _raises(lambda: L.maxt_joint_correction({}, family_size=2.5),
            'a fractional family size is not an integer family')
    _raises(lambda: L.maxt_joint_correction({}, family_size=True),
            'a boolean family size is not an integer family')


def test_joint_correction_gate_replays_and_refuses_tampering():
    tmp = tempfile.mkdtemp()
    _write_audit(tmp)
    good = A.audit_joint_correction([tmp])
    assert good['status'] == 'ok', good
    # A single explicit path (the CLI form) is one panel, not an iterable of
    # characters.
    assert A.audit_joint_correction(tmp)['status'] == 'ok'
    path = os.path.join(tmp, 'methodology_audit.json')
    base = json.load(open(path, encoding='utf-8'))

    def _mutate(mutator):
        rep = copy.deepcopy(base)
        mutator(rep)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(rep, fh)
        return A.audit_joint_correction([tmp])

    def _p(rep):
        rep['components']['compA'][0]['p_maxt'] = 0.99

    def _ci(rep):
        rep['components']['compA'][0]['ci_simultaneous'][0] += 1.0

    def _rank(rep):
        rep['components']['compA'][0]['maxt_rank'] = 3

    def _ready(rep):
        rep['family']['maxt']['family_ready'] = False

    def _only(rep):
        rep['family']['maxt']['maxt_only'] = []

    def _single(rep):
        rep['components']['compA'][0]['p_maxt_single'] = 1.0

    def _fp(rep):
        gid = next(iter(rep['family']['maxt']['groups']))
        rep['family']['maxt']['groups'][gid]['deltas_fp'] = 'deadbeef'

    for mut, why in ((_p, 'p_maxt'), (_ci, 'simultaneous interval'),
                     (_rank, 'rank'), (_ready, 'family_ready'),
                     (_only, 'maxt_only'), (_single, 'p_maxt_single'),
                     (_fp, 'deltas_fp')):
        bad = _mutate(mut)
        assert bad['status'] == 'fail', (why, bad)
        assert bad['violations'], (why, bad)

    # A legacy artifact without the block is unverifiable, never a pass.
    legacy = copy.deepcopy(base)
    legacy['family'].pop('maxt')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(legacy, fh)
    out = A.audit_joint_correction([tmp])
    assert out['status'] == 'warn', out
    assert any('no family.maxt block' in w for w in out['warnings'])
    # No artifact at all is not_verifiable_locally rather than a green run.
    assert A.audit_joint_correction([tempfile.mkdtemp()])['status'] == \
        'not_verifiable_locally'
    import inspect
    assert "'maxt': audit_joint_correction" in inspect.getsource(A._run_cli)
    assert "'joint_correction': audit_joint_correction" in inspect.getsource(A._run_cli)
