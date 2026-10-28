"""CPU review checks (split group 01; t_scaling_trend_primary_direction_and_floor .. t_doc_level_analysis)."""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

if not _HAVE_TORCH:
    import pytest as _pytest
    _pytest.skip('CPU torch dependencies are unavailable', allow_module_level=True)

def test_scaling_trend_primary_direction_and_floor():
    # Ragged panel: the paired per-seed evidence is six positive slopes, while
    # the seed-weighted WLS point means (the extra A seeds have high effects)
    # would give a negative slope.  The verdict must follow the paired slope.
    pts = [{'label': 'A', 'scale': 1.0, 'effects': [0.0] * 6 + [3.0] * 6,
            'seeds': list(range(12))},
           {'label': 'B', 'scale': 10.0, 'effects': [1.0] * 6, 'seeds': list(range(6))}]
    r = L.scaling_trend(pts, min_seeds=3, min_points=2)
    assert r['paired']['slope_mean'] == 1.0, r['paired']
    assert r['wls']['slope'] < 0.0, 'the constructed unpaired fit must disagree with the paired slope'
    assert r['significant'] is True and r['direction_slope'] == 1.0, r
    assert r['verdict'] == 'effect_increases_with_scale', r['verdict']
    assert r['benefit_grows_with_scale'] is False, r

    # Six complete seeds reject at alpha=0.05, but with min_seeds=7 the
    # pre-registered floor is not met: the trend must stay descriptive.
    def _pts(seeds):
        return [{'label': 'A', 'scale': 1.0, 'effects': [0.0] * seeds, 'seeds': list(range(seeds))},
                {'label': 'B', 'scale': 10.0, 'effects': [1.0] * seeds, 'seeds': list(range(seeds))},
                {'label': 'C', 'scale': 100.0, 'effects': [2.0] * seeds, 'seeds': list(range(seeds))}]
    r2 = L.scaling_trend(_pts(6), min_seeds=7, min_points=3)
    assert r2['p_primary'] < 0.05 and r2['underpowered'] is True, r2
    assert r2['significant'] is False and r2['verdict'] == 'underpowered', r2

    # A perfect three-point ladder is not "heterogeneous": lack-of-fit Q is
    # zero, so the trend verdict must survive.
    r3 = L.scaling_trend(_pts(6), min_seeds=3, min_points=3)
    assert r3['heterogeneity']['Q'] == 0.0 and r3['heterogeneity']['I2_percent'] == 0.0, r3['heterogeneity']
    assert r3['significant'] is True and r3['verdict'] == 'effect_increases_with_scale', r3

    # Extra A-only seeds must not make the paired trend look heterogeneous:
    # they contribute no paired slope in the first place.
    ragged = _pts(6)
    ragged[0] = {'label': 'A', 'scale': 1.0, 'effects': [0.0] * 6 + [10.0] * 6,
                 'seeds': list(range(12))}
    r5 = L.scaling_trend(ragged, min_seeds=3, min_points=3)
    assert r5['heterogeneity']['I2_percent'] > 75.0, r5['heterogeneity']
    assert r5['heterogeneity_paired']['I2_percent'] == 0.0, r5['heterogeneity_paired']
    assert r5['significant'] is True and r5['verdict'] == 'effect_increases_with_scale', r5

    # A two-point fit is a line; min_points=3 must not be reported as a trend.
    r4 = L.scaling_trend(_pts(6)[:2], min_seeds=3, min_points=3)
    assert r4['p_primary'] < 0.05 and r4['significant'] is False, r4
    assert r4['verdict'] == 'insufficient_points', r4['verdict']


def test_legacy_modes_bitexact():
    for mode, rhos in (('uniform', (0.0, 0.125, 0.5)), ('natural', (0.125, 0.5)), ('mask', (1.0,)), ('delete', (1.0,))):
        for rho in rhos:
            for jj in range(0, 9, 3):
                a = V10._corrupt_row(_PANEL, 128, rho, mode, jj, _VOCAB, 32)
                b = _legacy_corrupt_row(_PANEL, 128, rho, mode, jj, _VOCAB, 32)
                assert np.array_equal(a, b), f'legacy mode {mode} rho={rho} jj={jj} diverged from the reference algorithm'


def test_corruption_deterministic_and_paired():
    for mode in ('uniform', 'natural', 'span', 'doc', 'topical', 'mask', 'delete'):
        rho = 1.0 if mode in ('mask', 'delete', 'doc') else 0.5
        rows = [V10._corrupt_row(_PANEL, 128, rho, mode, jj, _VOCAB, 32) for jj in range(9)]
        again = [V10._corrupt_row(_PANEL, 128, rho, mode, jj, _VOCAB, 32) for jj in range(9)]
        for a, b in zip(rows, again):
            assert np.array_equal(a, b), f'{mode}: corruption must be deterministic (arm-pairing contract)'
        if mode != 'delete':
            for jj, r in enumerate(rows):
                orig = np.asarray(_PANEL[jj, :129], dtype=np.int64)
                n_far = 128 - 32
                assert np.array_equal(r[n_far:], orig[n_far:]), f'{mode}: the scored target zone must never be corrupted (jj={jj})'
                assert r.shape == (129,)


def test_new_modes_do_something():
    span = V10._corrupt_row(_PANEL, 128, 0.5, 'span', 1, _VOCAB, 32)
    orig = np.asarray(_PANEL[1, :129], dtype=np.int64)
    diff = int((span[:96] != orig[:96]).sum())
    assert 0 < diff <= 96, f'span must replace a nonzero, bounded portion of the far context (diff={diff})'
    doc = V10._corrupt_row(_PANEL, 128, 1.0, 'doc', 1, _VOCAB, 32)
    n_far = 96
    _src = (1 + max(1, _PANEL.shape[0] // 2)) % _PANEL.shape[0]
    donor = np.asarray(_PANEL[_src, :129], dtype=np.int64)
    assert not np.array_equal(doc[:n_far], orig[:n_far]), 'doc must replace the far context'
    assert any(np.array_equal(doc[:n_far], donor[o:o + n_far]) for o in range(0, 129 - n_far + 1)), 'doc far context must be ONE contiguous donor slice'
    top = V10._corrupt_row(_PANEL, 128, 0.5, 'topical', 1, _VOCAB, 32)
    assert not np.array_equal(top[:96], orig[:96]), 'topical must replace some tokens'
    nat = V10._corrupt_row(_PANEL, 128, 0.5, 'natural', 1, _VOCAB, 32)
    assert not np.array_equal(top, nat), 'topical (most-similar donor) should differ from natural (fixed-offset donor)'


def test_mode_validation():
    for bad in ('naturaln', '', 'DOC'):
        try:
            V10._corrupt_row(_PANEL, 128, 0.5, bad, 0, _VOCAB, 32)
            raise AssertionError(f'unknown mode {bad!r} must raise')
        except ValueError:
            pass
    for mode in ('mask', 'delete', 'doc'):
        try:
            V10._corrupt_row(_PANEL, 128, 0.5, mode, 0, _VOCAB, 32)
            raise AssertionError(f'{mode} must only admit rho=1.0')
        except ValueError:
            pass
    # An invalid corruption geometry must fail before the seeded RNG works:
    # a negative far context or rho outside [0,1] silently produced an
    # unchanged/partial row that still looked like a measured condition.
    for _kw in ({'eval_len': 64, 'rho': -0.1, 'target': 32},
                {'eval_len': 64, 'rho': 1.1, 'target': 32},
                {'eval_len': 64, 'rho': float('nan'), 'target': 32},
                {'eval_len': 32, 'rho': 0.5, 'target': 64},
                {'eval_len': 64, 'rho': 0.5, 'target': 0}):
        try:
            V10._corrupt_row(_PANEL, _kw['eval_len'], _kw['rho'], 'uniform',
                             0, _VOCAB, _kw['target'])
            raise AssertionError(f'_corrupt_row accepted invalid geometry {_kw!r}')
        except ValueError:
            pass
    try:
        V10._corrupt_row(_PANEL[:1, :64], 128, 0.5, 'uniform', 0, _VOCAB, 32)
    except ValueError:
        pass
    else:
        raise AssertionError('_corrupt_row silently used a panel shorter than eval_len')
    # A one-token vocabulary has no alternative token to draw.  The historical
    # rejection loop redrew the same value forever; the degenerate condition
    # must return the unchanged row instead of hanging the probe.
    _zeros = np.zeros((1, 65), dtype=np.int64)
    _zc = V10._corrupt_row(_zeros, 64, 1.0, 'uniform', 0, 1, 32)
    assert np.array_equal(_zc, _zeros[0]), _zc
    # A non-integral/boolean/non-positive vocabulary is malformed, not a token
    # space to truncate or index.
    for _bad_vocab in (0, -1, True, 1.5):
        try:
            V10._corrupt_row(_PANEL, 128, 0.5, 'uniform', 0, _bad_vocab, 32)
        except ValueError:
            pass
        else:
            raise AssertionError(f'_corrupt_row accepted vocab={_bad_vocab!r}')


def test_power_gate():
    with tempfile.TemporaryDirectory() as td:
        p6 = os.path.join(td, 'm6.json')
        m6 = _write_manifest(p6, list(range(6)))
        m6['primary_comparisons'] = [dict(m6['primary_comparisons'][0], name=f'c{i}') for i in range(4)]
        with open(p6, 'w', encoding='utf-8') as f:
            json.dump(m6, f)
        try:
            V12._load_manifest(p6)
            raise AssertionError('a 6-seed x Holm/4 manifest (floor 0.125 > 0.05) must be refused at load time')
        except SystemExit as e:
            assert 'cannot confirm' in str(e), str(e)[:200]
        p12 = os.path.join(td, 'm12.json')
        m12 = _write_manifest(p12, list(range(12)))
        m = V12._load_manifest(p12)
        assert m['seeds'] == list(range(12)) and len(m['manifest_sha256']) == 64
        # The frozen endpoint grid is exact: a fractional/bool/non-list
        # sequence length would score a different pre-registered endpoint.
        for _bad_lens in ([4096.5], [True], 4096):
            _bad = _write_manifest(os.path.join(td, 'bad_lens.json'), list(range(12)))
            _bad['eval'] = {'by_length_ppl': _bad_lens}
            with open(os.path.join(td, 'bad_lens.json'), 'w', encoding='utf-8') as f:
                json.dump(_bad, f)
            try:
                V12._load_manifest(os.path.join(td, 'bad_lens.json'))
            except SystemExit as e:
                assert 'by_length_ppl' in str(e), e
            else:
                raise AssertionError(f'eval.by_length_ppl={_bad_lens!r} was accepted')
        psm = os.path.join(td, 'smoke.json')
        _write_manifest(psm, [0], extra_stat={'descriptive_only': True})
        V12._load_manifest(psm)  # descriptive smoke manifests opt out
        # A truthy surrogate is not a boolean opt-out: the 6-seed plan still
        # cannot confirm anything and must be refused.
        pstr = os.path.join(td, 'm6_string.json')
        mstr = _write_manifest(pstr, list(range(6)),
                               extra_stat={'descriptive_only': 'false'})
        mstr['primary_comparisons'] = [dict(mstr['primary_comparisons'][0], name=f'c{i}') for i in range(4)]
        with open(pstr, 'w', encoding='utf-8') as f:
            json.dump(mstr, f)
        try:
            V12._load_manifest(pstr)
        except SystemExit as e:
            assert 'real boolean' in str(e), e
        else:
            raise AssertionError('a string descriptive_only opt-out was accepted')


def test_v12_malformed_manifest_and_lock_fail_closed():
    with tempfile.TemporaryDirectory() as td:
        # A scalar/list comparison cannot be read with .get(); it must be
        # named as malformed evidence instead of crashing the loader.
        p = os.path.join(td, 'bad_comps.json')
        m = _write_manifest(p, list(range(12)))
        m['primary_comparisons'] = ['not-an-object']
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(m, f)
        try:
            V12._load_manifest(p)
        except SystemExit as e:
            assert 'primary_comparisons[0]' in str(e) and 'object' in str(e), e
        else:
            raise AssertionError('a scalar primary comparison was accepted')
        # Nested binding blocks are read with .get(); a list cannot be a
        # tokenizer descriptor and must be named as malformed.
        p2 = os.path.join(td, 'bad_binding.json')
        m2 = _write_manifest(p2, list(range(12)))
        m2['binding'] = {'tokenizer': ['not', 'an', 'object']}
        with open(p2, 'w', encoding='utf-8') as f:
            json.dump(m2, f)
        try:
            V12._load_manifest(p2)
        except SystemExit as e:
            assert 'binding.tokenizer' in str(e), e
        else:
            raise AssertionError('a non-object tokenizer binding was accepted')
        # A valid-JSON but non-object freeze lock cannot attest anything;
        # .get() on it used to raise AttributeError after test access.
        outdir = os.path.join(td, 'freeze')
        os.makedirs(outdir)
        lp = os.path.join(outdir, V12._LOCK_NAME)
        for _payload in ([], 'null', 3):
            with open(lp, 'w', encoding='utf-8') as f:
                json.dump(_payload, f)
            try:
                V12._check_or_write_freeze_lock(
                    outdir, {'manifest_sha256': 'a' * 64, 'frozen_at': 'x'},
                    'slice-fp', stage='precheck')
            except SystemExit as e:
                assert 'not a JSON object' in str(e), e
            else:
                raise AssertionError(f'freeze lock payload {_payload!r} was accepted')


def test_admission_gate():
    m = _manifest_for_admission()
    good = {'a::s0': _rec('a_arm', 0, 'FP'), 'b::s0': _rec('b_arm', 0, 'FP'), 'a::s1': _rec('a_arm', 1, 'FP'), 'b::s1': _rec('b_arm', 1, 'FP')}
    adm, fp = V12._admit_confirmation_records(good, m)
    assert len(adm) == 4 and fp == 'FP'
    # off-whitelist seed: excluded, named, never counted
    raw = dict(good, **{'a::s9': _rec('a_arm', 9, 'FP')})
    adm2, _ = V12._admit_confirmation_records(raw, m)
    assert (u'a_arm', 9) not in adm2 and len(adm2) == 4
    # None data_fp (the None==None pairing hole): excluded
    raw = dict(good, **{'a::s2': _rec('a_arm', 2, None)})
    adm3, _ = V12._admit_confirmation_records(raw, m)
    assert (u'a_arm', 2) not in adm3
    # wrong manifest sha: excluded (stale revision)
    raw = dict(good, **{'a::s2': _rec('a_arm', 2, 'FP', msha='OLD')})
    adm4, _ = V12._admit_confirmation_records(raw, m)
    assert (u'a_arm', 2) not in adm4
    # wrong code stamp: excluded
    raw = dict(good, **{'a::s2': _rec('a_arm', 2, 'FP', code='v9.9')})
    adm5, _ = V12._admit_confirmation_records(raw, m)
    assert (u'a_arm', 2) not in adm5


def test_admission_refuses():
    m = _manifest_for_admission()
    # duplicate admissible records -> REFUSE (not overwrite)
    raw = {'a::s0': _rec('a_arm', 0, 'FP'), 'a::s0.0': _rec('a_arm', 0, 'FP'), 'b::s0': _rec('b_arm', 0, 'FP')}
    try:
        V12._admit_confirmation_records(raw, m)
        raise AssertionError('duplicate (variant, seed) must be refused, not overwritten')
    except SystemExit as e:
        assert 'duplicate' in str(e).lower() or 'ambiguous' in str(e).lower()
    # mixed fingerprints across seed pairs -> REFUSE (the slice must be ONE)
    raw = {'a::s0': _rec('a_arm', 0, 'FP1'), 'b::s0': _rec('b_arm', 0, 'FP1'), 'a::s1': _rec('a_arm', 1, 'FP2'), 'b::s1': _rec('b_arm', 1, 'FP2')}
    try:
        V12._admit_confirmation_records(raw, m)
        raise AssertionError('records spanning two test slices must be refused (pairwise-equal fps are not enough)')
    except SystemExit as e:
        assert 'slice' in str(e).lower()


def test_admission_refuses_reused_checkpoint_digest():
    m = _manifest_for_admission()
    _A, _B = 'a' * 64, 'b' * 64
    raw = {'a::s0': _rec('a_arm', 0, 'FP', ckpt=_A),
           'b::s0': _rec('b_arm', 0, 'FP', ckpt=_B)}
    adm, _ = V12._admit_confirmation_records(raw, m)
    assert len(adm) == 2, adm
    # One set of weights cannot be re-labelled as another arm/seed: that is a
    # duplicate measurement, not an independent repeat.
    raw = {'a::s0': _rec('a_arm', 0, 'FP', ckpt=_A),
           'b::s0': _rec('b_arm', 0, 'FP', ckpt=_A)}
    try:
        V12._admit_confirmation_records(raw, m)
        raise AssertionError('a reused checkpoint digest must be refused')
    except SystemExit as e:
        assert 'checkpoint' in str(e).lower() and 'reused' in str(e).lower(), e
    # Once the manifest (or any record) declares checkpoint binding, a record
    # without a valid digest is excluded instead of silently bypassing it.
    m2 = _manifest_for_admission()
    m2['binding'] = {'require_ckpt_sha256': True}
    raw = {'a::s0': _rec('a_arm', 0, 'FP'),
           'b::s0': _rec('b_arm', 0, 'FP', ckpt=_B)}
    adm, _ = V12._admit_confirmation_records(raw, m2)
    assert ('a_arm', 0) not in adm and ('b_arm', 0) in adm, adm
    raw = {'a::s0': _rec('a_arm', 0, 'FP', ckpt='not-a-digest')}
    adm, _ = V12._admit_confirmation_records(raw, m2)
    assert ('a_arm', 0) not in adm, adm


def test_recipe_signature():
    recipe = 'steps3000_sl512_bs12_nt8000000_lr0.0003_wd0.1_d256_L6_H8_Dh32_v8192_wu50_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee250_es128_detcpu-eager_dfdeadbeef_cscURRENT'
    sig = {'steps': 3000, 'sl': 512, 'bs': 12, 'nt': 8000000, 'lr': 0.0003, 'd': 256, 'L': 6}
    assert V12._recipe_matches_signature(recipe, sig) is True
    bad = dict(sig, steps=40)
    assert V12._recipe_matches_signature(recipe, bad) is False
    assert V12._recipe_matches_signature(None, sig) is False
    assert V12._recipe_matches_signature(recipe, None) is True  # no signature declared -> nothing to enforce
    # the SHIPPED v2 manifest's signature must match its own declared recipe
    with open(os.path.join(REPO, 'confirmation_frozen.json'), encoding='utf-8') as f:
        v2 = json.load(f)
    shipped = v2['binding']['training_recipe_signature']
    assert V12._recipe_matches_signature(recipe, shipped) is True, 'the shipped v2 signature must accept the canonical P3MT recipe string'


def test_freeze_lock():
    with tempfile.TemporaryDirectory() as td:
        man = {'manifest_sha256': 'AAA', 'frozen_at': '2026-10-17'}
        # two-phase lock: pre-access attestation is written BEFORE the test
        # slice is read, then bound to the slice fingerprint immediately after.
        pre = V12._check_or_write_freeze_lock(td, man, None, stage='preaccess')
        assert pre.get('stage') == 'preaccess' and pre.get('test_ids_fp') is None, pre
        lock = V12._check_or_write_freeze_lock(td, man, 'FP', stage='accessed')
        assert lock['manifest_sha256'] == 'AAA' and lock['test_ids_fp'] == 'FP'
        assert lock.get('stage') == 'accessed', lock
        # changed test slice -> refuse
        try:
            V12._check_or_write_freeze_lock(td, man, 'FP2')
            raise AssertionError('a changed test slice after first access must be refused')
        except SystemExit:
            pass
        # changed manifest -> refuse; V12_REFREEZE=1 downgrades with a warning
        man2 = {'manifest_sha256': 'BBB', 'frozen_at': '2026-10-17'}
        try:
            V12._check_or_write_freeze_lock(td, man2, 'FP')
            raise AssertionError('a changed manifest after first access must be refused')
        except SystemExit:
            pass
        os.environ['V12_REFREEZE'] = '1'
        try:
            V12._check_or_write_freeze_lock(td, man2, 'FP')
        finally:
            del os.environ['V12_REFREEZE']
        # Once a training fingerprint is locked, an analysis call without one
        # is a provenance downgrade and must be refused.
        V12._check_or_write_freeze_lock(td, man2, 'FP', train_fp='TRAIN')
        try:
            V12._check_or_write_freeze_lock(td, man2, 'FP', train_fp=None)
            raise AssertionError('a missing train_fp after it was locked must be refused')
        except SystemExit:
            pass


def test_freeze_lock_records_manifest_source_path():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'renamed_freeze.json')
        _write_manifest(p, [0, 1], extra_stat={'descriptive_only': True})
        m = V12._load_manifest(p)
        assert m['_path'] == os.path.abspath(p), m.get('_path')
        lock = V12._check_or_write_freeze_lock(td, m, None, stage='preaccess')
        assert lock['manifest_path'] == os.path.abspath(p), lock


def test_p1l_recipe_signature():
    # P5T evaluates checkpoints written by v7's length-phase writer, whose
    # recipe is the bespoke `p1l_...` format (not a run_cfg fingerprint).
    with open(os.path.join(REPO, 'confirmation_frozen.json'), encoding='utf-8') as f:
        v2 = json.load(f)
    sig = v2['binding']['training_recipe_signature']
    p1l = ('p1l_steps3000_tl512_v8192_bs12_lr0.0003_nt8000000_wu50_wd0.1_'
           'cl0.05_dlm10.0_mr4-csa_fixed_rope_dfdeadbeef')
    assert V12._recipe_knobs(p1l)['steps'] == '3000'
    assert V12._recipe_knobs(p1l)['sl'] == '512'
    assert V12._recipe_matches_signature(p1l, sig) is True, 'the P1L checkpoint recipe must satisfy the shipped signature'
    assert V12._recipe_matches_signature(p1l.replace('steps3000', 'steps40'), sig) is False
    assert V12._recipe_matches_signature('p1l_garbage', sig) is False


def test_aggregate_seed_normalization():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    base = dict(variant='csa_dynamic', run_cfg=fp, steps=1, tokens_seen=1, params=100, ppl=10.0)
    raw = {'a::seed0': dict(base, seed=0), 'b::seed0': dict(base, seed='0')}
    try:
        L.aggregate(raw)
        raise AssertionError('seed 0 and "0" are the same seed and must trip the ambiguous-duplicate guard, not be counted as n=2')
    except ValueError:
        pass
    # Fractional/boolean seeds are not extra draws: exclude them rather than
    # inflating n or emitting a non-integer seed in the panel summary.
    assert L.aggregate({'x::seedbad': dict(base, seed=1.5)}) == {}
    assert L.aggregate({'x::seedbad': dict(base, seed=True)}) == {}
    # A malformed arm id is unhashable and must be excluded, not crash the
    # whole panel aggregation.
    assert L.aggregate({'x::seed0': dict(base, seed=0, variant={'bad': 1})}) == {}
    assert L.aggregate({'x::seed0': dict(base, seed=0, variant='')}) == {}
    # A record whose optional diagnostics are explicitly null must not crash
    # the panel aggregate; it simply contributes no block-length readout.
    _null_stats = L.aggregate({'x::seed0': dict(base, seed=0, stats=None)})
    assert _null_stats['csa_dynamic']['n_seeds'] == 1, _null_stats
    assert 'avg_block_len' not in _null_stats['csa_dynamic'], _null_stats
    # A malformed history must yield an empty/omitted curve, not an exception
    # from the boolean reduction over a zero-column matrix.
    _bad_hist = L.aggregate({'x::seed0': dict(base, seed=0, ppl_history=[1])})
    assert _bad_hist['csa_dynamic'].get('ppl_curve') in (None, []), _bad_hist
    _bad_hist2 = L.aggregate({'x::seed0': dict(base, seed=0, ppl_history=[{}, 'x'])})
    assert _bad_hist2['csa_dynamic'].get('ppl_curve') in (None, []), _bad_hist2


def test_audit_resolution_classification():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'

    def _summary(deltas):
        s = {}
        for i, d in enumerate(deltas):
            s[f'a::seed{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1, tokens_seen=1, params=100, ppl=50.0 + d)
            s[f'b::seed{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1, tokens_seen=1, params=100, ppl=50.0)
        return s
    design = {'c': [('a', 'b', 'test')]}
    # n=6, effect > noise, p=0.0625 > alpha, but the exact-test floor (0.031)
    # is BELOW alpha -> a genuine null-ish reading, NOT a resolution ceiling.
    r = L.methodology_audit(_summary([1, 1, 1, 1, 1, -0.5]), design=design, verbose=False)
    assert 'c' in r['noise_level_components'] and 'c' not in r['resolution_limited_components'], r
    # n=4 strong effect -> the floor (0.125) is above alpha -> resolution-limited.
    r2 = L.methodology_audit(_summary([1, 1, 1, 0.9]), design=design, verbose=False)
    assert 'c' in r2['resolution_limited_components'], r2


def test_pair_reason_train_knobs_mixed():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    a = dict(variant='a', seed=0, run_cfg=fp, steps=1, tokens_seen=1, ppl=10.0, train_knobs=L.training_knobs())
    b = dict(variant='b', seed=0, run_cfg=fp, steps=1, tokens_seen=1, ppl=11.0)
    r = L.pair_reason(a, b)
    assert r is not None and 'unverifiable train_knobs' in r, f'one-sided train_knobs presence must be refused, got {r!r}'
    r2 = L.pair_reason({k: v for k, v in a.items() if k != 'train_knobs'}, b)
    assert r2 is None, 'both records missing the descriptor is the legacy path and may still pair'


def test_pair_reason_fingerprint_gate():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    base = dict(variant='a', seed=0, run_cfg=fp, steps=1, tokens_seen=1, ppl=10.0,
                data_fp='train-a', val_fp='val-a', eval_fp='eval-a')
    b = dict(base, variant='b')
    assert L.pair_reason(base, b) is None
    # Same run_cfg but a different corpus / validation slice is NOT a
    # controlled contrast, even though every numeric knob is identical.
    assert 'data_fp' in L.pair_reason(base, dict(b, data_fp='train-b'))
    assert 'val_fp' in L.pair_reason(base, dict(b, val_fp='val-b'))
    assert 'eval_fp' in L.pair_reason(base, dict(b, eval_fp='eval-b'))
    # The zero-GPU report builder has a local pair gate; it must mirror this.
    assert 'data_fp' in BR._pair_reason(base, dict(b, data_fp='train-b'))
    assert 'unverifiable val_fp' in BR._pair_reason(base, {k: v for k, v in b.items() if k != 'val_fp'})
    # One-sided presence is unverifiable, not silently compatible.
    c = dict(b)
    c.pop('val_fp')
    assert 'unverifiable val_fp' in L.pair_reason(base, c)
    # Nested records can share the full-val/eval fingerprints yet score a
    # different held-out row fold.  Both pairing twins must compare it.
    h = dict(b, holdout_row_fp='h1')
    assert L.pair_reason(h, dict(h, variant='a')) is None
    assert BR._pair_reason(h, dict(h, variant='a')) is None
    assert 'holdout_row_fp' in (L.pair_reason(h, dict(h, holdout_row_fp='h2')) or '')
    assert 'holdout_row_fp' in (BR._pair_reason(h, dict(h, holdout_row_fp='h2')) or '')
    assert 'unverifiable holdout_row_fp' in L.pair_reason(base, h)
    assert 'unverifiable holdout_row_fp' in BR._pair_reason(base, h)
    # Legacy records (both without fingerprints) keep the compatibility path.
    leg = {k: v for k, v in base.items() if k not in ('data_fp', 'val_fp', 'eval_fp')}
    assert L.pair_reason(leg, dict(leg, variant='b')) is None
    # aggregate must refuse to pool one group across two evaluation slices.
    summary = {
        'a::seed0': dict(base, variant='a'),
        'a::seed1': dict(base, variant='a', seed=1, ppl=10.5, val_fp='val-b'),
        'b::seed0': dict(b),
        'b::seed1': dict(b, seed=1, ppl=11.0),
    }
    try:
        L.aggregate(summary)
    except ValueError as e:
        assert 'different corpus/evaluation fingerprints' in str(e), e
    else:
        raise AssertionError('aggregate must reject a group pooled across validation slices')
    # Same train/validation but a different fixed evaluation slice is still a
    # mixed target set for the pooled group aggregate.
    eval_mix = {
        'a::seed0': dict(base, variant='a', eval_fp='eval-a'),
        'a::seed1': dict(base, variant='a', seed=1, ppl=10.5, eval_fp='eval-b'),
    }
    try:
        L.aggregate(eval_mix)
    except ValueError as e:
        assert 'eval_fp' in str(e), e
    else:
        raise AssertionError('aggregate must reject a group pooled across evaluation slices')
    # The fixed evaluation SUBSET is part of the same identity: two records
    # scored on different subsets have different target denominators even when
    # train/validation/eval fingerprints agree.
    subset_mix = {
        'a::seed0': dict(base, variant='a', eval_subset_fp='subset-a'),
        'a::seed1': dict(base, variant='a', seed=1, ppl=10.5,
                         eval_subset_fp='subset-b'),
    }
    try:
        L.aggregate(subset_mix)
    except ValueError as e:
        assert 'eval_subset_fp' in str(e), e
    else:
        raise AssertionError('aggregate must reject a group pooled across evaluation subsets')
    # Nested records on different held-out row folds are also different
    # samples, even when the full validation/eval fingerprints agree.
    holdout_mix = {
        'a::seed0': dict(base, variant='a',
                         ppl_estimation_rows='nested_validation_holdout',
                         ppl_holdout=10.0, holdout_row_fp='fold-a'),
        'a::seed1': dict(base, variant='a', seed=1, ppl=10.5,
                         ppl_estimation_rows='nested_validation_holdout',
                         ppl_holdout=10.5, holdout_row_fp='fold-b'),
    }
    try:
        L.aggregate(holdout_mix)
    except ValueError as e:
        assert 'holdout_row_fp' in str(e), e
    else:
        raise AssertionError('aggregate must reject a group pooled across held-out row folds')
    rows = [dict(base, by_len={8: {'ppl': 10.0}}),
            dict(base, variant='b', val_fp='val-b', by_len={8: {'ppl': 11.0}})]
    try:
        L.by_len_cells(rows, 8)
    except ValueError as e:
        assert 'different validation slices' in str(e), e
    else:
        raise AssertionError('by_len_cells must reject rows from different validation slices')


def test_m0_scale_analysis_descriptive():
    fp = 'steps10_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    summary = {}
    for v, base in (('csa_dynamic', 10.0), ('csa_fixed', 10.5),
                    ('hybrid_dynamic', 10.2), ('full_matched', 11.0),
                    ('full_sw128_matched', 10.8)):
        for seed in range(4):
            summary[f'{v}::seed{seed}'] = dict(
                variant=v, seed=seed, run_cfg=fp, steps=10, tokens_seen=10,
                ppl=base + 0.1 * seed, params=100, train_knobs=knobs,
                data_fp='d', val_fp='v')
    # Tier discovery is the B-tier ladder only, not the earlier RUN_SCALE /
    # RUN_LONG / RUN_ABL exploratory panels; explicit extra tiers are allowed
    # and duplicate declared seeds do not inflate the planned n.
    assert set(M0._tier_run_cfgs()) == {'S1', 'S2'}, M0._tier_run_cfgs()
    extra = M0._tier_run_cfgs({'S3': {'outdir': 'results_b_s3',
                                      'seeds': [2, 0, 2, 1]}})
    assert extra['S3']['outdir'] == 'results_b_s3', extra
    assert M0._declared_seed_list(extra['S3']) == [0, 1, 2], extra
    tmp = tempfile.mkdtemp()
    try:
        r = M0.scale_analysis({'S1': summary}, outdir=tmp)
        p = r['panels']['S1']
        assert p['n_seeds'] == 4 and p['descriptive_only'] is True, p
        names = {(c['name'], c['with'], c['without']) for c in p['contrasts']}
        assert ('dynamic_blocking', 'csa_dynamic', 'csa_fixed') in names, names
        assert ('hybrid_hierarchy', 'hybrid_dynamic', 'csa_dynamic') in names, names
        assert all(c['resolution_limited'] for c in p['contrasts'] if c['n_paired']), p['contrasts']
        assert os.path.exists(os.path.join(tmp, 'scale_analysis.json'))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_m0_result_diagnostics():
    summary = {
        'a::seed0': {'variant': 'a', 'seed': 0, 'ppl': 10.0, 'params': 100,
                     'curve_health': {'status': 'healthy', 'final_minus_best': 0.0},
                     'efficiency': {'status': 'ok', 'prefill_ms_p50': 2.0,
                                    'peak_mem_allocated_mb': 10.0},
                     'train_tokens_per_s': 1000.0,
                     'peak_memory': {'max_memory_allocated_mb': 12.0}},
        'a::seed1': {'variant': 'a', 'seed': 1, 'ppl': 10.2, 'params': 100,
                     'curve_health': {'status': 'overfitting', 'final_minus_best': 0.3},
                     'efficiency': {'status': 'ok', 'prefill_ms_p50': 2.2,
                                    'peak_mem_allocated_mb': 11.0},
                     'train_tokens_per_s': 990.0},
        'b::seed0': {'variant': 'b', 'seed': 0, 'ppl': 10.1, 'params': 250,
                     'curve_health': {'status': 'nonfinite'},
                     'efficiency': {'status': 'ok', 'prefill_ms_p50': 8.0,
                                    'peak_mem_allocated_mb': 30.0},
                     'train_tokens_per_s': 300.0,
                     'early_stop': {'stopped': True}},
    }
    d = M0._panel_result_diagnostics(summary)
    assert d['n_records'] == 3, d
    assert d['curve_health_counts']['healthy'] == 1, d
    assert d['records']['nonfinite'] == ['b::seed0'], d
    assert d['records']['overfitting'] == ['a::seed1'], d
    assert d['records']['final_above_best'] == ['a::seed1'], d
    assert d['records']['early_stopped'] == ['b::seed0'], d
    assert d['best_ppl_arm'] == 'a', d
    assert 'latency_2x_or_more' in d['cost']['b']['flags'], d
    assert 'params_2x_or_more' in d['cost']['b']['flags'], d
    assert 'memory_2x_or_more' in d['cost']['b']['flags'], d
    assert 'cost_increase_without_clear_gain' in d['cost']['b']['flags'], d
    assert d['train_curve_health_counts'].get('missing') == 3, d


def test_seed_plan():
    p = L.methodology_seed_plan(3)
    assert p['declared_contrasts'] >= 10, p
    assert p['min_seeds_for_family_significance'] > 3
    assert p['sufficient_for_family_rejection'] is False
    assert 'ablation_completeness' in p and p['ablation_complete'] is True, p
    req = p['min_seeds_for_family_significance']
    assert L.holm_min_attainable_p(req, p['declared_contrasts'], 0.05) <= 0.05
    assert L.holm_min_attainable_p(req - 1, p['declared_contrasts'], 0.05) > 0.05
    assert L.methodology_seed_plan(req)['sufficient_for_family_rejection'] is True
    # A fractional/bool plan size must not be truncated into a different n.
    for _bad_n in (req + 0.9, True, float('nan'), 'not-a-count'):
        _p = L.methodology_seed_plan(_bad_n)
        assert _p['n_seeds'] == 0, (_p, _bad_n)
        assert _p['sufficient_for_family_rejection'] is False, (_p, _bad_n)


def test_audit_family_holm_and_verified():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()

    def _summary(n, effect=-1.0):
        s = {}
        for i in range(n):
            s[f'csa_dynamic::seed{i}'] = dict(variant='csa_dynamic', seed=i, run_cfg=fp, steps=1, tokens_seen=1, ppl=50.0 + effect, params=1000, train_knobs=knobs)
            s[f'full_sw128_matched::seed{i}'] = dict(variant='full_sw128_matched', seed=i, run_cfg=fp, steps=1, tokens_seen=1, ppl=50.0, params=1001, train_knobs=knobs)
        return s
    # Only the baseline component is actually present in this panel, so the
    # correction family is its two declared contrasts, not every unmeasured
    # ablation claim.
    r3 = L.methodology_audit(_summary(3), verbose=False)
    assert r3['family']['effective_m'] == 2, r3['family']
    assert 'compression_vs_matched_dense' in r3['resolution_limited_baselines'], r3['resolution_limited_baselines']
    assert 'compression_vs_matched_dense' not in r3['verified_baselines']
    r6 = L.methodology_audit(_summary(6), verbose=False)
    assert r6['family']['effective_m'] == 2
    assert 'compression_vs_matched_dense' not in r6['verified_baselines'], 'm=2 needs n>=7 for a Holm-corrected rejection'
    # n=7: all seeds same sign now clears the family-corrected gate.
    r7 = L.methodology_audit(_summary(7), verbose=False)
    assert 'compression_vs_matched_dense' in r7['verified_baselines'], r7['verified_baselines']
    e = r7['baselines']['compression_vs_matched_dense'][0]
    assert e['p_holm_global'] > e['p_exact_signflip'], 'Holm-adjusted p must be >= raw p'
    # an incomplete component keeps ALL its declared slots (dummy p=1), so a
    # missing arm cannot shrink the family.
    design = {'c': [('a', 'b', 'x'), ('a', 'c', 'y')]}
    incomplete = {'a::seed0': dict(variant='a', seed=0, run_cfg=fp, steps=1, tokens_seen=1, ppl=49.0, params=100, train_knobs=knobs),
                  'b::seed0': dict(variant='b', seed=0, run_cfg=fp, steps=1, tokens_seen=1, ppl=50.0, params=100, train_knobs=knobs)}
    ri = L.methodology_audit(incomplete, design=design, baselines={}, verbose=False)
    assert ri['family']['declared_contrasts'] == 2 and ri['family']['effective_m'] == 2, ri['family']
    # parameter mismatch is a hard control failure for a baseline claim
    s = _summary(7)
    for i in range(7):
        s[f'full_sw128_matched::seed{i}']['params'] = 2000
    rbad = L.methodology_audit(s, verbose=False)
    assert 'compression_vs_matched_dense' not in rbad['verified_baselines'], rbad['verified_baselines']
    assert 'compression_vs_matched_dense' in rbad['unverified_baselines']


def test_ablation_design_loss_term():
    assert 'compression_regularizer' in L.ABLATION_DESIGN, 'the compression loss term must have a leave-one-out arm'
    pairs = L.ABLATION_DESIGN['compression_regularizer']
    assert any(a == 'csa_dynamic' and b == 'csa_dyn_nocomp' for a, b, _n in pairs), pairs
    cfgs = L.make_layer_cfgs(2, 'csa_dyn_nocomp')
    assert all(getattr(c, 'comp_lambda_mult', None) == 0.0 for c in cfgs), 'the nocomp arm must zero the loss multiplier'
    ref = L.make_layer_cfgs(2, 'csa_dynamic')
    assert all(getattr(c, 'comp_lambda_mult', 1.0) == 1.0 for c in ref), 'the full arm must keep the regularizer'
    # every declared invention still has at least one concrete contrast
    for comp, contrasts in L.ABLATION_DESIGN.items():
        assert contrasts and all(len(t) == 3 for t in contrasts), f'{comp}: malformed leave-one-out design'
    assert any(v == 'csa_dyn_nocomp' for v in L.ABL_VARIANTS), 'the loss ablation arm must be scheduled by RUN_ABL'
    # Every arm named by a declared contrast must be in the SAME panel as its
    # reference.  A declared-but-unscheduled arm silently turns a complete
    # leave-one-out design into a partial one at run time.
    declared = {x for cs in L.ABLATION_DESIGN.values()
                for a, b, _n in cs for x in (a, b)}
    missing = declared - set(L.RUN_ABL['variants'])
    assert not missing, f'declared ablation arms not scheduled by RUN_ABL: {sorted(missing)}'
    assert L.ablation_completeness_audit(L.ABLATION_DESIGN, L.RUN_ABL['variants'])['complete'] is True
    # Dynamic blocking must be decomposed by layer kind, not only measured as
    # "all-dynamic vs all-fixed".  Otherwise hybrid_hca_dyn is scheduled but
    # never contributes to any declared contrast, and the HCA boundary effect
    # cannot be separated from the CSA boundary effect.
    _dyn = {(a, b) for a, b, _n in L.ABLATION_DESIGN['dynamic_blocking']}
    assert ('hybrid_csa_dyn', 'hybrid_fixed') in _dyn, _dyn
    assert ('hybrid_hca_dyn', 'hybrid_fixed') in _dyn, _dyn
    assert any('hybrid_hca_dyn' in (a, b)
               for cs in L.ABLATION_DESIGN.values() for a, b, _n in cs), \
        'hybrid_hca_dyn is scheduled but belongs to no declared contrast'


def test_training_knobs_in_resume_fingerprint():
    cfg = dict(steps=1, seq_len=8, batch_size=1, n_train_tokens=10, lr=0.0003,
               weight_decay=0.1, warmup=1, comp_lambda=0.05, delta_lr_mult=10.0)
    tr = np.arange(16, dtype=np.int64)
    va = np.arange(8, dtype=np.int64).reshape(1, 8)
    fp1 = L._run_fp(dict(cfg), 8, 1, 1, 8, 16, tr, va)
    fp2 = L._run_fp(dict(cfg), 8, 1, 1, 8, 16, tr, va)
    assert fp1 == fp2 and '_tk' in fp1, (fp1, fp2)
    assert L.run_cfg_knobs(fp1).get('tk') == L.training_knobs_fingerprint()
    orig = L.training_knobs
    fp3 = None
    try:
        L.training_knobs = lambda: dict(orig(), grad_clip_norm=2.0)
        fp3 = L._run_fp(dict(cfg), 8, 1, 1, 8, 16, tr, va)
    finally:
        L.training_knobs = orig
    assert fp3 is not None and fp3 != fp1, 'changing optimizer/schedule/clip knobs must invalidate resume (no silent mixed panel)'
    # The sampled long-context segmenter's ceiling changes the cut geometry, so
    # it must invalidate a resumed cell even when the exact-limit stamp matches.
    _old_am = os.environ.get('CSA_ADAPTIVE_MAX_TOKENS')
    try:
        os.environ['CSA_ADAPTIVE_MAX_TOKENS'] = '2048'
        fp4 = L._run_fp(dict(cfg), 8, 1, 1, 8, 16, tr, va)
    finally:
        if _old_am is None:
            os.environ.pop('CSA_ADAPTIVE_MAX_TOKENS', None)
        else:
            os.environ['CSA_ADAPTIVE_MAX_TOKENS'] = _old_am
    assert fp4 != fp1 and '_am2048' in fp4, (fp1, fp4)
    # The task/synthetic recipe is part of the data identity.
    _de1 = L._run_fp(dict(cfg, data_extra={'n_pairs': 4}), 8, 1, 1, 8, 16, tr, va)
    _de2 = L._run_fp(dict(cfg, data_extra={'n_pairs': 8}), 8, 1, 1, 8, 16, tr, va)
    assert _de1 != _de2 and '_de' in _de1, (_de1, _de2)
    assert L._run_fp(dict(cfg), 8, 1, 1, 8, 16, tr, va) == fp1, 'unset data_extra must keep legacy fingerprints stable'
    # Early stopping changes the completed token budget, so the policy must
    # invalidate a resumed cell (and be part of the record's recipe).
    _es1 = L._run_fp(dict(cfg, early_stop_patience=2), 8, 1, 1, 8, 16, tr, va)
    _es2 = L._run_fp(dict(cfg, early_stop_patience=3), 8, 1, 1, 8, 16, tr, va)
    assert _es1 != _es2 and _es1 != fp1 and '_esp2' in _es1, (_es1, _es2, fp1)
    assert L._run_fp(dict(cfg), 8, 1, 1, 8, 16, tr, va) == fp1, 'unset early stop must keep legacy fingerprints stable'


def test_controlled_variables_all_seed_records():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    s = {}
    for i in range(3):
        s[f'a::seed{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=49.0, params=100, train_knobs=dict(knobs))
        s[f'b::seed{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=50.0, params=100, train_knobs=dict(knobs))
    # One seed of arm `a` silently ran with a different optimizer setting: the
    # panel-level audit must flag the mixed arm, even though the first record
    # looks identical and pair_reason refuses only that seed.
    s['a::seed0']['train_knobs'] = dict(knobs, grad_clip_norm=2.0)
    r = L.methodology_audit(s, design={'c': [('a', 'b', 'x')]}, baselines={}, verbose=False)
    cv = r['controlled_variables']['c']
    assert cv['train_knobs_equal'] is False, cv
    assert 'a' in cv['within_arm_mixed'], cv
    assert r['components']['c'][0]['n_paired'] == 2, r['components']['c'][0]


def test_training_knobs_must_be_complete():
    # A controlled recipe is attested by the FULL descriptor.  Two records
    # carrying the same partial dictionary cannot prove the missing
    # optimizer/schedule/augmentation fields were shared.
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    base = dict(variant='a', seed=0, run_cfg=fp, steps=1, tokens_seen=1,
                ppl=10.0, params=100, train_knobs=knobs)
    assert L.pair_reason(base, dict(base, variant='b')) is None
    # Both absent remains the documented legacy-vs-legacy path.
    assert L.pair_reason(dict(base, train_knobs=None),
                         dict(base, variant='b', train_knobs=None)) is None
    partial = {k: v for k, v in knobs.items()
               if k not in ('lr_schedule', 'augmentation')}
    ra = dict(base, train_knobs=partial)
    rb = dict(base, variant='b', train_knobs=dict(partial))
    why = L.pair_reason(ra, rb)
    assert why and 'omits controlled field' in why, why
    # The zero-GPU report twin must refuse exactly the same pair.
    rwhy = BR._pair_reason(ra, rb)
    assert rwhy and 'omits controlled field' in rwhy, rwhy
    # Malformed numeric fields are not a controlled recipe either.
    for bad in (dict(knobs, betas=[0.9, float('nan')]),
                dict(knobs, eps=0.0),
                dict(knobs, augmentation='')):
        assert L.pair_reason(dict(base, train_knobs=bad),
                             dict(base, variant='b', train_knobs=dict(bad))), bad
    # The tuned-baseline relaxed gate must not become a back door.
    tuned = {'a': 0, 'b': 0}
    # Give the relaxed rule its data fingerprints so it reaches the descriptor
    # check instead of stopping at missing corpus evidence.
    ra2 = dict(ra, data_fp='d', val_fp='v')
    rb2 = dict(rb, data_fp='d', val_fp='v')
    _relaxed = BT._relaxed_pair_reason(ra2, rb2, tuned)
    assert _relaxed and 'omits controlled field' in _relaxed, _relaxed
    # Panel-level report names the defect and refuses to call it controlled.
    s = {}
    for i in range(3):
        s[f'a::seed{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1,
                                tokens_seen=1, ppl=49.0, params=100,
                                train_knobs=dict(partial))
        s[f'b::seed{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1,
                                tokens_seen=1, ppl=50.0, params=100,
                                train_knobs=dict(partial))
    rep = L.methodology_audit(s, design={'c': [('a', 'b', 'x')]},
                              baselines={}, verbose=False)
    cv = rep['controlled_variables']['c']
    assert cv['controlled'] is False, cv
    assert set(cv['train_knobs_defects']) == {'a', 'b'}, cv
    assert any('train_knobs descriptor(s) are incomplete' in w
               for w in rep['warnings']), rep['warnings']


def test_underpowered_contrast_is_never_significant():
    # alpha=0.6 makes the raw exact p (0.5 at n=2) clear the threshold.  The
    # three-seed floor must still prevent a positive verdict.
    r = L.paired_effect_stats([-1.0, -1.2], alpha=0.6, min_n=3)
    assert r['underpowered'] is True, r
    assert r['significant'] is False, r
    assert r['std_defined'] is True, r
    # One seed has no estimable sample spread: keep the legacy 0.0 but say so.
    one = L.paired_effect_stats([-1.0])
    assert one['n'] == 1 and one['std'] == 0.0, one
    assert one['std_defined'] is False and one['significant'] is False, one
    zero = L.paired_effect_stats([])
    assert zero['std_defined'] is False and zero['significant'] is False, zero
    # A powered, resolvable, same-direction panel remains significant.
    r2 = L.paired_effect_stats([-1.0, -1.2, -0.8, -1.1, -0.9, -1.0],
                               alpha=0.05, min_n=3)
    assert r2['underpowered'] is False and r2['significant'] is True, r2
    assert r2['std_defined'] is True, r2


def test_report_pairs_cover_the_declared_ablation_design():
    pairs = L.declared_ablation_pairs()
    declared = {(a, b) for cs in L.ABLATION_DESIGN.values()
                for a, b, *_ in cs}
    assert set(pairs) == declared, (set(pairs) ^ declared)
    assert len(pairs) == len(set(pairs)), 'declared pair list must be de-duplicated'
    # The task-learned-boundary contrast is a declared innovation; the
    # console pair table must not silently omit it (the historical drift bug).
    assert ('csa_dynamic_task', 'csa_dynamic') in pairs, pairs
    # A malformed design yields no crash and no fabricated pairs.
    assert L.declared_ablation_pairs({'A': 'not-a-list'}) == []
    assert L.declared_ablation_pairs({'A': [('a', 'b', 'x'),
                                            ('a', 'b', 'again')]}) == [('a', 'b')]


def test_partial_ablation_cannot_verify_a_component():
    design = {'A': [('full', 'full-A', ''), ('full', 'full-A2', '')]}
    partial = {
        'ablation_completeness': {
            'components': {'A': {'complete': False}},
            'complete_components': [], 'incomplete_components': ['A'],
            'missing_arms': {'A': ['full-A2']}},
        'measured_components': ['A'],
        'verified_components': ['A'],
        'unverified_components': []}
    cov = L.ablation_coverage_across_panels({'p': partial}, design)
    row = cov['components']['A']
    assert row['panels_verified_raw'] == ['p'], row
    assert row['panels_verified'] == [], row
    assert row['panels_verified_incomplete'] == ['p'], row
    assert cov['verified_anywhere'] == [], cov
    assert cov['verified_only_incomplete'] == ['A'], cov
    assert any('whole innovation' in w for w in cov['warnings']), cov
    # Once the declared ablation is complete, the same verified panel counts.
    complete = dict(partial)
    complete['ablation_completeness'] = {
        'components': {'A': {'complete': True}},
        'complete_components': ['A'], 'incomplete_components': [],
        'missing_arms': {}}
    cov2 = L.ablation_coverage_across_panels({'p': complete}, design)
    assert cov2['verified_anywhere'] == ['A'], cov2
    assert cov2['verified_only_incomplete'] == [], cov2
    # A project-level "every innovation is useful" claim is only open when a
    # complete ablation verified every declared component.
    assert cov2['all_components_useful_allowed'] is True, cov2
    assert cov['all_components_useful_allowed'] is False, cov
    assert cov['unverified_components'] == ['A'], cov
    assert any('not verified' in r for r in cov['all_components_useful_reasons']), cov
    incomplete_only = dict(complete, verified_components=['A'],
                           ablation_completeness=partial['ablation_completeness'])
    cov3 = L.ablation_coverage_across_panels({'p': incomplete_only}, design)
    assert cov3['all_components_useful_allowed'] is False, cov3
    assert any('incomplete ablation' in r
               for r in cov3['all_components_useful_reasons']), cov3


def test_ids_fp_ignores_malformed_sidecar():
    # A valid-JSON list/scalar in the .idsfp.json sidecar is not a fingerprint
    # record.  It must be ignored (and the array hashed) rather than crashing
    # the whole cache load with AttributeError.
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'ids.npy')
        arr = np.memmap(path, dtype=np.int64, mode='w+', shape=(8,))
        arr[:] = np.arange(8)
        arr.flush()
        L._IDS_FP_MEMO.clear()
        expected = L._ids_fp(arr)
        side = path + '.idsfp.json'
        for bad in ([1, 2, 3], 'not-a-record', 7):
            with open(side, 'w', encoding='utf-8') as f:
                json.dump(bad, f)
            L._IDS_FP_MEMO.clear()
            assert L._ids_fp(arr) == expected, bad
        del arr
        # Reopening the same file without a valid sidecar still yields the
        # same content identity.
        L._IDS_FP_MEMO.clear()
        arr2 = np.memmap(path, dtype=np.int64, mode='r', shape=(8,))
        assert L._ids_fp(arr2) == expected
        del arr2


def test_ids_fp_memo_is_value_bound():
    # The memo key is an array identity, not the bytes.  A writable array
    # mutated in place must produce a NEW digest; otherwise a cleaning pass
    # would leave a stale fingerprint behind and resume/admission could bind
    # a different corpus under the old identity.
    arr = np.array([3, 1, 4, 1, 5], dtype=np.int64)
    before = L._ids_fp(arr)
    assert L._ids_fp(arr) == before
    arr[2] = 9
    after = L._ids_fp(arr)
    assert after != before, (before, after)
    # A read-only copy is the same pure function of the values (and may take
    # the memo/sidecar fast path); writable vs read-only is not an identity.
    ro = arr.copy()
    ro.setflags(write=False)
    assert L._ids_fp(ro) == after
    assert L._ids_fp(ro) == after
    # Storage width is transport, not content.
    assert L._ids_fp(arr.astype(np.int32)) == after
    assert L._ids_fp(arr.astype(np.uint16)) == after


def test_batch_transport_matches_plain_reference():
    ids = (np.arange(70000, dtype=np.int64) * 7 % 997)
    for dt in (np.int64, np.int32, np.uint16, np.uint8):
        rep = L.batch_contract_check(ids.astype(dt), 64, 8,
                                     torch.device('cpu'), seed=11,
                                     vocab=997, n_batches=3)
        assert rep['passed'] is True, (dt, rep)
        assert rep['checks']['transport_matches_int64_reference'] is True, (dt, rep)
        assert all(b.get('transport_matches_int64_reference') is True
                   for b in rep['batches']), (dt, rep['batches'])
        assert rep['transport']['index_dtype'] == 'int32', rep['transport']
    # A loader that silently changes one token must fail the transport check,
    # not merely the shift/vocabulary checks.
    _orig = L.batch_iter
    try:
        def _bad(train_ids, seq_len, batch_size, device, seed=0):
            for x, y in _orig(train_ids, seq_len, batch_size, device, seed=seed):
                x = x.clone()
                x[0, 0] = (int(x[0, 0]) + 1) % 997
                yield x, y
        L.batch_iter = _bad
        rep = L.batch_contract_check(ids, 64, 8, torch.device('cpu'),
                                     seed=11, vocab=997, n_batches=2)
        assert rep['passed'] is False, rep
        assert rep['checks']['transport_matches_int64_reference'] is False, rep
    finally:
        L.batch_iter = _orig


def test_aggregate_quarantines_bad_records_without_crashing():
    # One malformed/unmeasured/duplicate record must not make a completed
    # panel unreadable.  Valid readings survive; every exclusion is named and
    # counted, and different data/evaluation slices are SPLIT rather than
    # averaged.
    summary = {
        'csa::seed0': {'variant': 'csa', 'seed': 0, 'ppl': 10.0,
                       'run_cfg': 'cfg', 'data_fp': 'A'},
        'csa::seed1': {'variant': 'csa', 'seed': 1, 'ppl': 11.0,
                       'run_cfg': 'cfg', 'data_fp': 'A'},
        'csa::seed1_dup': {'variant': 'csa', 'seed': 1, 'ppl': 99.0,
                           'run_cfg': 'cfg', 'data_fp': 'A'},
        'csa::seed2': {'variant': 'csa', 'seed': 2, 'ppl': 12.0,
                       'run_cfg': 'cfg', 'data_fp': 'B'},
        'csa::error': {'variant': 'csa', 'seed': 3,
                       'error': 'training failed'},
        'csa::synth': {'variant': 'csa', 'seed': 4, 'ppl': 8.0,
                       'synthesized': True},
        'csa::badseed': {'variant': 'csa', 'seed': 1.5, 'ppl': 9.0},
    }
    agg, rep, admitted = L.aggregate_with_exclusions(summary)
    assert rep['n_input'] == len(summary), rep
    assert rep['n_excluded'] == 4, rep
    assert rep['n_admitted'] == 3, rep
    assert any('duplicate' in e['reason'] for e in rep['excluded']), rep
    assert any('non-finite' in e['reason'] or 'unmeasured' in e['reason']
               for e in rep['excluded']), rep
    assert any('non-integer seed' in e['reason'] for e in rep['excluded']), rep
    assert rep['fingerprint_split_groups'], rep
    # A != B are separate slices: two output keys, each with n=2 / n=1, never
    # one averaged n=3 group.
    assert len(agg) == 2, agg
    assert sorted(e['n_seeds'] for e in agg.values()) == [1, 2], agg
    assert set(admitted) == {'csa::seed0', 'csa::seed1',
                             'csa::seed2'}, admitted


def test_scale_geometry_overrides_are_validated():
    base = dict(L.RUN_S1)
    base_draws = base['steps'] * base['batch_size'] * base['seq_len']
    out = L.apply_scale_geometry_overrides(
        base, {'B_SEQ_LEN': '4096', 'B_BATCH_SIZE': '4'})
    assert out['seq_len'] == 4096 and out['batch_size'] == 4, out
    assert base['seq_len'] == L.RUN_S1['seq_len'], 'override must not mutate the module config'
    assert out['steps'] * out['batch_size'] * out['seq_len'] >= base_draws, out
    assert out['geometry_overrides']['steps_source'] == 'preserve_sampled_windows', out
    # An explicit step count always wins; a token-budget change derives steps
    # from the budget (ceiling), never silently keeps the old count.
    fixed = L.apply_scale_geometry_overrides(
        base, {'B_SEQ_LEN': '2048', 'B_STEPS': '17'})
    assert fixed['steps'] == 17 and fixed['geometry_overrides']['steps_source'] == 'explicit_env', fixed
    budget = L.apply_scale_geometry_overrides(
        base, {'B_SEQ_LEN': '256', 'B_BATCH_SIZE': '1',
               'B_N_TRAIN_TOKENS': '1000000'})
    assert budget['steps'] == (1000000 + 255) // 256, budget
    assert budget['geometry_overrides']['steps_source'] == 'from_declared_token_budget', budget
    for bad in ({'B_SEQ_LEN': '0'}, {'B_BATCH_SIZE': '-1'},
                {'B_STEPS': '1.5'}, {'B_VAL_SEQS': 'x'},
                {'B_SEQ_LEN': True}):
        try:
            L.apply_scale_geometry_overrides(base, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'scale geometry override accepted {bad!r}')
    # A budget override without the geometry needed to derive steps must fail
    # closed instead of silently keeping the old step count.
    try:
        L.apply_scale_geometry_overrides(
            {'n_train_tokens': 1000, 'steps': 10},
            {'B_N_TRAIN_TOKENS': '2000'})
        raise AssertionError('unrecomputable step override was silently accepted')
    except ValueError:
        pass
    # Future tiers are discovered from RUN_S<n>, not hard-coded S1/S2.
    import explib.m0 as M0
    old = os.environ.get('M0_PANEL')
    try:
        os.environ['M0_PANEL'] = 'S1,S2'
        assert M0._panel_chain_from_env() == ['S1', 'S2']
        os.environ['M0_PANEL'] = 'S9'
        try:
            M0._panel_chain_from_env()
            raise AssertionError('an undeclared scale panel must be refused')
        except SystemExit:
            pass
    finally:
        if old is None:
            os.environ.pop('M0_PANEL', None)
        else:
            os.environ['M0_PANEL'] = old


def test_dump_aggregate_writes_quarantine_report():
    # Integration: a completed panel with one bad record still writes
    # aggregate.json plus an explicit quarantine report, and the reader is
    # handed only the admitted records.
    summary = {
        'csa::seed0': {'variant': 'csa', 'seed': 0, 'ppl': 10.0,
                       'run_cfg': 'cfg', 'data_fp': 'A'},
        'csa::error': {'variant': 'csa', 'seed': 1, 'error': 'boom'},
        'csa::synth': {'variant': 'csa', 'seed': 2, 'ppl': 9.0,
                       'synthesized': True},
    }
    with tempfile.TemporaryDirectory() as td:
        _saved = (L._plot, L._print_table, L._print_pairs, L.methodology_audit)
        try:
            L._plot = lambda *a, **k: None
            L._print_table = lambda *a, **k: None
            L._print_pairs = lambda *a, **k: None
            L.methodology_audit = lambda *a, **k: {}
            L._dump_aggregate(td, summary)
        finally:
            (L._plot, L._print_table, L._print_pairs,
             L.methodology_audit) = _saved
        rep = json.load(open(os.path.join(td, 'aggregation_exclusions.json'),
                             encoding='utf-8'))
        assert rep['n_input'] == 3 and rep['n_admitted'] == 1, rep
        assert rep['n_excluded'] == 2, rep
        assert rep['reason_counts'], rep
        agg = json.load(open(os.path.join(td, 'aggregate.json'),
                             encoding='utf-8'))
        assert list(agg) == ['csa'], agg
        assert os.path.exists(os.path.join(td, 'summary.aggregatable.json'))


def test_failed_panel_reconciles_running_state():
    with tempfile.TemporaryDirectory() as td:
        cfg = {'seeds': [0], 'variants': ['full'], 'outdir': td,
               'n_train_tokens': 64, 'seq_len': 8}
        tr = np.arange(64, dtype=np.int64) % 97
        va = np.arange(16, dtype=np.int64).reshape(2, 8) % 97
        prov = L.experiment_provenance(cfg, train_ids=tr, val_batch=va,
                                       vocab=97)
        L.write_run_manifest(td, cfg, 'FP', prov, {}, status='running',
                             planned_cells=['full::seed0'],
                             started_at='2026-01-01T00:00:00')
        L._record_failed_panel(td, cfg, 'FP', prov, stage='preflight',
                               error='synthetic failure',
                               planned_cells=['full::seed0'])
        m = json.load(open(os.path.join(td, 'run_manifest.json'),
                           encoding='utf-8'))['panels']['FP']
        assert m['status'] == 'failed', m['status']
        assert m['completion']['status'] == 'failed', m['completion']
        assert m['planned_cells'] == ['full::seed0'], m['planned_cells']
        # A later failed pre-flight must not rewrite a COMPLETE panel.
        m['status'] = 'complete'
        m['completion']['status'] = 'complete'
        m['sentinel'] = 'keep-me'
        with open(os.path.join(td, 'run_manifest.json'), 'w',
                  encoding='utf-8') as f:
            json.dump({'manifest_version': 1, 'panels': {'FP': m}}, f)
        L._record_failed_panel(td, cfg, 'FP', prov, stage='preflight2',
                               error='another failure',
                               planned_cells=['full::seed0'])
        m2 = json.load(open(os.path.join(td, 'run_manifest.json'),
                            encoding='utf-8'))['panels']['FP']
        assert m2['status'] == 'complete' and m2['sentinel'] == 'keep-me', m2


def test_seed_coverage_warning():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    s = {}
    for i in range(3):
        s[f'a::seed{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=49.0, params=100, train_knobs=knobs)
        if i < 2:
            s[f'b::seed{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                    ppl=50.0, params=100, train_knobs=knobs)
    r = L.methodology_audit(s, design={'c': [('a', 'b', 'x')]}, baselines={}, verbose=False)
    cv = r['controlled_variables']['c']
    assert cv['seed_coverage_equal'] is False, cv
    assert any('seed/protocol coverage' in w for w in r['warnings']), r['warnings']
    assert r['components']['c'][0]['n_paired'] == 2, r['components']['c'][0]


def test_fairness_cross_arm_controls():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    obj = {'comp_reg_applies': True, 'comp_lambda_effective': 0.05,
           'delta_lr_effective': 10.0, 'boundary_policy': ['threshold'],
           'loss_terms': ['ce', 'comp_reg']}

    def _panel(*, knob_override=None, drop_seed=None, obj_override_arm=None):
        s = {}
        for i in range(3):
            for arm, ppl in (('a', 10.0), ('b', 11.0)):
                if arm == 'b' and i == drop_seed:
                    continue
                kk = dict(knobs)
                if knob_override == (arm, i):
                    kk['grad_clip_norm'] = 2.0
                oo = dict(obj)
                if obj_override_arm == arm:
                    oo['boundary_policy'] = ['reinforce']
                s[f'{arm}::seed{i}'] = dict(
                    variant=arm, seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                    ppl=ppl, params=100, train_knobs=kk, train_objective=oo)
        return s

    with tempfile.TemporaryDirectory() as td:
        def _run(summary):
            path = os.path.join(td, 'summary.json')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(summary, f)
            with contextlib.redirect_stdout(io.StringIO()):
                return A.audit_fairness(path, arms=['a', 'b'])

        ok = _run(_panel())
        assert ok['controlled_variables']['train_knobs_equal'] is True, ok['controlled_variables']
        assert ok['controlled_variables']['objective_equal'] is True, ok['controlled_variables']
        assert not any('train_knobs differ' in p for p in ok['problems']), ok['problems']
        # A different objective in one arm is a system-level confound; surface
        # it as a warning instead of silently calling the pair controlled.
        od = _run(_panel(obj_override_arm='b'))
        assert od['controlled_variables']['objective_equal'] is False, od['controlled_variables']
        assert any('training objectives differ' in w for w in od['warnings']), od['warnings']
        assert od['status'] == 'warn', od['status']
        # A seed-level objective change INSIDE one arm is a mixed protocol,
        # not merely a cross-arm disclosure: the arm mean pools two different
        # objectives and must not be called controlled.
        mixed_obj = {k: dict(v) for k, v in _panel().items()}
        mixed_obj['a::seed1']['train_objective'] = dict(obj, boundary_policy=['reinforce'])
        _mr, _ = L._variant_seed_map(mixed_obj)
        cv = L.controlled_variables_report(_mr, ['a', 'b'])
        assert cv['objective_equal'] is False, cv
        assert 'a' in cv['within_arm_mixed'], cv
        assert cv['controlled'] is False, cv
        assert 'train_objective' in cv['differences'], cv
        md = _run(mixed_obj)
        assert md['status'] == 'fail', md
        assert any('mix DIFFERENT' in p for p in md['problems']), md['problems']
        # A partially recorded objective descriptor is unverifiable, not a
        # pass; the arm must be named as partially recorded.
        part_obj = {k: dict(v) for k, v in _panel().items()}
        part_obj['b::seed1'].pop('train_objective')
        _pr, _ = L._variant_seed_map(part_obj)
        cv2 = L.controlled_variables_report(_pr, ['a', 'b'])
        assert cv2['objective_equal'] is False, cv2
        assert 'b' in cv2['partially_recorded'], cv2
        assert cv2['controlled'] is False, cv2
        # A different optimizer/clip in ONE arm must fail even when every arm
        # is internally consistent.
        bad = _run(_panel(knob_override=('b', 1)))
        assert bad['status'] == 'fail', bad
        assert any('train_knobs differ across arms' in p for p in bad['problems']), bad['problems']
        # Missing seed in one arm: arm means would be over different seeds.
        cov = _run(_panel(drop_seed=2))
        assert cov['status'] == 'fail', cov
        assert any('seed/protocol coverage differs' in p for p in cov['problems']), cov['problems']
        # Same recipe but a larger arm is a capacity confound, even though the
        # per-seed run_cfg/train_knobs gate is satisfied.
        cap = _panel()
        for _k in list(cap):
            if cap[_k]['variant'] == 'b':
                cap[_k]['params'] = 150
        rcap = _run(cap)
        assert rcap['status'] == 'fail', rcap
        assert any('parameter counts differ' in p for p in rcap['problems']), rcap['problems']
        miss = _panel()
        miss['b::seed0'].pop('params')
        rmiss = _run(miss)
        assert rmiss['status'] == 'warn', rmiss
        assert any('capacity matching cannot be verified' in w for w in rmiss['warnings']), rmiss


def test_tuned_pairing_keeps_optimizer_gate():
    knobs = L.training_knobs()
    a = dict(variant='a', steps=1, tokens_seen=1, data_fp='d', val_fp='v', params=100,
             train_knobs=knobs)
    b = dict(a, variant='b')
    assert BT._relaxed_pair_reason(a, b, {'a': 3, 'b': 3}) is None
    b2 = dict(b, train_knobs=dict(knobs, grad_clip_norm=2.0))
    why = BT._relaxed_pair_reason(a, b2, {'a': 3, 'b': 3})
    assert why and 'train_knobs' in why, why
    b3 = dict(b)
    b3.pop('train_knobs')
    why2 = BT._relaxed_pair_reason(a, b3, {'a': 3, 'b': 3})
    assert why2 and 'unverifiable train_knobs' in why2, why2
    # A paired seed does not imply a matched shared-trunk initialization.
    a4 = dict(a, init_core_fp='aaaa')
    b4 = dict(b, init_core_fp='bbbb')
    why3 = BT._relaxed_pair_reason(a4, b4, {'a': 3, 'b': 3})
    assert why3 and 'init_core_fp' in why3, why3
    b5 = dict(b)
    why4 = BT._relaxed_pair_reason(a4, b5, {'a': 3, 'b': 3})
    assert why4 and 'unverifiable init_core_fp' in why4, why4
    # Capacity must fail CLOSED: a missing params field used to slip through
    # the old int(None) try/except as if the pair were matched.
    b6 = dict(b)
    b6.pop('params')
    why5 = BT._relaxed_pair_reason(a, b6, {'a': 3, 'b': 3})
    assert why5 and 'unverifiable params' in why5, why5
    a7 = dict(a, param_breakdown={'total': 100, 'non_embedding': 80, 'active': 70})
    b7 = dict(b, param_breakdown={'total': 100, 'non_embedding': 80, 'active': 50})
    why6 = BT._relaxed_pair_reason(a7, b7, {'a': 3, 'b': 3})
    assert why6 and 'parameter breakdown' in why6, why6
    # A tuned contrast must score the same evaluation slice.
    a8 = dict(a, eval_fp='e1')
    b8 = dict(b, eval_fp='e2')
    why7 = BT._relaxed_pair_reason(a8, b8, {'a': 3, 'b': 3})
    assert why7 and 'eval_fp' in why7, why7
    assert BT._relaxed_pair_reason(dict(a, eval_fp='e1'), dict(b, eval_fp='e1'),
                                   {'a': 3, 'b': 3}) is None
    # The final PPL subset is part of the evaluation slice too.
    assert BT._relaxed_pair_reason(dict(a, eval_subset_fp='s1'),
                                   dict(b, eval_subset_fp='s2'),
                                   {'a': 3, 'b': 3}) is not None
    assert BT._relaxed_pair_reason(dict(a, eval_subset_fp='s1'),
                                   dict(b, eval_subset_fp='s1'),
                                   {'a': 3, 'b': 3}) is None
    # A held-out row fold must be the same fold on both arms.
    why8 = BT._relaxed_pair_reason(dict(a, holdout_row_fp='h1'),
                                   dict(b, holdout_row_fp='h2'), {'a': 3, 'b': 3})
    assert why8 and 'holdout_row_fp' in why8, why8
    # Independent tuning may change lr/wd/warmup, but NOT the rest of the
    # structured recipe.  A different seq_len/eval_subset/vocab or a one-sided
    # descriptor is unverifiable, never a matched tuned pair.
    hp = {'steps': 1, 'tokens_seen': 1, 'lr': 3e-4, 'weight_decay': 0.1,
          'warmup': 1, 'batch_size': 2, 'seq_len': 8, 'd': 8,
          'n_layers': 1, 'n_heads': 1, 'd_head': 8, 'mlp_ratio': 4.0,
          'eval_every': 1, 'eval_subset': 4, 'vocab_size': 16}
    a9 = dict(a, hyperparams=dict(hp))
    b9 = dict(b, hyperparams=dict(hp, lr=1e-3, weight_decay=0.05, warmup=2))
    assert BT._relaxed_pair_reason(a9, b9, {'a': 3, 'b': 3}) is None
    for _k, _v in (('seq_len', 16), ('batch_size', 4), ('eval_subset', 8),
                   ('vocab_size', 32), ('mlp_ratio', 8.0)):
        _bad = dict(b, hyperparams={**hp, _k: _v})
        _why = BT._relaxed_pair_reason(a9, _bad, {'a': 3, 'b': 3})
        assert _why and ('hyperparams.' + _k) in _why, (_k, _why)
    _why = BT._relaxed_pair_reason(a9, b, {'a': 3, 'b': 3})
    assert _why and 'unverifiable hyperparams' in _why, _why
    # A tuned contrast may not call a complete-case / seed-driven / n<3 panel
    # significant just because its Holm p is small.
    base_st = {'n': 3, 'p_exact_signflip': 0.02, 'p_holm': 0.03,
               'mean': 1.0, 'std': 0.2, 'underpowered': False,
               'within_seed_noise': False, 'seed_driven': False,
               'complete_pairs': True, 'n_dropped': 0}
    assert BT._tuned_significant(base_st, 0.03) is True
    for _bad in (dict(base_st, n=2, underpowered=True),
                 dict(base_st, within_seed_noise=True),
                 dict(base_st, seed_driven=True),
                 dict(base_st, complete_pairs=False),
                 dict(base_st, n_dropped=1),
                 dict(base_st, n_skipped=1),
                 dict(base_st, n_skipped_config_mismatch=1),
                 dict(base_st, n_skipped=-1),
                 dict(base_st, n_skipped=True),
                 dict(base_st, n_dropped=-1),
                 dict(base_st, n_input=-1),
                 dict(base_st, n_input=2),
                 dict(base_st, n_input=3, n_dropped=4),
                 dict(base_st, n_nonzero=4),
                 dict(base_st, n_nonzero=-1),
                 dict(base_st, n_nonzero=1)):
        assert BT._tuned_significant(_bad, 0.03) is False, _bad
    assert BT._tuned_significant(dict(base_st, n_skipped=0), 0.03) is True
    assert BT._tuned_significant(dict(base_st, n_input=3, n_dropped=0), 0.03) is True
    assert BT._tuned_significant(dict(base_st, n=6, n_nonzero=6), 0.03) is True
    assert BT._tuned_significant(dict(base_st, n=6, n_nonzero=3), 0.03) is False
    assert BT._tuned_significant(base_st, None) is False
    assert BT._tuned_significant(base_st, 0.5) is False
    # alpha is a gate parameter, not a free multiplier.
    for _bad_alpha in (True, 0, -0.1, 1.5, float('nan'), 'x'):
        assert BT._tuned_significant(base_st, 0.03, _bad_alpha) is False, _bad_alpha


def test_arm_mean_std_and_relative_effect():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    a_vals = [10.0, 10.2, 9.8, 10.1, 9.9]
    s = {}
    for i, ppl in enumerate(a_vals):
        s[f'a::seed{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=ppl, params=100, train_knobs=knobs)
        s[f'b::seed{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=10.0, params=100, train_knobs=knobs)
    r = L.methodology_audit(s, design={'c': [('a', 'b', 'x')]}, baselines={}, verbose=False)
    arm = r['arm_summary']['a']
    assert arm['n_seeds'] == 5 and arm['meets_min_seeds'] is True, arm
    assert abs(arm['ppl_mean'] - float(np.mean(a_vals))) < 1e-12
    assert abs(arm['ppl_std'] - float(np.std(a_vals, ddof=1))) < 1e-12
    assert r['arms_below_min_seeds'] == [], r['arms_below_min_seeds']
    e = r['components']['c'][0]
    assert e['n_paired'] == 5 and e['without_mean'] == 10.0, e
    assert abs(e['relative_effect_pct'] - 100.0 * e['mean'] / e['without_mean']) < 1e-9, e
    # Nested tuning records: the contrast's arm means, relative effect and
    # bootstrap sensitivity must use the SAME holdout estimand as the paired
    # delta, not the full-validation `ppl` that selected the hyper-parameters.
    hold = [10.0, 10.2, 9.8, 10.1, 9.9]
    nested = {}
    for i, ppl in enumerate(hold):
        nested[f'a::seed{i}'] = dict(s[f'a::seed{i}'], ppl=99.0 + i,
                                     ppl_estimation_rows='nested_validation_holdout',
                                     ppl_holdout=ppl)
        nested[f'b::seed{i}'] = dict(s[f'b::seed{i}'], ppl=1.0 + i,
                                     ppl_estimation_rows='nested_validation_holdout',
                                     ppl_holdout=10.0)
    rn = L.methodology_audit(nested, design={'c': [('a', 'b', 'x')]},
                             baselines={}, verbose=False)
    en = rn['components']['c'][0]
    assert en['without_mean'] == 10.0, en
    assert abs(en['with_mean'] - float(np.mean(hold))) < 1e-12, en
    assert abs(en['relative_effect_pct'] - 100.0 * en['mean'] / en['without_mean']) < 1e-9, en
    arm_n = rn['arm_summary']['a']
    assert arm_n['ppl_estimand'] == 'nested_validation_holdout', arm_n
    assert abs(arm_n['ppl_mean'] - float(np.mean(hold))) < 1e-12, arm_n
    assert arm_n['ppl_primary_missing'] == 0, arm_n


def test_v7_v8_v9_pairs_use_primary_estimand():
    import explib.scale as V8
    import explib.resume as V9
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()

    def _rec(v, seed, full, hold):
        r = {'variant': v, 'seed': seed, 'run_cfg': fp, 'steps': 1,
             'tokens_seen': 1, 'ppl': full, 'params': 100,
             'train_knobs': knobs,
             'ppl_estimation_rows': 'nested_validation_holdout'}
        if hold is not None:
            r['ppl_holdout'] = hold
        return r

    panel = {}
    for seed in (0, 1, 2):
        panel.setdefault('a', {})[seed] = _rec('a', seed, 100.0 + seed, 10.0 + seed)
        panel.setdefault('b', {})[seed] = _rec('b', seed, 1.0 + seed,
                                              12.0 + seed if seed else None)
    for fn in (V9.add_paired, V8._add_paired):
        comp = {}
        fn(comp, 'x', panel, 'a', 'b')
        st = comp['x']
        # Seed 0's b record has no holdout accounting, so it is excluded
        # rather than paired on the full-validation PPL (which would give a
        # huge positive delta from 100 vs 1).
        assert st['n'] == 2 and abs(st['mean'] + 2.0) < 1e-12, st


def test_paired_effect_seed_robustness():
    stable = L.paired_effect_stats([-1.0, -1.1, -0.9, -1.0])
    assert stable['seed_driven'] is False and stable['seed_robust'] is True, stable
    assert stable['ci95_low'] < stable['mean'] < stable['ci95_high'], stable
    # One outlier seed makes the aggregate mean positive, but dropping it
    # flips the sign: the direction is owned by a single seed.
    fragile = L.paired_effect_stats([-2.0, -2.0, -2.0, -2.0, 8.5])
    assert fragile['mean'] > 0 and fragile['seed_driven'] is True, fragile
    assert fragile['seed_robust'] is False, fragile
    assert 'UNSTABLE' in fragile['effect_vs_noise'], fragile
    # The audit must surface it instead of silently certifying the mean.
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    a_vals = [51.5, 51.5, 51.5, 51.5, 42.0]
    s = {}
    for i, ppl in enumerate(a_vals):
        s[f'a::seed{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=ppl, params=100, train_knobs=knobs)
        s[f'b::seed{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=50.0, params=100, train_knobs=knobs)
    r = L.methodology_audit(s, design={'c': [('a', 'b', 'x')]}, baselines={}, verbose=False)
    e = r['components']['c'][0]
    assert e['seed_driven'] is True and e['seed_robust'] is False, e
    assert any('SEED-DRIVEN' in w for w in r['warnings']), r['warnings']


def test_capacity_confound_warning():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    s = {}
    for i in range(3):
        s[f'a::seed{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=49.0, params=1200, train_knobs=knobs)
        s[f'b::seed{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=50.0, params=1000, train_knobs=knobs)
    r = L.methodology_audit(s, design={'c': [('a', 'b', 'x')]}, baselines={}, verbose=False)
    e = r['components']['c'][0]
    assert e['param_controlled'] is False and e['capacity_confounded'] is True, e
    assert any('CAPACITY' in w or 'capacity' in w for w in r['warnings']), r['warnings']


def test_component_pair_requires_same_mlp_ratio():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()

    def _hp(mlp):
        return {'steps': 1, 'tokens_seen': 1, 'lr': 0.0003, 'weight_decay': 0.1,
                'warmup': 1, 'batch_size': 1, 'seq_len': 8, 'd': 8, 'n_layers': 1,
                'n_heads': 1, 'd_head': 8, 'mlp_ratio': mlp, 'comp_lambda': 0.05,
                'delta_lr_mult': 10.0, 'dataset': 'unit', 'data_version': 'v'}
    s = {}
    for i in range(3):
        s[f'a::seed{i}'] = dict(variant='a', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=49.0, params=100, train_knobs=knobs, hyperparams=_hp(4.0))
        s[f'b::seed{i}'] = dict(variant='b', seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                                ppl=50.0, params=100, train_knobs=knobs, hyperparams=_hp(8.0))
    # Same run_cfg/train_knobs but a different MLP width is a capacity change,
    # not the module under test: a module ablation must not pair it.
    r = L.methodology_audit(s, design={'c': [('a', 'b', 'x')]}, baselines={}, verbose=False)
    e = r['components']['c'][0]
    assert e['controlled'] is False and e['n_paired'] == 0, e
    assert any('hyperparams.mlp_ratio' in w for w in r['warnings']), r['warnings']
    # A baseline contrast may legitimately widen the MLP for parameter
    # matching; there the parameter-count gate (not mlp_ratio equality) rules.
    for i in range(3):
        s[f'a::seed{i}']['params'] = 100
        s[f'b::seed{i}']['params'] = 100
    base_design = {'compression_vs_matched_dense': [('a', 'b', 'x')]}
    rb = L.methodology_audit(s, design={}, baselines=base_design, verbose=False)
    eb = rb['baselines']['compression_vs_matched_dense'][0]
    assert eb['n_paired'] == 3 and eb['controlled'] is True, eb


def test_ablation_completeness_leave_one_out():
    design = {'A': [('full', 'full-A', '')],
              'B': [('full', 'full-B', '')],
              'C': [('full', 'full-C', '')]}
    r = L.ablation_completeness_audit(design, ['full', 'full-A', 'full-B'])
    assert r['complete'] is False and r['incomplete_components'] == ['C'], r
    assert r['missing_arms']['C'] == ['full-C'], r
    r2 = L.ablation_completeness_audit(design, ['full', 'full-A', 'full-B', 'full-C'])
    assert r2['complete'] is True and r2['incomplete_components'] == [], r2


def test_ablation_completeness_full_factorial():
    arms = {(): 'none', ('A',): 'A', ('B',): 'B', ('C',): 'C',
            ('A', 'B'): 'AB', ('A', 'C'): 'AC', ('B', 'C'): 'BC'}
    r = L.ablation_completeness_audit(arms_by_combo=arms)
    assert r['mode'] == 'full-factorial' and r['complete'] is False, r
    assert r['missing_combos'] == [['A', 'B', 'C']], r
    arms[('A', 'B', 'C')] = 'ABC'
    r2 = L.ablation_completeness_audit(arms_by_combo=arms)
    assert r2['complete'] is True and r2['n_mapped'] == 8, r2
    # A mapped but not-run arm is still incomplete.
    r3 = L.ablation_completeness_audit(arms_by_combo=arms, present=list(arms.values())[:-1])
    assert r3['complete'] is False and r3['missing_arms'], r3


def test_full_factorial_rejects_reused_arm():
    arms = {(): 'none', ('A',): 'A', ('B',): 'B', ('C',): 'C',
            ('A', 'B'): 'AB', ('A', 'C'): 'AC', ('B', 'C'): 'BC',
            ('A', 'B', 'C'): 'AB'}
    r = L.ablation_completeness_audit(arms_by_combo=arms)
    assert r['mode'] == 'full-factorial' and r['complete'] is False, r
    assert 'AB' in r['reused_arms'], r
    arms[('A', 'B', 'C')] = 'ABC'
    assert L.ablation_completeness_audit(arms_by_combo=arms)['complete'] is True


def test_ablation_coverage_across_panels():
    design = {'A': [('full', 'full-A', '')], 'B': [('full', 'full-B', '')]}

    def _rep(*, complete=(), incomplete=(), missing=None, measured=(), verified=(),
             unverified=(), conflicts=(), harmful=(), entries=None, min_n=3):
        return {'ablation_completeness': {
                    'components': {c: {} for c in set(complete) | set(incomplete)},
                    'complete_components': list(complete),
                    'incomplete_components': list(incomplete),
                    'missing_arms': dict(missing or {})},
                'measured_components': list(measured),
                'verified_components': list(verified),
                'unverified_components': list(unverified),
                'direction_conflict_components': list(conflicts),
                'practically_harmful_components': list(harmful),
                'components': dict(entries or {}), 'min_n': min_n}

    reports = {
        'p1': _rep(complete=('A',), measured=('A',), verified=('A',)),
        'p2': _rep(incomplete=('A',), measured=('A',), missing={'A': ['full-A']}),
        'p3': _rep(unverified=('B',)),
    }
    cov = L.ablation_coverage_across_panels(reports, design)
    assert cov['components']['A']['panels_complete'] == ['p1'], cov
    assert cov['components']['A']['panels_usable'] == ['p1'], cov
    assert cov['components']['A']['panels_verified'] == ['p1'], cov
    assert cov['components']['A']['missing_arms'] == {'p2': ['full-A']}, cov
    assert cov['complete_anywhere'] == ['A'], cov
    assert cov['usable_anywhere'] == ['A'], cov
    assert cov['verified_anywhere'] == ['A'], cov
    assert cov['missing_everywhere'] == ['B'], cov
    assert cov['status'] == 'warn' and any('B' in w for w in cov['warnings']), cov
    # A complete panel that never verified is reported separately, not hidden.
    cov2 = L.ablation_coverage_across_panels(
        {'p': _rep(complete=('A', 'B'), measured=('A', 'B'), unverified=())}, design)
    assert cov2['complete_but_unverified'] == ['A', 'B'], cov2
    assert cov2['status'] == 'ok', cov2
    # All arms present but no controlled contrast shares enough seeds: this is
    # NOT a usable ablation even though the presence-only check is complete.
    cov3 = L.ablation_coverage_across_panels(
        {'p': _rep(complete=('A',), measured=())}, design)
    assert cov3['components']['A']['complete_anywhere'] is True, cov3
    assert cov3['components']['A']['usable_anywhere'] is False, cov3
    assert cov3['usable_anywhere'] == [], cov3
    assert cov3['status'] == 'warn' and any('shared seeds' in w for w in cov3['warnings']), cov3


def test_ablation_coverage_replication_and_contradiction():
    design = {'A': [('full', 'full-A', '')], 'B': [('full', 'full-B', '')]}

    def _rep(*, complete=(), incomplete=(), missing=None, measured=(), verified=(),
             unverified=(), conflicts=(), harmful=(), entries=None, min_n=3):
        return {'ablation_completeness': {
                    'components': {c: {} for c in set(complete) | set(incomplete)},
                    'complete_components': list(complete),
                    'incomplete_components': list(incomplete),
                    'missing_arms': dict(missing or {})},
                'measured_components': list(measured),
                'verified_components': list(verified),
                'unverified_components': list(unverified),
                'direction_conflict_components': list(conflicts),
                'practically_harmful_components': list(harmful),
                'components': dict(entries or {}), 'min_n': min_n}

    _full = dict(complete=('A', 'B'), measured=('A', 'B'))
    p1 = _rep(verified=('A', 'B'), **_full)
    p2 = _rep(verified=('B',), unverified=('A',), conflicts=('A',), **_full)
    cov = L.ablation_coverage_across_panels({'p1': p1, 'p2': p2}, design)
    # One panel verifying A is not enough: the other panel's corrected
    # opposite direction is contradictory evidence, not a second opinion.
    assert cov['components']['A']['panels_verified'] == ['p1'], cov
    assert cov['components']['A']['panels_direction_conflict'] == ['p2'], cov
    assert cov['components']['A']['replication_status'] == 'contradicted', cov
    assert cov['components']['B']['panels_replicated'] == ['p1', 'p2'], cov
    assert cov['components']['B']['replication_status'] == 'replicated', cov
    assert cov['replicated_anywhere'] == ['B'], cov
    assert cov['single_panel_verified'] == ['A'], cov
    assert cov['direction_conflict_anywhere'] == ['A'], cov
    assert cov['all_components_useful_allowed'] is False, cov
    assert any('contradicted by another panel' in r
               for r in cov['all_components_useful_reasons']), cov

    # A practically harmful removal is a contradiction even when the panel
    # classifier did not also put the component in the direction-conflict list.
    p2h = _rep(verified=('B',), unverified=('A',), harmful=('A',), **_full)
    covh = L.ablation_coverage_across_panels({'p1': p1, 'p2': p2h}, design)
    assert covh['practically_harmful_anywhere'] == ['A'], covh
    assert covh['components']['A']['replication_status'] == 'contradicted', covh
    assert covh['all_components_useful_allowed'] is False, covh

    # A powered opposite MEAN without correction-significance is a warning,
    # not a contradiction: do not let noise alone flip the project gate.
    p2o = _rep(verified=('B',), unverified=('A',), **_full,
               entries={'A': [{'controlled': True, 'n': 3,
                               'direction_consistent': False,
                               'observed_sign': 1, 'predicted_sign': -1}]})
    covo = L.ablation_coverage_across_panels({'p1': p1, 'p2': p2o}, design)
    assert covo['components']['A']['panels_opposite_mean'] == ['p2'], covo
    assert covo['components']['A']['replication_status'] == 'single_panel', covo
    assert covo['opposite_mean_anywhere'] == ['A'], covo
    assert covo['direction_conflict_anywhere'] == [], covo
    assert covo['all_components_useful_allowed'] is True, covo
    assert covo['status'] == 'warn', covo


def test_replication_independence_is_not_shared_draws():
    # A draw identity must bind seeds, evaluation slices and recipe/source
    # fields, while ignoring malformed records instead of crashing.
    ident = L.component_draw_identity(None, ['full'])
    assert ident['n_records'] == 0 and ident['seeds'] == [], ident

    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    recs = {}
    for i in range(3):
        for v, ppl in (('full', 50.0), ('full-A', 51.0)):
            recs[f'{v}::seed{i}'] = dict(
                variant=v, seed=i, run_cfg=fp, steps=1, tokens_seen=1,
                ppl=ppl, params=100, train_knobs=knobs, _code='cs1',
                val_fp='val1', eval_fp='eval1', data_fp='data1')
    rep = L.methodology_audit(
        recs, design={'A': [('full', 'full-A', 'ablate')]}, baselines={},
        verbose=False)
    draw = rep['component_draws']['A']
    assert draw['seeds'] == [0, 1, 2], draw
    assert draw['eval_fps'] == ['eval1'] and draw['val_fps'] == ['val1'], draw
    assert draw['code_values'] == ['cs1'], draw

    p1 = {'seeds': [0, 1, 2], 'eval_fps': ['eval1'], 'val_fps': ['val1'],
          'data_fps': ['data1'], 'code_values': ['cs1'],
          'train_knobs_values': ['knobs1']}
    p2 = {'seeds': [0, 1, 2], 'eval_fps': ['eval2'], 'val_fps': ['val2'],
          'data_fps': ['data1'], 'code_values': ['cs1'],
          'train_knobs_values': ['knobs1']}
    shared = L.replication_independence_audit({'p1': p1, 'p2': p2})
    assert shared['status'] == 'shared_training_draws', shared
    assert shared['panels_shared_seeds'] == ['p1', 'p2'], shared

    # Disjoint training seeds but a shared evaluation slice are NEW training
    # draws, not independent evaluation.
    p3 = dict(p2, seeds=[3, 4, 5], eval_fps=['eval1'], val_fps=['val1'])
    new_seeds = L.replication_independence_audit({'p1': p1, 'p3': p3})
    assert new_seeds['status'] == 'new_seeds_only', new_seeds
    assert new_seeds['panels_new_seeds'] == ['p1', 'p3'], new_seeds

    # Disjoint seeds and disjoint evaluation fingerprints are the only tier
    # that may be called independent replication.
    p4 = dict(p2, seeds=[3, 4, 5], eval_fps=['eval9'], val_fps=['val9'])
    independent = L.replication_independence_audit({'p1': p1, 'p4': p4})
    assert independent['status'] == 'independent_replication', independent
    assert independent['panels_independent'] == ['p1', 'p4'], independent

    # Missing seed identity is never treated as disjoint.
    unattested = L.replication_independence_audit(
        {'p1': p1, 'p5': {'seeds': [], 'eval_fps': ['eval9']}})
    assert unattested['status'] == 'unattested', unattested
    assert unattested['n_pairs'] == 1 and unattested['unattested_pairs'], unattested

    # A scalar identity field is one value, not an iterable of characters;
    # otherwise "eval1"/"eval2" would falsely share the character "e".
    scalar = L.replication_independence_audit({
        'p1': {'seeds': 0, 'eval_fps': 'eval1', 'val_fps': 'val1'},
        'p2': {'seeds': 1, 'eval_fps': 'eval2', 'val_fps': 'val2'}})
    assert scalar['status'] == 'independent_replication', scalar
    assert scalar['pairs']['p1|p2']['seed_overlap'] == [], scalar

    # A validation-only match is not enough to attest an independent outcome
    # slice; without eval/holdout fingerprints this stays new-seeds/unverified.
    val_only = L.replication_independence_audit({
        'p1': {'seeds': [0], 'val_fps': ['val1']},
        'p2': {'seeds': [1], 'val_fps': ['val2']}})
    assert val_only['status'] == 'new_seeds_only', val_only
    assert (val_only['pairs']['p1|p2']['tier']
            == 'new_seeds_unverified_evaluation'), val_only

    # A partially malformed seed list must not silently drop the bad entry and
    # become "disjoint"; the pair is unattested, not independent.
    bad_seed = L.replication_independence_audit({
        'p1': {'seeds': [0, 'not-a-seed'], 'eval_fps': ['eval1']},
        'p2': {'seeds': [1], 'eval_fps': ['eval2']}})
    assert bad_seed['status'] == 'unattested', bad_seed

    # Coverage must not call same-seed panels a replication, and must label
    # disjoint-seed/same-eval panels as a weaker new-seed repetition.
    def _rep(*, complete=(), measured=(), verified=(), draws=None):
        return {'ablation_completeness': {
                    'components': {c: {} for c in complete},
                    'complete_components': list(complete),
                    'incomplete_components': [],
                    'missing_arms': {}},
                'measured_components': list(measured),
                'verified_components': list(verified),
                'unverified_components': [],
                'direction_conflict_components': [],
                'practically_harmful_components': [],
                'components': {}, 'min_n': 3,
                'component_draws': dict(draws or {})}

    design = {'A': [('full', 'full-A', '')]}
    same = L.ablation_coverage_across_panels({
        'p1': _rep(complete=('A',), measured=('A',), verified=('A',),
                   draws={'A': p1}),
        'p2': _rep(complete=('A',), measured=('A',), verified=('A',),
                   draws={'A': p2})}, design)
    assert same['components']['A']['replication_status'] == 'replicated_shared_seeds', same
    assert same['independently_replicated_anywhere'] == [], same
    assert same['shared_seed_replicated_anywhere'] == ['A'], same
    assert any('share training seeds' in w for w in same['warnings']), same

    new_cov = L.ablation_coverage_across_panels({
        'p1': _rep(complete=('A',), measured=('A',), verified=('A',),
                   draws={'A': p1}),
        'p3': _rep(complete=('A',), measured=('A',), verified=('A',),
                   draws={'A': p3})}, design)
    assert new_cov['components']['A']['replication_status'] == 'replicated_new_seeds', new_cov
    assert new_cov['new_seed_replicated_anywhere'] == ['A'], new_cov

    ind_cov = L.ablation_coverage_across_panels({
        'p1': _rep(complete=('A',), measured=('A',), verified=('A',),
                   draws={'A': p1}),
        'p4': _rep(complete=('A',), measured=('A',), verified=('A',),
                   draws={'A': p4})}, design)
    assert ind_cov['components']['A']['replication_status'] == 'replicated', ind_cov
    assert ind_cov['independently_replicated_anywhere'] == ['A'], ind_cov

    # Legacy reports without draw metadata keep the old label but never get
    # promoted into the independent list.
    legacy = L.ablation_coverage_across_panels({
        'p1': _rep(complete=('A',), measured=('A',), verified=('A',)),
        'p2': _rep(complete=('A',), measured=('A',), verified=('A',))}, design)
    assert legacy['components']['A']['replication_status'] == 'replicated', legacy
    assert legacy['independently_replicated_anywhere'] == [], legacy


def test_panel_ablation_flag_does_not_overclaim():
    fp = 'steps1_sl8_bs1_nt10_lr0.0003_wd0.1_d8_L1_H1_Dh8_v16_wu1_cl0.05_dlm10.0_mtdefault_mrcsa_dynamic_ee1_es1_detcpu-eager_dfabc_csX'
    knobs = L.training_knobs()
    s = {}
    for i in range(3):
        for v, ppl in (('full', 50.0), ('full-A', 51.0)):
            s[f'{v}::seed{i}'] = dict(variant=v, seed=i, run_cfg=fp, steps=1,
                                      tokens_seen=1, ppl=ppl, params=100,
                                      train_knobs=knobs)
    design = {'A': [('full', 'full-A', '')], 'B': [('full', 'full-B', '')]}
    r = L.methodology_audit(s, design=design, baselines={}, verbose=False)
    # A itself is complete, but the panel as a whole omitted B entirely; the
    # panel-level flag must not call that a complete ablation.
    assert r['ablation_completeness']['complete'] is True, r['ablation_completeness']
    assert r['ablation_complete'] is False, r['ablation_complete']
    assert r['ablation_completeness']['omitted_components'] == ['B'], r['ablation_completeness']


def test_cosine_similarity_scalar_features_keep_sequence_axis():
    # A 1-D sequence means one scalar feature per position, so the result is
    # [T-1] (the same contract as [T, 1]), not a reduced scalar.  T<=1 must
    # yield a zero-width vector rather than 0.0.
    for T in (0, 1, 2, 5):
        x = torch.arange(T, dtype=torch.float32)
        got = L.cosine_similarity_consecutive(x)
        assert tuple(got.shape) == (max(T - 1, 0),), (T, got.shape, got)
    x = torch.tensor([1.0, 1.0, -1.0])
    assert L.cosine_similarity_consecutive(x).tolist() == [1.0, -1.0]


def test_adaptive_threshold_guard():
    # Row chunking keeps a large batch feasible; an explicit safety ceiling is
    # still honoured.
    L.causal_adaptive_threshold(torch.randn(64, 4096), 16)
    # Empty inputs keep their shape (batch x length), not a flat [0] tensor.
    for shape in ((0,), (0, 5), (3, 0), (0, 0)):
        got = L.causal_adaptive_threshold(torch.empty(shape), 16)
        assert tuple(got.shape) == shape, (shape, got.shape)
    _old = os.environ.get('CSA_ADAPTIVE_MAX_TOKENS')
    try:
        os.environ['CSA_ADAPTIVE_MAX_TOKENS'] = '1024'
        try:
            L.causal_adaptive_threshold(torch.randn(16000), 16)
            raise AssertionError('a row above the explicit safety ceiling must be refused')
        except RuntimeError:
            pass
    finally:
        if _old is None:
            os.environ.pop('CSA_ADAPTIVE_MAX_TOKENS', None)
        else:
            os.environ['CSA_ADAPTIVE_MAX_TOKENS'] = _old


def test_adaptive_threshold_values():
    # The production path row/column-chunks the sort.  That partition must not
    # change the closed-form per-row causal quantile: compare against the
    # unchunked definition for both a single row and a batch.
    def reference(sim, target):
        q = max(1.0 / max(int(target), 1), 0.001)
        squeeze = sim.dim() == 1
        x = (sim.unsqueeze(0) if squeeze else sim).detach().float()
        n = x.shape[-1]
        idx = torch.arange(n)
        pos = q * idx
        lo = pos.floor().long()
        frac = pos - lo
        lo_c = lo.clamp(max=n - 1)[..., None]
        hi_c = (lo + 1).clamp(max=n - 1)[..., None]
        use_hi = (frac > 0.0) & (lo + 1 <= idx)
        first = idx == 0
        masked = torch.where(idx[None, None, :] <= idx[None, :, None],
                             x[:, None, :], float('inf')).sort(dim=-1).values
        lo_v = masked.gather(-1, lo_c.expand(x.shape[0], n, 1)).squeeze(-1)
        hi_v = masked.gather(-1, hi_c.expand(x.shape[0], n, 1)).squeeze(-1)
        interp = lo_v * (1.0 - frac) + hi_v * frac
        oc = torch.where(use_hi, interp, lo_v)
        out = torch.where(first, torch.full_like(oc, float('-inf')), oc)
        return out.squeeze(0) if squeeze else out

    for shape in ((7,), (3, 11), (2, 300)):
        for target in (1, 4, 16, 128):
            sim = torch.randn(*shape)
            got = L.causal_adaptive_threshold(sim, target)
            want = reference(sim, target)
            assert torch.equal(got, want), (shape, target, (got - want).abs().max())


def test_empty_pool_geometry():
    dev = torch.device('cpu')
    ids = torch.empty(0, dtype=torch.long, device=dev)
    assert L._n_blocks_cached(ids) == 0, 'an empty block table has zero blocks'
    comp, last_tok, B = L.pool_variable_blocks(torch.empty(0, 4), torch.empty(0, 4),
                                               torch.empty(0, 4), torch.empty(0, 4), ids,
                                               torch.empty(4, 4), torch.empty(4, 4), 1)
    assert comp.shape == (0, 4) and last_tok.shape == (0,) and B == 0
    comp, last_tok, B = L.pool_blocks_single(torch.empty(0, 4), torch.empty(0, 4),
                                             torch.empty(4, 4), ids)
    assert comp.shape == (0, 4) and last_tok.shape == (0,) and B == 0


def test_segment_empty():
    bids, hon, counts = L._segment(0, [], 2, 128)
    assert bids.size == 0 and hon == {} and counts == []
    assert L._segment(1, [], 2, 128)[2] == [1]


def test_segment_uniform_stride_only_when_all_cuts():
    # A wanted cut at n-1 sits after the final token and cannot short-circuit
    # the walk.  {1} for n=2 has n-1 slots but leaves slot 0 unset, so the
    # first block must still run to the end (count 2), not take the short
    # uniform stride.  This is the boundary the uniform-stride fast path must
    # not absorb.
    bids, hon, counts = L._segment(2, [False, True], 1, 2)
    assert counts == [2] and bids.tolist() == [0, 0] and hon == {}, (bids, hon, counts)
    # n=3, two wanted slots including n-1: still NOT the all-cuts case.
    bids, hon, counts = L._segment(3, [False, True, True], 1, 2)
    assert counts == [2, 1] and bids.tolist() == [0, 0, 1] and hon == {1: 1.0}, (bids, hon, counts)
    bids, hon, counts = L._segment(3, [True, False, True], 1, 2)
    assert counts == [1, 2] and bids.tolist() == [0, 1, 1] and hon == {0: 1.0}, (bids, hon, counts)
    # All answer-forcing slots (0..n-2) present -> the uniform stride is exact.
    bids, hon, counts = L._segment(3, [True, True, True], 1, 2)
    assert counts == [1, 1, 1] and bids.tolist() == [0, 1, 2], (bids, hon, counts)
    bids, _, counts = L._segment(4, [True, True, False, True], 1, 3)
    assert counts == [1, 1, 2] and bids.tolist() == [0, 1, 2, 2], (bids, counts)


def test_shipped_manifest_consistency():
    with open(os.path.join(REPO, 'confirmation_frozen.json'), encoding='utf-8') as f:
        v2 = json.load(f)
    assert v2['manifest_version'] == 3
    n = len(v2['seeds'])
    m = len(v2['primary_comparisons'])
    assert n == 12 and v2['seeds'] == list(range(12))
    assert v2['statistics']['n_seeds'] == n
    assert v2['code_semantics_expected'] == L.CODE_SEMANTICS
    assert v2.get('eager_batch_expected') is False, 'v3 must bind the real per-row RoPE dispatch'
    assert v2.get('amp_dtype_expected') == 'bf16' and v2.get('amp_grad_scaler_expected') is False
    assert abs(v2['statistics']['p_floor'] - L.signflip_p_floor(n)) < 1e-15
    assert abs(v2['statistics']['min_attainable_p_holm'] - L.holm_min_attainable_p(n, m)) < 1e-12
    assert v2['statistics']['min_attainable_p_holm'] < 0.05
    assert v2['statistics']['min_consistency_for_first_gate'] == L.signflip_min_consistency(n, m)
    # v3 declares a reachable confirmation floor (8 complete pairs) and it is
    # mathematically able to reject under Holm/4.
    assert v2['statistics']['min_seeds_for_confirmation'] == 8
    assert L.holm_min_attainable_p(v2['statistics']['min_seeds_for_confirmation'], m) <= 0.05
    for k, want in v2['statistics']['power_first_holm_gate'].items():
        q = float(k.split('=')[1])
        got = L.signflip_power(n, m, q)
        assert abs(got - want) < 0.02, f'power table drift at {k}: manifest {want} vs code {got:.4f}'
    # the revision history must record the 6->12 change and its precondition
    hist = v2['revision']['history']
    assert any(h.get('version') == 1 for h in hist) and any(h.get('version') == 2 for h in hist) and any(h.get('version') == 3 for h in hist)
    # v5 pins the per-arm parameter-matching knob (two sink-matched full arms)
    per = v2.get('binding', {}).get('per_arm_recipe_signature') or {}
    assert set(per) == set(v2['arms']), per
    assert abs(float(per['full_rope']['mr']) - 6.76348876953125) < 1e-9, per
    assert abs(float(per['full_rope_sink']['mr']) - 6.763427734375) < 1e-9, per
    assert all(abs(float(per[a]['mr']) - 4.0) < 1e-12
               for a in v2['arms'] if a not in ('full_rope', 'full_rope_sink')), per
    # scope block present
    assert v2.get('scope', {}).get('not_covered'), 'scope.not_covered must name the dynamic/HCA claims that are out of scope'
    # the compute-budget caveat must be explicit, not buried in prose
    assert v2.get('compute_budget_disclosure', {}).get('flops'), 'v3 must disclose the dense-arm MLP/FLOPs mismatch'


def test_frozen_seeds():
    assert V12._frozen_seeds() == list(range(12)), 'phase seeds must come from the shipped manifest (single source of truth)'


def test_doc_level_analysis():
    n_seed, n_doc = 3, 4
    base_a = [[30.0 + i + 0.5 * s for i in range(n_doc)] for s in range(n_seed)]
    base_b = [[31.0 + i + 0.5 * s for i in range(n_doc)] for s in range(n_seed)]
    recs_a = {s: {'by_len': {'4096': {'ppl': 1.0, 'ppl_per_seq': base_a[s]}}} for s in range(n_seed)}
    recs_b = {s: {'by_len': {'4096': {'ppl': 1.0, 'ppl_per_seq': base_b[s]}}} for s in range(n_seed)}
    d = V12._doc_level_analysis(recs_a, recs_b, list(range(n_seed)), 4096)
    assert d is not None and d['n_docs'] == n_doc
    assert abs(d['per_doc_delta_mean'] - (-1.0)) < 1e-9, 'per-doc delta must be a - b = -1.0'
    assert abs(d['between_doc_sd']) < 1e-9, 'constant -1.0 shift across docs -> zero between-doc spread'

