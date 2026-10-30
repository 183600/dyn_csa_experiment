"""CPU checks for the confirmation-level SESOI / TOST readout.

The frozen confirmation family answers "can the zero effect be rejected?".
It does not answer "is the effect at least as large as the smallest effect the
study cares about?".  ``confirmation_practical_significance`` adds that
descriptive readout on the same per-seed pairs: a relative-SESOI log-ratio
margin, a paired-t TOST interval, and a one-sided beyond-margin p.  The two
Holm families it reports are separate from the frozen superiority family and
never touch ``p_exact_signflip``, ``p_holm`` or ``confirmed``.

No training or GPU is read; the end-to-end case writes synthetic by-length
records to a temporary directory.
"""
import contextlib
import copy
import inspect
import io
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

import explib.confirmation as V12


def _synthetic_manifest(td, arms, seeds, lengths):
    m = {'manifest_version': 3, 'frozen_at': '2026-01-01',
         'arms': list(arms), 'seeds': list(seeds),
         'primary_comparisons': [
             {'name': f'{arms[0]} vs {a}', 'a': arms[0], 'b': a,
              'endpoint': 'by_len.ppl@2048', 'expected_sign': 'negative'}
             for a in arms[1:]],
         'statistics': {'test': 'exact two-sided sign-flip',
                        'correction': 'holm step-down',
                        'descriptive_only': True,
                        'require_complete_coverage': True,
                        'min_seeds_for_confirmation': 2},
         'eval': {'by_length_ppl': list(lengths), 'n_seqs_clean': 4}}
    p = os.path.join(td, 'm.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(m, f)
    return V12._load_manifest(p)


def _record(arm, seed, msha, lengths, tail_delta):
    by = {}
    nll = 0.0
    prev = 1
    for Ln in lengths:
        n_tok = 4 * (Ln - 1)
        band_tok = n_tok if Ln == min(lengths) else 4 * (Ln - prev)
        nll += 4.0 * band_tok
        if Ln == max(lengths):
            nll += float(tail_delta.get(arm, 0.0))
        by[str(Ln)] = {'ppl': math.exp(nll / n_tok), 'n_tok': n_tok,
                       'nll_sum': nll, 'truncated': False, 'n_seq': 4}
        prev = Ln
    return {'variant': arm, 'seed': seed, 'split': 'test', 'data_fp': 'FP',
            'by_len': by, '_code': L.code_stamp(),
            'manifest_sha256': msha, 'n_seq': 4}


def _synthetic_stats(td, tail_delta=None, seeds=range(6)):
    tail_delta = tail_delta or {}
    m = _synthetic_manifest(td, ('a', 'b'), seeds, (512, 1024, 2048))
    outdir = os.path.join(td, 'res')
    os.makedirs(outdir)
    V12._check_or_write_freeze_lock(outdir, m, 'FP', stage='accessed')
    raw = {}
    for s in seeds:
        for arm in ('a', 'b'):
            raw[f'{arm}::s{s}'] = _record(
                arm, s, m['manifest_sha256'], (512, 1024, 2048), tail_delta)
    bp = os.path.join(outdir, 'bylen.json')
    with open(bp, 'w', encoding='utf-8') as f:
        json.dump(raw, f)
    p = os.path.join(td, 'stats.json')
    with contextlib.redirect_stdout(io.StringIO()):
        st = V12.finaltest_stats(
            manifest=m, bylen_path=bp,
            probe_path=os.path.join(td, 'no_probe.json'), out=p)
    return st, p


def test_policy_and_normalisation_fail_closed():
    p = L.CONFIRMATION_SESOI
    assert p['version'] == 'confirmation-sesoi-1', p
    assert p['scale'] == 'log_ratio' and p['relative_pct'] == 1.0, p
    pol = L.confirmation_sesoi_for_manifest(
        {'primary_comparisons': [{'name': 'x'}],
         'statistics': {'alpha': 0.05}})
    assert pol['source'] == 'library_policy', pol
    assert pol['declared_in_frozen_manifest'] is False, pol
    assert math.isclose(pol['margins']['x']['margin'], math.log(1.01)), pol
    assert pol['tost_conf'] == 0.95, pol
    stat_owned = L.confirmation_sesoi_for_manifest(
        {'primary_comparisons': [{'name': 'x'}],
         'statistics': {'alpha': 0.05,
                        'practical_significance': {'relative_pct': 2.0}}})
    assert stat_owned['source'] == 'manifest', stat_owned
    assert math.isclose(stat_owned['margins']['x']['margin'],
                        math.log(1.02)), stat_owned
    for bad in ({'relative_pct': 0}, {'relative_pct': -1},
                {'relative_pct': float('nan')}, {'relative_pct': True},
                {'relative_pct': 1, 'absolute_log_ratio': 0.01},
                {'scale': 'raw_ppl', 'relative_pct': 1},
                {'scale': 'log_ratio', 'absolute_ppl': 1},
                {'scale': 'bogus', 'relative_pct': 1}):
        try:
            L.normalise_confirmation_sesoi(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'a malformed SESOI was accepted: {bad!r}')
    try:
        L.normalise_confirmation_sesoi('1%')
    except ValueError:
        pass
    else:
        raise AssertionError('a string SESOI was accepted')


def test_manifest_override_and_unknown_comparison():
    m = {'primary_comparisons': [{'name': 'x'}, {'name': 'y'}],
         'statistics': {'alpha': 0.05},
         'practical_significance': {
             'scale': 'log_ratio', 'relative_pct': 2.0,
             'declared_in_frozen_manifest': True,
             'comparisons': {'x': {'relative_pct': 0.5}}}}
    pol = L.confirmation_sesoi_for_manifest(m)
    assert pol['source'] == 'manifest', pol
    assert pol['declared_in_frozen_manifest'] is True, pol
    assert math.isclose(pol['margins']['x']['margin'], math.log(1.005)), pol
    assert math.isclose(pol['margins']['y']['margin'], math.log(1.02)), pol
    bad = copy.deepcopy(m)
    bad['practical_significance']['comparisons'] = {'typo': {'relative_pct': 1.0}}
    try:
        L.confirmation_sesoi_for_manifest(bad)
    except ValueError as e:
        assert 'typo' in str(e), e
    else:
        raise AssertionError('a margin for an undeclared comparison was accepted')


def test_tost_and_beyond_sesoi_classification():
    margin = math.log(1.01)
    beyond = L.confirmation_practical_significance(
        [-0.05] * 12, margin=margin, expected_sign='negative',
        seeds=list(range(12)))
    assert beyond['status'] == 'beyond_sesoi', beyond
    assert beyond['beyond_sesoi'] is True and beyond['equivalent'] is False, beyond
    assert beyond['p_beyond'] < 0.025, beyond
    assert abs(beyond['relative_effect_pct'] + 4.8770575) < 1e-6, beyond

    equiv = L.confirmation_practical_significance(
        [-0.002] * 12, margin=margin, expected_sign='negative',
        seeds=list(range(12)))
    assert equiv['status'] == 'equivalent_within_sesoi', equiv
    assert equiv['equivalent'] is True and equiv['p_beyond'] > 0.9, equiv

    conflict = L.confirmation_practical_significance(
        [0.05] * 12, margin=margin, expected_sign='negative',
        seeds=list(range(12)))
    assert conflict['status'] == 'direction_conflict', conflict
    assert conflict['beyond_sesoi'] is False, conflict

    indeterminate = L.confirmation_practical_significance(
        [-0.03, 0.02] * 6, margin=margin, expected_sign='negative',
        seeds=list(range(12)))
    assert indeterminate['status'] == 'indeterminate', indeterminate

    zero = L.confirmation_practical_significance(
        [0.0] * 12, margin=margin, expected_sign='negative',
        seeds=list(range(12)))
    assert zero['status'] == 'zero_effect', zero
    assert zero['equivalent'] is None and zero['p_tost'] is None, zero

    nodir = L.confirmation_practical_significance(
        [-0.002] * 12, margin=margin, expected_sign=None,
        seeds=list(range(12)))
    assert nodir['status'] == 'equivalent_within_sesoi', nodir
    assert nodir['p_beyond'] is None, nodir

    one = L.confirmation_practical_significance(
        [-0.05], margin=margin, expected_sign='negative')
    assert one['status'] == 'not_estimable', one

    undeclared = L.confirmation_practical_significance(
        [-0.05] * 12, margin=None, expected_sign='negative')
    assert undeclared['status'] == 'not_declared', undeclared


def test_family_holm_and_confirmed_but_sub_sesoi():
    margin = math.log(1.01)
    b1 = L.confirmation_practical_significance(
        [-0.05] * 12, margin=margin, expected_sign='negative',
        seeds=list(range(12)))
    b2 = L.confirmation_practical_significance(
        [-0.002] * 12, margin=margin, expected_sign='negative',
        seeds=list(range(12)))
    comps = {'a': {'practical_significance': b1},
             'b': {'practical_significance': b2},
             'c': {'practical_significance': {'status': 'not_estimable'}}}
    pol = L.confirmation_sesoi_for_manifest(
        {'primary_comparisons': [{'name': n} for n in ('a', 'b', 'c')],
         'statistics': {'alpha': 0.05}})
    fam = L.confirmation_practical_family(
        comps, confirmed=['a', 'b'], policy=pol, alpha=0.05,
        family_names=['a', 'b', 'c'])
    assert fam['status'] == 'ok', fam
    assert fam['summary']['practically_meaningful'] == ['a'], fam
    assert fam['summary']['equivalent_within_sesoi'] == ['b'], fam
    assert fam['summary']['not_estimable'] == ['c'], fam
    assert fam['summary']['frozen_confirmed_but_not_beyond_sesoi'] == ['b'], fam
    assert fam['holm']['beyond_sesoi']['c']['p_raw'] == 1.0, fam
    assert fam['holm']['beyond_sesoi']['c']['m_effective'] == 3, fam
    assert fam['holm']['tost_equivalence']['c']['m_effective'] == 3, fam


def test_design_audit_requires_a_complete_margin_when_flagged():
    base = {'primary_comparisons': [{'name': 'x'}, {'name': 'y'}],
            'statistics': {'alpha': 0.05}}
    missing = L.confirmation_sesoi_design_audit(base)
    assert missing['status'] == 'not_declared', missing
    required = {'primary_comparisons': [{'name': 'x'}, {'name': 'y'}],
                'statistics': {'alpha': 0.05,
                               'require_practical_significance': True}}
    bad = L.confirmation_sesoi_design_audit(required)
    assert bad['status'] == 'fail' and bad['problems'], bad
    good = {'primary_comparisons': [{'name': 'x'}, {'name': 'y'}],
            'statistics': {'alpha': 0.05,
                           'require_practical_significance': True},
            'practical_significance': {'scale': 'log_ratio',
                                       'relative_pct': 1.0}}
    ok = L.confirmation_sesoi_design_audit(good)
    assert ok['status'] == 'ok' and ok['declared'] is True, ok
    partial = copy.deepcopy(good)
    partial['practical_significance']['comparisons'] = {'x': {'relative_pct': 1.0}}
    # A top-level default covers every name; to test a partial declaration,
    # remove the default unit and keep only the named override.
    partial['practical_significance'].pop('relative_pct')
    partial['practical_significance']['comparisons'] = {'x': {'relative_pct': 1.0}}
    bad2 = L.confirmation_sesoi_design_audit(partial)
    assert bad2['status'] == 'fail', bad2


def test_extreme_and_malformed_inputs_are_strict_json():
    block = L.confirmation_practical_significance(
        [1e308, 1e308, -1e308], margin=1.0, expected_sign='positive',
        seeds=[0, 1, 2], ppl_a=[1e308, 1e308, 1e308],
        ppl_b=[1e308, 1e308, 1e308])
    json.dumps(block, allow_nan=False)
    dropped = L.confirmation_practical_significance(
        [-0.05, float('nan'), -0.06], margin=math.log(1.01),
        expected_sign='negative', seeds=[0, 1, 2])
    assert dropped['n'] == 2 and dropped['n_dropped'] == 1, dropped
    for bad in (lambda: L.confirmation_practical_significance(
                    [-0.05] * 3, margin=0.0),
                lambda: L.confirmation_practical_significance(
                    [-0.05] * 3, margin=float('inf')),
                lambda: L.confirmation_practical_significance(
                    'abc', margin=0.1),
                lambda: L.confirmation_practical_significance(
                    [-0.05] * 3, margin=0.1, seeds=[0, 0, 1]),
                lambda: L.confirmation_practical_significance(
                    [-0.05] * 3, margin=0.1, expected_sign='sideways'),
                lambda: L.confirmation_practical_significance(
                    [-0.05] * 3, margin=0.1, tost_conf=1.0)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError('a malformed input was accepted')


def test_finaltest_emits_practical_block_and_audit_replays_it():
    with tempfile.TemporaryDirectory() as td:
        st, p = _synthetic_stats(td, tail_delta={'a': -20.0, 'b': 0.0})
        fam = st['practical_significance']
        assert fam['kind'] == 'confirmation_practical_significance_family', fam
        assert fam['status'] == 'ok', fam
        assert fam['summary']['frozen_confirmed'] == sorted(st['confirmed']), fam
        name = next(iter(fam['comparisons']))
        block = st['comparisons'][name]['practical_significance']
        assert block['kind'] == 'confirmation_practical_significance', block
        assert len(block['per_seed_ppl_a']) == block['n'], block
        assert len(block['per_seed_ppl_b']) == block['n'], block
        assert len(block['per_seed_log_ratio']) == block['n'], block
        assert block['margin_declared_in_frozen_manifest'] is False, block
        json.dumps(fam, allow_nan=False)
        # The small synthetic effect is statistically confirmed but below the
        # 1% practical margin, so the aggregate must flag it explicitly.
        assert name in fam['summary']['frozen_confirmed_but_not_beyond_sesoi'], fam
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_practical_significance(path=p)
        assert out['status'] == 'ok', out
        assert out['n_replayed'] == 1 and not out['problems'], out
        assert any('SESOI' in w for w in out['warnings']), out


def test_strong_effect_is_practically_meaningful():
    with tempfile.TemporaryDirectory() as td:
        st, p = _synthetic_stats(td, tail_delta={'a': -400.0, 'b': 0.0})
        name = next(iter(st['practical_significance']['comparisons']))
        block = st['comparisons'][name]['practical_significance']
        assert block['status'] == 'beyond_sesoi', block
        assert block['relative_effect_pct'] < -4.0, block
        assert name in st['practical_significance']['summary']['practically_meaningful'], st
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_practical_significance(path=p)
        assert out['status'] == 'ok', out
        assert not out['warnings'], out


def test_audit_detects_tampering_and_legacy_artifacts():
    with tempfile.TemporaryDirectory() as td:
        _st, p = _synthetic_stats(td, tail_delta={'a': -400.0, 'b': 0.0})
        good = json.load(open(p, encoding='utf-8'))
        name = next(iter(good['practical_significance']['comparisons']))

        def write(mut):
            q = os.path.join(td, 'bad.json')
            with open(q, 'w', encoding='utf-8') as f:
                json.dump(mut, f)
            with contextlib.redirect_stdout(io.StringIO()):
                return A.audit_practical_significance(path=q)

        with contextlib.redirect_stdout(io.StringIO()):
            assert A.audit_practical_significance(path=p)['status'] == 'ok'
        cases = []
        d = copy.deepcopy(good)
        d['comparisons'][name]['practical_significance']['p_tost'] = 0.99
        cases.append(d)
        d = copy.deepcopy(good)
        d['comparisons'][name]['practical_significance']['per_seed_ppl_b'][0] *= 1.5
        cases.append(d)
        d = copy.deepcopy(good)
        d['practical_significance']['summary']['practically_meaningful'] = []
        cases.append(d)
        d = copy.deepcopy(good)
        d['practical_significance']['confirmed'] = [name]
        cases.append(d)
        d = copy.deepcopy(good)
        d['practical_significance']['holm']['beyond_sesoi'][name]['p_holm'] = 1.0
        cases.append(d)
        for bad in cases:
            assert write(bad)['status'] == 'fail', bad
        legacy = copy.deepcopy(good)
        legacy.pop('practical_significance')
        assert write(legacy)['status'] == 'not_verifiable_locally'
        bare = copy.deepcopy(good)
        bare['practical_significance'] = {
            'kind': 'confirmation_practical_significance_family',
            'status': 'not_estimable', 'comparisons': {}, 'family_names': []}
        assert write(bare)['status'] == 'not_applicable'


def test_report_renders_practical_section():
    margin = math.log(1.01)
    b = L.confirmation_practical_significance(
        [-0.05] * 12, margin=margin, expected_sign='negative',
        seeds=list(range(12)))
    comps = {'a vs b': {'practical_significance': b}}
    pol = L.confirmation_sesoi_for_manifest(
        {'primary_comparisons': [{'name': 'a vs b'}],
         'statistics': {'alpha': 0.05}})
    fam = L.confirmation_practical_family(
        comps, confirmed=['a vs b'], policy=pol, alpha=0.05,
        family_names=['a vs b'])
    md = '\n'.join(V12._practical_significance_markdown(fam))
    assert '实际显著性' in md and 'SESOI' in md and 'TOST' in md, md
    assert '1%' in md and '描述性' in md, md
    assert '不是「效应恰好为零」' in md, md
    assert '不改变' in md and '`confirmed`' in md, md
    assert V12._practical_significance_markdown({}) == []


def test_cli_aliases_and_audit_documentation():
    assert hasattr(A, 'audit_practical_significance')
    src = inspect.getsource(A)
    assert 'practical_significance' in src, 'audit CLI is not registered'
    assert "'practical': audit_practical_significance" in src, src[-2000:]
    assert 'practical_significance' in (A.__doc__ or ''), 'audit gate is not documented'
