"""CPU checks for the position-band decomposition of by-length PPL cells.

The frozen by-length endpoint evaluates the same fixed rows at 512/1024/
2048/4096.  Because the LM is causal (and the segmenter is prefix-invariant),
the shorter-length cell is a prefix of the longer one; the difference of two
length totals is exactly the NLL of the newly covered target positions.
``length_position_decomposition`` uses that identity to compare arms *inside*
a fixed position band, so "the advantage grows with length" is separated from
the changing mix of early and late positions.

No training, checkpoint or GPU is read; the end-to-end case writes synthetic
by-length records to a temporary directory.
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


def _cell(rows):
    return {
        'ppl_by_seed': {str(s): math.exp(n / t) for s, (n, t) in rows.items()},
        'nll_by_seed': {str(s): n for s, (n, t) in rows.items()},
        'n_tokens_by_seed': {str(s): t for s, (n, t) in rows.items()},
    }


def _panels(arms=('a', 'b'), seeds=range(6), lengths=(512, 1024, 2048),
            tail_delta=None):
    """Synthetic cumulative cells with a known per-band NLL structure.

    The 512->1024 band is identical for every arm; the 1024->2048 band applies
    ``tail_delta[arm]`` extra NLL to that arm (negative = better).
    """
    tail_delta = tail_delta or {}
    by = {}
    for arm in arms:
        by[arm] = {}
        nll = 0.0
        prev = 1
        for Ln in lengths:
            n_tok = 4 * (Ln - 1)
            band_tok = n_tok if Ln == 512 else 4 * (Ln - prev)
            nll += 4.0 * band_tok
            if Ln == max(lengths):
                nll += float(tail_delta.get(arm, 0.0))
            by[arm][str(Ln)] = _cell(
                {s: (nll, n_tok) for s in seeds})
            prev = Ln
    return by


def test_policy_and_statuses_are_declared():
    p = L.LENGTH_POSITION_POLICY
    assert p['version'] == 'length-position-bands-1', p
    assert p['min_seeds'] >= 2 and p['min_lengths'] >= 2, p
    assert set(L.LENGTH_POSITION_STATUSES) == {'ok', 'not_estimable'}
    assert 'prefix' in p['identity'].lower(), p


def test_bands_reconstruct_cumulative_log_ratio():
    by = _panels(tail_delta={'a': -20.0, 'b': 0.0})
    r = L.length_position_decomposition(
        by, pairs=[{'name': 'a vs b', 'a': 'a', 'b': 'b'}])
    assert r['status'] == 'ok', r
    pair = r['pairs']['a vs b']
    assert pair['status'] == 'ok', pair
    assert list(pair['bands']) == ['512-1023', '1024-2047'], pair['bands']
    assert pair['max_reconstruction_error'] < 1e-12, pair
    # The first band is identical; the second carries the whole difference.
    first, second = (pair['bands'][k] for k in ('512-1023', '1024-2047'))
    assert abs(first['band_ppl_delta']['mean']) < 1e-12, first
    assert second['band_ppl_delta']['mean'] < 0.0, second
    assert pair['dominant_band_at_max_length'] == '1024-2047', pair
    # Total log ratio at L=2048 equals the sum of the two contributions.
    total = pair['total_by_length']['2048']
    assert math.isclose(
        total['mean'],
        first['mean_contribution_to_total_log_ratio_at_max']
        + second['mean_contribution_to_total_log_ratio_at_max'],
        rel_tol=1e-12, abs_tol=1e-12), pair


def test_position_matched_effect_isolates_the_extension_band():
    by = _panels(tail_delta={'a': -200.0, 'b': 0.0}, seeds=range(8))
    r = L.length_position_decomposition(
        by, pairs=[{'name': 'a vs b', 'a': 'a', 'b': 'b'}])
    pair = r['pairs']['a vs b']
    first, second = (pair['bands'][k] for k in ('512-1023', '1024-2047'))
    assert first['band_ppl_delta']['p_exact_signflip'] == 1.0, first
    assert second['band_ppl_delta']['p_exact_signflip'] < 0.05, second
    assert second['band_ppl_delta']['direction'] == 'negative', second
    # The interaction against the common-prefix band is the direct
    # "only the extension is different" readout.
    inter = second['interaction_vs_first_band']
    assert inter['n'] == 8, inter
    assert inter['p_exact_signflip'] < 0.05, inter
    assert inter['direction'] == 'negative', inter


def test_missing_seed_shrinks_the_band_instead_of_dropping_the_panel():
    by = _panels(lengths=(512, 1024, 2048))
    # Remove one seed from arm a's 2048 cell: both bands must use the 5
    # common seeds, and the dropped seed must be visible in the seed list.
    for key in ('ppl_by_seed', 'nll_by_seed', 'n_tokens_by_seed'):
        del by['a']['2048'][key]['5']
    r = L.length_position_decomposition(
        by, pairs=[{'name': 'a vs b', 'a': 'a', 'b': 'b'}])
    pair = r['pairs']['a vs b']
    assert pair['bands']['1024-2047']['n'] == 5, pair
    assert pair['bands']['1024-2047']['seeds'] == [0, 1, 2, 3, 4], pair
    # The arm curve uses the same full-coverage seed set, not a per-band
    # subset that changes composition.
    assert [b['n'] for b in r['arms']['a']['bands']] == [5, 5], r['arms']['a']


def test_non_monotone_nll_and_malformed_cells_fail_closed():
    by = _panels(lengths=(512, 1024))
    # A causal prefix cannot lose NLL when later positions are added.
    by['a']['1024']['nll_by_seed']['0'] = 1.0
    _nt = int(by['a']['1024']['n_tokens_by_seed']['0'])
    by['a']['1024']['ppl_by_seed']['0'] = math.exp(1.0 / _nt)
    try:
        L.length_position_decomposition(
            by, pairs=[{'name': 'a vs b', 'a': 'a', 'b': 'b'}])
    except ValueError as exc:
        assert 'non-monotone' in str(exc), exc
    else:
        raise AssertionError('non-monotone NLL was accepted')

    by = _panels(lengths=(512, 1024))
    by['a']['512']['ppl_by_seed']['0'] *= 1.01
    try:
        L.length_position_decomposition(
            by, pairs=[{'name': 'a vs b', 'a': 'a', 'b': 'b'}])
    except ValueError as exc:
        assert 'exp(nll_sum/n_tok)' in str(exc), exc
    else:
        raise AssertionError('inconsistent ppl/nll identity was accepted')

    by = _panels(lengths=(512, 1024))
    by['a']['512']['ppl_by_seed']['0x'] = by['a']['512']['ppl_by_seed']['0']
    try:
        L.length_position_decomposition(
            by, pairs=[{'name': 'a vs b', 'a': 'a', 'b': 'b'}])
    except ValueError as exc:
        assert 'integer key' in str(exc), exc
    else:
        raise AssertionError('non-integer seed key was accepted')

    by = _panels(lengths=(512, 1024))
    try:
        L.length_position_decomposition(
            by, pairs=[{'name': 'a vs b', 'a': 'a', 'b': 'a'}])
    except ValueError as exc:
        assert 'itself' in str(exc), exc
    else:
        raise AssertionError('self-pair was accepted')


def test_json_is_strict_and_replay_is_deterministic():
    by = _panels(tail_delta={'a': -20.0, 'b': 0.0})
    pairs = [{'name': 'a vs b', 'a': 'a', 'b': 'b'}]
    r1 = L.length_position_decomposition(by, pairs=pairs)
    r2 = L.length_position_decomposition(copy.deepcopy(by), pairs=pairs)
    assert json.dumps(r1, allow_nan=False) == json.dumps(r2, allow_nan=False)
    # A large but finite PPL panel must stay JSON-safe (no inf/NaN).
    big = 1e200
    big_by = {
        'a': {'512': _cell({0: (400.0 * math.log(big), 400)}),
              '1024': _cell({0: (800.0 * math.log(big), 800)})},
        'b': {'512': _cell({0: (401.0 * math.log(big), 400)}),
              '1024': _cell({0: (801.0 * math.log(big), 800)})},
    }
    rb = L.length_position_decomposition(
        big_by, pairs=[{'name': 'a vs b', 'a': 'a', 'b': 'b'}])
    json.dumps(rb, allow_nan=False)


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


def _synthetic_stats(td, arms=('a', 'b'), seeds=range(6),
                     lengths=(512, 1024, 2048), tail_delta=None):
    tail_delta = tail_delta or {}
    m = _synthetic_manifest(td, arms, seeds, lengths)
    outdir = os.path.join(td, 'res')
    os.makedirs(outdir)
    V12._check_or_write_freeze_lock(outdir, m, 'FP', stage='accessed')
    raw = {}
    for s in seeds:
        for arm in arms:
            raw[f'{arm}::s{s}'] = _record(
                arm, s, m['manifest_sha256'], lengths, tail_delta)
    bp = os.path.join(outdir, 'bylen.json')
    with open(bp, 'w', encoding='utf-8') as f:
        json.dump(raw, f)
    p = os.path.join(td, 'stats.json')
    with contextlib.redirect_stdout(io.StringIO()):
        st = V12.finaltest_stats(
            manifest=m, bylen_path=bp,
            probe_path=os.path.join(td, 'no_probe.json'), out=p)
    return st, p


def test_finaltest_stats_emits_replayable_bands_with_per_seed_accounting():
    with tempfile.TemporaryDirectory() as td:
        st, p = _synthetic_stats(
            td, tail_delta={'a': -200.0, 'b': 0.0})
        for arm in ('a', 'b'):
            for Ln in ('512', '1024', '2048'):
                cell = st['by_length'][arm][Ln]
                assert set(cell['nll_by_seed']) == set(cell['ppl_by_seed']), cell
                assert set(cell['n_tokens_by_seed']) == set(cell['ppl_by_seed']), cell
        block = st['length_position']
        assert block['kind'] == 'length_position_decomposition', block
        assert block['status'] == 'ok', block
        pair = block['pairs']['a vs b']
        assert pair['bands']['1024-2047']['band_ppl_delta']['p_exact_signflip'] < 0.05, pair
        # The synthetic second band is constant across seeds (zero SD); the
        # primary max-T helper refuses that family, so the exact score
        # fallback must still produce a family readout instead of dropping it.
        bf = pair['band_family_max_t']
        assert bf['status'] == 'ok', bf
        assert 'fallback' in bf['method'], bf
        assert bf['p_family'] < 0.05, bf
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_length_position(path=p)
        assert out['status'] == 'ok', out
        assert out['n_replayed'] == 1, out
        assert not out['problems'], out


def test_audit_rejects_tampered_band_blocks():
    with tempfile.TemporaryDirectory() as td:
        _st, p = _synthetic_stats(td, tail_delta={'a': -200.0, 'b': 0.0})
        good = json.load(open(p, encoding='utf-8'))
        with contextlib.redirect_stdout(io.StringIO()):
            assert A.audit_length_position(path=p)['status'] == 'ok'
        pair_name = next(iter(good['length_position']['pairs']))
        band_name = '1024-2047'

        def write(mut):
            q = os.path.join(td, 'bad.json')
            with open(q, 'w', encoding='utf-8') as f:
                json.dump(mut, f)
            with contextlib.redirect_stdout(io.StringIO()):
                return A.audit_length_position(path=q)

        cases = []
        d = copy.deepcopy(good)
        d['length_position']['pairs'][pair_name]['bands'][band_name][
            'band_ppl_delta']['p_exact_signflip'] = 0.99
        cases.append(d)
        d = copy.deepcopy(good)
        d['length_position']['pairs'][pair_name]['bands'][band_name][
            'mean_contribution_to_total_log_ratio_at_max'] *= 2.0
        cases.append(d)
        d = copy.deepcopy(good)
        d['length_position']['status'] = 'not_estimable'
        cases.append(d)
        d = copy.deepcopy(good)
        d['length_position']['confirmed'] = ['a vs b']
        cases.append(d)
        d = copy.deepcopy(good)
        d['by_length']['a']['1024']['nll_by_seed']['0'] *= 1.01
        cases.append(d)
        for bad in cases:
            assert write(bad)['status'] == 'fail', bad
        # Missing block is a legacy artifact, not a pass.
        legacy = copy.deepcopy(good)
        legacy.pop('length_position')
        assert write(legacy)['status'] == 'not_verifiable_locally'
        # A bare not_estimable block with no pair readouts is not_applicable.
        bare = copy.deepcopy(good)
        bare['length_position'] = {
            'kind': 'length_position_decomposition',
            'status': 'not_estimable', 'pairs': {},
            'reason': 'fewer than two arms'}
        assert write(bare)['status'] == 'not_applicable'


def test_audit_marks_a_replayed_not_estimable_block_as_not_applicable():
    by = _panels(lengths=(512, 1024, 2048))
    for key in ('ppl_by_seed', 'nll_by_seed', 'n_tokens_by_seed'):
        for seed in range(1, 6):
            del by['a']['2048'][key][str(seed)]
    block = L.length_position_decomposition(
        by, pairs=[{'name': 'a vs b', 'a': 'a', 'b': 'b'}])
    assert block['status'] == 'not_estimable', block
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'stats.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'length_position': block, 'by_length': by}, f)
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_length_position(path=p)
    assert out['status'] == 'not_applicable', out
    assert out['n_replayed'] == 0, out


def test_report_renders_position_bands_and_boundary():
    with tempfile.TemporaryDirectory() as td:
        _st, p = _synthetic_stats(td, tail_delta={'a': -200.0, 'b': 0.0})
        st = json.load(open(p, encoding='utf-8'))
        wrapped = {'finaltest': st}
        sp = os.path.join(td, 'stats.json')
        with open(sp, 'w', encoding='utf-8') as f:
            json.dump(wrapped, f)
        out = os.path.join(td, 'REPORT.md')
        original = V12.v12_analysis
        V12.v12_analysis = lambda out=None: wrapped  # noqa: ARG005
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                V12.build_report(out=out, stats_p=sp)
        finally:
            V12.v12_analysis = original
        with open(out, encoding='utf-8') as f:
            text = f.read()
    assert '长度位置分段' in text, text[:1200]
    assert 'band 512-1023' in text, text
    assert '1024-2047' in text, text
    assert '跨 band 扫描读数' in text, text
    assert '因果前缀恒等' in text, text
    assert '不进入 Holm 族' in text, text


def test_wiring_is_in_audit_cli_and_confirmation():
    assert 'length_position' in (A.__doc__ or ''), 'audit command not documented'
    assert "'length_position': audit_length_position()" in inspect.getsource(A.run_report)
    assert "'length_position': audit_length_position" in inspect.getsource(A)
    cli = inspect.getsource(A._run_cli)
    for alias in ('length_position', 'length_bands', 'position_bands'):
        assert alias in cli, alias
    assert 'length_position_decomposition' in inspect.getsource(V12.finaltest_stats)
    assert '长度位置分段' in inspect.getsource(V12.build_report)


def test_common_prefix_level_is_part_of_the_absolute_decomposition():
    # The shortest cell is a prefix *level*, not a difference of two lengths.
    # A pair that already differs there must still decompose: the prefix is an
    # explicit first contribution, and prefix + bands equals the total log
    # ratio at every common length.
    by = _panels(lengths=(512, 1024, 2048))
    for Ln in ('512', '1024', '2048'):
        cell = by['a'][Ln]
        for seed in list(cell['nll_by_seed']):
            nll = float(cell['nll_by_seed'][seed]) - 40.0
            cell['nll_by_seed'][seed] = nll
            cell['ppl_by_seed'][seed] = math.exp(
                nll / float(cell['n_tokens_by_seed'][seed]))
    r = L.length_position_decomposition(
        by, pairs=[{'name': 'a vs b', 'a': 'a', 'b': 'b'}])
    assert r['status'] == 'ok', r
    pair = r['pairs']['a vs b']
    assert pair['status'] == 'ok', pair
    prefix = pair['prefix']
    assert prefix['label'] == '1-511', prefix
    assert prefix['n'] == 6 and prefix['position_hi'] == 511, prefix
    assert prefix['mean_contribution_to_total_log_ratio_at_max'] < 0.0, prefix
    for Ln in (512, 1024, 2048):
        total = pair['total_by_length'][str(Ln)]['mean']
        max_tok = int(by['a']['2048']['n_tokens_by_seed']['0'])
        n_tok = int(by['a'][str(Ln)]['n_tokens_by_seed']['0'])
        # Stored contributions use the maximum-length token denominator.
        expected = (
            prefix['mean_contribution_to_total_log_ratio_at_max']
            + sum(row['mean_contribution_to_total_log_ratio_at_max']
                  for row in pair['bands'].values())) * (max_tok / n_tok)
        assert math.isclose(total, expected, rel_tol=1e-12, abs_tol=1e-12), (
            Ln, total, expected)
    assert pair['dominant_band_at_max_length'] == '1-511', pair
    assert pair['max_reconstruction_error'] < 1e-12, pair
    json.dumps(r, allow_nan=False)


def test_absurd_integer_accounting_and_bad_pair_names_fail_closed():
    by = _panels(lengths=(512, 1024))
    by['a']['512']['nll_by_seed']['0'] = 10 ** 400
    try:
        L.length_position_decomposition(
            by, pairs=[{'name': 'a vs b', 'a': 'a', 'b': 'b'}])
    except ValueError as exc:
        assert 'number' in str(exc) or 'finite' in str(exc), exc
    else:
        raise AssertionError('an integer beyond the float range was accepted')

    by = _panels(lengths=(512, 1024))
    by['a']['512']['n_tokens_by_seed']['0'] = 10 ** 400
    try:
        L.length_position_decomposition(
            by, pairs=[{'name': 'a vs b', 'a': 'a', 'b': 'b'}])
    except ValueError as exc:
        assert 'representable' in str(exc), exc
    else:
        raise AssertionError('an unrepresentable token count was accepted')

    by = _panels(lengths=(512, 1024))
    try:
        L.length_position_decomposition(
            by, pairs=[{'name': 'bad', 'a': ['a'], 'b': 'b'}])
    except ValueError as exc:
        assert 'arm a' in str(exc), exc
    else:
        raise AssertionError('a non-string pair arm was accepted')


def test_zero_effect_comparison_stays_strict_json_and_replayable():
    # All paired deltas are exactly zero.  The by-length decomposition and the
    # comparison diagnostics must still be strict-JSON safe: a degenerate
    # floor must not be passed to the planning helper as n=0 (which returns
    # NaN) and then leak into the artifact.
    with tempfile.TemporaryDirectory() as td:
        st, p = _synthetic_stats(td)
        json.dumps(st, allow_nan=False)
        assert st['length_position']['status'] == 'ok', st['length_position']
        pair = st['length_position']['pairs']['a vs b']
        assert pair['prefix']['mean_contribution_to_total_log_ratio_at_max'] == 0.0, pair
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_length_position(path=p)
        assert out['status'] == 'ok', out


def test_exact_huge_seed_labels_are_labels_not_float_measurements():
    # Seed keys are exact identities; this module never feeds them to an RNG.
    # A label beyond MAX_SEED or 2**53 must survive, like the ranking readout.
    huge = 10 ** 400
    def cell(n, t):
        return {'ppl_by_seed': {str(huge): math.exp(n / t),
                                '0': math.exp(n / t)},
                'nll_by_seed': {str(huge): n, '0': n},
                'n_tokens_by_seed': {str(huge): t, '0': t}}
    by = {'a': {'512': cell(2048.0, 512), '1024': cell(4096.0, 1024)},
          'b': {'512': cell(2048.0, 512), '1024': cell(4100.0, 1024)}}
    r = L.length_position_decomposition(
        by, pairs=[{'name': 'a vs b', 'a': 'a', 'b': 'b'}])
    pair = r['pairs']['a vs b']
    assert pair['status'] == 'ok', pair
    assert huge in pair['seeds'] and huge in pair['prefix']['seeds'], pair
    json.dumps(r, allow_nan=False)
