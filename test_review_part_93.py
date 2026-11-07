"""Exact sign-flip confidence set: the interval must agree with the exact p.

The frozen test is an exact sign-flip test, but the interval historically shown
beside it is a Student-t interval.  These checks cover the exact test
inversion, brute-force agreement, zero-membership consistency with the exact
p, shift/scale equivariance, resolution and degenerate panels, the artifact
and block replay gates, the v12 wiring, and the CLI/Makefile registration.
No training, no split read, no frozen p/Holm/confirmed change.
"""
import contextlib
import copy
import inspect
import io
import itertools
import json
import math
import os
import random
import tempfile

import pytest

from test_support import *  # noqa: F401,F403

REPO = os.path.dirname(os.path.abspath(__file__))

# A concrete 12-seed vector where the Student-t 95% interval excludes zero but
# the exact sign-flip p is 0.0518.  It motivates the exact set; it is a
# synthetic arithmetic example, not an experimental result.
DISAGREE12 = [
    -1.2795614472380512, 1.1845861854588706, 0.19932292240601032,
    0.90215805999999, 0.5116817221328843, 1.1546425466504586,
    0.3476453373498746, 0.5784130293189177, 0.24873614952275921,
    1.095111318193013, -0.39146819932337584, 1.0158470812014404,
]
SEEDS12 = list(range(12))


def _brute_p(d, mu):
    """Independent float enumeration of the two-sided sign-flip p at mu."""
    shifted = [float(x) - float(mu) for x in d]
    observed = abs(sum(shifted))
    count = 0
    for signs in itertools.product((-1, 1), repeat=len(shifted)):
        total = sum(s * x for s, x in zip(signs, shifted))
        if abs(total) >= observed - 1e-12:
            count += 1
    return count / float(1 << len(shifted))


def test_inversion_p_matches_bruteforce():
    d = [0.4, -0.25, 1.1, 0.05, -0.6, 0.9]
    for value in (0.0, 0.35, -1.2):
        got = L.exact_signflip_location_p(d, value)
        assert got['p'] == pytest.approx(_brute_p(d, value), abs=1e-15)
    assert L.exact_signflip_location_p(d, 0.0)['p'] == pytest.approx(
        L.exact_signflip_p(d)['p'], abs=1e-15)


def test_exact_ci_matches_bruteforce_acceptance_set():
    d = [0.4, -0.25, 1.1, 0.05, -0.6, 0.9]
    result = L.exact_signflip_ci(d)
    assert result['status'] in ('ok', 'warnings')
    lo, hi = result['ci']
    assert result['contains_zero'] == (result['p_at_zero'] >= 0.05)
    for mu in (lo - 0.2, lo, (lo + hi) / 2.0, hi, hi + 0.2):
        inside = any(a <= mu <= b for a, b in result['intervals'])
        assert inside == (_brute_p(d, mu) >= 0.05 - 1e-12), mu
    # At the endpoints the exact p is still above alpha; just outside it is not.
    assert _brute_p(d, lo) >= 0.05 - 1e-12
    assert _brute_p(d, lo - 1e-6) < 0.05 + 1e-12
    assert _brute_p(d, hi) >= 0.05 - 1e-12
    assert _brute_p(d, hi + 1e-6) < 0.05 + 1e-12


def test_exact_ci_cannot_disagree_with_exact_p_where_t_does():
    result = L.exact_signflip_ci(DISAGREE12, seeds=SEEDS12)
    exact_p = L.exact_signflip_p(DISAGREE12)['p']
    assert exact_p > 0.05
    assert result['p_at_zero'] == pytest.approx(exact_p, abs=1e-15)
    assert result['contains_zero'] is True
    t_ci = result['t_interval']['ci']
    assert not (t_ci[0] <= 0.0 <= t_ci[1])
    assert result['zero_relation']['relation'] == 't_excludes_exact_includes'
    assert any('anti-conservative' in w for w in result['warnings'])
    # The exact set must be the interval consistent with the exact test, and
    # the reported CI must contain the exact p-at-zero decision.
    assert result['ci'][0] <= 0.0 <= result['ci'][1]


def test_reported_float_endpoints_stay_inside_the_exact_acceptance_set():
    # ``float(Fraction)`` rounds to nearest.  The exact rational endpoint can
    # sit one ulp inside the interval, so a nearest-rounded lower/upper bound
    # can advertise a location whose exact sign-flip p is already below alpha.
    # DISAGREE12 used to report hi=0.891311359143354, where p=0.0498046875.
    cases = [
        (DISAGREE12, 0.05),
        ([-0.58, -1.03, -0.51, -2.4, 2.45], 0.20),
        ([0.4, -0.25, 1.1, 0.05, -0.6, 0.9, 0.15, -0.35], 0.10),
    ]
    for d, alpha in cases:
        result = L.exact_signflip_ci(d, alpha=alpha)
        assert result['unbounded'] is False, (d, alpha)
        for lo, hi in result['intervals']:
            p_lo = L.exact_signflip_location_p(d, lo)['p']
            p_hi = L.exact_signflip_location_p(d, hi)['p']
            assert p_lo is not None and p_lo >= alpha, (d, alpha, lo, p_lo)
            assert p_hi is not None and p_hi >= alpha, (d, alpha, hi, p_hi)


def test_shift_and_scale_equivariance():
    d = [0.4, -0.3, 1.2, 0.05, -0.8, 0.6, 0.2, -0.1, 0.9, -0.5, 0.3, 0.7]
    base = L.exact_signflip_ci(d)
    assert base['status'] in ('ok', 'warnings')
    for scale in (1e6, 1e-6):
        scaled = L.exact_signflip_ci([scale * x for x in d])
        assert scaled['ci'][0] == pytest.approx(scale * base['ci'][0],
                                               rel=1e-12, abs=1e-15)
        assert scaled['ci'][1] == pytest.approx(scale * base['ci'][1],
                                               rel=1e-12, abs=1e-15)
        # The exact p is a rational function of the supplied floats.  At an
        # exact tie the last mantissa bit of ``scale*x`` can add/remove a
        # sign pattern; the CI endpoints above stay scale-equivariant and the
        # mismatch is surfaced by p_consistency/warnings rather than hidden.
        assert scaled['p_at_zero'] == pytest.approx(base['p_at_zero'], abs=0.01)
    for shift in (2.5, -3.0):
        moved = L.exact_signflip_ci([x + shift for x in d])
        assert moved['ci'][0] == pytest.approx(base['ci'][0] + shift,
                                              abs=1e-12)
        assert moved['ci'][1] == pytest.approx(base['ci'][1] + shift,
                                              abs=1e-12)
        # Shifting moves the null: the p at zero of the moved vector is the
        # exact sign-flip p of the original vector at ``mu = shift``.
        assert moved['p_at_zero'] == pytest.approx(
            L.exact_signflip_location_p(d, shift)['p'], abs=1e-15)


def test_constant_panel_is_a_degenerate_point_not_precision():
    result = L.exact_signflip_ci([2.0] * 6)
    assert result['ci'] == [2.0, 2.0]
    assert result['degenerate'] is True
    assert result['status'] == 'warnings'
    assert any('single point' in w for w in result['warnings'])


def test_all_zero_panel_has_no_finite_interval():
    result = L.exact_signflip_ci([0.0] * 6)
    assert result['n_nonzero'] == 0
    assert result['resolution']['p_floor'] == 1.0
    assert result['resolution']['attainable'] is False
    assert result['unbounded'] is True
    assert result['ci'] is None
    assert result['contains_zero'] is True


def test_resolution_limited_panel_has_no_finite_ci():
    result = L.exact_signflip_ci([1.0, -0.5, 0.3, 0.2, -0.1])
    assert result['status'] == 'warnings'
    assert result['unbounded'] is True
    assert result['ci'] is None
    assert result['resolution']['attainable'] is False
    assert result['resolution']['min_n_for_alpha'] == 6
    assert any('resolution limit' in w for w in result['warnings'])


def test_zero_deltas_do_not_make_the_acceptance_set_unbounded():
    # A zero delta is sign-invariant at mu=0, so it lowers the test's
    # zero-null resolution (p_floor = 2/2**n_nonzero).  It is NOT
    # sign-invariant after centering at mu != 0: as |mu| grows only the
    # all-plus/all-minus patterns remain, so the far-tail floor is 2/2**n.
    # Unboundedness is decided by that tail floor.  Using n_nonzero here used
    # to report "whole real line" for this six-seed panel even though exact
    # p(mu) < 0.05 for mu < 0 or mu > 5.
    d = [5.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    result = L.exact_signflip_ci(d)
    assert result['status'] == 'ok'
    assert result['n_nonzero'] == 1
    assert result['resolution']['p_floor'] == 1.0
    assert result['resolution']['tail_p_floor'] == pytest.approx(2.0 / 64.0)
    assert result['resolution']['attainable'] is False
    assert result['resolution']['tail_attainable'] is True
    assert result['resolution']['ci_attainable'] is True
    assert result['unbounded'] is False and result['ci'] == [0.0, 5.0]
    assert result['contains_zero'] is True
    assert L.exact_signflip_location_p(d, -0.1)['p'] < 0.05
    assert L.exact_signflip_location_p(d, 5.1)['p'] < 0.05
    # The same decision must survive block replay: a block built from the
    # mixed-zero panel is valid, finite, and reproduces p_at_zero exactly.
    block = L.exact_signflip_ci_block(d, seeds=[0, 1, 2, 3, 4, 5])
    assert block['result']['unbounded'] is False
    verdict = L.validate_exact_signflip_ci_block(block)
    assert verdict['status'] in ('ok', 'warnings'), verdict
    assert block['result']['p_at_zero'] == pytest.approx(
        L.exact_signflip_p(d)['p'], abs=1e-12)
    # The all-zero panel keeps the documented conservative "no finite
    # interval" policy; it must not be confused with the mixed-zero case.
    zero = L.exact_signflip_ci([0.0] * 6)
    assert zero['unbounded'] is True and zero['ci'] is None


def test_invalid_and_large_inputs_fail_closed():
    for alpha in (0.0, 1.0, -0.1, True, 'x', float('nan')):
        with pytest.raises(ValueError):
            L.exact_signflip_ci([1.0, -1.0, 0.5, 0.2, -0.3, 0.7], alpha=alpha)
    for max_n in (1, 0, True, 19, '16'):
        with pytest.raises(ValueError):
            L.exact_signflip_ci([1.0, -1.0, 0.5, 0.2, -0.3, 0.7],
                                max_n=max_n)
    with pytest.raises(ValueError):
        L.exact_signflip_ci('not deltas')
    with pytest.raises(ValueError):
        L.exact_signflip_ci({'a': 1.0})
    with pytest.raises(ValueError):
        L.exact_signflip_ci([1.0, -1.0, 0.5], seeds=[0, 1])
    with pytest.raises(ValueError):
        L.exact_signflip_ci([1.0, -1.0, 0.5], seeds=[0, 0, 1])
    with pytest.raises(ValueError):
        L.exact_signflip_location_p([1.0, 2.0, 3.0], value=True)
    cap = L.exact_signflip_ci(list(range(1, 18)), max_n=16)
    assert cap['status'] == 'not_estimable'
    assert 'exceeds' in cap['reason']
    dropped = L.exact_signflip_ci(
        [1.0, float('nan'), 2.0, float('inf'), 3.0, 0.5])
    assert dropped['n'] == 4
    assert dropped['n_dropped'] == 2
    assert dropped['dropped_positions'] == [1, 3]
    assert dropped['status'] == 'warnings'


def test_unrepresentable_integers_fail_closed_without_formatting_crash():
    # CPython 3.12 refuses str()/repr()/json.dumps() on an integer with more
    # than 4300 decimal digits.  A future/scale caller can still hand the
    # public API an exact integer max_n or seed label, and an artifact is
    # untrusted input: both must produce a bounded diagnostic, not the
    # interpreter's integer-conversion error from inside an f-string.
    huge = 10 ** 10000
    for bad_max_n in (huge, -huge):
        with pytest.raises(ValueError, match='max_n'):
            L.exact_signflip_ci([1.0, 2.0, 3.0], max_n=bad_max_n)
    for seeds in ([huge, 1, 2], [1, 2, -huge]):
        with pytest.raises(ValueError, match='seeds'):
            L.exact_signflip_ci([1.0, 2.0, 3.0], seeds=seeds)

    artifact = L.build_exact_signflip_ci(DISAGREE12, seeds=SEEDS12)
    for path, value in ((('schema',), huge),
                        (('result', 'n'), huge),
                        (('inputs', 'max_n'), huge)):
        bad = copy.deepcopy(artifact)
        node = bad
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        # Hashing must survive the unrepresentable value so the validator can
        # name it; the strict-JSON and schema checks then fail closed.
        bad['content_sha256'] = L._esci_content_sha(bad)
        verdict = L.validate_exact_signflip_ci(bad, replay=False)
        assert verdict['status'] == 'fail', (path, verdict)
        json.dumps(verdict, allow_nan=False)

    # A structurally invalid interval must also fail without relying on the
    # content hash or on replay: an artifact can be re-hashed by an attacker.
    for bad_ci in ([1.0, 'x'], [2.0, 1.0], ['x', 'y'], [1.0]):
        bad = copy.deepcopy(artifact)
        bad['result']['ci'] = bad_ci
        bad['content_sha256'] = L._esci_content_sha(bad)
        verdict = L.validate_exact_signflip_ci(bad, replay=False)
        assert verdict['status'] == 'fail', (bad_ci, verdict)
        json.dumps(verdict, allow_nan=False)

    for where in ('artifact', 'inputs', 'result'):
        bad = copy.deepcopy(artifact)
        if where == 'artifact':
            bad[huge] = {'x': 1}
        elif where == 'inputs':
            bad['inputs'][huge] = 1
        else:
            bad['result'][huge] = 1
        bad['content_sha256'] = L._esci_content_sha(bad)
        verdict = L.validate_exact_signflip_ci(bad, replay=False)
        assert verdict['status'] == 'fail', (where, verdict)
        assert any('unrepresentable int' in problem
                   for problem in verdict['problems']), (where, verdict)
        json.dumps(verdict, allow_nan=False)


def test_artifact_roundtrip_hash_and_tamper():
    artifact = L.build_exact_signflip_ci(DISAGREE12, seeds=SEEDS12)
    good = L.validate_exact_signflip_ci(artifact)
    assert good['status'] in ('ok', 'warnings'), good
    assert isinstance(artifact['content_sha256'], str)
    assert len(artifact['content_sha256']) == 64
    # Change a result number: the content hash and the replay both fail.
    bad = copy.deepcopy(artifact)
    bad['result']['ci'][0] -= 0.01
    verdict = L.validate_exact_signflip_ci(bad)
    assert verdict['status'] == 'fail'
    assert any('content_sha256' in p for p in verdict['problems'])
    # A frozen-decision field is rejected even if the hash is refreshed.
    forbidden = copy.deepcopy(artifact)
    forbidden['result']['p_holm'] = 0.01
    forbidden['content_sha256'] = L._esci_content_sha(forbidden)
    verdict = L.validate_exact_signflip_ci(forbidden)
    assert verdict['status'] == 'fail'
    assert any('frozen-decision' in p for p in verdict['problems'])
    # An input edit that keeps the hash consistent is caught by replay.
    replay = copy.deepcopy(artifact)
    replay['inputs']['alpha'] = 0.1
    replay['content_sha256'] = L._esci_content_sha(replay)
    verdict = L.validate_exact_signflip_ci(replay)
    assert verdict['status'] == 'fail'
    assert any('replay' in p or 'differs' in p for p in verdict['problems'])


def test_block_roundtrip_and_replay():
    block = L.exact_signflip_ci_block(DISAGREE12, seeds=SEEDS12)
    assert L.validate_exact_signflip_ci_block(block)['status'] in ('ok', 'warnings')
    bad = copy.deepcopy(block)
    bad['result']['mean'] += 0.25
    verdict = L.validate_exact_signflip_ci_block(bad)
    assert verdict['status'] == 'fail'
    assert any('differs' in p for p in verdict['problems'])
    missing = copy.deepcopy(block)
    del missing['inputs']['alpha']
    assert L.validate_exact_signflip_ci_block(missing)['status'] == 'fail'
    # The block above the exact cap is a valid not_estimable block, not a fail.
    cap = L.exact_signflip_ci_block(list(range(1, 18)), max_n=16)
    assert L.validate_exact_signflip_ci_block(cap)['status'] == 'not_estimable'


def test_writer_noop_conflict_overwrite():
    first = L.build_exact_signflip_ci(DISAGREE12, seeds=SEEDS12)
    second = L.build_exact_signflip_ci(
        [0.3, -0.2, 0.9, 0.1, -0.5, 0.7, 0.2, -0.1, 0.8, -0.4, 0.25, 0.6],
        seeds=SEEDS12)
    with tempfile.TemporaryDirectory() as td:
        assert L.write_exact_signflip_ci(td, first)['status'] == 'written'
        assert L.write_exact_signflip_ci(td, first)['status'] == 'noop'
        assert L.write_exact_signflip_ci(td, second)['status'] == 'conflict'
        assert L.write_exact_signflip_ci(td, second, overwrite=True)['status'] == 'written'
        with open(os.path.join(td, 'exact_signflip_ci.json'), encoding='utf-8') as f:
            stored = json.load(f)
        assert L.validate_exact_signflip_ci(stored)['status'] in ('ok', 'warnings')


def test_markdown_renders_ci_and_boundary():
    result = L.exact_signflip_ci(DISAGREE12, seeds=SEEDS12)
    md = L.exact_signflip_ci_markdown(result)
    assert 'exact 95% set' in md
    assert 'Student-t interval' in md
    assert 'Boundary:' in md
    missing = L.exact_signflip_ci_markdown(
        {'status': 'not_estimable', 'reason': 'no data'})
    assert 'not estimable' in missing


def _synthetic_manifest(td, arms, seeds):
    m = {'manifest_version': 3, 'frozen_at': '2026-01-01',
         'arms': list(arms), 'seeds': list(seeds),
         'primary_comparisons': [
             {'name': f'{arms[0]} vs {a}', 'a': arms[0], 'b': a,
              'endpoint': 'by_len.ppl@4096', 'expected_sign': 'negative'}
             for a in arms[1:]],
         'statistics': {'test': 'exact two-sided sign-flip',
                        'correction': 'holm step-down',
                        'descriptive_only': True,
                        'require_complete_coverage': True,
                        'min_seeds_for_confirmation': 2}}
    p = os.path.join(td, 'm.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(m, f)
    return V12._load_manifest(p)


def _synthetic_record(arm, seed, ppl, msha, *, n_units=10):
    tokens = [100.0] * n_units
    per_unit_ppl = [ppl + 0.01 * i for i in range(n_units)]
    per_unit_nll = [math.log(v) * t for v, t in zip(per_unit_ppl, tokens)]
    pooled = math.exp(sum(per_unit_nll) / sum(tokens))
    return {'variant': arm, 'seed': seed, 'split': 'test', 'data_fp': 'FP',
            'by_len': {'4096': {
                'ppl': pooled, 'n_tok': int(sum(tokens)),
                'nll_sum': sum(per_unit_nll), 'truncated': False,
                'n_seq': n_units,
                'ppl_per_article': per_unit_ppl,
                'n_tokens_per_article': tokens,
                'nll_per_article': per_unit_nll,
                'article_ids': [f'a{i}' for i in range(n_units)]}},
            '_code': L.code_stamp(), 'manifest_sha256': msha}


def test_confirmation_stats_attach_exact_ci():
    arms = ('a_arm', 'b_arm', 'c_arm')
    with tempfile.TemporaryDirectory() as td:
        manifest = _synthetic_manifest(td, arms, range(8))
        outdir = os.path.join(td, 'res')
        os.makedirs(outdir)
        V12._check_or_write_freeze_lock(outdir, manifest, 'FP', stage='accessed')
        raw = {}
        for s in range(8):
            for k, arm in enumerate(arms):
                raw[f'{arm}::s{s}'] = _synthetic_record(
                    arm, s, 50.0 + 1.0 * k + 0.01 * s,
                    manifest['manifest_sha256'])
        bp = os.path.join(outdir, 'bylen.json')
        with open(bp, 'w', encoding='utf-8') as f:
            json.dump(raw, f)
        p = os.path.join(td, 'stats.json')
        with contextlib.redirect_stdout(io.StringIO()):
            stats = V12.finaltest_stats(
                manifest=manifest, bylen_path=bp,
                probe_path=os.path.join(td, 'no_probe.json'), out=p)
        comps = stats['comparisons']
        assert comps
        for name, row in comps.items():
            raw_block = row.get('exact_signflip_ci')
            log_block = row.get('exact_signflip_ci_log_ratio')
            assert isinstance(raw_block, dict), name
            assert isinstance(log_block, dict), name
            assert raw_block.get('kind') == 'exact_signflip_ci_block'
            assert L.validate_exact_signflip_ci_block(raw_block)['status'] in (
                'ok', 'warnings')
            result = raw_block['result']
            assert result['p_at_zero'] == pytest.approx(
                row['p_exact_signflip'], abs=1e-12)
            assert result['contains_zero'] == (result['p_at_zero'] >= 0.05)
        # The report renderer consumes the same blocks without a frozen field.
        md = V12._exact_signflip_ci_markdown(comps, alpha=0.05)
        text = '\n'.join(md)
        assert '精确符号翻转置信集' in text
        assert 'p_exact_signflip' in text
        assert 'p_holm' not in raw_block['result']


def test_audit_gate_embedded_standalone_and_tamper():
    block = L.exact_signflip_ci_block(DISAGREE12, seeds=SEEDS12)
    stats = {'comparisons': {'x': {'exact_signflip_ci': block}}}
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'stats.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(stats, f)
        ok = A.audit_exact_signflip_ci(path=p, verbose=False)
        assert ok['status'] in ('ok', 'warnings'), ok
        assert ok['n_blocks'] == 1
        assert ok['n_t_exact_disagreements'] == 1
        bad = copy.deepcopy(stats)
        bad['comparisons']['x']['exact_signflip_ci']['result']['ci'][0] -= 0.1
        pbad = os.path.join(td, 'bad.json')
        with open(pbad, 'w', encoding='utf-8') as f:
            json.dump(bad, f)
        assert A.audit_exact_signflip_ci(path=pbad, verbose=False)['status'] == 'fail'
        none = os.path.join(td, 'none.json')
        with open(none, 'w', encoding='utf-8') as f:
            json.dump({'comparisons': {'x': {'mean': 1.0}}}, f)
        assert A.audit_exact_signflip_ci(path=none, verbose=False)['status'] == 'not_applicable'
        malformed = os.path.join(td, 'malformed.json')
        with open(malformed, 'w', encoding='utf-8') as f:
            json.dump({'comparisons': {'x': {'exact_signflip_ci': 'not a block'}}}, f)
        assert A.audit_exact_signflip_ci(path=malformed, verbose=False)['status'] == 'fail'
        missing = A.audit_exact_signflip_ci(
            path=os.path.join(td, 'missing.json'), verbose=False)
        assert missing['status'] == 'not_verifiable_locally'
        artifact = L.build_exact_signflip_ci(DISAGREE12, seeds=SEEDS12)
        pa = os.path.join(td, 'artifact.json')
        with open(pa, 'w', encoding='utf-8') as f:
            json.dump(artifact, f)
        assert A.audit_exact_signflip_ci(path=pa, verbose=False)['status'] in (
            'ok', 'warnings')


def test_registry_cli_makefile_wiring():
    assert 'part_87_exact_signflip_ci.py' in L._PARTS
    assert 'part_53_audit_exact_signflip_ci.py' in A._PARTS
    assert 'exact_signflip_ci' in inspect.getsource(A.run_report)
    cli = inspect.getsource(A._run_cli)
    for alias in ('exact_signflip_ci', 'exact_ci', 'signflip_ci', 'seed_ci'):
        assert alias in cli, alias
    with open(os.path.join(REPO, 'Makefile'), encoding='utf-8') as f:
        makefile = f.read()
    assert 'exact-signflip-ci' in makefile
    assert 'exact_signflip_ci' in inspect.getsource(V12.finaltest_stats)
    assert '_exact_signflip_ci_markdown' in inspect.getsource(V12.build_report)
    # The protected paths were not moved by this work.
    assert os.path.isdir(os.path.join(REPO, '.github'))
    assert os.path.isfile(os.path.join(REPO, 'README.md'))
    assert os.path.isfile(os.path.join(REPO, 'README_zh.md'))


def test_exact_ci_coverage_smoke_under_symmetry():
    rng = random.Random(7)
    n = 6
    true_mean = 0.5
    covered = 0
    trials = 300
    for _ in range(trials):
        d = [true_mean + rng.gauss(0.0, 1.0) for _ in range(n)]
        result = L.exact_signflip_ci(d)
        lo, hi = result['ci']
        if lo <= true_mean <= hi:
            covered += 1
    # Exact test inversion should be at least nominal; with 300 draws the
    # Monte-Carlo error is small, so 0.90 is a conservative smoke bound.
    assert covered / trials >= 0.90
