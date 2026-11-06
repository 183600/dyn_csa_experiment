"""Data-blind split-half consistency across evaluation units.

These checks cover the deterministic assignment, the exact one-sided
primitive and its intersection-union composition, the direction and
degenerate cases, the artifact/block replay gates, the v12 wiring and the
audit/CLI/Makefile registration.  No training, no split read, no frozen
p/Holm/confirmed change.
"""
import contextlib
import copy
import inspect
import io
import itertools
import json
import math
import os
import tempfile

import pytest

from test_support import *  # noqa: F401,F403

REPO = os.path.dirname(os.path.abspath(__file__))
SEEDS = list(range(12))
UNITS = [f'doc{i:03d}' for i in range(16)]
TOKENS = [100.0 + 7.0 * i for i in range(16)]


def _brute_one_sided(deltas, direction):
    """Independent float enumeration of the one-sided sign-flip p."""
    observed = sum(deltas)
    count = 0
    for signs in itertools.product((-1, 1), repeat=len(deltas)):
        total = sum(s * x for s, x in zip(signs, deltas))
        if direction == 'negative':
            if total <= observed + 1e-12:
                count += 1
        else:
            if total >= observed - 1e-12:
                count += 1
    return count / float(1 << len(deltas))


def _rows_uniform(per_unit):
    """Twelve identical seed rows from one per-unit effect vector."""
    return [list(per_unit) for _ in range(12)]


# ---------------------------------------------------------------------------
# assignment
# ---------------------------------------------------------------------------

def test_assignment_is_deterministic_and_outcome_blind():
    a = L.unit_split_assignment(UNITS, TOKENS)
    b = L.unit_split_assignment(UNITS, TOKENS)
    assert a['digest'] == b['digest']
    assert a['halves'] == b['halves']
    other = L.unit_split_assignment(UNITS, TOKENS, salt='another-salt')
    assert other['digest'] != a['digest']
    # The outcome is not an input: a wildly different effect vector cannot
    # change which units share a half.
    assert L.unit_split_assignment(UNITS, TOKENS, salt=a['salt'])['digest'] == a['digest']
    merged = a['halves'][0]['indices'] + a['halves'][1]['indices']
    assert sorted(merged) == list(range(len(UNITS)))
    assert set(a['halves'][0]['indices']).isdisjoint(a['halves'][1]['indices'])
    assert a['n_units'] == len(UNITS)


def test_assignment_token_balance_and_validations():
    tokens = [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0]
    units = [f'u{i}' for i in range(8)]
    a = L.unit_split_assignment(units, tokens)
    shares = [h['token_share'] for h in a['halves']]
    # Greedy token balance keeps the two halves comparable for this panel.
    assert max(shares) < 0.6
    assert sum(shares) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        L.unit_split_assignment(['a', 'a', 'b', 'c'], [1, 1, 1, 1])
    with pytest.raises(ValueError):
        L.unit_split_assignment(units, [1, 1, 0, 1, 1, 1, 1, 1])
    with pytest.raises(ValueError):
        L.unit_split_assignment(units, [1] * 7)
    with pytest.raises(ValueError):
        L.unit_split_assignment(['a', 'b', 'c', 'd'], [1, 1, 1, 1],
                                salt='   ')


# ---------------------------------------------------------------------------
# exact one-sided primitive and IUT
# ---------------------------------------------------------------------------

def test_one_sided_matches_bruteforce_and_direction_symmetry():
    rng = __import__('random').Random(3)
    d = [rng.uniform(-1.0, 1.0) for _ in range(8)]
    neg = L._usr_one_sided(d, 'negative')
    pos = L._usr_one_sided(d, 'positive')
    assert neg['p_one_sided'] == pytest.approx(_brute_one_sided(d, 'negative'),
                                               abs=1e-15)
    assert pos['p_one_sided'] == pytest.approx(_brute_one_sided(d, 'positive'),
                                               abs=1e-15)
    flipped = L._usr_one_sided([-x for x in d], 'positive')
    assert neg['p_one_sided'] == pytest.approx(flipped['p_one_sided'], abs=1e-15)


def test_iut_is_the_max_and_both_halves_claim():
    # All units improve: with 12 identical rows the one-sided exact p is the
    # resolution floor, so the IUT rejects and the claim is both-halves.
    result = L.unit_split_replication(_rows_uniform([-0.05] * 16), TOKENS,
                                      unit_ids=UNITS, seeds=SEEDS,
                                      direction='negative', n_salts=0)
    assert result['status'] in ('ok', 'warnings')
    p0 = result['halves'][0]['test']['p_one_sided']
    p1 = result['halves'][1]['test']['p_one_sided']
    assert result['iut']['p_iut'] == pytest.approx(max(p0, p1))
    assert result['iut']['p_iut'] < 0.05
    assert result['iut']['claim'] == 'both_halves_support_direction'
    assert result['sign_agreement'] == 'both_negative'
    # The positive direction on the same data must not reject.
    pos = L.unit_split_replication(_rows_uniform([-0.05] * 16), TOKENS,
                                   unit_ids=UNITS, seeds=SEEDS,
                                   direction='positive', n_salts=0)
    assert pos['iut']['p_iut'] > 0.9
    assert pos['iut']['claim'] != 'both_halves_support_direction'


def test_concentrated_effect_is_not_a_both_halves_claim():
    assignment = L.unit_split_assignment(UNITS, TOKENS)
    i0 = assignment['halves'][0]['indices']
    i1 = assignment['halves'][1]['indices']
    effects = [0.0] * len(UNITS)
    for i in i0:
        effects[i] = -0.20
    for i in i1:
        effects[i] = +0.20
    result = L.unit_split_replication(_rows_uniform(effects), TOKENS,
                                      unit_ids=UNITS, seeds=SEEDS,
                                      direction='negative', n_salts=0)
    assert result['sign_agreement'] == 'discordant'
    assert result['iut']['claim'] == 'halves_discordant'
    assert result['iut']['p_iut'] > 0.05
    # The paired half difference is large and its descriptive CI is ordered.
    diff = result['half_difference']
    assert diff['mean'] == pytest.approx(-0.40)
    assert diff['ci95'][0] <= diff['ci95'][1]


# ---------------------------------------------------------------------------
# cleaning, resolution and fail-closed behaviour
# ---------------------------------------------------------------------------

def test_non_finite_row_is_dropped_and_named_and_replays():
    rows = _rows_uniform([-0.05] * 16)
    rows[3] = [None] * 16
    block = L.unit_split_replication_block(rows, TOKENS, unit_ids=UNITS,
                                           seeds=SEEDS, direction='negative',
                                           n_salts=0)
    result = block['result']
    assert result['n_seeds'] == 11
    assert [d['seed'] for d in result['dropped_seeds']] == [3]
    # The stored inputs are strict JSON: a missing cell is null, never NaN.
    json.dumps(block['inputs'], allow_nan=False)
    assert any(x is None for x in block['inputs']['log_ratios'][3])
    verdict = L.validate_unit_split_replication_block(block)
    assert verdict['status'] in ('ok', 'warnings'), verdict
    assert verdict['replayed'] is True


def test_resolution_and_degenerate_panels():
    short = L.unit_split_replication(_rows_uniform([-0.05] * 16)[:4], TOKENS,
                                     unit_ids=UNITS, seeds=list(range(4)),
                                     direction='negative', n_salts=0)
    assert short['status'] == 'not_estimable'
    assert 'min_n' in short['reason']
    zeros = L.unit_split_replication(_rows_uniform([0.0] * 16), TOKENS,
                                     unit_ids=UNITS, seeds=SEEDS,
                                     direction='negative', n_salts=0)
    assert zeros['status'] in ('ok', 'warnings')
    assert zeros['iut']['p_iut'] == 1.0
    assert zeros['iut']['claim'] != 'both_halves_support_direction'
    nodir = L.unit_split_replication(_rows_uniform([-0.05] * 16), TOKENS,
                                     unit_ids=UNITS, seeds=SEEDS,
                                     direction=None)
    assert nodir['status'] == 'not_applicable'
    toomany = L.unit_split_replication(
        _rows_uniform([-0.05] * 16)[:6], TOKENS, unit_ids=UNITS,
        seeds=list(range(6)), direction='negative', max_n=4, n_salts=0)
    assert toomany['status'] == 'not_estimable'
    too_few_units = L.unit_split_replication(
        [[-0.05, -0.04] for _ in range(12)], [100.0, 100.0],
        unit_ids=['a', 'b'], seeds=SEEDS, direction='negative', n_salts=0)
    assert too_few_units['status'] == 'not_estimable'
    with pytest.raises(ValueError):
        L.unit_split_replication(_rows_uniform([-0.05] * 16), TOKENS,
                                 unit_ids=UNITS, seeds=[0] * 12,
                                 direction='negative')


# ---------------------------------------------------------------------------
# artifact / block replay
# ---------------------------------------------------------------------------

def test_block_and_artifact_replay_tamper_and_writer():
    block = L.unit_split_replication_block(_rows_uniform([-0.05] * 16), TOKENS,
                                           unit_ids=UNITS, seeds=SEEDS,
                                           direction='negative', n_salts=8)
    assert L.validate_unit_split_replication_block(block)['status'] in (
        'ok', 'warnings')
    bad = copy.deepcopy(block)
    bad['result']['iut']['p_iut'] = 0.9
    assert L.validate_unit_split_replication_block(bad)['status'] == 'fail'
    missing = copy.deepcopy(block)
    del missing['inputs']['salt']
    assert L.validate_unit_split_replication_block(missing)['status'] == 'fail'
    smuggled = copy.deepcopy(block)
    smuggled['result']['confirmed'] = True
    assert L.validate_unit_split_replication_block(smuggled)['status'] == 'fail'
    # The IUT must be the maximum of the components.
    bad2 = copy.deepcopy(block)
    bad2['result']['iut']['p_iut'] = min(
        bad2['result']['halves'][0]['test']['p_one_sided'],
        bad2['result']['halves'][1]['test']['p_one_sided']) - 0.001
    assert L.validate_unit_split_replication_block(bad2)['status'] == 'fail'
    artifact = L.build_unit_split_replication(_rows_uniform([-0.05] * 16),
                                              TOKENS, unit_ids=UNITS,
                                              seeds=SEEDS, direction='negative',
                                              n_salts=4, source='unit-test')
    assert L.validate_unit_split_replication(artifact)['status'] in ('ok', 'warnings')
    with tempfile.TemporaryDirectory() as td:
        assert L.write_unit_split_replication(td, artifact)['status'] == 'written'
        assert L.write_unit_split_replication(td, artifact)['status'] == 'noop'
        other = L.build_unit_split_replication(_rows_uniform([-0.03] * 16),
                                               TOKENS, unit_ids=UNITS,
                                               seeds=SEEDS, direction='negative',
                                               n_salts=0)
        assert L.write_unit_split_replication(td, other)['status'] == 'conflict'
        assert L.write_unit_split_replication(td, other, overwrite=True)['status'] == 'written'
        with open(os.path.join(td, 'unit_split_replication.json'),
                  encoding='utf-8') as f:
            stored = json.load(f)
        assert L.validate_unit_split_replication(stored)['status'] in (
            'ok', 'warnings')


def test_split_sensitivity_is_descriptive_and_capped():
    result = L.unit_split_replication(_rows_uniform([-0.05] * 16), TOKENS,
                                      unit_ids=UNITS, seeds=SEEDS,
                                      direction='negative', n_salts=16)
    sens = result['split_sensitivity']
    assert sens['n_salts_requested'] == 16
    assert sens['n_salts_used'] <= 16
    assert sens['n_estimable'] == sens['n_salts_used']
    assert 0.0 <= sens['pass_fraction'] <= 1.0
    assert 'not a p-value' in sens['role']
    if sens['passing_salts_sample']:
        assert sens['passing_salts_sample'][0]['salt'] == result['assignment']['salt']
        assert (sens['passing_salts_sample'][0]['assignment_digest']
                == result['assignment']['digest'])
    # A zero-salt request disables the curve rather than inventing one.
    off = L.unit_split_replication(_rows_uniform([-0.05] * 16), TOKENS,
                                   unit_ids=UNITS, seeds=SEEDS,
                                   direction='negative', n_salts=0)
    assert off['split_sensitivity'] is None


def test_markdown_renders_table_and_boundary():
    result = L.unit_split_replication(_rows_uniform([-0.05] * 16), TOKENS,
                                      unit_ids=UNITS, seeds=SEEDS,
                                      direction='negative', n_salts=2)
    md = L.unit_split_replication_markdown(result)
    assert 'Split-half consistency' in md
    assert 'both-halves IUT' in md
    assert 'not independent replication' in md
    missing = L.unit_split_replication_markdown(
        {'status': 'not_estimable', 'reason': 'no data'})
    assert 'not_estimable' in missing


# ---------------------------------------------------------------------------
# v12 wiring and audit gate
# ---------------------------------------------------------------------------

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


def _run_synthetic(td, *, n_units=10, spread=0.01):
    arms = ('a_arm', 'b_arm', 'c_arm')
    manifest = _synthetic_manifest(td, arms, range(8))
    outdir = os.path.join(td, 'res')
    os.makedirs(outdir)
    V12._check_or_write_freeze_lock(outdir, manifest, 'FP', stage='accessed')
    raw = {}
    for s in range(8):
        for k, arm in enumerate(arms):
            rec = _synthetic_record(arm, s, 50.0 + 1.0 * k + spread * s,
                                    manifest['manifest_sha256'], n_units=n_units)
            raw[f'{arm}::s{s}'] = rec
    bp = os.path.join(outdir, 'bylen.json')
    with open(bp, 'w', encoding='utf-8') as f:
        json.dump(raw, f)
    p = os.path.join(td, 'stats.json')
    with contextlib.redirect_stdout(io.StringIO()):
        stats = V12.finaltest_stats(
            manifest=manifest, bylen_path=bp,
            probe_path=os.path.join(td, 'no_probe.json'), out=p)
    return stats


def test_confirmation_stats_attach_split_half_block():
    with tempfile.TemporaryDirectory() as td:
        stats = _run_synthetic(td)
        comps = stats['comparisons']
        assert comps
        for name, row in comps.items():
            block = row.get('unit_split_replication')
            assert isinstance(block, dict), name
            assert block.get('kind') == 'unit_split_replication_block', name
            verdict = L.validate_unit_split_replication_block(block)
            assert verdict['status'] in ('ok', 'warnings'), (name, verdict)
            result = block['result']
            assert result['iut']['p_iut'] == pytest.approx(max(
                result['halves'][0]['test']['p_one_sided'],
                result['halves'][1]['test']['p_one_sided']))
            assert row['unit_split_replication_unit']['granularity'] == 'article'
            # The frozen decisions are untouched.
            assert row['p_exact_signflip'] == pytest.approx(
                row['p_exact_signflip'])
            assert 'confirmed' not in result
        md = V12._unit_split_replication_markdown(comps)
        text = '\n'.join(md)
        assert 'split-half' in text
        assert 'p_exact_signflip' in text
        dumped = json.dumps(
            {k: v.get('unit_split_replication') for k, v in comps.items()})
        assert '"p_holm":' not in dumped
        assert '"confirmed":' not in dumped


def test_audit_gate_embedded_standalone_error_and_tamper():
    block = L.unit_split_replication_block(_rows_uniform([-0.05] * 16), TOKENS,
                                           unit_ids=UNITS, seeds=SEEDS,
                                           direction='negative', n_salts=2)
    with tempfile.TemporaryDirectory() as td:
        stats = {'comparisons': {'x': {'unit_split_replication': block}}}
        p = os.path.join(td, 'stats.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(stats, f)
        ok = A.audit_unit_split_replication(path=p, verbose=False)
        assert ok['status'] in ('ok', 'warnings'), ok
        assert ok['n_blocks'] == 1
        bad = copy.deepcopy(stats)
        bad['comparisons']['x']['unit_split_replication']['result']['iut']['claim'] = 'both_halves_support_direction'
        bad['comparisons']['x']['unit_split_replication']['inputs']['salt'] = 'tampered'
        pbad = os.path.join(td, 'bad.json')
        with open(pbad, 'w', encoding='utf-8') as f:
            json.dump(bad, f)
        assert A.audit_unit_split_replication(path=pbad, verbose=False)['status'] == 'fail'
        legacy = os.path.join(td, 'legacy.json')
        with open(legacy, 'w', encoding='utf-8') as f:
            json.dump({'comparisons': {'x': {'mean': 1.0}}}, f)
        assert A.audit_unit_split_replication(path=legacy, verbose=False)['status'] == 'not_applicable'
        err = os.path.join(td, 'err.json')
        with open(err, 'w', encoding='utf-8') as f:
            json.dump({'comparisons': {'x': {'unit_split_replication': {
                'kind': 'unit_split_replication_error',
                'status': 'not_estimable', 'reason': 'no units'}}}}, f)
        assert A.audit_unit_split_replication(path=err, verbose=False)['status'] == 'not_verifiable_locally'
        missing = A.audit_unit_split_replication(
            path=os.path.join(td, 'missing.json'), verbose=False)
        assert missing['status'] == 'not_verifiable_locally'
        artifact = L.build_unit_split_replication(
            _rows_uniform([-0.05] * 16), TOKENS, unit_ids=UNITS, seeds=SEEDS,
            direction='negative', n_salts=0)
        pa = os.path.join(td, 'artifact.json')
        with open(pa, 'w', encoding='utf-8') as f:
            json.dump(artifact, f)
        assert A.audit_unit_split_replication(path=pa, verbose=False)['status'] in (
            'ok', 'warnings')


def test_registry_cli_makefile_wiring_and_protected_paths():
    assert 'part_88_unit_split_replication.py' in L._PARTS
    assert 'part_54_audit_unit_split_replication.py' in A._PARTS
    assert 'unit_split_replication' in inspect.getsource(A.run_report)
    cli = inspect.getsource(A._run_cli)
    for alias in ('unit_split_replication', 'split_half', 'unit_split',
                  'unit_halves', 'unit_split_replication_artifact'):
        assert alias in cli, alias
    with open(os.path.join(REPO, 'Makefile'), encoding='utf-8') as f:
        makefile = f.read()
    assert 'unit-split' in makefile
    assert 'unit_split_replication' in inspect.getsource(V12.finaltest_stats)
    assert '_unit_split_replication_markdown' in inspect.getsource(V12.build_report)
    assert os.path.isdir(os.path.join(REPO, '.github'))
    for name in ('README.md', 'README_zh.md'):
        assert os.path.isfile(os.path.join(REPO, name))
