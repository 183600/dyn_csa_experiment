"""Validation -> test effect transfer (descriptive, zero GPU).

The frozen confirmation rule consumes the *sign* of the validation estimate,
but the report never showed how much of that estimate survives on the held-out
corpus.  These checks build the transfer block from synthetic per-seed PPL
cells on the real frozen comparison family, verify known attenuation /
reversal / missing-seed cases, and confirm that the audit gate replays the
stored sufficient statistics and rejects tampering.

No training, no data split, no frozen p/Holm/verdict is changed.
"""
import json
import math
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

REPO = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(REPO, 'confirmation_frozen.json')


def _manifest():
    with open(MANIFEST, encoding='utf-8') as fh:
        return json.load(fh)


def _fixture(*, n=12, validation_shift=-0.02, test_shift=-0.01,
             seed=20261031):
    """Per-arm PPL cells on the real frozen family, with known shifts."""
    man = _manifest()
    rng = random.Random(seed)
    seeds = list(range(n))
    arms = man['arms']
    # One base log-PPL vector per arm, plus a per-arm level shift.  A
    # comparison's validation log effect is vshift[a]-vshift[b]; the test
    # effect is tshift[a]-tshift[b].  Both are therefore known before the
    # call, independent of the tiny per-seed noise.
    base = {arm: [4.0 + rng.uniform(-0.01, 0.01) for _ in seeds]
            for arm in arms}
    vshift = {arm: validation_shift * i for i, arm in enumerate(arms)}
    tshift = {arm: test_shift * i for i, arm in enumerate(arms)}
    val = {arm: [math.exp(base[arm][i] + vshift[arm]
                          + rng.uniform(-0.0005, 0.0005))
                 for i in range(n)] for arm in arms}
    test = {arm: [math.exp(base[arm][i] + tshift[arm]
                           + rng.uniform(-0.0005, 0.0005))
                  for i in range(n)] for arm in arms}
    power = {'kind': 'power_validation', 'comparisons': {}}
    test_stats = {'finaltest': {'comparisons': {}, 'by_length': {}}}
    for comp in man['primary_comparisons']:
        name, a, b = comp['name'], comp['a'], comp['b']
        power['comparisons'][name] = {
            'pilot_seeds': seeds,
            'pilot_ppl_a': list(val[a]),
            'pilot_ppl_b': list(val[b]),
            'pilot_deltas': [val[a][i] - val[b][i] for i in range(n)],
        }
        test_stats['finaltest']['comparisons'][name] = {
            'seeds': seeds,
            'per_seed': [test[a][i] - test[b][i] for i in range(n)],
        }
    for arm in arms:
        test_stats['finaltest']['by_length'].setdefault(arm, {})['4096'] = {
            'ppl_by_seed': {str(s): test[arm][i] for i, s in enumerate(seeds)},
        }
    return man, power, test_stats


def _block(**kwargs):
    man, power, test_stats = _fixture(**kwargs)
    return man, L.validation_transfer(power, test_stats, man)


def _write(path, block):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump({'validation_transfer': block}, fh, allow_nan=False)


def test_transfer_builds_known_attenuation_on_the_real_family():
    man, block = _block(validation_shift=-0.02, test_shift=-0.01)
    assert block['kind'] == 'validation_transfer'
    assert block['status'] == 'ok', block.get('reason')
    assert block['family_size'] == 6 and block['n_ok'] == 6
    assert block['frozen_decision_changed'] is False
    assert block['summary']['n_opposite_sign'] == 0
    assert block['summary']['n_attenuated'] == 6
    for name in block['family']:
        rec = block['comparisons'][name]
        assert rec['status'] == 'ok'
        assert rec['n'] == 12
        assert rec['classification'] == 'same_sign_attenuated'
        assert rec['coverage_ok'] is True
        _lo, _hi = rec['difference_ci95']
        assert _lo * _hi > 0.0  # the paired split difference excludes zero
        assert abs(rec['test_mean_log_ratio']) < abs(rec['validation_mean_log_ratio'])
        assert rec['test_over_validation_ratio'] == \
            __import__('pytest').approx(0.5, abs=0.05)
        assert rec['difference_p_exact_signflip'] <= rec['difference_p_holm']
        json.dumps(rec, allow_nan=False)
    json.dumps(block, allow_nan=False)
    rep = L.replay_validation_transfer(block, manifest=man)
    assert rep['status'] == 'ok', rep['problems']


def test_transfer_reversal_is_named_not_hidden():
    man, block = _block(validation_shift=-0.02, test_shift=0.01)
    assert block['status'] == 'ok'
    assert block['summary']['n_opposite_sign'] == 6
    for name in block['family']:
        rec = block['comparisons'][name]
        assert rec['classification'] == 'opposite_sign'
        assert rec['direction_agreement'] is False
    assert L.replay_validation_transfer(block, manifest=man)['status'] == 'ok'


def test_transfer_missing_seed_is_named_and_gate_warns():
    man, power, test_stats = _fixture()
    name = man['primary_comparisons'][0]['name']
    for key in ('pilot_seeds', 'pilot_ppl_a', 'pilot_ppl_b',
                'pilot_deltas'):
        power['comparisons'][name][key] = power['comparisons'][name][key][1:]
    block = L.validation_transfer(power, test_stats, man)
    rec = block['comparisons'][name]
    assert rec['status'] == 'ok'
    assert rec['n'] == 11
    assert rec['coverage_ok'] is False
    assert rec['missing_validation_seeds'] == [0]
    assert block['summary']['n_incomplete_coverage'] == 1
    assert L.replay_validation_transfer(block, manifest=man)['status'] == 'ok'
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'stats.json')
        _write(p, block)
        gate = A.audit_validation_transfer(p)
    assert gate['status'] == 'warn', gate
    assert gate['n_incomplete_coverage'] == 1
    assert gate['n_ok'] == 6


def test_transfer_refuses_validation_delta_ppl_mismatch():
    man, power, test_stats = _fixture()
    name = man['primary_comparisons'][0]['name']
    power['comparisons'][name]['pilot_deltas'][0] += 0.25
    block = L.validation_transfer(power, test_stats, man)
    rec = block['comparisons'][name]
    assert rec['status'] == 'not_estimable'
    assert 'pilot_deltas' in rec['reason']
    assert block['status'] == 'partial'
    assert block['n_not_estimable'] == 1
    assert L.replay_validation_transfer(block, manifest=man)['status'] == 'ok'


def test_transfer_refuses_test_per_seed_mismatch():
    man, power, test_stats = _fixture()
    name = man['primary_comparisons'][0]['name']
    test_stats['finaltest']['comparisons'][name]['per_seed'][0] += 0.25
    block = L.validation_transfer(power, test_stats, man)
    rec = block['comparisons'][name]
    assert rec['status'] == 'not_estimable'
    assert 'per_seed' in rec['reason']
    assert block['status'] == 'partial'


def test_transfer_refuses_missing_pilot_ppl_cells():
    man, power, test_stats = _fixture()
    name = man['primary_comparisons'][0]['name']
    power['comparisons'][name].pop('pilot_ppl_a')
    block = L.validation_transfer(power, test_stats, man)
    rec = block['comparisons'][name]
    assert rec['status'] == 'not_estimable'
    assert 'pilot_ppl_a' in rec['reason']


def test_transfer_replay_detects_tampering():
    man, block = _block()
    name = block['family'][0]
    for key, value in (
            ('difference_mean', 9.0),
            ('classification', 'same_sign_amplified'),
            ('difference_p_exact_signflip', 0.5),
            ('validation_mean_log_ratio', 1.0)):
        bad = json.loads(json.dumps(block))
        bad['comparisons'][name][key] = value
        rep = L.replay_validation_transfer(bad, manifest=man)
        assert rep['status'] == 'fail', (key, rep)
        assert any(key in p for p in rep['problems']), (key, rep['problems'])
    bad = json.loads(json.dumps(block))
    bad['n_ok'] = 0
    assert L.replay_validation_transfer(bad, manifest=man)['status'] == 'fail'
    bad = json.loads(json.dumps(block))
    bad['frozen_decision_changed'] = True
    assert L.replay_validation_transfer(bad, manifest=man)['status'] == 'fail'
    bad = json.loads(json.dumps(block))
    bad['family'] = bad['family'][:1]
    assert L.replay_validation_transfer(bad, manifest=man)['status'] == 'fail'


def test_transfer_gate_missing_or_absent_block_is_not_verifiable():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'stats.json')
        with open(p, 'w', encoding='utf-8') as fh:
            json.dump({}, fh)
        gate = A.audit_validation_transfer(p)
        assert gate['status'] == 'not_verifiable_locally', gate
        gate = A.audit_validation_transfer(os.path.join(td, 'nope.json'))
        assert gate['status'] == 'not_verifiable_locally', gate


def test_transfer_gate_fails_tampered_block():
    man, block = _block()
    name = block['family'][0]
    block['comparisons'][name]['difference_mean'] = 4.0
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'stats.json')
        _write(p, block)
        gate = A.audit_validation_transfer(p)
    assert gate['status'] == 'fail', gate
    assert any('difference_mean' in x for x in gate['problems'])


def test_transfer_is_wired_into_the_report_and_cli():
    with open(os.path.join(REPO, 'explib/confirmation/_parts/part_03_finaltest_stats.py'),
              encoding='utf-8') as fh:
        stats_src = fh.read()
    assert "'validation_transfer'" in stats_src
    assert "_validation_transfer_markdown(st.get('validation_transfer'))" in stats_src
    assert "pilot_ppl_a" in stats_src and "pilot_log_ratio" in stats_src
    with open(os.path.join(REPO, '_audit_gates/_parts/part_05_audit_confirmation_verdicts.py'),
              encoding='utf-8') as fh:
        gate_src = fh.read()
    assert "'validation_transfer': audit_validation_transfer()" in gate_src
    assert "'transfer': audit_validation_transfer" in gate_src
    with open(os.path.join(REPO, 'Makefile'), encoding='utf-8') as fh:
        make = fh.read()
    assert 'transfer:\n\t$(PYTHON) audit_gates.py validation_transfer' in make
    assert A.audit_validation_transfer is not None


def test_transfer_report_section_is_rendered_as_descriptive():
    man, block = _block()
    text = '\n'.join(V12._validation_transfer_markdown(block))
    assert '验证集→test 的效应迁移' in text
    assert '不属于冻结族' in text
    assert 'p_exact_signflip' in text and 'confirmed' in text
    assert '同向，test 衰减' in text
    missing = '\n'.join(V12._validation_transfer_markdown(
        {'status': 'not_estimable', 'reason': 'no power artifact'}))
    assert '不可估计' in missing and 'no power artifact' in missing


def test_transfer_refuses_a_validation_artifact_from_another_manifest():
    man, power, test_stats = _fixture()
    man['manifest_sha256'] = 'a' * 64
    power['manifest_sha256'] = 'b' * 64
    out = L.validation_transfer(power, test_stats, man)
    assert out['status'] == 'not_estimable'
    assert 'manifest_sha256' in out['reason']
    assert out['n_not_estimable'] == 6
    assert all(r['status'] == 'not_estimable'
               for r in out['comparisons'].values())
    assert L.replay_validation_transfer(out, manifest=man)['status'] == 'ok'
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'stats.json')
        _write(p, out)
        gate = A.audit_validation_transfer(p)
    assert gate['status'] == 'not_verifiable_locally', gate


def test_transfer_gate_accepts_the_missing_power_placeholder_as_absent():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'stats.json')
        _write(p, {'kind': 'validation_transfer', 'status': 'not_estimable',
                   'reason': 'power_validation.json does not exist',
                   'frozen_decision_changed': False})
        gate = A.audit_validation_transfer(p)
    assert gate['status'] == 'not_verifiable_locally', gate
    assert 'power_validation.json' in gate['reason']
