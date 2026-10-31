"""CPU checks for exact training-data exposure (window draws vs unique data).

The training loader samples causal windows with replacement, so
``tokens_seen`` is a draw count and adjacent windows overlap.  This suite pins
the exact seeded interval union, its agreement with ``batch_iter``'s RNG
stream, the explicit estimate fallback, the coverage-budget inverse, the audit
replay/tamper behaviour, and the CLI/report wiring.  It reads no corpus and no
test split.
"""
import copy
import contextlib
import inspect
import io
import json
import math
import os
import tempfile

from test_support import *  # noqa: F401,F403

import pytest as _pytest


def _brute_coverage(n_tokens, seq_len, batch_size, steps, seed=0, world_size=1,
                    data_seeds=None):
    n_last = n_tokens - seq_len - 1
    idx_dtype = np.int32 if (n_last + seq_len + 1) < 2 ** 31 else np.int64
    seeds = (data_seeds if data_seeds is not None
             else [L.data_seed_for_rank(seed, r) for r in range(world_size)])
    starts = []
    for s in seeds:
        rng = np.random.default_rng(int(s))
        for _ in range(int(steps)):
            starts.append(rng.integers(0, n_last + 1, size=int(batch_size),
                                       dtype=idx_dtype))
    starts = np.concatenate(starts)
    covered = np.zeros(int(n_tokens), dtype=bool)
    for start in starts:
        covered[int(start) + 1:int(start) + 1 + int(seq_len)] = True
    return starts, covered


def _write(path, obj):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh)


def _record(block, **extra):
    rec = {'variant': 'csa', 'seed': 0, 'steps': 10,
           'steps_completed': 10, 'tokens_seen': 10 * 4 * 16,
           'data_seed': 0, 'distributed_world_size': 1,
           'data_exposure': block}
    rec.update(extra)
    return rec


def _audit(obj):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'summary.json')
        _write(path, obj)
        with contextlib.redirect_stdout(io.StringIO()):
            return A.audit_data_exposure(path)


def test_policy_status_and_boundary_are_declared():
    assert L.TRAINING_EXPOSURE_POLICY['version'] == 'training-data-exposure-1'
    assert L.TRAINING_EXPOSURE_POLICY['schema'] == 'training_window_coverage_v1'
    assert L.TRAINING_EXPOSURE_STATUSES == ('ok', 'not_estimable')
    assert 'not a probability sample' in L.TRAINING_EXPOSURE_POLICY['claim_boundary']
    assert 'must not' in L.TRAINING_EXPOSURE_POLICY['claim_boundary']
    block = L.training_window_exposure(1000, 16, 4, 10, seed=0)
    assert block['kind'] == 'training_data_exposure'
    assert block['status'] == 'ok'
    assert block['policy'] == L.TRAINING_EXPOSURE_POLICY['version']
    assert not ({'confirmed', 'p_holm', 'verified', 'verdict',
                 'p_exact_signflip'} & set(block))
    plan = L.training_coverage_budget(1000, 16, 4, target_coverage=0.9)
    assert plan['kind'] == 'training_coverage_budget'
    assert plan['exact'] is False
    assert plan['status'] == 'ok'


def test_exact_union_matches_bruteforce_including_one_token_windows():
    cases = [
        dict(n_tokens=200, seq_len=8, batch_size=5, steps=6, seed=3),
        dict(n_tokens=50, seq_len=1, batch_size=3, steps=10, seed=0),
        dict(n_tokens=17, seq_len=16, batch_size=2, steps=4, seed=4),
        dict(n_tokens=100, seq_len=4, batch_size=1, steps=40, seed=11),
    ]
    for case in cases:
        block = L.training_window_exposure(**case)
        starts, covered = _brute_coverage(**case)
        assert block['exact'] is True
        assert block['coverage_quality'] == 'measured'
        assert block['unique_starts'] == int(np.unique(starts).size)
        assert block['unique_tokens'] == int(covered.sum())
        assert block['draws'] == case['steps'] * case['batch_size']
        assert block['processed_tokens'] == block['draws'] * case['seq_len']
        assert block['window_positions'] == case['n_tokens'] - case['seq_len']
        assert block['target_positions'] == case['n_tokens'] - 1
        assert 0.0 <= block['token_coverage'] <= 1.0
        assert block['unique_tokens'] <= block['target_positions']
        assert block['unique_starts'] <= block['window_positions']
        assert block['max_start_multiplicity'] >= 1
        assert math.isclose(block['start_coverage'],
                            block['unique_starts'] / block['window_positions'])
        assert math.isclose(block['token_coverage'],
                            block['unique_tokens'] / block['target_positions'])


def test_start_stream_matches_batch_iter_exactly():
    n_tokens, seq_len, batch_size, steps, seed = 100, 8, 4, 5, 7
    ids = np.arange(n_tokens, dtype=np.int64)
    it = L.batch_iter(ids, seq_len, batch_size, torch.device('cpu'), seed=seed)
    starts_from_loader = []
    for _ in range(steps):
        x, _y = next(it)
        starts_from_loader.extend(x[:, 0].tolist())
    n_last = n_tokens - seq_len - 1
    idx_dtype = np.int32 if (n_last + seq_len + 1) < 2 ** 31 else np.int64
    starts_from_helper = L._tx_draw_starts(
        n_last, steps, batch_size, [seed], idx_dtype).tolist()
    assert starts_from_loader == starts_from_helper
    block = L.training_window_exposure(n_tokens, seq_len, batch_size, steps,
                                       seed=seed)
    _, covered = _brute_coverage(n_tokens, seq_len, batch_size, steps, seed)
    assert block['unique_tokens'] == int(covered.sum())


def test_distributed_union_uses_per_rank_streams_and_is_replayable():
    case = dict(n_tokens=200, seq_len=8, batch_size=5, steps=6, seed=3)
    block = L.training_window_exposure(**case, world_size=3)
    starts, covered = _brute_coverage(**case, world_size=3)
    assert block['draws'] == 6 * 5 * 3
    assert block['unique_starts'] == int(np.unique(starts).size)
    assert block['unique_tokens'] == int(covered.sum())
    assert block['inputs']['world_size'] == 3
    assert block['inputs']['data_seeds'] == [
        L.data_seed_for_rank(3, r) for r in range(3)]
    replay = L.training_window_exposure(
        **case, world_size=3, data_seeds=block['inputs']['data_seeds'])
    assert replay['unique_tokens'] == block['unique_tokens']
    assert replay['unique_starts'] == block['unique_starts']


def test_estimate_fallback_is_explicit_and_strict_json():
    block = L.training_window_exposure(1000, 16, 4, 1000, seed=0,
                                       max_exact_draws=10)
    assert block['exact'] is False
    assert block['coverage_quality'] == 'estimated'
    assert block['max_start_multiplicity'] is None
    assert any('exceeds max_exact_draws' in w for w in block['warnings'])
    assert 0 <= block['unique_tokens'] <= block['target_positions']
    assert 0.0 <= block['token_coverage'] <= 1.0
    assert 0.0 <= block['duplicate_token_fraction'] <= 1.0
    json.dumps(block, allow_nan=False)
    # A huge draw count must not materialise a per-draw array.
    huge = L.training_window_exposure(10 ** 12, 4096, 1024, 10 ** 6,
                                      seed=2, max_exact_draws=1000)
    assert huge['exact'] is False
    assert huge['unique_tokens'] <= huge['target_positions']
    assert huge['unique_tokens'] <= huge['processed_tokens']
    json.dumps(huge, allow_nan=False)


def test_degenerate_corpus_and_invalid_inputs_fail_closed():
    block = L.training_window_exposure(17, 16, 2, 4, seed=0)
    assert block['window_positions'] == 1
    assert block['start_coverage'] == 1.0
    assert block['token_coverage'] == 1.0
    assert block['unique_tokens'] == 16
    assert block['unique_starts'] == 1
    bad = [
        dict(n_tokens=16, seq_len=16, batch_size=2, steps=1),
        dict(n_tokens=100, seq_len=0, batch_size=2, steps=1),
        dict(n_tokens=100, seq_len=8, batch_size=-1, steps=1),
        dict(n_tokens=100, seq_len=8, batch_size=2, steps=0),
        dict(n_tokens=100, seq_len=8, batch_size=2, steps=1, world_size=0),
        dict(n_tokens=True, seq_len=8, batch_size=2, steps=1),
        dict(n_tokens=100, seq_len=8.0, batch_size=2, steps=1),
        dict(n_tokens=100, seq_len=8, batch_size=2, steps=1,
             max_exact_draws=-1),
    ]
    for kwargs in bad:
        with _pytest.raises(ValueError):
            L.training_window_exposure(**kwargs)
    for q in (0, 1, -0.1, 1.1, float('nan'), True, '0.9'):
        with _pytest.raises(ValueError):
            L.training_coverage_budget(1000, 16, 4, target_coverage=q)


def test_budget_is_the_monotone_inverse_of_coverage():
    prev = None
    for q in (0.1, 0.5, 0.9, 0.99):
        plan = L.training_coverage_budget(1000, 16, 4, target_coverage=q)
        assert plan['expected_draws'] == plan['expected_steps'] * 4
        assert plan['expected_processed_tokens'] == (
            plan['expected_steps'] * 4 * 16)
        assert plan['expected_unique_fraction'] >= q - 1e-12
        assert plan['expected_unique_fraction'] <= q + 0.05
        if prev is not None:
            assert plan['expected_steps'] >= prev
        prev = plan['expected_steps']
    assert L.training_coverage_budget(
        1000, 16, 4, target_coverage=0.9, max_steps=1)['achievable'] is False
    plan = L.training_coverage_budget(1000, 16, 4, target_coverage=0.9,
                                      world_size=2)
    assert plan['expected_draws'] == plan['expected_steps'] * 4 * 2


def test_audit_replays_a_clean_block_and_catches_tampering():
    block = L.training_window_exposure(1000, 16, 4, 10, seed=0)
    clean = _audit({'runs': [_record(block)]})
    assert clean['status'] == 'ok'
    assert clean['n_replayed'] == 1
    assert clean['n_ok'] == 1
    assert not clean['problems']
    mutations = [
        ('unique_tokens', lambda b: b.__setitem__('unique_tokens', b['unique_tokens'] + 1)),
        ('processed_tokens', lambda b: b.__setitem__('processed_tokens', 1)),
        ('inputs.seed', lambda b: b['inputs'].__setitem__('seed', 1)),
        ('schema', lambda b: b.__setitem__('schema', 'other')),
        ('coverage_quality', lambda b: b.__setitem__('coverage_quality', 'bogus')),
        ('confirm field', lambda b: b.__setitem__('confirmed', True)),
    ]
    for label, mutate in mutations:
        bad = copy.deepcopy(block)
        mutate(bad)
        out = _audit({'runs': [_record(bad)]})
        assert out['status'] == 'fail', label
        assert out['problems'], label


def test_audit_cross_checks_the_surrounding_record():
    block = L.training_window_exposure(1000, 16, 4, 10, seed=0)
    out = _audit({'runs': [_record(block, tokens_seen=999)]})
    assert out['status'] == 'fail'
    assert any('tokens_seen' in p for p in out['problems'])
    out = _audit({'runs': [_record(block, data_seed=1)]})
    assert out['status'] == 'fail'
    out = _audit({'runs': [_record(block, distributed_world_size=2)]})
    assert out['status'] == 'fail'
    out = _audit({'runs': [_record(block, steps_completed=9)]})
    assert out['status'] == 'fail'


def test_audit_legacy_missing_block_and_malformed_artifact():
    legacy = {'runs': [{'variant': 'csa', 'seed': 0, 'steps': 10,
                        'tokens_seen': 640}]}
    out = _audit(legacy)
    assert out['status'] == 'not_verifiable_locally'
    assert out['n_legacy'] == 1
    assert any('legacy' in w for w in out['warnings'])
    out = _audit({'runs': []})
    assert out['status'] == 'not_verifiable_locally'
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'summary.json')
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write('{not json')
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_data_exposure(path)
        assert out['status'] == 'fail'
    block = {'kind': 'training_data_exposure', 'status': 'not_estimable',
             'reason': 'no corpus', 'schema':
             L.TRAINING_EXPOSURE_POLICY['schema']}
    out = _audit({'runs': [_record(block)]})
    assert out['status'] == 'not_verifiable_locally'
    assert out['n_not_estimable'] == 1
    assert any('not_estimable' in w for w in out['warnings'])


def test_audit_accepts_a_directory_and_never_reads_the_test_split(monkeypatch):
    block = L.training_window_exposure(1000, 16, 4, 10, seed=0)

    def _boom(*_a, **_k):
        raise AssertionError('data-exposure audit must not read the test split')

    monkeypatch.setattr(L, 'load_wikitest', _boom)
    with tempfile.TemporaryDirectory() as td:
        _write(os.path.join(td, 'summary.json'), {'runs': [_record(block)]})
        with contextlib.redirect_stdout(io.StringIO()):
            out = A.audit_data_exposure(td)
        assert out['status'] == 'ok'
        assert out['n_replayed'] == 1


def test_cli_and_report_wiring_is_present():
    assert 'data_exposure' in (A.__doc__ or '')
    src = inspect.getsource(A._run_cli)
    assert "'data_exposure': audit_data_exposure" in src
    assert "'tokens_seen': audit_data_exposure" in src
    report_src = inspect.getsource(A.run_report)
    assert "res['data_exposure'] = audit_data_exposure()" in report_src
    assert 'training_window_exposure' in inspect.getsource(L.train_variant)
    assert 'data_exposure' in inspect.getsource(L.train_variant)
