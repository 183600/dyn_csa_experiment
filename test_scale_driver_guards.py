"""Scale/robustness checks for the driver and contrast-audit helpers.

These pin defects found while reviewing future scale-up paths:

* ``run_gapfill`` aggregated PPLs with ``statistics.stdev`` and only checked
  the result for finiteness.  For opposite finite extremes that call raises
  ``OverflowError`` before the guarded helper is ever reached.
* ``par.resolve_ctx`` materialised one slot per requested CPU worker, so a
  huge ``CSA_PAR`` value was a memory allocation instead of a clean refusal;
  malformed context metadata and payload cycles also escaped as raw errors.
* the single-factor audit formatted untrusted declaration values with
  ``repr``/``str`` and could hit Python 3.12's huge-integer limit, turning a
  malformed declaration into an exception rather than a ``fail`` verdict.

All checks are CPU-only and start no run.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403

import pytest

import par
import run_gapfill as G


def test_run_gapfill_extreme_spread_is_withheld_not_raised():
    assert G._safe_stdev([1.0, 2.0]) == pytest.approx(2 ** -0.5)
    assert G._safe_stdev([1.7e308, -1.7e308]) is None
    assert G._safe_stdev([1e308, 1.5e308]) == pytest.approx(
        3.535533905932738e307)
    assert G._safe_mean([10 ** 400, 10 ** 400]) is None


def test_run_gapfill_fold_recovers_a_finite_mean_from_an_overflowing_sum(tmp_path):
    live = {'full': {'seeds': [0], 'ppls': [1e308], 'n_seeds': 1,
                     'ppl_mean': 1e308, 'ppl_std': 0.0,
                     'ppl_primary': 1e308, 'nll_sum': 1e308,
                     'n_tokens': 1, 'ppl_curve': [[1, 1e308]]}}
    backup = {'full': {'seeds': [0, 1], 'ppls': [1e308, 1e308]}}
    (tmp_path / 'aggregate.json').write_text(json.dumps(live), encoding='utf-8')
    (tmp_path / 'aggregate.json.v6bak').write_text(
        json.dumps(backup), encoding='utf-8')
    G._fold_backup_into_aggregate(str(tmp_path))
    out = json.loads((tmp_path / 'aggregate.json').read_text(encoding='utf-8'))
    assert out['full']['seeds'] == [0, 1]
    assert out['full']['ppl_mean'] == 1e308
    assert out['full']['ppl_std'] == 0.0


def test_par_exact_integer_strings_keep_their_value():
    assert par._num_or_none('3') == 3
    assert par._num_or_none('3.0') == 3
    assert par._num_or_none(' 4 ') == 4
    huge = 10 ** 400
    assert par._num_or_none(str(huge)) == huge
    assert par._num_or_none('0') is None
    assert par._num_or_none('-1') is None


def test_par_refuses_an_unbounded_cpu_worker_count(monkeypatch):
    monkeypatch.setattr(par, 'gpu_slots', lambda *_a, **_k: ([], False))
    with pytest.raises(ValueError, match='safety limit'):
        par.resolve_ctx(str(par._MAX_CPU_WORKERS + 1))
    ctx = par.resolve_ctx('2')
    assert ctx['workers'] == 2 and ctx['slots'] == ['', '']


def test_par_ctx_active_treats_malformed_metadata_as_serial():
    assert par.ctx_active(None) is False
    assert par.ctx_active({'workers': 'not-a-number'}) is False
    assert par.ctx_active({'workers': 2}, n_cells='bad') is False
    assert par.ctx_active({'workers': 2}, n_cells=2) is True


def test_par_payload_normalisation_refuses_cycles_and_deep_nesting():
    cyclic = {}
    cyclic['self'] = cyclic
    with pytest.raises(ValueError, match='cyclic'):
        par._norm(cyclic)
    deep = {}
    cur = deep
    for _ in range(600):
        nxt = {}
        cur['x'] = nxt
        cur = nxt
    with pytest.raises(ValueError, match='nesting'):
        par._norm(deep)
    # A huge integer key is formatted through the bounded helper.
    assert par._norm({10 ** 10000: 1}) == {'<unrepresentable int>': 1}


def test_sweep_audit_turns_huge_malformed_values_into_a_fail_verdict():
    cases = (
        {10 ** 10000: {'factors': ['index_topk'], 'single_factor': True,
                       'a': 'csa_fix_topk8', 'b': 'csa_fixed'}},
        {'x': {'factors': ['warm_steps'], 'single_factor': True,
               'protocol_factor': 10 ** 10000,
               'a_level': 1, 'b_level': 2}},
        {'x': {'factors': ['warm_steps'], 'single_factor': True,
               'protocol_factor': 'warm_steps',
               'a_level': 10 ** 10000, 'b_level': 2}},
    )
    for decl in cases:
        rep = L.sweep_contrast_factor_audit(decl, verbose=False)
        json.dumps(rep, allow_nan=False)
    # The first declaration is a valid ordinary pair with an unprintable key;
    # the other two are invalid protocol declarations.
    assert L.sweep_contrast_factor_audit(cases[0], verbose=False)['status'] == 'ok'
    assert L.sweep_contrast_factor_audit(cases[1], verbose=False)['status'] == 'fail'
    assert L.sweep_contrast_factor_audit(cases[2], verbose=False)['status'] == 'fail'


def test_protocol_gate_and_metadata_attach_survive_malformed_containers():
    a = {'run_cfg': 'r', 'warm_steps': 5000}
    b = {'run_cfg': 'r', 'warm_steps': 0}
    assert 'object' in L.protocol_contrast_reason(a, b, 'not-a-declaration')
    mismatch = L.protocol_contrast_reason(
        a, b, {'protocol_factor': 'warm_steps', 'a_level': 1, 'b_level': 2})
    assert isinstance(mismatch, str) and 'do not match' in mismatch, mismatch
    import explib.rope as V7
    sentinel = ['not', 'a', 'mapping']
    assert V7.attach_contrast_factor_metadata(
        sentinel, V7.ROPE_CONTRAST_FACTORS, {'pairs': []}) is sentinel
    record = {'single_factor': True}
    L.attach_probe_factor_metadata(record, 'x', {'pairs': ['bad']})
    assert record['single_factor'] is False


def test_scale_ladder_unprintable_identity_counts_stay_json_safe():
    huge = 10 ** 10000
    base = {'steps': 1, 'batch_size': 1, 'seq_len': 1,
            'n_train_tokens': 1, 'lr': 1e-3, 'warmup': 0, 'seeds': [0]}
    for field in ('vocab', 'corpus_fp', 'fit_fp'):
        cfg = dict(base, **{field: huge})
        out = L.scale_ladder_audit([{'name': 'a', 'cfg': cfg, 'params': 1}])
        assert out['status'] == 'fail', (field, out)
        assert any(field in p and 'reportable integer bound' in p
                   for p in out['problems']), (field, out['problems'])
        json.dumps(out, allow_nan=False)
