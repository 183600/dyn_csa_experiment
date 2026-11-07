"""Regression checks for ambiguous per-trial budgets in the tuning artifact.

A tuning trial is only comparable at one horizon.  The artifact records the
per-seed budgets in ``budget_by_seed`` and a convenience ``steps`` /
``tokens_seen`` at the trial level.  A resumed or partially completed search
must not let the convenience field claim one seed's horizon for the whole
trial.
"""

import json
import math
import os
import shutil
import tempfile

import exp_lib as L
import baseline_tune as BT


def test_common_budget_requires_one_unanimous_value():
    assert BT._common_budget(
        {'0': {'steps': 3}, '1': {'steps': 3}}, 'steps') == 3
    assert BT._common_budget(
        {'0': {'steps': 3}, '1': {'steps': 4}}, 'steps') is None
    # A missing value is not filled in from the seeds that happen to have it.
    assert BT._common_budget(
        {'0': {'steps': 3}, '1': {}}, 'steps') is None
    assert BT._common_budget({'0': {'steps': True}}, 'steps') is None
    # Fractional values are not silently truncated into a different horizon.
    assert BT._common_budget({'0': {'steps': 3.5}}, 'steps') is None
    assert BT._common_budget({'0': {'steps': '3'}}, 'steps') == 3
    assert BT._common_budget({}, 'steps') is None


def test_mixed_seed_horizons_do_not_claim_a_common_trial_budget(monkeypatch):
    outdir = tempfile.mkdtemp(prefix='tune_mixed_budget_')
    trials = [{'lr': 1e-4, 'weight_decay': 0.1, 'warmup': 2}]
    protocol = dict(L.BASELINE_TUNING_PROTOCOL, arms=['a'], trials=trials,
                    search_seeds=[0, 1], final_seeds=[0, 1])
    base = dict(L.RUN, d=8, n_layers=1, n_heads=1, d_head=8, seq_len=8,
                batch_size=1, n_train_tokens=32, steps=3, warmup=1)

    def fake_run(cfg, seeds=None, guard=None, label=None, **kw):
        os.makedirs(cfg['outdir'], exist_ok=True)
        summary = {}
        for variant in cfg['variants']:
            for seed in seeds:
                # Seed 1 was resumed with a longer horizon than seed 0.
                steps = cfg['steps'] + (1 if int(seed) == 1 else 0)
                summary[f'{variant}::seed{seed}'] = {
                    'variant': variant, 'seed': int(seed),
                    'ppl': 10.0 + 0.1 * int(seed),
                    'steps': steps, 'tokens_seen': steps * 8,
                    'params': 100, 'run_cfg': 'same',
                }
        with open(os.path.join(cfg['outdir'], 'summary.json'), 'w',
                  encoding='utf-8') as fh:
            json.dump(summary, fh)

    monkeypatch.setattr(L, 'run', fake_run)
    monkeypatch.setattr(L, 'overfit_guard', lambda *a, **k: None)
    try:
        artifact = BT.run_tuning(base, outdir=outdir, protocol=protocol,
                                 final=False)
    finally:
        shutil.rmtree(outdir, ignore_errors=True)

    row = artifact['arms']['a']['trials'][0]
    assert row['budget_by_seed'] == {'0': [3, 24], '1': [4, 32]}
    # The per-seed table exposes the disagreement; the trial-level field must
    # not silently promote seed 0's shorter horizon to the whole trial.
    assert row['steps'] is None
    assert row['tokens_seen'] is None


def test_selection_pooling_survives_unrepresentable_nll_and_token_totals():
    # record_subset_ppl() reports nll_sum=None when one seed's float NLL sum
    # overflows, and a token total above 10**308 cannot become a float.  The
    # tuning driver used to sum those None values (TypeError) or convert the
    # total with float() (OverflowError) before reaching _safe_exp_ppl.
    huge = 10 ** 400
    stats = {
        0: {'nll': 1.0, 'nll_sum': None, 'n_tokens': huge},
        1: {'nll': 2.0, 'nll_sum': None, 'n_tokens': 3 * huge},
    }
    ppl, total_nll, total_tok, seed_nll = BT._selection_pooled_metrics(stats)
    assert total_nll is None and total_tok == 4 * huge, (total_nll, total_tok)
    # Token-weighted exact mean CE = (1*1e400 + 2*3e400) / (4e400) = 7/4.
    assert math.isclose(math.log(ppl), 1.75, rel_tol=1e-15), ppl
    assert seed_nll == {'0': None, '1': None}, seed_nll

    # Finite per-seed NLL totals whose cross-seed sum overflows are recovered
    # from the ratio, not by dividing an infinity.
    stats = {
        0: {'nll': 1.0, 'nll_sum': 1e308, 'n_tokens': 10 ** 308},
        1: {'nll': 1.0, 'nll_sum': 1e308, 'n_tokens': 10 ** 308},
    }
    ppl, total_nll, total_tok, seed_nll = BT._selection_pooled_metrics(stats)
    assert total_nll is None and total_tok == 2 * 10 ** 308, (total_nll, total_tok)
    assert math.isclose(ppl, math.e, rel_tol=1e-12), ppl
    assert seed_nll == {'0': 1e308, '1': 1e308}, seed_nll

    # The ordinary finite path keeps the historical total and ratio.
    stats = {0: {'nll': 1.0, 'nll_sum': 3.0, 'n_tokens': 4},
             1: {'nll': 2.0, 'nll_sum': 6.0, 'n_tokens': 4}}
    ppl, total_nll, total_tok, seed_nll = BT._selection_pooled_metrics(stats)
    assert total_nll == 9.0 and total_tok == 8, (total_nll, total_tok)
    assert math.isclose(ppl, math.exp(9.0 / 8.0), rel_tol=1e-15), ppl
    assert seed_nll == {'0': 3.0, '1': 6.0}, seed_nll
