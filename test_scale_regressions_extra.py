"""Regression checks for extreme-scale guards and the fusion control ladder.

These are zero-GPU checks.  They cover paths where a finite input can produce a
non-finite report value, a design gate that must propagate an upstream
rejection, or a deterministic placebo that must not depend on uninitialised
memory or on the expected norm of a Gaussian vector.
"""
import json
import math
import os
import sys as _sys

import numpy as np
import torch

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_support import *  # noqa: F401,F403


def test_initialization_audit_propagates_definition_site_failure():
    # No declared intervention keys: the definition-site audit must refuse this
    # contrast, and the initialization report must not call it "ok" merely
    # because no pair reached the parameter comparison loop.
    design = {'content_alignment': [('csa_fixed', 'csa_dynamic')]}
    rep = L.single_factor_initialization_audit(design, verbose=False)
    assert rep['design_status'] == 'fail', rep
    assert rep['status'] == 'fail', rep
    assert any('definition-site' in v for v in rep['violations']), rep


def test_random_fuse_kernel_is_deterministic_and_magnitude_matched():
    class _BadWeight:
        def detach(self):
            raise RuntimeError('force the deterministic fallback')

    w1 = L._frozen_random_fuse_weight(_BadWeight(), 8, 3, torch.float32)
    w2 = L._frozen_random_fuse_weight(_BadWeight(), 8, 3, torch.float32)
    assert torch.equal(w1, w2), 'the fallback placebo must not read uninitialised memory'
    norms = w1.reshape(8, -1).norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), norms

    # A hashed kernel has the same per-channel radius as the identity kernel
    # used by the learned treatment, for every kernel width (including 1).
    for kernel in (1, 2, 5):
        w = L._frozen_random_fuse_weight(torch.randn(16, 16), 8, kernel,
                                         torch.float32)
        norms = w.reshape(8, -1).norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), (
            kernel, norms)


def test_paired_effect_stats_withholds_unrepresentable_sample_sd():
    # The true sample SD is sqrt(2)*1.7e308, beyond float64 even though every
    # delta is finite.  The mean and the exact test survive; the SD-derived
    # fields become unavailable instead of infinite.
    rep = L.paired_effect_stats([1.7e308, -1.7e308])
    assert rep['std'] is None and rep['std_available'] is False, rep
    assert rep['sem'] is None and rep['cohens_dz'] is None, rep
    assert math.isfinite(rep['mean']) and math.isfinite(rep['p_exact_signflip']), rep
    for key in ('std', 'sem', 'cohens_dz'):
        assert rep[key] is None or math.isfinite(rep[key]), rep


def test_paired_delta_token_pooled_mean_is_finite_for_extreme_nll():
    # A-B itself overflows the row sum, but the token-pooled per-token mean is
    # representable.  The wider path must recover it; ordinary input stays on
    # the historical NumPy path.
    A = np.array([[1e308, 1.5e308], [1.2e308, 1.7e308]])
    B = np.array([[1e307, 1.5e307], [1.2e307, 1.7e307]])
    rep = L.paired_delta_variance_components(A, B, np.array([10.0, 20.0]))
    assert math.isfinite(rep['token_pooled_delta_nll_per_token']), rep
    json.dumps(rep, allow_nan=False)

    ordinary_a = np.array([[10.0, 20.0], [11.0, 21.0]])
    ordinary_b = np.array([[12.0, 22.0], [13.0, 23.0]])
    tok = np.array([10.0, 20.0])
    got = L.paired_delta_variance_components(ordinary_a, ordinary_b, tok)
    expected = float(np.mean(np.sum(ordinary_a - ordinary_b, axis=1)
                             / np.sum(tok)))
    assert math.isclose(got['token_pooled_delta_nll_per_token'], expected,
                        rel_tol=1e-12)


def test_scale_ladder_unrepresentable_budget_ratio_is_not_infinite():
    def tier(name, steps, params):
        return {'name': name,
                'cfg': {'steps': steps, 'batch_size': 1, 'seq_len': 1,
                        'n_train_tokens': steps, 'lr': 3e-4, 'warmup': 0,
                        'seeds': [0, 1]},
                'params': params}

    out = L.scale_ladder_audit([tier('S1', 1, 10 ** 300),
                                tier('S2', 10 ** 300, 1)])
    assert out['tokens_per_param_ratio'] is None, out
    assert out['budget_matched'] is False, out
    assert any('representable float range' in w for w in out['warnings']), out
    json.dumps(out, allow_nan=False)


def test_meta_analysis_opposite_extreme_effects_is_json_safe():
    points = [
        {'label': 'a', 'scale': 1.0, 'seeds': [0, 1],
         'effects': [1.7e308, -1.7e308]},
        {'label': 'b', 'scale': 2.0, 'seeds': [0, 1],
         'effects': [1.6e308, -1.6e308]},
    ]
    rep = L.meta_analyze_effects(points, min_seeds=2)
    for row in rep['per_point']:
        for key in ('std', 'se'):
            assert row[key] is None or math.isfinite(row[key]), (row, key)
        assert row['ci'] is None or all(math.isfinite(v) for v in row['ci']), row
    json.dumps(rep, allow_nan=False)


def test_unit_precision_design_reorders_an_overflowing_ci_product():
    base = [-1.0, -0.8, -1.2, -0.9, -1.1, -0.95]
    rep = L.unit_precision_design(
        10000, pilot_unit_deltas=[x * 1e307 for x in base])
    assert rep['status'] == 'ok', rep
    for key in ('ci_halfwidth', 'ci95_halfwidth'):
        assert math.isfinite(rep['precision'][key]), (key, rep)
    assert math.isfinite(rep['mde']['absolute']), rep
