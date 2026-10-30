"""Regression tests for the zero-GPU bug-fix pass.

Each test pins one concrete defect found by inspection:

* the aggregate audit gate did not recognise a child's singular
  ``warning``/``not_verifiable`` status or its non-empty ``warnings`` list;
* run-tracking verification crashed on a non-integer seed field instead of
  returning ``passed=False``;
* a constant non-zero paired panel emitted an infinite Cohen's d_z, which is
  not strict-JSON safe;
* the robust-location sensitivity check ignored the SIDE of the
  Hodges-Lehmann interval, so a positive raw effect with an entirely negative
  interval could still be reported as concordant;
* TOST decisions used ``alpha/2`` even when a stricter ``tost_conf`` was
  declared;
* the preprocessor audit re-fitted on a float64 view, losing exact int64 bits
  above 2**53 and marking a genuine train-only fit unverifiable;
* the full-width attention gate omitted the ``last_tok[topk_l]`` gather and
  its comparison temporary;
* an empty validation batch reached ``ndarray.min()`` and raised a raw NumPy
  error instead of the documented validation error.

These are CPU-only, zero-experiment checks.
"""
import json
import math
import os
import tempfile

import numpy as np

import audit_gates as A
import data_guard as DG
import exp_lib as L


def test_report_status_recognises_singular_warning_and_not_verifiable():
    rep = {
        'environment_identity': {'status': 'warning',
                                 'warnings': ['different stack'],
                                 'problems': []},
        'sensitivity': {'status': 'not_verifiable', 'problems': [],
                        'warnings': []},
        'clean': {'status': 'ok', 'warnings': [], 'problems': []},
    }
    out = A.audit_report_status(rep)
    assert out['status'] == 'warnings', out
    assert 'environment_identity' in out['gates_warned'], out
    assert 'sensitivity' in out['gates_not_verifiable_locally'], out
    assert out['exit_code'] == 0, out


def test_report_status_counts_child_warnings_even_without_problems():
    out = A.audit_report_status({
        'practical_significance': {
            'status': 'ok', 'problems': [],
            'warnings': ['confirmed but below the SESOI'],
        },
    })
    assert out['status'] == 'warnings', out
    assert out['gates_warned'] == ['practical_significance'], out


def test_verify_run_tracking_reports_malformed_seed_metadata():
    with tempfile.TemporaryDirectory() as td:
        cell = {
            'hyperparams': {
                'seed': 0, 'data_seed': 0,
                'rng': {'python': 0, 'numpy': 0, 'torch': 0, 'cuda': 0,
                        'tensorflow': 0, 'data': 0,
                        'dataloader_generator': 0},
            },
        }
        manifest = {'panels': {'cfg': {
            'planned_cells': ['a::seed0'],
            'cells': {'a::seed0': cell},
        }}}
        summary = {'a::seed0': {
            '_code': 'v1', 'ppl': 2.0,
            'hyperparams': {
                'seed': 0, 'data_seed': 0,
                'rng': {'python': 'bad', 'numpy': 0, 'torch': 0, 'cuda': 0,
                        'tensorflow': 0, 'data': 0,
                        'dataloader_generator': 0},
            },
        }}
        with open(os.path.join(td, 'run_manifest.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(manifest, f)
        with open(os.path.join(td, 'summary.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(summary, f)
        # The malformed field must be a recorded problem, not an exception
        # escaping the verifier (which would abort an archive audit).
        got = L.verify_run_tracking(td, run_cfg='cfg')
        assert got['passed'] is False, got
        assert any('seed metadata' in p for p in got['problems']), got
        json.dumps(got, allow_nan=False)


def test_constant_nonzero_paired_effect_has_no_infinite_dz():
    got = L.paired_effect_stats([1.0] * 6)
    assert got['cohens_dz'] is None, got
    assert got['std'] == 0.0 and got['mean'] == 1.0, got
    json.dumps(got, allow_nan=False)


def test_location_sensitivity_checks_interval_side_not_only_zero():
    # A few very large positive deltas carry the raw mean, while the robust
    # location interval is entirely negative: "excludes zero" is true on both
    # sides, but the directions conflict.
    deltas = [(-1.0 if i % 2 == 0 else -2.0) for i in range(30)] + [1000.0] * 6
    a = [2000.0 + x for x in deltas]
    b = [2000.0] * len(deltas)
    got = L.effect_sensitivity(a, b)
    assert got['raw']['direction'] == 'positive', got['raw']
    assert got['location']['direction'] == 'negative', got['location']
    assert got['location']['excludes_zero'] is True, got['location']
    assert got['concordance']['location_direction_agree'] is False, got
    assert got['status'] == 'location_discordant', got
    assert got['concordance']['all_agree'] is False, got


def test_tost_confidence_controls_the_equivalence_decision():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, 100)
    x = x - x.mean() + 0.001
    got = L.confirmation_practical_significance(
        list(x), margin=0.20, alpha=0.05, tost_conf=0.99,
        expected_sign='positive')
    # The old code used alpha/2 = 0.025 here even though the reported 99% CI
    # was wider than the margin.  Both one-sided p-values are > 0.005.
    assert math.isclose(got['alpha_one_sided'], 0.005, rel_tol=0.0,
                        abs_tol=1e-15), got
    assert got['equivalent'] is False, got
    assert got['status'] == 'indeterminate', got
    assert got['ci'][0] < -0.20 and got['ci'][1] > 0.20, got['ci']


def test_preprocessor_audit_preserves_exact_int64_fingerprint():
    X = np.array([[2 ** 53 + 1], [2 ** 53 + 3],
                  [2 ** 53 + 5], [2 ** 53 + 7]], dtype=np.int64)
    pre = DG.LeakageSafePreprocessor().fit(X)
    got = DG.audit_preprocessor_fit(pre, X)
    assert got['status'] == 'train_only_verified', got
    assert got['train_fp_match'] is True and got['stats_match_train'] is True


def test_full_width_gate_charges_the_indexer_gather_temporary():
    # n*k*8 bytes for last_tok[topk_l] plus the [n,k] comparison is not free;
    # this shape used to pass the 64 MiB gate and then allocate >100 MiB.
    assert L._attn_full_width_ok(171000, 32, 0, False) is False
    assert L._attn_full_width_ok(2048, 32, 128, False) is True
    assert L._attn_full_width_ok(128, 8, 16, True) is True


def test_error_analysis_rejects_empty_validation_batch_cleanly():
    class _Model:
        training = False

        def eval(self):
            return self

    with np.testing.assert_raises_regex(ValueError, 'non-empty'):
        L.analyze_errors(_Model(), np.empty((0, 4), dtype=np.int64), 8,
                         L.torch.device('cpu'))
    with np.testing.assert_raises_regex(ValueError, 'non-empty'):
        L.compare_token_errors(None, None, np.empty((0, 4), dtype=np.int64),
                               8, L.torch.device('cpu'))
