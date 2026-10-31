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
  error instead of the documented validation error;
* the beyond-SESOI superiority test used the TOST confidence level instead of
  the declared ``alpha/2``, so a looser ``tost_conf`` could create a false
  beyond-margin claim and a stricter one could suppress a true claim;
* the length audit read a large integral count through float64, rounding
  ``2**53 + 1`` to ``2**53`` and letting a malformed ``n_tok`` equal the
  ``n_seq*(L-1)`` formula it was supposed to fail;
* the overflow-safe pooled-NLL ratio took a token-weighted mean of the
  normalised NLLs instead of the ratio of sums, changing PPL whenever record
  lengths/rates differ;
* event counters, metadata counts, compute-control medians and reproduction
  counts were still round-tripped through float64, and an exact ``Fraction``
  count beyond float64 could raise ``OverflowError`` instead of comparing;
* the content fingerprint hashed the raw NaN payload bits of native float
  arrays, while duplicate detection (and the object/string paths) treat every
  NaN spelling as one missing-like value, so the same table could fingerprint
  differently by storage dtype and payload alone;
* the exact training-exposure replay buffered one array per step before
  concatenating, so with ``batch_size=1`` the draw cap did not bound memory
  (a permitted 20M-draw replay could hold 20M tiny ndarrays).

These are CPU-only, zero-experiment checks.
"""
import contextlib
import io
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


def test_beyond_sesoi_uses_superiority_level_not_tost_conf():
    margin = math.log(1.01)
    # p_beyond is ~0.0486: above the documented alpha/2=0.025 superiority
    # level but below the one-sided level implied by tost_conf=0.90.  The
    # latter must not create a beyond-margin claim.
    below = [3.0, 3.0] + [-3.0] * 7
    for conf in (None, 0.90, 0.99):
        got = L.confirmation_practical_significance(
            below, margin=margin, alpha=0.05, tost_conf=conf,
            expected_sign='negative')
        assert math.isclose(got['superiority_alpha_one_sided'], 0.025,
                            rel_tol=0.0, abs_tol=1e-15), got
        assert got['p_beyond'] > 0.025, got
        assert got['beyond_sesoi'] is False, got
        assert got['status'] != 'beyond_sesoi', got
    # p_beyond is ~0.0152: below alpha/2=0.025 but above the 0.005 level
    # implied by tost_conf=0.99.  A strict equivalence confidence must not
    # suppress a superiority claim that clears its own declared level.
    behind = [2.0] + [-4.0] * 5
    for conf in (None, 0.90, 0.99):
        got = L.confirmation_practical_significance(
            behind, margin=margin, alpha=0.05, tost_conf=conf,
            expected_sign='negative')
        assert got['p_beyond'] < 0.025, got
        assert got['beyond_sesoi'] is True, got
        assert got['status'] == 'beyond_sesoi', got
        assert got['equivalent'] is False, got
        # alpha_one_sided remains the TOST decision level, so the two
        # thresholds are not confused when tost_conf is non-default.
        expected_tost = 0.025 if conf is None else (1.0 - conf) / 2.0
        assert math.isclose(got['alpha_one_sided'], expected_tost,
                            rel_tol=0.0, abs_tol=1e-15), got
    # The mirrored positive-direction panel exercises beyond_worse with the
    # same threshold rule.
    got = L.confirmation_practical_significance(
        [-2.0] + [4.0] * 5, margin=margin, alpha=0.05, tost_conf=0.99,
        expected_sign='positive')
    assert got['p_beyond'] < 0.025 and got['beyond_sesoi'] is True, got


def test_length_audit_reads_large_counts_without_float64_rounding():
    big = 2 ** 53
    assert A._exact_count(big + 1) == big + 1
    assert A._exact_count(np.int64(big + 1)) == big + 1
    # Numeric strings are still not measurements, and non-integral floats are
    # still refused.
    assert A._exact_count(str(big + 1)) is None
    assert A._exact_count(1000.5) is None
    raw = {'arm': {'by_len': {'2': {
        'n_tok': big + 1, 'n_seq': big, 'eval_span': 1,
        'truncated': False}}}}
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'bylen.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(raw, f)
        with contextlib.redirect_stdout(io.StringIO()):
            got = A.audit_length(path)
    # The extra target token must make the n_seq*(L-1) identity fail; rounding
    # the count to 2**53 used to make this malformed cell pass as `ok`.
    assert got['status'] == 'fail', got
    assert any('n_tok' in p and 'n_seq*(L-1)' in p for p in got['problems']), got


def test_pooled_nll_ratio_preserves_unequal_token_weights():
    # The overflow fallback used to take a token-weighted mean of the
    # normalised NLLs; that only equals sum(nll)/sum(tokens) when every
    # record has the same NLL-per-token ratio.  Ordinary and scale paths must
    # both keep the ratio of sums.
    assert math.isclose(L._pooled_nll_ratio([1.0, 2.0], [10, 30]),
                        3.0 / 40.0, rel_tol=1e-15)
    pooled, nll_total, tok_total = L._pooled_ppl_from_records(
        [{'nll_sum': 1.0, 'n_tokens': 10},
         {'nll_sum': 2.0, 'n_tokens': 30}])
    assert math.isclose(pooled, math.exp(0.075), rel_tol=1e-15), pooled
    assert nll_total == 3.0 and tok_total == 40, (nll_total, tok_total)
    # Same check through the overflow fallback: the NLL total is inf and the
    # exact token total is beyond float64, but the represented ratio is 0.2.
    huge = 10 ** 308
    recs = [{'nll_sum': 1.5e308, 'n_tokens': huge},
            {'nll_sum': 0.5e308, 'n_tokens': 9 * huge}]
    pooled, nll_total, tok_total = L._pooled_ppl_from_records(recs)
    assert math.isclose(pooled, math.exp(0.2), rel_tol=1e-15), pooled
    assert nll_total is None, nll_total
    assert tok_total == 10 * huge, tok_total
    # record_subset_ppl shares the same overflow-safe ratio.
    subset = L.record_subset_ppl(
        {'nll_per_row': [1.5e308, 0.5e308],
         'n_tokens_per_row': [huge, 9 * huge]}, [0, 1])
    assert subset is not None and math.isclose(
        subset['nll'], 0.2, rel_tol=1e-15), subset


def test_subset_ppl_keeps_exact_token_denominator_at_scale():
    big = 2 ** 53 + 1
    rec = {'nll_per_row': [1.0, 2.0],
           'n_tokens_per_row': [big, big]}
    got = L.record_subset_ppl(rec, [0, 1])
    assert got['n_tokens'] == 2 * big, got
    assert math.isclose(got['nll'], 3.0 / (2 * big), rel_tol=1e-15), got
    # A token total beyond float64 is still an exact denominator and the
    # represented per-token ratio is not fabricated as zero by an intermediate
    # float conversion.
    from fractions import Fraction
    huge = 10 ** 400
    got = L.record_subset_ppl(
        {'nll_per_row': [1e300, 2e300],
         'n_tokens_per_row': [huge, huge]}, [0, 1])
    assert got['n_tokens'] == 2 * huge, got
    expected = float(Fraction(3 * 10 ** 300, 2 * huge))
    assert math.isclose(got['nll'], expected, rel_tol=1e-12), got


def test_event_metadata_and_config_counts_stay_exact_at_scale():
    big = 2 ** 53 + 1
    diag = L.diagnose_training_instability({
        'opt_diag_summary': {'nonfinite_loss_steps': big,
                             'nonfinite_grad_steps': 0,
                             'amp_scaler_overflows': 0, 'events': []}})
    evidence = [c['evidence'] for c in diag['checks']
                if c['check'] == 'forward_loss_numerics'][0]
    assert str(big) in evidence, evidence
    assert A._unit_meta_int({'vocab': big}, 'vocab') == big
    # Numeric-string metadata keeps its historical acceptance; a fractional
    # string is still not an integer.
    assert A._unit_meta_int({'vocab': str(big)}, 'vocab') == big
    assert A._unit_meta_int({'vocab': '3.5'}, 'vocab') is None
    # A huge exact parameter count is not rounded when the compute-control
    # median is formed.
    recs = {('a', 't', 0): {'params': big, 'tokens_seen': 10},
            ('b', 't', 0): {'params': big, 'tokens_seen': 10}}
    detail = L._compute_control_detail(recs, 'a', 'b')
    assert detail['params'] == {'a': big, 'b': big}, detail['params']


def test_exact_rational_counts_do_not_round_or_raise():
    from decimal import Decimal
    from fractions import Fraction
    huge = 10 ** 400
    assert L._repro_int(Fraction(huge, 1)) == huge
    assert L._repro_int(Decimal(huge)) == huge
    assert L._repro_int(Decimal('3.5')) is None
    assert L._integral_value(Decimal(huge)) == huge
    assert L._integral_value(Fraction(huge, 1)) == huge
    # Exact capacity verdict survives a pair whose float relative gap
    # underflows; a huge integer pair with one extra parameter is still
    # distinguished exactly.
    gap, within = L._param_ratio_pair(huge, huge + 1, 1e-9)
    assert gap is None and within is True, (gap, within)


def _length_strata_stats(block, seeds, n_units):
    return {'finaltest': {'comparisons': {'c1': {
        'doc_sampling': {'length_strata': block,
                         'pooled_ci_n_units': n_units},
        'per_seed': [{'seed': s} for s in seeds]}}}}


def _run_length_strata_audit(obj):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'test_stats.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f)
        with contextlib.redirect_stdout(io.StringIO()):
            return A.audit_length_strata(path)


def test_length_strata_audit_replays_nonmonotone_aggregate_order():
    # Equal-count bins are contiguous in *individual* length, but a two-unit
    # stratum can hold more tokens than the next one-unit stratum.  Replaying
    # the aggregated strata must preserve the declared order instead of
    # re-sorting them by summed token count (which would silently swap L2/L3).
    tok = np.array([1, 2, 3, 3, 4, 5, 6], dtype=np.int64)
    a = np.tile(tok * 3.0, (3, 1))
    b = a + tok
    block = L.length_stratified_effect(
        a, b, tok, n_strata=5, min_units_per_stratum=1)
    assert block['status'] == 'ok', block
    assert [s['n_tokens'] for s in block['strata']] == [1, 2, 6, 4, 11]
    got = _run_length_strata_audit(
        _length_strata_stats(block, list(range(3)), len(tok)))
    assert got['status'] == 'ok', got['problems']


def test_length_strata_audit_accepts_a_single_seed_and_huge_token_counts():
    # The generator emits an `ok` block for one seed too; there is simply no
    # seed-level interval.  Token counts beyond float64 must remain exact
    # denominators and token reference weights must not raise OverflowError.
    huge = 10 ** 400
    tok = [huge, 2 * huge]
    block = L.length_stratified_effect(
        [[1e300, 2e300]], [[1e300, 1e300]], tok,
        n_strata=2, min_units_per_stratum=1, reference='token')
    assert block['status'] == 'ok', block
    assert block['n_seeds'] == 1
    assert block['units']['n_tokens'] == 3 * huge
    assert math.isclose(block['standardized']['weights'][0], 1.0 / 3.0,
                        rel_tol=1e-15)
    got = _run_length_strata_audit(
        _length_strata_stats(block, [0], len(tok)))
    assert got['status'] == 'ok', got['problems']


def test_length_strata_accepts_numpy_and_huge_custom_weights():
    # A NumPy reference array is a legitimate numeric sequence; comparing it
    # with `reference == 'equal_strata'` used to raise "truth value of an
    # array is ambiguous".  Huge integer weights have a representable
    # normalisation even though `float(w)` overflows.
    tok = np.array([10, 1000], dtype=np.int64)
    a = np.tile(tok * 3.0, (3, 1))
    b = a + tok
    block = L.length_stratified_effect(
        a, b, tok, n_strata=2, min_units_per_stratum=1,
        reference=np.array([1, 2], dtype=np.int64))
    assert block['standardized']['weights'] == [1.0 / 3.0, 2.0 / 3.0]
    huge = 10 ** 400
    block = L.length_stratified_effect(
        a, b, tok, n_strata=2, min_units_per_stratum=1,
        reference=[huge, huge])
    assert block['standardized']['weights'] == [0.5, 0.5]


def test_length_strata_cochran_q_survives_subnormal_se():
    # A per-seed stratum SE around 1e-200 is finite, but se*se underflows to
    # zero.  The inverse-variance weights must be formed on a common scale,
    # not by dividing by zero.
    tiny = 1e-200
    a = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [tiny, tiny, 2 * tiny, 2 * tiny],
        [2 * tiny, 2 * tiny, 4 * tiny, 4 * tiny],
        [tiny, tiny, 2 * tiny, 2 * tiny],
    ])
    b = np.zeros_like(a)
    block = L.length_stratified_effect(
        a, b, [1, 1, 1, 1], n_strata=2, min_units_per_stratum=2)
    het = block['heterogeneity']
    assert block['status'] == 'ok', block
    assert isinstance(het['q'], float) and math.isfinite(het['q']), het
    assert het['reason'] is None, het
    assert 0.0 <= het['i2'] <= 1.0, het
    # The true tau^2 (~8e-402) is below the smallest subnormal; a fabricated
    # 0.0 would claim no between-stratum variance, so it must be withheld.
    assert het['tau2'] is None, het
    json.dumps(block, allow_nan=False)


def test_length_strata_audit_rejects_non_quantile_membership():
    tok = np.array([1, 2, 3, 3, 4, 5, 6], dtype=np.int64)
    a = np.tile(tok * 3.0, (3, 1))
    b = a + tok
    block = L.length_stratified_effect(
        a, b, tok, n_strata=5, min_units_per_stratum=1)
    obj = json.loads(json.dumps(_length_strata_stats(
        block, list(range(3)), len(tok))))
    strata = (obj['finaltest']['comparisons']['c1']['doc_sampling']
              ['length_strata']['strata'])
    # Swap two whole one-unit strata and relabel them.  The partition, token
    # sums and NLL sums all stay self-consistent, so only a check against the
    # declared equal-count bins can catch the fabricated membership.
    strata[0], strata[1] = strata[1], strata[0]
    for i, s in enumerate(strata):
        s['index'] = i
    got = _run_length_strata_audit(obj)
    assert got['status'] == 'fail', got
    assert any('declared equal-count length bins' in p
               for p in got['problems']), got['problems']


def test_length_strata_overall_sum_overflow_is_not_a_raw_exception():
    # Each stratum sum is finite (8 * 2e307 = 1.6e308), but their overall sum
    # (3.2e308) exceeds float64.  `math.fsum` raises OverflowError for that
    # exact sum; the descriptive block must report `not_estimable` instead of
    # aborting the whole final-test analysis.
    a = np.full((2, 16), 2e307)
    b = np.zeros((2, 16))
    block = L.length_stratified_effect(
        a, b, [1] * 16, n_strata=2, min_units_per_stratum=2)
    assert block['status'] == 'not_estimable', block
    assert 'overall summed NLL' in (block.get('reason') or ''), block
    json.dumps(block, allow_nan=False)


def test_length_strata_unrepresentable_token_weight_is_not_silently_zero():
    # The huge stratum has a representable pooled NLL (1e300 / 2e400 = 5e-101)
    # and its token share is 1.0 to float64.  The short stratum's share is
    # ~1e-400, which underflows to 0.0.  Replacing that weight with zero would
    # silently remove a whole stratum from the standardised estimate.
    tok = [1, 1, 10 ** 400, 10 ** 400]
    a = np.array([[2.0, 2.0, 1e300, 1e300]])
    b = np.zeros_like(a)
    block = L.length_stratified_effect(
        a, b, tok, n_strata=2, min_units_per_stratum=2, reference='token')
    assert block['status'] == 'not_estimable', block
    assert 'token reference weights' in (block.get('reason') or ''), block
    json.dumps(block, allow_nan=False)


def test_length_strata_tiny_log_effect_keeps_a_real_delta_ppl():
    # The per-token log effect is 1e-100, so exp(a) and exp(b) both round to
    # 1.0.  The raw ΔPPL is still representable and must not be reported as a
    # fake zero; the audit replay uses the same expm1 path.
    tok = [10 ** 400] * 4
    a = np.array([[1e300] * 4])
    b = np.zeros_like(a)
    block = L.length_stratified_effect(
        a, b, tok, n_strata=2, min_units_per_stratum=2)
    assert block['status'] == 'ok', block
    assert block['overall']['delta_ppl'] == 1e-100, block['overall']
    got = _run_length_strata_audit(
        _length_strata_stats(block, [0], len(tok)))
    assert got['status'] == 'ok', got['problems']


def test_unit_quantiles_support_token_totals_beyond_float64():
    # Every count is beyond float64, but each per-token effect is 1e-100 and
    # therefore representable.  Rejecting the panel during an unnecessary
    # float conversion lost a valid distribution; the exact-ratio fallback must
    # keep it and still produce a strict-JSON block.
    tok = [10 ** 400 * (i + 1) for i in range(8)]
    diff = np.array([[1e300 * (i + 1) for i in range(8)]])
    block = L.unit_quantile_profile(diff, tok, seeds=[7])
    assert block['status'] == 'ok', block
    assert block['n_tokens'] == sum(tok)
    assert math.isclose(block['mean']['token_pooled_log_ratio'], 1e-100,
                        rel_tol=1e-12, abs_tol=0.0)
    for row in block['quantiles']:
        assert math.isclose(row['log_ratio'], 1e-100,
                            rel_tol=1e-12, abs_tol=0.0)
    json.dumps(block, allow_nan=False)


def test_unit_quantiles_underflowing_per_token_effect_is_withheld():
    # 1e-300 / 1e400 is below the smallest subnormal float.  Reporting the
    # unit as a 0.0 effect would turn an unrepresentable measurement into a
    # fake no-effect observation.
    tok = [1] * 7 + [10 ** 400]
    diff = np.array([[1.0] * 7 + [1e-300]])
    block = L.unit_quantile_profile(diff, tok, seeds=[7])
    assert block['status'] == 'not_estimable', block
    assert 'per-token' in (block.get('reason') or ''), block
    json.dumps(block, allow_nan=False)


def test_unpaired_doc_ci_supports_token_totals_beyond_float64():
    # The naive `nll / n_tokens` and `ce * n_tokens` paths raise OverflowError
    # with a 1e400 token count even though the pooled CE (1e-100) is
    # representable.  The exact-ratio path keeps the point estimate.
    got = L._nll_doc_ci([1e300, 2e300], [10 ** 400, 2 * 10 ** 400])
    assert got is not None, got
    assert got['n_docs'] == 2, got
    assert math.isclose(got['doc_ce_mean'], 1e-100,
                        rel_tol=1e-12, abs_tol=0.0), got
    assert got['ppl'] == 1.0, got
    json.dumps(got, allow_nan=False)


def test_paired_doc_ci_supports_token_totals_beyond_float64():
    # The token total (3e400) is not representable, but the pooled log ratio
    # (1e300 / 1e400 = 1e-100) is.  The exact-rational fallback must keep the
    # point estimate instead of raising "int too large to convert to float";
    # exp(a)-exp(b) also needs expm1 here, or a 5e-101 PPL difference rounds
    # to a fake zero.
    tok = [10 ** 400, 2 * 10 ** 400]
    a = [[1e300, 2e300]]
    b = [[0.5e300, 1e300]]
    got = L._paired_nll_doc_ci(a, b, tok)
    assert got is not None, got
    assert math.isclose(got['delta_log_ratio'], 5e-101,
                        rel_tol=1e-12, abs_tol=0.0)
    assert math.isclose(got['delta_ppl'], 5e-101,
                        rel_tol=1e-12, abs_tol=0.0)
    assert got['delta_ppl_ci95'] is not None
    json.dumps(got, allow_nan=False)


def test_endpoint_sensitivity_withholds_unrepresentable_scale_statistics():
    # A finite repeat panel can have a representable mean/range/SD even when
    # the naive NumPy reductions overflow, and an actually unrepresentable
    # envelope must be unavailable rather than infinite (reports are strict
    # JSON).
    res = L.endpoint_measurement_resolution([1e308, 1.7e308])
    assert res['status'] == 'ok', res
    assert math.isfinite(res['mean']) and math.isfinite(res['sd'])
    assert math.isfinite(res['range']) and math.isfinite(res['resolution'])
    json.dumps(res, allow_nan=False)

    span = L.endpoint_measurement_resolution([-1.7e308, 1.7e308])
    assert span['status'] == 'not_verifiable', span
    assert span['range'] is None and span['sd'] is None
    assert span['resolution'] is None and span['resolves_mee'] is None
    json.dumps(span, allow_nan=False)

    # Bounds whose span/hole exceeds float64 stay structurally classifiable
    # but never serialise an infinity.
    head = L.endpoint_headroom('custom', control_level=-1e308,
                               metric='custom', direction='higher_is_better',
                               bounds=[-1e308, 1e308],
                               minimum_meaningful_effect=0.05)
    assert head['status'] == 'ok', head
    assert head['span'] is None and head['headroom'] is None
    assert head['headroom_covers_mee'] is True
    json.dumps(head, allow_nan=False)

    # A huge planned unit count must not raise from float(n) and an MDE that
    # overflows is withheld, not reported as an infinity.
    huge = L.endpoint_sensitivity(
        'accuracy', control_level=0.5, minimum_meaningful_effect=1e-6,
        repeat_values=[0.5, 0.5], paired_sd=1e308, n_units=10 ** 400)
    assert huge['status'] == 'not_verifiable', huge
    assert huge['mde']['mde'] is None, huge['mde']
    json.dumps(huge, allow_nan=False)

    # A numeric string is a provenance/serialisation artefact, not a level or
    # repeat measurement.  It must not be silently promoted by float().
    assert L.endpoint_headroom('accuracy', '0.9')['status'] == 'invalid'
    assert L.endpoint_measurement_resolution(
        ['0.9', '0.91'])['n_repeats'] == 0


def test_endpoint_resolution_target_is_checked_per_endpoint():
    # The requirement check used to read `cfg`/`pcfg` left over from the last
    # endpoint of the setup loop.  A target on the alphabetically-last endpoint
    # therefore masked a missing target on every other endpoint.
    manifest = {
        'primary_comparisons': [],
        'endpoint_sensitivity': {
            'require_measurement_resolution': True,
            'endpoints': {
                'accuracy': {},
                'ppl': {'max_relative_resolution': 1.0},
            },
        },
    }
    pilot = {'split': 'validation', 'endpoints': {
        'accuracy': {'control_level': 0.9, 'repeat_values': [0.9, 0.91]},
        'ppl': {'control_level': 30.0, 'repeat_values': [30.0, 30.01]},
    }}
    got = L.endpoint_sensitivity_audit(manifest, pilot, verbose=False)
    assert got['status'] == 'fail', got
    assert any('accuracy' in p and 'no MEE or max_relative_resolution' in p
               for p in got['problems']), got['problems']
    # A declared relative target that cannot be normalised (zero repeat mean)
    # is not silently a pass either.
    rel = L.endpoint_measurement_resolution(
        [-1e-3, 1e-3], max_relative_resolution=0.1)
    assert rel['status'] == 'not_verifiable', rel


def test_subnormal_panels_do_not_fabricate_zero_spread_or_certainty():
    # np.std squares subnormal values, underflowing to 0.  Both the endpoint
    # and ordered-unit helpers must retain a representable positive SD, and
    # withhold a spread/mean that is below the smallest float instead of
    # reporting a fake zero.
    tiny = [1e-300, -1e-300] * 6
    block = L.ordered_unit_inference(tiny, n_boot=50, n_perm=20)
    assert block['status'] == 'ok', block
    assert block['sd'] > 0.0 and block['iid_ci_width'] > 0.0, block
    json.dumps(block, allow_nan=False)
    res = L.endpoint_measurement_resolution(tiny)
    assert res['sd'] > 0.0 and res['resolution'] > 0.0, res

    subnormal = [0.0, 0.0, 0.0, 5e-324, 0.0, 0.0, 0.0, 5e-324]
    assert L._es_mean_sd(subnormal) == (None, None)
    assert L._od_mean_sd(subnormal) == (None, None)
    withheld = L.ordered_unit_inference(subnormal, n_boot=20, n_perm=10)
    assert withheld['status'] == 'not_estimable', withheld
    json.dumps(withheld, allow_nan=False)


def test_order_audit_rejects_unrepresentable_value_without_crashing():
    block = L.ordered_unit_inference(
        [0.1 * i for i in range(12)], n_boot=20, n_perm=10)
    obj = {'finaltest': {'comparisons': {'c1': {'doc_sampling': {
        'order_inference': block,
        'n_docs': 12, 'pooled_ci_n_units': 12}}}}}
    obj['finaltest']['comparisons']['c1']['doc_sampling'][
        'order_inference']['sufficient_statistics']['values'][0] = 10 ** 400
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'test_stats.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(obj, fh)
        with contextlib.redirect_stdout(io.StringIO()):
            got = A.audit_order_dependence(path)
    assert got['status'] == 'fail', got
    assert any('non-finite value' in p for p in got['problems']), got


def test_tiny_positive_headroom_and_mee_are_not_absolute_epsilon_events():
    # A 1e-13 headroom on a 1e-3-wide endpoint is small relative to that
    # endpoint but is not zero; an absolute 1e-12 epsilon used to call it
    # saturated.  Likewise an exact-zero resolution must resolve a tiny MEE.
    head = L.endpoint_headroom(
        'custom', control_level=0.0, metric='custom',
        bounds=[0.0, 1e-3], direction='higher_is_better',
        minimum_meaningful_effect=1e-13)
    assert head['exhausted'] is False and head['status'] != 'saturated', head
    exact = L.endpoint_measurement_resolution(
        [1e-13, 1e-13], minimum_meaningful_effect=1e-12)
    assert exact['resolves_mee'] is True and exact['status'] == 'ok', exact
    coarse = L.endpoint_measurement_resolution(
        [1e-13, 2e-13], minimum_meaningful_effect=1e-14)
    assert coarse['resolves_mee'] is False, coarse


def test_wilcoxon_drops_only_exact_zeros_at_mixed_scales():
    # A relative "zero" cutoff of 1e-15 * max|d| deleted every tiny but
    # genuine delta whenever the panel also contained a normal-sized effect.
    # The signed-rank test must keep those observations (and use the same
    # exact-zero convention as the Hodges-Lehmann interval), otherwise n and
    # the exact null distribution silently change.
    mixed = [1.0, 1e-16, -1e-16]
    got = L.wilcoxon_signed_rank_p(mixed)
    assert got['n_nonzero'] == 3, got
    # Ranks are 3, 1.5, 1.5; the exact two-sided p is 6/8.
    assert got['p'] == 0.75, got
    assert L.hodges_lehmann_interval(mixed)['n'] == got['n_nonzero']
    # A wholly subnormal panel is still a real, non-zero measurement panel.
    tiny = L.wilcoxon_signed_rank_p([1e-300, 2e-300, -3e-300])
    assert tiny['n_nonzero'] == 3 and math.isfinite(tiny['p']), tiny
    # Exact zeros remain absent observations, independent of the panel scale.
    zeros = L.wilcoxon_signed_rank_p([0.0, 1.0, -1.0, 0.0])
    assert zeros['n_nonzero'] == 2 and zeros['p'] == 1.0, zeros


def test_ordered_batch_means_se_is_on_the_input_scale():
    # The batch-means design effect is computed on a unit-scaled copy so it
    # cannot overflow.  Its SE is a measured quantity, not a scale-free ratio,
    # so it must be mapped back just like the i.i.d. and block-bootstrap SEs.
    vals = [2.0, -2.0] * 6
    block = L.ordered_unit_inference(vals, n_boot=40, n_perm=10, seed=3)
    assert block['status'] == 'ok', block
    raw = {g: L._od_batch_design(vals, g) for g in
           (row['n_batches'] for row in block['batch_means'])}
    for row in block['batch_means']:
        expected = raw[row['n_batches']]['se']
        assert math.isclose(row['se'], expected, rel_tol=1e-12), (row, expected)
    # Rescaling the whole panel rescales the SE, not the design effect.
    scaled = L.ordered_unit_inference([10.0 * v for v in vals],
                                      n_boot=40, n_perm=10, seed=3)
    assert scaled['batch_means'][0]['design_effect'] == (
        block['batch_means'][0]['design_effect'])
    assert math.isclose(scaled['batch_means'][0]['se'],
                        10.0 * block['batch_means'][0]['se'],
                        rel_tol=1e-12)


def test_pooled_order_uses_the_stable_ppl_difference_at_small_effects():
    # exp(a) and exp(b) both round to 1.0 for a tiny log ratio, so the old
    # direct `ea - eb` reported a fake zero point and centred the interval on
    # it.  The pooled readout must use the same expm1 path as the existing
    # cluster CI.
    a = np.array([[1e-100 * i for i in range(1, 9)],
                  [1.1e-100 * i for i in range(1, 9)]])
    b = np.zeros_like(a)
    tokens = [1] * 8
    pc = L._paired_nll_doc_ci(a, b, tokens)
    block = L.pooled_order_inference(a, b, tokens, n_boot=40, n_perm=0)
    assert block['status'] == 'ok', block
    assert math.isclose(block['point_ppl'], pc['delta_ppl'],
                        rel_tol=1e-12, abs_tol=0.0), block
    assert block['ppl']['iid_ci'][0] > 0.0, block['ppl']


def test_pooled_order_subnormal_influence_keeps_a_positive_width():
    # Squaring a 1e-200 influence underflows to zero; the cluster SE itself is
    # still representable.  Withholding it (or reporting a zero-width
    # interval) would turn a real panel into fake certainty.
    psi = [1e-200 * (-1) ** i for i in range(12)]
    block = L.pooled_order_inference_from_influence(
        psi, psi, delta_ppl=1e-200, delta_log_ratio=1e-200,
        n_seeds=2, n_boot=40, n_perm=0)
    assert block['status'] == 'ok', block
    for key in ('ppl', 'log_ratio'):
        ep = block[key]
        assert ep['iid_se'] > 0.0 and ep['iid_ci_width'] > 0.0, ep
    json.dumps(block, allow_nan=False)


def test_pooled_order_supports_token_totals_beyond_float64():
    # The pooled log ratio (1e-100) is representable even though the token
    # total (10**400) is not.  The exact-rational fallback must keep the point
    # and the influence instead of rejecting the whole panel.
    tokens = [10 ** 400 * (i + 1) for i in range(8)]
    a = np.array([[1e300 * (i + 1) + (1e300 if i % 2 else -1e300)
                   for i in range(8)] for _ in range(2)])
    b = np.array([[0.5e300 * (i + 1) for i in range(8)] for _ in range(2)])
    pc = L._paired_nll_doc_ci(a, b, tokens)
    block = L.pooled_order_inference(a, b, tokens, n_boot=40, n_perm=0)
    assert block['status'] == 'ok', block
    assert math.isclose(block['point_ppl'], pc['delta_ppl'],
                        rel_tol=1e-12, abs_tol=0.0), block
    assert math.isclose(block['point_log_ratio'], pc['delta_log_ratio'],
                        rel_tol=1e-12, abs_tol=0.0), block
    json.dumps(block, allow_nan=False)


def test_pooled_order_rejects_seed_count_mismatch():
    with np.testing.assert_raises(ValueError):
        L.pooled_order_inference_from_influence(
            [1.0, -1.0], [1.0, -1.0], delta_ppl=0.0,
            delta_log_ratio=0.0, seeds=[0, 1], n_seeds=3)


def test_training_exposure_stays_total_and_json_safe_beyond_float64():
    # A caller-supplied exact threshold must not disable the allocation guard:
    # astronomically many draws must fall back to the explicit estimate rather
    # than materialising a per-draw array.  A corpus beyond int64 cannot be
    # replayed, and huge exact counts must not overflow the ratio conversions.
    huge = L.training_window_exposure(
        10 ** 400, 16, 1, 1, max_exact_draws=10 ** 18)
    assert huge['exact'] is False
    assert huge['coverage_quality'] == 'estimated'
    assert 0 <= huge['unique_tokens'] <= huge['target_positions']
    assert huge['start_coverage'] == 0.0
    json.dumps(huge, allow_nan=False)

    many = L.training_window_exposure(
        10 ** 12, 4096, 1024, 10 ** 6, seed=2, max_exact_draws=10 ** 18)
    assert many['exact'] is False
    assert many['draws'] > L.TRAINING_EXPOSURE_HARD_MAX_DRAWS
    assert many['unique_tokens'] <= many['target_positions']
    json.dumps(many, allow_nan=False)

    plan = L.training_coverage_budget(
        10 ** 400, 4096, 1024, target_coverage=0.9, max_steps=10 ** 400)
    assert plan['status'] == 'ok', plan
    assert plan['expected_unique_fraction'] is not None
    assert 0.0 <= plan['expected_unique_fraction'] <= 1.0
    json.dumps(plan, allow_nan=False)


def test_training_exposure_replays_multirank_stream_without_step_buffering(
        monkeypatch):
    # The replay must preserve the per-step RNG stream and rank concatenation
    # order, but must not buffer one array per step: with batch_size=1 that
    # makes the draw cap a poor memory bound (20M tiny arrays).  Fill one
    # output array instead, as verified here by forbidding concatenate.
    seeds = [0, L.data_seed_for_rank(0, 1), L.data_seed_for_rank(0, 2)]
    got = L._tx_draw_starts(99, 7, 3, seeds, np.int32)
    expected = []
    for s in seeds:
        rng = np.random.default_rng(int(s))
        for _ in range(7):
            expected.extend(rng.integers(0, 100, size=3,
                                         dtype=np.int32).tolist())
    assert got.dtype == np.int32
    assert got.tolist() == expected

    class _NoConcatNumpy:
        # Patch only the replay's module-level numpy reference.  Patching
        # numpy.concatenate globally would also break NumPy's own RNG seeding.
        def __getattr__(self, name):
            if name == 'concatenate':
                raise AssertionError(
                    'exact replay must not buffer step arrays')
            return getattr(np, name)

    monkeypatch.setattr(L, '_tx_np', _NoConcatNumpy())
    again = L._tx_draw_starts(99, 7, 3, seeds, np.int32)
    assert again.tolist() == expected

    # A single draw still yields the one-element stream (the empty guard is
    # not allowed to drop a valid step).
    one = L._tx_draw_starts(0, 1, 1, [0], np.int32)
    assert one.tolist() == [0]


def test_data_exposure_audit_replays_huge_exact_counts_and_run_seed():
    block = L.training_window_exposure(
        10 ** 400, 16, 1, 1, max_exact_draws=10 ** 18)
    rec = {'variant': 'csa', 'seed': 0, 'steps': 1, 'steps_completed': 1,
           'tokens_seen': block['processed_tokens'], 'data_seed': 0,
           'distributed_world_size': 1, 'data_exposure': block}
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'summary.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump({'runs': [rec]}, fh)
        with contextlib.redirect_stdout(io.StringIO()):
            got = A.audit_data_exposure(path)
    assert got['status'] == 'ok', got
    assert got['n_replayed'] == 1

    bad = dict(rec, seed=1)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'summary.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump({'runs': [bad]}, fh)
        with contextlib.redirect_stdout(io.StringIO()):
            got = A.audit_data_exposure(path)
    assert got['status'] == 'fail', got
    assert any('record seed' in p for p in got['problems']), got['problems']


def test_ledger_append_refuses_a_deleted_middle_line_even_when_tail_is_intact():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'experiments.jsonl')
        for i in range(3):
            L.append_jsonl(path, {'kind': 'event', 'i': i})
        with open(path, encoding='utf-8') as fh:
            lines = fh.readlines()
        with open(path, 'w', encoding='utf-8') as fh:
            fh.writelines([lines[0], lines[2]])
        assert L.verify_experiment_ledger(path)['status'] == 'fail'
        with np.testing.assert_raises(RuntimeError):
            L.append_jsonl(path, {'kind': 'after'})


def test_native_and_object_temporal_matrices_share_a_content_fingerprint():
    native = np.array([['2020-01-01']], dtype='datetime64[D]')
    obj = np.array([[np.datetime64('2020-01-01')]], dtype=object)
    assert DG._row_keys(native)[0] == DG._row_keys(obj)[0]
    assert DG._matrix_fingerprint(native) == DG._matrix_fingerprint(obj)
    ns = np.array([['2020-01-01T00:00:00.000000001']],
                  dtype='datetime64[ns]')
    ns_obj = np.array([[np.datetime64('2020-01-01T00:00:00.000000001',
                                      'ns')]], dtype=object)
    assert DG._matrix_fingerprint(ns) == DG._matrix_fingerprint(ns_obj)


def test_nan_payloads_do_not_change_the_content_fingerprint():
    # IEEE-754 has many NaN bit patterns.  _row_keys/duplicate_mask already
    # treat them as one missing-like identity, so the content digest must too;
    # otherwise ``np.array([[nan]])`` and ``np.array([[nan]], dtype=object)``
    # fingerprint differently and two all-NaN tables differ only by payload.
    def _nan64(bits):
        return np.array([bits], dtype=np.uint64).view(np.float64)[0]

    def _nan32(bits):
        return np.array([bits], dtype=np.uint32).view(np.float32)[0]

    nan_a = _nan64(0x7FF8000000000001)
    nan_b = _nan64(0x7FF8000000000002)
    assert math.isnan(nan_a) and math.isnan(nan_b)
    assert DG._row_keys(np.array([[nan_a]]))[0] == \
        DG._row_keys(np.array([[nan_b]]))[0]
    assert DG.duplicate_mask(np.array([[nan_a], [nan_b]])).tolist() == \
        [False, True]
    fp_a = DG._matrix_fingerprint(np.array([[nan_a]]))
    assert fp_a == DG._matrix_fingerprint(np.array([[nan_b]]))
    assert fp_a == DG._matrix_fingerprint(np.array([[nan_a]], dtype=object))
    assert fp_a == DG._matrix_fingerprint(np.array([[None]], dtype=object))
    # The same payload concern exists for the narrower native dtypes (the
    # float64 conversion preserves a payload that the object path canonicalises).
    nan32_a = _nan32(0x7FC00001)
    nan32_b = _nan32(0x7FC00002)
    assert DG._matrix_fingerprint(np.array([[nan32_a]])) == \
        DG._matrix_fingerprint(np.array([[nan32_b]]))
    assert DG._matrix_fingerprint(np.array([[nan32_a]])) == \
        DG._matrix_fingerprint(np.array([[nan32_a]], dtype=object))
    # Missing is still distinct from an ordinary finite value.
    assert fp_a != DG._matrix_fingerprint(np.array([[1.0]]))


def test_paired_effect_stats_does_not_claim_an_overflowing_sd_is_noise():
    rep = L.paired_effect_stats([1.7e308, 1.7e308, -1.7e308])
    assert rep['std_available'] is False
    assert rep['within_seed_noise'] is False, rep


def test_pooled_influence_centeredness_does_not_overflow_fsum():
    huge = [1.7e308, -1.7e308] * 8
    got = L.pooled_order_inference_from_influence(
        huge, huge, delta_ppl=0.0, delta_log_ratio=0.0,
        seeds=[0, 1], n_seeds=2, n_boot=20, n_perm=5)
    assert got['status'] in ('ok', 'not_estimable'), got
    json.dumps(got, allow_nan=False)


def test_explicit_empty_order_layouts_and_unsorted_maps_fail_closed():
    vals = [0.1 * i for i in range(12)]
    with np.testing.assert_raises(ValueError):
        L.ordered_unit_inference(vals, block_lengths=[])
    with np.testing.assert_raises(ValueError):
        L.ordered_unit_inference(vals, n_batches=[])
    ids = np.asarray([[30, 30, 10, 10, 20, 20]], dtype=np.uint64)
    keys, inv = L.document_index(ids)
    with np.testing.assert_raises(ValueError):
        L.aggregate_nll_by_document(
            L.torch.arange(6, dtype=L.torch.float32).reshape(1, 6), ids,
            document_index_map=(keys[::-1], inv))


def test_plot_tolerates_a_sparse_aggregate(tmp_path):
    agg = {'full': {'ppl_primary': 1.0, 'ppl_std': 0.1,
                    'params': None, '_group_records': []}}
    L._plot({}, agg, str(tmp_path))
    assert os.path.exists(os.path.join(str(tmp_path), 'summary.png'))
