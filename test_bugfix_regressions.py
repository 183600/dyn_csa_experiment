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
  count beyond float64 could raise ``OverflowError`` instead of comparing.

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
