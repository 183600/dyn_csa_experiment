# DECISION_LOG — exploratory / confirmatory paper trail

This file is the human-readable decision log. It records **why** each version
ran what it ran and **what evidence level** the result belongs to. It is not a
results table: the numeric artifacts live in `results_*/` and in the immutable
release archive produced by `archive_release.py`.

> **History note.** The repository snapshot used for this reconstruction has a
> single squashed baseline commit, so the original per-commit decision log
> cannot be recovered from git. The entries below are reconstructed from
> `README.md`, `docs/design_notes.md`, `confirmation_frozen.json` and the
> driver docstrings. Missing details are marked *not recorded in this
> snapshot* rather than invented. The file is now tracked and guarded by the
> CI loop; it is also packed by `archive_release.py`.

## Evidence levels

- **Exploratory** — a panel whose variants, seeds, endpoints, hyperparameters or
  comparisons were chosen after looking at the same validation split. Its
  p-values (even Holm-corrected) are descriptive and cannot be promoted to
  confirmation by relabelling.
- **Confirmatory** — a pre-registered freeze (`confirmation_frozen.json`) with
  fixed arms, seeds, endpoints, correction and decision rule, evaluated on the
  held-out WikiText-103 test split exactly once. The current freeze covers
  **fixed-chunk CSA mechanism claims only** (see its `scope` block).
- **Evidence archive** — a released `release_<tag>.tar.gz` plus SHA-256
  manifest, containing per-seed records, configs, environment, checkpoint
  hashes, failure/coverage records and the freeze lock. The Git repo stays
  source-only by design (`docs/design_notes.md` §8).

## Timeline (reconstructed)

| version | what was run / decided | evidence level |
|---|---|---|
| v6 (`run_gapfill.py`) | core 9-variant table, leave-one-out ablation, scale probe, long runs; 3-seed floor; results kept out of Git | exploratory |
| v7 (`v7_supp.py`) | reviewer response: RoPE + QK-norm, dense warmup, 40k-step long run, scale re-seeding, length extrapolation, top-k sweep; analytic FLOPs / KV table | exploratory |
| v8 (`v8_supp.py`) | scale panel to 4 seeds; RoPE 20k long run | exploratory |
| v9 (`v9_supp.py`) | seq-2048 top-k sweep at batch size 1; third RoPE-20k seed | exploratory |
| v10 (`v10_supp.py`) | from-scratch random-indexer / all-block controls, distractor-injection probe (uniform / realistic-token / mask / delete + span / doc / topical), NIAH-natural, crossover scaling, equal-token and 6ND-proxy large arms | exploratory |
| v11 (`v11_supp.py`) | statistical close-out: distractor probe to n=6, crossover panels to n=4, d=384 paired arms; seed-level sign-flip with resolution floors | exploratory |
| v12 (`v12_supp.py`) | **freeze**: fixed-chunk CSA confirmation plan v2 (12 seeds, 4 primary comparisons, exact sign-flip + Holm, held-out test split, freeze lock, strict record admission, per-document PPL); v1 was rejected by the load-time power gate (6 seeds × Holm/4 has minimum attainable p=0.125) | confirmatory plan |
| review 2 | seed count revised 6→12 **before any test-split access**; scope block added; two-phase freeze lock; kernel mode refused for the frozen plan; evidence archiver; leakage/causality/fairness audits | frozen plan revision |
| review 3 | external audit (`科研审查_autodl运行决策.md`) found the confirmation path was not runnable: manifest stamp v11.126 vs code v11.127, `eager_batch_expected=true` with all four arms on the per-row RoPE loop, and P5T records missing `train_knobs` (admitted=0 after the one-shot test access). Manifest **v3** re-signed before any test access: stamp v11.128, eager binding false, P5T/probe records carry the provenance descriptor, per-arm `mr` pinned, equal-budget wording corrected, decision rule declares an n_effective floor of 8, adaptive 2-SE health threshold, reachable leakage statistics, and `v12_supp.py plan` as a zero-GPU CI pre-flight. The continuous-loop CI was disabled/cancelled during the fix. | frozen plan v3 + code hardening |
| review 3 (earlier same day) | pooled-NLL PPL + per-document NLL; pooled/bootstrap/Wilcoxon sensitivity; Holm in v7–v11 and `build_report`; objective/parameter/KV descriptors; opt-in task-learned boundary arm (`csa_dynamic_task`); synthesized records excluded from crossover; span-ΔNLL lookup fixed; kernel-stamp consistency; `DECISION_LOG.md` restored and guarded; dependency/Python declarations | code/evidence hardening |
| review 4 (2026-10-20) | no new experiment panels. Unified claim gate (`reportable_claim`), Holm family-resolution ceiling, NIAH folded into the Holm family, crossover fit refusal rules + slope CI, matched-init `init_core_fp`, measured compute-control direction (`system_level_only`), stale records fail-closed in `build_report`/v10, missing `truncated` treated as unverifiable, doc-bootstrap CI + <64-doc population gate, token-coverage estimate, efficiency sweep/pair audit, selection-attribution contract, evidence-release readiness audit. The frozen v12 manifest was NOT modified. | code/reporting/evidence hardening; all earlier panels remain exploratory |
| review 5 (2026-11-14) | external audit round 2 found no statistical or causality defects (exact sign-flip/Holm/TOST/Wilcoxon verified against independent references; 39-item causality suite passed; no data leakage). Fixed the residual guard/protocol gaps, all BEFORE any test-split access: (1) JS-divergence hard gate was DEAD (nats-scale threshold 0.75 > supremum ln2≈0.693) — gates now on the ln2-normalized scale with reachability validation; (2) `load_wikitest` cache-hit path compared only the alphabetically-first train tokenizer — now compares EVERY one, same as the cold path; (3) RoPE csa/hca layers silently ran eager under a kernel-enabled process (stamp pollution) — now refuse loudly; (4) `V12_ALLOW_UNHEALTHY` override persisted to artifacts (training_health.json + per-record `allow_unhealthy_override`); (5) ngram floor dual denominator (`ppl_matched_denominator` on the LM's R×(L−1) positions) and the audit compares on the matched scale; (6) test-internal duplicates no longer miscounted as train/val contamination (`_DocDeduplicator.snapshot/peek` attribution); (7) freeze lock mirrored to a git-tracked `confirmation_freeze_lock.json` — renaming/deleting the results dir can no longer silently reset the freeze; (8) post-access selective re-measurement closed: P5C refuses to retrain frozen arms after the lock exists, and P5T refuses to overwrite a cell measured on different `ckpt_sha256` (V12_REFREEZE=1 is the explicit, recorded exit); (9) implemented decision rule disclosed as stricter than the manifest wording (`decision_rule_applied`); (10) manifest **v4**: ee/es removed from the machine-checkable signature (P5C/p1l carries es=8 under the P5H-required diagnostics, so the v3 es=128 pin was uncheckable and wrong), tokenizer 8M-token fit cap documented; (11) pre-access power simulation implemented (`v12_supp.py power`, sign-flip enumeration on validation deltas) and required by P5T (V12_SKIP_POWER=1 recorded override); (12) probe records carry `manifest_sha256`, and probes skip health-gate-refused cells; (13) watchdog-killed cells book their real wall-clock into the TimeGuard ledger; (14) `eval_span` required at admission; bylen cells verify the checkpoint architecture/mr against the frozen binding; direction_ok refuses mean==0; misc robustness (successor-walk power-of-two invariant, `_pool_cpu` on the cosine n<=1 path, dead `_SinkWiden`, adaptive grid cap in the run fingerprint when it can affect values, `holm_min_attainable_p` overflow guard, kernel bincount int64). `CODE_SEMANTICS` stays `v11.128`: every change is a guard, disclosure or metadata field — no measurement semantics on any valid path changed, so no existing record is invalidated. | frozen plan v4 + guard hardening; no panel results touched |
| review 6 (2026-11-14, independent) | fresh full-repo methodological audit (statistics, kernels, data pipeline, drivers, report layer). Fixed: zero-width soft prior doubling every attention gradient (eager + kernel twin; latent — no scheduled arm uses index_topk=0); v12 CONFIRMED gate had drifted stricter than the pre-registered manifest v3 rule (param_matched is True + init-digest gates + a report fallback that re-required coverage_ok) — restored to the frozen text via one shared helper (the review-5 loop pass had instead kept the stricter implementation with a `decision_rule_applied` disclosure; this pass reverts the implementation to the registered rule — the manifest wording was never revised — and re-points that disclosure at the restored semantics); `_classify_group_status` never returned `measured` (powered non-noise contrasts were mislabeled `noise_level`); the v6 long-panel merge pooled/paired records across different run_cfg stamps; v8/v9(/v10/v11) driver-level `csa_fixed - full` contrasts published exact p-values without the parameter-control disclosure (arms differ ~24% in parameters); `pair_reason` accepted boolean fingerprints on both sides; report length table dropped JSON integral-float spellings; v12 smoke cache isolation was overridable; `load_wikitext` required the Hub client even on cache-hit/local-dump paths. | code/methodology hardening; frozen manifest NOT modified; no test-split access |
| review 7 (2026-11-14, independent) | six-track independent audit (statistics / data-leakage / v12 frozen-plan faithfulness / training+aggregation / attention algorithms / drivers+reports, ~25k randomized numeric cross-checks). Algorithm core verified CLEAN (causality read-gate, pooling, indexer, gathered attention, RoPE, dynamic segmenter prefix-invariance, kernel twins — no defects). Fixed: (1) `scaling_trend` heterogeneity Q was NOT variance-normalised (raw `sum(n*resid^2)` in effect units) — its chi2 "p", I2, tau2 and the `heterogeneous_trend` verdict flipped under a pure change of the effect unit (log10 vs ln vs percent); now a lack-of-fit F test normalised by the pooled within-scale pure-error variance (exact via the incomplete-beta F sf; fail-closed "unavailable" when no within-scale replication); unit-invariance regression test added; (2) manifest **v5**: the v4 revision note still claimed a strictly TIGHTER implemented decision rule, but review-6 had restored the implementation to the registered v3 text — v5 corrects the historical record (no normative field changed; precondition: still zero test-split access); (3) `load_wikitext` cache-hit path did not re-validate `tokenizer_fit_token_cap` against the current `min(TOKENIZER_FIT_TOKENS, n_train_tokens)` (stale-fit cache served under the same tag if the constant ever changes) — now a fail-closed rebuild check; (4) `par.py` watchdog could overwrite a child's valid `result.json` (written in its finally block before teardown) with `watchdog_timeout` AND double-book the cell's already-refunded TimeGuard seconds — now adopts the valid result and skips re-booking (verified by an integration simulation both ways); (5) `aggregate()` silently shrank `n_seeds` when planned seeds carried no usable measurement — entries now disclose `n_dropped_error_no_ppl` / `n_dropped_nonfinite_ppl` / `dropped_seeds` / `n_dropped_synthesized` plus a NOTE line; (6) `chi2_sf` exact branch truncated at 1000 series terms (rel err 8e-6 at df=1e5) and the Wilson-Hilferty branch (df>1e5) had up to 2.7% tail error — adaptive iteration cap (20*sqrt(a)+100) and the approximation now reserved for df>1e8 (verified <1e-7 vs scipy through df=1e8); (7) `_doc_bootstrap_ppl_ci` did not drop zero-token rows (comment claimed it did; `_combined_seed_doc_ci` did) — now aligned; (8) `measure_efficiency` prefix-prefill loop timed the first call at each new length (one-time kernel selection inside the p50/p95) — one untimed warmup per length added; (9) docs: `_DocDeduplicator` docstring overstated near-dup sensitivity (measured: 1-word edit of a <=30-word line exceeds Hamming-3) — reworded to the measured net; freeze-lock mirror `.gitignore` whitelist made explicit; README/README_zh stale "CI is disabled" claim replaced with the documented requirement + a "verify the actual state" instruction; `final_minus_best` mixed-basis disclosure field added; (10) dead code removed (`fuse_analysis.signflip`, `audit_gates` unused `ra, rb`, `par._KILL` write-only flag) and `error_analysis._compare_reason` now checks checkpoint A's position-table capacity symmetrically. `CODE_SEMANTICS` stays `v11.128`: every change is an analysis-layer bug fix, a guard, a disclosure field or documentation — no measurement semantics on any valid training/eval path changed, so no existing record is invalidated. | frozen plan v5 (documentation-only revision) + statistics fix + guard hardening; no test-split access |

## What the v12 freeze does and does not cover

Covered: fixed-chunk CSA vs parameter-matched dense, trained random-indexer and
trained all-block controls at L4096, on the held-out test split.

Not covered: dynamic chunking vs fixed chunking, semantic value of dynamic
boundaries, HCA advantages, dynamic-chunking scaling laws. Those claims stay
**exploratory** and require their own pre-registered freeze before a first
test-split access.

## Re-freeze rule

Any change to `confirmation_frozen.json` after the first test-split access
requires `V12_REFREEZE=1`, downgrades the earlier round to exploratory, and
must be recorded here with date, reason and the old/new manifest hashes. The
freeze lock (`results_lm_test/.freeze_lock.json`) is written **before** the
first read and updated with the test-slice fingerprint immediately after it;
the archive includes its hash. Do not delete the lock unless you can prove no
test-split record was ever written.

## Known limitations (not hidden)

- The validation split was reused for selection throughout v6–v11; those
  panels cannot be made confirmatory retroactively.
- Small seed counts (n=3–6) give exact sign-flip p floors of 0.25–0.031; many
  old cells are resolution-limited, not null.
- The default `csa_dynamic` gate is a cosine threshold with a learned scalar
  offset trained by the compression regularizer; it is **not** trained
  end-to-end by the LM loss. `csa_dynamic_task` is the opt-in score-function
  arm that makes the boundary policy task-trained.
- Efficiency claims from analytic FLOPs/KV are structural, not measured
  latency/throughput/memory; use the efficiency harness on real checkpoints.
- WikiText-103 has near-duplicate documents across official splits. The
  **test** loader removes exact/near duplicates against the train documents
  (`near_duplicate_cleanup` in the test meta); the **validation** split still
  contains near-duplicates and the leakage audit quantifies that overlap
  without removing it. Validation is also the split that was reused for all
  v6-v11 selection, so it cannot support confirmatory claims.

## 2026-10-19 — open-block panel (maximum block length × current-prefix reading)

The two open questions — does a maximum block length help, and does the
unreadable still-open block create an access bottleneck — were previously
tangled in separate ablations. They are now one controlled matrix with two
declared factors (`max_block_tokens`, `open_block_mode`) and a fixed recipe.
The main model is pure dynamic CSA so the three read rules compete for the
same `index_topk` slots; HCA reads every closed block and therefore cannot
carry the same-budget label (open HCA must declare an additional-read budget
and is not in the main matrix).

The prefix is causal by construction (block start and observed length only;
no block-final length/end or tail-relative position bias). `prefix_singletons`
is a length-1 pooled representation through the same pooling/projection
parameters, explicitly not raw-token K/V. The closed path is unchanged and is
covered by a before/after regression check. Position-table capacity is
decoupled from `max_block_tokens` (capacity 129, truncation) and recorded with
the truncation counts.

The 8-cell × 4-seed panel is exploratory until its own pre-registration:
4 seeds have an exact sign-flip p floor of 0.125, so a non-significant result
must be reported as resolution-limited, not as no effect. A 6-seed extension
of the key cells is the pre-declared path to a significance claim. The
outside-window panel is auxiliary and must not be pooled with the full-prefix
results; the extra-read-slot panel is a separate protocol if it is needed.


## 2026-10-20 — review-4 hardening (no new runs, frozen plan untouched)

The fourth review repeated the same methodological concerns at the reporting
layer: over-broad external claims, unmatched compute, multiplicity, stale
records and the absence of a real evidence release.  This revision does not
pretend to answer them with new data.  It changes what the code is allowed to
say:

- every "significant/confirmed" sentence must pass `exp_lib.reportable_claim`
  (n>=6, raw p, Holm p, effect > seed noise, family resolution floor);
- every crossover slope must pass `crossover_scaling_fit` (>=4 de-duplicated
  widths, no censored subset, training volume within 25%) and carries a CI;
- a pair whose shared-trunk initialization digest differs is refused, and a
  significant contrast whose extra compute favours the claimed winner is
  downgraded to `system_level_only`;
- stale length records are excluded from tables and conclusions, not quoted
  with a warning marker; a missing `truncated` flag is unverifiable;
- a "faster" efficiency claim requires same-shape/same-device measured
  prefill outside a +/-5% band;
- a selection-recall claim requires the artifact contract in
  `audit_gates.py selection`; until it exists the audit says the claim is
  unsupported;
- `RELEASES.md` with no complete row makes the repository explicitly
  source-only and not independently auditable.

The frozen confirmation manifest (`confirmation_frozen.json`, manifest v3)
and its power table were not changed; no test split was read.  The per-issue
table is `REVIEW_4_RESPONSE.md`, and the statistical conventions are in
`docs/design_notes.md` section 25.

## 2026-11-14 — review-6 independent full-repo audit (no new runs; rebased on the review-5 loop pass)

A fresh audit of the whole tree (statistical functions numerically verified
against scipy / exhaustive enumeration; kernel reference twins against naive
independent implementations; the data pipeline exercised end-to-end on a
synthetic local dump; every driver and the report layer read against their own
contracts) confirmed the statistical core (Holm, exact sign-flip incl. the
MITM/saddlepoint paths, Wilcoxon, bootstrap CIs, TOST, crossover fits,
pooled-PPL aggregation) and the leakage-critical contracts (test split read
only by P5T, train-only tokenizer fit, split disjointness, freeze-lock
ordering, resume semantics). It also found and fixed eight real defects. No
experiment was re-run, no numeric artifact was regenerated, the frozen
manifest was NOT modified, and no test-split access occurred.

1. **Zero-width soft prior doubled every attention gradient**
   (`exp_lib.gathered_attention`, `gathered_attention_open`,
   `csa_kernels.gathered_attention_kernel` / `_attn_bwd_ref`). With
   `index_topk == 0` the indexer returns a width-0 `soft` tensor, not `None`;
   the straight-through branch then adds an S path numerically identical to
   the hard A path (`soft_logits == logits` when there are no block lanes), so
   `dq`/`dk`/`dsink` were exactly 2x the contract (verified numerically in
   fp64 on both the eager path and the kernel reference twin). Width-0 is now
   normalised to "no prior" at all entry points (forward values bit-identical,
   gradients 1x). Latent only: no scheduled arm uses `index_topk=0` (the
   smallest is topk8), so no existing record was affected and
   `CODE_SEMANTICS` stays v11.128 — bumping the stamp would retire every panel
   for an edge case no panel measures.
2. **v12 CONFIRMED gate had drifted off the pre-registered rule.** A
   continuous-loop commit of 2026-11-12 changed `param_matched is not False`
   to `is True` and added `same_init_core_fp`/`init_core_fp_recorded` gates,
   and the report's legacy fallback additionally re-required `coverage_ok` —
   conditions the manifest v3 decision rule does not contain (v3 explicitly
   removed the all-12-seeds veto, and the manifest records the init digest as
   reported instrumentation, "reported and compared"). A
   stricter-than-registered gate is post-freeze drift: it can only fail to
   confirm what the pre-registered rule confirms. The primary gate and the
   fallback now call one shared `v12_supp._frozen_rule_confirms` implementing
   the manifest text exactly (strict `p_holm < alpha`, direction match,
   `param_matched is not false`, n_effective floor), the pinning test was
   rewritten to the frozen semantics, and the loop pass's
   `decision_rule_applied` disclosure now documents the restored rule
   (including which gates are deliberately NOT decision inputs).
3. **`exp_lib._classify_group_status` could never return `measured`.** The
   fall-through after the resolution-floor loop returned `noise_level`, so a
   controlled, powered, non-noise, resolution-OK, not-significant contrast was
   published in the bucket whose printed meaning is "effect within seed
   noise" — a false null reading. The final return is now `measured` (per its
   own docstring), and the descriptive `measured_components` tier no longer
   double-lists such a component.
4. **The v6 long-panel merge mixed configurations.** `merge_long_aggregate` /
   `_fold_backup_into_aggregate` (`run_gapfill.py`, the opt-in V6_SYNTH=1
   path) folded pre-retrain snapshot seeds into freshly re-measured entries
   and recomputed `dPPL_vs_full` across arms with different run_cfg / code
   stamps, with no identity check — exactly the pooling design_notes §3.1
   forbids. Both folds now require one shared `run_cfg` fingerprint
   (missing/differing => the fresh entry stands alone, loudly), and the
   paired-delta repair only runs within one fingerprint. Same-config folds
   (the recovery purpose) are unchanged.
5. **Driver-level contrasts lacked the parameter-control disclosure.** The
   v8/v9 `add_paired` helpers (also used by v10/v11) publish exact sign-flip
   p-values for `csa_fixed - full`-style contrasts whose arms differ ~24% in
   total parameters (11.22M vs 9.05M at the shared d=256/6L geometry) — above
   the repo's own 1% gate and without the `system_level_only` labelling
   `methodology_audit` applies to declared designs. The contrasts stay (they
   are honest system-level comparisons), but every stats record now carries
   `params_a`/`params_b`/`param_rel_diff`/`param_controlled` via the new
   `exp_lib.paired_param_control`, the console output discloses
   "capacity-confounded: system-level" when the gate fails, and
   `param_controlled is None` (unverifiable) is disclosed as well.
6. **`pair_reason` accepted boolean fingerprints on both sides.**
   `data_fp=True` on both records mapped to absent-vs-absent and paired as if
   controlled, while `assert_shared_validation_fp` correctly raises on a
   boolean `val_fp`. A boolean on either side is now `unverifiable` (also for
   `init_core_fp`).
7. **Report length table dropped JSON integral-float spellings.** `_len_key`
   promised "integral floats validated by value", but the string branch — the
   only one a JSON-loaded panel can reach — rejected "1024.0" and silently
   rendered the cell as "—". The canonical integral-float spelling is now
   accepted; fractional keys, huge lossy floats and duplicate spellings stay
   rejected/ambiguous.
8. **v12 smoke cache isolation was overridable** (`os.environ.setdefault`):
   an exported `WT103_CACHE_DIR` would redirect the smoke's cold build into
   the frozen cache — the exact hazard the guard exists for. The override is
   now unconditional. Separately, `load_wikitext` imported the Hub client at
   function entry, so even a warm cache or the air-gapped `WT103_RAW_TXT`
   fallback required `datasets`; the import moved into the Hub branch (a pure
   availability fix, no measurement semantics change).

Known items, reconciled with the review-5 loop pass that landed while this
audit was in flight: the frozen `training_recipe_signature.ee/es` dead letters
this audit flagged are now RESOLVED by the loop's manifest **v4** (pre-access
revision: ee/es removed from the machine-checkable signature, exactly because
they gate diagnostic-only evals that never touch the weights or the frozen
endpoint; `_recipe_matches_signature` keeps a compatibility skip for v3-era
manifests). The loop's decision-rule handling is superseded here: it had kept
the drifted stricter implementation with a `decision_rule_applied` disclosure,
while the manifest wording was never revised — this pass restores the
implementation to the registered rule (`param_matched is not false`, no
init-digest gates, no coverage_ok veto; one shared
`v12_supp._frozen_rule_confirms` for the primary gate and the report fallback)
and re-points the disclosure at the restored semantics. Still known-and-
accepted: `load_wikitest`'s cache-hit cross-binding degrades to a NOTE when
train-side cache files are absent (the loop pass additionally made the
cache-hit path compare every train tokenizer; the mandatory leakage audit and
P5T admission still refuse unverifiable states, so the confirmation path is
not exploitable through it).

Regression tests: `test_review_fixes.py` checks M1-M7 (zero-width prior,
boolean fingerprints, measured tier, cross-config merge, float length
spellings, param disclosure, frozen-rule helper) plus the rewritten
`t_audit_resolution_classification` and
`t_finaltest_decision_rule_matches_frozen_manifest`; full suite green
(253 + 15 checks).
