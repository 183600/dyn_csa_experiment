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

## 2026-10-20 — nested validation for tuned baselines (no new runs)

Seed-disjoint search/final partitions were not sufficient: hyper-parameters
were selected and the final tuned contrast was still measured on the same
validation documents. `baseline_tune.py` now tunes on a deterministic
SplitMix64 row fold and reports the tuned comparison on the disjoint fold.
Selection uses token-pooled PPL on the selection fold (arithmetic seed mean is
only a legacy fallback), and the final records keep full-validation PPL for
context while `tuned_comparison.json` uses the held-out row fold
(`estimation_rows=nested_validation_holdout`). The tuning artifact records row
fingerprints/counts and `baseline_tuning_audit` refuses a claimed nested run
whose final records lack held-out-row PPL or score a different fold.

The count-model floor now fits on a declared deterministic token prefix
(`SIMPLE_BASELINE_POLICY.max_fit_tokens`, 20M by default) and records
`n_fit_tokens`/`fit_truncated`, so the bigram table cannot explode when the
project moves to a multi-billion-token B-tier corpus. The WikiText-103 budget
is below the cap, so its floor is unchanged. No frozen manifest, test split,
or README was modified.
