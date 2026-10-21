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

## 2026-10-20 — binary metric / calibration / F1-averaging hardening (no new runs)

Changed only zero-GPU metric and audit paths; no panel, frozen manifest, test
split, README or CI configuration was touched. (1) Binary F1 was threshold-blind:
`classification_metrics` now records the decision threshold and its source, and
`binary_threshold_curve` / `select_binary_threshold` make validation-only
operating-point selection explicit (with a scale-safe cumulative sweep). A
threshold selected on the evaluation/test rows is rejected as leakage.
(2) Scores in [0,1] now carry Brier score and ECE; a below-chance ROC-AUC is
surfaced as a label-orientation warning. (3) `per_class` now carries TP/FP/FN,
so macro, micro and weighted F1 and accuracy are reproducible from the record;
forged aggregates fail the audit. (4) Boundary alignment now separates
macro-over-sequences F1 from micro F1 and stores the pooled counts. The old
audit asserted the micro harmonic identity against macro means, which is false
for unequal per-sequence precision/recall; micro is now checked from counts and
a macro record without counts is marked unverifiable instead of wrong.
(5) The temporary issue list was removed after the fixes.

## 2026-10-20 — scale-comparability hardening (no new runs, frozen plan untouched)

Changed only zero-GPU analysis/reporting paths; no panel, frozen manifest or
test split was touched. (1) Added a budget-independent `corpus_fp` (corpus
source + vocabulary + tokenizer-fit prefix) so a 5B and a 12B tier over the
same corpus no longer look "mixed"; `data_version` still identifies the exact
served bytes, and mixed provenance generations are refused. M0 now uses the
stable fingerprint and refuses legacy/new mixing. (2) Exact sign-flip
resolution now accounts for zero deltas: the attainable floor is
`2/2**(#nonzero)`, not `2/2**n`, in `exp_lib` and the report twin. (3) Scale
points carry declared-plan coverage (`n_planned`/`planned_seeds`,
`n_input`/`n_dropped`/`n_missing`); `meta_analyze_effects` refuses a pooled
"common effect" and `scaling_trend` refuses a confirmable trend when a point
is visible as an incomplete complete-case subset. Zero-within-variance scale
points are named instead of silently dropped from the DL weights. (4) M0
cross-tier per-seed pairs now pass `pair_reason`, and extreme-scale interactions
order by declared scale rather than label text. (5) The zero-aware floor is
also carried by `reportable_claim`, the shared v7/v12 exact helper and P5T's
effective-power fields; declared tiers with n=1, no measured effect, or zero
within-scale variance stay visible and block pooling. The temporary issue list
was removed after the fixes.

## 2026-10-20 — data-path hardening (no new experiment panels)

Four data-processing defects were fixed without reading any held-out test
record. (1) The nested validation folds used by `baseline_tune.py` were a
seeded row permutation, but validation rows are consecutive chunks of the token
stream: the permutation let selection see future chunks, estimated the final
contrast partly on earlier chunks, and repeatedly crossed the same document
boundary. `validation_row_split` now returns contiguous chronological blocks
(selection before evaluation); the seed only chooses between the balanced cut
positions when the row count is not divisible by the fold count.
`baseline_tuning_audit` refuses an artifact that declares a shuffled fold
policy, reverse-time ranges, or a non-chronological split. (2) Exact-duplicate
detection converted object columns through float64, so integers above 2**53
could become false duplicates or hide a real one; numeric object rows now use
canonical exact keys (integral floats still compare equal to the same integer,
`-0.0` is normalised, NaN/Inf stay distinct). (3) Near-duplicate document
filtering now quarantines skipped copies, closing transitive chains that could
let a near copy of a near copy cross from train into validation. (4) The
temporary issue list was removed after the fixes; the frozen manifest and test
split were not touched.

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
must be reported as resolution-limited, not as no effect. A **7-seed**
extension of the key cells is the pre-declared path to a family-corrected
significance claim: the primary family has three Holm comparisons, so the
family floor is 3·2/2^n (n=6 gives 0.094 > 0.05; n=7 gives 0.047). Six seeds
would only suffice for a single raw comparison, not the declared family. The
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

## 2026-10-20 — comparison/ablation audit hardening (no new runs)

Zero-GPU methodology fixes; no panel, frozen manifest or test split was
touched.

- **Open-block control gate:** beyond `run_cfg` and the compact
  hyperparameters, a pair now must also match the `train_knobs`
  optimizer/betas/eps/schedule/clipping/augmentation/precision descriptor, the
  derived `train_objective`, and the shared-trunk `init_core_fp`. A
  `variant::seedN` key whose record carries a different seed or variant is
  excluded and named instead of being paired under the wrong identity.
- **Open-block reporting:** the report now includes per-arm n / Mean ± SD /
  token-pooled PPL. Its three-comparison Holm family needs **n>=7**; the old
  "6-seed extension" wording was wrong (n=6 only suffices for a single raw
  comparison). The driver warns before training when the planned seed count
  cannot reach the declared family floor.
- **Dynamic blocking:** `hybrid_csa_dyn - hybrid_fixed` and
  `hybrid_hca_dyn - hybrid_fixed` now decompose the boundary effect by layer
  kind, so the scheduled `hybrid_hca_dyn` arm actually contributes to a
  declared contrast instead of being an orphan run.
- **Objective descriptors:** `train_objective` now accounts for per-layer
  `comp_lambda_mult`, so `csa_dyn_nocomp` is correctly reported as not
  applying comp_reg (previously it copied `comp_lambda=0.05`). Declared
  objective interventions (`compression_regularizer`, `task_learned_boundary`)
  are recorded as expected differences; any other objective change remains a
  `system_level_only` confound.
- **Run-time seed plan:** `run_gapfill.run_full` prints every phase's
  family-corrected minimum seed count before spending GPU time.
- **Fairness audit:** `audit_fairness` now surfaces cross-arm
  `train_objective` disagreement as a warning (system-level comparison unless
  the objective change is the declared intervention), alongside the existing
  run_cfg / train_knobs / hyperparameter checks.

## 2026-10-20 — nested validation for tuned baselines (no new runs)

Seed-disjoint search/final partitions were not sufficient: hyper-parameters
were selected and the final tuned contrast was still measured on the same
validation documents. `baseline_tune.py` now tunes on a deterministic
**chronological contiguous** row fold and reports the tuned comparison on the
later disjoint fold. Validation rows are consecutive chunks of the token
stream, so a seeded row permutation (the earlier implementation) would put
future chunks into selection, estimate the final contrast partly on earlier
chunks, and repeatedly cross the same document boundary; contiguous blocks
keep selection strictly before evaluation. The seed only chooses among the
balanced cut positions when the row count is not divisible by the fold count.
Selection uses token-pooled PPL on the selection fold (arithmetic seed mean is
only a legacy fallback), and the final records keep full-validation PPL for
context while `tuned_comparison.json` uses the held-out row fold
(`estimation_rows=nested_validation_holdout`). The tuning artifact records row
fingerprints/counts/ranges and `baseline_tuning_audit` refuses a claimed nested
run whose final records lack held-out-row PPL or score a different fold.

The count-model floor now fits on a declared deterministic token prefix
(`SIMPLE_BASELINE_POLICY.max_fit_tokens`, 20M by default) and records
`n_fit_tokens`/`fit_truncated`, so the bigram table cannot explode when the
project moves to a multi-billion-token B-tier corpus. The WikiText-103 budget
is below the cap, so its floor is unchanged. No frozen manifest, test split,
or README was modified.

## 2026-10-20 — nested-validation / statistics fail-closed pass (no new runs)

This pass changed only zero-GPU gates and helpers; no panel, frozen manifest,
or test split was touched. (1) `baseline_tuning_audit` no longer rejects a
valid `n_folds>2` chronological split merely because `covers_all_rows=False`
(only the selected blocks are used), and it now re-derives the declared fold
ranges/fingerprints from `(n_rows, seed, n_folds, selection_fold,
evaluation_fold)` so an edited metadata range cannot attest a different split.
(2) `record_subset_ppl` requires exact, unique, in-range row indices; a
fractional index is no longer truncated to a different row and a repeated row
cannot inflate the held-out token denominator. (3) `paired_effect_stats`
records `n_input`/`n_dropped`/`dropped_seeds`, and `reportable_claim` refuses
significance wording for a complete-case result whose missing pairs are
visible. (4) A partial vocab declaration across scale points is now
`unverifiable`, not `shared`, because the undeclared widths could be a
different tokenisation. (5) `alpha` is validated in the Holm/claim helpers,
Holm p-values below their raw p are refused, and boolean p-values can no longer
enter a family as `False = 0`. (6) Scale seed identity is an exact integer in
`_scale_points`/`paired_log_ratio`/`scale_interaction`, so `1.5` or `True`
cannot masquerade as a paired seed. (7) The count-model floor rejects
float/boolean token ids and fractional vocabularies instead of truncating them.
No frozen manifest, test split, README, or CI configuration was modified.

## 2026-10-20 — baseline/metric design gates (no new runs)

This pass changed only zero-GPU design/reporting gates; no panel, frozen
manifest, test split, README, or CI configuration was touched.

- `hypothesis_audit` now binds a hypothesis's `primary_metric` to its role in
  `METRIC_POLICY`. A metric that is declared only as a **secondary** endpoint
  (e.g. ROC-AUC under `binary_classification`) is rejected as the primary
  endpoint unless the hypothesis explicitly labels it
  `metric_role='secondary'` with a non-empty `metric_justification`. An
  optional `metric_policy` key binds the hypothesis to one policy. The binary
  policy now has a conditional primary, `pr_auc` at prevalence < 1e-3, rather
  than only a prose preference for PR-AUC.
- `metric_policy_audit` verifies classification records against the same
  contract: `f1_primary` must equal the field named by its averaging rule,
  macro/weighted F1 must be reproducible from `per_class`, multiclass records
  must expose macro/micro/weighted F1, and a 1:1000 task may not name F1 or
  ROC-AUC as its primary endpoint. A PPL record without NLL/token accounting
  is now explicitly unverifiable instead of silently accepted.
- `baseline_tuning_audit` fails closed when a final record omits the
  hyperparameters it ran (it cannot attest the selected config). The protocol
  declares its `search_design`; a declared `full_factorial` search must cover
  every declared cell, while a marginal/OFAT design is labelled as such.
- `methodology_audit` propagates tuning/floor warnings into the panel report
  and only promotes a baseline contrast to `tuned_verified_baselines` when the
  search is adequate: verified artifact, no grid-boundary/coarse warning, no
  truncated search, and no leave-one-seed-out instability. A verified but
  inadequate grid stays in `untuned_verified_baselines` with the reason.
- Tuned comparisons report a zero-aware family resolution floor
  (`holm_min_attainable_p_floors`): ties lower the attainable exact p, so a
  resolution-limited tuned contrast is not read as a null.
- The simple count-model protocol declares a full-factorial smoothing grid and
  the audit enforces that every declared cell was actually run.

## 2026-10-20 — result-analysis hardening (no new runs)

This pass changed only training diagnostics, zero-GPU report/error-analysis
paths and their tests; no panel, frozen manifest, test split, README or CI
configuration was touched. (1) `classify_training_health` now reports the
train-validation gap slope and `steps_since_best`, and attaches explicit
next actions (regularization/data/early stop for overfitting; data-pipeline,
LR and capacity checks for a stalled run; the NaN triage order for a
non-finite curve). A widening gap is called out even when the tail validation
slope is still small. (2) A systematic non-finite triage
(`diagnose_training_instability`) checks data validation evidence, the
forward/loss term (including the compression and boundary terms), the
attention/pooling guard, fp16 scaler overflow, and gradient explosion/LR, in
that order; unmeasured causes are `unknown`, never `clean`. Records and the
v7 long-run checkpoints carry the diagnosis, and the v12 one-shot test gate
now refuses a run with a recorded instability (overridable only through the
existing logged `V12_ALLOW_UNHEALTHY` path). (3) Early stopping is available
as an opt-in `train_variant` policy with a pure decision helper; the policy
enters the cell fingerprint and hyperparameters, completed steps are what
`tokens_seen`/timeguard accounting use, and planned versus completed steps are
both recorded. (4) Bad-case analysis now triages why a token is hard
(rare/annotation candidate, ambiguous context, confidently wrong blind spot,
target outside top-k) and writes the cause inventory plus a manual-review
guide into JSONL/Markdown; the report surfaces per-checkpoint error-analysis
directories rather than assuming one root summary. (5) The efficiency report
no longer draws a Pareto frontier across different token counts, devices,
torch builds or KV semantics; within a comparable group it shows per-arm PPL
mean ± SEM, latency/parameter/memory ratios, refuses significance claims from
sub-noise or zero-variance PPL differences, and flags accuracy gains that cost
≥1.5× latency.

Follow-up to the same pass (no new runs): `best_val_ppl`, `final_minus_best`
and the health classifier now use the SAME eval-subset rows, while
`final_full_val_ppl` is recorded separately; the report no longer invites
subtracting a subset best from a full-batch final. Early-stop records carry
`steps_completed`, the planned step count and the stop reason, and the report
shows the completed/planned budget so unequal-budget arms cannot look
equal-budget. The efficiency summary also treats zero within-seed variance as
unverifiable rather than as evidence of a significant difference.

Additional result-analysis hardening (no new runs): `analyze_errors` now tallies
the heuristic causes over the whole evaluated split (not only the worst
tokens) and records each cause's share/mean CE, while the worst-token
inventory keeps the manual-review samples. `audit_gates.training_health`
fails closed when a record's `instability_diagnosis` says clean while its own
event counters or validation curve say non-finite, and warns on unhealthy or
un-triaged legacy records. The efficiency summary also flags PPL comparisons
made across different `tokens_seen` budgets and falls back to the recorded
training peak memory when the timing record has none.

## 2026-10-21 — multiclass ranking metrics + imbalance audit (no new runs)

Changed only zero-GPU metric and audit paths; no panel, frozen manifest, test
split, README or CI configuration was touched. (1) `classification_metrics`
now computes one-vs-rest `pr_auc_macro`/`pr_auc_micro`/`pr_auc_weighted` and
`roc_auc_macro`/`roc_auc_micro`/`roc_auc_weighted` for 3+ class labels, plus
per-class PR-AUC/ROC-AUC, `undefined_classes`, `macro_micro_f1_gap`,
`minority_prevalence`, `imbalance_ratio`, and the non-zero one-vs-rest PR-AUC
chance floors/lifts (`pr_auc_macro_chance`, `pr_auc_weighted_chance`). The record must name
`ranking_averaging="one-vs-rest"`, because macro (equal class weight), micro
(pooled, frequency-dominated) and weighted (support) can disagree under
imbalance. (2) `METRIC_POLICY['multiclass_classification']` now declares those
endpoints and switches its conditional primary to `pr_auc_macro` when a class
is rarer than 1:1000. (3) `metric_policy_audit` derives the minority prevalence
from the full multiclass distribution (previously only two-class dicts were
read), requires PR-AUC at extreme imbalance, checks the one-vs-rest averaging
declaration and reproduces macro/weighted ranking aggregates from per-class
values. (4) Tests cover perfect/imperfect/undefined rankings, impossible
hand-edited aggregates, the PR-AUC chance floor and the macro-micro F1 gap.
The temporary issue list was removed after the fixes.

## 2026-10-21 — trivial classification floor recorded with every metric summary (no new runs)

`classification_metrics` now records `trivial_baseline`: the always-most-frequent
predictor's accuracy, balanced accuracy and macro/micro/weighted F1 in the same
label space. This makes the straw-man score explicit (at 99:1 imbalance a
constant classifier already has accuracy 0.99) instead of leaving a bare
accuracy/F1 to be read as model skill. `metric_policy_audit` recomputes the
floor from per-class supports and rejects a record whose floor label or
aggregates do not match. Tests cover binary and multiclass floors plus a forged
easier-than-majority floor. No panel, frozen manifest, test split, README or CI
configuration was touched.
