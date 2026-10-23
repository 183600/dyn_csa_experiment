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

## 2026-10-21 — tuned-family accounting, data transfer, lexical NIAH floor, real token-level folds (no new runs)

Zero-GPU experiment-design hardening; no panel, frozen manifest, test split,
README or CI configuration was touched. (1) The tuned power floor is now
computed from the protocol's **declared** method-vs-baseline contrasts rather
than `len(arms)-1`; an undeclared arm is rejected, and the same
`tuned_claim_allowed` verdict is exposed by the tuning and count-floor audits
and surfaced in the fairness gate. (2) Tuning trials now persist
`data_fp`/`val_fp`/`eval_fp`/`data_version`; the audit refuses a final estimate
on a different corpus/validation source, complementing the existing
horizon/exposure transfer checks. (3) `niah_copy_rule_baseline` adds an
untrained nearest-previous-key copy floor to every NIAH cell; the metric audit
recomputes it and warns when the learned model does not beat it. NIAH retrieval
hypotheses are now pre-registered and audited alongside the other hypothesis
families. (4) The document-group fold path was not actually runnable on a fresh
WikiText cache: `load_wikitext` wrote only document-level ids while
`cached_validation_row_groups` expected per-token ids. Validation now also
writes `val_token_doc_ids_*.npy` (validation only, so a 5B-token train sidecar
is not materialised); legacy caches are an explicit `groups_aware=False`
fallback, and a group-aware split that cannot be formed without cutting a
document falls back to full validation with the reason recorded rather than
crashing or silently using row-only folds. The temporary issue list was not
needed and no scratchpad remains.

## 2026-10-21 — NIAH trivial floors, primary-accuracy tests, document-group folds (no new runs)

Changed only zero-GPU metric/statistics/tuning paths; no panel, frozen
manifest, test split, README or CI configuration was touched. (1) NIAH-natural
now records `n_correct`, per-answer support/recall, macro recall (balanced
accuracy), the uniform chance floor over observed answer classes, the
majority-answer floor and both lifts; the v10 report shows them side by side.
`metric_policy_audit` recomputes the accuracy from counts and rejects forged
chance/majority/macro values, while legacy records without a floor are named
as unverifiable. (2) NIAH paired statistics now test top-1 accuracy (the
declared primary endpoint) as well as answer NLL; accuracy contrasts join the
same Holm family, so a significant secondary NLL cannot stand in for an
untested retrieval gain. (3) Nested tuning/evaluation row folds are now
document-group aware: the cached per-token validation document ids are folded
into row groups (rows sharing a document are unioned), and the tuning, final
and simple-baseline paths pass those groups to `validation_row_split`. A
missing sidecar is an explicit `groups_aware=False` fallback; a present but
inconsistent sidecar fails closed. Tests cover the retrieval floors, forged
NIAH aggregates, and document-crossing rows. The temporary issue list was
removed after the fixes.

## 2026-10-21 — baseline adequacy / conditional endpoints / degenerate cuts (no new runs)

Changed only zero-GPU baseline-audit and metric-audit paths; no panel, frozen
manifest, test split, README or CI configuration was touched. (1) The tuned
count-model floor now has the same adequacy gate as the neural tuning: a full
factorial grid is not enough if the selected alpha sits on the minimum/maximum
tested level or ties the runner-up. `simple_baseline_audit` records
`grid_adequacy`, `grid_boundary_hits`, `selection_margins` and
`selection_tie_arms`; `methodology_audit` exposes
`tuned_floor_claim_allowed` and refuses to describe a boundary/tied floor as a
well-tuned one (the arithmetic floor contrast remains visible). (2)
Multiclass one-vs-rest records now carry per-class `pr_auc_chance`,
`pr_auc_lift_over_chance`, `roc_auc_chance` and `roc_auc_lift_over_chance`;
the audit recomputes them from support/N and rejects forged lifts. Binary
records expose chance-relative ROC skill. (3) `classification_metrics` now
states a prevalence-aware `recommended_primary_metric`/`_average`; the audit
checks that a recorded recommendation matches the policy's conditional endpoint
(PR-AUC at <1:1000, F1 otherwise). (4) A hypothesis may declare `prevalence`;
`hypothesis_audit` then rejects a normal F1 primary at an extreme base rate
unless the conditional endpoint is named (or a justified secondary endpoint is
declared). (5) Degenerate binary operating points are visible: records expose
`predicted_positive_rate`, `degenerate_predictions`, score range and whether
the threshold falls inside it. A threshold outside the observed score range is
a metric-contract failure; an all-one-class cut is listed in
`degenerate_prediction_records` rather than silently passing as model quality.

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
imbalance. A row-normalized probability matrix additionally gets multiclass
Brier and top-label ECE; logits are reported as uncalibrated rather than
treated as probabilities. (2) `METRIC_POLICY['multiclass_classification']` now declares those
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

## 2026-10-21 — zero-information (uniform) floor sanity check (no new runs)

`simple_baseline_fit` now records `uniform_ppl` (the vocabulary size),
`best_ppl_vs_uniform` and `data.vocab_size`; `simple_baseline_audit` verifies
these agree and warns when the selected count model is worse than uniform
tokens. A count model that cannot beat zero-information sampling points at a
smoothing/tokenizer/data-path defect (or a signal-free sample), so it must not
be quoted as a baseline. The check is a warning rather than a hard failure
because a small/random evaluation sample can put the finite-sample count model
marginally above uniform. No panel, frozen manifest, test split, README or CI
configuration was touched.

## 2026-10-21 — metadata/audit robustness and binary endpoint correctness (no new runs)

Changed only zero-GPU metric, data-key and audit paths; no panel, frozen
manifest, test split, README or CI configuration was touched. (1) Binary
PR-AUC is scored against the POSITIVE class, so a correct record whose
positive endpoint is the common label (rare negative event) is no longer
reported as a misstated chance floor; when a record omits a separate
`prevalence`, the recorded PR-AUC floor itself still triggers the rare-event
endpoint check. Mixed-type binary labels (integer `1` vs string `"1"`) are now
turned into predictions as an object array instead of relying on NumPy dtype
promotion, which raises `DTypePromotionError` in NumPy 2. (2) The metric audit
fails closed on non-object manifest/per-class metadata and non-string declared
metric names instead of raising `TypeError`/`OverflowError` or aborting on
unhashable labels. (3) The tuning and count-model audits report scalar/non-
mapping protocol fields (trials, arms, grids, inherited tuning, selected
config) as evidence problems rather than crashing; the archive verifier
refuses malformed per-file manifest entries and unreadable/corrupt archives.
(4) Exact-duplicate detection now keys `Decimal('sNaN')` like every other
numeric NaN (missing-like) instead of treating its repr as a distinct
category. Tests cover the binary positive-floor orientation, mixed-type
predictions, malformed protocol/manifest shapes and signaling-NaN duplicates;
the temporary issue list was removed after the fixes.

## 2026-10-21 — baseline direction/budget provenance + micro PR-AUC chance floor (no new runs)

Changed only zero-GPU baseline-audit, metric-audit and fairness-audit paths;
no panel, frozen manifest, test split, README or CI configuration was touched.
(1) `baseline_tuning_audit` no longer trusts an artifact's selection
`direction` as free metadata: it must equal the pre-registered protocol
direction, and PPL selection must be lower-is-better. A flipped artifact could
otherwise crown the maximum-PPL trial as the tuned winner. Explicit blank
values for `selection_metric` / `selection_split` / `direction` are now
malformed evidence rather than silently falling through to the protocol
default, and both baseline audits bind the recorded split to the declared
protocol. Non-object protocols fail closed instead of raising.
(2) The count-model floor audit now rejects undeclared arms (an extra arm
could become the quoted `best_baseline` outside the equal-search design),
rejects malformed `arms` declarations, enforces the one implemented
`fit_token_policy` (`deterministic_prefix`), and checks that the reported floor
estimand (`best_ppl_estimation_rows`) matches the nested evidence actually
scored. Grid adequacy now also flags a selected interpolation weight at the
0/1 bigram–unigram corners, not only the alpha boundary.
(3) `classification_metrics` / `_multiclass_ranking_metrics` now record the
micro one-vs-rest PR-AUC chance floor (`1/K`) and its lift; the percentile
audit recomputes it from the class count and range-checks it. Trivial-floor
reproduction now requires exact non-negative supports (fractional/boolean
supports are rejected instead of truncated). The metric audit also verifies
that a record claiming `recommended_primary_available=True` actually carries
the recommended endpoint (and that `False` is not contradicted by a recorded
value).
(4) `audit_gates.audit_fairness` now checks cross-arm parameter totals:
differing exact totals are a capacity confound; missing/non-unique totals are
unverifiable, not a silent pass. Tests cover every new gate. The temporary
issue list was removed after the fixes.

## 2026-10-21 — comparison controls, attrition-aware claims, scale seed floor (no new runs)

Changed only zero-GPU analysis/audit/driver config paths; no panel result, frozen
manifest, test split, README or CI configuration was touched. (1) `build_report`
now exposes `n_dropped` and leave-one-seed-out sign stability in
`exact_signflip`, and `_sig_gate` refuses "significant" when a paired seed was
dropped by the controlled-comparison gate or when dropping one seed flips the
mean sign. The paired row and P0W warmup paths pass their planned pair count, so
a complete-case subset can no longer be quoted as a complete panel. (2) The
tuned-vs-tuned gate (`baseline_tune._relaxed_pair_reason`) still allows per-arm
lr/weight_decay/warmup, but now requires every other structured hyperparameter
(seq_len, batch/eval subset, model shape, vocab, precision, ...) to agree; a
one-sided descriptor is unverifiable. The tuned significance flag also requires
a complete, non-seed-driven panel at n>=3. (3) `scale_ladder_audit` now compares
the declared optimizer/betas/eps/clip/schedule-shape/augmentation/precision
descriptor across tiers (mismatch = fail, missing/partial = warning) and warns
whenever LR varies across tiers instead of silently accepting an LR-transfer
confound. (4) v10 Holm correction keeps unmeasured declared contrasts as dummy
p=1 slots instead of shrinking the family. (5) `reportable_claim` refuses
significance when `n_skipped_config_mismatch`/`n_skipped` seeds were excluded by
the control gate. (6) Prospective seed plans for the top-k sweep, length
extrapolation and all cross-size scale phases were raised from 2 to 3 seeds.
The temporary issue list was removed after the fixes.

## 2026-10-21 — leakage-safe data path hardening (no new runs)

Changed only the zero-GPU `data_guard.py` contract; no panel, frozen manifest,
test split, README or CI configuration was touched. (1) Split indices are
range-checked before the int64 cast, so unsigned/float indices above the int64
range fail as out-of-range rather than wrapping into a misleading negative
index. (2) Target encoding treats `NaT` as a missing label and accepts exact
Decimal/numeric targets; persisted target tables must retain a non-empty
`train_fp` and a consistent `n_train`, otherwise the in-sample full-table leak
guard would silently disappear after a reload. (3) Cleaning now refuses
`drop_invalid_labels`/non-default `label_conflicts` without labels, cleans
duplicate/label-conflict rows in categorical tables, applies train-only
Decimal target fences, and can keep duplicates explicitly
(`drop_duplicates=False`). (4) `split_and_prepare` binds the quality audit to
the split it actually returns: reserved `quality_kwargs` overrides are refused.
(5) Numerically stable mean/std/median/quantile paths keep fitted
preprocessing/target statistics finite and reloadable near the float64 limit,
and a divided transform avoids overflow for opposite-signed extremes. (6)
Signaling-NaN ordering keys are rejected before sorting, and persisted
category/frequency tables refuse truncated/empty key tables or legacy
non-native identities that can no longer be reconstructed. The temporary issue
list was removed after the fixes.

## 2026-10-21 — baseline budget transfer, floor truncation, NIAH imbalance endpoint (no new runs)

Changed only the zero-GPU baseline/metric/audit and NIAH record paths; no panel,
frozen manifest, test split, README or CI configuration was touched. (1) The
baseline-tuning audit now records and checks `budget_transfer`: the declared
`search_steps` must match the tuning trials and the final records' step budget,
and the trial/final token exposure must agree as well (equal steps do not imply
equal exposure if batch/sequence length changes). Equal search across arms did
not catch a search run at a shorter horizon than the reported final evaluation;
such a config is not the tuned optimum at the final budget. A mismatch is
disclosed and gates the "well-tuned baseline" claim, while legacy artifacts
without step/token evidence stay unverifiable rather than passing. (2) `simple_baseline_audit` now requires the recorded prefix to be
exactly `min(n_source_tokens, max_fit_tokens)`, rejects a `fit_truncated` flag
that contradicts the token counts, and propagates prefix-trained floors into
`tuned_floor_claim_allowed`. At scale the count-model floor can see far fewer
tokens than the neural panel; the arithmetic contrast stays valid, but it is no
longer called a floor trained on the full corpus. (3) `niah_retrieval` now
declares a conditional primary (`macro_recall`) below 1:1000 answer-class
prevalence; `retrieval_metrics` emits the recommendation/reason/availability,
the audit verifies it, and v10 NIAH records persist those fields. (4) Tests
cover horizon mismatch, truncated-floor contradictions, and NIAH hypothesis /
record alignment. The temporary issue list was removed after the fixes.

## 2026-10-21 — curve/NaN/bad-case/efficiency analysis hardening (no new runs)

Changed only zero-GPU result-analysis paths, their report rendering and tests;
no panel result, frozen manifest, test split, README or CI configuration was
touched. (1) `classify_training_health` / `early_stop_decision` now accept the
documented `val_ce`-only representation (deriving PPL only when the PPL field
is absent; an explicitly non-finite PPL is never papered over), order the tail
chronologically, collapse a repeated step to one evaluation, derive the
train-validation gap and its slope when a legacy record omits them, and mark a
two-point trend as low-resolution rather than presenting an interpolated slope
as a mechanism result. (2) The report now separates the number of curve points
from the number of documents per evaluation, which were previously both shown
under "curve rows". (3) `diagnose_training_instability` derives curve health
from `health_history` when the cached summary is missing, lets the curve
override a contradictory `healthy` cache on a non-finite point, and reports an
unmeasured curve as `unknown`. The health audit recomputes the curve and checks
the recorded status, best step/PPL, `final_minus_best`, `steps_since_best`,
event counters vs the event log, unknown health labels and empty
`healthy` histories. (4) Bad-case analysis now requires a measured training
frequency basis before labelling a target rare (empty/malformed `train_ids` no
longer manufactures all-rare cases), exposes same-prefix/different-target
ambiguity (label inconsistency vs genuine ambiguity) in JSONL/Markdown, makes
zero/negative error-analysis budgets explicit instead of clamping them to one,
and refuses paired per-token comparisons across different vocabularies.
(5) Efficiency summaries reject zero/non-positive latency, PPL, parameter and
memory readings instead of letting them dominate a frontier or a ratio, and
the accuracy/cost warning states the improvement direction without a negative
"gain". The temporary issue list was removed after the fixes.

## 2026-10-21 — comparison/ablation statistical hardening (no new runs)

Changed only zero-GPU analysis/audit/report paths and tests; no panel result,
frozen manifest, test split, README or CI configuration was touched. (1) A
controlled pair now requires a COMPLETE `train_knobs` descriptor. Two records
carrying the same partial dictionary used to pass `pair_reason` even though the
missing optimizer/betas/eps/schedule/augmentation fields were never compared;
a present-but-incomplete or malformed descriptor now fails in `exp_lib`,
`build_report`, `baseline_tune._relaxed_pair_reason` and the fairness audit,
while legacy-vs-legacy (both absent) stays the documented permissive path.
`controlled_variables_report` names the defective descriptor instead of
treating it as a controlled recipe. (2) `paired_effect_stats` no longer labels
an underpowered or resolution-limited contrast "significant" when a loose
alpha would otherwise let it through, and it exposes `std_defined` (a
single-seed arm still reports the legacy numeric 0.0, now explicitly marked as
not estimable; `arm_summary` adds `ppl_std_defined`). (3) The console paired
table is derived from `ABLATION_DESIGN` instead of a drifted duplicate list, so
every declared leave-one-out contrast (including the task-learned-boundary
ablation) is displayed. (4) Project-level ablation coverage only counts a
component as verified when a panel also contains its complete declared
ablation; verification through a partial ablation is reported separately and
does not open the "all components useful" claim. (5) `audit_methodology`
surfaces that project-level claim gate. Tests cover partial/malformed
descriptors, underpowered significance, the report/table parity, and partial
ablation coverage. The temporary issue list was removed after the fixes.
## 2026-10-21 — review-5: P1/P2 hardening before the one-shot test (no new runs)

This revision changed only zero-GPU driver/statistics/manifest paths and CI
configuration; no panel result, training recipe, arm, seed, endpoint or test
split was touched.  It is recorded as `revision.history` v4 in
`confirmation_frozen.json` and still predates any test-split access.

**P1 (methodology).**
(1) The frozen endpoint remains L4096, but every admitted comparison now
carries the full by-length curve (L512/L1024/L2048/L4096) computed from the
same records.  L512 is the training length; if its direction reverses, the
report must read the L4096 result as length-extrapolation robustness, not as
"CSA is better".
(2) Each primary comparison now declares `mechanism_claim_allowed`: only the
randidx-vs-learned contrast may be read as selection-mechanism evidence (and
only versus a fixed pseudo-random pattern); the main/aligned-dense contrasts
are system-level because MLP width/FLOPs, kernel, qk-norm/kv-norm width and
the learned sink differ, and allblocks is not reading-budget matched.
(3) The manifest's required pre-test power simulation is implemented as
`python v12_supp.py power`: exact conditional power over the 2^12 sign
patterns of the P5C validation deltas, a minimum same-sign count, and a
refusal when power at q=0.95 is below 0.8 (the q=0.90 sensitivity is
reported alongside; it is expected to be ~0.66 at n=12) (`V12_ALLOW_LOW_POWER=1` is the
explicit override).  It reads only the validation summary, never
`load_wikitest`.
(4) A complete-case subset can no longer be confirmed: v4 requires all 12
frozen seeds (`require_complete_coverage=true`), so cap skips, missing
checkpoints or health-gate refusals are reported as unconfirmed rather than
as a smaller-n confirmation.  The 8-pair mathematical floor is still
reported for transparency.

**P2 (engineering).**
(1) `run_full` now runs P5C -> P5H -> P5PW -> P5T before the exploratory
probe/scale panels.  A headroom stop sets `all_ok=False`, records the phase as
incomplete, prints `INCOMPLETE` (never `ALL PHASES DONE`) and extends the
shutdown delay so the log can be inspected.
(2) The P5H/P5T health gate treats a flat cosine tail
(`underfitting_or_stalled`) and `validation_improving_train_flat` as
recorded, non-blocking statuses; only nonfinite / overfitting /
validation_degrading / insufficient_data / unmonitored block the test.
(3) The scheduled continuous-loop workflow is disabled (`workflow_dispatch`
only) and the `fast_prompt()` wrapper that instructed the agent to scan for
issues, keep editing and commit after every change was removed.  An
unattended agent with push rights must not be able to mutate the frozen code
or manifest while the confirmation awaits its single test access.
(4) `python v12_supp.py power` and `phase P5PW` are the zero-GPU preflight;
`phase P5T` re-runs it, so a direct P5T cannot bypass the gate.

The temporary issue list was removed after the fixes.

## 2026-10-21 — review-6: sink 2x2, seeded random control, 48 test docs (no new runs)

Still before any test-split access; recorded as `revision.history` v5 in
`confirmation_frozen.json`. No training was run for this revision.

(1) **Sink 2x2.** Added `full_rope_sink` (same full-RoPE dense arm plus the
learned per-head sink) and `csa_fixed_rope_nosink`. The sink-matched contrast
`csa_fixed_rope - full_rope_sink` and the no-sink contrast
`csa_fixed_rope_nosink - full_rope` are both in the frozen family; their
difference is the sink-by-architecture interaction. `full_rope_sink` is
parameter-matched with `mr=6.763427734375` (the sink adds 48 parameters).
These comparisons are still multi-factor (kernel, qk-norm/kv-norm, read
fraction) and therefore `mechanism_claim_allowed=false`; they test whether the
v4 main result is a sink artifact, not whether the selection mechanism works.

(2) **Seeded random control.** Replaced the frozen fixed-hash `randidx` arm with
`csa_fixed_rope_randidx_seeded`: the pseudo-random block pattern is salted by
`(seed, layer)` (`seed*1000003 + layer + 1`), so the control is a random
variable across seeds rather than one arbitrary pattern shared by all seeds.
The legacy arm remains in code at salt 0 (bit-identical path) but is retired
from the frozen panel. The seeded arm is the only comparison eligible for a
selection-mechanism claim, and only versus that seeded pattern.

(3) **48 test documents.** `eval.n_seqs_clean` and `binding.test_slice.n_seqs`
raised from 16 to 48. This is still below the 64-document population-claim
floor and below the ~59-row L4096 ceiling on WikiText-103 (~245k test tokens /
4097), so population claims stay refused; the document-sampling readout now has
48 units instead of 16.

(4) **Family size.** m=4 -> 6 (two sink contrasts added; the randidx replacement
does not add a comparison). Minimum attainable Holm p 0.001953 -> 0.00293; the
8-pair resolution floor is 0.046875 < 0.05 and remains reported. The pre-test
power gate and complete-coverage rule are unchanged.

(5) **Dynamic/HCA.** Not added to v5 and not runnable here. A separate draft
manifest `confirmation_frozen_dynamic.json` records the design and its
precondition: a disjoint test corpus (or a disjoint, never-read document split),
a new freeze, a pre-test power simulation and the same admission/freeze-lock
machinery. `v12_supp.py` refuses `status=draft` (V12_ALLOW_DRAFT only permits
schema inspection, never a confirmation). Reusing the rows that the fixed v5
study reads would make the second study non-independent; the freeze lock does
not fix that.

The temporary issue list was removed after the fixes.

## 2026-10-22 — review-7: single-factor design gate (no new runs)

Still before any test-split access; no training was run for this revision.
Zero-GPU driver/audit/manifest-preflight paths only: no panel result, no
training recipe, no arm definition value, no seed, no endpoint, no test split
was touched, `CODE_SEMANTICS` stays `v11.128`, and `confirmation_frozen.json`
is byte-identical (its sha256 binding is unchanged).

**Problem.** Every existing pairing gate (`pair_reason` / `train_knobs` /
`hyperparams` / `init_core_fp` / `train_objective` / `_param_control` /
`_compute_control_detail`) verifies what two arms of a declared contrast
SHARE — none of them can see the attention-config DIFF, which is where a
"single factor" silently becomes two. An ablation arm that accidentally gains
a second knob, or whose declared intervention disappears, passes every
record-level gate. The frozen manifest's per-comparison
`mechanism_claim_allowed` flags were hand-written with no machine check
against the actual arm construction.

**Changes.**

(1) **Declared key-set contract** (`exp_lib.ABLATION_INTERVENTION_KEYS` /
`BASELINE_INTERVENTION_KEYS` / `CONTRAST_OUTSIDE_CFG_FACTORS`): every
declared contrast now declares the EXACT AttnCfg keys it is allowed to
differ in. `exp_lib.single_factor_design_audit` diffs both arms layer by
layer and refuses (fail-closed) on extra keys (a second factor), missing
keys (the intervention disappeared) or a missing declaration (a new contrast
without a factor contract). `kind` in the diff marks the contrast
architecture_level (operator swap + mlp-ratio parameter matching outside the
cfg) — the method-vs-matched-dense baselines are architecture-level, never
knob-level single factors.

(2) **Disclosed the previously-undeclared HCA-dynamic differences**: the
dynamic HCA layers train their boundary gates with `comp_lambda_mult=4`
(comp_lambda_effective 0.05->0.2), and their `block_size` field differs
inertly (dynamic segmentation never reads it). Both are now part of the
declared contract and of the ABLATION_DESIGN contrast notes; dynamic-vs-fixed
contrasts keep their §22.4 system-level reading.

(3) **Frozen-manifest flag verification** (`v12_supp.py plan`): each primary
comparison now gets a machine-derived `single_factor` verdict — cfg diff
inside ONE declared intervention family (`exp_lib.cfg_keys_single_factor`,
kind swaps excluded by definition) AND matching actual training objectives
(read off the built models) AND parameter totals matched. A
`mechanism_claim_allowed=true` on a non-single-factor comparison REFUSES the
plan before any P5C/P5T access; the conservative direction (False flag on a
config-single-factor contrast, e.g. allblocks' equal-read-budget caveat)
stays allowed with a note. Current manifest: all six flags verify
consistent; `randidx_seeded`'s `random_salt` is family-internal to
`indexer_mode` (it is the definition of the per-(seed, layer) draw).

(4) **Open-block matrix gate** (`exp_lib.open_block_matrix_audit`): the
declared 2-factor factorial (§24) must vary only along `max_block` /
`open_block_mode` / `open_block_scope` inside the matrix.

(5) **Sweep-contrast labeling** (`v9.SEQ2K_CONTRAST_FACTORS`): the v9/v10
seq2k paired tests now carry an explicit factor column — `topk8 - topk512`
is a single-factor dose response; the two `-m1` contrasts are multi-factor
block-geometry bundles (block_size 4->1, overlap 128->0, topk semantics)
at an equal 32-token budget and must not be read as topk dose responses.

(6) **Audit entry point**: `audit_gates.py single_factor` runs the design
contract + matrix gate + frozen-manifest cfg-level flag check (zero deps on
results_*), and is folded into `audit_gates.py report`.

Verification: the shipped design passes with 15 knob-level single-factor
and 3 architecture-level verified contrasts; the v12 preflight passes with
one conservative note (allblocks); mutating the manifest to flag the main
kind-swap comparison as mechanism evidence makes the preflight REFUSE. New
CPU tests cover the pass case, both fail-closed drifts, the flag-consistency
refusal and the sweep labeling.

## 2026-10-22 — driver-level capacity control + standalone seed floor (no new runs)

Zero-GPU driver/statistics hardening; no panel result, frozen manifest, test
split, README or CI configuration was touched.

(1) **Capacity control on ad-hoc driver contrasts.** `pair_reason` pins the
optimizer / LR schedule / step and token budget, but the v8/v9/v11 statistical
drivers built method-vs-dense comparisons with their own `add_paired` helpers
and quoted a p-value for `csa_fixed - full` even though plain `full` has about
13% fewer parameters than `csa_fixed` (measured on the d=384 panel). The helpers
now attach `params_a` / `params_b` / `param_relative_diff` /
`param_controlled` / `param_capacity_controlled` using the same
`_param_control` + `_param_control_detail` definitions as the methodology
audit, and method-vs-baseline calls opt into `require_param_match=True`. An
unmatched (or unverifiable) pair keeps its raw delta/p but is marked
`capacity_confounded=True`, `method_claim_allowed=False`, and is labelled
"raw comparison only" in the console/v11 report. The parameter-matched dense
arm remains a fair baseline; `full_matched` / `full_sw128_matched` are checked
by the same gate rather than assumed.

(2) **Standalone seed floor.** v8 P2R scheduled only seeds 0/1 and v9 P2T only
0/1, relying on a later driver to lift the key long-context / top-k panels to
n=3. A v8-only or v9-only run therefore reported those comparisons at n=2
(exact sign-flip p floor 0.5), below the project's own 3-seed reporting floor.
Both phases now schedule seeds 0-2, and v9 P2S3 lists the full 0-2 set (resume
skips already-measured cells), so each driver can stand alone at the floor.
Later seed top-ups remain resume-safe.

Verification: new CPU test L63 checks both driver implementations for the
matched/unmatched/unverifiable capacity verdicts; the extended seed-plan test
covers v8/v9/v11 phases. `python test_review_fixes.py` passes 363/363.

## 2026-10-22 — reproducibility hardening: seed normalisation, shared optimizer, environment gate (no new runs)

Zero-GPU code/test hardening; no panel result, frozen manifest, test split,
README or CI configuration was touched.

(1) **The seed the trainer actually uses is normalised.** `train_variant` used
to call `set_seed(seed)` for validation but kept the caller's original object
for the data stream and the record (a NumPy scalar or numeric string could
therefore be recorded differently from the integer it trained with). It now
binds the validated integer returned by `set_seed`. `overfit_guard` validates
its seed up front, and the overfit/reproducibility probes use the same
normalised value.

(2) **DDP rank data seeds are domain-safe.** The old `seed + rank` could
exceed the NumPy/PyTorch `[0, 2**32-1]` domain and could alias a different
declared seed (`seed=0, rank=1` vs `seed=1, rank=0`). Rank 0 keeps the
historical stream; other ranks use a deterministic SplitMix64 derivation
(`data_seed_for_rank`), recorded as `data_seed` on the training record.

(3) **Pre-flights share the production optimizer construction.**
`optimizer_param_groups` / `build_optimizer` now define the three AdamW
groups (decay / no-decay / delta-logit) once. `train_variant`, the small-batch
overfit probe and the duplicate-run reproducibility probe all use it, so a
grouping/`foreach` regression can no longer pass a hand-rolled probe while
breaking real training. `run()` additionally drives ONE fixed window through
the real `train_variant` loop (`production_sanity.json`) and refuses the panel
unless that loop applies updates, keeps the loss/gradient finite and reaches
the same loss/accuracy targets. The hand-rolled probe proves model/loss/label
plumbing; this companion proves optimizer/scheduler/loss-assembly wiring.

(4) **Execution-environment gate for paired seeds.**
`execution_fingerprint` hashes the reproducibility-relevant hardware/software
class (GPU model + compute capability, torch/CUDA/cuDNN builds, dependency
versions, TF32/reduced-precision policy, determinism flags, thread counts;
hostname/process id/UUID excluded). It is recorded per cell (`exec_fp`) and in
`run_hyperparameters`; `pair_reason` now refuses a paired contrast whose two
records were produced under different classes. Legacy records without the
digest keep the old permissive path, and the manifest exposes
`tracking_integrity.with_exec_fp`.

Verification: new CPU tests E1a1 (trainer seed normalisation), E1a4 (DDP rank
seed domain/collisions), E1a5 (environment pair gate), E3a1 (single optimizer
grouping shared by trainer and probes) and E3a2 (the real `train_variant`
loop overfits one fixed batch, and lr=0 fails it). `python
test_review_fixes.py` passes 372/372.

## 2026-10-22 — multiclass F1 for NIAH + pre-registered floor hypothesis (no new runs)

Zero-GPU metric/hypothesis hardening; no panel result, frozen manifest, test
split, README or CI configuration was touched.

**Problem.** Two experimental-design gaps remained. (1) NIAH-natural is a
single-label multiclass retrieval endpoint, but its record only carried top-1
accuracy and macro recall; it did not state a macro/micro/weighted F1, so a
multiclass F1 headline could not be audited. (2) "Beats the tuned count-model
floor" was enforced arithmetically but was not itself a pre-registered
directional hypothesis with a mechanism, endpoint and falsifier, unlike the
component and matched-dense families.

**Changes.** (1) `retrieval_metrics` now records per-class
`predicted`/`precision`/`f1` and named `f1_macro`, `f1_micro`, `f1_weighted`
(plus `f1_averaging`), all computed from the same top-1 labels. Micro F1 is
the pooled TP/FP/FN score and therefore equals accuracy on this single-label
endpoint; macro weights every observed answer class equally; weighted follows
answer support. `METRIC_POLICY.niah_retrieval` lists all three as secondary
endpoints and states why PR-AUC/ROC-AUC are not invented without per-class
scores. `metric_policy_audit` requires all three together, rejects a micro F1
that differs from accuracy, and recomputes the three aggregates from
per-class counts (legacy records without F1 are unverifiable, not a pass).
`v10_supp.py` persists the new fields and prints macro/weighted F1 with the
top-1/macro-recall/floors table. (2) `FLOOR_HYPOTHESIS_DESIGN` registers the
floor comparison as a directional hypothesis (Δ = method − floor,
`predicted_sign=-1`, primary `ppl`, falsifier covering non-comparability and
Holm failure). `methodology_audit` attaches the observed verdict/sign/direction
consistency to `simple_baseline.floor_hypothesis`; `audit_gates.hypotheses`
audits the new family. This does not relax any existing gate: a supported
hypothesis still requires the comparable rows, matching token denominator and
Holm-cleared method-arm contrast.

Verification: `python test_review_fixes.py` passes 372/372 (retrieval F1
identity/aggregate forgery and floor direction checks added);
`python test_open_block.py` passes 15/15; `python audit_gates.py hypotheses`
reports `status=ok` for all five hypothesis families.


## 2026-10-22 — result-analysis loop: curve checks, evidence-based NaN triage, printed bad cases, cost disclosure (no new runs)

Zero-GPU diagnostics/analysis hardening. No frozen manifest, test split, README
or CI configuration was touched; `CODE_SEMANTICS` is unchanged because the
changes only add evidence fields and diagnostic output.

(1) **Curve monitoring is now backed by an explicit data contract.** The
trainer already recorded `losses`/`ppl_history`/`curve_health`; the NaN triage's
first branch still trusted a free-text "data checked" string. `audit_token_data`
now checks dtype, range and NaN/Inf (training stream deterministically sampled
for scale, validation scanned in full), the first real batch verifies the
one-token causal target shift, and `diagnose_training_instability` consumes the
recorded `data_audit` instead of assuming the data were clean. A failed audit,
or a failed label-shift check, is an observed cause; a missing one is
`unknown`, never `not_observed`. `v7_supp.train_warmup` records the same audit.
Overfitting actions now name early stopping explicitly.

(2) **Bad cases are printed, not only written to disk.** `error_analysis.py`
prints the worst tokens (context, target, prediction, top-5, causes, prefix
ambiguity) and worst sequences by default (`--print-samples` /
`--print-sequences` control the budget). Per-token records and the Markdown
artifact now include the target's measured training frequency, so
"annotation/rare target" can be checked against a number.

(3) **Scale analysis reports cost alongside the final PPL.** `m0_supp` panels
now carry `result_diagnostics`: curve-health counts, non-finite/overfitting/
stalled/early-stopped/`final>best` records, and per-arm medians for PPL, params,
training throughput, prefill latency and peak memory, with `≥2x` cost flags.
S1/S2 train records opt into the two-length efficiency sweep
(`measure_efficiency_sweep`); the single-shape measurement remains.

(4) **Report/audit surfaces.** The NaN diagnostics table shows the data-audit
result next to the forward/fp16/gradient suspects, and
`audit_gates.training_health` fails closed on a present-and-failed data audit or
label-shift check (legacy records without the field keep their old warning
path).

Verification: `python test_review_fixes.py` passes 374/374 (data-audit/triage
and scale result-diagnostics tests added); `python v12_supp.py plan` still
admits 72/72 synthetic records with the same six comparisons.

## 2026-10-22 — baseline design preflight + tuned-baseline hypothesis (no new runs)

Zero-GPU experiment-design hardening; no panel result, frozen manifest, test
split, README or CI configuration was touched.

**Problem.** Equal-search tuning was audited only *after* the GPU panel had
run, and the tuned comparison itself had no pre-registered hypothesis.  The
protocol's 8-seed final partition was justified in prose, but nothing checked
the declaration before launch: a duplicated seed, an overlapping search/final
draw, an arm missing from a parameter-matching group, a non-chronological row
fold or a final partition too small for the declared Holm family could all be
discovered only from the finished artifact.  That wastes GPU time and makes an
underpowered contrast easy to launch as if it were confirmatory.

**Changes.** (1) `exp_lib.baseline_tuning_plan` / `simple_baseline_plan` /
`baseline_plan` now validate the *declared design* before any run: arm-group
coverage, unique trial cells, metric/split/direction (`ppl`, validation,
lower-is-better), disjoint search/final seed partitions, the chronological
nested row fold, and the exact seed count needed for the protocol's declared
Holm family.  Structural defects fail the plan; a resolvable but underpowered
final partition is labelled `descriptive_only` (`confirmatory_capable=False`)
rather than silently promoted.  The count-floor plan checks that the policy's
claimed full-factorial grid is actually complete and that the fit cap/policy
are explicit.  (2) `baseline_tune.py plan` runs the preflight and writes
`baseline_plan.json`; `simple`/`full` now refuse a structurally invalid design
before training.  (3) `TUNED_BASELINE_HYPOTHESIS_DESIGN` pre-registers the
tuned-baseline claim (equal search, parameter matching, disjoint held-out rows,
directional PPL endpoint, falsifier), and every PPL hypothesis now names
`metric_policy='lm_perplexity'` instead of relying on an implicit global
mapping.  `audit_gates.py hypotheses` audits the new family.

Verification: new CPU test L25c covers the shipped plan, overlap/duplicate/fold/
group defects, the underpowered-but-launchable case and the incomplete
count-model grid. `python baseline_tune.py plan` reports `status=ok`,
family m=4, final seeds 0-7 sufficient for n>=8.

## 2026-10-22 — hypothesis coverage + inclusive imbalance boundary + baseline ladder (no new runs)

Zero-GPU experiment-design hardening; no panel result, frozen manifest,
test-split access, README or CI configuration was touched.

**Problem.** (1) `hypothesis_audit` could verify that an existing hypothesis
was complete, but nothing checked that every declared contrast family actually
had one: a new ablation component or baseline family could be added to the
design with no pre-registered mechanism/direction, and the audit would still
report `ok`.  (2) The metric policy says the conditional endpoint switches at
"1:1000 or more imbalanced", which is prevalence `<= 1e-3`, but every code path
used a strict `< 1e-3`.  A panel with exactly one positive in 1000 therefore
kept the F1/accuracy headline instead of switching to PR-AUC / macro recall.
(3) The tuning preflight checked arm groups and seed partitions but not that a
group actually contains a baseline partner for its method arm.

**Changes.** (1) `exp_lib.hypothesis_coverage_audit` checks both directions
for the component and baseline designs: every declared family has a hypothesis
(missing), every hypothesis maps to a declared family (orphan), and no family
is empty.  `audit_gates.py hypotheses` now includes the coverage block (and a
direct `hypothesis_coverage` command).  (2) `_extreme_imbalance` is the single
inclusive boundary predicate (prevalence `<= threshold`), used by
`hypothesis_audit`, `retrieval_metrics`, `classification_metrics` and
`metric_policy_audit`; the policy text now states the inclusive boundary
explicitly.  (3) `_plan_arm_groups` now validates each group's `matched`
subset (inside the group, never the method arm) and refuses a method arm with
no baseline/control partner.  `baseline_tuning_plan` records the resulting
`tuning_groups` ladder so the preflight shows the actual method-vs-baseline
pairs before GPU time is spent.

Verification: new CPU tests H6a (coverage missing/orphan/empty, baseline
coverage, fail-closed inputs), H6b (exactly 1:1000 switches to PR-AUC / macro
recall and is rejected as an F1 hypothesis), and extended L25c (per-group
baseline ladder, method-only group refused, matched arm outside group refused).
`python test_review_fixes.py` passes 383/383; `python baseline_tune.py plan`
still reports `status=ok` with family m=4 and final seeds 0-7 sufficient.

## 2026-10-22 — train-only curve monitor + bounded bad-case scan + explicit cost/gain gate (no new runs)

Zero-GPU diagnostics/analysis hardening. No panel result, frozen manifest,
test split, README or CI configuration was touched; `CODE_SEMANTICS` is
unchanged because the additions only record evidence and bound an offline
diagnostic scan.

**Problem.** (1) Curve monitoring still depended on a validation grid.  A
panel with `eval_every=0` recorded per-step `losses` but reported only
`curve_health=unmonitored`, so a finite divergence or a spike that never
reached NaN stayed invisible; the NaN triage also had no train-curve suspect.
(2) The prefix-ambiguity scan in `analyze_errors` built a Python dictionary for
every evaluated position, so the offline bad-case pass could silently become a
multi-hundred-MB cost on a longer validation split.  (3) The efficiency summary
listed a small PPL delta and 2x latency/params/memory as separate tags, leaving
the reader to notice the combination that the methodology prompt explicitly
rules out.

**Changes.** (1) `exp_lib.classify_train_loss_curve` is the train-only fallback:
it reports `nonfinite`, `diverging`, `unstable_spikes`, `stalled`, `descending`
or `insufficient_data` from `losses`, with a noise-aware tail slope.  The
trainer records it as `train_curve_health` on every run and uses it to make
`curve_health` non-finite when only the training curve failed; the NaN triage
consumes it (deriving it from legacy `losses` when the field is absent) and adds
a `train_loss_curve` suspect/action.  `build_report.py` renders the new curve
next to the validation-health table.  (2) `analyze_errors` takes
`context_scan_max_positions` (CLI `--context-scan-positions`, default 2M; 0 =
unbounded) and records `scan_complete`, scanned row/position counts and a
sampling note, so a bounded ambiguity count can never be quoted as a whole-split
total.  (3) Both `m0_supp._panel_result_diagnostics` and
`build_report.efficiency_accuracy_summary` now emit
`cost_increase_without_clear_gain` when a sub-0.5%-or-within-noise PPL change
comes with a >=2x latency, parameter or memory increase; M0 also reports
per-arm seed SEM and `ppl_gain_within_seed_noise`.

Verification: new CPU tests A4m/A4m1 (train-only classifier and trainer
recording), A4m2 (bounded prefix scan and scan metadata), A4m3 (cost-gain flag)
and extended C21b.  `python test_review_fixes.py` passes 387/387;
`python test_open_block.py` passes 15/15.

## 2026-10-22 — unified baseline-claim ladder + observed tuned-baseline hypothesis (no new runs)

Zero-GPU methodology hardening; no panel result, frozen manifest, test split,
README or CI configuration was touched.

**Problem.** The audit computed every component of the "beats a tuned
baseline" claim — the tuned count-model floor, the equal-search tuning
artifact, and the parameter-matched dense contrast — but left them in separate
fields with no single machine-readable verdict. A reader (or a downstream
report) could quote the significant matched-dense row while ignoring that the
method had not beaten the count floor, or that the baseline was verified only
at its untuned default. In addition, the tuned-baseline hypothesis was
pre-registered in the launch plan but never attached to the observed panel
verdict, unlike the floor hypothesis.

**Changes.** (1) `exp_lib.baseline_claim_gate(report)` combines the panel audit
into an explicit three-rung ladder: tuned count-model floor → verified
equal-search tuning → tuned parameter-matched dense contrast. `claim_allowed`
is True only when every applicable rung passes on the same panel/rows; missing
evidence is `unverified` (warn) and counter-evidence (method below the floor,
or a baseline significantly better than the method) is `failed`. The gate is a
pure combination of existing artifacts — it adds no measurement and never reads
the test split. `methodology_audit` now records it as `baseline_claim_gate`,
prints each rung, and emits a failure warning. (2) `methodology_audit` attaches
the pre-registered `TUNED_BASELINE_HYPOTHESIS_DESIGN` entry to the observed
panel as `baseline_tuning.tuned_hypothesis`, with
`supported`/`partial`/`unresolved`/`falsified` status and sign consistency;
`falsified` is reported when a matched dense baseline is significantly better
than the method. This does not relax any existing gate: the previous
`tuned_verified_baselines` / `untuned_verified_baselines` split still drives
the verdict.

Verification: new CPU tests L70 (every rung required; floor failure and
direction conflict fail closed; untuned/verified stays unconfirmed) and L71
(tuned hypothesis is supported on a controlled 7-seed panel, falsified when the
dense baseline wins, and the full claim stays closed while the count floor is
unverified). `python test_review_fixes.py` passes 394/394.

## 2026-10-22 — seed audit, overfit-cache binding and trace verification (no new runs)

Zero-GPU reproducibility/tracking hardening; no panel result, frozen manifest,
test split, README or CI configuration was touched.

**Problem.** (1) The seed contract recorded child-process `PYTHONHASHSEED` but
not whether the *current* interpreter was launched with one; CPython consumes
that variable before `exp_lib` imports, so a normal `python driver.py` launch
remains hash-randomised even though every spawned child is pinned.  The
DataLoader check drew the worker hook in-process and never exercised real
worker processes.  (2) The small-batch overfit cache key covered code, data,
architecture and MLP width but not the execution target, so a CPU pass could
certify a CUDA run in the same process.  (3) A curve sidecar was referenced by
path only: a later edit or copied file could silently replace the curve behind
an otherwise complete manifest.

**Changes.** (1) `exp_lib.hashseed_contract` records the startup value and the
residual current-process gap in provenance/seed_contract (as a warning, not a
false hard failure).  New `dataloader_contract_check` iterates a real
`deterministic_dataloader` with worker processes twice at the same seed and
once at a different seed, checking sampler order plus per-item
Python/NumPy/Torch draws; `seed_contract_check(...,
check_dataloader_workers=True)` embeds it and `run()` enables it when
`dataloader_workers > 0`.  (2) `_overfit_probe_config` now includes the device,
determinism level and `exec_fp`, so a cached PASS cannot cross hardware or
software classes.  (3) `_run_cell` stores `curve_sha256` for sidecars;
`verify_run_tracking(outdir, run_cfg)` independently reads the manifest and
summary back and checks planned cells, code stamp, hyperparameters/RNG map,
data_version, final result and curve digest.  `run()` writes
`tracking_verification.json`, downgrades a panel to `incomplete` if the
read-back fails, and embeds the verdict in the manifest/ledger.

Verification: new CPU tests E5b (real worker-process contract), E5c (hashseed +
worker checks in the seed contract), E5d (overfit cache binds the execution
target) and extended E4b (curve digest and post-hoc tamper detection).
`python test_review_fixes.py` passes 397/397.

## 2026-10-22 — design preflight: token/parameter/read budgets, evidence levels, dynamic hypotheses (no new runs)

**Problem.** The frozen plan already disclosed that the dense reference is
parameter-matched but NOT compute/read matched, and the dynamic/HCA file was
only a draft with non-directional comparisons.  Three design defects remained
before any new GPU spend: (1) the token budget was easy to misread as
`n_train_tokens=8M` although a run consumes `3000*12*512=18.432M` over that
cache; (2) there was no single machine ledger proving which budget each arm
matched, so a mechanism claim could still be attached to a dense-only
comparison; (3) the dynamic draft had no explicit segmentation-quality /
retrieval-precision / long-range-coverage hypotheses and no dedicated
random-vs-random or read-matched controls.  There was also no code-level
registry preventing v6-v11 (selected on the same validation split) from being
re-labelled confirmatory.

**Changes.** (1) `exp_lib.comparison_budget_ledger` records per-arm
`train_cache_tokens`, `consumed_tokens`, real parameter totals and an analytic
read budget, classifies each comparison as selection/segmentation/task-policy/
sink/parameter-proxy/allblocks, and refuses `mechanism_claim_allowed=true`
unless token, parameter and read budgets match and a dedicated control exists.
(2) `exp_lib.evidence_level_audit` + `EVIDENCE_LEVELS` +
`EXPLORATORY_PANEL_REGISTRY` make v6-v11 exploratory by code; a draft manifest
or a draft comparison can never be confirmatory, and a registry panel summary
that says `confirmatory` fails the audit. (3) `exp_lib.dynamic_design_audit`
requires H1 segmentation quality (boundary F1), H2 retrieval precision (NIAH),
H3 long-range coverage (PPL vs a read-matched fixed arm), plus H4/H5, the
dedicated control arms and a disjoint-source/measured-calibration/passed-power
precondition. (4) `v12_supp.plan` now runs the budget/evidence preflight before
training; new `v12_supp dynamic-plan` audits the draft without treating it as a
confirmation; `audit_gates design` is the unified zero-GPU entry point.
(5) `v7_supp` gains rope-consistent dynamic control arms
(`csa_dynamic_rope_randidx_seeded`, `csa_dynamic_rope_task`,
`hybrid_dynamic_rope`, `csa_fixed_rope_topk{k}`) so the dynamic draft never
mixes rope and non-rope arms.

**No change to** `CODE_SEMANTICS`, `confirmation_frozen.json` arms/seeds/
endpoints/statistics, or any measured result.  The dynamic manifest remains
`status=draft`, `evidence_level=draft`, `runnable=false`; it must not be quoted
as a confirmation.  Verification: new CPU tests L82-L86; the frozen plan still
admits 72/72 synthetic records and the dynamic-design audit passes as a
non-runnable draft.

## 2026-10-22 — split-size reconciliation at scale (no new runs)

**Problem.** `data_guard._resolve_sizes` accepts split fractions whose sum is
within `1e-6` of 1.  That tolerance is harmless at small row counts, but at
large `n` a tiny overshoot changes integer floors: `0.01 + 1e-9 + 0.99` is
over 1 by `1e-9`, and at `n=742878783538` the exact floors sum to `n+741`.
The old code raised `inconsistent split sizes [...]` instead of returning a
valid chronological split, even though the requested fractions were accepted.
This would make a scaled-up corpus fail only after its row count grew.

**Changes.** `_resolve_sizes` now reconciles an over-allocated floor sum in
bulk, proportional to each split's slack above its one-row floor, rather than
raising.  Bulk matters: the accepted 1e-6 fraction slack can be ~1e12 rows on
a 1e18-row corpus, so a one-row-at-a-time loop would hang.  Empty/zero-fraction
splits are never touched, requested splits keep their one-row guarantee, and
the normal under-allocated path is unchanged.  Exact target sizes are computed
once as `Fraction`s, so no float product is re-evaluated during reconciliation.

Verification: extended `t_split_targets_use_exact_fraction_arithmetic` with
the large-n `0.01/1e-9/0.99` regression, the small-n one-row preservation case,
and the unchanged exact `0.8/0.1/0.1` result at `2**53+1`.  Both CPU suites
pass unchanged: `python test_review_fixes.py` 411 passed / 3 environment skips,
`python test_open_block.py` 15/15.

## 2026-10-22 — memo value identity + transport reference + aggregation quarantine + run-state reconciliation (no new runs)

Zero-GPU reproducibility/engineering hardening; no panel result, frozen
manifest, README or CI configuration was touched.

**Problem.** (1) `_ids_fp` memoised on `(data pointer, layout)` alone.  An
in-place write to a writable array left the old digest in the memo (and the
on-disk `.idsfp.json` sidecar path was also trusted), so the same array could
be fingerprinted as the pre-edit corpus.  That stale digest then enters
`run_cfg`, the manifest and the resume check — a pure function that was not
actually a function of the current values.  (2) `batch_contract_check` checked
shape/shift/repeatability but not the central transport claim: that narrowing
gather indices to int32, pinning/uploading the corpus and prefetching do not
change a token.  A wrong gather could still pass as “shifted and repeatable”.
(3) `_dump_aggregate` had a fallback that dropped some unusable records, but a
mixed evaluation-slice error (or a cross-protocol pairing error) could still
raise after training and leave the panel with no aggregate.  Nothing counted
the quarantine or stated the tolerance boundary.  (4) A failed pre-flight on an
existing panel only appended a ledger event; a panel killed in `running` state
stayed `running` forever, so run-state was not complete.  (5) Admission did
not bind a record to unique weights: the same checkpoint digest could be
re-labelled as another arm/seed.

**Changes.** (1) `_ids_fp` only trusts the memo/sidecar for read-only arrays
(`flags.writeable=False`); writable arrays are hashed fresh every call, so an
in-place mutation necessarily changes the digest.  Production memmaps keep the
fast path.  (2) `batch_contract_check` now builds an independent plain-int64
NumPy causal batch from the SAME seeded starts (no pin, no prefetch, no int32
narrowing) and compares every token and target; the report records the
transport dtype and per-batch equivalence.  A loader that changes one token
now fails the contract.  (3) `aggregate_with_exclusions` validates each record,
keeps the first measured record for a duplicated `(arm, protocol, seed)`,
splits groups that disagree on a corpus/evaluation fingerprint instead of
pooling them, retries a cross-protocol pairing group per arm, and returns a
JSON report (`n_input`, `n_admitted`, `n_excluded`, reason counts and named
keys).  `_dump_aggregate` always writes `aggregation_exclusions.json` (and
`summary.aggregatable.json` when anything was quarantined); no corrupt record is
silently averaged, and a bad record no longer crashes aggregation.  (4)
`_record_failed_panel` reconciles a `running`/`incomplete` entry to `failed`
while preserving its measured cells; an already `complete` panel is never
rewritten, and the failed attempt remains in the ledger.  (5) Confirmation
admission validates `ckpt_sha256` when present (or when
`binding.require_ckpt_sha256` is declared), excludes missing/malformed
digests, and refuses one digest reused by another `(variant, seed)`.  The
loaded manifest now carries `_path`, so the freeze lock records the actual
manifest file rather than the default filename.  (6) Final-test records now
carry `exec_fp` plus the compact environment snapshot, and resume treats a
present-but-different execution class as stale; evaluation provenance is no
longer weaker than training provenance.  (7) Scale geometry is
parameterised: `apply_scale_geometry_overrides` accepts validated
`B_SEQ_LEN`/`B_BATCH_SIZE`/`B_N_TRAIN_TOKENS`/`B_STEPS`/`B_VAL_SEQS`
overrides, preserves the sampled-window budget when only geometry changes,
records the override in the cfg, and the M0 driver discovers `RUN_S<n>`
panels instead of hard-coding S1/S2 (future tiers with no cap get their own
state file).

Verification: new CPU tests `L59b` (in-place mutation changes the id digest;
storage width/read-only status do not), `E2c3` (int64/uint16 reference
equivalence and a one-token-corrupting loader fails), `L59c` (corrupt,
unmeasured, duplicate and mixed-slice records are counted/split, not pooled),
`E4b2` (running → failed reconciliation; complete stays complete), `L53b`
(reused/missing checkpoint digests), and `L72b` (geometry override validation
plus future-panel discovery).  `python test_review_fixes.py` passes with the
same 3 environment skips; `python test_open_block.py` 15/15.

## 2026-10-22 — scale-safety review: deterministic reductions, row identity, artifact paths (no new runs)

**Problem.** Several helpers were correct at the small current scale but lost
information as counts/context grew. (1) `csa_kernels._det_scatter_add_multi`
and `_seg_apply` always accumulated in fp32 and cast to fp32 regardless of the
requested dtype; a global prefix from an early segment erased later small
segments, and float64 requests were silently downcast. (2) `_stable_topk_desc`
left its scatter buffer uninitialised for NaN scores. (3) Row identity in
`data_guard._row_keys` depended on numeric storage dtype (equal int64/float
rows had different keys), used only 64-bit digests (birthday collisions become
material at 1e9 rows), and `split_content_fingerprints` returned `None` after
a canonical-fingerprint failure. (4) `experiment_provenance` computed
`corpus_fp` before adding `data_extra`; a read-only view of writable storage
could keep a stale `_ids_fp` memo; `_ids_narrow` trusted only an upper bound
and wrapped large negative ids; `_audit_sample_flat` exceeded tiny budgets.
(5) `v7_supp.eval_length_gen` sliced a short panel but still allocated for the
requested length. (6) `m0_supp._ladder_params` used S1's `seq_len` for every
tier. (7) Result-dir audits resolved `_iter_result_dirs()` names against the
process CWD, and the document-overlap gate certified a split after scanning a
tiny fraction of a large train stream.

**Changes.** Deterministic scatter helpers now honour the requested output
dtype and accumulate floating sums in float64 (same stable order, then one
cast); non-finite `_stable_topk_desc` inputs delegate to `torch.sort`.
`_row_keys` hashes exactly representable integer rows through the same float64
bytes as float rows, keeps large-int precision, uses 128-bit row digests, and
provides a raw dtype/shape/byte fallback for failed content fingerprints;
native and object numeric rows share one identity. `experiment_provenance`
inserts `data_extra` before the composite fingerprint; `_ids_fp` walks the
base chain before trusting a read-only memo/sidecar; `_ids_narrow` checks both
int32 bounds while copying; `_audit_sample_flat` never exceeds its budget.
`eval_length_gen` reports an explicit cell error when the cache is shorter
than the requested span. `_ladder_params` uses `RUN_<tier>`'s sequence length
when present. Audit artifact loads resolve relative names against `REPO` (an
existing CWD path keeps its explicit meaning), `max_rows` caps validation/test
overlap scans with the sample size recorded, and a document-overlap scan below
25% coverage is `not_verifiable_locally` rather than clean.

Verification: new CPU tests L87–L93 plus the existing suite.
`python test_review_fixes.py` passes 426/429 with the same 3 DataLoader
environment skips; `python test_open_block.py` passes 15/15.

## 2026-10-22 — hypothesis/evidence/budget preflight hardening (no new runs)

**Problem.** The design preflight from the previous pass still allowed four
methodological loopholes. (1) A new dynamic comparison could omit
`evidence_level` and silently inherit the manifest level; its endpoint or
signed contrast was not tied to the declared hypothesis. (2) The declared
p-floor/Holm floor and the projected power were prose, not recomputed from the
frozen n/m/alpha. (3) A dynamic-vs-fixed mechanism claim only checked that a
`read_budget_calibration` object existed, so `status=pending` was enough.
(4) The H3 long-range-coverage claim had no PPL contrast against the concrete
read-matched fixed arm — it still relied on the fixed/dense references — and
`csa_fixed_rope_topk32` was not yet recognized as a single-factor
segmentation family.

**Changes.** (1) `exp_lib.planned_power_audit` recomputes the exact two-sided
floor, the Holm floor, the minimum-seed floor, the per-q `power_first_holm_gate`
table and the projected power; a plan whose projection is below
`min_power_for_test`, or a `power_gate.status=passed` label without numeric
power, fails. (2) `dynamic_read_budget_calibration` verifies the measured mean
dynamic block length, the `k_fixed = round(topk*mean/fixed_block_size)` formula
and the existence of the concrete `csa_fixed_rope_topk{k}` arm; pending
calibration is draft-only. (3) `dynamic_design_audit` now requires an explicit
`evidence_level` per comparison, an endpoint and signed orientation matching
the hypothesis (`contrast_arms`), a scheduled comparison for every required
hypothesis, a `by_len.ppl` dynamic-vs-read-matched-fixed contrast, and
`mechanism_claim_eligible=true` before any mechanism flag. H3/H5 are explicitly
system-level. (4) `comparison_budget_ledger` checks the served cache budget
globally, rejects boolean token counts, records the 6ND/local-window deltas and
design cost, and only permits a mechanism claim on a confirmatory manifest with
a measured calibration. (5) The dynamic draft records those fields and its
8-comparison power projection; it remains `status=draft`, `runnable=false`,
`promotion_ready=false`. No frozen v12 arm/seed/endpoint/statistic changed.

Verification: new CPU tests L94-L97 (explicit mapping/orientation, power and
calibration formula, mechanism-claim confirmation, evidence-family identity).
`python test_review_fixes.py` passes 430/433 with the same 3 DataLoader
environment skips; `python v12_supp.py dynamic-plan` reports the draft as
coherent but non-runnable.

## 2026-10-22 — scale cast/geometry hardening + malformed-design fail-closed (no new runs)

**Problem.** Four paths still assumed values that hold at the current scale.
(1) `data_guard._matrix_fingerprint` cast every signed-integer block through
float64 to int64 without a range guard.  `int64_max` maps to `2**63` (outside
int64), so the cast overflowed/warned and the same applied to the largest
`datetime64[ns]`; `_validated_index` guarded only the positive float boundary,
so a far-negative float reached the same implementation-defined cast before the
negativity check.  (2) The host pooling geometry builders (`_pool_parts_flat`,
`_pool_geom_flat_var`, `_pool_geom_flat_single`, `PoolGeometry`) packed every
table into int32; a sequence length / count beyond that range wrapped negative
instead of widening.  (3) `dynamic_design_audit` put comparison arm values into
a `frozenset` and tested them against the arm set without validating they were
strings, so a malformed list/dict arm raised `TypeError: unhashable type`
instead of failing closed.  (4) `_admit_confirmation_records` assumed the bylen
payload was an object and `_by_len_shape_reason` converted an unbounded
`n_seq` through float outside a guard, so malformed records could crash
admission rather than be named/excluded.  (5) `_det_segment_sum` — the eager
twin of the fused deterministic scatter — still accumulated a global fp32
prefix in the output dtype, so a large early segment erased later segments.

**Changes.** The fingerprint/index casts now guard both int64 boundaries and
route inexact values to the existing exact-tail path (identical digests, no
overflow warning).  Host pooling geometry selects int32 while the geometry and
its pairwise sums fit (`(2**31-2)//2`) and int64 otherwise; the H2D upload and
prefetch packer follow the chosen width, and `PoolGeometry` stores int64 when
needed (the Triton pool kernel, whose lane arithmetic is int32, now refuses
such a geometry explicitly instead of truncating it).  `dynamic_design_audit`
names a non-string/empty `a`/`b` and keeps auditing the remaining comparisons.
Confirmation admission rejects a non-object bylen payload with a clear
`SystemExit`, and the by-length `n_seq` read catches float overflow.
`_det_segment_sum` accumulates floating/complex sums in float64/complex128 and
casts only the final segment sums, so its output dtype is unchanged while the
later-segment precision matches the fused helper.

Verification: new CPU tests L98-L103 (unhashable arm values, int64/datetime
fingerprint boundaries, int32→int64 pool geometry, float index boundaries,
non-object confirmation payload, wide segment-sum accumulation).  `python
test_review_fixes.py` passes 436/439 with the same 3 DataLoader environment
skips; `python test_open_block.py` passes 15/15.

## 2026-10-22 — measurement identity, paired aggregation and confirmation parameter binding (no new runs)

**Problem.** Three analysis-layer paths let "paired" or "confirmed" mean less
than the frozen protocol says. (1) `_dump_aggregate` uses
`aggregate_with_exclusions`, whose old grouping split each variant first and
then called `aggregate` on one arm at a time. Every subgroup therefore lacked
the `full` baseline, so the shipped panel dumps silently lost
`dPPL_vs_full` / `n_paired` even though a direct `aggregate` computed them.
(2) Two records with different `_code` stamps were pooled as extra seeds by
`aggregate`, the arm summaries and `controlled_variables_report` when their
`run_cfg` happened to match; legacy records without the suffix were treated as
mergeable with current measurements. (3) `aggregate` and `methodology_audit`
tested the set `{False}` for truthiness, labelling every ordinary
full-validation arm as `nested_validation_holdout`. (4) Confirmation admission
did not require `params` even when the frozen plan discloses parameter
matching, so the decision rule's `param_matched is not false` could be
satisfied by `param_matched=None`; one seed's parameter count could also
certify an entire arm.

**Changes.** (1) `aggregate_with_exclusions` now groups by
`(protocol, fingerprint class)` ACROSS variants, so all arms of one controlled
class enter the same `aggregate` and paired dPPL survives; different classes
are still split with a shared `@fp<digest>` suffix and named in
`fingerprint_split_groups`, and the per-arm fallback still refuses a dPPL when
pairing is undefined. (2) `measurement_semantics_key` makes the `_code` stamp
part of `_cfg_fp` / `_cfg_fp_noparam`, the aggregation fingerprint class and
the arm-summary identity; `controlled_variables_report` exposes
`code_equal` / `code_values` and marks differing, within-arm-mixed or
one-sided stamps as uncontrolled (both absent stays the legacy `None` path).
(3) The nested flag is compared to `{True}` rather than tested for truthiness.
(4) When a non-descriptive manifest declares
`compute_budget_disclosure.parameter_matching`, admission requires a positive
`params` on every record (missing counts are named exclusions; descriptive
smoke plans and legacy plans without the disclosure keep the permissive path).
`finaltest_stats` computes `param_matched` only when every usable seed on both
sides carries a parameter count and each arm is internally consistent; it
records `params_by_seed`, `param_consistent`, `param_match_complete` and
`param_match_reason`, and the top-level `parameter_binding` states whether the
manifest requires the binding.  For a manifest that discloses parameter
matching, a confirmed verdict additionally requires
`param_match_complete is True`: `param_matched is not false` alone still
accepts `None`, which is the unknown-as-pass path being closed. The frozen decision text, manifest bytes, arms,
seeds, endpoints, alpha and Holm family are unchanged; `CODE_SEMANTICS` is not
bumped because no training/measurement value changes.

**Verification.** New CPU tests L104 (wrapper restores paired dPPL, mixed code
stamps split, nested estimand label), L105 (code semantics in arm/control
identity) and L106 (confirmation parameter binding + per-seed consistency).
`python test_review_fixes.py` passes 439/442 with the same 3 DataLoader
environment skips; `python test_open_block.py` passes 15/15;
`python v12_supp.py plan` admits 72/72 synthetic frozen cells and
`python v12_supp.py dynamic-plan` still reports the draft as non-runnable.

## 2026-10-22 — confirmation-verdict robustness: fragility + seed-drop Holm jackknife (no new runs)

Zero-GPU results-analysis hardening; no panel, frozen-manifest byte
(`confirmation_frozen.json`), test split, training path, `CODE_SEMANTICS`,
README or CI configuration was touched. The frozen `confirmed` verdict is not
recomputed; the new block is written beside it as a post-hoc diagnostic.

**Problem.** `finaltest_stats` reported a per-seed delta vector, a mean, an
exact sign-flip p, Holm p and a document bootstrap, but nothing quantified how
much the frozen verdict depended on one seed. `paired_effect_stats.seed_driven`
only flips the sign of the MEAN and was never wired into the frozen confirmation;
it also cannot see the case where the mean keeps its sign but the exact p (or
the whole Holm rejection set) crosses alpha because of one seed. A 12-seed panel
supported by 11 tiny same-direction deltas plus one very large one can therefore
look far more robust than it is. Separately, the per-document readout gave a
between-document interval but not whether a single document carried the mean.

**Changes.** (1) `exp_lib.paired_fragility` reuses the exact sign-flip test on
every subset and reports: `drop_fragility` (largest number of least-influential
|Δ| seeds removable while p<alpha), `flip_fragility` (smallest-|Δ| same-direction
signs that must flip before p>=alpha or the mean reverses; explicitly labelled a
greedy influence bound, not an exact minimum), per-seed LOO `p_shift=|Δp|` /
`removes_significance` / `flips_direction`, the t interval and its zero
exclusion, and `most_influential_seed` ranked by ABSOLUTE p shift (removing a
contrary seed that turns a null panel significant is single-seed dependence too,
which the signed p-increase alone hid), with `most_influential_tied_count` so an
equal-magnitude tie is not reported as one uniquely guilty seed. Non-finite
deltas drop their seed label with them; non-integer/duplicate/mismatched seeds
and invalid alpha fail closed.
(2) `exp_lib.holm_decision_jackknife` drops one seed at a time from every
comparison containing it, re-runs the exact test and the FULL Holm family, and
records `rejected_full`, per-fold `lost/gained`, `always/sometimes_rejected`,
`unstable_folds` and per-comparison `p_holm_min/max`. It replays the frozen
strict `p_holm < alpha` rule. `family_names` keeps a declared-but-omitted
comparison in the family at p=1, so a missing arm cannot shrink m and loosen the
correction. (3) `exp_lib.confirmation_robustness` composes both, keeps the
declared `family_size` (and refuses one smaller than the supplied comparison
set), and adds the equal-magnitude projection margin
(`min_same_sign_for_family_gate`, `same_sign_margin`) only as a planning
reference. (4) `v12_supp.finaltest_stats` attaches `stats['robustness']` AFTER
`confirmed` is computed; `confirmed`, raw p, Holm p, alpha and the family are
untouched. The complete-coverage gate is deliberately not replayed (dropping any
seed makes coverage false by construction); instead the block reports
`confirmed_without_coverage_full/by_fold/stable` and a `coverage_gate_note`. (5)
The v12 report renders a labelled post-hoc table (Δ mean [95% CI],
same-direction count, drop/flip fragility, LOO Holm range, most influential
seed, unstable seeds) via
the testable `_robustness_markdown` helper; when a Holm rejection is not
stable it additionally prints that the verdict may not be described as
"seed-robust" (while the pre-registered decision itself still stands). (6)
`_doc_level_analysis` adds
`doc_delta_negative/positive/zero`, `doc_a_better_fraction`,
`per_doc_delta_median` and `top_doc_abs_share` so a mean carried by one document
is visible; the main p5t table's document-CI cell also shows the share of
documents with Δ<0.

**Boundary.** These are descriptive sensitivity readouts, not a second primary
analysis: no new claim can be confirmed by them, and a fragile frozen verdict is
still reported as the pre-registered result — but the fragility must be printed
next to it. `flip_fragility` is greedy (definition shipped with the number);
`drop_fragility` is a deletion count, not an "add k events" fragility index.

**Verification.** New CPU tests L107 (fragility, seed alignment, invalid inputs,
LOO, n=6-vs-n=12 family jackknife, omitted comparison keeps its family slot),
L108 (v12 integration appends only `robustness`; `confirmed`/`holm` unchanged;
coverage gate not replayed; Markdown rendering; empty inputs) and L109
(document-level breadth + mismatched-row refusal).
`python test_review_fixes.py` passes 443/446 with the same 3 DataLoader
environment skips; `python v12_supp.py plan` still admits 72/72 synthetic frozen
records with the same six comparisons.

## 2026-10-22 — exact causal-quantile rank positions at scale (no new runs)

**Problem.** `_causal_quantile_exact` converted the integer column index to
float32 before computing the quantile rank (`pos = q * cols.float()`).  That
looks harmless at the default target of 16 (a power of two), but it is not a
valid index computation: for `target_block_tokens=41`, `1/41 * 41` rounds to
`0.99999994`, so the exact rank `1` became `0` with `frac≈1` and the quantile
interpolated from the wrong pair of order statistics.  The same cast also
loses the column itself once `c >= 2**24` (adjacent columns collapse to one
float32 value), which silently changes the dynamic-chunk boundary policy on a
long-context scale-up.

**Changes.** `exp_lib._causal_quantile_positions` now computes
`den = min(max(int(target_block_tokens), 1), 1000)` — exactly the branch where
`q = max(1/target, 0.001)` — and obtains `lo = c // den`, `rem = c % den` with
integer arithmetic (no float index).  `frac = rem / den` is cast to the work
dtype for interpolation.  `_causal_quantile_exact` uses this helper for both
the rank and the sort path, so an exact integer rank has `frac == 0` and uses
the lower order statistic directly.  The default target=16 path is value
identical (its float32 product was already exact); no frozen manifest, arm,
seed, endpoint, statistic or `CODE_SEMANTICS` changed.

**Verification.** New CPU test L110: an end-to-end quantile with target 41
selects exact ranks at columns 41/82 (the old float32 path returned
99.99999/199.99998 instead of 100/200), and the integer positions remain exact
at columns above `2**24` and `2**53`.  `python test_review_fixes.py` passes
444/447 with the same 3 DataLoader environment skips.

## 2026-10-22 — document-level evaluation from real document boundaries (no new runs)

Zero-GPU measurement/analysis hardening; no panel, frozen manifest byte
(`confirmation_frozen.json`), training path, `CODE_SEMANTICS`, test split,
README or CI configuration was touched. The pooled PPL is unchanged; this
entry fixes the UNIT used by the "per-document" uncertainty readout.

**Problem.** `eval_length_gen` returned `ppl_per_seq` and `_doc_level_analysis`
called each entry a "document", but the entries are fixed evaluation ROWS.
A row is an arbitrary window over the token stream: one WikiText document can
span two rows (so its loss is counted twice by a row bootstrap), while one row
can contain the tail of one document and the head of another. The test loader
persisted only a document-level sidecar (which documents contributed), so the
analysis had no way to map a scored token to its document. The result was
pseudoreplicated "document" CIs: the point estimate was right, but the
uncertainty was computed on the wrong sampling unit and the `n_docs` count
could differ from the number of documents actually evaluated.

**Changes.** (1) `load_wikitest` now records a per-TOKEN document-id sidecar
(`test_token_doc_ids_*.npy`, one id per served token; 48 x 4096 ids is ~1.6 MB)
alongside the existing document-level sidecar, binds it by count + fingerprint
in `test_meta.doc_ids`, and verifies both on every cache hit. The builder
checks that the per-token and per-document sidecars name the same document set
and that the per-token length equals the served slice exactly. Legacy caches
without it remain usable and explicitly fall back to row-level uncertainty;
a cache whose metadata DECLARES the sidecar but whose file is missing or
misaligned fails closed. (2) `exp_lib._token_doc_ids_for_capped_rows` builds
the mapping from the same per-row `len(ids)+1` lengths that produced the
token stream, so the EOS token is attributed to the document that emitted it.
(3) `exp_lib.document_index` gives a compact, deterministic document index for
arbitrary uint64 ids; `exp_lib.aggregate_nll_by_document` groups per-token NLL
by document and returns `nll_per_doc`, `n_tokens_per_doc`, `ppl_per_doc`,
`doc_ids`, `n_docs` and a fingerprint. It uses the shared deterministic
segmented-sum helper, so the grouping does not depend on chunk order.
(4) `v7_supp.eval_length_gen(..., doc_ids=...)` attaches those fields to every
by-length cell (including the truncated-`max_pos` branch) while leaving the
pooled `ppl`, `n_tok`, `nll_sum`, `nll_per_seq` and `ppl_per_seq` values
untouched. (5) `v12_supp.P5T` requests the mapping from `load_wikitest` and
passes it through; `_doc_level_analysis` now prefers `ppl_per_doc` and refuses
mixed document/row granularity or mismatched per-document token counts/id
order instead of pairing by position. The public `doc_sampling` dict records
`granularity`, and the P5T table/notes say "row" when the cache is legacy.
(6) `exp_lib._token_doc_ids_from_eos` reconstructs the same mapping on a
legacy cache from the construction contract (one EOS per complete document,
the cap-truncated last document having none) and the document sidecar; it is
used only when the EOS count/order and the document set both verify, so an
existing frozen cache does not need a rebuild just to get document-level
uncertainty. A failure to verify remains the explicit row-level fallback.
(7) Admission (`_by_len_shape_reason`) checks that a claimed document-level
cell's per-document token counts partition exactly `n_tok`; the audit gate
re-verifies the per-token sidecar's length/dtype/fingerprint. `eval_ppl`'s
per-ROW statistics keep their old name/behaviour but their docstring now says
"row", not "document".

**Boundary.** This is a data-processing correction to the uncertainty unit,
not a new endpoint: the primary pooled PPL and the frozen decision rule are
byte-for-byte unchanged, and no `CODE_SEMANTICS` bump is needed (no training
or measurement value changed). A legacy test cache without the sidecar is
mapped through the verified EOS construction contract when possible and
otherwise labelled row-level; it is never silently upgraded, and a fresh
cache always gets the document-level statistic.

**Verification.** New CPU tests L111 (per-token sidecar alignment/cap/EOS),
L112 (documents spanning rows and rows mixing documents; token-weighted pooled
PPL identity; truncated branch; one-doc-per-row equivalence with
`ppl_per_seq`; invalid mapping refusal) and L113 (analysis prefers document
cells, refuses mixed granularity and mismatched document units). The unchanged
L109 still covers the row-level fallback.
`python test_review_fixes.py` passes 447/450 with the same 3 DataLoader
environment skips.

## 2026-10-22 — token-pooled Δ interval at the document level (no new runs)

Zero-GPU statistics/measurement hardening; no panel, frozen manifest byte
(`confirmation_frozen.json`), training path, `CODE_SEMANTICS`, test split,
README or CI configuration was touched. The pooled point estimate and the
frozen decision rule are unchanged; this entry fixes the ESTIMAND of the
document-sampling interval printed next to the primary comparison.

**Problem.** `v12_supp._doc_level_analysis` built `ppl_per_doc`, averaged the
per-seed arm difference with EQUAL weight per document, and bootstrapped those
unit means. The primary endpoint is token-pooled PPL
(`exp(sum NLL / sum tokens)`), so when document lengths differ "the average
document's ΔPPL" and "the pooled ΔPPL" are different estimands. The report
printed the equal-weight interval next to the pooled Δ mean and the exact
sign-flip p, so the interval and the tested effect described two different
quantities. The single-arm diagnostic `_nll_doc_ci` already used a
cluster-robust ratio-of-totals interval; the paired comparison did not.

**Changes.** (1) `exp_lib._paired_nll_doc_ci(nll_a, nll_b, n_tokens)` computes
the mean over seeds of each seed's token-pooled PPL difference and a
document-cluster delta-method interval: per-seed influence
`exp(ce_a)*(a_i - ce_a*t_i)/T - exp(ce_b)*(b_i - ce_b*t_i)/T`, averaged over
seeds, variance `D/(D-1)*sum_i psi_i^2`, t critical value with `df=D-1`, and
the corresponding seed-averaged log-ratio / PPL-ratio interval. It is
deterministic (no bootstrap), conditions on the trained seeds, and validates
shapes, finiteness, positive token counts and `conf`. (2)
`v12_supp._doc_nll_matrix` checks that every per-document NLL vector is
finite, reconstructs the cell's recorded `nll_sum`, and is consistent with
`ppl_per_doc = exp(nll_i/t_i)`. (3) `_doc_level_analysis` attaches
`pooled_delta_ppl`, `pooled_delta_ppl_ci95`, `pooled_delta_log_ratio`,
`pooled_delta_log_ratio_ci95`, `pooled_ratio_ci95`, `pooled_ci_n_docs`,
`pooled_ci_df` and `pooled_ci_method`; if the vectors do not reproduce the
recorded per-seed pooled Δ, it emits `pooled_ci_unavailable_reason` instead of
a wrong interval. Existing fields (`per_doc_delta_*`, `per_doc_*_ci95`,
breadth counts) are unchanged. (4) Admission (`_by_len_shape_reason`) now
rejects a document cell whose `nll_per_doc` is non-finite or does not
partition its recorded `nll_sum`, and a cell whose `n_tokens_per_doc`
contains a zero-target document (the aggregator drops such documents rather
than reporting them as evaluation units). (5) The report uses the pure helper
`_doc_ci_markdown_cell`: the pooled cluster interval when available, otherwise
the explicitly labelled equal-weight unit bootstrap; the table header and a
note state which estimand each column is. (6) Row-level cells still get only
the labelled row bootstrap; no document interval is claimed for them.

**Boundary.** This is a descriptive document-sampling interval for the
observed document panel, not a second confirmatory test: the seed-level exact
sign-flip, Holm family, coverage gate and decision rule are untouched, and the
new interval is not used to confirm or reject any claim. The equal-weight
per-document mean and bootstrap remain available as a "mean document"
description; they are no longer presented as the interval of the token-pooled
effect. No `CODE_SEMANTICS` bump is needed because no training or measurement
value changed.

**Verification.** New CPU tests L115 (paired cluster CI: token weighting,
multi-seed point identity, degenerate/overflow shapes, input guards), L116
(document-level integration: pooled point equals the per-seed pooled Δ,
cluster CI present, equal-weight interval differs, inconsistent NLL/PPL
vectors are refused rather than silently used, row fallback stays row-level)
and L117 (report helper prefers the pooled interval and labels the fallback).
Extended L113 covers the new admission checks.
`python test_review_fixes.py` passes 451/454 with the same 3 DataLoader
environment skips; `python v12_supp.py plan` still admits 72/72 synthetic
frozen records with the same six comparisons.

## 2026-10-22 — device block orbit bound and pooled-CI provenance guard

Zero-GPU correctness review; no training run, frozen manifest, test split,
`CODE_SEMANTICS`, README or CI configuration was touched. Two latent bugs were
found and fixed.

**Bug 1 (scale).** `_device_segment_geometry` preallocates the successor orbit
as a power-of-two table sized from
`k_max = ceil(n / min_block) + 1`. That bound assumes `max_block >= min_block`
(the validated `AttnCfg` invariant), but the function is an exact replica of
`_segment`, which accepts any integer bounds. When `max_block < min_block`, the
successor `f(p) = min(p+max_block, max(p+min_block, cut+1))` takes the first
branch at every step, so the effective stride is `max_block`. For a long
sequence the preallocated orbit then covers only `~n/min_block` starts while
the true block count is `~n/max_block`; the device result silently truncates
and collapses every remaining token into one oversized final block. The fix
bounds the orbit by the smallest possible stride,
`max(1, min(max(min_block, 1), max(max_block, 1)))`, which is exact for
negative/zero bounds too (both segmenters floor the step at one).

**Bug 2 (provenance).** `_doc_level_analysis` published the token-pooled
cluster interval even when it could not reproduce the primary endpoint: if a
cell's recorded per-seed `ppl` was missing or non-finite, `_pt` became empty
and the `else` branch accepted `_pc` without the point-estimate identity
check. A missing primary value is not evidence that the document vectors match
it. The branch now emits `pooled_ci_unavailable_reason` and withholds the
interval, matching the existing refusal for mismatched vectors.

**Boundary.** Both fixes are descriptive/geometry corrections: the primary
pooled PPL, the frozen decision rule, the seed-level exact sign-flip and the
Holm family are unchanged, and no measurement value changes. `AttnCfg` already
rejects `max_block < min_block` for normal model configurations; the fix makes
the internal device replica honour its documented contract for any bounds the
host segmenter accepts, including future scale experiments that call the
segmenter directly.

**Verification.** Extended L116 (missing/non-finite primary `ppl` now refuses
the cluster interval) and new L118 (device vs host segmenter at `n=10003` for
`max_block < min_block`, dense/sparse cuts, honoured masks, and the valid
`max_block >= min_block` regression). A property test over 3,000 random
`(n, min_block, max_block, cut-table)` cases, including `n=10003` and
`max_block < min_block`, found no further device/host mismatch.
`python test_review_fixes.py` passes 452/455 with the same 3 DataLoader
environment skips; `python v12_supp.py plan` still admits 72/72 synthetic
frozen records with the same six comparisons.

## 2026-10-22 — matched-initialization evidence reaches the confirmation test (no new runs)

Zero-GPU validity/provenance hardening; no panel, frozen manifest byte
(`confirmation_frozen.json`), training or evaluation numeric path,
`CODE_SEMANTICS`, test split, README or CI configuration was touched.

**Problem.** The exact sign-flip confirmation is a paired-seed test: within a
seed the two arms must share the same shared-trunk initialization. Training
already records `init_core_fp` (a digest of the `tok`/`pos`/`head`/`norm`
initial tensors), and the generic `pair_reason` refuses a pair whose digests
differ or are one-sided. The confirmation path did not carry that evidence:
`_finaltest_bylen_cell` copied only `train_knobs/params/train_objective/
param_breakdown/rng_fp` from the training record, so `bylen.json` never had
`init_core_fp`. Three consequences followed. (1) `finaltest_stats` could not
check the pairing it claimed to check. (2) Its global
`same_init_core_fp = len(set(values)) <= 1` was both conceptually wrong —
the digest is a function of the seed, so a healthy panel HAS different values
across seeds — and vacuously true when no value existed, printing "same" for
"no evidence". (3) `_admit_confirmation_records` never ran the init check, so
a definitely mismatched pair could enter the test; with a legacy
`require_complete_coverage=false` manifest it could even confirm from the
remaining pairs.

**Changes.** (1) The checkpoint `extra` now stores `init_core_fp`, and
`_finaltest_bylen_cell` copies it from the training record (falling back to
the weight metadata) into every bylen record and the experiment event.  If
the checkpoint and the training record both carry the field but disagree, the
cell is REFUSED (the weights and the record describe different
initializations).  A missing digest emits an explicit WARNING and is never
invented.  A resumed bylen record whose digest differs from the expected one
— including a legacy record that has no field — is stale and is re-evaluated
rather than reused as an "unverified" confirmation measurement. (2) New
`exp_lib.paired_init_core_audit(records_a, records_b, seeds)` performs the
per-SEED two-arm check and returns four states (`verified`, `mismatch`,
`unverified`, `missing`) plus a tri-state `pairing_ok`: True only when every
requested pair is verified, False on any definite mismatch, None when the
evidence is merely incomplete. A one-sided/absent digest is unverified, not
equality. (3) Each primary comparison runs that audit; mismatched seeds are
named in `dropped_pairs` and excluded from the exact sign-flip test. If fewer
than two pairs remain the comparison is `omitted`, not invented. Per-
comparison `init_core_fp`, `same_init_core_fp` and `init_core_fp_recorded`
are written, and the top-level `validity` block now summarises the per-seed
two-arm digest sets (`init_core_fp_complete`, mismatch seeds) instead of a
cross-seed global set. (4) `confirmed` additionally requires
`same_init_core_fp is not False`. `None` (legacy records without the field)
keeps the permissive historical path but is rendered as "unverified", never
as "same". Under the shipped `require_complete_coverage=true` plan, dropping
any seed already fails coverage, so this gate only states the same fact; it
adds no effect-size threshold. (5) The P5T table gains an initialization-
pairing column and prints dropped seeds/reasons when a mismatch occurs.

**Boundary.** This is a pairing-validity correction, not a new endpoint or a
changed decision rule: the frozen arms/seeds/endpoints/alpha/Holm family,
the training numeric path and the test split are byte-for-byte unchanged.
The checkpoint gains one metadata field; existing checkpoints still resolve
the digest from their training record. A definite mismatch is no longer
treated as a pair; unverified legacy evidence is not relabelled as a match.
No `CODE_SEMANTICS` bump is needed because no loss or PPL value changes.

**Verification.** New CPU tests L119 (per-seed audit: healthy across-seed
digests, mismatch, one-sided, missing, vacuous-empty, seed validation) and
L120 (end-to-end `finaltest_stats`: matched panel is verified; a mismatched
seed is dropped by name and a complete-coverage plan becomes `omitted`;
absent/one-sided evidence is `None`, never True).
`python test_review_fixes.py` passes 454/457 with the same 3 DataLoader
environment skips; `python v12_supp.py plan` still admits 72/72 synthetic
frozen records with the same six comparisons.

## 2026-10-22 — scale-safe segmentation / sidecar construction / host fallbacks

Zero-GPU correctness review aimed at future scale; no training run, frozen
manifest byte, test split, `CODE_SEMANTICS`, README or CI configuration was
touched. The integer values and every measured PPL are unchanged; the fixes
remove per-block/per-token Python-object materialisation and one crash path
when the optional host geometry cannot be allocated.

**Bug 1 (uniform-stride segmentation).** `_segment` has a closed-form
uniform-stride shortcut. Its general successor-walk path already returns a
NumPy `counts` row for `n > 64`, but the shortcut always built
`[_stride] * (n // _stride)` as a Python list. For the m=1 / `block_size=1`
arm the stride is one, so a long sequence created one Python list element per
block before any consumer ran, and downstream `np.asarray(counts)` copied it
again. The shortcut now builds the same integers directly in an `int64`
NumPy row for large `n` and converts to the historical list only for `n <= 64`.
`bids`, `honoured`, counts and sum(n) are byte-for-byte the same values.

**Bug 2 (per-token document sidecar).** The cache builder's inline validation
path used `array.extend([doc_id] * take)` for every document. A long document
therefore allocated a temporary Python list with one pointer per token even
though the standalone capped-row helper had already been changed to
`np.repeat`. It now extends the `array('Q')` from `itertools.repeat`, so the
temporary object is O(1) and the stored uint64 ids are identical.

**Bug 3 (host-geometry fallback contract).** The host pooling-geometry
builders document "None when the host form cannot serve; the caller then
falls back to the on-device form". Their `except` lists caught runtime/type/
value/attribute errors but not `MemoryError` (or `OverflowError`), so a
large-but-valid geometry that exceeded host memory crashed the layer instead
of falling back. The six builders now treat allocation failure as "cannot
serve" and return `None`; the pinned-staging pools likewise fall back when
`pin_memory` raises `MemoryError`. No integer/value changes on the successful
path.

**Bug 4 (ladder parameter audit).** `m0_supp.run_params` iterated a literal
`('S1', 'S2', 'S3', 'S4')`, so a newly declared `SCALE_LADDER` tier was
silently absent from the planned-vs-observed parameter table even though the
panel resolver and `scale_ladder_audit` had discovered it. The tier list is
now derived from the ladder and sorted numerically (`S2` before `S10`), and a
tier without a recorded planning value is still measured and reported with an
explicit `None` ratio rather than dropped or raising `KeyError`.

**Boundary.** Bugs 1-3 are representation/transport corrections (same
integers, same cache bytes, same forward values) and bug 4 is an audit-only
coverage fix. No frozen arm/seed/endpoint/alpha/Holm family or decision rule
changes; `CODE_SEMANTICS` is not bumped.

**Verification.** New CPU tests L121 (uniform-stride counts are a large NumPy
row with exact values, the ≤64 list contract is preserved, and the host
geometry builders return `None` on a simulated `MemoryError`) and L122
(numeric ladder discovery, incl. S10 ordering and an unknown tier).
`python test_review_fixes.py` passes 456/459 with the same 3 DataLoader
environment skips; `python test_open_block.py` passes; `python v12_supp.py
plan` still admits 72/72 synthetic frozen records with the same six
comparisons.

## 2026-10-22 — run order as a design variable (blocked randomization; no new runs)

Zero-GPU design hardening; no training run, frozen manifest byte, test split,
`CODE_SEMANTICS`, README or CI configuration was touched.

**Problem.** The exact paired sign-flip test assumes within-seed exchangeability,
but calendar time was never a controlled design variable. `v7_supp.run_lenphase`
(the training entry point used by v12 P5C) scheduled
`[(v, seed) for v in variants for seed in seeds]`: every seed of arm A, then
every seed of arm B. The two arms of each pair were therefore trained in
different machine/time blocks, so a driver update, thermal throttling, a noisy
neighbour or a changed image could be perfectly confounded with arm. The
generic `exp_lib.run()` already happened to be seed-major, but a fixed arm
order still binds "first slot" to one arm and can turn a warm-up/cache effect
into an arm effect. This is a design defect, not a p-value problem.

**Change.** New pure helpers `exp_lib.blocked_run_order` /
`build_run_order_plan` / `run_order_balance` / `write_run_order`:

- one block per seed, cells inside a block contiguous;
- a stable SHA-256 base permutation (not Python's per-process `hash`) and a
  per-block cyclic rotation, so each arm occupies each slot roughly equally
  when blocks >= arms;
- the plan is built over ALL declared cells, so a resume cannot reshuffle it;
  `run_order.json` records the design, `blocks`, `planned_cells`,
  `cell_blocks` and `cell_variants` (inside the gitignored results dir);
- `v7_supp.run_lenphase` and `exp_lib.run()` now execute in plan order; resume
  keys, seeds, training values and `run_cfg` are unchanged.

New `audit_gates.py run_order` (also included in `run_report`) checks complete
blocks, position balance and the actual order from `experiments.jsonl`. A
complete arm-major execution is a hard FAIL even when the plan claims a blocked
design; a missing plan is `not_verifiable_locally`, never "order is fine". A
retrained cell is deduplicated to its last start, and parallel start-order
deviations are warnings only, because grouping (not wall-clock ranking) is what
the design guarantees.

**Boundary.** This changes only the order in which the same `(variant, seed)`
cells run and adds design metadata. It does not change a PPL, seed, recipe,
endpoint, alpha, Holm family, frozen arm or decision rule, and no
`CODE_SEMANTICS` bump is needed. Legacy panels keep their results; the audit
reports arm-major / unverifiable rather than retroactively rewriting them.
`run_order.json` is an artifact under `results_*/`, so it travels with the
evidence archive.

**Verification.** New CPU tests L123 (deterministic complete blocks, position
balance, differing design seeds, duplicate/unhashable/string-seed rejection,
plan-level audit and malformed/arm-major plans) and L124 (audit accepts a
planned execution, deduplicates a retrained cell, fails a completed arm-major
run, and returns not_verifiable when no plan/events exist).
`python test_review_fixes.py` passes 458/461 with the same 3 DataLoader
environment skips; `python audit_gates.py run_order` reports
`not_verifiable_locally` on this source-only tree (no `results_*/` artifacts);
`python v12_supp.py plan` still admits 72/72 synthetic frozen records with the
same six comparisons.

## 2026-10-22 — fixed-stride host tables, cached attachments and object-key scanning at scale

Zero-GPU scale review; no training run, frozen manifest byte, test split,
`CODE_SEMANTICS`, README or CI configuration was touched.

**Problem.** (1) `blocks_fixed` still built its host counts as
`[bs] * (n // bs)`, one Python list slot per block.  With the supported
`block_size=1` arm that is one Python pointer per token, even though
`_segment`'s uniform shortcut had already moved to a NumPy row.  (2) The same
tensor is cached in `_IDX_CACHE` with the counts row attached as `_pool_cpu`,
but `_cache_bytes`/`_idx_cache_total` charged only the tensor, so a length
sweep could retain roughly twice the advertised cache budget.  (3)
`segmentation_prefix_invariance` converted a large counts row back to a Python
list merely to prepend the zero start.  (4) `HybridAttention` used Python
`sum(numpy_row)` to verify the counts total while collecting cut tables; that
walk is O(blocks) in Python per layer/step.  (5) `data_guard._row_keys`
flattened the whole object-dtype table to a Python list only to reject complex
cells, then re-walked it block by block.

**Changes.** `blocks_fixed` builds the same integers in an `int64` NumPy row
and keeps the historical list only for `n <= 64`; `_cache_attachment_bytes`
charges the `_pool_cpu` counts row (NumPy `.nbytes`, or one pointer per list
entry) through both `_idx_cache_total` and `_cache_item_bytes`.
`segmentation_prefix_invariance` obtains starts as
`cumsum(counts) - counts` directly on the NumPy row.  The cut-table total uses
one vectorised `int64` sum of the same counts.  `_row_keys` rejects complex
values per bounded hash block instead of flattening the table.

**Boundary.** These are representation/transport/accounting corrections: the
same block ids, counts, cut positions, row digests and cache contents are
produced; no frozen arm/seed/endpoint/alpha/Holm family or decision rule
changes.  The `<=64` Python-list contract and all returned values are
unchanged.

**Verification.** New CPU tests L121b (NumPy fixed-stride counts with exact
values/list contract, attached-byte charging, prefix starts, linear run-order
audit) and L121c (complex object rows rejected across hash-block boundaries).
`python test_review_fixes.py` passes 460/463 with the same 3 DataLoader
environment skips; `python test_open_block.py` passes 15/15;
`python audit_gates.py run_order` still reports `not_verifiable_locally` on
this source-only tree; `python v12_supp.py plan` still admits 72/72 synthetic
frozen records with the same six comparisons.

## 2026-10-22 — source-tree identity and a complete dependency lock

Zero-GPU reproducibility/engineering hardening; no training run, frozen
manifest byte, test split, `CODE_SEMANTICS`, README or CI configuration was
touched.

**Problem.** The provenance block recorded `git_commit` plus a tracked-diff
hash, but those do not identify the bytes that actually ran: a dirty work tree
shares its commit, an untracked source file is only named in `git status`, and
a source tarball has no `.git` at all.  `CODE_SEMANTICS` is a measurement
semantics stamp, not a file hash (the continuous-loop header explicitly warns
that an edit can change semantics while the stamp still matches).  Separately,
`requirements.lock` pinned only the five direct dependencies, so the transitive
numerical stack could drift while the lock still looked "verified"; the
environment snapshot also recorded only a hand-picked subset of `CSA_*` flags.

**Changes.** `exp_lib.source_tree_manifest()` / `source_tree_summary()` hash a
deterministic, pruned set of measurement-relevant source files (non-test
`*.py`, `pyproject.toml`, `setup.py/cfg`, `requirements.txt`, both frozen
manifests) by relative path, byte count and per-file SHA-256.  The
machine-specific `requirements.lock` is deliberately excluded from the source
digest (the same source must not look different on a CPU test box and a CUDA
experiment box); its hash and contract are recorded separately in
`environment_snapshot()`.  Generated result/analysis/cache directories and
tests/docs/CI are excluded so a test edit cannot masquerade as a semantic
change.  `code_provenance()` records the summary under `code.source_tree`; it
is provenance only and does not enter `run_cfg`, resume fingerprints or
`exec_fp`.  `v12_supp.py plan` now writes the
full manifest plus `dependency_contract()` into
`analysis_v12/plan_snapshot.json`, binding the bytes and environment the plan
gate approved.

`exp_lib.parse_requirements_lock()` / `dependency_contract()` parse
`name==version` locks (options/comments ignored, hash-pinned continuation lines
accepted) and compare every pin with `importlib.metadata`; mismatched, missing
or malformed pins fail, `partial` records a lock that does not cover every
installed distribution.  `requirements.lock` is replaced by the complete
48-distribution `pip freeze` of the verified CPU environment (including
`torch==2.14.1+cpu` and the declared CPU wheel index), and
`environment_snapshot()` records the contract plus all `CSA_*` / `WT103_*` /
`FWEDU_*` / `M0_*` / `V12_*` run-control variables.

`audit_gates.py source` compares the current tree with a recorded
`source_tree` from the plan snapshot or panel provenance and reports
added/removed/changed files; no record is `not_verifiable_locally`, never a
silent pass.  `audit_gates.py dependencies` exposes the lock contract
(`partial` is a warning, mismatch/missing/invalid is a fail).  Both are part of
`audit_gates.py report`.

**Boundary.** These are evidence and environment-description changes only.
They do not alter a PPL, seed, recipe, endpoint, alpha, Holm family, frozen arm,
decision rule or P5T admission; no `CODE_SEMANTICS` bump is needed.  P5T does
not yet bind the tree digest in the freeze lock (the §49.2 boundary remains),
but the plan snapshot and every cell's provenance make the approved source and
the evidence-producing source comparable.

**Verification.** New CPU tests L137 (deterministic per-file/aggregate digest,
content change/removal, explicit include, escape rejection), L138 (options
ignored, match/mismatch/missing/malformed lock, absent lock), L139 (source
audit ok/fail/not_verifiable and dependency audit ok/fail) and L140 (the
tracked lock pins the five direct deps and ≥20 transitive distributions).
`python test_review_fixes.py` passes 476/479 with the same 3 DataLoader
environment skips; `python test_open_block.py` passes 15/15;
`python audit_gates.py source` reports `ok` against the fresh plan snapshot;
`python audit_gates.py dependencies` reports `ok` with 48/48 pinned and 0
unpinned installed distributions; `pip install --dry-run --no-deps
--ignore-installed -r requirements.lock` resolves every pin, and `pip check`
reports no broken requirements; `python v12_supp.py plan` still admits 72/72
synthetic frozen records with the same six comparisons.


## 2026-10-22 — single-factor labels now machine-derived (no new runs)

Zero-GPU audit hardening only; no arm value, recipe, seed, endpoint, frozen
manifest or test split was touched.

**Problem.** The review-7 gate verified that a contrast's actual cfg diff
equaled its DECLARED key set, but it still called every non-`kind` diff
"knob-level single-factor".  Two shipped dynamic-vs-fixed contrasts move the
segmentation bundle together with the HCA boundary-gate objective multiplier
(`comp_lambda_mult`) and an inert `block_size` field: enumerated keys were
accepted, yet the pair was multi-factor and the report counted it as a
single-factor ablation.  The v9/v10 seq2k `single_factor`/`factors` metadata
was also never compared with the real arm configs, and the open-block matrix
only checked one cap mode (and only extra keys).

**Changes.** (1) `exp_lib.single_factor_design_audit` now derives the level
from the diff: `single_factor` only when the keys sit inside ONE declared
family, `architecture_level` for `kind` swaps, and `system_level` only for the
two explicitly declared multi-factor HCA-dynamic pairs
(`CONTRAST_SYSTEM_LEVEL_DECLARATIONS`).  Any other multi-family bundle is a
violation; enumerating the keys is not a licence to call it single-factor.
(2) `exp_lib.sweep_contrast_factor_audit` rebuilds the v9/v10 seq2k arms and
checks `SEQ2K_CONTRAST_FACTORS` against the actual config diff (arms, factors,
`single_factor` flag).  `v9.add_paired` attaches the verification and clears an
unverified `single_factor=True` label instead of propagating metadata.
(3) `open_block_matrix_audit` checks every mode/cap/scope cell in both scopes
and fails on a missing axis key as well as a leaked key.
(4) `dynamic_design_audit` now requires every `mechanism_claim_eligible`
hypothesis to be tested by a single-factor cfg pair (H1/H2/H4 pass; H3/H5 stay
system-level), and a mechanism claim on a non-single-factor diff fails.
(5) `audit_gates.py single_factor` reports the new sweep section.

Verification: shipped design reports 13 knob-level single-factor, 3
architecture-level and 2 declared system-level pairs; 46 open-block intra-matrix
axes and all 3 seq2k declarations verify; the dynamic draft still audits `ok`
with H1/H2/H4 single-factor while the non-eligible H3/H5 family stays
system-level.  New CPU assertions cover the multi-family refusal, sweep drift,
omitted-record attachment and the mechanism-eligible/HCA mismatch.
`python test_review_fixes.py` passes 476/479 with the same 3 DataLoader
environment skips; `python test_open_block.py` passes 15/15; the
`audit_gates.py single_factor` subcommand reports `ok`.

## 2026-10-22 — the freeze lock now binds the measurement source tree (no new runs)

Zero-GPU reproducibility/evidence hardening; no training run, frozen manifest
byte, test split, `CODE_SEMANTICS`, README or CI configuration was touched.

**Problem.** The freeze lock (`results_lm_test/.freeze_lock.json`) bound the
manifest SHA-256 and the test-slice/training-corpus fingerprints, but not the
bytes of the code that produced the test measurements.  `manifest_sha256` pins
the **plan**, and `CODE_SEMANTICS` is a hand-maintained constant; the warning
at the top of `.github/workflows/continuous-loop.yml` records the exact
failure mode: an edit can change measurement semantics while the stamp still
matches, so P5T would admit it silently.  `source_tree_manifest()` already
gave the tree a content digest for the plan snapshot and per-cell provenance
(§50), but §49.2's boundary — "the analysis code itself is not hashed by the
freeze lock" — remained open.

**Changes.** `v12_supp._check_or_write_freeze_lock*` computes
`exp_lib.source_tree_manifest()` and writes the full manifest plus
`source_tree_sha256` / `source_tree_bound_at` when the lock is first created
(`preaccess`, before any test read).  Later calls compare the digest:

- `preaccess` / `accessed` (a real test evaluation) **refuse** a changed tree,
  naming the changed/added/removed files (bounded) and the two legitimate
  paths: restore the frozen tree, or `V12_REFREEZE=1` (the old digest is kept
  in `superseded` and `history`, and the earlier round is downgraded).
- `finaltest_stats` now calls the lock with `stage='analysis'`.  A changed
  analysis tree is refused by default; `V12_ALLOW_SOURCE_DRIFT=1` permits a
  **disclosed** re-analysis that keeps the ORIGINAL attestation and appends a
  `source_tree_drift` event (old/new digest, changed files, code stamp).  The
  run is then not a byte-identical replay and `audit_gates.py source` fails.
- A pre-source-binding lock that is still `preaccess` (no test read yet) is
  safely back-filled with the current tree; an already `accessed` legacy lock
  stays visibly unbound and the audit reports `not_verifiable_locally`, never
  "source is fine".  Injected manifests are validated with the same contract
  (non-empty SHA-256, no unreadable files).
- `audit_gates.audit_source_integrity()` now takes an explicit `lock_path`,
  discovers `.freeze_lock.json` under `results_*`, and prefers
  snapshot → freeze lock → panel provenance.  It reports file-level
  added/removed/changed and treats a recorded `source_tree_drift` as a fail.

`docs/design_notes.md` §49.2/§50.3 now cross-reference the resolved boundary
and §51 documents the full contract.

**Boundary.** Evidence/gate change only.  No PPL, seed, recipe, endpoint,
alpha, Holm family, frozen arm, decision rule, test split or checkpoint is
changed; `confirmation_frozen*.json` is untouched and `CODE_SEMANTICS` is not
bumped.  The digest does not enter `run_cfg`, resume fingerprints or
`exec_fp`, so existing cells' resume/pairing is unchanged.

**Verification.** New CPU tests L141 (first-write binding, drift refusal with
file names, disclosed analysis override, re-freeze rebinding and superseded
history, malformed injected manifest) and L142 (audit ok/fail against the lock,
legacy-lock unverifiable boundary, declared drift is a fail).
`python test_review_fixes.py` passes 478/481 with the same 3 DataLoader
environment skips; `python test_open_block.py` passes 15/15;
`python v12_supp.py plan` still admits 72/72 synthetic frozen records with the
same six comparisons; `python audit_gates.py source` reports `ok` against the
regenerated plan snapshot; `python audit_gates.py dependencies` reports `ok`
with 48/48 pinned and 0 unpinned installed distributions.

## 2026-10-23 — effect modification across evaluation lengths (no new runs)

Zero-GPU results-analysis hardening; no training run, no test read, no frozen
manifest byte, seed, recipe, endpoint, alpha, Holm family or decision rule is
touched.

**Problem.** The frozen P5T table reports a per-length exact sign-flip p for
512/1024/2048/4096, but the report had no test of whether the arm contrast
*changes* with length.  Reading "significant at 512, not significant at 4096"
as length dependence is the classic difference-between-significant-and-
non-significant error, and reading "significant at both" as a constant effect
does not test a zero difference either.  Absolute ΔPPL is also not comparable
across lengths because baseline PPL grows with the evaluation span.

**Changes.** `exp_lib.effect_modification()` takes one already-paired effect
per seed at each condition, aligns a single complete-case seed set, and
reports (i) descriptive simple effects, (ii) per-seed
`effect(condition) − effect(reference)` with exact sign-flip p, t interval,
paired bootstrap and a declared-family Holm adjustment, and (iii) an optional
per-seed OLS slope on a supplied numeric axis (never a regression on condition
means).  Missing/bool/non-finite cells drop the whole seed and are named.
`v12_supp._length_interaction_analysis()` applies it to PPL as
`log(PPL_a) − log(PPL_b)` with the training length 512 as reference (shortest
length if absent) and `log2(length)` as the axis; `finaltest_stats` attaches
`comparisons[*].length_interaction`, and `_length_interaction_markdown`
renders the post-hoc table after the per-length curve.

**Boundary.** The interaction and trend are post-hoc diagnostics: they are not
added to the frozen Holm family, do not change `confirmed`, `p`, `p_holm`,
α or the decision rule, and are not confirmatory evidence.  A non-significant
interaction is explicitly rendered as "no detected effect modification", not
as equivalent/constant effects.  No PPL, checkpoint, seed plan, manifest or
`CODE_SEMANTICS` is changed.

**Verification.** New CPU tests L149 (constant vs growing effects, Holm family,
per-seed trend, whole-seed dropping, reference/axis guards, malformed inputs)
and L150 (log-ratio definition, 512 reference and fallback, truncated-cell
seed drop, single-length refusal, Markdown caveats).
`python test_review_fixes.py` passes 486/489 with the same 3 DataLoader
environment skips; `python v12_supp.py plan` still admits 72/72 synthetic
frozen records with the same six comparisons; `python audit_gates.py source`
reports `ok` (23/23 files byte-identical to the regenerated plan snapshot).

## 2026-10-23 — the analysis unit is a raw line, not an article (no new runs)

Zero-GPU data-processing hardening; no training run, no test read, no frozen
manifest byte, seed, recipe, endpoint, alpha, Holm family or decision rule is
touched.

**Problem.** The HF `wikitext-103-raw-v1` builder yields one row per raw
**line** (a title heading, a paragraph, or a ` = = Section = = ` heading), not
per article.  The loaders' "document ids" and the per-token document map are
therefore line ids: the 60 articles of a WikiText validation split are ~2.5k
rows.  Three consequences were not gated. (1) Exact/SimHash deduplication ran
at line granularity, so an article whose lines were reworded or reordered
could cross a split. (2) The "per-document" bootstrap resampled paragraph
lines; the frozen 48×4097 prefix can contain hundreds of lines for a few dozen
articles, so the repo's 64-document population floor could be satisfied by
pseudoreplicates. (3) `build_tokenizer` set the ByteLevel pre-tokenizer but
not its inverse decoder: `decode` returned BPE surface text (`Ġthe`), and
`boundary_token_ids` therefore missed every space-prefixed punctuation token
(`Ġ.` `Ġ,` `Ġ:` …), undercounting the boundary mask on real BPE-8k.

**Changes. (1) Reversible tokenizer.** `exp_lib.build_tokenizer` now attaches
`decoders.ByteLevel()`. Encode/token ids are unchanged; decode is lossless, so
`boundary_token_ids` recognises space-prefixed punctuation. New caches persist
the decoder; legacy `tok_*.json` files (decoder null) are repaired **in
memory** by `ensure_bytelevel_decoder()` for audit purposes only, never
rewritten. (2) Unit primitives. `wikitext_article_groups` segments raw rows by
top-level ` = Title = ` headings (blank lines and section headings do not
split); `wikitext_article_text` / `wikitext_article_ids` give stable
article-level text and ids; `line_spans_from_token_doc_ids` and
`decode_token_runs` recover served units from a cache without re-reading the
raw test split. (3) `audit_gates.audit_document_unit` decodes the test/val
per-token map, classifies the served unit as line/article/unknown, reports
`n_lines` / `n_articles` / pseudoreplication and the true article count in the
frozen prefix, and sets `document_level_claims_allowed=False` for line units.
When raw train/valid dumps exist it additionally tests article-level leakage
the line index cannot see: a title whose lines are served in both splits is a
hard failure; near-duplicate articles missed by line-level cleanup are
warnings with examples. It never reads `wiki.test.raw`. (4) New caches record
`document_unit: "line"`, `document_unit_is_article: false` and a definition;
`_cached_split_problems` rejects an unknown unit or an article claim without
article-level sidecars. (5) The audit is wired into
`audit_gates.py document_unit` and `audit_report.json`.

**Boundary.** Token ids, split membership, training/evaluation slices, PPL,
`CODE_SEMANTICS`, `confirmation_frozen*.json` and CI are unchanged.  The
corrected boundary mask only affects newly built caches' `val_bnd`/`test_bnd`
diagnostics, not the LM loss.  Article-level sidecars are **not** written yet:
existing per-document vectors remain line-level, and the audit says so rather
than silently relabelling them.  A future freeze should persist
article/token-article sidecars before the one-shot test read and pass them to
`eval_length_gen`; this round only reports the true article breadth and
refuses unqualified document-level/population claims.  Missing raw dumps are
reported as `not_verifiable_locally`, never as clean.

**Verification.** New CPU tests L151 (heading segmentation, blank/section
non-splitting, preamble/no-heading refusal, stable article ids, token runs,
round-trip decode and space-punctuation boundary tokens) and L152 (line-unit
warning + claim gate, frozen-prefix article count, missing-raw
non-verification, line/article classification, and a title whose lines straddle
train/validation failing the audit).  `python test_review_fixes.py` passes
489/492 with the same 3 DataLoader environment skips;
`python audit_gates.py document_unit` reports `not_verifiable_locally` with no
cache; `python v12_supp.py plan` still admits 72/72 synthetic frozen records.

## 2026-10-23 — evaluation uncertainty is clustered by article, not by raw line (no new runs)

Zero-GPU results-analysis hardening; no training run, no test read, no frozen
manifest byte, seed, recipe, endpoint, alpha, Holm family, decision rule or
`CODE_SEMANTICS` is touched.

**Problem.** §53 of the design notes made the analysis unit explicit: the
HF `wikitext-103-raw-v1` builder yields one row per raw **line**, so the
loader's "document" sidecar is a line, and an article spans many rows.  The
audit (`audit_gates.document_unit`) could report the true article count and
refused unqualified document-level claims, but the per-comparison
`doc_sampling` readout still resampled the line-level `nll_per_doc` vectors.
That is pseudoreplication: several paragraphs of one article are treated as
independent sampling units, so the document bootstrap interval is too narrow
and `n_docs` overstates the independent corpus breadth.  The frozen `n=48`
"documents" are 48 fixed evaluation rows, not 48 articles.  As a concrete
check, the local 512x512 validation cache decodes to 2022 line runs inside 56
top-level articles (36.1 lines per article): the line bootstrap would treat
roughly 36 paragraphs as independent draws from a single article.

**Changes. (1) Article cluster primitives.** `exp_lib` now has
`wikitext_article_row_ids(rows)`, `wikitext_token_article_ids_from_line_map(
token_ids, line_ids, tok)`, `wikitext_token_article_ids_for_rows(rows,
lengths, cap_tokens)` and `aggregate_nll_by_article(nll, article_ids)`.
Cluster ids are keyed by `(group ordinal, normalised top-level " = Title = "
heading)`, not by the article body hash: the same frozen slice recovered from
raw rows or from decoded token runs must yield the same clusters, while a body
edit does not split one article and an ordinal disambiguates same-titled
articles.  Blank/uncovered rows get their own id and are never merged into a
neighbouring article.  `wikitext_article_ids()` (the content hash used by the
leakage audit) is unchanged; cluster label and content identity are now
separate concepts.  `load_wikitext_test_tokenizer()` reads the persisted test
tokenizer JSON (width-checked) without touching `wiki.test.raw`.

**(2) Additive evaluation fields.** `v7_supp.eval_length_gen` accepts an
optional `article_ids` mapping and appends `article_granularity='article'`,
`n_articles`, `nll_per_article`, `n_tokens_per_article`, `ppl_per_article`,
`article_ids`, `article_ids_fp` and article totals.  The line-level
`nll_per_doc` / `n_tokens_per_doc` / `ppl_per_doc` / `doc_ids` fields and every
pooled `ppl` / `nll_sum` / `n_tok` value stay bit-identical: the article
aggregation is a second grouping of the same per-token NLLs.  The frozen
line-level contract therefore remains auditable.

**(3) P5T reconstruction without a second test read.** The driver resolves the
article map from the already-loaded per-token line map and the persisted
ByteLevel decoder (`_resolve_test_article_ids`).  A legacy cache with no
persisted tokenizer or no top-level heading falls back to the line unit with
an explicit status/reason recorded in every P5T record's `article_grouping`;
it never guesses an article boundary and never re-reads the held-out split.
The decoder is repaired in memory for legacy caches, not written back.

**(4) Admission and analysis.** `_by_len_shape_reason` validates article
vectors with the same rigor as line vectors (equal lengths, positive integer
token counts that partition `n_tok`, finite NLLs that reconstruct `nll_sum`,
matching `n_articles`, unique ids, article count <= line-document count), and
refuses partially-written article fields.  `_doc_level_analysis` prefers
articles when every admitted record carries them, falls back to line-level
documents (with `article_fallback_reason`) when coverage is mixed, and refuses
to pool mixed units.  `_doc_nll_matrix` takes `unit='document'|'article'`.
The analysis also reports `n_line_docs`, the observed
`pseudoreplication_factor` (line units per article) and a
`small_cluster_warning` below 20 article clusters.  The population-claim gate
now uses the article count and refuses a population claim when article
coverage is mixed rather than substituting the larger line count.  Report
rendering names the unit ("文章"/"文档"/"行").

**Boundary.** This changes the evaluation *unit of uncertainty*, not the
point estimate or the frozen decision: the exact sign-flip over seeds remains
the confirmatory test, and no article field enters `run_cfg`, resume keys,
pairing identity or the manifested endpoint.  The article count over a
48×4097 prefix is expected to be well below the 64-unit population floor, so
population claims remain refused; the gain is that the refusal and any
reported interval now use the independent unit.  A cap-truncated final
article is one in-slice cluster whose label is reproducible within the frozen
slice; it is not claimed to equal the full-article content id.  No
article sidecar is persisted yet — a future freeze should write
`test_token_article_ids_*` before the one-shot read.  Historical v6-v11
panels are not backfilled: their records carry no article fields and remain
line-level; only new `eval_length_gen(article_ids=...)` callers get the
article readout.

**Verification.** New CPU tests L153 (blank rows not merged, title vs body
edits, raw-vs-decode identical clusters, legacy decoder repair, wrong-size
flat map refused, tokenizer width mismatch refused, driver helper reconstructs
the same map from the cache) and L154 (line fields bit-identical after the
additive article aggregation, article totals partition `n_tok`/`nll_sum`,
admission rejects partial/duplicate/inconsistent article cells, analysis
prefers articles, mixed coverage falls back with a recorded reason,
cross-seed article-id mismatch is refused).  `python test_review_fixes.py`
passes 493/496 with the same 3 DataLoader environment skips;
`python v12_supp.py plan` still admits 72/72 synthetic frozen records with the
same six comparisons; `python audit_gates.py source` reports `ok` (23/23 files byte-identical)
after regenerating the plan snapshot with `python v12_supp.py plan`.

## 2026-10-23 — validation article audit must trim the unserved sidecar tail (no new runs)

Zero-GPU audit correctness; no training run, no test read, no frozen
manifest byte, seed, recipe, endpoint or `CODE_SEMANTICS` is touched.

**Problem.** `exp_lib.load_wikitext` caps the validation token stream at
`val_seqs * seq_len + seq_len`, writes the per-token document map for that
whole cap (`val_token_doc_ids_*`), and serves only the first
`n_val * seq_len` tokens as `val_batch`.  `audit_gates._unit_decode_runs`
demanded exact equality between the map length and the served array, so on a
normally built cache the validation half of `document_unit` always failed
before decoding (e.g. 262656 sidecar ids vs 262144 served tokens in the local
512x512 cache).  The article-leakage/unit section then reported only the length
error instead of analysing the selection split, which defeats the audit that
motivated the article unit in the first place.

**Fix.** A longer sidecar is trimmed to the served prefix and the number of
unserved ids is recorded as `trimmed_unserved_tokens`; the map is written in
stream order, so the served tokens are its prefix.  A *shorter* sidecar is
still a hard error — that means some served token has no document identity.
No token, id, fingerprint or article boundary changes for the served slice.

**Verification.** Extended L152 with the real loader layout (one extra
sequence of sidecar ids): the audit decodes validation, records the trim and
still classifies the line unit; a one-id-short map remains refused.
`python test_review_fixes.py` passes 493/496 with the same 3 DataLoader
environment skips; `python v12_supp.py plan` still admits 72/72 synthetic
frozen records; `python audit_gates.py source` is `ok` after regenerating the
plan snapshot.

## 2026-10-23 — installable source tree + pinned upstream corpus revision (no new runs)

Zero-GPU reproducibility/engineering hardening. No training run, no test-split
read, no frozen manifest byte/seed/recipe/endpoint, no `CODE_SEMANTICS` change,
and no README/CI edit.

**Problem 1 (engineering).** `pyproject.toml` declared a setuptools build but
had no explicit module list. Setuptools' flat-layout auto-discovery found the
generated `wt103_cache/`, `analysis_*` and `results_*` directories and refused
with "Multiple top-level packages discovered"; `pip install .`/`python -m
build` failed. A repository whose only supported workflow is "run files in
place" cannot produce a reproducible wheel/sdist or a clean-environment smoke.
**Fix.** `[tool.setuptools] py-modules` now lists every measurement/driver
module explicitly (test entry points stay source-only), and the metadata
declares `readme`, license, keywords and classifiers. `MANIFEST.in` defines the
sdist contents (code, tests, frozen manifests, docs; no caches/results/CI),
`Makefile` groups the zero-GPU checks, and `.build/` is ignored. The new CPU
test L155 parses the metadata, fails if a new top-level module is not listed,
and performs a real `pip install --no-deps --no-build-isolation --target ...`
of a minimal source copy.

**Problem 2 (reproducibility).** `src_fp`/`train_fp` prove two runs saw the same
tokens, but not that a **fresh build** came from the same upstream snapshot.
`load_dataset('Salesforce/wikitext', ...)` and the FineWeb-Edu stream were called
without a revision, so a moved Hub branch would silently define a different
corpus for the same config (the cache metadata did not record the upstream
commit either). **Fix.** WikiText-103 (`b08601e0…`, aliased `wikitext` repo) and
FineWeb-Edu (`87f09149…`) now pin a full 40-hex Hub commit and pass
`revision=`; `WT103_DATASET_REVISION` / `FWEDU_DATASET_REVISION` accept only a
full commit (branch/tag/short prefix is refused). Every cache meta records
`dataset_source` (`provider`/`repo`/`config`/`revision`/`revision_pinned`), test
metas also record the train/validation source, and train/validation/test must
agree on one commit at build time. On cache hit a mismatched Hub revision
rebuilds train/validation and hard-refuses the frozen test cache; legacy caches
without the field remain readable but are reported as unverifiable, never as
clean. Local dumps are explicit `provider='local'` records (content still bound
by token fingerprints, upstream commit not claimed). Run/checkpoint provenance
also carries `configured_dataset_source`. The new
`python audit_gates.py data_source` subcommand (also in `report`) re-checks the
schema, commit pins and train/test pairing independently of the loader; L156
covers moving-reference refusal, env override, mismatched/legacy/local caches
and the audit verdicts, and L157 exercises the cold local-raw build/write path.

**Verification.** `python test_review_fixes.py` passes 496/499 with the same 3
DataLoader environment skips; `python v12_supp.py plan` still admits 72/72
synthetic frozen records; `python audit_gates.py source` is `ok` after the plan
snapshot refresh; `python audit_gates.py report` completes (its `data_source`
section correctly reports the current legacy `wt103_cache` as
`not_verifiable_locally`); a cold local-raw build was exercised and its meta
records `provider='local'`, and the subsequent cache hit serves the same bytes.

