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
