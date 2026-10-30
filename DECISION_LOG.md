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
| v7 (`python -m explib.rope`) | reviewer response: RoPE + QK-norm, dense warmup, 40k-step long run, scale re-seeding, length extrapolation, top-k sweep; analytic FLOPs / KV table | exploratory |
| v8 (`python -m explib.scale`) | scale panel to 4 seeds; RoPE 20k long run | exploratory |
| v9 (`python -m explib.resume`) | seq-2048 top-k sweep at batch size 1; third RoPE-20k seed | exploratory |
| v10 (`python -m explib.corruption`) | from-scratch random-indexer / all-block controls, distractor-injection probe (uniform / realistic-token / mask / delete + span / doc / topical), NIAH-natural, crossover scaling, equal-token and 6ND-proxy large arms | exploratory |
| v11 (`python -m explib.length`) | statistical close-out: distractor probe to n=6, crossover panels to n=4, d=384 paired arms; seed-level sign-flip with resolution floors | exploratory |
| v12 (`python -m explib.confirmation`) | **freeze**: fixed-chunk CSA confirmation plan v2 (12 seeds, 4 primary comparisons, exact sign-flip + Holm, held-out test split, freeze lock, strict record admission, per-document PPL); v1 was rejected by the load-time power gate (6 seeds × Holm/4 has minimum attainable p=0.125) | confirmatory plan |
| review 2 | seed count revised 6→12 **before any test-split access**; scope block added; two-phase freeze lock; kernel mode refused for the frozen plan; evidence archiver; leakage/causality/fairness audits | frozen plan revision |
| review 3 | an external audit document (not part of this source-only snapshot) found the confirmation path was not runnable: manifest stamp v11.126 vs code v11.127, `eager_batch_expected=true` with all four arms on the per-row RoPE loop, and P5T records missing `train_knobs` (admitted=0 after the one-shot test access). Manifest **v3** re-signed before any test access: stamp v11.128, eager binding false, P5T/probe records carry the provenance descriptor, per-arm `mr` pinned, equal-budget wording corrected, decision rule declares an n_effective floor of 8, adaptive 2-SE health threshold, reachable leakage statistics, and `python -m explib.confirmation plan` as a zero-GPU CI pre-flight. The continuous-loop CI was disabled/cancelled during the fix. | frozen plan v3 + code hardening |
| review 3 (earlier same day) | pooled-NLL PPL + per-document NLL; pooled/bootstrap/Wilcoxon sensitivity; Holm in v7–v11 and `build_report`; objective/parameter/KV descriptors; opt-in task-learned boundary arm (`csa_dynamic_task`); synthesized records excluded from crossover; span-ΔNLL lookup fixed; kernel-stamp consistency; `DECISION_LOG.md` restored and guarded; dependency/Python declarations | code/evidence hardening |
| review 4 (2026-10-20) | no new experiment panels. Unified claim gate (`reportable_claim`), Holm family-resolution ceiling, NIAH folded into the Holm family, crossover fit refusal rules + slope CI, matched-init `init_core_fp`, measured compute-control direction (`system_level_only`), stale records fail-closed in `build_report`/v10, missing `truncated` treated as unverifiable, doc-bootstrap CI + <64-doc population gate, token-coverage estimate, efficiency sweep/pair audit, selection-attribution contract, evidence-release readiness audit. The frozen v12 manifest was NOT modified. | code/reporting/evidence hardening; all earlier panels remain exploratory |
| review 5 (2026-10-27) | no new experiment panels. Cross-panel ablation replication now records sampling identity (training seeds, validation/evaluation/holdout fingerprints, code/knobs) and splits the old `replicated` label into independent draws, new-seeds/same-evaluation, shared seeds, and unattested. Shared draws are repeated measurements, not independent replication; the new labels are descriptive and do not alter p values, Holm, or `all_components_useful_allowed`. Legacy reports without identity keep the old label and are never promoted to independent replication. The frozen v12 manifest was NOT modified. | code/reporting/evidence hardening; all earlier panels remain exploratory |

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
record is `REVIEW_4_RESPONSE.md` (rebuilt from this log's review-4 entry and
the README), and the statistical conventions are in `docs/design_notes.md`
section 25.

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
`python -m explib.confirmation power`: exact conditional power over the 2^12 sign
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
(4) `python -m explib.confirmation power` and `phase P5PW` are the zero-GPU preflight;
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
machinery. `python -m explib.confirmation` refuses `status=draft` (V12_ALLOW_DRAFT only permits
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

(3) **Frozen-manifest flag verification** (`python -m explib.confirmation plan`): each primary
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
`python -m explib.corruption` persists the new fields and prints macro/weighted F1 with the
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
`unknown`, never `not_observed`. `explib.rope.train_warmup` records the same audit.
Overfitting actions now name early stopping explicitly.

(2) **Bad cases are printed, not only written to disk.** `error_analysis.py`
prints the worst tokens (context, target, prediction, top-5, causes, prefix
ambiguity) and worst sequences by default (`--print-samples` /
`--print-sequences` control the budget). Per-token records and the Markdown
artifact now include the target's measured training frequency, so
"annotation/rare target" can be checked against a number.

(3) **Scale analysis reports cost alongside the final PPL.** `explib.m0` panels
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
and scale result-diagnostics tests added); `python -m explib.confirmation plan` still
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
total.  (3) Both `explib.m0._panel_result_diagnostics` and
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
that variable before `exp_lib` imports, so a normal direct `python ...` launch
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
precondition. (4) `explib.confirmation.plan` now runs the budget/evidence preflight before
training; new `explib.confirmation dynamic-plan` audits the draft without treating it as a
confirmation; `audit_gates design` is the unified zero-GPU entry point.
(5) `explib.rope` gains rope-consistent dynamic control arms
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
(5) `explib.rope.eval_length_gen` sliced a short panel but still allocated for the
requested length. (6) `explib.m0._ladder_params` used S1's `seq_len` for every
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
environment skips; `python -m explib.confirmation dynamic-plan` reports the draft as
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
`python -m explib.confirmation plan` admits 72/72 synthetic frozen cells and
`python -m explib.confirmation dynamic-plan` still reports the draft as non-runnable.

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
reference. (4) `explib.confirmation.finaltest_stats` attaches `stats['robustness']` AFTER
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
environment skips; `python -m explib.confirmation plan` still admits 72/72 synthetic frozen
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
(4) `explib.rope.eval_length_gen(..., doc_ids=...)` attaches those fields to every
by-length cell (including the truncated-`max_pos` branch) while leaving the
pooled `ppl`, `n_tok`, `nll_sum`, `nll_per_seq` and `ppl_per_seq` values
untouched. (5) `explib.confirmation.P5T` requests the mapping from `load_wikitest` and
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

**Problem.** `explib.confirmation._doc_level_analysis` built `ppl_per_doc`, averaged the
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
`explib.confirmation._doc_nll_matrix` checks that every per-document NLL vector is
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
environment skips; `python -m explib.confirmation plan` still admits 72/72 synthetic
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
environment skips; `python -m explib.confirmation plan` still admits 72/72 synthetic
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
environment skips; `python -m explib.confirmation plan` still admits 72/72 synthetic
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

**Bug 4 (ladder parameter audit).** `explib.m0.run_params` iterated a literal
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
environment skips; `python test_open_block.py` passes; `python -m explib.confirmation
plan` still admits 72/72 synthetic frozen records with the same six
comparisons.

## 2026-10-22 — run order as a design variable (blocked randomization; no new runs)

Zero-GPU design hardening; no training run, frozen manifest byte, test split,
`CODE_SEMANTICS`, README or CI configuration was touched.

**Problem.** The exact paired sign-flip test assumes within-seed exchangeability,
but calendar time was never a controlled design variable. `explib.rope.run_lenphase`
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
- `explib.rope.run_lenphase` and `exp_lib.run()` now execute in plan order; resume
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
`python -m explib.confirmation plan` still admits 72/72 synthetic frozen records with the
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
this source-only tree; `python -m explib.confirmation plan` still admits 72/72 synthetic
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
`exec_fp`.  `python -m explib.confirmation plan` now writes the
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
reports no broken requirements; `python -m explib.confirmation plan` still admits 72/72
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

**Changes.** `explib.confirmation._check_or_write_freeze_lock*` computes
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
`python -m explib.confirmation plan` still admits 72/72 synthetic frozen records with the
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
`explib.confirmation._length_interaction_analysis()` applies it to PPL as
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
environment skips; `python -m explib.confirmation plan` still admits 72/72 synthetic
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
cache; `python -m explib.confirmation plan` still admits 72/72 synthetic frozen records.

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

**(2) Additive evaluation fields.** `explib.rope.eval_length_gen` accepts an
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
`python -m explib.confirmation plan` still admits 72/72 synthetic frozen records with the
same six comparisons; `python audit_gates.py source` reports `ok` (23/23 files byte-identical)
after regenerating the plan snapshot with `python -m explib.confirmation plan`.

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
environment skips; `python -m explib.confirmation plan` still admits 72/72 synthetic
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
DataLoader environment skips; `python -m explib.confirmation plan` still admits 72/72
synthetic frozen records; `python audit_gates.py source` is `ok` after the plan
snapshot refresh; `python audit_gates.py report` completes (its `data_source`
section correctly reports the current legacy `wt103_cache` as
`not_verifiable_locally`); a cold local-raw build was exercised and its meta
records `provider='local'`, and the subsequent cache hit serves the same bytes.


## 2026-10-23 — 补种子要按 look 序列检验（no new runs）

零 GPU 的对比实验方法硬化；不新增训练、不读 test split、不改冻结清单的
任何字节/种子/终点/α/Holm 决策规则，也不改 `CODE_SEMANTICS`。

**Problem.** §49.3 已把「探索性面板的 look history 不是机器可读的」列为
剩余边界。补种子流程（`topup` / `ABL_SEED_TARGET` / `CORE_SEED_TARGET`）
会让同一对照在 n=3→4→6→12 时被多次聚合、多次查看；最终 n 的朴素符号
翻转 p 没有计入「看了几次再决定继续加种子」的选择效应，属于 optional
stopping。仅在单面板内做 Holm 不能修正这一点。

**Fix.** `_dump_aggregate` 现在把每个新种子集合作为一次 look 写入
`<outdir>/look_history.json`（同一数据重复聚合去重；文件有界并在截断时
显式记录）。新增 `exp_lib.sequential_signflip_p`：在种子并集上联合枚举
符号翻转，对每个 look 取**最终效应方向**上的证据并取 max，再双侧化；
p 不小于最终 look 的普通精确 p。≤16 个种子精确枚举，更多时退回
Bonferroni-over-looks；同一种子跨 look 配对差不一致时旧 look 视为
stale 并剔除。`methodology_audit` 的 `p_holm_global` /
`p_holm_family` / `significant_global` / `verified` 改用序列 p，朴素
最终 p 保留为 `p_holm_global_naive`；TOST 等价也乘 look 数后再进 Holm。
新增 `audit_gates.py looks` 独立重算并核对每个 panel 的 look history 与
`methodology_audit.json`，把「朴素显著、序列不显著」列为降级清单。

**Boundary.** 只影响有 look history 的新聚合；历史 v6-v11 面板没有该
文件，审计保持原行为，不回填、不重跑、不把旧结果升级为已做序列修正。
该文件是分析事件记录，不进入 resume/pairing/run_cfg/冻结判定；「从未
查看中间 look」时该修正是保守的而非错误的。没有自动检测「换验证切片
复用 outdir」——那仍由 `pair_fp` / `_paired_seed_deltas` 的指纹门负责。

**Verification.** `python test_review_fixes.py` 通过 518/521，3 个
DataLoader 环境跳过；新增 L176（原语：单 look 等价、单调、方向性、
stale/大规模回退、malformed 拒绝）、L177（落盘/去重/stale 剔除/篡改 p
fail-closed/缺 audit 块 warning）、L178（降级门）。`python -m explib.confirmation
plan` 仍 72/72 合成记录、6 个比较；`python audit_gates.py source` 在刷新
plan snapshot 后 `ok`（23/23 文件逐字节一致）；`python audit_gates.py
sesoi` 仍通过；`python audit_gates.py looks` 在无结果树时明确
`not_verifiable_locally`（不是 pass）。

## 2026-10-23 — ordered dose-response design gate + exact paired trend (no new runs)

Zero-GPU design/analysis hardening. No training run, no test-split read, no
frozen manifest byte/seed/recipe/endpoint/α/Holm rule, and no
`CODE_SEMANTICS` change.

**Problem.** The mechanism panels contain ordered levels (top-k, block size,
retrieval/distractor ratio), but the reporting path only supports pairwise
contrasts. Testing the two endpoints (or the best adjacent pair) and calling
the result a dose-response loses the ordering, invites post-hoc endpoint and
direction selection, and makes a two-point line look like a mechanism result.
A significant pairwise difference is not evidence that the outcome varies
monotonically with the dose.

**Fix.** `exp_lib.py` now has two companion primitives:

- `dose_response_design_audit()` — a definition-point gate requiring ≥3
  distinct finite doses, distinct arms, an AttnCfg diff of exactly the declared
  `dose_key` versus the baseline, monotonic numeric dose-key values (inverse
  coding allowed), the same declared seed set at every level (default
  `min_seeds=3`), and one shared protocol identity. A partial protocol
  declaration or a bundled second factor fails closed. Optional
  `require_protocol`/`require_seeds` flags support legacy descriptive audits.
- `dose_response_trend()` — a within-seed OLS slope per seed over the declared
  doses, followed by ONE exact sign-flip test on those slopes (the existing
  `paired_effect_stats` contract: n, CI, p floor, underpowered,
  within-seed-noise). `direction='increasing'|'decreasing'` must be declared;
  a significant opposite-direction slope is `direction_conflict`, never
  relabelled. `shape='monotone'` additionally requires every adjacent mean
  step to move in the declared direction; a linear trend remains reportable
  without the stronger monotone claim (`shape_conflict`). Missing/non-finite
  readings are named complete-case exclusions, not zeros or silent subset
  means. Multiple ordered families must be corrected together via the existing
  Holm/global-multiplicity machinery; adjacent pairs are not separate
  hypotheses.

`audit_gates.py dose_response` reads an optional manifest
`dose_response_design` block (or an explicit in-process design) and reports
`not_applicable` when no ordered declaration exists. It was added to
`audit_gates.py report`. The frozen manifests are unchanged: the new section
is opt-in for future/ordered panels, not a retrofit of v12.

**Boundary.** Ordered doses are not randomized: a single-axis cfg diff rules
out a declared second factor, but not co-varying training time, memory or
implementation paths. A significant trend is association under the declared
ladder, not proof of mechanism; non-monotone or opposite-direction results
must be reported as such. The direction/shape declaration is machine-checked
for consistency but cannot prove it was written before the first look.

**Verification.** `python test_review_fixes.py` passes 529/529 (0 skipped in
this sandbox), including new L181 (ordered single-axis ladder, bundled second
factor, duplicate dose, seed mismatch, missing seed declaration, two points,
protocol mismatch, scrambled dose-key order, post-hoc `direction='auto'`,
invalid shape) and L182 (exact p=2/64, direction conflict, linear-vs-monotone
shape gate,
missing/non-finite complete-case exclusions, two-point refusal, protocol
conflict, invalid direction/shape). `python -m explib.confirmation plan` still admits
72/72 synthetic frozen records with the same six comparisons;
`python audit_gates.py source` is `ok` (23/23 source files byte-identical);
`python audit_gates.py dose_response` returns `not_applicable` because the
frozen plan has no ordered-dose declaration (explicitly not a pass).

## 2026-10-23 — 统计机器先校准再使用（no new runs）

零 GPU 的方法论硬化；不新增训练、不读 test split、不改冻结清单的
任何字节/种子/终点/α/Holm 决策规则，也不改 `CODE_SEMANTICS`。

**Problem.** 单元测试只能证明统计函数在给定 fixture 上返回预期形状，不能
证明它们在重复抽样下的频率学错误率符合名义水平。精确符号翻转的枚举、n=21
的 MITM、n>40 的鞍点尾、max-over-looks 的序列 p、TOST 的 margin 边界与
Holm 的 FWER 任何一处实现偏差，都可能让全部 fixture 通过、却让所有真实
面板的 p 系统性偏小。此前没有任何门直接检查「检验本身是否超发」。

**Fix.** `exp_lib.STATISTICAL_CALIBRATION_POLICY` 固定种子、重复次数、
α、功效下限与两条单侧校准阈值；新增
`statistical_calibration_suite()` 用**局部** `SeedSequence`/`default_rng`
流生成纯合成数据，对六条路径同时跑：

- 精确符号翻转枚举（n=12）、MITM（n=21）、鞍点尾（n=48，验证 n>40 分派）；
  每条路径还核对实际 `method` 确实是该路径，而不是在别的实现上做校准；
- 鞍点 p 与 n=21 精确 MITM p 的逐点最大/平均绝对误差；
- max-over-looks 序列 p（nested looks 6/9/12）；
- 配对 TOST 在 margin 边界（null mean=1，margin=1）与零均值（正控制）；
- Holm 六族全局 FWER 与「一个强信号 + 五个 null」的正控制。

每条路径都有**负控制**（`K ~ Binomial(n_sim, alpha)` 的上尾，防止超发）
和**正控制**（`K ~ Binomial(n_sim, power_floor)` 的下尾，防止「从不拒绝」的
检验蒙混过关）。校准失败判 fail；仅超出名义水平但仍在蒙特卡洛噪声内判
warning；保守行为通过。阈值固定于模拟前，不能看结果后再调。
`audit_gates.py calibration`（别名 `stats_calibration`）把结果写入
`audit_report.json` 的 `calibration` 键；校准不读结果树、不写
`results_*`、不触碰全局 RNG。`policy_fp` 绑定实际生效的
seed/n_sim/α/阈值，带覆盖参数运行不会复用默认策略的指纹。

**Boundary.** 校准只验证统计机器在理想独立配对、正态、同方差数据上的
行为，不验证真实面板的 exchangeability、随机化、配对完整性或测量同一性；
通过校准不能修复混淆、泄漏或 optional stopping，也不是机制证据。失败表示
该统计路径的声明不可信，不自动推翻旧结果，但基于它的新结论必须先修实现，
不能靠换种子或换阈值绕过。策略只属于分析工具，与冻结清单、P5T 的
α/Holm 规则和实验种子无关。

**Verification.** `python test_review_fixes.py` 通过 532/532（0 skipped），
新增 L183（二项尾已知值、保守通过、200/200 拒绝 fail、零功效 fail、
非法计数/α/阈值 fail-closed、两次运行确定、全局 RNG 不变、六条路径齐全）
与 L184（审计入口、报告接线、CLI 文档、过小模拟拒绝）。
`python audit_gates.py calibration` 在默认策略（seed=20261023, n_sim=2000）
下 status=`ok`：null 拒绝率 0.038–0.061，正控制 0.78–1.0，鞍点对精确
max |Δp|=0.0229、mean=0.00084。`python -m explib.confirmation plan` 仍 72/72 合成
冻结记录、6 个比较。

## 2026-10-23 — 区间估计器也要先校准（no new runs）

零 GPU 的方法论硬化；不新增训练、不读 test split、不改冻结清单的
任何字节/种子/终点/α/Holm 决策规则，也不改 `CODE_SEMANTICS`。这是
§61「统计机器先校准再使用」的区间估计器补全。

**Problem.** §61 只校准了 p 值路径，但报告还直接引用逐种子配对 t 区间、
配对 bootstrap 区间、按评测单位重采样的区间，以及合并 seed×document 两个
方差源的 `_combined_seed_doc_ci`。p 值校准不能推出区间校准：同一份数据上
5% 超发的检验可以伴随只有 90% 覆盖率的 95% 区间。实测旧
`paired_bootstrap_ci` 在 n=12 的百分位构造下覆盖率约 0.907，名义却是
95%；这会让报告中「95% bootstrap 区间」的标签失真。

**Fix.** 新增 `_bootstrap_t_interval()`，把 `bootstrap_mean_ci` 与
`paired_bootstrap_ci` 从百分位法改为 **bootstrap-t（studentized）**：
每次重采样用自身标准误标准化，再取 t 分位数反变换；RNG 仍是局部、固定
种子、分块生成。n<3、重采样方差为 0、或可用 studentized 抽样太少时显式
回退百分位法，并在 `method` 中写明 fallback。`bootstrap_mean_ci` 保持
`n_boot>=20` 的原有输入契约；`paired_bootstrap_ci` 保持 `n_boot>=2` 的
原有契约（小样本走 fallback），不静默改变估计量。

`STATISTICAL_CALIBRATION_POLICY` 新增区间名义覆盖率、模拟规模、bootstrap
抽取数、已知 DGP 参数和四条宽度上限。新增两条门：

- `calibration_interval_coverage_gate`：正确校准下覆盖数
  `K ~ Binomial(n_sim, nominal)`；只拒绝显著低于名义覆盖率的欠覆盖，
  保守的过覆盖通过；阈值在模拟前固定。
- `calibration_interval_width_gate`：平均宽度必须有限、为正且不超过预先
  声明的上限，拒绝「覆盖率 1 但宽度爆炸」的空洞区间。

`statistical_calibration_suite()` 新增四条路径：
`interval_seed_t`（逐种子配对 t）、`interval_paired_bootstrap`（配对
bootstrap-t）、`interval_unit_bootstrap`（24 单位 bootstrap-t）、
`interval_combined_seed_doc`（6 seeds × 24 docs，已知 seed/doc/token
方差，目标为 `exp(mu)`，在 log-PPL 尺度检查宽度）。每个模拟重复的
bootstrap 种子由局部 RNG 流派生，避免所有重复共享同一组重采样索引。
覆盖率与宽度分别进入 `coverage_control` / `width_control`，任一 fail 即
suite fail。

**Boundary.** 校准只覆盖上述四条被报告直接引用的路径；
`unpaired_bootstrap_ci`、`paired_row_bootstrap_delta`、
`_doc_bootstrap_ppl_ci` 仍是百分位近似，未进入覆盖率门，引用时必须保留
「近似/仅该抽样源」的限定。合成数据仍是理想独立正态单位，不验证真实
面板的相关结构、重尾或 seed×document 交互设定；通过门不等于区间在真实
面板上一定准确。宽度上限是防止退化解的工程门，不是最优精度声明。
这些改动只影响区间读数，不改变精确符号翻转 p、Holm 规则或冻结判定。

**Verification.** `python test_review_fixes.py` 通过 534/534（0 skipped，
新增 L186；L183 期望集合与 L184 断言扩展为 11 条路径）。
`python audit_gates.py calibration` 在默认策略（seed=20261023, n_sim=2000）
下 status=`ok`，policy_fp=`1a5b3de5…`：检验路径保持 §61 的结果；区间路径
覆盖率/平均宽度分别为 `interval_seed_t` 0.9525/1.2369、
`interval_paired_bootstrap` 0.9440/1.2848、`interval_unit_bootstrap`
0.9520/0.8354、`interval_combined_seed_doc` 0.9775/0.6568（log 尺度），
均通过且宽度低于上限。`python -m explib.confirmation plan` 仍 72/72 合成冻结记录、
6 个比较；`python audit_gates.py source` 在刷新快照后 status=`ok`
（23/23 文件字节一致），`audit_gates.py report` 的 `calibration` 键为
`ok`。

## 2026-10-24 — 分析口径的多重性要显式对账（no new runs）

零 GPU 的方法论硬化；不新增训练、不读 test split、不改冻结清单的
任何字节/种子/终点/α/Holm 决策规则，也不改 `CODE_SEMANTICS`。

**Problem.** §61/§62 校准了 p 值与区间，但同一份冻结 panel 仍可在多种
聚合口径之间选择：逐种子 mean/median/trimmed mean、token 池化 ΔPPL、
逐评测单元 mean/median/log-ratio/win-balance。这些口径回答不同估计量，本可
不一致；报告只展示其中一个时，读者看不到方向与零排除是否随口径改变，分析
者也可以挑选最有利的口径并称其「稳健」。一个超大文档可以让 token 池化
ΔPPL 为正、而等权逐单元中位数为负；此前 `doc_sampling` 虽逐项记录，却没有
显式的口径间对账块，也没有审计门检查该块是否被省略或篡改。

**Fix.** 新增
`exp_lib.analysis_choice_concordance(primary, alternatives)`：对主口径与
声明式口径集合计算 `direction_stable`、`precision_concordant`、
`effect_range_ppl`、逐口径 `sign`/`excludes_zero`/`available`，并给出
`stable`、`precision_sensitive`、`direction_unstable`、`zero_effect`、
`insufficient_specs` 五态。它**不计算 p 值、不做多重校正、不新增检验**，
只把口径选择显式化，避免把 forking paths 塞进显著性族。

`explib.confirmation._doc_level_analysis` 现把 8 条口径写入
`doc_sampling.analysis_choice_concordance`：逐种子 mean（主口径）、
median、20% trimmed mean，token 池化 ΔPPL，逐单元 mean、median、
log-ratio、win-balance（`(n_positive - n_negative) / n_units`，负号表示
a 更常更低；全平局为 0）。种子不足 5 时 trimmed mean 显式不可用，缺失区间显式留空。
报告新增「分析口径一致性」表，列出可用/总口径、主方向、同向/反向、区间
排除零计数、PPL 范围，并在方向翻转/精度敏感时点名具体口径、提示不得只报
单一聚合。`audit_gates.py analysis_choices`（别名 `specification_curve`、
`analysis_choice`）从 `analysis_v12/test_stats.json` 重放存储的口径行：
块内状态/计数/方向/零排除不可复现，或主口径点估计与冻结逐种子均值不符，
判 fail；方向翻转/精度敏感判 warning；旧产物缺少该块判
`not_verifiable_locally`，不冒充通过。

**Boundary.** 口径同向不是统计独立复现，也不是「换一种聚合仍显著」的
证明；`precision_sensitive` 在单位/估计量不同时是正常现象，不能读成方向
失败。该块只枚举当前报告实际使用的口径，不覆盖截尾比例、加权方案、文档
过滤、长度子集等未枚举的分析选择，因此不能证明「没有其他口径会翻转」。
逐种子/逐单元/token 池化区间条件在不同抽样源上；本块只比较是否排除零，
不比较宽度，也不合并两种不确定性。全部读数仍是 post-hoc 描述，不改变
精确符号翻转 p、Holm 决策、冻结清单或 P5T 的确认判定。

**Verification.** `python test_review_fixes.py` 通过 536/536（0 skipped，
新增 L188）：五态、零/不可用口径、CI-less 口径、重复名/非法 CI/布尔点/
非有限主口径/非法 `zero_tol` fail-closed、输入不变；审计稳定→ok、方向
翻转→warnings、状态篡改/主均值不符/重复名→fail、旧产物→
`not_verifiable_locally`；合成「token 池化正、逐单元中位数为负」panel 被
标为 `direction_unstable` 且报告保留 caveat。`python -m explib.confirmation plan`
仍 72/72 合成冻结记录、6 个比较；`python audit_gates.py analysis_choices`
在无 `analysis_v12/test_stats.json` 时返回 `not_verifiable_locally`（不冒充
通过）。


## 2026-10-25 — 证据归档自带测量源码并用内容摘要绑定（no new runs）

零 GPU 的发布工程硬化；不新增训练、不读 test split、不改冻结清单的
任何字节/种子/终点/α/Holm 决策规则，也不改 `CODE_SEMANTICS`。

**Problem.** §50/§51 已让冻结锁和运行清单绑定源码树，但
`archive_release.py` 的正式 release 仍只记录 `git_head`，tar 内没有测量
源码。工作树为 dirty、分支移动、提交被变基/删除，或报告在源码编辑后
才生成时，归档无法从自身字节重建当时的代码；`git_head` 只能证明某个
提交存在，不能证明归档里的结果与该提交在同一时刻被绑定。验证者被迫
信任打包者的陈述。

**Fix.** `plan`/`make` 复用 `exp_lib.source_tree_manifest()`（与冻结锁/
运行清单同一口径），把每个测量源码文件按仓库相对路径加入归档；manifest
新增 `identity`：`schema`、`code_semantics`、`code_stamp`、`source_tree`
（逐文件 `sha256`/`bytes`、`n_files`、总字节、聚合摘要）、顶层
`source_tree_sha256`、`dependency_contract`、`execution_fingerprint` 和
精简 `environment`。聚合摘要为
`sha256(sorted relpath NUL bytes NUL sha256 LF)`。`make` 在发布前用
`_release_identity_problems()` 复核实际写入 tar 的字节，最多只能校验已
打包源码；不一致（例如打包期间源码被编辑）则删除临时 tar 并拒绝发布。
`verify` 对归档内 manifest 执行同一检查，不依赖 Git、当前工作树或
`exp_lib`。旧归档无 `identity` 时仍校验文件哈希，但打印 NOTE，明确源码
绑定不可验证。

**Boundary.** 不是环境重建：`verify` 不重新安装依赖、不重算
`dependency_contract`，也不验证 `execution_fingerprint` 记录的主机/CUDA
栈。依赖与指纹是归档时声明，需配合发布记录和 `audit_gates.py`。源码树
口径只含测量相关源码/配置，不含测试、README、docs 和结果，绑定的是代码
身份而非说明文档版本。Git 提交仍保留，但不再是源码身份的唯一依据；
dirty 工作树可被如实绑定。数值、冻结清单、`CODE_SEMANTICS`、种子、
终点与 Holm 决策均不变。

**Verification.** `python test_review_fixes.py` 通过 L190：真实 `make`
的 tar 含 `exp_lib.py`/`archive_release.py` 等源码，`identity` 与 per-file
哈希一致，`verify` 通过；删除源码映射、篡改顶层源码摘要或清空
`code_semantics` 被身份检查拒绝；tar 内 manifest 的
`source_tree_sha256` 改为错误值后，即使文件哈希全匹配，`verify` 返回 1。
`python archive_release.py verify <archive>` 的旧归档路径保持兼容。

## 2026-10-25 — 跨 panel 消融门识别反证并报告复现状态（no new runs）

零 GPU 的方法学硬化；不新增训练、不读取 test split、不修改冻结清单的
字节/种子/终点/alpha/Holm 决策规则，也不改 `CODE_SEMANTICS`。

**Problem.** `ablation_coverage_across_panels` 只做覆盖计数：组件只要在
任一 panel 完整且 verified 就进入 `verified_anywhere`，即使另一个完整
panel 给出族校正后的相反方向或超过 SESOI 的有害移除效果，
`all_components_useful_allowed` 仍可能为真。这等于按 panel 挑选证据，
也会把单 panel verified 误读为已复现。

**Fix.** 每个组件的跨 panel 行新增
`panels_direction_conflict`、`panels_practically_harmful`、
`panels_opposite_mean`、`panels_measured_unverified`、
`panels_replicated` 和 `replication_status`；顶层新增
`replicated_anywhere`、`single_panel_verified`、
`direction_conflict_anywhere`、`practically_harmful_anywhere`、
`opposite_mean_anywhere`。`all_components_useful_allowed` 在任意
反证/实际有害时关闭，并在 reasons 中指名冲突 panel；incomplete 与
verified-only-incomplete 旧规则不变。`audit_methodology` 将跨 panel
列表提升到报告顶层。该门不重算 p、不合并效应，只消费已有族校正
verdict；SESOI/Holm 语义不变。

**Boundary.** `replicated` 只表示 >=2 个完整 panel 均 verified，不是
独立复现或外部效度：共享种子、切片、训练配方时误差仍相关。该门不做
meta-analysis，不估计合并效应或异质性；需要合并时仍用显式
`meta_analyze_effects`。反方向但未达族校正显著的均值只产生警告，不单独
关闭项目门，避免小样本噪声翻转；未声明方向的对比不算冲突。冲突 panel
即使 ablation 不完整也保留点名，不能用缺臂抹掉反证。

**Verification.** `python test_review_fixes.py` 通过 539/539（0 skipped，
新增 L191）：p1 verified、p2 完整但方向冲突时 A 为 `contradicted` 且项目
门关闭并点名 p2；B 在两个完整 panel verified 时为 `replicated`；
practically harmful 同样关闭门；非显著相反均值只产生
`opposite_mean` 警告。未改冻结清单、数值或 `CODE_SEMANTICS`。

## 2026-10-25 — train/validation 近重复段落门与覆盖诚实（no new runs）

零 GPU 的数据处理/方法学硬化；不新增训练、不读取 test split、不修改冻结
清单的字节/种子/终点/alpha/Holm 决策规则，也不改 `CODE_SEMANTICS`。

**Problem.** `audit_corpus_split` 与 `audit_split_discipline` 只检查
validation 行是否与 train 逐 token 完全相同。validation 复用训练文档中的
长段落但改写其余 token 时不会被发现；原始行级 `_DocDeduplicator` 也不能
保证跨行/半行复用被清除。同时，大 train 上只扫描一小部分，扫描干净仍可能
返回 `ok`，把「未发现」当成「已证明」。validation 是所有探索性 panel 的
选型切片，这会使选型与后续确认共享未披露的污染。

**Fix.** 新增 `_passage_overlap_from_segments()`：在既有 16 段确定性 train
窗口和 validation 行上计算 32-token shingle 哈希；报告 validation shingle
重叠比例、单行最大重叠、train 内部 A/B 重复基线、行级 train 内部基线、
覆盖比例和策略字段。集合级仅在 `val_overlap > 0.25` 且超出基线
`> 0.10` 时 fail；行级仅在 `max_row > 0.50` 且超出 row baseline
`> 0.10` 时 fail；覆盖低于 `min_scan_fraction` 返回 `not_verifiable`。
`audit_corpus_split` 复用已有 `_segments` 并新增 `passage_overlap` 块、
`exact_row_scan_sufficient` 和低覆盖 warning；`audit_split_discipline` 在
exact-row 检查后运行同一 helper。新参数全部进入 memo key 并做有限值/
`[0,1]` 校验。

**Boundary.** 32-token exact shingle 不是语义/改写检测：同义改写、翻译
回译、摘要复用仍可能漏掉，不能替代嵌入级近重复或人工审阅。扫描是样本，
覆盖足够也不证明未扫描区域干净；覆盖不足只标 `not_verifiable`，不冒充
通过。train 内部基线高时，与常见模板无法区分的泄漏仍可能漏掉；行级阈值
只抓「单行大半是训练段落」的强信号。该改动不改变数据构造、分词器、冻结
清单、种子、终点、alpha、Holm 或 `CODE_SEMANTICS`；不新增训练/测试读取。

**Verification.** `python test_review_fixes.py` 通过 542/542（0 skipped，
新增 L193）：干净 split ok；复制 train 段落后改一个 token（exact-row=0）
被 passage 行级门 fail；精确复制同时触发两个门；重复模板语料基线≈1、
超额 0 不误报；5% 覆盖降为 warnings/`not_verifiable`；短行
`not_applicable`；`audit_split_discipline` 对干净/污染 cache 分别 ok/fail；
NaN/负值/超范围参数被拒绝。`python -m explib.confirmation plan` 仍 72/72 合成冻结
记录、6 个比较。

## 2026-10-26 — 2x2 交互的 SESOI/TOST 等价性与设计门（no new runs）

零 GPU 的对比实验/消融方法学硬化；不新增训练、不读取 test split、不修改
冻结清单的字节/种子/终点/alpha/Holm 决策规则，也不改 `CODE_SEMANTICS`。

**Problem.** §59 的 `factorial_interaction_audit` 只有交互族 Holm 的
精确符号翻转：`p_holm>=alpha` 不是「两个模块可加」，n 大时一个亚 SESOI
的交互也可能显著。留一消融在 §56 已有预注册 margin + TOST，但交互层仍
把「未检出」与「实际可加」混在一起，也没有把「可检出但实际很小」与
「超过 SESOI 的子可加/协同」分开。

**Fix.** 新增 `INTERACTION_SESOI`、`_normalise_interaction_sesoi_design()`、
`interaction_practical_decision()` 和 `interaction_sesoi_audit()`。
margin 在自然对数 ratio-of-ratios 尺度上定义，`relative_pct=1.0` 换算为
`log(1.01)`；拒绝布尔/0/负值/NaN/inf/双单位/非字符串名。
`factorial_interaction_audit(..., sesoi=None)` 默认为每个可测交互计算
配对 TOST，并把 TOST p 放进独立的交互等价族做 Holm；不完整/缺失
factorial 保留 p=1 槽位。判定分为 `hypothesis_supported` /
`direction_conflict` / `sub_sesoi` / `additive` / `indeterminate` /
`underpowered` / `no_data` / `not_measured` / `not_declared`。
`additive` 要求 TOST 区间在 margin 内且族校正 TOST p<alpha；非显著且
未做等价检验一律 `indeterminate`，不能写成 additivity。交互单元新增
`p_tost_holm`、`practical_decision`、`sesoi`；顶层新增
`equivalent_interactions`、`hypothesis_supported_interactions`、
`sub_sesoi_interactions`、`indeterminate_interactions` 等列表。
设计期门 `audit_gates.py interaction_sesoi` 检查每个 factorial 的 margin
覆盖、单位、正性、拼写与方向假设；`audit_gates.py report` 纳入该块。

**Boundary.** `additive` 是预注册 margin 内的实际可加性，不是精确零
交互，也不是机制确认：容量/配方混杂仍由 `mechanism_claim_allowed`
独立把关。`sub_sesoi` 仍可检出，不能写成「无交互」；`indeterminate`
不能写成「可加」；`underpowered` 不能被等价性绕过。旧
`INTERACTION_HYPOTHESIS_DESIGN[*]['falsifier']` 文本不改写：
"not distinguishable from zero" 现在只是 `indeterminate`，只有
TOST 区间落在 margin 内的 `additive` 才是对「实际交互」的正反证。
交互族仍是探索性族，
不进入冻结 confirmatory 或留一消融 Holm 族；不新增训练、种子、终点或
α，不合并 panel，也不改变任何已落盘数值。

**Verification.** `python test_review_fixes.py` 通过 546/546（0 skipped，
新增 L195-L197）：交互 SESOI 设计门全通过/缺 margin/歧义单位/NaN/布尔/
双单位 fail、拼写 warning、空设计 `not_applicable`；正向支持/反向冲突/
亚 SESOI/精确可加/不确定/n<min_n/缺臂/`sesoi={}` 关闭/部分声明保留
p=1 槽位/无完整 seed 列表不重复/畸形声明 fail，以及三个重复 factorial
下超显著性与 TOST 两族 m=3 的 Holm；直接判定原语、
单次迭代器、相对 margin 换算、畸形族 p 的 fail-closed 与输入校验。
`python audit_gates.py interaction_sesoi` 返回 `ok`；`python -m explib.confirmation
plan` 仍 72/72 合成冻结记录、6 个比较。

## 2026-10-27 — 预注册方向性主张的三向判定：确认、反证与不确定（no new runs）

零 GPU 的确认性分析方法学硬化；不新增训练、不读取 test split、不修改
冻结清单的字节/种子/终点/alpha/Holm 决策规则，也不改 `CODE_SEMANTICS`。

**Problem.** §41/§42 的冻结确认规则只有 `confirmed` 与「其余」两态。当一个
比较在 Holm 校正后显著、但方向与预注册相反时，旧报告只能把它归入「未确认」
或「探索性」，这会掩盖证据已经与方向性主张冲突；反过来，直接写成「显著
更差」「实际等价」或「相反机制成立」又超出了 test 与 SESOI 声明能支持的
范围。

**Fix.** 新增 `exp_lib.directional_claim_verdicts()`：在同一个 Holm 拒绝
集合上，按预注册方向与既有有效性门做三向划分——`supported` / `refuted` /
`inconclusive`（另有 `not_testable`、`significant_nondirectional`）。反向
显著但覆盖/参数/初始化/样本下限未过时只记 `inconclusive` 并进入
`reverse_significant_unverified`；反向不显著进入
`opposite_nonsignificant`；无方向的显著读数进入
`significant_nondirectional`，永不进入 supported/refuted。参数门明确：
`param_matched=false` 始终失效，清单声明匹配契约时还要求
`param_match_complete=true`，缺记录不是通过。

`explib.confirmation.finaltest_stats` 写入 `verdicts`、`verdict_summary`、
`refuted`、`inconclusive`、`not_testable`、`opposite_nonsignificant`、
`reverse_significant_unverified`、`verdict_consistency`，并为
`not_confirmed_reasons` 补 `verdict`/`verdict_reason`/`gate_failures`；
冻结 `confirmed` 原值保留，不一致只警告并记录。`build_report` 增加
`_refutation_markdown`，单独渲染反证、显著反向但未过门、反向不显著三类，
阻止它们被「未确认/探索性」段吞掉。`audit_gates.py` 新增
`audit_confirmation_verdicts`：读取 `analysis_v12/test_stats.json`，用存储
的 `holm[*].p_holm`、方向与门独立重放三向划分，核对逐项 verdict、顶层
列表和 `confirmed == supported`；篡改/重复归类/不一致判 fail，缺三向字段
的旧 artifact 判 `not_verifiable_locally`，并接入 `report` 的 `verdicts`
键与 CLI 别名 `verdicts`/`verdict`/`three_way`。

**Boundary.** 反证不是新检验：它继承同一双侧 Holm 族的 FWER 控制，不另算
单侧 p、不增加 alpha。没有预注册 SESOI 时，`refuted` 只说明预注册的方向性
优势不成立，不是实际等价，也不能写成「显著更差」；`mechanism_claim_allowed
=false` 的 system-level 比较只否定所测系统差异，不证明相反机制。反向显著
但未过有效性门不是反证。旧 artifact 不追溯升级。该改动不改变任何已落盘
p/Holm/`confirmed` 数值，也不改变冻结清单或实验执行。

**Verification.** `python test_review_fixes.py` 通过 559/559（0 skipped，
新增 L206-L208）：L206 覆盖三向划分、反向显著但门未过、无方向/缺失/省略、
参数契约、样本下限、p==alpha、非法输入与入参不变；L207 覆盖
`finaltest_stats` 的 supported/refuted/inconclusive 三态、
`not_confirmed_reasons.verdict`、`verdict_consistency` 与反证/反向不显著
文案；L208 覆盖审计正常 `warnings`、列表/逐项 verdict 篡改 `fail`、旧
artifact/缺文件 `not_verifiable_locally` 以及 `run_report`/CLI 接线。
`python -m explib.confirmation plan` 仍 72/72 合成冻结记录、6 个比较。

## 2026-10-27 — 参数匹配基线的 SESOI/TOST 与设计期声明门（no new runs）

零 GPU 的对比实验方法学硬化；不新增训练、不读取 test split、不修改冻结
清单的字节/种子/终点/alpha/Holm 决策规则，也不改 `CODE_SEMANTICS`。

**Problem.** 留一消融与 2×2 交互已有预注册 SESOI + TOST，但
`compression_vs_matched_dense` 仍只按 Holm 显著性判 `verified`。大 n 下
一个 0.1% 的 PPL 差异可以显著，却不构成方法相对参数匹配稠密基线的实际
优势；反过来，非显著也没有等价性判定。`methodology_audit` 的默认 margin
只覆盖 `ABLATION_SESOI`，baseline 条目不会获得 SESOI。

**Fix.** 新增 `exp_lib.BASELINE_SESOI`（当前
`compression_vs_matched_dense` = 相对 1%）与
`DEFAULT_METHODOLOGY_SESOI`（消融 + baseline 并集）。`methodology_audit`
默认对 baseline 做相同的配对 TOST + 声明族 Holm；显著但低于 margin 进入
`sub_sesoi_baselines`，只有超过 margin 的 `superior` 才进入
`verified_baselines`，显式 `sesoi={}` 保持旧统计判定。`report['sesoi']`
新增 `declared_baselines`，不再把基线键混进 `declared_components`。
新增 `audit_gates.audit_baseline_sesoi()` /
`python audit_gates.py baseline_sesoi`：在设计期检查每个
`BASELINE_DESIGN` 对照的 margin 覆盖、单位、正性、拼写和
`RUN['variants']` 排程；`audit_gates.py report` 汇总新增
`baseline_sesoi`。

**Boundary.** 这是探索性方法学门，不进入冻结 confirmatory 的 p/Holm 族，
不改变任何已落盘数值。`sub_sesoi` 是可检出但小于实际边界，不能写成零
效应；`equivalent` 只说明 margin 内无可区分优势，不是方法更优；部分
baseline panel 的等价结果不能授权整条主张。margin 仍需在看结果前固定，
事后调整会使 TOST 失去预注册含义。

**Verification.** `python test_review_fixes.py` 通过 561/561（0 skipped，
新增 L210）：默认声明覆盖与排程、缺 margin/拼写/畸形/未排程臂的
fail 或 warning、0.2% 差异停在 `sub_sesoi` 且不进入
`verified_baselines`、2% 差异进入 `verified_baselines`，以及
`sesoi={}` 恢复旧判定。`python audit_gates.py baseline_sesoi` 返回
`ok`；`python audit_gates.py sesoi` 仍 `ok`；`python -m explib.confirmation plan`
仍 72/72 合成冻结记录、6 个比较。

## 2026-10-27 — 坏例桶的 row-cluster 发现/复现推断与重放门（no new runs）

零 GPU 的结果分析方法学硬化；不新增训练、不读取 test split、不修改冻结
清单的字节/种子/终点/alpha/Holm 决策规则，也不改 `CODE_SEMANTICS`。

**Problem.** `compare_token_errors` 的桶级报告按 token 计均值/样本量，长行
会对短行形成伪重复；而「最差桶」是看过同一批 validation 数据后挑出的，
再对同一个桶报 p 值/区间会受 winner's curse 影响。现有全样本桶 p 没有
多重性之外的校正，也没有说明桶是在哪一半上被选中的。

**Fix.** 新增 `exp_lib.paired_error_bucket_inference()`：以 validation row
为 cluster，先聚合 token，再对 cluster 做确定性 SHA-256 两半切分；只在
discovery 半边按方向选最差桶，在 disjoint replication 半边报告均值、t
区间、exact sign-flip p、同向性、shrinkage 和排名。
`replicated = same_sign AND replication 95% CI excludes 0`（不要求
replication p≤α，因为 3 个 cluster 的 exact p 下限是 0.25）。全样本桶 p
保留 Holm，但明确标记为 `post_hoc_full_sample_descriptive`。
`compare_token_errors(..., train_ids=..., val_bnd=...)` 构造 position /
target rarity / boundary 三个家族；缺少辅助输入时记录原因并省略。
`error_analysis.py` 的 `bad_cases.md`、`build_report.py` 的主报告渲染该
block；新增 `audit_gates.audit_error_case_inference()`（CLI
`error_case_inference` / `error_buckets`），从 `paired_errors.json` 存储的
cluster units 重放均值、exact p、Holm、切分、选桶、replication 符号/区间
与 `replicated`，并接入 `audit_gates.py report`。

**Boundary.** 这是固定 validation 文本上的**内部稳定性**检查，不是总体
推断、不是新数据、不是独立文章复现。WikiText 的 row 是原始行/窗口，同一
文章的不同行仍可能跨两半，row-level 切分不能消除文章级相关。
`replicated=True` 不是确认性证据，不进入冻结 p/Holm 族，不改变
`confirmed`；全样本 Holm 显著也不能写成「选择校正后的显著」。两半 p 值
有分辨率下限，小样本应以区间/符号为准。

**Verification.** `python test_review_fixes.py` 通过 `563/563`（0 skipped，
新增 L211/L211b）：覆盖确定性/不重叠切分、discovery 选桶、replication
复现、token 级伪重复不作为样本数、`no_data`/`no_candidate` 区分、非法输入
fail-closed、artifact 重放通过/篡改 `fail`/缺文件 `not_verifiable`、
辅助家族缺输入时记录原因，以及 EA/BR Markdown 的 `replicated` 与边界
文案。`python audit_gates.py error_case_inference` 对合成 artifact 返回
`ok`、篡改后 `fail`；缺文件返回 `not_verifiable`。`python -m explib.confirmation
plan` 的冻结计划/比较数不受影响。

## 2026-10-27 — 设计期功效把「分析能否运行」纳入：attrition-adjusted power（no new runs）

零 GPU 的设计期方法学硬化；不新增训练、不读取 test split、不修改冻结清单
的字节/种子/终点/alpha/Holm/覆盖规则，也不改 `CODE_SEMANTICS`。

**Problem.** §68 的 `paired_design_sensitivity`、`python -m explib.confirmation power` 与冻结
功效门都条件在完整面板上。当前清单声明 `require_complete_coverage=true`，
缺失一个配对就使该比较 unconfirmed，而不是用剩下的 k 个配对做小 n 确认；
但已有读数仍报 `signflip_power(n,m,q,alpha)`，没有报「可运行且拒绝」的联合
概率。n=12、m=6、q=0.95 的条件功效是 0.882；若配对完成率 p=0.95，完整覆盖
的可运行概率只有 `0.95^12=0.540`，联合功效降到 0.476，且任何名义 n 都无法
在 p=0.95 下达到 80%（n 增大提高条件功效，但 `p^n` 同时衰减）。

**Fix.** 新增 `exp_lib.attrition_power_audit()`：K ~ Binomial(n,p)，完整覆盖
时阈值 τ=n，否则 τ=max(Holm 下限, min_complete_pairs)；报告可运行概率
`P(K>=τ)`、联合功效 `Σ_{k>=τ} P(K=k)*signflip_power(k,m,q,alpha)`、期望完成
配对数、达到目标所需最低 p、所需名义 n，以及 p=1.00/0.99/0.97/0.95/0.90 的
压力表。`pair_retention` 必须显式声明；缺失时为 `not_declared`，不伪造通过；
`require_attrition_power=true` 时才 fail-closed。`python -m explib.confirmation power` 的每条
方向性对照新增 `attrition` 块（q 优先取 pilot sign rate，缺失时退回
`power_q_for_test`）；`audit_gates.audit_design_sensitivity()` 在每条对照和
顶层汇总中接入，并支持可选的 `statistics.attrition.pair_retention` /
`availability_target` / `require_attrition_power`。原有 q=0.95 精确条件功效
门不变。

**Boundary.** p 是设计假设，不是观测 attrition；独立同分布是最乐观情形，
成组失败或与效应相关的丢失会更差。条件功效仍是等幅符号检验投影，不把
|d_i| 不等幅或种子间相关变成精确 sign-flip。完整覆盖下的 `p^n` 是可运行
概率的上界，补名义种子不能把已发生的缺失变成随机可忽略损失；最低保留率
也必须在看 test 前固定，不能事后挑选。

**Verification.** `python test_review_fixes.py` 通过 `564/564`（0 skipped，
新增 L212）：完整覆盖联合功效、5% 损失 0.882→0.476、8 配对下限与完整覆盖
的不同失败模式、所需名义 n=13、最低保留率约 0.992、低于 Holm 下限的 warning、
未声明/要求声明/满足三态、非法输入，以及 audit 清单三态。`python
python -m explib.confirmation plan` 仍 72/72 合成冻结记录、6 个比较；`python audit_gates.py
report` 的 `sensitivity` 块已输出 attrition 压力表，当前清单未声明 retention，
因此保持 warning 而非绿色通过。

## 2026-10-28 — NIAH 机制对照的单因素声明与推理 override 审计（no new runs）

零 GPU 的设计/记录硬化；不新增训练、不读取 test split、不修改冻结清单的
字节/种子/端点/alpha/Holm 规则，也不改 `CODE_SEMANTICS`。

**Problem.** v10 的 NIAH 对照（randidx、allblocks、dense、from-scratch
random/all-block）进入统计与报告，却没有 `single_factor`/`factors` 声明，
也没有定义点审计。`allblocks` 的 H3 文本还写成「same top-k block budget」，
但该臂实际把 top-k 放大到全部块，等于把读取预算干预误写成等预算机制对照。

**Fix.** 新增 `exp_lib.NIAH_INFERENCE_ARM_OVERRIDES` 与
`apply_niah_inference_arm()`：推理臂切换时先把声明键重置为 checkpoint
基线，再应用当前臂 override，避免同 seed 跨臂复用时残留上一臂的键。
`python -m explib.corruption` 改用该共享表，不再在 probe 路径硬编码两行赋值。
新增 `NIAH_CONTRAST_FACTORS` 与 `niah_contrast_factor_audit()`：推理期
条目核对共享 override 的键集，trained/dense 条目通过 v7 工厂重算 AttnCfg
diff；声明和实际键集双向一致、单因素标签落在同一 cfg 族内才 verified。
`audit_gates.audit_single_factor()` 新增 `niah_contrasts` 块。v10 的 NIAH
NLL/accuracy 记录携带 factor metadata，审计未通过时保留
`declared_single_factor=True` 并把 `single_factor` 清零。H3 文本改为
read-rich all-block control，并明确不是等读取预算比较。

**Boundary.** randidx/allblocks 是同一训练 checkpoint 的推理期干预；
csa_randtrain/csa_alltrain 才是从头训练对照。allblocks 仍是单因素（只动
`index_topk`），但不能写成同预算下的选择优势；dense 是架构级对照。
NIAH 的 top-1 accuracy 仍是主端点，单因素声明不改变 Holm 族、种子、阈值
或冻结决定。

**Verification.** `python audit_gates.py single_factor` 返回整体 `ok`，
`niah_contrasts` 五项全部 verified；`python test_review_fixes.py` 通过
`565/565`（0 skipped，新增 L62c）。覆盖共享 override 的键集、
跨臂不残留、错误 factors/override 标签 fail、记录层 fail-closed。

## 2026-10-28 — CI 与解释器/平台纳入依赖锁契约（no new runs）

零 GPU 的可复现性硬化；不新增训练、不读取 test split、不修改冻结清单的
字节/种子/端点/alpha/Holm 规则，也不改 `CODE_SEMANTICS`。

**Problem.** `requirements.lock` 是完整 48 包闭包，但 CI 安装的是另一份
只有五个直接依赖的 `requirements-ci.txt`；pip 会自行解析 transitive 版本，
所以绿色 CI 不能证明锁环境通过。lock 本身也只钉包版本，没有声明生成它的
Python patch、实现和平台，无法阻止同一份 lock 在另一套解释器上被当作
“匹配”。

**Fix.** `requirements-ci.txt` 改为只含 `-r requirements.lock`，CI 与本地
共用同一份完整闭包。`requirements.lock` 头新增机器可读的
`# python-version: 3.12.15` / `# python-implementation: CPython` /
`# platform: Linux x86_64`；`parse_requirements_lock()` 解析这三条声明，
`dependency_contract()` 新增 `interpreter` 子契约并比较实际运行值，声明
不一致即 `mismatch`，旧 lock 无声明时报告 `unverified` 而非静默匹配。
`audit_gates.audit_dependencies()` 输出 declared/actual/problems 明细。
同时把 lock 与实际已验证环境对齐（`fonttools==4.68.0`、
`huggingface_hub==2.3.0`、`multidict==7.2.1`），并把开发期 `ruff` 明确
排除在运行时闭包覆盖之外。

**Boundary.** 解释器 clause 是声明，不是自动证明；CUDA/其他平台必须重新
生成自己的 lock 并改写三条 clause，不能复用 CPU lock 的“匹配”。该契约只
进入 provenance/audit，不参与训练、resume、pairing、P5T admission 或冻结
manifest。`requirements-ci.txt` 不再保存直接 pin，避免第二份列表再次漂移。

**Verification.** `python test_review_fixes.py` 通过 `567/567`（0 skipped；
新增 L138a 解释器/平台匹配、三条 clause 各自 mismatch、legacy `unverified`、
冲突声明 invalid；L140a CI 只引用 lock）。`python audit_gates.py
dependencies` 返回 `status=ok`、`contract_status=ok`、
`interpreter_status=match`，48 个 pin 与 50 个已安装分布一致（`ruff`
作为开发工具排除）。`python -m explib.confirmation plan` 复跑确认计划/快照未变。

## 2026-10-29 — P5T 最终测试评估顺序预注册与测量时间混杂审计（no new runs）

零 GPU 的设计/记录硬化；不新增训练、不读取 test split、不修改冻结清单的
字节/种子/端点/alpha/Holm 规则，也不改 `CODE_SEMANTICS`。

**Problem.** §48 已把训练面板改成 seed 分块随机顺序，但 P5T 的 clean
by-length 与 distractor probe 仍按 arm-major 循环（先跑完一个臂的全部
seed，再跑下一个臂）。这会让确认性终点的臂效应与测量时间/机器漂移完全
共线，配对符号检验无法修复；评估中断后的恢复也没有落盘顺序可供复核。

**Fix.** 新增 `exp_lib.build_evaluation_order_plan()`，复用
`blocked_run_order` 的 seed 区组、块内臂随机化与位置轮换，标签采用 P5T
事件键 `variant::s{seed}`。P5T 在 `preaccess` 锁之后、`load_wikitest`
之前分别写下 `clean_by_length_order.json` 与
`distractor_probe_order.json`；同清单恢复复用原计划，网格不匹配只有在
test 尚未绑定时才允许重写，绑定后拒绝事后洗牌。
`evaluation_cells_in_plan_order()` 要求计划与 (arm, seed) 网格完全一致。
新增 `audit_gates.audit_evaluation_order()`：核对完整区组、位置平衡、
计划覆盖、`created_at` 早于首条测量，以及 `experiments.jsonl` 的实际
顺序；clean 有测量无计划或实际 arm-major 判 fail，计划无事件报不可核验，
probe 缺计划只给 warning。该审计接入 `report` 并提供 `evaluation_order` /
`eval_order` / `measurement_order` CLI 别名。

**Boundary.** 顺序计划只控制测量时间与臂的共线，不修复已发生的漂移；
P5T 仍串行，不改变 checkpoint、test 行、PPL、种子、终点、α/Holm 或冻结
决定。`created_at` 与事件时间比较假设同一时钟口径，时钟回拨会 fail-closed。
旧 P5T 结果没有预计划时会被明确标为 fail/不可核验，这是证据强度标注，
不是追溯性修复；新的确认性轮次必须从预计划开始。

**Verification.** `python test_review_fixes.py` 通过 `577/577`（0 skipped；
新增 L213/L214/L215：计划确定性/区组/位置平衡/网格覆盖、P5T 预计划写入/
恢复复用/pre-access 重写/post-access 拒绝、审计的计划顺序/arm-major/事后
计划/缺计划/无事件/损坏计划）。`python -m explib.confirmation plan` 仍
72/72 合成冻结记录、6 个比较；`python audit_gates.py single_factor` 仍
`ok`。当前仓库没有 `results_lm_test`，所以
`python audit_gates.py evaluation_order` 报 `not_verifiable_locally` 而不是
绿色通过；合成测试覆盖了 ok/fail 分支。

## 2026-10-29 — 分析产物的内容寻址溯源与输入绑定（no new runs）

零 GPU 的可复现性硬化；不新增训练、不读取 test split、不修改冻结清单的
字节/种子/端点/alpha/Holm 规则，也不改 `CODE_SEMANTICS`。

**Problem.** `stamped_analysis` 已经记录 code/git/environment provenance，
但没有绑定（a）产物正文本身和（b）它实际读取的输入文件。手工改一个
p/verdict/curve，或在分析之后重写 `bylen.json`/manifest/summary，产物仍会
带着同一 code stamp 看起来“当前”；legacy 产物与刚重跑的结果也无法区分。
这使“报告中的数字来自磁盘上的哪份输入、哪段代码”只能靠人工记忆。

**Fix.** `stamped_analysis(..., inputs=...)` 现在可选记录每个输入文件的
仓库相对路径、SHA-256 与字节数（`analysis_input_manifest`）；同时写入
`content_sha256`，对结果正文（排除 `kind`/`created_at`/`code`/`provenance`/
`inputs` 等易变字段）做键排序、分隔符稳定的 SHA-256。新增
`verify_stamped_analysis()`：重算正文摘要并复核所有本机仍存在的输入哈希；
正文被改或输入在分析后被改判 fail，输入不在本机给 warning，legacy 无摘要
明确报 not_verifiable。确认统计入口 `finaltest_stats` 绑定
manifest/bylen/probe，`run_power` 绑定 manifest/summary。
新增 `audit_gates.audit_analysis_integrity()`（CLI `analysis_integrity` /
`artifact_integrity` / `content_integrity`）：只扫描带 stamp 的
`analysis_*/*.json` 与 `audit_report.json`，逐产物报告 ok/fail/legacy；
接入 `audit report`，并提供 `make integrity` 零 GPU 入口。

**Boundary.** 这是篡改/陈旧检测，不是密码学签名：能重写正文并同时重算
摘要的人仍可伪造。外部输入（仓库外路径）按绝对路径记录，换机器后可能
显示“输入不在本机”；这是显式 warning，不把归档产物误判为损坏。该机制
不重算任何统计量、不改变 p/Holm/confirmed 判定；缺少输入的归档产物仍可
作为证据使用，只是不能在本机完整复核推导链。

**Verification.** `python -m pytest -q` 通过 `581 passed, 3 skipped`
（新增 `test_review_part_13.py` 6 项：正文摘要稳定性/易变性隔离、正文与
输入篡改 fail、缺失输入 warning、malformed digest/inputs fail、输入路径
规范化、audit 扫描与 report/CLI 接线、实验事件携带摘要）。
`make integrity` 在无新分析产物的当前仓库返回 `warnings`（只点名 legacy
`audit_report.json`）而不是假绿；合成测试覆盖 ok/fail/not_verifiable 分支。

## 2026-10-29 — 实验事件账本的逐事件哈希链与尾锚点（no new runs）

零 GPU 的可复现性硬化；不新增训练、不读取 test split、不修改冻结清单的
字节/种子/端点/alpha/Holm 规则，也不改 `CODE_SEMANTICS`。

**Problem.** `experiments.jsonl` 是 `audit_run_order` /
`audit_evaluation_order` / 失败集中度等审计重建「实际执行顺序与失败」的
唯一依据，但它只是普通 append-only JSONL：删除一条 error 事件、改一个
`started_at`、删掉尾部截断记录，文件仍是合法 JSONL；`_read_experiment_events`
对坏行 `continue`，一行损坏会被静默跳过。分析产物已有正文/输入哈希，而
产生这些产物的执行账本没有完整性证明，顺序与失败读数因此可被单行编辑
改变而不留痕。

**Fix.** `append_jsonl()` 在原有 flock + 单写 + fsync 之上，为每条事件
写入保留的 `_ledger={schema,seq,prev,sha256}`：`sha256` 对
`domain || seq || prev || canonical_json(event without _ledger)` 计算，第一
条的 `prev` 为 64 个 0；每次追加在同一把锁内原子替换
`experiments.jsonl.anchor.json`（`count`/`last_sha256`/`bytes`），从而同时
检测编辑、插入、删除、重排与尾部截断。正常追加只读最后一行 + 锚点（O(1)），
legacy 前缀/崩溃窗口才全量重扫；若重扫发现锚点领先于账本（尾部被截断）
或锚点最后链接与重扫前缀不符，追加会拒绝并报警，而不是把更短的历史重新
锚定。新增
`exp_lib.verify_experiment_ledger()`：ok / warnings（缺锚点或锚点落后）/
fail（链断、正文改、坏 JSON、锚点损坏或 count 大于账本）/
not_verifiable_locally（空账本或全 legacy）；legacy 行以原始字节哈希作为
链接贡献，因此链可以从旧前缀之后开始；但链化开始后再出现 legacy 行判
fail，防止删掉尾部 `_ledger` 后把编辑过的链化事件伪装成 legacy 尾行。
调用方传入的 `_ledger` 一律忽略并
重写。新增 `audit_gates.audit_experiment_ledger()`（CLI
`experiment_ledger` / `event_ledger` / `ledger`），扫描 `results_*/` 与
`analysis_*/` 下的账本，接入 `report` 的 `experiment_ledger` 块。

**Boundary.** 这是 tamper-evident，不是密码学签名：同时重写账本、重算链
并替换锚点的人仍可伪造；它不能替代外部时间戳或只读归档。追加与锚点替换
不是同一个原子操作，二者之间被杀会留下「锚点落后」的 warning 而不是
fail；下一次追加会重扫并写回新锚点。若锚点反而领先于账本（截断）或与
重扫前缀不符，追加会拒绝并只打印 warning，避免把截断证据重新锚定掉。
legacy 事件被链绑定但没有写入时的
自证，审计报告 `n_legacy`，不把旧账本升级成新证据。该机制只覆盖执行
事件日志，不替代 `stamped_analysis` 的产物/输入哈希，也不替代
`archive_release.py` 的发布清单；不改变任何测量、p 值、Holm 族、顺序
审计语义或冻结决定。

**Verification.** 新增 `test_review_part_14.py` 6 项：链与锚点自洽、
正文编辑/删除中间事件/截断尾部 fail、锚点缺失/落后 warning 与损坏 fail、
legacy 前缀 + 无尾换行追加、NaN/NumPy/集合 JSON 安全与 `_ledger` 伪造被
覆盖、审计目录输入/混合 ok+fail/report/CLI 接线。`python -m pytest -q`
通过 `588 passed, 3 skipped`。当前仓库只有 legacy `analysis_audit`
账本，所以 `python audit_gates.py experiment_ledger` 返回
`not_verifiable_locally` 而不是绿色通过；`python -m explib.confirmation
plan` 的冻结计划/比较数不变。

## 2026-10-29 — 运行分配的前瞻性、设备槽位与臂×环境共线审计（no new runs）

零 GPU 的实验设计/准备硬化；不新增训练、不读取 test split、不修改冻结
清单的字节/种子/端点/alpha/Holm 规则，也不改 `CODE_SEMANTICS`。

**Problem.** §48 已把训练面板改成按种子分块并轮换臂位置，§76 又把 P5T
评估顺序预注册；但 `run_order.json` 的 `created_at` 从未与第一条 `cell`
事件比较，执行结束后补写的「看起来随机」的计划仍可通过 `audit_run_order`。
同时，多卡调度按空闲 worker 派发，同一 seed 的两臂可能固定落在不同
GPU；`exec_fp` 有意排除 GPU UUID/hostname，因此配对有效性门不会发现
「臂 × 设备」完全共线。`provenance.environment` 已经记录 host/CVD/GPU/
CUDA 栈，却没有任何审计读取它。

**Fix.** 新增 `exp_lib.build_run_order_plan(..., environment_slots=...)`：
按块内位置循环分配 `cell_environments`，使每个臂在每个槽位出现次数相差
不超过 1；不叠加块号旋转（会与位置旋转抵消）。`write_run_order` 对同一
planned_cells/blocks/design seed 的重建保持幂等，resume 不会用新时间戳
覆盖第一次写入的预注册计划。新增
`run_order_plan_timing(plan, events)`：计划晚于首条 cell 判 fail，缺时间戳
报 not_verifiable，坏时间戳 fail，时区不一致 not_verifiable。新增
`environment_identity` / `environment_balance_audit` /
`environment_plan_audit` / `run_allocation_audit`：两臂环境集合不相交且
各有 >=2 条记录判完全共线 fail；单臂样本不足或缺环境身份报
not_verifiable；>90% 集中在一个环境给 warning。`audit_gates.audit_run_allocation`
（CLI `run_allocation`/`allocation`/`environment_balance`）读取
`run_order.json` + `summary.json` + `experiments.jsonl`，接入
`audit_gates.py report` 的 `run_allocation` 块。

**Boundary.** 平衡是设计而非事后修复：完全共线不能用加种子或重加权拆开。
槽位是计划建议，实际调度可以不同，审计读实际环境快照而不是相信计划。
环境身份不能证明数值路径完全相同（时钟/MIG/邻居负载不在快照内），只能
排除完全共线。缺环境记录不是通过；旧面板报 not_verifiable。计划时间比较
假设同一时钟口径，跨时区不一致时拒绝猜测。该审计不改变任何 PPL、种子、
终点、alpha、Holm 族、`confirmed` 或 `CODE_SEMANTICS`。

**Verification.** 新增 `test_review_part_15.py` 5 项：槽位位置轮换的
臂×槽位平衡（含「块号旋转抵消位置旋转」回归）、计划覆盖/槽位合法性/
偏斜 fail、平衡/完全共线/样本不足/缺环境/跨主机共线状态、计划时间
前瞻性各分支、审计入口与 report/CLI 接线，以及空目录必须
not_verifiable。现有 `run_order`/`evaluation_order` 测试保持通过；
`python audit_gates.py run_allocation` 在当前无结果目录的仓库报
`not_verifiable_locally` 而不是假绿；`python -m explib.confirmation plan`
的冻结计划/比较数不变。

## 2026-10-29 — 内容消融的匹配安慰剂：对齐 vs 存在（本轮消融方法硬化，no new runs）

零 GPU 的设计/代码硬化；不新增训练、不读取 test split、不修改冻结清单的
字节/种子/端点/alpha/Holm 规则，也不改 `CODE_SEMANTICS`。

**Problem.** 内容消融只有 `zero` 对照（`csa_fix_zerocont` /
`csa_dyn_zerocont`）。它同时改变了「有没有内容」「内容是否与本块对齐」
和「行范数」三件事，因此 zero 变差不能唯一归因于「块内容携带了本块
信息」：任何非零 K/V、尺度或与位置无关的偏置都可能产生同样的 PPL 差。
在审查语境里，这就是「没有匹配安慰剂」的机制过度解释；现有 2×2 交互只
覆盖 dynamic_blocking × 三个组件，不回答内容对齐。

**Fix.** 新增 `content_mode='shift'` 与 `csa_fix_shiftcont` /
`csa_dyn_shiftcont`：保留 pooled 内容的形状、参数、读取预算和归一化
范数，只把每个块的内容换成前一块的内容（block 0 置零）。eager 单行、
CUDA batched ref 与 kernel 路径都走同一语义；indexer 仍读真实内容，只
改变注意力读取的 K/V。该变换严格因果、无 RNG，前缀不变性保持；shift +
open-block prefix 直接拒绝而不是静默变成另一种操作。新增
`ABLATION_DESIGN['content_alignment']`、`ABLATION_INTERVENTION_KEYS`、
`HYPOTHESIS_DESIGN`、`ABLATION_SESOI` 与 `RUN_ABL` 排程，构成
aligned / misaligned / absent 三级对照；并复用这四个臂新增
`dynamic_blocking_x_content_alignment` 2×2（aligned vs shifted × fixed vs
dynamic），交互方向预注册为 +1、独立 SESOI 与 `INTERACTION_SCOPE` 条目，
回答「动态分块使块更同质后，对齐收益是否变小」。新增
`CONTENT_CONTROL_DESIGN` + `exp_lib.content_control_audit()`：用真实
`make_layer_cfgs` 检查每级都有已排程臂、三级不复用臂、非 aligned 级与
base 的 cfg diff 恰好是 `content_mode` 且值匹配、两个留一对照都在
`ABLATION_DESIGN` 中声明；`audit_gates.py single_factor` 新增
`content_controls` 块，并提供 CLI `content_controls` /
`control_ladder` / `content_alignment` 与 `report` 顶层键；
`factorial` / `interaction_sesoi` 会自动覆盖新的 2×2。

**Boundary.** shift 不是完美安慰剂：相邻块在自然文本中仍可能语义相关，
shift≈pooled 只能说明本设计分辨率下未检出对齐效应；跨文档打乱、随机
投影、参数匹配随机内容等更强对照未实现。首个块无前驱，按 zero 处理，
是显式约定。indexer 仍看真实内容，因此该对照不能被解释成「选择器也
不看内容」的联合消融。新增臂是下一轮运行的排程，不是已有证据；旧面板
没有这些臂时 `content_alignment` 会被如实标为不完整。该改动不改变任何
PPL、种子、终点、α/Holm、冻结决定或 `CODE_SEMANTICS`。

**Verification.** `python -m pytest -q` 通过 `608 passed, 3 skipped`（新增
`test_review_part_16.py` 8 项：shift 因果/范数/梯度、单行/批量/3-D 索引
对拍、参数量不变但前向改变、prefix 因果、未知 mode/open-block 组合
fail-closed、eager 与 kernel ref 逐值对拍、控制阶梯缺 level/复用臂/多旋钮/
未排程/坏模式 fail、`single_factor`/`report`/CLI 接线）。`python
audit_gates.py content_controls` 与
`python audit_gates.py single_factor` 在当前仓库均为 `ok`；
`python audit_gates.py report` 正常写出；`python -m explib.confirmation
plan` 的冻结计划/比较数不受影响。

## 2026-10-29 — 均值置信区间之外的「下一个种子」预测读数（本轮结果分析硬化，no new runs）

零 GPU 的结果分析硬化；不新增训练、不读取 test split、不修改冻结清单的
字节/种子/端点/alpha/Holm 规则，也不改 `CODE_SEMANTICS`。

**Problem.** 所有种子级对比只报告均值的 t 置信区间
`mean ± t * sd / sqrt(n)`。它回答「均值估得多准」，但读者和报告措辞经常
把它当成「再训一个种子会落在哪里」。n=3–6 时两者的宽度差
`sqrt(n+1)`（n=6 约 2.65 倍）：一个在 6 个种子上可分辨的均值，仍可能让
下一个种子有相当概率落到相反方向。现有 `seed_driven` / LOO 是删除已观察
种子的回顾性诊断，不能替代前瞻性预测。

**Fix.** 新增 `exp_lib.new_seed_replication(deltas, conf=0.95,
direction=None)`：在显式 iid 正态、共同方差、种子可交换模型下，用
Student-t 预测分布给出
`mean ± t(conf, n-1) * sd * sqrt(1 + 1/n)` 的单新种子预测区间、
`P(new > 0)` / `P(new < 0)`、与观察均值同向的 `prob_same_sign`，以及
调用方声明方向时的 `prob_improvement`。`paired_effect_stats` 现在同时
输出 `prediction_interval_95`、`prob_new_seed_same_sign` 和完整的
`new_seed_replication`；`reportable_claim` 原样回显这两个告示字段但**不**
把它们变成拒绝理由；methodology audit 在均值 CI 后打印预测区间并明确
标注 `not a CI for the mean and not a posterior`；`step_00` 的对比记录
携带同一组字段。零方差面板与 n<2 返回 `available=False`，不生成零宽
「确定性」区间；非有限种子照旧丢弃并记录 `n_input/n_dropped`。

**Boundary.** 这是模型预测，不是后验概率，也不是「真实效应为正的概率」；
它不修复选择偏差、winner's curse、跨规模/跨语料外推或测量漂移，不能替代
真正的新种子/新运行。它仍以种子为分析单位，不把 token 当独立重复；共享
初始化之外的相关 RNG/优化状态会使 iid 预测偏乐观。该读数不重算 p、不重排
Holm 族、不改变任何 `confirmed` 判定或冻结决定，也不改变 `CODE_SEMANTICS`。

**Verification.** `python -m pytest -q` 通过 `621 passed, 3 skipped`（新增
`test_review_part_18.py` 7 项：预测区间比均值 CI 宽且逐值匹配 t 公式；
独立 `numpy.random.standard_t` 采样核对符号概率；1e-200–1e150 尺度不变性
与方向反转对称；`direction=-1/+1` 不颠倒；空/单种子/常数面板 fail-closed
且零均值时同向概率为 `None`；非法 `conf`/`direction`/字符串载荷拒绝；
`paired_effect_stats`/`reportable_claim` 回显与坏字段不被升级成门）。
`python -m explib.confirmation plan` 的冻结计划/比较数不受影响。

## 2026-10-29 — 总不确定性不是「较窄的那个区间」：配对 Δ 的 seed × evaluation-unit 两层 bootstrap（本轮结果分析硬化，no new runs）

零 GPU 的结果分析硬化；不新增训练、不读取 test split、不修改冻结清单的
字节/种子/端点/alpha/Holm 规则，也不改 `CODE_SEMANTICS`。

**Problem.** v12 的主比较同时报逐种子符号翻转 p（种子抽样）与
`doc_sampling.pooled_delta_ppl_ci95`（条件在已训练种子上、按评测文章/文档
聚类的 token 池化 Δ 区间）。两者各自只覆盖一个不确定性来源：cluster 区间
在 n=12 时可能很窄，逐种子区间又不含评测切片抽样。读者若引用较窄的那个，
就会把「条件在 12 个种子、48 个评测单位上」误读成「换一批种子和文档也这么
准」。现有的 `_combined_seed_doc_ci` 只处理**绝对 PPL**，
`paired_row_bootstrap_delta` 只重采样行且明确不含种子；配对 Δ 的两层联合
区间仍是缺口（review-4 第 14 条「seed-level 与 doc-level 仍是两层各自报告」
的直接残留）。

**Fix.** 新增 `exp_lib.paired_nll_two_way_bootstrap(nll_a, nll_b,
n_tokens, *, n_boot=2000, seed=20261018, conf=0.95)`：在 `[n_seeds,
n_units]` 的逐单位 NLL 上，每个 draw 独立重采样种子与评测单位，按抽中单位
重算每个抽中种子的 token 池化 PPL 差，再按种子权重平均，取 percentile
区间（同时给 log-ratio 与 ratio CI）。点估计仍为全 panel 的
`mean_s(PPL_a − PPL_b)`，与 `_paired_nll_doc_ci` 逐值一致；不拟合
hierarchical 模型，不假设两个方差分量可加。`_doc_level_analysis` 只在
cluster 区间已与记录主终点对齐后计算该块，点估计不能复现 pooled Δ 时显式
关闭并记录原因；`doc_sampling` 新增 `pooled_two_level_*` 字段、n_seeds/
n_units/n_boot 与「百分位区间在少 cluster 时可能偏窄」的 warning。
`_doc_ci_markdown_cell` 在同一格同时渲染 cluster 与两层区间；表下说明前者
条件在种子上、后者是总不确定性的描述性读数。新增
`audit_gates.audit_two_level_ci`（CLI `two_level_ci` / `two_level` /
`total_uncertainty` / `joint_ci`）：检查点估计复现 pooled Δ、点落在自己的
区间内、计数合法、方法名含 `two-way`；联合区间窄于条件 cluster 区间时给
warning（种子分量可能丢失）。缺块对旧产物是 warning，块自相矛盾才是 fail；
接入 `audit_gates.py report` 的 `two_level_ci` 顶层键。

**Boundary.** 这是 **superpopulation over seeds and units** 的描述性读数，
不是新的检验、不产生 p、不进入 Holm 族、不改变任何 SESOI 决策或
`confirmed`。它不替代真正的新种子/新评测切片，也不把 token 当独立重复；
单位内部相关只通过 cluster 保留。S=12 个种子 cluster 时 percentile 覆盖率
可能低于名义 95%，必须同时报告 S/D/n_boot 与 warning；要提高精度应增加独立
种子或文章，而不是缩窄区间。行级单元没有 token 池化 NLL 矩阵，不生成该块；
历史 v6–v11 产物不重跑、不回填，缺块在审计中如实报 warning/
not_verifiable，不把缺失当零方差。population-claim 的 64 单元门槛仍由
`unit_precision_design` 与报告总体声明门控制。

**Verification.** `python -m pytest -q` 通过 `631 passed, 3 skipped`（新增
`test_review_part_20.py` 5 项：点估计与 `_paired_nll_doc_ci` 逐值一致；
只有种子变化、文档内无变化时 cluster 区间退化为零而两层区间仍为正；
同 seed 可复现、conf 变宽、点估计不随重采样改变；形状/布尔/非有限/
分数 token/非法 conf/n_boot/seed fail-closed；单种子/单单位拒绝；v12 分析
接入且报告格同时含 cluster 与两层区间；审计的 ok/fail/warning/legacy/
not-verifiable 分支）。`python audit_gates.py two_level_ci` 在当前无
`analysis_v12/test_stats.json` 的仓库报 `not_verifiable_locally`，不假绿。
`python -m explib.confirmation plan` 的冻结计划/比较数不受影响。

## 2026-10-29 — 数据处理身份：近重复清理几何与 source→served 台账（本轮数据方法硬化，no new runs）

零 GPU 的数据处理/方法学硬化；不新增训练、不读取 test raw、不修改冻结
清单的字节/种子/端点/alpha/Holm 规则，也不改 `CODE_SEMANTICS`。

**Problem.** `dedup` 元数据此前只记录 `near_hamming_threshold` 和 policy
字符串，但清理集合实际由 exact hash、SimHash shingle 宽度、shingle 采样
上限、LSH band 数与 Hamming 阈值共同决定。任一参数不同，两个 cache 即使
token 指纹、`fit_fp` 与 policy 完全相同，也可能保留不同的评测文档；test
cache 仍会声称与 train 使用同一清理。源行数、删除数、服务行/文档/token
数又分散在 `sampling`、`dedup.splits`、`doc_ids`，没有可机器核对的闭合
台账。

**Fix.** 新增 `DEDUP_PROTOCOL_DEFAULTS` / `dedup_protocol_fp` /
`dedup_protocol_problems` / `dedup_protocol_from_meta`；`_DocDeduplicator`
接受完整 shingle/band 几何并在 `stats()` 中携带协议与指纹。新增两条
fail-closed 几何约束：`near_hamming_threshold < bands`（鸽巢候选完备性）
与 `bands` 整除 64（不留未索引高位）。train/validation meta 的 `dedup` 与
test meta 的 `near_duplicate_cleanup` 与 FineWeb 的 `dedup` 写
`protocol`/`protocol_fp`；test 冷构建从配对 train cache 读取协议重建
validation/test 索引，cache-hit 与冷构建都拒绝协议字符串之外的几何不一致。新增
`audit_gates.audit_data_processing`（`data_processing` /
`dedup_protocol` / `dedup` / `cleaning`，并进入 `report`）：重算指纹、
核对冗余字段、按 `(vocab, fit_fp)` 要求 train/val/test 协议一致，并检查
`nonempty_rows_scanned - rows_served == exact+near`、sampling 与 doc-id
计数、test 非空源文档台账、以及存在的 `.npy` 侧边长度。test meta 新增
`source_documents_nonempty` 使 held-out 台账精确闭合。

**Boundary.** 不重采样、不重新清理、不重跑，不改变任何 token、split
成员、清理决策或 PPL；旧 cache 不回填，legacy 缺完整协议只报
`not_verifiable_locally`/warning，记录非默认 threshold 的 legacy cache
无法知道 shingle/band 几何时拒绝读取。SimHash 仍是近似近重复；本改动
保证的是同一清理几何与 LSH 候选完备性，不是语义近重复的完备检测。台账
闭合不证明上游 raw dump 内容正确。

**Verification.** `python -m pytest -q` 通过 `640 passed, 3 skipped`（新增
`test_review_part_22.py` 7 项：默认协议/指纹、非默认几何、threshold≥bands
与 band 不整除 64 拒绝、协议校验器与 metadata 读取、干净台账 ok、
`protocol_fp` 篡改/跨 split 几何不一致/台账不闭合 fail、legacy 与缺缓存
`not_verifiable_locally`、`_filter_test_documents(protocol=...)` 用配对协议
重建索引）。`python audit_gates.py data_processing` 在当前无 `wt103_cache/`
的仓库报 `not_verifiable_locally`，不假绿。`python -m
explib.confirmation plan` 的冻结计划/比较数不受影响。

## 2026-10-29 — 复现是两次运行之间的比较：逐维身份诊断与可执行复现门（本轮可复现性/工程规范硬化，no new runs）

零 GPU 的可复现性/工程规范硬化；不新增训练、不读取 test split、不修改冻结
清单的字节/种子/端点/alpha/Holm 规则，也不改 `CODE_SEMANTICS`。

**Problem.** 仓库已有 `reproducibility_check`（同进程重复一小段训练流）、
`seed_contract_check`（RNG 来源）、`exec_fp` / `source_tree` /
`data_version` / `run_cfg` / `init_core_fp` 等逐记录身份字段，但**两份独立
产物之间**没有可执行的比较。README 中并行 vs 串行「identical numbers」只是
本地脚本侧检查，无法作为命令重放；`pair_reason` 是臂间配对门，不是同格重跑
门；直接比较 PPL 会把「环境不同」「数据切片不同」「代码字节不同」都压成
「数值漂移」，也会把「PPL 接近」误读成复现。缺指纹时没有任何机制阻止读者
把「没记录」当成「相等」。

**Fix.** 新增 `exp_lib.reproduction_diagnosis(original, rerun, *, rtol=1e-6,
atol=0.0)`：按 `code`（`source_tree.sha256` + `code_semantics` /
`code_stamp`）、`data`（两侧都存在的 `data_fp`/`val_fp`/`eval_fp`/
`data_version`/`corpus_fp`）、`config`（`run_cfg`/`protocol`/`warm_steps`/
`amp_dtype`/`eager_batch`，并复用 `training_knobs_reason` 与
`objective_reason`）、`environment`（`exec_fp`）、`initialization`
（`init_core_fp`）逐维比较；`rng_fp` 只作诊断（与 `pair_reason` 的既有约定
一致，架构差异本就消耗不同 RNG）。每维只有 `match`/`mismatch`/`unverified`
三种状态，缺失证据不是相等。`metric` 单独比较 `ppl` 与 `n_tokens`：token
数不同判 `metric_mismatch`；PPL 在 `atol + rtol*|a|` 内为
`within_tolerance`，严格相等才是 `identical`。判定优先级为
`*_changed` > `provenance_incomplete` > `metric_missing`/
`metric_mismatch` > `identical`/`within_tolerance` > `numerical_drift`，
`reproduced` 只在最后两档成立，`bit_identical` 额外要求 PPL 严格相等。
`synthesized` 记录（由旧 aggregate 重建、无权重/运行）直接 `not_comparable`。

新增 `exp_lib.compare_reproductions(original, rerun, ...)`：接受 summary
mapping、结果目录或 `summary.json`，按 `variant::seedN` 逐格匹配；只在一侧
出现的格子进入 `only_original`/`only_rerun` 并把汇总判为 `incomplete`，
没有共同格子判 `not_comparable`，任一格身份 mismatch 判 `not_reproduced`，
只有全部身份匹配而 PPL 漂移才判 `drift`。新增 CLI
`python -m explib.reproduce compare A B [--rtol/--atol/--json/--require
exact]`（0 复现 / 1 未复现 / 2 用法或输入错误）、`make reproduce A=... B=...`
与 `audit_gates.audit_reproduction`（CLI `reproduction`/`reproduce`/
`replication`/`rerun_check`，接受两个显式路径；并进入 `report` 的
`reproduction` 块）。没有显式两份目录时审计返回 `not_verifiable_locally`，
不猜哪两个目录是一对重跑。另在 `dataloader_contract_check` 构造 worker
iterator 前先探测 POSIX semaphore：受限沙箱缺信号量时返回干净的
`passed=False` 报告，避免 torch 部分初始化的
`_MultiProcessingDataLoaderIter.__del__` 在 GC 时抛 unraisable
`AttributeError`，把零 GPU 测试门随机变红。该改动只作用于 probe 的失败
路径，不改变任何训练/评测数值。

**Boundary.** 这是描述性身份比较，不是统计检验：没有 p 值/置信区间，不修复
选择偏差、测量误差或跨硬件数值差异；`rtol`/`atol` 必须由调用方声明并原样
回显。`exec_fp` 相同不等于逐位可复现，不同也不自动说明代码有错；工具只把
区别显式化，不替读者裁决。`rng_fp` 只诊断；`source_tree`、`exec_fp`、
`init_core_fp`、数据指纹缺失时结果是 `unverified`/
`provenance_incomplete`，不会降级成 `match`。该机制不重跑、不重算 PPL、不
读取 test split，也不改变任何冻结决定；它不能替代真正的新种子/新运行，更不
能把两份相同的旧产物变成独立证据。

**Verification.** `python -m pytest -q` 通过 `650 passed, 3 skipped`（新增
`test_review_part_23.py` 7 项：完全相同/容差内/漂移；源码/数据/配置/环境/
初始化各自的 `*_changed`；同种子不同 `rng_fp` 只诊断；缺源码摘要/`exec_fp`/
`init_core_fp`/数据身份 -> `provenance_incomplete`；token 数不同 ->
`metric_mismatch`；坏 PPL -> `metric_missing`；synthesized ->
`not_comparable`；非法容差/非 mapping fail-closed；批量报告的 `only_*`/
`counts`/`not_comparable`；CLI 0/1/2、`--require exact`、`--json` 落盘；
audit 的 ok/fail/not_verifiable 与 `_run_cli` 位置参数接线）。`python -m
explib.reproduce --help` 与 `python audit_gates.py reproduction` 在当前无
两份结果的仓库分别返回用法/`not_verifiable_locally`，不假绿。`python -m
explib.confirmation plan` 的冻结计划/比较数不受影响。

## 2026-10-29 — 消融的「模块有用」与「学习有用」：同容量随机 sink 安慰剂（本轮对比/消融方法硬化，no new runs）

零 GPU 的对比/消融方法学硬化；不新增训练、不读取 test split、不修改冻结
清单的字节/种子/端点/alpha/Holm 规则，也不改 `CODE_SEMANTICS`。

**Problem.** `ABLATION_DESIGN['attention_sink']` 的 on/off 对照同时改变了
三件事：是否有可训练 sink、是否有这份额外参数/容量、是否把额外注意力出口
交给优化器。因此 on/off 显著只能支持「sink 容量/存在性有用」，不能唯一归因
到「学到的 sink 吸收了 off-context 质量」；§79 已为内容轴加了 shift
安慰剂，§72 已为参数匹配基线加 SESOI，但参数化消融臂自身仍只有 on/off。

**Fix.** 新增 `AttnCfg.sink_mode ∈ {'learned','random'}`；`random` 保持与
learned 相同的 sink 张量形状/数量/总参数量/读取预算，但值以该层已初始化的
`W_q` 的 SHA-256 摘要作局部 RNG 种子抽取（归一化到
`||v||=sqrt(n_heads)`）并 `requires_grad=False`，且**不消耗全局 RNG**，
所以 treatment/placebo 的 trunk、W_q 与 `init_core_fp` 仍逐值一致。新增臂 `csa_fix_randomsink` /
`csa_dyn_randomsink` / `full_randomsink`，与 `csa_fixed` / `csa_dynamic` /
`full_sink` 构成 learned / frozen-random / absent 三级阶梯：
`learned−random` 是容量匹配的学习成分，`learned−absent` 是总效应，
`random−absent` 是容量/正则化成分。新增声明组件
`attention_sink_learning`、独立 `ABLATION_SESOI`、`HYPOTHESIS_DESIGN`
（`requires_param_match=True`）、`ABLATION_INTERVENTION_KEYS`
（只允许 `sink_mode`），以及容量匹配的
`dynamic_blocking_x_attention_sink_learning` 2×2、独立
`INTERACTION_SESOI` 与 `INTERACTION_HYPOTHESIS_DESIGN`，并加入
`INTERACTION_SCOPE` 的 sink-learning 对照范围。新增
`SINK_CONTROL_DESIGN` / `sink_control_audit()`：用真实 `make_layer_cfgs`
与 tiny `SmallGPT` 核验三级 role 齐全、臂已排程、treatment 就是 base、
placebo 与 base 的 cfg diff 恰好是 `sink_mode`、总/sink 参数与 sink 张量数
完全相等、placebo sink 冻结、absent diff 恰好是声明的缺席键且参数减少量
恰好等于 sink 参数量、两条留一对照与键集都已声明；缺 level/重复 role/
未排程/复用臂/多旋钮/未冻结/参数不匹配/未声明族全部 fail-closed。接入
`audit_gates.py sink_controls`（别名 `sink_placebo`）、`single_factor` 的
`sink_controls` 块与 `report` 顶层键；`SINGLE_FACTOR_CFG_FAMILIES` 增加
`{'sink_mode'}`，使 learned-vs-random 通过单因素契约。

**Boundary.** frozen-random 只破坏「学习」，不能排除「任何固定非零 sink 都
比零好」；后者是 `random−absent` 的 estimand。随机向量以 W_q 的哈希作局部
RNG 种子抽取，不是从任务分布独立抽签，换个初始化分布会换一个 placebo；
抽出值非有限/范数为零时显式回退到固定 `linspace`。on/off 仍移除参数，故 `attention_sink` 的
`requires_param_match` 仍是 False，on/off 结果不能单独写成学习机制证据。
fusion / boundary gate / dynamic gate 仍只有 on/off 或架构级读数，本阶梯
不是「所有模块都已有匹配安慰剂」的全局声明；`interaction_coverage` 继续把
未覆盖交互报 missing。新增臂是下一轮运行的排程，不是已有证据；旧面板缺臂
时 `ablation_completeness` 会如实报不完整。不改变任何冻结决定、种子、α、
终点或 `CODE_SEMANTICS`。

**Verification.** `python -m pytest -q` 通过 `661 passed, 3 skipped`（新增
`test_review_part_25.py` 5 项：random sink 同参数量/同 sink 张量数/冻结/
全局 RNG 中性/确定性/输出不等于 learned 或 off；动态臂前缀因果性；阶梯
设计门通过（含参数差恰好等于 sink 参数量、factorial、SESOI、hypothesis
覆盖）；缺 level/重复 role/未排程/复用臂/多旋钮/缺席臂多改容量/未声明族
fail-closed；audit 门与 `report` 接线）。`python audit_gates.py
sink_controls` 与 `python audit_gates.py single_factor` 在当前仓库均为
`ok`；`python -m explib.confirmation plan` 的冻结计划/比较数不受影响。

## 2026-10-29 — 冻结结论的 p 值必须可重算：从逐种子配对差重放 exact sign-flip 与 Holm（本轮可复现性/统计完整性硬化，no new runs）

零 GPU 的统计产物完整性硬化；不新增训练、不读取 test split、不修改冻结
清单的字节/种子/端点/alpha/Holm 规则，也不改 `CODE_SEMANTICS`。

**Problem.** `audit_gates.verdicts` 只从 `analysis_v12/test_stats.json`
已存的 `p_holm`、方向与有效性门重放三向判定，并明确声明不重算 p。于是
headline 统计量仍是被信任的输入：一个被编辑或在聚合中写错的 p，只要上层
判定按同一组数重放就无法发现。逐种子配对差虽然已由
`exact_sign_permutation` 写进 `comparisons[*].per_seed`，但没有门把它与
headline p 对账；`by_length_curve` 甚至没有保存产生 p 的向量；产物也没有
记录冻结族的比较名称，删除一个比较及其 Holm 槽位后，剩余内容仍可自洽，
而族大小从 m 缩到 m-1。

**Fix.** 新增 `audit_gates.audit_confirmation_pvalues(path=None)`（CLI
`pvalues` / `pvalue_replay` / `recompute_p`，可传 artifact 路径）：
逐比较从 `per_seed` 重算 exact two-sided sign-flip p（种子等长/唯一/整数，
delta 有限、拒绝布尔），对账 `p_exact_signflip`、`n`、`n_nonzero`、
`p_floor`、`mean`；逐长度重放 `by_length_curve` 的 p，并要求主端点曲线的
向量与 primary comparison 逐值一致；用 `manifest.primary_comparison_names`
核对预注册族集合（省略项保留 p=1 槽位），用 `exp_lib.holm_adjust` 重算整族
`p_holm`/`m_effective`，并用 `manifest.seeds` + 族大小重算
`min_attainable_p_holm`。任一不一致判 `fail`；没有 `per_seed` 的 legacy
产物是 `not_verifiable_locally`/`warnings`，缺证据不是通过。生产端只做
增量：`finaltest_stats` 在 manifest 下写 `primary_comparison_names` /
`family_size`，并在每个 by-length cell 写 `per_seed` / `n_nonzero` /
`p_floor` / `method`；原 p、端点、alpha、Holm 计算不变。新门接入
`audit_gates.py report` 的 `pvalue_replay` 顶层块。

**Boundary.** 重放的是「p 由这组 deltas 算出」的算术一致性，不是
deltas 的测量来源；后者仍由 P5T 的 admission / freeze lock / `data_fp` /
`ckpt_sha256` / `train_knobs` 门约束，两层不能互相替代。它不重算 PPL，
不重训，不读 test，不改变任何冻结 p/Holm/判定；若 `per_seed` 向量本身被
替换成另一组同样产生该 p 的数，本门不能发现，这是明确的剩余缺口（要闭合
需从 `bylen.json` 的逐 cell PPL 重算 deltas，并把 admission 规则一起纳入）。
它不是新检验，也不替代真正的复制运行。`verdicts` 的三向判定重放保持不变。

**Verification.** 新增 `test_review_part_26.py` 6 项：一致产物 ok；篡改
p、Holm、n 分别 fail；删除/新增族比较 fail、缺族名只 warning；缺
`per_seed` 为 not_verifiable、部分缺失为 warnings；篡改 by-length p 或
primary 向量 fail；重复种子/长度不符/布尔/NaN/p_floor/mean/过短向量全部
fail；CLI `pvalues PATH` 与 `report` 接线。完整 CPU 套件由
`665 passed, 3 skipped` 增至 `673 passed, 3 skipped`。`python
audit_gates.py pvalues` 在当前无 `analysis_v12/test_stats.json` 的仓库返回
`not_verifiable_locally`，不假绿。

## 2026-10-29 — 生成目录、输入绑定与完整案例：三处可复现性/推断缺口（本轮硬化，no new runs）

零 GPU；不新增训练、不读取 test split、不修改冻结清单字节/种子/端点/
alpha/Holm 规则，也不改 `CODE_SEMANTICS`。

**Problem 1（源码身份被构建目录污染）.** `source_tree_manifest()` 跳过
`build`/`dist` 却没有跳过 `.build`；而 `make install-check` 把整份源码复制到
`.build/install-check/`。于是同一份源码跑过打包检查后 `source_tree.sha256`
改变：冻结锁会误判「源码变了」，`explib.reproduce` 报 `code_changed`，
`archive_release.py` 还会打包重复副本。**Fix.** 跳过集加入
`.build`/`.tox`/`.nox`/`.eggs`/`.hypothesis`，并按 `*.egg-info` 后缀跳过
构建元数据；新增源码树回归测试（`.build/install-check`、`build/lib`、
`.tox/env`、`demo.egg-info`、`.eggs/pkg` 不改变摘要）。

**Problem 2（输入绑定可被整体删除）.** `stamped_analysis` 把 `inputs`
排除在 `content_sha256` 外，验证器只在 `inputs` 存在时检查。删掉整个
`inputs` 块后内容摘要不变，`audit_analysis_integrity` 仍报 `ok`。
**Fix.** 新增 `analysis_inputs_sha256()`，对输入清单的稳定投影
（label/path/exists/sha256/bytes；不含 abs_path/error）做 canonical 摘要；
`stamped_analysis` 总是写 `inputs`（无输入为 `{}`）与 `inputs_sha256`，
且 `inputs_sha256` 在 `content_sha256` 的哈希体内。验证器在
`inputs_sha256` 存在但 `inputs` 缺失/被清空/被改写时 fail；旧产物没有该
摘要时只给 warning，不假绿。

**Problem 3（完整案例子集被提升为显著/等价）.** `paired_effect_stats`
计算 `complete_pairs/n_dropped` 却未把完整配对作为 `significant` 的前置
条件；`ablation_practical_decision` 与 `interaction_practical_decision`
只在 `n<min_n` 时拒绝，TOST 可在 3/4 或 6/7 个幸存对上判
`equivalent`/`additive`。**Fix.** `significant` 增加 `complete_pairs`
前置；`effect_vs_noise` 优先标 `incomplete paired panel`；两个 practical
decision 在 superiority/TOST 之前返回 `incomplete_pairs`/
`inconclusive_incomplete_panel`，并回显 `complete_pairs`/`n_dropped`。
完整面板路径不变。

**Boundary.** 源码摘要跳过的是目录名（与既有 `build`/`dist` 约定一致），
若真实测量源码被放进同名目录会被忽略；`inputs_sha256` 证明记录清单未被
改，不证明输入文件内容（输入不在本机仍是 warning）；完整配对只排除「丢
种子后仍宣称确认」，不把种子变成随机总体抽样，也不替代新运行。三处改动
均不重算 PPL/p/Holm、不改任何冻结判定或 `CODE_SEMANTICS`。

**Problem 4（选择机制门未做多重校正）.** `audit_selection` 对多个
arm/length cell 各自用 `p<0.05` 置 `claim_allowed`；3 个 cell 各自
p=0.03125 时族级假阳性约 9.1%。**Fix.** 对显示出来的 cell 用
`holm_adjust` 调整（无法检验的 cell 保留 p=1 槽位），新增 `p_holm` 与
`claim_family`，`claim_allowed` 要求 `p_holm<0.05` 且 n>=6；单 cell 的
m=1 行为不变。

**Verification.** `test_review_part_10.py` 扩展源码树摘要测试并新增
`test_complete_case_survivors_are_not_promoted_to_a_claim`（6/7 幸存对
p<0.05 但 `significant=False`；ablation/interaction TOST 均为
`incomplete_pairs`；完整面板仍 `equivalent`）。`test_review_part_13.py`
新增 `test_analysis_input_binding_cannot_be_stripped`（删除/清空/改写
inputs 均 fail；移动 checkout 的 abs_path/error 不改变摘要；无输入产物仍
可验证）。`test_review_part_05.py` 的 selection 合同测试新增 3-cell 分支
（raw p=0.03125，Holm/3 后 `p_holm≈0.09375`，三个 `claim_allowed` 全
false，`claim_family.m==3`；单 cell 仍 true）。`requirements.lock` 补上
pytest 的 `iniconfig==2.3.1`/`pluggy==1.7.0`/`Pygments==2.21.0`，并把
本地项目自身的发行版从「未 pin 运行时依赖」中排除，使
`dependency_contract()` 在装过本项目后仍能为 `ok`。`python
audit_gates.py analysis_integrity`、`python audit_gates.py dependencies`
与全量 CPU 套件在本轮最终树通过。

## 2026-10-29 — 结果分析补上「方差从哪来」：seed × 评测单元矩估计分解（本轮结果分析硬化，no new runs）

零 GPU；不新增训练、不读取 test split、不修改冻结清单字节/种子/端点/
alpha/Holm 规则，也不改 `CODE_SEMANTICS`。

**Problem.** 结果表同时有 seed 级精确 sign-flip、document cluster CI 和
seed×unit 两层 bootstrap，但没有任何读数回答「面板均值的方差主要来自
种子、评测单元还是 seed×unit 交互」。方差来源不同，下一步完全不同：
seed 占比高时加评测文档只是把同一训练噪声估得更细；unit 占比高时加种子
修不了「结论依赖哪几个文档」；残差占比高时单独加任何一轴都不够。把两层
bootstrap 的区间宽度当作「再加种子会变窄」的依据，等于把未知的方差结构
当成已知。

**Fix.** 新增 `exp_lib.crossed_variance_components`：对完整平衡的
`[n_seeds, n_units]` 面板做 two-way ANOVA 矩估计
`σ²_seed=(MS_seed−MS_e)/D`、`σ²_unit=(MS_unit−MS_e)/S`、`σ²_e=MS_e`；
负的 raw 估计保留、estimate 截断到 0；报告三个 share/ICC、超总体与
固定单元两种均值方差、Satterthwaite 均值 CI、等效独立单元/设计效应、
三类预测区间，以及「下一格加种子还是加单元」的有限差建议。配对端点由
`paired_delta_variance_components` 在 `(nll_a−nll_b)/n_tokens` 上调用，
同时报告等权单元均值与 token 池化均值，避免把两个估计量混成一个。

**Wiring.** v12 的 `_doc_level_analysis` 把分解写进
`doc_sampling.variance_components`；v12 报告新增一节表格（缺失时写
「未验证」）；`audit_gates.py variance_components` 重放分量和、share、
`mean_variance`、`effective_n`、区间与 allocation 比例，篡改 fail，旧
artifact 只 warning。所有既有 p/Holm/`confirmed`/冻结族不变。

**Boundary.** 这是描述性矩估计，不是新的显著性检验；小面板上分量本身
不精确，没有 hierarchical/Bayes 后验；只接受完整平衡面板；`unit` 仍是
文章/行级文档/行，行级 readout 无单元 NLL 因而明确不产生该块；residual
混合了 seed×unit 交互与 token 抽样，不能全解释成测量误差；n_units<64 的
总体主张门不变。它不替代新运行，也不修复选择偏差或外推问题。

**Verification.** 新增 `test_review_part_27.py` 9 项：精确可加面板恢复
行/列样本方差；checkerboard 负矩估计 raw/truncation/share；常数面板
`not_estimable`；1e6 缩放不变性；非法输入与 conf 拒绝；配对 wrapper 与
手工 per-token 矩阵一致；报告渲染与缺块分支；审计门一致 artifact ok、
篡改 total/mean variance fail、无块 `not_verifiable_locally`。完整 CPU
套件与 `python audit_gates.py variance_components` 在本轮最终树通过。

## 2026-10-29 — 重放门只接受真实数字：p 值/种子/计数的类型防线（本轮可复现性硬化，no new runs）

零 GPU；不新增训练、不读取 test split、不修改冻结清单字节/种子/端点/alpha/Holm 规则，也不改 `CODE_SEMANTICS`。

**Problem 1（p 值重放接受被引号包住的数字）.** `audit_confirmation_pvalues` 的 `_pvr_finite` 用 `float(value)`，于是 `"0.05"` 这类 JSON 字符串会被当成测量值；`_pvr_replay_one` 对 `per_seed` 也直接 `float(x)`，并把 `seeds`、`n_effective` 直接 `int()`。结果：`"0.3"` 的 delta、`1.5` 的 seed（被截成 1）、`12.9` 的 n（被截成 12）都可能通过重放；`manifest.seeds` 只检查长度/唯一性，从不核对比较记录里的 seed 是否属于冻结白名单。

**Fix.** `_pvr_finite` 拒绝 bool/str/bytes/bytearray；新增 `_pvr_int` 只接受精确整数（整值浮点可以，`1.5`/`"12"` 不行）；delta 拒绝字符串；seed 校验整数、`[0, MAX_SEED]` 与白名单子集；`n_effective`/`n_nonzero` 用精确整数比较；`manifest.seeds` 先按同一规则校验并去重。比较与 `by_length_curve` 的 seed 都必须落在白名单内。

**Problem 2（方差分解审计只看 next_cell 的取值、不看推导）.** `audit_variance_components` 只验证 `next_cell ∈ {'seed','unit'}` 与非空 ratio，不重算有限差 `s/(S(S+1))` vs `u/(D(D+1))`，也允许整个 `allocation` 块缺席；`unit` 分量为 0 时 `recommended_seed_to_unit_ratio` 应为 null 的约束也没有检查。

**Fix.** `allocation` 必须存在且 status 只能为 `ok`/`not_decidable`；`ok` 时重算 current counts、`S/D`、next_cell 的有限差比较、`sqrt(seed/unit)` 以及 unit=0 时的 null；`not_decidable` 时要求 seed/unit 估计都为 0。

**Boundary.** 这是对“产物自己能否重放”的算术/类型一致性检查，不是新的显著性检验；它不证明 per_seed 向量来自真实测量（后者仍由 P5T admission/freeze lock 约束），也不改变任何冻结 p/Holm/判定。

**Verification.** 新增 `test_review_part_26.py` 2 项（引号数字、分数 seed、超白名单 seed、分数 n 全部 fail；manifest seed 类型/重复 fail）与 `test_review_part_27.py` 1 项（篡改 next_cell/ratio/current 或删除 allocation 均 fail）。完整 CPU 套件由 `686 passed, 3 skipped` 增至 `689 passed, 3 skipped`。

## 2026-10-29 — 证据归档的字节可复现性：`SOURCE_DATE_EPOCH`、内容摘要与元数据重放（本轮可复现性/工程规范硬化，no new runs）

零 GPU；不新增训练、不读取 test split、不修改冻结清单字节/种子/端点/
alpha/Holm 规则，也不改 `CODE_SEMANTICS`。

**Problem 1（归档自带墙钟，同一证据重建不是同一 tar）.** `archive_release.py`
默认 `tarfile.open(..., 'w:gz')` 会把构建时刻写进每个 member 的 mtime、内部
MANIFEST member 的 mtime 和 gzip header 的 MTIME；`created_at` 与
`environment.captured_at` 也是墙钟；`_collect` 按 `os.walk` 的返回顺序排列
文件，`checkpoint_hashes` 同样未排序。于是同一份 results/analysis/source
证据两次构建得到不同的 tar.gz 字节和不同的外部 `.sha256`。`verify` 只能
证明「这个 tar 的内部清单与内容一致」，不能证明「这个 tar 就是发布的
tar」；发布物无法被第三方重建后对哈希。另一个自指问题：第一次构建写出的
tar.gz 是未跟踪文件，第二次构建的 `git status` dirty 列表会因此变化，归档
自己使自己不可重建。

**Problem 2（内容身份被时间戳污染）.** 只有整个 tar 的哈希，没有与打包
元数据无关的内容摘要。未设 epoch 或换一个输出目录时，即使证据字节完全
相同也无法给出一个可比较的 identity；`verify` 也没有可重放的打包元数据
契约。

**Fix.** 引入 reproducible-builds 的 `SOURCE_DATE_EPOCH` 约定：`make` 新增
`--source-date-epoch N` 与 `--reproducible`；epoch 存在时（环境变量或参数），
所有 member 的 `mtime`、`uid/gid/uname/gname`、PAX atime/ctime 与 gzip
header MTIME 固定到该值，`created_at` 用该 epoch 的 UTC 表示；
`--reproducible` 缺 epoch 时 refuse，避免「以为可重建、实际仍是墙钟」。
`_collect` 和 `_ckpt_hash_table` 按仓库相对路径排序，MANIFEST 的 `files`
键顺序稳定；`_git_state` 用 pathspec 把归档输出目录从 dirty 列表排除，
使同一输出目录的二次构建不会被自己写出的产物改变。新增 metadata-free 的
`payload_sha256`（`sha256(sorted relpath NUL sha256 NUL bytes LF)`），只
覆盖入包文件的摘要/字节数，不覆盖时间戳与打包元数据；`verify` 重算它，
并重放 deterministic 块的 epoch/created_at/member mtime/uid/gid/uname/
gname/gzip MTIME。旧归档没有这些块时只提示，不假绿。

**Boundary.** 这是打包字节与元数据一致性的改进，不是新的测量、不是对
PPL/p/Holm 的重算，也不证明归档内容是真实运行产物（后者仍由 P5T 的
admission、freeze lock、`data_fp`/`ckpt_sha256`/`train_knobs` 门约束）。
`payload_sha256` 是内容身份，不是签名或时间戳证明；能重算它的人也能改它。
字节级重建假设同一份源码树、同一环境快照与同一输出目录：源码树中的未跟踪
文件或 dirty 列表变化仍会改变 MANIFEST；跨机器/跨 checkout 的路径与依赖
环境不同时，应比较 `payload_sha256`，而不是要求 tar 哈希相同。未设
`SOURCE_DATE_EPOCH` 时保持原有墙钟行为，归档仍可 verify，但不承诺 tar
字节可重建。它不替代真正的复制运行，也不改变任何冻结决定。

**Verification.** 新增 `test_review_part_28.py` 8 项：严格解析 epoch（负数/
小数/科学计数/日期/布尔/超范围全部拒绝）；同一 epoch 两次构建 tar.gz
字节完全相同、所有 member mtime/owner 被固定、gzip header 与 manifest
一致、`verify` 通过；不同 epoch 的 tar 字节不同但 `payload_sha256` 相同；
篡改文件摘要/算法/`payload_sha256` 被重放门拒绝；篡改 `created_at`/
`gzip_mtime`/member mtime/布尔类型/非确定性块借用 epoch 全部 fail；
`--reproducible` 缺 epoch 时 refuse 且不落盘；墙钟归档仍能 verify。完整
CPU 套件由 `689 passed, 3 skipped` 增至 `697 passed, 3 skipped`。

## 2026-10-29 — 2×2 交互的设计期功效与分辨率下限（本轮实验设计与准备硬化，no new runs）

零 GPU；不新增训练、不读取 test split、不修改冻结清单字节/种子/端点/
alpha/Holm 规则，也不改 `CODE_SEMANTICS`。

**Problem.** 交互已经有预注册 SESOI 与 TOST（§67/§84），但没有任何
**跑之前**的交互功效读数。主效应功效不能替代交互功效：交互是逐种子的
log ratio-of-ratios 差中之差，SD 更大；精确双侧 sign-flip 在 n 个种子上
最小只能给 `2/2^n`，m 个交互做 Holm 后是 `m·2/2^n`，**与效应量无关**。
当前 5 个声明的交互在 m=5 时：n=6 的最小 Holm p=0.15625，n=7 为
0.078125，n=8 才降到 0.0390625。也就是说 n≤7 的交互面板不可能在
alpha=0.05 下拒绝；此时「不显著」是分辨率不足，不是可加性证据。
`factorial_interaction_audit` 的 `min_n=3` 只是「可以计算」，不是
「可以拒绝」；把两者混同会让小样本 factorial 继续被过度解释。

**Fix.** 新增 `exp_lib.factorial_interaction_power(cells, ...)`：校验
`11/10/01/00` 四个等长、逐种子对齐的正 PPL 序列，计算逐种子
`d_i=log(PPL11/PPL10)−log(PPL01/PPL00)`；先检查 `m·2/2^n` 的 Holm
分辨率下限，不达标直接标 `resolution_limited` 并给出
`min_n_for_alpha`，不生成会被误读的 power 曲线；达标后复用
`paired_design_sensitivity`，以 `log(1+relative_pct/100)` 为 SESOI，
输出 SESOI 处功效、近似 MDE（并用 `expm1` 换回交互百分比）、配对 CI
半宽、所需 n，可选 `pair_retention` 附上完整覆盖下的 attrition 联合功效。
缺 cell、长度不齐、非有限/非正 PPL、布尔/字符串、非法 n/m/alpha/
target/SESOI 全部 fail-closed，不做 complete-case 交互。

新增 `exp_lib.factorial_power_audit(pilot=None, ...)`：对
`FACTORIAL_DESIGN` 的全部 factorial 做同一检查；`m_family` 默认等于
声明数（当前 5），`n_seeds` 可为标量或映射；无 pilot 时仍可先报
`family_resolution.min_n_for_alpha`（m=5 时为 8），但不把「没测」写成
通过。缺 pilot 是 `not_verifiable`，欠功效/分辨率不足默认只 warning，
`require_power=True` 才硬失败；未声明 SESOI 时 policy 默认 1% 会显式
标成 `policy_default` 并 warning，不冒充预注册。新增
`_min_n_for_holm_family(m, alpha)`：m=1→6，m=2/3→7，m=4/5/6→8，
m=7–10→9。

**Wiring.** `python audit_gates.py interaction_power [PILOT.json]`（别名
`factorial_power` / `power_2x2`，默认读
`analysis_v12/factorial_power_pilot.json`）只读 pilot 与声明表，不读
`results_*`/test；pilot 缺失时仍输出分辨率表并报
`not_verifiable_locally`。`audit_gates.py report` 新增
`interaction_power` 块，与 `interaction_sesoi` 并列：前者回答「设计能否
检出」，后者回答「检出后能否说可加」。pilot 格式为
`{"factorials": {name: {"cells": {...}, "relative_pct": 1.0}}}`。

**Boundary.** 这是设计期投影，不是 observed power：SESOI 处功效来自
pilot 的符号投影与正态 MDE 近似，`|d_i|` 不等时只是符号检验上界；pilot
来自 validation 且经过方法选择，存在 winner's curse，只能用于设计。
分辨率下限是硬约束：n 不足时只能加种子、缩小预注册交互族或承认面板只能
描述，不能换端点/挑种子/改用未校正 p。完整四 cell 是前提，缺 cell 不做
插补；MDE 的 `expm1` 百分比是交互尺度，不是任一臂 PPL 的百分比；
attrition 是独立配对保留模型，成组失败或与效应相关的丢失会更差。旧面板
不被追溯性改写，仍按 `factorial_interaction_audit` 报
`indeterminate`/描述性；不改任何 p/TOST/Holm/冻结判定或
`confirmation_frozen*.json`。

**Verification.** 新增 `test_review_part_29.py` 7 项：交互差值与
`paired_design_sensitivity` 逐值对账、m=5 的分辨率下限（n=5/6/7
`resolution_limited`，n=8 可分辨）、欠功效面板的 SESOI 功效/MDE/所需 n
与 `require_power` 硬门、方向镜像与 PPL 整体缩放不变性、缺 cell/长度
不齐/非法 PPL/非法参数的 fail-closed、attrition 接线，以及
audit gate/CLI/report 接线。完整 CPU 套件（`706 passed, 3 skipped`）与
`python audit_gates.py interaction_power`、`python audit_gates.py report`
在本轮最终树通过。

## 2026-10-29 — 消融族的多重校正要利用相关性：精确符号翻转 max-T 与同时区间（本轮对比/消融方法硬化，no new runs）

零 GPU；不新增训练、不读取 test split、不修改冻结清单字节/种子/端点/
alpha/Holm 规则，也不改 `CODE_SEMANTICS`；`.github/` 与所有 README 未触碰。

**Problem.** 消融表的多重校正只有 Holm step-down。Holm 合法，但它把每条
对比当成独立证据：`m` 条共享同一参照臂、同一组种子、同一批评测行的对比，
其逐种子配对差高度相关，Holm 仍对最小 p 乘 `m`，在强相关族上系统性
过度保守。反过来，若事后改用更宽松的校正、只报显著项，又等于用未预注册的
门槛替代 Holm。两种误读都需要一个可复核的中间读数：在不改变声明校正的
前提下，说明「Holm 在这个相关族上保守了多少」，并给出整族同时可读的区间。

**Fix.** 新增 `exp_lib.maxt_signflip_adjust`：在既有逐对比精确符号翻转的
同一个零假设下（配对差关于 0 对称、各种子符号独立等概率），枚举全部
`2**n` 个符号指派，得到**整族** max-T 的精确联合零分布；学生化用的
`sd_k` 在符号翻转下不变，所以这不是近似。按观测 `T` 从大到小做
Westfall-Young free step-down，调整 p 额外要求 `>=` 自身未调整 p 并
单调化；族最大值零分布的 `1-alpha` 分位点给出 `mean ± c*sd/sqrt(n)` 的
同时区间；该区间是 studentised 的 t 型近似（p 值精确，区间不是精确反演），
并显式返回临界值处的 `ci_tail_level`（n=5 时最小可达双侧尾概率
2/32=0.0625，达不到 95%，返回 `ci_simultaneous=None` 而不是把 93.75%
标成 95%）。

新增 `exp_lib.maxt_min_n_for_alpha`：给出最佳情形（完全相关对比）的种子
下限，alpha=0.05 时为 6（2/64），与 Holm 在 m=5 时的 8 并列显示；文件明确
标注这是乐观下界，独立对比会趋近 Holm 的 m*2/2**n。

新增 `exp_lib.maxt_joint_correction`：按精确种子集合分组，只接受完整、
受控的配对；`family_ready` 要求 `m_eligible == m_family` 且只有一个
种子组，否则 `status='subset_only'`，保留描述性 `p_maxt` 但
`significant_maxt=False`、`maxt_only` 为空——「跑完的对比越少、校正越宽」
正是要堵住的反方向。`n>20` 拒绝（不 Monte-Carlo 冒充精确）；零种子方差
且非零均值的退化对比整族 `not_estimable`，不制造 p。

**Wiring.** `methodology_audit` 在既有 `_holm_assign` 之后，把
`components` + `baselines` 的完整受控对比放进同一个 `m_family`，每个
entry 写 `p_maxt` / `p_maxt_single` / `p_raw_signflip_maxt` / `t_maxt` /
`maxt_rank` / `reject_maxt` / `significant_maxt` / `ci_simultaneous` /
`maxt_group`；`report['family']['maxt']` 汇总 `status`、`family_ready`、
`m_eligible`、`n_groups`、`best_case`、逐组 `critical_value`/
`ci_tail_level`/`ci_method`/`p_floor_single`/`deltas_fp`/`contrasts`、`n_holm_reject`/`n_maxt_reject`、
`holm_only`/`maxt_only` 与 warnings；顶层增加 `maxt_only_contrasts` /
`holm_only_contrasts` / `*_pairs`。`verified_components` 仍只由
`p_holm_global` 决定，`significant_maxt` 不进入
`_classify_group_status`；max-T 多拒绝时只写 warning，明说这是
「Holm 保守」的敏感度读数，要变成主张必须下一轮预注册 max-t。

新增 `audit_gates.py joint_correction [OUTDIR...]`（别名 `joint` / `maxt` /
`max_t`）：从落盘 `methodology_audit.json` 的逐种子 deltas 重算整族
max-T，逐项核对 `p_maxt`、`p_maxt_single`、原始 p、rank、reject、
同时区间、组号、单调性与 `holm_only`/`maxt_only`；缺 `family.maxt` 的
legacy 产物 `not_verifiable`，篡改判 `fail`。`audit_gates.py report` 增加
顶层 `joint_correction`，`audit_methodology` 的逐面板输出带
`joint_correction` / `maxt_only_contrasts` / `holm_only_contrasts`。

**Boundary.** 这是同一数据的第二个多重校正读数，不是新检验、不是新实验；
不改变任何 p/Holm/`verified`/`ablation_complete`/冻结判定，也不进入冻结
确认族。其有效性与逐对比符号翻转相同（配对内交换性：同数据顺序、同评测
行、同腐坏网格；初始化 RNG 按种子播种），不修复不满足交换性的面板，也
不把种子变成随机总体抽样。它不解决选择偏差、optional stopping（
`look_history` 的序列校正仍独立适用）或评测单位不确定性（两层
bootstrap/document-cluster 仍回答「换文档会怎样」）。`family_ready` 依赖
声明族本身；看结果后扩大族仍是 forking path，max-T 不能修复。
`ci_tail_level` 低于请求水平时不给有限区间。

**Verification.** 新增 `test_review_part_30.py` 11 项：m=1 与
`exact_signflip_p` 逐值对账（n=3..8）、p_floor/区间/排名；两条完全相关
对比下 max-T 拒绝而 Holm 不拒绝（n=6 时 .03125 vs .0625）；整体缩放、重复调用与名称序颠倒（并列 t 的确定性）的不变性；缺臂族 `subset_only` 且
`significant_maxt=False`；完整族写入但不提升 `verified`；多种子集合分组
与 excluded/退化对比的 fail-closed；空/不齐/布尔/NaN/n<2/超上限/小数上限/小数族大小/零方差/
非法 alpha 全部拒绝；极端量级 1e308/1e-300 与常数量级 p/t 一致；audit gate 干净产物 ok、篡改 p/区间/rank/
`family_ready`/`maxt_only`/`p_maxt_single` 判 fail、legacy 只 warn、
无产物 `not_verifiable_locally`，以及 CLI 别名接线。完整 CPU 套件在本轮
最终树通过。

## 2026-10-29 — 完整案例不是计划面板：缺失种子的部分识别、tipping point 与覆盖台账（本轮结果分析硬化，no new runs）

零 GPU；不新增训练、不读取 test split、不修改冻结清单字节/种子/端点/
alpha/Holm 规则，也不改 `CODE_SEMANTICS`；`.github/` 与所有 README 未触碰。

**Problem.** 既有规则只在 `complete_pairs=False` 时拒绝显著，却没有把
「计划了但没跑」「有记录但主终点不可用」「有记录但被控制门排除」分开，
也没有回答「剩下的 complete-case 方向需要多极端的未观测种子才能反转」。
在运行时限/失败/resume 下，缺失可能是臂相关的；把 complete-case 均值当成
计划面板均值是一个未声明的选择。

**Fix.** 新增 `missingness_sensitivity`：以计划种子帧 `N=n+u` 为分母，
在声明的未观测差范围 `[lo,hi]` 上给出总体均值的 sharp bounds
`m + (u/N)(x-m)`，并给出 `x0=-(n/u)m`、SESOI tipping
`xT=T+(n/u)(T-m)`、`tipping_ratio`、`tipping_in_range_*`；
`n_missing` 与 `n_excluded` 分开计数但都进入 `u`。默认范围是观察
`min/max` 并显式标成描述性；`N<=16` 时附 imputed 情景 p，`N>16` 跳过。
新增 `panel_missingness_audit` 统计 planned/present/usable/unusable/
missing 与 `unequal_arm_coverage`。

**Wiring.** `methodology_audit` 新增 `planned_cells` 参数（缺省从
`outdir/run_manifest.json` 自动读取），每个 contrast 写
`missingness`，面板写 `report['missingness']`；`step_00` 保存
`shared_seeds`/`skipped_seeds`，新 `step_02` 在既有 Holm/max-T/SESOI
之后附加该块并负责写盘/打印。新增
`audit_gates.py missingness`（别名 `missing`/`missing_data`/
`attrition_sensitivity`）从落盘 `observed_deltas`/`planned_seeds` 重放
bounds/tipping/status；篡改判 fail，legacy 报 not_verifiable；`report`
顶层与 `audit_methodology` 面板均带该块。它不改变任何 p/Holm/
`verified`/max-T 或冻结判定。

**Boundary.** 不检验 MCAR/MAR、不是新显著性检验、不把 `n_excluded`
当成无效应；默认观察范围之外的值不受保护；`unequal_arm_coverage` 不是
偏倚的证明，只说明 complete-case 子集可能由臂选择；不能替代补跑。

**Verification.** 新增 `test_review_part_31.py` 12 项：完整帧恒等、
已知 tipping/margin/范围端点、跨零 fragile、excluded 与 missing 分账、
缩放不变、畸形帧 fail-closed、面板 coverage 三态、无计划 `no_plan`、
`methodology_audit` 附加块但不改变判定、gate 重放/篡改/legacy、CLI/
report 接线。完整 CPU 套件在本轮最终树通过。

## 2026-10-29 — 单因素实验的初始化契约与融合安慰剂（本轮实验设计与准备硬化，no new runs）

零 GPU 的设计点硬化；不新增训练、不读取 test split、不修改冻结清单的
字节/种子/端点/alpha/Holm 规则，也不改 `CODE_SEMANTICS`；`.github/` 与所有
README 未触碰。

**Problem.** `local_fusion` 的 on/off 对照在 cfg 上只差 `fuse_mode`，但
learned 臂构造 `nn.Conv1d` 会消耗全局 torch RNG，使该臂从这一层起所有后续
块参数的初始化都与 no-fuse 臂不同；`single_factor_design_audit` 只看
AttnCfg diff，`init_core_fp` 只覆盖 trunk，二者都看不见这个第二因素。另外
`audit_gates.py single_factor` 之前没有读取动态/HCA 草稿，H1/H2/H4 这类
`mechanism_claim_eligible=true` 的对照若被换成多因素臂对，只有
`dynamic-plan` 会拒绝。

**Fix.** (1) 构造 `fuse_conv` 时保存/恢复全局 torch RNG；learned 保持
identity 初始化，random 用 W_q 哈希 + `b'fuse'` 的局部 generator 抽同 shape
kernel 并冻结，因此 learned/random/absent 的非 fuse 参数逐值相同。
(2) 新增 `AttnCfg.fuse_mode='random'`、`_frozen_random_fuse_weight`、臂
`csa_dyn_randfuse`/`hybrid_csa_dyn_randfuse`（并保留 `fuserand` 别名），
`FUSION_CONTROL_DESIGN` 与 `fusion_control_audit`：treatment learned /
capacity placebo frozen-random / absent identity 三级；逐 base 核验
`fuse_mode` 精确 diff、同参数、placebo 冻结、absent 参数差等于 fuse 参数、
三级非 fuse 初始化一致、两族声明与键集齐全。
(3) 新增 `single_factor_initialization_audit`：对 cfg 门判为 `single_factor`
的每条对照，同 seed 建 tiny 模型并逐参数对账；只放行声明干预对应的参数
（`fuse_conv`/`sink`/`delta_logit`），其余任何差异 fail。
(4) `audit_single_factor` 增加 `initialization`、`dynamic_study` 与
`fusion_controls` 块；`fusion_controls` 同时进 `report` 顶层；
`audit_gates.py fusion_controls`（别名 `fusion_placebo`/`local_fusion`）
接线。新增 `local_fusion_learning` 的 `ABLATION_DESIGN` /
`ABLATION_INTERVENTION_KEYS` / `HYPOTHESIS_DESIGN` / `ABLATION_SESOI`，并将
新臂排程到 `RUN_ABL`。

**Boundary.** 初始化审计只回答「同 seed 下非干预参数是否相同」，不检验
训练轨迹；random fusion 是固定非学习安慰剂，不是「任意固定 kernel 都比
identity 好」的检验；on/off 仍移除参数，故 `local_fusion` 的
`requires_param_match` 仍为 False。动态门只把单因素要求纳入本门，其他动态
设计问题保留在 `design_status`/`design_violations`。新臂是下一轮排程，不是
已有证据。

**Verification.** 新增 `test_review_part_32.py` 6 项：融合阶梯通过 shipped
design、缺 level/未排程/复用臂/多改一个键/重复 role fail-closed、learned
与 placebo/absent 的非 fuse 参数逐值相同且 placebo 冻结、故意消耗全局 RNG
的 factory 被初始化审计拒绝、`single_factor` 门包含 fusion/initialization/
dynamic_study、动态 H1 换成多因素臂对后该门 fail。完整 CPU 套件
`749 passed, 3 skipped`；`python -m explib.confirmation plan` 仍为
72/72 合成记录、6 个比较；`python audit_gates.py single_factor` 与
`python audit_gates.py report` 在当前树均为 `ok`。

## 2026-10-29 — 最终一步不是整条轨迹：配对评测历史的符号稳定性（本轮结果分析硬化，no new runs）

零 GPU 的结果分析硬化；不新增训练、不读取 test split、不修改冻结清单的
字节/种子/端点/alpha/Holm 规则，也不改 `CODE_SEMANTICS`；`.github/` 与所有
README 未触碰。

**Problem.** 冻结端点是固定最终评测，这是正确的；但同一批配对种子的
`ppl_history` 只用被画成 seed 平均曲线/做训练健康诊断。报告因此无法区分
「最终符号是持续 regime」与「最后一步才交叉」或「gap 仍在缩小」。现有
`within_seed_noise`、逐种子 LOO、`seed_driven`、max-T 都回答种子间稳定性，
不回答轨迹上的符号稳定性；而把轨迹上的多个评测步当独立重复检验又会引入
强相关读数，扩大 forking paths。

**Fix.** 新增 `exp_lib.paired_trajectory_stability(history_a, history_b, ...)`：
在**所有纳入种子、两条臂都有的共同评测步**上计算配对 log-PPL 差（与 NLL
差同尺度、对两臂整体缩放不变）和原始 PPL 差；尾窗固定为最后 3 个共同步
（`TRAJECTORY_POLICY`，不是看结果后挑的），给出 `stable` /
`late_crossing` / `final_point_only` / `zero_final` / `insufficient_history`，
以及全轨迹符号变化次数、最后变化步、每个种子的最终符号一致性、尾窗两臂
log 斜率、gap 斜率与 `widening`/`narrowing`/`flat`。历史最终符号与主端点
符号单独对比：历史用评测子集、主端点用完整固定验证行，两者不同样本，只
报 `primary_vs_history_sign` 并警告，不假设相等；同时报历史最终的 log 比
符号 `primary_vs_history_log_sign`，log 与原始 PPL 差符号相反时标
`scale_sensitive_final`，强制引用时点明估计量。`methodology_audit` 对每条
有配对 `deltas` 的对照写 `entry['trajectory_stability']`，面板写
`report['trajectory_stability']`（逐对照行 + late/narrowing/disagreement
名单），`_print_methodology_audit` 增加对应小节。无历史/畸形历史只标
`insufficient_history`/`invalid`，不阻断聚合，也不在 legacy 面板上伪造
stable。新增 `audit_gates.py trajectory`（别名 `traj`/
`trajectory_stability`）从落盘 `series`/`per_seed_tail` 重算 final sign、
尾窗 status、符号变化、gap 斜率/trend、逐种子一致计数与
`final_step_is_endpoint`，并核对面板计数；篡改判 `fail`，旧产物无该块判
`not_verifiable`。

**Boundary.** 不替代固定最终端点：`stable` 不是「效应真实」的证书，
`late_crossing` 也不是「无效」的证明；仍只由既有 p/Holm/TOST/max-T 与冻结
规则作判定。不是新检验：尾窗多个评测步高度相关，不产生轨迹 p 值、不进入
任何 Holm/max-T 族、不改变 `verified`/`significant`。历史与主端点可能测在
不同行上，符号不一致可能只是评测子集差异；一致也不证明主端点在轨迹上
稳定。共同步不足时显式不可用，丢步数落盘，不用不等 n 的步均值补造轨迹。
默认 3 点只是最近一段的描述性 convention；训练早期交叉不算 late crossing。
缺 `ppl_history` 的旧记录不回溯、不重跑。

**Verification.** 新增 `test_review_part_33.py` 13 项：稳定轨迹/主端点同向、
late crossing、`final_point_only`/`zero_final`、gap narrowing 警告、不等步
网格丢步与跳过种子、整体缩放不变性、重复步/非正 PPL/布尔 PPL/非整数步/
非法尾窗/重复种子/非数值主端点 fail-closed、共同步不足、历史在端点前结束时标
`final_step_is_endpoint=False`、log/raw 符号相反时 `scale_sensitive_final`
与两个 primary 对比字段分别正确、`methodology_audit`
有/无历史时所有判定承载字段逐值相同、late crossing 与 history/primary 不
一致的顶层 warning；审计门干净→`ok`、篡改 final sign→`fail`、旧产物→
`not_verifiable`。完整 CPU 套件在本轮最终树通过。

## 2026-10-29 — 预注册的不只是假设：失败/重跑处置与逐次尝试审计（本轮实验设计与准备硬化，no new runs）

零 GPU；不新增训练、不读取 test split、不修改 `confirmation_frozen*.json`
的字节/种子/端点/alpha/Holm 规则，也不改 `CODE_SEMANTICS`；`.github/` 与所有
README 未触碰。

**Problem.** 计划单元与测量单元不是同一个对象。一个 `(arm, seed)` cell 可能
先 OOM/timeout/NaN/被 runtime cap 截断，再在下一轮成功；已经成功的 cell 也
可能因 `run_cfg` 变化被 resume 逻辑删除重跑。原仓库把每次尝试留在
`experiments.jsonl`，却没有门回答「最终保留的是第几次成功、为什么」。
「跑不出数就再跑一次，直到有数」是一种选择机制，而且昂贵/动态臂更容易触发
cap/OOM，重试次数在臂间并不均等；§93 的 bounds 假设缺失集合已确定，无法
替代尝试层的审计。

**Fix.** 新增 `exp_lib.build_run_disposition_plan` /
`write_run_disposition` / `run_disposition_audit` 与
`audit_gates.py disposition`（别名 `attempts` / `retries` /
`run_disposition`）：

- 每面板在第一条 cell 之前写 `run_disposition.json`，声明 planned cells 与
  处置策略：默认 `retention=first_success`、`replacement=same_cell`、
  `max_attempts_per_cell=2`、`allow_supersede=false`；失败类分成可替换
  （infrastructure/resource_exhausted/timeout/runtime_cap）与不可替换
  （numerical/config/protocol），未声明/歧义类 fail-closed。
- 计划身份只覆盖 `planned_cells + policy`（`content_sha256`），resume 同内容
  no-op；不同内容必须作为带 `reason` 与旧 `content_sha256` 的 revision 写入，
  否则保留原计划并返回 conflict，绝不静默覆盖。`panel_start` 事件绑定计划
  摘要，使「计划先于第一次尝试」可由链化账本复核；审计另外比较
  `plan.created_at` 与最早 cell 的 `started_at`，计划晚于首条 cell 判
  reconstruction（策略显式允许时只 warning）。
- 审计从 `experiments.jsonl` + `summary.json`/`run_manifest.json` 重放每次
  尝试：保留值必须是第一次成功；未计划种子/臂、成功后改了 `run_cfg`、
  不可替换失败被重试、未知失败被重试、超过每 cell 或总尝试预算，分别判
  fail；链化账本哈希失败直接 fail；缺计划/缺 cell 事件
  `not_verifiable_locally`。失败文本用词边界匹配（`oom` 不命中
  `kaboom`），评测/探针 cell（`arm::s7`、`arm::seed3::niah_train`）只计数
  为 `n_ignored_cell_events`，不硬塞进 seed 帧。
- 逐臂 `by_arm` 与 `differential_attempts` 报告尝试率/重试率/每 cell 最大
  尝试数和 `concentrated`/`balanced`/`no_retries`。它是描述性读数，不产生
  p 值：少量 cell/臂下做差性流失检验会把噪声当机制。
- `audit_gates.py report` 增加顶层 `disposition`；该门不重算 PPL/p/Holm，
  不进入冻结族。
- 设计前馈 `audit_gates.py design` 读 `statistics.disposition_policy`：
  confirmatory 必须 `first_success` / `same_cell` / 不允许 supersede，否则
  violate；缺声明只 warning 并提示尚未预注册失败/替换规则；缺
  `max_total_attempts` 也 warning。

**Boundary.** 不是新检验，也不修复已经发生的选择偏差；它让「保留的是哪次
尝试」可见。可替换失败的重试是允许的，前提是失败类已声明、回到同一 cell、
在尝试预算内且所有尝试都留在账本。账本 tamper-evident 而非签名。缺计划时
只能退回 observed/caller-supplied 帧，后者无法发现事后加 cell，因此不假绿。
与 §93 missingness 互补：本门回答「保留的是哪次尝试」，§93 回答「未观测
种子能把结论推到哪」。

**Verification.** 新增 `test_review_part_34.py` 20 项：干净单次 ok、可替换
失败重试 warn、第二次成功被保留 fail、保留第一次成功只 warn、未计划测量/
尝试 fail、key/seed 不一致/重复 run_id/未知 status/布尔 seed/非对象事件
fail-closed、NaN/未知失败重试 fail、cell 与总尝试预算、supersede 默认 fail
与显式允许、计划写入器绑定/冲突/revision、重试集中单臂为描述性、无计划
not_verifiable、评测/探针 cell 只计数、计划晚于首条 cell 判 reconstruction、
cap 状态归入 runtime_cap、损坏/篡改计划判 fail、`design` 前馈拒绝
confirmatory 的 last_success/supersede，以及 gate/CLI 接线与账本篡改 fail。

## 2026-10-30 — 训练时去掉不等于推理时在用：同 checkpoint lesion 与操作检查（本轮对比/消融方法硬化，no new runs）

零 GPU 的设计/分析硬化；不新增训练、不读取 test split、不修改
`confirmation_frozen*.json` 的字节/种子/端点/alpha/Holm 规则，也不改
`CODE_SEMANTICS`；`.github/` 与所有 README 未触碰。

**Problem.** `ABLATION_DESIGN` 的 on/off 与 §84/§94 的同容量 placebo 都是
**between-model** 对照：它们比较两个分别训练的模型。显著差异只能说明「带
模块训练改变了模型」，不能说明「训练出来的模型在推理时使用该模块」。一个
模块可以只是训练期优化支架；learned sink 初始为零、learned fusion 初始为
identity，也可能在训练后仍是 no-op。终点表无法区分「模块无用」与「模块在
推理时没被使用」；反向的「被使用但冗余」或「被使用但有害」也没有读数。
§94 的初始化审计只保证训练前非干预参数一致，仍没有同 checkpoint 的干预。

**Fix.** 新增 `exp_lib.component_lesion` / `lesion_sensitivity` /
`lesion_effect` / `lesion_panel_audit` / `lesion_design_audit`：

- `LESION_DESIGN` 白名单声明 `attention_sink`（`use_sink`/`full_sink`）、
  `local_fusion`（移除 `fuse_conv`）、`content_pooling`
  （`content_mode='zero'`）的适用臂与干预键。`component_lesion` 复制每层
  `AttnCfg`、临时替换声明的子模块，并在 `finally` 恢复**原对象**；权重
  `state_dict` 不变，异常也恢复。
- `component_lesion_state` 区分参数存在/功能初值；**零值 sink 仍是
  `present`/`zero_valued`**（sink 槽以 `exp(0)` 进入 softmax 分母，不是
  no-op），只有 fusion 的 identity kernel 是 `trivial`。
- `lesion_sensitivity` 在同一输入上比较 intact/lesion logits，要求
  `changed=True` 且 `weights_unchanged=True`；`lesion_effect` 对
  `log(PPL_lesion/PPL_intact)` 做 exact sign-flip + 配对 TOST，输出
  `necessary`/`harmful`/`redundant`/`engaged_below_sesoi`/`inert`/
  `not_checked`/`invalid`/`underpowered`/`inconclusive`。操作检查未通过时
  `claim_allowed=False`，PPL 零差不得写成冗余。
- `lesion_panel_audit` 重放
  `{component, arm, seed, checkpoint_sha256, intact_ppl, lesion_ppl,
  sensitivity, eval_fp}`：同组 seed 唯一、`eval_fp` 一致、checkpoint 身份
  存在、权重未变；缺操作检查只 warning，多个 training seed 共用同一
  checkpoint 也只 warning（伪重复，不是独立样本）。`lesion_design_audit`
  用真实工厂
  构造每个声明臂，检查组件存在、lesion 至少作用一层、臂已排程且属于留一
  法族、机制键被 `ABLATION_INTERVENTION_KEYS` 覆盖。fusion 的 identity
  初值只 warning（必须到训练后 checkpoint 再做操作检查）。
- `audit_gates.py lesion`（别名 `inference_lesion` / `lesion_panel` /
  `component_lesion`）扫描 `results_*/lesion.json` 与
  `analysis_*/lesion.json`；无产物 `not_verifiable_locally`。`report` 增加
  顶层 `lesion`。该块不改变 `methodology_audit` 的 Holm/`verified`/max-T/
  `ablation_complete`，不进入冻结确认族。

**Boundary.** 它需要已保存的 checkpoint；本快照没有 `results_*/`，所以没有
任何 lesion 数值。lesion 固定权重，回答「这个训练好的网络在推理时是否依赖
它」，不替代 `intact - absence` 的训练效应，也不定位子电路。操作检查只说明
干预改变了输出，不是机制证据。面板内 `eval_fp` 必须一致；fusion 的 lesion
刻意不改 `cfg.fuse_mode`，以免把 row/batched 后端切换混进干预。`redundant`
是 SESOI 内等价，不是绝对零效应；`inert` 是本次前向不变，不是组件永远无用。
缺 checkpoint 的 seed 不能插补，§93 的 missingness bounds 仍独立适用。

**Verification.** 新增 `test_review_part_35.py` 8 项：设计门接受 shipped
三类组件且每臂可构造/可干预；context 异常恢复原 cfg/`fuse_conv` 对象且权重
指纹不变、未知/缺席组件 fail-closed；零值 sink 仍 `present` 且 lesion 改变
输出、lesion 内改 sink 不再影响输出；identity fusion 初值 `trivial`、
非 identity lesion 改变输出且 lesion 内改 kernel 不再影响输出；content
lesion 改前向不改权重；`lesion_effect` 全部判定与长度/布尔/非正 PPL/重复
种子/非法 alpha 的 fail-closed；面板门对 `eval_fp` 混杂、重复 seed、缺
checkpoint、权重变化、未声明臂、NaN PPL 判 fail、无记录
`not_verifiable`；`audit_gates lesion PATH` 干净重放 ok、篡改 fail。完整
CPU 套件在本轮最终树通过。

## 2026-10-30 — 报的是哪一个 checkpoint：端点选择的可追溯性与跨臂一致性（本轮结果分析/实验设计硬化，no new runs）

零 GPU 的设计/分析硬化；不新增训练、不读取 test split、不修改
`confirmation_frozen*.json` 的字节/种子/端点/alpha/Holm 规则，也不改
`CODE_SEMANTICS`；`.github/` 与所有 README 未触碰。

**Problem.** 训练记录同时保存 `steps_completed`、`early_stop`、
`best_step`/`best_val_ppl`/`final_minus_best` 与 `checkpoints`，但一个
`ppl` 字段没有说明它来自固定最终步、验证集最佳步、早停步还是被 cap 截断
的步。这四种是不同 estimand。逐臂挑验证集最佳点再比较，是在比较两个被
选择的最大值（winner's curse）；A 的最终步 vs B 的最佳步连预算和选择规则
都不同；在选点用的同一批 validation 行上报告该点则是乐观偏差。§95 只问
最终符号在轨迹上是否持续，§97 只问同一 checkpoint 在推理时是否使用组件，
都没有回答「这个 checkpoint 是怎么选出来的」。

**Fix.** 新增 `exp_lib.endpoint_selection_audit` 与策略
`ENDPOINT_SELECTION_POLICY`：

- 逐 `(arm, tag, seed)` 记录读取 `steps_planned`/`steps_completed`/
  `reported_step`/`selection_rule`/`selection_split`/`eval_split`/
  `checkpoint`/`early_stop`/`curve_health`（以及 train_variant 写在顶层的
  `best_step`/`best_val_ppl`/`final_minus_best`）。
- 规则推断：显式声明优先；`checkpoint.mode`、早停、完成步与计划步、
  报告步与最佳步的关系依次给出 `final_step` / `best_validation_step` /
  `early_stopped_best` / `early_stopped_stop` / `truncated_step` /
  `intermediate_step` / `unknown`。只有 `steps` 的旧/rope 记录按运行长度
  读成 `final_step`，但输出 `steps_completed_inferred=true`；
  `cap_truncated=true` 优先判 `truncated_step`。既无 `steps_completed` 也
  无 `steps` 才是 `unknown`，不是 `final_step`。
- 配对比较：同臂跨种子混规则、跨臂规则不一致、`selection_split='test'`、
  选择 split 与评测 split 相同、`checkpoint.step` 与报告步不符、
  `checkpoint.mode='both'` 判 fail；数据依赖规则在独立 split 上只 warn；
  同配对种子完成步数不同、部分 checkpoint 身份、负
  `final_minus_best` 警告。
- manifest 可声明 `endpoint_policy`（`kind`/`selection_split`/
  `comparison`/`arms`/`require_checkpoint_identity`）；
  `require_preregistered=True` 时无声明或声明与记录不符判 fail，数据依赖
  规则无 `selection_split` 也判 fail。旧清单不会被猜成 `final_step`。
- `audit_gates.py endpoint_selection`（别名 `endpoint` / `checkpoint` /
  `checkpoint_selection`）扫描 `results_*/summary.json`，同目录
  `run_manifest.json` 的 `endpoint_policy` 参与对账；无产物
  `not_verifiable_locally`。`audit_gates.py report` 增加顶层
  `endpoint_selection`。
- `methodology_audit` 新增 `report['endpoint_selection']`：普通最终步面板
  `ok`；早停/最佳步面板 `warn` 并给一条顶层 warning；未知规则
  `not_verifiable`，不伪造 clean。该块不改变 p/Holm/max-T/`verified`/
  `ablation_complete`，不进入冻结确认族。

**Boundary.** 这不是新的显著性检验：没有 p 值/区间/Holm 成员。它不证明
一个 `final_step` 训练正确，也不证明一个最佳步结果错误；只证明「两个臂
的端点规则是否可比、来源是否被声明」。在独立 validation 上选点、独立
test 上报告是允许的设计，本门只要求声明并给 warning；同 split 选点+报告、
在 test 上选点、跨臂规则不一致才是 fail。`checkpoint.sha256` 是自报来源，
不是权重审计；`ckpt_mode` 是「保存了哪些 checkpoint」，不是「引用了哪
个」，不能单独当规则证据。已经按「挑最好 checkpoint」产生的面板不能被
本门修复，只能被标出并降级/重跑。缺 checkpoint 的 seed 不能插补，
§93/§96/§97 的边界仍独立适用。

**Verification.** 新增 `test_review_part_36.py` 12 项：干净最终步面板 ok；
跨臂 final/best 不一致、同臂混规则判 fail；早停最佳/早停停止/截断/最终
四类规则与预算警告；独立 split 只 warn、同 split 或 test 选点判 fail、缺
selection_split 警告；manifest 的 final/best 声明、`require_preregistered`、
arms 匹配与畸形声明；checkpoint step 不符、`mode='both'`、部分身份与
`require_checkpoint_identity`；长度/布尔/非正 PPL/重复记录/非法规则/非法
split 的 fail-closed；tag 不跨池与 summary key 兼容；单臂/未知规则
`not_verifiable`；audit gate 干净重放 ok、篡改 fail、缺产物
`not_verifiable_locally` 与 CLI/report 接线；以及 `methodology_audit` 在
有/无端点字段下所有判定承载字段逐值相同、只在早停面板增加 warning。完整
CPU 套件在本轮最终树通过（830 passed, 3 skipped）。

## 2026-10-30 — 同一份证据要重建出同一份报告：schema 2 时间戳、证据输入绑定与源码树卫生（本轮可复现性/工程规范硬化，no new runs）

零 GPU 的可复现性/工程规范硬化；不新增训练、不读取 test split、不修改
`confirmation_frozen*.json` 的字节/种子/端点/alpha/Holm 规则，也不改
`CODE_SEMANTICS`；`.github/` 与所有 README 未触碰。

**Problem.** `stamped_analysis` 的 `created_at` 是墙钟，聚合器又把
`generated_at` 写进被哈希的 body，所以同一份结论连续生成两次会得到不同的
`content_sha256`；内容寻址只能发现篡改，不能把重建识别为同一结果。更关键
的是 `run_report` 调 `stamped_analysis('audit_report', res)` 时没有传
`inputs`，顶层审计报告因此 `inputs == {}`：改动 `results_*/summary.json`、
`analysis_*/test_stats.json` 或冻结清单后，报告自身仍然验证通过，`body` 与
证据之间没有机器可检的绑定。跨报告也没有「同一证据重建」的比较器，
`explib.reproduce compare` 只覆盖训练结果目录。工程侧还有三处遗留：
`.tmp_lint/`（vendored pyflakes + 拼接草稿）和
`archive_test_dcgf_cjz/out/*.tmp27904` 被 Git 跟踪；`MANIFEST.in` 手写测试
白名单，漏掉三个顶层回归测试；没有 `.gitattributes` 固定 LF，而
`source_tree_manifest()` 按字节计算源码身份。运行记录也没有命令行，复现者
只能从 README 猜 driver/mode/flag。

**Fix.**

- 新增 `exp_lib.source_date_epoch` / `reproducible_timestamp`（与
  `archive_release._source_date_epoch` 同语义），并引入
  `content_schema = 2`：schema 2 把 `generated_at` 视为 volatile，同时把
  `content_schema` 本身纳入摘要；旧记录没有该字段，继续按旧规则验证，
  不追溯作废已发布哈希。设 `SOURCE_DATE_EPOCH` 时 `created_at` /
  `generated_at` 固定为 UTC，同一 checkout 上的 stamp 可字节重建。
- 新增 `audit_gates.audit_evidence_inputs(root)`：确定排序地枚举
  `results_*/`、`analysis_*/` 下的文本/JSON 证据（`.json/.jsonl/.csv/.tsv/
  .md/.txt/.yaml/.yml`）、冻结清单和 `RELEASES.md`，返回
  `evidence:<relpath> -> path` 与 `scope`（文件数/字节数/上限/跳过项）。
  `run_report` 把它们传给 `stamped_analysis(inputs=...)`，并把 `scope`
  写进 body；`analysis_integrity` 随即可重放输入字节。`analysis_audit/`
  被排除，因为报告会向该 ledger 追加事件，且它另有 hash-chain/anchor 审计。
  `run_report` 的 `analysis_integrity` 还排除即将被替换的
  `audit_report.json` 自身：否则新报告会审计上一份报告的新鲜度，证据一变
  就出现 fail→ok 的一次性滞后；独立 `analysis_integrity` 子命令仍审计它。
- 新增 `exp_lib.analysis_report_diff` 与
  `python -m explib.reproduce report A B`：分开 body / inputs / code /
  environment 四个维度，给出 JSON pointer 级 body 差异，判定
  `reproduced` / `body_changed` / `inputs_changed` / `provenance_changed` /
  `schema_changed` / `unverified` / `invalid`；`invalid` 优先，一侧摘要与
  自身 body 不符是篡改。`--ignore /experiment_ledger`（或
  `--ignore-ledger`）用于只追加审计 trace。
- `environment_snapshot()` 增加 `invocation`（argv/entrypoint/`-m` 模块/
  参数个数），token、password、secret、api-key、credential、auth 的值脱敏
  为 `***`；它不进入 `execution_fingerprint()`，不改变同环境配对。
- 工程卫生：删除被跟踪的 `.tmp_lint/` 与 `archive_test_dcgf_cjz/`，加入
  `.gitignore`；`MANIFEST.in` 改为 `include test_*.py` 并纳入 `Makefile` /
  `.gitattributes`；新增 `.gitattributes` 固定 `text=auto eol=lf` 与二进制
  标记；`prune` 缓存/草稿目录。

**Boundary.** 这不是新的统计检验，不改变任何 PPL/p/Holm/`verified`/冻结
判定。`content_sha256` 是内容身份而非签名；`inputs_sha256` 只覆盖枚举到的
文本/JSON 证据，超过 64 MiB 的文件、二进制 checkpoint、`.npy` 缓存与
`analysis_audit` ledger 不在其中，它们仍由 run record / release manifest /
ledger chain 负责。`reproduced` 只表示 body 与输入绑定一致；源码/环境变化
单独报 `provenance_changed`，不会被「数字相同」掩盖。`SOURCE_DATE_EPOCH`
只固定时间字段，不虚构原始运行时间；不设时仍记录墙钟。仓库仍无随附
evidence release，本轮绑定不能替代 `RELEASES.md` 的外部归档与 verifier。
删除的两个目录全仓库无引用，是历史迭代的本地草稿，不是实验输入。

**Verification.** 新增 `test_review_part_37.py` 9 项：epoch 解析与 archive
解析器逐值一致（含 bool/负数/小数/日期/科学计数/超范围/空白）；固定 epoch
下时间戳与整份 stamp 字节稳定；legacy 的 `generated_at` 仍参与摘要而
schema 2 不参与；证据枚举的确定性、扩展名、上限与 `analysis_audit` 排除；
输入被编辑后 `verify_stamped_analysis` 判 fail；report diff 对
body/inputs/code 三类变化分别判定、篡改判 `invalid`、`/experiment_ledger`
可忽略；CLI 退出码 0/1/2；argv 脱敏与 `-m` 模块识别；以及 `git ls-files`
无 `.tmp_lint`/`archive_test_*`/缓存/结果目录、`.github/` 与 README 仍在、
MANIFEST 覆盖全部顶层 `test_*.py`、`.gitattributes` 固定 LF。最终树上用
`SOURCE_DATE_EPOCH=1700000000` 连续生成两份 `audit_report.json`，
`python -m explib.reproduce report A B --ignore-ledger` 返回 `reproduced`
（body/inputs/code/environment 全 match；唯一增长的是被 ignore 的
`analysis_audit` ledger 事件数）。完整 CPU 套件在本轮最终树通过。

## 2026-10-30 — held-out 只评估一次：test access 计数、预注册 look 预算与 effective verdict（本轮实验设计/统计完整性硬化，no new runs）

零 GPU 的设计/分析硬化；不新增训练、不读取 test split、不修改
`confirmation_frozen*.json` 的字节/种子/端点/alpha/Holm 规则，也不改
`CODE_SEMANTICS`；`.github/` 与所有 README 未触碰。

**Problem.** 冻结锁 `_check_or_write_freeze_lock` 早已把 held-out 访问
写成机器可读账本：`test_reads`、`history`、`superseded`、
`source_tree_drift`。第二次 P5T `preaccess` 会被拒绝，除非同时设置
`V12_REFREEZE=1` 与 `V12_ALLOW_TEST_REREAD=1`；显式重读会自增
`test_reads` 并追加 `reread`。但 `finaltest_stats` 只以
`stage='analysis'` 复用同一把锁做身份核对，**从未读取这些计数器**。因此
一个被显式允许的第二次 test 评估，仍可按 12 个种子的名义
`alpha=0.05` 产出 `confirmed`；访问后 `V12_REFREEZE=1` 改清单/切片只在
控制台打印「前一轮降级为探索性」，最终统计仍可能把它标成确认；分析阶段
的源码漂移也只是警告。§58 给探索性补种子加了 look history，§96 给训练
失败/重跑加了 disposition，§98 给 checkpoint 选择加了端点来源，§99 给
报告重建加了证据绑定，但「held-out 访问了几次」本身没有下游消费者。

**Fix.**

- 新增 `exp_lib.TEST_ACCESS_POLICY` / `test_access_policy` /
  `confirmatory_access_audit`。默认 `single_look`（`max_looks=1`）；计划可
  在 `statistics.test_access_policy` 预注册 `bonferroni` 多次 look 预算。
  策略键名白名单、`max_looks` 整数边界、`single_look` 必须 1 次、未知
  method/键/ schema 全部 fail-closed；`_load_manifest` 在首读前校验。
- `confirmatory_access_audit` 校验 `test_reads` 与 `history` 的 `reread`
  数一致、`test_reads>0` 必须绑定 test 切片、空锁/类型错误/负数
  fail-closed；`superseded` 中 prior `test_ids_fp` 非空才判「访问后改
  计划」，prior 为 `null` 是正常的首读前预注册修订。它还与
  `test_stats.json` 的 manifest 哈希、test slice 指纹、alpha 对账，并
  重放已存 `test_access`/`confirmed_effective`，不一致即 fail。
- 多次 look 的 α 分配用**计划** `max_looks`：
  `alpha_effective = alpha/max_looks`，
  `p_holm_effective = min(1, p_holm*max_looks)`；未预注册的重读不给任何
  effective 确认，只给 `p_holm_bonferroni_descriptive` 描述性读数。提前
  停止不退 α，避免「看一次就停」再次变成选择。
- `finaltest_stats` 追加 `stats['test_access']` 与
  `stats['confirmed_effective']`，打印降级警告；**不改** `confirmed`、
  `alpha`、`p_holm` 或任何测量值。`REPORT_v12` 的确认行改用
  `confirmed_effective`，访问不合法时增加降级段并抑制反证段（反证也是
  方向性确认陈述）。
- `audit_gates.py test_access`（别名 `test_access_audit` / `heldout` /
  `test_looks`）扫描 `results_*/.freeze_lock.json` 与
  `analysis_*/test_stats.json`；`audit_gates.py report` 增加顶层
  `test_access`。没有锁/统计工件一律 `not_verifiable_locally`。
- 当前 `confirmation_frozen.json` 哈希与锁不一致时，不把工作树里的策略
  当作锁的策略：回退单次 look 并告警，防止访问后补声明 `max_looks=2`
  把重读洗白。`_check_or_write_freeze_lock` 的 `reread` 事件记录
  `V12_TEST_REREAD_REASON`（缺失为 `None`，审计输出
  `n_rereads_without_reason`）。

**Boundary.** 这不是新的显著性检验：没有新 p 值、区间或 Holm 成员，
也不改写已发生的访问；`fail` 只表示「该访问模式不能支持确认性主张」。
锁是防误改而非防恶意，能重写锁的人也能改 `test_reads`。一次 test 读取
仍可跑多种分析口径，那由 §63 与 §51/§99 负责；本门只数访问，不数分析
变体。Bonferroni 是对同一批最终数据换 checkpoint/代码/口径的保守上界，
不是累积信息序贯设计，Pocock/OBF 的信息分数不适用。首读前的预注册修订
放行；源码漂移只告警（测量树仍由锁绑定，分析树由 source gate 单独判）。
没有随附 release，本门只能约束「你自己的 artifact」，不能替代
`RELEASES.md` 的外部归档。

**Verification.** 新增 `test_review_part_38.py` 16 项：默认策略、顶层与
`statistics` 声明、锁内重命名计划策略；畸形策略 fail-closed；
`descriptive_only` 计划归 `not_applicable`；单次读取 ok 且冻结 p 不变；
未预注册重读降级并给描述性 Bonferroni p；预注册 `bonferroni` 按计划数
分配 α 并筛比较（含提前停止）；超 `max_looks` fail；访问后改计划/切片
fail、首读前修订放行；空锁/类型/负数/零读取带指纹/重读计数不一致/
`history` 非列表 fail；工件 manifest/slice/alpha 不符、零读取却带
confirmed、非法 p、存储块篡改 fail；审计门三态与 CLI/report 接线；
`V12_TEST_REREAD_REASON` 写入 history；`_load_manifest` 首读前拒绝畸形
策略。完整 CPU 套件在本轮最终树通过（868 passed, 3 skipped）。

## 2026-10-30 — 分析的「看」也要入链：look-history attestation 与聚合审计的失败退出（本轮可复现性/工程规范硬化，no new runs）

零 GPU 的可复现性/工程规范硬化；不新增训练、不读取 test split、不修改
`confirmation_frozen*.json` 的字节/种子/端点/alpha/Holm 规则，也不改
`CODE_SEMANTICS`；`.github/` 与所有 README 未触碰。

**Problem.** §58 的 optional-stopping 校正完全依赖
`results_*/look_history.json`，但它是普通可变 JSON：`audit_look_history`
只重算剩余条目的 p 并和 `methodology_audit.json` 对账。删掉较早的一次
look 后，文件仍是一条合法的短序列，`p_sequential` 退化成最终 p，校正被
静默取消；截断/旧文件覆盖也没有交叉证据。工程侧同样有一个「没有失败
状态」的问题：`python audit_gates.py report` 汇总 60 个子审计写进
`audit_report.json`，但顶层没有 `status`，子审计 `fail` 时命令仍退出 0，
`make audit` 不能作为闸门。

**Fix.** (1) `record_panel_look` 为每条 look 计算 `look_sha256`，在
`look_history.json` 写 `attestation`（schema/kind/ledger/n_looks/
truncated_oldest_looks），并向哈希链
`experiments.jsonl` 追加 `panel_look` 事件（look、n_contrasts、
look_sha256、recorded_at）。首次升级旧目录时
`_attest_look_events` 为保留的历史 look 补事件，避免旧面板永久缺证据。
(2) 新增 `exp_lib.verify_look_history_attestation`：legacy（无
attestation/事件）不追溯判 fail；有事件无 attestation、历史 look 无事件、
事件落在保留区间之外、ledger 链/锚点失败判 fail；attestation 在但 ledger
缺失、条目 digest 不符判 warning；合法的 `truncated_oldest_looks` 允许
早期事件没有条目，但多余的未声明早期事件判 fail。`audit_look_history`
并入该块，`audit_gates.py report` 的顶层 `looks` 能看到 attestation。
(3) 新增 `audit_gates.audit_report_status`：任一子审计 `fail` → 顶层
`status='fail'`、`exit_code=1`；有 warning/problems → `'warnings'`；
`not_verifiable_locally`/`not_applicable` 只计数、不升级为 fail。
`run_report` 在落盘前写顶层 `status`/`audit_status`，CLI 的 `report`
与其它子命令一样在真实失败时退出 1。

**Boundary.** 不是新的显著性检验：没有新 p/区间/Holm 成员，不改变
`p_sequential`、`verified`、冻结 `confirmed` 或任何测量。attestation 是
tamper-evident 不是签名：能重写 ledger 并重建链/锚点的人仍能伪造；本门
让意外截断、旧文件覆盖、单行删除和升级后缺事件可见。内容 digest 不符只
判 warning（p 重算通常已能判 fail，digest 是第二层）；顶层状态只汇总，
不代替子审计证据；`panel_look` 是 tracking 事件，不进 `run_cfg`/resume
指纹，不改变数值路径。`make audit` 仍不读 test split、不训练、不需要
GPU；source-only checkout 缺产物时保持 `warnings` 而不是 `fail`。源码
改动后应先 `make plan` 重建 plan snapshot，否则 `source` 门会按设计判
fail（表示代码已变而计划快照未变）。

**Verification.** 新增 `test_review_part_39.py` 10 项：记录后 digest/
attestation 与 ledger 事件一致且重复聚合不重复记事件；截断历史判 fail；
删除 attestation 判 fail；ledger 缺失判 warnings；改写历史 delta 并把 p
改成自洽值时由 digest 判 warnings；篡改 ledger 事件体使链重放失败；旧
look 目录首次升级自动补事件；纯 legacy 历史保持 `legacy`；聚合状态的
ok/warnings/fail/空报告；CLI 在顶层 fail 时退出 1。完整 CPU 套件在本轮
最终树通过（881 passed, 3 skipped）。

## 2026-10-30 — 排序也要有分辨率：固定臂集合的配对种子 bootstrap（本轮结果分析硬化，no new runs）

零 GPU 的结果分析硬化；不新增训练、不读取 test split、不修改
`confirmation_frozen*.json` 的字节/种子/端点/α/Holm 规则，也不改
`CODE_SEMANTICS`；`.github/` 与所有 README 未触碰。

**Problem.** 六个冻结臂只有预注册的**逐对**比较和 Holm 族；报告仍以逐臂
`by_length` 均值表呈现结果，读者会把观测最低均值读成「最优臂」。这是
六选一的选择极值：12 个 seed 上几个 PPL 点的差距可能小于 seed 训练噪声，
而逐对检验显著也不回答「第一名与第二名的顺序是否分辨」。已有读数各自
回答别的问题——单对照 bootstrap 区间、max-T 同时区间（§92）、方差分解
（§89）、口径对账（§63）——没有一个给出 P(best)、排名区间、前二差距、
逐对顺序概率或「删一个 seed 后赢家是否改变」。

**Fix.** 新增 `exp_lib.ARM_RANK_POLICY` / `arm_rank_inference`：对固定臂
集合、按 seed 均值端点排序，用**跨臂共享 seed 下标**的配对 bootstrap
重采样；输出 `p_best`/`p_top2`/`mean_rank`/`rank_ci95`/前二差距及其区间/
`pairwise_p_better`/归一化最优熵/有效候选数/留一种子赢家稳定性。全同臂或
前二均值完全相等判 `undifferentiated` 并共享名次；K=2 判 `pairwise_only`；
共有种子少于 `min_seeds`（默认 5）判 `not_estimable`。`decisive` 还要求
前二差距区间排除零且留一稳定；`tie_tol` 只给 `practical_tie`，与统计可分辨性分开。数值在
共同仿射尺度上计算以防极端有限值溢出；固定 `n_boot=4000`/seed 可重放。
`finaltest_stats` 在唯一端点、主比较臂并集上写入 `stats['arm_ranking']`
（含原始逐 seed 值）；`build_report` 新增排序分辨率表与边界段；
`audit_gates.py arm_ranking` 从块内原始值独立重放并逐字段比对，拒绝夹带
`confirmed`/`p_holm`/`verified` 等确认性字段，并入顶层 `report` 与
`make audit`。

**Boundary.** 不是新的显著性检验：没有新 p 值/区间/Holm 成员，不做 FWER
控制，不改变 exact sign-flip p、Holm、`verified` 或冻结 `confirmed`。
`p_best` 是 bootstrap 频率，不是后验/精确抽样概率；n=12 的百分位区间可能
偏窄。臂集合固定且非随机化，容量/配方/计算量不同，排序不是架构因果排序；
只重采样观测 seed，不重采样评测文档，12 个 seed 不是 seed 总体样本。高
`p_best` 不替代独立复现；留一稳定只排除单 seed 决定赢家这一小类脆弱性。
只在所有主比较共享一个端点时计算，否则明确 `not_estimable`，不混合端点。

**Verification.** 新增 `test_review_part_42.py` 16 项：清晰分离/重叠面板/
`better='max'`/前二差距与 `tie_tol`/全同臂/均值相等但方差不同/seed 键交集
与覆盖/K=2 与种子不足/可重放与换 bootstrap seed/极端有限值/非法输入
fail-closed/审计重放与 `p_best`、rank CI、确认性字段篡改/缺块与畸形块/
bare `not_estimable`/CLI、report、`finaltest_stats`、报告接线/合成三臂
六 seed 端到端落盘审计，以及合成 stats 的 `REPORT_v12` 排序表渲染。
完整 CPU 套件在本轮最终树通过（916 passed, 3 skipped）。

## 2026-10-30 — 排序的另一个抽样轴：评测单元重采样下的固定臂排序（本轮结果分析硬化，no new runs）

零 GPU 的结果分析硬化；不新增训练、不读取 test split、不修改
`confirmation_frozen*.json` 的字节/种子/端点/α/Holm 规则，也不改
`CODE_SEMANTICS`；`.github/` 与所有 README 未触碰。

**Problem.** §102 的排序分辨率只重采样观测到的种子，并在边界里明确写出
「不重采样评测文档」。但报告里的「哪个臂最好」同时条件在 12 个训练种子和
冻结的 48 个评测单元上；若干长文档占掉大部分 token 时，token 池化端点会
把很大权重放在这些文档上，一个在多数短文档上略差的臂可能靠 1 个长文档
赢下观测最优。已有读数各自回答别的问题——cluster/two-level bootstrap 是
均值差的区间，`unit_influence` 是删一个单元对均值差的影响，
`variance_components` 拆的是 seed × unit 的方差——没有一个给出「换一批
评测文档后排序/赢家是否改变」。这正是固定臂排序缺失的第二个抽样轴。

**Fix.** 新增 `exp_lib.ARM_RANK_UNIT_POLICY` /
`arm_rank_unit_sensitivity`：输入逐臂、逐种子、逐单元 NLL/token 矩阵，按
`exp(sum NLL/sum tokens)` 重建与 §102 同口径的逐种子 token 池化 PPL，再对
种子算术均值排序。三条配对轴：seed-only（与 §102 相同，共用 seed 下标）、
unit-only（所有臂/种子共用同一组 unit 下标）、two-way（两轴同时重采样）。
每条轴输出 `p_best`/`p_top2`/`mean_rank`/`rank_ci95`/`rank_sd`/
`p_unique_winner`/`p_best_observed_winner`/`top2_gap_*`；并报告 top-two gap
的 seed/unit/two-way SD 与方差占比、`dominant_axis`。状态带为 `stable` /
`seed_sensitive` / `unit_sensitive` / `both_sensitive` / `undifferentiated` /
`not_estimable`：seed 轴 0.60，unit/two-way 0.90，`lead_level` 0.60；
`stable` 要求观测赢家在三轴都达到 band 且都是该轴 argmax。逐单元矩阵必须
重建 `finaltest_stats` 的逐种子 PPL，否则拒绝（`endpoint_values_match`）；
优先 article 单元，全部记录都有才用，否则退回行级 document 并在报告中
明示；`n_units<64` 时 `population_claim_allowed=false`。固定
`n_boot=2000`/seed `20261018` 可重放；排序在 `logp-max(logp)` 尺度上算，
`1e300` 级 token 也不产生 inf/NaN。

**Boundary.** 不是新的显著性检验：没有新 p 值/区间/Holm 成员，不做 FWER
控制，不改变 exact sign-flip p、Holm、`verified` 或冻结 `confirmed`。
`p_best` 是 bootstrap 频率，不是后验/精确抽样概率；重采样的是观测单元，
不是语言总体样本，48 单元仍低于 64-unit 总体门槛。行级 document 是
WikiText 一行，同一文章多行会伪重复；必须报告 `unit_granularity`，不能把
行级读数冒充文章级。两轴在 two-way 中不独立，gap 方差占比是描述性读数，
不是 ANOVA。臂集合固定且容量/配方/计算量不同，排序不是架构因果排序；
高 `P(best)` 不替代独立复现；`stable` 不是各臂等效。只在冻结主比较共享
一个端点且逐单元矩阵可对齐时计算，否则 `not_estimable`，不混合端点/单元。

**Verification.** 新增 `test_review_part_43.py` 14 项：策略/状态声明；
清晰分离→`stable`、三轴 P(best)≈1、rank CI [1,1]、严格 JSON、同 seed 可
重放；`better='max'` 方向；seed 交集与 `dropped_seeds` 具名报告；单元主导面板→seed-only P(best)≈1 但 unit-only 明显下降、
`unit_sensitive`、gap 分解主导轴为 unit；端点交叉核对拒绝错误估计量并
接受一致矩阵；逐种子 PPL 的**算术均值**排序（几何均值会选错赢家）被
单独锁定；13 类非法载荷 + 7 类非法参数 fail-closed；ragged 单元、token 数不一致、
unit id 顺序/部分缺失、种子/单元不足的拒绝；`1e300` token 级有限值不产生
NaN/inf；审计对合法块 `ok`、篡改 `p_best`/夹带 `confirmed` 判 `fail`、
缺块 `not_verifiable_locally`、bare `not_estimable` `not_applicable`；
`run_report`/CLI 接线；合成六 seed 三臂逐单元面板的端到端落盘 → 审计
`ok`；以及 `REPORT_v12` 渲染「排序的评测单元敏感性」表与边界段。完整
CPU 套件在本轮最终树通过。
