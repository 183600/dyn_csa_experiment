# Follow-up confirmatory study: Dynamic CSA / HCA

## Status and scope

This is a **new, independent protocol**. It does not amend `confirmation_frozen.json`,
does not change the v12 confirmatory estimand, and must use a new output directory
and a separately frozen manifest. Until the manifest is signed/frozen and the
new study is run, all results are exploratory.

The repository has `csa_fixed_rope`, `csa_dynamic_rope`, `hybrid_fixed_rope`,
and the pure-HCA arms `hca_fixed_rope` / `hca_dynamic_rope` (every layer is
`kind='hca'`). Do not label `hybrid_fixed_rope` as a pure HCA arm: it
alternates CSA and HCA layers. `validate_followup_design.py --verify-registry`
imports the variant registry and rejects a hybrid arm declared as HCA, a
fixed/dynamic mismatch, a missing variant, or a Dense control that is not a
`full`-attention arm; `--require-ready` implies that check.

## Required arms

| Factor | Level | Meaning |
|---|---|---|
| Attention family | CSA | compressed-block sparse attention |
| Attention family | HCA | hierarchical compressed attention; pure HCA, not hybrid |
| Segmentation | fixed | fixed block boundaries |
| Segmentation | dynamic | online/causal dynamic boundaries |
| Reference | Dense, parameter matched | match trainable parameter count |
| Reference | Dense, FLOPs matched | match measured training FLOPs |

Required factorial cells:

1. CSA + fixed segmentation
2. CSA + dynamic segmentation
3. HCA + fixed segmentation
4. HCA + dynamic segmentation

Add two distinct Dense reference arms. Parameter matching and compute matching
are different estimands and must never be conflated. A parameter-matched Dense
arm does not establish equal-compute efficiency; a FLOPs-matched Dense arm
need not have equal parameter count.

## Fairness controls

For all primary sparse factorial cells:

- same tokenizer, train/validation corpus fingerprints, train-token order, seed,
  optimizer, learning-rate schedule, batch size, sequence length, context
  position encoding, normalization, and training-token budget;
- same model width/depth and output head unless the registered comparison
  explicitly requires otherwise;
- identical checkpoint selection and validation cadence;
- report trainable parameter count, measured training FLOPs/token, total
  training FLOPs, measured inference FLOPs/token, peak allocated memory,
  throughput, and wall-clock time;
- profile all arms with the same profiler, hardware, software stack, warm-up
  policy, and measurement repetitions;
- for FLOPs matching, tune only predeclared capacity knobs and require the
  measured total training-FLOPs ratio to be within 1% of the target;
- do not use wall-clock time as a substitute for FLOPs matching.

## Mechanism-isolation controls (review 10, issues 2/3)

The factorial sparse cells above answer *system-level* questions: a significant
dynamic-vs-fixed or HCA-vs-CSA difference does not by itself attribute the
effect to the segmentation or attention mechanism. Run these controls before
any mechanism wording:

- **Same-backbone isolation.** A selector-only contrast must hold backbone,
  width/depth, output head, KV projection, normalization (q/kv norm width and
  placement), position encoding, sink policy, attention kernel (manual eager vs
  SDPA), and dtype identical. A single-factor version of the current frozen
  dense comparison is required; the v12 dense arm differs in MLP width, FLOPs,
  kernel, q/kv normalization and sink, so it is a system-level reference only.
- **Random-selection variants.** The current random-indexer control is a FIXED
  deterministic hash pattern (identical for every seed/layer/step) and its
  indexer projections are dead parameters (~1.8% capacity), so it supports only
  "learned selection beats this fixed pattern". Add at least: a per-forward
  re-randomized selector, a distance-matched random selector, a nearest-block
  selector, and a uniform-stride selector; repeat with connection-pattern seeds
  independent of the training seed so training noise and pattern noise can be
  separated.  **These four selectors are now implemented** (review-14):
  `lightning_indexer` modes `random_each` / `random_dist` / `nearest` /
  `stride`, registered as the rope variants `csa_fixed_rope_randidx_each`,
  `csa_fixed_rope_randidx_dist`, `csa_fixed_rope_nearidx`,
  `csa_fixed_rope_strideidx` (registry-only; declare them in this design's
  arms before training).  `random_each` re-randomizes every forward via a
  content-keyed hash of the hidden state (recompute-stable, no RNG stream);
  `random_dist` weights the same hash by a causal distance kernel
  (tau = 512 tokens).  **Connection-pattern seeds independent of the
  training seed are wired now** (review-15): `AttnCfg.pattern_seed`
  (default 0 = review-14 bits exactly; fail-closed on every non-hash mode)
  mixes an independent integer into the row key of `random_each` /
  `random_dist`, and the registry exposes the `..._randidx_each_ps<N>` /
  `..._randidx_dist_ps<N>` twins, so one training seed can be rerun under
  many connection patterns.  Declare the controls (and their pattern seeds)
  in `design.selection_controls`; the validator cross-checks each declared
  seed against the resolved registry cfgs.
- **Reading-scope controls.** A learned-top-k vs all-block contrast changes
  reading scope and the STE soft prior, not only selection; pair it with a
  same-scope learned-vs-random contrast, and report block/read counts.
- **Three budget questions, answered separately.** (i) equal trainable
  parameters (active, after dead parameters), (ii) equal training tokens
  (same optimizer/LR-schedule shape and data order), (iii) equal profiler-
  measured training FLOPs (within 1%). Never describe one answer as another.
- **Kernel/dtype parity** is an acceptance gate (below); a performance
  difference measured across two different kernels is a system property and
  must be labelled as such.

Record both analytical FLOPs estimates and profiler-measured FLOPs. The measured
value is the primary compute-matching gate; analytical estimates are a
cross-check. If the profiler cannot reliably count an operator (e.g. custom
Triton kernels), disclose the unsupported operators and refuse to call the arm
FLOPs-matched until the count is independently reconciled.

**The measured-FLOPs gate tool is wired** (review-16): `followup_flops_gate.py`.
`measure` counts training-step FLOPs with `torch.utils.flop_counter.FlopCounterMode`
(aten-dispatch level — the ONLY counter family that attributes
`scaled_dot_product_attention`; `torch.profiler(with_flops=True)` covers only
mm/bmm/addmm and silently drops the attention kernels, so the profiler pass runs
alongside purely to DISCLOSE what it missed). A second identical block must
reproduce the count exactly (shape-based counting is deterministic; any drift
means the computation changed and the number certifies nothing). The custom CSA
kernel path refuses to measure at all (custom Triton kernels never dispatch
aten ops and would be undercounted). `gate` compares two artifacts with the 1%
tolerance on the total training FLOPs (`flops_per_step x declared_total_steps`);
CPU runs are recorded as smoke evidence and never certify — only runs on the
declared certifying device (default `cuda`) can produce a `FLOPS_MATCHED`
verdict.

## Evaluation population

Use the complete held-out test split where feasible. If a sampled document panel
is required, sample at least 64 distinct documents, stratified by document
length, and freeze the document IDs before model evaluation. The 64-document
floor is a minimum population gate, not a claim that 64 documents are always
sufficient for every effect size.

Do not treat multiple windows from one document as independent documents.
Aggregate token NLL within document, then report:

- pooled NLL/token and PPL;
- per-document NLL/PPL distribution;
- document-level bootstrap confidence intervals (at least 10,000 resamples);
- seed-level paired effects;
- truncation/token coverage and the exact denominator.

The test set must not be used for hyperparameter selection, early stopping,
capacity tuning, threshold selection, or code-path selection.

## Primary hypotheses and statistics

Predeclare four primary tests:

1. Dynamic minus Fixed within CSA.
2. Dynamic minus Fixed within HCA.
3. HCA minus CSA under Fixed segmentation.
4. The segmentation × attention-family interaction.

Use paired comparisons by seed. The same seed must use the same data order and
training recipe across paired arms. Use exact two-sided sign-flip tests,
Holm correction across the four primary hypotheses, and report raw and adjusted
p-values, paired effect sizes, per-seed differences, and confidence intervals.

A 12-seed design gives an exact two-sided sign-flip resolution floor of
`2 / 2^12`; the validator checks that the minimum attainable p-value can clear
the first Holm threshold. This is a mathematical resolution check, not a power
guarantee. Before test access, run a simulation-based power analysis using
plausible paired effect sizes and the observed validation-set variance. Increase
the seed count if the target power is not reached.

**The simulation tool is wired** (review-16): `followup_power_sim.py`
projects per-hypothesis Holm power under a declared effect model (`normal`
mean/sd or a `bootstrap` over the observed validation deltas), using the exact
statistics the confirmatory analysis will use (`exp_lib.exact_signflip_p` +
`exp_lib.holm_adjust`); `--seed-sweep` reports the minimum seed count that
clears the declared target. The artifact JSON binds to the validator:
`statistics.power_simulation_completed = true` now REQUIRES
`statistics.power_simulation_artifact` (a path to the tool's output) whose
study id, hypothesis family, seed panel, alpha, target power and minimum
simulated power all match the plan — a bare boolean no longer passes
`--require-ready`. The declared effect sizes / variance inputs are recorded in
the artifact's `effect_model` block so the projection is auditable.

**Stopping rule.** This is a single-look, fixed-n study: exactly one analysis
after all registered seeds complete (`stopping_rule =
"single_look_fixed_n"`, `n_looks = 1`, `interim_analysis_planned = false`).
No interim look, optional stopping, or sample-size re-estimation on the test
split is registered. Adding looks or changing n after seeing test data would
require a new pre-registration with its own alpha-spending plan; it cannot be
folded into this one. The validator refuses a `--require-ready` plan without
these fields.

For document uncertainty, resample documents (not windows) within each seed.
Report seed-level and document-level uncertainty separately; neither substitutes
for the other.

## Algorithmic acceptance gates before training

Before any confirmatory training:

- dynamic segmentation prefix invariance: extending a sequence must not change
  cuts for an already observed prefix;
- future-token perturbation: changing future tokens must not change earlier
  segmentation, attention outputs, or gradients;
- all-block equivalence: when the sparse selector includes every readable block,
  sparse attention must match the independent dense/reference implementation
  within declared dtype-specific tolerances;
- forward and backward comparisons for FP32 and BF16, with explicit tolerance
  tables and finite-gradient assertions;
- no duplicate selected block contributes twice to the normalization;
- block/window overlap semantics are explicitly declared and tested;
- HCA pure-arm tests establish that the model actually uses HCA in every
  intended layer (not a hybrid alternation);
- test both eager and Triton/reference kernel paths where supported.

These are correctness tests, not evidence that one method is better.

## Freeze and reporting rules

1. Implement missing pure-HCA variants and tests first.
2. Run CPU correctness gates and GPU/Triton parity gates.
3. Tune capacity only on training/validation data.
4. Complete simulation-based power analysis.
5. Freeze a new manifest, source commit, variant registry, seeds, document IDs,
   metrics, hypotheses, correction family, and FLOPs measurement protocol.
6. Only then access the held-out test split.
7. Write outputs to a new directory, e.g. `results_followup_confirmatory/`.
8. Never rewrite v12 results or silently revise the frozen v12 manifest.
9. A failed or incomplete arm remains missing; do not silently shrink the
   paired sample size or synthesize measurements.
10. Claims about Dynamic CSA, HCA, or equal-compute superiority are permitted
    only if the corresponding preregistered comparison passes its gates.

## Running the design validator

```bash
python validate_followup_design.py followup_design.template.json --verify-registry
python validate_followup_design.py followup_design.json --require-ready --repo .
python followup_power_sim.py power_sim_spec.json --seed-sweep 12,16,20,24,32
python followup_flops_gate.py measure --variant full_rope_flops_matched \
    --mlp-ratio 2.69140625 --device cuda:0 --out measure_dense.json
python followup_flops_gate.py gate measure_csa.json measure_dense.json
```

The template is intentionally a draft.  The pure-HCA entries are real
variants, the four selection controls and their pattern-seeded twins are
registered, and the FLOPs-matched Dense arm now has a registry entry plus an
ANALYTICAL capacity plan (`full_rope_flops_matched`; the run cfg routes it
through `exp_lib.flops_matched_mlp_ratio`, whose frozen-config solution is
mr = 689/256 = 2.69140625).  The remaining blockers are the
profiler-MEASURED FLOPs verification (now runnable via
`followup_flops_gate.py` on the certifying device; the analytical plan is
never the certificate), the pre-test power simulation (now runnable via
`followup_power_sim.py`, but it must be run on real validation-derived
effect/variance inputs and its artifact bound into
`statistics.power_simulation_artifact`), the algorithmic gates, and the
freeze metadata.  It must therefore still fail `--require-ready` until those
are completed and recorded.
