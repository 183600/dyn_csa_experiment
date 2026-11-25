# Dynamic-Block CSA + HCA — WikiText-103 controlled study

**English** | [中文说明](README_zh.md)


Controlled study of compressed-block sparse attention (CSA / HCA, fixed and
dynamic blocking) vs dense attention in a small from-scratch LM (BPE-8k,
WikiText-103). This repository contains the experiment code and the
protocols needed to RE-RUN the experiments on a machine of your own. It does
**not** contain the numeric results: no evidence release is bundled (see
`RELEASES.md`), so no PPL number in the generated reports is independently
auditable from this Git tree alone. Claims are limited to the frozen
scope in `confirmation_frozen.json`; everything else is exploratory.

> **Version naming.** `v12` is the driver/study generation; `CODE_SEMANTICS =`
> `v11.128` is the library measurement-semantics stamp that the frozen plan
> binds. They are not the same numbering scheme, and the mismatch is intentional.
> The frozen manifest is at `manifest_version: 5` (see its `revision.history`; v4 removed the uncheckable ee/es pins and documented the tokenizer token cap, v5 corrected the v4 note's description of the implemented decision rule back to the registered v3 wording — no normative field changed; and v6 (2026-11-24) registered the conservative evidence-completeness and minimum-power gates described below — all still strictly before any test-split access);
> per the manifest's v3 `ci_loop` note the continuous-loop CI is supposed to stay
disabled until the frozen commit is signed and P5T has run — check the workflow's
> actual state (`gh workflow list`) before relying on this.

> **Evidence levels.** v6-v11 are exploratory (selected on the same validation
split). Only the frozen v12 fixed-chunk comparisons are confirmatory. See
`DECISION_LOG.md` and `docs/design_notes.md` §17/§20.

> **Evidence status (read this before quoting a number).** `results_*/`,
> `analysis_*/` and `REPORT*.md` are intentionally not tracked, and **no
> formal evidence release is bundled** (`RELEASES.md` has no row yet). The
> generated reports are reproducible from artifacts you produce yourself, but
> the numbers in any report you have not regenerated are not independently
> auditable. A claim becomes auditable only when its `release_<tag>.tar.gz`
> plus manifest is published and `archive_release.py verify` passes.

## Review-10 hardening (what changed)

The review-10 response is implementation/reporting only: no new training panel,
no frozen-manifest byte change, no measurement-semantics change.  The
item-by-item disposition is in `docs/design_notes.md` §28.  Highlights:

- **Evidence status is printed, not implied.** Every report builder now starts
  with an `evidence_status_banner()` line from the `RELEASES.md` index: while no
  complete release row exists (the current state), the banner says the numbers
  are **not independently auditable**.  A release row must contain tag / date /
  commit / archive URL / archive SHA-256 / manifest SHA-256 / verifier; audit
  functions and freeze locks are not a substitute for published artifacts.
- **System-level scope, not mechanism attribution.** `REPORT_v12` prints the
  frozen manifest's `interpretation_caveat` and `compute_budget_disclosure`
  next to the primary table; every frozen comparison carries
  `single_factor_contrast: false` / `system_level_only: true`.  CONFIRMED means
  "system-level arm difference under this frozen recipe", never "the learned
  selector causes the advantage".
- **Random control is a fixed deterministic hash** (`(row*12.9898 + col*78.233)`,
  identical for every seed/layer/step) — not a per-forward random draw, and not
  a general random-sparse baseline.  Its indexer projections are dead
  parameters (202,752, ≈1.83%); `param_breakdown` now reports
  `dead_params`/`active = total − dead`, and the report prints per-arm active
  parameters.
- **Re-freeze cannot create a new confirmation.** If the freeze lock shows a
  post-test `V12_REFREEZE=1` (`superseded`), the current round is forced to
  `confirmatory_round=false`, `confirmed=[]`, with a report warning: the same
  test slice has already been observed.
- **Fix-recipe boundary.** The frozen confirmation is a frozen-default-
  hyperparameter comparison; it is **not** a both-sides-tuned best-method
  comparison (`BASELINE_TUNING_PROTOCOL.note`).
- **NIAH task-validity gate.** NIAH-natural now records the chance level, runs a
  `nofar` no-far-information control, stores distance-bucketed accuracy, and
  requires a recorded positive-control solver before it can support mechanism
  wording.  No such positive control exists in the current panels, so NIAH stays
  descriptive-only.
- **Corruption audit.** `rho` is nominal: reports print realised coverage /
  unique positions / overlap / actually-changed tokens / no-op rows per mode
  (spans are disjoint by construction; donor collisions can make changed <
  covered), so modes are not compared as equal-strength corruption.
- **Statistics.** Exact sign-flip is a test of a SHARP sign-symmetry null;
  sharing seed/data order/init is a control, not randomization.  Sign counts,
  leave-one-seed-out stability and within-seed-noise are reported separately
  from significance, and the power artifact now includes q_hat, a Wilson lower
  bound, leave-one-seed-out and outlier sensitivity.  A `V12_SKIP_POWER=1`
  access additionally requires `V12_ACCEPT_LOW_POWER=1` when the manifest
  declares a power gate.
- **Units and length scope.** The frozen 16 evaluation units are fixed
  windows/rows, not documents; `48` is the arm×seed record count (4×12), not 48
  documents; the 64-document floor is a house population gate with no
  statistical derivation.  PPL@4096 scores nested-prefix targets, so it is not
  a same-target context-length comparison.

## Review-11 hardening (what changed)

The review-11 response is verification-first: no new training panel, no frozen
manifest byte change, no test access.  The item-by-item disposition (including
the claims that are factually stale at HEAD) is `docs/design_notes.md` §29.
Real code fixes in this pass:

- **Paired-panel bug fixed.** The report grouped seeds for a sign-flip test
  using the per-seed `init_core_fp`, which split every correct panel into one
  group per seed and returned n=1.  Initialization is now checked only within
  a seed pair; a 5-seed panel again yields n=5.
- **Stale records fail closed.** `sec_stats`, `sec_flops`, the health/efficiency
  tables, per-seed pairing and the v8–v11 panel readers now reject records whose
  code stamp is not current (previously several paths still rendered old values).
- **Multiplicity made visible in the standalone reports.** v10/v11 publish the
  family resolution floor and a cross-panel BH-FDR + Bonferroni view; v10's
  ΔNLL/NIAH tables now print Holm p-values instead of raw p-values under a
  "Holm" heading.
- **Mechanism wording narrowed.** v10/v11/fuse no longer call a random/all-block
  contrast an "independent contribution" or a "mechanism conclusion"; they state
  the attribution boundary and the missing selection-attribution artifact.
- **Stopping rule registered.** The follow-up protocol/validator now require a
  single fixed-n look; interim looks or sample-size changes need a new
  pre-registration.
- **Report hygiene.** v9/open-block/fuse reports print the evidence-status
  banner; the "documents" mislabel on `eval_subset_rows` is fixed; the
  efficiency section states that energy is not measured.

## Review-12 hardening (what changed)

Verification-first pass over a 16-item external methodological review.  No new
training panel, no test-split access, no frozen-manifest byte change, no
measurement-semantics change (`CODE_SEMANTICS` stays `v11.128`).  The
item-by-item disposition (including claims that are factually stale at HEAD)
is `docs/design_notes.md` §30.  The review's design-boundary points are
accepted and remain limitations; this pass fixes the code/report defects that
were still reproducible:

- **Training-objective descriptor was incomplete.** `train_objective` ignored
  the per-layer `comp_lambda_mult`, so `csa_dyn_nocomp` (mult=0) was recorded
  as `comp_reg_applies=True` with a 0.05 effective weight, and hybrid HCA
  layers' 4x weight was invisible.  It now records the per-layer effective
  weights, `objective_reason` compares the vector, and v7/v9 paired contrasts
  carry `objective_matched` / `objective_differences` and print the
  system-level boundary.  Dense-vs-sparse pairs are still paired (that is the
  comparison), but they can no longer be silently read as mechanism-clean.
- **Sign-test robustness reaches the artifacts.** `aggregate`, the v7 stats
  artifact and v9's `add_paired` now serialize sign counts, the p-resolution
  floor, paired and unpaired bootstrap CIs, Wilcoxon p and the explicit
  sign-symmetry/seed-level assumption.  The v7 stats table prints them next to
  `p` / `p_Holm`, and its Holm family cannot shrink when a declared comparison
  is absent (missing slots enter as p=1).
- **Length-extrapolation reporting.** The v7 length section and v10's P3MT now
  use cross-seed **token-pooled PPL** (`exp(ΣNLL/Σtokens)`) whenever the
  records carry NLL/token totals, fall back explicitly (`*`) otherwise, print
  the seed count and ratio basis, and replace “mechanism-consistent”
  conclusions with descriptive system-level readings.  The section is labelled
  as a **512-training-length extrapolation probe**, not long-context training.
  P3MP probe cells likewise publish pooled PPL/median/IQR.
- **Multiplicity coverage completed.** v10's global BH/Bonferroni family now
  includes the scale-endpoint and equal-token p-values it displays (previously
  m excluded them, making the printed q-values anti-conservative); P3T gained
  `p_Holm` and the resolution floor; v9 gained its own global BH/Bonferroni
  view rather than only within-family Holm.
- **Resource disclosure per arm.** A new report section prints, for each
  recorded arm, total / attention / indexer / MLP / active / dead parameters,
  layer count, structural KV elements per token, the actual compression
  weights and the delta-LR multiplier, and states that these are structure-only
  counts — parameter matching is not compute or active-capacity matching, and
  per-arm FLOPs are not measured.
- **Old “falsified” wording removed.** The P0-3 RoPE block no longer says
  “P0-3 的证伪” and the boundary-F1 summary no longer calls the cuts
  “genuinely semantic”; both are exploratory directional readings.
- **Evidence auditability.** `evidence_release_status` now schema-checks a
  release row (ISO date, 7–40 hex commit, http(s)/s3/gs URL, two 64-hex
  digests, non-placeholder verifier) instead of accepting any seven
  non-placeholder cells; a junk row can no longer flip every report banner to
  `released`.  `archive_release.py` now packs the TimeGuard
  `run_time_state*.json(.k)` ledger the docstring promises, refuses an
  analysis-only bundle (at least one `summary.json`/`run_manifest.json` is
  required), and records the full GPU/driver/TF32/pip-freeze environment.
  The v7 report ends with an input manifest (per-file SHA-256 + git HEAD) and
  writes `analysis_v7/report_inputs.json`, so that report is bound to the exact
  results/analysis batch that produced it; the v10/v11/v12 builders do not yet
  emit this manifest.
- **Crossover scope.** The v10 fit text now states that d is bound to depth
  (128/4L … 512/10L) and that batch/seq/eval grids differ, so the slope is not
  a single-scale-variable estimate; “pre-registered” was corrected to an
  exploratory pre-declared criterion.

Still acknowledged as design limitations (not solved by code): small WikiText
models only, parameter- but not FLOPs-matched dense controls, dense/CSA q/k
normalization and attention-scale differences (CSA: L2-normalize→RMSNorm,
scale 1.0; dense: RMSNorm only, scale 1/sqrt(d_head)), no true
same-projection/same-sink dense control, short training budgets, 2–6 seed
exploratory panels, the 16-window frozen panel, hard top-k selection with no
attention gradient on unselected blocks, no oracle/selection recall artifact on
the real panel (the `mech_selection.py` probe exists and passes its audit
contract on smoke, but the artifact itself still requires a GPU run over the
trained checkpoints),
partial dynamic-block per-sample geometry, no censored crossover regression,
source-only (no released evidence; `archive_release.py record` exists but no
archive is published), unlocked CUDA environment, and no energy measurement.

## Review-14 hardening (what changed)

Verification-first pass over a six-category external methodology review (no
new training panel, no test-split access, no frozen-manifest byte change, no
measurement-semantics change — `CODE_SEMANTICS` stays `v11.128`, and
`v12_supp.py plan` still admits 48/48 records).  The item-by-item disposition is
`docs/design_notes.md` §31.  The review's design-boundary findings (dynamic
chunking not task-learned, no Dynamic/HCA arm in the freeze, param-vs-compute
dense confound, 27.5% power, 16-window panel) all hold and were already
disclosed; this pass closes the two gaps that were actually fixable:

- **The four protocol-required random-selection controls exist now.**
  `lightning_indexer` gains `random_each` (per-forward re-randomized,
  content-keyed and recompute-stable — no RNG stream), `random_dist`
  (distance-matched random, causal decay tau=512 tokens), `nearest`
  (recency) and `stride` (uniform spacing) modes, registered as the rope
  variants `csa_fixed_rope_randidx_each` / `csa_fixed_rope_randidx_dist` /
  `csa_fixed_rope_nearidx` / `csa_fixed_rope_strideidx` (registry-only;
  default panels train none of them).  The frozen fixed-hash `random` path
  is bit-identical to before (N110 asserts it).
- **The selection-attribution artifact now has a producer.**
  `mech_selection.py` is the mechanism probe behind
  `audit_gates.py selection`: per-seed recall@k / mass covered / k-n_blocks
  against a same-layer all-block attention-mass oracle, plus descriptive
  PPL readouts (learned vs oracle-forced vs allblocks vs fixed-hash random
  at the same k).  `python mech_selection.py smoke` builds an artifact on a
  tiny synthetic model and passes the audit contract end to end; `run`
  probes real checkpoints on the VALIDATION stream or synthetic ids and
  refuses anything that touches `results_lm_test`.  Until it is run on the
  real panel, the selection-recall claim gate stays `not_verifiable_locally`.
- **A real reporting defect fixed.**  `_uses_batched_eager` tested
  `indexer_mode` for truthiness while the dispatch requires `== 'learned'`,
  so random-indexer layers were misreported as taking the batched twin; the
  eligibility check now mirrors the dispatch.  Dead-parameter accounting
  (`param_breakdown`) covers every non-learned mode, not just `random`.

## Repository layout

The repository keeps only **source code, driver scripts and design
documents**. All experiment result directories (`results_*/`), zero-GPU
analysis directories (`analysis_*/`), per-version reports (`REPORT*.md`) and
run-state files are **not tracked** — the driver scripts below regenerate them
on whatever machine runs the experiments, so you can simply reproduce them
locally whenever needed (see the commands in the next section). `.gitignore`
already keeps those path families out of the repository (`docs/` retains only
`design_notes.md`; the former `compute_plan.md` was retired).

| path | what |
|---|---|
| `exp_lib.py` | **the library — and the source of truth**: model, CSA/HCA attention, pooling, indexer (learned + the five selection-control modes), segmenters, training loop, aggregation and statistics |
| `mech_selection.py` | **selection-attribution mechanism probe** (review-14): the producer of `analysis_mechanism/selection_attribution.json` — learned-vs-oracle recall@k / mass covered / random baseline per seed, plus learned / oracle-forced / allblocks / fixed-hash-random PPL readouts; `smoke` self-checks the `audit_gates.py selection` contract, `run` probes real checkpoints (validation stream or synthetic ids only; the frozen test split is refused) |
| `csa_kernels.py` | **the fused-kernel path for the three core operators** (the retired compute-plan M0 work package): block-compression pooling, indexer top-k selection and sparse attention as Triton kernels (plus a CPU-runnable torch-reference twin of the same algorithmic decomposition); opt-in via `CSA_KERNEL`, off by default — see "Triton kernel path (M0/G0)" below |
| `run_gapfill.py` | **the v6 driver**: the minimum-publishable supplementary runs, resumable, guarded by a run-time cap, pushes after every phase, auto-shuts the instance down |
| `par.py` | **task-level parallel executor (opt-in)**: one GPU card per independent run — see "Task-level parallelism" below |
| `v7_supp.py` | **the v7 driver**: reviewer-response supplementary suite — P0R RoPE+QK-norm, P0W dense-warmup, P0E 40k-step long run, P2S scale re-seeding, P1L length extrapolation, P1T topk sweep |
| `v8_supp.py` | **the v8 driver**: P1S (scale panel seed completion to 4 seeds) and P2R (RoPE+QK-norm 20k-step long run) |
| `v9_supp.py` | **the v9 driver**: P2T (the seq-2048 topk sweep at `batch_size=1`) and P2S3 (third seed for the RoPE 20k run) |
| `v10_supp.py` | **the v10 driver**: P3M length-extrapolation mechanism + distractor-injection probe, P3S inversion-crossover scaling, P3T topk sweep to 4 seeds |
| `v11_supp.py` | **the v11 driver**: statistical close-out — distractor probe to n=6 (P4MT/P4MP), crossover panels to n=4 (P4S), d=384 paired arms to n=4 (P4F) |
| `v12_supp.py` | **the v12 confirmation driver (reviewer-response + review-2 hardening)**: P5C trains all four frozen arms at the manifest seed list, P5MP runs the extended corruption-mode probe (uniform/realistic-token/mask/delete + review-2 span/doc/topical + NIAH-natural), P5S2/P5S3 add the token-matched and 6ND-param-proxy scale arms (exploratory; not in the frozen primary family), P5T performs the FROZEN final evaluation on the held-out WikiText-103 test split per `confirmation_frozen.json` — with a load-time plan-power gate (a plan that cannot mathematically confirm is refused), strict record admission (seed whitelist / unique records / one frozen test slice / recipe signature / manifest sha), a first-access freeze lock, and per-row PPL records (the **evaluation window is the unit**; true document clusters via `load_wikitest(with_doc_ids=True)` + `eval_ppl(doc_index=...)` are a follow-up, not what P5T currently does) |
| `open_block_supp.py` | **open-block panel driver**: maximum block length (32/64/128/no cap) × current-block reading rule (`closed` / `prefix_summary` / `prefix_singletons`), with a shared top-k budget, causal prefix construction, outside-window auxiliary panel, paired per-seed deltas and interaction statistics — see `docs/design_notes.md` §24 |
| `test_open_block.py` | **open-block CPU gates**: config/segmenter, optimized-vs-reference prefix pooling (forward + gradients), prefix invariance, future perturbation/gradient, candidate masks, edge cases, closed regression, singleton semantics and HCA budget declaration |
| `confirmation_frozen.json` | **pre-registered confirmation plan, manifest v5 (revision history v1–v6)** (tracked): frozen arms, seeds 0-11 (revised 6->12 BEFORE any test-split access; power table recorded), endpoints, exact sign-flip + Holm, decision rule, machine-checkable binding block and a scope declaration — see `docs/design_notes.md` §17 and `DECISION_LOG.md` |
| `RELEASES.md` | **evidence release index**: where each immutable `release_<tag>.tar.gz` + SHA-256 manifest is published; the Git repo itself stays source-only |
| `pyproject.toml` / `requirements.txt` | Python ≥3.12 and dependency declarations; install explicitly (no import-time pip install) |
| `DECISION_LOG.md` | **decision log** (tracked; reconstructed 2026-10-18 from the README/design notes/manifest because the snapshot has one squashed commit): why each version ran what it ran, what was seen, and what was decided next — the exploration/confirmation paper trail; formal releases additionally archive per-seed results via `archive_release.py` |
| `audit_gates.py` | **review-2/4 verification entry points** (zero-experiment): data-leakage audits (tokenizer provenance, vocab identity, train/test overlap), CPU causality unit checks (prefix invariance, label shift, decode consistency), block-attribution / boundary-vs-random / length-consistency / baseline-fairness / failure-concentration / budget-reconciliation audits and prefill-decode-kv-growth measurement utilities |
| `archive_release.py` | **immutable evidence archiver**: packs per-release artifacts (results, analyses, reports, freeze lock, decision log, environment, checkpoint hashes, git diff/status hashes) into a tar.gz with a per-file SHA-256 manifest, an independent `verify` subcommand, and a `record` subcommand that binds a verified, uploaded archive URL + both digests into `RELEASES.md` — the git repo stays source-only (design_notes §8) while every formal release gets an auditable fixed archive |
| `m0_supp.py` | **the M0/S1 driver (adopted B tier)**: entry points for the scale-up roadmap — `params` (zero-GPU S1–S4 parameter audit), `data` (FineWeb-Edu + 32k-BPE corpus cache), `smoke` (end-to-end self-check on the B-tier data path), `kernel` (the G0 validation suite: kernel/eager numerical comparison for the three operators plus the sparse-attention throughput benchmark), `full` (the S1 panel); see "B-tier M0/S1" below |
| `build_report.py` | zero-GPU report builder: regenerates the v7 report from `results_*/` + `analysis_*/` artifacts (every number is computed from disk; protocol constants and analytic FLOPs assumptions are explicit inputs, not measurements) |
| `fuse_analysis.py` | zero-GPU fuse-vs-no-fuse driver: recomputes per-layer boundary alignment (tol=1), block-length stats and PPL trajectories from a `results_*/summary.json` |
| `docs/design_notes.md` | **design and implementation notes**: why the block-read gate is strict and must be synced across five sites, the dynamic segmenter's prefix-invariance requirement, the pairing/measurability rules, the reporting conventions, and the contracts that keep attention row-chunking equivalent to its unbatched form |

> **`exp_lib.py` is the single source of truth — edit it directly.**

### Regenerating the experiment artifacts

Result directories and reports are not kept in the repository. To obtain them
again:

```bash
python run_gapfill.py smoke      # end-to-end self-check, zero GPU cost
python run_gapfill.py full       # the v6 main experiment (bounded by the run-time cap)
python v7_supp.py ...            # per-version supplementary runs; subcommands in each script's docstring
python build_report.py           # rebuild the report from on-disk results_*/ + analysis_*/
```

## Reproducing (any CUDA box)

```bash
# 0) environment — Python 3.12 (see pyproject.toml / requirements.txt):
#    For the verified CPU test/reproduction environment use the lock (includes torch):
pip install -r requirements.lock
#    On a CUDA box install the matching torch wheel first, then the ranges:
#      python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch
#      pip install -r requirements.txt
#    Regenerate and archive a machine-specific lock with `python -m pip freeze > requirements.lock`;
#    requirements.lock records the verified CPU environment, not the paper-scale CUDA one.
#    Torch note: the lock pins torch==2.14.1+cpu.  The scatter paths cast their
#    int32 indices to int64, so they no longer rely on torch >= 2.14 scatter
#    behaviour; the supported range remains torch>=2.8,<3 (requirements.txt).
# The code no longer pip-installs at import time.  Set CSA_AUTO_INSTALL=1 only
# for a throwaway environment.  The Hugging Face mirror is opt-in via
# CSA_HF_MIRROR=1; the default endpoint is the official Hub.
#
# If the Hub is unreachable, `load_wikitext` falls back to a local raw dump:
# point `WT103_RAW_TXT` at a directory holding `wiki.train.raw` /
# `wiki.valid.raw` (the search also tries `./wt103_raw` next to the repo and
# `/root/wt103_raw`).  The frozen P5T confirmation additionally needs
# `wiki.test.raw` in the same directory: `load_wikitest` reads it as the
# held-out split and checks test documents against both train and validation.
# Tokenising the raw files reproduces the `wt103_cache/*.npy` cache
# token-for-token, so the fallback is a pure availability path and never
# changes a measurement.
# When `wt103_cache/` is already populated the corpus is served from the cache
# and no dump or network access is needed at all.  Set `WT103_CACHE_DIR` to
# relocate the cache; the loaders, `audit_leakage` and P5T all resolve the same
# variable.  `v12_supp.py smoke` deliberately uses an isolated cache under its
# smoke outdir so it can never leave a differently-tokenised file in the frozen
# cache.  Test-cache hits verify the persisted tokenizer bytes (sha256 recorded
# in test_meta) in addition to the train fit fingerprint.

# 1) validate the whole pipeline end to end (smoke run, no push):
#    (covers dense / fixed / dynamic arms AND the leave-one-out ablation
#     forward paths: random indexer, zero content, sink off, learned fusion,
#     HCA layers, dense+sink control)
python run_gapfill.py smoke
python v10_supp.py smoke           # also covers the new corruption modes + NIAH + controls
python test_open_block.py         # open-block causality/reference/mask gates (CPU)
python open_block_supp.py smoke    # open-block matrix plumbing on a tiny synthetic run
python open_block_supp.py main     # 8 cells x 4 seeds; resumable
python open_block_supp.py report   # paired deltas + interaction + exact sign-flip/Holm
python open_block_supp.py extended # contingency: all 4 caps x all 3 modes
python v12_supp.py smoke           # also covers the frozen final-test roundtrip

# 2) the real supplementary study (bounded by the run-time cap):
nohup python run_gapfill.py full > run_v6.log 2>&1 &
tail -f run_v6.log
```

`run_gapfill.py full` runs, in order (each phase is resumable; re-running the
same command skips completed `variant::seed` pairs).  The cheap, statistically
foundational panels run first so a run-time-capped box secures them before the
20k-step long runs; every phase schedules at least the 3-seed floor the
methodology audit requires (`min_n`), and seed-completion passes (P2S / P3T /
P4SS / P5C …) top individual panels up to n=4–6:

| phase | output dir | what |
|---|---|---|
| P-CORE | `results_lm_v3_1500` | 9-variant core table, 1500 steps x 3 seeds |
| P-ABL | `results_lm_v4_abl` | leave-one-out ablation table — in-panel references (`full` / `csa_fixed` / `csa_dynamic` / `hybrid_fixed` / `hybrid_dynamic`) + `csa_fix_*` / `csa_dyn_*` module-off arms, fuse/HCA/topk arms, 1500 steps x 3 seeds |
| P-SCALE | `results_lm_v5_scale` | d=384 / 8 layers / seq-1024 probe, 5 variants x 3 seeds x 4000 steps |
| P-LONG | `results_lm_v3_long` | hybrid_fixed / hybrid_dynamic / hybrid_csa_dyn at seeds 0-2 → 3-seed long-run table |

`full` is bounded by the run-time cap described at the end of this section.

### Task-level parallelism (one GPU per run)

Every driver can run its independent cells — one `(variant, seed)` training
run each — in parallel, **one card per run**, on multi-GPU boxes.  It is
strictly opt-in and off by default, so the serial behaviour above is exactly
what you get when the variable is unset:

```bash
# one worker per visible GPU (recommended):
CSA_PAR=auto nohup python v11_supp.py full > run_v11.log 2>&1 &

# only 2 cards, only these cards:
CSA_PAR=2 CSA_PAR_GPUS=0,3 python v10_supp.py phase P3SL

# back to 100% serial (the default): unset CSA_PAR
```

What it guarantees (see `docs/design_notes.md` §13 for the full contracts):

- **identical numbers** — each cell seeds its own RNG in its own process, so
  the parallel pass is designed to reproduce the serial pass; the bit-for-bit
  equivalence check is a local scripts-side test, not a CI-enforced guarantee;
  fingerprints,
  resume rules and `CODE_SEMANTICS` are unchanged;
- **merge-safe artifacts** — all `summary.json` updates go through an
  flock-guarded load-modify-write, so concurrently finishing runs never lose
  each other's records; aggregation / plots / git push / shutdown happen once,
  in the parent;
- **honest accounting under the run-time cap** — a child pre-books its
  projected time into the shared `TimeGuard` state and refunds the difference
  after the run, so concurrently launched cells still respect the cap, and the
  cap keeps metering **GPU-seconds** (4 cards x 1 h = 4 h against the cap);
- **same failure semantics as serial** — a failed cell writes its error
  record and the phase continues; a crashed child aborts the phase like a
  serial exception would, completed cells stay on disk and re-running
  resumes.

Per-cell logs land under `par_jobs/<phase>_<timestamp>/` (gitignored); the
main log reports start/done/heartbeat lines and tails a failed child's log.
Smoke modes always run serially regardless of `CSA_PAR`.

### B-tier M0 / S1 (scale-up entry point, adopted tier B)

The adopted B-tier scale-up plan (entry point `m0_supp.py`; the former
`docs/compute_plan.md` is retired) adds a second
corpus path and the S1–S4 scale ladder on top of the WikiText-103 mechanism
layer; `m0_supp.py` is the entry point:

```bash
python m0_supp.py params   # zero GPU: S1–S4 parameter audit vs the plan
python m0_supp.py data     # build / verify the FineWeb-Edu cache (M0_TOKENS=…)
python m0_supp.py smoke    # end-to-end self-check on the B-tier data path
python m0_supp.py kernel   # G0 suite: kernel/eager numerical comparison + throughput
python m0_supp.py full     # the S1 panel (5 arms × 4 seeds; CSA_PAR=auto works)
M0_PANEL=S2 python m0_supp.py full   # the S2 panel (5 arms × 3 seeds × 12B; M2_CAP_HOURS override)
```

- Corpus: FineWeb-Edu (default `sample-10BT` config), self-trained 32k BPE;
  npy cache + sha256 content fingerprint, same discipline as `wt103_cache/`.
  Token ids are cached at 2 bytes/token and served int32 — every batch widens
  back to int64, so nothing downstream changes.
- Offline / air-gapped machines: point `FWEDU_LOCAL_JSONL` at local jsonl/txt
  documents (pure availability fallback — different corpus content yields a
  different fingerprint, never a confusable cache).
- `full` is gated by `M0_CAP` (state file `run_time_state_m0.json`,
  `M0_CAP_HOURS` overrides): before the Triton kernels pass the G0 gate the
  eager throughput projections are refused — that is the plan's own
  precondition of that plan, not a defect.

### Triton kernel path (M0/G0)

`csa_kernels.py` implements the three core operators of the scale-up plan as
fused kernels — block-compression pooling (per-channel masked softmax over
[A-lanes | overlap lanes] with position embeddings), indexer scoring + top-k
selection (first-max extraction per round, which is exactly the order of
torch's stable descending sort, tie = earliest index), and sparse attention
(flash-style online softmax over [top-k block | sliding-window] lanes with a
sink as the initial normalizer mass). Every operator also has a torch
reference twin of the same algorithmic decomposition that runs on any device
and is what the CPU-side semantic validation exercises.

The kernel path is an **independent variant**, not a bitwise rewrite: the G0
gate is statistical equivalence against eager (same seed, same data), while
the eager path itself is untouched — with `CSA_KERNEL` unset the run-config
fingerprint is byte-identical to the historical one, and with it on the
fingerprint and `_code` stamp gain a `-tk1:<backend>` suffix so kernel records
only ever pair with kernel records (design_notes §3.1). Backward passes are
fixed-order reductions throughout — no atomics; cross-destination accumulations
go through stable-sort segment sums. The straight-through soft prior of the
eager path cancels in the forward value and is reconstructed as a dual A/S
Jacobian in the backward, matching the eager autograd graph.

```bash
python m0_supp.py kernel                 # anywhere: ref-backend semantics check
CSA_KERNEL=1 python m0_supp.py kernel    # CUDA + triton: full G0 suite
CSA_KERNEL=1 python m0_supp.py full      # kernel-mode S1 panel (after G0)
```

- `CSA_KERNEL` unset / `0`: eager everywhere (default; nothing changes);
  `1`: require CUDA + triton (hard error if missing); `auto`: triton when
  available, else the torch reference; `ref`: force the reference backend
  (CPU-debuggable, same dispatch and autograd wrappers).
- Per-operator overrides for GPU debugging: `CSA_KERNEL_POOL` / `_IDX` /
  `_ATTN` ∈ {`triton`, `ref`}; `CSA_KERNEL_TF32=0` switches `tl.dot` to
  ieee fp32 for bit-level triage (default tf32 for tensor-core throughput).
- Kernel mode uses its own TimeGuard state file (`*.k`): eager and kernel
  throughput calibrations never contaminate each other.
- Supported layer kinds: `csa` / `hca` with the learned indexer and no fuse
  conv — exactly the B-tier arms (`csa_fixed`, `csa_dynamic`,
  `hybrid_dynamic`). Anything else fails loudly rather than silently mixing
  implementations within a panel.

### v11 statistical close-out suite

```bash
python v11_supp.py smoke
nohup python v11_supp.py full > run_v11.log 2>&1 &
python v11_supp.py report         # rebuild REPORT_v11.md from disk (zero GPU)
```

Phases: P4MT (RoPE-arm seeds 0-5, exact P1L/P3MT recipe; resume skips
checkpoints whose code stamp is still current), P4MP (probe resume for
the same seed set, eval-only, stale cells re-measured), P4SS/P4SL
(crossover panels seeds 0-3, same resume rule), P4F (d=384
`csa_fixed`/`full` seeds 0-3 under one config, so the panel pairs at
n=4).

### v10 mechanism / scaling / stats suite

```bash
# v10: distractor-injection mechanism probe + crossover scaling + topk seeds
python v10_supp.py smoke
nohup python v10_supp.py full > run_v10.log 2>&1 &
python v10_supp.py analysis       # zero-GPU stats -> analysis_v10/
python v10_supp.py report         # rebuild REPORT_v10.md from disk (zero GPU)
```

Phases (each resumable): P3MT (retrain the two RoPE arms, the P1L recipe),
P3MTC (**from-scratch selector controls under the exact same budget**:
`csa_fixed_rope_randidx` / `csa_fixed_rope_allblocks`; the latter is a
same-CSA all-block *selection* control, **not** a dense twin — it retains KV
compression/pooling, cosine q/k with scale 1.0, sink and the window branch),
P3MP (eval-only far-distance distractor
injection: paired corrupted contexts, clean 512-token target, learned-selection
vs dense vs random-indexer vs see-everything arms, **uniform /
realistic-token (natural) / mask / delete + review-2 span / doc / topical
corruption modes** + the NIAH-natural far-necessity task), P3T
(topk sweep seeds 2/3 → n=4), P3SS/P3SL (d=128/4L 8000-step and d=512/10L
4000-step crossover panels), P3SL2/P3SL3 (**token-matched and
6ND-parameter-proxy-budget large arms** — a scale-confound *probe*: 6ND is a
proxy, not measured FLOPs, and only a token-matched slice is actually
matched; see docs/design_notes.md §16).
See `docs/design_notes.md` for the design rationale behind the
paired-corruption protocol, the pre-registered crossover criterion, the
corruption-mode/ΔNLL semantics (§15) and the intervention-vs-trained-control
distinction (§18).

### v12 confirmation suite (exploration / confirmation split)

```bash
python v12_supp.py smoke           # end-to-end self-check (tiny ckpts + a smoke freeze manifest)
python v12_supp.py plan            # zero-GPU pre-flight: stamps + implementation binding + admission + power, NO test read
python v12_supp.py phase P5C       # train all four frozen arms at the manifest seed list (resumable; 0-5 main arms covered by P3MT/P4MT)
python v12_supp.py phase P5MP      # extended probe (all modes + NIAH + controls) at the manifest seed list
python v12_supp.py phase P5S2      # token-matched large scale arm (8000 steps)
python v12_supp.py phase P5S3      # 6ND-param-proxy-budget large arm (param-ratio steps; NOT measured FLOPs)
python v12_supp.py phase P5T       # FROZEN final test on wiki.test.raw (requires the freeze manifest)
python v12_supp.py analysis        # zero-GPU: shared v10 stats + frozen final-test stats
python v12_supp.py report          # rebuild REPORT_v12.md from disk (zero GPU)
```

P5T refuses to run without `confirmation_frozen.json`, refuses a manifest whose
smallest attainable Holm p exceeds alpha (a plan that cannot confirm is a design
defect, not conservatism — the seed count was revised 6 -> 12 before any
test-split access for exactly this reason), refuses records that fail the
strict admission gate (seed whitelist, uniqueness, non-null test-slice
fingerprint, code/recipe match, manifest sha binding), and writes the freeze
lock (`.freeze_lock.json`) **before** the first test-split read, then binds the
test-slice fingerprint immediately after it. Kernel mode is refused for this
eager-only frozen plan. The test split
(`wiki.test.raw`, via `load_wikitest`) is read by P5T **only** — every earlier
phase reads train/valid exclusively. After the first test-split evaluation, no
frozen comparison or seed may be added (selection bias by construction; see
`DECISION_LOG.md`).

**Plan revision v6 (2026-11-24, still before any test access)** closes three
evidence gaps that a source-level review flagged:

- **Minimum-power gate.** `statistics.power_gate` (q=0.80, min_power=0.80)
  turns the pre-access power simulation from a printed warning into a gate:
  every primary comparison's panel-specific power in `p5c_power.json` must be
  recorded and at or above the floor before P5T may read the test split. A
  miss refuses the access unless `V12_ACCEPT_LOW_POWER=1` is set; the decision
  is written to `results_lm_test/power_gate.json` and carried into
  `test_stats.json`. The 12-seed design's own q=0.80 power under Holm/4 is
  0.275, so the gate normally fires: the honest remedies are added seeds (a
  further recorded revision) or an explicit, auditable acceptance of a
  resolution-limited confirmation.
- **Verified parameter match.** A comparison is CONFIRMED only if
  `param_matched is true` (a measured <=1% gap). The v5 wording (`is not
  false`) let an unverifiable count confirm; that is no longer accepted.
- **Per-seed matched initialization.** `init_core_fp` is a CONFIRMED gate:
  both arms of every usable pair must record the same shared-trunk digest.
  The per-comparison check was also fixed to compare arms **within** a seed —
  it previously pooled all seeds into one set, so it reported `False` on every
  correct 12-seed panel and could not detect a real mismatch.

Both tightenings can only withhold a CONFIRMED label, never create one; no
arm, seed, endpoint, alpha or Holm family changed.

**Scope (review-2 issue 4)**: the confirmation covers ONLY fixed-chunk CSA
mechanism claims. Dynamic chunking vs fixed chunking, semantic boundary value,
HCA advantages and dynamic-chunking scaling laws are NOT covered by this
freeze and need their own pre-registration.

**Verification entry points (review-2 issue 11)** — zero-experiment audits:

```bash
python audit_gates.py causality   # CPU unit audit: prefix invariance / label shift / decode consistency
python audit_gates.py leakage     # tokenizer provenance, vocab identity, train/test overlap
python audit_gates.py report      # run every locally-verifiable audit -> audit_report.json
python audit_gates.py selection   # selection-recall contract (refuses unsupported mechanism claims when absent)
python audit_gates.py efficiency_pairs  # measured latency claims: same shape/device/KV required
python audit_gates.py release     # evidence-release readiness (no complete row => source-only)
python archive_release.py plan v13 --dirs results_lm_test analysis_v12   # dry-run an evidence archive
```

### v8 / v9 close-out suites

```bash
# v8: scale-panel seed completion (4 seeds) + RoPE 20k long run
python v8_supp.py smoke
python v8_supp.py full            # or: python v8_supp.py phase P1S / P2R
python v8_supp.py analysis        # zero-GPU stats -> analysis_v8/

# v9: complete topk sweep (bs=1) + RoPE-20k third seed + report
python v9_supp.py smoke
nohup python v9_supp.py full > run_v9.log 2>&1 &
python v9_supp.py report          # rebuild REPORT_v9.md from disk (zero GPU)
```

See `docs/design_notes.md` for the design rationale
and the pairing/measurability rules these phases rely on.

### v7 reviewer-response suite

The v7 suite (see `docs/design_notes.md` for the design rationale and the
phase-command layout) runs the reviewer-concern experiments and rebuilds the report:

```bash
# zero-GPU: analytic FLOPs/KV-cache + paired sign-flip stats + report rebuild
python v7_supp.py analysis
# one phase at a time (resumable), or the whole chain:
python v7_supp.py phase P0E   # 40k-step long run
python v7_supp.py phase P2S   # scale re-seeding to 4 seeds
python v7_supp.py phase P1L   # length extrapolation
python v7_supp.py phase P1T   # seq-2048 topk sweep
python build_report.py        # regenerate REPORT_v7.md from artifacts (zero GPU)
```

Every driver meters its own accelerator time against a configured run-time
cap: it refuses to start a run that would pass the cap, and hard-truncates a
run that reaches it mid-flight.  On a multi-GPU box, `CSA_PAR=auto` runs the
same phases with one independent run per card (the cap then meters aggregate
GPU-seconds, i.e. N parallel runs consume it N times faster in wall clock) — a truncated cell leaves no partial
measurement and is retried on the next pass. When all phases finish the
script pushes and shuts the instance down (cancel with
`pkill -f v6autoshutdown`; suppress with `V6_NO_SHUTDOWN=1`, pushes with
`V6_NO_PUSH=1`).

> All quantities are taken from the tables that `build_report.py` /
> `fuse_analysis.py` and the per-version `v*_supp.py report` builders compute
> on the fly from the on-disk `results_*/` and `analysis_*/` artifacts; no
> quantity is ever hard-coded in the repository prose.

## Review-4 hardening (what changed)

See the review-4/review-5 rows in `DECISION_LOG.md` for the item-by-item list
(there is no `REVIEW_4_RESPONSE.md` in this source snapshot).  The main additions are
machine gates rather than new prose:

- **Claim wording** now goes through `exp_lib.reportable_claim`, which since the
  review-10 split reports **three separate verdicts**: statistical significance
  (n>=6, raw p, Holm p, family resolution floor), seed **stability**
  (|mean|>std, leave-one-seed-out), and **practical value** (only when a
  pre-declared `min_effect` is supplied; v12 pre-registers none, so v12 makes no
  practical-value claim). |mean|>std is a stability diagnostic, not a
  significance test.  `attach_holm` reports
  `family_min_attainable_p_holm`, so `p_holm=1.0` from an oversized family is
  printed as a resolution ceiling, not as a null result.  v10's NIAH contrasts
  are now inside the Holm family.
- **Crossover scaling** uses `exp_lib.crossover_scaling_fit`: >=4 de-duplicated
  widths, no censored subset, training volume within 25%, slope CI/p/R²/LOO,
  claim level fixed to descriptive.  Otherwise it refuses and says why.
- **Matched initialization and compute**: every training record carries
  `init_core_fp` (tok/pos/head/norm) and `pair_reason` refuses a pair whose
  digests differ.  `_compute_control_detail` reports measured time/token (or
  the 6ND proxy) and its direction; a significant contrast whose extra compute
  favours the claimed winner becomes `system_level_only`, not `verified`.
- **Stale records fail closed**: `build_report.sec_p1l` and v10 exclude any arm
  with no current code stamp (no table cell, no ratio, no conclusion);
  `_agg_entry` refuses a multi-group aggregate with no current stamp.  Missing
  `truncated` flags are treated as unverifiable, not full-length.
- **Uncertainty unit**: `ppl_ci95_doc_bootstrap` resamples the fixed
  evaluation rows/windows, not documents (a 4096-token window can span
  document boundaries).  v12 therefore labels that interval as
  window-sampling and refuses population-level claims below 64 rows; genuine
  document-level CIs need the loader `doc_index` path (used by the follow-up
  study, not by the frozen v12 records).
- **Efficiency**: the length phase records `measure_efficiency`; the new
  `measure_efficiency_sweep` covers batch x length, and
  `audit_gates.efficiency_pairs` only permits a "faster" claim when token
  count, device, torch version, KV semantics match and the ratio exceeds ±5%.
  Analytic FLOPs are labelled structural, never as measured speed.
- **Mechanism**: `audit_gates.selection` defines the selection-recall artifact
  contract and reports `not_verifiable_locally` (with an explicit "unsupported
  mechanism claim" note) until that artifact exists.  Boundary alignment,
  block/read diagnostics and NIAH remain the available direct readouts.
- **Evidence**: `audit_gates.release` parses `RELEASES.md`; with no complete
  row the repository is declared source-only/not auditable.  `archive_release.py`
  now also packs `pyproject.toml`, cache/tokenizer metadata and `run_*.log`.

## Review-3 hardening (what changed)

- **PPL aggregation**: training/length records now carry `nll_sum`,
  `n_tokens` and per-row NLL (per-document NLL when `eval_ppl` is called with
  a `doc_index` table); `aggregate` reports `ppl_pooled =
  exp(ΣNLL/Σtokens)` as `ppl_primary` (with the arithmetic seed mean,
  median/IQR, seed-level t interval and document bootstrap kept alongside).
  The final PPL is still measured on the full fixed validation batch;
  `eval_subset` only controls the in-training curve.
- **Statistics**: v7–v11 analysis and `build_report.py` attach Holm
  step-down over the displayed family (omitted comparisons keep a dummy p=1
  slot). Report prose may say “statistically significant” only when n≥6, raw
  p<0.05, Holm p<0.05 and the family resolution floor is reachable;
  |mean|>std and leave-one-seed-out stability are reported separately as
  **stability**, never as the significance criterion.
- **Sign-flip assumptions**: `paired_effect_stats` now also reports the sign
  split, a paired bootstrap interval, an exact Wilcoxon sensitivity p, and an
  unpaired bootstrap interval when both arms are supplied. The primary p
  remains seed-level.
- **Objectives/parameters**: each record carries a `train_objective`
  descriptor (does `comp_reg` actually apply? is the delta-LR effective? what
  boundary policy?) plus `param_breakdown` and
  `kv_cache_elements_per_token`. `methodology_audit` flags same-run_cfg but
  different-objective contrasts as system-level, not isolated mechanism.
- **Boundary learning**: the default `csa_dynamic` remains a cosine threshold
  with a learned scalar offset trained by `comp_reg`; it is **not**
  task-learned. The new opt-in `csa_dynamic_task` samples cuts and adds a
  score-function term, and is scheduled in the ablation panel.
- **Synthesized records**: `run_gapfill.py full` no longer reconstructs the
  long summary unless `V6_SYNTH=1`; crossover statistics exclude synthesized
  seeds entirely and refuse a slope with fewer than 3 independent points.
- **Reproducibility**: `pyproject.toml` declares Python ≥3.12,
  `requirements.txt` plus a verified CPU `requirements.lock` are tracked,
  import-time pip install and the HF mirror are opt-in (`CSA_AUTO_INSTALL=1`, `CSA_HF_MIRROR=1`), and
  `archive_release.py` packs `DECISION_LOG.md`, the dependency declarations,
  the freeze lock and a SHA-256 manifest. See `RELEASES.md`.
