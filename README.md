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
> The frozen manifest is at `manifest_version: 3` (see its `revision.history`).
> P5T is deferred, and the continuous-loop workflow is currently re-enabled,
> gated by the repository variable `CONTINUOUS_LOOP_ENABLED`.

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

## Study at a glance

The drivers are organized by study generation. Each has a `smoke` mode for
plumbing checks, a training mode for the real panels, and zero-GPU
analysis/report subcommands where applicable. v6 keeps its historical script
entry point; v7 onwards live under `explib/`.

| generation | scope | entry point | generated artifacts |
|---|---|---|---|
| v6 | core CSA/HCA table, leave-one-out ablations, scale probe, long runs | `python run_gapfill.py full` | `results_lm_v3_1500/`, `results_lm_v4_abl/`, `results_lm_v5_scale/`, `results_lm_v3_long/` |
| v7 | RoPE + QK-norm, length extrapolation, top-k sweep | `python -m explib.rope` | `REPORT_v7.md` |
| v8 | scale-panel seed completion and the 20k RoPE run | `python -m explib.scale` | `analysis_v8/` |
| v9 | seq-2048 top-k completion and the third 20k seed | `python -m explib.resume` | `REPORT_v9.md` |
| v10 | distractor-injection mechanism probe, crossover scaling, selector controls | `python -m explib.corruption` | `analysis_v10/`, `REPORT_v10.md` |
| v11 | statistical close-out of the v10/v11 panels | `python -m explib.length` | `analysis_v11/`, `REPORT_v11.md` |
| v12 | frozen fixed-chunk confirmation | `python -m explib.confirmation` | `analysis_v12/`, `REPORT_v12.md` |
| M0 | FineWeb-Edu data path, kernel G0 gate, S1/S2 scale panels | `python -m explib.m0` | `results_b_s1/`, `results_b_s2/`, `analysis_m0/` |
| open-block | maximum block length × open-block reading rule | `python -m explib.open_block` | `analysis_open_block/` |

Run a driver with `--help` for its exact subcommands. Training is never
implicit: it starts only in a mode named `full`, `phase <name>`, `main`, or
`auto`.

## Contents

- [Study at a glance](#study-at-a-glance)
- [Start here](#start-here)
- [Quick start](#quick-start)
- [Where to look](#where-to-look)
- [Repository layout](#repository-layout)
  - [Regenerating the experiment artifacts](#regenerating-the-experiment-artifacts)
- [Running the experiments](#running-the-experiments)
  - [Environment](#environment)
  - [Data and cache fallbacks](#data-and-cache-fallbacks)
  - [Validate the pipeline before training](#validate-the-pipeline-before-training)
  - [Run the v6 supplementary study](#run-the-v6-supplementary-study)
  - [Task-level parallelism (one GPU per run)](#task-level-parallelism-one-gpu-per-run)
  - [B-tier M0 / S1 (scale-up entry point, adopted tier B)](#b-tier-m0--s1-scale-up-entry-point-adopted-tier-b)
  - [Triton kernel path (M0/G0)](#triton-kernel-path-m0g0)
  - [v11 statistical close-out suite](#v11-statistical-close-out-suite)
  - [v10 mechanism / scaling / stats suite](#v10-mechanism--scaling--stats-suite)
  - [v12 confirmation suite (exploration / confirmation split)](#v12-confirmation-suite-exploration--confirmation-split)
  - [v8 / v9 close-out suites](#v8--v9-close-out-suites)
  - [v7 reviewer-response suite](#v7-reviewer-response-suite)
- [Measurement guardrails](#measurement-guardrails)

## Start here

All commands below run from the repository root. A checkout is healthy when
`make test` passes.

| goal | start with | starts training? |
|---|---|---|
| verify that a checkout is healthy | `python -m pip install -r requirements.lock`, then `make test` | no |
| validate the frozen plan without touching the test split | `make plan` | no |
| run the open-block causality/reference gates | `python -m pytest -q test_open_block.py` | no |
| try the pipeline end to end | `python run_gapfill.py smoke` | smoke run only |
| run or resume a study panel | [Study at a glance](#study-at-a-glance) | yes |
| rebuild a report from local artifacts | `python build_report.py` or `python -m explib.<driver> report` | no |
| check whether two runs agree | `python -m explib.reproduce compare ORIGINAL RERUN` | no |
| understand why a measurement is defined this way | the [Where to look](#where-to-look) table | no |

## Quick start

Python 3.12 is required. `requirements.lock` is the verified CPU environment
used by the tests and the zero-GPU analysis path. On a CUDA machine, install
the torch wheel that matches the local driver first, then the ranges in
`requirements.txt` (see [Environment](#environment)).

```bash
python -m pip install -r requirements.lock
make test                              # complete CPU pytest suite
make plan                              # frozen-manifest preflight; no test-split read
python -m explib.confirmation --help   # list subcommands without starting a run
```

None of these commands starts a training run. Training is always explicit:
invoke a driver's `full`, `phase <name>`, or `auto` mode when you intend to
run it, and start with that driver's `smoke` mode when validating a change to
the pipeline.

### Where to look

| question | document |
|---|---|
| How do I run or re-run an experiment? | this README |
| Which command and subcommand should I use? | [Study at a glance](#study-at-a-glance) and the per-driver sections below |
| Why is a measurement or comparison defined this way? | `docs/design_notes.md` |
| What was decided, and in what order? | `DECISION_LOG.md` |
| Where is the released evidence? | `RELEASES.md` |
| How were the review issues resolved? | `REVIEW_4_RESPONSE.md` |

## Repository layout

The repository keeps only **source code, driver scripts and design
documents**. All experiment result directories (`results_*/`), zero-GPU
analysis directories (`analysis_*/`), per-version reports (`REPORT*.md`) and
run-state files are **not tracked** — the driver scripts below regenerate them
on whatever machine runs the experiments, so you can simply reproduce them
locally whenever needed (see the commands in the next section). `.gitignore`
already keeps those path families out of the repository (`docs/` retains only
the design notes, `docs/design_notes.md`).

| path | what |
|---|---|
| `exp_lib.py` | **compatibility shim** that re-exports the split implementation in `explib/` |
| `_split_loader.py` | shared loader for the split packages: runs each `_parts/` fragment in the caller's module globals, in declared order, with the fragment path as the compile filename (so tracebacks point at the reviewable source) |
| `explib/` | **the library — and the source of truth**: model, CSA/HCA attention, pooling, indexer, segmenters, training loop, aggregation and statistics |
| `csa_kernels.py` | **compatibility shim** that re-exports the split implementation in `_csa_kernels/` |
| `_csa_kernels/` | **the fused-kernel path for the three core operators** (the retired compute-plan M0 work package): block-compression pooling, indexer top-k selection and sparse attention as Triton kernels (plus a CPU-runnable torch-reference twin of the same algorithmic decomposition); opt-in via `CSA_KERNEL`, off by default — see "Triton kernel path (M0/G0)" below |
| `run_gapfill.py` | **the v6 driver**: core/ablation/scale/long panels, resumable, guarded by a run-time cap, pushes after every phase, auto-shuts the instance down |
| `par.py` | **task-level parallel executor (opt-in)**: one GPU card per independent run — see "Task-level parallelism" below |
| `explib/rope/` | **the v7 driver** (`python -m explib.rope`): reviewer-response supplementary suite — P0R RoPE+QK-norm, P0W dense-warmup, P0E 40k-step long run, P2S scale re-seeding, P1L length extrapolation, P1T topk sweep |
| `explib/scale/` | **the v8 driver** (`python -m explib.scale`): P1S (scale panel seed completion to 4 seeds) and P2R (RoPE+QK-norm 20k-step long run) |
| `explib/resume/` | **the v9 driver** (`python -m explib.resume`): P2T (the seq-2048 topk sweep at `batch_size=1`) and P2S3 (third seed for the RoPE 20k run) |
| `explib/corruption/` | **the v10 driver** (`python -m explib.corruption`): P3M length-extrapolation mechanism + distractor-injection probe, P3S inversion-crossover scaling, P3T topk sweep to 4 seeds |
| `explib/length/` | **the v11 driver** (`python -m explib.length`): statistical close-out — distractor probe to n=6 (P4MT/P4MP), crossover panels to n=4 (P4S), d=384 paired arms to n=4 (P4F) |
| `explib/confirmation/` | **the v12 confirmation driver** (`python -m explib.confirmation`; reviewer-response + review-2 hardening): P5C trains all six frozen arms at the manifest seed list, P5MP runs the extended corruption-mode probe (uniform/realistic-token/mask/delete + review-2 span/doc/topical + NIAH-natural), P5S2/P5S3 add the token-matched and 6ND-param-proxy scale arms, P5T performs the FROZEN final evaluation on the held-out WikiText-103 test split per `confirmation_frozen.json` — with a load-time plan-power gate (a plan that cannot mathematically confirm is refused), strict record admission (seed whitelist / unique records / one frozen test slice / recipe signature / manifest sha), a first-access freeze lock, and per-document PPL records |
| `explib/open_block/` | **open-block panel driver** (`python -m explib.open_block`): maximum block length (32/64/128/no cap) × current-block reading rule (`closed` / `prefix_summary` / `prefix_singletons`), with a shared top-k budget, causal prefix construction, outside-window auxiliary panel, paired per-seed deltas and interaction statistics — see `docs/design_notes.md` §24 |
| `explib/reproduce/` | **cross-run reproduction comparator** (`python -m explib.reproduce compare ORIGINAL RERUN`): matches two panel summaries cell by cell and reports whether they share source bytes, data slice, configuration, execution environment and matched initialization, plus whether the final PPL is exact/within a declared tolerance. Missing provenance is `provenance_incomplete`, never a pass; the same check is exposed as `audit_gates.py reproduction ORIGINAL RERUN` — see `docs/design_notes.md` §83 |
| `test_open_block.py` | **open-block CPU gates**: config/segmenter, optimized-vs-reference prefix pooling (forward + gradients), prefix invariance, future perturbation/gradient, candidate masks, edge cases, closed regression, singleton semantics and HCA budget declaration |
| `test_review_part_*.py`, `test_review_fixes.py` | **CPU review pytest suite**: compatibility entry point plus the split test modules (run with `make test` or `pytest`) |
| `confirmation_frozen.json` | **pre-registered confirmation plan** (tracked; `manifest_version: 3`, revision history v1-v5): frozen arms and seeds 0-11 (revised 6->12 BEFORE any test-split access; power table recorded), endpoints, exact sign-flip + Holm, decision rule, machine-checkable binding block and a scope declaration — see `docs/design_notes.md` §17 and `DECISION_LOG.md` |
| `confirmation_frozen_dynamic.json` | **separate draft pre-registration for dynamic chunking and HCA** (`manifest_version: 4`; not frozen): H1 segmentation quality, H2 retrieval precision and H3 long-range coverage, plus the controls that must be scheduled before a disjoint test corpus is reserved — see `DECISION_LOG.md` |
| `RELEASES.md` | **evidence release index**: where each immutable `release_<tag>.tar.gz` + SHA-256 manifest is published; the Git repo itself stays source-only |
| `pyproject.toml` / `requirements.txt` / `requirements.lock` | Python ≥3.12 and dependency declarations; install explicitly (no import-time pip install). `requirements.lock` is the verified CPU environment |
| `DECISION_LOG.md` | **decision log** (tracked; reconstructed 2026-10-18 from the README/design notes/manifest because the snapshot has one squashed commit): why each version ran what it ran, what was seen, and what was decided next — the exploration/confirmation paper trail; formal releases additionally archive per-seed results via `archive_release.py` |
| `REVIEW_4_RESPONSE.md` | **review-4 per-issue audit** (tracked): the 21 methodological issues, what this repository already enforced, what this revision changed, and what still cannot be answered without new experiments/artifacts |
| `audit_gates.py` | **compatibility shim** for `_audit_gates/`, the **review-2/4/7 verification entry points** (zero-experiment): data-leakage audits (tokenizer provenance, vocab identity, train/test overlap), CPU causality unit checks (prefix invariance, label shift, decode consistency), block-attribution / boundary-vs-random / length-consistency / baseline-fairness / failure-concentration / budget-reconciliation audits, the review-7 single-factor design gate (declared contrasts' arm-cfg diffs vs their declared intervention key sets; frozen-manifest mechanism_claim_allowed flags) and prefill-decode-kv-growth measurement utilities |
| `archive_release.py` | **immutable evidence archiver**: packs per-release artifacts (results, analyses, reports, freeze lock, decision log, environment, checkpoint hashes) into a tar.gz with a per-file SHA-256 manifest and an independent `verify` subcommand — the git repo stays source-only (design_notes §8) while every formal release gets an auditable fixed archive |
| `explib/m0/` | **the M0/S1 driver (adopted B tier)** (`python -m explib.m0`): entry points for the scale-up roadmap — `params` (zero-GPU S1–S4 parameter audit), `data` (FineWeb-Edu + 32k-BPE corpus cache), `smoke` (end-to-end self-check on the B-tier data path), `kernel` (the G0 validation suite: kernel/eager numerical comparison for the three operators plus the sparse-attention throughput benchmark), `full` (the S1 panel); see "B-tier M0/S1" below |
| `build_report.py` | **compatibility shim** for `_build_report/`, the zero-GPU report builder: regenerates the v7 report from `results_*/` + `analysis_*/` artifacts (every number is computed from disk; protocol constants and analytic FLOPs assumptions are explicit inputs, not measurements) |
| `fuse_analysis.py` | zero-GPU fuse-vs-no-fuse driver: recomputes per-layer boundary alignment (tol=1), block-length stats and PPL trajectories from a `results_*/summary.json` |
| `Makefile` | zero-GPU entry points: `make test`, `make plan`, `make audit`, `make integrity`, `make reproduce`, `make install-check`, `make build`, `make clean` |
| `docs/design_notes.md` | **design and implementation notes**: why the block-read gate is strict and must be synced across five sites, the dynamic segmenter's prefix-invariance requirement, the pairing/measurability rules, the reporting conventions, and the contracts that keep attention row-chunking equivalent to its unbatched form |

> **The measurement library lives in `explib/`; edit those modules directly.**
> `exp_lib.py` (and `csa_kernels.py`, `audit_gates.py`, `build_report.py`,
> `data_guard.py`) are compatibility shims that re-export the split packages.
>
> **Driver entry points.** The former top-level v7–v12, M0/S1 and open-block
> scripts now live in modules under `explib/`: use `python -m explib.rope`,
> `python -m explib.scale`, `python -m explib.resume`,
> `python -m explib.corruption`, `python -m explib.length`,
> `python -m explib.confirmation`, `python -m explib.m0`, and
> `python -m explib.open_block`. The subcommands are unchanged.

### Regenerating the experiment artifacts

Result directories and reports are not kept in the repository. To obtain them
again:

```bash
python run_gapfill.py smoke            # end-to-end self-check; no research measurement
python run_gapfill.py full             # the v6 main experiment (bounded by the run-time cap)
python -m explib.<driver> --help       # exact subcommands for a v7-v12/M0/open-block driver
python build_report.py                 # rebuild the report from on-disk results_*/ + analysis_*/
```

If the same cells were executed twice (parallel vs serial, a second machine, or
a re-run after a code change), verify the second copy instead of trusting the report:

```bash
python -m explib.reproduce compare results_lm_v6 results_lm_v6_rerun
# exit 0 only when source bytes, data slice, config, environment, matched
# initialization and the declared PPL tolerance all line up; use
# --require exact for a bit-identical demand.
```

The verifier reports each dimension separately, so `environment_changed`,
`data_changed`, `code_changed` and `numerical_drift` are not collapsed into one
"did not reproduce" message. A missing fingerprint is
`provenance_incomplete` (never a pass).

## Running the experiments (any machine)

The commands below are grouped by side effect: environment, zero-GPU checks,
then training. Run them from the repository root. Training panels need a GPU;
tests, analysis, report builders and audit gates run on CPU.

### Environment

Python 3.12 is required (see `pyproject.toml`).

- **Verified CPU environment (tests and zero-GPU analysis):**

  ```bash
  python -m pip install -r requirements.lock
  ```

  The lock is a complete pinned closure and includes the CPU torch wheel.

- **CUDA machine:** install the torch wheel that matches the local driver
  first, then the compatibility ranges:

  ```bash
  python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch
  python -m pip install -r requirements.txt
  ```

  The supported range is `torch>=2.8,<3`. The lock pins `torch==2.14.1+cpu`
  for the CPU path only; regenerate a machine-specific lock with
  `python -m pip freeze > requirements.lock` and archive it with the release.
  The scatter paths cast int32 indices to int64, so they do not rely on a
  particular torch ≥ 2.14 scatter behaviour.

- **No hidden installs:** importing the library never installs packages.
  `CSA_AUTO_INSTALL=1` is an explicit opt-in for a throwaway environment.

- **Hugging Face mirror:** the official Hub is the default;
  `CSA_HF_MIRROR=1` opts into `hf-mirror.com` when the official endpoint is
  unreachable.

### Data and cache fallbacks

- `load_wikitext` first tries the Hub. If the Hub is unreachable, set
  `WT103_RAW_TXT` to a directory containing `wiki.train.raw` and
  `wiki.valid.raw`; the loader also checks `./wt103_raw` next to the repo and
  `/root/wt103_raw`.
- Only those two files are read, and tokenising them reproduces the
  `wt103_cache/*.npy` cache token-for-token. The fallback is an availability
  path, not a different measurement.
- Once `wt103_cache/` is populated, the corpus is served from the cache and no
  dump or network access is needed. `WT103_CACHE_DIR` relocates the cache; the
  loaders, `audit_leakage` and P5T all resolve the same variable.
- `python -m explib.confirmation smoke` uses an isolated cache under its smoke
  outdir, so it cannot leave a differently tokenised file in the frozen cache.
- Test-cache hits verify the persisted tokenizer bytes (sha256 recorded in
  `test_meta`) in addition to the train fit fingerprint.
- The v12 P5T final test reads `wiki.test.raw` via `load_wikitest` (same search
  path). No earlier phase reads that split. After changing the train corpus,
  delete the whole `wt103_cache/` and rebuild it; the test cache is not rebuilt
  silently.

### Validate the pipeline before training

These checks exercise the plumbing without producing a research measurement.
The smoke run covers dense / fixed / dynamic arms and the leave-one-out
ablation forward paths (random indexer, zero content, sink off, learned
fusion, HCA layers, dense+sink control):

```bash
python run_gapfill.py smoke
python -m explib.corruption smoke   # corruption modes + NIAH + controls
python -m pytest -q test_open_block.py   # open-block causality/reference/mask gates (CPU)
python -m explib.open_block smoke   # open-block matrix plumbing on a tiny synthetic run
python -m explib.confirmation smoke # frozen final-test roundtrip
```

The remaining open-block modes are real runs:

```bash
python -m explib.open_block pilot     # one cell, short run
python -m explib.open_block main      # 8 cells x 4 seeds; resumable
python -m explib.open_block aux       # outside-window auxiliary panel
python -m explib.open_block extended  # contingency: all 4 caps x all 3 modes
python -m explib.open_block report    # paired deltas + interaction + exact sign-flip/Holm
```

### Run the v6 supplementary study

```bash
nohup python run_gapfill.py full > run_v6.log 2>&1 &
tail -f run_v6.log
```

`run_gapfill.py full` runs the phases below in order. Each phase is resumable:
re-running the same command skips completed `variant::seed` pairs. The core,
ablation and scale panels are scheduled before the 20k-step long runs, and
every phase starts at the 3-seed floor required by the methodology audit
(`min_n`). Seed-completion passes (P2S / P3T / P4SS / P5C …) later top
individual panels up to n=4–6:

| phase | output dir | what |
|---|---|---|
| P-CORE | `results_lm_v3_1500` | 9-variant core table, 1500 steps x 3 seeds |
| P-ABL | `results_lm_v4_abl` | leave-one-out ablation table — in-panel references (`full` / `csa_fixed` / `csa_dynamic` / `hybrid_fixed` / `hybrid_dynamic`) + `csa_fix_*` / `csa_dyn_*` module-off arms, fuse/HCA/topk arms, 1500 steps x 3 seeds |
| P-SCALE | `results_lm_v5_scale` | d=384 / 8 layers / seq-1024 probe, 5 variants x 3 seeds x 4000 steps |
| P-LONG | `results_lm_v3_long` | hybrid_fixed / hybrid_dynamic / hybrid_csa_dyn at seeds 0-2 → 3-seed long-run table |

### Task-level parallelism (one GPU per run)

Every driver can run its independent cells — one `(variant, seed)` training
run each — in parallel, **one card per run**, on multi-GPU boxes.  It is
strictly opt-in and off by default, so the serial behaviour above is exactly
what you get when the variable is unset:

```bash
# one worker per visible GPU (recommended):
CSA_PAR=auto nohup python -m explib.length full > run_v11.log 2>&1 &

# only 2 cards, only these cards:
CSA_PAR=2 CSA_PAR_GPUS=0,3 python -m explib.corruption phase P3SL

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

The adopted B-tier scale-up plan (entry point `python -m explib.m0`; the former
compute-plan document is retired) adds a second
corpus path and the S1–S4 scale ladder on top of the WikiText-103 mechanism
layer; `python -m explib.m0` is the entry point:

```bash
python -m explib.m0 params   # zero GPU: S1–S4 parameter audit vs the plan
python -m explib.m0 data     # build / verify the FineWeb-Edu cache (M0_TOKENS=…)
python -m explib.m0 smoke    # end-to-end self-check on the B-tier data path
python -m explib.m0 kernel   # G0 suite: kernel/eager numerical comparison + throughput
python -m explib.m0 full     # the S1 panel (5 arms × 4 seeds; CSA_PAR=auto works)
M0_PANEL=S2 python -m explib.m0 full   # the S2 panel (5 arms × 3 seeds × 12B; M2_CAP_HOURS override)
python -m explib.m0 auto     # UNATTENDED: G0 gate (fail-closed) -> certificate re-check ->
                           #   panel chain starts by itself; default panel S1,
                           #   M0_AUTO_CHAIN=S1,S2 sequences both; CSA_KERNEL=1 after a
                           #   green gate (explicit setting wins); a failed gate starts no
                           #   panel and leaves a 40-min diagnosis window
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
python -m explib.m0 kernel                 # anywhere: ref-backend semantics check
CSA_KERNEL=1 python -m explib.m0 kernel    # CUDA + triton: full G0 suite
CSA_KERNEL=1 python -m explib.m0 full      # kernel-mode S1 panel (after G0)
python -m explib.m0 auto                   # unattended: G0 suite -> certificate -> panel chain
                                         #   (the "after G0, may B-tier start?" decision is
                                         #   now machine-made: certificate present = both
                                         #   G0 criteria passed = the panel starts)
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
python -m explib.length smoke
nohup python -m explib.length full > run_v11.log 2>&1 &
python -m explib.length report         # rebuild REPORT_v11.md from disk (zero GPU)
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
python -m explib.corruption smoke
nohup python -m explib.corruption full > run_v10.log 2>&1 &
python -m explib.corruption analysis       # zero-GPU stats -> analysis_v10/
python -m explib.corruption report         # rebuild REPORT_v10.md from disk (zero GPU)
```

Phases (each resumable): P3MT (retrain the two RoPE arms, the P1L recipe),
P3MTC (**from-scratch selector controls under the exact same budget**:
`csa_fixed_rope_randidx` / `csa_fixed_rope_allblocks`; the latter doubles as
the item-by-item-aligned dense twin), P3MP (eval-only far-distance distractor
injection: paired corrupted contexts, clean 512-token target, learned-selection
vs dense vs random-indexer vs see-everything arms, **uniform /
realistic-token (natural) / mask / delete + review-2 span / doc / topical
corruption modes** + the NIAH-natural far-necessity task), P3T
(topk sweep seeds 2/3 → n=4), P3SS/P3SL (d=128/4L 8000-step and d=512/10L
4000-step crossover panels), P3SL2/P3SL3 (**token-matched and
compute-matched large arms** — the scale-confound fix; see
docs/design_notes.md §16).
See `docs/design_notes.md` for the design rationale behind the
paired-corruption protocol, the pre-registered crossover criterion, the
corruption-mode/ΔNLL semantics (§15) and the intervention-vs-trained-control
distinction (§18).

### v12 confirmation suite (exploration / confirmation split)

```bash
python -m explib.confirmation smoke           # end-to-end self-check (tiny ckpts + a smoke freeze manifest)
python -m explib.confirmation plan            # zero-GPU pre-flight: stamps + implementation binding + admission + power, NO test read;
                                              #   on pass, archives the audit snapshot analysis_v12/plan_snapshot.json
python -m explib.confirmation auto            # UNATTENDED: the same chain as `full`, made explicitly pause-free —
                                              #   the plan gate runs first (fail-closed) and its approval evidence
                                              #   (manifest sha256 / code stamp / git tree / arms / seeds /
                                              #   primary comparisons / alpha) is archived for ASYNCHRONOUS human
                                              #   review instead of blocking the launch
python -m explib.confirmation phase P5C       # train all six frozen arms at the manifest seed list (resumable; 0-5 main arms covered by P3MT/P4MT)
python -m explib.confirmation phase P5MP      # extended probe (all modes + NIAH + controls) at the manifest seed list
python -m explib.confirmation phase P5S2      # token-matched large scale arm (8000 steps)
python -m explib.confirmation phase P5S3      # compute-matched large scale arm (param-ratio steps)
python -m explib.confirmation phase P5T       # FROZEN final test on wiki.test.raw (requires the freeze manifest)
python -m explib.confirmation analysis        # zero-GPU: shared v10 stats + frozen final-test stats
python -m explib.confirmation report          # rebuild REPORT_v12.md from disk (zero GPU)
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

**Scope (review-2 issue 4)**: the confirmation covers ONLY fixed-chunk CSA
mechanism claims. Dynamic chunking vs fixed chunking, semantic boundary value,
HCA advantages and dynamic-chunking scaling laws are NOT covered by this
freeze and need their own pre-registration.

**Verification entry points (review-2 issue 11)** — zero-experiment audits:

```bash
python audit_gates.py causality   # CPU unit audit: prefix invariance / label shift / decode consistency
python audit_gates.py leakage     # tokenizer provenance, vocab identity, train/test overlap
python audit_gates.py single_factor  # design-time gate: every declared contrast's arm-cfg diff equals its declared intervention key set; open-block matrix axes; frozen-manifest mechanism_claim_allowed flags vs the actual cfg diff; also the learned/frozen boundary-gate, sink, content and fusion control ladders
python audit_gates.py sesoi       # design-time gate: every ablation component has a valid declared SESOI and is fully scheduled
python audit_gates.py unit_influence  # result-analysis gate: unit-influence block reproduces pooled Δ; flags one-unit-driven comparisons
python audit_gates.py looks       # look-history gate: recompute per-look and exact max-over-looks sequential p; flags seed top-ups that only look significant at the final n
python audit_gates.py experiment_ledger  # replay the experiments.jsonl hash chain + tail anchor; edited/deleted/reordered/truncated events fail
python audit_gates.py run_allocation  # prospective run-order plan + arm×device balance; post-hoc plans and complete arm/device confounding fail
python audit_gates.py report      # run every locally-verifiable audit -> audit_report.json
python audit_gates.py selection   # selection-recall contract (refuses unsupported mechanism claims when absent)
python audit_gates.py efficiency_pairs  # measured latency claims: same shape/device/KV required
python audit_gates.py release     # evidence-release readiness (no complete row => source-only)
python archive_release.py plan v13 --dirs results_lm_test analysis_v12   # dry-run an evidence archive
```

### v8 / v9 close-out suites

```bash
# v8: scale-panel seed completion (4 seeds) + RoPE 20k long run
python -m explib.scale smoke
python -m explib.scale full            # or: python -m explib.scale phase P1S / P2R
python -m explib.scale analysis        # zero-GPU stats -> analysis_v8/

# v9: complete topk sweep (bs=1) + RoPE-20k third seed + report
python -m explib.resume smoke
nohup python -m explib.resume full > run_v9.log 2>&1 &
python -m explib.resume report          # rebuild REPORT_v9.md from disk (zero GPU)
```

See `docs/design_notes.md` for the design rationale
and the pairing/measurability rules these phases rely on.

### v7 reviewer-response suite

The v7 suite (see `docs/design_notes.md` for the design rationale and the
phase-command layout) runs the reviewer-concern experiments and rebuilds the report:

```bash
# zero-GPU: analytic FLOPs/KV-cache + paired sign-flip stats + report rebuild
python -m explib.rope analysis
# one phase at a time (resumable), or the whole chain:
python -m explib.rope phase P0E   # 40k-step long run
python -m explib.rope phase P2S   # scale re-seeding to 4 seeds
python -m explib.rope phase P1L   # length extrapolation
python -m explib.rope phase P1T   # seq-2048 topk sweep
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

## Measurement guardrails

The analysis path fails closed: absent, stale, or mismatched provenance is
reported as `not_verifiable_locally` / `provenance_incomplete`, never as a
pass. The main contracts are:

- **Claim wording** goes through `exp_lib.reportable_claim`: n≥6, raw p, Holm
  p, |mean|>std, and the family-resolution floor. `attach_holm` reports the
  minimum attainable Holm p, so an oversized family is shown as a resolution
  ceiling rather than a null result.
- **Pairs must be matched** on data slice, recipe, code stamp and
  initialization (`init_core_fp`); `pair_reason` refuses a pair whose
  digests differ. A contrast whose measured compute favours the claimed
  winner is labelled `system_level_only`, not `verified`.
- **Crossover scaling** requires at least four de-duplicated widths, no
  censored subset, training volume within 25%, and reports slope CI/p/R²/LOO;
  otherwise it refuses and states why. The claim level is descriptive.
- **Ablations** carry a declared SESOI (relative PPL margin, default 1% for
  exploratory panels) and are judged with paired TOST plus family-adjusted
  equivalence p. A non-significant contrast is never relabelled "removable"
  without TOST; a sub-SESOI effect is `sub_sesoi`, not `verified`.
- **Unit influence and repeated looks** are explicit. Pooled ΔPPL carries an
  exact leave-one-unit-out readout, and repeated seed sets are judged by the
  exact max-over-looks sign-flip p recorded in `look_history.json`; neither
  changes a frozen p/Holm/`confirmed` verdict.
- **Design and release gates** are executable: `audit_gates.py` covers
  leakage, causality, single-factor contrasts, SESOI scheduling, unit
  influence, look history, allocation and evidence readiness;
  `archive_release.py verify` checks a published archive independently.
  With no complete row in `RELEASES.md`, the repository stays source-only.

The item-by-item review disposition is in `REVIEW_4_RESPONSE.md`; the full
design rationale is in `docs/design_notes.md` (§25, §56–§58, §84), and the
chronology is in `DECISION_LOG.md`.
