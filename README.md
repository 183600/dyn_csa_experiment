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
> The frozen manifest is at `manifest_version: 3` (see its `revision.history`);
> the continuous-loop CI is intentionally disabled until that signed commit is
> fixed and P5T has run.

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

## Repository layout

The repository keeps only **source code, driver scripts and design
documents**. All experiment result directories (`results_*/`), zero-GPU
analysis directories (`analysis_*/`), per-version reports (`REPORT*.md`) and
run-state files are **not tracked** — the driver scripts below regenerate them
on whatever machine runs the experiments, so you can simply reproduce them
locally whenever needed (see the commands in the next section). `.gitignore`
already keeps those path families out of the repository (`docs/` retains only
`design_notes.md`).

| path | what |
|---|---|
| `exp_lib.py` | **compatibility shim** that re-exports the split implementation in `explib/` |
| `explib/` | **the library — and the source of truth**: model, CSA/HCA attention, pooling, indexer, segmenters, training loop, aggregation and statistics |
| `csa_kernels.py` | **compatibility shim** that re-exports the split implementation in `_csa_kernels/` |
| `_csa_kernels/` | **the fused-kernel path for the three core operators** (the retired compute-plan M0 work package): block-compression pooling, indexer top-k selection and sparse attention as Triton kernels (plus a CPU-runnable torch-reference twin of the same algorithmic decomposition); opt-in via `CSA_KERNEL`, off by default — see "Triton kernel path (M0/G0)" below |
| `run_gapfill.py` | **the v6 driver**: the minimum-publishable supplementary runs, resumable, guarded by a run-time cap, pushes after every phase, auto-shuts the instance down |
| `par.py` | **task-level parallel executor (opt-in)**: one GPU card per independent run — see "Task-level parallelism" below |
| `explib/rope/` | **the v7 driver** (`python -m explib.rope`): reviewer-response supplementary suite — P0R RoPE+QK-norm, P0W dense-warmup, P0E 40k-step long run, P2S scale re-seeding, P1L length extrapolation, P1T topk sweep |
| `explib/scale/` | **the v8 driver** (`python -m explib.scale`): P1S (scale panel seed completion to 4 seeds) and P2R (RoPE+QK-norm 20k-step long run) |
| `explib/resume/` | **the v9 driver** (`python -m explib.resume`): P2T (the seq-2048 topk sweep at `batch_size=1`) and P2S3 (third seed for the RoPE 20k run) |
| `explib/corruption/` | **the v10 driver** (`python -m explib.corruption`): P3M length-extrapolation mechanism + distractor-injection probe, P3S inversion-crossover scaling, P3T topk sweep to 4 seeds |
| `explib/length/` | **the v11 driver** (`python -m explib.length`): statistical close-out — distractor probe to n=6 (P4MT/P4MP), crossover panels to n=4 (P4S), d=384 paired arms to n=4 (P4F) |
| `explib/confirmation/` | **the v12 confirmation driver** (`python -m explib.confirmation`; reviewer-response + review-2 hardening): P5C trains all four frozen arms at the manifest seed list, P5MP runs the extended corruption-mode probe (uniform/realistic-token/mask/delete + review-2 span/doc/topical + NIAH-natural), P5S2/P5S3 add the token-matched and 6ND-param-proxy scale arms, P5T performs the FROZEN final evaluation on the held-out WikiText-103 test split per `confirmation_frozen.json` — with a load-time plan-power gate (a plan that cannot mathematically confirm is refused), strict record admission (seed whitelist / unique records / one frozen test slice / recipe signature / manifest sha), a first-access freeze lock, and per-document PPL records |
| `explib/open_block/` | **open-block panel driver** (`python -m explib.open_block`): maximum block length (32/64/128/no cap) × current-block reading rule (`closed` / `prefix_summary` / `prefix_singletons`), with a shared top-k budget, causal prefix construction, outside-window auxiliary panel, paired per-seed deltas and interaction statistics — see `docs/design_notes.md` §24 |
| `test_open_block.py` | **open-block CPU gates**: config/segmenter, optimized-vs-reference prefix pooling (forward + gradients), prefix invariance, future perturbation/gradient, candidate masks, edge cases, closed regression, singleton semantics and HCA budget declaration |
| `test_review_part_*.py`, `test_review_fixes.py` | **CPU review pytest suite**: compatibility entry point plus the split test modules (run with `make test` or `pytest`) |
| `confirmation_frozen.json` | **pre-registered confirmation plan v2** (tracked): frozen arms, seeds 0-11 (revised 6->12 BEFORE any test-split access; power table recorded), endpoints, exact sign-flip + Holm, decision rule, machine-checkable binding block and a scope declaration — see `docs/design_notes.md` §17 and `DECISION_LOG.md` |
| `RELEASES.md` | **evidence release index**: where each immutable `release_<tag>.tar.gz` + SHA-256 manifest is published; the Git repo itself stays source-only |
| `pyproject.toml` / `requirements.txt` / `requirements.lock` | Python ≥3.12 and dependency declarations; install explicitly (no import-time pip install). `requirements.lock` is the verified CPU environment |
| `DECISION_LOG.md` | **decision log** (tracked; reconstructed 2026-10-18 from the README/design notes/manifest because the snapshot has one squashed commit): why each version ran what it ran, what was seen, and what was decided next — the exploration/confirmation paper trail; formal releases additionally archive per-seed results via `archive_release.py` |
| `audit_gates.py` | **compatibility shim** for `_audit_gates/`, the **review-2/4/7 verification entry points** (zero-experiment): data-leakage audits (tokenizer provenance, vocab identity, train/test overlap), CPU causality unit checks (prefix invariance, label shift, decode consistency), block-attribution / boundary-vs-random / length-consistency / baseline-fairness / failure-concentration / budget-reconciliation audits, the review-7 single-factor design gate (declared contrasts' arm-cfg diffs vs their declared intervention key sets; frozen-manifest mechanism_claim_allowed flags) and prefill-decode-kv-growth measurement utilities |
| `archive_release.py` | **immutable evidence archiver**: packs per-release artifacts (results, analyses, reports, freeze lock, decision log, environment, checkpoint hashes) into a tar.gz with a per-file SHA-256 manifest and an independent `verify` subcommand — the git repo stays source-only (design_notes §8) while every formal release gets an auditable fixed archive |
| `explib/m0/` | **the M0/S1 driver (adopted B tier)** (`python -m explib.m0`): entry points for the scale-up roadmap — `params` (zero-GPU S1–S4 parameter audit), `data` (FineWeb-Edu + 32k-BPE corpus cache), `smoke` (end-to-end self-check on the B-tier data path), `kernel` (the G0 validation suite: kernel/eager numerical comparison for the three operators plus the sparse-attention throughput benchmark), `full` (the S1 panel); see "B-tier M0/S1" below |
| `build_report.py` | **compatibility shim** for `_build_report/`, the zero-GPU report builder: regenerates the v7 report from `results_*/` + `analysis_*/` artifacts (every number is computed from disk; protocol constants and analytic FLOPs assumptions are explicit inputs, not measurements) |
| `fuse_analysis.py` | zero-GPU fuse-vs-no-fuse driver: recomputes per-layer boundary alignment (tol=1), block-length stats and PPL trajectories from a `results_*/summary.json` |
| `Makefile` | zero-GPU entry points: `make test`, `make plan`, `make audit`, `make install-check`, `make build`, `make clean` |
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
python run_gapfill.py smoke      # end-to-end self-check, zero GPU cost
python run_gapfill.py full       # the v6 main experiment (bounded by the run-time cap)
python -m explib.rope ...        # per-version supplementary runs; subcommands in each module's help
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
# The code no longer pip-installs at import time.  Set CSA_AUTO_INSTALL=1 only
# for a throwaway environment.  The Hugging Face mirror is opt-in via
# CSA_HF_MIRROR=1; the default endpoint is the official Hub.
#
# If the Hub is unreachable, `load_wikitext` falls back to a local raw dump:
# point `WT103_RAW_TXT` at a directory holding `wiki.train.raw` /
# `wiki.valid.raw` (the search also tries `./wt103_raw` next to the repo and
# `/root/wt103_raw`).  Only those two files are read, and tokenising them
# reproduces the `wt103_cache/*.npy` cache token-for-token, so the
# fallback is a pure availability path and never changes a measurement.
# When `wt103_cache/` is already populated the corpus is served from the cache
# and no dump or network access is needed at all.  Set `WT103_CACHE_DIR` to
# relocate the cache; the loaders, `audit_leakage` and P5T all resolve the same
# variable.  `python -m explib.confirmation smoke` deliberately uses an isolated cache under its
# smoke outdir so it can never leave a differently-tokenised file in the frozen
# cache.  Test-cache hits verify the persisted tokenizer bytes (sha256 recorded
# in test_meta) in addition to the train fit fingerprint.

# 1) validate the whole pipeline end to end (smoke run, no push):
#    (covers dense / fixed / dynamic arms AND the leave-one-out ablation
#     forward paths: random indexer, zero content, sink off, learned fusion,
#     HCA layers, dense+sink control)
python run_gapfill.py smoke
python -m explib.corruption smoke  # also covers the new corruption modes + NIAH + controls
python test_open_block.py         # open-block causality/reference/mask gates (CPU)
python -m explib.open_block smoke     # open-block matrix plumbing on a tiny synthetic run
python -m explib.open_block pilot     # one cell, short run
python -m explib.open_block main      # 8 cells x 4 seeds; resumable
python -m explib.open_block aux       # outside-window auxiliary panel
python -m explib.open_block report    # paired deltas + interaction + exact sign-flip/Holm
python -m explib.open_block extended  # contingency: all 4 caps x all 3 modes
python -m explib.confirmation smoke   # also covers the frozen final-test roundtrip

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
python -m explib.confirmation phase P5C       # train all four frozen arms at the manifest seed list (resumable; 0-5 main arms covered by P3MT/P4MT)
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
python audit_gates.py single_factor  # design-time gate: every declared contrast's arm-cfg diff equals its declared intervention key set; open-block matrix axes; frozen-manifest mechanism_claim_allowed flags vs the actual cfg diff
python audit_gates.py sesoi       # design-time gate: every ablation component has a valid declared SESOI and is fully scheduled
python audit_gates.py unit_influence  # result-analysis gate: unit-influence block reproduces pooled Δ; flags one-unit-driven comparisons
python audit_gates.py looks       # look-history gate: recompute per-look and exact max-over-looks sequential p; flags seed top-ups that only look significant at the final n
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

## Review-4 hardening (what changed)

See the review-4 entry in `DECISION_LOG.md` and `docs/design_notes.md` §25 for
the item-by-item record.  The main additions are machine gates rather than new
prose:

- **Claim wording** now goes through `exp_lib.reportable_claim` (n>=6, raw p,
  Holm p, |mean|>std, family resolution floor).  `attach_holm` reports
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
- **Doc-level uncertainty**: `ppl_ci95_doc_bootstrap` is displayed; v12 adds a
  per-comparison document Δ 95% CI and refuses population-level claims below
  64 documents.
- **Efficiency**: the length phase records `measure_efficiency`; the new
  `measure_efficiency_sweep` covers batch x length, and
  `audit_gates.efficiency_pairs` only permits a "faster" claim when token
  count, device, torch version, KV semantics match and the ratio exceeds ±5%.
  Analytic FLOPs are labelled structural, never as measured speed.
- **Mechanism**: `audit_gates.selection` defines the selection-recall artifact
  contract and reports `not_verifiable_locally` (with an explicit "unsupported
  mechanism claim" note) until that artifact exists.  Boundary alignment,
  block/read diagnostics and NIAH remain the available direct readouts.
- **Ablation practical significance**: every declared ablation component now
  carries a declared SESOI (relative PPL margin, default 1%; exploratory
  panels only) and is adjudicated with paired TOST plus family-adjusted
  equivalence p.  A
  non-significant contrast is never relabelled "removable" without TOST; a
  statistically significant but sub-SESOI effect is `sub_sesoi`, not
  `verified`; a TOST-inside-margin result on a COMPLETE declared ablation
  is `equivalent` (positive evidence of removability).  `audit_gates.py sesoi` fails closed if a component has no
  declared margin or the ablation is not fully scheduled.  This is exploratory
  and does not modify the frozen confirmatory manifest.
- **Single-unit influence**: the pooled ΔPPL now carries an exact
  leave-one-unit-out readout (`doc_sampling.unit_influence`): the most
  influential article/line, the Δ without it, direction stability across all
  folds, a delete-one-unit jackknife interval/SE and its ratio to the
  cluster-robust SE.  A comparison whose direction or magnitude depends on one
  evaluation unit is flagged in `REPORT_v12.md`; `audit_gates.py
  unit_influence` independently checks the block against the reported pooled
  estimate and warns on fragile/stale artifacts.  It is post-hoc and never
  changes the frozen p/Holm/`confirmed` verdict (design_notes §57).
- **Optional stopping / seed top-ups**: every aggregation with a new seed set
  records each measured declared ablation/baseline contrast in
  `<outdir>/look_history.json`.  When a contrast has more than
  one look, `methodology_audit` judges it by the exact **max-over-looks**
  sign-flip p in the final effect direction (joint sign enumeration over the
  seed union), and the family-Holm `verified` gate uses that sequential p; the
  naive final-n p is retained only as `p_holm_global_naive`.  TOST equivalence
  gets a Bonferroni-over-looks correction as well.  Stale looks (a re-run seed
  whose delta changed) are excluded and named.  `audit_gates.py looks`
  independently recomputes the per-look and sequential p and flags
  naive-significant / sequential-null claims.  This is exploratory only and
  never changes the frozen manifest (design_notes §58).
- **Evidence**: `audit_gates.release` parses `RELEASES.md`; with no complete
  row the repository is declared source-only/not auditable.  `archive_release.py`
  now also packs `pyproject.toml`, cache/tokenizer metadata and `run_*.log`.

## Review-3 hardening (what changed)

- **PPL aggregation**: training/length records now carry `nll_sum`,
  `n_tokens` and per-document NLL; `aggregate` reports `ppl_pooled =
  exp(ΣNLL/Σtokens)` as `ppl_primary` (with the arithmetic seed mean,
  median/IQR, seed-level t interval and document bootstrap kept alongside).
  The final PPL is still measured on the full fixed validation batch;
  `eval_subset` only controls the in-training curve.
- **Statistics**: v7–v11 analysis and `build_report.py` attach Holm
  step-down over the displayed family (omitted comparisons keep a dummy p=1
  slot). Report prose may say “significant” only when n≥6, raw p<0.05, Holm
  p<0.05 and |mean|>std.
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
