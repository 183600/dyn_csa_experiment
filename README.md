# Dynamic-Block CSA + HCA — WikiText-103 controlled study

Controlled study of compressed-block sparse attention (CSA / HCA, fixed and
dynamic blocking) vs dense attention in a small from-scratch LM (BPE-8k,
WikiText-103). This repository contains the experiment code and everything
needed to reproduce the results on a machine of your own.

## Repository layout

仓库只保留**源码、驱动脚本与设计文档**。所有实验结果目录（`results_*/`）、
零 GPU 分析目录（`analysis_*/`）、逐版报告（`REPORT*.md`）与预算账本
（`autodl_budget*.json`）都**不入库**——它们由下面的驱动脚本在跑实验的
机器上重新生成，需要时本地复现即可（见下一节的命令）。`.gitignore` 已把这几类
路径挡在仓库外（预算账本按前缀匹配，因为 `CostGuard.state_path` 由调用方
命名；`docs/` 只保留 `design_notes.md`）。

| path | what |
|---|---|
| `exp_lib.py` | **the library — and the source of truth**: model, CSA/HCA attention, pooling, indexer, segmenters, training loop, aggregation and statistics |
| `run_gapfill.py` | **the v6 driver**: the minimum-publishable supplementary runs, resumable, budget-guarded, pushes after every phase, auto-shuts the AutoDL instance down |
| `v7_supp.py` | **the v7 driver**: reviewer-response supplementary suite — P0R RoPE+QK-norm, P0W dense-warmup, P0E 40k-step long run, P2S scale re-seeding, P1L length extrapolation, P1T topk sweep |
| `v8_supp.py` | **the v8 driver**: P1S (scale panel seed completion to 4 seeds) and P2R (RoPE+QK-norm 20k-step long run) |
| `v9_supp.py` | **the v9 driver**: P2T (the seq-2048 topk sweep at `batch_size=1` with per-block gradient checkpointing) and P2S3 (third seed for the RoPE 20k run) |
| `v10_supp.py` | **the v10 driver**: P3M length-extrapolation mechanism + distractor-injection probe, P3S inversion-crossover scaling, P3T topk sweep to 4 seeds |
| `v11_supp.py` | **the v11 driver**: statistical close-out — distractor probe to n=6 (P4MT/P4MP), crossover panels to n=4 (P4S), d=384 `full` arm to n=4 (P4F) |
| `build_report.py` | zero-GPU report builder: regenerates the v7 report from `results_*/` + `analysis_*/` artifacts (every number is computed from disk, never hard-coded) |
| `fuse_analysis.py` | zero-GPU fuse-vs-no-fuse driver: recomputes per-layer boundary alignment (tol=1), block-length stats and PPL trajectories from a `results_*/summary.json` |
| `mon3.py` | tiny AutoDL progress watcher used by the drivers |
| `prep_cache.py` | one-off helper that builds `wt103_cache/*.npy` from the raw WikiText-103 dump |
| `bonus_watcher.sh`, `driver_v7.sh` | shell wrappers used on the AutoDL box (budget watch / v7 launch) |
| `docs/design_notes.md` | **design and implementation notes**: why the block-read gate is strict and must be synced across five sites, the dynamic segmenter's prefix-invariance requirement, the pairing/measurability rules, the reporting conventions, and the transient-bound contracts that keep attention row-chunking equivalent to its unbatched form |

> **`exp_lib.py` 是唯一的代码来源（source of truth），直接改它。**

### 重新生成实验产物

结果目录、报告与预算账本都不在仓库里。要重新得到它们：

```bash
python run_gapfill.py smoke      # 端到端自检，零 GPU 消耗
python run_gapfill.py full       # v6 主实验（受下方 cap 约束）
python v7_supp.py ...            # 各版本补充实验，子命令见对应脚本的 docstring
python build_report.py           # 从落盘的 results_*/ + analysis_*/ 重建报告
```

## Reproducing (AutoDL or any CUDA box)

```bash
# 0) environment — Python 3.12, torch 2.8.0+cu128, plus:
pip install datasets tokenizers matplotlib
# (the code also pip-installs these itself on import; HF_ENDPOINT is set to
#  hf-mirror.com inside exp_lib.py — change it if you are outside CN)
#
# If the Hub is unreachable, `load_wikitext` falls back to a local raw dump:
# point `WT103_RAW_TXT` at a directory holding `wiki.train.raw` /
# `wiki.valid.raw` (the search also tries `./wt103_raw` next to the repo and
# `/root/wt103_raw`).  Only those two files are read, and tokenising them
# reproduces the committed `wt103_cache/*.npy` token-for-token, so the
# fallback is a pure availability path and never changes a measurement.
# When `wt103_cache/` is already populated the corpus is served from the cache
# and no dump or network access is needed at all.

# 1) validate the whole pipeline end to end (no budget use, no push):
python run_gapfill.py smoke

# 2) the real supplementary study (hours of accelerator time, bounded by the cap below):
nohup python run_gapfill.py full > run_v6.log 2>&1 &
tail -f run_v6.log
```

`run_gapfill.py full` runs, in order (each phase is resumable; re-running the
same command skips completed `variant::seed` pairs):

| phase | output dir | what |
|---|---|---|
| P-LONG | `results_lm_v3_long` | hybrid_fixed / hybrid_dynamic / hybrid_csa_dyn at seed 2 → 3-seed long-run table |
| P-CORE | `results_lm_v3_1500` | 9-variant core table, 1500 steps x 3 seeds |
| P-SCALE | `results_lm_v5_scale` | d=384 / 8 layers / seq-1024 probe, 5 variants x 2 seeds x 4000 steps |

`full` books against the shared ledger described at the end of this section.

### v11 statistical close-out suite

```bash
python v11_supp.py smoke
nohup python v11_supp.py full > run_v11.log 2>&1 &
python v11_supp.py report         # rebuild REPORT_v11.md from disk (zero GPU)
```

Phases: P4MT (RoPE-arm seeds 3-5, exact P1L/P3MT recipe), P4MP (probe
resume for the new seeds, eval-only), P4SS/P4SL (crossover panels seeds
2/3), P4F (d=384 `full` seeds 2/3). Books against
`autodl_budget_state_v11.json` (default cap ¥30); `V11_*` env vars mirror
the earlier ones.

### v10 mechanism / scaling / stats suite

```bash
# v10: distractor-injection mechanism probe + crossover scaling + topk seeds
python v10_supp.py smoke
nohup python v10_supp.py full > run_v10.log 2>&1 &
python v10_supp.py analysis       # zero-GPU stats -> analysis_v10/
python v10_supp.py report         # rebuild REPORT_v10.md from disk (zero GPU)
```

Phases (each resumable): P3MT (retrain the two RoPE arms, the P1L recipe),
P3MP (eval-only far-distance distractor injection: paired corrupted
contexts, clean 512-token target, learned-selection vs dense vs
random-indexer vs see-everything arms), P3T (topk sweep seeds 2/3 → n=4),
P3SS/P3SL (d=128/4L 8000-step and d=512/10L 4000-step crossover panels).
See `docs/design_notes.md` for the design rationale behind the
paired-corruption protocol and the pre-registered crossover criterion. v10
books against its own ledger (`autodl_budget_state_v10.json`, default cap
¥60); `V10_*` env vars mirror the earlier ones.

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
and the pairing/measurability rules these phases rely on. v8/v9 book against their own ledgers
(`autodl_budget_state_v8.json` / `_v9.json`); `V8_*` / `V9_*` env vars mirror
the v6/v7 ones.

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

`gathered_attention` and `_block_token_attn` shrink their row chunk to fit a
per-layer transient budget (`_attn_transient_budget`), which is what lets the
seq-2048 topk sweep complete on a single accelerator card.

The CostGuard (`exp_lib.py`, SECTION 8.5) books GPU time to
`autodl_budget_state.json`, refuses to start runs that would pass
`BUDGET["total_yuan"] × margin`, and hard-truncates a run at the absolute cap.
Edit `BUDGET["price_per_hour"]` to match the rented GPU. When all phases
finish the script pushes and shuts the instance down (cancel with
`pkill -f v6autoshutdown`; suppress with `V6_NO_SHUTDOWN=1`, pushes with
`V6_NO_PUSH=1`).

## Two accounting notes (v6)

1. **`results_lm_v3_long/summary.json` is reconstructed** from an
   `aggregate.json` snapshot: reconstructed per-run records are flagged
   `"synthesized": true`. All reported **means** (PPL, block length,
   boundary F1, delta) are exact (mean-of-means identity); across-seed
   **stds of the secondary metrics** are approximate because the old
   per-seed spread is unrecoverable. PPL stds are exact.
2. **`results_lm_v3_1500` is a fresh directory** rather than an update of
   `results_lm_v3`: the old dir holds the 500-step v3 table, and mixing step
   counts in one summary would silently corrupt the paired statistics. The
   500-step table remains in `results_lm_v3` as the "budget regime" column.

## Headline results so far

> **量值一律以 `build_report.py` / `fuse_analysis.py` 从 `results_*/`、
> `analysis_*/` 现算出的表格为准**：正文不给量值，报告给量值。

- 短训（小预算）：CSA 早于 dense 收敛，且这个早期优势在全部消融臂上都不动
  （随机 indexer、content 置零、无 sink、top-k 8/64 …），说明它来自架构而非
  检索质量。
- 收敛（大预算、多种子）：优势**反转** —— dense 的渐近 PPL 更好，即
  "学得快、渐近差"。
- v7 reviewer response（见 `REPORT_v7.md`）：每个被质疑的混淆项都被证伪或给出
  上界 —— RoPE+QK-norm 对两条臂帮助相同（P0-3）；dense warmup 救不回稀疏臂，
  且过暖有害（P0-1）；长程跑显示差距**在收窄但未交叉**（P0-2，按保守口径陈述）；
  纯 DSA m=1 比压缩 CSA **更差**（P1：压缩不是瓶颈）。一个正向发现：CSA+RoPE
  的长度外推显著优于 dense-RoPE（P1L）。
- v10（见 `REPORT_v10.md`）：给上面那个正向发现补了机制 —— distractor 注入探针
  （配对远端上下文破坏 + 共享权重消融臂）显示外推稳健性来自**稀疏归纳偏置本身**
  （随机 indexer 同样抗噪；"全看得见"臂显著退化），而不是学到的检索质量；反转
  的交叉点在 4 个规模上都做了定位（方向：模型越大越早反转；log-log 斜率只作
  方向性证据）。
- 完整表格见各 `results_*/aggregate.csv`，图见 `summary.png` /
  `length_gen.png`。

> **这些产物不在仓库里。** 上面的 `results_*/`、`REPORT_*.md` 与
> `analysis_*/` 都由驱动脚本在跑实验的机器上生成，仓库只保留源码与驱动。
> 想复核某个数字，先按上一节重新生成对应面板，再跑 `build_report.py` /
> `fuse_analysis.py`。本节引用的 `REPORT_v7.md` / `REPORT_v10.md` /
> `REPORT_v11.md` 同理——它们是构建产物，不是仓库文件。
