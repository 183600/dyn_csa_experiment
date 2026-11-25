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
| v12 (`v12_supp.py`) | **freeze**: fixed-chunk CSA confirmation plan, manifest v5 with revision history v1–v6 (12 seeds, 4 primary comparisons, exact sign-flip + Holm, held-out test split, freeze lock, strict record admission, per-window PPL); v1 was rejected by the load-time power gate (6 seeds × Holm/4 has minimum attainable p=0.125) | confirmatory plan |
| review 2 | seed count revised 6→12 **before any test-split access**; scope block added; two-phase freeze lock; kernel mode refused for the frozen plan; evidence archiver; leakage/causality/fairness audits | frozen plan revision |
| review 3 | external audit (`科研审查_autodl运行决策.md`) found the confirmation path was not runnable: manifest stamp v11.126 vs code v11.127, `eager_batch_expected=true` with all four arms on the per-row RoPE loop, and P5T records missing `train_knobs` (admitted=0 after the one-shot test access). Manifest **v3** re-signed before any test access: stamp v11.128, eager binding false, P5T/probe records carry the provenance descriptor, per-arm `mr` pinned, equal-budget wording corrected, decision rule declares an n_effective floor of 8, adaptive 2-SE health threshold, reachable leakage statistics, and `v12_supp.py plan` as a zero-GPU CI pre-flight. The continuous-loop CI was disabled/cancelled during the fix. | frozen plan v3 + code hardening |
| review 3 (earlier same day) | pooled-NLL PPL + per-document NLL; pooled/bootstrap/Wilcoxon sensitivity; Holm in v7–v11 and `build_report`; objective/parameter/KV descriptors; opt-in task-learned boundary arm (`csa_dynamic_task`); synthesized records excluded from crossover; span-ΔNLL lookup fixed; kernel-stamp consistency; `DECISION_LOG.md` restored and guarded; dependency/Python declarations | code/evidence hardening |
| review 4 (2026-10-20) | no new experiment panels. Unified claim gate (`reportable_claim`), Holm family-resolution ceiling, NIAH folded into the Holm family, crossover fit refusal rules + slope CI, matched-init `init_core_fp`, measured compute-control direction (`system_level_only`), stale records fail-closed in `build_report`/v10, missing `truncated` treated as unverifiable, doc-bootstrap CI + <64-doc population gate, token-coverage estimate, efficiency sweep/pair audit, selection-attribution contract, evidence-release readiness audit. The frozen v12 manifest was NOT modified. | code/reporting/evidence hardening; all earlier panels remain exploratory |
| review 5 (2026-11-14) | external audit round 2 found no statistical or causality defects (exact sign-flip/Holm/TOST/Wilcoxon verified against independent references; 39-item causality suite passed; no data leakage). Fixed the residual guard/protocol gaps, all BEFORE any test-split access: (1) JS-divergence hard gate was DEAD (nats-scale threshold 0.75 > supremum ln2≈0.693) — gates now on the ln2-normalized scale with reachability validation; (2) `load_wikitest` cache-hit path compared only the alphabetically-first train tokenizer — now compares EVERY one, same as the cold path; (3) RoPE csa/hca layers silently ran eager under a kernel-enabled process (stamp pollution) — now refuse loudly; (4) `V12_ALLOW_UNHEALTHY` override persisted to artifacts (training_health.json + per-record `allow_unhealthy_override`); (5) ngram floor dual denominator (`ppl_matched_denominator` on the LM's R×(L−1) positions) and the audit compares on the matched scale; (6) test-internal duplicates no longer miscounted as train/val contamination (`_DocDeduplicator.snapshot/peek` attribution); (7) freeze lock mirrored to a git-tracked `confirmation_freeze_lock.json` — renaming/deleting the results dir can no longer silently reset the freeze; (8) post-access selective re-measurement closed: P5C refuses to retrain frozen arms after the lock exists, and P5T refuses to overwrite a cell measured on different `ckpt_sha256` (V12_REFREEZE=1 is the explicit, recorded exit); (9) implemented decision rule disclosed as stricter than the manifest wording (`decision_rule_applied`); (10) manifest **v4**: ee/es removed from the machine-checkable signature (P5C/p1l carries es=8 under the P5H-required diagnostics, so the v3 es=128 pin was uncheckable and wrong), tokenizer 8M-token fit cap documented; (11) pre-access power simulation implemented (`v12_supp.py power`, sign-flip enumeration on validation deltas) and required by P5T (V12_SKIP_POWER=1 recorded override); (12) probe records carry `manifest_sha256`, and probes skip health-gate-refused cells; (13) watchdog-killed cells book their real wall-clock into the TimeGuard ledger; (14) `eval_span` required at admission; bylen cells verify the checkpoint architecture/mr against the frozen binding; direction_ok refuses mean==0; misc robustness (successor-walk power-of-two invariant, `_pool_cpu` on the cosine n<=1 path, dead `_SinkWiden`, adaptive grid cap in the run fingerprint when it can affect values, `holm_min_attainable_p` overflow guard, kernel bincount int64). `CODE_SEMANTICS` stays `v11.128`: every change is a guard, disclosure or metadata field — no measurement semantics on any valid path changed, so no existing record is invalidated. | frozen plan v4 + guard hardening; no panel results touched |
| review 6 (2026-11-14, independent) | fresh full-repo methodological audit (statistics, kernels, data pipeline, drivers, report layer). Fixed: zero-width soft prior doubling every attention gradient (eager + kernel twin; latent — no scheduled arm uses index_topk=0); v12 CONFIRMED gate had drifted stricter than the pre-registered manifest v3 rule (param_matched is True + init-digest gates + a report fallback that re-required coverage_ok) — restored to the frozen text via one shared helper (the review-5 loop pass had instead kept the stricter implementation with a `decision_rule_applied` disclosure; this pass reverts the implementation to the registered rule — the manifest wording was never revised — and re-points that disclosure at the restored semantics); `_classify_group_status` never returned `measured` (powered non-noise contrasts were mislabeled `noise_level`); the v6 long-panel merge pooled/paired records across different run_cfg stamps; v8/v9(/v10/v11) driver-level `csa_fixed - full` contrasts published exact p-values without the parameter-control disclosure (arms differ ~24% in parameters); `pair_reason` accepted boolean fingerprints on both sides; report length table dropped JSON integral-float spellings; v12 smoke cache isolation was overridable; `load_wikitext` required the Hub client even on cache-hit/local-dump paths. | code/methodology hardening; frozen manifest NOT modified; no test-split access |
| review 7 (2026-11-14, independent) | six-track independent audit (statistics / data-leakage / v12 frozen-plan faithfulness / training+aggregation / attention algorithms / drivers+reports, ~25k randomized numeric cross-checks). Algorithm core verified CLEAN (causality read-gate, pooling, indexer, gathered attention, RoPE, dynamic segmenter prefix-invariance, kernel twins — no defects). Fixed: (1) `scaling_trend` heterogeneity Q was NOT variance-normalised (raw `sum(n*resid^2)` in effect units) — its chi2 "p", I2, tau2 and the `heterogeneous_trend` verdict flipped under a pure change of the effect unit (log10 vs ln vs percent); now a lack-of-fit F test normalised by the pooled within-scale pure-error variance (exact via the incomplete-beta F sf; fail-closed "unavailable" when no within-scale replication); unit-invariance regression test added; (2) manifest **v5**: the v4 revision note still claimed a strictly TIGHTER implemented decision rule, but review-6 had restored the implementation to the registered v3 text — v5 corrects the historical record (no normative field changed; precondition: still zero test-split access); (3) `load_wikitext` cache-hit path did not re-validate `tokenizer_fit_token_cap` against the current `min(TOKENIZER_FIT_TOKENS, n_train_tokens)` (stale-fit cache served under the same tag if the constant ever changes) — now a fail-closed rebuild check; (4) `par.py` watchdog could overwrite a child's valid `result.json` (written in its finally block before teardown) with `watchdog_timeout` AND double-book the cell's already-refunded TimeGuard seconds — now adopts the valid result and skips re-booking (verified by an integration simulation both ways); (5) `aggregate()` silently shrank `n_seeds` when planned seeds carried no usable measurement — entries now disclose `n_dropped_error_no_ppl` / `n_dropped_nonfinite_ppl` / `dropped_seeds` / `n_dropped_synthesized` plus a NOTE line; (6) `chi2_sf` exact branch truncated at 1000 series terms (rel err 8e-6 at df=1e5) and the Wilson-Hilferty branch (df>1e5) had up to 2.7% tail error — adaptive iteration cap (20*sqrt(a)+100) and the approximation now reserved for df>1e8 (verified <1e-7 vs scipy through df=1e8); (7) `_doc_bootstrap_ppl_ci` did not drop zero-token rows (comment claimed it did; `_combined_seed_doc_ci` did) — now aligned; (8) `measure_efficiency` prefix-prefill loop timed the first call at each new length (one-time kernel selection inside the p50/p95) — one untimed warmup per length added; (9) docs: `_DocDeduplicator` docstring overstated near-dup sensitivity (measured: 1-word edit of a <=30-word line exceeds Hamming-3) — reworded to the measured net; freeze-lock mirror `.gitignore` whitelist made explicit; README/README_zh stale "CI is disabled" claim replaced with the documented requirement + a "verify the actual state" instruction; `final_minus_best` mixed-basis disclosure field added; (10) dead code removed (`fuse_analysis.signflip`, `audit_gates` unused `ra, rb`, `par._KILL` write-only flag) and `error_analysis._compare_reason` now checks checkpoint A's position-table capacity symmetrically. `CODE_SEMANTICS` stays `v11.128`: every change is an analysis-layer bug fix, a guard, a disclosure field or documentation — no measurement semantics on any valid training/eval path changed, so no existing record is invalidated. | frozen plan v5 (documentation-only revision) + statistics fix + guard hardening; no test-split access |

| review 8 (2026-11-23, patch merged onto the loop HEAD) | three review fixes rebased onto the current main: (1) the closed CSA straight-through S path masks zero-probability compressed-block lanes before softmax, so a lower-precision all-zero prior can no longer leave every usable block logit at one finite sentinel and leak a uniform-softmax gradient into q/K (sliding-window lanes untouched); (2) `gathered_attention_open` masks zero-prior candidate lanes before AND after the log-prior add (a finite MIN sentinel can keep the add's identity gradient while rounding to the sentinel); (3) the RoPE boundary fallback scores sampled Bernoulli actions with the original pre-sigmoid policy logits instead of `logit(clamp(sigmoid(z)))`, whose derivative is zero once the gate saturates. Tests N68-N71 are registered in `test_review_fixes.py`; the closed-path regression uses a float32 prior on float64 logits because a same-dtype MIN sentinel saturates the add and `clamp_min_` already cuts the gradient (the original window-only case passes either way). No new runs, frozen manifest untouched, no test-split access. | code hardening; exploratory panels unchanged |

| review 9 (2026-11-25, independent re-audit) | full verification of an external review's defect list against the current HEAD before fixing anything. Confirmed ALREADY FIXED upstream (no change needed): all six par.py dispatcher claims (parent-logical→token `CUDA_VISIBLE_DEVICES` mapping incl. UUID/MIG, worker-index slot scheduling so CPU workers truly run in parallel, validated+deduped `CSA_PAR_GPUS` with no silent all-device fallback, `mkdtemp`-unique run dirs, fail-closed rc=0-but-invalid results via `_INVALID_RESULT_STATUSES`, atomic result/job writes via `_atomic_json`); the RoPE boundary fallback honoured-mask length (`T-1`, with the empty-sequence `gate=None` path); GEMM bitwise scoping (`lightning_indexer` refuses causal-ends narrowing because a GEMM is not bitwise column-independent across shapes, the eager batched twin is declared non-bitwise and carried in `training_knobs` rather than the `_code` stamp, and the environment snapshot records per-GPU UUID/arch plus precision flags); and both claimed hotspots (`causal_adaptive_threshold` is 128 MiB chunk-budgeted with the 8192-token exact cap and a 1024-sample long-context path; `lightning_indexer` fuses selection per query chunk and never materializes the full [n,B] score table). Real defects fixed in this pass: (1) import-time `os.chdir(REPO)` in `v7_supp`/`run_gapfill`/`m0_supp`/`baseline_tune`/`error_analysis` violated the design-notes §7 import-purity contract (test isolation, foreign-cwd library imports) — moved to each CLI `__main__` entry; `par.py` children already run with `cwd=REPO`, so no dispatch path changes; (2) design_notes §22.5 now states the complete straight-through gradient contract: the indexer weights get their ONLY gradient via the soft S path, while the main attention path (q projection, KV pooling/compression/head projections, sink) receives BOTH the hard and the soft gradient — previously only the indexer half was documented. `CODE_SEMANTICS` stays `v11.128`: no measurement semantics on any valid training/eval path changed, so no existing record is invalidated; frozen manifest untouched; no test-split access. | import hygiene + documentation; exploratory panels unchanged |
| review 10 (2026-11-25, external methodological review) | verification-first pass over the reviewer's 15-item list against the CURRENT HEAD; no training panel, no frozen-manifest byte, no measurement-semantics change (`CODE_SEMANTICS` stays v11.128). Already correct / factual corrections recorded: the score-function sign is `+(CE-baseline)*logpi` in both docs and code (not negative as the review claimed; now extracted to `exp_lib.boundary_policy_term` with a two-action toy test `t_score_function_term_sign_two_action_toy`); the default `csa_dynamic` is documented as NOT task-learned (§22.5, README, here); the random-indexer control is still the FIXED deterministic hash `(row*12.9898+col*78.233)` in HEAD and every ref (the claimed per-(seed,layer) change does not exist; the reviewer read a stale snapshot); the Holm family is m=4 (not six); the frozen test panel is 16 windows (not 48 documents; 48 = arms x seeds). Real gaps fixed: (1) evidence status banner in every report builder from the `RELEASES.md` index (source-only => numbers not independently auditable); (2) frozen manifest caveats + `compute_budget_disclosure` printed next to the v12 primary table, per-comparison `single_factor_contrast=false` / `system_level_only=true`, and CONFIRMED wording limited to system-level differences; (3) honest parameter accounting: `param_breakdown` reports `dead_params`/`active` for random-indexer arms (202,752 dead, ~1.83%), `finaltest_stats` reports per-arm active counts/relative gap, and the report prints the fixed-hash/dead-parameter caveat; (4) `reportable_claim` split into statistical / stability / practical verdicts with |mean|>std demoted to a stability diagnostic, `build_report._sig_gate` updated, and the v12 CONFIRMED rule documented as NOT using that threshold (corrected the false 'same gate' claim); (5) explicit sign-symmetry null + assumption boundary + per-comparison sign counts/leave-one-out stability/within-seed-noise in `test_stats.json`; (6) power sensitivity (q_hat + Wilson lower bound, leave-one-seed-out and outlier-seed power) in `p5c_power.json`, `V12_SKIP_POWER=1` now additionally requires `V12_ACCEPT_LOW_POWER=1` when a power gate is declared and records `power_gate.json(status=skipped)`, report renders `accepted_low_power`, and the false 'symbol projection is an upper/lower bound' docstrings are corrected; (7) corruption audit: `_corrupt_row(return_audit=True)`/`_corruption_audit` record realised coverage/unique/overlap/changed/no-op/donor (spans were already disjoint; rho is nominal) and the report prints the realised-geometry table; (8) NIAH chance level + `nofar` no-far-information control + distance-bucketed accuracy + a task-validity machine gate that refuses mechanism wording without a positive control; (9) v12 test-probe Delta-NLL rebuilt from stored nll_sum/n_tok (same-arm clean baseline) instead of absolute PPL only; (10) post-test re-freeze cannot confirm: `round_is_confirmatory` forces `confirmatory_round=false`/`confirmed=[]` and the report warns that re-freezing does not restore test independence; (11) unit wording fixed (16 windows / 64-floor house convention / frozen `per_document` field erratum), length-extrapolation scope notes (nested-prefix targets, read fraction, alternative explanations) added to reports and §16; (12) broken `REVIEW_4_RESPONSE.md` references removed, README manifest-v2 and 'compute-matched' wording corrected, README_zh brought in line. No arm, seed, endpoint, alpha, Holm family or test-split access changed; no released evidence exists, so no number is claimed auditable. | code/reporting/documentation hardening; frozen manifest untouched; no test-split access |

| review 12 (2026-11-25, external methodological review) | verification-first pass over 16 review items against HEAD; no runs, no test access, no frozen-manifest byte, `CODE_SEMANTICS` stays v11.128. Fixed real defects: `train_objective` ignored per-layer `comp_lambda_mult` (nocomp recorded as applied/0.05; hybrid 4x invisible) and `objective_reason`/v7/v9 pairs did not disclose objective differences; sign-test robustness (paired/unpaired bootstrap, Wilcoxon, sign counts, assumption) was computed then discarded by `aggregate`/v7/v9; v7 length and v10 P3MT/P3MP used arithmetic seed means instead of token-pooled PPL and still carried mechanism wording; v10's global BH/Bonferroni family omitted its scale/equal-token p-values and P3T lacked `p_Holm`; v9 had no global view; release-row parser accepted any seven non-placeholder cells (`aaaa`/`bbbb` => released); archive omitted the TimeGuard ledger and could certify an analysis-only bundle; no report-to-input binding. Added `sec_resources`, report input manifest, N98–N103 regressions. Design limits (exploration/confirmation mixing, small n, multi-factor contrasts, validation reuse, source-only) remain documented, not disguised. | statistics/reporting/evidence hardening; frozen manifest untouched; no test-split access |

| review 13 (2026-11-25, six-item review) | verification-first against HEAD; no runs, no test access, frozen manifest untouched. Fixed: v10 P3MP never probed the P3MTC scratch controls (`PROBE.variants` omitted them), ΔNLL had no trained-control rows, and the by-length table hid their clean records; scale report lacked p floors / numeric token ordering / per-arm 6ND alignment / capacity and LR confounds / insufficient-eval wording; distractor report lacked an explicit clean-far benefit table; `allblocks` was still called an item-wise aligned dense control despite keeping compressed KV/pooling/cosine-qk, and the missing q/k + scale disclosure was added to the v12 claim scope; a valid release could break CI because the test hard-coded `no_release`; added `archive_release.py record` and git diff/status hashes. Residual design limits (no release, 2–6 seeds, no same-projection dense run) remain documented. | reporting/wiring/evidence hardening; no test-split access |

## 2026-11-25 — review 13: six-item verification (scale confound / mechanism controls / distractor / small-n / architecture / reproducibility)

逐项判断与处置（全部对照 HEAD，没有新增训练面板、没有读 test split、没有改冻结清单）：

1. **规模混杂 — 部分成立，已把控制做进报告。** P3SS/P3SL 的 49.15M vs 24.58M token 差异仍存在；`P3SL_TOK`（等 token）与 `P3SL_FLOP`（6ND 代理，非实测 FLOPs）已在代码/报告中，但 v10 的 scale 表原先不显示 p 下限、按字符串排序 token 切片、不显示逐臂 6ND 比值、不显示容量差/LR 轨迹/评估点不足。现在表内加入预算口径、总步数、参数差、p 下限、平台化种子，切片按 token 数值排序并分「LR 对齐/仅 token 对齐」两组，另列逐臂 6ND 比值；缺失时明确写「控制未产出，端点仍受混杂」，不写「优势随规模反转」。
2. **机制对照 — 代码已有 P3MTC 同预算从头训练，但 v10 P3MP 从未探测它们。** `PROBE.variants` 只含两个主臂，导致 `retrain-gap`/`csa_randtrain` 在 v10-only 运行中永远不可测。现在 `PROBE` 纳入两个 scratch 变体并只跑其 `learned` 臂（避免再次引入测试时替换的混杂），ΔNLL 增加两条从头训练对照，by-length 表并列 P3MTC 记录；缺数据时报告明确写「对照未测量」。未重训，故这些行仍是待产出。
3. **干扰鲁棒性 — 设计已有 mask/delete/NIAH，但缺一个把「远端收益」直接读出来的表。** 新增 P3M-3b：收益 = NLL_污染 − NLL_干净（= ΔNLL），正值表示干净远端有用；与干净 NLL 并列，用于区分「鲁棒」与「本来就不用远端」。报告同时列 7 种腐坏模式并声明 `doc` 是文档级自然干扰、`uniform` 不等于自然干扰；NIAH 仍受正对照缺失的机器门限制，只作描述性。
4. **小样本/多重性 — 部分成立，已把分辨率写进 scale 表。** v10 scale 臂只有 n=2（floor 0.500），v11 补到 n=4（floor 0.125），都达不到 0.05；现在两张 scale 表都有 p 下限列，并在 floor>0.05 时打印警告。仍不把探索性 p 值升级为确认；确认路径仍是 v12 冻结清单 + 独立 test split。
5. **dense/CSA 差异 — 成立，已停止虚假对齐声明。** `csa_fixed_rope_allblocks` 仍保留 KV 压缩/池化、cosine q/k + scale=1.0、sink、滑窗，**不是** dense 对照；v10/v7/README/design_notes 中「逐项对齐的 dense」已删除，`arch-aligned` 对比改名为 `system-level`，并在 v12 claim scope/报告里补上 q/k 归一化（CSA：L2→RMSNorm；dense：仅 RMSNorm）与 scale（1.0 vs 1/sqrt(d_head)）的披露。另注册了未训练的 `full_rope_cos`（cosine scale 的 dense），但它仍不是精确单因子对照，必须随残差一起报告；真正同投影/同归一化/同 sink 的 dense 对照仍需新训练。
6. **公开可复核性 — 成立，但本轮只能修机制不能变出结果。** 仓库仍无 `results_*`，因此没有可发布归档；已修复「一发布 release 就挂 CI」的硬编码 `no_release` 测试，新增 `archive_release.py record`（verify 通过后把 URL/两个 SHA-256/commit 绑定进 RELEASES.md），归档 manifest 增加 `git_diff_sha256`/`git_status_sha256`，README 补 `wiki.test.raw`、纠正「每份报告都有输入清单」和「v12 已做逐文档聚类」的过度声明。独立 test split 路径本身已存在（`load_wikitest` + 一次性冻结锁），但 16-window slice 不是全 test split。

测试：N104–N109。`CODE_SEMANTICS` 不变（无测量语义改动）；冻结清单未改；未读 test split。

## 2026-11-25 — review 12: 16-item verification; statistics/reporting/evidence defects fixed

逐项处置见 `docs/design_notes.md` §30。没有新增训练面板、没有读 test split、
没有改冻结清单字节、没有改测量语义（`CODE_SEMANTICS` 仍为 `v11.128`）。

审查的 16 条中，**设计边界类成立但不能用代码消除**：v6–v11 的探索/确认混合、
2–6 seeds 的分辨率、非正交多因素对照、validation 反复用于选择、source-only
无发布工件。其余多数已在 v10/v11 硬化；本轮修掉仍能在 HEAD 复现的缺陷：

1. `train_objective` 漏掉每层 `comp_lambda_mult`：`csa_dyn_nocomp`（mult=0）
   仍记录为 `comp_reg_applies=True`、有效权重 0.05；hybrid HCA 的 4× 不可见。
   现在记录每层有效权重，`objective_reason` 比较向量；v7/v9 对比携带
   `objective_matched`/`objective_differences` 并打印系统级边界。
2. 符号翻转的稳健性字段被丢弃：`aggregate`、v7 stats、v9 `add_paired` 现在
   落盘 sign counts、p 下限、配对/非配对 bootstrap、Wilcoxon、零假设文本；
   v7 `sec_stats` 并排列出，且声明族不会收缩（缺失槽位 p=1）。
3. PPL 聚合：v7 长度节、v10 P3MT/P3MP 改 token-pooled；v9/v10/v11 面板表加
   pooled/median/IQR；长度节改为描述性读数并删除「机制一致/滤噪」措辞。
4. 多重性：v10 全局族补 scale/equal-token p；P3T 补 `p_Holm`/下限；v9 新增
   全局 BH/Bonferroni；P2S 补 `p_Holm`/下限。
5. 资源披露：新增 `sec_resources` 逐臂打印 total/attention/indexer/MLP/
   active/dead/layers/KV/实际压缩权重/ΔLR，并声明参数量匹配 ≠ 计算/有效容量匹配。
6. 证据：release 行 schema 严格化；归档补 TimeGuard 账本、拒绝 analysis-only
   归档、补全环境快照；报告新增输入清单（逐文件 SHA-256 + git HEAD）与
   `analysis_v7/report_inputs.json`。
7. 措辞：P0-3 去「证伪」；boundary-F1 去「genuinely semantic」；v10 标题去
   「机制验证」；「预注册」改为探索性前声明；v10/v11 的 synthesized 断言改为
   扫描整份 summary。

测试：N98–N103。`CODE_SEMANTICS` 不变，因为所有改动都在统计/报告/归档层，
没有改变任何有效训练/评测路径上的数值语义。

## 2026-11-25 — review 11: verification-first disposition; real stale/paired-statistics defects fixed

An external methodological review (21 numbered concerns + a "keep" list) was
checked item by item against HEAD before changing anything.  No training panel
was run, no test split was read, `confirmation_frozen.json` was not modified,
and no measurement semantics changed (`CODE_SEMANTICS` stays `v11.128`).  The
full disposition table is `docs/design_notes.md` §29; the important split is:

**Real code defects fixed (this is the part that changes what the code is
allowed to say):**

1. **Paired-panel grouping bug.** `build_report._pair_condition_tail` included
   the per-SEED `init_core_fp` in the key used to group seeds for one sign-flip
   test.  A correct panel has a different init digest for each seed, so every
   correct panel was split into one group per seed and `paired_row` returned
   n=1 (P0R/P1T/P2S).  The digest is now checked only within a seed pair by
   `_pair_reason`, which is the correct level.  Regression:
   `t_report_paired_row_per_seed_init`.
2. **Stale records were still quotable through report paths that did not check
   the code stamp.**  `sec_stats` rendered an old `analysis_v7/stats.json`
   unchanged; `sec_training_health`/`sec_efficiency` did not filter `_code` or
   `synthesized`; `per_seed_records` (and therefore P2S) had no code filter;
   v8/v9/v10/v11 panel readers defaulted to `require_code=False`.  All of these
   now fail closed on a stale/missing stamp (`sec_stats`, `sec_flops`,
   health/efficiency, `per_seed_records`, `ppl_by_seed(require_code=True)`,
   `v9_supp.add_paired`).  Regression: `t_report_stale_records_are_not_quoted`.
3. **v10/v11 family multiplicity.**  v10 discarded `attach_holm`'s family-level
   resolution metadata and v10/v11 had no cross-panel FDR view; v10's ΔNLL/NIAH
   tables printed raw p under a "Holm" heading.  The reports now publish the
   family floor (`m * 2/2^n`), a BH-FDR + Bonferroni view over all displayed
   p-values, and `p_Holm` columns in those tables.
4. **Mechanism wording.**  v10/v11 said the learned selector had an
   "independent contribution" and used "mechanism conclusion" headings; fuse
   analysis used the same heading.  These now read as system-level contrasts
   with an explicit attribution boundary (random is a fixed hash, all-blocks
   changes reading scope/STE, dense changes representation/capacity, and the
   `audit_gates.py selection` artifact is absent).
5. **Stopping rule.**  The follow-up protocol now registers a single fixed-n
   look (`single_look_fixed_n`, `n_looks=1`,
   `interim_analysis_planned=false`); `validate_followup_design.py` refuses a
   `--require-ready` plan without it.
6. **Report hygiene.**  The three report builders that lacked the
   `evidence_status_banner()` now print it (`v9`, open-block, fuse); the
   training-health table no longer calls `eval_subset_rows` "documents"; the
   efficiency section states that the Pareto frontier is descriptive,
   cross-shape/device comparisons are not a speed claim, and energy is not
   measured.

**Acknowledged limitations, not code-fixable (must not be presented as
solved):** external validity beyond small WikiText models; parameter- but not
FLOPs-matched dense controls; 1500/20k-step budgets; 2–6 seed exploratory
panels; the frozen 16-window evaluation panel; the hard top-k/no-grad
selection path (unselected blocks get no attention gradient); the absence of
an oracle/soft-selection/selection-recall artifact; dynamic-block geometry
being only partially persisted; no Tobit/censored crossover regression; the
source-only repo having no released evidence archive; the CUDA environment not
being locked; and energy not being measured.  These require new runs, new
artifacts or a new pre-registration.

**Factual corrections retained so the next review does not repeat them:**
`eval_subset=128` is the in-training curve subset, not the final PPL (the final
PPL uses the full fixed validation batch; the frozen panel is 16 windows);
`n=2 -> R^2=1` and "no slope CI" are false at HEAD (the fit requires >=4
de-duplicated widths and reports CI/LOO; k=2 reports no R^2); import-time pip
install and the HF mirror are opt-in; Holm within families and a cross-panel
BH/Bonferroni audit exist; window bootstrap exists; `48` is 4 arms x 12 seeds,
not 48 documents; and absPE truncation is flagged, not silent.

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
table was historically called `REVIEW_4_RESPONSE.md` (not present in this source snapshot; see the review-4/review-5 rows here), and the statistical conventions are in
`docs/design_notes.md` section 25.

## 2026-11-14 — review-6 independent full-repo audit (no new runs; rebased on the review-5 loop pass)

A fresh audit of the whole tree (statistical functions numerically verified
against scipy / exhaustive enumeration; kernel reference twins against naive
independent implementations; the data pipeline exercised end-to-end on a
synthetic local dump; every driver and the report layer read against their own
contracts) confirmed the statistical core (Holm, exact sign-flip incl. the
MITM/saddlepoint paths, Wilcoxon, bootstrap CIs, TOST, crossover fits,
pooled-PPL aggregation) and the leakage-critical contracts (test split read
only by P5T, train-only tokenizer fit, split disjointness, freeze-lock
ordering, resume semantics). It also found and fixed eight real defects. No
experiment was re-run, no numeric artifact was regenerated, the frozen
manifest was NOT modified, and no test-split access occurred.

1. **Zero-width soft prior doubled every attention gradient**
   (`exp_lib.gathered_attention`, `gathered_attention_open`,
   `csa_kernels.gathered_attention_kernel` / `_attn_bwd_ref`). With
   `index_topk == 0` the indexer returns a width-0 `soft` tensor, not `None`;
   the straight-through branch then adds an S path numerically identical to
   the hard A path (`soft_logits == logits` when there are no block lanes), so
   `dq`/`dk`/`dsink` were exactly 2x the contract (verified numerically in
   fp64 on both the eager path and the kernel reference twin). Width-0 is now
   normalised to "no prior" at all entry points (forward values bit-identical,
   gradients 1x). Latent only: no scheduled arm uses `index_topk=0` (the
   smallest is topk8), so no existing record was affected and
   `CODE_SEMANTICS` stays v11.128 — bumping the stamp would retire every panel
   for an edge case no panel measures.
2. **v12 CONFIRMED gate had drifted off the pre-registered rule.** A
   continuous-loop commit of 2026-11-12 changed `param_matched is not False`
   to `is True` and added `same_init_core_fp`/`init_core_fp_recorded` gates,
   and the report's legacy fallback additionally re-required `coverage_ok` —
   conditions the manifest v3 decision rule does not contain (v3 explicitly
   removed the all-12-seeds veto, and the manifest records the init digest as
   reported instrumentation, "reported and compared"). A
   stricter-than-registered gate is post-freeze drift: it can only fail to
   confirm what the pre-registered rule confirms. The primary gate and the
   fallback now call one shared `v12_supp._frozen_rule_confirms` implementing
   the manifest text exactly (strict `p_holm < alpha`, direction match,
   `param_matched is not false`, n_effective floor), the pinning test was
   rewritten to the frozen semantics, and the loop pass's
   `decision_rule_applied` disclosure now documents the restored rule
   (including which gates are deliberately NOT decision inputs).
3. **`exp_lib._classify_group_status` could never return `measured`.** The
   fall-through after the resolution-floor loop returned `noise_level`, so a
   controlled, powered, non-noise, resolution-OK, not-significant contrast was
   published in the bucket whose printed meaning is "effect within seed
   noise" — a false null reading. The final return is now `measured` (per its
   own docstring), and the descriptive `measured_components` tier no longer
   double-lists such a component.
4. **The v6 long-panel merge mixed configurations.** `merge_long_aggregate` /
   `_fold_backup_into_aggregate` (`run_gapfill.py`, the opt-in V6_SYNTH=1
   path) folded pre-retrain snapshot seeds into freshly re-measured entries
   and recomputed `dPPL_vs_full` across arms with different run_cfg / code
   stamps, with no identity check — exactly the pooling design_notes §3.1
   forbids. Both folds now require one shared `run_cfg` fingerprint
   (missing/differing => the fresh entry stands alone, loudly), and the
   paired-delta repair only runs within one fingerprint. Same-config folds
   (the recovery purpose) are unchanged.
5. **Driver-level contrasts lacked the parameter-control disclosure.** The
   v8/v9 `add_paired` helpers (also used by v10/v11) publish exact sign-flip
   p-values for `csa_fixed - full`-style contrasts whose arms differ ~24% in
   total parameters (11.22M vs 9.05M at the shared d=256/6L geometry) — above
   the repo's own 1% gate and without the `system_level_only` labelling
   `methodology_audit` applies to declared designs. The contrasts stay (they
   are honest system-level comparisons), but every stats record now carries
   `params_a`/`params_b`/`param_rel_diff`/`param_controlled` via the new
   `exp_lib.paired_param_control`, the console output discloses
   "capacity-confounded: system-level" when the gate fails, and
   `param_controlled is None` (unverifiable) is disclosed as well.
6. **`pair_reason` accepted boolean fingerprints on both sides.**
   `data_fp=True` on both records mapped to absent-vs-absent and paired as if
   controlled, while `assert_shared_validation_fp` correctly raises on a
   boolean `val_fp`. A boolean on either side is now `unverifiable` (also for
   `init_core_fp`).
7. **Report length table dropped JSON integral-float spellings.** `_len_key`
   promised "integral floats validated by value", but the string branch — the
   only one a JSON-loaded panel can reach — rejected "1024.0" and silently
   rendered the cell as "—". The canonical integral-float spelling is now
   accepted; fractional keys, huge lossy floats and duplicate spellings stay
   rejected/ambiguous.
8. **v12 smoke cache isolation was overridable** (`os.environ.setdefault`):
   an exported `WT103_CACHE_DIR` would redirect the smoke's cold build into
   the frozen cache — the exact hazard the guard exists for. The override is
   now unconditional. Separately, `load_wikitext` imported the Hub client at
   function entry, so even a warm cache or the air-gapped `WT103_RAW_TXT`
   fallback required `datasets`; the import moved into the Hub branch (a pure
   availability fix, no measurement semantics change).

Known items, reconciled with the review-5 loop pass that landed while this
audit was in flight: the frozen `training_recipe_signature.ee/es` dead letters
this audit flagged are now RESOLVED by the loop's manifest **v4** (pre-access
revision: ee/es removed from the machine-checkable signature, exactly because
they gate diagnostic-only evals that never touch the weights or the frozen
endpoint; `_recipe_matches_signature` keeps a compatibility skip for v3-era
manifests). The loop's decision-rule handling is superseded here: it had kept
the drifted stricter implementation with a `decision_rule_applied` disclosure,
while the manifest wording was never revised — this pass restores the
implementation to the registered rule (`param_matched is not false`, no
init-digest gates, no coverage_ok veto; one shared
`v12_supp._frozen_rule_confirms` for the primary gate and the report fallback)
and re-points the disclosure at the restored semantics. Still known-and-
accepted: `load_wikitest`'s cache-hit cross-binding degrades to a NOTE when
train-side cache files are absent (the loop pass additionally made the
cache-hit path compare every train tokenizer; the mandatory leakage audit and
P5T admission still refuse unverifiable states, so the confirmation path is
not exploitable through it).

Regression tests: `test_review_fixes.py` checks M1-M7 (zero-width prior,
boolean fingerprints, measured tier, cross-config merge, float length
spellings, param disclosure, frozen-rule helper) plus the rewritten
`t_audit_resolution_classification` and
`t_finaltest_decision_rule_matches_frozen_manifest`; full suite green
(253 + 15 checks).

## 2026-11-23 — review-8 zero-prior S-path and RoPE REINFORCE fixes (no new runs; frozen plan untouched)

Three implementation defects reported by executable CPU tests were fixed on top
of the current loop HEAD. No experiment was re-run, no numeric artifact was
regenerated, `confirmation_frozen.json` was NOT modified and no test-split
access occurred. Forward values on every valid path are unchanged; only the
straight-through gradient of zero-prior lanes and the saturated boundary-policy
gradient change.

1. **Closed CSA zero-prior lanes** (`exp_lib.gathered_attention`). A block lane
   whose prior is exactly zero is absent from the S-path support. With a
   lower-precision prior (e.g. float32 prior on float64 logits) the sentinel
   stays finite, the `add_` + `clamp_min_` pair keeps an identity gradient, and
   a row whose every usable lane is a zero-prior block lane turns into a
   uniform sentinel softmax — a spurious S-path gradient into q/K that the
   reference kernel twin already refuses. Zero-prior block lanes are now
   masked before softmax; sliding-window lanes are never masked because the
   prior does not apply to them. (With a same-dtype MIN sentinel the add
   saturates and the old code is already gradient-free, so the regression test
   uses the mixed dtype that exposes the defect.)
2. **Open-block zero-prior lanes** (`exp_lib.gathered_attention_open`). The
   open path adds the log prior without the closed path's `clamp_min_`, so a
   finite MIN sentinel can survive the add while keeping the add's identity
   gradient. Zero-prior candidate lanes are now masked both before and after
   the add; the post-add mask makes them constants in the S path. The hard A
   path is untouched.
3. **RoPE REINFORCE policy gradient**
   (`v7_supp.HybridAttentionRoPE._single_rope`). The fallback reconstructed the
   policy logits with `logit(clamp(sigmoid(z)))`, whose derivative is zero
   whenever the gate saturates at either clamp edge. The sampled Bernoulli
   action is now scored with the original pre-sigmoid affine logits, matching
   `exp_lib._single`.

Merge notes (the patch was written against an older main): the paper-trail
file `docs/code_review_fixes.md` is folded into this entry instead of adding a
new `docs/` file — the current `.gitignore` whitelists `docs/design_notes.md`
only, and `DECISION_LOG.md` is the tracked place for review rationale. The new
test functions were not registered in `main()` in the patch; N68-N71 now run in
CI. The closed-path test was strengthened: its original window-only case passes
both before and after the fix, so N68 adds the all-block-lane mixed-dtype case
that actually fails on the unfixed tree.

Deliberately not changed: the dynamic-boundary score formula, block/window
overlap semantics and the frozen experiment design. Those are method choices
that need an explicit scientific decision (and, where applicable, a
pre-registration amendment), not an implementation-only fix. Equal-FLOPs
controls, broader document sampling and full GPU/Triton validation remain
experimental follow-up work.

Regression coverage: `test_review_fixes.py` N68-N71 (closed mixed-dtype
all-block-lane zero prior, sliding-window S-path preservation, open all-zero
candidate prior, saturated RoPE REINFORCE). Local run on the rebased tree:
357/363 checks passed; the 5 errors are missing optional `matplotlib` /
`tokenizers` imports on this box and the single skip is a sandbox
POSIX-semaphore restriction — the CI job installs those dependencies. The
zero-GPU preflight (`python v12_supp.py plan`) still admits 48/48 records.

## 2026-11-24 — review-9: evidence-completeness gates, per-seed init fix and a minimum-power gate (no new runs; no test-split access)

An external source-level review raised nine methodological points. Seven are
design limitations that the repository already discloses and (where code can
enforce it) machine-gates; this entry does not relabel them:

- **No evidence release / non-auditable numbers** — `RELEASES.md` has no row,
  `results_*/` and `analysis_*/` are gitignored, and `audit_gates.py release`
  reports `no_release`. That is a deliberate source-only policy, not a hidden
  claim; nothing here makes a numeric claim auditable.
- **v6-v11 reused the same validation split** — recorded in the manifest
  `purpose` and treated as exploratory by construction; not promotable.
- **The confirmation covers fixed-chunk CSA only** — `scope.not_covered`
  names dynamic chunking, HCA and scaling laws, and
  `FOLLOWUP_CONFIRMATORY_PROTOCOL.md` / `followup_design.template.json` /
  `validate_followup_design.py` pre-register the 2x2 CSA/HCA factorial plus
  both dense references. That study is still a DRAFT (`TODO_FLOPS_MATCHED_DENSE`,
  no power simulation, gates unset) and must not be cited as evidence.
- **Parameter-matched, not compute-matched, Dense** — `compute_budget_disclosure`
  and every `interpretation_caveat` state the +69% MLP / ~1.87x FLOPs /
  25%->3.1% read-fraction mismatch; `finaltest_stats` refuses a non-descriptive
  plan that drops the disclosure.
- **Default `csa_dynamic` boundaries are not task-learned** — the
  `gate.detach() > 0.5` hard cut and the `comp_reg`-only offset gradient are
  documented in design_notes §22.5; `csa_dynamic_task` is the opt-in
  score-function alternative and is not part of the frozen confirmation.
- **16 evaluation windows (not documents) do not cover the language population** —
  the frozen panel is 16 fixed 4096-token rows/windows of one stream slice; a
  window can span articles and an article can contribute several windows.
  `MIN_DOCS_FOR_POPULATION_CLAIM = 64` is a house population gate with no
  statistical derivation (and also counts rows/windows); stats report
  between-window variance and set `population_claim_allowed = False` at 16.
- **The randidx / allblocks controls are not single-variable ablations** —
  the fixed hash pattern (with 202,752 dead projection parameters; the v5
  text's 201,216 was 1,536 short and is corrected in v6, see design_notes
  §27.6) and the 1024-block read are disclosed in each `interpretation_caveat`.

Two points were real implementation defects and are fixed, and the review's
power-gate recommendation is now normative:

1. **`same_init_core_fp` pooled all seeds into one set.** A correct 12-seed
   panel gives every seed its own shared-trunk initialization, so the field was
   guaranteed `False`: it could not detect a real arm mismatch and it
   mislabelled every correct panel. `finaltest_stats` now compares the two arms
   WITHIN each seed and reports `init_core_fp_pairs_checked` /
   `init_core_fp_mismatch_seeds` / `init_core_fp_missing_seeds`; the panel-level
   `validity` block groups by seed (`init_core_fp_distinct`,
   `init_core_fp_conflict_seeds`). The per-seed digest is now a CONFIRMED gate
   (manifest v6): a missing or mismatching digest is unverifiable and cannot
   confirm.
2. **Unverifiable provenance could confirm.** The v5 rule
   `param_matched is not false` treated a missing parameter count exactly like
   a measured <=1% match. Manifest v6 requires `param_matched is True`; a
   measured mismatch or an unverifiable `None` both withhold CONFIRMED.
3. **Low power was only a warning.** `run_finaltest` checked that
   `p5c_power.json` existed and was bound to the plan, then printed a warning
   and consumed the one-shot split. Manifest v6 adds
   `statistics.power_gate` (q=0.80, min_power=0.80): every primary
   comparison's panel-specific power must be recorded and at or above the floor
   or the access is refused; `V12_ACCEPT_LOW_POWER=1` is the explicit override,
   written to `results_lm_test/power_gate.json` and carried into
   `test_stats.json`. The 12-seed design's own q=0.80 Holm power is 0.275, so
   the gate normally fires — that is the point: the operator must add seeds
   (with a further recorded revision) or consciously accept a
   resolution-limited confirmation. The override does not make the design
   adequately powered.

Plan revision **v6** (still strictly before any test-split access: no
`results_lm_test/`, no freeze lock, no mirror, no evidence row) registers the
new `decision_rule`, `validity_assumptions` and `power_gate`, and records that
the change is a conservative tightening: it can withhold a CONFIRMED label but
never create one. No arm, seed, endpoint, alpha or Holm family changed; no
measurement semantics changed, so `CODE_SEMANTICS` stays `v11.128` and no
existing record is invalidated.

Not changed on purpose: the arm set (adding Dynamic/HCA arms would be a new
pre-registered study, not a v12 amendment), the 12-seed count, the 16-window
evaluation panel, the dense-arm capacity reallocation, and the default dynamic
segmenter's gradient path. Those are scientific decisions that need their own
registered design (the follow-up protocol) and, for the Power question, GPU
runs — not an implementation-only patch.

Regression coverage: `test_review_fixes.py` N96
(`_power_gate_problem` floor/override/malformed cases), the rewritten
`t_finaltest_decision_rule_matches_frozen_manifest` (per-seed digest across
seeds = True, one-seed mismatch = False, missing digest/params = not
confirmed), the updated `_frozen_rule_confirms` tests, and the v6 manifest
assertions in `t_shipped_manifest_consistency`.

## Review 14 (2026-11-25): verify the six-category methodology review; wire the four random-selection controls and the selection-attribution probe

Verification-first pass, no new training panel, no test-split access, no frozen
manifest byte change.  The review's six categories map to HEAD as follows:

1. "Dynamic chunking is not task-trained" — TRUE by the default arm's
   definition (`gate.detach() > 0.5`; `delta_logit` trained by `comp_reg` only)
   and already documented as such; `csa_dynamic_task` is the opt-in REINFORCE
   boundary arm.  No code change possible without a new study.
2. "The frozen confirmation does not cover Dynamic/HCA" — TRUE by design
   (`scope.not_covered`); the follow-up protocol, pure-HCA variants and the
   validator already exist.  Amending the frozen manifest is forbidden by
   pre-registration discipline.
3. Control confounds — TRUE and disclosed (1.87x training FLOPs, kernel /
   normalization / sink differences, 32x allblocks read, dead parameters).
   The actionable part was the MISSING control family: the protocol requires a
   per-forward re-randomized selector, a distance-matched random selector, a
   nearest-block selector and a uniform-stride selector — none existed.
4. Statistics (power 27.5%, sharp-null assumptions) — TRUE; the v6 manifest
   power gate (review 13) is normative and the robustness fields
   (bootstrap/Wilcoxon/LOO, assumptions text) shipped in review 12.
5. "16 evaluation windows are not documents" — TRUE and disclosed
   (unit wording, `population_claim_allowed=False`, 64-document house gate,
   doc-level path reserved for the follow-up).
6. "No oracle / recall@k / mass-covered mechanism evidence" — TRUE: the
   `audit_gates.audit_selection` contract existed but no producer ever
   generated the artifact.

Fixed this round:

1. **Four selection-control modes** in `lightning_indexer`
   (`random_each` / `random_dist` / `stride` / `nearest`), registered as the
   rope variants `csa_fixed_rope_randidx_each` / `_randidx_dist` /
   `nearidx` / `strideidx` (registry-only; nothing trains them by default).
   `random_each` is a content-keyed hash of the hidden state: the pattern
   re-randomizes every forward as H evolves, yet is a fixed function of the
   saved inputs, so activation recompute and resume reproduce it without any
   RNG stream (`.detach()` keeps it out of the gradient path).
   `random_dist` multiplies the same hash by a causal distance kernel
   (tau = 512 tokens); `nearest`/`stride` are integer geometry controls.  All
   four flow through the existing causal/top-k/STE machinery and are refused
   by CSA_KERNEL and open-block paths exactly like the legacy hash.
2. **`mech_selection.py`** — the selection-attribution mechanism probe:
   learned-vs-oracle recall@k, mass covered and k/B baseline per seed, plus
   descriptive PPL readouts (learned / oracle-forced / allblocks /
   fixed-hash-random at the same k).  It writes the exact
   `analysis_mechanism/selection_attribution.json` contract that
   `audit_selection` validates (`smoke` passes the audit end to end), refuses
   any path that touches `results_lm_test`, and reads only the validation
   stream or synthetic ids.
3. **A real reporting defect found by line-by-line review**:
   `_uses_batched_eager` tested `indexer_mode` for truthiness while the
   forward dispatch requires `== 'learned'`, so random layers were
   misreported as taking the batched twin.  Fixed to the equality check.
4. **Generalized dead-parameter accounting** — every non-learned
   `indexer_mode` bypasses W_DQ/W_DK/W_w, so `param_breakdown` now counts
   them all as dead (the note text changed accordingly).

Deliberately not changed: the frozen arms/seeds/panel, the default dynamic
segmenter's gradient path, the 12-seed count, the 16-window panel, and the
dense capacity reallocation — all of them need their own registered design
and GPU runs, not an implementation patch.

`CODE_SEMANTICS` stays `v11.128`: the learned path and the legacy fixed-hash
path are bit-identical (N110 asserts `True == 'random'`); `v12_supp.py plan`
still admits 48/48 synthetic records after the change.

Regression coverage: `test_review_fixes.py` N110 (mode normalization,
selection semantics of all four modes vs independent formula/geometry
replication, legacy bit-identity, fail-closed modes), N111 (tiny-model
forward/backward per mode, batched-twin mirror fix, rope registry entries,
finite forwards), N112 (`mech_selection` smoke produces an audit-accepted
artifact); `t_param_breakdown_reports_dead_indexer_params` (L5b) extended to
the new modes.  `test_open_block.py` 15/15 and `validate_followup_design.py`
registry checks still pass.

## Review 15 (2026-11-26): independent re-verification of the six-category review + per-function exp_lib audit; pattern-seed wiring, mass_covered fix, FLOPs-matched dense planning

Decision: verify every review-14 disposition claim independently against HEAD
before changing anything, then perform the requested per-function audit of the
dynamic segmenter / CSA-HCA forward / gradient paths.  No new training panel,
no test-split access, no frozen-manifest byte change, `CODE_SEMANTICS` stays
`v11.128`, `v12_supp.py plan` must keep admitting 48/48 records.

Outcomes:

1. **`mech_selection.mass_covered` was a real metric defect** (implementation
   contradicted the `audit_gates.audit_selection` contract): the denominator
   used the readable-block COUNT instead of the row's total attention mass,
   deflating the metric by ~n_readable and making it incomparable with the
   k/n_blocks random baseline.  Fixed (denominator = the softmax normaliser
   over readable blocks); N114 locks the formula against an independent
   recomputation.  Scientific impact: the pre-fix smoke read
   `mass 0.09 < base 0.13` purely from the bug; the corrected smoke reads
   `mass 0.38 > base 0.13`.
2. **Connection-pattern seeds are wired** (FOLLOWUP protocol previously
   declared them "NOT yet wired"): `AttnCfg.pattern_seed` (0 = review-14 bits
   exactly, guard skips the shift entirely; any non-hash mode with a nonzero
   seed fails closed in BOTH AttnCfg and lightning_indexer — the frozen
   fixed-hash arm stays bit-identical), `lightning_indexer` /
   `_indexer_trainable` / `_single` pass-through, registry twins
   `csa_fixed_rope_randidx_{each,dist}_ps<N>`, `design.selection_controls`
   in the follow-up template with a validator cross-check.  N113.
3. **FLOPs-matched Dense planning** replaces `TODO_FLOPS_MATCHED_DENSE`:
   `analytic_attn_flops_token` / `analytic_train_flops_token` (the disclosure
   accounting: attention core + MLP, projections excluded for every arm;
   dense's optional window is `cfg.window`, not CSA's `sliding_window`)
   reproduce 1,229,824 vs 2,297,296 FLOPs/token/layer (1.87x) exactly;
   `flops_matched_mlp_ratio` solves the dense knob (frozen-config solution
   mr = 689/256 = 2.69140625); `full_rope_flops_matched` registered and routed
   through `variant_mlp_ratio(flops_matched=..., flops_match_ref=...)`,
   mutually exclusive with parameter matching (fail-closed).  The analytical
   equality is planning only — the 1% profiler-measured gate stays the sole
   "equal compute" acceptance; the follow-up template still fails
   `--require-ready` by design.  N115.
4. Per-function audit findings recorded in design_notes §32.4: the STE soft
   prior, REINFORCE sign/baseline ordering, per-row/layer logp accumulation,
   the four control-mode selection semantics, and the batched-twin dispatch
   mirror were all verified sound (no gradient-estimation or
   control-equivalence defect beyond item 1).

Regression coverage: N113 (bit-identity at seed 0 vs the review-14 formula,
reproducibility, distinctness, fail-closed modes, registry twins, model-path
consumption), N114 (contract-formula lock + smoke), N115 (analytic numbers,
solver, routing, mutual exclusion, registry, finite forward).

## Review 16 (2026-11-26): retire v12 confirmatory evidence as a claim basis; land the two pre-freeze tooling blockers (power simulation, measured-FLOPs gate)

**Trigger.** The owner decided to retire (抛弃) the v12 experimental results
as the basis for any methodological claim, and asked for the remaining
actionable fixes ahead of the follow-up confirmatory runs. Per the review-14/15
dispositions, all six design-boundary findings are disclosed rather than
hidden; with the v12 evidence retired, the only scientifically valuable
modifications are (a) the forward-looking tooling the follow-up protocol still
lacked, and (b) an explicit retirement record so no reader mistakes v12 for
current evidence. The algorithm core (`gate.detach()` chunking, CSA/HCA
forward, REINFORCE arm) is intentionally untouched — review-15 verified it
sound, and the boundary is scope, not implementation.

**Decisions.**

1. **v12 evidence retirement is declared, not rewritten.**
   `confirmation_frozen.json` stays byte-identical (rewriting a frozen
   pre-registration is tampering, worse than the disclosed limitation it would
   "fix"); `v12_supp.py plan` and the CI preflight keep running against it.
   The retirement lives in the claim layer: README/README_zh evidence-levels
   blockquotes, RELEASES.md status note, this entry, design_notes §33.1. No
   current or future statement may cite v12 numbers (CONFIRMED labels
   included) as support for CSA/HCA/Dynamic effectiveness or equal-compute
   superiority; confirmatory evidence can only come from
   FOLLOWUP_CONFIRMATORY_PROTOCOL.md.
2. **`followup_power_sim.py` lands the pre-test simulation-based power
   analysis** the protocol required but no tool provided. Effect model:
   declared normal mean/sd or a bootstrap over observed validation deltas,
   recorded in the artifact. Statistics are the exact confirmatory ones
   (`exp_lib.exact_signflip_p` + `holm_adjust`; the batched enumeration is
   contract-locked to the scalar path, N116). `--seed-sweep` recommends the
   minimum seed count clearing the declared target power; the conservative
   analytic projection (q=0.80 @ 12 seeds = 0.2749) is recorded as a
   cross-check column.
3. **The validator binds the artifact, not the boolean (N117).**
   `statistics.power_simulation_completed = true` now requires
   `statistics.power_simulation_artifact` naming the tool's output; study id,
   hypothesis family, seed panel, alpha, target power and minimum simulated
   power are cross-checked against the plan. Tampering with any field fails
   `--require-ready`.
4. **`followup_flops_gate.py` lands the measured-FLOPs acceptance gate.**
   Empirical driver: `torch.profiler(with_flops=True)` attributes only
   mm/bmm/addmm and silently drops every SDPA attention kernel, while
   `torch.utils.flop_counter.FlopCounterMode` (aten-dispatch level) counts
   SDPA with the same non-halved convention as the repo's analytical model.
   Measurement = FlopCounterMode over K post-warm-up training steps
   (fwd+bwd+optimizer); a repeat block must reproduce the count exactly; the
   active custom CSA Triton kernel path refuses to measure (invisible to
   op-level counting); the profiler pass runs only to DISCLOSE the
   unattributed ops. `gate` applies the 1% tolerance to total training FLOPs
   (`flops_per_step x declared_total_steps`); CPU runs are smoke evidence and
   never certify. First smoke numbers recorded in design_notes §33.3: at a
   tiny geometry the per-row eager CSA implementation executes ~3x the
   op-level FLOPs of the dense SDPA path — an implementation cost the
   analytical planning model cannot see, which is precisely why the protocol
   demands measured FLOPs as the sole equal-compute acceptance.
5. **No change to:** the frozen manifest (zero bytes), `CODE_SEMANTICS`
   (`v11.128`), the variant registry, the algorithm core, or the CI gates.
   `test_review_fixes.py` 420/420 (N116–N119 added), `test_open_block.py`
   15/15, `v12_supp.py plan` 48/48.

**What this does NOT do.** It does not run the power simulation on real
validation-derived effect sizes (the inputs must come from the follow-up
validation panel, then be bound into the design); it does not run the FLOPs
measurement on the certifying device (GPU access required for a certificate);
it does not run the algorithmic gates; and it does not freeze the follow-up
manifest. All four remain preconditions for `--require-ready`, by design.
