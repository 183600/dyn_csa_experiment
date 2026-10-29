# 评审四（review 4）逐条核对与修复记录

基线：review-4 修订（2026-10-20）。本文件由 `DECISION_LOG.md` 的 review-4
条目、README 的 Review-4 段与 `docs/design_notes.md` §25 重建，只记录
**代码/文档层面**的状态；`results_*/`、`analysis_*/`、`REPORT*.md` 不入库，
因此所有涉及具体 PPL 数值的判断都必须由读者在自己的产物上重跑。

图例：**FIXED** = 本轮把原先的软约束改成机器闸门；**PARTIAL** = 已有实质
机制，但仍有本轮无法消除的边界；**UNVERIFIABLE** = 缺实验产物，代码只能
提供可执行入口。

| # | 评审问题 | 当前判定 | 本轮改动（代码/文档） | 仍存在的边界 |
|---|---|---|---|---|
| 1 | 外部有效性：小模型结论被外推到 LLM/预训练/DeepSeek/生产 | PARTIAL | 5 个报告构建器的标题改为范围声明（非 DeepSeek-V4 全量复现）；README/README_zh 明示不外推；冻结清单 `scope` 未变 | 跨语料/跨规模只有 B 档计划（`explib/m0/`，`python -m explib.m0`），未运行；本快照无结果产物 |
| 2 | 参数/计算/训练状态未同时匹配 | PARTIAL | `_param_control_detail` 报告 non-embedding/active 参数量；`_compute_control_detail` 用实测 time/token（或 6ND 代理）给出 compute 比与方向；compute 偏袒赢家且 >1.5× 时 `system_level_only`；v12 拒绝缺少 compute disclosure 的清单；主比较保留 1.87× FLOPs 披露 | 没有等 FLOPs/等 GPU 时间/等能耗的**运行**；代理不是测量；v12 主比较仍是 capacity reallocation |
| 3 | 训练预算不足以支持「最终/渐近线」结论 | PARTIAL | 报告中的「渐近线坐实/反转」改为窗口内读数；n<6 时强制写「描述性、不足以判定」；best/final 字段与曲线已存在 | 40k 长跑只有 n=2，无学习率续训/独立复现；未补跑 |
| 4 | 样本量小，p 下限被误读 | PARTIAL（解释层已修，样本量未变） | `reportable_claim` 统一 n≥6 + 原始 p + Holm + mean>std + 族级下限；v10/v11/`build_report` 全部走同一闸门；`underpowered` 与 `resolution_limited` 分开返回 | 探索性面板仍多为 n=2–6；没有新增种子 |
| 5 | 多重比较未控制 | PARTIAL | v10 NIAH 对照并入 Holm 族重新校正；`attach_holm` 输出族级最小可达 p 并在 p_holm=1.0 旁打印分辨率上限；v10/v11 的「强版本」不再用原始 p | 没有跨面板的全局 FDR；探索性比较族仍很大 |
| 6 | 结果导向补实验/选择性停止 | PARTIAL | 探索/确认标签、冻结后禁加种子、`n_effective`/`missing_seeds`/`skipped_reasons` 保留；DECISION_LOG 记录修订前提 | v7–v11 没有事前预注册或停止规则；只能事后标注 |
| 7 | 长上下文：absPE 截断、外推≠长上下文训练 | FIXED | `truncated` 标记进入记录；RoPE-only 比较；`audit_length` 校验同长度同目标；**缺失 `truncated` 现按不可验证处理**（不再 fail-open） | 仍是 train@512 的长度外推，不是长上下文训练 |
| 8 | 干扰注入不能证明普遍抗噪 | PARTIAL | 7 种腐坏模式（uniform/natural/mask/delete/span/doc/topical）+ NIAH；报告措辞改为 `realistic-token`，删去「更自然/自然语言」的强表述 | 没有主题/语义相近/对抗干扰；不能外推为普遍抗噪 |
| 9 | random/all-blocks 不能隔离机制 | PARTIAL | allblocks 明确写成「压缩块全读、读取更多、非 dense token attention、非等预算」；`explib/rope/`（v7）注释同步修正 | 没有读取预算匹配的 allblocks 变体；选择与读取容量仍混在一起 |
| 10 | hard top-k/分块改变优化过程 | PARTIAL | 文档明确 ST 只作用于已选集合、默认 `csa_dynamic` 边界不是任务训练；新增 `audit_gates.selection` 契约，缺工件时拒绝「选择器找到相关信息」的表述 | 没有 oracle/soft selection/蒸馏/recall 工件；REINFORCE 方差未分析 |
| 11 | 动态分块混淆块数/读取量 | PARTIAL | `audit_attribution` 从 `stats` 重算块数/块长/p95，并给 fixed-vs-dynamic 块数比与 10% 匹配判据 | 没有 matched-read-token 的再切分实验；读取 token 数仍无直接字段 |
| 12 | 交叉点 scaling 统计基础弱 | FIXED（分析层） | `crossover_scaling_fit`：≥4 个去重宽度、任一点删失即拒绝、训练量差 >25% 拒绝、重复 d 保留 token-matched、斜率 CI/p/R²/留一范围；报告被拒绝时不再报斜率 | 没做 Tobit/生存回归；点数少，仍是描述性趋势，不是 scaling law |
| 13 | 训练/评估数据独立性 | PARTIAL | 分词器 train-only、test 近重复清理、train/val/test 指纹、泄漏审计 hard-refuse；DECISION_LOG 更正「不删除近重复」的旧说法；新增 `tokens_seen_unique_estimate`/覆盖率估计与记账说明 | 估计假设独立均匀窗口抽样，不是精确计数；validation 被反复用于选择 |
| 14 | eval_subset 与测量误差 | PARTIAL | 文档/代码明确最终 PPL 用完整固定验证批，`eval_subset` 只控曲线；`ppl_ci95_doc_bootstrap` 进入 P2 表；v12 加文档 Δ cluster CI，并新增 seed×unit 两层 bootstrap 的总不确定性读数与 `audit_gates.two_level_ci`；<64 文档禁止总体主张 | 冻结 test 的 n=48 仍低于 64 单元总体门槛；两层 bootstrap 是设计型重采样，不是联合 hierarchical/随机效应模型，12 个种子 cluster 时百分位覆盖率有限；未做 token 级测量误差模型 |
| 15 | 配对检验的独立性/匹配 | PARTIAL | 新增 `init_core_fp`（tok/pos/head/norm）并纳入 `pair_reason`；v12 输出 `same_init_core_fp`；`rng_fp` 仍按设计只报告不强制 | 没有逐张量全参数匹配；架构导致的 RNG 差异仍属于臂 |
| 16 | 陈旧/重构记录可能进入结论 | FIXED | `sec_p1l` 与 v10 按长度表整臂排除无当前戳记录（不列表/不算 ratio）；`_agg_entry` 多组无当前戳或显式旧 run_cfg 直接拒绝；`build_report` 的 Monte-Carlo 旧文案更正 | 单条无 `run_cfg` 的历史 aggregate 仍走兼容路径；synthesized 记录仍可用于曲线展示 |
| 17 | 结果产物不入库，无法独立审计 | PARTIAL | README/README_zh 增加「证据状态」段；`audit_gates.release` 解析 `RELEASES.md`，无完整行即 `no_release`；归档增加 logs/cache meta/pyproject；新增 `explib.reproduce` 跨运行复现比较器（源码/数据/配置/环境/初始化逐维对账） | **本快照没有随附 release**；没有真实数值可复核，只有可执行归档/比较工具 |
| 18 | 复现环境不可控 | PARTIAL | import-time pip install 早已移除；HF mirror 仅 opt-in；`requirements.lock`/`pyproject.toml` 在位；CI 安装 `requirements-ci.txt` → `requirements.lock`；归档记录 pip freeze/GPU | 仍没有跨硬件逐位复现；lock 不是 CUDA 参考环境 |
| 19 | 效率结论缺真实测量 | PARTIAL | v7 长度阶段开启 `measure_eff`；新增 `measure_efficiency_sweep`（batch×length、CUDA event、环境元数据）；`audit_gates.efficiency_pairs` 只在同 token 数/同设备/同 KV/±5% 外才允许「更快」；FLOPs 文案改为结构性估计 | 本机无 GPU，未产生实测数字；没有 energy/真实 KV cache；dense 基线优化程度未逐项对齐 |
| 20 | 缺机制层面直接验证 | PARTIAL | NIAH、boundary-vs-random、block stats 已在；新增 selection-attribution 契约与 `audit_gates.selection`；缺工件时明确「未支持」 | 没有 selection recall/precision、与 dense attention 权重的相关性、oracle 分割/检索上界、selector 稳定性实验 |
| 21 | 方向性证据被写成因果结论 | PARTIAL | 「压缩不是瓶颈」「容量红利」「优势随长度放大」「渐近线坐实」等段落加显著性/口径限制或改写；`system_level_only` 阻止把目标/计算混杂的显著结果标成 `verified` | 报告是动态生成的，仍可能有未覆盖的措辞；需要在真实产物上重跑后复核 |

## 本轮新增/修改的机器闸门（便于 review）

- `exp_lib.reportable_claim`、`attach_holm.family_min_attainable_p_holm`
- `exp_lib.crossover_scaling_fit`、`exp_lib.loglog_regression` 的 CI 路径接入 v10/v11
- `exp_lib.init_core_fingerprint` + `pair_reason` 的 init gate
- `exp_lib._param_control_detail`、`exp_lib._compute_control_detail`、`system_level_only`
- `exp_lib.paired_nll_two_way_bootstrap` + `audit_gates.two_level_ci`
- `exp_lib.reproduction_diagnosis` / `compare_reproductions` + `python -m explib.reproduce` + `audit_gates.reproduction`
- `exp_lib.measure_efficiency_sweep`
- `build_report.sec_p1l` / `_agg_entry` 的 stale fail-closed
- `explib/corruption/`（v10）的 NIAH-Holm 合并、`_all_sig` 走统一闸门、按长度表 stale 排除
- `explib/confirmation/`（v12）的 compute disclosure 拒绝、init/compute 字段、文档 CI 列、总体主张门槛
- `audit_gates.py` 的 `selection` / `release` / `efficiency_pairs`，以及 `attribution` 的块数对照
- `archive_release.py` 追加 logs / cache meta / pyproject
- review 闸门测试覆盖 L1–L11，现拆分为 `test_review_part_*.py` 并由
  `test_review_fixes.py` 兼容入口统一运行（`make test`）

## 不能由本仓库单独回答的部分

1. 任何具体 PPL 数值：没有结果产物，也没有证据归档。
2. 等 FLOPs / 等 GPU 时间 / 等能耗的公平比较：需要新的训练运行。
3. 大模型、其他语料、长上下文训练、生产系统：需要 B 档/独立预注册。
4. 选择器 recall/oracle、软选择、蒸馏：需要新的机制实验 artifact。
5. 真实推理延迟/吞吐/KV cache/能耗：需要带 checkpoint 的 GPU 测量。
