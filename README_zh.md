# Dynamic-Block CSA + HCA — WikiText-103 对照实验

[English](README.md) | **中文说明**

在小型从零训练语言模型（BPE-8k，WikiText-103）中，对**压缩块稀疏注意力**
（CSA / HCA，固定分块与动态分块）与稠密注意力进行受控对比研究。本仓库包含
实验代码以及在自己机器上**重跑实验**所需的协议。仓库**不包含数值结果**：没有随仓库发布证据归档（见 `RELEASES.md`），因此仅凭 Git 树无法独立复核任何报告中的 PPL 数值。

> **证据等级**：v6–v11 为探索性（在同一 validation split 上选择后观察）；只有 v12 冻结的固定分块比较是确认性。详见 `DECISION_LOG.md` 与 `docs/design_notes.md` §17/§20。
>
> **v12 证据已退役（2026-11-26，owner 决策）**：v12 确认轮被取代——四臂不含 Dynamic/HCA、Dense 对照是参数匹配非算力匹配、随机对照是固定哈希、测试面板仅 16 窗口（均为已披露边界），owner 决定退役该轮数字作为任何方法学声明的依据。本仓库今后不得引用 v12 结果（含 CONFIRMED 标签）支持 CSA/HCA/Dynamic 有效性或等算力主张；确认性证据只能来自 `FOLLOWUP_CONFIRMATORY_PROTOCOL.md`。冻结清单 `confirmation_frozen.json` 字节不动（历史预注册记录，改写即篡改）；`v12_supp.py plan` 与 CI 预检照旧对它运行（design_notes §33.1）。
>
> **版本命名**：`v12` 是驱动/研究代次；`CODE_SEMANTICS = v11.128` 是冻结清单绑定的库测量语义戳，两者故意不同号。冻结清单当前为 `manifest_version: 5`（见其 `revision.history`；v4 移除了不可机检的 ee/es 钉定并补充了 tokenizer 拟合的 token 上限说明，v5 把 v4 注记中对已实现判定规则的描述更正回注册的 v3 措辞——规范性字段未变；v6（2026-11-24）注册了保守的“证据完整性 + 最小功效”闸门，详见下文——均仍在任何 test 访问之前）；按清单 v3 的 `ci_loop` 注记，continuous-loop CI 应保持停用直至签署后的提交固定且 P5T 完成——依赖此约定前请用 `gh workflow list` 核实工作流的实际状态。

> **证据状态（引用数字前必读）**：`results_*/`、`analysis_*/`、`REPORT*.md` 有意不入库，且**没有随仓库发布证据归档**（`RELEASES.md` 尚无记录）。报告可由你自己产生的产物重建，但任何你没有重建过的报告数值都无法独立复核。只有当 `release_<tag>.tar.gz` 与清单发布到只读位置并通过 `archive_release.py verify` 时，相应主张才可审计。

## 评审十一硬化（review 11）

逐项核验见 `docs/design_notes.md` §29。本轮不新增训练面板、不改冻结清单、不读 test。
真正的代码修复：

- **配对面板 bug**：报告曾把每个 seed 不同的 `init_core_fp` 当作跨 seed 分组键，
  正确面板被拆成每组 1 个 seed（n=1）。现在只在 seed 对内校验初始化，5 seed 面板恢复 n=5。
- **陈旧记录 fail-closed**：`sec_stats`、`sec_flops`、健康/效率表、逐 seed 配对及 v8–v11
  面板读取都要求当前 code stamp；旧分析/旧记录不再进入主表。
- **多重性可见**：v10/v11 打印族级分辨率下限和跨面板 BH-FDR + Bonferroni；
  v10 的 ΔNLL/NIAH 表补 `p_Holm`，不再在“Holm”标题下只印原始 p。
- **机制措辞收紧**：v10/v11/fuse 不再把 random/all-block 对照写成“独立贡献/机制结论”，
  改为系统级差异 + 归因边界 + 缺失 selection-attribution 的说明。
- **停止规则**：follow-up 协议/校验器要求单次固定 n 分析；中途看数或改样本量需新预注册。
- **报告卫生**：v9/open-block/fuse 补证据状态横幅；`eval_subset_rows` 不再误称 documents；
  效率节明确 energy 未测。

仍属研究设计限制、不能靠代码消除的：小模型/WikiText 外部有效性、参数匹配但非 FLOPs 匹配、
dense/CSA 的 q/k 归一化与 attention scale 不同（CSA 先 L2-normalize 再 RMSNorm、scale=1.0；
dense 仅 RMSNorm、scale=1/sqrt(d_head)）、缺同投影/同 sink 的 dense 对照、短训练预算、
2–6 seed 探索面板、16 窗口冻结面板、hard top-k 未选块无注意力梯度、缺 oracle/selection recall、
动态分块逐样本几何不完整、无删失回归、source-only 无发布（`archive_release.py record` 已有，
但尚无归档）、CUDA 环境未锁定、未测 energy。

## 评审十二硬化（review 12）

对 16 条外部方法学意见做**核验优先**的处置：不新增训练面板、不读 test、不改冻结清单字节、不改测量语义（`CODE_SEMANTICS` 仍为 `v11.128`）。逐项判定（含在当前 HEAD 已不成立的事实）见 `docs/design_notes.md` §30。审查的设计边界意见（探索/确认混合、小 n、多因素、validation 复用、source-only）成立且保留为限制；本轮修的是仍可复现的代码/报告缺陷：

- **训练目标描述符漏了每层压缩权重**：`train_objective` 原先只记 `comp_lambda`，`csa_dyn_nocomp`（mult=0）仍显示 `comp_reg_applies=True`、有效权重 0.05，hybrid HCA 的 4× 也不可见。现在记录每层有效权重，`objective_reason` 比较权重向量；v7/v9 对比携带 `objective_matched`/`objective_differences` 并打印系统级边界（dense-vs-sparse 仍配对，但不能再被静默读成机制归因）。
- **符号翻转的稳健性字段落盘**：`aggregate`、v7 stats、v9 `add_paired` 现在保存符号计数、p 下限、配对/非配对 bootstrap CI、Wilcoxon 与显式零假设；v7 stats 表并排打印，且声明族不会因缺少某行而缩小（缺失槽位按 p=1 进入 Holm）。
- **PPL 聚合**：v7 长度节与 v10 P3MT 改为跨 seed 的 **token-pooled PPL**（`exp(ΣNLL/Σtokens)`；缺 NLL 才标 `*` 回退），打印 n 与 ratio 口径；v9/v10/v11 面板表增加 pooled/median/IQR；长度结论改为描述性系统级读数，明确「512 训练长度外推探针，不是长上下文训练」，删除「机制一致/滤噪」措辞。
- **多重性补全**：v10 的全局 BH/Bonferroni 族补入其展示的 scale 端点与等 token 切片 p 值（此前 m 偏小，打印的 q 值偏乐观）；P3T 补 `p_Holm` 与分辨率下限；v9 增加自己的全局 BH/Bonferroni 视图。
- **逐臂资源披露**：新报告节打印每臂 total/attention/indexer/MLP/active/dead、层数、结构 KV elem/token、实际压缩权重与 ΔLR 倍数，并声明「参数量匹配 ≠ 计算/有效容量匹配，逐臂 FLOPs 未实测」。
- **旧「证伪」措辞**：P0-3 不再写「P0-3 的证伪」；boundary-F1 不再称分块「genuinely semantic」；两者都降为探索性方向读数。
- **证据可审计性**：`evidence_release_status` 对发布行做 schema 校验（ISO 日期、7–40 hex commit、http(s)/s3/gs URL、两个 64-hex 摘要、非占位 verifier），垃圾行不再把所有报告横幅翻成 `released`；`archive_release.py` 补打 TimeGuard `run_time_state*.json(.k)` 账本，拒绝只有 analysis 的归档（至少一个 `summary.json`/`run_manifest.json`），环境快照补 GPU 数量/名称/UUID/算力/驱动/TF32 与 pip freeze 截断标志；v7 报告末尾新增输入清单（逐文件 SHA-256 + git HEAD）并写 `analysis_v7/report_inputs.json`，把该报告绑定到具体 results/analysis 批次；v10/v11/v12 报告尚未生成该清单。
- **crossover 范围**：v10 明确 d 与层数绑定（128/4L…512/10L）、batch/seq/eval 网格不同，斜率不是单一规模变量；「预注册」改为「分析前声明的探索性判据」。

## 评审十四硬化（review 14）

对六类外部方法学意见做**核验优先**的处置：不新增训练面板、不读 test、不改冻结清单字节、不改测量语义（`CODE_SEMANTICS` 仍为 `v11.128`；`v12_supp.py plan` 仍 48/48 通过）。逐项判定见 `docs/design_notes.md` §31。审查的设计边界结论（动态分块非任务学习、冻结臂不含 Dynamic/HCA、dense 参数≠计算匹配、27.5% 功效、16 窗口面板）均成立且此前已披露；本轮修掉的是两个真正可修的缺口和一个逐行核验发现的真实缺陷：

- **协议要求的四个随机选择对照已实现**：`lightning_indexer` 新增 `random_each`（每前向重随机：隐状态 content 哈希，可重算复现、无 RNG 流）、`random_dist`（距离匹配随机：因果衰减 tau=512 tokens）、`nearest`（近邻）、`stride`（均匀间隔）四个模式，并注册为 rope 变体 `csa_fixed_rope_randidx_each`/`_randidx_dist`/`nearidx`/`strideidx`（仅注册表，默认面板不训练）。冻结的固定哈希 `random` 路径逐位不变（N110 断言）。
- **selection-attribution 工件有了生成器**：`mech_selection.py` 是 `audit_gates.py selection` 契约背后的机制探针——逐 seed 的 recall@k / mass covered / k-n_blocks（oracle = 同层 all-block 读的块级注意力质量），外加 learned / oracle-forced / allblocks / 同 k 固定哈希随机的描述性 PPL 读数；`smoke` 在小合成模型上端到端通过审计契约，`run` 只读验证流或合成 ids、拒绝任何触及 `results_lm_test` 的路径。在真实面板上跑过之前，选择召回主张门槛仍为 `not_verifiable_locally`。
- **真实报告缺陷修复**：`_uses_batched_eager` 对 `indexer_mode` 只做真值检查而实际分派要求 `== 'learned'`——random 臂会被误报为走了批处理孪生；已改为与分派一致的相等检查。死参数计数（`param_breakdown`）泛化到全部非 learned 模式。

## 评审十五硬化（review 15）

对同一份六类意见做**独立复核**（不采信 review-14 自述），并按外部要求对 `exp_lib.py` 的动态分段器 / CSA-HCA 前向 / 梯度回传 / 训练目标做逐函数审查。逐项判定见 `docs/design_notes.md` §32。零训练、零 test 访问、冻结清单零字节改动、`CODE_SEMANTICS` 保持 `v11.128`。六类设计边界结论复核后均成立且已披露；本轮修掉三个仍可修的缺口：

- **`mech_selection.py` 的 mass_covered 是真实指标缺陷**：分母误用可读块**计数**而非该行可读块**总注意力质量**，指标被系统性缩小约 n_readable 倍且与 k/n_blocks 基线不可比——review-14 smoke 的 mass=0.09 < base=0.13 纯属该 bug 的假象（修复后同一 smoke 为 mass=0.38 > base=0.13）。分母已改为 softmax 归一化项，与 `audit_gates.audit_selection` 契约一致；N114 用独立实现的公式对拍锁定。
- **连接模式种子已接通**（原「NOT yet wired」）：`AttnCfg.pattern_seed`（0 = review-14 逐位不变；非哈希模式配非零种子一律 fail-closed）把独立整数混入 `random_each`/`random_dist` 行键，同一训练 seed 可在多个连接模式下重跑；注册 `_ps<N>` 变体族，followup 模板新增 `design.selection_controls` 声明，验证器对账（N113）。
- **FLOPs 匹配 Dense 臂有了解析容量计划**：`analytic_attn_flops_token`/`analytic_train_flops_token` 精确复现冻结披露数字（1,229,824 vs 2,297,296 FLOPs/token/层，1.87x），`flops_matched_mlp_ratio` 解出 `full_rope_flops_matched` 的旋钮——冻结配置解 **mr = 689/256 = 2.69140625**——替换 `TODO_FLOPS_MATCHED_DENSE` 占位符；经 run cfg 的 `flops_matched` 集合路由、与参数匹配互斥 fail-closed。**解析等式不是证书**：1% profiler 实测门仍是「等计算」措辞的唯一验收，模板按设计仍 `--require-ready` 失败。

## 评审十六硬化（review 16）

owner 退役 v12 证据作为主张依据（见顶部引注），follow-up 协议中两个仍缺工具的 pre-freeze 阻塞项落地。零训练、零 test 访问、冻结清单零字节改动、`CODE_SEMANTICS` 保持 `v11.128`、`v12_supp.py plan` 48/48、`test_review_fixes.py` 420/420、`test_open_block.py` 15/15（详见 `docs/design_notes.md` §33）：

- **`followup_power_sim.py`——测试集访问前的仿真功效分析**：协议要求「合理配对效应量 + 验证集实测方差」的仿真，而 v12 的精确枚举只针对已观测增量。工具在声明的效应模型（正态均值/方差，或 bootstrap 重采样观测验证增量）下仿真四假设 Holm 族，检验统计量与确认性分析同源（`exp_lib.exact_signflip_p` + `holm_adjust`，批处理与标量契约对拍锁定，N116）；`--seed-sweep` 报告达到目标功效的最小种子数。
- **验证器绑定功效工件（N117）**：`power_simulation_completed = true` 必须携带 `power_simulation_artifact`（工具输出 JSON），study/假设族/种子面板/alpha/目标/min_power 六项对账，篡改即拒——布尔位不再能拍脑袋置。
- **`followup_flops_gate.py`——实测 FLOPs 接受门**：测量用 `torch.utils.flop_counter.FlopCounterMode`（aten 调度层，唯一能归因 `scaled_dot_product_attention` 的计数器——`torch.profiler(with_flops=True)` 只覆盖 mm/bmm/addmm，注意力内核一律静默漏计，profiler 通道降级为披露表）；重复块必须逐位复现；自定义 CSA Triton 内核路径拒绝测量；`gate` 对总训练 FLOPs 施加 1% 容差，CPU 冒烟永不签证书。CPU 冒烟实测已给出诚实的理由：tiny 几何下逐行 eager CSA 实现的 op 级计算量约为 dense SDPA 路径的 3 倍——解析规划模型看不到的实现开销。

## 评审十七硬化（review 17）

对一份十一部分的外部方法学审查（实现正确性 / 实验设计 / 统计推断）做核验优先的复核。审查的设计边界结论对照 HEAD 复核后**全部成立且已披露**（v12 覆盖缺口、v6-v11 验证集复用、稠密臂混杂、动态分块非任务学习、12 种子功效、冻结面板以窗口为单位、固定哈希随机对照、能耗未测）；逐项判定见 `docs/design_notes.md` §34。零训练、零 test 访问、冻结清单零字节改动、`CODE_SEMANTICS` 保持 `v11.128`。仍可在代码层面修复的一个缺口本轮关闭：

- **同骨架稠密隔离对照注册为正式臂**：follow-up 协议的机制隔离一节要求稠密对照的「单因素版本」，但此前只有两个容量参照（`full_rope` 参数匹配、`full_rope_flops_matched` 算力规划），v12 式混杂（q/k 归一化、注意力缩放、sink、kernel）没有对照臂能移除。`full_rope_cos_sink`（registry-only，不训练任何面板）是携带 CSA 臂 q/k 管线策略的稠密注意力：共享 qk RMSNorm、scale 1.0 的余弦归一化 q/k、可学习的逐头 sink 与 RoPE；稠密层**没有 sink 才走 SDPA**，因此 sink 同时把它路由到与 CSA 各层相同的手写 eager `_sink_attn_loop` kernel 族。残余差异（逐 token K/V 对比压缩块 KV 管线、q/k 归一化次序、逐 token 滑窗分支、投影结构）即被检验的机制本身，随臂披露而非隐藏。follow-up 模板以 `dense_same_backbone` 控制项声明之，`validate_followup_design.py` **强制要求**存在 `same_backbone` 稠密对照并逐字段核验声明的变体在每层真的解析出 `full_cosine/full_sink/qk_norm/rope`——普通稠密臂不再能冒用该隔离标签（N120）。

改动后的验证：`test_review_fixes.py` 421/421（新增 N120）、`test_open_block.py` 15/15、`audit_gates.py causality` 七个 rope 臂全过、`validate_followup_design.py --verify-registry` 接受模板（其余失败项均是有意的 pre-freeze 阻塞）、`v12_supp.py plan` 不受本轮影响。

## 评审十八硬化（review 18）

对同一份十一部分审查（其第二节的动态分块归因消融表）做核验优先的复核。审查的量化指控对照 HEAD **复算全部成立且与披露一致**（12 seed/Holm-4 功效 q=0.80 为 0.2749、精确符号翻转 p 下限 2/2^12=0.000488、randidx 死参数 202,752/1.83%、解析 FLOPs 1,229,824 vs 2,297,296/1.87x、FLOPs 匹配 mr=689/256=2.69140625）；逐项判定见 `docs/design_notes.md` §35。零训练、零 test 访问、冻结清单零字节改动、`CODE_SEMANTICS` 保持 `v11.128`、`v12_supp.py plan` 仍 48/48。仍可修的一个结构性缺口本轮关闭：

- **分段（边界生成）对照族落地**：审查要求区分「dynamic 增益来自边界位置、压缩率变化还是任务信号」，但 HEAD 的 follow-up 注册表只有选择级对照、没有任何边界级对照（`csa_dyn_nocomp`/`csa_dynamic_task` 只有非 rope 孪生；随机边界与 oracle 边界完全不存在）。本轮注册五个 registry-only rope 臂（`v7_supp.SEGMENT_CONTROL_VARIANTS`）：`csa_dynamic_nocomp_rope`（消融 B，comp 正则归零）、`csa_dynamic_task_rope`/`hca_dynamic_task_rope`（消融 D，REINFORCE 臂——其采样离散切分是唯一拿到 LM 损失梯度的边界；默认 `csa_dynamic_rope` 仍必须描述为余弦阈值+comp-reg 偏移，不得称为任务学习）、`csa_dynamic_randbnd_rope`（消融 E，`exp_lib.random_boundary_cut_table` 内容键控哈希切分：可重算、无 RNG 流、构造性率匹配、`pattern_seed` 可分离模式噪声）、`csa_dynamic_oraclebnd_rope`（消融 F，在 boundary-F1 分析既有的标点/eos 参考位置切分：`set_boundary_token_ids`+trunk 级刷新，未注册即拒绝）。`AttnCfg` 对 random/oracle 作用在稠密或非动态层、reinforce 作用在非余弦门上均 fail-closed。
- **修复一个 rope 路径真 bug**：`SmallGPTRoPE.trunk` 此前不把逐层 `last_boundary_logp` 聚合到训练器消费的模型级 `boundary_logp`——rope REINFORCE 臂的边界策略项会被静默丢弃（此前无 rope 臂使用 reinforce，故未暴露）。现在按基类同语义聚合（N124 锁定）。
- **设计记录绑定对照**：模板新增 `design.segmentation_controls`（五臂+机制标签+披露 note）；`validate_followup_design.py` 逐条对照注册表核验（threshold 臂不得冒用 `task_learned_boundary` 标签、oracle 必须写明参考源与注册要求、未知机制拒绝），并**强制要求** `random_boundary` 与 `task_learned_boundary` 两类对照——没有它们，注册的 2×2 因子无法把 dynamic 增益归因到任何东西。

改动后的验证：`test_review_fixes.py` 427/427（新增 N121–N126）、`test_open_block.py` 15/15、`audit_gates.py causality` PASS、`mech_selection.py smoke` 数值逐位不变、`validate_followup_design.py --verify-registry` 仅剩有意 pre-freeze 阻塞、`v12_supp.py plan` 不受影响。

## 评审十九硬化（review 19）

对「等计算量 / 动态分段 / 统计功效」十一部分审查做核验优先的复核。审查的量化指控对照 HEAD **复算基本全部成立**（12 seed/Holm-4 功效 q=0.80 为 0.2749、q=0.85/0.90/0.95 → 0.4435/0.6590/0.8816；精确符号翻转 p 下限 2/2^12=0.000488、Holm 下限 0.001953125；randidx 死参数 202,752/1.8283%；参数总量 11,089,904 vs 11,088,512；MLP 隐藏 1024 vs 1731/+69.04%；解析 FLOPs 1,229,824 vs 2,297,296/1.868x；冻结面板 16 窗口 = 8,192 token）；设计边界结论（dense 是参数匹配而非等计算、动态分块是余弦阈值+压缩正则、固定哈希随机选择器带死参数、27.5% 功效、窗口为评估单位、STE 选择、单一数据集、结果未归档）**全部成立且此前已披露**；逐项判定见 `docs/design_notes.md` §36。零训练、零 test 访问、冻结清单零字节改动、`CODE_SEMANTICS` 保持 `v11.128`、`v12_supp.py plan` 仍 48/48。本轮找到并关闭的是**披露缺陷**（不是测量缺陷）：

- **冻结散文里有一个本仓库复现不出来的比值**：`compute_budget_disclosure.flops` 与主比较的 `interpretation_caveat` 都写「attention 3.2x at L512」，但解析模型在 L512 下给出 dense/csa = 524,288/181,248 = **2.8927x**（dense/hca 为 3.7654x）；3.2x 的真实来源是 dense@L512 除以 hca@L2048（524,288/163,840），属**跨长度混算**，本仓库没有任何 L512 配置给出 3.2x。清单按预注册纪律字节不变，勘误进文档与报告：`exp_lib.compute_budget_recompute` 逐项重算披露数字，`exp_lib.audit_compute_disclosure` 按句（`; , . ( )` 切分）与句内最近关键词（attention/FLOPs/MLP）给每个 `Nx`/`N%` 归类并与复算值比对（>1% 即 mismatch），`v12_supp._emit_compute_budget_audit` 在 caveat 之后打印「口径复算」段与 mismatch 表，把 3.2x 标为**预注册文本勘误**。无法归类的数字（"32x more blocks"、"100%"、"0.013% apart"）不参与审计；复算不可用时整份审计返回 `unavailable`，不猜数。
- **一处 docstring 数字笔误**：`flops_matched_mlp_ratio` 的 docstring 写 2,297,523（比解析值多 227），改为 2,297,296，并补记 attention 比 2.8927x。

改动后的验证：新增 N127（复算逐项等于冻结清单、2,297,523 笔误不再出现、未注册变体 fail-closed）与 N128（审计必须且只能标出 3.2x 一条 mismatch、1.87x 与 +69% 通过、跨长度 3.2x 算术、不可归类数字不产生结论、复算不可用时不打印勘误、报告端渲染）**均通过**；受影响子集复跑全绿（N115/N118/N119、L1/L3*/L7*/L10/L12、N86/N96/N97/N99、C6/C20b/J4c/A6d）。本次沙箱对单次调用有时限，长训练回路用例（C5b、N103 等）未能在一次调用内跑完，故**未**声称全绿。其余仍需 GPU 的项（Triton 路径 custom_op + `register_flop` 的实测等计算量、任务驱动分段的训练证据、≥64 文档面板与文档级 bootstrap、≥20 种子的 80% 功效、真实 checkpoint 上的选择归因、能耗）不产生任何新的性能主张。

## 仓库结构

仓库只保留**源码、驱动脚本与设计文档**。所有实验结果目录（`results_*/`）、
零 GPU 分析目录（`analysis_*/`）、逐版报告（`REPORT*.md`）与运行状态文件
都**不入库**——它们由驱动脚本在跑实验的机器上重新生成，需要时本地复现即可
（见下文命令）。`.gitignore` 已把这几类路径挡在仓库外（`docs/` 只保留
`design_notes.md`；原 `compute_plan.md` 已退役）。

| 路径 | 说明 |
|---|---|
| `exp_lib.py` | **核心库，也是唯一的代码来源**：模型、CSA/HCA 注意力、池化、indexer（learned + 五种选择对照模式）、分段器、训练循环、聚合与统计 |
| `mech_selection.py` | **选择归因机制探针**（review 14）：`analysis_mechanism/selection_attribution.json` 的生成器——逐 seed 的 learned-vs-oracle recall@k / mass covered / 随机基线，外加 learned / oracle-forced / allblocks / 同 k 固定哈希随机的 PPL 读数；`smoke` 端到端自检 `audit_gates.py selection` 契约，`run` 只读验证流或合成 ids（拒绝一切触及 `results_lm_test` 的路径） |
| `csa_kernels.py` | **三算子融合内核路径**（已退役算力规划的 M0 工作包）：块压缩池化、索引 top-k 选择与稀疏注意力的 Triton 内核（附带同一套算法分解、任意设备可跑的 torch 参考孪生实现）；`CSA_KERNEL` 选开、默认关闭——见下文「Triton 内核路径（M0/G0）」 |
| `run_gapfill.py` | **v6 驱动**：最小可发表补充实验，支持断点续跑，受运行时上限约束，每阶段完成后自动 push，结束后自动关机 |
| `par.py` | **任务级并行执行器（可选）**：每个独立实验占一张 GPU 卡——见下文「任务级并行」 |
| `v7_supp.py` | **v7 驱动**：回应 reviewer 的补充实验套件——P0R RoPE+QK-norm、P0W 稠密预热、P0E 40k 步长跑、P2S 规模重播种、P1L 长度外推、P1T topk 扫描 |
| `v8_supp.py` | **v8 驱动**：P1S（规模面板补齐至 4 seeds）与 P2R（RoPE+QK-norm 20k 步长跑） |
| `v9_supp.py` | **v9 驱动**：P2T（seq-2048 的 topk 扫描，`batch_size=1`）与 P2S3（RoPE 20k 长跑的第三个 seed） |
| `v10_supp.py` | **v10 驱动**：P3M 长度外推机制 + 干扰项注入探针、P3S 交叉面板扩规模、P3T topk 扫描补至 4 seeds |
| `v11_supp.py` | **v11 驱动**：统计收尾——干扰探针扩到 n=6（P4MT/P4MP）、交叉面板扩到 n=4（P4S）、d=384 配对臂扩到 n=4（P4F） |
| `v12_supp.py` | **v12 确认驱动（评审响应 + 评审二轮加固）**：P5C 按清单种子列表训练全部四个冻结臂，P5MP 跑扩展腐坏模式探针（uniform/realistic-token/mask/delete + 评审二轮新增 span/doc/topical + NIAH-natural），P5S2/P5S3 补等 token 与 6ND 参数量代理预算的大规模臂，P5T 按 `confirmation_frozen.json` 在 held-out 的 WikiText-103 test split 上做**冻结后一次性确认评估**——含加载期功效硬门槛（数学上不可能确认的清单直接拒绝）、严格记录准入（种子白名单/唯一记录/单一冻结切片/配方签名/manifest 哈希）、首次访问冻结锁与**逐行/窗口** PPL 落盘（不是逐文档：单位见 design_notes §20/§27.2） |
| `open_block_supp.py` | **未结束块读取面板驱动**：最大块长（32/64/128/无强制上限）× 当前未结束块读取方式（`closed` / `prefix_summary` / `prefix_singletons`）；共享 top-k 预算、因果前缀构造、窗口外辅助面板、逐种子配对差与交互统计——见 `docs/design_notes.md` §24 |
| `test_open_block.py` | **未结束块 CPU 门禁**：配置/分段器、优化前缀池化与逐查询参考的前向/梯度一致、前缀不变性、未来扰动/未来梯度、候选掩码、边界情形、closed 回归、单 token 语义与 HCA 预算声明 |
| `confirmation_frozen.json` | **预注册冻结清单 v2**（入库）：冻结臂、种子 0-11（在任何 test-split 访问**之前**由 6 修订为 12，含功效表）、终点、精确符号翻转 + Holm、决策规则、机器可校验的绑定块与范围声明——见 `docs/design_notes.md` §17 与 `DECISION_LOG.md` |
| `DECISION_LOG.md` | **决策日志**（入库；因快照只有一个压缩提交，2026-10-18 依据 README/设计说明/清单重建）：每个版本为什么跑/不跑什么、看到什么、据此决定了什么——探索/确认分离的书面轨迹；正式版本的逐种子证据另由 `archive_release.py` 归档 |
| `audit_gates.py` | **评审二轮的可执行审计入口**（零实验）：数据泄漏（分词器来源/词表一致性/文档重叠）、纯 CPU 因果性单元审计（前缀不变性/标签移位/解码一致性）、分块归因/边界 vs 随机基线/长度口径/基线公平/失败集中度/预算对账与 prefill-decode-内存测量工具 |
| `archive_release.py` | **不可变证据归档器**：把每个正式版本的产物（结果/分析/报告/冻结锁/决策日志/环境/权重哈希）打包为带逐文件 sha256 清单的 tar.gz，附独立 verify 子命令——Git 主仓仍只承载源码（design_notes §8），归档发布到只读位置 |
| `m0_supp.py` | **M0/S1 驱动（已选定 B 档）**：扩规模路线图的代码入口——`params`（零 GPU 的 S1–S4 参数量核算）、`data`（FineWeb-Edu + 32k BPE 语料缓存）、`smoke`（B 档数据路径端到端自检）、`kernel`（G0 验证套件：三算子 kernel/eager 数值对照 + 稀疏注意力吞吐基准）、`full`（S1 主面板）；见下文“B 档 M0/S1” |
| `build_report.py` | 零 GPU 报告构建器：从 `results_*/` + `analysis_*/` 产物重建 v7 报告（所有数字现算，绝不硬编码） |
| `fuse_analysis.py` | 零 GPU fuse 对比驱动：从 `results_*/summary.json` 重算逐层边界对齐（tol=1）、块长统计与 PPL 曲线 |
| `docs/design_notes.md` | **设计与实现笔记**：块读门为何必须严格、且五处位置必须同步；动态分段器的前缀不变性要求；配对/可测量规则；报告约定；保证注意力按行分块与不分批形式等价的契约 |

> **`exp_lib.py` 是唯一的代码来源，直接改它。**

### 重新生成实验产物

结果目录与报告都不在仓库里。要重新得到它们：

```bash
python run_gapfill.py smoke      # 端到端自检，零 GPU 消耗
python run_gapfill.py full       # v6 主实验（受运行时上限约束）
python v7_supp.py ...            # 各版本补充实验，子命令见对应脚本的 docstring
python build_report.py           # 从落盘的 results_*/ + analysis_*/ 重建报告
```

## 复现方法（任意 CUDA 机器）

```bash
# 0) 环境 — Python 3.12（见 pyproject.toml / requirements.txt）：
#    CPU 复现/测试请使用已验证的 lock（含 torch）：
pip install -r requirements.lock
#    CUDA 机器请按本机 CUDA 版本先装 torch，再安装其余依赖：
#      python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch
#      pip install -r requirements.txt
#    随后用 `python -m pip freeze > requirements.lock` 生成本机 lock 并归档；
#    requirements.lock 记录的是已验证的 CPU 环境，不是论文规模的 CUDA 环境。
#    torch 说明：lock 固定 torch==2.14.1+cpu；scatter 路径已把 int32 索引
#    转成 int64，不再依赖 torch >= 2.14 的 scatter 行为，支持范围仍为
#    requirements.txt 中的 torch>=2.8,<3。
# 代码不再在 import 时自动安装依赖；如确需临时环境可显式设置
# CSA_AUTO_INSTALL=1。Hugging Face 镜像为 opt-in（CSA_HF_MIRROR=1），
# 默认走官方 Hub。
#
# 如果 Hub 不可达，`load_wikitext` 会回退到本地原始文本：
# 将 `WT103_RAW_TXT` 指向包含 `wiki.train.raw` / `wiki.valid.raw` 的目录
# （搜索路径还包括仓库旁的 ./wt103_raw 和 /root/wt103_raw）。
# 只读这两个文件，对其分词可逐 token 复现 `wt103_cache/*.npy` 缓存，
# 因此该回退是纯可用性路径，不会改变任何测量结果。
# 若 `wt103_cache/` 已有缓存，语料直接由缓存提供，完全无需下载。
# v12 P5T 的独立最终测试另读 `wiki.test.raw`（load_wikitest，同一目录
# 搜索路径；该 split 此前任何阶段都不读）。更换 train 语料后请删除整个
# wt103_cache/ 重建（test 缓存不会跟随静默重建）。

# 1) 端到端校验整个流水线（smoke，不 push）：
#    （除 dense / 固定 / 动态臂外，还覆盖去一法消融的全部独有前向路径：
#     随机 indexer、零内容池化、关 sink、learned fusion、HCA 层、dense+sink 对照）
python run_gapfill.py smoke
python v10_supp.py smoke           # 同时覆盖新腐坏模式 + NIAH + 从头训练对照
python test_open_block.py         # 未结束块因果性/参考实现/掩码门禁（CPU）
python open_block_supp.py smoke    # 未结束块矩阵的微型合成自检
python open_block_supp.py main     # 8 配置 × 4 种子；可续跑
python open_block_supp.py report   # 配对差 + 交互效应 + 精确符号翻转/Holm
python open_block_supp.py aux      # 窗口外辅助面板
python open_block_supp.py extended # 应急：4 上限 × 3 模式
python v12_supp.py smoke           # 同时覆盖冻结最终测试的完整往返

# 2) 正式补充实验（受运行时上限约束）：
nohup python run_gapfill.py full > run_v6.log 2>&1 &
tail -f run_v6.log
```

`run_gapfill.py full` 按顺序运行（每阶段可断点续跑；重跑同一命令会跳过
已完成的 `variant::seed` 组合）。便宜且统计上根本性的面板排在最前，
受运行时上限约束的机器先拿到 n≥3 的核心/消融/规模表，20k 步长跑最后；
每个阶段都至少排满方法学审计要求的 3-seed 下限（`min_n`），个别面板
再由补种子阶段（P2S / P3T / P4SS / P5C …）加到 n=4–6：

| 阶段 | 输出目录 | 内容 |
|---|---|---|
| P-CORE | `results_lm_v3_1500` | 9 变体核心表，1500 步 × 3 seeds |
| P-ABL | `results_lm_v4_abl` | 去一法消融表——面板内参照臂（`full` / `csa_fixed` / `csa_dynamic` / `hybrid_fixed` / `hybrid_dynamic`）+ `csa_fix_*` / `csa_dyn_*` 去模块臂，以及 fuse/HCA/topk 臂，1500 步 × 3 seeds |
| P-SCALE | `results_lm_v5_scale` | d=384 / 8 层 / seq-1024 探针，5 变体 × 3 seeds × 4000 步 |
| P-LONG | `results_lm_v3_long` | hybrid_fixed / hybrid_dynamic / hybrid_csa_dyn，seeds 0-2 → 三 seed 长跑表 |

`full` 受本节末尾描述的运行时上限约束。

### 任务级并行（每个实验一张卡）

在多卡机器上，所有驱动都可以并行运行独立单元——每个 `(variant, seed)`
训练一个进程，**一卡一进程**。严格可选、默认关闭：不设置变量时行为与上面
的串行描述完全一致。

```bash
# 每张可见 GPU 一个 worker（推荐）：
CSA_PAR=auto nohup python v11_supp.py full > run_v11.log 2>&1 &

# 只有 2 张卡、且只用这两张：
CSA_PAR=2 CSA_PAR_GPUS=0,3 python v10_supp.py phase P3SL

# 回到 100% 串行（默认）：不设 CSA_PAR
```

它保证的性质（完整契约见 `docs/design_notes.md` §13）：

- **数字完全一致**——每个单元在自己的进程里独立播种 RNG，并行结果与串行
  逐位一致；指纹、续跑规则与 `CODE_SEMANTICS` 均不变；
- **合并安全的产物**——所有 `summary.json` 更新都走 flock 保护的
  读-改-写，并发结束的实验不会互相丢记录；聚合 / 画图 / git push / 关机
  只在父进程做一次；
- **运行时上限下的诚实计量**——子进程先把预计用时预扣到共享 `TimeGuard`
  状态，跑完后退还差值，因此并发启动的单元依然尊重上限；上限按
  **GPU-秒** 计量（4 卡 × 1 小时 = 计 4 小时）；
- **失败语义与串行一致**——失败单元写入错误记录后阶段继续；子进程崩溃则
  像串行异常一样中止该阶段，已完成单元留在磁盘上，重跑即续。

每个单元的日志在 `par_jobs/<phase>_<timestamp>/`（已 gitignore）；主日志
输出 start/done/heartbeat 行，并尾部回显失败子进程的日志。smoke 模式
无论 `CSA_PAR` 如何都强制串行。

### B 档 M0 / S1（扩规模入口，已选定 B 档）

B 档规模计划（入口 `m0_supp.py`；原 `docs/compute_plan.md` 已退役）在 WikiText-103 机制层之上新增第二条语料路径与
S1–S4 规模阶梯，代码入口为 `m0_supp.py`：

```bash
python m0_supp.py params   # 零 GPU：S1–S4 参数量核算（对照规划值）
python m0_supp.py data     # 构建 / 校验 FineWeb-Edu 缓存（M0_TOKENS=… 可调 cap）
python m0_supp.py smoke    # 端到端自检：B 档数据路径 + 三条注意力路径
python m0_supp.py kernel   # G0 套件：kernel/eager 数值对照 + 吞吐基准
python m0_supp.py full     # S1 主面板（5 臂 × 4 seeds；CSA_PAR=auto 可用）
M0_PANEL=S2 python m0_supp.py full   # S2 主面板（5 臂 × 3 seeds × 12B；M2_CAP_HOURS 可覆盖）
```

- 语料：FineWeb-Edu（默认 `sample-10BT` 配置），自训 32k BPE；npy 缓存 + sha256
  内容指纹，纪律与 `wt103_cache/` 完全一致；token id 以 2 字节落盘、int32 供给，
  取批后照旧回宽 int64，下游路径不变；
- 离线 / 内网机器：`FWEDU_LOCAL_JSONL` 指向本地 jsonl/txt 文档（纯可用性回退——
  不同语料内容必然产生不同指纹，不会与真缓存混淆）；
- `full` 受 `M0_CAP` 上限约束（状态文件 `run_time_state_m0.json`，`M0_CAP_HOURS`
  可覆盖）：Triton 内核通过 G0 判据之前，eager 路径的预计耗时会被拒绝启动——
  这是规划文档自身的前提判断（§3），不是缺陷。

### Triton 内核路径（M0/G0）

`csa_kernels.py` 把规模计划的三个核心算子实现为融合内核——块压缩
池化（[A 泳道 | overlap 泳道] 拼接后的逐通道掩码 softmax + 位置嵌入）、
索引打分 + top-k 选择（逐轮首达最大值提取，与 torch 稳定降序排序严格同
序，tie 取最早下标）、稀疏注意力（[top-k 块 | 滑窗] 泳道上的 flash 式在线
softmax，sink 作为归一化初值质量）。每个算子另有同一套算法分解的 torch
参考孪生实现，任意设备可跑，CPU 侧的语义验证走的正是它。

内核路径是**独立变体**而非逐位置写：G0 判据是与 eager 的统计等价（同
种子、同数据）；eager 路径本身零改动——`CSA_KERNEL` 未设置时 run_cfg
指纹与历史版本逐字节一致，开启后指纹与 `_code` 戳追加 `-tk1:<backend>`
后缀，内核记录只与同模式记录配对（design_notes §3.1）。反向全部为固定
顺序归约（无原子加），跨目的地累加走稳定排序段和；eager 直通 soft
先验在前向值上严格抵消，反向按 A/S 双路 jacobian 重建，与 eager 的
autograd 图一一对应。

```bash
python m0_supp.py kernel                 # 任意设备：ref 后端语义验证
CSA_KERNEL=1 python m0_supp.py kernel    # CUDA + triton：完整 G0 套件
CSA_KERNEL=1 python m0_supp.py full      # 内核模式 S1 主面板（G0 通过后）
```

- `CSA_KERNEL` 未设置 / `0`：全部 eager（默认，什么都不变）；`1`：强制
  CUDA + triton（缺席即硬报错）；`auto`：可用则 triton，否则 torch 参考；
  `ref`：强制参考后端（CPU 可调试，分发与 autograd 封装同一套）。
- 逐算子覆盖（GPU 排查用）：`CSA_KERNEL_POOL` / `_IDX` / `_ATTN` ∈
  {`triton`, `ref`}；`CSA_KERNEL_TF32=0` 把 `tl.dot` 切到 ieee fp32
  （逐位口径排查；默认 tf32 走张量核吞吐）。
- 内核模式使用独立的 TimeGuard 状态文件（`*.k`）：两档吞吐标定互不污染。
- 支持的层类型：learned indexer、无 fuse 卷积的 `csa` / `hca`——恰好
  覆盖 B 档的 `csa_fixed` / `csa_dynamic` / `hybrid_dynamic`；其余臂
  硬报错，绝不在同一面板里静默混入两种实现。

### v11 统计收尾套件

```bash
python v11_supp.py smoke
nohup python v11_supp.py full > run_v11.log 2>&1 &
python v11_supp.py report         # 从磁盘重建 REPORT_v11.md（零 GPU）
```

阶段：P4MT（RoPE 臂 seeds 0-5，完全沿用 P1L/P3MT 配方；续跑时跳过代码
指纹仍是当前版本的 checkpoint）、P4MP（同 seed 集的探针续跑，仅评估，
过期单元重新测量）、P4SS/P4SL（交叉面板 seeds 0-3，同一续跑规则）、
P4F（d=384 的 `csa_fixed`/`full`，seeds 0-3，同一 config，使面板配对到
n=4）。

### v10 机制 / 扩规模 / 统计套件

```bash
# v10：干扰注入机制探针 + 交叉扩规模 + topk 补 seed
python v10_supp.py smoke
nohup python v10_supp.py full > run_v10.log 2>&1 &
python v10_supp.py analysis       # 零 GPU 统计 -> analysis_v10/
python v10_supp.py report         # 从磁盘重建 REPORT_v10.md（零 GPU）
```

阶段（均可续跑）：P3MT、P3MTC、P3MP、P3T、P3SS/P3SL、P3SL2/P3SL3。详见驱动脚本与 `docs/design_notes.md`。

### v12 确认套件（探索 / 确认分离）

```bash
python v12_supp.py smoke           # 端到端自检（微型 ckpt + 冻结清单副本）
python v12_supp.py plan            # 零 GPU 预检：语义戳 + 实现绑定 + 准入 + 功效，不读 test
python v12_supp.py phase P5C       # 按清单种子训练全部四个冻结臂（可续跑；主臂 0-5 由 P3MT/P4MT 覆盖）
python v12_supp.py phase P5MP      # 扩展探针（全部模式 + NIAH + 对照），种子取自清单
python v12_supp.py phase P5S2      # 等 token 大规模臂（8000 步）
python v12_supp.py phase P5S3      # 6ND 参数量代理预算大规模臂（参数量比例缩步）
python v12_supp.py phase P5T       # 冻结后的最终测试（读 wiki.test.raw，需冻结清单）
python v12_supp.py analysis        # 零 GPU：共享 v10 统计 + 冻结确认性统计
python v12_supp.py report          # 从磁盘重建 REPORT_v12.md（零 GPU）
```

P5T 在 `confirmation_frozen.json` 缺失时拒绝运行，代码语义戳与清单不一致
时同样拒绝（必须显式重冻结），**最小可达 Holm p > α 的清单直接拒绝**
（数学上不可能确认 = 设计缺陷，种子数 6→12 的修订因此发生），记录准入
失败（种子白名单/歧义重复/混切片/配方签名/manifest 哈希）时拒绝，首次
test 访问写入冻结锁 `.freeze_lock.json`（此后清单/test 变更即拒绝，
V12_REFREEZE=1 显式改冻结并把前轮降级为探索性）。test split
（`wiki.test.raw`，经 `load_wikitest`）**只**被 P5T 读取——此前所有
phase 只读 train/valid。首次 test split 评估之后禁止追加冻结比较或种子
（构造性选择偏差，见 `DECISION_LOG.md`）。

**清单修订 v6（2026-11-24，仍在任何 test 访问之前）** 补上审查指出的三处证据
缺口，且只收紧、不放松：

- **最小功效硬门**：`statistics.power_gate`（q=0.80，min_power=0.80）把访问前
  功效模拟从“打印警告”变成闸门——每个主比较的 panel 功效（`p5c_power.json`）
  必须落盘且不低于下限，否则 P5T 拒绝读取 test split；显式
  `V12_ACCEPT_LOW_POWER=1` 是唯一覆盖，决定写入
  `results_lm_test/power_gate.json` 并进入 `test_stats.json`。12-seed 设计自身
  q=0.80 的 Holm 保守功效仅 0.275，因此该门通常会触发：诚实做法是加种子并记录
  修订，或显式接受“分辨率受限”的确认。
- **参数匹配必须可验证**：CONFIRMED 现要求 `param_matched is true`（实测
  ≤1%）；v5 的“非 false”让无法验证的 None 与已验证等价，现不再接受。
- **逐 seed 初始化配对**：`init_core_fp` 成为 CONFIRMED 闸门，每一对可用配对的
  两臂必须记录相同的共享主干摘要；同时修复了聚合缺陷——旧实现把所有 seed 并成
  一个集合，正确的 12-seed 面板必然 False，既发现不了真正的臂间不匹配、又会
  误报正确面板。

**范围声明**：本确认只覆盖**固定分块 CSA 机制**的主张；动态分块优于固定分块、动态边界的语义价值、HCA 的优劣与动态分块的规模规律**不在本确认内**，需要各自的预注册冻结。

**默认 `csa_dynamic` 不是任务学习边界**（design_notes §22.5）：硬切分来自
`gate.detach() > 0.5`，LM 交叉熵不能通过离散切分反传；`delta_logit` /
`fuse_conv` 只从压缩率正则 `comp_reg` 得到梯度。它是「余弦阈值 + 由压缩率
正则训练的标量偏移」。可选 `csa_dynamic_task` 才是按 score-function
`+(CE − baseline)·logπ / n_decisions` 训练边界的臂（CE 为最小化成本），且
**不在**本次固化确认内。

**可执行审计入口（零实验）**：

```bash
python audit_gates.py causality   # 纯 CPU 单元审计：前缀不变性 / 标签移位 / 解码一致性
python audit_gates.py leakage     # 分词器来源 / 词表一致性 / train-test 文档重叠
python audit_gates.py report      # 运行本地可核验的全部审计 -> audit_report.json
python archive_release.py plan v13 --dirs results_lm_test analysis_v12   # 证据归档 dry-run
```

### v8 / v9 收尾套件

```bash
# v8：规模面板补齐 4 seeds + RoPE 20k 长跑
python v8_supp.py smoke
python v8_supp.py full            # 或：python v8_supp.py phase P1S / P2R
python v8_supp.py analysis        # 零 GPU 统计 -> analysis_v8/

# v9：补全 topk 扫描（bs=1）+ RoPE-20k 第三 seed + 报告
python v9_supp.py smoke
nohup python v9_supp.py full > run_v9.log 2>&1 &
python v9_supp.py report          # 从磁盘重建 REPORT_v9.md（零 GPU）
```

设计依据与这些阶段依赖的配对/可测量规则见 `docs/design_notes.md`。

### v7 回应 reviewer 套件

v7 套件（设计依据与阶段/子命令布局见 `docs/design_notes.md`）运行
reviewer 关切的实验并重建报告：

```bash
# 零 GPU：解析式 FLOPs/KV-cache + 成对符号翻转统计 + 报告重建
python v7_supp.py analysis
# 逐阶段运行（可续跑），或一次跑完整链：
python v7_supp.py phase P0E   # 40k 步长跑
python v7_supp.py phase P2S   # 规模重播种至 4 seeds
python v7_supp.py phase P1L   # 长度外推
python v7_supp.py phase P1T   # seq-2048 topk 扫描
python build_report.py        # 从产物重建 REPORT_v7.md（零 GPU）
```

所有驱动都按配置的运行时上限计量自己的加速器时间：超过上限的运行拒绝
启动，运行中途到达上限则被硬截断。多卡机器上 `CSA_PAR=auto` 以一卡一
实验并行运行相同阶段（上限此时按 GPU-秒总账计量，即 N 路并行按 N 倍
墙钟速度消耗）——被截断的单元不会留下不完整测量，会在下一轮重试。全部
阶段完成后脚本自动 push 并关机（取消关机：`pkill -f v6autoshutdown`；
禁止关机：`V6_NO_SHUTDOWN=1`；禁止 push：`V6_NO_PUSH=1`）。

> 量值一律以 `build_report.py` / `fuse_analysis.py` 与各 `v*_supp.py report`
> 构建器从 `results_*/`、`analysis_*/` 落盘产物现算出的表格为准，仓库正文
> 不写死任何量值。

## 评审十硬化（review 10，本轮）

详见 `docs/design_notes.md` §28 与英文 README 的 Review-10 hardening 节；
要点（只改实现/报告/文档，不改冻结清单字节与测量语义）：
- 所有报告头打印证据状态：无完整 `RELEASES.md` 发布行时，数字**不能独立审计**。
- v12 主表旁原样打印冻结清单的 `interpretation_caveat` / 计算预算披露；每个
  主比较标 `single_factor_contrast=false` / `system_level_only=true`——CONFIRMED
  只能读作「该冻结配方下的系统级臂间差异」。
- 随机选择对照是**固定确定性哈希**（非按 seed/layer 变化、非每次前向重随机），
  其 indexer 投影是死参数（202,752，≈1.83%）；`param_breakdown` 现报告
  `dead_params`/`active`，报告逐臂打印有效参数。
- 冻结锁含 `superseded` 时当前轮强制 `confirmatory_round=false`、`confirmed=[]`：
  test 看过之后重冻结不恢复独立性。
- 固定配方确认 ≠ 两侧同搜索预算调参后的最佳方法比较。
- NIAH 增加 chance、`nofar` 无远端信息对照、距离分桶与任务有效性门；缺少
  正对照时 NIAH 只能作描述性读数。
- 腐坏探针报告**实际覆盖/实际改变 token/空操作行**：ρ 是名义值，span 段
  互不重叠但受段长量化，donor 撞车使 changed < covered，跨模式同 ρ 不是
  同强度。
- 精确符号翻转检验的零假设是**符号关于 0 对称**；共享 seed/数据顺序/初始化
  是控制而非随机化。功率产物增加 q_hat/Wilson 下界/留一与异常种子敏感性；
  `V12_SKIP_POWER=1` 在清单声明功效门时必须同时 `V12_ACCEPT_LOW_POWER=1`。
- 单位：冻结 v12 的 16 个评测单位是固定窗口/行，不是 16 篇文档；`48` 是
  arms×seeds=4×12 的记录数，不是 48 篇文档。PPL@L 是嵌套前缀目标，不是
  同目标跨长度比较。

## 评审四硬化（本轮）

逐条核对见 `DECISION_LOG.md` 的评审 4/5 行（本快照没有 `REVIEW_4_RESPONSE.md`）。核心变化是把原先的软约束变成机器闸门：

- 报告措辞统一走 `exp_lib.reportable_claim`：review-10 起拆成**统计显著性**
  （n≥6、原始 p、Holm p、族级最小可达 p）、**稳定性**（|mean|>std、
  留一法符号稳定）与**实际价值**（仅当预先声明 `min_effect`；v12 未预注册，
  故不作实际价值声明）三个独立判决；|mean|>std 是稳定性诊断、不是显著
  性判据。v12 的 CONFIRMED 规则不使用该阈值。`attach_holm` 输出族级
  分辨率上限，v10 的 NIAH 对照并入同一 Holm 族。
- 交叉点标度改用 `crossover_scaling_fit`：≥4 个去重宽度、任一点删失即
  拒绝、训练量差 >25% 拒绝、斜率带 95% CI；否则不报告斜率。
- 每条训练记录新增 `init_core_fp`，`pair_reason` 拒绝共享 trunk 初始化
  不同的配对；`_compute_control_detail` 报告实测计算量比及方向，偏袒
  赢家的混杂结果标为 `system_level_only` 而非 `verified`。
- 陈旧记录 fail-closed：无当前代码戳的按长度臂整臂排除，不列表、不算
  ratio；缺失 `truncated` 标记按不可验证处理。
- P2 表显示 row/window bootstrap CI（非文档）；v12 主比较表的行/窗口 Δ CI
  单位是固定评测窗口，不是文档。`MIN_DOCS_FOR_POPULATION_CLAIM=64` 是仓库
  自设最小总体门槛（无统计推导，且计数单位是行/窗口）；16 个窗口不覆盖
  语言总体，总体主张一律拒绝。
- 长度阶段记录实测效率；新增 `measure_efficiency_sweep` 与
  `audit_gates.py efficiency_pairs`，只有同形状/同设备/同 KV 语义且
  比值超出 ±5% 才允许说「更快」。
- 新增 `audit_gates.py selection`：选择召回工件缺失时明确「学习选择器
  找到相关信息」为未支持主张；`release` 在 `RELEASES.md` 无完整行时
  声明仓库 source-only、数值不可独立复核。
