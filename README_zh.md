# Dynamic-Block CSA + HCA — WikiText-103 对照实验

[English](README.md) | **中文说明**

在小型从零训练语言模型（BPE-8k，WikiText-103）中，对**压缩块稀疏注意力**
（CSA / HCA，固定分块与动态分块）与稠密注意力进行受控对比研究。本仓库包含
实验代码以及在自己机器上复现结果所需的全部内容。

> **证据等级**：v6–v11 为探索性（在同一 validation split 上选择后观察）；只有 v12 冻结的固定分块比较是确认性。详见 `DECISION_LOG.md` 与 `docs/design_notes.md` §17/§20。
>
> **版本命名**：`v12` 是驱动/研究代次；`CODE_SEMANTICS = v11.125` 是冻结清单绑定的库测量语义戳，两者故意不同号。

## 仓库结构

仓库只保留**源码、驱动脚本与设计文档**。所有实验结果目录（`results_*/`）、
零 GPU 分析目录（`analysis_*/`）、逐版报告（`REPORT*.md`）与运行状态文件
都**不入库**——它们由驱动脚本在跑实验的机器上重新生成，需要时本地复现即可
（见下文命令）。`.gitignore` 已把这几类路径挡在仓库外（`docs/` 只保留
`design_notes.md` 与 `compute_plan.md`）。

| 路径 | 说明 |
|---|---|
| `exp_lib.py` | **核心库，也是唯一的代码来源**：模型、CSA/HCA 注意力、池化、indexer、分段器、训练循环、聚合与统计 |
| `csa_kernels.py` | **三算子融合内核路径**（docs/compute_plan.md §3 的 M0 工作包）：块压缩池化、索引 top-k 选择与稀疏注意力的 Triton 内核（附带同一套算法分解、任意设备可跑的 torch 参考孪生实现）；`CSA_KERNEL` 选开、默认关闭——见下文「Triton 内核路径（M0/G0）」 |
| `run_gapfill.py` | **v6 驱动**：最小可发表补充实验，支持断点续跑，受运行时上限约束，每阶段完成后自动 push，结束后自动关机 |
| `par.py` | **任务级并行执行器（可选）**：每个独立实验占一张 GPU 卡——见下文「任务级并行」 |
| `v7_supp.py` | **v7 驱动**：回应 reviewer 的补充实验套件——P0R RoPE+QK-norm、P0W 稠密预热、P0E 40k 步长跑、P2S 规模重播种、P1L 长度外推、P1T topk 扫描 |
| `v8_supp.py` | **v8 驱动**：P1S（规模面板补齐至 4 seeds）与 P2R（RoPE+QK-norm 20k 步长跑） |
| `v9_supp.py` | **v9 驱动**：P2T（seq-2048 的 topk 扫描，`batch_size=1`）与 P2S3（RoPE 20k 长跑的第三个 seed） |
| `v10_supp.py` | **v10 驱动**：P3M 长度外推机制 + 干扰项注入探针、P3S 交叉面板扩规模、P3T topk 扫描补至 4 seeds |
| `v11_supp.py` | **v11 驱动**：统计收尾——干扰探针扩到 n=6（P4MT/P4MP）、交叉面板扩到 n=4（P4S）、d=384 配对臂扩到 n=4（P4F） |
| `v12_supp.py` | **v12 确认驱动（评审响应 + 评审二轮加固）**：P5C 按清单种子列表训练全部四个冻结臂，P5MP 跑扩展腐坏模式探针（uniform/realistic-token/mask/delete + 评审二轮新增 span/doc/topical + NIAH-natural），P5S2/P5S3 补等 token 与 6ND 参数量代理预算的大规模臂，P5T 按 `confirmation_frozen.json` 在 held-out 的 WikiText-103 test split 上做**冻结后一次性确认评估**——含加载期功效硬门槛（数学上不可能确认的清单直接拒绝）、严格记录准入（种子白名单/唯一记录/单一冻结切片/配方签名/manifest 哈希）、首次访问冻结锁与逐文档 PPL 落盘 |
| `open_block_supp.py` | **未结束块读取面板驱动**：最大块长（32/64/128/无强制上限）× 当前未结束块读取方式（`closed` / `prefix_summary` / `prefix_singletons`）；共享 top-k 预算、因果前缀构造、窗口外辅助面板、逐种子配对差与交互统计——见 `docs/design_notes.md` §24 |
| `test_open_block.py` | **未结束块 CPU 门禁**：配置/分段器、优化前缀池化与逐查询参考的前向/梯度一致、前缀不变性、未来扰动/未来梯度、候选掩码、边界情形、closed 回归、单 token 语义与 HCA 预算声明 |
| `confirmation_frozen.json` | **预注册冻结清单 v2**（入库）：冻结臂、种子 0-11（在任何 test-split 访问**之前**由 6 修订为 12，含功效表）、终点、精确符号翻转 + Holm、决策规则、机器可校验的绑定块与范围声明——见 `docs/design_notes.md` §17 与 `DECISION_LOG.md` |
| `DECISION_LOG.md` | **决策日志**（入库；因快照只有一个压缩提交，2026-10-18 依据 README/设计说明/清单重建）：每个版本为什么跑/不跑什么、看到什么、据此决定了什么——探索/确认分离的书面轨迹；正式版本的逐种子证据另由 `archive_release.py` 归档 |
| `audit_gates.py` | **评审二轮的可执行审计入口**（零实验）：数据泄漏（分词器来源/词表一致性/文档重叠）、纯 CPU 因果性单元审计（前缀不变性/标签移位/解码一致性）、分块归因/边界 vs 随机基线/长度口径/基线公平/失败集中度/预算对账与 prefill-decode-内存测量工具 |
| `archive_release.py` | **不可变证据归档器**：把每个正式版本的产物（结果/分析/报告/冻结锁/决策日志/环境/权重哈希）打包为带逐文件 sha256 清单的 tar.gz，附独立 verify 子命令——Git 主仓仍只承载源码（design_notes §8），归档发布到只读位置 |
| `m0_supp.py` | **M0/S1 驱动（已选定 B 档）**：扩规模路线图的代码入口——`params`（零 GPU 的 S1–S4 参数量核算）、`data`（FineWeb-Edu + 32k BPE 语料缓存）、`smoke`（B 档数据路径端到端自检）、`kernel`（G0 验证套件：三算子 kernel/eager 数值对照 + 稀疏注意力吞吐基准）、`full`（S1 主面板）；见下文“B 档 M0/S1”与 `docs/compute_plan.md` |
| `build_report.py` | 零 GPU 报告构建器：从 `results_*/` + `analysis_*/` 产物重建 v7 报告（所有数字现算，绝不硬编码） |
| `fuse_analysis.py` | 零 GPU fuse 对比驱动：从 `results_*/summary.json` 重算逐层边界对齐（tol=1）、块长统计与 PPL 曲线 |
| `docs/design_notes.md` | **设计与实现笔记**：块读门为何必须严格、且五处位置必须同步；动态分段器的前缀不变性要求；配对/可测量规则；报告约定；保证注意力按行分块与不分批形式等价的契约 |
| `docs/compute_plan.md` | **算力规划**：扩规模路线图（0.14B–1.3B 阶梯、FineWeb-Edu、Triton 内核前提）、按 H800 卡时折算的预算口径与 M0–M4 里程碑判据；原申请表草稿不再入库 |

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

B 档（docs/compute_plan.md §4–§6）在 WikiText-103 机制层之上新增第二条语料路径与
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

`csa_kernels.py` 把 compute_plan §3 的三个核心算子实现为融合内核——块压缩
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

阶段（均可续跑）：P3MT（重训两个 RoPE 臂，P1L 配方）、P3MTC（**同预算
从头训练的选择器对照**：`csa_fixed_rope_randidx` / `csa_fixed_rope_allblocks`，
后者同时充当逐项对齐的 dense 孪生臂）、P3MP（仅评估的远距离干扰注入：
成对污染上下文、干净的 512-token 目标，对比 learned-selection / dense /
random-indexer / see-everything 各臂，**uniform / realistic-token（natural）/
mask / delete + 评审二轮新增 span / doc / topical 腐坏模式** + NIAH-natural
远端必要信息任务）、P3T（topk 扫描 seeds 2/3 → n=4）、P3SS/P3SL（
d=128/4L 8000 步与 d=512/10L 4000 步交叉面板）、P3SL2/P3SL3（**等 token
与 6ND 参数量代理预算的大模型臂**——规模混杂修复，代理 ≠ 实测等
FLOPs，见 `docs/design_notes.md` §16）。成对污染协议、预注册交叉判据、
腐坏模式/ΔNLL 口径（§15）与「测试时干预 vs 从头训练对照」的区别
（§18）见 `docs/design_notes.md`。

### v12 确认套件（探索 / 确认分离）

```bash
python v12_supp.py smoke           # 端到端自检（微型 ckpt + 冻结清单副本）
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

**范围声明（评审二轮·意见四）**：本确认只覆盖**固定分块 CSA 机制**的主张；
动态分块优于固定分块、动态边界的语义价值、HCA 的优劣与动态分块的规模
规律**不在本确认内**，需要各自的预注册冻结。

**可执行审计入口（评审二轮·意见十一，零实验）**：

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
