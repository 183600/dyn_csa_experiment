# Dynamic-Block CSA + HCA — WikiText-103 对照实验

[English](README.md) | **中文说明**

本仓库包含实验代码，以及在自己机器上**重跑**一项受控研究所需的协议。该研究
在小型从零训练语言模型（BPE-8k，WikiText-103）中，对比**压缩块稀疏注意力**
（CSA / HCA，固定分块与动态分块）与稠密注意力。

这是一个**只承载源码**的仓库：没有随仓库发布数值证据归档（见
`RELEASES.md`）。因此，任何你没有用自己产物重建过的报告 PPL 数值，仅凭
Git 树无法独立复核。仓库中的主张仅限 `confirmation_frozen.json` 冻结的
范围；其余均为探索性。

> **版本命名**：`v12` 是驱动/研究代次；`CODE_SEMANTICS = v11.128` 是冻结
> 清单绑定的库测量语义戳。两套编号故意不同。冻结清单当前为
> `manifest_version: 3`（见其 `revision.history`）。

> **证据等级**：v6–v11 为探索性（在同一 validation split 上观察后继续
> 扩展面板）；只有 v12 冻结的固定分块比较是确认性。详见 `DECISION_LOG.md`
> 与 `docs/design_notes.md` §17/§20。

> **证据状态（引用数字前必读）**：`results_*/`、`analysis_*/` 与
> `REPORT*.md` 有意不入库，且**没有随仓库发布证据归档**
> （`RELEASES.md` 尚无记录）。报告可由你自己产生的产物重建，但没有重建过
> 的报告数值无法独立复核。只有当 `release_<tag>.tar.gz` 与清单发布到只读
> 位置，并通过 `archive_release.py verify` 时，相应主张才可审计。

## 目录

- [研究一览](#研究一览)
- [从这里开始](#从这里开始)
- [快速开始](#快速开始)
  - [示例流程](#示例流程)
- [从哪里读起](#从哪里读起)
- [仓库结构](#仓库结构)
  - [核心代码](#核心代码)
  - [驱动模块](#驱动模块)
  - [测试与冻结输入](#测试与冻结输入)
  - [证据与设计记录](#证据与设计记录)
  - [工具](#工具)
  - [重新生成实验产物](#重新生成实验产物)
- [复现方法（任意机器）](#复现方法任意机器)
  - [环境](#环境)
  - [数据与缓存回退](#数据与缓存回退)
  - [训练前校验流水线](#训练前校验流水线)
  - [运行 v6 补充实验](#运行-v6-补充实验)
  - [任务级并行（每个实验一张卡）](#任务级并行每个实验一张卡)
  - [B 档 M0 / S1（扩规模入口，已选定 B 档）](#b-档-m0--s1扩规模入口已选定-b-档)
  - [Triton 内核路径（M0/G0）](#triton-内核路径m0g0)
  - [v11 统计收尾套件](#v11-统计收尾套件)
  - [v10 机制 / 扩规模 / 统计套件](#v10-机制--扩规模--统计套件)
  - [v12 确认套件（探索 / 确认分离）](#v12-确认套件探索--确认分离)
  - [v8 / v9 收尾套件](#v8--v9-收尾套件)
  - [v7 回应 reviewer 套件](#v7-回应-reviewer-套件)
  - [常见注意事项](#常见注意事项)
- [测量护栏](#测量护栏)
- [延伸阅读](#延伸阅读)

## 研究一览

驱动按研究代次组织。每个驱动都有用于流水线自检的 `smoke` 模式、用于正式
面板的训练模式，以及在适用时提供零 GPU 的分析/报告子命令。v6 保留历史
脚本入口；v7 及之后位于 `explib/` 下。

| 代次 | 范围 | 入口 | 生成的产物 |
|---|---|---|---|
| v6 | 核心 CSA/HCA 表、去一法消融、规模探针、长跑 | `python run_gapfill.py full` | `results_lm_v3_1500/`、`results_lm_v4_abl/`、`results_lm_v5_scale/`、`results_lm_v3_long/` |
| v7 | RoPE + QK-norm、长度外推、top-k 扫描 | `python -m explib.rope` | `REPORT_v7.md` |
| v8 | 规模面板补种子与 20k RoPE 长跑 | `python -m explib.scale` | `analysis_v8/` |
| v9 | seq-2048 top-k 收尾与第三个 20k seed | `python -m explib.resume` | `REPORT_v9.md` |
| v10 | 干扰项注入机制探针、交叉扩规模、选择器对照 | `python -m explib.corruption` | `analysis_v10/`、`REPORT_v10.md` |
| v11 | v10/v11 面板的统计收尾 | `python -m explib.length` | `analysis_v11/`、`REPORT_v11.md` |
| v12 | 冻结的固定分块确认 | `python -m explib.confirmation` | `analysis_v12/`、`REPORT_v12.md` |
| M0 | FineWeb-Edu 数据路径、内核 G0 门、S1/S2 规模面板 | `python -m explib.m0` | `results_b_s1/`、`results_b_s2/`、`analysis_m0/` |
| open-block | 最大块长 × 未结束块读取规则 | `python -m explib.open_block` | `analysis_open_block/` |

训练从不隐式启动：只有名为 `full`、`phase <name>`、`main` 或 `auto` 的模式
才会开始训练。各驱动的模式见下表；`--help` 打印驱动说明，argparse 驱动的
子命令也会一并列出。

| 驱动 | 模式 |
|---|---|
| `python run_gapfill.py` | `smoke`、`overfit`、`topup`、`full`（默认 `full`） |
| `python -m explib.rope` | `smoke`、`analysis`、`phase P0R` / `P0W` / `P0E` / `P2S` / `P1L` / `P1T`、`full` |
| `python -m explib.scale` | `smoke`、`analysis`、`phase P1S` / `P2R`、`full` |
| `python -m explib.resume` | `smoke`、`analysis`、`report`、`phase P2T` / `P2S3`、`full` |
| `python -m explib.corruption` | `smoke`、`analysis`、`report`、`phase <name>`、`full` |
| `python -m explib.length` | `smoke`、`analysis`、`report`、`phase <name>`、`full` |
| `python -m explib.confirmation` | `smoke`、`plan`、`dynamic-plan`、`power`、`phase <name>`、`analysis`、`report`、`full`、`auto` |
| `python -m explib.open_block` | `smoke`、`pilot`、`main`、`aux`、`extended`、`report` |
| `python -m explib.m0` | `params`、`data`、`smoke`、`kernel`、`analysis`、`full`、`auto` |
| `python -m explib.reproduce` | `compare 原始目录 重跑目录`、`report OLD.json NEW.json` |

## 从这里开始

所有命令均在仓库根目录执行。`make test` 通过即表示检出可用。

| 目标 | 先运行 | 会启动训练吗？ |
|---|---|---|
| 检查代码与环境是否正常 | `python -m pip install -r requirements.lock`，再 `make test` | 不会 |
| 预检冻结清单，不读 test split | `make plan` | 不会 |
| 运行未结束块因果性/参考实现门禁 | `python -m pytest -q test_open_block.py` | 不会 |
| 端到端试跑整条流水线 | `python run_gapfill.py smoke` | 仅 smoke 小跑 |
| 运行或续跑正式面板 | [研究一览](#研究一览)中的驱动表 | 会 |
| 从本地产物重建报告 | `python build_report.py`，或该驱动提供 `report` 模式时使用它 | 不会 |
| 比较两次运行是否一致 | `python -m explib.reproduce compare 原始目录 重跑目录` | 不会 |
| 理解某个测量为什么这样定义 | [从哪里读起](#从哪里读起) | 不会 |

## 快速开始

需要 Python 3.12。`requirements.lock` 是测试与零 GPU 分析路径使用的已验证
CPU 环境；CUDA 机器请先安装与本机驱动匹配的 torch wheel，再安装
`requirements.txt` 中的范围（见[环境](#环境)）。

```bash
python -m pip install -r requirements.lock
make test                              # 完整 CPU pytest 套件
make plan                              # 冻结清单预检；不读取 test split
python -m explib.confirmation --help   # 列出子命令，不启动实验
```

以上命令都不会启动训练。训练始终需要显式调用驱动的 `full`、`phase <name>`
或 `auto` 模式；修改流水线后，先用该驱动的 `smoke` 模式验证。

### 示例流程

在动正式面板前先校验检出：

```bash
make test                              # CPU 套件
python run_gapfill.py smoke            # v6 流水线自检，不产生研究测量
python -m explib.confirmation plan     # 冻结清单预检，不读 test
```

逐阶段运行冻结确认链（每阶段可续跑；只有 `P5T` 读取 test split）：

```bash
python -m explib.confirmation plan
python -m explib.confirmation phase P5C
python -m explib.confirmation phase P5MP
python -m explib.confirmation phase P5S2
python -m explib.confirmation phase P5S3
python -m explib.confirmation phase P5T
python -m explib.confirmation analysis
python -m explib.confirmation report
```

在相信重跑报告前，先与原始产物比较：

```bash
python -m explib.reproduce compare results_lm_v6 results_lm_v6_rerun
```

### 从哪里读起

| 问题 | 文档 |
|---|---|
| 实验如何运行或重跑？ | 本 README |
| 该用哪个命令和子命令？ | [研究一览](#研究一览)与下文各驱动小节 |
| 某个测量或比较为什么这样定义？ | `docs/design_notes.md` |
| 做过哪些决定，顺序如何？ | `DECISION_LOG.md` |
| 已发布的证据在哪里？ | `RELEASES.md` |
| 评审问题是如何处理的？ | `REVIEW_4_RESPONSE.md` |

## 仓库结构

仓库只保留**源码、驱动脚本与设计文档**。所有实验结果目录（`results_*/`）、
零 GPU 分析目录（`analysis_*/`）、逐版报告（`REPORT*.md`）与运行状态文件
都**不入库**——它们由驱动脚本在跑实验的机器上重新生成。`.gitignore` 已把
这几类路径挡在仓库外；`docs/` 只保留固定设计说明。

### 核心代码

| 路径 | 说明 |
|---|---|
| `exp_lib.py` | **兼容 shim**，重新导出拆分后的 `explib/` 实现 |
| `_split_loader.py` | 拆分包的共享加载器：按声明顺序把 `_parts/` 片段执行到调用方模块的 globals 中，并以片段路径作为 compile 文件名（traceback 指向可审查的源码） |
| `explib/` | **核心库，也是唯一的代码来源**：模型、CSA/HCA 注意力、池化、indexer、分段器、训练循环、聚合与统计 |
| `csa_kernels.py` | **兼容 shim**，重新导出 `_csa_kernels/` 实现 |
| `_csa_kernels/` | **三算子融合内核路径**：块压缩池化、索引 top-k 选择与稀疏注意力的 Triton 内核（附带同一套算法分解、任意设备可跑的 torch 参考孪生实现）；`CSA_KERNEL` 选开、默认关闭——见下文 [Triton 内核路径](#triton-内核路径m0g0) |
| `run_gapfill.py` | **v6 驱动**：核心/消融/规模/长跑面板，支持断点续跑，受运行时上限约束，每阶段完成后自动 push，结束后自动关机 |
| `par.py` | **任务级并行执行器（可选）**：每个独立实验占一张 GPU 卡——见下文[任务级并行](#任务级并行每个实验一张卡) |
| `data_guard.py` / `_data_guard/` | 防泄漏预处理与编码基础函数；shim 重新导出拆分包 |

### 驱动模块

| 路径 | 说明 |
|---|---|
| `explib/rope/` | **v7 驱动**（`python -m explib.rope`）：回应 reviewer 的补充实验套件——P0R RoPE+QK-norm、P0W 稠密预热、P0E 40k 步长跑、P2S 规模重播种、P1L 长度外推、P1T topk 扫描 |
| `explib/scale/` | **v8 驱动**（`python -m explib.scale`）：P1S（规模面板补齐至 4 seeds）与 P2R（RoPE+QK-norm 20k 步长跑） |
| `explib/resume/` | **v9 驱动**（`python -m explib.resume`）：P2T（seq-2048 的 topk 扫描，`batch_size=1`）与 P2S3（RoPE 20k 长跑的第三个 seed） |
| `explib/corruption/` | **v10 驱动**（`python -m explib.corruption`）：P3M 长度外推机制 + 干扰项注入探针、P3S 交叉面板扩规模、P3T topk 扫描补至 4 seeds |
| `explib/length/` | **v11 驱动**（`python -m explib.length`）：统计收尾——干扰探针扩到 n=6（P4MT/P4MP）、交叉面板扩到 n=4（P4S）、d=384 配对臂扩到 n=4（P4F） |
| `explib/confirmation/` | **v12 确认驱动**（`python -m explib.confirmation`；评审响应 + 评审二轮加固）：P5C 按清单种子列表训练全部六个冻结臂，P5MP 跑扩展腐坏模式探针（uniform/realistic-token/mask/delete + 评审二轮新增 span/doc/topical + NIAH-natural），P5S2/P5S3 补等 token 与 6ND 参数量代理预算的大规模臂，P5T 按 `confirmation_frozen.json` 在 held-out 的 WikiText-103 test split 上做**冻结后一次性确认评估**——含加载期功效硬门槛（数学上不可能确认的清单直接拒绝）、严格记录准入（种子白名单/唯一记录/单一冻结切片/配方签名/manifest 哈希）、首次访问冻结锁与逐文档 PPL 落盘 |
| `explib/open_block/` | **未结束块读取面板驱动**（`python -m explib.open_block`）：最大块长（32/64/128/无强制上限）× 当前未结束块读取方式（`closed` / `prefix_summary` / `prefix_singletons`）；共享 top-k 预算、因果前缀构造、窗口外辅助面板、逐种子配对差与交互统计——见 `docs/design_notes.md` §24 |
| `explib/m0/` | **M0/S1 驱动（已选定 B 档）**（`python -m explib.m0`）：扩规模路线图的代码入口——`params`（零 GPU 的 S1–S4 参数量核算）、`data`（FineWeb-Edu + 32k BPE 语料缓存）、`smoke`（B 档数据路径端到端自检）、`kernel`（G0 验证套件：三算子 kernel/eager 数值对照 + 稀疏注意力基准）、`analysis`、`full`（S1 主面板）与 `auto`；见下文 [B 档 M0/S1](#b-档-m0--s1扩规模入口已选定-b-档) |
| `explib/reproduce/` | **跨运行复现比较器**（`python -m explib.reproduce compare 原始目录 重跑目录`）：按 `variant::seedN` 逐格比较两份 panel，分别报告源码字节、数据切片、配置、执行环境、匹配初始化与最终 PPL 容差；缺指纹一律 `provenance_incomplete`，绝不当作通过。同一检查也暴露为 `audit_gates.py reproduction 原始目录 重跑目录`——见 `docs/design_notes.md` §83 |

### 测试与冻结输入

| 路径 | 说明 |
|---|---|
| `test_open_block.py` | **未结束块 CPU 门禁**：配置/分段器、优化前缀池化与逐查询参考的前向/梯度一致、前缀不变性、未来扰动/未来梯度、候选掩码、边界情形、closed 回归、单 token 语义与 HCA 预算声明 |
| `test_review_part_*.py`、`test_review_fixes.py` | **CPU 评审 pytest 套件**：兼容入口与拆分后的测试模块（用 `make test` 或 `pytest` 运行） |
| `confirmation_frozen.json` | **预注册确认清单**（入库；`manifest_version: 3`，修订历史 v1–v5）：冻结臂与种子 0-11（在任何 test-split 访问**之前**由 6 修订为 12，含功效表）、终点、精确符号翻转 + Holm、决策规则、机器可校验的绑定块与范围声明——见 `docs/design_notes.md` §17 与 `DECISION_LOG.md` |
| `confirmation_frozen_dynamic.json` | **动态分块与 HCA 的独立预注册草案**（`manifest_version: 4`；尚未冻结）：H1 分段质量、H2 检索精度、H3 长程覆盖，以及冻结前必须排期的对照——见 `DECISION_LOG.md` |

### 证据与设计记录

| 路径 | 说明 |
|---|---|
| `RELEASES.md` | **证据发布索引**：每个不可变 `release_<tag>.tar.gz` 与 SHA-256 清单发布在哪里；Git 仓库本身只承载源码 |
| `DECISION_LOG.md` | **决策日志**（入库；因快照只有一个压缩提交，2026-10-18 依据 README/设计说明/清单重建）：每个版本为什么跑/不跑什么、看到什么、据此决定了什么——探索/确认分离的书面轨迹；正式版本的逐种子证据另由 `archive_release.py` 归档 |
| `REVIEW_4_RESPONSE.md` | **评审四逐条回应**（入库）：21 个方法学问题、仓库已强制执行的部分、本轮修订了什么，以及哪些仍需新实验/产物才能回答 |
| `docs/design_notes.md` | **设计与实现笔记**：块读门为何必须严格、且五处位置必须同步；动态分段器的前缀不变性要求；配对/可测量规则；报告约定；保证注意力按行分块与不分批形式等价的契约 |

### 工具

| 路径 | 说明 |
|---|---|
| `audit_gates.py` | **兼容 shim**，实现位于 `_audit_gates/`；即**评审二/四/七轮的可执行审计入口**（零实验）：数据泄漏（分词器来源/词表一致性/文档重叠）、纯 CPU 因果性单元审计（前缀不变性/标签移位/解码一致性）、分块归因/边界 vs 随机基线/长度口径/基线公平/失败集中度/预算对账，评审七的单因素设计门（声明对照的臂配置 diff 对声明的干预键集；冻结清单 mechanism_claim_allowed 标志对账）与 prefill-decode-内存测量工具 |
| `archive_release.py` | **不可变证据归档器**：把每个正式版本的产物（结果/分析/报告/冻结锁/决策日志/环境/权重哈希）打包为带逐文件 sha256 清单的 tar.gz，附独立 verify 子命令——Git 主仓仍只承载源码（design_notes §8），归档发布到只读位置 |
| `build_report.py` | **兼容 shim**，实现位于 `_build_report/`；即零 GPU 报告构建器：从 `results_*/` + `analysis_*/` 产物重建 v7 报告（所有数字现算，绝不硬编码） |
| `fuse_analysis.py` | 零 GPU fuse 对比驱动：从 `results_*/summary.json` 重算逐层边界对齐（tol=1）、块长统计与 PPL 曲线 |
| `baseline_tune.py` | 等搜索量基线调参驱动：所有臂使用同一份声明网格、只在 validation 上选择，并提供零 GPU 的 `plan`/`report` 审计路径 |
| `error_analysis.py` | 零训练坏例分析：对已保存 checkpoint 只在 validation split 上统计最差 token/序列并做启发式归因 |
| `Makefile` | 零 GPU 入口：`make test`、`make plan`、`make audit`、`make manipulation`、`make integrity`、`make env-check`、`make reproduce`、`make report-diff`、`make install-check`、`make build`、`make clean` |
| `pyproject.toml` / `requirements.txt` / `requirements.lock` | Python ≥3.12 与依赖声明；需显式安装（无 import 时 pip install）。`requirements.lock` 是已验证的 CPU 环境 |

> **测量库位于 `explib/`，直接编辑其中的模块。**
> `exp_lib.py`（以及 `csa_kernels.py`、`audit_gates.py`、`build_report.py`、
> `data_guard.py`）只是重新导出拆分包的兼容 shim。
>
> **驱动入口。** 历史顶层的 v7–v12、M0/S1 与未结束块脚本现已迁入 `explib/`
> 下的模块：使用 `python -m explib.rope`、`python -m explib.scale`、
> `python -m explib.resume`、`python -m explib.corruption`、
> `python -m explib.length`、`python -m explib.confirmation`、
> `python -m explib.m0`、`python -m explib.open_block`。子命令不变。

### 重新生成实验产物

结果目录与报告都不在仓库里。要重新得到它们：

```bash
python run_gapfill.py smoke       # 端到端自检，不产生研究测量
python run_gapfill.py full        # v6 主实验（受运行时上限约束）
python -m explib.<driver> --help  # 驱动说明；适用时列出子命令
python -m explib.<driver> report  # 仅对提供 report 模式的驱动重建报告
python build_report.py            # 从落盘的 results_*/ + analysis_*/ 重建 v7 报告
```

如果同一批格子跑了两遍（串行 vs 并行、换一台机器、或修改代码后重跑），
不要直接相信报告，先比较第二份产物：

```bash
python -m explib.reproduce compare results_lm_v6 results_lm_v6_rerun
# 仅当源码字节、数据切片、配置、执行环境、匹配初始化与声明的 PPL 容差
# 全部对齐时退出码才为 0；`--require exact` 要求逐位相同。
```

该检查逐维报告，因此 `environment_changed` / `data_changed` /
`code_changed` / `numerical_drift` 不会被压成一句「没复现」；缺指纹是
`provenance_incomplete`，不会被当成通过。

## 复现方法（任意机器）

以下命令按副作用分组：先准备环境，再跑零 GPU 校验，最后才启动训练。
均在仓库根目录执行。训练面板需要 GPU；测试、分析、报告构建与审计门在
CPU 上运行。

### 环境

需要 Python 3.12（见 `pyproject.toml`）。

- **已验证的 CPU 环境（测试与零 GPU 分析）：**

  ```bash
  python -m pip install -r requirements.lock
  ```

  lock 是完整的固定依赖闭包，包含 CPU 版 torch。

- **环境身份会被核对，而不只是「装上了」：** `make env-check` 会把本地各
  panel 记录的 `env_fp_v2` / `env_fp_v3` 与当前栈比较。v3 在 v2 的
  lock 闭包之上加入 BLAS/LAPACK/SIMD 的构建与运行时身份，以及未 pin
  （且非工具类）分布的实际版本，因此「同一把锁」不能悄悄代替「同一套
  数值后端」。

- **CUDA 机器：** 先安装与本机驱动匹配的 torch wheel，再安装兼容范围：

  ```bash
  python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch
  python -m pip install -r requirements.txt
  ```

  支持范围为 `torch>=2.8,<3`。lock 中的 `torch==2.14.1+cpu` 只用于 CPU
  路径；在实验机器上用 `python -m pip freeze > requirements.lock` 生成
  该机器的 lock，并随发布归档。scatter 路径已把 int32 索引转成 int64，
  不依赖某个特定的 torch ≥ 2.14 scatter 行为。

- **不会隐式安装：** import 库时不会安装任何包。只有显式设置
  `CSA_AUTO_INSTALL=1` 才会为一次性环境安装缺失依赖。

- **Hugging Face 镜像：** 默认走官方 Hub；官方端点不可达时可显式设置
  `CSA_HF_MIRROR=1` 使用 `hf-mirror.com`。

### 数据与缓存回退

- `load_wikitext` 先尝试 Hub。若 Hub 不可达，将 `WT103_RAW_TXT` 指向包含
  `wiki.train.raw` 与 `wiki.valid.raw` 的目录；加载器还会检查仓库旁的
  `./wt103_raw` 和 `/root/wt103_raw`。
- 只读这两个文件；对其分词可逐 token 复现 `wt103_cache/*.npy` 缓存。
  该回退是纯可用性路径，不会改变测量结果。
- 若 `wt103_cache/` 已有缓存，语料直接由缓存提供，无需下载或原始文本。
  `WT103_CACHE_DIR` 可迁移缓存位置；加载器、`audit_leakage` 与 P5T
  解析同一个变量。
- `python -m explib.confirmation smoke` 在 smoke 输出目录下使用隔离缓存，
  不会把不同分词结果的缓存写进冻结缓存。
- 命中 test 缓存时，除 train fit 指纹外，还会校验落盘分词器字节
  （sha256 记录在 `test_meta` 中）。
- v12 P5T 的独立最终测试另读 `wiki.test.raw`（`load_wikitest`，同一目录
  搜索路径）；此前任何阶段都不读该 split。更换 train 语料后请删除整个
  `wt103_cache/` 重建（test 缓存不会跟随静默重建）。

### 训练前校验流水线

以下命令只验证流水线，不产生研究测量。smoke 运行覆盖 dense / 固定 /
动态臂，以及去一法消融的全部独有前向路径（随机 indexer、零内容池化、
关 sink、learned fusion、HCA 层、dense+sink 对照）：

```bash
python run_gapfill.py smoke
python -m explib.corruption smoke        # 腐坏模式 + NIAH + 对照
python -m pytest -q test_open_block.py   # 未结束块因果性/参考实现/掩码门禁（CPU）
python -m explib.open_block smoke        # 未结束块矩阵的微型合成自检
python -m explib.confirmation smoke      # 冻结最终测试的完整往返
```

其余未结束块模式是真实运行：

```bash
python -m explib.open_block pilot     # 单格短程运行
python -m explib.open_block main      # 8 配置 × 4 种子；可续跑
python -m explib.open_block aux       # 窗口外辅助面板
python -m explib.open_block extended  # 应急：4 上限 × 3 模式
python -m explib.open_block report    # 配对差 + 交互效应 + 精确符号翻转/Holm
```

### 运行 v6 补充实验

```bash
nohup python run_gapfill.py full > run_v6.log 2>&1 &
tail -f run_v6.log
```

`run_gapfill.py full` 按顺序运行下列阶段（每阶段可断点续跑；重跑同一命令
会跳过已完成的 `variant::seed` 组合）。核心、消融与规模面板排在 20k 步
长跑之前；每个阶段都至少排满方法学审计要求的 3-seed 下限（`min_n`），
个别面板再由补种子阶段（P2S / P3T / P4SS / P5C …）加到 n=4–6：

| 阶段 | 输出目录 | 内容 |
|---|---|---|
| P-CORE | `results_lm_v3_1500` | 9 变体核心表，1500 步 × 3 seeds |
| P-ABL | `results_lm_v4_abl` | 去一法消融表——面板内参照臂（`full` / `csa_fixed` / `csa_dynamic` / `hybrid_fixed` / `hybrid_dynamic`）+ `csa_fix_*` / `csa_dyn_*` 去模块臂，以及 fuse/HCA/topk 臂，1500 步 × 3 seeds |
| P-SCALE | `results_lm_v5_scale` | d=384 / 8 层 / seq-1024 探针，5 变体 × 3 seeds × 4000 步 |
| P-LONG | `results_lm_v3_long` | hybrid_fixed / hybrid_dynamic / hybrid_csa_dyn，seeds 0-2 → 三 seed 长跑表 |

`python run_gapfill.py overfit` 是正式面板前的小批量过拟合自检（损失应能
拟合、流水线保持有限）；`topup` 只补齐缺失种子，不重跑已完成单元。

### 任务级并行（每个实验一张卡）

在多卡机器上，所有驱动都可以并行运行独立单元——每个 `(variant, seed)`
训练一个进程，**一卡一进程**。严格可选、默认关闭：不设置变量时行为与上面
的串行描述完全一致。

```bash
# 每张可见 GPU 一个 worker（推荐）：
CSA_PAR=auto nohup python -m explib.length full > run_v11.log 2>&1 &

# 只有 2 张卡、且只用这两张：
CSA_PAR=2 CSA_PAR_GPUS=0,3 python -m explib.corruption phase P3SL

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

B 档规模计划（入口 `python -m explib.m0`；原算力规划文档已退役）在
WikiText-103 机制层之上新增第二条语料路径与 S1–S4 规模阶梯：

```bash
python -m explib.m0 params   # 零 GPU：S1–S4 参数量核算（对照规划值）
python -m explib.m0 data     # 构建 / 校验 FineWeb-Edu 缓存（M0_TOKENS=… 可调 cap）
python -m explib.m0 smoke    # 端到端自检：B 档数据路径 + 三条注意力路径
python -m explib.m0 kernel   # G0 套件：kernel/eager 数值对照 + 基准
python -m explib.m0 analysis # 零 GPU：S1/S2 描述性配对统计
python -m explib.m0 full     # S1 主面板（5 臂 × 4 seeds；CSA_PAR=auto 可用）
M0_PANEL=S2 python -m explib.m0 full   # S2 主面板（5 臂 × 3 seeds × 12B；M2_CAP_HOURS 可覆盖）
python -m explib.m0 auto     # 无人值守：G0 门（fail-closed）→ 证书复核 → 面板链自动启动；
                           #   默认 S1，M0_AUTO_CHAIN=S1,S2 顺序串行（中间不关机）；
                           #   G0 通过后默认 CSA_KERNEL=1（显式设置优先）；门失败则
                           #   不启动面板、不 push、留诊断窗口后关机
```

- 语料：FineWeb-Edu（默认 `sample-10BT` 配置），自训 32k BPE；npy 缓存 +
  sha256 内容指纹，纪律与 `wt103_cache/` 完全一致；token id 以 2 字节落盘、
  int32 供给，取批后照旧回宽 int64，下游路径不变；
- 离线 / 内网机器：`FWEDU_LOCAL_JSONL` 指向本地 jsonl/txt 文档（纯可用性
  回退——不同语料内容必然产生不同指纹，不会与真缓存混淆）；
- `full` 受 `M0_CAP` 上限约束（状态文件 `run_time_state_m0.json`，
  `M0_CAP_HOURS` 可覆盖）：Triton 内核通过 G0 判据之前，eager 路径的预计
  耗时会被拒绝启动——这是规划文档自身的前提判断，不是缺陷。

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
python -m explib.m0 kernel                 # 任意设备：ref 后端语义验证
CSA_KERNEL=1 python -m explib.m0 kernel    # CUDA + triton：完整 G0 套件
CSA_KERNEL=1 python -m explib.m0 full      # 内核模式 S1 主面板（G0 通过后）
python -m explib.m0 auto                   # 无人值守：G0 套件 → 证书 → 面板链
                                         #   （「G0 之后 B 档是否继续」改由机器裁决：
                                         #   证书存在 = 双判据全过 = 面板启动）
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
python -m explib.length smoke
nohup python -m explib.length full > run_v11.log 2>&1 &
python -m explib.length report         # 从磁盘重建 REPORT_v11.md（零 GPU）
```

阶段：P4MT（RoPE 臂 seeds 0-5，完全沿用 P1L/P3MT 配方；续跑时跳过代码
指纹仍是当前版本的 checkpoint）、P4MP（同 seed 集的探针续跑，仅评估，
过期单元重新测量）、P4SS/P4SL（交叉面板 seeds 0-3，同一续跑规则）、
P4F（d=384 的 `csa_fixed`/`full`，seeds 0-3，同一 config，使面板配对到
n=4）。

### v10 机制 / 扩规模 / 统计套件

```bash
# v10：干扰注入机制探针 + 交叉扩规模 + topk 补 seed
python -m explib.corruption smoke
nohup python -m explib.corruption full > run_v10.log 2>&1 &
python -m explib.corruption analysis       # 零 GPU 统计 -> analysis_v10/
python -m explib.corruption report         # 从磁盘重建 REPORT_v10.md（零 GPU）
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
python -m explib.confirmation smoke           # 端到端自检（微型 ckpt + 冻结清单副本）
python -m explib.confirmation plan            # 零 GPU 预检：语义戳 + 实现绑定 + 准入 + 功效，不读 test；
                                              #   通过后自动留档审计快照 analysis_v12/plan_snapshot.json
python -m explib.confirmation dynamic-plan    # 零 GPU 预检动态/HCA 草稿：设计自洽，不是确认
python -m explib.confirmation power           # 零 GPU：用 P5C validation 差值做条件功效（不读 test）
python -m explib.confirmation auto            # 无人值守：与 full 同链，但显式去掉了「P5T 前人工停顿」——
                                              #   plan 门先跑（fail-closed），批准证据（清单哈希/
                                              #   代码戳/git 树/arms/seeds/主比较/α）自动落盘，
                                              #   人工核对变为事后异步，不再阻塞链的启动
python -m explib.confirmation phase P5C       # 按清单种子训练全部六个冻结臂（可续跑；主臂 0-5 由 P3MT/P4MT 覆盖）
python -m explib.confirmation phase P5MP      # 扩展探针（全部模式 + NIAH + 对照），种子取自清单
python -m explib.confirmation phase P5S2      # 等 token 大规模臂（8000 步）
python -m explib.confirmation phase P5S3      # 6ND 参数量代理预算大规模臂（参数量比例缩步）
python -m explib.confirmation phase P5T       # 冻结后的最终测试（读 wiki.test.raw，需冻结清单）
python -m explib.confirmation analysis        # 零 GPU：共享 v10 统计 + 冻结确认性统计
python -m explib.confirmation report          # 从磁盘重建 REPORT_v12.md（零 GPU）
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
python audit_gates.py single_factor  # 定义点单因素门：每条声明对照的臂配置 diff 恰等于声明的干预键集；开放块矩阵只沿声明轴变化；冻结清单的 mechanism_claim_allowed 标志与实际 cfg diff 对账；并核验 learned/frozen 边界门、sink、content、fusion 的容量/目标匹配对照阶梯
python audit_gates.py sesoi       # 设计期门：每个消融组件都有合法已声明 SESOI，且声明对照完整排程
python audit_gates.py unit_influence  # 结果分析门：逐单元影响块必须复现主终点 pooled Δ；标记单单元主导/陈旧产物
python audit_gates.py looks       # look-history 门：复核逐 look p 与精确 max-over-looks 序列 p；标记「补种子后才显著」的对照
python audit_gates.py experiment_ledger  # 重放 experiments.jsonl 的哈希链与尾锚点；编辑/删除/重排/截断事件判 fail
python audit_gates.py run_allocation  # 运行分配门：计划时间戳须早于首条 cell；臂×设备完全共线或计划覆盖不全判 fail
python audit_gates.py report      # 运行本地可核验的全部审计 -> audit_report.json
python audit_gates.py selection   # 选择召回契约：缺产物时拒绝不支持的选择机制主张
python audit_gates.py efficiency_pairs  # 延迟主张必须同 shape/device/KV
python audit_gates.py release     # 证据发布就绪：无完整记录 => source-only
python archive_release.py plan v13 --dirs results_lm_test analysis_v12   # 证据归档 dry-run
```

### v8 / v9 收尾套件

```bash
# v8：规模面板补齐 4 seeds + RoPE 20k 长跑
python -m explib.scale smoke
python -m explib.scale full            # 或：python -m explib.scale phase P1S / P2R
python -m explib.scale analysis        # 零 GPU 统计 -> analysis_v8/

# v9：补全 topk 扫描（bs=1）+ RoPE-20k 第三 seed + 报告
python -m explib.resume smoke
nohup python -m explib.resume full > run_v9.log 2>&1 &
python -m explib.resume report          # 从磁盘重建 REPORT_v9.md（零 GPU）
```

设计依据与这些阶段依赖的配对/可测量规则见 `docs/design_notes.md`。

### v7 回应 reviewer 套件

v7 套件（设计依据与阶段/子命令布局见 `docs/design_notes.md`）运行
reviewer 关切的实验并重建报告：

```bash
# 零 GPU：解析式 FLOPs/KV-cache + 成对符号翻转统计 + 报告重建
python -m explib.rope analysis
# 逐阶段运行（可续跑），或一次跑完整链：
python -m explib.rope phase P0E   # 40k 步长跑
python -m explib.rope phase P2S   # 规模重播种至 4 seeds
python -m explib.rope phase P1L   # 长度外推
python -m explib.rope phase P1T   # seq-2048 topk 扫描
python build_report.py            # 从产物重建 REPORT_v7.md（零 GPU）
```

所有驱动都按配置的运行时上限计量自己的加速器时间：超过上限的运行拒绝
启动，运行中途到达上限则被硬截断。多卡机器上 `CSA_PAR=auto` 以一卡一
实验并行运行相同阶段（上限此时按 GPU-秒总账计量，即 N 路并行按 N 倍
墙钟速度消耗）——被截断的单元不会留下不完整测量，会在下一轮重试。全部
阶段完成后脚本自动 push 并关机（取消关机：`pkill -f v6autoshutdown`；
禁止关机：`V6_NO_SHUTDOWN=1`；禁止 push：`V6_NO_PUSH=1`）。

> 量值一律以 `build_report.py` / `fuse_analysis.py` 与各驱动 `report`
> 构建器从 `results_*/`、`analysis_*/` 落盘产物现算出的表格为准，仓库正文
> 不写死任何量值。

### 常见注意事项

- **在仓库根目录运行。** 驱动按检出路径解析 `results_*/`、`analysis_*/`
  与冻结清单。
- **先看 `--help` 再选模式。** `python -m explib.<driver> --help` 打印驱动
  说明，模式列表见[研究一览](#研究一览)。注意
  `python -m explib.<driver> smoke --help` 不是帮助命令——位置式驱动会忽略
  第二个 token，直接开始 smoke 运行。
- **test split 是一次性的。** 只有 P5T 读 `wiki.test.raw`；不要在其外围
  再写读 test 的脚本。
- **陈旧缓存不会静默重建。** 更换 train 语料后先删除 `wt103_cache/` 再
  重建，然后才比较两次运行。
- **报告本身不等于证据。** 引用数字前先发布并校验归档
  （`archive_release.py`）。

## 测量护栏

分析路径采用 fail-closed：缺失、陈旧或不匹配的来源信息一律报为
`not_verifiable_locally` / `provenance_incomplete`，绝不当作通过。主要契约：

- **报告措辞**统一走 `exp_lib.reportable_claim`（n≥6、原始 p、Holm p、
  `|mean|>std`、族级分辨率下限）。`attach_holm` 同时给出族级最小可达 p，
  过大的比较族会显示为分辨率上限，而不是零结果。
- **配对必须匹配**数据切片、配方、代码戳与初始化（`init_core_fp`）；
  `pair_reason` 拒绝摘要不一致的配对。若实测计算量方向偏袒声称的赢家，
  该对照标为 `system_level_only`，不进入 `verified`。
- **交叉点标度**要求至少 4 个去重宽度、无删失子集、训练量差不超过 25%，
  并报告斜率 CI/p/R²/LOO；否则拒绝并说明原因，主张级别固定为描述性。
- **消融**必须声明 SESOI（探索性面板默认相对 PPL 1%），并用配对 TOST
  加族校正等价 p 判定。没有 TOST 时，不显著的结果不会被改写成「可删」；
  显著但小于 SESOI 的结果是 `sub_sesoi`，不是 `verified`。
- **声明的消融必须是真实干预。** 仅有配置差不够：
  `audit_gates.py manipulation` 用真实工厂建两臂，要求初始化前向改变
  （`forward` 模式）或可训练参数/梯度/目标路径改变（`training` 模式）。
  惰性对照会 fail-closed，因此后续 PPL 零差不能被写成「模块不必要」。
- **单元影响与重复 look** 是显式读数。pooled ΔPPL 带精确的
  leave-one-unit-out 诊断；新增种子集合按 `look_history.json` 中的精确
  max-over-looks 符号翻转 p 判定。两者都不改变冻结的 p/Holm/`confirmed`
  判定。
- **设计与发布门**可执行：`audit_gates.py` 覆盖泄漏、因果性、单因素对照、
  SESOI 排程、单元影响、look 历史、运行分配与证据就绪；
  `archive_release.py verify` 可独立校验已发布归档。当 `RELEASES.md`
  没有完整记录时，仓库保持 source-only。

## 延伸阅读

- `docs/design_notes.md`——完整设计依据与测量契约（通常可从 §25、
  §56–§58、§84 读起）。
- `REVIEW_4_RESPONSE.md`——逐条评审处置。
- `DECISION_LOG.md`——跑过什么、决定了什么的时间线。
- `RELEASES.md`——证据发布索引（当前为 source-only）。
