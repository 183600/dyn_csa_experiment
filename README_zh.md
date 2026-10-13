# Dynamic-Block CSA + HCA — WikiText-103 对照实验

在小型从零训练语言模型（BPE-8k，WikiText-103）中，对**压缩块稀疏注意力**
（CSA / HCA，固定分块与动态分块）与稠密注意力进行受控对比研究。本仓库包含
实验代码以及在自己机器上复现结果所需的全部内容。

**English** | [中文说明](README.md)

## 仓库结构

仓库只保留**源码、驱动脚本与设计文档**。所有实验结果目录（`results_*/`）、
零 GPU 分析目录（`analysis_*/`）、逐版报告（`REPORT*.md`）与运行状态文件
都**不入库**——它们由驱动脚本在跑实验的机器上重新生成，需要时本地复现即可
（见下文命令）。`.gitignore` 已把这几类路径挡在仓库外（`docs/` 只保留
`design_notes.md`）。

| 路径 | 说明 |
|---|---|
| `exp_lib.py` | **核心库，也是唯一的代码来源（source of truth）**：模型、CSA/HCA 注意力、池化、indexer、分段器、训练循环、聚合与统计 |
| `run_gapfill.py` | **v6 驱动**：最小可发表补充实验，支持断点续跑，受运行时上限约束，每阶段完成后自动 push，结束后自动关机 |
| `par.py` | **任务级并行执行器（可选）**：每个独立实验占一张 GPU 卡——见下文"任务级并行" |
| `v7_supp.py` | **v7 驱动**：回应 reviewer 的补充实验套件——P0R RoPE+QK-norm、P0W 稠密预热、P0E 40k 步长跑、P2S 规模重播种、P1L 长度外推、P1T topk 扫描 |
| `v8_supp.py` | **v8 驱动**：P1S（规模面板补齐至 4 seeds）与 P2R（RoPE+QK-norm 20k 步长跑） |
| `v9_supp.py` | **v9 驱动**：P2T（seq-2048 的 topk 扫描，`batch_size=1`）与 P2S3（RoPE 20k 长跑的第三个 seed） |
| `v10_supp.py` | **v10 驱动**：P3M 长度外推机制 + 干扰项注入探针、P3S 交叉面板扩规模、P3T topk 扫描补至 4 seeds |
| `v11_supp.py` | **v11 驱动**：统计收尾——干扰探针扩到 n=6（P4MT/P4MP）、交叉面板扩到 n=4（P4S）、d=384 配对臂扩到 n=4（P4F） |
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
# 0) 环境 — Python 3.12, torch 2.8.0+cu128，另需：
pip install datasets tokenizers matplotlib
# （代码在 import 时也会自行安装；exp_lib.py 内把 HF_ENDPOINT 指向
#  hf-mirror.com——中国大陆以外请自行修改）
#
# 如果 Hub 不可达，`load_wikitext` 会回退到本地原始文本：
# 将 `WT103_RAW_TXT` 指向包含 `wiki.train.raw` / `wiki.valid.raw` 的目录
# （搜索路径还包括仓库旁的 ./wt103_raw 和 /root/wt103_raw）。
# 只读这两个文件，对其分词可逐 token 复现 `wt103_cache/*.npy` 缓存，
# 因此该回退是纯可用性路径，不会改变任何测量结果。
# 若 `wt103_cache/` 已有缓存，语料直接由缓存提供，完全无需下载。

# 1) 端到端校验整个流水线（smoke，不 push）：
python run_gapfill.py smoke

# 2) 正式补充实验（受运行时上限约束）：
nohup python run_gapfill.py full > run_v6.log 2>&1 &
tail -f run_v6.log
```

`run_gapfill.py full` 按顺序运行（每阶段可断点续跑；重跑同一命令会跳过
已完成的 `variant::seed` 组合）：

| 阶段 | 输出目录 | 内容 |
|---|---|---|
| P-LONG | `results_lm_v3_long` | hybrid_fixed / hybrid_dynamic / hybrid_csa_dyn，seed 2 → 三 seed 长跑表 |
| P-CORE | `results_lm_v3_1500` | 9 变体核心表，1500 步 × 3 seeds |
| P-SCALE | `results_lm_v5_scale` | d=384 / 8 层 / seq-1024 探针，5 变体 × 2 seeds × 4000 步 |

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

阶段（均可续跑）：P3MT（重训两个 RoPE 臂，P1L 配方）、P3MP（仅评估的
远距离干扰注入：成对污染上下文、干净的 512-token 目标，对比
learned-selection / dense / random-indexer / see-everything 各臂）、
P3T（topk 扫描 seeds 2/3 → n=4）、P3SS/P3SL（d=128/4L 8000 步与
d=512/10L 4000 步交叉面板）。成对污染协议与预注册交叉判据的设计依据见
`docs/design_notes.md`。

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
