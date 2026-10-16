# 决策日志（Decision Log）

本文件记录**为什么在某个时间点跑/不跑某个实验**——评审指出「多轮追加实验
若在看到结果后追加种子、挑选比较或调整假设，再把普通 p 值当作预先固定
实验的证据，就会有选择偏差；当前没有公开结果和完整决策记录」。本日志
就是对这一点的制度化回应：每个版本追加实验的动机、看到了什么、据此决定
了什么，全部按时间顺序落档；v12 起的确认性评估以冻结清单
（`confirmation_frozen.json`）预注册，任何「看了结果再改计划」的操作必须
以新条目显式记录（并使该轮降级回探索性证据）。

**规则**：
1. 探索性面板（v6→v11 及 v10/v12 扩展）的 p 值一律标注「探索性」，
   Holm 校正后的 p 也只用于防止过度宣称，不作为确认。
2. 确认性证据只能来自 `confirmation_frozen.json` 冻结后的 test split
   一次性评估（v12 P5T）。
3. 在 test split 评估之后再追加冻结比较/种子 = 选择偏差，禁止；若确需
   新的确认，必须新开冻结清单并在本日志记录决策。

---

## 2024-11 前后 — v6 gapfill（基线面板）
- **决策**：跑 P-LONG（20k 步长跑）、P-CORE（9 变体 × 1500 步 × 5 种子）、
  P-SCALE（d=384/8L）。动机：建立 CSA/HCA vs dense 的基线差距与
  参数匹配对照（`full_matched`/`full_cos`/`full_sw128_matched`/sink 等）。
- **看到**：长跑与大方向见 REPORT_v6（本地重建）。此轮全部为初始设计，
  属于探索性证据。

## v7 — 评审响应 #1（RoPE / warmup / 长外推 / 规模复播 / topk / 40k 长跑）
- **决策**：P0R 把 RoPE+QK-norm 引入两臂（`*_rope` 变体）；P0W dense
  warmup；P1L 长度外推（512 训练 → 512..4096 评测，3000 步、3 种子）；
  P2S 规模面板补种；P1T seq-2048 topk 扫描；P0E 40k 步长跑。
- **动机**：外部评审质疑 abs-PE 位置截断与 warmup 协议。
- **看到**：P1L 出现唯一 sparse 正面信号（CSA+RoPE 长外推优于 dense+RoPE），
  n=3，p 值分辨率下限 0.250 —— 明确标注为附带观察。
- **追加决策**：不立即加种子（预算优先给 v8 的统计收官），保留为假设。

## v8 — 统计收官 #1
- **决策**：P1S 规模面板补到 4 种子；P2R RoPE 20k 长跑（2 种子起步）。
- **动机**：3-4 种子的精确符号翻转检验 p 值下限只有 0.25/0.125。

## v9 — 统计收官 #2
- **决策**：P2T 完整 5 点 topk 扫描（bs=1，新目录自洽面板）；P2R 第三
  种子。
- **动机**：bs=3 的 topk 扫描与 bs=1 的 m=1 基线不可比，重建面板。

## v10 — 机制升级 + 规模读数
- **决策**：P3MT 重训两 RoPE 臂并持久化 ckpt；P3MP 远距干扰注入探针
  （4 臂，ρ 网格，同腐坏严格配对）；P3S 新增 d=128/4L 与 d=512/10L 交叉
  定位面板；P3T topk 补到 n=4。
- **动机**：把 P1L 的附带观察升级为机制性主结果；4 点规模交叉读数。
- **已知混杂（本轮评审指出，已在 v10 修订中处理）**：
  1. P3SS（49.15M token）与 P3SL（24.58M token）训练量不对齐 →
     跨规模终点差异不能归因于规模（v10 修订：等 token 切片 + P3SL2/P3SL3）。
  2. P3MP 的 randidx/allblocks 是测试时干预 → 只能回答依赖性
     （v10 修订：P3MTC 同预算从头训练对照）。
  3. uniform 随机 token 干扰 ≠ 自然干扰，且未区分「滤噪」与「少用远端」
     （v10 修订：natural/mask/delete 模式 + ΔNLL + NIAH-natural）。

## v11 — 统计收官 #3
- **决策**：P4MT/P4MP 把探针相关面板补到 n=6（p 值下限 0.031，唯一能
  跨过 0.05 的种子数）；P4S/P4F 交叉与 d=384 面板补到 n=4。
- **动机**：使主要对照至少在分辨率上具备 0.05 的可能。

## 2026-10-16 — v10 修订 + v12（本轮：评审 5 点的修复与预注册）
- **决策（代码）**：
  - `v7_supp.py`：新增 `csa_fixed_rope_randidx` / `csa_fixed_rope_allblocks`
    两个与 `csa_fixed_rope` 逐字段同构的从头训练变体；
    `exact_sign_permutation` 增补逐种子 Δ、d_z、p 值下限（纯增量字段）。
  - `exp_lib.py`：新增 `load_wikitest`（held-out test split，冻结后专用）、
    `holm_adjust`、`signflip_p_floor`。**未改任何数值路径，`CODE_SEMANTICS`
    保持 v11.125**（既有面板/ckpt 全部续跑有效）。
  - `v10_supp.py`：P3MTC 阶段（同预算对照）；干扰探针扩展
    uniform/natural/mask/delete 四模式（uniform 路径与历史逐字节一致，
    历史 cell 键不带 mode 后缀，向后兼容）；NIAH-natural 远端必要信息
    任务；P3SL2/P3SL3 等 token/等计算预算规模臂 + 等 token 切片 + 收敛
    （平台期）诊断；分析按 mode 分组、输出 ΔNLL、Holm（显式标注探索性）。
  - `v11_supp.py`：P4MT/P4MP 显式钉死原始两臂与 uniform 协议（v11 语义
    不随 v10 扩展漂移）；分组键带 mode。
  - `v12_supp.py`（新）：P5C（对照补到 n=6）、P5MP（扩展探针 n=6）、
    P5S2/P5S3（规模对齐臂）、P5T（冻结后的 test split 一次性确认）。
  - `confirmation_frozen.json`（新）：预注册主比较 4 项、终点 L4096、
    种子 0-5、精确符号翻转 + Holm、方向一致性决策规则；代码戳 v11.125。
- **决策（统计纪律）**：v6→v11 一切 p 值标注探索性；确认性证据只出自
  P5T；test split 评估后禁止追加冻结比较/种子。
- **未做的事（显式）**：没有回填/修改任何历史数值；没有把任何新对照的
  预期方向写进冻结清单以外的文件；没有让 P5T 在 manifest 缺失或代码戳
  不匹配时静默放行（两者都会 hard refuse）。

## 2026-10-16 — 确认协议的机器可读方向与族固定（P5T 执行前）
- **背景**：冻结清单的决策规则要求「p_holm < 0.05 **且** 方向与 validation
  估计一致」，但清单里只有自然语言 `hypothesis_direction`，分析代码无法核对
  方向；同时 Holm 族大小此前由「恰好有有限 p 值的比较」派生，缺一个臂会
  悄悄缩小族、放宽校正。
- **决策（清单）**：为每个主比较补 `expected_sign`（`negative` / `positive`
  / `null` 表示非方向性机制读数），与既有 `hypothesis_direction` 文字一致；
  这是**在任何 test split 评估之前**对预注册的可核对化，不改变任何比较的
  臂、终点或假设方向。
- **决策（代码）**：`v12_supp.py` 的确认统计改为——Holm 在**清单声明的**族
  （主比较条数，缺失臂以 p=1.0 计入，族不缩水）上校正；只有「拒绝零假设
  且方向符合 `expected_sign`」才记为确认；配对要求同一种子两臂记录的
  `data_fp` 一致（否则该种子不进配对），杜绝跨 test-slice 配对。
- **诊断**：若 n 个种子与 m 元族下最小可达 Holm p > 0.05，运行日志与报告
  显式提示「该方案在构造上无法确认」——这是保守预注册的性质，不是数据结论。
- **未做的事（显式）**：没有改动任何主比较的臂/终点/假设方向；没有新增或
  删除种子；没有回填历史数值。

## 待办（按优先级，供 CI 循环调度）
1. `python v12_supp.py phase P5C`（对照 n=6；0-2 若 P3MTC 已跑会自动跳过）
2. `python v12_supp.py phase P5MP`（扩展探针 n=6）
3. `python v12_supp.py phase P5S2` / `P5S3`（规模对齐臂，各 2 种子起步）
4. 确保 v11 P4MT 已把两主臂补到 6 种子（否则先跑 `python v11_supp.py phase P4MT`）
5. `python v12_supp.py phase P5T`（冻结确认；前置：1、4）
6. `python v12_supp.py report`
