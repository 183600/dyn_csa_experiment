# 设计与实现要点

本文件记录代码库中若干**非显然的设计决定**以及支撑它们的判据。目的不是
记录改动历史，而是让后来者（和评审）能判断这些决定为什么成立、在什么
条件下会被破坏。

## 1. 因果性：块读取门限

判据是**前缀不变性** `f(x)[:T'] == f(x[:T'])`。这是因果语言模型的硬约束，
也是本仓库所有稀疏臂必须满足的性质。它同时是统计/探针评测可复现的
前提——若同一文本在不同窗口长度下得到不同段块，跨长度的比较就无意义。

### 1.1 唯一一致的判据是严格不等式

所有稀疏臂只读**已结束**的块摘要：

```
READABLE(b, pos)  ==  pos > last_tok[b]
```

必须**严格**。`pos >= last_tok[b]` 会让 token 读到它自己所属的块；尾块在
长度 T 的窗口里持有 `[start, T-1]`，其池化摘要因此是 T 的函数，窗口内每个
读到它的 token 的 logit 都会随总长度变化。

严格形式是唯一对固定与动态两种分段器同时成立的选择：

- 固定分段器：非尾块 `end = (b+1)*bs - 1` 与 T 无关；
- 动态分段器：切割表本身前缀不变（见 §2），非尾块端点亦与 T 无关；
- 尾块：`last_tok[tail] == T-1 >= pos` 对窗口内一切 `pos` 自动失败——
  **无需任何「该块是否已闭合」的特判**。

### 1.2 五个门限点必须同步

判据分散在多处（top-k 选择、gather、软注意力掩码、孪生实现等），任何一处
不一致都会造成臂间不对称。所有读取都经由同一入口 `block_readable`。

## 2. 动态分段器

### 2.1 切割表必须前缀不变

动态分段由每条序列的相似度信号导出切割位置。切割表必须满足：文本前缀
`x[:T']` 的切割表是 `x[:T]` 切割表的前缀（对 `T' <= T` 内的非尾块）。否则
同一 token 在不同评测长度下属于不同块，跨长度面板不可比。

### 2.2 低相似度对开边界的位置

边界在相似度**低**的两个相邻 token 之间张开，切割槽下标的归属（边界属于
左块还是右块）必须全仓库一致——`blocks_from_cuts` 是唯一出口。

## 3. 统计可比性

### 3.1 配对只允许同配置

两条记录相减（ΔPPL、配对 p 值）的前提是它们由**同一配置指纹**产生：
`run_cfg` 必须相等，且训练量字段（`steps` / `tokens_seen`）必须一致。指纹里
含代码语义戳（`exp_lib.CODE_SEMANTICS`），代码语义变化会使旧记录失效并
触发重算，而不是被静默复用。

### 3.2 「已测量」的定义

一条记录只有携带有限数值的 `ppl`（非 NaN、非缺省）才是测量。`error`
记录、`synthesized` 重构记录（由旧聚合表回填、无权重、无 `run_cfg`）、
同 (variant, protocol, seed) 的歧义重复，都**不计为**独立观测。

### 3.3 多协议面板按 (variant, protocol) 分组

同一目录下混入不同协议的记录（如不同 warmup 步数）时，按
`(variant, protocol)` 分组报告，绝不跨协议池化（跨协议池化会伪造额外的
种子数）。

## 4. 报告口径

- 全仓库的 `±` 一律是样本标准差（ddof=1）。
- 精确符号翻转 p 值在 n 个种子下的下限为 `2/2^n`；报告中的 n 一律从落盘
  产物实时计算，不写死。
- 每个配对比较同时报告：逐种子 Δ（`per_seed`）、效应量 d_z
  （`cohens_dz`，配对差均值/样本标准差）与该 n 的 p 值下限——「未显著」
  必须能区分是分辨率上限还是数据结论。
- 一个报告内的多个比较按 **Holm step-down**（`exp_lib.holm_adjust`）在
  显式声明的族内校正；族必须注明「预注册」（冻结清单）或「未预注册」
  （探索性，校正后的 p 只防过度宣称，不作确认）。
- 报告分布与配对差，而不是极值。
- 被截断的格（时长截断、未完成）不进入长上下文均值。
- 交叉判据的平滑在端点不得有偏：边缘用截断卷积核，避免端点处的系统性
  偏移。

## 5. 注意力行块化（`q_chunk`）

gather 实现的注意力按行分块计算。行块化必须与不分块的形式逐位一致
（同一表达式的分组归约），这是可断言、可检验的契约。sink softmax 的拆分
（sink 质量 = 1 − soft 列之和）必须发生在 softmax 归约**之后**，与未拆分
形式共用同一个 log-sum-exp，才不引入 ULP 级漂移。

逐位契约的适用范围是**前向值（以及 q 侧梯度）**：实测（torch 2.8 CPU，
q_chunk ∈ {128, 1024, 7}）输出与 `q.grad` 逐位相同。但 `_take_2d` 的反向
是向 k/v 栈的累加（gather 的 scatter 反向），其归约分组随分块边界变化，
同一输入在 CPU 上即可测得 ULP 级的 `k/v.grad` 漂移。因此 `q_chunk` 的
默认值是既有测量值的组成部分：调高默认值会让新跑的格与落盘旧记录不再
可比。同理，稀疏路径（池化、indexer、gather 注意力）不得跨 batch 行合并：
改变任何 matmul/归约的形状分组都会引入同类漂移，`_pre` 的 (B·T, d)
批量投影是仓库已选定并计入全部记录的既定形状。

## 6. 确定性约定

- 每个 (panel, seed, 评测步) 的随机源都由这些元数据派生，不读时钟。
- 浮点归约的分组长度与分组边界固定（不随设备/线程数漂移），同一输入
  在同一面板内走同一条归约顺序。
- 反向路径不出现整张 scatter-add（其累加顺序依赖线程数），需要 scatter
  的位置用固定顺序的形式表达。

## 6.1 传输层批量化不改值

以下实现只改**传输/存储形式**，不改任何到达算子的数值，因此不
触碰 §3.1 的可比性（`CODE_SEMANTICS` 不需要 bump）：

- `HybridAttention.forward`（及 v7 的 RoPE 镜像）把逐行的
  `blocks_from_cuts` 上传与 honoured 掩码上传合并为每层一次堆叠
  （`_hoist_blocks_from_cuts`）。块表仍由 `blocks_from_cuts` 唯一
  出口产生（以 CPU device 调用即跳过上传）；块 id 是整数、honoured
  掩码是精确的 0.0/1.0，堆叠后逐行切片与逐行上传逐位相同。门控硬决策
  以 `.cpu().numpy()` 零拷贝视图传给宿主侧分段器，布尔值与 `.tolist()`
  形式相同。
- `batch_iter` 的常驻语料（GPU 或 pinned host）以 int32 存储（token id
  < 2³¹，有 max 守卫），每次取批后**立即** `.long()` 还原，yield 出的
  batch 与 int64 形式逐位相同。是否驻留 GPU 的阈值判断仍以原 int64
  字节数为准（行为集合不变）。
- `_h2d_async`（及 `_hoist_blocks_from_cuts` 的批量上传）经 `_h2d_stage`
  的**常驻 pinned 暂存池**做 H2D：调用方把与原形式完全相同的值写进
  池化缓冲后入队复制；H2D 是逐位 memcpy，与每次现 pin 的形式到达设备的
  值完全一致。每个槽位由 CUDA event 守卫（上次入队未完成则不复用，
  全部忙时回落原 pin-per-call 形式），槽位内容在复用前必被整体覆写，
  因此不存在传输中的数据竞争；池不命中时的回落路径就是原实现。
- `batch_iter` 的窗口 gather 索引在 `n + seq_len + 1 < 2³¹`（有守卫，
  越界即回落 int64）时以 int32 上传/消费：每个下标都是精确整数，
  gather 返回值与 int64 索引形式逐位相同，gather 后照例 `.long()` 还原，
  yield 出的 batch 不变。
- `gathered_attention`（及 v7 的 `_rope_attn`）的 `topk_l` 在 topk 选择
  已是 int32 时跳过 int64 上行：高级索引与相等比较作用于同一组精确
  整数，返回值逐位相同。
- `HybridAttention.forward`（及 v7 的 RoPE 镜像）的门控信号 /
  `blocks_from_cuts` 提升路径对**一切** batch 大小启用（含 B=1，如
  seq-2048 topk 扫描）：B=1 时逐行内联形式与提升形式的差异仍仅在
  上传/下载的传输形状——门控布尔值、块 id、honoured 掩码逐一相同，
  §6.1 的原有论证对 B=1 逐字成立；`_gate_signal` 与 `_single` 内联
  门控的算子序列本就逐行相同。
- `CSA_CKPT`（默认关闭）的激活重算是「重算同一纯函数」的 transport
  形式：`mlp` 级包的 MLP 前向无 RNG、是输入与权重的纯函数，反向用
  保存的输入按同一确定性分解重算，前向值与梯度与直调形式逐位一致
  （同 ce_loss / gathered_attention / _pool_trainable 的既有
  checkpoint 论证）。`block` 级只覆盖无 gate_mean 侧信道的臂
  （chunking != 'cosine_learnable'）：gate_mean 经属性逃逸出块，
  checkpoint 初向前向的 no_grad 会剥掉它的图，故 cosine_learnable
  臂自动回退到 mlp 级（v9 _V9_CKPT 的同一限制）。块内既有的池化 /
  gather 注意力 / sink 分块 checkpoint 在反向重算时照旧包裹（嵌套
  非重入 checkpoint 是 torch 支持的形式），`_CKPT_NEST` 使内层 MLP
  不重复包裹。开关不改任何到达算子的值，与 CSA_PAR / CSA_CPU_THREADS
  同类，不进入 run_cfg 指纹（§3.1）。
- `load_fineweb_edu` 的语料 served 经 `_npy_serve_int32` 由 mmap 分块
  转换（而非整载入再 `.astype`）：磁盘上的 uint16/int32 与 served 的
  int32 是同一组精确整数，served 字节与直接 astype 形式逐一相同
  （`_ids_fp` 的内容指纹与 `batch_iter` 的对象身份键都不变），只压
  峰值宿主内存。

## 6.2 宿主侧算法的整数等价

`_segment`（动态分段器的扫描闭式）只做整数/布尔运算，其实现可在
「输出整数完全一致」的前提下自由重写（局部变量绑定、numpy 视图直读
等）。任何重写都以随机化对拍（含 n≤64 缓存键路径）验证 bids /
honoured / counts 三者与旧实现逐一相同。

同类的布尔/整数等价：

- 池化 gather 几何（`_pool_ordered` / `pool_blocks_single` 的 g/mask/b_off/
  _pos_a）是 block-ids 对象自身切分表（`_pool_cpu` 的 counts）与若干形状
  标量的精确整数/布尔函数，缓存于该对象上（`_pool_geom`，键含全部形状
  标量，上限 8 项）与 `_pool_parts` 的缓存同理按位安全。固定分段器在全
  面板内共享同一 block-ids 对象；动态分段器的逐行对象不命中，按原形式
  重建（值一致）。缓存张量只在 inference_mode(False) 下构建，下游只读
  （gather / 取反掩码视图），eval 期首填亦可服务训练。
- 门控与余弦切割的布尔表以 `.cpu().numpy()` 零拷贝视图（而非
  `.tolist()`）传给 `_segment` / `blocks_from_cuts`：同一组布尔值，
  分段输出的 bids / honoured / counts 逐一相同。
- `gathered_attention`（及 v7 的 `_rope_attn`）接受 `topk_distinct`
  关键字：调用方保证 topk 每行块 id 互不相同（HCA「读所有块」臂的
  arange 展开）时，`first_occurrence_mask` 恒为全 True，跳过它得到的
  keep 掩码与计算形式逐位相同。学习式 indexer 路径允许重复（无效槽
  以 pad 块填充），不得使用此关键字。

## 6.3 宿主侧纯函数的记忆化

以下三处是**纯宿主侧**的进程内记忆化，不改任何到达算子的数值、不改
RNG 流、不改返回内容，因此不触碰 §3.1 的可比性（`CODE_SEMANTICS`
不需要 bump）：

- `_ids_fp`：语料字节的 sha256 摘要。驱动每个 phase 都会对同一批
  数组重算一次（`run()` 的 run_cfg 指纹、`load_wikitext` 的缓存
  校验）。以（数据指针、shape、strides、dtype）
  为键并持有数组强引用（指针在条目存活期内不可能被复用）做记忆化；
  输入字节相同则摘要必然相同。约定：调用方不就地修改这些数组
  （全仓库无此用法）。
- `load_wikitext`：以（cache_dir、tag、四个缓存文件的 size+mtime）
  为键记忆返回元组。命中时返回对象保持身份——`batch_iter` 的
  `_PINNED_HOST_CACHE` 以对象身份为键，下游行为与未记忆化逐位一致。
  缓存文件被外部重建时 size/mtime 变化，记忆自动失效并重新校验。
- `variant_mlp_ratio`：matched variant 的 MLP 加宽比是其全部形状
  参数的纯函数（参数量只依赖形状；构造过程在 `fork_rng` 内进行，
  命中时跳过的一次性 RNG 抽取本来就会被丢弃，调用方 RNG 流不变）。
- `_finfo_min`：`torch.finfo(dtype).min` 是 dtype 的纯函数（每次调用
  返回同一常量），按 dtype 记忆化，不涉及任何张量值。

## 7. 落盘与续跑

- 所有 JSON 产物原子写（临时文件 + `os.replace`），读者永远不会看到
  半个文件。
- 续跑键为 `variant::seed`；指纹不匹配、无指纹、或不可计量的记录一律
  先丢弃再重算，失败的重算不得在原地留下旧代码测得的数字。
- 模块导入必须零副作用：不读写状态文件、不下载、不改随机种子。依赖产物的
  检查在产物缺席时必须可跳过而不是失败。

## 8. 仓库边界

仓库只承载**源码、驱动脚本与设计文档**：实验结果目录、分析目录、逐版
报告、运行状态文件都不入库（`.gitignore` 白名单保证，`docs/` 仅留本文档）。
实测数字一律由零 GPU 的报告构建器从落盘产物现算，不写死在任何仓库
文件中——写死的数字在仓库内没有可复核的来源，代码一改就会与报告静默
矛盾。

## 9. 结果聚合的容错边界

`params`（参数量）对一条测量记录是可选元数据：缺失时配对照常进行，
参数对齐的标签降级为「未知」而不是令整节报告失败。坏的一格不拖垮整份
报告，但必须在运行日志里被点名。

## 10. `bnd_rand` 的两种口径不得混池

随机切割基线有两种口径（固定块数 / 每序列独立切），统计含义不同，
任何聚合、配对、绘图都必须分开处理。

## 11. 窗口掩码的带宽包含对角线

窗口掩码的 live 区间是 `(i - w, i]`——对角线（token 读自己）包含在内，
带宽参数 `w` 数的是包含对角线的列数。

## 12. `by_len` 的键类型

JSON 往返会按名字把整数键写成字符串；`by_len`（及类似的分长度表）的键
因此 int 与 str 并存，所有读取必须两种都认。

## 13. 任务级并行（`par.py`，一张卡跑一个独立 run）

调度单位与开关：

- **cell 即 run**：`L.run` / `run_lenphase` / `run_probe` / `run_warmup`
  的最内层 `(variant, seed)`（warm 另含 `warm_steps`）就是一个独立 run，
  cell 之间无任何数据依赖，因此并行化以 cell 为粒度。每个 cell 在
  **子进程**里执行，父进程按槽位给子进程设
  `CUDA_VISIBLE_DEVICES=<卡号>`——子进程里 `exp_lib.DEVICE` 自然解析到
  那张卡，模型与训练代码零改动。`CSA_PAR=auto`（每张可见卡一个
  worker）或 `CSA_PAR=N`（N 个 worker）开启；`CSA_PAR_GPUS=0,2`
  限定卡池。**不设 `CSA_PAR` 时全部驱动保持原串行行为**（smoke 模式
  显式钉死 `par=False`，不受环境变量影响）。CPU 机器上 `CSA_PAR`
  会开 CPU worker（主要用于测试）。
- **数值等价**：每个 cell 在 `set_seed(seed)` 后自进程内训练，RNG 与
  进程外完全隔离；指纹（`run_cfg`/`_recipe`/`_probe_fingerprint`）只依赖
  落盘语料内容与确定性标签，与计算它的进程无关。因此并行与串行产出的
  每个数字逐位一致（已按 §13.1 的对照测试验证），`CODE_SEMANTICS`
  不因并行化 bump。

并发安全契约：

- **共享 JSON 一律锁内改**：所有 cell 对 `summary.json` /
  `distractor.json` 的写入走 `locked_json_update`（flock 互斥 +
  载入-修改-原子写回），读取走 `locked_json_read`。两个 cell 同时完成
  时谁也不会覆盖谁。`TimeGuard` 状态文件沿用自带的 flock。
- **预订-退款记账**：并行 cell 开训前把 `estimate_seconds` 的估计值
  `book()` 进共享状态（不动 run 计数、不动标定），训完 `record_run`
  记实际秒数后全额退款差额。收敛不变量：`booked_seconds` 恒等于实际
  秒数之和，而并发启动的子进程也能看到共享余量在收缩，从而继续尊重
  运行时上限。多卡并行时上限按 **GPU 秒** 计（4 卡并行 1 小时消耗
  4 GPU 小时）。
- **父进程独占**：聚合（`_finish_panel`/`aggregate.json`/CSV/图）、
  `git push`、自动关机、以及 dispatch 前的语料缓存预热只在父进程做；
  子进程只跑自己的 cell。续跑检查（含 stale 记录的丢弃）在 dispatch
  之前由父进程一次性完成，子进程不做二次丢弃。
- **失败语义与串行对齐**：cell 内训练异常 → 写 error record、phase
  继续（同串行）；子进程基础设施失败（rc≠0，如导入失败/summary 不可
  解析）→ 父进程抛异常中止该 phase（同串行异常），已完成 cell 保留，
  重跑续跑。父进程收到 Ctrl-C 会杀掉整组子进程；被杀的 cell 不留
  半条记录（记录只在训练完成后写入），与串行崩溃语义一致。
- 驱动级差异：`module` 字段告诉子进程导入哪个驱动模块以复现补丁链
  （如 v10 的 `P3T` 需要梯度检查点开关，job 里带 `ckpt_flag`）；v7 的
  NIAH phase 没有并行路径，`CSA_PAR` 开着时打印说明并串行回退。
- job/日志/结果文件落 `par_jobs/<phase>_<时间戳>/<序号>_<cell>/`
  （已 gitignore），父进程日志只报 start/done/心跳，失败时贴子日志尾部。

### 13.1 并行正确性的验证方式

对照测试（`scripts` 侧，不入库）：同一份小面板分别以 `par=False` 与
`CSA_PAR=2`（CPU worker）各跑一遍，逐字段比较 `ppl`、`losses`、
`ppl_history`、`by_len`、`ppl_mean`、warmup `ppl` 与聚合表，并断言
`TimeGuard` 的 run 计数不被预订膨胀、子进程日志数量与 cell 数一致、
全 skip 的重跑不 spawn 任何子进程。

## 14. 代码语义戳的唯一出口是 `code_stamp()`

`exp_lib.code_stamp()` 的 docstring 明确要求：一切「记录是否当前」的判定
（`result_is_current` 的 code 实参、`_cs...` 后缀匹配、checkpoint 的
`code` 字段）都必须以它为准。eager 模式下它逐字节等于 `CODE_SEMANTICS`；
内核模式（`CSA_KERNEL`）下追加 `-tk1:<backend>` 后缀。因此 v7 驱动链的
`CKPT_CODE`（warmup / NIAH / lenphase 的记录戳与 checkpoint 戳）以及
v8/v10/v11/fuse_analysis 的 `_cs` 后缀与 `_code` 比较一律经
`code_stamp()` 解析——裸比 `CODE_SEMANTICS` 会把内核测得的记录错盖
eager 戳（与真正的 eager 记录错误配对，§3.1 的污染方向），或把内核记录
误判为陈旧使 resume 静默失效、面板被标 `_stale_cfg`。零 GPU 的
`build_report.py` 不 import exp_lib、只服务 eager 的 v7 面板，仍按源码正
则读取 `CODE_SEMANTICS`，属同一值。

同一口径的另一处：`csa_kernels.sparse_layer_forward` 与 eager `_single`
同口径收集 `attn._stats` / `attn._cuts`（同一组块 id，bincount 与切点逐
一相同，仅 eval 期触发），内核模式的记录因此携带与 eager 完全一致的块
长分布与边界对齐诊断；缺了它，内核面板的机制性指标会静默缺席。

## 15. 干扰探针的腐坏模式与 ΔNLL 口径（v10 修订 + 评审二轮·意见六）

远距干扰注入探针（P3MP/P4MP/P5MP）的腐坏模式，全部只依赖
(eval_len, 行号, ρ, mode, span_len)，与臂/模型种子无关——同一 (行, ρ,
mode) 的腐坏在臂间**逐字节相同**，配对成立：

- `uniform`（历史协议）：远端 ρ 比例 token 换成均匀随机 token（同 token
  重采样）。历史 cell 键**不带** mode 后缀、记录不带 `mode` 字段，读到
  时按 uniform 解释——既有面板的 resume 与统计完全不变。
- `natural`（**概念更名：realistic-token**）：替换 token 取自同一验证面板
  **另一条真实序列**的同位置。它保持 unigram/词频统计，但**逐位置拼接
  会破坏句法、语义与篇章结构**——评审二轮·意见六判定：token 来源真实
  ≠ 保持自然文本分布。因此它只支持「**对真实语料 token 替换的鲁棒性**」
  主张，不支持「对自然长文档干扰的鲁棒性」；报告标签一律写
  realistic-token。与原 token 撞车不重采样（撞车本来就是自然统计
  的一部分）。
- `span`（评审二轮新增）：远端 k 个 token 换成供体行的**连续**片段
  （段长 PROBE_SPAN_LEN=32）：段内保持局部句法，段间与跨位置连贯性
  被破坏——介于 realistic-token 与整段替换之间的中间档。
- `doc`（新增，ρ 恒 1.0）：远端整段换成供体行的连续等长切片——「完整
  无关文档上下文」的长度/位置受控实现（协议不允许改变 eval_len，故以
  等长替换实现「插入无关文档」的科学问题）。
- `topical`（新增）：供体行 = 与本行远端 unigram Jaccard 相似度最高的
  其他行（确定性、无 RNG），再做同位置替换——「主题相近但答案无关」
  的最廉价近似。
- `mask`：远端整体替换为 `<pad>`（保留长度/位置，抹掉信息）。ρ 恒 1.0。
  `<pad>` 对远端上下文是**分布外输入**——本模式只作互补探针，与
  `delete` 成对解读。
- `delete`：输入只剩目标区（RoPE 相对位置自洽）。ρ 恒 1.0。同时改变
  长度与上下文结构——与 mask 一样是互补探针，不能单独识别纯信息效应。

**为什么必须有 mask/delete**：只看污染鲁棒性无法区分「滤噪」与「本来就
少用远端」（关键替代解释）。mask/delete 给出干净远端上下文的收益幅度
（NLL_干净 − NLL_屏蔽），是解读一切鲁棒性数字的参照系。**为什么必须有
NIAH-natural**：检索任务（key→value 耦合埋在远端自然文本、查询在目标区、
干扰只加在草料上）直接测「长距离信息利用」，与鲁棒性互补。

**ΔNLL 口径**：ΔNLL = NLL_污染 − NLL_干净（nats/目标 token，逐种子配
对；干净基线 = 同臂同参数化的 uniform ρ=0 cell）。报告必须同时给 ΔNLL
与干净 NLL——某臂 ΔNLL 小 + 干净 NLL 差 = 「少用远端」，不是「更抗噪」。

**向后兼容约定**：`_fp_norm` 归一化时剥掉 `mode` 字段（指纹本体不含
mode），但分析分组的 key **显式**带 mode——uniform 与 natural 在同 ρ
下绝不并池。非 uniform 记录的 `probe_params` 内嵌 `mode`，resume 检查
按「同 mode 才算 current」。新增模式（span/doc/topical）走同一套键与
resume 规则，历史面板不受影响。

## 16. 规模面板的预算对齐（v10 修订 + 评审二轮·意见七）

P3SS（d=128/4L，8000 步）与 P3SL（d=512/10L，4000 步）的累计训练
token 相差一倍：跨规模**终点**差异混杂了训练进度，不能单独归因于规模。
规模面板因此有三个口径，缺一不可：

1. **等 token 切片**（零 GPU，`_equal_token_slices`）：同一批曲线在相同
   累计 token 处逐种子配对（gap_L − gap_S），历史数据立即可算；
2. **等 token 臂**（`P3SL_TOK`，8000 步）：大模型训练到与小模型相同的
   token；
3. **6ND 参数量代理预算臂**（`P3SL_FLOP`）：步数 = small_steps ×
   N_small/N_large（6·N·D **代理**；参数量在 CPU 上确定性计数，运行期
   推导并打印审计轨迹；下限 250 步保住 warmup 后的学习率调度）。

**评审二轮·意见七的口径约束（必须遵守）**：
- 等参数量代理**不等于**实测等 FLOPs——稀疏选择、池化、注意力与激活
  重算的成本不与参数量成比例；触发步数下限时还会偏离原目标预算。报告
  一律称「6ND 参数量代理预算臂」，并**并列**给出：实际 token 数、参数
  量、估算 FLOPs（`audit_gates.train_flops_estimate`）与 TimeGuard 实测
  耗时。**等 token 与等算力是两个不同的问题，必须分别陈述**；不得把
  代理预算写成严格匹配。
- 「最后四分之一 eval 窗口改善 <1%」（`_panel_convergence` 的 plateau
  诊断）**不能证明收敛**——学习率衰减、平台期、评测噪声或优化停滞都
  可能满足该条件。它只允许「**预算内平台期后交叉**」
  （crossing-after-plateau-under-budget）的读数；「渐近优势反转」「收敛
  后更差」需要更长训练、合理学习率续训及重复验证。

## 17. 探索 / 确认分离与冻结清单（v12 + 评审二轮·意见一/二/三/四）

v6→v11 的全部面板（含多轮补种子、追加比较）都围绕同一个 validation
split 展开——按选择偏差的构造方式，这属于**探索性**证据；其 p 值（含
Holm 校正后）不得写成确认。确认性证据只能来自：

- **冻结清单**（`confirmation_frozen.json`，入库、带日期与版本）：
  预注册主比较、臂、种子数、终点、检验与校正规则、决策规则；
- **held-out test split**（`exp_lib.load_wikitest` 读 wiki.test.raw）：
  此前任何 phase 都不读它（历史代码只读 train/valid）；P5T 是唯一读者；
- **一次性评估**（v12 P5T）：代码语义戳与冻结清单不一致时 hard refuse
  （重冻结必须显式改清单）；test split 评估之后**禁止**追加冻结比较或
  种子（那是选择偏差的定义），缺种子就如实报小 n。

**评审二轮后的三条硬约束（v12_supp 代码化，不再只是约定）**：

1. **功效门槛（意见一）**：加载清单即校验「该方案是否数学上可能确认」
   （min attainable Holm p = m·2/2^n 与 α 比较；`exp_lib.holm_min_
attainable_p`）。数学上不可能确认的清单被拒绝——这不是保守，是构造
   缺陷。v1 清单（6 种子 × 双侧 × Holm/4，最小可达 p=0.125）在此门槛
   下不合规；v2 已在**未读任何 test 数据**的前提下把种子修订为 12 并
   记录功效表（数学上可拒绝 ≠ 已足功效，清单里逐 q 报功效）。
2. **严格准入与绑定（意见二/三）**：确认统计入口只接纳通过「冻结种子
   白名单、唯一记录、非空 data_fp、单一冻结 test 切片、_code 与
   recipe 的机器校验（清单 binding.training_recipe_signature）、
   manifest_sha256 绑定」的记录；歧义重复与混切片直接拒绝。每条记录
   另存 ckpt_sha256。首次 test 访问写 `.freeze_lock.json`（清单哈希 +
   切片指纹），此后清单/切片变更即拒绝（V12_REFREEZE=1 显式改冻结，
   前轮降级为探索性并记入 DECISION_LOG）。清单修订后同目录旧结果
   自动作废（resume 的 stale 判定含 manifest 哈希）。
3. **范围声明（意见四）**：清单 `scope` 块明示——本确认只覆盖**固定
   分块 CSA 机制**的主张；动态分块优于固定分块、动态边界的语义价值、
   HCA 的优劣、动态分块的规模规律**不在本确认内**，需要各自的预注册
   冻结。冻结主表的正/负结果都不得外溢为这些主张的证据。

`DECISION_LOG.md`（入库）按时间顺序记录每个版本为什么跑/不跑什么、看
到什么、据此决定了什么——评审要求「完整决策记录」的落点（2026-10-17
  已恢复并追加评审二轮条目；正式版本的逐种子结果/配置/失败记录/曲线/
报告/环境/权重哈希另由 `archive_release.py` 打包为带 sha256 清单的
不可变归档——仓库边界 §8 仍只承载源码，归档发布到只读位置）。

test split 的缓存纪律与主语料相同（npy + sha256 + 原子写），另有两点：
分词器用与 `load_wikitext` **完全相同的拟合过程**重建（同一份 train 前
50k 行，确定性 BPE），并与任何在场的 train 侧缓存做词表一致性硬校验
（不一致即 refuse；`audit_gates.py leakage` 独立复核落盘副本与 train/
test 文档重叠基线）；若更换 train 语料，请删除整个 `wt103_cache/` 重建
（README 同款约定）——test 缓存不会跟随静默重建，这是显式边界。

## 18. 测试时干预 ≠ 从头训练对照（v10 修订 + 评审二轮·意见五）

`run_probe_cell` 的 `randidx` / `allblocks` 臂是**测试时干预**：它们回答
「已训练模型是否依赖原选择器」，不回答「学习式选择是否优于训练充分的
随机选择/全块读取」——干预改变了模型训练时适应的输入与注意力分布，
性能下降可能来自失配。因此机制面板必须同时有：

- 同预算**从头训练**对照（`csa_fixed_rope_randidx` /
  `csa_fixed_rope_allblocks`，与 `csa_fixed_rope` 逐字段同构，仅选择方
  式不同；P3MTC/P5C 训练）；
- **干预与重新训练的性能差**（评审二轮更名：原「mismatch 分解」→
  `retrain-gap`）= 测试时干预 − 同预算从头对照（同 L、同 ρ）。**对
  越低越好的损失，正值说明干预比重新训练更差**；该差距本身**可能包含**
  训练—测试失配的代价，也混合了优化适应与参数差异——它是原始性能
  差读数，**不是**已完成的因果失配分解。评审二轮·意见五的原表述
  「正值是不能归因于失配的性能损失部分」方向错误，已从代码注释与
  报告口径中删除；简单相减不得解释为因果分解。
- `csa_fixed_rope_allblocks` 同时充当**逐项对齐的 dense**（与
  `full_rope` 只剩 KV 压缩投影与分块读取方式之差）——评审点 5 的
  「在尽可能相同的投影、归一化、位置编码下分别改变分块方式与选择方式」
  的落点。`arch-aligned` 对比（full_rope − allblocks 孪生）因此是比
  dense−csa 更干净的分块/压缩效应读数。

## 19. 对比实验 / 消融 / 统计可靠性的自动审计

`exp_lib` 在每个面板聚合时（`_dump_aggregate`）对 `summary.json` 跑一次
零 GPU 的 `methodology_audit`，结果同时打印并写入
`<outdir>/methodology_audit.json`；面板开训前 `run()` 还会用
`methodology_seed_plan` 打印该面板的种子预算。它把评审要求的三件事变成
可复核的读数：

- **受控变量**：每条对比先用 `pair_reason` 做严格同 `run_cfg` 配对——
  学习率、权重衰减、`delta_lr_mult`、训练步数与 token 数、warmup、comp
  正则、模型形状、数据指纹与确定性档全部一致，只有层架构（`variant`）
  不同；`run_cfg_knobs` 只是把该串解码成可读字段便于展示，**真正的门槛仍是
  整串相等**。`run_cfg` 不承载的优化器/调度设置由独立的 `training_knobs()`
  描述符补足：优化器（AdamW/foreach）、betas/eps、LR 调度形状
  （warmup→cosine 的 floor/幅度）、梯度裁剪与数据增强一律提为模块常量
  （`OPT_ADAMW_*` / `GRAD_CLIP_NORM` / `LR_FLOOR_MULT` / `LR_COSINE_AMP`），
  随每条记录落盘；`train_variant` 实际用的就是这些常量，因此所有臂天然
  共享。`pair_reason` 对两条都带该字段的记录要求逐字段相等；**只有一条
  带该字段、另一条不带也判为 unverifiable**——否则新旧记录混用时，换了
  优化器的那条会静默通过，正是本门槛要防的假提升。两条都不带（纯历史
  记录）才保留旧的兼容路径。被拒的种子不计入 n，并在 `skipped_reasons`
  里点名。`BASELINE_DESIGN` 另把主方法与参数对齐稠密基线
  （`full_sw128_matched` / `full_matched`）的对比纳入同一套门槛，并额外
  要求两者参数量相对差 ≤1%（`param_controlled`）；参数不匹配的基线主张
  不计为 verified。
- **消融完整性**：`ABLATION_DESIGN` 声明式地列出每个创新点的留一对照。
  去一法只在**固定分块**下做是不充分的（`csa_fix_*`），因此同一组去一法
  在**动态分块**（提出的主方法）下也有对应臂 `csa_dyn_randidx` /
  `csa_dyn_zerocont` / `csa_dyn_nosink`（与 `csa_dynamic` 逐字段同构，仅
  待验证模块不同）；压缩正则这一**损失项**也有 `csa_dyn_nocomp`
  （`comp_lambda_mult=0`，同一 `run_cfg` 的 `cl` 不变，配对门槛因此仍能
  通过），对应 `compression_regularizer` 去一法。去一法要成立，**「Full」
  与各「Full−X」臂必须落在同一面板里**——`pair_reason` 的配对与 `methodology_audit` 的审计都按单面板
  `summary.json` 进行，参照臂不在面板内时审计只能报「缺臂」。因此
  `RUN_ABL` 自含参照臂（`full` / `csa_fixed` / `csa_dynamic` /
  `hybrid_fixed` / `hybrid_dynamic`；`csa_dynamic` 与 `hybrid_csa_dyn`
  即各去一法的「Full」），并由 v6 驱动的 **P-ABL 阶段**实际排程
  （`run_gapfill.py full`，seeds 0-2 ≥ `min_n`；`RUN_ABL` 声明的 5-seed
  全集留给后续补种子 pass）。v6 smoke 同步覆盖消融臂的全部独有前向路径
  （随机 indexer / 零内容 / 无 sink / fuse / HCA / dense+sink），使 P-ABL
  不会在烧掉数小时后才暴露坏臂。
- **验证分级（不再把「有对照」写成「已证明」）**：`verified_components`
  只收**受控 + 参数受控（基线）+ 族内 Holm 校正后 p<α + 效应大于种子噪声**
  的创新点；仅达到 `min_n` 的对照列入 `measured_components`（描述性），
  单种子/欠功效列入 `underpowered_components`，完全没有受控对比列入
  `unverified_components`，缺哪个臂会被逐条 warn。审计为每个创新点导出
  `controlled_variables`（`run_cfg_equal` / `train_knobs_equal` /
  `param_matched`）与 `component_seed_requirements`。
- **统计可靠性**：`paired_effect_stats` 给逐种子 Δ 的 mean±std（ddof=1）、
  `cohens_dz`、精确符号翻转 p 与该 n 的 p 下限，并显式标出
  `underpowered`（n<`min_n`）、`resolution_limited`（精确检验在该 n 的
  最小可达 p 仍 > α，即**无论效应多大都不可能达到显著性**——这是分辨率
  上限、不是零结果）与 `within_seed_noise`（|mean| ≤ std，即 +0.2% 的
  均值配 0.5% 的方差这种噪声级「提升」）；`significant` 同时要求 p<α 且
  效应大于种子噪声，避免大 n 的 p 值把亚噪声效应包装成真实增益。
- **多重比较**：审计对整个声明族（`ABLATION_DESIGN`，以及面板里实际排了
  参数对齐稠密臂时的 `BASELINE_DESIGN`）做 **Holm step-down**，每个对比
  同时给出 `p_holm_global`。族按「本面板能测到的声明」划定：一个组件至少
  有一条完整对比（两臂都在场）才进入族；一旦进入，它的**全部**声明槽位都
  保留（缺失臂用 dummy p=1 占位），因此不完整的消融不能靠少报比较来「买」
  更低的 p。没有排基线臂的纯消融面板不把未测试的基线主张计入族。每个创新
  点另有族内 `p_holm_component` 供细读，但 `verified` 以全局保守口径为准。`methodology_seed_plan` 给出该族在 α 下**数学上可能拒绝**
  所需的最小 n（α=0.05、单比较 n≥6；m 元族约需 2ⁿ ≥ 2m/α），开训前就把
  n=3–5 的面板标为描述性，避免事后把分辨率上限读成零结果。
- **排程侧同一下限**：v6/v7 驱动的每个面板阶段至少排 `min_n` 个种子
  （P-CORE / P-ABL / P-SCALE / P-LONG seeds 0-2，P0W 每 warm 条件 3
  seeds），更细的面板由补种子阶段加到 n=4–6；v7 `train_warmup` 的优化器 /
  betas / eps / LR 调度 / 裁剪全部取自 `exp_lib` 模块常量并向记录写
  `train_knobs`，消除第二份会静默漂移的字面量。需要族级确认的面板应按
  `methodology_seed_plan` 的返回值排种子，而不是沿用「3 个种子即可」的
  旧口径。达到族级下限的实际路径是 `python run_gapfill.py topup`
  （`ABL_SEED_TARGET=<n>` 可覆盖默认值）：它沿用同一 `RUN_ABL` 配置与
  同一 `summary.json`，resume 跳过已完成单元，只补缺失的种子；参数对齐
  稠密基线所在的 P-CORE 可用 `CORE_SEED_TARGET=<n>` 一并补齐。

`exact_signflip_p` 对 n≤20 穷举 2ⁿ 符号翻转（确定性、与各驱动套件同一
检验），n>20 用确定性正态近似；`signflip_p_floor` 给出单比较的分辨率下限，
`holm_min_attainable_p` / `signflip_min_n_for_holm` 给出族级下限，使
「未显著」能区分是分辨率上限还是数据结论。

## 20. 统计检验的假设、边界与外部有效性（评审二轮·意见九/十一新增）

本节把精确符号翻转检验的**有效性前提**与两类抽样不确定性显式化——
「种子编号相同」本身不提供任何统计保证。

- **交换性假设（意见九 a）**：精确符号翻转检验的 p 值正确性依赖配对内
  交换性：同一种子下，两臂共享除臂本身之外的一切随机因素——同数据顺序
  （P3MT 协议：同种子同数据顺序）、同评测行、同腐坏网格（腐坏只依赖
  (eval_len, 行号, ρ, mode)，与臂无关）；初始化 RNG 按种子播种，臂间
  架构差异对 RNG 流的扰动**属于臂的一部分**（这是合理的：被检验的就是
  整个臂的差）。种子间方差主要反映训练随机性。若未来引入臂相关的数据
  顺序或采样，本假设即被破坏，检验必须重新论证或更换。
- **文档抽样不确定性（意见九 b）**：最终测试的干净 by-length 评估默认
  16 条行。种子层方差 ≠ 文档层方差：`eval_length_gen` 现在落盘
  `ppl_per_seq`（逐文档 PPL，池化 PPL 的归约不变），`finaltest_stats`
  输出 `doc_sampling`（between-doc 与 within-doc-across-seed 两层分解）。
  16 行**不足以覆盖语言数据总体分布**——面向总体的主张需要更大的行面板
  与文档×种子的分层分析；在此之前，主张范围限于「该冻结 test 切片」。
- **小种子面板一律描述性（意见九 c）**：双侧精确符号翻转的 p 下限 =
  2/2^n：n=4 → 0.125，n=3 → 0.25，n=2 → 0.5。S1×4 / S2×3 / hero×2
  （compute_plan §5 阶梯）**全部不可能达到 0.05 显著性**，无论效应多大
  ——这些面板只能作为描述性证据与方向性读数；确认性结论必须等到种子数
  与族大小满足功效门槛（`exp_lib.holm_min_attainable_p` /
  `signflip_power`，冻结清单 v2 的 12 种子口径）。小规模补种子不能替代
  大规模结论自身的不确定性声明。
- **外部有效性边界（意见十一）**：全部机制结论出自 WikiText-103 小模型
  （BPE-8k，d≤512）。跨语料、跨规模、跨任务的推广是**主张外的**——
  compute_plan 的 B 档规模面板（FineWeb-Edu、0.14B+）是为此设计的独立
  验证路径，其判据见 §16 的口径约束；在此之前，标题与摘要不得把机制
  结论写成通用规律。
- **九领域审计入口（意见十一）**：`audit_gates.py` 把数据泄漏（分词器
  拟合源/词表一致性/文档重叠）、自回归正确性（前缀不变性/标签移位/解码
  一致性，纯 CPU 单元审计）、分块归因、边界对齐 vs 随机基线、长度外推
  评分口径、基线公平、失败集中度、预算对账与效率测量统一为可执行子命
  令——「不能排除」的问题必须有可运行的核验入口，而不是留在评审回复
  的清单里。
