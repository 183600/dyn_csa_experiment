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
- `run()` 在第一个单元开训前就以 `status='running'` 写 `run_manifest.json`
  （代码/环境/数据版本、超参、计划单元），结束时改为 `complete` 并列出
  `missing_cells`。这样进程被 cap/机器杀掉时仍能追溯这一轮到底准备跑什么，
  而不是只在全部结束后的汇总里出现。
- 代码版本除 `git_commit` 外还记录 worktree 的 status/diff 摘要
  （`git_status_sha256` / `git_diff_sha256`）：同一个 commit 下的未提交
  修改不会与干净工作树被误认成同一个代码版本（只存摘要，不落盘 diff 内容）。
- 数据版本由 `_data_version` 汇总 dataset/protocol、token 数、词表与
  train/val 内容指纹，随 provenance、每个单元与 manifest 落盘；它与
  `rng_fp`、`run_cfg` 一起构成「代码 + 数据 + 配置 + 随机源」的可追溯键。
- 小批量过拟合与同机重复运行探针只在**通过**时才作为进程内缓存命中；
  缓存的失败会在下一次调用重跑并再次硬停，避免上层 catch 后在同一进程内
  静默绕过安全检查。缓存键包含**完整探针配置**（种子、步数、形状、收敛
  阈值、架构、代码戳）以及语料指纹：一次宽松探针的通过不能认证之后的严格
  探针。
- 所有真实语料训练路径都先跑**数据路径**的小批量过拟合检查，而不只跑合成
  数据检查：`run()`、长度/检查点阶段（v7 `run_lenphase`）、warmup 阶段与
  NIAH 自训练循环分别在实际 token 流或实际任务批上确认损失能迅速下降、
  next-token accuracy 接近 100%、梯度有限且参数确实更新。
- 自定义阶段（warmup / lenphase / probe）同样写 `experiments.jsonl` 的
  `panel_start` / `panel_finish` / `cell` 事件：代码戳、配置、种子、数据
  版本、provenance、超参、RNG 指纹、训练曲线与最终评估结果都在自动记录里，
  不需要手工誊写。`run_manifest.json` 额外给出 `tracking_integrity` 计数，
  让缺失 provenance / 超参 / RNG 指纹 / 曲线的单元一眼可见。
- 面板轴（seeds / variants）在任何探针、指纹或 manifest 之前先校验并去重：
  重复 seed 不会再把同一个 `variant::seed` 训练两次，缺省/空 seed 列表也不会
  出现「训练了 fallback seed、manifest 却计划 0 个单元」的错位。
- 每个面板在训练前运行显式的 `seed_contract_check`：同一种子必须复现
  Python/NumPy/Torch CPU/每张 CUDA/TF（若已导入）/DataLoader generator 与
  worker hook 的抽样，换种子必须逐流改变；报告写入 `seed_contract.json`
  并进入 manifest 的 `evidence`。它比 3 步损失复现更快，且能点名是哪一路
  RNG 没有播种。
- 环境快照记录 Python/NumPy/Torch/TF 版本、cuDNN 确定性开关、TF32 与
  float32 matmul 精度、线程数、`CUDA_VISIBLE_DEVICES` 以及每张可见 GPU 的
  UUID/显存/计算能力；另附 `requirements*.txt` / `pyproject.toml` 的内容
  摘要与已安装包版本，避免同一 commit 下换 wheel 的运行被误认成同一环境。
  跨硬件漂移不能被消除，但可以被解释和追溯。
- `run()` 在加载语料前先按面板第一个种子固定数据构造路径（分词器拟合、
  缓存重建、未来 DataLoader 的随机源）；每个训练单元随后在 `train_variant`
  内重新按自己的种子定界，因此面板级播种不会改变任何记录。
- 模块导入必须零副作用：不读写状态文件、不下载、不改随机种子。依赖产物的
  检查在产物缺席时必须可跳过而不是失败。

## 8. 仓库边界

仓库只承载**源码、驱动脚本与设计文档**：实验结果目录、分析目录、逐版
报告、运行状态文件都不入库（`.gitignore` 白名单保证，`docs/` 仅留 `design_notes.md`；原 `compute_plan.md` 已退役并从仓库删除）。
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

代码落点：`boundary_alignment(..., rand_kind=...)` 接受
`matched_cut_count`（固定块数/匹配切割数）与
`per_sequence_independent`（每序列独立切），每个 cell 记录
`bnd_rand_kind` 与 `bnd_rand_definition`；独立切口径的闭式值是覆盖率近似，
因此同时记 `bnd_rand_exact: False`，不参与精确口径的池化。`aggregate` 遇到
同一臂混用两种口径时只报 `boundary_rand_kind_conflict`，不产出
`boundary_excess_dyn`；`fuse_analysis` 同样把混口径的层序列整体屏蔽。

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
  里点名。`training_knobs()` 的摘要还进入 `_run_fp` 的 `_tk` 字段：改变
  优化器 / betas / eps / 调度 / 裁剪 / 增强常量会像改变 lr 一样使旧
  `run_cfg` 失效并按正常 resume 规则重训，而不是留下旧优化器记录冒充新
  协议。`run_cfg` 相等也**不**证明语料与评测切片相同（它只记录 token 数、
  序列长度与形状）。因此 `pair_reason` 还要求 `data_fp` / `val_fp` /
  `eval_fp` 两条都带且逐项相等；只有一条带指纹判为 unverifiable。扩展规模
  时同一 outdir 可能先后落入不同 cap 的缓存，这一步防止把两个不同验证切片
  的 PPL 当成同一切片配对。`aggregate` 在 (variant, protocol) 组内也做同一
  检查：组内出现多个 `data_fp`/`val_fp` 直接拒绝，而不是把不可比的 PPL
  池化成一个 `ppl_pooled`。历史面板只有新旧记录都不带指纹时走兼容路径。
  面板级 `controlled_variables_report` 检查**该臂的全部种子记录**
  （不是只看第一条）的 `run_cfg`、`train_knobs` 与结构化 `hyperparams`
  关键字段，并点名「同一条臂内不同种子用了不同协议」的混合情况。
  模块消融的**逐种子配对**还额外过 `_hyperparam_reason`：`run_cfg` 不
  承载的 `mlp_ratio` 等结构化字段必须逐项相等（两边都缺才走历史兼容
  路径）；同一 run_cfg 但 MLP 宽度不同的两条记录不会进入模块效应统计。
  基线对比不使用该条门槛——它的 MLP 宽度本来就是为参数量对齐而改的，
  由 `param_controlled` 单独把关。
  `BASELINE_DESIGN` 另把主方法与参数对齐稠密基线
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
  不会在烧掉数小时后才暴露坏臂。`ablation_completeness_audit` 把「完整性」
  变成机器读数：留一法下每个创新点必须**全部声明的对照都两臂在场**才算
  complete，缺哪个臂会被点名；它也接受 `arms_by_combo` 的全因子 2^k 映射
  （全因子是更昂贵的替代方案），8 种组合少一种（例如只有 A/B/C 的单独臂
  和 AB/AC/BC，漏掉 C 或 ABC）会给出精确的缺失组合，而不是把 7/8 面板
  描述成全因子。全因子模式另要求**每个组合一个独立臂**：同一个臂 id 被
  映射到两个组合（例如 A 与 AB 复用同一条 run）会进入 `reused_arms` 并
  判为不完整，因为这样的表无法把效应归因到组合。部分组合因此不能再支撑
  「每个模块都有用」的结论。
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
  效应大于种子噪声，避免大 n 的 p 值把亚噪声效应包装成真实增益。每条
  对照还落盘 `with_mean±with_std` / `without_mean±without_std`、
  `relative_effect_pct`（Δ 占基线 PPL 的百分比）、`relative_noise_pct`
  与 `signal_to_noise`，因此「0.2% 的提升配 0.5% 的种子标准差」可以直接
  从记录复核，而不必反推原始数字。`paired_effect_stats` 还给出 Δ 的 95%
  置信区间（t 区间；3–5 个种子时不用正态近似）与**去一种子稳定性**
  （`loo_means` / `seed_driven` / `seed_robust`）：若丢掉任意一个种子就使
  均值变号，则该方向由单个种子主导，审计写入 SEED-DRIVEN 警告——显著 p
  不等于机制效应稳定。`arm_summary` 对每条臂按 (variant, protocol) 给出
  n、mean±std 与 `meets_min_seeds`，关键实验低于 `min_n` 的臂进入
  `arms_below_min_seeds`。模块消融的参数量差异 >1% 时另标
  `capacity_confounded`：增益可能来自新增参数而非机制，报告必须给出参数
  差或补参数对齐对照（基线对比本来就把参数对齐作为硬门槛）。
  `signflip_min_consistency` / `signflip_power` 的二项尾概率走正则化不完全
  贝塔 / log 空间（不再是 `comb(n,k)/2**n` 的整数转浮点），因此补种子计划
  扩到上千个种子也不会溢出；`paired_effect_stats` 的去一种子扫描是 O(n)
  单遍（总数减单个种子），不会因规模扩大变成平方复杂度。
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
检验）；20<n≤40 用 meet-in-the-middle 精确计数（两半各 2^(n/2) 个带符号和，
两次二分查找统计 |sum|≥观测值的符号指派），补种子面板扩大到几十个种子时
仍是**精确** p。n>40 使用确定性的 Lugannani-Rice **鞍点尾近似**：零假设下
`S=Σ s_i d_i` 的矩母函数为 `Π cosh(t d_i)`，求解 `K'(t)=|S|` 后按鞍点公式
取尾概率；其方差是符号翻转零假设的 `Σ d_i²`，**不是**以观测均值为中心的
`n·s²`（后者会随效应增大而缩小分母，给出反保守的 p）。鞍点 p 另被钳制在
精确检验下限 `2/2ⁿ` 之上，并由返回值的 `method` 明确标注。零 GPU 报告
构建器 `build_report.py` 走**同一确定性公式**（n≤40 MITM，n>40 鞍点），
不再使用随机 Monte-Carlo，报告与统计审计不再出现两套 p 值。
`signflip_p_floor` 给出单比较的分辨率下限，
`holm_min_attainable_p` / `signflip_min_n_for_holm` 给出族级下限，使
「未显著」能区分是分辨率上限还是数据结论。

G0 的训练级等价检验 `tost_paired` 使用**真正的配对 t**（df=n−1）：两侧
单边 p 由 Student t 尾概率给出，CI 用 t 临界值而非把样本 se 当作已知正态
z（n=6 时 90% t 区间比 z 区间宽约 20%，用 z 会把「未判等价」误判为等价）；
零方差样本恰好落在 margin 边界时取 t 极限 P(T≥0)=0.5，不会把边界相等
判成等价。

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
  （已退役算力规划的 S1/S2 阶梯）**全部不可能达到 0.05 显著性**，无论效应多大
  ——这些面板只能作为描述性证据与方向性读数；确认性结论必须等到种子数
  与族大小满足功效门槛（`exp_lib.holm_min_attainable_p` /
  `signflip_power`，冻结清单 v2 的 12 种子口径）。小规模补种子不能替代
  大规模结论自身的不确定性声明。
- **外部有效性边界（意见十一）**：全部机制结论出自 WikiText-103 小模型
  （BPE-8k，d≤512）。跨语料、跨规模、跨任务的推广是**主张外的**——
  B 档规模面板（`m0_supp.py`；FineWeb-Edu、0.14B+）是为此设计的独立
  验证路径，其判据见 §16 的口径约束；在此之前，标题与摘要不得把机制
  结论写成通用规律。
- **九领域审计入口（意见十一）**：`audit_gates.py` 把数据泄漏（分词器
  拟合源/词表一致性/文档重叠）、自回归正确性（前缀不变性/标签移位/解码
  一致性，纯 CPU 单元审计）、分块归因、边界对齐 vs 随机基线、长度外推
  评分口径、基线公平、失败集中度、预算对账与效率测量统一为可执行子命
  令——「不能排除」的问题必须有可运行的核验入口，而不是留在评审回复
  的清单里。

## 21. 实验设计与准备的三条硬门槛

### 21.1 预注册假设与方向一致的验证

`exp_lib.HYPOTHESIS_DESIGN` / `BASELINE_HYPOTHESIS_DESIGN` 为每个声明式
对照写出 statement / mechanism / `predicted_sign` / primary metric /
falsifier。Δ 的符号沿用「带模块 − 去模块」：`predicted_sign=-1` 表示
预期降低 PPL。审计在 `verified` 之外新增 `direction_conflict`：一个
族内 Holm 校正后显著、但方向与预注册相反的结果，只能算**反对该机制**
的证据，不能事后改写成支持。没有进入假设表的组件不施加方向门槛，但会
进入 `hypotheses_missing` 警告。

### 21.2 基线必须与主方法同预算调参

`exp_lib.BASELINE_TUNING_PROTOCOL` 声明同一组搜索 trial、同一验证集选择
指标（PPL，越低越好）与相同的 final seeds；`baseline_tune.py` 对
`full_matched` / `full_sw128_matched` / `csa_fixed` / `csa_dynamic`
逐臂跑同一网格并落盘 `baseline_tuning.json`。`baseline_tuning_audit`
要求：同一 trial 网格、同一 trial 数、只在 validation 上选择、选中项
确为该臂最优，且 `selected_metric_value` 必须是**该选中 config 自己**的
trial 值（不能拿另一个 trial 的最小值冒充）；trial 记录暴露
steps/tokens 时还要求跨臂训练预算一致。`methodology_audit` 把统计上显著但没有该产物的基线声明
放进 `untuned_verified_baselines` 并给出警告——未调参的弱基线只能
证明复杂模型比稻草人强；只有 `tuned_verified_baselines` 才支持
「优于精心调优的基线」。调参后的逐臂最优配置允许 lr/wd/warmup 不同，
因此其配对由 `baseline_tune.py` 的 `tuned_comparison.json` 按
「data/val 指纹、steps、tokens、参数量一致 + 等搜索预算」单独报告，
绝不与严格同 run_cfg 的架构对照混池。

搜索本身也必须跨种子：单一种子的 validation PPL 会把种子噪声选成最优
超参。`BASELINE_TUNING_PROTOCOL.search_seeds`（默认 `[0,1,2]`）声明同一组
搜索种子，每个 trial 的 `ppl` 是这些种子的**均值**，`seed_ppls` / `ppl_std`
落盘逐种子值与样本标准差；选择按均值而非单次抽样。`baseline_tuning_audit`
在记录暴露 `seeds`/`seed_ppls` 时额外要求：聚合值确为逐种子均值、标准差
确为样本标准差、**所有臂的所有 trial 使用同一搜索种子集**。final seeds
另检查逐臂覆盖一致，并给出该调参族（主方法与各基线两两对比）的精确检验
分辨率下限；种子不足时标为 resolution-limited，而不是读成零效应。

### 21.3 指标口径显式化

`exp_lib.METRIC_POLICY` 声明每个任务的主指标、方向、聚合方式与不平衡
处理；`metric_policy_audit` / `audit_gates.py metrics` 核验落盘记录：

- LM 以 PPL 为主指标（token-weighted），next-token accuracy 只是诊断；
- 边界检测的正类只占切割位置的一小部分，禁止报 cut accuracy，必须
  同时给 precision / recall / F1 与匹配块长的随机基线；
- NIAH 是 8192 路检索，主指标为 top-1 accuracy，必须同时给 answer NLL；
  不能把 top-1 标签伪造成 PR-AUC/ROC-AUC；
- 通用分类入口 `classification_metrics` 同时返回 macro / micro / weighted
  F1、balanced accuracy、逐类 recall；二分类有分数时另给 PR-AUC 与
  ROC-AUC，并在正类比例低于 1:1000 时显式提示以 PR-AUC 为主。

### 21.4 先立最简单的基线：计数模型地板

复杂机制的第一道对照不是另一个 Transformer，而是**调过参的最简基线**。
`exp_lib.SIMPLE_BASELINE_POLICY` 定下训练集计数的 unigram / bigram 模型
（unigram 是 context-free 线性计数模型；bigram 用 unigram 插值），平滑
系数与插值权重只在 validation 上选，**不读 test**。`baseline_tune.py
simple` / `full`（默认先跑 simple）落盘 `simple_baseline.json`：

- `simple_baseline_fit` / `ngram_lm_score` 是纯函数：按行边界评分（每行
  首 token 只由 unigram 预测，不跨行借用前一行末 token），PPL 为
  token-weighted，与 LM 指标口径一致；`baseline_tune.py simple` 用
  `val_batch[:eval_subset]` 的同一批行与神经面板比较，并把完整验证缓存的
  指纹写进 `data.cache_val_fp`（比较门槛）与 `data.eval_fp`（实际评分
  切片）；
- `simple_baseline_audit` 要求 selected config 确为最优、selected 值确为
  **该 config** 的 trial 值、各 trial 评估 token 数一致、只选在
  validation；缺失产物只报 `not_verifiable_locally`；
- `methodology_audit` 在 val 指纹一致时把面板主方法 PPL 与地板并排报告：
  主方法没打过地板会给出警告（`panel_beats_simple_floor=False`），
  指纹不一致则标 `comparable_to_panel=False` 并禁止引用该比较。

地板比较不是「机制优于稠密基线」的证据；它只是排除「复杂模型连最简
计数模型都不如」这一最基本的设计失败。稠密基线的同预算调参仍由 21.2
单独把关，两条证据不得混池。

### 21.5 假设完整性与指标对齐的机器审计

`hypothesis_audit` 把 21.1 的约定变成机器读数：每个声明式对照必须写全
statement / mechanism / `predicted_sign` / `primary_metric` / falsifier；
`primary_metric` 必须是 `METRIC_POLICY` 真正定义的端点；`predicted_sign=0`
只允许「是否存在差异」的探索性假设，`_classify_group_status` 会把它归入
`direction_unspecified`，不能被写成方向性优势。`methodology_audit`
输出 `hypothesis_audit` 与 `hypotheses_incomplete`，缺字段/指标未声明
在解释任何数字之前就会出现在 warnings 里。

`classification_metrics` 的 `positive_label` 参数显式指定正类（默认仍为
`labels[-1]`，与 sklearn 对齐），并新增 `f1_positive` / `f1_primary` /
`f1_primary_average`、`pr_auc_prevalence` 与
`pr_auc_lift_over_prevalence`：PR-AUC 必须对照正类流行率读取，macro F1
对请求的每个标签取平均（样本中缺席的标签按 F1=0 计入，不会被静默丢掉）。
`metric_policy_audit` 另外检查：F1 记录必须注明 averaging 口径；1:1000
以下只报 ROC-AUC 或缺 PR-AUC 流行率底线的记录会告警。


## 22. 评审三（review 3）后的统计与证据硬化

本节记录本轮实际改动的口径，避免把「代码里有字段」误读成「结论已确认」。

### 22.1 PPL 聚合与不确定性

- 训练/长度评估现在同时落盘 `nll_sum`、`n_tokens`、逐文档 `nll_per_row` /
  `n_tokens_per_row`（`eval_ppl(..., return_stats=True)`；长度面板另存
  `nll_per_seq`）。
- `aggregate` 的主聚合改为 `ppl_pooled = exp(ΣNLL / Σtokens)`，并保留
  `ppl_mean_arithmetic`、`ppl_median`、IQR、seed-level t 区间与文档
  bootstrap 区间。旧记录缺 NLL 时明确回退为 `arithmetic_seed_mean_fallback`，
  不伪装成 pooled。
- 报告仍打印 seed 标准差，但必须标成 **seed-level**；文档 bootstrap 才是
  评测文档抽样不确定性。`eval_subset` 只控制训练过程中的曲线评估；最终
  PPL 仍用完整固定验证批（历史口径），该区别在记录中由 `eval_rows` /
  `eval_fp` 显式化。
- 文档级区间必须针对**同一个估计量**：token-pooled PPL 的区间不能用
  「逐文档 CE 的未加权标准误」近似（文档长度不同时两者不是同一个量）。
  `_nll_doc_ci` 因此改用 documents-as-clusters 的稳健方差与 t 临界值；
  v12 的逐文档配对分析另给 `bootstrap_mean_ci` 的文档重采样区间（对逐
  文档 PPL 差与 log 比分别报告）。这些区间**以已训练的种子为条件**，
  只量化文档抽样不确定性，不能替代 seed-level 精确符号翻转检验。

### 22.2 符号翻转检验的假设与敏感性

- `paired_effect_stats` 现同时给出符号计数、paired bootstrap 区间、
  Wilcoxon signed-rank 敏感性 p（n≤16 精确）、以及调用方提供两臂数值时的
  **unpaired** bootstrap 区间。
- 这些都只是敏感性读数；主检验仍是 seed-level 精确符号翻转。相同 seed
  不等于随机分配处理，架构导致的 RNG/优化轨迹差异属于「臂」的一部分。
  p 值永远是 seed-level，不是 token-level。

### 22.3 族级校正与显著性措辞

- v7/v8/v9/v10/v11 的 analysis 与 `build_report.py` 都对**展示出来的**
  比较族做 Holm step-down；被省略/无效的比较保留 dummy p=1 槽位，不能靠
  少报比较来缩小 m。
- 报告正文的「显著」必须同时满足：n≥6（否则精确检验下限 >0.05 或
  族校正不可达）、原始 p<0.05、Holm p<0.05、|mean|>std、且通过
  `run_cfg`/`train_knobs` 配对门禁。任一不满足只能写「方向性 / 名义显著 /
  未过校正」，不得写「证伪」「机制性」「更稳固」。

### 22.4 训练目标、参数与 KV 的公平性字段

- 每条记录新增 `train_objective`（comp_reg 是否实际生效、每层
  `comp_lambda_mult` 的实际权重、delta_logit 的 LR 倍数是否实际生效、
  boundary policy）与 `param_breakdown`
  （total/attention/MLP/embedding/head/active）、
  `kv_cache_elements_per_token`。`comp_reg_applies` 现在按**实际乘子**
  计算：`csa_dyn_nocomp` 的乘子为 0，必须报告 comp_reg 未生效，而不是
  照抄配置里的 `comp_lambda=0.05`。
- `methodology_audit` 对同一 `run_cfg` 但目标不同的对照给出
  `objective_confounded` 警告与 `mechanism_claim_allowed=False`。这不会
  阻止配对（那会让所有 dense-vs-sparse 对比消失），但会阻止把它写成
  「隔离机制」。例外只限**声明的目标型创新点**：
  `OBJECTIVE_INTERVENTION_KEYS` 中 `compression_regularizer` 的
  comp-reg 项与 `task_learned_boundary` 的 boundary-policy 项是干预本身，
  记录为 `objective_expected_differences`；任何未声明的目标差异仍然
  判为 system-level。`audit_fairness` 也在面板级报告跨臂
  `objective_equal`，目标不一致作为警告而不是静默通过。
- 参数量匹配容差统一为 1%（`_param_control` 与报告共用）；冻结确认的
  准入还检查 params 相对差与完整 seed 覆盖。

### 22.5 动态边界的梯度来源（重要）

- 默认 `csa_dynamic` 的硬切分是 `gate.detach() > 0.5`，在 `no_grad` 中
  生成块边界；LM 交叉熵**不能**通过离散边界反传。`delta_logit` 与
  `fuse_conv` 只从 `comp_reg` 得到梯度，`W_DQ/W_DK/W_w` 走的是 soft
  attention 的 straight-through 代理，不是 top-k 身份。因此默认臂应描述为
  「余弦阈值 + 由压缩率正则训练的标量偏移」。
- 新增 opt-in `csa_dynamic_task`：切分决策按 gate 概率采样，训练损失加入
  score-function 项 `-(CE - baseline) * logπ / n_decisions`。这才是「任务
  损失可训练边界」的臂；它已加入 `ABL_VARIANTS` / `ABLATION_DESIGN`。
- `csa_dyn_nocomp` 把 `comp_lambda_mult` 置零，同时冻结了 `delta_logit`
  的唯一梯度来源；它是「损失项移除」臂，不是「仅关正则、其余完全不变」
  的臂，报告必须这样写。

### 22.6 合成记录、长度与规模

- `run_gapfill.synthesize_long_summary` 生成的 `synthesized` 记录只用于
  历史曲线展示；v10/v11 的 crossover 现在**先剔除**这些 seed，而不是
  事后标记。拟合至少需要 3 个独立、预算可比的点，否则只报
  `insufficient_points`，不报斜率/幂律。
- 长度面板统一在 512 训练、在 512–4096 评估，`n_seq` 写入 recipe 与记录；
  这是**长度外推探针**，不是长上下文训练。块数/选择率随长度变化，报告
  必须并列给出。
- 交叉规模面板的 batch/seq/steps/eval 网格并不相同；报告保留
  「方向性、非幂律」措辞，不把 4 个点拟合成 scaling law。

### 22.7 证据归档与环境

- `archive_release.py` 现在打包 `DECISION_LOG.md`、`requirements.txt`、
  `requirements.lock`、冻结锁、逐文件 SHA-256 清单、checkpoint 哈希、
  `pip freeze` 与环境快照；`verify` 独立复核。
- `exp_lib` 不再在 import 时自动 pip install；`HF_ENDPOINT` 默认官方，
  `CSA_HF_MIRROR=1` 才使用镜像。`pyproject.toml` 声明 Python ≥3.12，
  `requirements.txt` 给兼容范围，发布时用 `pip freeze > requirements.lock`。
- 仍存在的边界：WikiText-103 官方 split 之间有近重复文档，审计只量化、
  不删除；16 条 test 文档不覆盖语言总体；实际推理延迟/吞吐/峰值显存必须
  在真实 checkpoint 上测量，解析 FLOPs/KV 只作结构性读数。

## 23. 训练健康、NaN 排查与坏案例分析（结果分析层）

### 23.1 曲线必须用于判断，而不只是被记录

- `train_variant` 在每个 `eval_every` 点同时评估验证 PPL 与**固定训练
  batch** 的 CE（固定 batch 使用独立 RNG 流，不改变训练 batch 顺序）。
  记录 `train_ce`、`val_ce`、`gap`、验证子集的逐文档 95% CI，以及
  `curve_health`：`healthy` / `overfitting` / `underfitting_or_stalled` /
  `validation_degrading` / `nonfinite` / `insufficient_data` /
  `unmonitored`。
- 同时记录 `best_step`、`best_val_ppl`、`final_minus_best`。最终表格仍用
  预注册的 final-step PPL；best 数值只作诊断，不能在未预注册时拿来挑选
  后再冒充固定预算比较。
- 曲线使用的 `eval_subset_rows` 与最终 full-val 行数不同，报告必须并列，
  不能把固定子集的波动读成 full-val 的变化。

### 23.2 NaN / Inf 的系统排查顺序

1. 每个（抽样）batch 进入模型前检查 dtype、形状、token id 范围；未来浮点
   特征必须检查 `isfinite`。
2. 前向 loss 非有限：记录 step、CE、comp_reg、PG 项、LR 和 batch 指纹，
   **跳过该 step 的更新**，不把 NaN 写进参数。
3. 反传后检查裁剪前梯度范数；非有限时清空梯度并跳过更新（DDP 下用
   all-reduce 保证所有 rank 同分支，避免集合通信死锁）。
4. fp16 记录 GradScaler 的 scale 最小值、overflow/skipped step 次数；bf16
   路径记录是否有非有限值。
5. 参数抽样检查发现非有限值时**中止**，而不是继续训练。
6. `nan_to_num` 调用点保留原数值语义，但默认计数；`CSA_NAN_AUDIT=1`
   统计被清洗的元素数，`CSA_STRICT_NAN=1` 直接抛错并打印调用位置。
7. 若仍复现：查数据中的 NaN/Inf、`log(0)`/除零、softmax 数值溢出，再查
   fp16 溢出，最后查梯度爆炸或学习率过大。诊断顺序写进记录的
   `opt_diag_summary`，不依赖事后猜测。

### 23.3 坏案例分析

- `exp_lib.analyze_errors` 与 `error_analysis.py` 对保存的 checkpoint 输出
  逐 token CE、argmax、top-k、entropy、margin、位置、文档行号、解码上下文；
  同时给出最差序列、位置分桶、罕见目标 token、边界 token 分桶。
- 输出 `bad_cases.jsonl` / `bad_cases.md` / `summary.json`。`--compare-ckpt`
  额外输出逐 token 配对差 `paired_errors.json` / `paired_bad_cases.jsonl`，
  回答「某臂在这些位置上系统性更差」而不是只比较总 PPL。分析默认只在
  validation 上进行；冻结 test 的逐样本分析必须走 v12 的一次性访问协议，
  不能在 test 上反复看错例再改设计。
- 看到高 CE 先分清：标注/移位错误、近重复、文本本身极模糊、罕见 token、
  长距离依赖、块边界/格式问题，还是模型盲区。没有这一步，0.1–0.5 PPL 的
  差异无法对应到具体行为。

### 23.4 效率与精度的权衡

- 主训练在保存 checkpoint 后实测 prefill p50/p95、tokens/s、allocator
  峰值；没有 KV cache 时明确标成 `prefix_prefill`，不得写成增量 decode。
- `build_report.py` 的 Measured efficiency 表与 PPL/参数量并排；解析 FLOPs
  只作结构性读数。微小且不显著的 PPL 涨点若伴随数量级延迟/显存成本，
  不能称为改进。
- 训练峰值显存按几何配置缓存，用于校准 `_auto_ckpt_mode` 的解析估算
  （只允许上调需求，避免低估导致 OOM）。

### 23.5 冻结确认的训练健康门禁

- P5C 的长度阶段可在 `diag_eval_every>0` 时做**仅诊断**的固定验证切片
  评估；它不参与 checkpoint 选择、不改变冻结臂/种子/终点。
- v12 新增 `P5H` 训练健康阶段；`P5T` 在写冻结锁、读取 test split 之前
  重新检查冻结臂的 `curve_health`。缺失/非有限/过拟合/停滞时拒绝消耗
  一次性 test 访问；`V12_ALLOW_UNHEALTHY=1` 是显式、会打印的覆盖开关。

## 24. Open-block reading: maximum block length × current-prefix representation

### 24.1 Questions and arms

The open-block panel separates two factors that the earlier single-knob
ablations conflated:

1. **Maximum block length** (`max_block_tokens`): does forcing a cut at a
   maximum length help, with the reading rule held fixed?
2. **Reading the still-open block** (`open_block_mode`): does the wait for the
   current block to close create an information-access bottleneck?

For a query position `t`, let `s_t` be the start of its (still open) dynamic
block and `P_t = [s_t, t]`.  All arms share the same dynamic cut table; only
the read stage changes:

| id | read rule |
|---|---|
| `closed` | only blocks with `end < t` (the historical strict gate) |
| `prefix_summary` | closed candidates + one causal pooling of `P_t` |
| `prefix_singletons` | closed candidates + one length-1 pooled representation per token in `P_t` |

`prefix_singletons` is **not** raw-token K/V.  With one A lane the softmax is
exactly 1 and no position bias survives, so the pooled value is
`Xa[u] = x_u @ W_aKV` (then `core_kv = normalize(.)` and `W_kvhead`), i.e. a
length-1 compressed representation through the same module and parameters.
The singleton path deliberately uses `overlap=0`: with previous-block overlap
lanes it would no longer be a length-1 representation.  The summary path keeps
the configured overlap convention.  A raw-token variant would be a separate
`prefix_rawtokens` mode and is not part of this panel.

The prefix is a temporary representation for the current query.  It does
not modify `bid`; for the next query it grows with `t`, and once the block
closes the ordinary closed-block summary takes over.  A final block summary
and a past temporary summary of the same block are never both retained.

### 24.2 Causality is enforced at construction, not by loosening the mask

`block_readable()` keeps its strict `pos > last_tok` rule.  The prefix path is
a separate, causal candidate source.  For query `t`, the current block start,
observed prefix length, pooling inputs/weights/normalizer, indexer key,
attention key/value, and position/block-length features all use only
`x_{<= t}`.  In particular:

- the current block's **final** length/end is never read;
- `B_pos_a` is indexed by `u - s_t` (known at `u <= t`), never by a
  block-tail-relative position;
- the previous-block overlap lanes (when `overlap > 0`) are the same causal
  lanes the closed pooling uses: block `b-1` is entirely `< s_b`;
- `max_block=None` is converted at the segmenter entry point to an explicit
  bound equal to the current sequence length — no hidden large constant.

`prefix_summary` for the full prefix includes the same previous-block overlap
convention as the closed pooling, so at `t = end-1` it equals that block's
closed summary up to floating-point error.  The auxiliary `outside_window`
scope builds the prefix from `U_t = P_t \ W_t` only and disables the overlap
lanes (they may lie inside the local window).  If `U_t` is empty there is no
prefix candidate.

Two implementations exist for the prefix pooling: the optimized cumulative
form (`causal_prefix_pool`) and a per-query stable-softmax reference
(`causal_prefix_pool_reference`).  The CPU suite compares their forward
values and gradients, and checks the block-end equality above.

### 24.3 Budget, masks and implementation paths

The main panel uses the **shared top-k** rule for CSA:
`closed candidates ∪ prefix candidates` compete for the same `index_topk`
slots.  There is no separate prefix scorer; all candidates use the same
`qI/kI/w` path.  `_indexer_selection` pads with invalid slots when fewer than
`K` candidates exist; padding is marked invalid and is never read, and valid
selected ids are unique.

`prefix_singletons` uses the existing local-token K/V table, because a
length-1 pooled block is exactly `normalize(Ca_raw[u])` at the same module
parameters.  `prefix_summary` computes its own pooled summary and then the
same `W_kvhead`/normalization path as a closed block.  A summary never
materialises the per-singleton `[T, P]` candidate table; the singleton table
is guarded at 16M elements so a pathological no-cap block fails explicitly
instead of silently exhausting memory.

The open path is implemented in the eager row loop.  `CSA_KERNEL` and the
batched eager twin do not implement it and fail rather than silently running
a different read rule; the same guards refuse the kernel path for a
non-default `(max_block, pool_position_capacity)` geometry, because the
kernel's pool-geometry cache keys on `max_block` and predates the capacity
decoupling.  `eager_batch=False` is therefore the controlled choice for this
panel (all arms, including `closed`).  HCA has no top-k: it reads every closed block.  An open HCA
layer must declare `selection_budget=all_closed_plus_prefix` (an **additional**
read slot); the main matrix uses pure CSA so all three arms share one K.

### 24.4 Position capacity is decoupled

`max_block_tokens` controls forced cuts only.  `pool_position_capacity` (default
129) controls the `B_pos_a/B_pos_b` tables for every arm; positions beyond the
capacity are truncated by `pool_position_policy='truncate'`.  The report records
`position_truncated_tokens` / `position_truncated_blocks` and the fraction of
block tokens affected.  The default `(max_block=128, capacity=129)` is the
historical configuration, so the closed path is unchanged.

### 24.5 Running the matrix

`open_block_supp.py` defines the main matrix:

```
max_block: 32  → closed
           64  → closed
          128  → closed, prefix_summary, prefix_singletons
         None  → closed, prefix_summary, prefix_singletons
```

Commands:

```bash
python test_open_block.py                 # K1-K14 causality/reference/mask/analysis gates
python open_block_supp.py smoke           # all 8 cells, synthetic, a few steps
python open_block_supp.py pilot --seeds 0 --steps 100
python open_block_supp.py main --seeds 0,1,2,3 --steps 1500
python open_block_supp.py report
python open_block_supp.py aux             # outside-window auxiliary panel
python open_block_supp.py extended        # contingency: 4 caps x 3 modes
```

`main` is resumable and refuses a record whose code stamp / recipe
fingerprint no longer matches.  `report` computes paired per-seed NLL deltas:
`summary - closed @128`, `singletons - summary @128`, and the interaction
`(summary - closed | None) - (summary - closed | 128)`; it reports per-arm
`n`/mean/sample SD and token-pooled PPL, plus paired mean, sample SD, `d_z`,
exact sign-flip p, Holm-adjusted p within the declared primary family, and the
attainable-p floor.  At 4 seeds the raw floor is 0.125; this is a resolution
limit, not evidence of no effect.  The primary family has three Holm
comparisons, so the family-corrected seed floor is 7 (3·2/2^7 = 0.047); six
seeds only suffice for one raw comparison and are not enough for a family
claim.  A controlled comparator permits only the declared factors
(`max_block`, `open_block_mode`, `open_block_scope`) to differ and rejects
undeclared recipe/layer differences: the compact hyperparameters, the
`train_knobs` optimizer/schedule/clipping/augmentation/precision descriptor,
the derived training objective, and the matched-initialization digest must all
agree.  A cell whose key seed/variant disagrees with the record is excluded
and named rather than silently paired.

### 24.6 Recorded diagnostics

Per validation sequence/layer (only when `compression_report` enables stats):

- prefix age `a_t = t - s_t + 1`, P50/P95/max;
- outside-window prefix length and nonzero fraction; prefix length;
- position-capacity truncation;
- valid prefix candidates, prefix selection rate, selected closed count,
  selected total, unused slots, and the number of history blocks that a
  closed-only top-k would have selected but the open selection displaced;
- attention mass assigned to prefix slots.

Efficiency (tokens/s, peak memory, prefix-prefill latency) comes from the
existing `measure_efficiency` record.  Prefix candidates can displace history
blocks: the main result measures "is a prefix useful under a fixed K", not
"is extra information useful if the budget is increased".

### 24.7 Interpretation boundaries

- A better `closed` arm at a larger cap does not by itself prove the cap is
  good; it only supports a block-length effect under this recipe.
- A `prefix_summary` gain at `None` that disappears at 128 supports an
  interaction with waiting, but must be read with block lengths, selection
  rates and window coverage.
- If prefix candidates are rarely selected, that says the selection path did
  not choose them; it does not show that prefix information is useless.
- If the gain exists only with additional read slots, it is an extra-budget
  effect and must not be pooled with the equal-K results.
- `full_prefix` intentionally includes tokens already inside the local
  window (a duplicate information path).  The auxiliary `outside_window`
  panel isolates the window-exterior gap; the two scopes must not be mixed in
  one comparison.
- All formal arms train from scratch; a checkpoint from a closed run switched
  to an open mode is a diagnostic, not a substitute for training adaptation.

## 25. 评审四（review 4）后的证据门槛：把「有字段」变成「有闸门」

本轮不再新增实验面板，而是把四个评审反复指出的「软约束」改成机器可检的
闸门，并明确仍然无法由本仓库回答的部分。

### 25.1 统一的显著性措辞闸门

`exp_lib.reportable_claim` 是报告正文「significant / confirmed」的唯一
判据：n≥6、原始 p<α、Holm p<α、|mean|>std、族级最小可达 Holm p≤α 全部
满足才返回 `allowed=True`；否则返回 `underpowered` /
`resolution_limited` / `not_significant` 与逐条原因。v10/v11 的
「强版本成立」判定、`build_report._sig_gate` 与 v12 的确认规则都走同一
口径。`attach_holm` 现在同时输出 `family_min_attainable_p_holm` 与
`family_resolution_limited`：当 m 个比较 × n 个种子的最小可达 Holm p
已经 >α 时，p_holm=1.0 是分辨率上限，不是零结果，报告必须这样写。
v10 的 NIAH-natural 对照此前在 Holm 之后才构造，已改为并入同一族重新
校正，不再出现「标题写 Holm、表里是原始 p」的路径。

### 25.2 交叉点标度拟合的拒绝规则

`exp_lib.crossover_scaling_fit` 取代了各驱动里的 `np.linalg.lstsq`：

- 重构件（`synthesized`）不进入拟合；
- 任一点删失（窗口内未交叉）即拒绝拟合，而不是丢掉删失点后只拟合
  更快交叉的子集；
- 需要 ≥4 个**去重后**的宽度点；同一 d 出现多个面板时保留训练 token
  最多的（token-matched）臂并列出被丢弃者；
- 训练量最大/最小 >1.25 时拒绝，报告 `training_volume_confounded`；
- 通过时斜率、95% CI、p、R²、留一斜率范围全部来自
  `loglog_regression`，并固定 `claim_level='descriptive'`。

因此 n=2 的恒等式、删失偏倚、d=512 三个 4000/8000/flop 面板互相冒充
独立点这三条路径都被堵住。剩余边界：没有实现 Tobit/生存回归；点数
仍少，结论只能是描述性趋势。

### 25.3 初始化的可核对性与计算预算

`init_core_fingerprint` 对每臂共享的 trunk 参数（tok/pos/head/norm）
取哈希，作为训练记录字段 `init_core_fp`；`pair_reason` 要求成对记录
两边都有且相等，否则判 unverifiable/different。这样「按 seed 配对」
不再只是声明：至少共享部分的初始化被机器核对了。架构差异导致的
RNG 流差异仍属于「臂」的一部分，因此 `rng_fp` 仍只报告不强制。

`_param_control_detail` 在总参数量之外报告 non-embedding 与 active
参数量差（random-indexer 的死参数、dense 的 MLP 重分配因此在表里可见）。
`_compute_control_detail` 用实测 `train_time_s/tokens_seen`（无计时时
退回 6ND 代理）给出 compute 比与「哪一臂更贵」；当更贵的一臂恰好是
声称的赢家且比值 >1.5 时，`mechanism_claim_allowed=False`，
`_classify_group_status` 返回 `system_level_only` 而不是 `verified`。
v12 冻结统计另外拒绝缺少 `compute_budget_disclosure` 的非描述性清单。

### 25.3b 训练 token 记账

`tokens_seen` 是**窗口抽取数**，不是唯一 token 数：训练窗口独立均匀采样、
彼此重叠，8M token 缓存上 3000 步 × 12 × 512 并不等于消费了 18.4M 个不同
token。记录现在同时写 `tokens_seen_unique_estimate`、
`tokens_seen_unique_frac_estimate` 与 `token_accounting_note`——按独立均匀
窗口抽样的覆盖率估计 `1-(1-seq/N)^draws`，是估计而不是测量；任何「等 token
训练」的主张应同时给出抽取数与覆盖率估计。

### 25.4 陈旧记录与长度截断的 fail-closed

- `build_report.sec_p1l` 与 v10 的按长度表：没有任何记录携带当前代码
  语义戳的臂**整臂排除**，不列表、不算 ratio、不进入外推结论；表下
  明示被排除的臂。此前「没有新记录就引用旧值并加 ⚠」的路径已删除。
- `_agg_entry`：多配置组且无当前戳时返回 None（拒绝引用），单条显式
  `run_cfg` 非当前戳时同样拒绝。
- 长度 cell 缺少 `truncated` 字段时按「不可验证」处理（不进入 ratio），
  `audit_gates.length` 也把缺失标记记为 problem；`truncated=False` 才是
  全长测量。

### 25.5 文档抽样、效率与选择归因

- `ppl_ci95_doc_bootstrap` 进入 P2 表；v12 主比较表新增「文档 Δ 95% CI」
  列。`MIN_DOCS_FOR_POPULATION_CLAIM=64`，低于它时
  `population_claim_allowed=False`，主张只针对冻结的评测切片。
- v7 长度阶段现在记录 `measure_efficiency`；新增
  `measure_efficiency_sweep`（batch × length、CUDA event 计时、设备/
  torch/triton/TF32 元数据）与 `audit_gates.efficiency_pairs`：只有同
  token 数、同设备/torch、同 KV 语义、且比值超出 ±5% 才允许说
  「更快」。解析 FLOPs 的措辞已改为「结构性估计，不是实测加速」。
- `audit_gates.selection` 定义了选择归因工件的契约（recall@k、
  mass_covered、随机基线、per-seed 检验）。没有该工件时审计返回
  `not_verifiable_locally`，并明确「学习选择器找到了相关信息」是
  **未支持的机制主张**。这不能替代真正的 oracle/recall 实验，但它把
  「没有直接机制证据」变成了可执行、不可静默跳过的状态。

### 25.6 证据归档

`audit_gates.release` 解析 `RELEASES.md`：没有任何同时给出 commit、
归档 URL、两个 SHA-256 与 verifier 的行时，状态为 `no_release`，
并声明所有报告数值不可独立复核。`archive_release.py` 现在额外打包
`pyproject.toml`、cache/tokenizer 元数据与 `run_*.log`（存在时）。
本快照仍是 source-only：仓库**没有**随附证据归档，这一点在 README、
README_zh 与 `RELEASES.md` 中均已明示。

## 26. 基线地板、调参网格与开放块假设（本轮设计硬化）

本轮不新增 GPU 面板，只修正「简单基线是否被公平对待」与「调参是否真的
覆盖了最优区间」这两类设计缺陷。

### 26.1 计数模型地板必须用嵌套验证

`baseline_tune.py simple` 以前在整个验证切片上选择 unigram/bigram 的
平滑与插值系数，并在同一批行上报告 PPL。这与神经面板的嵌套验证不一致：
神经臂的最终估计使用靠后的 holdout 行，而计数模型已经用这些行调过参，
地板因此获得了额外的选择优势。

现在 `simple_baseline_fit` 接受 `selection_rows` / `evaluation_rows`：
- 平滑/插值系数只在较早的时间连续行折上选择；
- 被选中的配置只在不相交的较晚行折上报告 `holdout_ppl`；
- `best_ppl` 是**选中臂的 holdout PPL**，不是选择折上的最小值；
- 工件记录 `row_split`（policy、n_rows、folds、行范围、行指纹）、
  `selection_fp`/`eval_fp`（token 指纹）与 `n_eval_tokens`。

`simple_baseline_audit` 会拒绝以下情况：声称嵌套但缺少行折元数据；
行折不可由 `(n_rows, seed, n_folds, selection_fold, evaluation_fold)`
复现；选择行与评估行重叠或顺序颠倒；`best_ppl` 不是 holdout 值；
嵌套报告使用了与协议声明不同的 fold 布局；holdout token 数不一致。
`methodology_audit` 还要求地板的评估行指纹与面板的 `holdout_row_fp`
一致，并且**每个 seed 的 token 分母**一致，否则拒绝比较。默认平滑网格
从 3×3 扩展为 6 个 alpha × 6 个插值系数（计数表共享，额外成本只是一次
打分），避免「地板只是被测过的几个值里最不差的」。

工件查找顺序是：面板目录 → 面板父目录 → `results_baseline_tuning/`。
最终调参面板位于 `<outdir>/final/`，而 `baseline_tuning.json` /
`simple_baseline.json` 写在 `<outdir>/`；此前只查 final 目录会把两个门都
静默降级为 `not_verifiable_locally`。

### 26.2 「击败地板」必须是 seed 级对比

`panel_beats_simple_floor` 仍保留为逐点读数，但新增 `floor_contrast`：
把每个神经 seed 的 PPL 减去确定性的地板 PPL，做精确符号翻转检验，
报告均值、SD、p、p 下限与 95% CI，并给出
`beats_floor_significant` / `beats_floor_pointwise_not_significant` /
`beats_floor_pointwise_underpowered` / `does_not_beat_floor`。
只有第一种可以写成「显著优于简单基线」；逐点领先但未过检验时只写
「点估计更低，尚未证明」。

参考臂按**主端点（token-pooled PPL）**选择，而不是按算术 seed 均值；
两者在 token 数不同时可以给出不同排序。

嵌套记录把 `ppl` 换成 holdout 值时，必须同时把 `nll_sum`/`n_tokens`
换成 `nll_sum_holdout`/`n_tokens_holdout`。否则 pooled 聚合与指标核验会
拿 holdout 的 PPL 去对全验证集的 NLL 记账，二者描述不同样本；`_pooled_
ppl_from_records` 与 `metric_policy_audit` 也识别
`ppl_estimation_rows='nested_validation_holdout'` 并优先使用 holdout 记账。

### 26.3 调参网格要有边界覆盖

`BASELINE_TUNING_PROTOCOL.trials` 从 4 个试次扩展为 8 个，学习率覆盖
`5e-5` 到 `6e-4`，weight decay 覆盖 `0.01` 到 `0.2`。原先前四个试次
保持原顺序，历史工件仍可声明为完整网格的严格子集。

`baseline_tuning_audit` 新增 `grid_levels` / `grid_coverage` /
`grid_boundary_hits` / `grid_adequacy`。边界判定使用**该臂实际搜索过的
层**（`searched_levels`），而不只是声明的完整网格：一个被预算截断、
只跑了前两层的臂，其最大值就是搜索边界。若选中的 lr 或 weight decay
落在实际搜索的最小/最大层（且至少 3 层），说明最优可能位于网格之外；
若 lr 只有 1–2 层，也会提示网格过粗。选中值不在声明/实际搜索层上则是
problem，不是「调过参」。warning 明确要求扩展网格并重跑等宽搜索，不能
被误读成 pass。审计同时要求同一臂的各 trial（以及各臂）在**相同数量的
评估 token** 上选择；评估子集不同的 PPL 不能放进同一个 argmin。

### 26.4 指标口径从「声明」变成「按记录核验」

`metric_policy_audit` 现在不仅检查策略表字段，还核对落盘记录：
- 语言模型记录的 `ppl` 必须等于 `exp(nll_sum/n_tokens)`（或由
  `nll_per_row`/`n_tokens_per_row` 求和得到），否则说明报的不是声明的
  token-weighted PPL；没有 NLL 记账的旧记录只跳过核验，不伪造通过。
- 边界对齐的 `bnd_f1` 必须落在 [0,1] 且等于 `bnd_prec`/`bnd_rec` 的
  调和平均；随机基线也必须在 [0,1]。这样「F1 用了别的分母/口径」会被
  点名。
- NIAH 的 `acc` 必须在 [0,1]，`nll` 有限，且 query 数是正整数；缺失
  query 数的记录视为不可用。

### 26.5 开放块面板的预注册假设

`OPEN_BLOCK_HYPOTHESIS_DESIGN` 为主家族的三个对比登记了
statement / mechanism / predicted_sign / primary_metric / falsifier：
1. 开放块的前缀 summary 优于完全忽略开放块；
2. 在相同 top-k 预算下，singleton 表示优于 pooled summary；
3. 取消最大块长后，summary 的收益应更大（交互项为负）。

`open_block_supp.py report` 调用 `hypothesis_audit`，把审计与逐条假设写入
`stats.json` 和 Markdown；`audit_gates.hypotheses` 也覆盖这张表。描述性
单元（其它块长 × 读取规则）仍不进入 Holm 家族，也不得被写成方向性结论。

### 26.6 零信息地板：uniform token

计数模型至少要优于从词表均匀采样。`simple_baseline_fit` 现在把
`uniform_ppl`（= 词表大小）、`best_ppl_vs_uniform` 和 `data.vocab_size`
写入工件；`simple_baseline_audit` 校验三者一致，并在最佳计数模型 PPL
高于 uniform 时给出 warning：这可能只是小样本/无信号样本的噪声，也可能
是 smoothing/tokenizer/data 路径坏了，不能直接当地板使用。它不把
「PPL 比 uniform 高一点」判成 hard failure，以免在随机小样本上误杀。

## 27. 地板门槛、选择不确定性与评估行重采样（本轮硬化）

本轮不新增 GPU 面板，只修正「击败简单基线」与「调参赢家」这两类容易
被事后选择放大的读数。

### 27.1 方法臂自己必须过地板门槛

`methodology_audit` 原先按 token-pooled PPL 选一个**最好的面板臂**与计数
模型地板比较。这是一个事后选择：如果提出的方法输给稠密基线，而另一个
臂赢了地板，整块面板仍会显示「击败地板」。现在预先取
`BASELINE_DESIGN` 左侧的所有方法臂，逐臂与同一地板做 seed 级对比，并
对这一方法臂族做 Holm 校正。只有每个声明方法臂都达到
`beats_floor_holm`，`method_arms_beat_floor` 才为真；任一声明方法臂低于
地板即 `method_arm_does_not_beat_floor`，并给出独立 warning。原
`panel_reference_arm` / `panel_beats_simple_floor` 保留为逐点显示，但不再
单独承担方法主张。

### 27.2 地板不是常数：共享行重采样

`simple_baseline_fit` 现在为选中配置记录 `reported_nll_per_row` /
`reported_n_tokens_per_row`（与报告 PPL 同一评估行），审计核验其 token
总和与 NLL 总和。`methodology_audit` 和 `baseline_tune._run_final` 使用
`paired_row_bootstrap_delta`：对两条臂和所有共享 seed 重采样**同一批
行**，得到 `mean_s(PPL_a - PPL_b)` 的评估行 95% CI。该区间只覆盖评估行
抽样不确定性，不替代 seed 级精确符号翻转检验，也不伪装成总不确定性。
当记录缺少 per-row 记账时返回 `unverifiable`，绝不从别的样本造一个窄区间。

### 27.3 调参赢家的选择不确定性

`baseline_tune` 仍按最小 multi-seed 均值选配置，但记录
`selection_margin`、`selection_margin_se`、`selection_margin_in_se`、
`selection_decisive` 与 `runner_up`。审计从 trials 重新计算该 margin；
若赢家与亚军的差距小于一个标准误，给出 tie/uncertain warning。记录值与
重算值不一致是 problem（失败关闭），防止手改 artifact 把平局写成确定
最优。

### 27.4 因子覆盖与交互

`baseline_tuning_audit` 新增 `grid_factorial`：每个臂实际搜索的组合数、
完整笛卡尔积大小、覆盖率，以及每一对超参数的 `full_pairwise` / 缺失
组合。默认网格是 one-factor-at-a-time，因此审计会明确写出
`grid_interaction_untested`（如 `a:lr|weight_decay`），提示 OFAT 赢家
不等于联合最优；要么把搜索声明为 marginal/OFAT，要么补齐交互单元。

### 27.5 空 final seed 分区

`run_tuning(final=True)` / `_run_final` 现在拒绝空的 `final_seeds`。此前
空列表会落到 `L.run(..., seeds=[])` 的默认 seed 0，可能重新使用调参
draw，而 artifact 不记录这一替换；现在直接报错。

## 28. 主估计量、配对选择不确定性与跨规模种子身份（本轮硬化）

本节不新增 GPU 面板，只修正三处会让「配对」名不副实的数据流。

### 28.1 嵌套 holdout 的逐种子统计必须用主估计量

嵌套调参记录同时保存全验证集 `ppl` 与 held-out 行 `ppl_holdout`；
`ppl_estimation_rows='nested_validation_holdout'` 时，报告的主估计量是
后者。此前 `aggregate` 的 dPPL、`_print_pairs` 与
`methodology_audit` 的 `_paired_seed_deltas` 仍读原始 `ppl`，等于把
「用来选超参的行」重新放回配对差。现在统一走
`_record_primary_ppl` / `_paired_primary_delta`：

- 依次接受 `ppl_holdout`、`nll_sum_holdout/n_tokens_holdout`、
  `nll_per_row_holdout/n_tokens_per_row_holdout`；
- 声明了 nested 但三者全缺时返回 **None**（不可验证），绝不回退到
  全验证 `ppl`；
- `aggregate` 的 `ppl_mean/std/median/seed_t` 也改为按主估计量计算，并
  记录 `ppl_primary_estimand` 与 `ppl_primary_missing`。若同组有记录缺少
  holdout 记账，整组显式回退到全验证值并打印 warning，避免 `ppl_primary`
  与误差条描述两个样本；
- `build_report` 的 `paired_row`（P1T/P2S 等）与 P0W 逐种子差同样改用
  主估计量。

### 28.2 调参 margin 用配对差，不用两个边际 SE

调参 trials 在**同一组 search seeds** 上评估，赢家与亚军的差是配对差。
`tuning_selection_margin` 因此优先对共同 seed 的逐种子差取
`SD/sqrt(n)`；每种子同向的稳定优势不会被两个边际 SD 的 `hypot` 误判成
tie。只有 trial 缺 `seed_ppls` 时才回退到独立边际 SE，并把
`selection_margin_method` 落盘；审计按该方法核验
`selection_margin_se`，旧 artifact（无 method）可匹配任一实现，新
artifact 必须匹配自己声明的方法。

### 28.3 跨规模效应不能按位置配对

`_scale_points` 过去在缺 `seeds` 时把位置 `0..n-1` 当成 seed id。
`meta_analyze_effects` 只用点内重复，问题较小；但 `scaling_trend` 的
complete-seed 斜率与 `interaction_extremes` 会把不同规模的「第 0 个
effect」当成同一次训练随机抽样。现在每个点记录 `seeds_declared`：

- 任一点缺 seed 身份时，`scaling_trend` 的 paired path 与 per-seed slope
  全部撤回，`seed_pairing='unattested'`，只保留点级 WLS（并明确
  `primary_inference`）；
- 极端规模交互返回 `seed_identity_unattested`，不做位置配对；
- `meta_analyze_effects` 仍可做点内随机效应汇总，但输出
  `seed_pairing=declared|partial|positional_only`；
- `scale_interaction` 对未声明 seed 的调用标记 `seed_pairing='positional'`。

### 28.4 规模阶梯的种子计划核验

`scale_ladder_audit` 现在拒绝重复/负数 seed，并输出
`seed_set_shared`、`seed_set_intersection`、`n_common_seeds`。不同 tier
的额外 seed 不算对其它 tier 的重复；跨规模配对只能用交集，审计给出
warning 而不是把两套 seed 集合当成同一族。

### 28.5 符号翻转规划与 Holm 的边界一致

`signflip_min_consistency` 由 `< alpha` 改为与 `holm_adjust` /
`holm_min_attainable_p` 一致的 `<= alpha`（含 1e-15 浮点容差）。因此
「最小可达 p 恰好等于 alpha」的计划不再被错误地判为不可能确认、也不再
报零功效。

### 28.6 低功效规模点不得进入「共同效应」与趋势确认

- `meta_analyze_effects` 仍逐点报告随机效应，但只要任一参与合并的规模点
  `n < min_seeds`，`poolable=False`，并在 `underpowered_points` 与
  `note` 中逐点点名；pooled 数字只能作为描述性摘要。
- `scaling_trend` 的无配对（WLS point-mean）路径同样执行 `min_seeds`：
  任一点 `n < min_seeds` 即 `underpowered=True`，即使三点恰好落在一条
  直线上、WLS 的 p 为 0，也不得标为 `significant`。paired 路径沿用
  完整 seed 数的 `paired_effect_stats.underpowered`。

### 28.7 审计侧：指纹、行数字段与缺失测量的可见性

- `audit_leakage` 新增 c) 的真实实现：从 `train_ids_*.npy` /
  `test_ids_*.npy` 重新计算 `_ids_fp`，与 meta 的 `src_fp` 比较；并核对
  test meta 的 `fit_fp` 与 train meta 一致。缺任一侧的 fit 指纹按
  不可验证失败关闭，不再只比较 vocab。
- `audit_length` 要求同一 cell 的 `n_seq` 与 `len(ppl_per_seq)` 一致；
  显式 `n_seq` 非法时即使 per-document 向量给出合理长度也记为 problem。
- `audit_failures` 新增 `unmeasured_by_arm` / `unmeasured_samples`：
  NaN/缺失 PPL 的记录既不是成功也不是 error，必须作为「缺失测量」进入
  生存偏差读数，不能从两侧同时消失。

## 29. 规模证据的可比性：语料身份、数据缺失与对照配对（本轮硬化）

本节不新增 GPU 面板，也不修改冻结清单或 test split；它修正的是「规模
分析把两个不可比的东西放在一起」的数据流。

### 29.1 语料/分词器身份与训练预算分离

`data_version` 是**实际服务字节**的身份（含 `requested_train_tokens`、
`train_tokens`、`train_fp`、split 长度），因此 S1（5B）与 S2（12B）
在同一语料、同一分词器上也会不同。过去 M0 把它当作跨规模
`corpus_fp`，于是合法的 token/参数规模阶梯被判为 `mixed` 而拒绝。

现在每条记录另存预算无关的 `corpus_fp`：语料来源 + 词表宽度 +
**分词器拟合前缀**（前 `TOKENIZER_FIT_TOKENS` 个 train id，即 BPE 实际
拟合的文本）的指纹。训练 cap 增大时该前缀不变，所以共享语料/分词器的
不同预算 tier 得到相同 `corpus_fp`；换语料、换分词器或改变拟合前缀仍然
得到不同指纹。`data_version` 语义保持不变（精确服务字节）；旧记录没有
`corpus_fp` 时 M0 回退到 `data_version`，同一面板混用两代 provenance 则
拒绝，而不是让新子集代表整个面板。

### 29.2 零差值降低符号翻转的分辨率

精确符号翻转检验中，`d_i = 0` 的符号不改变统计量，因此双侧 p 的分辨率
下限是 `2 / 2**(#nonzero)`，不是 `2 / 2**n`。p 值原本就是正确的；错的
是 `paired_effect_stats` 报告的 `p_floor` 与 `resolution_limited` 标签。
现在 `exact_signflip_p` 返回 `n_nonzero` 与可达下限，paired 统计、
`reportable_claim`、`build_report` 的本地孪生实现以及 v7–v12 共用的
`exact_sign_permutation` 都使用它；全零向量报 `p_floor=1`。P5T 的
`p_floor_effective` / `min_attainable_p_holm_effective` 也按实际非零差
计算，而不是按配对数计算。

### 29.3 规模点的可见缺失与零方差

规模点过去只汇报过滤后的 `n`。`_scale_points` 现在接受
`n_planned` / `planned_seeds`，并记录 `n_input`、`n_dropped`、
`n_missing`、`n_unexpected`、`complete_pairs`。`meta_analyze_effects`
在任一规模点不完整时禁止 `poolable=True`；`scaling_trend` 返回
`incomplete_scale_point` 且不标 `significant`。没有声明计划的旧调用保持
原行为（`complete_pairs=True`）。

同一规模内逐种子效应完全相同的点（within-scale variance = 0）无法形成
`1/se^2` 权重。它不是缺失点，也不应被静默丢弃；现在它保留在
`per_point`，列入 `zero_variance_points`，并阻止合并声明，直到有方差
模型或更多种子。

`n=1` 或全部效应非有限的规模点无法进入合并权重，但也不能因此从族里
消失：前者进入 `underpowered_points`（即使它不在 `usable` 里），后者
进入 `empty_points`；两者都阻止 `poolable`。声明过的 tier 因此不会因为
「算不出 SE」而被静默删除。

### 29.4 跨规模逐种子配对必须过受控变量门

M0 的跨规模 log-PPL 效应过去只按 `(variant, seed)` 取 PPL。种子编号相同
不代表两次训练共享优化器、步数、数据或初始化。现在每个 seed 先过
`exp_lib.pair_reason`（run_cfg / train_knobs / 精度模式 / 数据与评测
指纹 / `init_core_fp` / steps/tokens）；被拒的 seed 会打印并排除，不计入
规模点。极端规模交互按声明的 `scale` 排序，不再按标签字符串排序
（否则 `S10` 会排在 `S2` 之前）。

### 29.5 规模阶梯的功效是「声明族 + 共同种子」的属性

`scale_ladder_audit` 过去用原始符号翻转下限 `2/2**n` 标记每个 tier 的
`descriptive_only`，忽略了预注册族大小。n=6 的 tier 能到 0.031，但面对
`family_m=3` 的 Holm 族最小可达 p 是 0.094，仍然无法确认任何比较。现在
每个 tier 同时报告 `p_floor_family_holm` / `descriptive_only_basis`，并按
族下限判定 `descriptive_only`；`family_min_seeds` 直接来自
`signflip_min_n_for_holm(family_m)`。

跨 tier 的配对对比只能用种子集合的交集，而不是任一层自己的 n。审计新增
`n_common_seeds`、`common_seed_paired_p_floor`、
`common_seed_min_attainable_p_holm` 与
`common_seeds_sufficient_for_family`；共同种子不足以达到族下限时给出
warning，并明确「某一层多跑的 seed 不能修补另一层缺的 seed」。这样在
GPU 时间被消耗前就能看出阶梯是否具备可确认的配对证据。

M0 的跨规模配对同时改为读取记录的**主估计量**：嵌套调参记录声明
`ppl_estimation_rows='nested_validation_holdout'` 时，逐种子值取
`ppl_holdout`（或 holdout NLL 记账），不再读仍包含选参行的全验证
`ppl`；声明嵌套却缺 holdout 记账的记录按「未测量」处理，而不是静默回退
到全验证值。这样规模效应与面板汇总/报告使用的是同一个样本。

### 29.6 共享种子的跨规模合并不能假装各点是独立研究

`meta_analyze_effects` 的 DerSimonian-Laird 合并 CI 把每个规模点当作独立
研究。当各 tier 使用同一组训练 seed 时，逐点效应是配对抽样而非独立抽样，
这个 CI 可能偏窄。现在输出 `seed_dependence`（`shared_seeds_paired` /
`shared_seeds_unpaired` / `unattested`）并在 note 中点明独立性假设；当所有
点都声明 seed 身份且存在跨全部点的共同 seed 时，额外给出
`paired_common_effect`：对每个共同 seed 取跨规模效应的等权平均，再做精确
符号翻转检验。它是依赖感知的敏感性读数，不替代异质性检验，也不改变
DL 合并值的定义。这样「大模型上效应更大/更小」的合并区间不会被误读为
独立重复实验的精度。

### 29.7 交互检验的 `significant` 必须接受调用方的最低 n 门槛

`scale_interaction` 的底层 `paired_effect_stats.significant` 只回答
「p<alpha 且效应大于种子噪声」，不知道调用方声明的 `min_n`。因此 n=6、
`min_n=8` 时它会返回 `significant=True`，与同一对象的
`underpowered=True` 及 `claim_level` 文本矛盾。现在保留原始检验旗标
`significant_raw`，但交互层 `significant` 同时要求未 underpowered 且未
resolution-limited；调用方给出的最低 n 门槛不能再只停留在说明文字里。

### 29.8 开放块主族的族下限也必须使用非零差值计数

`open_block_supp._family_holm_floor` 过去只接收各主对比的配对 `n`。零差值
不改变符号翻转 p，却降低可达分辨率；当某主对比有大量 ties 时，按 `n`
计算会把族下限说得比实际更细。现在 helper 同时接受旧式整数与结果字典：
字典形式优先使用 `n_nonzero`（全零映射为 1，下限 1），越界/缺失才回退
`n`。`analyse` 改为传入完整主对比结果，而不是只传 `n` 列表。

## 30. M0 规模分析由声明驱动，不再写死 S1/S2（本轮硬化）

`m0_supp.scale_analysis` 过去在三个位置硬编码 S1/S2：默认 summary 发现、
跨规模点的计划种子数、预算阶梯审计。以后增加 S3/S4 时，新 tier 会从
「计划 vs 实际」检查中静默消失。现在从 `exp_lib.SCALE_LADDER` 的 tier
名发现对应的 `RUN_<TIER>` 配置（不会误收 `RUN_SCALE` / `RUN_LONG` /
`RUN_ABL` 这些早期探索面板），并允许调用方显式补充 tier。

每个跨规模点同时带 `n_planned` 与 `planned_seeds`：前者用于旧记录/外部
tier 的计数降级，后者让 `_scale_points` 能区分「计划中的 seed 缺失」与
「计划外的多余 seed」。计划 seed 先去重再计数，重复声明不会伪造更大的 n。
分析输出的 notes 也按实际 tier/n 生成，不再写死「S1 4 seeds / S2 3
seeds」。

## 31. 行重采样 bootstrap 的覆盖率/抽数/种子不可静默替换（本轮硬化）

`paired_row_bootstrap_delta` 过去把 `conf` 原样写进输出，却在内部校验失败
时回退到 95% 计算；传入 `conf=1.0` 时会得到一个 95% 区间却标成 `conf=1.0`。
现在 `conf` 必须是 (0,1) 内的实数，`n_boot` 必须是 >=20 的整数，`seed`
必须是整数；任一非法就返回 `status='unverifiable'` 并给出 `reason`，不再
生成区间。覆盖率、抽数和 RNG 种子都是区间身份的一部分，不能为了「有输出」
而替换成另一个区间。

## 32. 调参选择不确定性的种子去重（本轮硬化）

`tuning_selection_margin` / `tuning_selection_stability` 从 `seed_ppls` 取
配对搜索种子。JSON 可以同时出现 `"0"` 与 `0`，旧实现后写覆盖前写，把一个
重复抽样静默折叠成一个观测，进而低估配对 SE。现在若同一 seed 以两种写法
出现，margin 返回 `method='invalid_seed_ppls'` 且不给 SE/判定（不回退到
独立边际 SE），stability 返回 `status='unavailable'` 并说明重复 seed。
缺失 `seed_ppls` 的旧 artifact 仍走带标签的独立 SE 回退；只有「有 per-seed
证据但证据非法」才失败关闭。

交互层 `significant` 同时要求 `complete_pairs is not False`：只要有一个 seed
因非有限配对被丢弃，即使剩余非零对达到 p<alpha，也不能把 complete-case
子集的结果当作完整面板的交互结论；原始检验旗标仍在 `significant_raw`。

## 33. 方法审计的臂均值/相对效应也必须用主估计量（本轮硬化）

`methodology_audit` 的配对差已经走 `_paired_primary_delta`，但同一对比的
`with_mean`/`without_mean`、`relative_effect_pct` 与 unpaired bootstrap 仍读
原始 `ppl`。嵌套调参记录的全验证 `ppl` 仍包含选参行，因此会出现「配对 Δ
用 holdout，分母却用全验证」的混合口径。现在这些量统一读
`_record_primary_ppl`；`arm_summary` 也按主估计量计算 `ppl_mean`/`ppl_std`
并记录 `ppl_estimand`、`ppl_primary_missing`。若同一臂中有记录声明嵌套却
缺 holdout 记账，则整臂显式回退到全验证值并记 warning（与 aggregate 同一
约定），绝不把两个样本混在一个均值里。

## 34. 规模点 effects 字段非法时不得静默删除 tier（本轮硬化）

`_scale_points` 过去对非可迭代的 `effects`（例如误传一个标量）执行
`continue`，该 tier 就从族里消失，剩余点看起来像完整阶梯。`effects=None`
仍是「已声明的空测量」，但数字/布尔/字符串/bytes 等非法形式现在直接抛
`ValueError`；调用方必须给出测量向量或显式 `None`。这样规模分析不会因为
一个格式错误而偷偷改变族大小或「共同效应」的可见性。

同一约定也应用到 `_combined_seed_doc_ci`（seed × document 两层区间）：
`conf` 必须是 (0,1) 实数、`n_boot` 是 >=20 的整数、`seed` 是整数；否则
返回 `status='unverifiable'` 与 `reason`，不再构造一个标签与实际覆盖率
不一致的区间。

## 35. 二分类阈值、校准与 F1 聚合口径（本轮硬化）

### 35.1 F1 不是无阈值指标

`f1_positive` / `precision` / `recall` / `balanced_accuracy` 都依赖一个决策
阈值。旧的 `classification_metrics` 在只有分数时隐式使用 0.5，却没有把这个
约定写进记录：一个排序很好但正类分数整体低于 0.5 的模型会被读成弱模型；
更危险的是，如果阈值是在评测行上调出来的，审计看不到任何痕迹。现在：

- `classification_metrics(..., threshold=...)` 显式接收阈值，并输出
  `decision_threshold` / `decision_threshold_source`（`default_0.5`、
  `caller_threshold`、`caller_predictions`、`argmax` 或
  `validation_selection`）。默认 0.5 仍是报告约定，但不再隐式。
- `binary_threshold_curve` 给出每个候选切点的 TP/FP/FN/TN、precision、
  recall、F1、specificity、balanced accuracy 与 Youden J；
  `select_binary_threshold` 返回选定切点及其相对 0.5 的
  `threshold_sensitivity`。两者都明确要求：切点只能在**与评测行不相交的
  验证行**上选择。
- 扫描按分数排序后用累积计数一次完成，复杂度 O(n log n + k)，并支持
  `max_thresholds` 上限与 `threshold_grid_truncated` 标记，避免在
  B-tier 规模上把每个候选阈值都重扫全样本。
- `metric_policy_audit` 对二分类记录检查：阈值必须在 [0,1]；来源必须是
  声明集合之一；`validation_selection` 必须同时声明
  `threshold_selection_split="validation"` 与
  `threshold_selection_metric`；评测/测试行上的阈值选择按泄漏拒绝。
  报告正类 F1 却没有 `positive_label` 时也无法识别正类，记为不可复核。
- 在 1:100 以上的不平衡（少数类 < 1e-2）仍使用 0.5 默认切点时给出
  warning：这不是错误，但必须同时报告 PR-AUC，且不能把一个未调优的切点
  当成模型上限。1:1000 以上仍按条件主指标切换为 PR-AUC。

### 35.2 排序好不等于概率可用

二分类分数若落在 [0,1]，`classification_metrics` 额外给出 Brier score 与
等宽分箱 ECE（`calibration` 中带每个非空箱的 confidence/accuracy/gap）。
它们进入 `binary_classification` 的 secondary 端点与记录级范围审计。这样
「PR-AUC 提升」不会被误读成「概率可用于阈值决策」；分数若不在 [0,1]，
校准返回 `status="unavailable"` 而不是硬算一个没有概率含义的 Brier。
ROC-AUC 低于 0.5 会显式提示检查正类列方向，而不是直接把模型判为弱。

### 35.3 macro / micro / weighted 必须由 per-class 计数复现

`classification_metrics` 的 `per_class` 现在带 `tp`/`fp`/`fn`，并输出
`precision_micro` / `recall_micro`。`metric_policy_audit` 在计数齐全时：

- 由每个类的 TP/FP/FN 重算 precision/recall/F1，和记录值不一致即失败；
- 由计数总和重算 micro precision/recall/F1，和 `f1_micro` 等字段不一致即
  失败；
- 由正确计数比例重算 accuracy；
- 继续由 per-class F1 与 support 复现 macro / weighted F1。

这使「多分类 F1 用的是 macro、micro 还是 weighted」不只是文字声明，而是
可由记录本身验证的口径。

### 35.4 边界 F1 的 macro 与 micro 是两件事

边界对齐过去只落盘逐序列 precision/recall/F1 的均值（macro over
sequences），审计却要求
`bnd_f1 == 2*bnd_prec*bnd_rec/(bnd_prec+bnd_rec)`。该恒等式只对 micro
（池化 TP/FP/FN）成立，对 macro 不成立：均值 F1 一般不等于均值
precision/recall 的调和平均。现在 `boundary_alignment` 可选返回逐序列
TP/FP/FN 计数，`compression_report` 同时落盘：

- `bnd_prec` / `bnd_rec` / `bnd_f1`：逐序列 macro 口径，并声明
  `bnd_f1_average="macro_over_sequences"`；
- `bnd_prec_micro` / `bnd_rec_micro` / `bnd_f1_micro` 与池化
  `bnd_tp`/`bnd_fp`/`bnd_fn`：可由计数直接复现的 micro 口径。

审计只对 micro 声明强制调和关系；macro 记录若没有 micro 计数，会明确标为
`macro_f1_reconstructible=False`，不再用一个错误的恒等式把正确读数判错，
也不假装能离线复现它。

### 35.5 多分类也要有 threshold-free 排序口径

多分类记录过去只有 accuracy 与 macro/micro/weighted F1。F1 依赖决策规则；
当某个类只占 1:1000 时，micro-F1 会被多数类主导，macro-F1 又无法说明排序
质量。现在 `classification_metrics` 对 3 类以上的列式分数矩阵额外给出
one-vs-rest 的 `pr_auc_macro` / `pr_auc_micro` / `pr_auc_weighted` 与
`roc_auc_macro` / `roc_auc_micro` / `roc_auc_weighted`，并把每类的
PR-AUC/ROC-AUC 放进 `per_class`：

- **macro**：只在 one-vs-rest 端点有定义的类上等权平均（缺失类列入
  `undefined_classes`，不伪造 0/1 分）；
- **weighted**：按类 support 加权；
- **micro**：把所有 one-vs-rest 决策展平后池化，会被多数类主导。

`macro`、`micro`、`weighted` 在类别不平衡时可能给出不同排名，因此记录必须
同时声明 `ranking_averaging="one-vs-rest"`。`metric_policy_audit` 现在也对
多分类记录取最小类 prevalence；若低于 1:1000，条件主指标切换为
`pr_auc_macro`，只报 F1 或只报 ROC-AUC 会被标为不可作为主结论。per-class
值与 support 存在时，macro/weighted 聚合必须能被 per-class 值复现，手改
聚合值会被审计拒绝。`METRIC_POLICY['multiclass_classification']` 同时记录
`macro_micro_f1_gap`，让 macro 与 micro 的分歧显式可见。one-vs-rest PR-AUC
还记录其非零 chance floor：`pr_auc_macro_chance`（有定义类的平均 prevalence）
与 `pr_auc_weighted_chance`（support 加权），以及对应的 lift；在 1:1000
数据上 PR-AUC=0.01 附近可能只是 chance，不能直接当作有技能。若输入是
行归一化的概率矩阵，还记录多分类 Brier（sum-of-squares）与 top-label ECE；
logits 会返回 `calibration.status="unavailable"`，不会把 logit 当概率硬算。

### 35.6 分类指标必须带 trivial floor

只报 accuracy/F1 而不报「只猜多数类」的分数，无法判断模型是否真的学到了
信号；1:99 不平衡时多数类分类器的 accuracy 已经是 0.99。现在
`classification_metrics` 默认把 `trivial_baseline` 写入记录：
`strategy="most_frequent"`（并列时按声明 label 顺序取第一个）、该类的
accuracy、balanced accuracy 与 macro/micro/weighted F1。它在同一 label
空间上计算，不读取分数或预测结果，因此不能通过调阈值「变强」。审计在
per-class support 存在时会重算该 floor；若记录把 floor 的 label 改成不是
最多数的类，或聚合值无法由 support 复现，按不可复核处理。

## 36. 检索端点的平凡基线、主端点检验与文档组行折（本轮硬化）

本轮不新增 GPU 面板；改动都发生在零 GPU 的指标/统计/调参路径。

### 36.1 NIAH 不能只报 top-1 accuracy

NIAH-natural 是多分类检索端点（8k 词表中的答案 token）。只报 micro top-1
accuracy 无法区分「真的检索到了远端信息」与「答案类别频率/运气」：
高频答案会主导 micro 指标，稀有答案的错误被平均掉。`retrieval_metrics`
现在对同一批 query 同时记录：

- `n_correct` 与 `acc`（top-1，micro over queries）；
- `balanced_accuracy` / `macro_recall`（每个实际出现的答案类别等权）；
- `chance_accuracy = 1/n_observed_answer_classes`（均匀猜）；
- `majority_accuracy`（总是猜出现最多的答案 token）与对应 lift；
- `answer_support` / `per_class_recall`，供审计重算。

`v10_supp.py` 的 NIAH cell 记录写入这些字段，报告表并列展示 micro acc、
macro-recall、uniform chance 与 majority floor。`metric_policy_audit` 对
新记录重算 `acc = n_correct/n_queries`、chance/majority floor 与 macro
recall；手改 accuracy 或 floor 会失败。没有这些字段的旧记录被列为
`legacy_without_trivial_floor` 并给 warning，不能当作「已经证明有技能」。

### 36.2 检索的主端点必须真的被检验

之前 NIAH 的配对统计只检验 answer NLL，但 `METRIC_POLICY` 声明的主端点是
top-1 accuracy；这会出现「NLL 显著、accuracy 无差异」却被写成检索收益的
错位。`v10_supp.py stats` 现在同时构造逐种子配对 accuracy 对比
（`accuracy_contrasts`），并与 NLL 对比一起进入同一个 Holm 族；报告分别
列出两种端点，避免用次级 NLL 代替主端点结论。

### 36.3 嵌套验证行折必须是文档组感知的

固定长度的验证行可以跨文档边界；只按行切分仍可能把同一篇文档的一部分
放在 selection fold、另一部分放在 evaluation fold。缓存中已有逐 token 的
`val_doc_ids` sidecar，`cached_validation_row_groups` 将其折叠为每行一个
文档组，并把任何共享文档的行 union 成同一组；`baseline_tune` 的
tuning/final/simple 三条路径都通过 `_nested_row_split` 把这个组向量传给
`validation_row_split`。sidecar 缺失时显式记录 `groups_aware=False` 与
原因，不冒充文档不相交；sidecar 存在但长度/指纹不一致时 fail closed。
`row_split` 元数据会保留 `groups_aware`、`n_groups` 与来源，审计仍可复现
行指纹。该实现覆盖 WikiText 主研究；FineWeb 若需要组感知切分，调用方应
显式提供组向量。

## 37. 单因素设计契约：在定义点核验每条对照的因素结构（本轮硬化）

### 37.1 问题：记录级门看不见「因素」本身

评审三/四给对照配对加了一系列**记录级**门：`pair_reason`（run_cfg 逐字节
相等）、`train_knobs`（优化器/调度/裁剪/增广全描述符）、`hyperparams`
（结构超参）、`init_core_fp`（匹配初始化）、`train_objective`（实际训练目
标）、`_param_control`（参数量对齐）、`_compute_control_detail`（算力对
齐）。这些门都核验「两个臂**共享**哪些东西」；而一条对照之所以是对照，
恰恰因为它的两个臂**允许**差一个因素——那个差异（AttnCfg 的键）没有任何
门在看。若某天一条消融对照的臂定义被多改了一个旋钮（例如顺手改了
`index_topk` 或 `fuse_kernel`），所有记录级门都会照常放行：那个多出来的
旋钮本身就成了对照的「因素」，报告却仍按原声明解读。反向漂移同样成立：
声明的干预键若被改得不再差异（干预消失），对照变成两个同构臂，也无人报错。

### 37.2 契约：每条对照声明它允许差的键集

`exp_lib.ABLATION_INTERVENTION_KEYS` / `BASELINE_INTERVENTION_KEYS` 为
`ABLATION_DESIGN` / `BASELINE_DESIGN` 的每条对照显式声明「允许差」的
AttnCfg 键集（`CONTRAST_OUTSIDE_CFG_FACTORS` 另行声明 cfg 之外的带外因
素，如稠密基线的 mlp 加宽参数对齐，仍由记录级 `_param_control` 把关）。
`exp_lib.single_factor_design_audit` 逐层 diff 两个臂的实际配置并核验：

- **实际差集必须恰好等于声明键集**：多一个键 = 第二个因素混进单因素对照
  （refuse）；少一个键 = 干预消失（refuse）；两侧漂移都在跑 GPU 之前被抓。
- **缺声明即 fail-closed**：往设计里加对照而不声明键集，本身就是错误——
  没有默认放行。
- **分级**：`kind` 出现在差集 = 换注意力算子（另有 mlp 加宽/投影/sink 语
  义），标记 `architecture_level`，永远不冒充 knob 级单因素；其余为
  `single_factor`。方法 vs 参数对齐稠密基线因此是 architecture_level 而
  不是单因素，与 §22.4 的 objective 门结论一致。

契约同时把两处此前**未声明**的 HCA 动态臂差异写进明面：动态 HCA 层的
边界门以 `comp_lambda_mult=4` 训练（comp_lambda_effective 0.05→0.2），以及
`block_size` 在动态分段下的 inert 差（`blocks_from_cuts` 不读它）。这两条
差此前只在代码与 `train_objective` 读数里可见、未进对照声明；现在它们是
声明键集的一部分，且 `ABLATION_DESIGN` 的对照注释与
`CONTRAST_OUTSIDE_CFG_FACTORS` 都写明其语义。dynamic-vs-fixed 的对照因此
继续保持 §22.4 的 system-level 读数——契约的职责是不漏报，不是放宽。

### 37.3 冻结清单：`mechanism_claim_allowed` 从手写断言变成可核验属性

v12 的清单为每条 primary comparison 手写 `mechanism_claim_allowed` 与
`interpretation_caveat`；此前没有任何机器检查把标志与臂定义对账。现在
`v12_supp.py plan`（零 GPU，读不到任何 test 数据）为每条比较计算：

- attention-cfg diff 键集是否落在**一个**声明的干预族内
  （`exp_lib.cfg_keys_single_factor`；`kind` 互换按定义不属于任何族）；
- 两臂由已构建模型读出的**实际**训练目标是否一致（`train_objective`，
  同一记录级口径）；
- 参数总量是否在 1% 对齐内（既有检查）。

三者同时成立才判 `single_factor=True`。**标志为 True 而判非单因素 =>
plan 直接 refuse**（机制主张不允许挂在多因素比较上）；标志为 False 而判
单因素 => 保持保守标志并记 note（caveat 可依赖 cfg 之外的约束，如
allblocks 读严格更多信息、等读预算不成立——保守方向永远不是错误）。当前
清单核验结果：`randidx_seeded`（唯一 True 标志）确为单因素
（`indexer_mode` + 逐 (seed,layer) 抽取的 `random_salt`，后者是随机抽取的
定义而非第二干预）；main/sink-matched/no-sink/aligned-dense 四条 kind 互换
与 allblocks 的保守 False 全部一致。`audit_gates.py single_factor` 子命令
给出同一口径的 cfg 级读数（不建模型，零依赖），并汇入 `report` 汇总审计。

### 37.4 扫描对照与开放块矩阵

- v9/v10 的 seq2k 扫描配对检验新增因素列（`v9.SEQ2K_CONTRAST_FACTORS`）：
  `topk8 - topk512` 是单因素剂量响应（只差 `index_topk`）；两条 m1 对照同
  时改 block_size(4→1)/overlap(128→0) 与 topk 语义（逐 token 读取）——等
  token 预算但**多因素块几何束**，只能读作 sweep 点之间的系统级比较，报告
  明示不得读作 topk 剂量响应。
- `exp_lib.open_block_matrix_audit` 核验开放块 2x2 矩阵（§24 声明的
  max_block × 读取规则）只沿两根声明轴变化：同 cap 的 mode 对照只差
  `open_block_mode`、同 mode 的 cap 对照只差 `max_block`、scope 对照只差
  `open_block_scope`；任何第三键泄漏进矩阵内 diff 都会让交互统计读到未声
  明因素而 refuse。

本节全部改动为零 GPU 的定义点/清单门与标注：不改任何臂的测量语义
（`CODE_SEMANTICS` 不变）、不改冻结清单字节、不动任何 results_*。

## 38. 多分类 F1 口径与地板假设（本轮设计硬化）

本轮不新增 GPU 面板，只补齐两处「指标必须对齐研究目标、比较必须有
预注册假设」的缺口。`CODE_SEMANTICS` 与冻结清单均未改动。

### 38.1 NIAH 的 macro/micro/weighted F1

NIAH-natural 是单标签多分类检索端点。此前它报告 top-1 accuracy、
macro recall（balanced accuracy）、均匀/多数类地板和 answer NLL，但
没有 F1；一个多分类指标不注明 macro/micro/weighted 无法复核。现在
`retrieval_metrics` 从同一批 top-1 标签计算每类 TP/FP/FN，并同时返回：

- `f1_macro`：只在观测到的答案类上等权平均；
- `f1_micro`：池化 TP/FP/FN，单标签任务下恒等于 top-1 accuracy；
- `f1_weighted`：按答案支持度加权。

`per_class` 增加 `predicted`/`precision`/`f1`，`metric_policy_audit`
对 NIAH 记录重算三个聚合（有 per-class 计数时），并强制
`f1_micro == acc`。缺少 F1 的旧记录标为 legacy（不是通过）；声称 F1
却没有 per-class 计数的记录标为不可复核。PR-AUC/ROC-AUC 仍不伪造：
top-1 记录没有每类分数，硬造 one-vs-rest 分数会改变被测对象。
`v10_supp.py` 的 NIAH 记录与报表同步持久化并展示 macro/weighted F1。

### 38.2 「战胜调参过的计数地板」是独立预注册假设

`FLOOR_HYPOTHESIS_DESIGN.method_beats_tuned_count_floor` 把最基本的
比较写成可审计假设：Δ = 方法臂 PPL − 地板 PPL，`predicted_sign=-1`，
主指标 `ppl`（`lm_perplexity`），falsifier 是任一声明方法臂不低、未过
Holm、或行/token 分母不可比。`methodology_audit` 把该假设与
`method_arm_floor_status` 绑定，输出 `observed_status`
（supported/falsified/unresolved/unverified）、`observed_sign` 与
`direction_consistent`；正向显著不再能被自动读成支持声明机制。
`audit_gates.hypotheses` 也覆盖该假设族。比较仍需先通过
§26/§27 的可比性、token 分母与 Holm 门槛。

## 39. 设计前馈：预算台账、证据等级与动态分块假设（本轮方法硬化）

本轮不读 test、不新增 GPU 结果；只把「实验开始前就必须回答的设计问题」
变成零 GPU、可拒绝运行的机器合约。`CODE_SEMANTICS` 与
`confirmation_frozen.json` 均未改动；动态/HCA 仍只是 draft，不能据此
产生确认性主张。

### 39.1 token 预算不等于 train-cache 大小

`confirmation_frozen.json` 的 `binding.training_recipe_signature.nt` 是
**训练缓存被服务的 token 数**（8M），而一次 run 实际消耗的是
`steps*bs*seq_len = 3000*12*512 = 18.432M`（约 2.3 个 epoch）。把二者
当成同一个「token 预算」会把冻结清单误判成超预算。新的
`exp_lib.comparison_budget_ledger` 同时记录
`train_cache_tokens`、`consumed_tokens`、`epochs_over_cache` 与逐臂
`consumed_tokens`，并要求同一对照的两臂**消耗 token 完全相同、缓存相同**。
任何 per-arm 的 steps/bs/seq 覆盖都会让共享预算失效并拒绝。

### 39.2 预算控制是三维的：token、参数、读取/计算

仅说「parameter-matched dense」不够：dense 可能通过加宽 MLP 匹配总量，
但训练 FLOPs、注意力读取比例和 kernel 仍不同；反过来，稀疏臂读的
block 少也不等于参数少。台账对每条对照分别输出：

- `token_matched`：`consumed_tokens` 与 `train_cache_tokens` 相同；
- `param_matched`：真实建出的 `SmallGPT` 参数总量在 1% 内；
- `read_matched`：按 `kind`/`block_size`/`target_block_tokens`/`index_topk`
  解析出的每 query 读取 token 数在 10% 内（动态臂标记为 estimate）；
- `compute_proxy_matched`：当前以参数×token 的 6ND 代理为准（显式声明
  它不是实测 FLOPs）；
- `dedicated_control`：selection / segmentation / task-policy / sink /
  parameter-proxy / allblocks / multi-factor。

只有 `mechanism_claim_allowed=true` 同时满足 token、参数、读取预算匹配，
且 cfg diff 属于一个已声明单因素族、并存在专用对照时，台账才放行。
`csa_fixed_rope_randidx_seeded` 因此可以支撑「学习式选择 vs 同读取预算
的随机选择」；主对比（vs `full_rope`）仍是 system-level：参数匹配但读取/
FLOPs/kernel 不匹配，`mechanism_claim_allowed` 必须保持 false。
`csa_fixed_rope_allblocks` 被明确标为 information-rich all-block control，
不是等预算对照。

### 39.3 证据等级是代码属性，不是报告措辞

`exp_lib.EVIDENCE_LEVELS` 定义 confirmatory / exploratory / descriptive /
draft 四级；`EXPLORATORY_PANEL_REGISTRY` 把 v6-v11（以及 open-block、
scale 等在同一 validation split 上选择/比较出来的面板）登记为
exploratory。`evidence_level_audit` 的硬规则：

- `status=draft` 或 `frozen_at=null` 的清单永远不能声明 confirmatory；
- 单条 comparison 不能靠改 label 越过 manifest 等级；
- confirmatory manifest 必须同时有 `require_power_gate`、
  `require_complete_coverage`、非空 `scope.confirmatory_claims` 与冻结时间；
- registry 里的结果目录若在 summary 里写 `confirmatory/confirmed`，直接 fail。

因此「v6-v11 的 p 值更小」不是升级理由；升级只能来自新的冻结清单 +
disjoint test + 完整覆盖 + test 前功效门槛。

### 39.4 动态分块必须按假设拆开，而不是只跟 dense 比

动态分块相对固定分块的增益至少有三种互斥解释，draft 现在把它们写成三条
可证伪假设：

- **H1 切分质量**：`metric_policy=boundary_alignment`，主端点
  `boundary_f1.tol1@4096`；关键是 **random-vs-random** 对照——两臂都用
  seeded random selector，只有 fixed/dynamic 边界不同，才能把「边界」从
  「学习式选择」里分离出来。
- **H2 检索精度**：`niah_retrieval`，主端点 `niah.top1_accuracy@L4096`；
  对照是同一动态分块下的 learned top-k vs seeded random top-k，读取预算
  相同，并同时报告 uniform/majority 平凡地板。
- **H3 长程覆盖**：`lm_perplexity`，主端点 `by_len.ppl@4096`；必须与
  **读取预算匹配的固定臂**比，而不是只跟 dense/参数代理比。动态块长度是
  数据相关的，所以 draft 要求先在 validation 上测出 realized mean block
  length，再按 `k_fixed = round(topk_dynamic * mean_dynamic_block_tokens /
  fixed_block_size)` 训练具体的 `csa_fixed_rope_topk{k_fixed}`；在测量前
  `topk32` 只是候选，不能写成 read-matched。H3 现在有**专门的
  `by_len.ppl` 动态 vs read-matched fixed 对照**，不再只靠「动态 vs 固定」
  或「动态 vs dense」；审计要求这条 PPL 对照存在。
- H4 task-learned boundary 与 H5 HCA hierarchy 作为独立假设记录；HCA 读
  每个 closed block，因此被明确标成 system-level，不得当作等读取机制对照。

每条 comparison 还必须显式声明 `evidence_level`，并通过
`hypothesis_id` 绑定到假设的 `endpoint`、`predicted_sign` 与
`contrast_arms`（谁减谁、方向是否反转）。H3/H5 标为
`mechanism_claim_eligible=false`：即使预算恰好匹配，也不能被引用为边界/
选择机制证据。新增一条对照却没有声明等级、端点漂移、方向反转写错或假设
没有对应比较，都会在设计阶段被拒绝。

### 39.5 功效与读取校准是冻结前的同一道门

`statistics.p_floor`、`min_attainable_p_holm`、`min_seeds_for_confirmation`
与 `power_first_holm_gate` 不再只是散文：`planned_power_audit` 用 exact
sign-flip / Holm 公式重算并逐项对账。`min_power_for_test=0.8` 还必须真的
被 `signflip_power(n, m, q, alpha)` 达到；只写 `power_gate.status=passed`
而没有数值 power，或把 q 降到 0.90 使投影功效 0.789 < 0.8，都会 fail。

动态读取校准由 `dynamic_read_budget_calibration` 验证：`status=measured`
必须带 realized mean block length、`fixed_block_size`、`topk_dynamic` 和
公式算出的 `matched_fixed_topk`，且 manifest 必须真的包含
`csa_fixed_rope_topk{k_fixed}`。pending 状态只能存在于 draft；非 draft
清单的 `promotion_ready` 需要 measured 校准 + passed power gate + 完整
scope 声明同时成立。

### 39.6 入口

```bash
python audit_gates.py design          # 主清单 + 动态 draft 的预算/证据/假设审计
python v12_supp.py plan               # 冻结清单 preflight（现在也包含预算台账与证据等级）
python v12_supp.py dynamic-plan       # 动态 draft 的设计 preflight（不读 test）
python test_review_fixes.py           # L82-L97 覆盖上述合约
```

新增对照前先过 `design`：它会指出该对照属于 token/parameter/read 哪一维
不匹配、缺哪个专用臂、以及是否被错误地标成 confirmatory。

## 40. 值身份、传输对拍、聚合隔离与运行态对账（本轮硬化）

### 40.1 纯函数记忆化必须绑定当前值

`_ids_fp` 的 memo 键是 `(数据指针, shape, strides, dtype)`，它标识的是
**数组对象**而不是内容。对一个可写数组做原地清理后，旧实现仍返回写入前
的摘要；该摘要会进入 `run_cfg`、manifest 与 resume 判定，等于把两份不同
语料当成同一身份。现在只有 `flags.writeable=False` 的数组才允许命中
memo 或 `.idsfp.json` sidecar；可写数组每次重新哈希。生产语料是只读
memmap，因此快速路径不变。测试同时核对：原地改值必变、只读/可写副本与
不同整数宽度（uint16/int32/int64）在值相同时摘要相同。

### 40.2 传输层批量化要有独立对拍

`batch_contract_check` 过去只检查 shape、标签右移、同种子重复与词表范围；
一个错误的 gather 仍可能“右移且可重复”。现在它从**同一组 seeded starts**
另建一条最朴素的 int64 NumPy 因果窗口（不 pin、不 prefetch、不窄化索引），
逐 token 比较真实 `batch_iter` 的输出，并在批次记录中保存
`transport_matches_int64_reference`。这样 int32 索引 / pinned host /
预取线程的“只改传输、不改值”声明是被测出来的，而不是注释里的假设。

### 40.3 聚合的容错边界必须显式

`aggregate` 仍然 fail-closed（重复 seed、混语料/混评测切片直接拒绝）。
但一个已经训练完的面板不能因为一条损坏记录而完全没有聚合输出。
`aggregate_with_exclusions` 逐条校验，命名并计数所有隔离项：

- 非对象、无有效 PPL、合成重建、非整数 seed 的记录剔除；
- 同一 `(variant, protocol, seed)` 只保留首个可测记录，后者作为歧义重复
  剔除（不覆盖、不静默合并）；
- 同一组内语料/评测指纹不同时，**按指纹类别拆分**输出键，绝不跨切片平均；
- 仍然触发跨协议配对错误时，按臂分别聚合（保留边际读数，不生成 dPPL）。

`_dump_aggregate` 总会写 `aggregation_exclusions.json`，有隔离项时另写
`summary.aggregatable.json`。报告的 `n_input` / `n_admitted` /
`n_excluded` / `reason_counts` / `excluded` 就是容错边界；没有记录被当成
0 或凭空补一个 seed。

### 40.4 run-state 对账

`_record_failed_panel` 现在会把同一 `run_cfg` 下停留在 `running`（进程被
杀/后续 preflight 拒绝）的条目改为 `failed`，同时保留已测 cell；已经是
`complete` 的面板不会被后来的失败 preflight 改写，失败尝试只进
`experiments.jsonl`。这样“进程退出”与“面板完成”不再是同一个状态。

### 40.5 准入绑定唯一权重与真实清单路径

确认准入在记录携带 `ckpt_sha256`（或清单声明
`binding.require_ckpt_sha256`）时要求合法 SHA-256，并拒绝同一权重摘要被
另一个 `(variant, seed)` 复用：一套权重不是两次独立测量。`_load_manifest`
把 `_path` 带到加载后的清单上，冻结锁因此记录实际使用的清单文件，而不是
默认文件名。

### 40.6 面向放大：几何参数化

`apply_scale_geometry_overrides` 接受 `B_SEQ_LEN` / `B_BATCH_SIZE` /
`B_N_TRAIN_TOKENS` / `B_STEPS` / `B_VAL_SEQS`，非法值直接报错；只改几何
而未显式给步数时按原采样窗口预算重算 steps（ceiling），改 token 预算时
从预算重算，显式 `B_STEPS` 永远优先。覆盖值写入 cfg，因此进入 `run_cfg`
与 manifest。M0 驱动改为发现 `RUN_S<n>`，新增 S3/S4 不需要再改
`_panel_chain_from_env`；没有声明 cap 的未来档位使用自己的
`run_time_state_s<n>.json`，不会与 S1 共用预算。

### 40.7 最终评估也带机器环境

P5T 的 by-length 记录以前只有代码戳/配方/数据/权重哈希，没有执行环境。
现在每条评估记录与 `experiments.jsonl` 事件都带 `exec_fp` 与紧凑的
`environment` 快照；resume 的 stale 判定把「已记录但不同的执行类别」
视为过期。这样“同一份权重在不同机器/精度栈上评出来的数”不会被当作本轮
当前机器的测量静默复用，训练与评估的 provenance 口径一致。

## 41. 测量语义身份、配对聚合与确认参数绑定（本轮硬化）

本轮不新增 GPU 面板，不改冻结清单的臂/种子/终点/α/族，也不改训练数值路径
（`CODE_SEMANTICS` 不变）。修正的是分析层三处会让「配对」和「确认」名不副实
的数据流。

### 41.1 聚合包装器不得在配对前拆散臂

`_dump_aggregate` 走的是 `aggregate_with_exclusions`。旧实现先按 `variant`
分组、再按指纹类拆，`aggregate(sub)` 的每个 `sub` 只剩一个臂，于是
`dPPL_vs_full` / `dPPL_vs_sw128m` 永远不会生成——所有面板 dump 只有边际
均值，没有配对差（直接调用 `aggregate` 才有）。现在先按
`(protocol, sample/measurement class)` **跨臂**分组：同一 class 内的 csa 与
`full` 一起进入 `aggregate`，配对差恢复；class 不同才拆，输出键追加
`@fp<digest>`，且同一 class 的各臂共享同一后缀。跨协议/配对失败时仍逐臂
回退并记 `pairing_split_groups`，此时明确不产生 dPPL，而不是伪造一个。

### 41.2 `_code` 是测量身份，不是装饰

`run_cfg` 通常内嵌代码语义戳，但历史记录可能没有后缀。两条不同 `_code`
的记录不是同一实验的两个种子：预处理、归约顺序或内核路径可能不同。现在：

- `aggregate` 的 `_cfg_fp` / `_cfg_fp_noparam` 含
  `measurement_semantics_key(rec)`；
- `aggregate_with_exclusions` 的指纹类含 `_code`；缺失戳是独立的
  `<unverified>` 类，不是会与已验证记录合并的通配符；
- `methodology_audit` 的 arm summary 分组身份含 `_code`（与 val/eval/
  holdout 指纹并列）；
- `controlled_variables_report` 新增 `code_equal` / `code_values`：
  跨臂不同、臂内不同种子不同、或部分记录缺戳都使
  `controlled=False`；两边都缺仍是 legacy `None`，而不是失败或通过。

这些改动不改变任何数值，只让不可比的记录被分组或点名，而不是静默平均。

### 41.3 estimand 标签的集合真值 bug

`{False}` 在 Python 中为真。`aggregate` 与 `methodology_audit` 的 arm summary
曾因此把普通全验证记录标成 `nested_validation_holdout`。估计量本身是正确的
（`_pooled_ppl_from_records` 用单个布尔推断），错的只是标签；但标签会进入
报告与下游消费，仍属统计口径缺陷。现在只有 `_nested_flags == {True}` 才标
nested，其余一律 `full_validation`（missing/mixed 的显式 fallback 不变）。

### 41.4 确认性参数绑定

冻结决策规则写的是 `param_matched is not false`；若记录没有 `params`，
`param_matched=None` 会被读成「不是 false」而通过。冻结清单已经披露
`compute_budget_disclosure.parameter_matching`，因此参数计数是测量身份的
一部分，而不是可选的展示字段：

- `_admit_confirmation_records` 对这类清单要求每条记录携带正的 `params`；
  缺失/非法即**点名剔除**。`statistics.descriptive_only=true` 的 smoke 清单
  与没有该披露的历史清单保留原 permissive path。
- `finaltest_stats` 的 `param_matched` 只在**每个可用种子、两侧**都有参数、
  且同一臂自身各 seed 的参数完全一致时才计算；否则
  `param_match_complete=False` 并给出 `param_match_reason`，不再让一个种子的
  `params` 认证整臂，也不把「未知」当匹配。
- 每条比较落盘 `params_by_seed` / `param_consistent` /
  `param_match_complete`；顶层 `parameter_binding` 记录该清单是否要求绑定、
  以及有多少可采纳记录缺参数。对披露参数匹配的清单，确认判定还要求
  `param_match_complete is True`——`param_matched is not false` 单独成立时
  仍会接受 `None`，那正是「未知当通过」的路径。

冻结决策文字与 `confirmation_frozen.json` 字节均未改动：本改动只是让「未知」
不能走进那条规则；shipped plan 的 72/72 合成记录仍全部通过预检。

### 41.5 入口与验证

```bash
python v12_supp.py plan              # 冻结预检（参数绑定 + 原有功效/预算门）
python v12_supp.py dynamic-plan      # 动态 draft 仍不可运行
python test_review_fixes.py          # L104-L106 覆盖上述合约
python test_open_block.py            # 15/15
```

L104 覆盖「包装器恢复配对 dPPL + 混代码戳拆分 + nested 标签」；L105 覆盖
代码语义进入 arm/control 身份；L106 覆盖确认准入的参数绑定与逐种子一致性。

## 42. 确认结论的稳健性：脆弱性指数与逐种子 Holm 重算（本轮结果分析硬化）

本轮只改结果分析层，不新增 GPU 面板，不改冻结清单的臂/种子/终点/α/族，
也不改训练或测量数值（`CODE_SEMANTICS` 不变）。目标是把「显著」之外的第二个
问题写进报告：**这个判定有多依赖个别种子？**

### 42.1 为什么需要它

`finaltest_stats` 已经报告逐种子 Δ、均值、精确符号翻转 p、Holm p 与逐文档
区间，但没有任何读数回答「删掉一个种子后，Holm 拒绝集合是否还成立」。
`paired_effect_stats.seed_driven` 只检查**均值符号**是否被单个种子翻转，
既没有进入冻结确认分析，也看不到「均值不翻、但 p 因一个种子越过 α」的情形。
一个 12 种子的冻结 panel 若由 11 个微小同向差 + 1 个极大同向差支撑，均值与
p 都可能很好看，但结论对那一个种子高度敏感。报告若不写这件事，读者会把
「确认」误读成「稳健」。

### 42.2 两个核心脆弱性读数（外加逐种子 LOO 与区间）

`exp_lib.paired_fragility(deltas, seeds=...)` 复用同一个精确符号翻转检验，
对每个子集重算 p（确定性、无 Monte-Carlo），给出：

- `drop_fragility`：反复删除当前 |Δ| 最小的种子、p 仍 < α 的最大次数。
  `0` 表示删掉最弱的一个种子就已经不显著；数值越大，判定越不依赖单点。
- `flip_fragility`：按 |Δ| 从小到大翻转**同向**种子的符号，直到 p ≥ α 或
  均值方向反转所需的翻转次数。它是贪心影响上界，**不是**精确最小翻转集；
  `definition` 字段随数值一起落盘，避免被读成精确最小值。
- `loo`：逐种子删除后的 mean/p、`p_increase`、`p_shift=|Δp|`、
  `removes_significance`、`flips_direction`；`most_influential_seed` 按
  **绝对 p 位移**取最大（不是只看 p 增大：删除一个反向种子让 null panel
  变显著，同样是单点依赖）；等幅数据下所有种子并列，`most_influential_tied_count`
  明确写出并列数，报告不把任意一个种子写成唯一驱动。
- 回归到均值的 t 区间 `ci95_low/high` 与 `ci_excludes_zero`，用来看
  「精确检验显著」与「区间排除 0」是否一致；报告的稳健性表同时打印
  `Δ mean [95% CI]`，不让读者只拿到一个 p 值。

数值本身不进入任何判定：函数只返回诊断字典。

### 42.3 族级：删一个种子后重算整个 Holm

`exp_lib.holm_decision_jackknife(named_paired, family_names=...)` 对种子并集
的每个种子做一次 fold：包含该种子的比较删掉该对后重算精确 p，不包含的
比较保留全 panel p；随后对整个族重算 Holm，并用与冻结决策一致的严格
`p_holm < alpha` 回放拒绝集合。输出 `rejected_full`、每个 fold 的
`rejected/lost/gained`、`always_rejected`、`sometimes_rejected`、
`unstable_folds` 与逐比较的 `p_holm_min/max`。

`family_names` 是**声明族**：没有可算 Δ 的比较仍以 p=1 占位。若省略这一点，
一个缺失臂会让 jackknife 的 m 从 6 缩到 5，Holm 校正反而变松——这正是
`holm_adjust` / `attach_holm` 一直在防的「少报比较缩小族」。

### 42.4 与冻结决策的边界

- `v12_supp.finaltest_stats` 只在原有 `confirmed` 算完之后追加
  `stats['robustness']`；`confirmed`、`p`、`p_holm`、Holm 族、α 与
  `min_seeds_for_confirmation` 全部不改写。报告中的新表明确标注
  「post-hoc 诊断；不改变上面的冻结判定」。当
  `all_rejections_stable=false` 时另打印「解读提示」：判定仍按预注册规则
  成立，但**不得**被描述为「对种子稳健」，引用时必须同时报告敏感度。
- `require_complete_coverage` **不在** jackknife 内重放：删任何一个种子都会
  使覆盖失败，若重放则该门按构造必然失败，等于没有信息。因此另设
  `confirmed_without_coverage_full` /
  `confirmed_without_coverage_by_fold` /
  `confirmed_without_coverage_stable`，只重放方向 + 参数 + 最小 n 的判定，
  并在 `coverage_gate_note` 说明覆盖门仍属于冻结决策。
- `confirmation_robustness` 的 `family_size` 取冻结清单声明的 m；若传入值
  小于实际比较数则**拒绝**，不靠缩小族来放松 Holm。逐比较另给
  `min_same_sign_for_family_gate`（等幅投影下界，仅作规划参照；
  `same_sign_margin` 不替代精确检验）。

### 42.5 文档层广度：均值是否被单个文档支撑

`v12_supp._doc_level_analysis` 新增 `doc_delta_negative/positive/zero`、
`doc_a_better_fraction`、`per_doc_delta_median` 与
`top_doc_abs_share = max|Δ_doc| / Σ|Δ_doc|`。这些是描述性读数（文档不是臂的
随机化单位），但能暴露「一个文档贡献了几乎全部均值差」的赢法；逐文档
bootstrap 区间仍然以已训练种子为条件。p5t 主表的「文档 Δ 95% CI」单元格
同时显示 `Δ<0` 的文档占比，避免只看均值与区间。

### 42.6 入口与验证

```bash
python test_review_fixes.py     # L107-L109
python v12_supp.py plan         # 冻结预检不变：72/72 合成记录、6 个主比较
```

L107 覆盖 drop/flip/LOO、种子对齐、非法输入、族门投影（n=6 无法通过
Holm/6 时返回 None），以及
「删除 n=6 比较的一个种子后 Holm 拒绝丢失、n=12 比较保持」和
`family_names` 不缩小 m；L108 覆盖 v12 集成只追加 `robustness`、不修改
`confirmed`/`holm`、覆盖门不被重放，以及 Markdown 渲染；L109 覆盖文档层
广度读数与行长不一致时的拒绝。


## 43. 文档级评测必须按真实文档边界聚合（本轮数据口径硬化）

固定长度的评测行是**流上的窗口**，不是文档。一行可能只覆盖某个文档的
一部分（同一文档被两行各计一次），也可能同时包含前一个文档的尾部和后
一个文档的头部。把它当作「文档」做 bootstrap，会系统性地把独立性假设
写错：点估计仍是 token 池化 PPL（不受影响），但区间和 `n_docs` 描述的是
行抽样，而不是文档抽样。

因此：

- `load_wikitest` 除文档级 sidecar 外，另落**逐 token** 文档 id
  `test_token_doc_ids_*.npy`（一个 served token 一个 id）。它由产生 token
  流的同一组 `len(ids)+1` 长度展开，EOS 归属发出它的文档；写入前校验
  长度恰等于 served slice、逐 token 与逐文档两个 sidecar 的文档集合一致；
  meta 记录 count + fingerprint，缓存命中时重新核验。把声明了该 sidecar
  却缺失/错位的缓存视为损坏（fail closed），而不是静默退回行级。
- `eval_length_gen(doc_ids=...)` 用 `document_index` + 确定性分段求和把
  **同一个** per-token NLL 表按文档聚合，输出 `nll_per_doc` /
  `n_tokens_per_doc` / `ppl_per_doc` / `doc_ids` / `n_docs`。池化 `ppl`、
  `n_tok`、`nll_sum`、`nll_per_seq`、`ppl_per_seq` 不变；新字段只是同一
  批预测的另一种分组。
- `v12_supp._doc_level_analysis` 优先使用 `ppl_per_doc`；若记录混合了
  文档级与行级、或逐文档 token 数/文档顺序不一致，则拒绝配对，而不是
  按位置硬配。`granularity` 明确写出 `document` 或 `row`；报告中的
  「文档/行 Δ 95% CI」和「% Δ<0」使用对应的单位。
- 准入要求：声明 `doc_granularity=document` 的单元，逐文档 token 数之和
  必须恰等于 `n_tok`，`n_docs` 必须等于逐文档向量长度。审计端独立重算
  per-token sidecar 的长度、dtype、负值与指纹。
- 旧缓存可**不重建**恢复映射：构造契约保证每篇完整文档恰好一个 EOS，被
  token cap 截断的最后一篇没有 EOS；`_token_doc_ids_from_eos` 按 EOS 位置
  和文档级 sidecar 顺序展开。只有 EOS 计数/顺序与文档集合都通过校验时才
  采用；否则仍显式降级为行级并打印原因。这样冻结后的 test cache 不必为了
  文档级不确定度而重读/重建。
- 边界：旧缓存没有逐 token sidecar、且 EOS 重建也无法通过校验时，文档级
  读数明确降级为**行级**并打印说明；池化 PPL 与冻结判定不变，
  `CODE_SEMANTICS` 不因本改动而升版。

入口与验证：

```bash
python test_review_fixes.py     # L111 逐 token 对齐、L112 跨行/混合文档聚合、L113 粒度拒绝
python v12_supp.py plan         # 冻结预检不受影响
```

## 44. 文档级 Δ 区间必须与主终点同口径（token 池化 cluster）

主终点是 token 池化的 PPL：`exp(Σ NLL / Σ tokens)`。文档长度不同，它
不等于「每个文档 PPL 的等权平均」。因此「平均文档的 ΔPPL」和「池化
ΔPPL」是两个不同的估计量，不能把前者的区间印在后者的点估计旁边。

本轮修正：

- `exp_lib._paired_nll_doc_ci(nll_a, nll_b, n_tokens)` 以文档为 cluster，
  先对每个种子计算 token 池化 CE 差，再对种子取平均。单种子影响函数为
  `exp(ce_a)·(a_i − ce_a·t_i)/T − exp(ce_b)·(b_i − ce_b·t_i)/T`，
  种子平均后的方差为 `D/(D−1)·Σ_i ψ_i²`，用 `df=D−1` 的 t 临界值；同时
  给出种子平均 log-ratio 与 PPL ratio 区间。它不重采样、结果确定，且
  明确条件在已训练的种子上：种子间训练随机性仍由精确符号翻转检验负责。
- `v12_supp._doc_nll_matrix` 在使用前校验逐文档 NLL 有限、逐文档 NLL 之和
  与单元记录的 `nll_sum` 一致、`ppl_per_doc = exp(nll_i/t_i)` 一致。任一
  不通过就拒绝计算，并写出 `pooled_ci_unavailable_reason`；绝不从错位的
  向量算出一个看起来正常的区间。
- `_doc_level_analysis` 新增 `pooled_delta_ppl` / `pooled_delta_ppl_ci95` /
  `pooled_delta_log_ratio` / `pooled_delta_log_ratio_ci95` /
  `pooled_ratio_ci95` / `pooled_ci_n_docs` / `pooled_ci_df` /
  `pooled_ci_method`。旧的 `per_doc_delta_mean` 与等权 bootstrap
  `per_doc_delta_ppl_ci95` 保留，只作为「平均文档」的描述量。
- 准入端 `_by_len_shape_reason` 增加：`nll_per_doc` 必须有限，且其和必须
  与记录的 `nll_sum` 一致（容差 1e-5 相对）；`n_tokens_per_doc` 的每一项
  必须为正。零目标文档已在聚合端被丢弃，若仍作为评测单元声明，说明该
  单元的文档向量与计分流不一致，不能准入。
- 报告端 `_doc_ci_markdown_cell`：有 pooled cluster 区间时优先显示并标明
  「token 池化」；没有时才显示「等权单元 bootstrap」，并在表头/表注说明
  两种口径。行级记录仍只能得到行 bootstrap，不会冒充文档区间。
- 边界：这是描述性的文档抽样区间（且条件在已训练种子上），不进入冻结
  决策；精确符号翻转、Holm 族、覆盖门和 `confirmed` 均不变。池化点估计
  和 `CODE_SEMANTICS` 也不因本改动而改变。

入口与验证：

```bash
python test_review_fixes.py     # L113 准入、L115 cluster CI、L116 集成、L117 报告口径
python v12_supp.py plan         # 冻结预检不受影响
```

## 45. 设备端分块轨道上界与 pooled-CI 来源守卫

两个零 GPU 的潜在缺陷，规模扩大后都会变成静默错误，因此一并修正。

### 45.1 设备端后继轨道必须按「最小有效步长」定上界

`_device_segment_geometry` 用倍增法在设备上预分配后继轨道，原先按
`ceil(n / min_block) + 1` 估算块数上界。该上界默认
`max_block >= min_block`（`AttnCfg` 的正常约束），但本函数承诺是
`_segment` 的逐整数复刻，而宿主分段器接受任意整数上下界。当
`max_block < min_block` 时，后继
`f(p) = min(p+max_block, max(p+min_block, cut+1))` 恒取第一支，实际步长
是 `max_block`；长序列下真实块数约为 `n/max_block`，预分配轨道只覆盖
`n/min_block` 个起点，剩余块被截断，所有后续 token 被并进一个超长末块。
设备端与宿主端的块表因此不同，且没有任何报错。

修正：按最小可能步长定上界
`max(1, min(max(min_block, 1), max(max_block, 1)))`。`max_block` 为
`None` 时已在函数入口转换为 `n`；上下界为 0 或负数时宿主每一步也至少
前进 1，该公式同样成立。修正不改变 `AttnCfg` 已有的
`max_block >= min_block` 校验，只让内部复刻在宿主可接受的任意边界下
保持逐整数一致。

### 45.2 没有主终点 `ppl` 时不得发布 pooled cluster 区间

`_doc_level_analysis` 只用文档 NLL 向量计算 cluster 区间，然后把它与
单元记录的逐种子池化 `ppl` 差核对，确保点估计与冻结主比较同口径。
原实现中，若 `ppl` 缺失或非有限，`_pt` 为空，`else` 分支会直接接受
`_pc`，跳过这层核对，发布一个无法与主终点绑定的区间。缺失的主终点值
不是「文档向量已核对」的证据。

修正：`_pt` 为空时写出 `pooled_ci_unavailable_reason` 并拒绝该区间；
只有在逐种子 `ppl` 有限且与文档向量复现的池化 Δ 一致时才发布。

入口与验证：

```bash
python test_review_fixes.py     # L116 缺失 ppl 拒绝、L118 设备/宿主分块一致（n=10003）
python test_open_block.py       # 分块与开块回归
python v12_supp.py plan         # 冻结预检不变：72/72、6 个主比较
```

## 46. 配对初始化的证据必须随权重走，并按种子核验

确认性检验的单位是「同种子、两臂」的配对差。精确符号翻转检验要求配对内
交换性，其中「两臂从同一个 shared-trunk 初始化出发」是可核验的必要条件。
本轮不新增实验、不改冻结清单的臂/种子/终点/α/族，也不改训练或评测数值；
修正的是这条证据从训练记录到确认分析之间的断链，以及一个会「真空通过」
的读法。

### 46.1 三个具体缺陷

1. **证据没有进入 test 记录。** `train_variant` 已经把
   `init_core_fp`（`tok`/`pos`/`head`/`norm` 的初始化摘要）写入训练记录，
   但 `_finaltest_bylen_cell` 的 `_ck_prov` 只搬运
   `train_knobs/params/train_objective/param_breakdown/rng_fp`，bylen 记录
   从未携带该字段。确认统计从 `bylen.json` 现算，因此根本看不到初始化
   证据；P5T 的 checkpoint 也没有保存它。
2. **全局「所有值相等」是错的。** `init_core_fp` 是种子的函数：健康面板
   里不同种子的摘要本来就不同，同一种子两臂才应相同。`finaltest_stats`
   原先用「admitted 记录的全部值只有一种」作为 `same_init_core_fp`，健康
   面板会被判为 False；更糟的是字段缺失时集合为空，
   `len(values) <= 1` 为 True，于是「没有证据」被打印成「匹配」。
3. **确认准入没有执行配对检查。** 通用的 `pair_reason` 会拒绝
   `init_core_fp` 不一致或单边的配对，但 `_admit_confirmation_records`
   只检查了数据/语料/行数/配方/代码/参数，没有调用它。于是初始化不一致的
   种子仍可能进入确认比较；在 `require_complete_coverage=false` 的历史
   清单下，甚至可以用剩余配对继续确认。

### 46.2 修正

- **随权重走。** checkpoint 的 `extra` 增加 `init_core_fp`；
  `_finaltest_bylen_cell` 从训练记录（旧 checkpoint 则从权重元数据）取出
  该值写入 bylen 记录与 experiment event。若 checkpoint 与训练记录都带该
  字段但不相等，说明权重与训练记录不是同一次初始化，直接 REFUSE 该 cell；
  缺失时打印 WARNING，明确该配对将显示为「未验证」，而不是伪造一个摘要。
  续跑时，已有 bylen 记录的 `init_core_fp` 与当前期望值不一致（包括旧记录
  根本没有该字段）会被判为 stale 并重新评测，而不是把一个「未验证」的
  旧测量直接复用进确认面板。
- **纯函数审计。** `exp_lib.paired_init_core_audit(records_a, records_b,
  seeds)` 按种子给出 `verified` / `mismatch` / `unverified` / `missing`
  四态，返回 `pairing_ok` 三值：全部核验为 True，存在确定不一致为 False，
  证据不完整为 None。单边或缺失的摘要不参与相等判断。
- **确认层执行。** 每个主比较先跑该审计；`mismatch` 的种子被**点名剔除**
  （写入 `dropped_pairs`），不进入精确符号翻转检验。若剔除后不足两对，
  比较按 `omitted` 记录而不是伪造结果。逐比较落盘
  `init_core_fp`/`same_init_core_fp`/`init_core_fp_recorded`；顶层
  `validity` 改为按种子的两臂摘要集合汇总，`init_core_fp_complete`
  明确面板是否每个种子、每个臂都有证据。
- **判定边界。** `confirmed` 增加
  `same_init_core_fp is not False`：确定不一致的配对不能确认。`None`
  （历史记录未携带该字段）保持原有 permissive 路径，但报告只写
  「未验证」，不再写「相同」。在 shipped plan 的
  `require_complete_coverage=true` 下，剔除任何一个种子本来就会让覆盖门
  失败，因此该 gate 只是把同一事实写进判定，不引入新的效应量阈值。
- **报告。** P5T 主表新增「初始化配对」列（✔ / ✘ / —未验证），并在有
  剔除时列出种子与原因。`validity.note` 明确 `rng_fp` 仍然只报告不强制
  （架构对 RNG 的消耗属于臂本身），而初始化这一部分现在是「测量并执行」，
  不再是「真空为真」。

### 46.3 边界

这是测量/分析层的有效性修正，不是新终点：`confirmation_frozen.json`
字节、冻结臂/种子/终点/α/Holm 族、训练数值路径、test split 与
`CODE_SEMANTICS` 均未改变。checkpoint 只多一个元数据字段，已有 checkpoint
仍可从训练记录回退读取。新 gate 不会把「未验证」判成失败，只阻止「确定
不一致」被当成配对。

### 46.4 入口与验证

```bash
python test_review_fixes.py     # L119 三值审计、L120 确认层剔除/未验证语义
python v12_supp.py plan         # 冻结预检不变
python v12_supp.py smoke        # P5T 往返仍覆盖，smoke 记录不可入确认
```

## 47. 面向放大的分块/侧车/宿主几何：不要按 token 或 block 建 Python 列表

本节不改变任何冻结配置或测量值；四处在长序列/长文档下会随规模放大的
表示或失败路径被修正。

### 47.1 均匀步长分块也必须遵守大行 NumPy 契约

`_segment` 的通用后继表路径已在 `n > 64` 时返回 NumPy `counts` 行，避免
为每个 block 建 Python 对象；但均匀步长闭式捷径仍无条件执行
`[_stride] * (n // _stride)`。m=1 / `block_size=1` 臂的步长恰为 1，长序列
因此先产生一个每 block 一个指针的 Python 列表，随后消费端
`np.asarray(counts)` 还会再复制一次。现在捷径先用 `int64` NumPy 行构造
同一组整数，仅 `n <= 64` 时转回历史 list 形式；`bids`、`honoured`、
`counts` 与 `sum(counts)` 的值逐项不变。

### 47.2 逐 token 文档 sidecar 不得为每篇文档建临时列表

缓存构建的验证流原先用 `array.extend([doc_id] * take)` 写逐 token id。
长文档会临时分配「每个 token 一个指针」的 Python 列表，尽管独立的
capped-row helper 已改用 `np.repeat`。现在改用
`array.extend(itertools.repeat(doc_id, take))`：临时对象 O(1)，写出的
uint64 id 序列完全相同。

### 47.3 宿主几何的 OOM 必须走既有回退，而不是让整层崩溃

池化几何的宿主构建器承诺「宿主形式不能服务时返回 `None`，调用方回退到
设备端形式」。原 `except` 覆盖 RuntimeError/TypeError/ValueError/
AttributeError，但 `MemoryError`（以及 `OverflowError`）不在此列：规模
放大后宿主侧一次大分配失败会直接终止，而不是使用值完全相同的设备端
几何。六个宿主构建器现把分配失败也视为「不能服务」；pinned staging 池在
`pin_memory` 因显存/锁页内存不足抛 `MemoryError` 时同样回退。成功路径的
整数与布尔值不变。

### 47.4 参数审计的档位必须从阶梯发现（S10 也按数值排序）

`m0_supp.run_params` 过去遍历字面量 `('S1', 'S2', 'S3', 'S4')`。新增
`SCALE_LADDER` 档位时，面板解析器和 `scale_ladder_audit` 已能发现，但
planned-vs-observed 参数量表会静默漏掉该档。现在档位列表由
`SCALE_LADDER` 生成并按数值排序（`S2` 在 `S10` 前）；没有记录规划值的
档位仍然测量并写出，只是 `ratio_to_plan=None`，不删除也不 KeyError。

边界：47.1–47.3 是表示/传输层修正（整数、缓存字节、前向值均不变），
47.4 是审计覆盖面修正；冻结清单、臂/种子/终点/α/Holm 族、训练数值路径与
`CODE_SEMANTICS` 均不变。

入口与验证：

```bash
python test_review_fixes.py     # L121 大行分块 + MemoryError 回退；L122 阶梯发现/排序
python test_open_block.py       # 分块与开块回归
python v12_supp.py plan         # 冻结预检不变：72/72、6 个主比较
```
