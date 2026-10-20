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

- 每条记录新增 `train_objective`（comp_reg 是否实际生效、delta_logit 的
  LR 倍数是否实际生效、boundary policy）与 `param_breakdown`
  （total/attention/MLP/embedding/head/active）、
  `kv_cache_elements_per_token`。
- `methodology_audit` 对同一 `run_cfg` 但目标不同的对照给出
  `objective_confounded` 警告与 `mechanism_claim_allowed=False`。这不会
  阻止配对（那会让所有 dense-vs-sparse 对比消失），但会阻止把它写成
  「隔离机制」。
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
python test_open_block.py                 # K1-K9 causality/reference/mask gates
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
`(summary - closed | None) - (summary - closed | 128)`; it reports mean,
sample SD, `d_z`, exact sign-flip p, Holm-adjusted p within the declared
primary family, and the attainable-p floor.  At 4 seeds the floor is 0.125;
this is a resolution limit, not evidence of no effect.  A controlled
comparator permits only the declared factors (`max_block`, `open_block_mode`,
`open_block_scope`) to differ and rejects undeclared recipe/layer differences.

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
