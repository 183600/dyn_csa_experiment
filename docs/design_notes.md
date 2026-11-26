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
  检查在产物缺席时必须可跳过而不是失败。驱动脚本的 `os.chdir(REPO)` 只在
  CLI 入口（`__main__`）执行，不在导入期；`par.py` 子进程由派发方显式给定
  `cwd=REPO`，库调用方（audit 门 / 测试）保留自己的工作目录。

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
（NLL_干净 − NLL_屏蔽），是解读一切鲁棒性数字的参照系。**为什么必须有 NIAH-natural**：检索任务（key→value 耦合埋在远端自然文本、
查询在目标区、干扰只加在草料上）用于测「长距离信息利用」。但把随机 token
的 key→value 对插入自然文本**不等于**模型具备该格式下的检索能力：小模型
未在该格式上训练时，chance 水平可能只是任务不适配，而不是注意力失败。
因此 NIAH 增加了机器判据（review 10 issue 13）：chance 基线（1/可用
value token 数）、`nofar` 无远端信息对照（摧毁远端耦合后准确率应退回
chance）、距离分桶准确率（short→long 梯度），并要求记录一个**能解任务
的正对照**。三者缺一时 NIAH 只能作描述性读数，不得作为「长距离利用」的
机制证据；当前面板没有正对照，故该门保持关闭。合成 NIAH 的格式训练/评测
路径在 v7_supp 中已实现但未列入 phase（可显式作为探索性正对照运行，不
触碰冻结确认）。

**ΔNLL 口径**：ΔNLL = NLL_污染 − NLL_干净（nats/目标 token，逐种子配
对；干净基线 = 同臂同参数化的 uniform ρ=0 cell）。报告必须同时给 ΔNLL
与干净 NLL——某臂 ΔNLL 小 + 干净 NLL 差 = 「少用远端」，不是「更抗噪」。

**ρ 是名义值，必须同时报告实现覆盖（review 10, issue 12）**：
- `uniform`/`natural`/`topical` 选 k = max(1, round(ρ·n_far)) 个位置；
  donor 模式写入的 token 可能与原文相同，因此**实际改变 token 数 ≤ k**；
- `span` 的段**互不重叠**（远端按等宽 tile 切分、每 tile 抽一段），实现
  覆盖槽位 = `n_seg × span_len`（通常 > k，是 span_len 的量化），
  `自然/主题/文档` 供体撞车同样使 changed < covered；
- `doc`/`mask` 恒 ρ=1、覆盖整个远端；`delete` 是**移除**，定义上
  changed = n_far 且行变短。
因此同一 ρ 在不同模式下**不是同强度腐坏**，跨模式比较只能以「实现
覆盖/实际改变比例」列与 ΔNLL 一起读；`_corruption_audit` 把这些量
（覆盖槽位/唯一位置/重叠/实际改变 token/空操作行/供体行）落盘并在报告
中打印，禁止只报 nominal ρ。另：`uniform` 是强分布外扰动、`natural`
只替换 token 不保持句法、`topical` 的 Jaccard 只是主题代理且不保证答案
无关、donor 单位是**固定评测窗口**（可能跨文章）——这些都写进报告限制段。

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

**review 10（issue 14）补充的不可分离性（必须与上表并列写出）**：
- 规模面板同时改变宽度、深度、seq/batch 与训练步数（P3SS 128/4L/8000 vs
  P3SL 512/10L/4000；d=384 面板另用 seq=1024/bs=8）。**不存在只改宽度或
  只改深度的对照**，因此「规模」效应无法在 2–3 个规模点内分离成宽度/
  深度/预算三项。
- 等 token 切片对齐的是**累计 token**；不同面板的学习率进度形状与数据
  重复次数并不相同（8000 步面板在相同 token 处经历了不同的 LR 与重复
  模式）。`P3SL_TOK` 对齐了步数/LR 形状，但**不**对齐参数分配。
- 交叉点不是模型属性：它是**当前评测网格与观察窗内**的首次“转正且不再
  回穿”，窗后行为未知、删失点不能进入拟合；log-log 拟合用的是**seed
  均值轨迹**的交叉位置，不是逐种子交叉位置的平均。
- 6ND 是**参数量代理**，不是实测 FLOPs；「最后四分之一 eval 改善 <1%」
  不是收敛证明。规模结论只能写成**预算内描述性趋势**，不得写成
  scaling law。

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
- `csa_fixed_rope_allblocks` 是 **同架构的全块读取对照**（与
  `csa_fixed_rope` 只差 `index_topk`），因此 `csa_fixed_rope` −
  `csa_fixed_rope_allblocks` 是同投影/同归一化/同位置编码下只改选择方式
  的落点，评审点 5 的选择侧结论应建立在这一对上。
- 它**不是** dense 对照：相对 `full_rope` 仍差 KV 压缩投影、块池化、
  q/k 归一化位置与宽度（CSA 为 L2-normalize→RMSNorm，dense 仅 RMSNorm）、
  attention scale（1.0 vs 1/sqrt(d_head)）、sink、滑窗分支、kernel 与
  MLP 宽度（参数匹配改变的是 MLP 容量分配）。`full_rope` − allblocks
  只能作**系统级**读数，代码中已由 `arch-aligned` 更名为
  `system-level`，不得再写成压缩/分块的单因素效应。

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

- **零假设与符号对称性（review 10 issue 6 修正）**：精确双侧符号翻转
  检验的 p 值在**尖锐零假设**下精确：每个种子的配对差的符号分布关于 0
  对称（配对差分布对称于 0，或随机分配处理使符号可交换）。它**不是**
  「两套架构平均效果相同」这一弱零假设的检验——除非差值分布对称于其均
  值，否则平均相等不能推出符号可交换。共享种子编号、数据顺序、评测行与
  腐坏网格是**控制**（减少 nuisance 变异、让配对有意义），**不是**随机
  分配意义上的交换性，不能使检验自动有效。机器可检的控制项：
  `data_fp`/`train_fp`/`n_seq`/配方签名/`_code`/manifest sha/逐种子
  `init_core_fp`；`rng_fp` 只报告不强制（架构导致的 RNG 消耗属于臂）。
  `finaltest_stats` 现在把该零假设文本、逐比较符号计数、留一法均值符号
  稳定性与 `|mean|≤std` 稳定性诊断一并写入 `validity`；这些只限定稳定性
  读法，不改变预注册决策规则。若差值分布强烈偏斜/厚尾，精确符号翻转
  检验与等幅投影功效表都可能偏离，p 值只对上述尖锐零假设成立。
- **行/窗口抽样不确定性（意见九 b + review 10 issue 9）**：最终测试的干
  净 by-length 评估默认 16 条**固定窗口**（不是 16 篇文档；一个 4096
  token 窗口可跨文章，一篇文章也可贡献多行）。`ppl_per_seq` 是**逐行/
  窗口** PPL，池化 PPL 的归约不变；`finaltest_stats` 的 `doc_sampling`
  单位明确标为 `fixed evaluation windows (sequence rows), NOT documents`，
  并给出 between-window 与 within-window-across-seed 两层分解。真正的
  文档聚类需要 loader `doc_index` 路径（follow-up 研究的要求），冻结
  v12 不提供。`MIN_DOCS_FOR_POPULATION_CLAIM=64` 是**仓库自设的最小
  总体门槛**（无统计推导；64 不是总体推断突然有效的分界线），且它计数
  的也是行/窗口；抽样代表性与区间精度比固定条数更关键。16 行**不足以
  覆盖语言数据总体分布**——主张范围限于「该冻结 test 切片」。
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
- 报告仍打印 seed 标准差，但必须标成 **seed-level**；行/窗口 bootstrap（非文档）才是
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
- 报告正文的「显著」只由**统计证据**门决定：n≥6（否则精确检验下限
  >0.05 或族校正不可达）、原始 p<0.05、Holm p<0.05、族级分辨率可达、
  且通过 `run_cfg`/`train_knobs` 配对门禁。`|mean|>std` 与留一法符号
  稳定性是**稳定性**诊断（报告为 `stable`/`stability`），**不是**显著性
  判据：低于一个种子标准差的效应可能是真实且可估计的小效应，不得据此
  说它「不真实」「只是噪声」或「不显著」。**实际价值**需要**预先声明**
  的最小有意义改善（`min_effect`）；v12 未预注册该阈值，因此 v12 报告
  不得作「实际价值」声明，只能同时给出估计、CI、校正 p、稳定性与主张
  范围。任一门槛不满足只能写「方向性 / 名义显著 / 未过校正」，不得写
  「证伪」「机制性」「更稳固」。

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
- straight-through 的完整梯度契约：前向值只取 hard 选择注意力，但反向里
  `attn + (soft_attn - soft_attn.detach())` 让一切进入 `soft_attn` 的
  参数都多得到一份 soft 路径梯度。即除了 indexer 权重（`W_DQ/W_DK/W_w`，
  这是它们唯一的梯度来源）之外，主注意力路径的参数（q 投影、KV 池化/压
  缩/头投影与 sink）会**同时**收到 hard 与 soft 两条梯度。这是有意的
  surrogate 估计器设计：eager 与 kernel 参考实现按同一契约对拍
  （`test_review_fixes.py` 的 `t_kernel_*` 梯度测试锁定该契约），不是
  「只有 indexer 获得梯度」。收紧或放宽该契约属于测量语义变更，必须更新
  `CODE_SEMANTICS` 并重跑相关面板。
- 新增 opt-in `csa_dynamic_task`：切分决策按 gate 概率采样，训练损失加入
  score-function 项 `+(CE - baseline) * logπ / n_decisions`（CE 是被最小化的
  成本：advantage 为正时该项的梯度下降会**降低**该切分模式的采样概率；
  写成 `-adv` 会强化坏切分）。这才是「任务损失可训练边界」的臂；它已加入
  `ABL_VARIANTS` / `ABLATION_DESIGN`。
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
  不删除；16 条 test **窗口/行（非文档）**不覆盖语言总体；实际推理延迟/吞吐/峰值显存必须
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

`exp_lib.reportable_claim` 是报告正文「statistically significant」的
共享闸门（review 10 issue 8 起拆成三个独立判决）：
- `allowed` / `statistical_significance`：n≥6、原始 p<α、Holm p<α、
  族级最小可达 Holm p≤α 且测量值格式合法——**只**回答统计显著性；
- `stable` / `stability`：`|mean|>std` 与留一法符号稳定性——**稳定性**
  诊断，不参与统计显著性判定，也不得反过来写「无效应」；
- `practical`：只有调用方传入**预先声明**的 `min_effect` 才给出是否达到
  最小有意义改善；否则记为未测量（`None`），不得由 p 值推断。
`build_report._sig_gate` 同步改为只要求统计显著性，并把 mean/std/稳定性
交给表格与报告文字单独呈现。**v12 的 CONFIRMED 规则不使用
`|mean|>std`**：它严格按冻结清单的 decision_rule（p_holm<α、方向符合、
`param_matched is True`、逐种子 `init_core_fp` 相等、n≥8），由
`v12_supp._frozen_rule_confirms` 单一实现；`reportable_claim` 与
`build_report._sig_gate` 只服务探索性面板的措辞。`attach_holm` 同时输出
`family_min_attainable_p_holm` 与 `family_resolution_limited`：当 m 个
比较 × n 个种子的最小可达 Holm p 已 >α 时，p_holm=1.0 是分辨率上限，
不是零结果，报告必须这样写。v10 的 NIAH-natural 对照此前在 Holm 之后才
构造，已改为并入同一族重新校正。

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

## 26. 评审五（review 5，2026-11-14）：把残余的守卫缺口关死

本轮不新增实验面板；统计核心（精确符号翻转 / Holm / TOST / Wilcoxon /
功效表 / 交叉点拟合）经独立参照逐函数验证**无数学错误**，因果性与数据
隔离（39 + 50 项检查）亦通过。修的是审计发现的守卫与协议缺口——全部
发生在任何 test-split 访问之前，`CODE_SEMANTICS` 保持 `v11.128`（本轮
只改守卫、披露与元数据字段，任何合法路径的测量值不变，既有记录不失效）。

### 26.1 死门限与浅校验

- **JS 散度硬门**（`audit_corpus_split`）：nats 尺度上界是 ln2≈0.693，
  旧 `shift_fail=0.75` 永远不可达——「永远不会触发的门不是门」。
  `categorical_shift_report` 现返回 `js_divergence_normalized`（js/ln2，
  [0,1]），门限改在归一化尺度上判定，并对阈值本身做可达性校验。
- **`load_wikitest` 命中路径**曾只与字典序第一个 train tokenizer 比对
  全词表；现与缓存目录里的**每一个** train tokenizer 比对（与冷构建
  路径同规则）。
- **RoPE×内核**：`HybridAttentionRoPE.forward` 的 rope csa/hca 分支
  永远走逐行 eager 循环，但从不检查 `CSA_KERNEL`——内核模式下该臂的
  记录会带着 `-tk1` 内核戳而数值实为 eager（假"kernel vs eager"对照）。
  现在拒绝运行（v12 计划本就全局拒绝 CSA_KERNEL，m0/B 档臂全部非
  rope，无既有合法路径受影响）。
- by-length 准入中 `eval_span` 改为**必填**：缺字段 = 无法证明评了
  全长，不可采纳（诚实写入方 `eval_length_gen` 恒带该字段）。
- P5T 的 bylen cell 现显式复核 checkpoint 的 (d, L, H, Dh) 与 per-arm
  `mr` 是否等于冻结绑定值——p1l 配方字符串是注入常量，不能证明权重
  的真实形状（探针路径早已硬拒绝，此处对齐）。

### 26.2 「一次性访问」的两条旁路（本轮最重要的修复）

- **目录级**：`.freeze_lock.json` 只活在 gitignore 的结果目录里，改名/
  删除目录即静默重置冻结。锁现在**镜像到 git 管理的
  `confirmation_freeze_lock.json`**：镜像在场而本地锁缺席 → 拒绝
  （V12_REFREEZE=1 显式重冻结并在锁内保留 superseded 历史）；镜像与
  本地锁身份不一致 → 拒绝。镜像应随证据提交入库。
- **cell 级**：锁绑定计划/切片/语料，但不绑定 checkpoint 集合——
  「看结果→挑一个 (arm, seed) 用 P5C 重训→重跑 P5T 原键覆盖」曾畅通
  无阻。现在：P5C 在锁存在后拒绝重训冻结臂；P5T 拒绝在**不同
  ckpt_sha256** 的权重上重测已有 cell（覆盖须 V12_REFREEZE=1，且新
  记录携带 `replaced_prior_ckpt_sha256` 审计字段）。

### 26.3 覆盖开关与加严规则的落盘披露

- `V12_ALLOW_UNHEALTHY=1` 现写进 `training_health.json`
  （`v12_allow_unhealthy_used`）、每条被放行 cell 的 bylen 记录
  （`allow_unhealthy_override` + `health_status_override`）与
  experiments.jsonl 事件——不再只是 stdout。
- 决策规则的版本轨迹（按顺序读）：review-5 的实现曾**擅自**比清单 v3 更严
  （额外要求 same_init_core_fp 与 init_core_fp_recorded；param_matched 必须
  为 True 而非"非 false"）。只会少确认，但属于未注册偏离，review-6 已把实现
  恢复到注册文字，清单 **v5** 更正了历史注记。**v6（2026-11-24，仍在任何
  test 访问之前）**把加严正式注册：CONFIRMED 现要求 param_matched is True
  （不可验证的 None 不再等同已验证），且逐 seed 的 init_core_fp 必须记录并在
  每一对内相等；实现与清单文字一致，不再有"实现更严"的偏离。v6 同时修掉
  一个真实缺陷：旧的 `same_init_core_fp` 把**所有 seed、两个臂**的摘要并进
  一个集合，而正确的 12-seed 面板每个 seed 本来就有各自的共享主干初始化，
  所以该字段在正确面板上恒为 False，既无法发现真正的臂间不匹配、又会误报
  正确面板。现在改为逐 seed 比较两臂（`init_core_fp_pairs_checked` /
  `init_core_fp_mismatch_seeds`），面板级 validity 也改为按 seed 分组。
  只会少确认、不会多确认，`test_stats.json` 仍携带 `decision_rule_applied`
  （清单原文 + 实现闸门清单 + 说明）。
- 清单 **v4**（仍在任何 test 访问之前）：ee/es 移出可机检配方签名
  （它们是诊断监控旋钮；P5C/p1l 在 P5H 所需诊断下合法携带 es=8，
  v3 的 es=128 钉定在 p1l 路径上不可校验且与实际记录不符）；
  tokenizer.fit 补充 8M token 上限说明。
- `v12_supp.py power`：把清单 power_note 的义务变成可执行物——访问前
  在冻结臂的验证集逐种子 delta 上做 2^n 符号翻转枚举（固定 |d_i|），
  报告 Holm 保守阈值下的功效(q) 曲线与最小同向种子数，落盘
  `p5c_power.json`；P5T 在读 test 前要求该工件在场
  （V12_SKIP_POWER=1 为显式、落盘记录的覆盖）。**v6** 起清单
  `statistics.power_gate`（q=0.80，min_power=0.80）把它升级为硬门：逐比较
  的 panel 功效缺失或低于下限时 P5T **拒绝**访问，除非显式
  `V12_ACCEPT_LOW_POWER=1`；该决定写入 `results_lm_test/power_gate.json`
  并带进 `test_stats.json`。12-seed 设计自身 q=0.80 的 Holm 保守功效只有
  0.275，所以这道门通常会触发——诚实做法是加种子并记录修订，或显式接受
  「分辨率受限」的确认（覆盖不等于设计已有足够功效）。

### 26.4 度量口径与记账

- **ngram 地板双分母**：`ngram_lm_score` 保留历史口径（含每行首
  token，R×L），同时返回 `ppl_matched_denominator`（剔除行首 token，
  与神经 LM 的 R×(L−1) 评分位置一致）；`methodology_audit` 的
  beats-floor 比较改用匹配分母并标注口径，旧工件无该字段时回退并
  标注「历史口径，地板略被抬高」。
- **test 去重归属**：`_DocDeduplicator.snapshot()/peek()` 提供只读
  探测；test 行之间的内部重复被剔除时计入
  `test_internal_{exact,near}_removed`，不再误记为 train/val 污染
  （剔除行为本身逐字节不变）。
- **探针记录**携带 `manifest_sha256`（v12 分析侧校验一致性）；健康门
  拒绝的 (arm, seed) 不再被探针消耗一次性访问。
- **TimeGuard**：被 watchdog 击杀的 cell 以 `record_run(elapsed,
  steps_done=0)` 记账真实墙钟（不动步速标定）——kill-and-retry 不再
  系统性低估 GPU 预算。
- 探针腐坏 ρ→种子的 1/2²⁰ 量化在冻结网格 {0, .125, .25, .5} 上无
  碰撞（精确二进制分数）；非二进制网格扩展时需注意。

### 26.5 其余健壮性

`_successor_starts` 向量倍增路径补 power-of-two 不变量断言（当前探测
长度恒为 32，满足）；`blocks_from_cosine` 的 n≤1 捷径补 `_pool_cpu`
挂载（与 `blocks_from_cuts` 同口径）；删除死代码 `_SinkWiden` 与
`eval_ppl` 的 `_span` 死赋值、par 的 `_num_or_none`；
`gathered_attention_open` 的 `_soft_live` 与闭式路径同样检查
requires_grad（前向值不变，省无谓子图）；`CSA_ADAPTIVE_MAX_TOKENS`
网格上限在「能影响阈值取值」时（seq 超过精确上限或显式设置）进入
run 指纹（默认主线路径指纹逐字节不变）；`holm_min_attainable_p` 的
ldexp 溢出改为守卫返回 1.0；鞍点近似的实测误差界（p∈[0.002,0.04] 内
≤2%）写进 docstring；内核路径 `bincount` 补 `.long()`（整数值不变）。

## 27. 评审六（2026-11-24）：把「已披露」变成「已拒绝」，并关闭单位错标

本节记录本轮代码级修复。它不移除任何既有主张，也不改变 v12 冻结清单的
数值字段；所有改动都在**首次 test 访问之前**，且由
`test_review_fixes.py` 的 N87–N95 与既有 377 项测试共同锁住。

### 27.1 计算量匹配：代理比值不得当证书

`_compute_control_detail` 过去在只有 `params × tokens`（6ND）代理时，对
参数对齐的一对臂返回 `compute_matched=True`。参数对齐的 sparse/dense 臂
可以相差 ~1.87× 实际 FLOPs/token（清单 `compute_budget_disclosure` 自己
就写了这一点），所以那个 `True` 会被下游读成「等计算量」。现在：

- 代理路径只报告 `proxy_ratio_a_over_b`，`compute_matched=None`、
  `compute_verifiable=False`，并明确写「equal compute is NOT established」；
- 只有**实测**的 per-token 墙钟落在容差内才置 `compute_matched=True`
  （`compute_verifiable=True`），且 `flops_matched` 仍保持 `None`——
  墙钟不是 FLOPs；
- 聚合报告在代理路径新增 `compute_unverified` 警告清单，要求把该对比写成
  「参数对齐、计算量未验证」。

P0 的「实测 FLOPs 匹配 Dense 臂」仍需 GPU profiler 与新的冻结清单，本轮
没有伪造一个解析式匹配臂。

### 27.2 文档/窗口单位：新增真正的文档聚类路径

v12 的 16 条评测行是从去重后 test 流头部切出的**固定窗口**，不是 16 篇
文档；一个 4096-token 窗口可以跨多篇文档。旧代码把它们叫
`per-document`、`doc_sampling`、`document bootstrap`，算术没错但单位错标。
本轮：

- `load_wikitest(..., with_doc_ids=True)` 返回 `(ids, vocab, tok, bnd,
  doc_index, doc_ids)`：`doc_index` 是逐 token 的文档序号矩阵，
  由构建时记录的 `test_doc_lengths_<tag>.npy` 精确展开（缓存命中时校验
  文档数/总 token/覆盖；缺失则 fail-closed，不重读 test 去重建）；
- `eval_ppl(..., doc_index=...)` 与 `eval_length_gen(..., doc_index=...)`
  把逐 token NLL 按真实文档聚合，输出 `doc_nll/doc_ntok/n_docs/doc_ppl/
  doc_ci`；`cluster_unit` 明确写 `document (...)`；
- 逐 token 序号在流内单调，聚合走确定性分段和（无 CUDA atomics）；
- v12 的 `doc_sampling` 现在明确标 `unit = fixed evaluation windows`，
  `ci_method` 改为「over rows/windows, NOT over documents」，并保留
  `n_docs` 字段做兼容；真正的文档级结论属于 follow-up 清单。

这回答了「16 行不足以支撑总体泛化」的代码侧部分：单位不再被误标，且
后续研究有了可执行的文档级聚合入口；v12 自身仍只声明覆盖该冻结切片。

### 27.3 纯 HCA 臂与 follow-up 注册表校验

`hybrid_*_rope` 是 CSA/HCA 交替层，不能回答 HCA-vs-CSA。新增
`hca_fixed_rope` / `hca_dynamic_rope`（每一层都是 `kind='hca'`），并加
`PURE_HCA_VARIANTS` / `FACTORIAL_VARIANTS`；`validate_followup_design.py`
新增 `--verify-registry`（`--require-ready` 隐含）：导入仓库注册表，逐臂
检查变体可解析、层类型与声明族一致、fixed/dynamic 与声明一致、Dense 控制
确为 `full` 层。hybrid 冒充 HCA、TODO 变体、缺失变体都会在设计阶段被拒绝。

`followup_design.template.json` 的 HCA 单元已指向真实变体；FLOPs-matched
Dense 仍是 TODO——因为协议要求**实测** FLOPs，不能用一个未标定的解析式
变体代替。

### 27.4 一次性 test 访问：从「preaccess」到「reading」

旧流程先写 `stage='preaccess'`、再 `load_wikitest`、成功后才写
`stage='accessed'`。若读取后形状/泄漏审计失败，锁仍是 preaccess，
`_lock_proves_test_access` 返回 False，重试会**再次读取** test——一次性
访问名存实亡。现在：

- 读前写 `stage='reading'`（不是 preaccess）；`_lock_proves_test_access`
  对 reading 返回 True；
- 新增 `_refuse_repeated_test_read`：若锁停在 reading 且没有绑定
  `test_ids_fp`，任何重试都在读之前被拒绝（`V12_REFREEZE=1` 是显式、
  打印的覆盖，并把上一轮降级为探索性）；
- 失败路径不再因为「preaccess 可改清单」而放行；一旦开始读，清单就不能
  再改。

### 27.5 准入与签名的 fail-closed 收口

- `_recipe_matches_signature` 对结构计数（steps/sl/bs/nt/v/wu/d/L/H/Dh/
  ee/es）改为**精确整数**比较：`steps=3000.002`、`bs=12.000001` 不再借
  浮点容差混入；只有 lr/wd/cl/dlm/mr 保留格式化容差。
- `_load_manifest` 要求带 `binding.training_recipe_signature` 的清单
  必须有 `manifest_version` 且为 5；未来 schema 不得按旧语义解释。
- 清单声明 `binding.checkpoint.hash` 时，真实 P5T 记录必须携带 64-hex
  `ckpt_sha256` 才可进入确认统计（`plan` 的合成记录显式标
  `plan_synthetic`，只跳过这一项）。
- 清单声明 `eval.per_document` 时，准入要求每个可用 `by_len` 单元都带
  `ppl_per_seq`（长度 = 冻结 `n_seqs_clean`、值为正有限）——不再允许
  报告阶段发现「文档不确定性」字段缺失。
- 所有可采纳记录的 `train_knobs.det` 必须一致；混合确定性模式不能进入
  同一确认对比。

### 27.6 其它已核实修复

- `ce_loss`：目标列数多于输入列数时直接拒绝；分母只数**被评分**的列
  （旧代码在 targets 更宽时会静默把 CE 缩小 `ids_cols/targets_cols`）。
- `csa_kernels`：融合池化的公开入口现在拒绝非连续/非单调 `bid`（eager
  会内部排序，融合核不会，旧路径会静默池化错误的 token 集合）；
  `sparse_layer_forward` 对 `open_block_mode != 'closed'` 直接拒绝，不再
  依赖 `HybridAttention.forward` 的唯一守卫。
- `audit_gates.audit_causality` 覆盖 `csa_dynamic_rope`、`hca_fixed_rope`、
  `hca_dynamic_rope`，并新增「固定长度下扰动未来 token」的直接前缀检查；
  HCA 的末位浮点弥散（~1e-6）在 1e-4 容差内，决策与块表相同。
- **更正一个披露数字**：randidx 臂的死参数为
  6 × (16,384 `W_DQ` + 16,384 `W_DK` + 1,024 `W_w`) = **202,752**
  （11,089,904 的 1.83%），冻结清单里写的 201,216 少算了 1,536
  （6 × 256）。清单本身是已发布预注册，本轮不改其字节；以本笔记与
  `count_params` 实测为准，报告引用时写 202,752 / ≈1.8%。
- `v12_supp.py` 的异常兜底不再对 `plan/smoke/power/analysis/report`
  执行 `git add -A` + commit + pull --rebase + push：零 GPU 预检失败绝不
  应把开发树发布到 main（本次审计中真实发生过一次）。长训练模式要推送
  部分结果需显式 `V12_AUTO_PUSH=1`。

## 28. review 10（外部方法学审查）的逐项处置与事实更正

本轮不新增训练面板、不改冻结清单字节、不改任何测量语义。逐项处置：

1. **证据缺口（issue 1）**：仓库仍为 source-only。新增 `exp_lib.evidence_
   release_status` / `evidence_status_banner` 与 `build_report` 的本地
   孪生：所有报告头必须打印「无完整证据发布 ⇒ 数字不能独立审计」；完整
   发布行的判据是 tag/date/commit/URL/archive sha/manifest sha/verifier
   均非占位，且 `archive_release.py verify` 通过。审计函数不能替代真实
   工件，本仓库未声称数字可审计。
2. **主对照非单因素（issue 2）**：冻结清单已有 `interpretation_caveat`
   与 `compute_budget_disclosure`；本轮把它们**原样打印在 v12 报告主表
   旁**，并给每个主比较写入 `single_factor_contrast=false` /
   `system_level_only=true` / `claim_scope`。CONFIRMED 只能读作「该冻结
   配方下的系统级臂间差异」；机制归因需要 selection-attribution 工件与
   同骨干/同投影/同归一化/同 kernel 的隔离对照。
3. **随机选择对照（issue 3）**：审查声称「新版已改为按 (seed, layer)
   变化的伪随机模式」——**在当前 HEAD 与所有 ref 中不存在**；实现仍是
   固定确定性哈希 `(row*12.9898 + col*78.233)`，对每 seed/layer/step
   完全相同（冻结清单 caveat 已如实写明），不是每次前向重随机。随机臂
   的 `W_DQ/W_DK/W_w` 是**死参数**（202,752，≈1.83%）：本轮让
   `param_breakdown` 计入 `dead_params` 并令 `active=total−dead`，
   `finaltest_stats` 逐臂报告 active/相对差/`active_param_matched`，报告
   打印「total 匹配 ≠ 有效容量匹配」。因此该对照最多支持「学习选择优于
   该固定伪随机连接模式」；不支持「选择器找到语义相关信息」，也不代表
   所有随机稀疏基线。
4. **动态分块语义（issue 4）**：默认 `csa_dynamic` 不是任务学习边界——
   §22.5 已写明 `gate.detach()>0.5` + `comp_reg` 训练偏移；冻结范围不含
   动态/HCA；follow-up 仍是 draft（`--require-ready` 拒绝）。本轮仅在
   README_zh 补齐与英文 README 相同的「非任务学习」负向声明。
5. **score-function 符号（issue 5）**：当前文档与代码均为
   `+(CE − baseline)·logπ / n_decisions`（CE 是最小化成本，detached），
   **不是**审查所称的负号；`exp_lib.boundary_policy_term` 把该式抽成
   可测试函数，训练循环原样调用。新增两动作玩具测试
   `t_score_function_term_sign_two_action_toy`：低损失动作概率在正确符号
   下上升、在反号反事实下下降——把「文档疑点」变成可执行核验。
6. **符号翻转假设（issue 6）**：§20 已改写为「尖锐零假设 = 配对差符号
   关于 0 对称」；明确指出共享 seed/数据顺序/评测行/初始化只是控制，
   不提供随机分配意义上的交换性。`finaltest_stats.validity` 新增
   `null_hypothesis` / `assumption_boundary` / `machine_checked_controls`
   与逐比较 `positive/negative/zero` 符号计数、留一法均值符号稳定性、
   `within_seed_noise`（稳定性诊断，非显著性判定）。
7. **功效设计（issue 7）**：审查称门槛用 q=0.95——**事实错误**：冻结
   门槛是 `statistics.power_gate {q:0.80, min_power:0.80}`，q=0.95 只是
   四列设计辅助表之一。真实缺口是：验证集挑选的效应可能高估 test 效应、
   固定 |Δ| 的符号枚举不建模训练随机性、低功效 override 的报告呈现。
   本轮给 `p5c_power.json` 增加 `power_sensitivity`（观测同向率 q_hat 与
   Wilson 下界处的功效、留一 seed 功效范围、去掉最大 |Δ|/替换为中位幅
   度的功效、均匀缩放的说明），`V12_SKIP_POWER=1` 在清单声明
   `power_gate` 时必须同时 `V12_ACCEPT_LOW_POWER=1` 并写
   `power_gate.json(status=skipped)`；v12 报告打印 `accepted_low_power`。
   另更正 `signflip_power`/`signflip_min_consistency` 文档中「等幅投影
   是精确功效上界/下界」的错误断言（8×(+100)+4×(−1) 反例：精确 p=0.0078
   vs 等幅投影 0.388，方向可反转）。
8. **`|mean|>std` 混用（issue 8）**：`reportable_claim` 拆成
   statistical/stability/practical 三个独立判决（见 §25.1）；
   `_sig_gate` 只要求统计显著性；v12 CONFIRMED **从不**使用该稳定性
   阈值（§25.1 已更正原先「同一口径」的错误表述）。v12 未预注册
   `min_effect`，因此 v12 报告不得作「实际价值」声明。
9. **单位（issue 9）**：冻结 v12 的 16 个评测单位是**固定窗口/行**，
   不是文档；`doc_sampling` 明确 `unit=fixed evaluation windows`。
   `48` 是 arms×seeds=4×12 的记录数，**不是 48 篇文档**；冻结清单无
   「48 documents」字段。`MIN_DOCS_FOR_POPULATION_CLAIM=64` 是仓库自设
   最小总体门槛（无统计推导，且计数单位是行/窗口）。DECISION_LOG 与
   build_report 中的「文档」措辞已改为「行/窗口（非文档）」；冻结清单
   `eval.per_document` 的字段名属预注册字节，不改，按 §27.2 与本
   errata 解释为逐行/窗口。
10. **长度外推（issue 10）**：PPL@Ln 是**前缀嵌套目标**（512 ⊂ 1024 ⊂
    …），不是同目标跨长度比较；读取比例随长度下降（512 25% → 4096
    3.1%）。报告已并列写出目标定义、块数/top-k/读取比例与替代解释；
    局部窗口基线（`full_sw128*` 已注册）、逐长度远端移除 ΔNLL、固定目标
    位置跨长度探针与距离分桶选择读数**尚未在冻结面板提供**，因此长长度
    优势只能作为「该配方下的描述性外推读数」，不能归因为「更会利用长
    上下文」。
11. **干扰探针（issue 11）**：验证集 P3MP 早已以 ΔNLL/ΔΔNLL + 干净基线
    为核心，非绝对 PPL；但 v12 test probe 原先只有绝对 PPL。本轮在
    `_test_probe_stats` 用已存储的 `nll_sum/n_tok` 重建同臂干净 NLL 与
    `delta_nll`（描述性、不进 Holm），报告并列打印。§15 补充 uniform 的
    OOD 性质、donor 单位=固定窗口、topical 只是主题代理且不保证答案无关、
    mask/delete 改变长度/分布、跨模式同 ρ 不同强度等限制。
12. **span 覆盖（issue 12）**：审查描述的「允许重叠」在 HEAD 已不存在
    （`6646c91` 起按 tile 切分、段不重叠）；真实缺口是**未记录**实现覆盖
    /实际改变 token 数。本轮 `_corrupt_row(return_audit=True)` +
    `_corruption_audit` 落盘覆盖槽位/唯一位置/重叠/实际改变/空操作行/
    供体行，报告打印「实际腐坏几何」表；§15 同步说明 ρ 是名义值。
13. **NIAH（issue 13）**：新增 chance 基线（1/可用 value token 数）、
    `nofar` 无远端信息对照（摧毁远端耦合，准确率应退回 chance）、距离
    分桶准确率（short→long 梯度）与任务有效性机器门：clean 显著高于
    chance + nofar 与 chance 一致 + 记录了解题正对照，三者缺一则 NIAH
    只能作描述性读数、不得作为「长距离利用」的机制证据。当前面板没有
    正对照，因此门必然为 false——这是诚实收紧，不是新结果。
14. **规模/收敛（issue 14）**：§16 补充宽度/深度/预算不可分离、等 token
    切片不控制 LR 形状与数据重复、交叉点依赖评测窗、拟合是均值轨迹而非
    逐 seed 交叉均值等不可分离性；README 的「compute-matched」措辞改为
    「6ND 参数量代理预算（非实测 FLOPs）」。
15. **冻结/调参/健康门（issue 15）**：新增 `round_is_confirmatory`：
    冻结锁含 `superseded`（post-test `V12_REFREEZE=1`）时，**当前轮**
    也强制 `confirmatory_round=false`、`confirmed=[]`，报告显著警告
    「重冻结不恢复 test 独立性」；只保留 `would_confirm_under_decision_
    rule` 作探索性记录。`BASELINE_TUNING_PROTOCOL` 的「固定配方确认 ≠
    两侧调参后的最佳方法比较」限制补入 README/本笔记；health-gate 的
    override 已持久化，v7 P1L 的「为取得 health 证据而重训」补写实验事件。
    另修正若干处指向不存在的 `REVIEW_4_RESPONSE.md` 的引用与
    `confirmation_frozen.json` 的「plan v2」陈旧措辞。

**审查中与当前 HEAD 不符的事实**（保留记录，避免下一轮重复）：`Holm 六比较`
（实为 m=4）、`48 documents`（实为 4×12 条 arm-seed 记录）、`-(CE-baseline)`
（实为 `+`）、默认 `csa_dynamic` 被称作 task-learned（文档明确否认）、
随机模式按 (seed, layer) 变化（实为固定哈希，所有 ref 一致）。

## 29. review 11（外部方法学审查）的逐项核验与代码处置

本轮是**核验优先**的处置：先逐条对照当前 HEAD 的代码/产物，再决定改什么。
没有新增训练面板，没有读取 test split，没有改 `confirmation_frozen.json`，
也没有改任何测量语义（`CODE_SEMANTICS` 仍为 `v11.128`）。审查中的方向性
结论大部分成立，但若干“代码里没有”的具体断言在当前 HEAD 已不成立；
真正的代码缺陷集中在校验/报告层，本轮已修并加了回归测试。

| # | 审查意见 | 是否成立（对照 HEAD） | 本轮处置 |
|---|---|---|---|
| 1 | 外部有效性：小模型/WikiText 不能外推到 LLM、预训练、DeepSeek、生产 | **成立（研究设计边界）**；README 顶部、§20、冻结清单 `scope` 均已声明 | 代码不能消除；报告/README 继续把主张限定在冻结范围 |
| 2 | 未同时控制参数量、FLOPs、显存、优化难度 | **部分成立**；`full_matched`/`full_sw128_matched` 做了 1% 参数量匹配，`compute_budget_disclosure` 明说 +69% MLP / ~1.87x FLOPs / 读取比例不匹配；等 FLOPs 臂仍在 follow-up 的 TODO | 未新增算力臂（需新 run）；报告继续把 v12 主比较标为 `system_level_only` |
| 3 | 1500 步/20k/40k 预算不足以支持“最终/渐近线”结论 | **部分成立**；报告已拒绝“渐近线”判定，只报窗口内读数 | 无代码改动（需要更长训练/LR 续训的新实验） |
| 4 | 2–4 seeds 的精确检验 p 下限 | **成立（设计限制）**；`reportable_claim`/`signflip_p_floor` 已机器化并拒绝把未显著读成无效 | 无代码改动（补种子是新实验）；报告继续打印下限 |
| 5 | 大量比较未做 FWER/FDR；exact 只保证单检验 | **部分成立**；族内 Holm 一直存在，`build_report` 也有跨面板 BH/Bonferroni；但 v10/v11 独立报告没有族级下限/全局视图，v10 的 ΔNLL/NIAH 表只印原始 p | **已修**：v10/v11 用 `attach_holm` 保留族级分辨率元数据并写进报告；新增跨面板 BH-FDR + Bonferroni；v10 的 ΔNLL/NIAH 表补 `p_Holm` 列 |
| 6 | v7–v11 结果导向/选择性停止；预注册不足 | **部分成立**；v12 有冻结清单/一次性 test，但 follow-up 没有停止规则 | **已修**：`followup_design.template.json` 增加 `single_look_fixed_n`/`n_looks=1`/`interim_analysis_planned=false`；`validate_followup_design.py` 在 `--require-ready` 路径强制；协议写明 |
| 7 | 长度外推：absPE 超过 max_pos 只取最后一段，和 RoPE 不可比 | **不成立（事实错误）**：`SmallGPT.trunk` 对超表输入直接 raise；v7 评估显式 `truncated=True`，报告把 absPE 排除出外推对比 | 无代码改动；保留警告 |
| 8 | 干扰注入只测均匀随机 token，不能代表真实干扰 | **部分成立**；实现已有 uniform/natural/mask/delete/span/doc/topical 七种模式，并记录实际覆盖/改变几何；冻结 v12 探针确实只用 uniform | 无代码改动（冻结字节不动）；报告继续限定为“该腐坏模式下的响应” |
| 9 | random/allblocks/dense 对照不能隔离机制 | **成立（设计限制）**；random 是固定哈希且有死参数，allblocks 改变读取范围/STE，dense 同时改表示/参数/核 | **已修报告措辞**：v10/v11 不再写“独立贡献/机制结论”，改“系统级差异”，并写明需要 selection-attribution 工件；fuse 报告标题同步降级 |
| 10 | hard top-k/no_grad；未选块无注意力梯度；无 oracle/soft/recall | **成立（方法定义 + 证据缺口）**；`audit_gates.py selection` 是 fail-closed 合同，当前没有工件 | 未伪造证据；直接机制验证需要新探针/新测量 |
| 11 | 动态分块混杂；未报告每样本块数/长度/读取 token | **部分成立**；`compression_report` 已有块数/长度/分位数和边界 F1，`audit_gates` 有块数匹配检查；逐样本读取 token 未落盘 | 未新增探针；报告保留现有聚合诊断并说明限制 |
| 12 | 交叉点拟合点少、混杂、删失、平滑依赖、n=2 R²=1、无 CI | **部分成立**；HEAD 已强制 ≥4 个去重宽度、拒绝删失/训练量混杂、给斜率 CI/LOO；`n=2` 不会给 R² | **已修**：v10 交叉点新增 ±0/±1/±2 平滑敏感性并写进报告；Tobit/逐 seed 斜率仍是方法扩展 |
| 13 | tokenizer/数据独立性、验证集复用、无独立 test 流程 | **不成立（事实错误）**；tokenizer 只从 train 前缀拟合，缓存命中 fail-closed 校验，`load_wikitest` 有一次性 test 流程（只是尚未执行） | 无代码改动 |
| 14 | `eval_subset=128` 太小、只把 seed 当独立单位、无序列 bootstrap | **部分成立/表述过时**；128 只控制训练曲线，最终 PPL 用完整固定验证批；窗口 bootstrap 已实现；冻结面板确实是 16 个固定窗口 | **已修**：训练健康表的“documents/document-level”误标改为“evaluation rows/windows”；16 窗口限制继续披露 |
| 15 | paired seed 不是严格 block；sign-flip 零假设是符号对称 | **成立（假设边界）**；§20 已写明；本轮还发现并修复一个真正的配对 bug | **已修**：`build_report._pair_condition_tail` 把**每 seed 不同**的 `init_core_fp` 当成跨 seed 分组键，导致正确面板被拆成每组 1 个 seed、n 从 4/5 掉到 1；现在只在 seed 内由 `_pair_reason` 校验，新增回归测试；v10/v11 报告补假设边界文字 |
| 16 | 旧代码/旧配方/重构记录仍可能被引用 | **成立（真实代码缺陷）**：`sec_stats` 不看 analysis 的 code，`sec_training_health`/`sec_efficiency` 不过滤 `_code`/`synthesized`，`per_seed_records` 无 code 过滤，v8/v9 面板读取默认不过滤 | **已修**：`sec_stats`/`sec_flops` 的 analysis 戳不匹配即整节拒绝；health/efficiency 过滤 stale/synthesized；`per_seed_records` 要求当前 `_code`；`ppl_by_seed(require_code=True)` 供 v8/v9/v10/v11；`v9_supp.add_paired` 拒绝 stale code；`_CUR_CS` 不可读时 fail-closed |
| 17 | results/analysis/REPORT 不入库，无法独立审计 | **成立（证据缺口）**；仓库是 source-only，`RELEASES.md` 无发布行；`archive_release.py` 机制存在 | 代码不能替代真实产物；报告头继续打印“不能独立审计” |
| 18 | 依赖/环境未锁定，import 时装包/HF mirror | **部分成立/表述过时**：import 时 pip install 与 HF mirror 都已改为显式 opt-in；CPU lock 精确；CUDA 环境仍未 lock | 无代码改动；发布流程应归档 `pip freeze`/GPU 信息（`archive_release.py` 已做） |
| 19 | 效率结论缺真实测量 | **部分成立**；`measure_efficiency` 有 warmup/重复/peak memory，报告区分解析 FLOPs；energy 未测 | **已修报告**：明确“Pareto 描述性、跨 shape/device 不可比、energy 未测”；未新增 GPU 测量 |
| 20 | 缺 model-behavior/mechanism 直接验证 | **成立（研究设计限制）**；selection recall/oracle/distillation 未实现；NIAH 有 chance/nofar/正对照门，当前门为 false | 未伪造机制证据；`audit_gates.py selection` 继续 fail-closed |
| 21 | 方向性证据被写成因果结论 | **部分成立**；v12 已有 system-level 限制，但 v10/v11/fuse 仍有“独立贡献/机制结论”字样 | **已修**：v10/v11/fuse 的标题/结论改为系统级读数并加归因边界；`build_report` 去掉“效应>噪声”当显著性门槛的旧措辞；`methodology_audit` 显式输出 `significant_p_only`/`stable_global` 两个判决 |

**审查中与 HEAD 不符、应保留记录的事实**：`eval_subset=128` 不是最终 PPL
评估（最终用完整固定验证批；冻结面板是 16 个窗口）；`n=2 → R²=1`/“无斜率
CI”在 HEAD 不成立（≥4 点、CI/LOO、k=2 不给 R²）；“import 时自动装包/强制
HF mirror”已改为 opt-in；“没有 Holm/FDR”“没有 window bootstrap”“48
documents”“absPE 静默截断”均不成立。审查对**“参数匹配 ≠ 计算匹配”“hard
top-k 的选择梯度只覆盖被选块”“缺少 selection-attribution”“source-only
不可审计”**的判断成立，这些是本轮保留的限制声明，不用代码伪装成已解决。

本轮代码改动（全部零实验、零 test 访问）：

- `build_report.py`：修复 per-seed init 分组 bug；stats/flops/health/
  efficiency/per-seed pairing 的 stale-code 门；单位措辞；效率/因果措辞。
- `exp_lib.py`：`ppl_by_seed(require_code=...)`；`methodology_audit` 分离
  `significant_p_only` 与 `stable_global`。
- `v8_supp.py` / `v9_supp.py` / `v10_supp.py` / `v11_supp.py`：面板读取
  要求当前 code；v9 paired 拒绝 stale code；v10/v11 族级分辨率 + 跨面板
  BH/Bonferroni；v10 p_Holm 列、平滑敏感性、系统级措辞；v11 系统级措辞。
- `fuse_analysis.py` / `open_block_supp.py` / `v9_supp.py`：报告头补
  `evidence_status_banner()`；fuse 标题降级。
- `followup_design.template.json` / `validate_followup_design.py` /
  `FOLLOWUP_CONFIRMATORY_PROTOCOL.md`：单次固定 n 的停止规则。
- `test_review_fixes.py`：新增 per-seed init、stale-record、stopping-rule
  回归测试（N97 等）。

## 30. review 12（外部方法学审查）的逐项核验与代码处置

本轮仍是**核验优先**：先逐条对照当前 HEAD 的代码/产物，再决定改什么。没有新增
训练面板，没有读取 test split，没有改 `confirmation_frozen.json`，也没有改任何
测量语义（`CODE_SEMANTICS` 仍为 `v11.128`）。审查的 16 条意见中，设计边界类
（探索/确认未分离、小 n、多因素混杂、validation 复用、source-only）**成立但
不能用代码消除**；其余条目大多已在 v10/v11 硬化，本轮修掉的是**仍能在当前
代码中复现的缺陷**。

| # | 审查意见 | 对照 HEAD 的结论 | 本轮处置 |
|---|---|---|---|
| 1 | 探索/确认未严格分开；v6–v11 后验选择；旧报告「机制被证伪/负结果更稳固」应降级 | **成立（研究设计边界）**；报告头已标 exploratory，v12 是唯一冻结确认 | 修掉残余的**正向机制归因**（长度外推的「机制一致/滤掉远距噪声」、P0-3 的「证伪」、`learned boundaries … semantic`）；不能追溯预注册 |
| 2 | 2–4 seeds 分辨率不足；v7–v11 无族级校正 | **部分成立/表述过时**：`reportable_claim` 的 n≥6 门、Holm 与全局 BH/Bonferroni 已存在；但 v7 `sec_stats` 不打印 n/下限，v10 全局族漏掉 scale/equal-token，v9 无全局视图 | `sec_stats` 增加 n/符号/p 下限/配对与非配对 bootstrap/Wilcoxon 与声明族；v10 全局族补全；v9 增加全局 BH/Bonferroni；P2S/P3T 补 `p_Holm` |
| 3 | 符号翻转的适用前提未说明；配对只是同 seed | **部分成立/已披露**：`paired_effect_stats` 返回 `unit='seed'`、对称零假设、LOO/CI/Wilcoxon，v10/v11 已打印；v7/v9 未打印且字段被丢弃 | v7/v9 报告补固定假设边界句；`aggregate`、v7 stats、v9 `add_paired` 落盘符号计数、`p_floor`、配对/非配对 bootstrap、Wilcoxon、`assumption` |
| 4 | 训练目标不同（comp_reg、delta LR 10×） | **成立（真实混杂）**：`train_objective` 已记录，但**漏掉每层 `comp_lambda_mult`**（`csa_dyn_nocomp` 的 mult=0 仍被记为 applied=True/0.05；hybrid HCA 的 4× 不可见），且 v7 配对与 v8–v10 `add_paired` 不消费目标描述符 | `train_objective` 记录每层有效权重；`objective_reason` 比较权重向量；v7/v9 对比携带 `objective_matched`/`objective_differences` 并在报告标注系统级边界（仍可配对，不做机制归因） |
| 5 | 参数匹配不全局；RoPE 面板容量不匹配；缺 8 项资源表 | **部分成立**：`full_matched`/`full_sw128_matched` 存在；当前 P0R/P1L 的 `full_rope` 已按 `csa_fixed_rope` 宽度构建（CPU 复核 11,088,512 vs 11,089,904，<1%）；旧记录仍有容量差警告。没有任何报告逐臂打印 breakdown/KV/目标 | 新增 `sec_resources`：逐臂 total/attention/indexer/MLP/active/dead/layers/KV elem/token/comp_reg 权重/ΔLR/tokens；明确「参数量匹配 ≠ 计算/有效容量匹配」，FLOPs 未逐臂实测。v12 的 active 差继续披露且不改变冻结判定规则 |
| 6 | 动态分块未端到端学习 | **成立（默认臂的定义），但已如实文档化**：`gate.detach()>0.5` + `comp_reg` 只训练标量偏移；`csa_dynamic_task` 是 opt-in 的 REINFORCE 边界臂，不在冻结范围 | 仅改残余措辞（boundary F1 不再是「genuinely semantic」）；默认臂继续描述为「余弦相似度 + 压缩率正则的阈值分块」 |
| 7 | 许多比较同时改多个因素 | **成立（设计边界）**：现有消融不是正交因子设计；v12 主比较标 `single_factor_contrast=false`/`system_level_only=true` | 不伪造正交性；`sec_resources` 把容量/目标/读取差异逐臂列出；正交 2×2 设计仍是 follow-up 草案 |
| 8 | 随机 indexer 只是评估时干预 | **不成立（当前 HEAD）**：P3MP 的 `ac.indexer_mode='random'` 是**测试时干预**，但 P3MTC 已从头训练 `csa_fixed_rope_randidx`/`_allblocks`，报告也明确区分两者并给出 retrain-gap。仍缺 oracle-block、同块数不同边界、冻结选择器 | 无代码改动（报告措辞已足够）；新增对照需要新训练面板 |
| 9 | 长上下文混淆（训练 512、位置编码、截断、目标区） | **成立（设计边界）**：absPE 超表会 raise/截断并标 `truncated`，报告只用 RoPE 臂比外推，但结论段仍有机制措辞、算术平均、无 n | `sec_p1l`/v10 P3MT 改为跨 seed **token-pooled PPL**（缺 NLL 才标 `*` 回退），打印 n/ratio 口径，把「结论」改为描述性读数，明确「512 训练长度外推探针，不是长上下文训练」；删除「机制一致/滤噪」解释 |
| 10 | crossover 证据弱、点少、判据后验 | **部分成立**：拟合已拒绝 <4 点/删失/训练量比>1.25，给斜率 CI/LOO/平滑敏感性，并声明描述性；但 d 与层数绑定、batch/seq 网格不同，v10 曾称判据「预注册」 | 报告明确横轴不是单一规模变量（128/4L…512/10L）、斜率混入深度/优化/评测网格；改「预注册」为「分析前声明的探索性判据」 |
| 11 | validation 反复用于选择；无最终 test | **成立（设计边界）**：v6–v11 全为 exploratory，v12 一次性 test 流程存在但**尚未执行**（无 `results_lm_test/`、无冻结锁、RELEASES 无行） | 无代码可填补；报告/README 继续限制主张；不能把 v12 的保护回填给 v6–v11 |
| 12 | PPL 用算术平均，未用 pooled NLL；缺 median/IQR/配对差 | **部分成立**：`aggregate` 主口径已是 pooled（含 median/IQR/CI），但 v7 长度/P2S、v9–v11 面板表仍用算术平均且不打印 pooled | v7/v10/v11 主表改为 pooled（或显式回退），v9 面板增加 pooled/median/IQR；`aggregate` 与 v7/v9 artifact 保留逐 seed 值与配对差 |
| 13 | 评估集小、噪声低估 | **部分成立**：最终 PPL 用完整固定验证批（`eval_subset=128` 只控制训练曲线）；行/窗口 bootstrap 已实现，但 v10/v11 不打印；token-level CI 不存在 | 长度/P2S 表打印行/窗口 CI 或 n；明确「CI 单位是行/窗口，不是 token/文档」；v10 探针仍是 16 seq，限制不变 |
| 14 | FLOPs 分析不能推出效率结论 | **部分成立**：解析 FLOPs 只含 attention；v7 报告有实测 prefill/prefix-prefill/峰值显存并声明 energy 未测；v10–v12 不展示 | `sec_resources` 增加结构资源与「非实测」声明；v10–v12 未新增效率测量（需要真实 checkpoint/GPU） |
| 15 | 结果不可审计、无不可变证据包 | **成立（source-only）**：`archive_release.py` 存在，但无发布行/工件；release 解析过宽（任意 7 个非占位单元即 `released`），归档可只含 analysis，且漏掉 TimeGuard 账本 | 严格 release 行 schema（commit/URL/两个 64-hex/日期/verifier）；`cmd_make` 要求至少一个 `summary.json`/`run_manifest.json`；归档补 `run_time_state*.json(.k)`；报告新增输入清单（逐文件 SHA-256 + git HEAD + `analysis_v7/report_inputs.json`） |
| 16 | 依赖/环境/数据可复现性风险 | **部分成立/表述过时**：import 时 pip install 与 HF mirror 已改 opt-in；tokenizer 仅 train 前缀拟合并 fail-closed 校验；CPU lock 精确，CUDA 环境未锁 | 归档环境快照补 GPU 数量/名称/UUID/算力/驱动/TF32 与 pip freeze 截断标志；CUDA lock 仍须在发布时生成 |

**审查中与 HEAD 不符、应保留记录的事实**：`v11.125` 在当前 HEAD 不存在
（`CODE_SEMANTICS='v11.128'`）；`eval_subset=128` 不是最终 PPL 的评估行数；
「随机 indexer 只在评估时替换」不成立（已有从头训练对照）；`n=2 → R²=1`
不成立（拟合至少 4 点）；「没有 Holm/FDR/窗口 bootstrap」不成立。审查对
**探索/确认混合、小 n、目标/容量不匹配、动态边界非任务学习、source-only
不可审计**的判断成立，这些是本轮保留的限制声明，不用代码伪装成已解决。

本轮代码改动（全部零实验、零 test 访问）：

- `exp_lib.py`：`train_objective` 每层压缩权重 + `objective_reason` 向量比较；
  `aggregate` 透传符号检验的稳健性字段；`evidence_release_status` 严格 schema；
  boundary-F1 措辞降级。
- `build_report.py`：严格 release 解析；`sec_stats` 声明族/n/下限/符号/CI/
  Wilcoxon + 假设边界；P2S `p_Holm`/下限/目标披露；P0-3 去「证伪」；长度节
  pooled + 描述性措辞；新增 `sec_resources` 与输入清单；`_objective_reason`。
- `v7_supp.py` / `v8_supp.py` / `v9_supp.py` / `v10_supp.py` / `v11_supp.py`：
  v7 stats 稳健性字段；v8 面板 pooled/median/IQR；v9 稳健性+目标披露+
  全局多重性+pooled 列；v10 全局族补全、P3T Holm/下限、P3MT/P3MP pooled、
  机制标题/预注册/synthesized 措辞、crossover 混杂披露；v11 pooled 列+
  Holm 列+合成记录措辞。
- `archive_release.py`：TimeGuard 账本、run evidence 门槛、完整环境快照。
- `audit_gates.py`：release URL 接受 `s3://`/`gs://` 对象键。
- `test_review_fixes.py`：N98–N103 回归测试。

`CODE_SEMANTICS` 保持 `v11.128`：所有改动都是统计/报告/归档层，没有改任何
训练或评测路径上的数值语义，因此不会使已有记录失效。冻结清单字节未改。

## 31. review 14（外部方法学审查）的逐项核验与代码处置

本轮仍是**核验优先**：先逐条对照当前 HEAD 的代码/产物，再决定改什么。没有新增
训练面板，没有读取 test split，没有改 `confirmation_frozen.json` 的任何字节，
冻结计划预检（`v12_supp.py plan`）在本轮改动后仍 48/48 通过。审查的六类意见
对照 HEAD 的结论与处置如下：

| # | 审查意见 | 对照 HEAD 的结论 | 本轮处置 |
|---|---|---|---|
| 1 | 动态分块未被任务损失端到端训练（`gate.detach()>0.5`，delta_logit/fuse_conv 主要由 comp_reg 训练） | **成立（默认臂的定义），且已如实文档化**（README review-3、§22.5）；opt-in `csa_dynamic_task` 是 REINFORCE 边界臂，不在冻结范围 | 无代码可改：这是设计决定，不是缺陷；措辞边界继续维持「余弦相似度+压缩率正则的阈值分块」 |
| 2 | 冻结实验未覆盖 Dynamic/HCA（无 `csa_dynamic_rope`/纯 HCA/动态 vs 固定确认性对照） | **成立（设计边界）**：`scope.not_covered` 明确声明；FOLLOWUP 协议 + 纯 HCA 变体 + 验证器已存在 | 不改冻结清单（预注册纪律）；这些主张仍需其自己的冻结研究 |
| 3.1 | Dense 参数匹配 ≠ 计算量公平（1.87× 训练 FLOPs、kernel/归一化/sink 差异） | **成立且已披露**（`compute_budget_disclosure` + 逐对比对 `interpretation_caveat` + `reporting_rule` 禁止「等计算」措辞） | 无代码可改；FLOPs 匹配 Dense 仍是 follow-up 的 `TODO_FLOPS_MATCHED_DENSE`（需 GPU 实测） |
| 3.2 | 随机对照是固定哈希模式（同一 seed/层/步），且带 202,752 个死参数；「更合适的对照应含每次前向重随机、距离匹配、均匀间隔」 | **成立**；固定哈希与死参数已披露，但审查建议的四类对照**此前不存在** | **本轮实现**：`lightning_indexer` 新增 `random_each`/`random_dist`/`stride`/`nearest` 四个选择模式（见下），并以 `csa_fixed_rope_randidx_each`/`_dist`/`nearidx`/`strideidx` 注册为 rope 变体，满足 FOLLOWUP 协议 mechanism-isolation 一节的四类对照要求 |
| 3.3 | allblocks 读取预算不匹配（32× 读取、STE 软先验差） | **成立且已披露**（caveat 中 32x 与 log(1/32) vs log(1/1024) 均已写明） | 无代码可改：这是臂的定义；同读取预算对照由 FOLLOWUP 的 same-scope 条款要求 |
| 4.1 | 12 seeds 的 q=0.80 Holm 保守功效仅 27.5% | **成立**；manifest v6 已把功效模拟变成闸门（power_gate，miss 即拒绝或显式 V12_ACCEPT_LOW_POWER=1） | 已由 review-13 完成；无残余代码缺口 |
| 4.2 | sign-flip 尖锐零假设的适用前提；需 CI/bootstrap 敏感性 | **成立且已文档化**（`validity_assumptions`；配对/非配对 bootstrap、Wilcoxon、LOO 已在 review-12 落盘并打印） | 已在前几轮完成；无残余代码缺口 |
| 5 | 测试集 16 窗口非独立文档；训练 512 vs 评估 4096 是长度外推；嵌套前缀目标不同 | **成立（设计边界）**：单位披露（窗口≠文档）、`population_claim_allowed=False`、64 文档 house 门槛、doc-level 路径（`doc_index`）留给 follow-up；训练/评估长度差与嵌套前缀口径均在 README/报告声明 | 无代码可改；完整测试集/≥64 分层文档抽样已写入 FOLLOWUP 协议 |
| 6 | 缺 Oracle Block Selection / Recall@K / Mass Covered / 同预算随机 / 选择消融的直接机制证据 | **成立**：`audit_gates.audit_selection` 只定义了工件契约，报告一直 `not_verifiable_locally`——**工件从未被生成过** | **本轮实现** `mech_selection.py`：机制探针（oracle=同层 all-block 读的块级注意力质量分布；recall@k/mass_covered/random_baseline 逐 seed 工件 + oracle-forced/allblocks/固定哈希随机 PPL 读数），smoke 模式产出的工件通过 audit_selection 的结构校验；探针拒绝读取 test split |

工程三项（8M 训练缓存、18.4M 窗口抽取量 ≠ 唯一 token、CUDA 环境未锁）在 HEAD
分别由冻结绑定 `training_data_fingerprint`、§25.3b 的唯一 token 覆盖率字段、
以及 `archive_release.py` 的 GPU/驱动/TF32 环境快照覆盖，均属已披露边界，本轮
不改。

本轮另修复一个**逐行核验中发现的真实缺陷**：`_uses_batched_eager` 对
`indexer_mode` 只做真值检查，而 `HybridAttention.forward` 的实际分派要求
`== 'learned'`——`random` 臂（以及任何非 learned 模式）会被误报为走了批处理
孪生路径。已改为与分派一致的相等检查（N111 回归覆盖）。

新选择模式的语义（详见 `lightning_indexer` docstring 与代码注释）：

- `random_each`——per-forward 重随机：行键是隐状态的确定性 float64 归约过同一
  GLSL 哈希，模式随 H 演化逐步/逐层/逐 seed/逐行改变，同时保持「固定输入→固定
  输出」，激活重算与续跑逐位可复现（无 RNG 流；跨设备逐位稳定性不声明、也不需要）。
  `.detach()` 保证随机对照不给自己的分数引入梯度路径。
- `random_dist`——同一 content 哈希乘因果距离衰减核（tau=512 tokens，
  `_RAND_DIST_TAU_TOKENS`），使抽样带局部性先验（「距离匹配」的随机对照）。
- `nearest`——recency 对照：每行读最近 k 个可读块。
- `stride`——均匀间隔对照：行自适应步长 `max(1, readable_i // k)` 的整数函数。

四个模式全部走既有 `_indexer_selection` 的 top-k/因果收窄/STE 软先验机器，因此
与固定哈希臂共享同一选择语义约定；CSA_KERNEL 与 open-block 路径对非 learned
模式的既有拒绝自动覆盖它们；死参数计数（`param_breakdown`）从 `== 'random'`
泛化为 `!= 'learned'`。

本轮代码改动（全部零实验、零 test 访问）：

- `exp_lib.py`：`_random_mode_of` 模式归一化（未知值 fail-closed）；
  `lightning_indexer` 四个新模式；`AttnCfg` 白名单扩展；`_pre` 的 qI/w_idx
  跳过条件从 `== 'random'` 泛化为 `!= 'learned'`；`_uses_batched_eager`
  相等检查修复；`param_breakdown` 死参数与文案泛化。
- `v7_supp.py`：四个 rope 对照变体注册（registry-only，默认面板不训练）。
- `mech_selection.py`（新文件）：selection-attribution 机制探针（run/smoke/
  audit 三个子命令；产出并校验 `analysis_mechanism/selection_attribution.json`）。
- `test_review_fixes.py`：N110–N112 新回归 + L5b 断言更新（死参数文案）。

`CODE_SEMANTICS` 保持 `v11.128`：learned 路径与冻结的固定哈希路径逐位不变
（N110 的 True=='random' bit-identity 断言、`v12_supp.py plan` 48/48 通过），
新增模式是 opt-in 的新臂，未进入任何默认面板或冻结臂。冻结清单字节未改。

## 32. review 15（外部六类方法学复审的独立复核 + 逐函数代码审查）的逐项核验与代码处置

本轮是**对 review-14 处置的独立复核**：同一份六类意见先被逐条对照 HEAD 重新
验证（不采信 review-14 的自述），再对 `exp_lib.py` 的动态分段器、CSA/HCA
前向、梯度回传与训练目标做逐函数审查（用户明确要求的「梯度估计偏差、对照
实现不等价或其他会影响科学结论的实现问题」）。没有新增训练面板，没有读取
test split，没有改 `confirmation_frozen.json` 的任何字节，`CODE_SEMANTICS`
保持 `v11.128`（所有改动均为 opt-in 新路径或纯函数新增；learned 路径与冻结
固定哈希路径逐位不变，N113 断言 pattern_seed=0 的 bit-identity）。

### 32.1 六类意见对照 HEAD 的独立复核结论

| # | 审查意见 | 独立复核结论（相对 review-14 之后的 HEAD） | 本轮处置 |
|---|---|---|---|
| 1 | 动态分块未被任务损失端到端训练 | **成立且属实**：`_single`/`_gate_signal`/`_gate_signals` 三处的默认 `threshold` 路径均为 `gate.detach() > _HALF`（exp_lib 5358/5397/5481/5491 行族）；`delta_logit` 的唯一梯度来源是 comp_reg 的 STE（`gate_mean = (nblk-1 + _shs - _shs.detach())/T`）。`csa_dynamic_task`（REINFORCE）实现正确：`boundary_policy_term` 的符号、EMA 基线（先取优势后更新基线）、逐行逐层 logp 累积均无误，但不在冻结范围 | 设计边界维持披露口径，无代码缺陷可修 |
| 2 | 冻结实验未覆盖 Dynamic/HCA | **成立**：`confirmation_frozen.json` 的 4 臂确无 dynamic/HCA；`scope.not_covered` 已声明；FOLLOWUP 协议 + 纯 HCA 变体 + 验证器存在 | 不改冻结清单；本轮把 FOLLOWUP 的两处「NOT yet wired / TODO」落地（见 32.3） |
| 3.1 | Dense 参数匹配 ≠ 计算量公平 | **成立**：`compute_budget_disclosure` 的 1.23M/2.30M/1.87x 数字本轮被解析模型**精确复现**（181,248 + 4·256²·4 = 1,229,824；524,288 + 4·256²·6.76348876953125 = 2,297,296） | 见 32.3 的 FLOPs 匹配 Dense planning 臂 |
| 3.2 | 随机对照是固定哈希 + 死参数；四类对照缺失 | **review-14 已补**：`random_each`/`random_dist`/`nearest`/`stride` 四模式存在且语义正确（本轮逐模式复核：nearest=可读块最大 id、stride=行自适应步长、random_dist=tau=512 因果核）；**遗留缺口**：协议要求的「连接模式种子独立于训练种子」未接通（内容哈希无种子旋钮） | **本轮实现** pattern_seed（见 32.2） |
| 3.3 | allblocks 读取预算不匹配 | **成立且已披露**（32x 读取、STE 软先验 log(1/32) vs log(1/1024) 均在 caveat） | 臂定义不变 |
| 4.1 | 12 seeds 功效 27.5% | **成立**；power_gate（v6）已在清单与代码落地 | 已覆盖，无残余 |
| 4.2 | sign-flip 零假设与 CI | **成立且已披露**；`paired_bootstrap_ci`/`unpaired_bootstrap_ci`/LOO/Wilcoxon 均在统计工件里 | 已覆盖，无残余 |
| 5 | 16 窗口非独立文档 | **成立**（`n_seqs: 16`）；§27.2 的 doc_index 文档聚类路径存在；FOLLOWUP 的 64 文档门槛存在 | 设计边界维持披露 |
| 6 | 缺 Oracle/Recall@K/Mass Covered | **review-14 已建生产者** `mech_selection.py`；但本轮逐行审查发现其 **mass_covered 实现与自身契约矛盾**（见 32.2 的缺陷一） | **本轮修复**该缺陷 |
| 工程 | 8M 缓存 / 18.4M 窗口量 / CUDA 未锁 | 均为已披露边界（冻结 data fingerprint、§25.3b 唯一 token 覆盖率、archive 环境快照） | 不改 |

### 32.2 逐函数深查发现的两个真实缺陷（本轮全部修复）

**缺陷一（影响科学结论的指标 bug）：`mech_selection._probe_arm` 的
mass_covered 分母用错了。** 契约（`audit_gates.audit_selection` docstring 与
工件 `ARTIFACT_DEFINITION`）定义 mass_covered 为「落在学习式选择内的 oracle
注意力质量**分数**」，分母应是该行可读块上的总质量（softmax 归一化后 ≈1）；
实现却除以了**可读块计数**（`read_np[r].sum()`），把指标系统性缩小约
n_readable 倍且逐行权重不一致——review-14 的 smoke 读数 mass=0.0917 <
base=0.125 完全是该缺陷造成的假象（修复后同一 smoke 为 mass=0.3799，学习式
选择在未训练模型上就已覆盖 38% 的 oracle 质量，明显高于均匀随机地板
12.5%）。若带着该 bug 跑真实面板，「选择器是否找到重要块」的机制结论会被
整体翻转。修复：分母改为 `mass_np[r][read_np[r]].sum()`（逐行总质量），并加
NaN/非正守卫；N114 用独立实现的契约公式对拍锁定。

**缺陷二（协议要求未接通）：连接模式种子与训练种子不独立。**
`random_each`/`random_dist` 的模式是隐状态内容的确定函数，没有任何旋钮能把
「模式噪声」与「训练噪声」分开重跑。修复：
- `AttnCfg.pattern_seed`（默认 0 = review-14 逐位不变；`_ps=0` 时完全跳过
  位移，op 链不变）；非哈希模式（含冻结固定哈希 `random`、nearest/stride、
  learned）配非零种子一律 fail-closed 拒绝——冻结臂的 bit-identity 是硬约束；
- `lightning_indexer(..., pattern_seed=)` 把种子混入行键
  （`_hk + ps*0.618...`），`_indexer_trainable` 与模型前向调用点透传；
- v7_supp 注册 `csa_fixed_rope_randidx_each_ps<N>` / `..._dist_ps<N>` 变体族
  （registry-only）；
- `validate_followup_design.py` 对 `design.selection_controls` 声明的
  pattern_seed 与注册表解析值做机器对账；
- N113 锁定：ps=0 bit-identity、同种子可复现、异种子异模式、错误模式
  fail-closed、模型前向真实消费 cfg.pattern_seed。

### 32.3 FOLLOWUP 协议遗留 TODO 的落地

- **FLOPs 匹配 Dense 臂（原 `TODO_FLOPS_MATCHED_DENSE`）**：新增解析模型
  `analytic_attn_flops_token` / `analytic_train_flops_token`（注意力核+MLP，
  投影/池化/embedding 对所有臂一致地排除；注意 dense 的可选滑窗是
  `cfg.window` 而非 CSA 的 `sliding_window`），求解器
  `flops_matched_mlp_ratio` 解析解出容量旋钮——冻结配置下
  **mr = 689/256 = 2.69140625**（hidden 689：Dense 注意力在 L512 每层每
  token 多花 343,040 FLOPs，MLP 从 1024 缩到 689 来支付）。注册
  `full_rope_flops_matched` 变体；run cfg 通过 `flops_matched` 集合 +
  `flops_match_ref` 走 `variant_mlp_ratio` 的新路由，与参数匹配集
  **互斥 fail-closed**（同臂同列是配置错误——两种 estimand 不得混同）。
  模板 JSON 的 TODO 已替换。**边界维持 §27.1 的纪律：解析等式不是证书**，
  只有 profiler 实测总训练 FLOPs 落在参照 1% 内才允许「等计算」措辞；模板
  仍按设计 `--require-ready` 失败（measured 验证、功效模拟、算法门、冻结
  元数据仍是阻塞项）。
- **模式种子对照族**：模板 `design.selection_controls` 声明固定哈希 /
  random_each(ps 0/1/2) / random_dist(ps1) / nearest / stride，与验证器对账。

### 32.4 逐函数审查中确认无问题的部分（记录备查）

- STE 软先验：前向值 = 硬选择（`(soft_attn - soft_attn.detach()).add_(attn)`
  精确相消），梯度经 soft_attn 直通；零先验块被掩出 S 路径支撑集，与
  reference/Triton 反向一致；
- `_indexer_trainable` 的 checkpoint 重算对全部选择模式无 RNG 依赖（含
  pattern_seed——纯配置整数的函数）；
- REINFORCE 项符号与 EMA 基线更新次序（先优势后更新）正确；
- 四个对照模式的 top-k 语义与 `_indexer_selection` 的无放回/因果收窄共享
  同一机器；`_uses_batched_eager` 的相等检查修复（review-14）仍在位；
- `_full_batched` 对 `cfg.window`（dense 滑窗）与 CSA `sliding_window`
  （局部分支）的分派互不串扰。

### 32.5 本轮改动清单（全部零实验、零 test 访问、冻结清单零字节改动）

- `exp_lib.py`：`AttnCfg.pattern_seed` 字段 + fail-closed 校验；
  `lightning_indexer`/`_indexer_trainable`/`_single` 调用点的种子透传；
  `analytic_attn_flops_token` / `analytic_train_flops_token` /
  `flops_matched_mlp_ratio` 三个纯函数；`variant_mlp_ratio` 的
  flops_matched 路由（互斥 fail-closed）与两处调用点透传。
- `mech_selection.py`：mass_covered 分母修复 + 契约文本明确分母；
  replay 包装接受 pattern_seed（签名兼容）。
- `v7_supp.py`：`_ps<N>` 变体族与 `full_rope_flops_matched` 注册。
- `validate_followup_design.py`：`selection_controls` 注册表对账、factorial
  臂 pattern_seed 一致性检查、FLOPs 匹配 Dense 声明的 planning 来源要求。
- `followup_design.template.json`：TODO 替换 + selection_controls 声明。
- `test_review_fixes.py`：N113–N115 三个新回归。
- `FOLLOWUP_CONFIRMATORY_PROTOCOL.md` / `README.md` / `README_zh.md` /
  本节：对应文档更新。

`CODE_SEMANTICS` 保持 `v11.128`；`v12_supp.py plan` 在本轮改动后仍须
48/48 通过（CI 门未动）。


## 33. review 16（旧证据退役 + 两项 pre-freeze 工具落地）的决策与实现

### 33.1 背景：v12 证据退役（owner 决策，2026-11-26）

v12 冻结确认轮的四臂（`csa_fixed_rope` / `full_rope` /
`csa_fixed_rope_randidx` / `csa_fixed_rope_allblocks`）不含 Dynamic/HCA 臂，
其 Dense 对照是参数匹配而非算力匹配，随机对照是固定哈希模式，测试面板
只有 16 个窗口——这些边界在 §17/§20/§22 中已如实披露，且历史结论本身
无实现缺陷（review-15 逐函数复核）。owner 决定**退役 v12 证据作为任何
方法学声明的依据**：本仓库此后不得引用 v12 的数字（含 CONFIRMED 标签）
支持 CSA/HCA/Dynamic 的有效性或等算力声明；确认性证据只能来自
follow-up 协议（§31/§32 注册的 2×2 全因子 + 双 Dense 参照 + 机制探针）。

处置方式：**冻结清单 `confirmation_frozen.json` 一个字节都不动**（它是
历史预注册记录，改写等于篡改；v12_supp 的 plan 预检与 CI 门继续对它
运行）；退役只发生在证据声明层——README 的证据等级说明、RELEASES.md
的状态行、DECISION_LOG 的决策条目。这遵循本仓库的一贯原则：废弃靠
声明与索引，不靠改写历史（与 re-freeze 不得 CONFIRMED 同源，§28）。

### 33.2 pre-freeze 阻塞项的工具化：followup_power_sim.py

协议要求测试集访问前完成「基于合理效应量与验证集实测方差的仿真功效
分析」。v12 的 `_power_enumeration` 是对已观测增量的精确枚举（事后设计
辅助），回答不了「这个种子数能不能检出预注册效应」。新工具补齐缺口：

- 效应模型两种：`normal`（声明均值/标准差）或 `bootstrap`（重采样观测
  验证增量）；四个主假设各自的效应与方差独立声明，随 artifact 存档，
  投影可审计；
- 检验统计量与确认性分析完全同源：每 replicate 的 p 用
  `exp_lib.exact_signflip_p`、族校正用 `exp_lib.holm_adjust`（批处理路径
  `_batch_exact_signflip_p` 与标量契约逐行一致，N116 锁定，n≤20 走
  2^n 向量化枚举，n>20 回退 MITM/saddlepoint）；
- 输出每假设功效 + Monte-Carlo Wilson 95% CI、min_power、
  `--seed-sweep` 的最小达标种子数推荐、保守解析投影（q=0.80@12seeds
  =0.2749，与审查报告的 27.5% 一致，交叉参照）；
- fail-closed：alpha>0.05、布尔种子数、零方差、<200 replicate、假设族
  不完整等一律 invalid_spec；同 spec 同 rng_seed 产物逐字节确定。

**验证器绑定（N117）**：`statistics.power_simulation_completed = true`
现在必须携带 `statistics.power_simulation_artifact`（工具输出 JSON 路径），
且 study id / 假设族 / 种子面板 / alpha / 目标功效 / min_power 全部与
设计对账；篡改任何一项都被拒绝——布尔位不再是可以拍脑袋置的证据。

### 33.3 pre-freeze 阻塞项的工具化：followup_flops_gate.py

FLOPs 匹配 Dense 的 1% 接受门此前只有解析规划（review-15），无测量路径。
关键实测发现驱动了设计：

- `torch.profiler(with_flops=True)` 只给 mm/bmm/addmm 归因，
  **SDPA 注意力内核（CPU flash / CUDA flash / cudnn）一律零归因**——
  只用 profiler 会系统性漏掉注意力计算量，且静默漏掉；
- `torch.utils.flop_counter.FlopCounterMode` 在 aten 调度层计数，含
  SDPA（按张量形状的满公式 4*B*H*Tq*Tk*D，与本仓库解析模型同为非因果
  口径）；计数基于形状，确定可复现；
- 自定义 CSA Triton 内核不经过 aten 调度，对任何 op 级计数器不可见——
  活跃内核路径直接拒绝测量（fail-closed），宁可拒证不可少计。

工具结构：`measure`（warm-up 后计数 K 步训练步 = fwd+bwd+optimizer；
第二同构块必须逐位复现，否则说明计算本身变了、数字不是测量）+
`gate`（两 artifact 的总训练 FLOPs 比值 ≤1% 才 FLOPS_MATCHED；CPU 运行
只算 smoke 证据永不签证书；per-step 与 total 不一致的 artifact 拒收）。
profiler 通道降级为披露表：记录「带时间却无 flops 归因」的算子清单，
显式展示调度级计数器必须覆盖什么。冒烟实测（tiny 几何）：
full_rope 403M FLOPs/步、csa_fixed_rope 1.17G FLOPs/步——**逐行 eager
实现的 op 级实际计算量约为 dense SDPA 路径的 3 倍**，解析模型（仅注意力
核+MLP，排除投影/池化/索引器投影）看不到这项实现开销。这正是协议坚持
「实测 FLOPs 为唯一等算力接受门」的原因，也再次确认 kernel/dtype parity
必须是与 FLOPs 门并行的独立验收。

### 33.4 本轮改动清单（零实验、零 test 访问、冻结清单零字节改动）

- 新增 `followup_power_sim.py`（仿真功效分析 + 种子扫描 + artifact）；
- 新增 `followup_flops_gate.py`（FlopCounterMode 测量 + 1% 门 +
  fail-closed 政策）；
- `validate_followup_design.py`：功效 artifact 绑定（study/族/种子/
  alpha/目标/min_power 六项对账，unreadable 拒绝）；
- `test_review_fixes.py`：N116–N119 四个新回归（420/420 全绿）；
- `FOLLOWUP_CONFIRMATORY_PROTOCOL.md`：两个工具的用法与绑定语义；
- `README.md` / `README_zh.md`：review-16 节 + v12 证据退役声明；
- `RELEASES.md` / `DECISION_LOG.md`：退役状态行与决策条目。

`CODE_SEMANTICS` 保持 `v11.128`；`v12_supp.py plan` 48/48 与
`test_open_block.py` 15/15 未受影响（本轮零核心库改动）。

## 34. review 17（外部方法学审查：实现正确性 / 实验设计 / 统计推断）的逐项核验与代码处置

审查分三个层面（实现正确性、实验设计有效性、统计推断有效性），共十一个主要部分。
本轮立场与历次一致：核验优先，设计边界如实成立则保持披露，只有代码层面
可修的缺口才动手。零训练、零 test 访问、冻结清单零字节改动、
`CODE_SEMANTICS` 保持 `v11.128`。

### 34.1 十一部分逐项判定（对照 HEAD 复核）

| # | 审查主张 | 复核结论 | 现状 |
|---|---|---|---|
| 1 | v12 确认轮未覆盖 Dynamic CSA / HCA；v12 的 CONFIRMED 不能外推 | 成立（设计边界，已披露） | `confirmation_frozen.json` arms = csa_fixed_rope / full_rope / randidx / allblocks，scope.not_covered 原文即声明 dynamic/HCA 不在冻结范围；2026-11-26 owner 决策已退役 v12 作为主张依据（README 引注），Dynamic/HCA 证据只能出自 follow-up 协议的 2×2 全因子 + 双 Dense 参照（模板已声明四臂） |
| 2 | v6-v11 同一验证集重复使用 → 自适应过拟合风险 | 成立（已披露） | README「Evidence levels」明示 v6-v11 为 exploratory；探索/确认分离 + P5T 冻结锁 + 单次访问策略是现行机制；审查建议的「先冻结再访问 test」正是 follow-up 协议 §Freeze and reporting rules 的条目 1-9 |
| 3 | Dense 对照非单因素（Q/K 归一化、scale、KV、sink 均不同）；参数量匹配 ≠ FLOPs 匹配 | 成立（已披露），**本轮关闭最后一个可修缺口** | 三对照中：原始 Dense（`full_rope`，参数匹配）与 FLOPs 匹配（`full_rope_flops_matched` + 实测门）已在；**同骨架 Dense 此前缺失** —— 仅有 `full_rope_cos`（余弦归一化、无 sink、SDPA kernel）。本轮注册 `full_rope_cos_sink`（见 34.2），模板声明 `dense_same_backbone`，验证器强制要求并逐字段核验（N120） |
| 4 | `csa_dynamic` 的边界由余弦阈值 + 压缩正则决定，LM 交叉熵不经由离散切分反传；`csa_dynamic_task` 才是任务损失路线 | 成立（已披露且命名已分离） | gate = sigmoid(前缀均值 + delta_logit)（纯因果前缀统计），硬切分 detach，LM 梯度不可达；`make_layer_cfgs` 注释与 README 明示 threshold 臂「must not be described as task-learned」；`boundary_policy='reinforce'` 臂经 `boundary_policy_term`（REINFORCE，EMA baseline，符号由 N120 前的符号锁测试锁定）把 (CE-baseline)*logp/n 并入训练损失；`train_objective` 描述符逐层记录 comp_lambda_mult |
| 5 | 12 种子 Holm 保守功效 ≈0.275；p_min=2/2^12 ≠ 功效 | 成立（已披露且已工具化） | v6 修订的 minimum-power gate（q=0.80）在 P5T 加载期强制执行；`followup_power_sim.py`（review-16）用与确认性分析同源的 `exact_signflip_p`+`holm_adjust` 做预冻结功效仿真 + `--seed-sweep` 推荐最小种子数；验证器把功效工件六项对账绑定进 `--require-ready`（N117）。审查建议「功效模拟先行、按需加种子」即协议现状 |
| 6 | 16 窗口面板把窗口当独立样本；应按文档聚类 | 成立（已披露，路径已具备） | P5T 记录单位 = 评估窗口（README v12 行明示）；`eval_ppl(doc_index=...)` 支持真文档级 NLL/token 聚类（t_doc_index_eval_aggregates_real_documents 锁定，含采样评估与截断/错位 fail-closed）；follow-up 协议 §Evaluation population 明确 64 文档下限、文档级 bootstrap ≥10k 次、窗口不得当独立文档——v12 冻结面板不改（预注册不可变），文档级统计由 follow-up 承接 |
| 7 | PPL 应 token-pooled，不应窗口算术平均 | 成立（已修复：review-3/12） | `aggregate` 以 `ppl_pooled=exp(ΣNLL/Σtok)` 为 `ppl_primary`，种子均值/中位/IQR 并列；记录携带 nll_sum/n_tokens/逐行 NLL（doc_index 时逐文档）；review-12 起 v7/v10 长度节同样统一 pooled 并标注回退 |
| 8 | 固定哈希随机对照 ≠ 一般随机稀疏基线；死参数 1.83% | 成立（已披露；对照族已补齐） | `lightning_indexer` 的 'random' 模式 = (row*12.9898+col*78.233) 冻结哈希（本仓复核实测在 exp_lib.py）；review-14 补齐 `random_each`/`random_dist`/`nearest`/`stride` 四模式，review-15 加 `pattern_seed`（连接模式种子独立于训练种子）；Oracle/recall = `mech_selection.py`（review-14/15，mass_covered 分母 bug 已修，N114 锁定），真实面板工件待 GPU 运行（README 明示 not_verifiable_locally） |
| 9 | 效率结论需实测：FLOPs/吞吐/延迟/显存/KV/能耗分开测 | 成立（部分已工具化；能耗未测是硬件边界） | `followup_flops_gate.py`（review-16）：FlopCounterMode 实测 + 1% 门 + 自定义内核拒绝 + CPU 永不签证书；`measure_efficiency(_sweep)` 记录吞吐/时延；`audit_gates.efficiency_pairs` 的同形/同设备/KV 语义配对门；能耗未测持续披露 |
| 10 | 动态分块需边界语义对齐、Fixed-vs-Dynamic 消融矩阵、长度泛化三层证据 | 成立（协议承接；探针已有） | 边界对齐：`fuse_analysis.py`/boundary-F1（v7/v10 已跑，结论保持探索性措辞）；审查的 A-F 六格矩阵由 follow-up 2×2 全因子 + `dense_same_backbone`/`allblocks`/随机对照/Oracle 承接；长度外推 = v7 P1L / v10 P3M（README 明示为 512 训练长度外推探针，非长上下文训练） |
| 11 | 逐函数代码审计（分段器因果性 / CSA-HCA 前向 / top-k 梯度 / 池化 / 训练目标 / 评估 token 集 / 统计） | 复核未见新缺陷（既有审计链完整） | 分段器：gate 为前缀均值（只依赖 ≤i 的相似度），`audit_gates causality` 前缀不变性 + 未来扰动 + 严格不等式门、`test_open_block` 前缀不变性/未来梯度；CSA/HCA 前向：`csa_kernels.py` torch 参考孪生 + G0 统计等价门 + 行分块等价契约；HCA = 每层读全部粗块（`pool_blocks_single`、无 top-k）+ 滑窗 + sink，与 CSA（变长块、overlap 池化、top-k）机制不同；top-k 硬选择的 STE 软先验在前向值中抵消、反向重建对偶 Jacob（README M0/G0 节）；训练目标：review-12 已修 comp_lambda_mult 记录；评估 token 集：P5T 准入门（同切片指纹/配方签名）+ `ce_loss`/`eval_ppl` 目标列契约；统计：N1-N119 全链回归 |

### 34.2 本轮唯一的代码改动：同骨架稠密隔离对照

- `v7_supp.py`：注册 `full_rope_cos_sink` =
  `AttnCfgRope(kind='full', rope=True, rope_dim=16, qk_norm=True,
  full_cosine=True, full_sink=True)`（registry-only；不进任何历史面板）。
  控制变量：共享 qk RMSNorm、余弦归一化 q/k @ scale 1.0、可学习逐头 sink、
  RoPE；sink 的存在使稠密路径必然走手写 `_sink_attn_loop`（无 sink 才走
  SDPA），kernel 族与 CSA 对齐。残余差异（逐 token K/V vs 压缩块 KV 管线、
  q/k 归一化次序、逐 token 滑窗分支、投影结构）= 被检验机制本身，随臂注释
  披露。`ALIGNED_DENSE_VARIANTS` 增补该臂。
- `followup_design.template.json`：`dense_controls` 新增
  `{"id": "dense_same_backbone", "matching": "same_backbone",
  "variant": "full_rope_cos_sink", ...note 披露控制项与残余项...}`。
- `validate_followup_design.py`：设计层新增「必须声明 same_backbone 稠密
  对照」；registry 层对该 matching 逐字段核验 full_cosine/full_sink/qk_norm/
  rope 全真，且 note 必须写明匹配内容（scale/sink/kernel）与残余机制项——
  普通稠密臂不能冒用隔离标签。
- `test_review_fixes.py`：N120（注册解析/模块 sink+qk_norm 实建/前向反向
  有限/full_rope 语义不变/模板声明/验证器接受-缺失拒绝-冒用拒绝）。
- `FOLLOWUP_CONFIRMATORY_PROTOCOL.md`：Required arms 表加 same-backbone
  行；机制隔离一节标注「已注册」并列出残余差异清单；验证器一节同步。
- `README.md` / `README_zh.md`：review-17 hardening 节。
- 冻结清单 `confirmation_frozen.json` 零字节改动；`v12_supp.py` / P5T /
  历史面板 arm 列表零改动（`full_rope` 保持无 sink，冻结语义不受影响）。

### 34.3 结论

审查的十一部分中，十个属于「设计边界成立且已披露 / 既有工具已覆盖」；
本轮独立复核未发现新的实现缺陷；唯一可修缺口（协议要求单因素稠密对照但
注册表与模板均无该臂）本轮以 `full_rope_cos_sink` + 模板声明 + 验证器强制
+ N120 回归关闭。审查建议的 P0 三项（区分研究假设、公平对照组、四因素
矩阵）中，第 1、3 项在仓库现状与 follow-up 协议中已满足，第 2 项的最后一
个空洞（Same-backbone Dense）至此补齐；P1/P2 各项分别由 mech_selection /
followup_power_sim / followup_flops_gate / archive_release 承接，均需 GPU
与真实数据运行后才能产生证据（工具存在 ≠ 实验完成，README 一贯口径不变）。

### 34.4 改动后的验证记录

`test_review_fixes.py` 421/421（N120 新增，历史 420 项全绿无回归）；
`test_open_block.py` 15/15；`audit_gates.py causality` PASS（七个 rope 臂：
label shift + 前缀不变性 + 严格可读门 + 解码一致性）；
`validate_followup_design.py followup_design.template.json --verify-registry`
零注册错误（剩余 8 项均为有意的 pre-freeze 阻塞：功效仿真未跑 + 六个
算法门未记录）；`v12_supp.py plan` 不受本轮影响；冻结清单字节比对不变。
