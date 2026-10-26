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

### 7.1 上游语料版本是内容指纹之外的另一层身份

`src_fp` / `train_fp` 证明两次运行**看到了同一批 token**；它不能证明一次
**全新构建**来自同一个上游快照。WikiText-103 与 FineWeb-Edu 都是可变的
Hub 仓库，同一配置在分支移动后重建会得到不同语料，而旧记录的内容指纹
对「新语料」只会表现为另一组数字，不会指出上游已经变了。

因此两类语料都固定一个**完整 40 位 commit**：

- WikiText-103：`WIKITEXT_HF_REVISION`（`Salesforce/wikitext`，
  `wikitext-103-raw-v1`；镜像别名 `wikitext` 指向同一 commit）；
- FineWeb-Edu：`FINEWEB_EDU_HF_REVISION`（`HuggingFaceFW/fineweb-edu`）。

`load_dataset(..., revision=<commit>)` 直接请求该 commit。需要镜像/离线副本
时用 `WT103_DATASET_REVISION` / `FWEDU_DATASET_REVISION` 覆盖，但只接受
完整 40 位十六进制 commit——分支名、tag、短前缀都会移动，不能作为数据身份。
非法值在加载前硬报错。commit 固定的是 Hub 树；若某个 builder 另行抓取外部
归档，最终到达训练循环的字节仍以 `src_fp` / `fit_fp` 内容指纹为准。两层
身份互补：commit 让**上游变化可见**，内容指纹让**实际字节可对账**，不能用
前者替代后者。

每个 cache meta 新增 `dataset_source`（test meta 另记
`train_dataset_source` / `validation_dataset_source`）：

```
{'provider': 'huggingface', 'repo': ..., 'config': ...,
 'revision': '<40-hex>', 'revision_pinned': true}
```

- 缓存命中时，若 meta 记录的 revision 与当前固定值不同，train/validation
  缓存按既有纪律**重建**（并在日志里点名），test 缓存则**硬拒绝**（冻结
  切片不能静默换上游）；
- test meta 的 train/validation/test 三个来源必须属于同一 commit，冷构建
  在写盘前检查，混快照直接拒绝；
- 没有该字段的旧缓存按 legacy 处理：打印一次 NOTE，继续可读（旧结果必须
  仍可复现），但审计明确写「上游 commit 不可本地验证」，绝不算作已证明；
- 本地 `wiki.*.raw` / `WT103_RAW_TXT` / `FWEDU_LOCAL_JSONL` 记为
  `provider='local'`：内容仍由 token 指纹绑定，但上游 commit 不被声称，
  审计给 warning。

`python audit_gates.py data_source`（也进入 `report`）独立于 loader 重算这
一层：逐 cache 校验 schema/commit，配对 train/test，汇总目录内出现的
revision。它与 §8 的证据分级一致：**不可验证就说不可验证**，不用「文件
存在」代替「来源已核验」。

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
  v12 的逐文档配对分析另给 `bootstrap_mean_ci` 的文档重采样
  studentized 区间（对逐文档 PPL 差与 log 比分别报告）。这些区间**以
  已训练的种子为条件**，
  只量化文档抽样不确定性，不能替代 seed-level 精确符号翻转检验。

### 22.2 符号翻转检验的假设与敏感性

- `paired_effect_stats` 现同时给出符号计数、paired bootstrap-t 区间、
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
- **分级由实际 diff 推导**：`kind` 出现在差集 = 换注意力算子（另有 mlp
  加宽/投影/sink 语义），标记 `architecture_level`，永远不冒充 knob 级单
  因素；差集落在**一个**已声明干预族内才是 `single_factor`。方法 vs 参数
  对齐稠密基线因此是 architecture_level 而不是单因素，与 §22.4 的
  objective 门结论一致。
- **多族键集必须显式声明 system-level**：仅把多个键枚举进
  `ABLATION_INTERVENTION_KEYS` 不等于单因素。差集跨两个以上家族时，只有
  在 `CONTRAST_SYSTEM_LEVEL_DECLARATIONS` 中声明的对照才能记为
  `system_level`；否则 refuse。否则一条两因素消融会继续被报告成单因素。

契约同时把两处此前**未声明**的 HCA 动态臂差异写进明面：动态 HCA 层的
边界门以 `comp_lambda_mult=4` 训练（comp_lambda_effective 0.05→0.2），以及
`block_size` 在动态分段下的 inert 差（`blocks_from_cuts` 不读它）。这两条
差此前只在代码与 `train_objective` 读数里可见、未进对照声明；现在它们是
声明键集的一部分，且 `ABLATION_DESIGN` 的对照注释与
`CONTRAST_OUTSIDE_CFG_FACTORS` 都写明其语义。`hybrid_dynamic` /
`hybrid_hca_dyn` 两条对照同时移动分段束与 HCA 边界门的
`comp_lambda_mult`，因此显式列入 `CONTRAST_SYSTEM_LEVEL_DECLARATIONS`，
继续按 §22.4 的 system-level 读数，不再计入 knob 级单因素；其余
dynamic-vs-fixed 对照（只差分段束）仍为单因素。契约的职责是不漏报，不是
放宽。

### 37.2b 扫描对照：元数据要与真实臂 diff 对账

`exp_lib.sweep_contrast_factor_audit` 把 `v9.SEQ2K_CONTRAST_FACTORS` 的
`a`/`b`/`factors`/`single_factor` 声明逐条重建臂配置并核验：声明的因素键
必须恰好等于实际 diff；标成 `single_factor=True` 的还必须落在一个声明家
族内。`v9.add_paired` 在写 comparison 记录时附带
`factor_declaration_verified`，声明不匹配时清零 `single_factor` 标签，避
免手写元数据冒充已验证的单因素结论。

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

动态研究清单同样在定义点核验“需要是单因素”的假设：
`dynamic_design_audit` 对每条 `mechanism_claim_eligible=true` 的假设要求
其 primary comparison 的 cfg diff 落在一个单因素族内。当前 H1（随机选择
下的固定 vs 动态分段 → `dynamic/chunking/target_block_tokens`）、H2
（动态臂内的学习 vs 随机选择 → `indexer_mode/random_salt`）与 H4（任务
边界策略 → `boundary_policy`）通过；H5 的 HCA 互换（`kind` + 多键）与 H3
的稠密参数代理保持 system-level，不能挂机制主张。若把可挂机制的假设换成
多因素臂对，audit 直接 refuse。

### 37.4 扫描对照与开放块矩阵

- v9/v10 的 seq2k 扫描配对检验新增因素列（`v9.SEQ2K_CONTRAST_FACTORS`）：
  `topk8 - topk512` 是单因素剂量响应（只差 `index_topk`）；两条 m1 对照同
  时改 block_size(4→1)/overlap(128→0) 与 topk 语义（逐 token 读取）——等
  token 预算但**多因素块几何束**，只能读作 sweep 点之间的系统级比较，报告
  明示不得读作 topk 剂量响应。声明本身由 37.2b 的 audit 与实际臂 diff 对
  账，漂移时 refuse。
- `exp_lib.open_block_matrix_audit` 核验开放块 2x2 矩阵（§24 声明的
  max_block × 读取规则）只沿两根声明轴变化：同 cap 的 mode 对照只差
  `open_block_mode`、同 mode 的 cap 对照只差 `max_block`、scope 对照只差
  `open_block_scope`。审计覆盖每个 mode/cap/scope 单元和两种 scope，不再
  只查 closed 读取规则；任何第三键泄漏（extra）或声明轴不再差异（missing）
  都会 refuse。

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

> **单元补注（§53）。** WikiText 的 row 本身是原始**行**（标题/段落），一篇
> 文章跨多个 row；本节所说的「文档边界」是 cache 中的 row 边界，不是文章
> 边界。§53 给出按顶层 ` = Title = ` 标题的文章级重分组与审计入口；本节
> 已冻结的逐 token 行分组不变（点估计也不变）。

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

## 48. 运行顺序也是实验设计变量：按种子分块并随机化

### 48.1 问题：配对检验假设了「同一时间块」而驱动没有保证它

精确符号翻转检验的交换性前提（§20）要求同一种子下两臂除臂本身外一切
随机因素相同。代码把数据顺序、初始化 RNG、评测行与腐坏网格都按种子对齐
了，但**日历时间**从未进入设计的机器检查：`v7_supp.run_lenphase`
（v12 P5C 的训练入口）原来的单元格顺序是
`[(v, seed) for v in variants for seed in seeds]`，即「臂 A 的全部种子跑完，
再跑臂 B 的全部种子」。同一 pair 的两臂因此落在相隔数小时甚至数天的两个
机器时间块里；驱动更新、GPU 温度/降频、邻居负载、集群镜像变化都可能制造
一个与臂完全共线的差异，而逐种子配对检验无法把这种差异与臂效应分开。
这不是「p 值不够小」的问题，是设计层面的混杂。

通用 `exp_lib.run()` 的 `_plan_run` 恰好是 seed-major（同一种子的各臂相邻），
说明两个驱动早已不一致；但即使是 seed-major，固定不变的臂顺序仍可能把
「每个 block 的第一个位置」与某个臂绑定（例如基线总是先跑），把预热/缓存
效应读成臂效应。因此需要的不只是「改成 seed-major」，而是**完整区组 +
位置随机化**。

### 48.2 设计：随机化完整区组（一个种子一个 block）

`exp_lib.blocked_run_order(cells, block_key=..., seed=...)` 实现：

1. 每个 block（默认 `seed`）内的单元格连续执行，使同一 pair 的两臂尽量
   处在同一机器时间块；
2. block 内顺序由一个**稳定的 SHA-256 置换**决定，不依赖 Python 的
   `hash()`（后者每个进程随机化，会让 resume 得到不同计划）；
3. 当每个 block 含同一组 treatment 时，使用「随机基排列 + 逐 block 循环
   旋转」：`offset = (design_seed + block_index) % n_treatments`。这样在
   block 数 ≥ 臂数时，每个臂在每个位置出现次数之差不超过 1；不是每个
   block 独立洗牌（独立洗牌可能恰好让某臂总在第一，仍是系统性顺序效应）。
4. block 顺序按键排序（不是随机），保证并行/续跑看到同一计划；设计种子
   固定为面板第一个种子，因此同一 `(variants, seeds)` 的 `run_order.json`
   可复现。

`exp_lib.build_run_order_plan` 把计划写成 `run_order.json`（在
`results_*/` 内，已被 `.gitignore` 覆盖），字段包括 `design`
（name/block/design_seed/position_balance）、`variants`、`blocks`（每个
block 的臂顺序）、`planned_cells`、`cell_blocks` 与 `cell_variants`。
计划对**全部声明单元格**生成，而不是只对 resume 后待跑的单元格生成，
否则续跑会重新洗牌、让实际顺序与最初计划不一致。

入口接线：

- `v7_supp.run_lenphase`（v12 P5C 使用）先写计划，再按 `planned_cells`
  的顺序执行；`_write_phase_manifest(planned_cells=...)` 也使用同一顺序。
- `exp_lib.run()`（v6/v8/v9/v10/m0 等通用面板）在 `_plan_run` 之后写同一
  计划，并把待跑单元格按计划顺序排序；训练数值、resume 键和 `run_cfg`
  均不变。

### 48.3 审计：计划完整区组 + 实际顺序 + arm-major 硬失败

`audit_gates.py run_order`（零 GPU）对每个 `results_*` 目录：

- 读 `run_order.json`，检查每个 block 是否含全部臂各一次（完整区组）、
  每个臂的位置计数是否平衡、`design.name` 是否是声明的随机化设计；
- 读 `experiments.jsonl` 的 `kind='cell'` / `status='ok'` 事件，按
  `started_at` 重建实际顺序；同一单元格重训时取最后一次开始时间（resume
  合法，不应被误报为重复）；
- **arm-major 硬失败**：若完整实际序列的臂游程数等于臂数（A 的全部种子
  连续，再 B 的全部种子连续……），即使 `run_order.json` 声称是区组设计，
  也判 fail——这说明驱动没有执行计划；
- 无 `run_order.json` 时从事件推断顺序：能证明是 arm-major 则 fail，
  否则报 `not_verifiable_locally`，绝不把「没有计划」写成「顺序没问题」；
- 实际顺序偏离计划排列只给 warning（并行调度可以合法地改变开始顺序），
  完整区组与 arm-major 判定不受并行影响。

该审计不改变任何 PPL、种子、终点、α、Holm 族或冻结清单；它只把
「时间是否是替代解释」变成可复核的设计读数。对已经用旧顺序跑完的面板，
审计会如实报 arm-major / not_verifiable，而不会追溯性改写其结果；新面板
从下一次运行起使用区组计划。

### 48.4 入口与验证

```bash
python test_review_fixes.py       # L123 计划/位置平衡/畸形输入；L124 审计的区组、
                                  #     重训去重、arm-major、无计划不可核验
python audit_gates.py run_order   # 零 GPU：计划 + 实际顺序审计
python v12_supp.py plan           # 冻结预检不变：72/72、6 个主比较
```

边界：48 是运行顺序的设计与审计，不触碰训练/评测数值路径、
`CODE_SEMANTICS`、`confirmation_frozen.json` 或任何冻结臂/种子/终点；
`run_order.json` 只存在于被忽略的结果目录，证据归档时随 `results_*/`
一起打包。

## 49. 实验设计审查的剩余边界（不在本轮改动）

§48 修复的是运行时间混杂；下列设计/准备问题仍是显式边界，不应被
「已有大量统计门槛」掩盖：

1. **测试切片是前缀而不是概率样本。** 冻结的 48 篇文档是
   `wiki.test.raw` 中前若干条非重复行；清单已拒绝总体推断（<64 文档）并把
   估计量限定为该冻结切片，所以这是外部有效性边界，不是隐藏缺陷。未来若
   要做总体推断，应在冻结前用记录种子的、按长度分层的文档抽样，并把当前
   前缀切片保留为可复现锚点。
2. **分析代码本身没有被冻结锁哈希。** 清单钉住 test/correction/endpoint/
   alpha/decision rule，P5T 也检查这些声明字符串；但测试访问后的分析实现
   改动不会改变 `manifest_sha256` 或 `CODE_SEMANTICS`。下一次冻结可在首次
   test 读取前写入分析函数的规范化摘要。（§51 已在首次写入时绑定完整
   测量源码树摘要，并在之后的评估/分析调用中拒绝未经声明的漂移。）
3. **探索性面板的 look history 不是机器可读的。** 补种子记录在
   decision log、registry 也保证探索性面板不能升级为确认性，但没有逐面板
   的 `max_seeds` / look count。前瞻性的 panel analysis plan 能让「看了
   几次再决定继续」可见，即使仍只作探索性证据。
   （**§58 已处理**：每次新种子集合的聚合写入 `look_history.json`，并用
   精确 max-over-looks 序列 p 收紧探索性面板的 `verified`。）
4. **实际显著性仍依赖判断。** 报告拒绝 `|mean| <= std` 并给出相对效应/
   噪声，但没有预先注册的最小实际效应量；若决策是关于实际而非仅统计
   优势，未来冻结应写明 SESOI（smallest effect size of interest）。

## 50. 源码树身份与依赖锁契约（本轮可复现性硬化）

§49.2 指出「分析代码本身没有被冻结锁哈希」是剩余边界；本轮不改变 P5T
的冻结判定，而是把**源码字节**和**依赖环境**变成可记录、可重算、可审计的
证据，使「同一 commit / 同一 CODE_SEMANTICS」不再被误当作「同一份代码 /
同一套数值栈」。

### 50.1 问题

1. `git_commit` 只描述仓库历史：工作区有未提交改动时，两个 run 可以共享
   同一个 commit；从源码 tar 包运行时甚至没有 commit。`git_diff_sha256`
   只覆盖 tracked diff，未跟踪的新源码文件不在内容摘要里。
2. `CODE_SEMANTICS` 是测量语义戳，不是文件哈希（CI header 已明确警告：
   agent 可以在不 bump 戳的情况下改测量语义）。P5T 的 admission 绑定
   manifest sha，但不绑定 `exp_lib.py` 的字节。
3. `requirements.lock` 原先只钉住 5 个直接依赖，transitive 依赖仍随
   PyPI 最新版本漂移；同一个 lock 不能重建同一条数值路径。
4. 环境快照只采集了少数 `CSA_*` 开关；未记录的 run-control 变量会让两次
   「相同配置」的运行产生无法解释的差异。

### 50.2 改动（只增加证据，不改测量语义）

- `exp_lib.source_tree_manifest()` / `source_tree_summary()`：对
  **非生成、非测试**的 `*.py`（`test_*.py` / `conftest.py` 除外）以及
  `pyproject.toml`、`setup.py/cfg`、`requirements.txt`、两份冻结清单
  做确定性 SHA-256。`requirements.lock` 是**机器特定**的环境记录（README
  要求每台机器重新生成），不进入源码树摘要，否则 CPU 测试机与 CUDA 实验
  机上的同一份源码会被误判为不同；lock 的哈希与契约由
  `environment_snapshot()` 单独记录。摘要算法包含相对路径、字节数和逐文件哈希，因此重命名、
  截断、增删文件都可区分；生成目录（`results_*` / `analysis_*` /
  `__pycache__` / cache 等）被剪枝，不会把大结果树变成 provenance I/O
  扫描。
- `code_provenance()` 的 `code.source_tree` 记录 `sha256` / `n_files` /
  `bytes` / `scope`。它只进入 provenance，不进入 `run_cfg`、resume
  fingerprint 或 `exec_fp`，所以既有 cell 的 resume/pairing 完全不变。
- `v12_supp.py plan` 的 `analysis_v12/plan_snapshot.json` 额外落盘完整
  `source_tree` 和 `dependency_contract`，让「机器在批准计划时看到的是
  哪份源码/哪个环境」可异步复核。
- `exp_lib.parse_requirements_lock()` / `dependency_contract()`：解析
  `name==version` lock（忽略注释与 `--extra-index-url` 等选项），逐条与
  `importlib.metadata` 的已安装版本比对，并报告未 pin 的已安装分布。
  `mismatch` / `missing` / `invalid` 是失败；`partial` 表示所有 pin 都匹配
  但 lock 不是完整环境记录。
- `requirements.lock` 改为当前 CPU 测试环境的**完整 `pip freeze`**
  （48 个分布，含 `torch==2.14.1+cpu`，并在文件头声明 CPU wheel index），
  不再只有 5 个直接依赖。
- `audit_gates.py source`：把当前源码树与 `plan_snapshot.json`（或
  `results_*/summary.json` 的 provenance）记录的 `source_tree` 比对；
  有完整 manifest 时给出 added/removed/changed 文件；无记录时明确
  `not_verifiable_locally`，绝不把 git commit 当作源码哈希。
- `audit_gates.py dependencies`：执行 lock 契约；`partial` 作为 warning，
  `mismatch`/`missing`/`invalid` 为 fail。两者都并入 `audit_gates.py report`。
- `environment_snapshot()` 增加 `dependency_contract` 与
  `env_flags`（所有 `CSA_*` / `WT103_*` / `FWEDU_*` / `M0_*` / `V12_*`），
  补齐「未记录开关」这一类不可解释差异。

### 50.3 边界

源码树摘要是**内容证据**，不替代 `CODE_SEMANTICS`、冻结 manifest 或
`run_cfg`：它不参与训练/评测数值、seed、resume、pairing 或 P5T 的
admission 判定。它也不把测试文件、README、docs 或 CI 配置纳入测量身份；
这些文件的改动不会让一个已冻结面板看起来「代码变了」。§51 起 P5T 在
**首次写入 freeze lock 时**绑定完整源码树摘要，并在后续评估/分析调用中
核验；plan snapshot 与每个 cell 的 provenance 仍保留同一摘要，便于定位
漂移文件。

### 50.4 复核

```bash
python audit_gates.py source        # 与 plan_snapshot 的源码树比对
python audit_gates.py dependencies  # lock 与已安装版本比对
python v12_supp.py plan             # 预检不变：72/72、6 个主比较；快照含源码树/依赖契约
python test_review_fixes.py         # L137–L140 覆盖摘要、lock 解析、门禁与完整 lock
python -m pip install --dry-run --no-deps -r requirements.lock  # 验证每条 pin 可解析
```

零 GPU、不读 test split；不改变 `confirmation_frozen*.json`、冻结臂/种子/
终点/alpha/Holm family，也不需要 bump `CODE_SEMANTICS`。

## 51. 冻结锁绑定测量源码树：计划相同不再等于代码相同

### 51.1 问题：manifest 哈希与语义戳都钉不住代码字节

冻结清单的 `manifest_sha256` 钉住的是**计划**（臂、种子、终点、检验、
决策规则），`CODE_SEMANTICS` 钉住的是一个手工维护的语义常量。两者都不是
源码哈希。CI 的 continuous-loop header 已明确记录这一危险：一次编辑可以
改变测量语义而戳仍匹配，P5T 会静默接纳。§49.2 因此把「分析代码没有被
冻结锁哈希」列为剩余边界，§50 只把源码树摘要写进 plan snapshot 和逐 cell
provenance，仍未让它参与 freeze lock 的判定。

### 51.2 改动：首次访问时绑定，之后按阶段拒绝/披露

`v12_supp._check_or_write_freeze_lock*` 现在：

- 在**第一次写入**（`preaccess`，即任何 test 读取之前）把
  `exp_lib.source_tree_manifest()` 的完整清单写入锁的 `source_tree`，
  并写 `source_tree_sha256` / `source_tree_bound_at`。清单沿用 §50 的范围：
  非测试/非生成的 `*.py`、`pyproject.toml`、`requirements.txt`、两份冻结
  清单；README/docs/tests/CI/lock 不在测量身份内。
- 后续 `preaccess` / `accessed`（真实 test 评估）在源码摘要不同时**硬拒绝**，
  错误信息列出 changed/added/removed 文件（有界），并提示两条合法路径：
  恢复冻结时的源码，或 `V12_REFREEZE=1` 有意识地重冻结（前一轮降级为
  探索性，旧摘要进入 `superseded` 与 `history`）。
- `finaltest_stats` 的分析调用使用 `stage='analysis'`。默认同样硬拒绝；
  只有显式 `V12_ALLOW_SOURCE_DRIFT=1` 才允许**披露式**分析重跑：锁保留
  **原始** `source_tree`（不追溯改写认证），另写 `source_tree_drift`
  事件（旧/新摘要、变化文件、代码戳）。因此该轮不是 byte-identical
  replay，`audit_gates.py source` 会判 fail 而不是静默通过。
- 旧版本写下的锁没有 `source_tree`：若仍是 `preaccess`（尚未读 test），
  安全地补绑当前树；已 `accessed` 的旧锁保持**未绑定**，审计报
  `not_verifiable_locally`，绝不把「没有哈希」写成「源码没问题」。
- 注入的 `source_tree` 也要通过同一契约（必须有非空 sha256、不能有
  unreadable 文件）；畸形清单不能被当作认证。

`audit_gates.audit_source_integrity()` 的取证顺序变为
`plan_snapshot` → 显式 freeze lock → 发现的 `.freeze_lock.json` →
panel provenance。它比较完整逐文件清单，报告 added/removed/changed，
并把锁里的 `source_tree_drift` 作为 fail 证据。没有源码摘要时仍是
`not_verifiable_locally`。

### 51.3 边界

这是**证据/门禁**改动：不改变任何 PPL、种子、训练/评测数值路径、冻结臂、
终点、α、Holm 族或决策规则；不修改 `confirmation_frozen*.json`；也不需要
bump `CODE_SEMANTICS`。源码树摘要不进入 `run_cfg`、resume fingerprint 或
`exec_fp`，因此旧 cell 的 resume/pairing 不变。它只让「同一 manifest +
同一语义戳」不再被误当作「同一份测量代码」。

### 51.4 复核

```bash
python test_review_fixes.py       # L141 锁绑定/漂移拒绝/披露/重冻结；L142 审计与旧锁边界
python audit_gates.py source      # 有锁时按锁内源码树核验；无摘要则 not_verifiable
python v12_supp.py plan           # 预检不变：72/72、6 个主比较
```

## 52. 结果分析要检验「效应是否随条件改变」，而不是比较两个显著性（本轮方法硬化）

### 52.1 问题：两个 p 值不能相减

P5T 已经报告同一批冻结记录在 512/1024/2048/4096 上的逐长度曲线。
旧报告把每个长度的 p 值并排打印，读者很自然地会做两种错误推断：

1. 「L512 显著、L4096 不显著 ⇒ 优势只在训练长度成立」；
2. 「两个长度都显著 ⇒ 效应不随长度变化」。

第一种是经典的 **difference between significant and non-significant is
not itself significant**：两个简单效应的显著性差异不是它们之差的检验。
第二种同样没有回答「差值是否为 0」。小样本、不同基线 PPL、以及 PPL 的
乘性尺度会让这两种读法都不可靠。

### 52.2 改动：逐种子效应修饰（interaction）与趋势

`exp_lib.effect_modification(effects_by_condition, reference=...,
condition_values=..., seeds=..., min_n=6)` 接收「每个条件一个逐种子配对
效应」的映射，并做三件事：

- **简单效应**：每个条件的 `paired_effect_stats`（描述性；不把两个简单
  效应的显著性当成交互）。
- **交互**：对每个非参考条件计算逐种子
  `effect(condition) − effect(reference)`，复用精确符号翻转、t 区间、
  配对 bootstrap，并在**声明**的非参考条件族内做 Holm；缺失/非法成员以
  p=1 占位，避免缩小族放松校正。
- **趋势**：若提供严格递增的数值轴，则先对**每个种子**拟合 OLS 斜率，
  再对逐种子斜率做精确符号翻转；绝不对条件均值回归（那会丢掉配对）。

条件向量的缺失值按**整颗种子**剔除，并在 `dropped_seeds` 点名；同一个
交互面板里的简单效应、交互与斜率因此使用同一个完整种子集合。布尔值、
非有限值不是测量。`condition_values` 缺失/不单调/少于三个不同值时不报
斜率，但交互仍然可算，原因写进 `trend_reason`。

`v12_supp._length_interaction_analysis()` 把每个长度的 PPL 转成
`log(PPL_a) − log(PPL_b)`（正 = a 更差），以训练长度 512 为参考（不在
面板时退回最短长度），轴为 `log2(eval length)`。这样跨长度的效应可比，
且交互不会被基线 PPL 随长度增长所污染。结果作为
`comparisons[*].length_interaction` 落盘，并在报告的「全长度曲线」之后
渲染为 post-hoc 表。

### 52.3 解释边界

- 交互与趋势是 **post-hoc 诊断**：不进入冻结 Holm 族、不改变 `confirmed`、
  不改变任何 p/Holm/决策规则。
- **不显著的交互 ≠ 效应不变/等价**；报告必须同时给出交互 p、区间和
  「未检测到效应修饰」的措辞，不能写成「各长度相同」。
- 简单效应仍是描述性的；「一个长度显著、另一个不显著」不再被当作长度
  依赖的证据，必须看交互行。
- `n < min_n` 时交互 p 只作描述，不能称为已确认；这与报告层 n≥6 的措辞
  规则一致。
- 逐长度曲线的 Holm 族仍只含冻结主比较；交互是额外的敏感度读数，不能
  被用来追加确认性主张。

### 52.4 入口与验证

```bash
python test_review_fixes.py     # L149 effect_modification；L150 长度交互/渲染
python v12_supp.py analysis     # 在 test_stats.json 增加 length_interaction（不读新数据）
python v12_supp.py report       # 全长度曲线后追加 post-hoc 交互表
```

L149 覆盖：恒定效应的零交互、线性增长效应、Holm 族、逐种子趋势、缺失
种子整列剔除、数值键/参考解析、非单调/缺轴拒绝、空数据与畸形输入。
L150 覆盖：`log(PPL_a/PPL_b)` 口径、512 参考与回退、截断单元导致整颗
种子剔除、负/单长度拒绝，以及 Markdown 必须保留「不显著 ≠ 相同」和
「不改变冻结判定」的边界。

## 53. 分析单元：WikiText 的「文档」是原始行，不是文章（本轮数据口径硬化）

§43 把「固定行 → 逐 token 文档映射」做对了，但映射里的 `document` 仍是
**原始数据行**。上游 `wikitext-103-raw-v1` builder 的 `_generate_examples`
逐行 yield：一行标题、若干行段落、若干行 ` = = Section = = ` 小节标题都各是
一个 row；空行只是文章内部的分节空行，不是文章边界。实测 WikiText-2
validation 的 60 篇文章对应 2 461 个非空 row、60 个顶层 ` = Title = ` 标题。
因此旧口径下的：

- **去重**是行级去重：一篇被改写/重排的文章，只要每行都不同，就绕过了
  精确 + SimHash 行索引；
- **per-document bootstrap** 重采样的是段落行而不是独立文章；冻结前缀里
  48 条 4097-token 行可能覆盖几十篇文章、上千行，「<64 文档」的总体门槛
  被行数虚构地满足；
- 冻结清单 §49.1 所说的「48 篇文档是前若干条非重复行」本身就把 row 与
  article 混用了。

本轮把单元变成可执行概念，而不是再换一个未经验证的标签。

### 53.1 分词器必须可逆：ByteLevel decoder 缺失

`build_tokenizer` 只设置了 `ByteLevel(add_prefix_space=True)` 预分词器，没有
设置其逆 `ByteLevel` decoder。后果是：

- `Tokenizer.decode` 返回 BPE 表面形式（`Ġthe`），不是原文；任何把 token
  id 映射回原始行/文章的数据审计都无法工作；
- `boundary_token_ids` 对每个 id 调用 `decode` 后检查是否全为标点。
  带前导空格的标点 token（`Ġ.` `Ġ,` `Ġ:` `Ġ;` …）解码带 `Ġ`，被 `strip()`
  后仍不是纯标点，于是**边界 token 掩码漏掉这些 token**。在真实
  BPE-8k 上这是系统性少计，不是显示问题。

修正：`build_tokenizer` 在训练前挂上 `decoders.ByteLevel()`；`encode` 与
token id 完全不变，但 `decode` 变成字节级可逆，`boundary_token_ids` 能识别
空格前缀标点。旧缓存的分词器 JSON 里 `decoder: null`，由
`ensure_bytelevel_decoder()` 在审计的**内存**中补齐，不重写旧文件；新缓存的
`tok_*.json` 自带 decoder。**token id、训练流、PPL 均不变**；只有新缓存
的 `val_bnd` / `test_bnd` 掩码会包含此前漏掉的标点 token（边界诊断口径
修正，属于数据属性，不是主终点）。

### 53.2 单元原语与审计

- `exp_lib.wikitext_article_groups(rows)`：空行忽略，每个顶层
  ` = Title = ` 标题开启一篇新文章；` = = Section = = ` 不切分。返回
  `row_group`（空行为 -1）、`n_articles`、`group_sizes`、`preamble_groups`、
  `valid` 与问题列表。没有顶层标题时返回单个 group 且 `valid=False`，不
  静默把每行当成文章。
- `wikitext_article_text()` / `wikitext_article_ids()`：文章正文按行保序
  拼接后取内容哈希；任何一行改动都会改变文章 id。
- `line_spans_from_token_doc_ids()` / `decode_token_runs()`：把 per-token
  id 映射还原成「一段连续同 id = 一个服务单元」并解码；这是在不重读
  `wiki.test.raw` 的前提下把已冻结切片重新分组的唯一路径。
- `audit_gates.audit_document_unit(cache_dir=..., seq_len=..., raw_dir=...)`：
  1. 解码 test / validation 的 per-token 映射，按「每段是否以顶层标题
     开头」判定服务单元是 `line` 还是 `article`，并用解码文本的内容哈希
     复核 id 与字节自洽；切片末尾被 cap 截断的最后一行为允许的 cap 伪影；
  2. line 单元按顶层标题重分组，报告 `n_lines / n_articles`、
     pseudoreplication 倍数，以及冻结前缀（`n_seqs × seq_len`）里真正独立
     的文章数，并用它判定仓库的 64-document 总体门槛；
  3. 输出 `document_level_claims_allowed`：只有文章级单元才允许未加限定的
     「per-document」区间/总体主张；line 单元返回 `warnings` 并给出明确
     理由；
  4. 当本地有 `wiki.train.raw` / `wiki.valid.raw` 时，把 raw 行哈希映射回
     缓存中实际服务的行，检查**行级精确去重看不见的文章级泄漏**：同一篇文章
     的行跨越 train/validation（标题相同但行不同）判 `fail`；再用与 loader
     相同的 SimHash LSH 检查整篇被改写的近重复文章，命中给 warning 并列出
     样例；
  5. `wiki.test.raw` **永不读取**（那会消耗一次性 test 访问）；test 的文章
     分组只从已落盘的 id + per-token 映射 + 分词器恢复。缺缓存/缺 raw 时
     明确返回 `not_verifiable_locally`，不会当成 clean。

### 53.3 元数据与边界

新构建的 WikiText train/validation 与 test cache 在 meta 里写
`document_unit: "line"`、`document_unit_is_article: false` 和
`document_unit_definition`；`_cached_split_problems` 拒绝未知单元，以及
「声明 article 却没有 article 级侧车」的自相矛盾缓存。旧 cache 没有该字段，
读取路径不受影响，审计会从解码结果推断。

边界：

- **不改变任何 token id、split 成员、训练/评测切片或 PPL**；不修改
  `confirmation_frozen*.json`、`CODE_SEMANTICS`、fixture 或 CI 配置。
- 文章级侧车（`*_article_ids` / `*_token_article_ids`）**尚未**写入缓存：
  本轮的 per-document 向量仍然是行级。要把评测不确定度真正改成文章级，
  应在**下一次冻结前**写入文章级侧车并让 `eval_length_gen(doc_ids=...)`
  使用它；冻结后的旧 cache 只报告真实文章数与「不允许文档级主张」，
  不静默改冻结分析。
- 文章级泄漏检查依赖 raw train/valid；没有 raw 时只报告单元，不声称
  已排除文章级泄漏。test 侧不读 raw，因此 test 内部的分组是完整的，
  但 test 与 train 的**部分文章**交叉（若存在）只能由未来缓存中的文章 id
  侧车证明。

入口与验证：

```bash
python test_review_fixes.py          # L151 单元原语 + 可逆分词器；L152 单元审计/文章泄漏
python audit_gates.py document_unit  # 单命令审计（缺缓存 -> not_verifiable_locally）
python audit_gates.py report         # 汇总新增 document_unit 段
```

L151 覆盖：顶层标题切分、空行不切分、section 标题不切分、preamble、
无标题回退、非字符串拒绝、文章 id 稳定性、per-token run、ByteLevel 解码
往返与空格前缀标点进入边界集。L152 覆盖：line 单元 → `warnings` 且
`document_level_claims_allowed=False`、冻结前缀文章数、缺 raw 的
`not_verifiable_locally`、line/article 结构分类，以及「同一文章标题的行
跨 train/validation」被判 `fail` 的文章级泄漏。

> **后续（§54）。** 本节只把「行 vs 文章」做成审计与拒绝主张；§54 进一步
> 把文章聚类接进评测记录与结果分析：`nll_per_article` 等字段与行级字段
> 并存，`doc_sampling` 在可用时按文章 cluster 重采样。§53.3 所说的
> 「per-document 向量仍是行级」指**冻结的行级契约未改**；新增的文章向量
> 是追加证据，不替换旧字段。

## 54. 评测不确定性必须按文章聚类，而不是按原始行（本轮统计口径硬化）

§53 把「分析单元是原始行，不是文章」变成可审计的诊断，但**主分析仍然
按行级 per-document 向量做 bootstrap**。这不只是标签问题：一行是文章内的
一个段落/标题行，同一篇文章的多行被当作独立样本重采样，等于把段落当成了
独立的语料抽样单位。区间会因此偏窄，`n_docs` 也会被行数虚构地抬高。§53
把这个边界写清楚了，本轮把文章级聚类真正接进评测记录与结果分析，同时
**不改冻结清单、不改训练、不改主终点 PPL、不改任何 line 字段**。

### 54.1 问题：行级 bootstrap 的伪重复

冻结评测的 48 条固定窗口按逐 token 行映射聚合后，得到的是该切片覆盖的
raw lines（通常多于 48 条）的「per-document」向量；每条向量其实是一个
raw row（标题、段落或 section 标题），一篇 WikiText 文章通常跨多行。
于是：

- 同一篇文章的若干段落被当成独立抽样单位，文章内相关被忽略；
- 文章数远小于行数，真正的独立单位数被高估；
- `doc_sampling` 的区间、`doc_a_better_fraction`、top-share 等描述量
  都作用在错误的单位上；
- 冻结清单把 48 条评测行写成「48 篇文档」；行级 `n_docs` 计的是行数，
  二者都不是文章数。

本地 512×512 验证缓存（2026-10-23 的当前工作区）解码后是 2022 条行级
run、56 篇顶层文章，平均 36.1 行/篇。也就是说，行级 bootstrap 会把同一篇
文章里约 36 个段落当成独立抽样单位；这正是伪重复的量级。

§53 的审计已经能报告真实文章数与 `document_level_claims_allowed=False`，
但那只阻止了主张，并没有让区间本身变对。本轮补上后者。

### 54.2 文章聚类的构造：标题 + 组序号

新增 `exp_lib.wikitext_article_row_ids(rows)`：按顶层 ` = Title = ` 标题
把 raw rows 分组成文章，再为每篇文章生成一个**聚类 id**。id 的键是

```
(组序号, 归一化后的顶层标题)
```

而不是文章正文的内容哈希。原因：

- 同一个冻结切片既可以从 raw rows 构造，也可以从已落盘的逐 token 行映射
  解码后重建；两者的正文可能有无害的空白差异，但标题一致。用标题做键，
  两条路径得到同一组聚类 id。
- 正文编辑不应把同一篇文章拆成两个 cluster；标题/组序号才是文章身份。
- 组序号保证同一标题的两篇文章（罕见）不会合并。

无标题的 preamble 用「空白归一化后的正文」做键，并报告
`preamble_groups`；它被当作一个独立 cluster（保守：不与其在切片外的
续文合并）。`wikitext_article_ids()`（内容哈希，用于泄漏审计）保持不变，
两个 id 的用途不同：前者是**聚类标签**，后者是**内容身份**。

### 54.3 不需要重读 test split 的重建

`exp_lib.wikitext_token_article_ids_from_line_map(token_ids, line_ids, tok)`：

1. `line_spans_from_token_doc_ids` 把逐 token 行映射切成连续 run；
2. `decode_token_runs` 用持久化的 ByteLevel decoder 解出每一行的文本
   （legacy `decoder: null` 只在内存中补，不重写缓存）；
3. `wikitext_article_groups` 按顶层标题分组；
4. 把每个 run 的 token 写成其文章的聚类 id。

因此 P5T 在**已经读过** test slice 之后，只需读 `test_tok_*.json` 就能
重建文章映射，不会第二次访问 `wiki.test.raw`，也不修改任何缓存字节。
`load_wikitext_test_tokenizer()` 只读 tokenizer JSON，并核对宽度；缺
tokenizer 的 legacy cache 明确回退到行级，不猜 id 空间。

`wikitext_token_article_ids_for_rows(rows, lengths, cap_tokens)` 是
cache-build 对偶：直接从 raw rows + 每行 token 长度构造同样的映射，供未来
在首次 test 读取前落文章侧车。

### 54.4 评测记录：文章字段是**追加**的

`v7_supp.eval_length_gen(..., article_ids=...)` 在原有
`nll_per_doc` / `n_tokens_per_doc` / `ppl_per_doc` / `doc_ids` **之外**
追加：

```
article_granularity = 'article'
n_articles
nll_per_article
n_tokens_per_article
ppl_per_article
article_ids
article_ids_fp
nll_per_article_total
n_tokens_article_total
```

行级字段逐位不变（同一 `nll_cols`，只是多一次分组求和）。池化 `ppl`、
`nll_sum`、`n_tok`、`ppl_per_seq` 完全不变；冻结的 line 契约仍可复核。
`article_ids` 是聚类标签的十六进制形式，使用与 line 相同的定长格式。

`_by_len_shape_reason` 对文章字段做与行级字段同等的准入检查：向量等长、
token 计数为正整数且精确分割 `n_tok`、NLL 有限且重构 `nll_sum`、
`n_articles` 一致、`article_ids` 无重复、文章数不得超过行级文档数。
部分写入或自相矛盾的文章字段被拒绝，而不是被当作「没有文章字段」。

### 54.5 结果分析：文章优先、回退显式

`_doc_level_analysis` 的单位选择顺序：

1. 每条记录的每个 cell 都有完整文章向量 → `granularity='article'`；
2. 只有部分记录有 → 若全部有行级向量，回退到 `'document'` 并在
   `article_fallback_reason` 里说明；否则拒绝混合估计量；
3. 没有文章/行级向量 → 退回 `'row'`。

文章模式下，`_doc_nll_matrix(..., unit='article')` 读取
`nll_per_article` / `n_tokens_per_article` / `ppl_per_article`，逐文章
配对、按 token 池化、对文章做 cluster bootstrap。`n_docs` 字段保留为
「单位数」的历史名字，同时给出 `unit` / `n_units` / `n_articles` /
`pooled_ci_unit`，避免下游把文章数误读成行数。`_doc_ci_markdown_cell`
对文章显示「文章」，对行级显示「文档/行」。文章 cluster 少于 20 时，
`small_cluster_warning` 明确说明百分位区间很粗、必须连同 n 报告，不能当作
总体区间。

种群门槛（`population_claim_allowed`）优先使用文章数：只要任一 cell 有
文章向量就用文章单位；若文章覆盖在不同 cell 间不一致，则**不**用更大的
行数替代，直接拒绝种群主张。这与「文章是独立单位」的统计含义一致。由于
48 条 4097-token 行通常只覆盖几十篇文章，门槛几乎必然仍是 `n<64`；
本轮的贡献是让拒绝理由和区间都建立在正确单位上，而不是放宽门槛。

### 54.6 边界

- **不改任何 token、split、训练、checkpoint、PPL 点估计、种子、alpha、
  Holm family、决策规则或 `confirmation_frozen*.json`**；不需要 bump
  `CODE_SEMANTICS`。
- 文章聚类是**分析单位**，不是新数据；文章 id 不进入 `run_cfg`、resume
  指纹或 pairing 键。
- 切片末尾被 cap 截断的文章仍是一个 cluster，但其 id 只基于切片内可见的
  标题/组序号；这是可复现的切片内身份，不声称与全文文章 id 相同。
- 切片从文章中间开始且没有顶层标题时，`valid_for_clustering=False`，明确
  回退行级；不把「无标题」误判成「一篇文章」。
- 文章级侧车尚未写入缓存；本轮在 P5T 分析时从行映射重建。未来应在首次
  test 读取前写 `test_token_article_ids_*`，让缓存自身携带该单位。
- 历史探索性面板（v6–v11）的落盘记录没有文章字段，仍按行级分析；本轮
  不重跑、不回填，也不把旧记录重新标记成文章级。`eval_length_gen` 的
  `article_ids=` 是给新面板/新分析的入口。
- 文章级区间仍只描述冻结切片的文章抽样，不是语言总体；`n<64` 时种群
  主张继续拒绝。

### 54.7 复核

```bash
python test_review_fixes.py     # L153 文章原语/重建；L154 评测字段 + 分析偏好
python v12_supp.py plan         # 零 GPU、不读 test，冻结计划不变
python audit_gates.py report     # 其余审计不变
```

L153 覆盖：空白行不并入文章、标题变更改变聚类、正文变更不改变聚类、
raw/lengths 与 decode 两条路径得到同一映射、legacy decoder 内存修复、
错误总长度的扁平映射被拒绝、tokenizer 宽度不匹配被拒绝、驱动 helper
从缓存重建同一文章映射。L154 覆盖：文章字段追加后行级字段逐位不变、
文章 token/NLL 精确重构池化总量、准入拒绝部分/重复/不一致字段、分析
优先文章单位、混合覆盖回退行级并留痕、跨种子文章 id 不一致被拒绝。

## 55. 抽样设计：source-order prefix 不是概率样本（本轮数据方法硬化）

### 55.1 问题

§49.1 已经承认：冻结的 test slice 是 test split 按源顺序切出的**前若干
行**，不是对 test 文章总体的概率抽样；train/validation 同样如此。
`load_wikitext` / `load_wikitest` 的构造规则是确定的、可复现的、泄漏安全
的，但它回答的是「源顺序下最先出现的足量 token 是什么」，而不是「从文章
总体中随机抽到的样本是什么」。因此：

- 数据覆盖只到源文件前部；后部文章、长尾长度层可能完全没有进入切片；
- 固定行数/ token 数不是 inclusion probability，不能把切片均值直接外推到
  WikiText 文章总体；
- 现有缓存只在 `split_strategy` 里描述「官方 split + 去重」，没有字段让
  读者区分「确定性前缀」与「可复核的概率样本」；
- §54 的文章聚类修正了**分析单位**，但没有修正**单位如何进入切片**。

本轮补上的是抽样层：不改 token、split、cap、去重、训练或冻结计划，只
增加可复核的抽样计划原语、缓存侧抽样元数据和一个独立审计。

### 55.2 通用分层计划：`document_stratified_plan`

```python
plan = document_stratified_plan(
    sizes, n_select, seed=0, n_strata=4,
    allocation='proportional', min_per_stratum=1,
    source_fp='<corpus snapshot fingerprint>')
```

- 单位按 `(size, index)` 排序后切成 `n_strata` 个**等数量**层；
- 每层用由 `seed + stratum + source_fp + n_units + n_select` 哈希出的
  独立 RNG 做**无放回**抽样；
- `allocation` 支持 `proportional`（按层内单位数）、`equal`、以及显式的
  `size_weighted`（按层内 size 总和；只有在分析权重也按 size 调整时才
  应使用）；
- 当 `n_select >= n_strata * min_per_stratum` 时保证每层至少一个单位；
- 计划记录 selected indices、每层 n_units/n_selected、`sizes_fp`、
  `source_fp` 和 `plan_fp`（去掉 `plan_fp` 后的 canonical JSON SHA-256）。

`sampling_plan_problems(plan)` 独立复核：index 范围/重复/排序、层计数
之和、`all_strata_represented`、`selected_without_replacement`、
`plan_fp` 是否与内容一致。任何冻结后被改写或层计数不再闭合的计划都会被
拒绝。`select_sampled_items(items, plan)` 只按已校验的 indices 取数，
不重新抽样。

### 55.3 文章级抽样：`wikitext_article_sampling_plan`

```python
rep = wikitext_article_sampling_plan(
    rows, n_articles, seed=0, n_strata=4,
    size_unit='rows',           # 或 'chars'
    allocation='proportional', source_fp=None)
```

- 先用 `wikitext_article_groups` 按顶层 ` = Title = ` 标题切成完整文章；
  没有标题的语料直接 `ValueError`，绝不把每行当成伪文章；
- 以文章行数（或字符数）分层，抽中的是**整篇文章**；
- 返回 `selected_article_indices`、`selected_article_ids`、
  `selected_article_sizes` 和排序后的 `selected_row_indices`，调用方可以
  只物化这些行，不会把一篇文章拆到 selected/unselected 两侧；
- `plan` 字段就是 55.2 的计划，`plan_fp` 与之绑定。

这是给**下一次 freeze** 的入口：目前冻结的 v12 缓存不重采样，新计划也
不会自动替换现有 prefix 切片。

### 55.4 缓存侧抽样元数据

新构建的 train/validation/test meta 增加 `sampling`（追加字段，不改
`src_fp`/`val_fp`/`test_fp`/`file_sigs`/token ids）：

```text
sampling.train.design                 = 'source_order_prefix'
sampling.train.selection_rule         = 'first_nonempty_rows_..._until_token_cap'
sampling.train.source_rows            = 上游 split 行数
sampling.train.nonempty_rows_scanned  = 实际扫到的非空行数
sampling.train.rows_served            = 实际进入 token 流的行数
sampling.train.tokens_served          = 实际 token 数
sampling.train.token_cap              = 请求的 cap
sampling.train.cap_reached            = 是否因 cap 停止
sampling.train.duplicate_documents_skipped = {exact, near}
sampling.train.representativeness_claim_allowed = false
sampling.train.population_claim_allowed         = false
```

validation 额外记录 `sequences_served/sequences_requested`；test 额外记录
`rows_after_dedup`、`tokens_tokenised`。legacy 缓存没有该字段时，审计将其
标为 `prefix_inferred_legacy`，不补写、不改字节。

### 55.5 审计：`audit_gates.audit_sample_design`

```bash
python audit_gates.py sample_design
python audit_gates.py report        # 汇总里新增 sample_design
```

审计逻辑：

1. 读取每个 meta 的 `sampling` 块；缺失则明确标记
   `prefix_inferred_legacy`；
2. 对 `source_order_prefix`：`representativeness_claim_allowed` 和
   `population_claim_allowed` 必须为 false；若元数据反而声称可外推，直接
   `fail`。这是**硬门**，不是脚注；
3. 对 `stratified_document`：调用
   `sampling_plan_problems(plan)`；计划无效、`source_fp` 与 block 不一致、
   层未覆盖、或 `population_claim_allowed=true` 但独立单位少于 64，均
   `fail`；
4. 若 `tokens_served < token_cap` 且 `cap_reached != true`，报告「源提前
   耗尽」，防止把实际更短的训练/评测流当成声明的长度；
5. 若本地存在 `wiki.train.raw` / `wiki.valid.raw`，用 doc-id sidecar 把
   served 行映射回文章：
   - 行覆盖率 = served raw lines / raw nonempty lines；
   - 文章覆盖率 = 至少一行被 serve 的文章数 / raw 文章数；
   - served 与 source 的文章长度分布 KS 统计量，阈值取双样本 5% 临界值
     `1.36 * sqrt((n1+n2)/(n1*n2))`（上限 1.0）；
   - 按 source 文章长度四分位报告覆盖率；某层完全没有 served 文章时给出
     warning；
6. **永不读取 `wiki.test.raw`**。test 的 raw coverage 固定
   `not_applicable`，只用 build-time metadata 报告 source/served 行数。
   这是 §17 一次性 held-out 访问纪律的延续。

状态语义：`ok` 只表示声明与可复核的证据一致；`warnings` 包含
「prefix 不可外推」「源提前耗尽」「长度分布偏移」「某长度层未覆盖」；
`not_verifiable_locally` 用于缺缓存/缺 raw；`fail` 只给自相矛盾或过度
声称的元数据。

### 55.6 边界

- **不改** token、split、cap、去重、训练、checkpoint、PPL 点估计、种子、
  alpha、Holm family、决策规则或 `confirmation_frozen*.json`；不 bump
  `CODE_SEMANTICS`。
- 现有 v12 冻结切片仍是 prefix；本轮**不重采样、不回填、不重跑**。§49.1
  的外部效度限制继续成立，只是现在有机器可读的审计证据。
- `source_order_prefix` 是有效的研究设计（预注册的固定切片），只是不能
  冒充概率样本；它的重复实验随机性来自训练种子，不来自文章抽样。
- 分层计划只处理「哪些单位进入切片」；进入切片后的 token 截断、文章被
  cap 从中间切断、近重复删除仍按原规则执行，并在审计中分开报告。
- 概率抽样的 `representativeness_claim_allowed=true` 只表示抽样机制可
  复核，不等于 48 行/几十篇文章足以支持总体区间；`population_claim_allowed`
  继续要求至少 64 个独立单位（§45/§54 的门槛）。

### 55.7 复核

```bash
python test_review_fixes.py   # L158 计划确定性/防篡改；L159 整文章抽样；L160 审计
python audit_gates.py sample_design
python audit_gates.py report
```

L158 覆盖：同参数逐位复现、无放回/排序/层计数闭合、`source_fp` 改变选择、
计划被篡改后 `plan_fp`/层计数检查拒绝、非法 n/seed/NaN/allocation 被拒。
L159 覆盖：只选整篇文章、返回行索引与文章组完全一致、无标题语料拒绝。
L160 覆盖：prefix 块标 warning 且 claims 为 false、prefix 上过度声称判
fail、合法 stratified 计划通过、少于 64 单位时 population claim 判 fail、
test 块只读 metadata 且 `raw_coverage=not_applicable`。

## 56. 消融要判「实际等价/可删」，不能把不显著当成零（本轮方法硬化）

### 56.1 问题：显著性检验不能回答「模块能不能删」

§19/§25 的门槛解决的是「有没有可检出的差异」：`verified` 要求受控、
Holm 校正后 p<α 且效应大于种子噪声。它没有、也不能回答消融真正要问的
第二个问题：

> 去掉这个模块后，PPL 的变化是否小到可以认为它**实际**不重要？

旧审计把没有校正显著性的对照放进 `measured` / `noise_level` /
`resolution_limited`，这三个标签都不是「等价」。审稿人或下游读者很容易
把 `noise_level` 读成「模块无用」，但 n=3–6 的符号翻转检验本来就没有
足够分辨率排除一个中等效应；这正是 §49.4 点名的剩余边界。反过来，大 n
下一个 0.05% 的 PPL 差异也可能显著，把它算成「模块有用」同样是把统计
显著性误当成实际重要性。**两种错误方向都需要一个预先固定的最小实际
效应量（SESOI）和等价性检验，而不是继续加一句「不显著不代表无差异」。**

### 56.2 设计：预注册 relative SESOI + 配对 TOST + 族校正

`ABLATION_SESOI` 为 `ABLATION_DESIGN` 的每个组件声明一个相对 margin
（默认 1% of the without-arm PPL）和 rationale。对**尚未看结果的新面板**，
这是预注册；对已经跑完的探索性面板，这是本轮分析开始前固定的 convention
（不是事后从结果反推），并且仍不升级为确认性证据。margin 是**相对
without 臂主终点 PPL 的百分比**，不是从观察到的种子方差反推的——后者会让
噪声大的面板更容易「等价」，正好把不确定性变成结论。相对 margin 在每条
对照上换算为绝对 PPL 边界：

```
margin_ppl = without_arm_ppl * relative_pct / 100
```

对逐种子配对差 Δ = with − without（PPL，负 = 模块有用）同时做两件事：

1. **优越性**：沿用 §19 的受控配对、精确符号翻转与 Holm；只有
   `significant_global` 且 `|Δ| > margin_ppl` 才算 `superior`；
2. **等价性**：`tost_paired(deltas, margin_ppl, conf=1-2α)` 的两个单边
   检验都通过，且 TOST p 经同一 Holm 族校正后仍 < α，才算 `equivalent`。
   TOST 的 p 取两个单边 p 的最大值，再对声明族做 Holm——交集-并集检验
   下这是保守且可复核的做法。

判定优先级（`ablation_practical_decision`）：

| decision | 含义 |
|---|---|
| `superior` | 校正后显著、方向符合声明、幅度 > SESOI，且不由单个种子翻转 |
| `inferior` | 校正后显著、方向相反、幅度 > SESOI（真正的有害/反向证据） |
| `sub_sesoi` | 统计上可检出，但 abs(Δ) ≤ SESOI（大 n 的微小差异；不算实际有用） |
| `equivalent` | 没有显著差异 **且** TOST 区间整体落在 ±SESOI 内、族校正后 p<α |
| `seed_driven_improvement` | 显著且 >SESOI，但去掉一个种子会翻转均值方向；不升级为实际结论 |
| `detectable_difference` | 方向未声明（`direction=0`）时的显著差异，只报区间 |
| `indeterminate` / `underpowered` / `no_data` / `not_declared` | 数据或设计不足以判「有用」或「可删」，不得写成 null |

`equivalent` 与「不显著」的区别是硬门：没有 TOST 通过，绝不产生
`equivalent`；没有预注册 margin，连判定入口都不开（`not_declared`）。

### 56.3 接入 `methodology_audit`

- 新增 `methodology_audit(..., sesoi=None)`：`None` 用 `ABLATION_SESOI`，
  传 `{}` 显式关闭，传 mapping 则覆盖（仍受同一校验）。
- 每条受控对照的 `entry['sesoi']` 记录 margin、TOST p/CI、族校正 p、
  `decision`/`claim`/`reason`；`entry['practical_decision']` 供分类器使用。
- `_classify_group_status` 只在**声明了 margin** 时收紧：
  - 显著但全部 `sub_sesoi` → `sub_sesoi`，不进 `verified`；
  - 显著但存在非 `superior` 的实际判定 → `practically_indeterminate`；
  - 全部 `equivalent` → `equivalent`，不再落入 `noise_level`；
  - 未声明 SESOI 的自定义设计保持旧行为，避免破坏历史面板。
- 报告新增顶层 `sesoi` 块与
  `practically_useful_components` / `equivalent_components` /
  `sub_sesoi_components` / `practically_harmful_components` /
  `practically_inconclusive_components`，并把 `equivalent`、`sub_sesoi`
  与 `verified` 分开打印。
- `audit_gates.py sesoi` 在设计期检查：每个 `ABLATION_DESIGN` 组件都有
  margin、单位唯一、有限且 >0、能换算成正 PPL 边界、没有拼错组件名、且
  声明的留一对照在 `RUN_ABL` 中完整排程。缺声明直接 `fail`。

### 56.4 边界

- 只作用于探索性消融面板；**不改** `confirmation_frozen*.json`、冻结臂/
  种子/终点/α/Holm 族或决策规则，也不改任何训练/评测数值路径。
- 对历史面板，1% 是现在固定下来、用于重新阅读旧结果的 convention，**不是**
  当年第一次 test 访问前写下的预注册；因此它不能把探索性证据升级为
  确认性。真正的确认性 SESOI 必须在首次 test 读取前进入冻结清单。
- 1% 是本研究对架构级消融预先采用的 convention；具体面板可在**看结果前**
  用 `sesoi=` 传入更小/更大的 margin，但必须记录 rationale。它不是从数据
  估计的，也不能在看到 `p` 后调整。
- TOST 用配对 t 区间（与 `tost_paired` 的既有假设一致），不是精确符号
  翻转的等价版本；它在小 n 下偏保守/依赖近似，因此 `n<min_n` 一律
  `underpowered`，且 `equivalent` 仍要求族校正后 p<α。
- `equivalent` 的结论是「在该 SESOI 下实际可删」，不是「总体严格零」；
  报告必须同时给出 margin、TOST CI、Holm p 与 n，不能只写「无显著差异」。
- **组件级「可删」要求完整消融**：如果某组件只测了声明对照的一部分，即使
  已测的那条 TOST 等价，也只标为描述性/不确定，不能把部分消融升级为
  「整个模块可删」。`_classify_group_status` 的 `equivalent` 分支因此额外
  要求所有声明对照都在场。
- `sub_sesoi` 不是「模块无用」：它只说明效应小于预先声明的实际重要性
  阈值；若研究问题关心更小效应，应换一个**预先注册**的更小 margin。

### 56.5 入口与验证

```bash
python test_review_fixes.py          # L169 判定原语；L170 审计分级；L171 声明门
python audit_gates.py sesoi          # 零 GPU：SESOI 声明 + 排程完整性
python audit_gates.py report         # 汇总里新增 sesoi
```

L169 覆盖：等价/亚 SESOI/反向有害/未声明/欠功效、相对 margin 换算、零
baseline、畸形声明 fail-closed。L170 覆盖：2% 效应仍 `verified`；0.2%
可检出效应被降为 `sub_sesoi` 且不进 `verified`；精确零效应成为
`equivalent`（不是 null）；反向 1.6% 成为 `harmful`；`sesoi={}` 恢复旧
行为；未测声明对照仍占 Holm 槽位。L171 覆盖：缺组件、零/NaN margin、
拼错组件名都在设计期被发现。

## 57. 结果稳健性要问到「单个评测单位」：leave-one-unit-out 影响与 jackknife（本轮结果分析硬化）

### 57.1 问题：种子稳健不等于评测单位稳健

§42 的脆弱性/逐种子删除回答的是「结论是否依赖某一个训练种子」；§54 的
cluster-robust 区间回答的是「按文章聚类后评测抽样的不确定性有多宽」。
两者都不能回答一个更直接的结果分析问题：

> 报告的 pooled ΔPPL 是否主要由**一个评测文章/文档**贡献？删掉它，方向
> 或幅度还成立吗？

旧报告只有 `top_doc_abs_share = max|Δ_unit| / Σ|Δ_unit|` 这一描述量。
它能提示某个单位的逐文档 PPL 差很大，但（a）逐文档 PPL 的等权差与主终点
的 token 池化 Δ 不是同一估计量；（b）它没有做「删除该单位后重算」；
（c）cluster-robust 区间可能在一个高杠杆单位存在时仍然很窄。把
`top_doc_abs_share` 当作「单文档影响」会同时犯 §54 已经指出的估计量混用
错误。

### 57.2 方法：在同一个 pooled 估计量上做精确 delete-one-unit

新增 `exp_lib.paired_nll_unit_influence(nll_a, nll_b, n_tokens, *,
unit_ids=..., conf=0.95, top_k=5)`：

- 输入仍是 `[n_seeds, n_units]` 的逐单位 NLL 与共享 token 数；单位可以是
  文章、行级文档或固定行。主终点保持

  ```
  Δ = mean_s [ exp(Σ_i a_si / T) − exp(Σ_i b_si / T) ]
  ```

  不做逐文档等权平均。
- 对每个单位 i 精确重算 `Δ_{-i}`（去掉 i 后的 token 池化 PPL 差，再对
  种子取均值），得到 `loo_delta_ppl` / `loo_delta_log_ratio` 与
  `loo_change_*`。这与「把逐单位 PPL 差做一次 bootstrap」回答的问题不同：
  后者重采样等权单位，前者保持主终点的 token 权重。
- 同时暴露 `_paired_nll_doc_ci` 已经在用、但此前只用于方差求和的
  delta-method influence `psi_i` / `psi_lr_i`，便于检查局部影响。
- 用全部 LOO 折计算 delete-one-unit jackknife：

  ```
  se_jack = sqrt((D−1)/D · Σ_i (Δ_{-i} − mean(Δ_{-i}))²)
  ```

  并给出 t(df=D−1) 区间；同时报告它与 cluster-robust SE 的比值。比值远大于
  1 是「高杠杆单位」警告，而不是新的显著性检验。
- 方向稳定性按**尺度无关的 log 效应**判断：`sign_stable_log_ratio` 要求
  所有有限 LOO 折与全 panel 的 `Δlog PPL` 同号；`single_unit_dominates`
  表示删除一个单位使 log 效应变化 ≥100%。`fragile_to_one_unit` 是二者的
  并集；全 panel 效应恰为 0 时返回 `None`（没有方向可翻转，不能伪造
  stable/fragile）。
- `top_k` 按 |Δlog PPL 变化| 排序；`unit_ids` 必须与单位数等长且唯一。
  某个 LOO 折因剩余 token 池化的 PPL 溢出而不可用时，该折标为不可用，
  jackknife 明确关闭，而不是把 inf/nan 当成一个点。

### 57.3 接入 v12 分析、报告与审计

- `v12_supp._doc_level_analysis` 只在 pooled cluster CI 已经与记录的主
  终点逐值对齐时计算影响块（否则影响属于另一个估计量）；单位 id 取
  `article_ids`（优先）或 `doc_ids`，要求唯一且顺序已跨臂/种子核对。
  行级（`row`）单元没有 token 池化主终点，不生成该块。
- 结果写入 `comparisons[*].doc_sampling.unit_influence`：
  `full_delta_*`、`delta_method_influence_*`、`loo_*`、`jackknife_*`、
  `se_ratio_*`、`most_influential_unit`、`top_units` 与
  `fragile_to_one_unit`；`label_source` 区分 `provided` 与
  `index_fallback`。不可用时写 `unit_influence_unavailable_reason`，
  不静默省略。
- `v12_supp._doc_influence_markdown` 在 REPORT_v12 的结论稳健性表之后
  渲染「单元影响诊断」：最敏感单位、删除后 Δ、最大单单位变化、删除后
  是否同向、jackknife 区间与标记。`单单元主导` 行会额外给出「引用时必须
  同时报告该敏感度，不能描述为对评测文档稳健」的提示。
- `audit_gates.audit_unit_influence`（`python audit_gates.py unit_influence`）
  独立检查每个影响块：`full_delta_ppl` 必须复现报告 pooled Δ、单位 id/
  LOO 向量长度必须闭合、`fragile_to_one_unit` 必须是 bool/null。块自相
  矛盾判 `fail`；单单元主导或 pooled CI 存在但没有影响块的陈旧产物判
  `warnings`；没有产物判 `not_verifiable_locally`。该审计**不**把 fragile
  改写成「未确认」——冻结决策规则没有这条阈值。

### 57.4 边界

- **不改**任何 token、split、训练、checkpoint、PPL 点估计、种子、alpha、
  Holm 族、决策规则或 `confirmation_frozen*.json`；不需要 bump
  `CODE_SEMANTICS`。`confirmed` 仍只由 §41/§42 的冻结规则产生。
- 这是 **post-hoc 描述**：条件在已训练种子上；单位不是随机化分配；删除
  一个单位后描述的是「冻结切片减一个单位」，不是语言总体，也不支持
  `population_claim_allowed`。
- `fragile_to_one_unit=False` 只表示「没有单个单位的删除会翻转方向或
  使 log 效应变化 ≥100%」，不是「对所有文章稳健」的证书；仍然必须与
  cluster 区间、n、文章数和 §54 的小 cluster 警告一起读。
- 100% 阈值是预先写明的描述性 convention（不是从这批结果反推的检验阈
  值）；若删除一个单位只造成 20% 变化，报告应显示数值而不是把它藏进
  「稳定」标签。
- 行级单元没有 token 池化 NLL 矩阵，故不生成影响块；历史 v6–v11 记录
  没有该字段，不重跑、不回填，也不把旧记录标成已做单单元敏感性。
- `unit_ids` 缺失/重复时分析层回退到索引标签并明确记录，而不是猜是哪
  篇文章；这会降低可读性，但不改变数值。

### 57.5 入口与验证

```bash
python test_review_fixes.py             # L173 原语；L174 v12 接入/渲染；L175 审计
python audit_gates.py unit_influence    # 零 GPU：核对已落盘影响块
python audit_gates.py report            # 汇总里新增 unit_influence
```

L173 覆盖：精确 LOO 与逐折 `_paired_nll_doc_ci` 逐位一致、jackknife SE
手算一致、单单位主导/方向翻转、均匀效应零变化、全零效应返回 undefined、
<3 单位拒绝、LOO 溢出折标为不可用、形状/token/id/选项 fail-closed。
L174 覆盖：v12 分析在文档与文章单元上写入正确 id、影响块点估计与
`pooled_delta_ppl` 一致、报告表含「单单元主导」与「不改变冻结判定」
边界、均匀效应不被误标、无影响块时不伪造表格。L175 覆盖：fragile 块
→ `warnings`、一致块 → `ok`、篡改点估计 → `fail`、pooled CI 存在但
无影响块 → `warnings`、缺产物 → `not_verifiable_locally`。

## 58. 补种子不是免费重跑：把 look history 变成可复算的序列检验（本轮对比实验方法硬化）

### 58.1 问题：同一个面板被分析了不止一次

§19 的族校正解决的是「一个面板内同时比较 m 条声明」的多重性；§42/§57
解决的是「结论是否由某一个训练种子或评测单位主导」。它们都默认**最终
种子集合只被检验了一次**。但本仓库的补种子流程不是这样工作的：

- `run_gapfill.py full` 先以 seeds 0-2 聚合、写 `summary.json` /
  `methodology_audit.json`；
- 看到结果后，`topup`（`ABL_SEED_TARGET` / `CORE_SEED_TARGET`）再把同一
  面板补到 n=4、6、12，每补一次就再聚合、再看一次。

这正是 optional stopping / 多次 look：如果每次都在「这次还不显著」之后
继续加种子，直到某一次 p<α，那么最终 n 下的朴素 p 不再是实验真正花费的
错误率。§49.3 已经把「探索性面板的 look history 不是机器可读的」列为
剩余边界；本轮把它变成落盘、可复算、会真正收紧判定的证据，而不是再加
一句「补种子要小心」。

### 58.2 定义：一次 look 是一个新的种子集合

`exp_lib.record_panel_look(outdir, recs)` 在 `_dump_aggregate` 每次聚合时
运行，对每个**已测声明对照**记录：

- 当前参与配对的种子集合与逐种子配对差 `d_i = with − without`（主终点、
  与 `_paired_seed_deltas` 同一受控配对门）；
- 该 look 的精确符号翻转 p、均值、run 配置键与删除的配对数。

写进 `<outdir>/look_history.json`。同一份数据重复聚合**不会**新增 look；
新种子、新臂或某个种子的配对差被重跑改变才会新增。文件有上限
（`LOOK_HISTORY_MAX_LOOKS=500`），截断时显式记录
`truncated_oldest_looks`，不静默丢掉证据。

这是**分析事件**记录，不进入 `run_cfg`、resume 键、配对身份、`src_fp`
或冻结清单；它不改变任何训练/评测数值。

### 58.3 方法：最终方向上的 max-over-looks 精确符号翻转

`exp_lib.sequential_signflip_p(looks)` 的输入是同一对照按时间排序的
look 序列，最后一个 look 是当前最终种子集合。设最终总效应
`T_final = |Σ_i d_i|`，方向 `s = sign(Σ_i d_i)`。零假设下每个种子的符号
独立等概率翻转，同一条符号向量投影到每个 look 的种子上：

```
T_k(ε) = s · Σ_{i ∈ look_k} ε_i d_i
p_one  = P( max_k T_k(ε) ≥ T_final )
p_seq  = min(1, 2 · p_one)
```

- 这是**联合**零分布，不是把各 look 的 p 当独立检验再做一次校正；look
  之间的相关性（共享种子）被精确复现。
- 取 max 且用**最终方向**，而不是「所有 look 里最小的双侧 p」：一个早期
  look 即使更极端、方向相反，也不能替最终的相反方向「买」显著。最终
  look 一定在 max 内，所以 `p_seq ≥ p_final`；这保证修正只会变保守。
- 双侧用 `2·p_one`：对任意符号路径，
  `P(max_k |S_k| ≥ T) ≤ 2·P(max_k S_k ≥ T)`，是有效的保守上界。
- 种子并集 ≤ `LOOK_SEQUENTIAL_ENUM_MAX_N=16` 时按上式精确枚举；更大时
  退回 `min(1, K · p_final)` 的 Bonferroni-over-looks（有效但保守），并
  在 `method` 里标明。K=1 时直接等于普通精确 p（没有额外损失）。
- 同一种子在两个 look 的 `d_i` 不一致（记录被重跑/编辑）时，旧 look 视为
  **stale**，在 `methodology_audit` 中剔除并点名；公开原语则退回
  Bonferroni 并把冲突种子写进 `inconsistent_seeds`。绝不把两个估计量的
  配对差混进同一个联合零分布。

### 58.4 接入 `methodology_audit`：verified 用序列 p

- `methodology_audit(..., look_history=None)`：默认从
  `<outdir>/look_history.json` 读取；也可显式传入（审计导出的副本）。
  每个对照条目新增 `look_analysis`（逐 look 的 n/seeds/p、`p_sequential`、
  `method`、`n_stale_looks`）与 `p_look_adjusted`。
- 族校正改为两列：`p_holm_global_naive` 用最终 n 的朴素 p（保留，供
  对照/诊断），`p_holm_global` / `p_holm_family` 用 `p_look_adjusted`。
  `significant_global` 因此要求**序列 p** 也过 Holm；`verified` /
  `system_level_only` / SESOI 的统计前提全部沿用这条更保守的判定。
- TOST 等价同样受多次 look 影响：对 K>1 的对照，`_apply_sesoi_holm` 把
  原始 TOST p 先做 `min(1, K·p)` 再进 Holm，写入
  `sesoi.p_tost_look_adjusted` / `n_looks`。`equivalent` 不能再靠某一次
  恰好很紧的补种子 look 取得。
- 报告顶层新增 `look_history` 块（`panel_looks`、被修正的对照数、被
  降级的朴素显著项、`invalid`）并把降级写进 warnings；控制台打印
  `analysis looks: K`。`REPORT_v12` 若读取该审计块会得到同一口径。
- `audit_gates.audit_look_history`（`python audit_gates.py looks`）独立
  复核：look 编号严格递增、seed/delta 长度与唯一性、逐 look p 可重算、
  `methodology_audit.json` 的 `p_sequential` 与重算一致、补种子是否嵌套、
  以及「朴素显著但序列不显著」的降级清单。结构不一致判 `fail`，缺少
  audit 块判 `warnings`，没有任何 look 文件判 `not_verifiable_locally`。

### 58.5 边界

- **不改**冻结确认清单：`confirmation_frozen*.json`、P5T 的 p/α/Holm/
  决策规则、种子白名单、终点与 `CODE_SEMANTICS` 全部不动。该层只约束
  探索性面板的**重新阅读与措辞**；v12 确认性判定仍按冻结规则。
- 只对**有 look history 的新聚合**生效。历史 v6-v11 面板没有该文件，
  `methodology_audit` 保持原行为（并在缺文件时不做任何猜测），不重跑、
  不回填，也不把旧结论标成已做序列修正。
- 「聚合过」不必然等于「人看过」：若中间 look 从未被查看，序列修正是
  保守的而非错误的；若确实按 p 决定继续补种子，它就是必须的修正。该
  文件把「有没有多次分析」变成可复核事实，但无法证明分析者的意图。
- 序列 p 的条件是「look 序列是同一对照、同一主终点、同一评测切片」；
  stale look 被剔除，但**没有**自动检测「换了一个验证切片后仍复用同一
  outdir」——这仍由 `pair_reason` 的 `val_fp`/`eval_fp` 门和 §19 负责。
- 并集 >16 个种子用 Bonferroni 回退：仍有效，但比精确 max 保守；报告
  必须读 `method`，不能把两种 p 当成同一精度。
- 这是族级「最终方向是否有过证据」的 p，不替代效应量、SESOI、种子
  稳健性（§42）或单单元影响（§57）；它只回答「多次补种子后，最终 n 的
  p 还能不能按原样引用」。

### 58.6 入口与验证

```bash
python test_review_fixes.py          # L176 原语；L177 落盘/去重/stale/审计；L178 降级门
python audit_gates.py looks          # 零 GPU：逐面板 look history 与序列 p
python audit_gates.py report         # 汇总里新增 looks
```

L176 覆盖：单 look 等于普通精确 p、嵌套 look 的 `p_seq ≥ p_final`、
具体降级向量、早期反向 look 不能替最终方向取得显著、stale 种子回退
Bonferroni、>16 种子回退、确定性与 malformed 输入 fail-closed。L177
覆盖：6→10 补种子产生 2 个 look、重复聚合不新增、stale delta 被剔除
并 warn、篡改逐 look p 判 fail、缺 audit 块判 warnings、无产物判
`not_verifiable_locally`。L178 覆盖：朴素显著/序列不显著的降级被审计
点名，audit 的序列 p 与重算不一致判 fail。

## 59. 2×2 因子交互：相加性不是默认假设

### 59.1 为什么需要

留一（leave-one-out）族回答的是「单独拿掉这个模块会怎样」，不能回答
「两个模块的收益是否重叠」。完全可能出现：两个模块各自显著有用，但同时
打开时第二个几乎不再增加收益。把两个单因素结论直接相加，会高估组合收益。
交互检验把「第二个模块在第一个已开启时是否仍然有用」变成预注册问题。

### 59.2 设计与估计量

只使用 `RUN_ABL` 已经排程的臂，不新增训练预算：

| 交互 | control/control | active/control | control/active | active/active |
|---|---|---|---|---|
| dynamic_blocking × learned_selection | `csa_fix_randidx` | `csa_dyn_randidx` | `csa_fixed` | `csa_dynamic` |
| dynamic_blocking × content_pooling | `csa_fix_zerocont` | `csa_dyn_zerocont` | `csa_fixed` | `csa_dynamic` |
| dynamic_blocking × attention_sink | `csa_fix_nosink` | `csa_dyn_nosink` | `csa_fixed` | `csa_dynamic` |

每个种子上的估计量是 log PPL 尺度的差之差：

```
I_s = (logPPL_11 - logPPL_10) - (logPPL_01 - logPPL_00)
```

正 `I` = 次可加/冗余（第二因子在第一个开启后帮助变小）；负 `I` = 协同。
log 尺度避免「基线 PPL 更高所以绝对效应更大」的机械放大。三个交互都
预注册 `predicted_sign=+1`：两个机制被假设为在争夺同一部分块相关性/上下文
外质量信息，而不是纯粹互补。

### 59.3 为什么只有这三组

`hybrid × dynamic` 被排除：固定层加 HCA 只改 `kind/block_size`，动态层加
HCA 同时改 `chunking/dynamic/target_block_tokens/comp_lambda_mult`；两条轴
不是同一组 cfg 改动，交互会把「HCA」与「动态配方」混在一起。只有
`dynamic_blocking` 对另外三个因子在 fixed/dynamic 两水平的 cfg diff 完全
相同，2×2 才可识别。

### 59.4 设计点门与记录级门

`factorial_design_audit()`（定义点、零 GPU）检查：

1. 四个 cell 完整、映射到四个不同的臂，且全部在 `RUN_ABL['variants']`；
2. 每个因子的 control→active AttnCfg diff 在另一因子的 control/active 两个
   水平上**完全相同**；否则该轴不是并行操纵；
3. diff 不含 `kind`，且每个键属于 `ABLATION_INTERVENTION_KEYS` 的声明族；
4. 每个 factorial 都有完整、metric-aligned 的 `INTERACTION_HYPOTHESIS_DESIGN`
   条目；
5. 复用 `ablation_completeness_audit(arms_by_combo=...)` 的全因子模式，
   缺 cell / 重用臂会被精确点名。

`factorial_interaction_audit()`（记录级）对每个种子要求四臂共享 protocol
tag、通过 `pair_reason` 与 `_hyperparam_reason`、使用同一 primary estimand、
PPL 有限，然后才计算 `I_s`；被拒的种子逐条列出，不静默取子集平均。
参数容量不通过时交互仍可报告，但标 `mechanism_claim_allowed=False`。

### 59.5 统计与族隔离

- 对 `I_s` 使用与主审计相同的精确符号翻转检验和种子级 CI；报告
  `ratio_of_ratios=exp(mean(I))`，而不是把「两次显著性」相互比较。
- 交互有**独立**的 Holm 族：不进入留一族，不改变 `family.effective_m`，
  也不改变冻结确认规则。缺失/不可控的 factorial 保留 dummy p=1 槽位，
  所以缺臂不能缩小校正族来「买」显著。
- 非显著交互只说明「当前种子数下没检出偏离相加」，不是「效应严格可加」；
  与主审计一样受精确检验分辨率下限约束。

### 59.6 入口与验证

```bash
python audit_gates.py factorial        # 定义点：并行操纵/完整性/假设
python audit_gates.py single_factor    # 同时包含 factorial_design 视图
python audit_gates.py methodology      # 面板 summary 中的交互族
python audit_gates.py report           # 汇总新增 factorial / interactions
python test_review_fixes.py            # L60a/L60b
```

L60a 覆盖： shipped 三组设计通过、缺 cell / 重用臂 / 非并行轴 / 未排程臂
fail-closed。L60b 覆盖：常值正/负交互的精确 p、独立族大小、缺 cell 保留
槽位、run_cfg 不一致丢弃全部种子、容量混淆时禁止机制声明。

## 60. 有序剂量设计：dose-response 不是「挑两个端点比一下」

### 60.1 为什么需要

top-k、块大小、retrieval ratio、distractor 比例这类机制面板天然是**有序水平**。
只挑两个端点做一次配对检验，会同时丢掉两件事：

1. **顺序信息**：如果机制真的随剂量单调变化，三点以上的趋势比端点差更有
   说服力；反过来，端点显著但中间不动，也不支持「剂量响应」的措辞。
2. **选择自由度**：端点、方向和水平数量如果都是看完结果再定，那么
   「显著」可以被事后搜索出来。方向与形状必须先于读数写入设计。

两个点永远可以连成一条线，所以两点对比只能叫 endpoint contrast，不能叫
dose-response。

### 60.2 设计点契约

`exp_lib.dose_response_design_audit(levels, direction=..., dose_key=...)` 在
训练/读数之前核验：

- 至少 3 个有限、互异、按剂量排序的水平；两点直接 fail（并提示改用端点
  对比）；
- 每个水平声明 `label`、`dose`、`arm`；所有 arm 互异；
- 基线 arm 到每个剂量 arm 的 AttnCfg diff **恰好等于** `{dose_key}`：多一个
  键 = 混入第二因素，少一个键 = 干预消失；
- `dose_key` 在每个水平上是有限标量，且其值随声明剂量单调（允许反向编码，
  不允许乱序）；重复值说明两个「剂量」实际是同一操纵；
- 每个水平声明**同一批种子**（默认 `min_seeds=3`），因为趋势是种子内配对；
- 若声明测量协议（eval 长度、checkpoint、recipe 等身份），每个水平必须
  声明同一个协议；否则趋势会把不同测量混在一起。

这是**设计声明门**，不读结果，也不能证明方向确实是在第一次看数之前写的；
它只保证一旦写下 declaration，机器能发现「不是单轴 / 不是同一批种子 /
不是同一测量协议 / 不是有序剂量」的声明。

### 60.3 估计量与检验

记录级 `dose_response_trend(levels, direction=..., shape='linear'|'monotone')`
对每个在**所有水平**都有有限读数的种子计算 OLS 斜率：

```
b_s = Σ_i (x_i - x̄)(y_{s,i} - ȳ_s) / Σ_i (x_i - x̄)^2
```

主检验是对 `{b_s}` 的**一次双侧精确符号翻转**，与主审计使用同一
`paired_effect_stats` 口径（n、mean、std、CI、p floor、underpowered、
within-seed-noise 都保留）。不是把每个相邻水平都做一次检验再取最小 p：
那样既没有校正，也会把「中间某个相邻差最大」误当成整体剂量响应。

- `direction` 必须是预注册的 `increasing`/`decreasing`；显著但方向相反时
  判 `direction_conflict`，不能改写成「反向剂量响应」来救回显著。
- `shape='linear'` 只要求方向；`shape='monotone'` 额外要求每个相邻水平的
  **平均**步进都在声明方向上。形状不满足时判 `shape_conflict`，但仍可
  报告线性趋势，且 `claim_allowed=False`。
- 缺失/非有限读数按完整案例剔除并逐个点名；不补零、不静默取子集平均。
- 协议身份不一致（例如不同 eval 长度）判 `protocol_conflict`，即使 p 很小
  也不允许因果/机制措辞。
- 多个有序族应把各自的 p 作为**一个声明的族**交给
  `global_multiplicity_audit`/Holm；本函数只产生一条 p。

### 60.4 边界

- 有序剂量不是随机分配：`dose` 与训练时长、显存、实现路径等仍可能共变；
  单轴 cfg diff 只排除**声明层面**的第二因素，不证明剂量本身外生。
- 显著趋势只说明「操纵与结果同向」，不证明机制；反向/非单调结果同样
  有信息，必须按预注册方向报告，而不是事后改假设。
- 该层不改变 `confirmation_frozen*.json`、P5T 的 p/α/Holm/终点、种子白名单
  或 `CODE_SEMANTICS`；它只约束新的有序面板/未来冻结清单中的可选
  `dose_response_design` 声明。
- 清单没有 `dose_response_design` 时，`audit_gates.py dose_response` 返回
  `not_applicable`，不是 pass；没有声明就没有可审计的剂量假设。

### 60.5 入口与验证

```bash
python audit_gates.py dose_response   # 零 GPU：读清单可选声明；无声明 not_applicable
python test_review_fixes.py           # L181 设计门；L182 精确趋势
```

L181 覆盖：三点单轴通过、额外 cfg 键、重复剂量、种子集不一致、缺种子声明、
两点、协议变化、剂量轴乱序、`direction='auto'`、非法 shape 全部
fail-closed。L182 覆盖：6 种子精确 p=2/64、方向冲突、linear vs monotone
形状门、缺失/非有限读数点名剔除、两点拒绝、协议冲突、非法
direction/shape 的调用错误。

## 61. 统计机器校准：先证明检验本身不超发，再谈显著

### 61.1 为什么需要

本仓库的结论最终落在一串统计程序上：精确符号翻转、n=21–40 的
meet-in-the-middle（MITM）、n>40 的 Lugannani–Rice 鞍点尾、max-over-looks
的序列 p、配对 TOST 与 Holm。单元测试能证明这些函数在**给定输入**上返回
预期形状，但不能证明它们的**频率学错误率**在重复抽样下符合名义水平。
一个实现错误（枚举漏项、鞍点公式符号、序列 max 的方向、TOST 边界用了单侧
而不是双侧）完全可能让所有 fixture 测试通过，却让零假设下的 p 系统性偏小。
一旦如此，后面所有「p<0.05」「Holm 后仍显著」都失去意义。

所以统计机器本身必须像训练/评测代码一样被校准：先用**已知答案**的合成
数据检验它，再允许它去检验真实面板。校准不是又一层结果审计，而是分析
工具的计量学。

### 61.2 对象与策略

`exp_lib.STATISTICAL_CALIBRATION_POLICY` 固定以下先验参数（改动会改变
`policy_fp`，必须在看结果前写入；指纹绑定实际生效的 seed/n_sim/α/阈值，
带覆盖参数运行不会复用默认策略的指纹）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `simulation_seed` | `20261023` | 模拟种子的根；派生子流 |
| `n_sim` | 2000 | 每个路径的重复次数 |
| `alpha` | 0.05 | 与主审计相同的名义水平 |
| `power_floor` | 0.60 | 正控制要求的最低灵敏度 |
| `control_level` | 0.001 | 单侧校准门的失败阈值 |
| `warning_level` | 0.01 | 仅告警、不判失败的阈值 |
| `saddlepoint_max_abs_error` | 0.03 | 鞍点 p 与精确 p 的最大允许偏差 |
| `saddlepoint_mean_abs_error` | 0.005 | 平均允许偏差 |

所有随机性来自 `numpy.random.SeedSequence(seed).spawn(k)` 派生的**局部**
`default_rng` 流；校准不读、不写全局 NumPy/Torch RNG，因此不会扰动训练或
其他面板的可复现性。输出只依赖策略与输入（模拟出的 p 值），不依赖时钟，
同一版本下逐字节可复现。

### 61.3 负控制与正控制

只做零假设校准是不够的：一个永远不拒绝的检验会完美通过所有 type-I 检查。
每个路径因此必须同时通过两个门。

**负控制（type-I）**：数据由 `N(0,1)` 生成，真实效应为零。若检验校准，
拒绝次数 `K ~ Binomial(n_sim, alpha)`。单侧上尾

```
p_anti = P(K >= k_obs | alpha)
```

在 `p_anti < control_level` 时判 fail（观测超发率显著高于名义水平），
在 `p_anti < warning_level` 时只告警；低于 alpha 的保守行为通过。阈值
在模拟前固定，不能看完 `k_obs` 再挑一个刚好不判失败的 `control_level`。

**正控制（灵敏度）**：数据由已知均值移动的 `N(delta,1)` 生成。若检验有
至少 `power_floor` 的功效，拒绝次数 `K ~ Binomial(n_sim, power_floor)`。
单侧下尾

```
p_below = P(K <= k_obs | power_floor)
```

在 `p_below < control_level` 时判 fail（灵敏度显著低于声明下限）；观测率
低于下限但仍在蒙特卡洛噪声内时告警。正控制的 delta 是设计选择，不是从
实验里估计出来的效应量。

两个门都是**单侧筛查**：通过只表示在 `n_sim` 次重复下没有检出足以判失败的
系统性偏差，不能读成「错误率精确等于 α 已被证明」；1 个百分点量级的偏差
需要更大的 `n_sim` 或专门的等价性检验。这里选择固定预算的筛查，是为了让
校准能在零 GPU 审计里常规运行，而不是把「未检出」包装成「已证明」。

### 61.4 被校准的检验路径与鞍点精度

| 路径 | 种子数 | 负控制 | 正控制 | 额外检查 |
|---|---|---|---|---|
| 精确符号翻转枚举 | 12 | `N(0,1)` | `N(1.0,1)` | 方法确为 enumeration |
| MITM 精确路径 | 21 | `N(0,1)` | `N(0.8,1)` | 方法确为 MITM |
| 鞍点尾（n>40 分派） | 48 | `N(0,1)` | `N(0.5,1)` | 方法确为 saddlepoint |
| 鞍点近似精度 | 21 | — | — | 与精确 MITM p 的逐点最大/平均绝对误差 |
| max-over-looks 序列 p | 12，looks 6/9/12 | `N(0,1)` | `N(0.9,1)` | 同一批种子嵌套 look |
| 配对 TOST | 12 | `mean=1.0`（恰在 margin=1.0） | `mean=0` | `conf=0.90` |
| Holm 族 | m=6 | `Uniform(0,1)` p 值 | 一个 `1e-6` + 五个 null | 全局 FWER / 至少一次拒绝 |

鞍点精度是**近似正确性**门，不是显著性门：它逐点比较
`_signflip_saddlepoint_p` 与 `exact_signflip_p`（n=21 时后者是 MITM 精确值），
限制最大绝对误差与平均绝对误差；仅当所有 draw 都通过才记为 ok。

默认策略在 `seed=20261023, n_sim=2000` 下的一个实际运行结果（用于回归
定位，不作为科学证据）：

```
exact enumeration   null 0.061   positive 0.8825
MITM                null 0.0495  positive 0.942
saddlepoint n=48    null 0.038   positive 0.9165
sequential          null 0.042   positive 0.7845
TOST boundary       null 0.0485  positive 0.888
Holm FWER           null 0.051   positive 1.0
saddlepoint vs exact max |Δp| 0.0229, mean |Δp| 0.00084
```

这些数字随固定种子确定；它们说明当前实现没有可检出的系统性超发，也
保留了可用功效。它们**不**说明任何真实面板的效应存在。

### 61.5 边界

- 校准只验证统计机器在**理想独立配对、正态、同方差**数据上的频率学行为。
  它不验证真实面板的 exchangeability、随机化、配对完整性或测量同一性；
  那些仍由 `pair_fp`、`src_fp`、run-order、leakage 与 protocol 门负责。
- 通过校准不能把有混淆的设计变成因果证据，也不能修复选择偏差、泄漏或
  optional stopping；它只排除「分析函数本身算错」这一小类问题。
- 一个显著结果仍可能来自真实但非因果的共变；校准不是机制证据。
- 失败表示该统计路径的声明不可信，不表示旧结果一定错；但基于该路径的
  新结论必须先修复实现，不能靠换种子/换阈值绕过。
- 策略参数、种子与模拟数据都是分析工具的一部分；它们与
  `CODE_SEMANTICS`、冻结清单、P5T 的 α/Holm 规则和实验种子无关。校准不
  读取结果树，不新增训练，不写 `results_*`。

### 61.6 入口与验证

```bash
python audit_gates.py calibration     # 零 GPU、纯合成数据；fail 时退出码 1
python audit_gates.py stats_calibration
python test_review_fixes.py           # L183 策略/门/确定性；L184 审计接线
```

`audit_gates.py calibration` 也作为 `calibration` 键进入
`audit_gates.py report` 的 `audit_report.json`。L183 覆盖：二项尾已知值、
保守检验通过、200/200 拒绝判 fail、零功效判 fail、非法计数/α/阈值
fail-closed、两次运行逐字节一致、全局 RNG 状态不变、全部六条路径存在且
控制门通过。L184 覆盖：审计入口的种子/规模/状态、报告字典接线、CLI 文档
与过小模拟的拒绝。

## 62. 区间也要校准：覆盖率不等于「检验通过」，宽度不能靠放宽（本轮结果分析硬化）

### 62.1 问题：检验校准了，区间还没校准

§61 只校准了 p 值路径。报告同时大量引用区间估计：逐种子配对 t 区间
（`paired_effect_stats.ci95_*`）、配对 bootstrap 区间、按评测单位
（document/unit）重采样的区间，以及把训练种子与评测文档两个方差源合并的
`_combined_seed_doc_ci`。p 值校准不能推出区间校准：

- 一个 p 检验可以在零假设下保持 5% 超发，而同一份数据上的 95% 区间却只有
  90% 覆盖率——两者是不同的估计量；
- 反过来，一个无限宽的区间 `[-inf, +inf]` 覆盖率恒为 1，能通过任何只检查
  覆盖率的门，却不传递任何不确定性信息；
- 小样本百分位 bootstrap 的欠覆盖是已知的有限样本现象，不是实现崩溃：
  在 n=12、名义 95% 下，本仓库旧的 `paired_bootstrap_ci` 实测覆盖率约
  0.907（固定种子模拟），而配对 t 区间约 0.952。这个差距足以让报告里的
  「95% bootstrap 区间」名不副实。

因此需要把区间当成与检验并列的统计机器：既检查覆盖率，也检查区间不是
空洞的，并把不合格的构造替换掉，而不是把名义水平改小。

### 62.2 方法：两个门 + studentized bootstrap

`STATISTICAL_CALIBRATION_POLICY` 新增区间校准参数（名义覆盖率 0.95、
每个路径的重复次数沿用 `n_sim`、bootstrap 抽取数、已知数据生成过程与
宽度上限）。新增两条门：

- `calibration_interval_coverage_gate(K, n_sim, nominal=0.95, ...)`：
  正确校准下 `K ~ Binomial(n_sim, nominal)`。用单侧下尾
  `P(X <= K | nominal)` 判 fail/warning；**高于**名义覆盖率的保守行为通过。
  阈值在模拟前固定。
- `calibration_interval_width_gate(mean_width, max_width)`：
  平均宽度必须是有限正数且不超过声明的上限。上限是 DGP 单位下的设计值，
  不是看完结果再写的描述；它专门拒绝「覆盖率为 1 但宽度爆炸」的退化解。

每个模拟重复的 bootstrap 种子由同一个局部 RNG 流派生，而不是所有重复共用
一个固定种子；否则覆盖率会条件在单一一组重采样索引上，蒙特卡洛独立性不足。

配对与单位 bootstrap 从百分位法改为 **bootstrap-t（studentized）**：
每个重采样用自身的标准误标准化 `(mean* - mean)/se*`，再取 t 分布的分位数
反变换回原尺度。它在 n=12 附近的覆盖率显著优于百分位法，且不改变主检验
（精确符号翻转 p）或冻结判定。若重采样的标准差为 0 或可用 t 统计量太少
（常量输入、极少数不同取值），函数显式回退到百分位区间，并在返回的
`method` 里写明 fallback，不静默改变估计量。

四条路径：

| 路径 | 构造 | 目标 | 宽度上限 |
|---|---|---|---|
| `interval_seed_t` | `paired_effect_stats` 的逐种子配对 t 区间 | 总体均值 0 | 2.0 |
| `interval_paired_bootstrap` | 配对 bootstrap-t（`paired_bootstrap_ci`） | 总体均值 0 | 2.0 |
| `interval_unit_bootstrap` | 单位/bootstrap-t（`bootstrap_mean_ci`，24 个单位） | 总体均值 0 | 2.0 |
| `interval_combined_seed_doc` | `_combined_seed_doc_ci`（6 seeds × 24 docs，已知 seed/doc/token 方差） | `exp(mu)`，在 log-PPL 尺度检查宽度 | 1.2（log 宽度） |

### 62.3 默认策略下的实际校准

`seed=20261023, n_sim=2000, n_boot=400` 的结果（仅用于回归定位，不是
科学证据）：

```
interval_seed_t             coverage 0.9525  mean_width 1.2369
interval_paired_bootstrap   coverage 0.9440  mean_width 1.2848
interval_unit_bootstrap     coverage 0.9520  mean_width 0.8354
interval_combined_seed_doc  coverage 0.9775  mean_width 0.6568 (log)
```

四条路径的覆盖率都在名义水平附近（最低 0.944，差异在蒙特卡洛噪声内），
宽度都低于声明上限。combined 区间偏保守（0.9775 > 0.95），这是合并两个
方差源时用
`S-1` 自由度 t 分位数的已知保守性；它通过覆盖率门，但报告仍应把它读作
保守区间，而不是「精确 95%」。

### 62.4 边界

- 校准数据仍是理想正态/独立单位；它不验证真实面板的相关结构、重尾、
  文档内相关或 seed×document 交互的设定是否正确。真实面板的覆盖率仍依赖
  这些假设。
- 这里只校准四条被报告直接引用的路径。`unpaired_bootstrap_ci`、逐行
  `paired_row_bootstrap_delta`、`_doc_bootstrap_ppl_ci` 等百分位区间仍是
  近似读数，未进入本轮覆盖率门；引用它们时必须保留「近似/仅该抽样源」
  的限定。
- 通过覆盖率门不等于区间在真实面板上一定准确；欠覆盖只在固定 `n_sim`
  下可检出的量级上被判 fail，不能证明误差率恰好等于 5%。
- 宽度上限是防止退化解的工程门，不是「最优精度」声明；加宽区间以换取
  覆盖率仍可能是合理的保守选择。
- 这些改动只影响报告中的区间读数，不改变精确符号翻转 p、Holm 决策规则、
  冻结清单、`CODE_SEMANTICS` 或 P5T 的确认判定；也不新增训练或读取
  test split。

### 62.5 入口与验证

```bash
python audit_gates.py calibration     # 零 GPU；同时打印检验与区间校准
python test_review_fixes.py           # L183/L184 扩展 + L186
```

L186 覆盖：覆盖率门的已知值（0.95 通过、0.70 fail、保守通过）、非法计数/
名义水平/阈值 fail-closed、宽度门拒绝 0/负/NaN/inf/超上限/布尔值、四条
区间路径存在且宽度非空洞、bootstrap-t 的确定性与常量输入显式回退。
完整 CPU 套件 534/534 通过（0 skipped）。

## 63. 同一 panel 的多种分析口径要显式对账（本轮结果分析硬化）

### 63.1 问题：口径选择本身是不可见的结果分析自由度

§61/§62 校准了 p 值与区间，但报告仍可在同一份冻结 panel 上选择多种聚合：

- 逐种子 mean / median / 20% trimmed mean；
- token 池化 ΔPPL（与冻结主终点同口径）；
- 逐评测单元（article/document/row）等权 mean / median；
- 逐单元 log-ratio 与 win-balance。

这些口径回答不同问题，本来就不应给出同一个数。但在只报其中一个时，读者
看不到方向是否一致、区间是否同样排除零；更糟的是，分析者可以在一组口径
中挑选最有利的一个，把它写成「稳健」。逐种子均值显著、而逐文档中位数
方向相反的面板是真实可能出现的：一个超长/超大文档可以主导 token 池化
Δ，却改变不了等权中位数的符号。此前 `doc_sampling` 已分别记录这些读数，
但没有一个显式的「口径间对账」块，也没有审计门检查它是否被悄悄省略。

### 63.2 方法：只对账，不新增检验

`exp_lib.analysis_choice_concordance(primary, alternatives)` 接收主口径与
一组声明式口径（`spec`、`estimand`、`unit`、`scale`、`point`、`ci95`），
返回：

- `direction_stable`：主口径非零，且所有可用且非零的口径点估计与主口径同号；
- `precision_concordant`：所有带两侧区间的可用口径对「是否排除零」一致；
- `status`：`stable`、`precision_sensitive`（同向但零排除不一致）、
  `direction_unstable`（方向矛盾）、`zero_effect`、`insufficient_specs`；
- `effect_range_ppl`：仅在 PPL-Δ 口径内给出 min/max/span，不把 log-ratio
  或 win-balance 混进同一尺度；
- 逐口径的 `sign`、`excludes_zero`、`available/reason`，以及反向/跨零的
  口径名单。

`v12_supp._doc_level_analysis` 现在把 8 条口径写入
`doc_sampling.analysis_choice_concordance`：逐种子 mean（主口径）、
median、trimmed mean，token 池化，逐单元 mean、median、log-ratio、
win-balance（`(n_positive - n_negative) / n_units`，负号表示 a 更常更低；全平局为 0）。
种子数不足 5 时 trimmed mean 显式不可用，未计算的区间显式留空，不伪造。

报告新增「分析口径一致性」表，列出可用/总口径、主方向、同向/反向、
区间排除零计数与 PPL 范围；方向翻转或精度敏感时附具体口径名单和引用提示。
若被确认的比较在该块中 fragile，确认段落还会单独提示不能只报单一聚合。
该表**不新增 p 值、不进入 Holm 族、不做多重校正、不改变冻结判定**。把
更多口径塞进显著性族只会扩大 forking paths，而不是控制它。

`audit_gates.py analysis_choices`（别名 `specification_curve` /
`analysis_choice`）从 `analysis_v12/test_stats.json` 重放存储的口径行：
若块内 `status`/计数/方向/零排除无法由 `analysis_choice_concordance`
复现，或主口径点估计与冻结逐种子均值不一致，判 fail；方向翻转或精度敏感
只判 warning，并要求报告携带 caveat；旧产物缺少该块时判
`not_verifiable_locally`，不冒充通过。

### 63.3 边界

- 口径一致不是统计独立复现，也不是「换一种聚合仍显著」的证明。同向只说明
  这些描述量没有互相矛盾；`precision_sensitive` 在单位/估计量不同时是
  正常现象，不能读成方向失败。
- 该块不覆盖所有可能的口径：截尾比例、加权方案、文档过滤、长度子集、
  模型选择等仍是未枚举的 forking paths；它只把当前报告实际使用的口径
  显式化，不能证明「没有其他口径会翻转」。
- 逐种子/逐单元/token 池化的区间条件在不同抽样源上；本块只比较
  「是否排除零」，不比较区间宽度，也不把两种不确定性合并。
- 全部读数仍是 post-hoc 描述；不改变精确符号翻转 p、Holm 决策、冻结清单、
  `CODE_SEMANTICS` 或 P5T 的确认判定，不新增训练，也不读取 test split。

### 63.4 入口与验证

```bash
python audit_gates.py analysis_choices   # 零 GPU；读 analysis_v12/test_stats.json
python v12_supp.py report                # 报告新增「分析口径一致性」表
python test_review_fixes.py              # L188
```

L188 覆盖：稳定/精度敏感/方向翻转/零效应/口径不足五种状态、零与不可用
口径的显式记录、CI-less 口径只参与方向、重复名/非法 CI/布尔点/非有限主
口径/非法 `zero_tol` fail-closed、输入不被修改；审计入口的稳定→ok、
方向翻转→warnings、状态篡改/主均值不符/重复名→fail、旧产物→
`not_verifiable_locally`、报告与 CLI 接线；以及「token 池化均值为正但逐
单元中位数为负」的合成 panel 被标为 `direction_unstable` 且报告保留
caveat。完整 CPU 套件 536/536 通过（0 skipped）。

## 64. 证据归档必须自带测量源码并用内容摘要绑定

### 64.1 问题：Git HEAD 不能证明归档里的代码

§50/§51 已让冻结锁和运行清单绑定测量源码树，但 `archive_release.py` 的
正式归档仍只记录 `git_head`。发布物本身不含源码；当工作树是 dirty、
分支已移动、提交被变基/删除，或报告在最后一次源码编辑之后才生成时，
一个「可审计」归档无法从自身字节重建当时的测量代码。只写 `git_head`
还有一个更隐蔽的失败模式：归档里的结果和该提交的源码都真实存在，但二者
从未在同一时刻被绑定，验证者只能相信打包者的陈述。正式 release 因此需要
把测量源码入包，并让 `verify` 从包内字节独立重算身份。

### 64.2 方法：一个源码清单、一个内容摘要、一次发布前复核

`archive_release.py` 复用
`exp_lib.source_tree_manifest()`（与冻结锁/运行清单同一扫描口径，排除
`test_*.py`、`conftest.py`、`results_*`、`analysis_*` 等），在 `plan`
和 `make` 时把每个源码文件按仓库相对路径加入归档。manifest 新增
`identity` 块，记录：

- `schema`、`code_semantics`、`code_stamp`；
- `source_tree`（文件 → `sha256`/`bytes`、`n_files`、总字节数和聚合
  `sha256`）与顶层 `source_tree_sha256`；
- `dependency_contract`（锁文件契约的当前读数）和
  `execution_fingerprint`；
- 精简的 `environment` 字段。

聚合摘要使用与源码树清单一致的算法：
`sha256(sorted relpath NUL bytes NUL sha256 LF)`。`make` 在写 manifest
前用 `_release_identity_problems()` 对**实际写入 tar 的字节**复核：每个
源码树条目必须在 per-file map 中且摘要/大小一致，聚合摘要必须能重算；
不一致（例如打包期间源码被编辑）就删除临时 tar 并拒绝发布。`verify`
对归档内的 manifest 执行同一检查，因此校验不依赖 Git、当前工作树或
`exp_lib`。没有 `identity` 的旧归档仍可校验文件哈希，但会打印 NOTE：
其源码绑定不可从归档本身验证。

### 64.3 边界

- 这不是环境重建：`verify` 校验源码字节和摘要，不重新安装依赖、不重算
  `dependency_contract`，也不验证 `execution_fingerprint` 所记录的主机/
  CUDA 栈是否为真。依赖契约和指纹是归档时的声明，需配合发布记录和
  `audit_gates.py` 使用。
- 源码树口径是「测量相关源码和配置」，不含测试、README、docs 和结果
  文件；它绑定的是代码身份，不是论文/说明文档的版本。
- Git 提交仍保留在 manifest 中，但它不再是源码身份的唯一依据；dirty
  工作树可以被如实绑定，而不需要先制造一个干净提交。
- 该改动不改变任何数值、冻结清单、`CODE_SEMANTICS`、种子、终点或
  Holm 决策；也不新增训练或读取 test split。

### 64.4 入口与验证

```bash
python archive_release.py plan  <tag> --dirs results_lm_test analysis_v12
python archive_release.py make  <tag> --dirs results_lm_test analysis_v12
python archive_release.py verify archive/release_<tag>.tar.gz
python test_review_fixes.py              # L190
```

L190 覆盖：真实 `make` 的 tar 中含 `exp_lib.py`/`archive_release.py` 等
源码，manifest 的 `identity` 与 per-file 哈希一致，`verify` 通过；从
per-file map 删除源码、篡改顶层源码摘要或清空 `code_semantics` 会被
身份检查拒绝；把 tar 内 manifest 的 `source_tree_sha256` 改成错误值后，
即使所有文件哈希仍匹配，`verify` 也返回 1。

## 65. 跨 panel 消融门必须识别反证，并区分单面板与复现

### 65.1 问题：覆盖计数会把反证平均掉

`ablation_coverage_across_panels` 原来的项目级读数是「某组件在任意
panel 完整且 verified」即计入 `verified_anywhere`；只要所有组件都在
某处通过，`all_components_useful_allowed` 就为真。若组件 A 在 p1 完整
且方向一致地通过，却在 p2 完整、方向显著相反（或移除效果超过 SESOI 而
有害），旧门仍会允许「每个创新都有用」的项目级声明。这不是复现，而是
按 panel 挑选有利证据：一个反证 panel 不能因为另一个 panel 通过就被
平均掉。反过来，单 panel verified 也不等于独立复现，报告应明确标成
`single_panel`，而不是让读者自行假设已经复制。

### 65.2 方法：反证优先，其次才是复现

对每个声明的组件，`ablation_coverage_across_panels` 在原有
complete / usable / measured / verified / unverified 覆盖读数之外增加：

- `panels_direction_conflict`：该 panel 的
  `direction_conflict_components` 点名了此组件——族校正后显著、但方向
  与预注册假设相反；
- `panels_practically_harmful`：该 panel 的
  `practically_harmful_components` 点名了此组件——移除效果超过预注册
  SESOI 且方向为害；
- `panels_opposite_mean`：至少一个 controlled、n≥min_n 的对比均值为反
  方向；这是描述性警告，不是校正后的反证；
- `panels_measured_unverified`：测过但没有任何 verified 对比的面板；
- `panels_replicated` 与 `replication_status`：至少两个完整 panel 均
  verified 才是 `replicated`；否则按 `contradicted` / `single_panel` /
  `not_verified` / `not_measured` 分级。

顶层新增 `replicated_anywhere`、`single_panel_verified`、
`direction_conflict_anywhere`、`practically_harmful_anywhere` 和
`opposite_mean_anywhere`。`all_components_useful_allowed` 在任何
反证/实际有害存在时关闭，并在 `all_components_useful_reasons` 点名
冲突 panel；旧的 incomplete / verified-only-incomplete 规则保持不变。
`audit_methodology` 把上述跨 panel 列表提升到报告顶层，便于
`audit_gates.py report` 直接消费。

该门不重新计算 p 值，不新增统计检验，也不合并 panel 效应；它只重排
已有 panel 级、族校正后的 verdict。`verified`、SESOI 与 Holm 的语义
完全不变。

### 65.3 边界

- **不是独立复现，也不是外部效度。** `replicated` 只表示两个完整 panel
  都通过；若它们共享种子、验证切片、训练配方或调参历史，误差仍相关。
  该字段不能替代独立数据/独立实验室的 replication。
- **不做 meta-analysis。** 不估计跨 panel 的合并效应、异质性
  (`I²`/`tau²`) 或预测区间；需要合并效应时仍应使用显式的
  `meta_analyze_effects`，并先证明各 panel 的可合并性。
- **反证优先于复现计数。** 冲突 panel 即使 ablation 不完整也仍然点名，
  因为显著反方向结果不能被「该 panel 缺一个臂」抹掉；它不能被当作
  `verified`，也不能被忽略。
- **只对预注册方向负责。** `direction_unspecified` 不算
  `direction_conflict`；未声明方向的 panel 不会被强行判为反证。
  `panels_opposite_mean` 仅为警告，不单独关闭项目门，避免小样本噪声
  翻转结论。
- 该改动不改变冻结清单、种子、终点、alpha、Holm 决策或任何数值结果；
  不新增训练、不读取 test split、不改 `CODE_SEMANTICS`。

### 65.4 入口与验证

```bash
python audit_gates.py report             # 报告含 replicated/single_panel/...
python test_review_fixes.py              # L191
```

L191 覆盖：p1 完整 verified、p2 完整但显著反方向时 A 被标为
`contradicted`、项目门关闭且在 reasons 中点名 p2；B 在两个完整 panel
通过时为 `replicated`；`practically_harmful` 同样关闭门；一个 powered
但非显著的相反均值只产生 `opposite_mean` 警告，不关闭项目门。完整 CPU
套件 539/539 通过（0 skipped）。

## 66. train/validation 近重复段落审计：覆盖诚实与语料内基线校准

### 66.1 问题：精确行匹配和「干净样本」都不足以证明切分干净

现有 `audit_corpus_split` 只在一个确定性 train 样本上检查 validation 行是否
逐 token 完全相同，`audit_split_discipline` 也只做 exact-row overlap。两条
路径都漏掉两种情况：

1. validation 行复用了训练文档中的一段长段落，但行内其它 token 不同，因此
   不是 exact row；原始行级 `_DocDeduplicator` 在行粒度工作，也不能保证
   跨行/半行的段落复用被清除。
2. train 很大时扫描预算只覆盖一小部分；旧逻辑在扫描干净时仍返回 `ok`，
   把「未发现」误写成「证明没有」。validation 是所有探索性 panel 的选型
   切片，这个 fail-open 会直接影响模型选择。

### 66.2 方法：shingle 重叠、语料内基线、显式覆盖

新增 `_passage_overlap_from_segments()`（`audit_gates.py`）：

- 对同一组确定性 train 窗口（默认最多 2M/4M tokens，16 段等距）和
  validation 行计算 32-token 滑动 shingle 的 uint64 多项式哈希；
- 统计 validation 唯一 shingle 在 train 中出现的比例，以及单个 validation
  行的最大重叠比例；
- 用扫描样本的两个不相交部分（A/B）计算 train 内部的重复段落基线；只有
  `val_overlap > 0.25` 且 `val_overlap - baseline > 0.10` 才按集合级违规
  处理。行级同样减去由 train 内部 chunk 估计的 row baseline，避免把
  WikiText 的模板/重复段落误判为泄漏；
- 覆盖不足 `min_scan_fraction`（默认 0.25）时返回 `not_verifiable`，绝不
  返回 `ok`。

`audit_corpus_split` 复用已有的 `_segments`，把 passage 报告写入
`report['passage_overlap']`；集合级或行级违规进入 `hard_failures`（strict
时拒绝训练），覆盖不足进入 warnings。旧的 exact-row 扫描也显式记录
`exact_row_scan_sufficient`：扫描干净但覆盖不足时降为 warnings。
`audit_split_discipline` 在 exact-row 检查后运行同一 helper，并把结果写入
`rec['passage_overlap']`；违规 fail，覆盖不足 warning。

参数（`passage_shingle`、`min_scan_fraction`、`max_val_shingle_overlap`、
`max_passage_excess_overlap`、`max_val_row_shingle_overlap`）全部进入
`audit_corpus_split` 的 memo key，并做有限值/分数范围校验；NaN、负值、
`>1` 的「阈值」不会静默关闭门。

### 66.3 边界

- **不是语义/改写检测。** 32-token exact shingle 对逐段复制和轻微编辑
  敏感，但对同义改写、翻译回译、摘要复用不敏感；它不能替代
  SimHash/嵌入级近重复或人工审阅。
- **不是全量证明。** 扫描是确定性的均匀样本；覆盖不足会明确标成
  `not_verifiable`，但覆盖足够也不证明未扫描区域一定干净。需要更强保证时
  应增大 `max_scan_tokens` 或做全量扫描。
- **基线不是万能校准。** train 内部基线高说明语料自身重复多；此时集合级
  超额阈值仍可能漏掉与常见模板无法区分的泄漏。行级阈值 0.50 只针对
  「单行大半是训练段落」的强信号。
- 该改动不改变任何数据构造、分词器、冻结清单、种子、终点、alpha、Holm
  决策或 `CODE_SEMANTICS`；不新增训练、不读取 test split；只增加零 GPU
  的审计读数和 fail-closed 行为。

### 66.4 入口与验证

```bash
python audit_gates.py split              # 每个 cache 含 passage_overlap
python test_review_fixes.py              # L193
python v12_supp.py plan
```

L193 覆盖：干净随机 split 的 passage `ok`；复制一段 train 后只改一个 token
（exact-row 仍为 0）被行级重叠抓住并 hard fail；精确复制同时触发 exact-row
和 passage；重复模板语料的 train 内部基线接近 1、超额为 0，不误报；扫描
覆盖 5% 时 exact-row 与 passage 都降为 warnings/`not_verifiable`；短行返回
`not_applicable`；`audit_split_discipline` 对同一 cache 的干净/污染版本
分别 ok/fail；NaN/负值/超范围参数被拒绝。完整 CPU 套件 542/542 通过
（0 skipped）。

## 67. 2x2 交互的「不显著」也不等于「可加」：预注册交互 SESOI + TOST

### 67.1 问题：交互 p≥α 被读成 additivity

§59 的 `factorial_interaction_audit` 只做精确符号翻转 + 交互族 Holm：
`p_holm≥α` 只说明当前种子数下没有检出交互，不能证明两个模块的实际
交互为零。另一方面，n 很大时一个 0.1% 的交互也可能「显著」，却不足以
支撑子可加/协同的机制主张。§56 已给留一消融加了 SESOI + TOST，但交互层
仍缺少同一区分，于是报告容易把「未检出」写成「可加」，或把亚 SESOI 的
可检出交互写成机制证据。

### 67.2 方法：log ratio-of-ratios 上的预注册 margin

新增 `INTERACTION_SESOI`（`exp_lib.py`）与
`_normalise_interaction_sesoi_design()`。交互 estimand 保持 §59 的定义：

```text
I_s = (log PPL_11 - log PPL_10) - (log PPL_01 - log PPL_00)
```

margin 也定义在自然对数交互尺度上。`relative_pct=1.0` 换算为
`log(1.01)`，即比值比落在 `[1/1.01, 1.01]`；也接受
`absolute_log_ratio`（仍是 log 尺度，不是 PPL）。布尔、0、负值、NaN/inf、
同时声明两个单位、非字符串名都会 fail closed，不会静默退化成「没有
margin」。

`interaction_practical_decision()` 对逐种子交互值计算配对 TOST，并给出
机器可读的 `decision`：

- `hypothesis_supported`：族校正后显著且 `I` 超过 margin，方向与预注册
  一致（当前三个假设均为正、子可加）；
- `direction_conflict`：族校正后显著、超过 margin，但方向相反；
- `sub_sesoi`：族校正后显著，但 `|I|≤margin`——可检出，不构成实际交互；
- `additive`：TOST 区间落在 margin 内，且（若有）Holm 校正后的 TOST p
  小于 α——这是「实际可加」的正证据，不是「交互恰好为零」；
- `indeterminate`：既没有超过 margin 的交互，也没有落入 margin 的等价
  区间；
- `underpowered` / `no_data` / `not_measured` / `not_declared`。

`factorial_interaction_audit(..., sesoi=None)` 默认读取
`INTERACTION_SESOI`；传入 `{}` 可显式关闭实际显著性层。每个可测交互的
原始 TOST p 先进入**独立的交互等价族**，再做 Holm；缺失/不完整 factorial
保留 p=1 槽位，不能通过少报一个交互来缩小等价性校正族。超显著性检验族
（§59）与等价性族互不混用：`p_holm` 仍是「交互是否非零」的族校正 p，
`p_tost_holm` 才是「能否声称实际可加」的族校正 p。TOST 是 t 近似，没有
精确符号翻转的 `2/2**n` 分辨率下限，因此 `sesoi.family` 不报告
`min_attainable_p_holm`。

顶层新增：

- `equivalent_interactions`
- `hypothesis_supported_interactions`
- `sub_sesoi_interactions`
- `indeterminate_interactions`
- `not_measured_interactions`
- `not_declared_interactions`
- `sesoi.family` / `sesoi.policy`

每个 `factorials[*].interaction` 仍保留 §59 的 `p_holm`、
`significant_family`、`mechanism_claim_allowed`，并新增
`p_tost_holm`、`practical_decision` 与 `sesoi` 块。容量/配方控制仍是
`mechanism_claim_allowed` 的独立硬条件：等价性只说明「交互小」，不能绕开
参数/容量混杂。

### 67.3 设计期门

`interaction_sesoi_audit()`（exp_lib）与
`python audit_gates.py interaction_sesoi` 在跑 GPU 前检查：

- 每个声明的 factorial 都有 margin；缺一个即 fail；
- 单位唯一、有限、>0；相对 margin 会换算为 log 尺度；
- 没有为拼写错误的 factorial 写 margin（warning，防止真 factorial 静默
  失去 margin）；
- 对应 `INTERACTION_HYPOTHESIS_DESIGN` 存在且 `predicted_sign∈{-1,0,1}`；
- 交互声明为空时返回 `not_applicable`，而不是假装通过。

`audit_gates.py report` 现在包含 `interaction_sesoi` 块。

### 67.4 边界

- **不是精确零交互检验。** `additive` 的区间是预注册 margin 内的实际
  可加性；等价于「小到本研究不关心」，不是「I=0」。
- **不是交互的机制确认。** 显著交互仍可能来自容量、实现或训练配方差异；
  只有 `mechanism_claim_allowed=True` 才允许机制措辞。现有
  `param/breakdown` 控制继续生效。
- **不改变冻结确认清单。** 交互族仍是 §59 的探索性族，不进入冻结
  confirmatory 或留一消融 Holm 族；新 margin 只约束交互结果的措辞与
  决策，不新增训练、不改种子/终点/α，也不改 `CODE_SEMANTICS`。
- **TOST 依赖 t 近似与配对可交换性。** 种子数很小或差异严重偏态时，
  区间覆盖率仍受 §62 的校准边界约束；`underpowered` 不会被等价性绕过。
- **只在 panel 内校正。** 交互超显著性族与交互 TOST 族都是单 panel 的
  声明族；跨 panel 的交互复制/反证仍应走 §65 的跨 panel 读数和显式
  meta-analysis，不能把多个 panel 的 `additive` 直接当作全局等价。
- **一个显著但亚 SESOI 的交互不是零。** `sub_sesoi` 明确保留「可检出」，
  不能写成「无交互」；`indeterminate` 也不能写成「可加」。
- **旧 falsifier 的读法。** `INTERACTION_HYPOTHESIS_DESIGN[*]['falsifier']`
  里的 “not distinguishable from zero” 现在必须经新判定解释：只有
  `additive`（TOST 区间在 margin 内）才是否定「实际交互」的正证据；
  `indeterminate` 只是数据不足，不自动算 falsifier。本改动不改写既有
  hypothesis 文本，避免把新决策伪装成对旧预注册的事后重写。
- 默认 1% 是本研究对架构级交互预先采用的实际阈值，与 §56 的消融 1%
  对齐；面板若要更小/更大 margin，必须显式声明并走同一设计门。

### 67.5 入口与验证

```bash
python audit_gates.py interaction_sesoi   # 设计期：覆盖/单位/方向
python audit_gates.py report              # 汇总含 interaction_sesoi
python test_review_fixes.py               # L195-L197
python v12_supp.py plan                   # 冻结计划预检不受影响
```

L195 覆盖默认声明全通过、绝对 log margin、缺 margin/歧义单位/NaN/布尔/
双单位 fail、拼写错误 warning、空设计 `not_applicable` 与 CLI 包装。
L196 覆盖正向支持/反向冲突/亚 SESOI/精确可加/不确定/n<min_n/缺臂/
`sesoi={}` 关闭/部分声明保留未声明 factorial 的 p=1 槽位/无完整 seed 时
列表不重复/畸形声明 fail，以及三个重复 factorial 下超显著性与 TOST
两个族的 m=3 Holm。L197 覆盖直接判定原语、单次迭代器、相对 margin 换算、
畸形族 p 的 fail-closed、方向/ margin 输入校验与 normaliser 幂等。
完整 CPU 套件 546/546 通过（0 skipped）；`python v12_supp.py plan` 仍
72/72 合成冻结记录、6 个比较。

## 68. 设计期功效必须对实际 margin 负责：MDE、精度与所需种子数

### 68.1 问题：能检出「非零」不等于能检出「值得关心」

已有两层功效检查：

- `planned_power_audit` 检查冻结的 n/m/α、`2/2^n` 与 Holm 下限是否自洽，
  以及声明的 q 处投影功效是否达到 `min_power_for_test`；
- `v12_supp.py power` 在 validation 差值上枚举 `2^12` 个符号模式，给出
  条件功效与最小同向种子数。

两者都回答「这个计划在某个 q 下有多大概率通过第一道 Holm 门」，但没有
回答实验设计更基本的问题：**n=12、m=6 能分辨多大的实际效应？** 一个很小
的效应在 n 足够大时也会显著，却可能远低于研究真正关心的最小实际效应
（SESOI）。§56/§67 已给留一消融和 2×2 交互加了 SESOI + TOST，但确认性
主对照本身仍没有 MDE、区间半宽或「在 SESOI 处的功效」读数。预注册若只写
「12 seeds, q=0.95」，读者无法判断这 12 个种子是为哪个效应量准备的。

### 68.2 方法：把 pilot SD、SESOI 与 sign-flip 计划连接起来

新增 `exp_lib.paired_design_sensitivity()`：

```python
rep = paired_design_sensitivity(
    12, 6, alpha=0.05, target_power=0.80,
    pilot_deltas=validation_deltas, expected_sign='negative',
    sesoi=0.5, target_halfwidth=0.2,
    require_sesoi_power=False)
```

输入是设计参数、一个 pilot/validation 配对差值向量（或只给 `pilot_sd`）、
可选 SESOI 与目标区间半宽。输出包括：

- 精确符号翻转/Holm 下限：`p_floor`、`min_attainable_p_holm`；
- 功效曲线：`q=0.50 ... q=0.99` 处的 `signflip_power`；
- `min_q_for_target_power`：达到目标功效所需的最小同向率 q；
- pilot 同向率 `q_sign_rate`、正态隐含的 `q_normal_implied`、当前 n 下的
  功效，以及在目标功效下所需的最小 n；
- **MDE**：在配对正态符号模型下
  `MDE ≈ sigma * Phi^-1(q_target)`，其中 `q_target` 是达到目标功效所需的
  最小 q；同时报告 `MDE / SESOI`；
- **精度**：计划 n 下的配对 t 95% CI 半宽
  `t_{0.975,n-1} * sigma / sqrt(n)`，以及达到目标半宽所需的 n；
- **SESOI 处的功效**：`q_sesoi = Phi(|SESOI| / sigma)`，再用同一个
  `signflip_power(n, m, q_sesoi, alpha)` 计算功效与所需 n。

`require_sesoi_power=True` 时，缺 SESOI、缺 pilot SD、非方向性对照或
SESOI 处功效不足都判 fail；默认 false 时只给 warning，并把
`sesoi.status` 标成 `not_declared` / `not_verifiable` / `no_direction` /
`underpowered`。审计**不会替研究虚构一个 margin**。

### 68.3 接线

- `v12_supp.py power` 的每条对照新增 `sensitivity` 块，用冻结的 n、m 与
  同一条 validation 差值向量计算上述读数；不改变原有 q=0.95 功效门槛。
  若 sensitivity 自身 fail（例如声明了 SESOI 但计划在 margin 处欠功效），
  `all_pass` 会变 false，避免“声明了实际显著性却完全不检查”。
- `python audit_gates.py sensitivity` 读取冻结清单和（可选的）
  `analysis_v12/power_validation.json`；每条主对照输出同一结构。没有
  power 产物时状态是 `not_verifiable`，不是绿色通过。
- `python audit_gates.py report` 汇总里新增 `sensitivity` 块。
- 现有冻结清单尚未声明主对照 SESOI，因此当前读数会明确 warning 并停在
  `not_verifiable`；这是在暴露设计缺口，不是把缺失当成通过。下一次冻结
  可以在首次 test 访问前显式声明 margin 并打开 `require_sesoi_power`。

### 68.4 边界

- **MDE 与 q_sesoi 是正态符号模型近似。** `signflip_power` 本身是等幅
  符号检验投影；`|d_i|` 不等时它是真实 sign-flip 功效的上界。MDE 不是精确
  sign-flip 效应量，只是设计期标尺，不能替代 `v12_supp.py power` 的
  条件功效或最终数据。
- **不是事后功效。** 函数只接受 pilot/validation 差值或显式 SD，不读取
  test 结果；审计入口也不调用 `load_wikitest`。它不改变任何 PPL、种子、
  终点、α、Holm 族、决策规则或 `CODE_SEMANTICS`。
- **pilot 不是确认证据。** pilot 同向率高只说明设计前提暂时可行；不能把
  pilot 的 p 值、MDE 或 SESOI 功效写进确认性结论。
- **文档级精度不是种子级精度。** 这里的 `sigma` 是逐种子配对差值的 SD；
  文档/文章层面的聚类区间需要 document-cluster SD，不能把本函数的 t 区间
  当作 §54/§57 的单位级区间。
- **没有 SESOI 就没有实际显著性结论。** 当前主对照仍只能报告统计方向与
  区间；`sensitivity` 的 warning 是把 §49.4 的边界显式化，而不是新增一个
  任意阈值。
- **n 是计划种子数，不是 attrition 后的样本量。** 所需 n 搜索假设 pilot
  q/SD 固定，且不建模记录缺失、seed 崩溃或 `min_complete_pairs` 低于计划
  seeds 的情形；把这些损失纳入后，真实 MDE 只会更宽。需要保守设计时，应
  用可完成配对数而不是名义 seed 数重跑本函数。
- **n_max 是搜索上限。** 若在 `n_max` 内达不到目标功效，所需 n 返回
  `None`；这表示当前设计参数下不可行，不是“需要更多种子”的自动许可。

### 68.5 入口与验证

```bash
python audit_gates.py sensitivity   # 设计期：功效曲线/MDE/精度/SESOI
python audit_gates.py report        # 汇总含 sensitivity
python test_review_fixes.py         # L198/L199
python v12_supp.py plan             # 冻结计划预检不变
```

L198 覆盖：12×6 的精确下限、功效曲线与公开 `signflip_power` 对账、
pilot 同向率/方向、MDE 与 t 区间半宽、目标半宽所需 n、SESOI 充足/不足/
仅警告/非方向性、缺 pilot、非法 α/target/SD/margin/±方向/字符串 n_max，
以及 n=6,m=4 的分辨率失败。L199 覆盖 manifest + pilot artifact + SESOI 门的接线、缺 pilot 或对照名不匹配时的 `not_verifiable`，以及无 SESOI 时的 warning。完整 CPU 套件 550/550 通过（0 skipped）；
`python v12_supp.py plan` 仍 72/72 合成冻结记录、6 个比较。
