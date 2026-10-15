#!/usr/bin/env python3
"""CSA/HCA 三算子融合内核 —— docs/compute_plan.md §3 的 M0 内核工作包。

三个核心算子（与 exp_lib.py 的 eager 实现一一对应）：

  1. 块压缩 pool（`pool_variable_blocks` / `pool_blocks_single`）：
     逐通道的「段内 gather + 位置嵌入加法 + 掩码 softmax + 加权和」融合为
     单个内核族；A 泳道（块自身 token）与 B 泳道（前块尾部 overlap 重读）
     拼接后做同一维 softmax，与 eager 的 cat-then-softmax 语义一致。
  2. 索引 top-k 选择（`lightning_indexer` 的打分 + `_indexer_selection`）：
     qI·kI 打分（relu 后按 indexer 头权聚合）、因果掩码、log-sum-exp、
     以及与 torch 稳定降序排序完全一致的逐轮首达最大值提取（tie 取最早
     下标），一次融合完成，输出 idx / valid / soft。
  3. 稀疏注意力（`gathered_attention`）：按 [top-k 块 | 滑窗] 泳道做
     flash 式在线 softmax 注意力；sink 列作为在线归约的初值质量；soft
     先验只进入梯度路径（eager 的直通估计器在前向值上严格抵消，本模块
     的反向按同一分解重建 A/S 双路 jacobian）。

数值契约（docs/design_notes.md §5/§6 的内核侧对应物）：

- 内核路径是**独立变体**，目标是与 eager 的统计不可区分（kernel/eager
  同种子配对对照），不承诺逐位一致；eager 路径本身一字未动。
- 反向全部为固定顺序归约：不使用原子加；跨目的地的累加（dk_blk、
  B_pos_b 梯度）经稳定排序 + 段和表达，累加顺序与线程数无关。
- 选择（top-k 下标）与整型几何不可微，反向只经 soft / 打分路径回传。

后端选择（环境变量 `CSA_KERNEL`）：

- 未设置或 '0'：完全走 eager（默认；`exp_lib` 的既有指纹不变）。
- '1'：强制 Triton 后端（需要 CUDA 与 triton 可导入，否则报错）。
- 'auto'：CUDA + triton 可用时用 Triton，否则用 torch 参考后端。
- 'ref'：torch 参考后端（任意设备；与 Triton 内核同一套算法分解的
  张量表达，用于 CPU 侧语义验证与调试）。

内核变体标识：`KERNEL_STAMP`。内核模式开启时 exp_lib 的 run_cfg 指纹与
`_code` 戳追加 `-tk1:<backend>`，eager 记录的指纹逐字节不变（配对纪律，
design_notes §3.1）。G0 数值对照与微基准的驱动入口是
`python m0_supp.py kernel`。
"""
import math
import os

import torch
import torch.nn.functional as F

KERNEL_STAMP = 'tk1'
_NEG_INF = float('-inf')


def _env_mode():
    return (os.environ.get('CSA_KERNEL') or '').strip().lower()


_TRITON_STATE = {'checked': False, 'ok': False, 'version': None}


def triton_importable():
    """进程内一次性的 triton 可导入性检查（只在 CUDA 设备在场时有意义）。"""
    if not _TRITON_STATE['checked']:
        _TRITON_STATE['checked'] = True
        try:
            import triton
            _TRITON_STATE['version'] = getattr(triton, '__version__', 'unknown')
            _TRITON_STATE['ok'] = True
        except Exception:
            _TRITON_STATE['ok'] = False
    return _TRITON_STATE['ok']


def kernel_backend():
    """解析当前进程的内核后端：'' = 关闭（eager），'triton' 或 'ref'。"""
    mode = _env_mode()
    if mode in ('', '0', 'off', 'none', 'false'):
        return ''
    if mode == 'ref':
        return 'ref'
    want_triton = mode in ('1', 'true', 'on', 'triton')
    if want_triton:
        if not (torch.cuda.is_available() and triton_importable()):
            raise RuntimeError(
                "CSA_KERNEL=1 requires CUDA and an importable triton, "
                f"but cuda={torch.cuda.is_available()} triton={triton_importable()}; "
                "unset CSA_KERNEL to stay on the eager path, or use CSA_KERNEL=ref for the "
                "torch-reference backend (CPU-debuggable)")
        return 'triton'
    if mode == 'auto':
        if torch.cuda.is_available() and triton_importable():
            return 'triton'
        return 'ref'
    raise ValueError(f"unknown CSA_KERNEL value {os.environ.get('CSA_KERNEL')!r} "
                     "(expected unset/'0' (eager), '1' (force triton), 'auto', or 'ref')")


def kernel_enabled():
    return kernel_backend() != ''


def kernel_stamp():
    """进入 run_cfg 指纹 / `_code` 戳的内核标识；eager 模式返回空串。"""
    backend = kernel_backend()
    return '' if backend == '' else f'-{KERNEL_STAMP}:{backend}'


_ENV_CACHE = {}


def _env_cached(name):
    """进程内缓存的环境变量读取：这些开关（CSA_KERNEL_TF32 /
    CSA_KERNEL_{POOL,IDX,ATTN} / CSA_KERNEL_IDX_CHUNK）在仓库内从不被
    运行期改写（只有 CSA_KERNEL 本身按设计每次 forward 重读，不走这里），
    与 _TRITON_STATE 同为 cache-once 模式。"""
    v = _ENV_CACHE.get(name, None)
    if v is None:
        v = os.environ.get(name)
        _ENV_CACHE[name] = v
    return v


def _finfo_min(dtype):
    return torch.finfo(dtype).min if dtype.is_floating_point else 0.0


def _next_pow2(n):
    p = 1
    while p < int(n):
        p <<= 1
    return max(p, 1)


def _det_scatter_add_multi(n_rows, indices, values_list, device, dtype):
    """_det_scatter_add 的多值表形式：同一组键只排一次序、只做一次段定位，
    各值表分别段和（段和沿列独立归约——每张表的结果与逐张调用
    _det_scatter_add 逐位一致，共享的只是排序与段结构）。values_list 中的
    None 原样返回 None。dk/dv（或 dksw/dvsw）的键逐元素相同，共用同一组
    排序与段结构。"""
    idx = indices.reshape(-1).to(torch.int64)
    n_rows = int(n_rows)
    outs = [None if v is None else torch.zeros((n_rows, *v.shape[1:]), device=device, dtype=dtype) for v in values_list]
    if idx.numel() == 0:
        return outs
    order = torch.argsort(idx, stable=True)
    idx_s = idx.index_select(0, order)
    end = torch.zeros_like(idx_s, dtype=torch.bool)
    if idx_s.numel() > 1:
        end[:-1] = idx_s[1:] != idx_s[:-1]
    end[-1] = True
    seg_end = torch.nonzero(end, as_tuple=False).reshape(-1)
    seg_id = idx_s.index_select(0, seg_end)
    keep = (seg_id >= 0) & (seg_id < n_rows)
    seg_id_k = seg_id[keep]
    lo_idx = seg_end[:-1]
    for values, out in zip(values_list, outs):
        if values is None:
            continue
        val_s = values.index_select(0, order).to(torch.float32)
        csum = val_s.cumsum_(dim=0)
        hi = csum.index_select(0, seg_end)
        lo = torch.zeros_like(hi)
        if seg_end.numel() > 1:
            lo[1:] = csum.index_select(0, lo_idx)
        seg_sum = hi - lo
        out.index_copy_(0, seg_id_k, seg_sum[keep].to(dtype))
    return outs


def _det_scatter_add(n_rows, indices, values, device, dtype):
    """确定性 scatter-累加（values 为 [N] 或 [N, D]）：同键的值按稳定排序
    后的固定顺序做前缀和，段和 = 段尾前缀 − 段首前的前缀。

    design_notes §6 的「需要 scatter 的位置用固定顺序的形式表达」在 torch
    侧的落点：CUDA 的 index_add_/index_put_(accumulate=True) 走原子加，
    累加顺序依赖线程调度；本实现先把 (key, value) 平铺后按 key 稳定排序，
    段内顺序因此只由 (key, 原下标) 决定，cumsum 的分组与线程数无关。
    """
    return _det_scatter_add_multi(n_rows, indices, [values], device, dtype)[0]


def _seg_structure(keys, n_rows):
    """跨头共享的段归约结构（与 _det_scatter_add_multi 同一套固定顺序
    归约）。注意力反向的 dk/dv 键含头维（base·nh + h）：同一批 base 键对
    每个头只差一个常量偏移，稳定排序给出同一排列、同一批段边界，因此
    argsort 与段定位全部头共用一次，段内累加顺序与逐头排序形式逐元素
    一致。"""
    idx = keys.reshape(-1).to(torch.int64)
    n_rows = int(n_rows)
    order = torch.argsort(idx, stable=True)
    idx_s = idx.index_select(0, order)
    end = torch.zeros_like(idx_s, dtype=torch.bool)
    if idx_s.numel() > 1:
        end[:-1] = idx_s[1:] != idx_s[:-1]
    if idx_s.numel():
        end[-1] = True
    seg_end = torch.nonzero(end, as_tuple=False).reshape(-1)
    seg_id = idx_s.index_select(0, seg_end)
    keep = (seg_id >= 0) & (seg_id < n_rows)
    return (order, seg_end, seg_id[keep], keep)


def _seg_apply(seg, values, out_flat, h, nh):
    """按共享段结构把一个头的值表归约进扁平梯度缓冲的 head=h 行：值经
    同一排列 gather 后 cumsum，段和写入 seg_id·nh + h——即旧键
    (base·nh + h) 排序定位到的同一批行；每行每头恰写一次，与旧形式的
    零初始化后 += 同值。"""
    order, seg_end, seg_id_k, keep = seg
    val_s = values.index_select(0, order).to(torch.float32)
    csum = val_s.cumsum_(dim=0)
    hi = csum.index_select(0, seg_end)
    lo = torch.zeros_like(hi)
    if seg_end.numel() > 1:
        lo[1:] = csum.index_select(0, seg_end[:-1])
    seg_sum = hi - lo
    out_flat.index_copy_(0, seg_id_k * int(nh) + int(h), seg_sum[keep])


class PoolGeometry:
    """一个 batch 的逐行池化几何（全部为整型张量，值与 eager 的
    `_pool_parts(_host)` 完全同源——都来自分段器自己的 counts 表）。

    形状记号：R = batch 行数，Bm = 行间最大块数，T = 序列长。
    counts[r, b]   块内 token 数（pad 块为 0）
    ends[r, b]     块尾（排他）；pad 块与前一末尾一致
    starts[r, b]   块首
    last_tok[r, b] 块内最后 token 下标（pad 块置 T-1：t > T-1 恒假，
                   与「pad 块永不可读」一致）
    ov_start[r, b] B 泳道起点（前块尾部 overlap 窗）
    ov_len[r, b]   B 泳道长度
    starts_prev[r, b] 前块块首（b=0 时为 0，与 eager 的 F.pad 语义一致）
    b_off[r, b, j]  B 泳道位置嵌入行（ov_start - starts_prev + j，截断到
                   B_pos_b 行数内）——按需惰性构建
    n_blocks[r]    该行真实块数
    """

    def __init__(self, counts_rows, overlap, T, device, max_block):
        import numpy as np
        R = len(counts_rows)
        Bm = max((len(c) for c in counts_rows), default=0)
        Bm = max(Bm, 1)
        # 全部表都是整型：先按行填入 counts，之后用 2D numpy 一次性求
        # ends/starts/overlap 几何（整数的 cumsum/移位/clamp 沿轴逐位精确，
        # 与逐行循环同一套公式同一批值——逐位一致）。
        counts_np = np.zeros((R, Bm), dtype=np.int64)
        nblk_np = np.zeros((R,), dtype=np.int64)
        for r, counts in enumerate(counts_rows):
            n = len(counts)
            nblk_np[r] = n
            if n:
                counts_np[r, :n] = counts
        ends_full = np.cumsum(counts_np, axis=1, dtype=np.int64)
        starts_full = ends_full - counts_np
        colmask = np.arange(Bm)[None, :] < nblk_np[:, None]
        ends_np = np.where(colmask, ends_full, 0)
        starts_np = np.where(colmask, starts_full, 0)
        last_np = np.where(colmask, np.clip(ends_full - 1, 0, T - 1), T - 1)
        max_len = max(1, int(counts_np.max(initial=0)))
        if overlap is not None:
            end_prev_full = np.concatenate([np.full((R, 1), -1, dtype=np.int64), ends_full[:, :-1]], axis=1)
            starts_prev_full = np.concatenate([np.zeros((R, 1), dtype=np.int64), starts_full[:, :-1]], axis=1)
            ov_start_full = np.maximum(end_prev_full - int(overlap), starts_prev_full)
            ov_len_full = end_prev_full - ov_start_full
            starts_prev_np = np.where(colmask, starts_prev_full, 0)
            ov_start_np = np.where(colmask, ov_start_full, 0)
            ov_len_np = np.where(colmask, ov_len_full, 0)
            # 与逐行形式相同的 max_ov 口径：只统计列 1..n-1（列 0 无 overlap，
            # 其 ov_len 恒负/零且被逐行形式的 [1:] 切片排除）。
            if Bm > 1:
                max_ov = int(np.where(colmask[:, 1:], ov_len_full[:, 1:], 0).max(initial=0))
            else:
                max_ov = 0
        else:
            ov_start_np = np.zeros((R, Bm), dtype=np.int64)
            ov_len_np = np.zeros((R, Bm), dtype=np.int64)
            starts_prev_np = np.zeros((R, Bm), dtype=np.int64)
            max_ov = 0
        W_b = max_ov if overlap is not None else 0
        # 8 张 int32 几何表打包进一条宿主缓冲、一次 H2D（同样的 int32 值、
        # 同样的 int64→int32 窄化，transport only），设备侧按只读视图切分。
        flat = np.empty(7 * R * Bm + R, dtype=np.int32)
        for _i, _a in enumerate((counts_np, ends_np, starts_np, last_np,
                                 ov_start_np, ov_len_np, starts_prev_np)):
            flat[_i * R * Bm:(_i + 1) * R * Bm] = _a.reshape(-1)
        flat[7 * R * Bm:] = nblk_np
        _buf = torch.from_numpy(flat).to(device)
        _v = _buf[:7 * R * Bm].view(7, R, Bm)
        self.R, self.Bm, self.T = R, Bm, T
        self.counts = _v[0]
        self.ends = _v[1]
        self.starts = _v[2]
        self.last_tok = _v[3]
        self.ov_start = _v[4]
        self.ov_len = _v[5]
        self.starts_prev = _v[6]
        self.n_blocks = _buf[7 * R * Bm:]
        self.max_len = int(max_len)
        self.W_b = int(W_b)
        self.overlap = overlap
        self.max_block = int(max_block)
        self._b_off = None
        self._bposb_rows = None
        self._ga_cache = None
        self._g_b = None
        self._i64 = None

    def gather_geom(self):
        """A 泳道的纯几何 gather 常量 (g_a [R,Bm,L] int64, av_all [R,Bm,L,1]
        bool)——内容是 geom 的纯整型函数，只读消费；缓存论证与 b_off /
        _GEOM_CACHE 同（消掉每次 fwd/bwd 调用对 [R,Bm,L] 的重建）。"""
        if self._ga_cache is None:
            jA = torch.arange(self.max_len, device=self.starts.device)
            g_a = (self.starts.to(torch.int64)[:, :, None] + jA[None, None, :]).clamp(max=int(self.T) - 1)
            av_all = jA[None, None, :, None] < self.counts.to(torch.int64)[:, :, None, None]
            self._ga_cache = (g_a, av_all)
        return self._ga_cache

    def gather_geom_b(self):
        """B 泳道 gather 常量 g_b [R,Bm,W_b] int64（W_b == 0 时返回 None）。"""
        if int(self.W_b) <= 0:
            return None
        if self._g_b is None:
            jB = torch.arange(self.W_b, device=self.starts.device)
            self._g_b = (self.ov_start.to(torch.int64)[:, :, None] + jB[None, None, :]).clamp(max=int(self.T) - 1)
        return self._g_b

    def i64_views(self):
        """geom 标量表的 int64 形式（int64 原型的 .to 是 no-op，非原型的
        cast 结果在此缓存，避免每次调用逐张重cast）。"""
        if self._i64 is None:
            self._i64 = dict(starts=self.starts.to(torch.int64),
                             counts=self.counts.to(torch.int64),
                             ov_start=self.ov_start.to(torch.int64),
                             ov_len=self.ov_len.to(torch.int64),
                             n_blocks=self.n_blocks.to(torch.int64))
        return self._i64

    def b_off(self, bposb_rows):
        """[R, Bm, W_b] 的 B 泳道位置嵌入行号（带 eager 同款截断）。"""
        if self._b_off is not None and self._bposb_rows == int(bposb_rows):
            return self._b_off
        j = torch.arange(self.W_b, device=self.starts.device, dtype=torch.int32)
        raw = self.ov_start[:, :, None] - self.starts_prev[:, :, None] + j[None, None, :]
        self._b_off = raw.clamp(max=int(bposb_rows) - 1).long()
        self._bposb_rows = int(bposb_rows)
        return self._b_off


# 几何/常数表缓存：键 = 内容的纯整型函数，缓存值只读消费（与 exp_lib 的
# _pool_parts/_arange_exp_cache 同一论证）。fix 分块臂每步传入完全相同的
# counts。
_GEOM_CACHE = {}
_GEOM_CACHE_CAP = 16
_ARANGE_CACHE = {}
_ARANGE_CACHE_CAP = 8
_ARANGE_CACHE_BYTES = [0]
_ARANGE_CACHE_BUDGET = 512 << 20


def _full_arange_evict(need=0):
    # 与 _GEOM_CACHE 同一套 FIFO 淘汰：(B, T, Bm) 组合数无天然上界
    # （例如 eval 长度扫描），缓存条目数与驻留字节都必须有上界。
    while _ARANGE_CACHE and (len(_ARANGE_CACHE) >= _ARANGE_CACHE_CAP or
                             _ARANGE_CACHE_BYTES[0] + need > _ARANGE_CACHE_BUDGET):
        _k0 = next(iter(_ARANGE_CACHE))
        _t0 = _ARANGE_CACHE.pop(_k0)
        _ARANGE_CACHE_BYTES[0] -= _t0.numel() * _t0.element_size()


def _pool_geometry_cached(counts_rows, overlap, T, device, max_block):
    # 键 = 行 counts 对象的身份（条目强引用这些对象，因此 id 在条目存活期间
    # 唯一且稳定，不可能撞上复用）；fix 臂全程复用同一分段器出口对象必命中，
    # 动态臂每步拿到新对象必 miss——命中语义与内容键一致（counts 表按约定
    # 不可变，与 exp_lib `_pool_parts` 缓存于对象身份同一论证）。
    key = (tuple(id(c) for c in counts_rows),
           None if overlap is None else int(overlap), int(T),
           str(device), int(max_block))
    ent = _GEOM_CACHE.get(key)
    if ent is not None:
        refs, geom = ent
        if len(refs) == len(counts_rows) and all(a is b for a, b in zip(refs, counts_rows)):
            return geom
    if len(_GEOM_CACHE) >= _GEOM_CACHE_CAP:
        _GEOM_CACHE.pop(next(iter(_GEOM_CACHE)))
    geom = PoolGeometry(counts_rows, overlap, T, device, max_block)
    _GEOM_CACHE[key] = (tuple(counts_rows), geom)
    return geom


def _full_arange_cached(B, T, Bm, device):
    key = (int(B), int(T), int(Bm), str(device))
    out = _ARANGE_CACHE.get(key)
    if out is None:
        _full_arange_evict(int(B) * int(T) * int(Bm) * 4)
        out = torch.arange(Bm, device=device, dtype=torch.int32)[None, None, :].expand(B, T, Bm).contiguous()
        _ARANGE_CACHE[key] = out
        _ARANGE_CACHE_BYTES[0] += out.numel() * out.element_size()
    return out


# ============================================================================
# 算子 1：块压缩 pool（`pool_variable_blocks` / `pool_blocks_single` 的内核路径）
# ============================================================================

_POOL_CHUNK_BYTES = 64 << 20


def _pool_chunk_b(R, Ltot, c):
    per_block = max(1, int(R) * int(Ltot) * int(c) * 4)
    return max(1, min(int(_POOL_CHUNK_BYTES) // per_block, 4096))


def _gather_rows(x, idx):
    """x [R, S, D] 按 idx [R, ..., k]（int64）在 S 维 gather，保留尾维。"""
    R, _S, D = x.shape
    flat = idx.reshape(R, -1)
    out = x.gather(1, flat.unsqueeze(-1).expand(R, flat.shape[1], D))
    return out.view(*idx.shape, D)


def _pool_fwd_ref(Xa, Xb, Za, Zb, geom, B_pos_a, B_pos_b):
    """池化前向的 torch 参考实现（按块分块，内存有界）。

    eager 语义对齐点：
    - 泳道 logits：Za[g] + B_pos_a[j]（A 泳道）/ Zb[g] + B_pos_b[b_off]
      （B 泳道），无效泳道在位置嵌入加法**之后**置 -inf；
    - softmax 维度 = [A 泳道 | B 泳道] 拼接后的泳道维，逐通道独立；
    - softmax 后 nan_to_num（全 -inf 泳道行 → 0），pad 块 comp = 0；
    - X 的 gather 下标带 eager 同款截断，但无效泳道 sc = 0，贡献为 0。
    """
    R, T, c = Xa.shape
    Bm = geom.Bm
    dev = Xa.device
    dtype = Xa.dtype
    L, W = geom.max_len, geom.W_b
    has_b = Xb is not None and W > 0
    assert Za.shape[-1] == c, 'pool 语义要求 X 与 Z 同宽（kv_dim）'
    comp = torch.zeros((R, Bm, c), device=dev, dtype=dtype)
    jA = torch.arange(L, device=dev)
    g_a, av_all = geom.gather_geom()
    pos_a = jA.clamp(max=int(B_pos_a.shape[0]) - 1)
    Bp_a = B_pos_a[pos_a].to(dtype)
    if has_b:
        jB = torch.arange(W, device=dev)
        g_b = geom.gather_geom_b()
        b_off = geom.b_off(B_pos_b.shape[0]).to(dev)
        Bp_b = B_pos_b[b_off].to(dtype)
    ch = _pool_chunk_b(R, L + (W if has_b else 0), c)
    for b0 in range(0, Bm, ch):
        b1 = min(b0 + ch, Bm)
        av = av_all[:, b0:b1]
        # 这些 logit 张量都是本轮新算出的临时量（无别名），fill 原地进行——
        # 值与出位版逐位相同。
        logit_a = _gather_rows(Za, g_a[:, b0:b1]) + Bp_a[None, None] * av
        logit_a.masked_fill_(~av, _NEG_INF)
        if has_b:
            vm = jB[None, None, :, None] < geom.ov_len.to(torch.int64)[:, b0:b1, None, None]
            logit_b = _gather_rows(Zb, g_b[:, b0:b1]) + Bp_b[:, b0:b1] * vm
            logit_b.masked_fill_(~vm, _NEG_INF)
            WinZ = torch.cat([logit_a, logit_b], dim=2)
        else:
            WinZ = logit_a
        sc = torch.nan_to_num(torch.softmax(WinZ, dim=2))
        sc_a, sc_b = sc[:, :, :L], (sc[:, :, L:] if has_b else None)
        part = (_gather_rows(Xa, g_a[:, b0:b1]) * sc_a).sum(dim=2)
        if has_b:
            part = part + (_gather_rows(Xb, g_b[:, b0:b1]) * sc_b).sum(dim=2)
        comp[:, b0:b1] = part
    return comp


def _pool_bwd_ref(Xa, Xb, Za, Zb, geom, B_pos_a, B_pos_b, bid, dcomp,
                  need_xa, need_xb, need_za, need_zb, need_bpa, need_bpb):
    """池化反向（两个后端共用；确定性——块升序分块处理，B_pos_b 的跨块
    累加走稳定排序段和）。

    梯度分解（逐通道独立，与 eager 的 autograd 图逐一对应）：
      gsc[b,l,f]    = dcomp[b,f] · X[g_l, f]
      Srow[b,f]     = Σ_l gsc[b,l,f] · sc[b,l,f]
      glogit[b,l,f] = sc · (gsc − Srow)
      gX[t,f]       = sc[own lane] · dcomp[b(t)]（另加 B 泳道：sc_b · dcomp[b(t)+1]）
      gZ[t,f]       = glogit[own lane]（另加 B 泳道项）
      gB_pos_a[j,f] = Σ_{r,b} glogit_a[r,b,j,f]
      gB_pos_b[o,f] = Σ_{lanes: b_off=o} glogit_b
    A 泳道与 token 一一对应（g_a = starts + j 是块内双射）；每个 token 至多
    出现在一个 B 泳道（overlap 窗被夹在前块内部），因此 gXb/gZb 是纯 gather。
    """
    R, T, c = Xa.shape
    Bm = geom.Bm
    dev = Xa.device
    dtype = Xa.dtype
    L, W = geom.max_len, geom.W_b
    has_b = Xb is not None and W > 0
    gXa = torch.zeros_like(Xa) if need_xa else None
    gZa = torch.zeros_like(Za) if need_za else None
    # W_b == 0（overlap=0 / 单块行）时 B 泳道为空：eager 的 autograd 图里
    # 这些张量仍在（空 gather/空求和），拿到的是零梯度张量而不是 None——
    # None 会让 AdamW 整参数跳过（连 weight decay 都不施加），零张量则与
    # eager 的更新逐步一致，退化几何下两模式不得分流。
    gXb = torch.zeros_like(Xb) if (need_xb and Xb is not None) else None
    gZb = torch.zeros_like(Zb) if (need_zb and Zb is not None) else None
    gBpa = torch.zeros_like(B_pos_a) if need_bpa else None
    jA = torch.arange(L, device=dev)
    g_a, av_all = geom.gather_geom()
    pos_a = jA.clamp(max=int(B_pos_a.shape[0]) - 1)
    Bp_a = B_pos_a[pos_a].to(dtype)
    if has_b:
        jB = torch.arange(W, device=dev)
        g_b = geom.gather_geom_b()
        b_off = geom.b_off(B_pos_b.shape[0]).to(dev)
        Bp_b = B_pos_b[b_off].to(dtype)
    bid_l = bid.to(torch.int64)
    t_idx = torch.arange(T, device=dev)[None, :].expand(R, T)
    r_idx = torch.arange(R, device=dev)[:, None].expand(R, T)
    _i64 = geom.i64_views()
    starts_l = _i64['starts']
    ov_start_l = _i64['ov_start']
    nblk_l = _i64['n_blocks'][:, None]
    gBpb_keys, gBpb_vals = [], []
    ch = _pool_chunk_b(R, L + (W if has_b else 0), c)
    # 广播用的零标量只读，逐 chunk 重建是同一常量——循环外只建一次。
    zero32 = torch.zeros((), device=dev, dtype=torch.float32)
    for b0 in range(0, Bm, ch):
        b1 = min(b0 + ch, Bm)
        av = av_all[:, b0:b1]
        # 这些 logit 张量都是本轮新算出的临时量（无别名），fill 原地进行——
        # 值与出位版逐位相同。
        logit_a = _gather_rows(Za, g_a[:, b0:b1]) + Bp_a[None, None] * av
        logit_a.masked_fill_(~av, _NEG_INF)
        if has_b:
            vm = jB[None, None, :, None] < geom.ov_len.to(torch.int64)[:, b0:b1, None, None]
            logit_b = _gather_rows(Zb, g_b[:, b0:b1]) + Bp_b[:, b0:b1] * vm
            logit_b.masked_fill_(~vm, _NEG_INF)
            WinZ = torch.cat([logit_a, logit_b], dim=2)
        else:
            WinZ = logit_a
        sc = torch.nan_to_num(torch.softmax(WinZ, dim=2)).to(torch.float32)
        sc_a, sc_b = sc[:, :, :L], (sc[:, :, L:] if has_b else None)
        dc = dcomp[:, b0:b1].to(torch.float32)
        gsc_a = dc[:, :, None] * _gather_rows(Xa, g_a[:, b0:b1]).to(torch.float32)
        if has_b:
            gsc_b = dc[:, :, None] * _gather_rows(Xb, g_b[:, b0:b1]).to(torch.float32)
            Srow = (gsc_a * sc_a).sum(dim=2) + (gsc_b * sc_b).sum(dim=2)
            glogit_b = sc_b * (gsc_b - Srow[:, :, None])
        else:
            Srow = (gsc_a * sc_a).sum(dim=2)
            glogit_b = None
        glogit_a = sc_a * (gsc_a - Srow[:, :, None])
        if need_xa or need_za:
            in_ch = (bid_l >= b0) & (bid_l < b1)
            sel = bid_l.clamp(b0, b1 - 1)
            jA_t = (t_idx - starts_l.gather(1, sel)).clamp(0, L - 1)
            dc_sel = dcomp.gather(1, sel[:, :, None].expand(R, T, c)).to(torch.float32)
            if need_xa:
                # sc_a/glogit_a are chunk-local ([R, b1-b0, L, c]); rebase the
                # global clamped block id into chunk coordinates before the
                # lane gather (starts_l/dcomp stay global-indexed above).
                sc_a_sel = _take_lane(sc_a, r_idx, sel - b0, jA_t)
                gXa += torch.where(in_ch[:, :, None], sc_a_sel * dc_sel, zero32).to(dtype)
            if need_za:
                gl_a_sel = _take_lane(glogit_a, r_idx, sel - b0, jA_t)
                gZa += torch.where(in_ch[:, :, None], gl_a_sel, zero32).to(dtype)
        if has_b and (need_xb or need_zb):
            b2 = (bid_l + 1).clamp(0, Bm - 1)
            j2 = t_idx - ov_start_l.gather(1, b2)
            ok2 = (j2 >= 0) & (bid_l + 1 < nblk_l) & (bid_l + 1 < Bm)
            ok2 = ok2 & (b2 >= b0) & (b2 < b1)
            b2c = b2.clamp(b0, b1 - 1)
            j2c = j2.clamp(0, W - 1)
            dc2 = dcomp.gather(1, b2c[:, :, None].expand(R, T, c)).to(torch.float32)
            if need_xb:
                sc_b_sel = _take_lane(sc_b, r_idx, b2c - b0, j2c)
                gXb += torch.where(ok2[:, :, None], sc_b_sel * dc2, zero32).to(dtype)
            if need_zb:
                gZb += torch.where(ok2[:, :, None], _take_lane(glogit_b, r_idx, b2c - b0, j2c), zero32).to(dtype)
        if need_bpa:
            gsum = glogit_a.sum(dim=(0, 1))
            # 前向把 pos_a 钳到 B_pos_a 末行（见上方 clamp）：超程泳道的
            # 梯度必须按同一钳位语义累加进末行，而不是按形状截断或报错。
            _rows = min(int(gsum.shape[0]), int(gBpa.shape[0]))
            gBpa[:_rows] += gsum[:_rows].to(B_pos_a.dtype)
            if _rows < int(gsum.shape[0]):
                gBpa[_rows - 1] += gsum[_rows:].sum(dim=0).to(B_pos_a.dtype)
        if need_bpb and has_b:
            gBpb_keys.append(b_off[:, b0:b1].reshape(-1))
            gBpb_vals.append(glogit_b.reshape(-1, c))
    gBpb = None
    if need_bpb:
        # 同 gXb/gZb：W_b == 0 时 keys/vals 为空，段和返回全零——与 eager
        # 空泳道图的零梯度一致（HCA 把同一 B_pos 作 a/b 两参传入，零项
        # 不改变 autograd 对同一张量的梯度求和）。
        keys = torch.cat(gBpb_keys) if gBpb_keys else torch.zeros(0, device=dev, dtype=torch.long)
        vals = torch.cat(gBpb_vals) if gBpb_vals else torch.zeros(0, c, device=dev, dtype=torch.float32)
        gBpb = _det_scatter_add(B_pos_b.shape[0], keys, vals, dev, B_pos_b.dtype)
    return gXa, gXb, gZa, gZb, gBpa, gBpb


def _take_lane(lane4, r_idx, blk_idx, lane_idx):
    """[R, B, L, c] 泳道张量按 (r, blk_idx[r,t], lane_idx[r,t]) 取 [R, T, c]——
    纯 gather，无累加（A 泳道双射 / B 泳道至多一处的两处用途都满足）。"""
    R, B, L, c = lane4.shape
    row = (r_idx.reshape(-1) * B + blk_idx.reshape(-1).clamp(0, B - 1))
    col = lane_idx.reshape(-1).clamp(0, L - 1)
    out = lane4.reshape(R * B, L, c)[row, col]
    return out.reshape(r_idx.shape[0], r_idx.shape[1], c)


class FusedPool(torch.autograd.Function):
    """块压缩 pool 的可微封装：前向走内核（Triton 或 torch 参考），反向为
    共用的确定性 torch 分解（见 _pool_bwd_ref 的推导注释）。"""

    @staticmethod
    def forward(ctx, impl, Xa, Xb, Za, Zb, bid, geom, B_pos_a, B_pos_b):
        if impl == 'triton':
            comp = _pool_fwd_triton(Xa, Xb, Za, Zb, geom, B_pos_a, B_pos_b)
        else:
            comp = _pool_fwd_ref(Xa, Xb, Za, Zb, geom, B_pos_a, B_pos_b)
        ctx.save_for_backward(Xa, Xb, Za, Zb, B_pos_a, B_pos_b, bid)
        ctx.geom = geom
        return comp

    @staticmethod
    def backward(ctx, dcomp):
        Xa, Xb, Za, Zb, B_pos_a, B_pos_b, bid = ctx.saved_tensors
        geom = ctx.geom
        if dcomp is None:
            return (None,) * 9
        dcomp = dcomp.contiguous()
        ng = ctx.needs_input_grad
        gXa, gXb, gZa, gZb, gBpa, gBpb = _pool_bwd_ref(
            Xa, Xb, Za, Zb, geom, B_pos_a, B_pos_b, bid, dcomp,
            ng[1], ng[2], ng[3], ng[4], ng[7], ng[8])
        return (None, gXa, gXb, gZa, gZb, None, None, gBpa, gBpb)


def pool_variable_blocks_kernel(Xa, Xb, Za, Zb, bid, geom, B_pos_a, B_pos_b, impl=None):
    """`exp_lib.pool_variable_blocks` 的内核路径（批量化：R 行一次过）。
    返回 (comp [R, Bm, c], last_tok [R, Bm] int32, Bm)。"""
    impl = impl or _op_impl('POOL')
    comp = FusedPool.apply(impl, Xa, Xb, Za, Zb, bid, geom, B_pos_a, B_pos_b)
    return comp, geom.last_tok, geom.Bm


def pool_blocks_single_kernel(X, Z, bid, geom, B_pos, impl=None):
    """`exp_lib.pool_blocks_single` 的内核路径（HCA：仅 A 泳道）。"""
    impl = impl or _op_impl('POOL')
    comp = FusedPool.apply(impl, X, None, Z, None, bid, geom, B_pos, B_pos)
    return comp, geom.last_tok, geom.Bm


# ============================================================================
# 算子 2：索引打分 + top-k 选择（`lightning_indexer` + `_indexer_selection`）
# ============================================================================

def _indexer_fwd_ref(qI, kI, w_idx, last_tok, topk):
    """torch 参考实现（与 Triton 内核同一套算法分解）。

    返回 (idx [R,T,k] int32, valid [R,T,k] bool, soft [R,T,k])。

    与 eager 的语义对齐点：
    - 打分：scores[t,b] = Σ_i relu((qI[t,i]·kI[b,i])·iscale)·w[t,i]；
    - lse 的掩码是**仅因果**（~causal → -inf，非有限值保留原样进入
      logsumexp——与 eager 的 masked_fill_(~causal) 完全一致）；
    - top-k 掩码是因果 ∧ 有限（cf），与 eager 的 `cf = causal & finite`
      一致；tie 规则 = 稳定降序排序的最早下标——逐轮「首达最大值」提取
      与 torch.sort(stable=True, descending=True) 的前缀严格同序；
    - 有效槽位不足 k 时，尾部以块 0 填充，valid=False——与 eager 的
      compaction+padding 逐位一致（eager 的 pad_blk 恒为 0：池化给出的
      last_tok 单调不减，首个因果块必为块 0，两种形式对所有池化输入
      同值；此处直接取 eager 形式）；
    - soft 在无效槽位直接置 0（eager 在下游乘 keep=0，值等价）。
    """
    R, T, nIH, hd = qI.shape
    Bm = kI.shape[1]
    iscale = hd ** -0.5
    dev = qI.device
    dtype = qI.dtype
    t_idx = torch.arange(T, device=dev)[None, :, None]
    # causal 只在 t 分块切片上消费：循环内按片做同一整数比较（last_tok 的
    # int64 升型循环外一次），布尔值与全幅形式逐格相同，[R, T, Bm] 全幅
    # 掩码不物化。
    lt64 = last_tok.to(torch.int64)
    # fp32 升型整体只做一次（逐片 cast 的值与整张 cast 后取片逐位一致）：
    # 否则每个头都重复拷贝一次 qI / kI / w_idx 的 fp32 版本。
    qIf = qI if qI.dtype == torch.float32 else qI.to(torch.float32)
    kIf = kI if kI.dtype == torch.float32 else kI.to(torch.float32)
    wf = w_idx if w_idx.dtype == torch.float32 else w_idx.to(torch.float32)
    k_out = min(int(topk), Bm)
    idx = torch.empty((R, T, k_out), device=dev, dtype=torch.int32)
    valid = torch.empty((R, T, k_out), device=dev, dtype=torch.bool)
    soft = torch.empty((R, T, k_out), device=dev, dtype=torch.float32)
    # 逐行独立的 (r,t) 归约（mask/lse/sort/topk）：按 t 分块与整幅计算的
    # 每行值逐位一致（同 kernel、同归约顺序），中间张量按 t 块封顶。
    t_chunk = max(1, min(T, (256 << 20) // max(R * Bm * 32, 1)))
    for t0 in range(0, T, t_chunk):
        t1 = min(t0 + t_chunk, T)
        causal_c = t_idx[:, t0:t1] > lt64[:, None, :]
        scores = torch.zeros((R, t1 - t0, Bm), device=dev, dtype=torch.float32)
        for i in range(nIH):
            # mul/relu 原地（s_i 是本轮新出的临时量，relu(x)>0 ⟺ x>0 逐位一致）；
            # add_ 与 += 同一个核。
            s_i = torch.einsum('rth,rbh->rtb', qIf[:, t0:t1, i, :], kIf[:, :, i, :])
            if dtype != torch.float32:
                # eager 的逐头舍入结构：einsum 输出按 dtype 舍入一次，mul_ 再
                # 舍入一次（bf16/fp16 张量乘 python 标量按 fp32 计算后舍回），
                # 末段 einsum 才按 fp32 累加、输出舍入一次。低精度下漏掉这两次
                # 中间舍入会翻转近 tie 的选择次序。
                s_i = s_i.to(dtype)
                s_i = s_i * iscale
            else:
                s_i *= iscale
            F.relu_(s_i)
            # wf 必须与 s_i 切到同一个 t 分块：s_i 是 [R, t1-t0, Bm] 的块内
            # 临时量，漏掉 t0:t1 切片时 [R, T, 1] 的 wf 在分块域（R·Bm 大、
            # t_chunk < T）直接广播失败，整块域两者是同一批行。
            scores.add_(s_i * wf[:, t0:t1, i:i + 1])
        scores = scores.to(dtype)
        masked_lse = torch.where(causal_c, scores, torch.full_like(scores, _NEG_INF))
        lse = torch.logsumexp(masked_lse, dim=-1)
        cf = causal_c & torch.isfinite(scores)
        work = torch.where(cf, scores, torch.full_like(scores, _NEG_INF))
        # pad 恒为块 0（eager 形式；池化 last_tok 单调时与「首个因果块」
        # 同值——那种形式的首个因果块必为块 0）。
        pad_blk = torch.zeros((R, t1 - t0), device=dev, dtype=torch.int64)
        # 逐轮「首达最大值」提取 == 稳定降序排序的前缀（tie 取最早下标，
        # 见上文 docstring）：一次 torch.sort 替代 k 轮全宽 max/where/min 循环。
        sval, sidx = torch.sort(work, dim=-1, descending=True, stable=True)
        top_val = sval[:, :, :k_out]
        top_idx = sidx[:, :, :k_out]
        # 与 eager 的 keep = cf.gather(...)（cf = causal & isfinite）同口径：
        # 有限的极端负分是有效选择，只有 -inf 填充位无效。
        has = top_val > float('-inf')
        valid[:, t0:t1] = has
        idx[:, t0:t1] = torch.where(has, top_idx, pad_blk[:, :, None]).to(torch.int32)
        sv = top_val - lse[:, :, None]
        sv = torch.where(has & torch.isfinite(lse)[:, :, None],
                         torch.exp(sv), torch.zeros_like(sv))
        soft[:, t0:t1] = torch.where(sv == sv, sv, torch.zeros_like(sv))
    return idx, valid, soft.to(dtype)


def _indexer_bwd_ref(gs, idx, valid, soft, qI, kI, w_idx, last_tok,
                     need_dq, need_dk, need_dw, t_chunk=128):
    """索引反向（两后端共用；按 t 分块，块内 einsum，固定顺序累加）。

    选择本身不可微；梯度只经 soft 回传：
      soft_j = exp(s_{idx_j} − lse)，lse = logsumexp(where(causal, s, −inf))
      ⇒ dL/ds_b = Σ_{j: valid_j ∧ idx_j=b} gs_j·soft_j − e_b·Gt，
         e_b = exp(masked_b − lse)，Gt = Σ_j gs_j·soft_j
    （e 的掩码与 lse 同基：仅因果；无因果行 / 非有限 lse 行的梯度为 0，
    与 eager 中 nan_to_num 的零通带一致。有效选择彼此互不相同，因此
    scatter 不存在同址碰撞，值与写入顺序无关。）
    再经 scores = Σ_i relu(iscale·qI·kI)·w 链到 qI / kI / w：
      g = ds·w_i·[s_i>0]·iscale；dqI_i = g @ kI_i；dkI_i = g^T @ qI_i；
      dw_i = Σ_b ds·relu(iscale·qI·kI)。
    """
    R, T, nIH, hd = qI.shape
    Bm = kI.shape[1]
    iscale = hd ** -0.5
    dev = qI.device
    dtype = qI.dtype
    t_idx = torch.arange(T, device=dev)[None, :, None]
    # 同 _indexer_fwd_ref：causal 按 t 分块现算（同一整数比较），全幅
    # [R, T, Bm] 掩码不物化。
    lt64 = last_tok.to(torch.int64)
    gsf = gs.to(torch.float32)
    softf = soft.to(torch.float32)
    Gt = (gsf * softf).sum(dim=-1)
    if idx.numel():
        # 一次升型供全循环复用（同一批整数）：pad 槽位（重复的 pad_blk）投到
        # 哑列 Bm——scatter_ 对重复下标的胜者未定义，但哑列上全部候选值都是
        # 0，且有效槽位彼此互不相同——切片后即精确。
        idx_l = idx.to(torch.int64)
    dqI = torch.zeros_like(qI) if need_dq else None
    dkI = torch.zeros_like(kI) if need_dk else None
    dw = torch.zeros_like(w_idx) if need_dw else None
    # kI / w_idx 与 t 分块无关：fp32 升型在分块循环外只做一次（整型内容
    # 不变，逐位一致），消掉每个 t-chunk × 每头的重复拷贝。
    kIf = kI if kI.dtype == torch.float32 else kI.to(torch.float32)
    wf = w_idx if w_idx.dtype == torch.float32 else w_idx.to(torch.float32)
    for t0 in range(0, T, t_chunk):
        t1 = min(t0 + t_chunk, T)
        # Mk 随 t 分块构建（全幅 [R, T, Bm+1] 不物化）：逐 chunk 的 scatter
        # 键/值就是全幅形式对应行的切片，逐行无碰撞——切片后的值逐位一致。
        Mk = torch.zeros((R, t1 - t0, Bm + 1), device=dev, dtype=torch.float32)
        if idx.numel():
            dst = torch.where(valid[:, t0:t1], idx_l[:, t0:t1], Bm)
            Mk.scatter_(2, dst, (gsf[:, t0:t1] * softf[:, t0:t1]) * valid[:, t0:t1].to(torch.float32))
        Mk = Mk[:, :, :Bm]
        q_c = qI[:, t0:t1].to(torch.float32)
        s_list = []
        for i in range(nIH):
            # 原地 relu：后面用到的 (s_i > 0) 与 relu(s_i)>0 逐位同真，
            # dw 用的 relu(s_i) 即本张量自身——不再保留 relu 前的副本。
            s_i = torch.einsum('rth,rbh->rtb', q_c[:, :, i, :], kIf[:, :, i, :])
            if dtype != torch.float32:
                # 与前向（_indexer_fwd_ref）同一逐头舍入结构：反向的 relu
                # 掩码与下方 softmax 归一化必须作用在前向实际使用的舍入
                # 分数上，梯度才与（舍入后的）前向函数共轭。
                s_i = s_i.to(dtype)
                s_i = s_i * iscale
            else:
                s_i *= iscale
            s_list.append(F.relu_(s_i))
        # fp32 累加器（与前向的 scores 容器同口径——即使 s_i 已按 dtype
        # 舍入，跨头累加也在 fp32 进行）。
        s_all = torch.zeros((R, t1 - t0, Bm), device=dev, dtype=torch.float32)
        for i in range(nIH):
            s_all.add_(s_list[i] * wf[:, t0:t1, i:i + 1])
        if dtype != torch.float32:
            # 前向末段 einsum 的输出舍入：lse/e 必须从前向实际选择的
            # 同一批舍入分数重算。
            s_all = s_all.to(dtype)
        causal_c = t_idx[:, t0:t1] > lt64[:, None, :]
        masked = torch.where(causal_c, s_all, torch.full_like(s_all, _NEG_INF))
        lse = torch.logsumexp(masked, dim=-1)
        row_ok = torch.isfinite(lse)
        e = torch.where(row_ok[:, :, None], torch.exp(masked - lse[:, :, None]), torch.zeros_like(masked))
        ds = Mk - e * Gt[:, t0:t1, None]
        ds = torch.where(causal_c & torch.isfinite(s_all), ds, torch.zeros_like(ds))
        for i in range(nIH):
            s_i = s_list[i]
            g = ds * wf[:, t0:t1, i:i + 1] * (s_i > 0) * iscale
            if need_dq:
                dqI[:, t0:t1, i, :] += torch.einsum('rtb,rbh->rth', g, kIf[:, :, i, :]).to(dtype)
            if need_dk:
                dkI[:, :, i, :] += torch.einsum('rtb,rth->rbh', g, q_c[:, :, i, :]).to(dtype)
            if need_dw:
                # s_i 已在上方原地 relu（relu(relu(x)) 与 relu(x) 逐位一致）。
                dw[:, t0:t1, i] += (ds * s_i).sum(dim=-1).to(dtype)
    return dqI, dkI, dw


class FusedIndexer(torch.autograd.Function):
    """索引打分 + top-k 选择的可微封装：idx/valid 不可微（离散选择），
    soft 的梯度经 _indexer_bwd_ref 回传到 qI / kI / w。"""

    @staticmethod
    def forward(ctx, impl, qI, kI, w_idx, last_tok, topk):
        if impl == 'triton':
            idx, valid, soft = _indexer_fwd_triton(qI, kI, w_idx, last_tok, topk)
        else:
            idx, valid, soft = _indexer_fwd_ref(qI, kI, w_idx, last_tok, topk)
        ctx.save_for_backward(qI, kI, w_idx, last_tok, idx, valid, soft)
        ctx.topk = int(topk)
        return idx, valid, soft

    @staticmethod
    def backward(ctx, gidx, gvalid, gs):
        qI, kI, w_idx, last_tok, idx, valid, soft = ctx.saved_tensors
        if gs is None:
            return (None,) * 6
        ng = ctx.needs_input_grad
        dqI, dkI, dw = _indexer_bwd_ref(
            gs, idx, valid, soft, qI, kI, w_idx, last_tok,
            need_dq=bool(ng[1]), need_dk=bool(ng[2]), need_dw=bool(ng[3]),
            t_chunk=int(_env_cached('CSA_KERNEL_IDX_CHUNK') or 128))
        return (None, dqI, dkI, dw, None, None)


def lightning_indexer_kernel(qI, kI, w_idx, last_tok, topk, impl=None):
    """`exp_lib.lightning_indexer`（learned 模式、return_mask=False）的内核
    路径。    qI [R,T,nIH,hd]、kI [R,Bm,nIH,hd]、w_idx [R,T,nIH]、
    last_tok [R,Bm]。返回 (idx, valid, soft)。"""
    impl = impl or _op_impl('IDX')
    return FusedIndexer.apply(impl, qI, kI, w_idx, last_tok, int(topk))


# ============================================================================
# 算子 3：稀疏注意力（`gathered_attention` 的内核路径）
# ============================================================================

_ATTN_REF_BUDGET = 64 << 20


def _attn_t_chunk(R, M, nh, hd):
    per_t = max(1, int(R) * int(M) * int(nh) * int(hd) * 4)
    return max(1, min(int(_ATTN_REF_BUDGET) // per_t, 256))


def _attn_fwd_ref(q, k_blk, v_blk, k_sw, v_sw, topk_idx, valid, last_tok, soft, sink, w, scale):
    """torch 参考实现（按 t 分块 × 逐头，内存有界）。

    泳道布局与 eager 一致：[top-k 块泳道 | 滑窗泳道]，共 M = nb + w 条。
    前向值 = **A 路径**（plain softmax 注意力）——eager 的直通估计器
    `attn = A + (S − S.detach())` 在前向值上严格等于 A；S（soft 先验）
    路径只进入梯度（见 _attn_bwd_ref）。sink 经 [sink | lanes] 拼接 softmax
    后丢弃 sink 列：P_l = e_l / (e_sink + Σ e)。
    无 sink 且全泳道无效的行：softmax 在全 MINL 上给出均匀分布，eager
    以 anyrow 乘法清零——本实现以同样的 anyrow 语义清零。
    """
    R, T, nh, hd = q.shape
    nb = topk_idx.shape[-1]
    M = nb + w
    dev = q.device
    dtype = q.dtype
    MINL = _finfo_min(dtype)
    out = torch.empty_like(q)
    t_idx = torch.arange(T, device=dev)[None, :, None]
    if valid is None:
        valid = torch.ones_like(topk_idx, dtype=torch.bool)
    topk_l = topk_idx.to(torch.int64)
    win_raw = (t_idx - (w - 1) + torch.arange(w, device=dev)[None, None, :]).expand(R, T, w)
    sink_v = torch.nan_to_num(sink).to(dtype) if sink is not None else None
    tc = _attn_t_chunk(R, M, nh, hd)
    # 头在外、t-chunk 在内：各 (t,h) 的前向计算相互独立，循环顺序交换
    # 不改变任何元素值。chunk 内的 src_sel/lt_sel/sel_ok/src_win/win_ok/
    # anyrow 全部与 h 无关——在头循环外按 chunk 各算一次（同值）。
    lt_l = last_tok.to(torch.int64)
    chunks = []
    for t0 in range(0, T, tc):
        t1 = min(t0 + tc, T)
        src_sel = topk_l[:, t0:t1]
        lt_sel = lt_l.gather(1, src_sel.reshape(R, -1)).view(R, t1 - t0, nb)
        sel_ok = valid[:, t0:t1] & (t_idx[:, t0:t1, :] > lt_sel)
        src_win = win_raw[:, t0:t1, :].clamp(min=0)
        win_ok = win_raw[:, t0:t1, :] >= 0
        lane_ok = torch.cat([sel_ok, win_ok], dim=-1)
        anyrow = (sel_ok.any(dim=-1) | win_ok.any(dim=-1)) if sink_v is None else None
        chunks.append((t0, t1, src_sel, src_win, lane_ok, anyrow))
    for h in range(nh):
        q_all = q[:, :, h, :].to(torch.float32)
        kb_all = k_blk[:, :, h, :].to(torch.float32)
        kw_all = k_sw[:, :, h, :].to(torch.float32)
        vb_all = v_blk[:, :, h, :].to(torch.float32)
        vw_all = v_sw[:, :, h, :].to(torch.float32)
        for (t0, t1, src_sel, src_win, lane_ok, anyrow) in chunks:
            qh = q_all[:, t0:t1]
            Kb = _gather_rows(kb_all, src_sel)
            Kw = _gather_rows(kw_all, src_win)
            Vb = _gather_rows(vb_all, src_sel)
            Vw = _gather_rows(vw_all, src_win)
            zA = torch.cat([torch.einsum('rtd,rtmd->rtm', qh, Kb), torch.einsum('rtd,rtmd->rtm', qh, Kw)], dim=-1) * scale
            zA.masked_fill_(~lane_ok, MINL)
            if sink_v is None:
                P_A = torch.softmax(zA, dim=-1) * (anyrow[:, :, None] if anyrow is not None else 1.0)
            else:
                zext = torch.cat([sink_v[h].expand(R, t1 - t0, 1), zA], dim=-1)
                P_A = torch.softmax(zext, dim=-1)[..., 1:]
            Vl = torch.cat([Vb, Vw], dim=-2)
            out[:, t0:t1, h, :] = torch.einsum('rtm,rtmd->rtd', P_A, Vl).to(dtype)
    return out


def _attn_bwd_ref(q, k_blk, v_blk, k_sw, v_sw, topk_idx, valid, last_tok, soft, sink, w, scale, dO,
                  need_dq, need_dkb, need_dvb, need_dksw, need_dvsw, need_dsoft, need_dsink):
    """torch 参考反向（与 eager 的 autograd 图逐步对应；确定性——
    dk_blk / dk_sw / dv_blk / dv_sw 的跨目的地累加全部经稳定排序段和）。

    直通估计器的梯度分解（前向值 = A，梯度 = A 路 + S 路之和）：
      g_l         = ⟨dO, V_l⟩                （两条路径共享同一个 g）
      ĝ_l         = g_l · anyrow（无 sink 时；有 sink 时 = g_l）
      dzA_l       = P_A_l · (ĝ_l − GA)，GA = Σ_j P_A_j ĝ_j
      dzS_l       = P_S_l · (ĝ_l − GS)，GS = Σ_j P_S_j ĝ_j
      dz_l        = dzA_l + dzS_l
      dq          = Σ_l dz_l · K_l · scale
      dk_l        = dz_l · q · scale
      dV_l        = (P_A_l · anyrow) · dO   （前向值只经 A，见上）
      dλ_l        = dzS_l（块泳道）；dsoft_l = dλ_l · [soft ≥ 1e-12] / soft
      dsink_h     = −Σ_{r,t} (P_sink^A·GA + P_sink^S·GS)
    """
    R, T, nh, hd = q.shape
    nb = topk_idx.shape[-1]
    Bm = k_blk.shape[1]
    dev = q.device
    dtype = q.dtype
    MINL = _finfo_min(dtype)
    M = nb + w
    t_idx = torch.arange(T, device=dev)[None, :, None]
    if valid is None:
        valid = torch.ones_like(topk_idx, dtype=torch.bool)
    topk_l = topk_idx.to(torch.int64)
    win_raw = (t_idx - (w - 1) + torch.arange(w, device=dev)[None, None, :]).expand(R, T, w)
    sink_v = torch.nan_to_num(sink).to(dtype) if sink is not None else None
    dq = torch.zeros_like(q) if need_dq else None
    dsoft = torch.zeros_like(soft) if (need_dsoft and soft is not None) else None
    dsink = torch.zeros_like(sink) if (need_dsink and sink is not None) else None
    r_idx = torch.arange(R, device=dev)[:, None, None]
    tc = _attn_t_chunk(R, M, nh, hd)
    # 头在外、t-chunk 在内：各 (t,h) 独立。chunk 内的几何/掩码/soft 常量
    # 与 h 无关——头循环外按 chunk 各算一次（同值；soft_l[:, :, :nb] 同时
    # 兼作逐头重置的 sf cast）。
    # 键值梯度碎片按头收集、逐头归约：键含头维的部分（+h）跨头只差常量
    # 偏移，稳定排序的排列与段边界跨头完全相同（见 _seg_structure），故
    # 段结构按不含头维的 base 键预建一次、全部头共用；头内条目保持
    # chunk 升序，组内累加顺序与逐头排序 flush 逐元素一致。
    dk_blk32 = torch.zeros((R * Bm * nh, hd), device=dev, dtype=torch.float32) if need_dkb else None
    dv_blk32 = torch.zeros((R * Bm * nh, hd), device=dev, dtype=torch.float32) if need_dvb else None
    dk_sw32 = torch.zeros((R * T * nh, hd), device=dev, dtype=torch.float32) if need_dksw else None
    dv_sw32 = torch.zeros((R * T * nh, hd), device=dev, dtype=torch.float32) if need_dvsw else None
    # dk/dv 共享同一组键（同 (chunk,h) 构造的 keyb），dksw/dvsw 共享 keyw。
    dk_vals, dv_vals = [], []
    dksw_vals, dvsw_vals = [], []
    lt_l = last_tok.to(torch.int64)
    chunks = []
    for t0 in range(0, T, tc):
        t1 = min(t0 + tc, T)
        src_sel = topk_l[:, t0:t1]
        lt_sel = lt_l.gather(1, src_sel.reshape(R, -1)).view(R, t1 - t0, nb)
        sel_ok = valid[:, t0:t1] & (t_idx[:, t0:t1, :] > lt_sel)
        src_win = win_raw[:, t0:t1, :].clamp(min=0)
        win_ok = win_raw[:, t0:t1, :] >= 0
        lane_ok = torch.cat([sel_ok, win_ok], dim=-1)
        anyrow = (sel_ok.any(dim=-1) | win_ok.any(dim=-1)) if sink_v is None else None
        soft_l = None
        if soft is not None:
            soft_l = torch.zeros((R, t1 - t0, nb + w), device=dev, dtype=torch.float32)
            soft_l[:, :, :nb] = soft[:, t0:t1, :].to(torch.float32)
        # 键基按需（int 精确）：不含头维的 base 键——(r,src)/窗口部分与
        # h 无关，头维在归约落行时才以 seg_id·nh + h 加回（_seg_apply）。
        keyb = r_idx.expand(R, t1 - t0, nb) * Bm + src_sel if (need_dkb or need_dvb) else None
        keyw = r_idx.expand(R, t1 - t0, w) * T + src_win if (need_dksw or need_dvsw) else None
        chunks.append((t0, t1, src_sel, src_win, lane_ok, soft_l, anyrow, keyb, keyw))
    seg_b = _seg_structure(torch.cat([_c[7].reshape(-1) for _c in chunks]), R * Bm) if (need_dkb or need_dvb) else None
    seg_w = _seg_structure(torch.cat([_c[8].reshape(-1) for _c in chunks]), R * T) if (need_dksw or need_dvsw) else None

    def _flush(h, vals_lists, bufs, seg):
        # 逐头归约后清空碎片列表，释放本轮引用的全部 chunk 碎片。
        # dk/dv（或 dksw/dvsw）键逐元素相同，共享同一次排序段和；bufs 中
        # 的 None（未请求的梯度）跳过。
        for vl, buf in zip(vals_lists, bufs):
            if vl and buf is not None:
                _seg_apply(seg, torch.cat(vl), buf, h, nh)
        for vl in vals_lists:
            vl.clear()

    for h in range(nh):
        q_all = q[:, :, h, :].to(torch.float32)
        dO_all = dO[:, :, h, :].to(torch.float32)
        kb_all = k_blk[:, :, h, :].to(torch.float32)
        kw_all = k_sw[:, :, h, :].to(torch.float32)
        vb_all = v_blk[:, :, h, :].to(torch.float32)
        vw_all = v_sw[:, :, h, :].to(torch.float32)
        for (t0, t1, src_sel, src_win, lane_ok, soft_l, anyrow, keyb, keyw) in chunks:
            qh = q_all[:, t0:t1]
            dOh = dO_all[:, t0:t1]
            Kb = _gather_rows(kb_all, src_sel)
            Kw = _gather_rows(kw_all, src_win)
            Vb = _gather_rows(vb_all, src_sel)
            Vw = _gather_rows(vw_all, src_win)
            Kl = torch.cat([Kb, Kw], dim=-2)
            Vl = torch.cat([Vb, Vw], dim=-2)
            zA = torch.cat([torch.einsum('rtd,rtmd->rtm', qh, Kb), torch.einsum('rtd,rtmd->rtm', qh, Kw)], dim=-1) * scale
            zA.masked_fill_(~lane_ok, MINL)
            g = torch.einsum('rtd,rtmd->rtm', dOh, Vl)
            gh = g * (anyrow[:, :, None] if anyrow is not None else 1.0)
            P_S = GS = dzS = P_sinkS = P_sinkA = None
            if soft_l is not None:
                # S 路径（soft 先验）与 sink 无关——eager 的直通估计器在
                # use_sink=False 时同样有 S 路梯度。
                lam = torch.log(soft_l.clamp_min(1e-12))
                lam = lam.masked_fill(soft_l <= 0, MINL)
                zS = zA.clone()
                zS[:, :, :nb] = (zA[:, :, :nb] + lam[:, :, :nb]).clamp_min(MINL)
            if sink_v is None:
                P_A = torch.softmax(zA, dim=-1)
                P_A_f = P_A * (anyrow[:, :, None] if anyrow is not None else 1.0)
                GA = (P_A * gh).sum(dim=-1)
                dzA = P_A * (gh - GA[:, :, None])
                if soft_l is not None:
                    P_S = torch.softmax(zS, dim=-1)
                    GS = (P_S * gh).sum(dim=-1)
                    dzS = P_S * (gh - GS[:, :, None])
            else:
                zext = torch.cat([sink_v[h].expand(R, t1 - t0, 1), zA], dim=-1)
                P_full = torch.softmax(zext, dim=-1)
                P_A = P_full[..., 1:]
                P_sinkA = P_full[..., 0]
                P_A_f = P_A
                GA = (P_A * gh).sum(dim=-1)
                dzA = P_A * (gh - GA[:, :, None])
                if soft_l is not None:
                    zextS = torch.cat([sink_v[h].expand(R, t1 - t0, 1), zS], dim=-1)
                    P_fullS = torch.softmax(zextS, dim=-1)
                    P_S = P_fullS[..., 1:]
                    P_sinkS = P_fullS[..., 0]
                    GS = (P_S * gh).sum(dim=-1)
                    dzS = P_S * (gh - GS[:, :, None])
                if dsink is not None:
                    acc = (-(P_sinkA * GA)).sum(dim=(0, 1))
                    if GS is not None:
                        acc = acc + (-(P_sinkS * GS)).sum(dim=(0, 1))
                    dsink[h] += acc.to(dtype)
            dz = dzA if dzS is None else dzA + dzS
            if dq is not None:
                dq[:, t0:t1, h, :] += torch.einsum('rtm,rtmd->rtd', dz, Kl).mul(scale).to(dtype)
            if dsoft is not None and dzS is not None:
                dl = dzS[:, :, :nb]
                sf = soft_l[:, :, :nb]
                dsoft[:, t0:t1, :] += torch.where(sf >= 1e-12, dl / sf.clamp_min(1e-12), torch.zeros_like(dl)).to(dtype)
            if need_dkb or need_dksw:
                dkl = torch.einsum('rtm,rtd->rtmd', dz, qh) * scale
                if need_dkb:
                    dk_vals.append(dkl[:, :, :nb].reshape(-1, hd))
                if need_dksw:
                    dksw_vals.append(dkl[:, :, nb:].reshape(-1, hd))
            if need_dvb or need_dvsw:
                dvl = torch.einsum('rtm,rtd->rtmd', P_A_f, dOh)
                if need_dvb:
                    dv_vals.append(dvl[:, :, :nb].reshape(-1, hd))
                if need_dvsw:
                    dvsw_vals.append(dvl[:, :, nb:].reshape(-1, hd))
        _flush(h, [dk_vals, dv_vals], [dk_blk32, dv_blk32], seg_b)
        _flush(h, [dksw_vals, dvsw_vals], [dk_sw32, dv_sw32], seg_w)
    dk_blk = dk_blk32.to(dtype).view(R, Bm, nh, hd) if need_dkb else None
    dk_sw = dk_sw32.to(dtype).view(R, T, nh, hd) if need_dksw else None
    dv_blk = dv_blk32.to(dtype).view(R, Bm, nh, hd) if need_dvb else None
    dv_sw = dv_sw32.to(dtype).view(R, T, nh, hd) if need_dvsw else None
    return (dq, dk_blk, dv_blk, dk_sw, dv_sw, dsoft, dsink)


# ============================================================================
# 算子 3 的可微封装 + 公共入口
# ============================================================================

class FusedSparseAttention(torch.autograd.Function):
    """稀疏注意力的可微封装：前向 = A 路径（与 eager 的直通估计器前向值
    严格一致）；反向按 _attn_bwd_ref 的 A+S 双路分解（triton 后端为同公式
    的四内核版：统计 / dq / dk-dv / dsoft）。"""

    @staticmethod
    def forward(ctx, impl, q, k_blk, v_blk, k_sw, v_sw, topk_idx, valid, last_tok, soft, sink, w, scale):
        if impl == 'triton':
            out = _attn_fwd_triton(q, k_blk, v_blk, k_sw, v_sw, topk_idx, valid, last_tok, soft, sink, w, scale)
        else:
            out = _attn_fwd_ref(q, k_blk, v_blk, k_sw, v_sw, topk_idx, valid, last_tok, soft, sink, w, scale)
        ctx.save_for_backward(q, k_blk, v_blk, k_sw, v_sw, topk_idx, valid, last_tok, soft, sink)
        # 反向必须跑与前向同一个后端：前向拿到的是调用方解析好的 impl
        # （gathered_attention_kernel 经 _op_impl('ATTN')），反向若在环境
        # 变量上重新解析，显式 impl= 的调用（bench/selfcheck）就会 silently
        # 混成 triton 前向 + torch 参考反向。
        ctx.impl = impl
        ctx.w = int(w)
        ctx.scale = float(scale)
        return out

    @staticmethod
    def backward(ctx, dO):
        q, k_blk, v_blk, k_sw, v_sw, topk_idx, valid, last_tok, soft, sink = ctx.saved_tensors
        if dO is None:
            return (None,) * 13
        dO = dO.contiguous()
        impl = ctx.impl
        if impl == 'triton':
            grads = _attn_bwd_triton(q, k_blk, v_blk, k_sw, v_sw, topk_idx, valid, last_tok, soft, sink,
                                     ctx.w, ctx.scale, dO, ctx.needs_input_grad)
        else:
            ng = ctx.needs_input_grad
            grads = _attn_bwd_ref(q, k_blk, v_blk, k_sw, v_sw, topk_idx, valid, last_tok, soft, sink,
                                  ctx.w, ctx.scale, dO,
                                  bool(ng[1]), bool(ng[2]), bool(ng[3]), bool(ng[4]), bool(ng[5]),
                                  bool(ng[9]), bool(ng[10]))
        dq, dk_blk, dv_blk, dk_sw, dv_sw, dsoft, dsink = grads
        return (None, dq, dk_blk, dv_blk, dk_sw, dv_sw, None, None, None, dsoft, dsink, None, None)


def gathered_attention_kernel(q, k_blk, v_blk, topk_idx, last_tok, k_sw, v_sw, w, scale,
                              soft=None, sink_logits=None, sel_valid=None, impl=None):
    """`exp_lib.gathered_attention` 的内核路径（批量化）。

    q [R,T,nh,hd]、k_blk/v_blk [R,Bm,nh,hd]、k_sw/v_sw [R,T,nh,hd]、
    topk_idx [R,T,nb] int、sel_valid [R,T,nb] bool（None = 全有效，HCA）、
    soft [R,T,nb]（None = 无先验，HCA）、sink_logits [nh]。

    前置约定：每行 topk_idx 的块 id 互不相同（无放回选择——索引器出口
    与 HCA 的 arange 展开恒满足）。eager 的 gathered_attention 默认做
    first-occurrence 去重，本路径不做；含重复块 id 的行会被重复计数，
    与 eager 产生分歧——直接调用时必须先自行去重。
    """
    impl = impl or _op_impl('ATTN')
    return FusedSparseAttention.apply(impl, q, k_blk, v_blk, k_sw, v_sw, topk_idx,
                                      sel_valid, last_tok, soft, sink_logits, int(w), float(scale))


def _op_impl(op):
    """逐算子的后端覆盖：CSA_KERNEL_POOL / _IDX / _ATTN ∈ {'', 'triton', 'ref'}
    （空 = 跟随全局 CSA_KERNEL）。用于 GPU 上的分算子排查。"""
    forced = (_env_cached(f'CSA_KERNEL_{op}') or '').strip().lower()
    if forced in ('triton', 'ref'):
        return forced
    return kernel_backend() or 'ref'


# ============================================================================
# Triton 内核（全部惰性构建：模块导入零副作用，CPU 机器不触碰 triton）
# ============================================================================

_KERNELS = None


def _dot_prec():
    """tl.dot 的输入精度：默认 ieee（逐位口径的 fp32 FMA——eager 的 fp32
    einsum 也是 ieee，默认必须保证 kernel/eager 是同一函数，否则 indexer
    的近 tie 选择会被 ~1e-3 的 tf32 噪声翻转、且同一 softmax 内滑窗泳道
    (tl.dot) 与块泳道 (tl.sum) 精度不对称）；CSA_KERNEL_TF32=1 时显式
    切换到 tf32（与 eager 的差异属容差口径）。
    bf16 输入不受此开关影响（张量核对 bf16 输入精确、fp32 累加）。"""
    return 'tf32' if (_env_cached('CSA_KERNEL_TF32') or '0').strip().lower() not in ('0', 'false', 'off') else 'ieee'


def _kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    import types
    import triton
    import triton.language as tl

    ns = types.SimpleNamespace(triton=triton, tl=tl)

    # ---------- 算子 1：pool 前向 ----------
    @triton.jit
    def pool_fwd(Xa_ptr, Xb_ptr, Za_ptr, Zb_ptr, Bpa_ptr, Bpb_ptr,
                 starts_ptr, counts_ptr, ov_start_ptr, ov_len_ptr, starts_prev_ptr,
                 comp_ptr, T, C, Bm, BPA_ROWS, BPB_ROWS,
                 BLOCK_C: tl.constexpr, BLOCK_L: tl.constexpr, BLOCK_W: tl.constexpr, HAS_B: tl.constexpr):
        pid = tl.program_id(0)
        nc = tl.cdiv(C, BLOCK_C)
        pid_c = pid % nc
        rb = pid // nc
        b = rb % Bm
        r = rb // Bm
        cs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        cm = cs < C
        st = tl.load(starts_ptr + r * Bm + b).to(tl.int64)
        ct = tl.load(counts_ptr + r * Bm + b).to(tl.int64)
        jA = tl.arange(0, BLOCK_L).to(tl.int64)
        a_ok = jA < ct
        posA = tl.minimum(jA, BPA_ROWS - 1)
        gA = tl.minimum(st + jA, T - 1)
        za = tl.load(Za_ptr + (r * T + gA[:, None]) * C + cs[None, :], mask=a_ok[:, None] & cm[None, :], other=0.0).to(tl.float32)
        bpa = tl.load(Bpa_ptr + posA[:, None] * C + cs[None, :], mask=cm[None, :], other=0.0).to(tl.float32)
        la = tl.where(a_ok[:, None], za + bpa, float('-inf'))
        m = tl.max(la, 0)
        if HAS_B:
            ovs = tl.load(ov_start_ptr + r * Bm + b).to(tl.int64)
            ovl = tl.load(ov_len_ptr + r * Bm + b).to(tl.int64)
            spv = tl.load(starts_prev_ptr + r * Bm + b).to(tl.int64)
            jB = tl.arange(0, BLOCK_W).to(tl.int64)
            b_ok = jB < ovl
            boff = tl.minimum(ovs - spv + jB, BPB_ROWS - 1)
            gB = tl.minimum(ovs + jB, T - 1)
            zb = tl.load(Zb_ptr + (r * T + gB[:, None]) * C + cs[None, :], mask=b_ok[:, None] & cm[None, :], other=0.0).to(tl.float32)
            bpb = tl.load(Bpb_ptr + boff[:, None] * C + cs[None, :], mask=cm[None, :], other=0.0).to(tl.float32)
            lb = tl.where(b_ok[:, None], zb + bpb, float('-inf'))
            m = tl.maximum(m, tl.max(lb, 0))
        # 全掩泳道兜底取 0（ea=exp(-inf-0)=0、Z=0 走 Zs 守卫）；有限极值
        # 不钳位（见 idx_topk 的同口径注释）。
        m = tl.where(m == float('-inf'), 0.0, m)
        ea = tl.exp(la - m[None, :])
        Z = tl.sum(ea, 0)
        if HAS_B:
            eb = tl.exp(lb - m[None, :])
            Z += tl.sum(eb, 0)
        Zs = tl.where(Z == 0.0, 1.0, Z)
        xa = tl.load(Xa_ptr + (r * T + gA[:, None]) * C + cs[None, :], mask=a_ok[:, None] & cm[None, :], other=0.0).to(tl.float32)
        acc = tl.sum(ea * xa, 0) / Zs
        if HAS_B:
            xb = tl.load(Xb_ptr + (r * T + gB[:, None]) * C + cs[None, :], mask=b_ok[:, None] & cm[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(eb * xb, 0) / Zs
        # store 基址升 int64：(r·Bm + b)·C 在 R·Bm·C ≥ 2³¹ 的超大配置下按
        # int32 求值会溢出（gather 侧都经 gA/gB 的 int64 广播，只有这条
        # 输出链是纯 int32）。
        tl.store(comp_ptr + (r * Bm + b).to(tl.int64) * C + cs, acc.to(comp_ptr.dtype.element_ty), mask=cm)

    ns.pool_fwd = pool_fwd

    # ---------- 算子 2：索引打分 ----------
    # T0/TF：打分按 query 行分块启动（scores 缓冲封顶）；T0 是本块的全局行
    # 偏移，TF 是完整序列长（qI/w 的行 stride），scores/输出按块内 T 寻址。
    # T0=0、TF=T 时寻址与不分块形式完全一致（同一批全局行、同一批值）。
    # HDP：点积泳道宽（>= 16 的 2 幂）；hd < HDP 时 d 泳道零填充——零乘积
    # 向点积累加恰好贡献 +0.0，分数逐位不变，hd 不再要求是 2 的幂或 >= 16。
    @triton.jit
    def idx_score(qI_ptr, kI_ptr, w_ptr, scores_ptr, T0, TF, T, Bm, ISCALE,
                  N_IH: tl.constexpr, HD: tl.constexpr, HDP: tl.constexpr, BT: tl.constexpr, BB: tl.constexpr,
                  DOT: tl.constexpr):
        pid = tl.program_id(0)
        n_t = tl.cdiv(T, BT)
        pid_t = pid % n_t
        r = pid // n_t
        t = pid_t * BT + tl.arange(0, BT)
        tm = t < T
        ti = t.to(tl.int64)
        tg = (T0 + t).to(tl.int64)
        d = tl.arange(0, HDP)
        dm = d < HD
        for b0 in range(0, Bm, BB):
            bcol = b0 + tl.arange(0, BB)
            bm = bcol < Bm
            bi = bcol.to(tl.int64)
            acc = tl.zeros((BT, BB), dtype=tl.float32)
            for i in tl.static_range(N_IH):
                q_i = tl.load(qI_ptr + ((r * TF + tg[:, None]) * N_IH + i) * HD + d[None, :], mask=tm[:, None] & dm[None, :], other=0.0).to(tl.float32)
                w_i = tl.load(w_ptr + (r * TF + tg) * N_IH + i, mask=tm, other=0.0).to(tl.float32)
                k_t = tl.load(kI_ptr + ((r * Bm + bi[:, None]) * N_IH + i) * HD + d[None, :], mask=bm[:, None] & dm[None, :], other=0.0).to(tl.float32)
                s = tl.dot(q_i, tl.trans(k_t), input_precision=DOT)
                if qI_ptr.dtype.element_ty == tl.float32:
                    s = s * ISCALE
                else:
                    # 复现 eager 的逐头舍入结构（einsum 输出按输入 dtype
                    # 舍入一次、乘 iscale 再舍入一次）；低精度下漏掉这两次
                    # 中间舍入会翻转近 tie 的选择次序。ref 后端的同款结构
                    # 见 _indexer_fwd_ref。
                    s = s.to(qI_ptr.dtype.element_ty).to(tl.float32)
                    s = (s * ISCALE).to(qI_ptr.dtype.element_ty).to(tl.float32)
                acc += tl.where(s > 0.0, s, 0.0) * w_i[:, None]
            out_s = acc
            if qI_ptr.dtype.element_ty != tl.float32:
                # 末段 einsum 的输出舍入（fp32 累加后按 dtype 舍入一次），
                # 与 eager/ref 的选择输入逐位对齐。
                out_s = acc.to(qI_ptr.dtype.element_ty).to(tl.float32)
            tl.store(scores_ptr + (r * T + ti[:, None]) * Bm + bi[None, :], out_s, mask=tm[:, None] & bm[None, :])

    ns.idx_score = idx_score

    # ---------- 算子 2：top-k 提取（逐轮首达最大值；tie = 最早下标） ----------
    # T0/TF 与 idx_score 同义：因果比较用全局位置 T0+t，scores 按块内行、
    # idx/valid/soft 输出按全局行寻址。逐轮的软分直接取本轮最大值 m：has
    # 为真时 masked[pick] 与 m 是同一个 fp32 值（work 只在非因果/非有限
    # 泳道与 masked 不同，而那些泳道使 has 为假），has 为假时 sv 被紧随的
    # where/nan 守卫清零——两种情形都与旧的全宽 extract 归约同值。
    @triton.jit
    def idx_topk(scores_ptr, last_tok_ptr, idx_ptr, valid_ptr, soft_ptr, T0, TF, T, Bm,
                 K: tl.constexpr, BB: tl.constexpr):
        pid = tl.program_id(0)
        r = pid // T
        t = pid % T
        b = tl.arange(0, BB).to(tl.int64)
        bm = b < Bm
        rt = (r * T + t).to(tl.int64)
        tg = (T0 + t).to(tl.int64)
        rt_out = r * TF + tg
        s = tl.load(scores_ptr + rt * Bm + b, mask=bm, other=float('-inf')).to(tl.float32)
        lt = tl.load(last_tok_ptr + r * Bm + b, mask=bm, other=TF - 1).to(tl.int64)
        causal = (tg > lt) & bm
        finite = (s - s) == 0.0
        masked = tl.where(causal, s, float('-inf'))
        has_nan = tl.sum(tl.where(masked != masked, 1.0, 0.0), 0) > 0.0
        has_pinf = tl.sum(tl.where(masked == float('inf'), 1.0, 0.0), 0) > 0.0
        m0 = tl.max(masked, 0)
        # 全掩行兜底取 0（exp(-inf-0)=0、lse=-inf，与 eager 的全 -inf
        # logsumexp 同值）；有限行一律用真 max——钳位哨兵会在「全部有效
        # logit < 哨兵」时背离 eager 的归一化。
        m_safe = tl.where(m0 == float('-inf'), 0.0, m0)
        e = tl.exp(masked - m_safe)
        Ssum = tl.sum(e, 0)
        lse = m_safe + tl.log(Ssum)
        lse = tl.where(has_nan, float('nan'), tl.where(has_pinf, float('inf'), lse))
        work = tl.where(causal & finite, s, float('-inf'))
        BIG = Bm + 1
        # pad 恒为块 0（eager 形式；池化 last_tok 单调时与「首个因果块」
        # 同值——那种形式的首个因果块必为块 0）。
        pad = tl.zeros([], tl.int64)
        for j in tl.static_range(K):
            m = tl.max(work, 0)
            # work 的泳道非有限即 -inf：与 eager 的 cf = causal & isfinite
            # 同口径（有限的极端负分也是有效选择）。
            has = m > float('-inf')
            pick = tl.min(tl.where(work == m, b, BIG), 0)
            work = tl.where(has & (b == pick), float('-inf'), work)
            use = tl.where(has, pick, pad)
            sv = tl.exp(m - lse)
            sv = tl.where(has, sv, 0.0)
            sv = tl.where(sv == sv, sv, 0.0)
            tl.store(idx_ptr + rt_out * K + j, use.to(tl.int32))
            tl.store(valid_ptr + rt_out * K + j, has.to(tl.int8))
            tl.store(soft_ptr + rt_out * K + j, sv.to(soft_ptr.dtype.element_ty))

    ns.idx_topk = idx_topk

    _KERNELS = ns
    return ns


# ---------------------------------------------------------------------------
# 注意力 Triton 内核：前向（flash 式在线 softmax，sink 作初值质量）与
# 反向四内核（stats / dq / dk-dv / dsoft——dsoft 是跨头求和，(r,b,h) 网格
# 表达不了，独立成一个 (r, t 块) 网格、每泳道单写者、寄存器定序累加的
# 内核）。块泳道按 (t, j) 逐条 gather（选择
# 逐行不同，无法共享 K tile）；滑窗泳道以共享 K_sw tile 的标准带状 flash
# 循环处理。全部累加为固定顺序（无原子加）。
#
# 两个化简（值等价，见模块 docstring 的语义对齐点）：
# - anyrow：w ≥ 1 时每个 t 都有对角滑窗泳道（src = t, j = w-1），因此
#   eager 的 anyrow 恒为真——内核直接按 1 处理（launcher 断言 w ≥ 1）；
# - 前向值 = A 路径；S 路径（soft 先验）只出现在反向的 jacobian 里。
# ---------------------------------------------------------------------------

_MINL32 = -3.402823466e38


def _attn_fwd_triton(q, k_blk, v_blk, k_sw, v_sw, topk_idx, valid, last_tok, soft, sink, w, scale):
    K = _kernels()
    if getattr(K, 'attn_fwd', None) is None:
        _attach_attn_kernels(K)
    triton = K.triton
    R, T, nh, hd = q.shape
    nb = topk_idx.shape[-1]
    Bm = k_blk.shape[1]
    assert int(w) >= 1 and hd >= 16 and hd & (hd - 1) == 0, 'triton attention 前向要求 w >= 1 且 hd 为 2 的幂且 >= 16（其余形状走 ref 后端）'
    q = q.contiguous()
    k_blk, v_blk = k_blk.contiguous(), v_blk.contiguous()
    k_sw, v_sw = k_sw.contiguous(), v_sw.contiguous()
    topk_idx = topk_idx.contiguous()
    valid_c = valid.contiguous() if valid is not None else topk_idx
    last_tok = last_tok.contiguous()
    soft_c = soft.contiguous() if soft is not None else topk_idx
    sink_c = torch.nan_to_num(sink).contiguous() if sink is not None else q
    out = torch.empty_like(q)
    BT, BW = 64, 64
    grid = (R * nh * triton.cdiv(T, BT),)
    K.attn_fwd[grid](q, k_blk, v_blk, k_sw, v_sw, topk_idx, valid_c, last_tok, soft_c, sink_c,
                     out, T, Bm, int(w), float(scale),
                     NB=nb, NH=nh, HD=hd, BT=BT, BW=BW,
                     HAS_VALID=valid is not None, HAS_SINK=sink is not None,
                     DOT=_dot_prec(), num_warps=4)
    return out


def _attn_bwd_triton(q, k_blk, v_blk, k_sw, v_sw, topk_idx, valid, last_tok, soft, sink, w, scale, dO, needs):
    K = _kernels()
    if getattr(K, 'attn_fwd', None) is None:
        _attach_attn_kernels(K)
    triton = K.triton
    R, T, nh, hd = q.shape
    nb = topk_idx.shape[-1]
    Bm = k_blk.shape[1]
    dev = q.device
    dtype = q.dtype
    assert int(w) >= 1 and hd >= 16 and hd & (hd - 1) == 0
    q = q.contiguous()
    k_blk, v_blk = k_blk.contiguous(), v_blk.contiguous()
    k_sw, v_sw = k_sw.contiguous(), v_sw.contiguous()
    dO = dO.contiguous()
    topk_idx = topk_idx.contiguous()
    valid_c = valid.contiguous() if valid is not None else topk_idx
    last_tok = last_tok.contiguous()
    soft_c = soft.contiguous() if soft is not None else topk_idx
    sink_c = torch.nan_to_num(sink).contiguous() if sink is not None else None
    sink_arg = sink_c if sink_c is not None else q
    need_dq = bool(needs[1])
    need_dkb = bool(needs[2])
    need_dvb = bool(needs[3])
    need_dksw = bool(needs[4])
    need_dvsw = bool(needs[5])
    need_dsoft = bool(needs[9]) and soft is not None
    need_dsink = bool(needs[10]) and sink is not None
    BT, BW = 64, 64
    # soft 统计量（mS/ZS/GS）只在 HAS_SOFT 下被 attn_bwd_stats 写、被
    # attn_bwd_dq / dkdv 内核读；soft=None（HCA 臂）时给 1 元素占位缓冲。
    mA = torch.empty((R, T, nh), device=dev, dtype=torch.float32)
    ZA = torch.empty((R, T, nh), device=dev, dtype=torch.float32)
    GA = torch.empty((R, T, nh), device=dev, dtype=torch.float32)
    if soft is not None:
        mS = torch.empty((R, T, nh), device=dev, dtype=torch.float32)
        ZS = torch.empty((R, T, nh), device=dev, dtype=torch.float32)
        GS = torch.empty((R, T, nh), device=dev, dtype=torch.float32)
    else:
        mS = ZS = GS = torch.empty(1, device=dev, dtype=torch.float32)
    K.attn_bwd_stats[(R * nh * triton.cdiv(T, BT),)](
        q, k_blk, v_blk, k_sw, v_sw, topk_idx, valid_c, last_tok, soft_c, sink_arg, dO,
        mA, ZA, mS, ZS, GA, GS, T, Bm, int(w), float(scale),
        NB=nb, NH=nh, HD=hd, BT=BT, BW=BW,
        HAS_VALID=valid is not None, HAS_SOFT=soft is not None, HAS_SINK=sink is not None,
        DOT=_dot_prec(), num_warps=4)
    dq = None
    if need_dq:
        dq = torch.empty_like(q)
        K.attn_bwd_dq[(R * nh * triton.cdiv(T, BT),)](
            q, k_blk, v_blk, k_sw, v_sw, topk_idx, valid_c, last_tok, soft_c, sink_arg, dO,
            mA, ZA, mS, ZS, GA, GS, dq, T, Bm, int(w), float(scale),
            NB=nb, NH=nh, HD=hd, BT=BT, BW=BW,
            HAS_VALID=valid is not None, HAS_SOFT=soft is not None, HAS_SINK=sink is not None,
            DOT=_dot_prec(), num_warps=4)
    dk_blk = dv_blk = dk_sw = dv_sw = dsoft = None
    if need_dkb or need_dvb:
        # attn_bwd_dkdv_blocks 对网格内每个 (r,b,h) 无条件 store（NEED_DK/
        # NEED_DV 与这里的分配条件一致），因此 dk/dv 缓冲无需零初始化。
        dk_blk = torch.empty((R, Bm, nh, hd), device=dev, dtype=dtype) if need_dkb else q
        dv_blk = torch.empty((R, Bm, nh, hd), device=dev, dtype=dtype) if need_dvb else q
        flat = topk_idx.reshape(R, T * nb)
        # 稳定排序只依赖键值与位置。order 保留 int64：lane 值域是
        # [0, T·nb)，T·nb ≥ 2³¹ 的超大配置下窄化到 int32 会翻负、内核再
        # 按 lane//NB 寻址即越界。
        order = flat.argsort(dim=1, stable=True).contiguous()
        # 整数直方图（等价于逐行 bincount(minlength=Bm) 的 scatter_add_ 形
        # 式）：行号并入键后一次 bincount——整数计数精确且与写入顺序无关
        # （整数加法可交换、无舍入）。
        _keys = flat.to(torch.int64) + (torch.arange(R, device=dev, dtype=torch.int64) * Bm)[:, None]
        cnt = torch.bincount(_keys.reshape(-1), minlength=R * Bm).reshape(R, Bm)
        offs = torch.zeros((R, Bm + 1), device=dev, dtype=torch.int64)
        offs[:, 1:] = cnt.cumsum(dim=1)
        del flat, _keys, cnt
        K.attn_bwd_dkdv_blocks[(R * Bm * nh,)](
            q, k_blk, v_blk, topk_idx, valid_c, last_tok, soft_c, sink_arg, dO,
            mA, ZA, mS, ZS, GA, GS,
            dk_blk, dv_blk, order, offs, T, Bm, nb, float(scale),
            NH=nh, HD=hd, HAS_VALID=valid is not None, HAS_SOFT=soft is not None,
            HAS_SINK=sink is not None,
            NEED_DK=need_dkb, NEED_DV=need_dvb, num_warps=1)
    if need_dsoft:
        # dsoft 是跨头求和（ref 反向在头循环里 +=），(r,b,h) 网格的
        # dkdv_blocks 表达不了——由独立内核完成：每个 (t,j) 泳道恰好一个
        # 写者、头按固定顺序寄存器累加，每个 (t,j) 都被无条件 store。
        dsoft = torch.empty((R, T, nb), device=dev, dtype=dtype)
        K.attn_bwd_dsoft[(R * triton.cdiv(T, BT),)](
            q, k_blk, v_blk, topk_idx, valid_c, last_tok, soft_c, dO,
            mS, ZS, GS, dsoft, T, Bm, float(scale),
            NB=nb, NH=nh, HD=hd, BT=BT, HAS_VALID=valid is not None, num_warps=4)
    if need_dksw or need_dvsw:
        # attn_bwd_dkdv_window 对每个 (r,p,h) 同样无条件 store——empty 即可。
        dk_sw = torch.empty((R, T, nh, hd), device=dev, dtype=dtype) if need_dksw else q
        dv_sw = torch.empty((R, T, nh, hd), device=dev, dtype=dtype) if need_dvsw else q
        K.attn_bwd_dkdv_window[(R * T * nh,)](
            q, k_sw, v_sw, sink_arg, dO, mA, ZA, mS, ZS, GA, GS,
            dk_sw, dv_sw, T, int(w), float(scale),
            NH=nh, HD=hd, HAS_SINK=sink is not None, HAS_SOFT=soft is not None,
            NEED_DK=need_dksw, NEED_DV=need_dvsw, BW=64, num_warps=4)
    dsink = None
    if need_dsink:
        sk = sink_c.to(torch.float32).view(1, 1, nh)
        ZAs = torch.where(ZA == 0, torch.ones_like(ZA), ZA)
        p_sink_a = torch.exp(sk - mA) / ZAs
        acc = -(p_sink_a * GA).sum(dim=(0, 1))
        if soft is not None:
            ZSs = torch.where(ZS == 0, torch.ones_like(ZS), ZS)
            p_sink_s = torch.exp(sk - mS) / ZSs
            acc = acc - (p_sink_s * GS).sum(dim=(0, 1))
        dsink = acc.to(dtype)
    return (dq, dk_blk, dv_blk, dk_sw, dv_sw, dsoft, dsink)


def _attach_attn_kernels(ns):
    import triton
    import triton.language as tl

    @triton.jit
    def attn_fwd(q_ptr, kb_ptr, vb_ptr, ksw_ptr, vsw_ptr, idx_ptr, valid_ptr, lt_ptr, soft_ptr, sink_ptr, out_ptr,
                 T, Bm, W, SCALE,
                 NB: tl.constexpr, NH: tl.constexpr, HD: tl.constexpr, BT: tl.constexpr, BW: tl.constexpr,
                 HAS_VALID: tl.constexpr, HAS_SINK: tl.constexpr, DOT: tl.constexpr):
        pid = tl.program_id(0)
        n_t = tl.cdiv(T, BT)
        pid_t = pid % n_t
        rh = pid // n_t
        h = rh % NH
        r = rh // NH
        t0 = pid_t * BT
        t = t0 + tl.arange(0, BT)
        tm = t < T
        ti = t.to(tl.int64)
        d = tl.arange(0, HD)
        qt = tl.load(q_ptr + ((r * T + ti[:, None]) * NH + h) * HD + d[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
        if HAS_SINK:
            sk = tl.load(sink_ptr + h).to(tl.float32)
            sk = tl.where(sk == sk, sk, 0.0)
            m = tl.zeros((BT,), tl.float32) + sk
            l = tl.full((BT,), 1.0, tl.float32)
        else:
            m = tl.full((BT,), -1e30, tl.float32)
            l = tl.zeros((BT,), tl.float32)
        acc = tl.zeros((BT, HD), tl.float32)
        p_lo = tl.maximum(t0 - (W - 1), 0)
        p_hi = tl.minimum(t0 + BT, T)
        for p0 in range(p_lo, p_hi, BW):
            p = p0 + tl.arange(0, BW)
            pm = p < p_hi
            kt = tl.load(ksw_ptr + ((r * T + p[:, None].to(tl.int64)) * NH + h) * HD + d[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
            z = tl.dot(qt, tl.trans(kt), input_precision=DOT) * SCALE
            ok = pm[None, :] & (p[None, :] <= ti[:, None]) & (p[None, :] > ti[:, None] - W)
            z = tl.where(ok, z, float('-inf'))
            mn = tl.maximum(m, tl.max(z, 1))
            alpha = tl.exp(m - mn)
            pa = tl.exp(z - mn[:, None])
            l = l * alpha + tl.sum(pa, 1)
            vt = tl.load(vsw_ptr + ((r * T + p[:, None].to(tl.int64)) * NH + h) * HD + d[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
            acc = acc * alpha[:, None] + tl.dot(pa, vt, input_precision=DOT)
            m = mn
        for j in tl.static_range(NB):
            src = tl.load(idx_ptr + ((r * T + ti) * NB + j), mask=tm, other=0).to(tl.int64)
            lt_j = tl.load(lt_ptr + r * Bm + src, mask=tm, other=0).to(tl.int64)
            ok = tm & (ti > lt_j)
            if HAS_VALID:
                vj = tl.load(valid_ptr + ((r * T + ti) * NB + j), mask=tm, other=0)
                ok = ok & (vj != 0)
            kj = tl.load(kb_ptr + ((r * Bm + src[:, None]) * NH + h) * HD + d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
            zj = tl.sum(qt * kj, 1) * SCALE
            zj = tl.where(ok, zj, float('-inf'))
            mn = tl.maximum(m, zj)
            alpha = tl.exp(m - mn)
            pj = tl.exp(zj - mn)
            l = l * alpha + pj
            vjv = tl.load(vb_ptr + ((r * Bm + src[:, None]) * NH + h) * HD + d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
            acc = acc * alpha[:, None] + pj[:, None] * vjv
            m = mn
        if HAS_SINK:
            out = acc / l[:, None]
        else:
            lsafe = tl.where(l == 0.0, 1.0, l)
            out = acc / lsafe[:, None]
        tl.store(out_ptr + ((r * T + ti[:, None]) * NH + h) * HD + d[None, :], out.to(out_ptr.dtype.element_ty), mask=tm[:, None])

    ns.attn_fwd = attn_fwd

    @triton.jit
    def attn_bwd_stats(q_ptr, kb_ptr, vb_ptr, ksw_ptr, vsw_ptr, idx_ptr, valid_ptr, lt_ptr, soft_ptr, sink_ptr, dO_ptr,
                       mA_ptr, ZA_ptr, mS_ptr, ZS_ptr, GA_ptr, GS_ptr,
                       T, Bm, W, SCALE,
                       NB: tl.constexpr, NH: tl.constexpr, HD: tl.constexpr, BT: tl.constexpr, BW: tl.constexpr,
                       HAS_VALID: tl.constexpr, HAS_SOFT: tl.constexpr, HAS_SINK: tl.constexpr, DOT: tl.constexpr):
        pid = tl.program_id(0)
        n_t = tl.cdiv(T, BT)
        pid_t = pid % n_t
        rh = pid // n_t
        h = rh % NH
        r = rh // NH
        t0 = pid_t * BT
        t = t0 + tl.arange(0, BT)
        tm = t < T
        ti = t.to(tl.int64)
        d = tl.arange(0, HD)
        qt = tl.load(q_ptr + ((r * T + ti[:, None]) * NH + h) * HD + d[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
        dOt = tl.load(dO_ptr + ((r * T + ti[:, None]) * NH + h) * HD + d[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
        if HAS_SINK:
            sk = tl.load(sink_ptr + h).to(tl.float32)
            sk = tl.where(sk == sk, sk, 0.0)
            mA = tl.zeros((BT,), tl.float32) + sk
            ZA = tl.full((BT,), 1.0, tl.float32)
            mS = tl.zeros((BT,), tl.float32) + sk
            ZS = tl.full((BT,), 1.0, tl.float32)
        else:
            mA = tl.full((BT,), -1e30, tl.float32)
            ZA = tl.zeros((BT,), tl.float32)
            mS = tl.full((BT,), -1e30, tl.float32)
            ZS = tl.zeros((BT,), tl.float32)
        GAr = tl.zeros((BT,), tl.float32)
        GSr = tl.zeros((BT,), tl.float32)
        p_lo = tl.maximum(t0 - (W - 1), 0)
        p_hi = tl.minimum(t0 + BT, T)
        for p0 in range(p_lo, p_hi, BW):
            p = p0 + tl.arange(0, BW)
            pm = p < p_hi
            kt = tl.load(ksw_ptr + ((r * T + p[:, None].to(tl.int64)) * NH + h) * HD + d[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
            vt = tl.load(vsw_ptr + ((r * T + p[:, None].to(tl.int64)) * NH + h) * HD + d[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
            z = tl.dot(qt, tl.trans(kt), input_precision=DOT) * SCALE
            ok = pm[None, :] & (p[None, :] <= ti[:, None]) & (p[None, :] > ti[:, None] - W)
            z = tl.where(ok, z, float('-inf'))
            g = tl.dot(dOt, tl.trans(vt), input_precision=DOT)
            mnA = tl.maximum(mA, tl.max(z, 1))
            aA = tl.exp(mA - mnA)
            ZA = ZA * aA + tl.sum(tl.exp(z - mnA[:, None]), 1)
            GAr = GAr * aA + tl.sum(tl.exp(z - mnA[:, None]) * g, 1)
            mA = mnA
            if HAS_SOFT:
                mnS = tl.maximum(mS, tl.max(z, 1))
                aS = tl.exp(mS - mnS)
                ZS = ZS * aS + tl.sum(tl.exp(z - mnS[:, None]), 1)
                GSr = GSr * aS + tl.sum(tl.exp(z - mnS[:, None]) * g, 1)
                mS = mnS
        for j in tl.static_range(NB):
            src = tl.load(idx_ptr + ((r * T + ti) * NB + j), mask=tm, other=0).to(tl.int64)
            lt_j = tl.load(lt_ptr + r * Bm + src, mask=tm, other=0).to(tl.int64)
            ok = tm & (ti > lt_j)
            if HAS_VALID:
                vj = tl.load(valid_ptr + ((r * T + ti) * NB + j), mask=tm, other=0)
                ok = ok & (vj != 0)
            kj = tl.load(kb_ptr + ((r * Bm + src[:, None]) * NH + h) * HD + d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
            vjv = tl.load(vb_ptr + ((r * Bm + src[:, None]) * NH + h) * HD + d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
            zj = tl.sum(qt * kj, 1) * SCALE
            zj = tl.where(ok, zj, float('-inf'))
            g = tl.sum(dOt * vjv, 1)
            mnA = tl.maximum(mA, zj)
            aA = tl.exp(mA - mnA)
            eA = tl.exp(zj - mnA)
            ZA = ZA * aA + eA
            GAr = GAr * aA + eA * g
            mA = mnA
            if HAS_SOFT:
                sf = tl.load(soft_ptr + ((r * T + ti) * NB + j), mask=tm, other=0.0).to(tl.float32)
                lam = tl.where(sf > 0.0, tl.log(tl.maximum(sf, 1e-12)), -3.402823466e38)
                zS = tl.maximum(zj + lam, -3.402823466e38)
                zS = tl.where(ok, zS, float('-inf'))
                mnS = tl.maximum(mS, zS)
                aS = tl.exp(mS - mnS)
                eS = tl.exp(zS - mnS)
                ZS = ZS * aS + eS
                GSr = GSr * aS + eS * g
                mS = mnS
        ZAs = tl.where(ZA == 0.0, 1.0, ZA)
        ZSs = tl.where(ZS == 0.0, 1.0, ZS)
        base = (r * T + ti) * NH + h
        tl.store(mA_ptr + base, mA, mask=tm)
        tl.store(ZA_ptr + base, ZA, mask=tm)
        tl.store(GA_ptr + base, GAr / ZAs, mask=tm)
        if HAS_SOFT:
            tl.store(mS_ptr + base, mS, mask=tm)
            tl.store(ZS_ptr + base, ZS, mask=tm)
            tl.store(GS_ptr + base, GSr / ZSs, mask=tm)

    ns.attn_bwd_stats = attn_bwd_stats

    @triton.jit
    def attn_bwd_dq(q_ptr, kb_ptr, vb_ptr, ksw_ptr, vsw_ptr, idx_ptr, valid_ptr, lt_ptr, soft_ptr, sink_ptr, dO_ptr,
                    mA_ptr, ZA_ptr, mS_ptr, ZS_ptr, GA_ptr, GS_ptr, dq_ptr,
                    T, Bm, W, SCALE,
                    NB: tl.constexpr, NH: tl.constexpr, HD: tl.constexpr, BT: tl.constexpr, BW: tl.constexpr,
                    HAS_VALID: tl.constexpr, HAS_SOFT: tl.constexpr, HAS_SINK: tl.constexpr, DOT: tl.constexpr):
        pid = tl.program_id(0)
        n_t = tl.cdiv(T, BT)
        pid_t = pid % n_t
        rh = pid // n_t
        h = rh % NH
        r = rh // NH
        t0 = pid_t * BT
        t = t0 + tl.arange(0, BT)
        tm = t < T
        ti = t.to(tl.int64)
        d = tl.arange(0, HD)
        qt = tl.load(q_ptr + ((r * T + ti[:, None]) * NH + h) * HD + d[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
        dOt = tl.load(dO_ptr + ((r * T + ti[:, None]) * NH + h) * HD + d[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
        st = (r * T + ti) * NH + h
        mA = tl.load(mA_ptr + st, mask=tm, other=0.0)
        ZA = tl.load(ZA_ptr + st, mask=tm, other=1.0)
        GA = tl.load(GA_ptr + st, mask=tm, other=0.0)
        ZAs = tl.where(ZA == 0.0, 1.0, ZA)
        # soft 统计量只在 HAS_SOFT 下参与 dq——其余情形 mS/ZS/GS 为 1 元素占位
        # 缓冲（启动器侧），读取全部跳过。
        if HAS_SOFT:
            mS = tl.load(mS_ptr + st, mask=tm, other=0.0)
            ZS = tl.load(ZS_ptr + st, mask=tm, other=1.0)
            GS = tl.load(GS_ptr + st, mask=tm, other=0.0)
            ZSs = tl.where(ZS == 0.0, 1.0, ZS)
        else:
            mS = 0.0
            GS = 0.0
            ZSs = 1.0
        dqacc = tl.zeros((BT, HD), tl.float32)
        p_lo = tl.maximum(t0 - (W - 1), 0)
        p_hi = tl.minimum(t0 + BT, T)
        for p0 in range(p_lo, p_hi, BW):
            p = p0 + tl.arange(0, BW)
            pm = p < p_hi
            kt = tl.load(ksw_ptr + ((r * T + p[:, None].to(tl.int64)) * NH + h) * HD + d[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
            vt = tl.load(vsw_ptr + ((r * T + p[:, None].to(tl.int64)) * NH + h) * HD + d[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
            z = tl.dot(qt, tl.trans(kt), input_precision=DOT) * SCALE
            ok = pm[None, :] & (p[None, :] <= ti[:, None]) & (p[None, :] > ti[:, None] - W)
            z = tl.where(ok, z, float('-inf'))
            eA = tl.exp(z - mA[:, None])
            g = tl.dot(dOt, tl.trans(vt), input_precision=DOT)
            dz = eA / ZAs[:, None] * (g - GA[:, None])
            if HAS_SOFT:
                eS = tl.exp(z - mS[:, None])
                dz += eS / ZSs[:, None] * (g - GS[:, None])
            dqacc += tl.dot(dz, kt, input_precision=DOT) * SCALE
        for j in tl.static_range(NB):
            src = tl.load(idx_ptr + ((r * T + ti) * NB + j), mask=tm, other=0).to(tl.int64)
            lt_j = tl.load(lt_ptr + r * Bm + src, mask=tm, other=0).to(tl.int64)
            ok = tm & (ti > lt_j)
            if HAS_VALID:
                vj = tl.load(valid_ptr + ((r * T + ti) * NB + j), mask=tm, other=0)
                ok = ok & (vj != 0)
            kj = tl.load(kb_ptr + ((r * Bm + src[:, None]) * NH + h) * HD + d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
            vjv = tl.load(vb_ptr + ((r * Bm + src[:, None]) * NH + h) * HD + d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
            zj = tl.sum(qt * kj, 1) * SCALE
            zj = tl.where(ok, zj, float('-inf'))
            eA = tl.exp(zj - mA)
            g = tl.sum(dOt * vjv, 1)
            dz = eA / ZAs * (g - GA)
            if HAS_SOFT:
                sf = tl.load(soft_ptr + ((r * T + ti) * NB + j), mask=tm, other=0.0).to(tl.float32)
                lam = tl.where(sf > 0.0, tl.log(tl.maximum(sf, 1e-12)), -3.402823466e38)
                zS = tl.maximum(zj + lam, -3.402823466e38)
                zS = tl.where(ok, zS, float('-inf'))
                eS = tl.exp(zS - mS)
                dz += eS / ZSs * (g - GS)
            dqacc += dz[:, None] * kj * SCALE
        tl.store(dq_ptr + ((r * T + ti[:, None]) * NH + h) * HD + d[None, :], dqacc.to(dq_ptr.dtype.element_ty), mask=tm[:, None])

    ns.attn_bwd_dq = attn_bwd_dq

    @triton.jit
    def attn_bwd_dkdv_blocks(q_ptr, kb_ptr, vb_ptr, idx_ptr, valid_ptr, lt_ptr, soft_ptr, sink_ptr, dO_ptr,
                             mA_ptr, ZA_ptr, mS_ptr, ZS_ptr, GA_ptr, GS_ptr,
                             dkb_ptr, dvb_ptr, csr_ptr, cso_ptr,
                             T, Bm, NB, SCALE,
                             NH: tl.constexpr, HD: tl.constexpr,
                             HAS_VALID: tl.constexpr, HAS_SOFT: tl.constexpr, HAS_SINK: tl.constexpr,
                             NEED_DK: tl.constexpr, NEED_DV: tl.constexpr):
        pid = tl.program_id(0)
        h = pid % NH
        rb = pid // NH
        b = rb % Bm
        r = rb // Bm
        seg0 = tl.load(cso_ptr + r.to(tl.int64) * (Bm + 1) + b)
        seg1 = tl.load(cso_ptr + r.to(tl.int64) * (Bm + 1) + b + 1)
        d = tl.arange(0, HD)
        kb_v = tl.load(kb_ptr + ((r * Bm + b) * NH + h) * HD + d).to(tl.float32)
        vb_v = tl.load(vb_ptr + ((r * Bm + b) * NH + h) * HD + d).to(tl.float32)
        dk_acc = tl.zeros((HD,), tl.float32)
        dv_acc = tl.zeros((HD,), tl.float32)
        # 循环不变量：本程序只服务块 b，last_tok 在段循环外只载一次（同一值）。
        lt_b = tl.load(lt_ptr + r * Bm + b).to(tl.int64)
        for k in range(seg0, seg1):
            # r*(T*NB) 在 R·T·NB ≥ 2³¹ 的超大配置下按 int32 求值会溢出
            # （int64 的 k 要到加法才参与升型）——先升 r 再连乘，整条链
            # 保持 int64（其余寻址都经 src/ti 的 int64 广播）。
            lane = tl.load(csr_ptr + r.to(tl.int64) * T * NB + k).to(tl.int64)
            tt = lane // NB
            jj = lane % NB
            if HAS_VALID:
                # Match attn_fwd / attn_bwd_stats: validity AND causality.
                ok = (tl.load(valid_ptr + (r * T + tt) * NB + jj) != 0) & (tt > lt_b)
            else:
                ok = tt > lt_b
            if ok:
                q_v = tl.load(q_ptr + ((r * T + tt) * NH + h) * HD + d).to(tl.float32)
                dO_v = tl.load(dO_ptr + ((r * T + tt) * NH + h) * HD + d).to(tl.float32)
                z = tl.sum(q_v * kb_v) * SCALE
                st = (r * T + tt) * NH + h
                mA = tl.load(mA_ptr + st)
                ZA = tl.load(ZA_ptr + st)
                GA = tl.load(GA_ptr + st)
                ZAs = tl.where(ZA == 0.0, 1.0, ZA)
                PA = tl.exp(z - mA) / ZAs
                g = tl.sum(dO_v * vb_v)
                dz = PA * (g - GA)
                if HAS_SOFT:
                    GS = tl.load(GS_ptr + st)
                    mS = tl.load(mS_ptr + st)
                    ZS = tl.load(ZS_ptr + st)
                    ZSs = tl.where(ZS == 0.0, 1.0, ZS)
                    sf = tl.load(soft_ptr + (r * T + tt) * NB + jj).to(tl.float32)
                    lam = tl.where(sf > 0.0, tl.log(tl.maximum(sf, 1e-12)), -3.402823466e38)
                    zS = tl.maximum(z + lam, -3.402823466e38)
                    PS = tl.exp(zS - mS) / ZSs
                    dzS = PS * (g - GS)
                    dz += dzS
                if NEED_DK:
                    dk_acc += dz * q_v * SCALE
                if NEED_DV:
                    dv_acc += PA * dO_v
        if NEED_DK:
            tl.store(dkb_ptr + ((r * Bm + b) * NH + h) * HD + d, dk_acc.to(dkb_ptr.dtype.element_ty))
        if NEED_DV:
            tl.store(dvb_ptr + ((r * Bm + b) * NH + h) * HD + d, dv_acc.to(dvb_ptr.dtype.element_ty))

    ns.attn_bwd_dkdv_blocks = attn_bwd_dkdv_blocks

    @triton.jit
    def attn_bwd_dsoft(q_ptr, kb_ptr, vb_ptr, idx_ptr, valid_ptr, lt_ptr, soft_ptr, dO_ptr,
                       mS_ptr, ZS_ptr, GS_ptr, dsoft_ptr,
                       T, Bm, SCALE,
                       NB: tl.constexpr, NH: tl.constexpr, HD: tl.constexpr, BT: tl.constexpr,
                       HAS_VALID: tl.constexpr):
        # dsoft = Σ_h dzS_h / sf —— ref 反向在头循环里 += 的同一个跨头求和。
        # dkdv_blocks 的网格是 (r, b, h)，表达不了它：同一 (t, j) 泳道会被
        # nh 个程序写同一地址（末写者胜、且永远拼不出头求和）。故独立成
        # 内核：网格 (r, t 块)，每个 (t, j) 恰好一个写者，头贡献按
        # static_range(NH) 的固定顺序在寄存器里累加——无原子、无跨程序
        # 写，归约顺序与 ref 的头升序一致；逐头贡献先舍入到输出 dtype 再
        # 累加，与 ref 的 `.to(dtype)` 后 += 同一步骤。
        pid = tl.program_id(0)
        n_t = tl.cdiv(T, BT)
        pid_t = pid % n_t
        r = pid // n_t
        t0 = pid_t * BT
        t = t0 + tl.arange(0, BT)
        tm = t < T
        ti = t.to(tl.int64)
        d = tl.arange(0, HD)
        for j in tl.static_range(NB):
            src = tl.load(idx_ptr + ((r * T + ti) * NB + j), mask=tm, other=0).to(tl.int64)
            lt_j = tl.load(lt_ptr + r * Bm + src, mask=tm, other=0).to(tl.int64)
            ok = tm & (ti > lt_j)
            if HAS_VALID:
                vj = tl.load(valid_ptr + ((r * T + ti) * NB + j), mask=tm, other=0)
                ok = ok & (vj != 0)
            sf = tl.load(soft_ptr + ((r * T + ti) * NB + j), mask=tm, other=0.0).to(tl.float32)
            lam = tl.where(sf > 0.0, tl.log(tl.maximum(sf, 1e-12)), -3.402823466e38)
            acc = tl.zeros((BT,), tl.float32)
            for h in tl.static_range(NH):
                st = (r * T + ti) * NH + h
                q_v = tl.load(q_ptr + ((r * T + ti[:, None]) * NH + h) * HD + d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
                k_v = tl.load(kb_ptr + ((r * Bm + src[:, None]) * NH + h) * HD + d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
                dO_v = tl.load(dO_ptr + ((r * T + ti[:, None]) * NH + h) * HD + d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
                v_v = tl.load(vb_ptr + ((r * Bm + src[:, None]) * NH + h) * HD + d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
                mS = tl.load(mS_ptr + st, mask=tm, other=0.0)
                ZS = tl.load(ZS_ptr + st, mask=tm, other=1.0)
                GS = tl.load(GS_ptr + st, mask=tm, other=0.0)
                ZSs = tl.where(ZS == 0.0, 1.0, ZS)
                z = tl.sum(q_v * k_v, 1) * SCALE
                zS = tl.maximum(z + lam, -3.402823466e38)
                zS = tl.where(ok, zS, float('-inf'))
                PS = tl.exp(zS - mS) / ZSs
                g = tl.sum(dO_v * v_v, 1)
                dzS = PS * (g - GS)
                ds_h = tl.where(sf >= 1e-12, dzS / tl.maximum(sf, 1e-12), 0.0)
                acc += ds_h.to(dsoft_ptr.dtype.element_ty).to(tl.float32)
            tl.store(dsoft_ptr + ((r * T + ti) * NB + j), acc.to(dsoft_ptr.dtype.element_ty), mask=tm)

    ns.attn_bwd_dsoft = attn_bwd_dsoft

    @triton.jit
    def attn_bwd_dkdv_window(q_ptr, ksw_ptr, vsw_ptr, sink_ptr, dO_ptr,
                             mA_ptr, ZA_ptr, mS_ptr, ZS_ptr, GA_ptr, GS_ptr,
                             dksw_ptr, dvsw_ptr, T, W, SCALE,
                             NH: tl.constexpr, HD: tl.constexpr, HAS_SINK: tl.constexpr,
                             HAS_SOFT: tl.constexpr,
                             NEED_DK: tl.constexpr, NEED_DV: tl.constexpr, BW: tl.constexpr):
        pid = tl.program_id(0)
        h = pid % NH
        rp = pid // NH
        r = rp // T
        p = (rp % T).to(tl.int64)
        d = tl.arange(0, HD)
        k_v = tl.load(ksw_ptr + ((r * T + p) * NH + h) * HD + d).to(tl.float32)
        v_v = tl.load(vsw_ptr + ((r * T + p) * NH + h) * HD + d).to(tl.float32)
        dk_acc = tl.zeros((HD,), tl.float32)
        dv_acc = tl.zeros((HD,), tl.float32)
        t_hi = p + W
        if t_hi > T:
            t_hi = T
        # tt 沿 [p, p+W) 连续——按 [BW, HD] 分片矢量载入替代逐 tt 标量循环：
        # 片内固定树形归约、片间按 tt 升序累加（与逐 tt 循环同为固定顺序
        # 归约，内核变体口径见模块 docstring）。掩码泳道经 other=0 载入后
        # z=0、PA=exp(0)/1、g=0，各贡献项（dz·q、PA·dO）恰为 0。
        for tt0 in range(p, t_hi, BW):
            tt = tt0 + tl.arange(0, BW)
            tm2 = tt < t_hi
            tt64 = tt.to(tl.int64)
            q_t = tl.load(q_ptr + ((r * T + tt64[:, None]) * NH + h) * HD + d[None, :], mask=tm2[:, None], other=0.0).to(tl.float32)
            dO_t = tl.load(dO_ptr + ((r * T + tt64[:, None]) * NH + h) * HD + d[None, :], mask=tm2[:, None], other=0.0).to(tl.float32)
            z_t = tl.sum(q_t * k_v[None, :], 1) * SCALE
            st = (r * T + tt64) * NH + h
            mA_t = tl.load(mA_ptr + st, mask=tm2, other=0.0)
            ZA_t = tl.load(ZA_ptr + st, mask=tm2, other=1.0)
            GA_t = tl.load(GA_ptr + st, mask=tm2, other=0.0)
            ZAs_t = tl.where(ZA_t == 0.0, 1.0, ZA_t)
            PA_t = tl.exp(z_t - mA_t) / ZAs_t
            g_t = tl.sum(dO_t * v_v[None, :], 1)
            dz_t = PA_t * (g_t - GA_t)
            if HAS_SOFT:
                GS_t = tl.load(GS_ptr + st, mask=tm2, other=0.0)
                mS_t = tl.load(mS_ptr + st, mask=tm2, other=0.0)
                ZS_t = tl.load(ZS_ptr + st, mask=tm2, other=1.0)
                ZSs_t = tl.where(ZS_t == 0.0, 1.0, ZS_t)
                PS_t = tl.exp(z_t - mS_t) / ZSs_t
                dz_t += PS_t * (g_t - GS_t)
            if NEED_DK:
                dk_acc += tl.sum(dz_t[:, None] * q_t * SCALE, 0)
            if NEED_DV:
                dv_acc += tl.sum(PA_t[:, None] * dO_t, 0)
        if NEED_DK:
            tl.store(dksw_ptr + ((r * T + p) * NH + h) * HD + d, dk_acc.to(dksw_ptr.dtype.element_ty))
        if NEED_DV:
            tl.store(dvsw_ptr + ((r * T + p) * NH + h) * HD + d, dv_acc.to(dvsw_ptr.dtype.element_ty))

    ns.attn_bwd_dkdv_window = attn_bwd_dkdv_window


# ============================================================================
# Triton 启动封装（pool / indexer）
# ============================================================================

def _pool_fwd_triton(Xa, Xb, Za, Zb, geom, B_pos_a, B_pos_b):
    K = _kernels()
    triton = K.triton
    R, T, C = Xa.shape
    Bm = geom.Bm
    L, W = geom.max_len, geom.W_b
    has_b = Xb is not None and W > 0
    Xa = Xa.contiguous()
    Za = Za.contiguous()
    if has_b:
        Xb = Xb.contiguous()
        Zb = Zb.contiguous()
    B_pos_a = B_pos_a.contiguous()
    B_pos_b = (B_pos_b if has_b else B_pos_a).contiguous()
    BLOCK_C = min(64, _next_pow2(C))
    BLOCK_L = _next_pow2(L)
    BLOCK_W = max(_next_pow2(W), 1)
    comp = torch.empty((R, Bm, C), device=Xa.device, dtype=Xa.dtype)
    grid = (R * Bm * triton.cdiv(C, BLOCK_C),)
    K.pool_fwd[grid](Xa, Xb if has_b else Za, Za, Zb if has_b else Za, B_pos_a, B_pos_b,
                     geom.starts, geom.counts, geom.ov_start, geom.ov_len, geom.starts_prev,
                     comp, T, C, Bm, int(B_pos_a.shape[0]), int(B_pos_b.shape[0]) if has_b else 1,
                     BLOCK_C=BLOCK_C, BLOCK_L=BLOCK_L, BLOCK_W=BLOCK_W, HAS_B=has_b,
                     num_warps=8 if BLOCK_C * BLOCK_L >= 4096 else 4)
    return comp


# scores 分块缓冲的字节上限（fp32 [R, T_c, Bm]）：打分/top-k 两内核按
# query 行分块启动，scores 不按全幅 [R, T, Bm] 物化——每行的值与全幅
# 形式逐位一致（每个 (r, t) 程序只读写自己的行）。
_IDX_SCORES_CAP_BYTES = 128 << 20


def _indexer_fwd_triton(qI, kI, w_idx, last_tok, topk):
    K = _kernels()
    triton = K.triton
    R, T, nIH, hd = qI.shape
    Bm = kI.shape[1]
    assert Bm <= 8192, 'Bm > 8192 超出单程序泳道寄存器上限——走 ref 后端'
    qI = qI.contiguous()
    kI = kI.contiguous()
    w_idx = w_idx.contiguous()
    last_tok = last_tok.contiguous()
    # 打分泳道宽补到 >= 16 的 2 幂（零填充泳道对点积恰好贡献 +0.0，分数
    # 逐位不变），hd 不再要求是 2 的幂或 >= 16。
    HDP = max(_next_pow2(hd), 16)
    k_out = min(int(topk), Bm)
    BB2 = _next_pow2(Bm)
    idx = torch.empty((R, T, k_out), device=qI.device, dtype=torch.int32)
    valid_i8 = torch.empty((R, T, k_out), device=qI.device, dtype=torch.int8)
    soft = torch.empty((R, T, k_out), device=qI.device, dtype=qI.dtype)
    BT, BB = 64, 64
    T_c = max(1, min(T, _IDX_SCORES_CAP_BYTES // max(R * Bm * 4, 1)))
    scores = torch.empty((R, T_c, Bm), device=qI.device, dtype=torch.float32)
    for t0 in range(0, T, T_c):
        tc = min(T_c, T - t0)
        K.idx_score[(R * triton.cdiv(tc, BT),)](qI, kI, w_idx, scores, t0, T, tc, Bm, hd ** -0.5,
                                                N_IH=nIH, HD=hd, HDP=HDP, BT=BT, BB=BB,
                                                DOT=_dot_prec(), num_warps=4)
        K.idx_topk[(R * tc,)](scores, last_tok, idx, valid_i8, soft, t0, T, tc, Bm,
                              K=k_out, BB=BB2, num_warps=8 if BB2 >= 1024 else 4)
    return idx, valid_i8.view(torch.bool), soft


# ============================================================================
# HybridAttention 的稀疏内核路径（exp_lib.forward 的批量分发放到这里，
# eager 路径零改动）
# ============================================================================

def sparse_layer_forward(attn, x, pre_all, gate_sigs, pre_blocks, impl=None):
    """csa/hca 层的批量化内核前向（对应 eager 的 forward 稀疏分支 + _single）。

    输入：attn = HybridAttention 实例；x [B, T, d]；pre_all/gate_sigs/pre_blocks
    = exp_lib.HybridAttention.forward 已算好的批量提升（门控信号 / 块几何）。
    返回 outs [B, T, d]，并按 eager 同口径设置 attn.last_gate_mean。

    语义对齐（相对 eager 的 _single）：
    - q / core_kv / kv_local / k_blk / k_sw 的归一化与 W_kvhead 的行拼接
      GEMM 逐行独立，批量化不改值；
    - csa：pool_variable_blocks → lightning_indexer → gathered_attention(soft)；
      hca：pool_blocks_single → arange 展开（pad 泳道由 last_tok=T-1 的
      不可读门自动清零）→ gathered_attention(无 soft)；
    - 门控正则 gate_mean 与 eager 的 (nblk-1 + Σ gate·hon)/T 逐行同式。
    """
    cfg = attn.cfg
    # 逐算子解析后端：显式 impl= 强制全部三算子（selfcheck/bench 口径）；
    # impl=None 时各算子吃自己的 CSA_KERNEL_{POOL,IDX,ATTN} 覆盖（空则跟随
    # 全局 CSA_KERNEL）。
    impl_pool = impl or _op_impl('POOL')
    impl_idx = impl or _op_impl('IDX')
    impl_attn = impl or _op_impl('ATTN')
    B, T, d = x.shape
    dev = x.device
    if cfg.fuse_mode != 'none':
        raise RuntimeError('CSA_KERNEL 不支持 fuse_mode != "none"（csa_dyn_fuse 臂）——该臂请回退 eager（ unset CSA_KERNEL ）')
    if cfg.indexer_mode != 'learned':
        raise RuntimeError('CSA_KERNEL 只支持 learned indexer（indexer_mode != "learned" 的臂请回退 eager）')
    if cfg.chunking == 'cosine_learnable':
        if pre_blocks is None or gate_sigs is None:
            raise RuntimeError('cosine_learnable 层需要 forward 提升的门控信号与块几何')
        bids = [pb[0] for pb in pre_blocks]
        nblks = [pb[1] for pb in pre_blocks]
        hons = [pb[2] for pb in pre_blocks]
        bid_full = torch.stack(bids, 0).to(torch.int32).contiguous()
        counts_rows = [getattr(bd, '_pool_cpu', None) for bd in bids]
        if any(c is None for c in counts_rows):
            raise RuntimeError('块几何缺少 _pool_cpu counts——分段器出口被绕过')
        counts_rows = [c[0] for c in counts_rows]
    elif pre_blocks is not None:
        bids = [pb[0] for pb in pre_blocks]
        nblks = [pb[1] for pb in pre_blocks]
        hons = None
        bid_full = torch.stack(bids, 0).to(torch.int32).contiguous()
        counts_rows = [getattr(bd, '_pool_cpu', None) for bd in bids]
        if any(c is None for c in counts_rows):
            raise RuntimeError('块几何缺少 _pool_cpu counts——分段器出口被绕过')
        counts_rows = [c[0] for c in counts_rows]
    else:
        base = _exp_blocks_fixed(T, cfg.block_size, dev)
        counts = getattr(base, '_pool_cpu', None)
        if counts is None:
            raise RuntimeError('blocks_fixed 未携带 _pool_cpu——分段器出口被绕过')
        # blocks_fixed 经 _IDX_CACHE 全局缓存（同一对象贯穿全部层/步），
        # bid_full 是该对象内容的纯整型函数（transport only）——按 (B, device)
        # 缓存在对象上，消掉每层每步的 [B, T] int32 expand+contiguous 分配
        # 及其在 autograd 图里的逐层持有。
        _bkey = (int(B), str(dev))
        _bc = getattr(base, '_bidfull_cache', None)
        if _bc is None:
            _bc = {}
            base._bidfull_cache = _bc
        bid_full = _bc.get(_bkey)
        if bid_full is None:
            with torch.inference_mode(False):
                bid_full = base[None, :].to(torch.int32).expand(B, T).contiguous()
            _bc[_bkey] = bid_full
        counts_rows = [counts[0]] * B
        nblks = [len(counts[0])] * B
        hons = None
    if attn._stats is not None:
        # 与 eager `_single` 同口径的块统计：输入是同一组块 id（int32 与
        # int64 的 bincount/不等比较结果逐一相同），每条目同样一行一条。
        # 只在 eval（compression_report 经 enable_block_stats 开启）时触发；
        # 没有它内核模式的记录会丢掉全部块长/边界诊断。
        for _b in range(B):
            attn._stats.append(torch.bincount(bid_full[_b]))
        if attn._cuts is not None:
            for _b in range(B):
                attn._cuts.append((bid_full[_b, 1:] != bid_full[_b, :-1]).nonzero(as_tuple=True)[0] + 1)
    overlap = cfg.overlap if cfg.kind == 'csa' else None
    geom = _pool_geometry_cached(counts_rows, overlap, T, dev, cfg.max_block)
    Ca = pre_all['Ca']
    Za = pre_all['Za']
    q = F.normalize(pre_all['q'].view(B, T, attn.nh, attn.hd), dim=-1)
    if cfg.kind == 'hca':
        comp, last_tok, _Bn = pool_blocks_single_kernel(Ca, Za, bid_full, geom, attn.B_pos_a, impl=impl_pool)
    else:
        comp, last_tok, _Bn = pool_variable_blocks_kernel(Ca, pre_all['Cb'], Za, pre_all['Zb'], bid_full, geom,
                                                          attn.B_pos_a, attn.B_pos_b, impl=impl_pool)
    attn_kv = torch.zeros_like(comp) if cfg.content_mode == 'zero' else comp
    core_kv = F.normalize(attn_kv, dim=-1)
    kv_local = F.normalize(Ca, dim=-1)
    Bm = geom.Bm
    kv = core_kv.shape[-1]
    _kvh = attn.W_kvhead(torch.cat([core_kv.reshape(B * Bm, kv), kv_local.reshape(B * T, kv)], 0))
    _kb, _vb = attn._split(_kvh[:B * Bm])
    _kw, _vw = attn._split(_kvh[B * Bm:])
    k_blk = _kb.view(B, Bm, attn.nh, attn.hd)
    v_blk = _vb.view(B, Bm, attn.nh, attn.hd)
    k_sw = _kw.view(B, T, attn.nh, attn.hd)
    v_sw = _vw.view(B, T, attn.nh, attn.hd)
    k_blk = F.normalize(k_blk, dim=-1)
    k_sw = F.normalize(k_sw, dim=-1)
    if cfg.kind == 'hca':
        topk_idx = _full_arange_cached(B, T, Bm, dev)
        valid = None
        soft = None
    else:
        nIH = cfg.n_index_heads
        hd = cfg.c_index // nIH
        qI = pre_all['qI'].view(B, T, nIH, hd)
        w_idx = pre_all['w_idx']
        kI = (comp.reshape(B * Bm, kv) @ attn.W_DK).view(B, Bm, nIH, hd)
        topk_idx, valid, soft = lightning_indexer_kernel(qI, kI, w_idx, last_tok, cfg.index_topk, impl=impl_idx)
    sink = attn.sink if cfg.use_sink else None
    out = gathered_attention_kernel(q, k_blk, v_blk, topk_idx, last_tok, k_sw, v_sw,
                                    cfg.sliding_window, 1.0,
                                    soft=soft, sink_logits=sink, sel_valid=valid, impl=impl_attn)
    outs = attn.W_o(out.reshape(B, T, attn.nh * attn.hd))
    attn.last_gate_mean = _gate_mean_batched(attn, cfg, T, B, gate_sigs, hons, nblks)
    return outs


def _gate_mean_batched(attn, cfg, T, B, gate_sigs, hons, nblks):
    if cfg.chunking != 'cosine_learnable':
        return None
    if not attn.need_reg:
        return None
    if gate_sigs is None or hons is None or T <= 1:
        return torch.zeros((), device=attn.B_pos_a.device if attn.B_pos_a is not None else torch.device('cpu'))
    gate_stack = torch.stack([gs[0] for gs in gate_sigs], 0)
    hon_stack = torch.stack(list(hons), 0).to(gate_stack.dtype)
    nblk_t = torch.tensor([float(nb - 1) for nb in nblks], device=gate_stack.device, dtype=gate_stack.dtype)
    shs = (gate_stack * hon_stack).sum(dim=1)
    per_row = (nblk_t + shs - shs.detach()) / T
    gm = per_row.sum() / B
    return gm


def _exp_blocks_fixed(n, block_size, device):
    import exp_lib as L
    return L.blocks_fixed(n, block_size, device)


# ============================================================================
# G0 自检：算子级 kernel/eager 数值对照 + 模型级前反向对照 + 微基准
# （驱动入口：python m0_supp.py kernel）
# ============================================================================

import contextlib


@contextlib.contextmanager
def kernel_env(mode):
    """临时设置 CSA_KERNEL（exp_lib 的分发按每次 forward 读取环境变量）。"""
    old = os.environ.get('CSA_KERNEL')
    if mode is None:
        os.environ.pop('CSA_KERNEL', None)
    else:
        os.environ['CSA_KERNEL'] = mode
    try:
        yield
    finally:
        if old is None:
            os.environ.pop('CSA_KERNEL', None)
        else:
            os.environ['CSA_KERNEL'] = old


def _rand_counts(T, min_block, max_block, gen):
    counts = []
    p = 0
    while p < T:
        c = int(torch.randint(int(min_block), int(max_block) + 1, (1,), generator=gen).item())
        c = max(1, min(c, T - p))
        counts.append(c)
        p += c
    return counts


def _host_bid(counts):
    """带 _pool_cpu 附件的宿主侧块表（分段器出口的同一挂载约定）。"""
    import numpy as np
    bid = torch.from_numpy(np.concatenate([np.full(c, i, dtype=np.int64) for i, c in enumerate(counts)]))
    bid._pool_cpu = (counts, max(counts), max(counts[:-1]) if len(counts) > 1 else 0)
    return bid


def _rel(a, b, pad_mask=None):
    """稳健相对误差：分母带张量自身尺度的下限（近零条目不放大噪声），
    再取逐元素最大值（pad 位置可屏蔽）。"""
    af = a.detach().to(torch.float64)
    bf = b.detach().to(torch.float64)
    if pad_mask is not None:
        af = af[pad_mask]
        bf = bf[pad_mask]
    if af.numel() == 0:
        return 0.0
    denom = bf.abs() + 1e-4 * float(bf.abs().max())
    if not bool(denom.max()):
        # reference is all-zero: identical-zero => rel err 0; any nonzero
        # absolute difference => rel err 1 (4-decimal floor, saturating).
        return 0.0 if not bool(af.abs().max()) else 1.0
    return float(((af - bf).abs() / denom).max())


def selfcheck_operators(device=None, dtype=torch.float32, impl='ref', seed=0, verbose=True):
    """三算子 kernel/eager 对照（同种子、同输入）。

    - pool：comp 值 + (Xa/Xb/Za/Zb/B_pos_a/B_pos_b) 梯度；
    - indexer：idx/valid 精确相等（int）+ soft 值 + (H/W_DQ/W_DK/W_w) 梯度；
    - attention：out 值 + (q/k_blk/v_blk/k_sw/v_sw/soft/sink) 梯度。
    返回 {name: max_rel_err}；任何 int 不一致直接断言失败。
    """
    import exp_lib as L
    device = device or torch.device('cpu')
    gen = torch.Generator().manual_seed(seed + 17)
    R, T = 3, 96
    min_block, max_block, overlap = 4, 8, 8
    kv, c_index, nIH = 32, 32, 4
    hd = c_index // nIH
    nh, d_head, d_model = 4, 16, 48
    topk, win = 8, 16
    results = {}

    def rp(*shape):
        return (torch.randn(*shape, generator=gen, device=device, dtype=torch.float32) * 0.5).to(dtype)

    # ---- 算子 1：pool ----
    counts_rows = [_rand_counts(T, min_block, max_block, gen) for _ in range(R)]
    Bm = max(len(c) for c in counts_rows)
    geom = PoolGeometry(counts_rows, overlap, T, device, max_block)
    Xa0, Xb0 = rp(R, T, kv), rp(R, T, kv)
    Za0, Zb0 = rp(R, T, kv), rp(R, T, kv)
    Bpa0, Bpb0 = rp(max_block + 1, kv), rp(max_block + 1, kv)
    leaves = [Xa0, Xb0, Za0, Zb0, Bpa0, Bpb0]
    gcrit = rp(R, Bm, kv)
    for tag, mode in (('eager', None), (impl, impl)):
        L_ = [t.detach().clone().requires_grad_(True) for t in leaves]
        if mode is None:
            outs = []
            for r in range(R):
                bid = _host_bid(counts_rows[r])
                c, _lt, _Bn = L.pool_variable_blocks(L_[0][r], L_[1][r], L_[2][r], L_[3][r], bid, L_[4], L_[5], overlap, monotonic=True)
                outs.append(c)
            comp = torch.stack([torch.nn.functional.pad(o, (0, 0, 0, Bm - o.shape[0])) for o in outs], 0)
        else:
            bid_full = torch.stack([_host_bid(c) for c in counts_rows], 0).to(device).to(torch.int32).contiguous()
            bid_full._pool_cpu = None
            comp, _lt, _B = pool_variable_blocks_kernel(L_[0], L_[1], L_[2], L_[3], bid_full, geom, L_[4], L_[5], impl=mode)
        (comp * gcrit).sum().backward()
        if mode is None:
            ref = ([comp] + [t.grad for t in L_])
        else:
            ker = ([comp] + [t.grad for t in L_])
    names = ['comp', 'gXa', 'gXb', 'gZa', 'gZb', 'gBpa', 'gBpb']
    for i, nm in enumerate(names):
        pad = None
        results[f'pool.{nm}'] = _rel(ker[i], ref[i])
    if verbose:
        print(f'[selfcheck:{impl}:{str(dtype)}] pool  ' + '  '.join(f'{k.split(".")[-1]}={v:.2e}' for k, v in results.items() if k.startswith('pool.')))
    # Multi-chunk regression: shrinking _POOL_CHUNK_BYTES exercises the b0>0
    # rebasing in the chunked backward (the default chunk size would keep this
    # tiny case single-chunk).
    _saved_chunk = _POOL_CHUNK_BYTES
    try:
        # two blocks per chunk: lane indices must be rebased into chunk
        # coordinates at b0 > 0, so a second block keeps the test discerning.
        globals()['_POOL_CHUNK_BYTES'] = 3 * (int(geom.max_len) + int(geom.W_b)) * 32 * 4 * 2
        L2_ = [t.detach().clone().requires_grad_(True) for t in leaves]
        bid2 = torch.stack([_host_bid(c) for c in counts_rows], 0).to(device).to(torch.int32).contiguous()
        bid2._pool_cpu = None
        comp2, _lt2, _B2 = pool_variable_blocks_kernel(L2_[0], L2_[1], L2_[2], L2_[3], bid2, geom, L2_[4], L2_[5], impl=impl)
        (comp2 * gcrit).sum().backward()
        ker2 = [comp2] + [t.grad for t in L2_]
    finally:
        globals()['_POOL_CHUNK_BYTES'] = _saved_chunk
    for i, nm in enumerate(names):
        results[f'pool_mc.{nm}'] = _rel(ker2[i], ref[i])
    if verbose:
        print(f'[selfcheck:{impl}:{str(dtype)}] pool(mc)  ' + '  '.join(f'{k.split(".")[-1]}={v:.2e}' for k, v in results.items() if k.startswith('pool_mc.')))

    # ---- 算子 2：indexer ----
    counts_i = [_rand_counts(T, 4, 8, gen) for _ in range(R)]
    Bm_i = max(len(c) for c in counts_i)
    geom_i = PoolGeometry(counts_i, None, T, device, max_block)
    H = rp(R, T, d_model)
    comp_kv = rp(R, Bm_i, kv)
    last_tok = geom_i.last_tok
    W_DQ = rp(d_model, c_index)
    W_DK = rp(kv, c_index)
    W_w = rp(nIH, d_model)
    gs = rp(R, T, topk)
    leaves_i = [H, W_DQ, W_DK, W_w]
    for tag, mode in (('eager', None), (impl, impl)):
        L_ = [t.detach().clone().requires_grad_(True) for t in leaves_i]
        if mode is None:
            idxs, softs, valids = [], [], []
            for r in range(R):
                _box = []
                _m, idx, soft = L.lightning_indexer(L_[0][r], comp_kv[r], last_tok[r].to(torch.int64), L_[1], L_[2], L_[3], nIH, topk, return_mask=False, out_valid=_box)
                idxs.append(idx)
                softs.append(soft)
                valids.append(_box[0])
            idx = torch.stack(idxs, 0)
            soft = torch.stack(softs, 0)
            valid_e = torch.stack(valids, 0)
            valid = valid_e
        else:
            qI = (L_[0].reshape(R * T, d_model) @ L_[1]).view(R, T, nIH, hd)
            w_idx = torch.nn.functional.linear(L_[0].reshape(R * T, d_model), L_[3]).view(R, T, nIH)
            kI = (comp_kv.reshape(R * Bm_i, kv) @ L_[2]).view(R, Bm_i, nIH, hd)
            idx, valid, soft = lightning_indexer_kernel(qI, kI, w_idx, last_tok, topk, impl=mode)
        (soft * (gs * valid)).sum().backward()
        if mode is None:
            ref_i = [idx, soft] + [t.grad for t in L_]
        else:
            ker_i = [idx, soft] + [t.grad for t in L_]
    assert torch.equal(ref_i[0].to(torch.int64), ker_i[0].to(torch.int64)), 'indexer idx 不一致（选择语义被破坏）'
    assert torch.equal(valid_e, valid), 'indexer valid 不一致（选择语义被破坏）'
    results['idxer.soft'] = _rel(ker_i[1] * valid, ref_i[1] * valid)
    for i, nm in enumerate(['gH', 'gWDQ', 'gWDK', 'gWw']):
        results[f'idxer.{nm}'] = _rel(ker_i[2 + i], ref_i[2 + i])
    if verbose:
        print(f'[selfcheck:{impl}:{str(dtype)}] idxer  ' + '  '.join(f'{k.split(".")[-1]}={v:.2e}' for k, v in results.items() if k.startswith('idxer.')))

    # ---- 算子 3：attention ----
    q0 = F.normalize(rp(R, T, nh, d_head), dim=-1)
    kb0 = F.normalize(rp(R, Bm_i, nh, d_head), dim=-1)
    vb0 = rp(R, Bm_i, nh, d_head)
    ksw0 = F.normalize(rp(R, T, nh, d_head), dim=-1)
    vsw0 = rp(R, T, nh, d_head)
    sink0 = rp(nh)
    soft0 = ker_i[1].detach().clone().requires_grad_(True)
    topk_idx = ker_i[0].detach().to(torch.int64)
    go = rp(R, T, nh, d_head)
    leaves_a = [q0, kb0, vb0, ksw0, vsw0, soft0]
    # 用非 1 的 scale（d_head**-0.5 惯例）对照——dq 反向必须与 dk 一样乘
    # scale；此处使用非 1 scale 的对照。
    asc = float(d_head) ** -0.5
    for tag, mode in (('eager', None), (impl, impl)):
        L_ = [t.detach().clone().requires_grad_(True) for t in leaves_a]
        sk = sink0.detach().clone().requires_grad_(True) if mode is not None else sink0.detach().clone().requires_grad_(True)
        if mode is None:
            outs = []
            for r in range(R):
                outs.append(L.gathered_attention(L_[0][r], L_[1][r], L_[2][r], topk_idx[r], last_tok[r].to(torch.int64), L_[3][r], L_[4][r], win, asc, soft=L_[5][r], sink_logits=sk, sel_valid=valid[r]))
            out = torch.stack(outs, 0)
        else:
            out = gathered_attention_kernel(L_[0], L_[1], L_[2], topk_idx.to(torch.int32), last_tok, L_[3], L_[4], win, asc, soft=L_[5], sink_logits=sk, sel_valid=valid, impl=mode)
        (out * go).sum().backward()
        if mode is None:
            ref_a = [out] + [t.grad for t in L_] + [sk.grad]
        else:
            ker_a = [out] + [t.grad for t in L_] + [sk.grad]
    names_a = ['out', 'gq', 'gkb', 'gvb', 'gksw', 'gvsw', 'gsoft', 'gsink']
    for i, nm in enumerate(names_a):
        results[f'attn.{nm}'] = _rel(ker_a[i], ref_a[i])
    if verbose:
        print(f'[selfcheck:{impl}:{str(dtype)}] attn  ' + '  '.join(f'{k.split(".")[-1]}={v:.2e}' for k, v in results.items() if k.startswith('attn.')))

    # ---- 算子 3 退化配置对照：soft=None（HCA 臂）与 sink=None（use_sink=False）----
    # 这两条路径的反向分解与默认配置不同（A 路单项 / 无 sink 列的 S 路），
    # 必须与 eager 逐步对齐。
    for cfg_tag, use_soft, use_sink in (('nosoft', False, True), ('nosink', True, False)):
        for tag, mode in (('eager', None), (impl, impl)):
            L_ = [t.detach().clone().requires_grad_(True) for t in leaves_a]
            sk = sink0.detach().clone().requires_grad_(True)
            sf = L_[5] if use_soft else None
            sk_arg = sk if use_sink else None
            if mode is None:
                outs = []
                for r in range(R):
                    outs.append(L.gathered_attention(L_[0][r], L_[1][r], L_[2][r], topk_idx[r], last_tok[r].to(torch.int64), L_[3][r], L_[4][r], win, 1.0,
                                                     soft=None if sf is None else sf[r], sink_logits=sk_arg, sel_valid=valid[r]))
                out = torch.stack(outs, 0)
            else:
                out = gathered_attention_kernel(L_[0], L_[1], L_[2], topk_idx.to(torch.int32), last_tok, L_[3], L_[4], win, 1.0,
                                                soft=sf, sink_logits=sk_arg, sel_valid=valid, impl=mode)
            (out * go).sum().backward()
            row = [out] + [t.grad for t in L_]
            if use_sink:
                row = row + [sk.grad]
            if mode is None:
                ref_d = row
            else:
                ker_d = row
        names_d = ['out', 'gq', 'gkb', 'gvb', 'gksw', 'gvsw'] + (['gsoft'] if use_soft else []) + (['gsink'] if use_sink else [])
        idxs_d = [0, 1, 2, 3, 4, 5] + ([6] if use_soft else []) + ([7] if use_sink else [])
        for i, nm in zip(idxs_d, names_d):
            results[f'attn_{cfg_tag}.{nm}'] = _rel(ker_d[i], ref_d[i])
        if verbose:
            print(f'[selfcheck:{impl}:{str(dtype)}] attn_{cfg_tag}  ' + '  '.join(f'{k.split(".")[-1]}={v:.2e}' for k, v in results.items() if k.startswith(f'attn_{cfg_tag}.')))
    return results


def selfcheck_model(variant='csa_dynamic', impl='ref', seed=0, device=None, T=64, B=2, verbose=True):
    """模型级对照：同种子构建 SmallGPT，eager vs 内核模式的前向 loss 与
    全部参数梯度的相对误差（配对纪律：同一配置指纹口径的两种实现）。"""
    import exp_lib as L
    if device is None:
        # triton 内核只接受 CUDA 指针——不显式传 device 时按后端选择。
        device = torch.device('cuda') if (impl == 'triton' and torch.cuda.is_available()) else torch.device('cpu')
    d, n_layers, n_heads, d_head = 64, 2, 4, 16
    vocab = 128
    ids = torch.randint(0, vocab, (B, T), generator=torch.Generator().manual_seed(seed + 5)).to(device)
    tgts = torch.randint(0, vocab, (B, T), generator=torch.Generator().manual_seed(seed + 6)).to(device)
    outs = {}
    grads = {}
    for mode in (None, impl):
        with kernel_env(mode):
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                cfgs = L.make_layer_cfgs(n_layers, variant)
                model = L.SmallGPT(vocab, d, n_layers, n_heads, d_head, T, cfgs).to(device)
            ce = model.ce_loss(ids, tgts, vocab)
            if model.comp_reg.requires_grad:
                ce = ce + 0.05 * model.comp_reg
            model.zero_grad(set_to_none=True)
            ce.backward()
            outs[mode] = float(ce.detach())
            grads[mode] = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    res = {'ce_abs_diff': abs(outs[None] - outs[impl])}
    worst = 0.0
    norm_num = 0.0
    norm_den = 0.0
    for n in grads[None]:
        if n in grads[impl]:
            a = grads[impl][n].detach().to(torch.float64)
            b = grads[None][n].detach().to(torch.float64)
            worst = max(worst, _rel(a, b))
            norm_num += (a - b).norm().item() ** 2
            norm_den += b.norm().item() ** 2
    res['grad_max_rel'] = worst
    res['grad_norm_rel'] = (norm_num ** 0.5) / max(norm_den ** 0.5, 1e-30)
    if verbose:
        print(f'[selfcheck:{impl}] model({variant})  ce eager={outs[None]:.6f} kernel={outs[impl]:.6f} (|Δ|={res["ce_abs_diff"]:.2e})  '
              f'grad_norm_rel={res["grad_norm_rel"]:.2e}  grad_max_rel={worst:.2e}')
    return res


def bench_attention(device, impl='triton', R=8, T=8192, nh=12, hd=64, topk=32, win=128,
                    Bm=None, iters=20, dtype=torch.float32, verbose=True):
    """稀疏注意力算子的微基准（判据口径见 docs/compute_plan.md §6）。

    计数口径：前向 = 4·R·T·(topk+win)·nh·hd（QK^T 与 PV 各 2·hd/泳道），
    反向按 2.5× 前向估计（统计重算 + dq + dk/dv 三遍）。有效速率 =
    (前向 + 2.5×前向) / 实测 (前向+反向) 墙钟——与稠密臂的口径可比
    （均按注意力项计数 / 墙钟）。
    """
    device = torch.device(device)
    gen = torch.Generator().manual_seed(1234)
    Bm = Bm or max(T // 4, topk + 1)
    q = F.normalize(torch.randn(R, T, nh, hd, generator=gen, device=device, dtype=dtype), dim=-1)
    k_blk = F.normalize(torch.randn(R, Bm, nh, hd, generator=gen, device=device, dtype=dtype), dim=-1)
    v_blk = torch.randn(R, Bm, nh, hd, generator=gen, device=device, dtype=dtype)
    k_sw = F.normalize(torch.randn(R, T, nh, hd, generator=gen, device=device, dtype=dtype), dim=-1)
    v_sw = torch.randn(R, T, nh, hd, generator=gen, device=device, dtype=dtype)
    lt = torch.randint(0, T, (R, Bm), generator=gen, device=device, dtype=torch.int32).clamp(max=T - 1)
    sel = torch.stack([torch.randperm(Bm, generator=gen, device=device)[:topk] for _ in range(R * T)]).view(R, T, topk).to(torch.int32)
    valid = torch.ones(R, T, topk, device=device, dtype=torch.bool)
    soft = torch.rand(R, T, topk, generator=gen, device=device, dtype=dtype).clamp_min(1e-4)
    sink = torch.zeros(nh, device=device, dtype=dtype)
    q = q.requires_grad_(True)
    k_blk = k_blk.requires_grad_(True)

    def step():
        out = gathered_attention_kernel(q, k_blk, v_blk, sel, lt, k_sw, v_sw, win, 1.0,
                                        soft=soft, sink_logits=sink, sel_valid=valid, impl=impl)
        g = torch.randn_like(out)
        out.backward(g)
        q.grad = None
        k_blk.grad = None

    for _ in range(3):
        step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    flops = (1.0 + 2.5) * 4.0 * R * T * (topk + win) * nh * hd
    tflops = flops / dt / 1e12
    if verbose:
        print(f'[bench:{impl}] R={R} T={T} Bm={Bm} topk={topk} w={win} nh={nh} hd={hd}  {dt * 1e3:.2f} ms/step(fwd+bwd)')
    return tflops
