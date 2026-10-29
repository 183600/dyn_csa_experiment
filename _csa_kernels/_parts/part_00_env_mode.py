#!/usr/bin/env python3
"""CSA/HCA 三算子融合内核工作包。

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

- 内核路径是**独立变体**，判据是四层等价口径：算子级**数值容差**（explib.m0.run_kernel 的预声明
  阈值：前向 ce <1e-5、梯度范数相对误差 <1e-4、逐条目阈值按后端分级）
  加训练级**预注册 margin 的 TOST 等价检验**（exp_lib.tost_paired）。
  「统计不可区分」**不是**通过条件——未发现显著差异 ≠ 等价（小样本下
  很大的实际差异也可能不显著）；也不承诺逐位一致；eager 路径本身一字
  未动。
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
design_notes §3.1）。G0 数值对照的驱动入口是
`python -m explib.m0 kernel`。
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
                "torch-reference backend (CPU-checkable)")
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
    # The prefix-sum difference below subtracts two GLOBAL prefixes.  In the
    # response dtype (often fp32) a large earlier segment rounds away the
    # small values of every later segment (e.g. 64 * 1e12 followed by 1000 *
    # 1.0 gives 0 instead of 1000), and the downcast to fp32 also discarded
    # the caller's requested precision.  Accumulate in float64 and cast only
    # the final segment sums: the stable sort still fixes the addition order,
    # so the reduction stays deterministic, while long-context dk/dv sums no
    # longer lose short segments as the number of blocks grows.
    _acc = torch.complex128 if dtype.is_complex else torch.float64
    for values, out in zip(values_list, outs):
        if values is None:
            continue
        val_s = values.index_select(0, order).to(_acc)
        csum = val_s.cumsum(dim=0)
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
    归约）。注意力反向的 dk/dv 键不含头维：同一批键对每个头给出同一
    排列、同一批段边界，因此 argsort 与段定位全部头共用一次，段内累加
    顺序与逐头排序形式逐元素一致。"""
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


def _seg_apply(seg, values, out_flat):
    """按共享段结构把一个头的值表归约进逐头扁平行：值经同一排列
    gather 后 cumsum，段和写入 seg_id 行（调用方保证 out_flat 已清零；
    未被任何键覆盖的行保持零，与「零初始化全头缓冲后按 seg_id·nh+h
    写入、末尾一次 cast」逐头 cast 的参考形式一致）。段和用 float64
    累加后再写回 out_flat 的 dtype：共享的前缀和会被大段稀释，fp32 下
    后续小段会变成 0；累加顺序仍由同一稳定排序固定，所以结果确定。"""
    order, seg_end, seg_id_k, keep = seg
    _acc = (torch.complex128 if out_flat.dtype.is_complex
            else torch.float64)
    val_s = values.index_select(0, order).to(_acc)
    csum = val_s.cumsum(dim=0)
    hi = csum.index_select(0, seg_end)
    lo = torch.zeros_like(hi)
    if seg_end.numel() > 1:
        lo[1:] = csum.index_select(0, seg_end[:-1])
    seg_sum = hi - lo
    out_flat.index_copy_(0, seg_id_k, seg_sum[keep].to(out_flat.dtype))


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
        # 8 张几何表打包进一条宿主缓冲、一次 H2D。  int32 is the historical
        # transport width and stays byte-identical while every table value (and
        # the `starts + offset` sums built later) fits; beyond that the same
        # exact integers use int64 instead of wrapping to negative indices.
        # The bound is half the int32 range so a pairwise sum cannot overflow.
        _safe_i32 = (2 ** 31 - 2) // 2
        _need_i64 = max(int(T), int(W_b), int(counts_np.max(initial=0))) > _safe_i32
        _idx_dtype = np.int64 if _need_i64 else np.int32
        flat = np.empty(7 * R * Bm + R, dtype=_idx_dtype)
        for _i, _a in enumerate((counts_np, ends_np, starts_np, last_np,
                                 ov_start_np, ov_len_np, starts_prev_np)):
            flat[_i * R * Bm:(_i + 1) * R * Bm] = _a.reshape(-1)
        flat[7 * R * Bm:] = nblk_np
        import exp_lib as L
        _tdtype = torch.int64 if _need_i64 else torch.int32
        # Build outside inference mode: the same geometry object is cached and
        # later saved for autograd by the indexer/attention wrappers.  An
        # inference tensor created by an eval pass cannot be saved for a
        # subsequent training backward.
        with torch.inference_mode(False):
            staged, commit = L._h2d_stage((int(flat.size),), _tdtype, device)
            if staged is not None:
                staged.numpy()[:] = flat
                _buf = commit()
            else:
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
        self.idx_dtype = _idx_dtype
        # `max_block=None` is the no-cap configuration: the counts table already
        # encodes every forced cut, so the geometry is well defined.  Keep the
        # declaration as None instead of crashing on int(None) (the no-cap
        # open-block arms are a supported scale configuration).
        self.max_block = None if max_block is None else int(max_block)
        self._b_off = None
        self._bposb_rows = None
        self._ga_cache = None
        self._g_b = None
        self._i64 = None

    def gather_geom(self):
        """A 泳道的纯几何 gather 常量 (g_a [R,Bm,L] int64, av_all [R,Bm,L,1]
        bool)——内容是 geom 的纯整型函数，只读消费；缓存论证与 b_off /
        _GEOM_CACHE 同。"""
        if self._ga_cache is None:
            # The lazy caches can first be filled by an inference-mode eval;
            # keep them normal tensors so a later training pass can save them.
            with torch.inference_mode(False):
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
            with torch.inference_mode(False):
                jB = torch.arange(self.W_b, device=self.starts.device)
                self._g_b = (self.ov_start.to(torch.int64)[:, :, None] + jB[None, None, :]).clamp(max=int(self.T) - 1)
        return self._g_b

    def i64_views(self):
        """geom 标量表的 int64 形式（int64 原型的 .to 是 no-op，非原型的
        cast 结果在此缓存）。"""
        if self._i64 is None:
            with torch.inference_mode(False):
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
        with torch.inference_mode(False):
            j = torch.arange(self.W_b, device=self.starts.device, dtype=torch.int32)
            raw = self.ov_start[:, :, None] - self.starts_prev[:, :, None] + j[None, None, :]
            self._b_off = raw.clamp(max=int(bposb_rows) - 1).long()
            self._bposb_rows = int(bposb_rows)
        return self._b_off


_GEOM_CACHE = {}
_GEOM_CACHE_CAP = 16
_GEOM_CACHE_BYTES = [0]
_GEOM_CACHE_BUDGET = 256 << 20
_ARANGE_CACHE = {}
_ARANGE_CACHE_CAP = 8
_ARANGE_CACHE_BYTES = [0]
_ARANGE_CACHE_BUDGET = 512 << 20


def _arange_nbytes(t):
    # Physical size of `t`: an expand view's storage is smaller than its
    # logical shape, and the ref backend keeps that view.
    try:
        return int(t.untyped_storage().nbytes())
    except Exception:
        return int(t.numel()) * int(t.element_size())


def _full_arange_evict(need=0):
    while _ARANGE_CACHE and (len(_ARANGE_CACHE) >= _ARANGE_CACHE_CAP or
                             _ARANGE_CACHE_BYTES[0] + need > _ARANGE_CACHE_BUDGET):
        _k0 = next(iter(_ARANGE_CACHE))
        _t0 = _ARANGE_CACHE.pop(_k0)
        _ARANGE_CACHE_BYTES[0] -= _arange_nbytes(_t0)


def _geom_nbytes(geom):
    # Charge the real (physical) storage of every integer table a geometry
    # entry holds, including its lazily-built gather constants.  The eight
    # scalar tables are slices of ONE packed `_buf`, and the int64 views can
    # share storage too, so charge each underlying storage exactly once
    # rather than once per view.
    seen = {}
    def _charge(t):
        if t is None:
            return
        try:
            st = t.untyped_storage()
            key = (str(st.device), int(st.data_ptr()), int(st.nbytes()))
            if key not in seen:
                seen[key] = int(st.nbytes())
        except Exception:
            key = ('logical', id(t))
            if key not in seen:
                seen[key] = int(t.numel()) * int(t.element_size())
    for name in ('counts', 'ends', 'starts', 'last_tok', 'ov_start', 'ov_len',
                 'starts_prev', 'n_blocks', '_b_off', '_g_b'):
        _charge(getattr(geom, name, None))
    ga = getattr(geom, '_ga_cache', None)
    if ga is not None:
        for t in ga:
            _charge(t)
    i64 = getattr(geom, '_i64', None)
    if i64:
        for t in i64.values():
            _charge(t)
    return sum(seen.values())


def _geom_evict(need=0):
    while _GEOM_CACHE and (len(_GEOM_CACHE) >= _GEOM_CACHE_CAP or
                           _GEOM_CACHE_BYTES[0] + int(need) > _GEOM_CACHE_BUDGET):
        _k0 = next(iter(_GEOM_CACHE))
        _e0 = _GEOM_CACHE.pop(_k0)
        _GEOM_CACHE_BYTES[0] -= int(_e0[2])


def _pool_geometry_cached(counts_rows, overlap, T, device, max_block):
    key = (tuple(id(c) for c in counts_rows),
           None if overlap is None else int(overlap), int(T),
           str(device), None if max_block is None else int(max_block))
    ent = _GEOM_CACHE.get(key)
    if ent is not None:
        refs, geom, charged = ent
        if len(refs) == len(counts_rows) and all(a is b for a, b in zip(refs, counts_rows)):
            cur = _geom_nbytes(geom)
            if cur != charged:
                _GEOM_CACHE_BYTES[0] += cur - charged
                _GEOM_CACHE[key] = (refs, geom, cur)
                if _GEOM_CACHE_BYTES[0] > _GEOM_CACHE_BUDGET:
                    _geom_evict(0)
            return geom
        _GEOM_CACHE.pop(key, None)
        _GEOM_CACHE_BYTES[0] -= int(charged)
    geom = PoolGeometry(counts_rows, overlap, T, device, max_block)
    need = _geom_nbytes(geom)
    if need <= _GEOM_CACHE_BUDGET:
        _geom_evict(need)
        _GEOM_CACHE[key] = (tuple(counts_rows), geom, need)
        _GEOM_CACHE_BYTES[0] += need
    return geom


def _full_arange_cached(B, T, Bm, device, contiguous=True):
    # `contiguous=False` returns a stride-0 expand view: every row holds the
    # same exact integers, which is all the torch-reference backend reads
    # (advanced indexing / gathers / comparisons).  The triton attention
    # kernels interpret the table by lane index.  Callers that need a dense
    # table ask for `contiguous=True`.
    key = (int(B), int(T), int(Bm), str(device), bool(contiguous))
    out = _ARANGE_CACHE.get(key)
    if out is None:
        # Cache-normal tensor even when first filled from eval inference mode;
        # FusedSparseAttention saves this table for a later training backward.
        with torch.inference_mode(False):
            base = torch.arange(Bm, device=device, dtype=torch.int32)[None, None, :].expand(B, T, Bm)
            out = base.contiguous() if contiguous else base
        if not contiguous:
            # Shape (B,T,Bm) with stride (0,0,1) is the row-constant arange
            # table; mark it so the attention launcher can recognise it.
            out._idx_arange = True
        _need = _arange_nbytes(out)
        _full_arange_evict(_need)
        if _need <= _ARANGE_CACHE_BUDGET:
            _ARANGE_CACHE[key] = out
            _ARANGE_CACHE_BYTES[0] += _need
    return out


# ============================================================================
# 算子 1：块压缩 pool（`pool_variable_blocks` / `pool_blocks_single` 的内核路径）
# ============================================================================

_POOL_CHUNK_BYTES = 64 << 20
_POOL_CHUNK_HARD_BYTES = 512 << 20


def _pool_chunk_b(R, Ltot, c, temps=1):
    per_block = max(1, int(R) * int(Ltot) * int(c) * 4)
    ch = max(1, min(int(_POOL_CHUNK_BYTES) // per_block, 4096))
    return max(1, min(ch, int(_POOL_CHUNK_HARD_BYTES) // max(1, int(temps) * per_block)))


def _gather_rows(x, idx):
    """x [R, S, D] 按 idx [R, ..., k]（int64）在 S 维 gather，保留尾维。"""
    R, _S, D = x.shape
    flat = idx.reshape(R, -1)
    out = x.gather(1, flat.unsqueeze(-1).expand(R, flat.shape[1], D))
    return out.view(*idx.shape, D)


def _pool_fwd_ref(Xa, Xb, Za, Zb, geom, B_pos_a, B_pos_b):
    """池化前向的 torch 参考实现（按块分块）。

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
    # 每个块列都被下方的分块循环恰好写一次，故直接 empty。
    comp = torch.empty((R, Bm, c), device=dev, dtype=dtype)
    jA = torch.arange(L, device=dev)
    g_a, av_all = geom.gather_geom()
    pos_a = jA.clamp(max=int(B_pos_a.shape[0]) - 1)
    # Within the table the gather index is arange(L).
    Bp_a = (B_pos_a[:L] if L <= B_pos_a.shape[0] else B_pos_a[pos_a]).to(dtype)
    if has_b:
        jB = torch.arange(W, device=dev)
        g_b = geom.gather_geom_b()
        b_off = geom.b_off(B_pos_b.shape[0]).to(dev)
        Bp_b = B_pos_b[b_off].to(dtype)
    ch = _pool_chunk_b(R, L + (W if has_b else 0), c, temps=6)
    for b0 in range(0, Bm, ch):
        b1 = min(b0 + ch, Bm)
        av = av_all[:, b0:b1]
        # 这些 logit 张量都是本轮新算出的临时量（无别名），fill 原地进行——
        # 值与出位版逐位相同。无效泳道随即被 masked_fill_ 覆盖为 -inf
        # （被覆盖 lanes 的最终值与乘前无关，含 NaN/inf 边角——masked_fill
        # 一律改写）。
        logit_a = _gather_rows(Za, g_a[:, b0:b1]) + Bp_a[None, None]
        logit_a.masked_fill_(~av, _NEG_INF)
        if has_b:
            vm = jB[None, None, :, None] < geom.ov_len.to(torch.int64)[:, b0:b1, None, None]
            logit_b = _gather_rows(Zb, g_b[:, b0:b1]) + Bp_b[:, b0:b1]
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
    # 这些张量仍在（空 gather/空求和），拿到的是零梯度张量而不是 None；
    # 零张量与 eager 的更新逐步一致，退化几何下两模式不得分流。
    gXb = torch.zeros_like(Xb) if (need_xb and Xb is not None) else None
    gZb = torch.zeros_like(Zb) if (need_zb and Zb is not None) else None
    gBpa = torch.zeros_like(B_pos_a) if need_bpa else None
    jA = torch.arange(L, device=dev)
    g_a, av_all = geom.gather_geom()
    pos_a = jA.clamp(max=int(B_pos_a.shape[0]) - 1)
    # Within the table the gather index is arange(L).
    Bp_a = (B_pos_a[:L] if L <= B_pos_a.shape[0] else B_pos_a[pos_a]).to(dtype)
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
    _bpb_rows = int(B_pos_b.shape[0]) if need_bpb else 0
    # Work in at least fp32; fp64 inputs must not be silently downcast (finite
    # products can overflow fp32 and turn inf-inf into NaN gradients).
    _acc = torch.promote_types(dtype, torch.float32)
    gBpb32 = torch.zeros((_bpb_rows, c), device=dev, dtype=_acc) if need_bpb else None
    gBpb_tmp = torch.zeros((_bpb_rows, c), device=dev, dtype=_acc) if need_bpb else None
    ch = _pool_chunk_b(R, L + (W if has_b else 0), c, temps=8)
    # 单 chunk（ch >= Bm）时 in_ch / b2 的 chunk 范围条件恒真（geom 与 bid
    # 同源——Bm 由同一分段表的 counts 给出，bid 值恒 < Bm）：where(全真,
    # x, 0) 逐位返回 x（含 ±0/NaN）。
    single_chunk = ch >= Bm
    # 广播用的零标量只读，循环外建立。
    zero_acc = torch.zeros((), device=dev, dtype=_acc)
    for b0 in range(0, Bm, ch):
        b1 = min(b0 + ch, Bm)
        av = av_all[:, b0:b1]
        # 这些 logit 张量都是本轮新算出的临时量（无别名），fill 原地进行——
        # 值与出位版逐位相同。无效泳道随即被 masked_fill_ 覆盖为 -inf
        # （被覆盖 lanes 的最终值与乘前无关，masked_fill 一律改写）。
        logit_a = _gather_rows(Za, g_a[:, b0:b1]) + Bp_a[None, None]
        logit_a.masked_fill_(~av, _NEG_INF)
        if has_b:
            vm = jB[None, None, :, None] < geom.ov_len.to(torch.int64)[:, b0:b1, None, None]
            logit_b = _gather_rows(Zb, g_b[:, b0:b1]) + Bp_b[:, b0:b1]
            logit_b.masked_fill_(~vm, _NEG_INF)
            WinZ = torch.cat([logit_a, logit_b], dim=2)
        else:
            WinZ = logit_a
        sc = torch.nan_to_num(torch.softmax(WinZ, dim=2)).to(_acc)
        sc_a, sc_b = sc[:, :, :L], (sc[:, :, L:] if has_b else None)
        dc = dcomp[:, b0:b1].to(_acc)
        gsc_a = dc[:, :, None] * _gather_rows(Xa, g_a[:, b0:b1]).to(_acc)
        if has_b:
            gsc_b = dc[:, :, None] * _gather_rows(Xb, g_b[:, b0:b1]).to(_acc)
            Srow = (gsc_a * sc_a).sum(dim=2) + (gsc_b * sc_b).sum(dim=2)
            glogit_b = sc_b * (gsc_b - Srow[:, :, None])
        else:
            Srow = (gsc_a * sc_a).sum(dim=2)
            glogit_b = None
        glogit_a = sc_a * (gsc_a - Srow[:, :, None])
        if need_xa or need_za:
            in_ch = None if single_chunk else (bid_l >= b0) & (bid_l < b1)
            sel = bid_l.clamp(b0, b1 - 1)
            jA_t = (t_idx - starts_l.gather(1, sel)).clamp(0, L - 1)
            dc_sel = dcomp.gather(1, sel[:, :, None].expand(R, T, c)).to(_acc)
            if need_xa:
                # sc_a/glogit_a are chunk-local ([R, b1-b0, L, c]); rebase the
                # global clamped block id into chunk coordinates before the
                # lane gather (starts_l/dcomp stay global-indexed above).
                sc_a_sel = _take_lane(sc_a, r_idx, sel - b0, jA_t)
                gXa += (sc_a_sel * dc_sel).to(dtype) if single_chunk else torch.where(in_ch[:, :, None], sc_a_sel * dc_sel, zero_acc).to(dtype)
            if need_za:
                gl_a_sel = _take_lane(glogit_a, r_idx, sel - b0, jA_t)
                gZa += gl_a_sel.to(dtype) if single_chunk else torch.where(in_ch[:, :, None], gl_a_sel, zero_acc).to(dtype)
        if has_b and (need_xb or need_zb):
            b2 = (bid_l + 1).clamp(0, Bm - 1)
            j2 = t_idx - ov_start_l.gather(1, b2)
            ok2 = (j2 >= 0) & (bid_l + 1 < nblk_l) & (bid_l + 1 < Bm)
            if not single_chunk:
                ok2 = ok2 & (b2 >= b0) & (b2 < b1)
            b2c = b2.clamp(b0, b1 - 1)
            j2c = j2.clamp(0, W - 1)
            dc2 = dcomp.gather(1, b2c[:, :, None].expand(R, T, c)).to(_acc)
            if need_xb:
                sc_b_sel = _take_lane(sc_b, r_idx, b2c - b0, j2c)
                gXb += torch.where(ok2[:, :, None], sc_b_sel * dc2, zero_acc).to(dtype)
            if need_zb:
                gZb += torch.where(ok2[:, :, None], _take_lane(glogit_b, r_idx, b2c - b0, j2c), zero_acc).to(dtype)
        if need_bpa:
            gsum = glogit_a.sum(dim=(0, 1))
            # 前向把 pos_a 钳到 B_pos_a 末行（见上方 clamp）：超程泳道的
            # 梯度必须按同一钳位语义累加进末行，而不是按形状截断或报错。
            _rows = min(int(gsum.shape[0]), int(gBpa.shape[0]))
            gBpa[:_rows] += gsum[:_rows].to(B_pos_a.dtype)
            if _rows < int(gsum.shape[0]):
                gBpa[_rows - 1] += gsum[_rows:].sum(dim=0).to(B_pos_a.dtype)
        if need_bpb and has_b:
            _k = b_off[:, b0:b1].reshape(-1)
            _v = glogit_b.reshape(-1, c)
            _seg = _seg_structure(_k, _bpb_rows)
            gBpb_tmp.zero_()
            _seg_apply(_seg, _v, gBpb_tmp)
            gBpb32 += gBpb_tmp
    gBpb = None
    if need_bpb:
        # 同 gXb/gZb：W_b == 0 时无任何键，段和为全零——与 eager
        # 空泳道图的零梯度一致（HCA 把同一 B_pos 作 a/b 两参传入，零项
        # 不改变 autograd 对同一张量的梯度求和）。fp32 累加器按上述
        # 固定顺序求和，末尾一次 cast——与 _det_scatter_add 同口径。
        gBpb = gBpb32.to(B_pos_b.dtype)
    return gXa, gXb, gZa, gZb, gBpa, gBpb


def _take_lane(lane4, r_idx, blk_idx, lane_idx):
    """[R, B, L, c] 泳道张量按 (r, blk_idx[r,t], lane_idx[r,t]) 取 [R, T, c]——
    纯 gather，无累加（A 泳道双射 / B 泳道至多一处的两处用途都满足）。"""
    R, B, L, c = lane4.shape
    row = (torch.arange(R, device=r_idx.device)[:, None] * B
           + blk_idx.clamp(0, B - 1)).reshape(-1)
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
    if Xa.shape[0] == 0 or Xa.shape[1] == 0:
        # Same empty-sequence contract as the eager path: no lanes to pool,
        # hence zero blocks (do not run the gather geometry, whose clamp to
        # T-1 would index an empty time dimension with -1).
        return (Xa.new_empty((Xa.shape[0], 0, Xa.shape[-1])),
                torch.empty((Xa.shape[0], 0), dtype=torch.int32, device=Xa.device),
                0)
    impl = impl or _op_impl('POOL')
    comp = FusedPool.apply(impl, Xa, Xb, Za, Zb, bid, geom, B_pos_a, B_pos_b)
    return comp, geom.last_tok, geom.Bm


def pool_blocks_single_kernel(X, Z, bid, geom, B_pos, impl=None):
    """`exp_lib.pool_blocks_single` 的内核路径（HCA：仅 A 泳道）。"""
    if X.shape[0] == 0 or X.shape[1] == 0:
        return (X.new_empty((X.shape[0], 0, X.shape[-1])),
                torch.empty((X.shape[0], 0), dtype=torch.int32, device=X.device),
                0)
    impl = impl or _op_impl('POOL')
    comp = FusedPool.apply(impl, X, None, Z, None, bid, geom, B_pos, B_pos)
    return comp, geom.last_tok, geom.Bm


# ============================================================================
# 算子 2：索引打分 + top-k 选择（`lightning_indexer` + `_indexer_selection`）
# ============================================================================

def _stable_topk_desc(x, k):
    """Stable descending top-k of the last dim, tie = earliest column.

    Returns (values, indices) with exactly the first k columns of
    torch.sort(x, dim=-1, descending=True, stable=True): the k-th value is
    found with topk, all strictly larger entries are kept and the remaining
    slots are filled from equal entries in column order (the stable tie
    rule), then the k candidates are ordered by a stable argsort.
    """
    B = x.shape[-1]
    kk = min(int(k), B)
    if kk <= 0:
        empty = x[..., :0]
        return empty, torch.empty(x.shape[:-1] + (0,), dtype=torch.long, device=x.device)
    if kk >= B:
        val, idx = torch.sort(x, dim=-1, descending=True, stable=True)
        return val, idx
    # The rank/scatter construction below needs a finite k-th value: with NaN
    # in a row, comparisons against kth are all false, rank slot 0 is never
    # written and `buf` keeps uninitialised indices (a crash or garbage
    # selection depending on allocator contents).  Delegate the non-finite
    # case to sort, whose stable NaN ordering is the contract this helper
    # documents ("exactly the first k columns of torch.sort").
    if not bool(torch.isfinite(x).all()):
        val, idx = torch.sort(x, dim=-1, descending=True, stable=True)
        return val[..., :kk], idx[..., :kk]
    lead = x.shape[:-1]
    m = 1
    for _s in lead:
        m *= int(_s)
    dev = x.device
    flat = x.reshape(m, B)
    # Clone the single k-th column before discarding the source table.  The
    # two branches below produce the same k-th value.
    if kk * 4 >= B:
        kth = torch.kthvalue(flat, B - kk + 1, dim=1).values.unsqueeze(1)
    else:
        kth = torch.topk(flat, kk, dim=1, sorted=True).values[:, -1].unsqueeze(1).clone()
    # Same candidate construction as exp_lib._rank_blocks: strictly greater
    # entries plus the earliest equal-to-kth entries.  Prefix ranks use int32
    # (B < 2^31), the same exact integers; the sentinel is filled in place.
    # torch < 2.14 requires an int64 scatter index, so only the index is
    # widened below; `cols` stays int32 to match the destination dtype
    # (scatter_ also requires src and self to share a dtype).
    gt = flat > kth
    eq = flat == kth
    need = (kk - gt.sum(dim=1)).clamp_min(0).to(torch.int32)
    eq_cum = torch.cumsum(eq, dim=1, dtype=torch.int32)
    sel = gt | (eq & (eq_cum <= need[:, None]))
    del eq_cum
    rank = torch.cumsum(sel, dim=1, dtype=torch.int32)
    rank -= 1
    rank.masked_fill_(~sel, kk)
    cols = torch.arange(B, device=dev, dtype=torch.int32)[None, :].expand(m, B)
    buf = torch.empty(m, kk + 1, dtype=torch.int32, device=dev)
    buf.scatter_(1, rank.long(), cols)
    cand = buf[:, :kk].long()
    order = torch.argsort(flat.gather(1, cand), dim=1, stable=True, descending=True)
    cand = cand.gather(1, order)
    cand = cand.reshape(lead + (kk,))
    return x.gather(-1, cand), cand

def _indexer_fwd_ref(qI, kI, w_idx, last_tok, topk, need_soft=True):
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
    # 工作精度 = max(dtype, fp32)（fp32/bf16/fp16 输入取 fp32；fp64 输入
    # 全 fp64，前向/反向同一精度合同）。升型整体只做一次（逐片 cast 的值
    # 与整张 cast 后取片逐位一致）。
    acc = torch.promote_types(dtype, torch.float32)
    qIf = qI if qI.dtype == acc else qI.to(acc)
    kIf = kI if kI.dtype == acc else kI.to(acc)
    wf = w_idx if w_idx.dtype == acc else w_idx.to(acc)
    k_out = max(0, min(int(topk), Bm))
    idx = torch.empty((R, T, k_out), device=dev, dtype=torch.int32)
    valid = torch.empty((R, T, k_out), device=dev, dtype=torch.bool)
    # Soft is a training-only prior (the straight-through S path is dead
    # without gradients).
    soft = torch.empty((R, T, k_out), device=dev, dtype=acc) if need_soft else None
    # 逐行独立的 (r,t) 归约（mask/lse/sort/topk）：按 t 分块与整幅计算的
    # 每行值逐位一致（同 kernel、同归约顺序），中间张量按 t 分块。
    t_chunk = max(1, min(T, (256 << 20) // max(R * Bm * 32, 1)))
    for t0 in range(0, T, t_chunk):
        t1 = min(t0 + t_chunk, T)
        causal_c = t_idx[:, t0:t1] > lt64[:, None, :]
        scores = torch.zeros((R, t1 - t0, Bm), device=dev, dtype=acc)
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
            # 临时量，整块域两者是同一批行。
            scores.add_(s_i * wf[:, t0:t1, i:i + 1])
        scores = scores.to(dtype)
        # where 的标量分支与 full_like/zeros_like 逐位同值（标量按 dtype
        # 精确表示、元素级选择同一比较结果）。
        if need_soft:
            masked_lse = torch.where(causal_c, scores, _NEG_INF)
            lse = torch.logsumexp(masked_lse, dim=-1)
            # Only lse survives.
            del masked_lse
        cf = causal_c & torch.isfinite(scores)
        work = torch.where(cf, scores, _NEG_INF)
        # pad 恒为块 0（eager 形式；池化 last_tok 单调时与「首个因果块」
        # 同值——那种形式的首个因果块必为块 0）。
        pad_blk = torch.zeros((R, t1 - t0), device=dev, dtype=torch.int64)
        # 逐轮「首达最大值」提取 == 稳定降序排序的前缀（tie 取最早下标，
        # 见上文 docstring）。
        top_val, top_idx = _stable_topk_desc(work, k_out)
        # 与 eager 的 keep = cf.gather(...)（cf = causal & isfinite）同口径：
        # 有限的极端负分是有效选择，只有 -inf 填充位无效。
        has = top_val > float('-inf')
        valid[:, t0:t1] = has
        idx[:, t0:t1] = torch.where(has, top_idx, pad_blk[:, :, None]).to(torch.int32)
        if need_soft:
            sv = top_val - lse[:, :, None]
            sv = torch.where(has & torch.isfinite(lse)[:, :, None],
                             torch.exp(sv), 0.0)
            soft[:, t0:t1] = torch.where(sv == sv, sv, 0.0)
    return idx, valid, (soft.to(dtype) if need_soft else None)


def _indexer_bwd_ref(gs, idx, valid, soft, qI, kI, w_idx, last_tok,
                     need_dq, need_dk, need_dw, t_chunk=None):
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
    if t_chunk is None:
        # t_chunk 按每 chunk 的 [R,tc,Bm] 级临时量（Mk、s_list、s_all、
        # masked、e、ds 及逐元素中间量）确定。
        # chunk 只划分独立的 (r,t) 行（dkI 的跨 chunk 累加分组属低阶位
        # 口径，与 CSA_KERNEL_IDX_CHUNK 覆盖同理）。
        t_chunk = max(1, min(T, (192 << 20) // max(int(R) * int(Bm) * 4 * (int(nIH) + 8), 1)))
    t_idx = torch.arange(T, device=dev)[None, :, None]
    # 同 _indexer_fwd_ref：causal 按 t 分块现算（同一整数比较），全幅
    # [R, T, Bm] 掩码不物化。
    lt64 = last_tok.to(torch.int64)
    # 工作精度 = max(dtype, fp32)：fp32 / bf16 / fp16 输入取 fp32，
    # fp64 输入全 fp64，前向/反向同一精度合同。
    acc = torch.promote_types(dtype, torch.float32)
    gsf = gs.to(acc)
    softf = soft.to(acc)
    Gt = (gsf * softf).sum(dim=-1)
    if idx.numel():
        # 一次升型供全循环复用（同一批整数）：pad 槽位（重复的 pad_blk）投到
        # 哑列 Bm——scatter_ 对重复下标的胜者未定义，但哑列上全部候选值都是
        # 0，且有效槽位彼此互不相同——切片后即精确。
        idx_l = idx.to(torch.int64)
    dqI = torch.zeros_like(qI) if need_dq else None
    dkI = torch.zeros_like(kI) if need_dk else None
    dw = torch.zeros_like(w_idx) if need_dw else None
    # kI / w_idx 与 t 分块无关：工作精度升型在分块循环外只做一次（整型
    # 内容不变，逐位一致）。
    kIf = kI if kI.dtype == acc else kI.to(acc)
    wf = w_idx if w_idx.dtype == acc else w_idx.to(acc)
    for t0 in range(0, T, t_chunk):
        t1 = min(t0 + t_chunk, T)
        # Mk 随 t 分块构建（全幅 [R, T, Bm+1] 不物化）：逐 chunk 的 scatter
        # 键/值就是全幅形式对应行的切片，逐行无碰撞——切片后的值逐位一致。
        Mk = torch.zeros((R, t1 - t0, Bm + 1), device=dev, dtype=acc)
        if idx.numel():
            dst = torch.where(valid[:, t0:t1], idx_l[:, t0:t1], Bm)
            Mk.scatter_(2, dst, (gsf[:, t0:t1] * softf[:, t0:t1]) * valid[:, t0:t1].to(acc))
        Mk = Mk[:, :, :Bm]
        q_c = qI[:, t0:t1].to(acc)
        s_list = []
        for i in range(nIH):
            # 原地 relu：后面用到的 (s_i > 0) 与 relu(s_i)>0 逐位同真，
            # dw 用的 relu(s_i) 即本张量自身。
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
        # 工作精度累加器（与前向的 scores 容器同口径——即使 s_i 已按
        # dtype 舍入，跨头累加也在工作精度进行）。
        s_all = torch.zeros((R, t1 - t0, Bm), device=dev, dtype=acc)
        for i in range(nIH):
            s_all.add_(s_list[i] * wf[:, t0:t1, i:i + 1])
        if dtype != torch.float32:
            # 前向末段 einsum 的输出舍入：lse/e 必须从前向实际选择的
            # 同一批舍入分数重算。
            s_all = s_all.to(dtype)
        causal_c = t_idx[:, t0:t1] > lt64[:, None, :]
        # 同 _indexer_fwd_ref：标量分支与 full_like/zeros_like 形式逐位一致。
        masked = torch.where(causal_c, s_all, _NEG_INF)
        lse = torch.logsumexp(masked, dim=-1)
        row_ok = torch.isfinite(lse)
        e = torch.where(row_ok[:, :, None], torch.exp(masked - lse[:, :, None]), 0.0)
        ds = Mk - e * Gt[:, t0:t1, None]
        ds = torch.where(causal_c & torch.isfinite(s_all), ds, 0.0)
        # The per-chunk normalizer/selector tables are dead once ds exists.
        del masked, lse, row_ok, e, Mk, s_all
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


_SHAPE_FALLBACK_WARNED = set()


def _warn_shape_fallback(op, detail):
    key = (str(op), str(detail))
    if key in _SHAPE_FALLBACK_WARNED:
        return
    _SHAPE_FALLBACK_WARNED.add(key)
    print(f'[kernel] {op}: {detail} — falling back to the torch-reference backend for this shape '
          '(the reference twin implements the same algorithm; values are the same up to the '
          'declared backend tolerance)')


def _indexer_impl_for_shape(impl, kI):
    # The scoring kernel already loops over block columns during the call, and
    # the runner-up path below streams the top-k over block chunks.  Long-context
    # shapes (Bm can be tens of thousands) therefore stay on the Triton
    # scorer instead of silently degrading to the torch reference twin.
    if impl == 'triton' and kI.dtype == torch.float64:
        # The Triton scorer loads q/k/w into fp32 lanes, while the reference
        # twin promotes to fp64.  A finite fp64 value can overflow fp32 and an
        # fp64 near-tie can change the selected block, so fail closed to the
        # reference implementation instead of silently changing the function.
        _warn_shape_fallback(
            'indexer', 'fp64 inputs require the torch-reference backend '
                       '(the Triton scorer computes in fp32)')
        return 'ref'
    return impl


class FusedIndexer(torch.autograd.Function):
    """索引打分 + top-k 选择的可微封装：idx/valid 不可微（离散选择），
    soft 的梯度经 _indexer_bwd_ref 回传到 qI / kI / w。"""

    @staticmethod
    def forward(ctx, impl, qI, kI, w_idx, last_tok, topk, need_soft=True):
        # A caller may request the no-soft form only when no gradient can
        # reach it; when any differentiable input needs the prior, keep it.
        # `ctx.needs_input_grad` must be used here: torch.is_grad_enabled()
        # is False inside a custom autograd Function's forward, so it cannot
        # detect that a backward will be built.
        need_soft = bool(need_soft) or any(ctx.needs_input_grad[1:4])
        if impl == 'triton' and any(
                t.dtype == torch.float64 for t in (qI, kI, w_idx)):
            _warn_shape_fallback(
                'indexer', 'fp64 inputs require the torch-reference backend '
                           '(the Triton scorer computes in fp32)')
            impl = 'ref'
        else:
            impl = _indexer_impl_for_shape(impl, kI)
        if impl == 'triton':
            idx, valid, soft = _indexer_fwd_triton(qI, kI, w_idx, last_tok, topk, need_soft=need_soft)
        else:
            idx, valid, soft = _indexer_fwd_ref(qI, kI, w_idx, last_tok, topk, need_soft=need_soft)
        ctx.save_for_backward(qI, kI, w_idx, last_tok, idx, valid, soft)
        ctx.topk = int(topk)
        return idx, valid, soft

    @staticmethod
    def backward(ctx, gidx, gvalid, gs):
        qI, kI, w_idx, last_tok, idx, valid, soft = ctx.saved_tensors
        if gs is None:
            return (None,) * 7
        ng = ctx.needs_input_grad
        _ic = (_env_cached('CSA_KERNEL_IDX_CHUNK') or '').strip()
        dqI, dkI, dw = _indexer_bwd_ref(
            gs, idx, valid, soft, qI, kI, w_idx, last_tok,
            need_dq=bool(ng[1]), need_dk=bool(ng[2]), need_dw=bool(ng[3]),
            t_chunk=int(_ic) if _ic else None)
        return (None, dqI, dkI, dw, None, None, None)


def lightning_indexer_kernel(qI, kI, w_idx, last_tok, topk, impl=None, need_soft=True):
    """`exp_lib.lightning_indexer`（learned 模式、return_mask=False）的内核
    路径。    qI [R,T,nIH,hd]、kI [R,Bm,nIH,hd]、w_idx [R,T,nIH]、
    last_tok [R,Bm]。返回 (idx, valid, soft)。"""
    impl = impl or _op_impl('IDX')
    return FusedIndexer.apply(impl, qI, kI, w_idx, last_tok, int(topk), bool(need_soft))


# ============================================================================
# 算子 3：稀疏注意力（`gathered_attention` 的内核路径）
# ============================================================================

_ATTN_REF_BUDGET = 256 << 20


def _attn_t_chunk(R, M, nh, hd):
    # 头循环在 t-chunk 之外；chunk 只划分独立的 (r,t) 行，宽度不影响任何
    # 元素值。
    per_t = max(1, 6 * int(R) * int(M) * int(hd) * 4)
    return max(1, min(int(_ATTN_REF_BUDGET) // per_t, 256))


def _attn_window_pair(k_sw, v_sw, q, w):
    """Normalise the sliding-window K/V pair to the kernel's shape contract.

    The public eager entry point documents ``w=0`` as "block-only attention"
    with ``k_sw=v_sw=None``.  The kernel twin used to subscript ``None`` while
    building its per-head views, and its backward allocated the window
    gradients with a hard-coded ``[R, T, nh, hd]`` shape even when the caller
    supplied a legitimately zero-width tensor.  Both are transport/shape bugs,
    not algorithm changes: values for a zero-width window are never read, so
    an empty tensor is the exact representation of "no window lanes".  A
    non-zero window without its K/V tables is a caller error and is refused
    here instead of failing later with an opaque ``NoneType`` error.
    """
    R, T, nh, hd = q.shape
    try:
        w = int(w)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'attention window width must be an integer, got {w!r}') from None
    if w < 0:
        raise ValueError(f'attention window width must be >= 0, got {w}')
    if (k_sw is None) != (v_sw is None):
        raise ValueError('attention needs BOTH k_sw and v_sw (or neither)')
    if k_sw is None:
        if w != 0:
            raise ValueError(f'attention window w={w} requires k_sw/v_sw tensors')
        # Any zero-length row dimension is unused when w == 0; use the
        # canonical empty form so gathers, broadcasts and dgrad shapes all
        # agree with the caller's zero-width window.
        z = q.new_zeros((int(R), 0, int(nh), int(hd)))
        return z, z
    if (len(tuple(k_sw.shape)) != 4
            or int(k_sw.shape[0]) != int(R)
            or int(k_sw.shape[2]) != int(nh)
            or int(k_sw.shape[3]) != int(hd)):
        raise ValueError(
            'attention k_sw/v_sw must have shape [R, T_window, nh, hd]; got '
            f'{tuple(k_sw.shape)} vs q={tuple(q.shape)}')
    if tuple(v_sw.shape) != tuple(k_sw.shape):
        raise ValueError(
            'attention k_sw and v_sw must have the same shape; got '
            f'{tuple(k_sw.shape)} vs {tuple(v_sw.shape)}')
    if int(w) > 0 and int(k_sw.shape[1]) != int(T):
        raise ValueError(
            f'attention window w={w} indexes positions 0..{int(T) - 1}, but '
            f'k_sw has {int(k_sw.shape[1])} row(s); supply one K/V row per '
            'sequence position')
    return k_sw, v_sw


def _attn_fwd_ref(q, k_blk, v_blk, k_sw, v_sw, topk_idx, valid, last_tok, soft, sink, w, scale):
    """torch 参考实现（按 t 分块 × 逐头）。

    泳道布局与 eager 一致：[top-k 块泳道 | 滑窗泳道]，共 M = nb + w 条。
    前向值 = **A 路径**（plain softmax 注意力）——eager 的直通估计器
    `attn = A + (S − S.detach())` 在前向值上严格等于 A；S（soft 先验）
    路径只进入梯度（见 _attn_bwd_ref）。sink 经 [sink | lanes] 拼接 softmax
    后丢弃 sink 列：P_l = e_l / (e_sink + Σ e)。
    无 sink 且全泳道无效的行：softmax 在全 MINL 上给出均匀分布，eager
    以 anyrow 乘法清零——本实现以同样的 anyrow 语义清零。
    """
    R, T, nh, hd = q.shape
    w = int(w)
    k_sw, v_sw = _attn_window_pair(k_sw, v_sw, q, w)
    nb = topk_idx.shape[-1]
    M = nb + w
    dev = q.device
    dtype = q.dtype
    # fp64 attention must stay fp64; downcasting to fp32 silently overflows
    # finite inputs.  fp16/bf16 still work in fp32, matching eager's work dtype.
    acc = torch.promote_types(dtype, torch.float32)
    MINL = _finfo_min(acc)
    out = torch.empty_like(q)
    t_idx = torch.arange(T, device=dev)[None, :, None]
    if valid is None:
        valid = torch.ones_like(topk_idx, dtype=torch.bool)
    win_raw = (t_idx - (w - 1) + torch.arange(w, device=dev)[None, None, :]).expand(R, T, w)
    # q/K/V are deliberately computed in float32 above; the sink logits
    # must use the same work dtype or torch.cat promotes to the input dtype
    # and the softmax/einsum then mixes (for fp16/bf16/fp64 input) dtypes.
    sink_v = torch.nan_to_num(sink).to(acc) if sink is not None else None
    tc = _attn_t_chunk(R, M, nh, hd)
    # t-chunk 在外、头在内：各 (t,h) 的前向计算相互独立，循环顺序交换
    # 不改变任何元素值。chunk 内的 lt_sel/sel_ok/src_win/win_ok/anyrow
    # 全部与 h 无关，按 chunk 各算一次。
    lt_l = last_tok.to(torch.int64)
    q_heads = [q[:, :, h, :] for h in range(nh)]
    kb_heads = [k_blk[:, :, h, :] for h in range(nh)]
    kw_heads = [k_sw[:, :, h, :] for h in range(nh)]
    vb_heads = [v_blk[:, :, h, :] for h in range(nh)]
    vw_heads = [v_sw[:, :, h, :] for h in range(nh)]
    for t0 in range(0, T, tc):
        t1 = min(t0 + tc, T)
        src32 = topk_idx[:, t0:t1]
        lt_sel = lt_l.gather(1, src32.to(torch.int64).reshape(R, -1)).view(R, t1 - t0, nb)
        sel_ok = valid[:, t0:t1] & (t_idx[:, t0:t1, :] > lt_sel)
        src_win = win_raw[:, t0:t1, :].clamp(min=0)
        win_ok = win_raw[:, t0:t1, :] >= 0
        lane_ok = torch.cat([sel_ok, win_ok], dim=-1)
        anyrow = (sel_ok.any(dim=-1) | win_ok.any(dim=-1)) if sink_v is None else None
        src_sel = src32.to(torch.int64)
        for h in range(nh):
            qh = q_heads[h][:, t0:t1].to(acc)
            Kb = _gather_rows(kb_heads[h], src_sel).to(acc)
            Kw = _gather_rows(kw_heads[h], src_win).to(acc)
            Vb = _gather_rows(vb_heads[h], src_sel).to(acc)
            Vw = _gather_rows(vw_heads[h], src_win).to(acc)
            zA = torch.cat([torch.einsum('rtd,rtmd->rtm', qh, Kb), torch.einsum('rtd,rtmd->rtm', qh, Kw)], dim=-1)
            if scale != 1.0:
                # x·1.0 is the IEEE identity; the fresh cat output is safe to
                # scale in place (same elementwise kernel and values as the
                # out-of-place form, and the CatBackward is unaffected).
                zA.mul_(scale)
            zA.masked_fill_(~lane_ok, MINL)
            if sink_v is None:
                P_A = torch.softmax(zA, dim=-1) * (anyrow[:, :, None] if anyrow is not None else 1.0)
            else:
                zext = torch.cat([sink_v[h].expand(R, t1 - t0, 1), zA], dim=-1)
                P_A = torch.softmax(zext, dim=-1)[..., 1:]
            Vl = torch.cat([Vb, Vw], dim=-2)
            out[:, t0:t1, h, :] = torch.einsum('rtm,rtmd->rtd', P_A, Vl).to(dtype)
        del src32, lt_sel, sel_ok, src_win, win_ok, lane_ok, anyrow, src_sel
    return out
