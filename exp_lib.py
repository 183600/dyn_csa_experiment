import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
# PYTHONHASHSEED only takes effect when it is set before the interpreter
# starts; setting it here is a best-effort propagation to child processes
# (par.py workers, resumed runs).  The experiment itself never depends on
# set/dict iteration order, but pinning it closes the last Python-level
# nondeterminism source on repeat runs of the same machine.
os.environ.setdefault('PYTHONHASHSEED', '0')
_cpu_thr = os.environ.get('CSA_CPU_THREADS', '').strip() or '1'
os.environ.setdefault('OMP_NUM_THREADS', _cpu_thr)
os.environ.setdefault('MKL_NUM_THREADS', _cpu_thr)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
del _cpu_thr
import sys, subprocess
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except (ValueError, OSError):
            pass
_NEEDED = []
for _pkg in ('datasets', 'tokenizers', 'matplotlib'):
    try:
        __import__(_pkg)
    except ImportError:
        _NEEDED.append(_pkg)
if _NEEDED:
    print(f"[setup] installing missing package(s): {', '.join(_NEEDED)}")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *_NEEDED])
    except Exception as _pip_err:
        print(f"[setup] could not install {', '.join(_NEEDED)} ({type(_pip_err).__name__}: {_pip_err}) — continuing; a code path that actually needs them will raise ImportError")
import torch
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
import gc, math, os, re, time, json, random, csv, tempfile, itertools, hashlib, collections
from math import comb
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[setup] device = {DEVICE}   torch = {torch.__version__}')
QUICK = False
RUN_CAP = dict(total_hours=58.333333333333336, margin=0.93, already_hours=0.0, state_path='run_time_state.json')
CODE_SEMANTICS = 'v11.125'
CKPT_CODE = CODE_SEMANTICS
# ---- 三算子内核路径（docs/compute_plan.md §3 的 M0 工作包；实现与数值契约
# 见 csa_kernels.py）。CSA_KERNEL 未设置时本模块的执行路径与指纹逐字节不变。
try:
    import csa_kernels as _CK
except Exception as _ck_err:
    _CK = None
    _CK_ERR = _ck_err


def _kernel_stamp():
    """内核变体标识（进入 run_cfg 指纹与 `_code` 戳）。eager 模式返回空串，
    既有记录的指纹逐字节不变；内核模式下记录只与同模式记录配对（§3.1）。"""
    if _CK is None:
        return ''
    try:
        return _CK.kernel_stamp()
    except Exception:
        return ''

def code_stamp():
    """当前代码+内核模式写进每条记录的 `_code` 戳（eager = CODE_SEMANTICS
    本身，与历史逐字节一致；内核模式追加变体后缀）。一切「记录是否当前」
    的判定（result_is_current 的 code 实参、`_cs...` 后缀匹配）都必须以
    本值为准——裸比 CODE_SEMANTICS 会把内核模式的全部记录误判为陈旧，
    使 resume 静默失效、面板被标 `_stale_cfg`。"""
    return f'{CODE_SEMANTICS}{_kernel_stamp()}'

# ---- 可选激活重算（CSA_CKPT，默认关闭）----
# 与 ce_loss / gathered_attention / _pool_trainable 的既有 checkpoint 同型
# （design_notes §6.1「重算同一纯函数」的 transport 论证）：被包的前向无
# RNG、是输入与权重的纯函数，反向用保存的输入按同一确定性分解重算，前向
# 值与梯度与直调形式逐位一致。未设置时执行路径与指纹逐字节不变；开关本身
# 不改任何到达算子的值，与 CSA_PAR / CSA_CPU_THREADS 同类，不进入
# run_cfg 指纹（§3.1）。
#   mlp   —— 每个 MLP 前向包一层非重入 checkpoint：fc1 输出 [B·T, 4·d]
#            不保存，反向用保存的输入按同一确定性分解重算 fc1+gelu。
#   block —— 在 mlp 之上，对无 gate_mean 侧信道的臂（chunking !=
#            'cosine_learnable'）整块包一层 checkpoint：块输入 [B, T, d]
#            成为唯一保存的层间激活。cosine_learnable 臂的 gate_mean 经
#            属性逃逸出块（trunk 的 comp_reg 读它），而 checkpoint 的初向
#            前向在 no_grad 下运行会剥掉它的图（v9 _V9_CKPT 的同一限
#            制），故这些臂自动回退到 mlp 级并提示一次。块内既有的池化 /
#            gather 注意力 / sink 分块 checkpoint 在反向重算时照旧包裹
#            （嵌套非重入 checkpoint 是 torch 支持、v9 驱动已在使用的形
#            式）；_CKPT_NEST 使内层 MLP 不再重复包裹。
_CKPT_NEST = [0]
_CKPT_WARNED = set()

def _ckpt_mode():
    raw = os.environ.get('CSA_CKPT', '').strip().lower()
    if raw in ('', '0', 'off', 'none', 'false'):
        return ''
    if raw in ('mlp', 'block'):
        return raw
    raise ValueError(f"unknown CSA_CKPT value {os.environ.get('CSA_CKPT')!r} (expected unset/'0' (off), 'mlp', or 'block')")

_CKPT = {'on': _ckpt_mode()}
RUN = dict(seq_len=512, batch_size=12, n_train_tokens=1000000 if QUICK else 8000000, steps=500 if QUICK else 1500, warmup=50, lr=0.0003, weight_decay=0.1, comp_lambda=0.05, delta_lr_mult=10.0, eval_every=250, eval_subset=128, seeds=[0] if QUICK else [0, 1, 2, 3, 4], outdir='results_lm_v3_1500', variants=['full', 'full_matched', 'full_cos', 'full_sw128', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic', 'hybrid_fixed', 'hybrid_dynamic'])
# 消融臂：`csa_fix_*` 是固定分块下对学习式 indexer / 池化内容 / sink / top-k
# 的去一法对照；`csa_dyn_*` 是同一组去一法在**动态分块**（主方法）下的对照——
# 只报固定分块下的去一法无法支撑「该模块在提出的动态方法里各自有用」。两组
# 都是「与 csa_dynamic / csa_fixed 逐字段同构，仅待验证模块不同」。
ABL_VARIANTS = ['hybrid_csa_dyn', 'hybrid_hca_dyn', 'csa_dyn_fuse', 'hybrid_csa_dyn_fuse', 'csa_fix_randidx', 'csa_fix_zerocont', 'csa_fix_nosink', 'csa_fix_topk8', 'csa_fix_topk64', 'csa_dyn_randidx', 'csa_dyn_zerocont', 'csa_dyn_nosink', 'csa_dyn_nocomp', 'full_sink']
# 去一法要成立，「Full」与各「Full−X」臂必须落在**同一个面板**里：ABLATION_DESIGN
# 的每条对照都靠 pair_reason 在同一 run_cfg 下逐种子配对，methodology_audit 也按
# 单面板 summary 审计——参照臂不在面板内时该面板只能报「缺臂」，撑不起任何
# 「各模块都有用」的结论。因此消融面板自含参照臂（csa_dynamic / hybrid_csa_dyn
# 即去一法的「Full」；full 供 full_sink 的 dense sink 对照）。参照臂列在最前：
# 历史机器上 resume 直接跳过，全新机器上被 cap 截断时它们最先完成，缺的
# 只是 abl 臂而不是整套对照。
ABL_REFS = ['full', 'csa_fixed', 'csa_dynamic', 'hybrid_fixed', 'hybrid_dynamic']
RUN_ABL = dict(RUN, outdir='results_lm_v4_abl', variants=ABL_REFS + ABL_VARIANTS, seeds=[0] if QUICK else [0, 1, 2, 3, 4])
RUN_LONG = dict(RUN, n_train_tokens=2000000 if QUICK else 110000000, steps=500 if QUICK else 20000, warmup=200, eval_every=1000, seeds=[0] if QUICK else [0, 1, 2, 3], outdir='results_lm_v3_long', variants=['full', 'full_matched', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic', 'hybrid_fixed', 'hybrid_dynamic', 'hybrid_csa_dyn'])
RUN_SCALE = dict(RUN, d=384, n_layers=8, n_heads=12, d_head=32, seq_len=1024, batch_size=8, n_train_tokens=2000000 if QUICK else 40000000, steps=500 if QUICK else 4000, warmup=100, eval_every=500, seeds=[0] if QUICK else [0, 1, 2], outdir='results_lm_v5_scale', variants=['full', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic', 'hybrid_dynamic'])
# ---- B 档规模阶梯（docs/compute_plan.md §4–§5；档位已选定 B 进取）----
# 以下全部为规划值：步数 / 批大小 / 学习率等训练量与超参按
# docs/compute_plan.md 的口径滚动修正；参数量以 SmallGPT 实例化统计（见
# m0_supp.py params，S4 按规划的解析外推口径）。
B_VOCAB = 32768
SCALE_LADDER = {'S1': dict(d=768, n_layers=12, n_heads=12, d_head=64), 'S2': dict(d=1024, n_layers=16, n_heads=16, d_head=64), 'S3': dict(d=1536, n_layers=20, n_heads=24, d_head=64), 'S4': dict(d=2048, n_layers=24, n_heads=32, d_head=64)}
B_ARMS = ['full_matched', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic', 'hybrid_dynamic']
RUN_S1 = dict(RUN, dataset='fineweb', vocab_size=B_VOCAB, d=768, n_layers=12, n_heads=12, d_head=64, seq_len=8192, batch_size=8, n_train_tokens=5000000000, steps=76294, warmup=2000, eval_every=4000, eval_subset=32, val_seqs=512, seeds=[0, 1, 2, 3], outdir='results_b_s1', variants=B_ARMS)
M0_CAP = dict(total_hours=700.0, margin=0.93, already_hours=0.0, state_path='run_time_state_m0.json')
RUN_S2 = dict(RUN_S1, d=1024, n_layers=16, n_heads=16, n_train_tokens=12000000000, steps=183105, warmup=4000, eval_every=8000, seeds=[0, 1, 2], outdir='results_b_s2')
M2_CAP = dict(total_hours=500.0, margin=0.93, already_hours=0.0, state_path='run_time_state_m2.json')

_CPU_DEV = torch.device('cpu')
_DEV_STR = {}

def _dev_str(device):
    # str(device) is a pure formatting function of the device object: the same
    # string every time, host-side only — cache it per object.
    s = _DEV_STR.get(device)
    if s is None:
        s = str(device)
        _DEV_STR[device] = s
    return s

def block_readable(pos, last_tok):
    return pos[:, None] > last_tok[None, :]

def cosine_similarity_consecutive(H, eps=1e-08):
    h = F.normalize(H, dim=-1, eps=eps)
    return (h[1:] * h[:-1]).sum(dim=-1)

# LRU memo of exact segmenter outputs (OrderedDict, O(1) eviction by recency)
# — a pure cache: hits return the identical (bids, honoured, counts) triple
# either policy would return, so no value downstream can change.
_SEGMENT_CACHE = collections.OrderedDict()
_SEGMENT_CACHE_CAP = 4096
# Per-length host position table arange(n + 1) reused across segmenter calls
# (read-only).  Pure cache — the same integers `np.arange(n + 1)` returns
# every call.
_SEG_POS_CACHE = {}
_SEG_POS_CACHE_CAP = 16

def _seg_pos(n):
    pos = _SEG_POS_CACHE.get(n)
    if pos is None:
        pos = np.arange(n + 1, dtype=np.int64)
        if len(_SEG_POS_CACHE) >= _SEG_POS_CACHE_CAP:
            _SEG_POS_CACHE.pop(next(iter(_SEG_POS_CACHE)))
        _SEG_POS_CACHE[n] = pos
    return pos

def _segment(n, want_cut_list, min_block, max_block):
    if n <= 0:
        # No tokens -> no blocks.  Returning the empty triple here (instead of
        # indexing ends[-1] on an empty array) keeps the public entry points
        # total for empty/negative-length inputs.
        return (np.empty(0, dtype=np.int64), {}, [])
    cacheable = n <= 64
    # uint8 cast of the booleans is exactly 0/1 per slot — the same byte string
    # `bytes(bytearray(1 if _c else 0 ...))` built, without a Python-level
    # generator per cut slot (host-side integer/bool equivalent rewrite; the
    # cached bids/honoured/counts the key guards are unchanged).
    key = (n, min_block, max_block, np.asarray(want_cut_list, dtype=np.uint8).tobytes()) if cacheable else None
    if cacheable:
        hit = _SEGMENT_CACHE.get(key)
        if hit is not None:
            _SEGMENT_CACHE.move_to_end(key)
            return hit
    # Successor-table form of the original greedy scan: with p the start of
    # the current block, the next block start is f(p) = min(p + max_block,
    # max(p + min_block, next_cut[p] + 1), n) (never less than p + 1), where
    # next_cut[p] is the earliest wanted-cut slot >= p (slot ci is the
    # boundary between tokens ci and ci+1, scanned when the scan reaches
    # token ci + 1).  f is a pure integer function of p, so the whole table
    # is built by vectorised integer ops and the scan itself is then a plain
    # table walk from 0 — the same block starts the token-by-token form
    # produced.  A cut at q honours the LAST wanted slot that was already
    # pending by then (the largest slot in [p, q - 1], i.e.
    # slots[bisect_right(slots, q - 1) - 1]) — computed vectorised over the
    # block ends, the same dict the pointer form built.
    _BIG = n + 1
    # `want_cut_list` arrives as a python list (per-row `.tolist()`) or as a
    # numpy bool row (the batched `_hoist_blocks_from_cuts` path); normalising
    # through np.asarray is a no-op for the latter.  Same booleans either way.
    slots = np.nonzero(np.asarray(want_cut_list, dtype=bool))[0]
    ns = len(slots)
    pos = _seg_pos(n)
    if ns:
        si = np.searchsorted(slots, pos, side='left')
        nc = np.where(si < ns, slots[np.minimum(si, ns - 1)], _BIG)
    else:
        nc = np.full(n + 1, _BIG, dtype=np.int64)
    nxt = np.minimum(pos + int(max_block), np.maximum(pos + int(min_block), nc + 1))
    np.minimum(nxt, n, out=nxt)
    np.maximum(nxt, pos + 1, out=nxt)
    # The successor table is a contiguous int64 array; a memoryview reads the
    # same exact integers as `nxt.tolist()` for each successor.  The fallback
    # to tolist covers any non-contiguous / non-buffer dtype.
    try:
        nxt_walk = nxt.data if nxt.flags['C_CONTIGUOUS'] else nxt.tolist()
    except (AttributeError, BufferError, ValueError):
        nxt_walk = nxt.tolist()
    starts = []
    ap = starts.append
    p = 0
    while p < n:
        ap(p)
        p = nxt_walk[p]
    sa = np.asarray(starts, dtype=np.int64)
    ends = np.empty(len(sa), dtype=np.int64)
    ends[:-1] = sa[1:]
    ends[-1] = n
    counts = (ends - sa).tolist()
    honoured = {}
    if ns:
        sel = (ends < n) & (nc[sa] <= ends - 1)
        if sel.any():
            hj = np.searchsorted(slots, ends[sel] - 1, side='right')
            # dict.fromkeys over the `.tolist()` view builds the identical
            # {python-int slot: 1.0} mapping (first-occurrence order, duplicate
            # slots collapsed), with the same keys, values and insertion order.
            honoured = dict.fromkeys(slots[hj - 1].tolist(), 1.0)
    bids = np.repeat(np.arange(len(counts), dtype=np.int64), np.asarray(counts, dtype=np.int64))
    if cacheable:
        while len(_SEGMENT_CACHE) >= _SEGMENT_CACHE_CAP:
            _SEGMENT_CACHE.popitem(last=False)
        _SEGMENT_CACHE[key] = (bids, honoured, counts)
    return (bids, honoured, counts)

_H2D_POOL = {}
_H2D_POOL_BYTES = [0]
_H2D_POOL_CAP = 96 << 20
_H2D_POOL_SLOTS = 4

def _h2d_stage(shape, dtype, device):
    # Pinned-staging pool for CUDA uploads — transport only (design notes
    # §6.1).  The caller fills the returned host buffer with the exact values
    # it would otherwise hand to `torch.tensor(..., device=device)` and
    # `commit()` enqueues the H2D copy; the copy is bit-exact regardless of
    # which staging buffer (if any) it leaves from, so every value reaching
    # the device is identical.  Each slot is guarded by a CUDA event recorded
    # after its last enqueue, so a slot is never rewritten while an earlier
    # async copy from it is still in flight; when every slot is busy (or the
    # pool cannot serve at all) the caller falls back to a direct pinned copy,
    # again with identical values.
    if device.type != 'cuda' or not torch.cuda.is_available():
        return (None, None)
    shape = tuple(int(_s) for _s in shape)
    if not shape or 0 in shape:
        return (None, None)
    key = (_dev_str(device), dtype, shape)
    ent = _H2D_POOL.get(key)
    if ent is None:
        est = int(np.prod(shape, dtype=np.int64)) * torch.empty((), dtype=dtype).element_size() * _H2D_POOL_SLOTS
        if _H2D_POOL_BYTES[0] + est > _H2D_POOL_CAP:
            return (None, None)
        try:
            with torch.inference_mode(False):
                slots = [[torch.empty(shape, dtype=dtype).pin_memory(), None] for _ in range(_H2D_POOL_SLOTS)]
        except (RuntimeError, TypeError):
            return (None, None)
        _H2D_POOL_BYTES[0] += est
        ent = {'slots': slots, 'rr': 0}
        _H2D_POOL[key] = ent
    slots = ent['slots']
    n = len(slots)
    for _ in range(n):
        i = ent['rr'] % n
        ent['rr'] += 1
        buf, ev = slots[i]
        if ev is None or ev.query():
            def commit(_buf=buf, _slot=slots[i]):
                with torch.inference_mode(False):
                    dev_t = _buf.to(device, non_blocking=True)
                    ev2 = _slot[1]
                    if ev2 is None:
                        ev2 = torch.cuda.Event()
                    ev2.record()
                    _slot[1] = ev2
                return dev_t
            return (buf, commit)
    return (None, None)

def _h2d_async(host_src, dtype, device):
    # Same values as torch.tensor(host_src, dtype=dtype, device=device):
    # pure transport of exact (integer or gate-mask) values.  The
    # inference_mode(False) guard keeps the result a normal tensor even when the
    # caller runs under inference_mode (eval), so it can later serve as an
    # autograd index tensor.  CUDA uploads stage through the pooled pinned
    # buffers of `_h2d_stage` (same bit-exact copy); the fallback form below
    # remains available.
    with torch.inference_mode(False):
        t = torch.as_tensor(host_src, dtype=dtype)
        if t.device == device:
            return t
        if t.device.type == 'cuda' or device.type != 'cuda' or not torch.cuda.is_available():
            return t.to(device)
        staged, commit = _h2d_stage(t.shape, t.dtype, device)
        if staged is not None:
            staged.copy_(t)
            return commit()
        try:
            t = t.pin_memory()
        except RuntimeError:
            pass
        return t.to(device, non_blocking=t.is_pinned())

def blocks_from_cuts(n, want_cut_list, min_block, max_block, device, return_count=False, return_honoured=False):
    if n == 0:
        base, cnt, hon = (torch.empty(0, dtype=torch.long, device=device), 0, {})
    elif n == 1:
        base, cnt, hon = (torch.zeros(1, dtype=torch.long, device=device), 1, {})
    else:
        bids, hon, counts = _segment(n, want_cut_list, min_block, max_block)
        base = _h2d_async(bids, torch.long, device)
        cnt = len(counts)
        _mx = max(counts)
        base._pool_cpu = (counts, _mx, max(counts[:-1]) if cnt > 1 else 0)
    if not (return_count or return_honoured):
        return base
    out = [base]
    if return_count:
        out.append(cnt)
    if return_honoured:
        out.append(hon)
    return tuple(out)

def _hoist_blocks_from_cuts(T, gate_lists, min_block, max_block, device, want_honoured, hon_dtype):
    # Batched form of the per-row `blocks_from_cuts` + honoured-mask uploads in
    # the cosine_learnable path.  `blocks_from_cuts` stays the single exit for
    # the cut table (called with a CPU device it returns the host tensor and
    # performs no H2D).  Every value reaching the device — the integer block
    # ids (with the `_pool_cpu` tag re-attached to each device row) and the
    # 0.0/1.0 honoured mask — is identical to the per-row form.
    bids_cpu = []
    nblks = []
    hons = []
    for _g in gate_lists:
        _b, _nb, _hon = blocks_from_cuts(T, _g, min_block, max_block, _CPU_DEV, return_count=True, return_honoured=True)
        bids_cpu.append(_b)
        nblks.append(_nb)
        hons.append(_hon)
    # When the staging pool serves, the rows are written straight into the
    # pinned buffer via a single `torch.stack(out=...)` call; the values
    # uploaded are exactly the stacked rows either way.
    staged, commit = _h2d_stage((len(bids_cpu), T), torch.long, device)
    if staged is not None:
        torch.stack(bids_cpu, out=staged)
        bid_stack = commit()
    else:
        bid_stack = _h2d_async(torch.stack(bids_cpu), torch.long, device)
    hon_stack = None
    if want_honoured and T > 1:
        arr = None
        staged_h, commit_h = _h2d_stage((len(gate_lists), T - 1), hon_dtype, device)
        if staged_h is not None:
            try:
                arr = staged_h.numpy()
                arr[:] = 0.0
            except TypeError:
                # a numpy-less dtype (e.g. bfloat16) — fall back to a host array
                arr = None
                staged_h = commit_h = None
        if arr is None:
            arr = np.zeros((len(gate_lists), T - 1), dtype=np.float32)
        for _i, _hon in enumerate(hons):
            if _hon:
                # iterating the honoured dict yields its (integer) slot keys
                _sl = np.fromiter(_hon, dtype=np.int64, count=len(_hon))
                _sl = _sl[(0 <= _sl) & (_sl < T - 1)]
                arr[_i, _sl] = 1.0
        # the honoured mask is exact 0.0/1.0, identical in any float dtype
        hon_stack = commit_h() if commit_h is not None else _h2d_async(arr, hon_dtype, device)
    out = []
    for b in range(len(bids_cpu)):
        row = bid_stack[b]
        row._pool_cpu = getattr(bids_cpu[b], '_pool_cpu', None)
        out.append((row, nblks[b], None if hon_stack is None else hon_stack[b]))
    return out

def blocks_from_cosine(H, tau, min_block, max_block, sim=None, return_count=False):
    n = H.shape[0]
    dev = H.device
    if n <= 1:
        z = torch.zeros(n, dtype=torch.long, device=dev)
        return (z, 1) if return_count and n == 1 else ((z, 0) if return_count else z)
    if sim is None:
        sim = cosine_similarity_consecutive(H)
    # zero-copy numpy view; the segmenter sees the same booleans as a list
    cut_list = (sim < tau).cpu().numpy()
    return blocks_from_cuts(n, cut_list, min_block, max_block, dev, return_count=return_count)

def causal_adaptive_threshold(sim, target_block_tokens):
    if sim.numel() == 0:
        return torch.empty(0, dtype=sim.dtype, device=sim.device)
    # 闭式因果分位数按行分块；超限时 fail-fast（见下方 row_chunk）。
    n = sim.shape[-1]
    _rows_total = int(sim.shape[0]) if sim.dim() == 2 else 1
    _row_chunk_est = max(1, (256 << 20) // max(n * n * 4, 1))
    _peak_bytes = min(_rows_total, _row_chunk_est) * n * n * 16
    if _peak_bytes > 1 << 30:
        raise RuntimeError(
            f'causal_adaptive_threshold 的因果分位数闭式为 O(rows·T²)（单块峰值 ~{_peak_bytes / 2**30:.1f} GB）: '
            'adaptive 分段器不支持该序列长度——请改用 chunking="cosine_learnable"')
    q = max(1.0 / max(target_block_tokens, 1), 0.001)
    # 1-D (single row) or 2-D (batched rows) — every op is rowwise
    # independent, so row-chunking the batch dim is value-identical per row:
    # same kernels, same per-row reduction order.
    squeeze = sim.dim() == 1
    x = (sim.unsqueeze(0) if squeeze else sim).detach().float()
    n = x.shape[-1]
    idx = _arange_cache(n, sim.device)
    pos = q * idx
    lo = pos.floor().long()
    frac = pos - lo
    lo_c = lo.clamp(max=n - 1)[..., None]
    hi_c = (lo + 1).clamp(max=n - 1)[..., None]
    use_hi = (frac > 0.0) & (lo + 1 <= idx)
    first_row = idx == 0
    # row chunking keeps the per-chunk sort input bounded.
    row_chunk = max(1, (256 << 20) // max(n * n * 4, 1))
    out = torch.empty_like(x)
    for r0 in range(0, x.shape[0], row_chunk):
        b = min(row_chunk, x.shape[0] - r0)
        masked = torch.where(idx[None, None, :] <= idx[None, :, None],
                             x[r0:r0 + b, None, :], float('inf'))
        srt = masked.sort(dim=-1).values
        lo_v = srt.gather(-1, lo_c.expand(b, n, 1)).squeeze(-1)
        hi_v = srt.gather(-1, hi_c.expand(b, n, 1)).squeeze(-1)
        interp = lo_v * (1.0 - frac) + hi_v * frac
        oc = torch.where(use_hi, interp, lo_v)
        out[r0:r0 + b] = torch.where(first_row, torch.full_like(oc, float('-inf')), oc)
    out = out.to(sim.device)
    return out.squeeze(0) if squeeze else out

def blocks_fixed(n, block_size, device):
    key = (_dev_str(device), 'blkfix', int(n), int(block_size))
    hit = _IDX_CACHE.get(key)
    if hit is not None:
        return hit
    with torch.inference_mode(False):
        out = torch.arange(n, device=device) // int(block_size)
        if n > 0:
            bs = int(block_size)
            counts = [bs] * (n // bs)
            _rem = n % bs
            if _rem:
                counts.append(_rem)
            out._pool_cpu = (counts, min(bs, n), bs if len(counts) > 1 else 0)
    _cache_put(_IDX_CACHE, key, out, _MASK_CACHE_BUDGET_BYTES, _idx_cache_total)
    return out
_MASK_CACHE = {}
_CAUSAL_DONOR = -1
_CAUSAL_TRI = None

def _causal_donor(device, width):
    global _CAUSAL_TRI
    dev_s = _dev_str(device)
    held = _CAUSAL_TRI
    if held is not None and held[0] == dev_s and (held[1].shape[0] >= int(width)):
        return (held[1], dev_s)
    tri = _mask_buffer(int(width), device, 'causal')
    _PINNED_KEYS.clear()
    _PINNED_KEYS.add((dev_s, 'causal', _CAUSAL_DONOR))
    _CAUSAL_TRI = (dev_s, tri)
    _cache_put(_MASK_CACHE, (dev_s, 'causal', _CAUSAL_DONOR), tri, _mask_budget(width), _mask_cache_total)
    return (tri, dev_s)
_IDX_CACHE = {}
_IDX_CACHE_BYTES = 0
_MASK_CACHE_BUDGET_BYTES = 192 << 20

def _mask_budget(n):
    # [T,T] fp32 mask 单张占 4·T² 字节；pinned 的 causal donor 与
    # bandw/cw/cwf 三种形式需要同时驻留，上限随 T 放大让这几张 mask
    # 共存；只影响缓存命中，不影响任何值。
    return max(_MASK_CACHE_BUDGET_BYTES, 6 * int(n) * int(n) * 4)

def _cache_bytes(t):
    if t is None:
        return 0
    if isinstance(t, (tuple, list)):
        return sum((_cache_bytes(_v) for _v in t))
    try:
        nbytes = int(t.numel()) * int(t.element_size())
        # Views can share storage: a stride-0 expand (the HCA arange
        # expansion) has numel O(n·m) but occupies O(m).  Charging the full
        # numel would make that one entry look larger than the whole cache
        # budget, so every later _cache_put would evict all other tables and
        # rebuild them on the next call — cache thrash instead of reuse.  The
        # resident storage is the correct (conservative) charge for a view;
        # min() keeps a sliced view of a larger buffer from over-counting.
        try:
            nbytes = min(nbytes, int(t.untyped_storage().nbytes()))
        except Exception:
            pass
        return nbytes
    except Exception:
        return 0

def _mask_cache_total(cache):
    return sum((_cache_bytes(v) for v in cache.values()))

def _idx_cache_total(cache):
    return sum((_cache_bytes(v) + 128 * len(k) for k, v in cache.items()))

def _is_pinned_key(key):
    return key in _PINNED_KEYS
_PINNED_KEYS = set()

def _evictable_limit(cache, limit, total_of):
    pinned = sum((_cache_bytes(v) for k, v in cache.items() if _is_pinned_key(k)))
    return limit - pinned if pinned < limit else 0

def _cache_put(cache, key, tensor, limit, total_of):
    cache[key] = tensor
    total = total_of(cache)
    evictable = _evictable_limit(cache, limit, total_of)
    while total > limit and total > evictable and (len(cache) > 1):
        oldest = next((k for k in cache if not _is_pinned_key(k)), None)
        if oldest is None:
            break
        cache.pop(oldest)
        total = total_of(cache)
    return total
_ATB_CACHE = {}
_ATB_EPOCH = 0

def _attb_bump_epoch():
    global _ATB_EPOCH
    _ATB_EPOCH += 1
    _ATB_CACHE.clear()
_HALF = 0.5

def _arange_cache(n, device):
    key = (_dev_str(device), 'arange', int(n))
    a = _IDX_CACHE.get(key)
    if a is None:
        with torch.inference_mode(False):
            a = torch.arange(n, device=device)
        _cache_put(_IDX_CACHE, key, a, _MASK_CACHE_BUDGET_BYTES, _idx_cache_total)
    return a

def _window_geometry(n, w, device):
    key = (_dev_str(device), 'wingeo', int(n), int(w))
    hit = _IDX_CACHE.get(key)
    if hit is not None:
        return hit
    pos = _arange_cache(n, device)
    rel = _range_cache(0, w, device)
    with torch.inference_mode(False):
        # int32 arithmetic throughout (design notes §6.1): every position is an
        # exact integer in [-(w-1), n) and the sequence lengths this repo uses
        # lie far inside the int32 range, so the table holds bitwise the same
        # integers as the int64 form — index_select accepts int32 directly.
        # The validity mask is read off the pre-clamp table, so it sees the
        # same negative positions either way.
        win_idx = pos.to(torch.int32)[:, None] - (w - 1) + rel.to(torch.int32)[None, :]
        valid = win_idx >= 0
        win_idx.clamp_(min=0)
        pair = (win_idx, valid)
    _cache_put(_IDX_CACHE, key, pair, _MASK_CACHE_BUDGET_BYTES, _idx_cache_total)
    return pair

def _range_cache(lo, hi, device):
    if lo == 0:
        return _arange_cache(hi, device)
    key = (_dev_str(device), 'arange2', int(lo), int(hi))
    a = _IDX_CACHE.get(key)
    if a is None:
        with torch.inference_mode(False):
            a = torch.arange(lo, hi, device=device)
        _cache_put(_IDX_CACHE, key, a, _MASK_CACHE_BUDGET_BYTES, _idx_cache_total)
    return a

def _arange_exp_cache(n, m, device):
    # Cached int32 (n, m) arange expansion for the HCA read-everything arm:
    # the same exact integers as _arange_cache(m, device).unsqueeze(0).
    # expand(n, m).to(torch.int32) (index transport, design notes §6.1); int32
    # in, so the int64 upcast is skipped downstream (also §6.1).  Built outside
    # inference mode so a table first filled during eval can still serve as an
    # autograd index tensor in training; only ever read downstream (row slices
    # feeding gathers), never mutated.
    #
    # The table is stored as a stride-0 expand of one [m] arange instead of a
    # materialised repeat: every row holds the same exact integers, advanced
    # indexing / slicing / cat read them identically, and nothing downstream
    # writes to the table, so the resident footprint is O(m) rather than O(n·m)
    # (the largest single index table in the HCA arm at long T).
    key = (_dev_str(device), 'arangeexp', int(n), int(m))
    a = _IDX_CACHE.get(key)
    if a is None:
        with torch.inference_mode(False):
            base = torch.arange(int(m), device=device, dtype=torch.int32)
            a = base.expand(int(n), int(m))
        _cache_put(_IDX_CACHE, key, a, _MASK_CACHE_BUDGET_BYTES, _idx_cache_total)
    return a

def _causal_window_mask_build(T, window, device, dtype):
    # Direct construction of the causal sliding-window additive mask in the
    # target dtype: allowed lanes (i-window < j <= i) get exactly 0.0 and every
    # other lane exactly -inf — the same two constants the causal + band
    # addition and the elementwise cast produce.  A boolean scratch is
    # triangularised twice to fill the two masked regions.
    with torch.inference_mode(False):
        m = torch.zeros((T, T), device=device, dtype=dtype)
        far = torch.ones((T, T), device=device, dtype=torch.bool)
        far.tril_(-int(window))
        m.masked_fill_(far, float('-inf'))
        far.fill_(True).triu_(1)
        m.masked_fill_(far, float('-inf'))
    return m

def causal_window_mask_f(T, window, device, dtype):
    """causal_window_mask 的目标 dtype 形式（逐元素 cast 对 0/-inf 精确；
    缓存键含 dtype，mask 下游只读）。dtype 相同则无新分配。"""
    key = (_dev_str(device), 'cwf', int(T), int(window), str(dtype))
    if window <= 0 or window >= T:
        base = causal_mask(T, device)
        if base.dtype == dtype:
            return base
        m = _MASK_CACHE.get(key)
        if m is None:
            with torch.inference_mode(False):
                m = base.to(dtype)
            _cache_put(_MASK_CACHE, key, m, _mask_budget(T), _mask_cache_total)
        return m
    m = _MASK_CACHE.get(key)
    if m is None:
        m = _causal_window_mask_build(T, window, device, dtype)
        _cache_put(_MASK_CACHE, key, m, _mask_budget(T), _mask_cache_total)
    return m

_ATTN_TRANSIENT_CAP_BYTES = 2 << 30

def _attn_transient_budget(device):
    key = (_dev_str(device), _ATB_EPOCH)
    if key in _ATB_CACHE:
        return _ATB_CACHE[key]
    val = None
    try:
        if device.type == 'cuda':
            free, _total = torch.cuda.mem_get_info(device)
            # 预算随空闲量缩放，并封顶 _ATTN_TRANSIENT_CAP_BYTES；行分块是
            # 纯行划分（任意宽度下逐行值逐位相同，见 gathered_attention 的
            # chunk 注释），因此预算变化只影响分块宽度，不改变任何值。
            val = max(64 << 20, min(int(free) // 4, _ATTN_TRANSIENT_CAP_BYTES))
    except Exception:
        val = None
    _ATB_CACHE[key] = val
    return val

def _tri_full(n, device, k):
    with torch.inference_mode(False):
        m = torch.empty((n, n), device=device)
        m.fill_(float('-inf'))
        torch.triu(m, k, out=m) if k > 0 else torch.tril(m, k, out=m)
    return m

def _mask_buffer(n, device, kind):
    if kind == 'causal':
        return _tri_full(n, device, 1)
    key = (_dev_str(device), kind, int(n))
    buf = _MASK_CACHE.get(key)
    if buf is not None:
        return buf
    if kind != 'zero':
        raise ValueError(f'unknown mask kind {kind!r}')
    with torch.inference_mode(False):
        buf = torch.zeros((n, n), device=device)
    _cache_put(_MASK_CACHE, key, buf, _mask_budget(n), _mask_cache_total)
    return buf

def causal_mask(T, device):
    tri, _dev_s = _causal_donor(device, T)
    if tri.shape[0] == int(T):
        return tri
    return tri[:T, :T]

def window_band(T, window, device):
    if window <= 0 or window >= T:
        return _mask_buffer(T, device, 'zero')
    key = (_dev_str(device), 'bandw', int(T), int(window))
    buf = _MASK_CACHE.get(key)
    if buf is None:
        buf = _tri_full(T, device, -int(window))
        _cache_put(_MASK_CACHE, key, buf, _mask_budget(T), _mask_cache_total)
    return buf

def causal_window_mask(T, window, device):
    if window <= 0 or window >= T:
        return causal_mask(T, device)
    key = (_dev_str(device), 'cw', int(T), int(window))
    m = _MASK_CACHE.get(key)
    if m is None:
        m = _causal_window_mask_build(T, window, device, torch.float32)
        _cache_put(_MASK_CACHE, key, m, _mask_budget(T), _mask_cache_total)
    return m

def _window_block_offset(n, w, n_blk, device):
    key = (_dev_str(device), 'winoff', int(n), int(w), int(n_blk))
    hit = _IDX_CACHE.get(key)
    if hit is not None:
        return hit
    win_idx, _win_valid = _window_geometry(n, w, device)
    with torch.inference_mode(False):
        out = (win_idx + int(n_blk)).to(torch.int32)
    _cache_put(_IDX_CACHE, key, out, _MASK_CACHE_BUDGET_BYTES, _idx_cache_total)
    return out

def _take_rows(x, idx):
    return torch.index_select(x, 0, idx)

def _take_2d(x, idx):
    return torch.index_select(x, 0, idx.flatten()).view(idx.shape[0], idx.shape[1], *x.shape[1:])

def first_occurrence_mask(idx):
    if idx.shape[1] <= 1:
        return torch.ones_like(idx, dtype=torch.bool)
    # Both branches compute exactly the same booleans ("keep only the first
    # occurrence of each value per row"): the triu form marks an element a
    # duplicate iff an equal element sits at an earlier column; the stable
    # argsort form keeps, inside each run of equal sorted values, the element
    # that stability places first — which is the earliest original column.
    # Identical output on either branch.
    if idx.shape[1] <= 64:
        # rowwise-independent: chunk rows to bound the transient [rows,k,k]
        # comparison matrix; per-row results are identical either way.
        k = idx.shape[1]
        row_chunk = max(1, (128 << 20) // max(k * k, 1))
        if row_chunk >= idx.shape[0]:
            dup = (idx[:, :, None] == idx[:, None, :]).triu_(1).any(dim=1)
            return ~dup
        out = torch.empty_like(idx, dtype=torch.bool)
        for r0 in range(0, idx.shape[0], row_chunk):
            r1 = min(r0 + row_chunk, idx.shape[0])
            dup = (idx[r0:r1, :, None] == idx[r0:r1, None, :]).triu_(1).any(dim=1)
            out[r0:r1] = ~dup
        return out
    order = torch.argsort(idx, dim=1, stable=True)
    sorted_idx = idx.gather(1, order)
    first = torch.ones_like(sorted_idx, dtype=torch.bool)
    first[:, 1:] = sorted_idx[:, 1:] != sorted_idx[:, :-1]
    keep = torch.empty_like(first)
    keep.scatter_(1, order, first)
    return keep

def _block_order(block_ids, n):
    if block_ids.numel() <= 1:
        return (_arange_cache(block_ids.numel(), block_ids.device), True)
    if bool((block_ids[1:] >= block_ids[:-1]).all()):
        return (_arange_cache(block_ids.numel(), block_ids.device), True)
    return (torch.argsort(block_ids, stable=True), False)

def _n_blocks_cached(block_ids):
    # block_ids is treated as immutable by the pooling caches (_pool_gpu /
    # _pool_geom), so its max+1 is a constant of the object — cache it per
    # object.  First call per object computes the identical value.
    B = getattr(block_ids, '_pool_nblocks', None)
    if B is None:
        B = int(block_ids.max().item()) + 1
        try:
            block_ids._pool_nblocks = B
        except (AttributeError, RuntimeError):
            pass
    return B

def pool_variable_blocks(Xa, Xb, Za, Zb, block_ids, B_pos_a, B_pos_b, overlap, n_blocks=None, monotonic=False):
    n, Fd = Xa.shape
    dev = Xa.device
    B = int(n_blocks) if n_blocks is not None else _n_blocks_cached(block_ids)
    order, _ident = ((_arange_cache(block_ids.numel(), block_ids.device), True) if monotonic else _block_order(block_ids, n))
    if _ident:
        Xas, Xbs, Zas, Zbs = (Xa, Xb, Za, Zb)
    else:
        Xas = _take_rows(Xa, order)
        Xbs = _take_rows(Xb, order)
        Zas = _take_rows(Za, order)
        Zbs = _take_rows(Zb, order)
    return _pool_ordered(Xas, Xbs, Zas, Zbs, block_ids, B_pos_a, B_pos_b, overlap, n, Fd, dev, B, order, order_is_identity=_ident)

_POOL_PARTS_HOST = True
_POOL_PARTS_BUCKET = 256
_POOL_GEOM_BUCKET = 1024
# Block-chunking of the block-pooling gathers/mixes: the chunk boundaries are
# a pure function of the shapes, so the forward and the checkpoint recompute
# split identically.  Each block's pooled vector is computed from its own
# tokens/overlap lanes only, so processing blocks in any grouping yields the
# same per-block values bit-for-bit.
_POOL_TRANSIENT_BUDGET_BYTES = 256 << 20

def _pool_block_chunk(B, per_block_elems, elem_size):
    # Number of blocks per chunk; >= 1 and never more than B (so short
    # sequences run one chunk).
    _per = max(1, int(per_block_elems) * int(elem_size))
    return max(1, min(int(B), _POOL_TRANSIENT_BUDGET_BYTES // _per))

def _pool_parts_flat(counts_l, B, ov, n, pad_to=1):
    # Host-side integer flat of the pooling parts (design notes §6.2): every
    # value is an exact integer function of the cut table's own counts list
    # (block_ids._pool_cpu[0] — the same list that generated block_ids), so
    # each value is the same integer the bincount/cumsum form produces
    # (cumsum / shifts / clamps on int64 are exact on either side).  Returns
    # (flat_int64 ndarray, total) — flat is zero-padded to a pad_to multiple
    # and the padded tail is never read — or None when the host form cannot
    # serve; the caller then falls back to the on-device form, whose values
    # are identical.
    try:
        B = int(B)
        n = int(n)
        counts_np = np.asarray(counts_l, dtype=np.int64)
        if counts_np.shape != (B,) or n <= 0:
            return None
        ends_np = np.cumsum(counts_np, dtype=np.int64)
        nseg = 8 if ov is not None else 4
        total = nseg * B
        pad = (-total) % int(pad_to) if pad_to > 1 else 0
        # Flat layout: [counts | ends | starts | last_idx (| end_prev |
        # starts_prev | ov_start | ov_len)] followed by the zero pad; the
        # integers uploaded are identical to the per-segment form.
        flat = np.empty(total + pad, dtype=np.int64)
        flat[0:B] = counts_np
        flat[B:2 * B] = ends_np
        starts_np = flat[2 * B:3 * B]
        np.subtract(ends_np, counts_np, out=starts_np)
        np.clip(ends_np - 1, 0, n - 1, out=flat[3 * B:4 * B])
        if ov is not None:
            end_prev = flat[4 * B:5 * B]
            end_prev[0] = -1
            end_prev[1:] = ends_np[:-1]
            starts_prev = flat[5 * B:6 * B]
            starts_prev[0] = 0
            starts_prev[1:] = starts_np[:-1]
            ov_start = flat[6 * B:7 * B]
            np.subtract(end_prev, int(ov), out=ov_start)
            np.maximum(ov_start, starts_prev, out=ov_start)
            np.subtract(end_prev, ov_start, out=flat[7 * B:8 * B])
            np.maximum(flat[7 * B:8 * B], 0, out=flat[7 * B:8 * B])  # block 0 无 overlap：几何表自身保持非负
        if pad:
            flat[total:] = 0
        return (flat, total)
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return None

def _pool_parts_host(block_ids, B, ov, n, counts_l):
    # Host-side build of the pooling geometry (see _pool_parts_flat for the
    # exact-integer argument).  The whole geometry rides one flat staged
    # upload (transport only, §6.1); the parts are views into that single
    # buffer (they are only ever read downstream, never mutated),
    # and the bucket-padding tail is never read.  `_h2d_async` builds the
    # buffer outside inference mode, so a part first filled during eval can
    # still serve training, exactly like the on-device form.  Returns None
    # when the host form cannot serve; the caller then falls back to the
    # on-device form, whose values are identical.
    _built = _pool_parts_flat(counts_l, B, ov, n, _POOL_PARTS_BUCKET)
    if _built is None:
        return None
    try:
        flat, total = _built
        buf = _h2d_async(flat, torch.long, block_ids.device)
        # the bucket pad is sliced off first; the unbound segment views are
        # read-only views into buf.
        return buf[:total].view(8 if ov is not None else 4, int(B)).unbind(0)
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return None

def _pool_geom_flat_var(counts_l, B, ov, n, max_len, ov_hi, pa_rows, pb_rows):
    # Host-side integer/bool flats of the variable-block gather geometry
    # (design notes §6.2): g_a/mask_a/g_b/mask_b/b_off/pos_a are exact
    # integer/bool functions of the cut table's own counts (the same counts
    # _pool_parts_flat already mirrors) plus the shape scalars, so the numpy
    # form below produces values identical to the on-device build in
    # _pool_ordered — only the location of the integer math and the
    # transport change (§6.1).  Returns (flat_i, flat_b, n_a, n_b, W_b,
    # max_len) — flat_i holds [g_a | g_b | b_off | pos_a] and flat_b holds
    # [mask_a | mask_b] — or None when the host form cannot serve; the
    # caller then falls back to the on-device form, whose values are
    # identical.
    try:
        B = int(B)
        n = int(n)
        ov = int(ov)
        max_len = int(max_len)
        W_b = max(int(ov_hi), 0)
        pa_rows = int(pa_rows)
        pb_rows = int(pb_rows)
        counts_np = np.asarray(counts_l, dtype=np.int64)
        if counts_np.shape != (B,) or n <= 0 or max_len <= 0 or pa_rows <= 0 or pb_rows <= 0:
            return None
        ends_np = np.cumsum(counts_np, dtype=np.int64)
        starts_np = ends_np - counts_np
        pos = np.arange(max_len, dtype=np.int64)
        gp = starts_np[:, None] + pos[None, :]
        mask_a_np = gp < ends_np[:, None]
        np.minimum(gp, n - 1, out=gp)
        op = np.arange(W_b, dtype=np.int64)
        end_prev = np.empty(B, dtype=np.int64)
        end_prev[0] = -1
        end_prev[1:] = ends_np[:-1]
        starts_prev = np.empty(B, dtype=np.int64)
        starts_prev[0] = 0
        starts_prev[1:] = starts_np[:-1]
        ov_start = np.maximum(end_prev - ov, starts_prev)
        ov_len = np.maximum(end_prev - ov_start, 0)
        mask_b_np = op[None, :] < ov_len[:, None]
        g_b_np = np.minimum(ov_start[:, None] + op[None, :], n - 1)
        b_off_np = np.minimum(ov_start[:, None] - starts_prev[:, None] + op[None, :], pb_rows - 1)
        pos_a_np = pos if max_len <= pa_rows else np.minimum(pos, pa_rows - 1)
        n_a = B * max_len
        n_b = B * W_b
        flat_i = np.empty(n_a + 2 * n_b + max_len, dtype=np.int64)
        flat_i[0:n_a] = gp.ravel()
        flat_i[n_a:n_a + n_b] = g_b_np.ravel()
        flat_i[n_a + n_b:n_a + 2 * n_b] = b_off_np.ravel()
        flat_i[n_a + 2 * n_b:] = pos_a_np
        flat_b = np.empty(n_a + n_b, dtype=np.bool_)
        flat_b[0:n_a] = mask_a_np.ravel()
        flat_b[n_a:] = mask_b_np.ravel()
        return (flat_i, flat_b, n_a, n_b, W_b, max_len)
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return None

def _pool_geom_host_var(counts_l, B, ov, n, max_len, ov_hi, pa_rows, pb_rows, dev):
    # Host-side build of the variable-block gather geometry (see
    # _pool_geom_flat_var for the exact-integer argument).  The int64 parts
    # ride one flat staged upload and the two bool masks a second one (the
    # same staged pinned pool _pool_parts_host uses); the returned tensors
    # are read-only views into those buffers (the same aliasing argument the
    # parts make — downstream they feed gathers / mask multiplies /
    # negated-mask fills only).  Both flats are bucket-padded so the staged
    # pool's shape keys stay bounded as (B, max_len, W_b) varies across
    # steps; the pad tails are never read.  The fixed segmenter keeps its
    # build-once/cache path.  Returns None when the host form cannot serve;
    # the caller then falls back to the on-device form, whose values are
    # identical.
    _built = _pool_geom_flat_var(counts_l, B, ov, n, max_len, ov_hi, pa_rows, pb_rows)
    if _built is None:
        return None
    try:
        B = int(B)
        flat_i, flat_b, n_a, n_b, W_b, max_len = _built
        _pi = (-len(flat_i)) % _POOL_GEOM_BUCKET
        if _pi:
            flat_i = np.concatenate((flat_i, np.zeros(_pi, dtype=np.int64)))
        buf = _h2d_async(flat_i, torch.long, dev)
        g_a = buf[0:n_a].view(B, max_len)
        g_b = buf[n_a:n_a + n_b].view(B, W_b)
        b_off = buf[n_a + n_b:n_a + 2 * n_b].view(B, W_b)
        pos_a = buf[n_a + 2 * n_b:n_a + 2 * n_b + max_len]
        _pb = (-len(flat_b)) % _POOL_GEOM_BUCKET
        if _pb:
            flat_b = np.concatenate((flat_b, np.zeros(_pb, dtype=np.bool_)))
        mbuf = _h2d_async(flat_b, torch.bool, dev)
        # the bucket pad tail is never read: every view has an explicit end.
        return (g_a, mbuf[0:n_a].view(B, max_len), g_b, mbuf[n_a:n_a + n_b].view(B, W_b), b_off, pos_a)
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return None

def _pool_geom_flat_single(counts_l, B, n, max_len, p_rows):
    # Single-block-form twin of _pool_geom_flat_single's variable sibling
    # (pool_blocks_single's g/mask/pos_a flats): same exact-integer host
    # build (§6.2).  Returns (flat_i, mask_np, n_g, max_len) or None.
    try:
        B = int(B)
        n = int(n)
        max_len = int(max_len)
        p_rows = int(p_rows)
        counts_np = np.asarray(counts_l, dtype=np.int64)
        if counts_np.shape != (B,) or n <= 0 or max_len <= 0 or p_rows <= 0:
            return None
        ends_np = np.cumsum(counts_np, dtype=np.int64)
        starts_np = ends_np - counts_np
        pos = np.arange(max_len, dtype=np.int64)
        gp = starts_np[:, None] + pos[None, :]
        mask_np = gp < ends_np[:, None]
        np.minimum(gp, n - 1, out=gp)
        pos_a_np = pos if max_len <= p_rows else np.minimum(pos, p_rows - 1)
        n_g = B * max_len
        flat_i = np.empty(n_g + max_len, dtype=np.int64)
        flat_i[0:n_g] = gp.ravel()
        flat_i[n_g:] = pos_a_np
        return (flat_i, mask_np, n_g, max_len)
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return None

def _pool_geom_host_single(counts_l, B, n, max_len, p_rows, dev):
    # Single-block-form twin of _pool_geom_host_var (pool_blocks_single's
    # g/mask/_pos_a): same exact-integer host build (§6.2), same staged
    # transport (§6.1) with bucket-padded flats (bounded pool shape keys,
    # unread pad tails), same None-means-device-fallback contract.
    _built = _pool_geom_flat_single(counts_l, B, n, max_len, p_rows)
    if _built is None:
        return None
    try:
        B = int(B)
        flat_i, mask_np, n_g, max_len = _built
        _pi = (-len(flat_i)) % _POOL_GEOM_BUCKET
        if _pi:
            flat_i = np.concatenate((flat_i, np.zeros(_pi, dtype=np.int64)))
        buf = _h2d_async(flat_i, torch.long, dev)
        mask_np = mask_np.ravel()
        _pb = (-mask_np.size) % _POOL_GEOM_BUCKET
        if _pb:
            mask_np = np.concatenate((mask_np, np.zeros(_pb, dtype=np.bool_)))
        mask = _h2d_async(mask_np, torch.bool, dev)[:n_g].view(B, max_len)
        return (buf[0:n_g].view(B, max_len), mask, buf[n_g:n_g + max_len])
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return None

def _pool_parts(block_ids, B, ov, n):
    # Integer block geometry shared by the pooling paths.  Every value is an
    # exact integer function of (block_ids, B, ov, n), so caching the tensors on
    # the block-ids object is bitwise safe; the fixed segmenter hands out ONE
    # shared block-ids object for every row/layer/step.  Tensors are built
    # outside inference mode so a cache first filled during eval can still serve
    # training (an inference tensor saved for backward would raise).
    key = (int(B), -1 if ov is None else int(ov), int(n))
    cache = getattr(block_ids, '_pool_gpu', None)
    if cache is None:
        cache = {}
        try:
            block_ids._pool_gpu = cache
        except (AttributeError, RuntimeError):
            pass
    else:
        hit = cache.get(key)
        if hit is not None:
            return hit
    _cpu = getattr(block_ids, '_pool_cpu', None)
    if _POOL_PARTS_HOST and _cpu is not None and len(_cpu[0]) == int(B):
        _parts = _pool_parts_host(block_ids, B, ov, n, _cpu[0])
        if _parts is not None:
            if len(cache) < 8:
                cache[key] = _parts
            return _parts
    with torch.inference_mode(False):
        counts = torch.bincount(block_ids, minlength=B)
        ends = torch.cumsum(counts, 0)
        starts = ends - counts
        last_idx = (ends - 1).clamp(0, n - 1)
        if ov is None:
            parts = (counts, ends, starts, last_idx)
        else:
            end_prev = F.pad(ends[:-1], (1, 0), value=-1)
            starts_prev = F.pad(starts[:-1], (1, 0), value=0)
            ov_start = torch.clamp(end_prev - ov, min=starts_prev)
            ov_len = (end_prev - ov_start).clamp_min(0)  # block 0 无 overlap：几何表自身保持非负
            parts = (counts, ends, starts, last_idx, end_prev, starts_prev, ov_start, ov_len)
    if len(cache) < 8:
        cache[key] = parts
    return parts

def _pool_geom_put(block_ids, key, geom):
    # The gather geometry is an exact integer/bool function of the block-ids
    # object's own cut table (the `_pool_cpu` counts) plus the shape scalars
    # in the key, so caching it on the block-ids object is bitwise safe — the
    # same argument `_pool_parts` makes for its cached counts/ends.  The fixed
    # segmenter hands out ONE shared block-ids object for every row/layer/
    # step; the dynamic segmenter's per-row objects simply miss and rebuild
    # (identical values).  Every cached tensor is only read downstream
    # (gathers / negated-mask views / index selects), never mutated.
    cache = getattr(block_ids, '_pool_geom', None)
    if cache is None:
        cache = {}
        try:
            block_ids._pool_geom = cache
        except (AttributeError, RuntimeError):
            return geom
    if len(cache) < 8:
        cache[key] = geom
    return geom

def _prefetch_pool_geometry(attn, pre_blocks, n, dev):
    # Batched pool-geometry prefetch for the per-row sparse path (transport
    # only, design notes §6.1/§6.2).  Each row's parts / gather geometry is
    # the same exact-integer host build the lazy path performs inside
    # _pool_parts / _pool_ordered / pool_blocks_single (the same flat
    # builders above, the same per-row `_pool_cpu` counts); here the whole
    # batch's flats are concatenated into ONE staged upload per dtype class
    # (one int64, one bool) per layer, and the per-row views are attached to
    # each row's block-ids object under exactly the
    # keys the lazy lookups use — so _pool_parts / _pool_geom hit them
    # verbatim, including on the _pool_trainable recompute in backward.
    # Rows whose host build cannot serve are skipped; their lazy path falls
    # back to the value-identical on-device build.  The concatenated tails
    # are bucket-padded (bounded staged-pool shape keys; pad never read).
    if not _POOL_PARTS_HOST:
        return
    cfg = attn.cfg
    _csa = cfg.kind == 'csa'
    ov = int(cfg.overlap) if _csa else None
    pa_rows = int(attn.B_pos_a.shape[0])
    pb_rows = int(attn.B_pos_b.shape[0]) if _csa else 1
    rows = []
    for _pb in pre_blocks:
        bid, nblk = _pb[0], int(_pb[1])
        _cpu = getattr(bid, '_pool_cpu', None)
        if _cpu is None or len(_cpu[0]) != nblk or nblk <= 0:
            rows.append(None)
            continue
        counts_l = _cpu[0]
        max_len = _cpu[1]
        if _csa:
            _ov_hi = min(ov, _cpu[2]) if nblk > 1 else -1
            _g = _pool_geom_flat_var(counts_l, nblk, ov, n, max_len, _ov_hi, pa_rows, pb_rows)
        else:
            _g = _pool_geom_flat_single(counts_l, nblk, n, max_len, pa_rows)
        _p = _pool_parts_flat(counts_l, nblk, ov, n)
        if _g is None or _p is None:
            rows.append(None)
            continue
        rows.append((bid, nblk, _p, _g))
    if not any(rows):
        return
    _ints = []
    _bools = []
    for _r in rows:
        if _r is None:
            continue
        _ints.append(_r[2][0])
        _ints.append(_r[3][0])
        _bools.append(_r[3][1].ravel())
    flat_i = np.concatenate(_ints)
    _pi = (-len(flat_i)) % _POOL_GEOM_BUCKET
    if _pi:
        flat_i = np.concatenate((flat_i, np.zeros(_pi, dtype=np.int64)))
    buf_i = _h2d_async(flat_i, torch.long, dev)
    flat_b = np.concatenate(_bools)
    _pbk = (-len(flat_b)) % _POOL_GEOM_BUCKET
    if _pbk:
        flat_b = np.concatenate((flat_b, np.zeros(_pbk, dtype=np.bool_)))
    buf_b = _h2d_async(flat_b, torch.bool, dev)
    off_i = 0
    off_b = 0
    for _r in rows:
        if _r is None:
            continue
        bid, nblk, _p, _g = _r
        _ptot = _p[1]
        parts = buf_i[off_i:off_i + _ptot].view(8 if _csa else 4, nblk).unbind(0)
        off_i += _ptot
        if _csa:
            _fi, _fb, n_a, n_b, W_b, max_len = _g
            g_a = buf_i[off_i:off_i + n_a].view(nblk, max_len)
            g_b = buf_i[off_i + n_a:off_i + n_a + n_b].view(nblk, W_b)
            b_off = buf_i[off_i + n_a + n_b:off_i + n_a + 2 * n_b].view(nblk, W_b)
            pos_a = buf_i[off_i + n_a + 2 * n_b:off_i + n_a + 2 * n_b + max_len]
            off_i += n_a + 2 * n_b + max_len
            mask_a = buf_b[off_b:off_b + n_a].view(nblk, max_len)
            mask_b = buf_b[off_b + n_a:off_b + n_a + n_b].view(nblk, W_b)
            off_b += n_a + n_b
            geom = (g_a, mask_a, g_b, mask_b, b_off, pos_a)
            pkey = (nblk, ov, int(n))
            gkey = ('vo', nblk, ov, int(n), pa_rows, pb_rows)
        else:
            _fi, mask_np, n_g, max_len = _g
            g = buf_i[off_i:off_i + n_g].view(nblk, max_len)
            pos_a = buf_i[off_i + n_g:off_i + n_g + max_len]
            off_i += n_g + max_len
            mask = buf_b[off_b:off_b + n_g].view(nblk, max_len)
            off_b += n_g
            geom = (g, mask, pos_a)
            pkey = (nblk, -1, int(n))
            gkey = ('so', nblk, int(n), pa_rows)
        try:
            _c1 = getattr(bid, '_pool_gpu', None)
            if _c1 is None:
                _c1 = {}
                bid._pool_gpu = _c1
            _c1[pkey] = parts
            _c2 = getattr(bid, '_pool_geom', None)
            if _c2 is None:
                _c2 = {}
                bid._pool_geom = _c2
            _c2[gkey] = geom
        except (AttributeError, RuntimeError):
            pass

def _pool_ordered(Xas, Xbs, Zas, Zbs, block_ids, B_pos_a, B_pos_b, overlap, n, Fd, dev, B, order, order_is_identity=False):
    counts, ends, starts, last_idx, end_prev, starts_prev, ov_start, ov_len = _pool_parts(block_ids, B, overlap, n)
    ov = overlap
    _cpu = getattr(block_ids, '_pool_cpu', None)
    if _cpu is not None and len(_cpu[0]) == B:
        max_len = _cpu[1]
        _ov_hi = min(ov, _cpu[2]) if B > 1 else -1
        _gkey = ('vo', int(B), int(ov), int(n), int(B_pos_a.shape[0]), int(B_pos_b.shape[0]))
        _geom = getattr(block_ids, '_pool_geom', None)
        _geom = None if _geom is None else _geom.get(_gkey)
    elif ov_len.numel():
        _cmin, max_len, _ov_lo, _ov_hi = (int(v) for v in torch.stack([counts.min(), counts.max(), ov_len.min(), ov_len.max()]).tolist())
        _geom = None
    else:
        _cmin, max_len = (int(v) for v in torch.stack(torch.aminmax(counts)).tolist())
        _ov_lo = _ov_hi = None
        _geom = None
    if _geom is not None:
        g_a, mask_a, g_b, mask_b, b_off, _pos_a = _geom
    else:
        # Host form first (the same exact-integer build _pool_parts_host makes
        # for the parts — design notes §6.2): the host form produces values
        # identical to the on-device build below.  None means fall back to the
        # on-device form, whose values are identical.
        _host = None
        if _POOL_PARTS_HOST and _cpu is not None and len(_cpu[0]) == B:
            _host = _pool_geom_host_var(_cpu[0], B, ov, n, max_len, _ov_hi, B_pos_a.shape[0], B_pos_b.shape[0], dev)
        if _host is not None:
            g_a, mask_a, g_b, mask_b, b_off, _pos_a = _host
            _pool_geom_put(block_ids, _gkey, _host)
        else:
            # built outside inference mode for the same reason as _pool_parts: a
            # cache first filled during eval must still serve training (an
            # inference tensor saved for backward would raise); the mode guard
            # changes no values.
            with torch.inference_mode(False):
                pos = _range_cache(0, max_len, dev)
                _gp = starts[:, None] + pos[None, :]
                mask_a = _gp < ends[:, None]
                # These three index tensors feed gathers only; once the masks (which
                # need the unclamped values) are built, the clamps run in place — the
                # same clamp on the same values.
                g_a = _gp.clamp_(max=n - 1)
                _W_b = max(_ov_hi if _ov_hi is not None else 0, 0)
                op = _range_cache(0, _W_b, dev)
                mask_b = op[None, :] < ov_len[:, None]
                g_b = (ov_start[:, None] + op[None, :]).clamp_(max=n - 1)
                b_off = (ov_start[:, None] - starts_prev[:, None] + op[None, :]).clamp_(max=B_pos_b.shape[0] - 1)
                _lim = B_pos_a.shape[0] - 1
                _pos_a = pos if max_len <= B_pos_a.shape[0] else pos.clamp(max=_lim)
            if _cpu is not None and len(_cpu[0]) == B:
                _pool_geom_put(block_ids, _gkey, (g_a, mask_a, g_b, mask_b, b_off, _pos_a))
    # Every block's pooled vector is a function of its own block lanes and
    # overlap lanes only (the softmax normalization and the weighted sum are
    # both within one block), so the blocks are processed in chunks: each
    # chunk's gathers/mix are bit-identical to the same rows of the unchunked
    # form, and `comp` is written row-for-row.
    _Fd = int(Xas.shape[1])
    _Wb = int(g_b.shape[1])
    # Conservative per-block element count: Xam+Zam+Xbm+Zbm + WinZ + sc
    # (float lanes) plus the two boolean masks.
    _per_elems = (4 * int(max_len) + 4 * _Wb) * _Fd + 8 * (int(max_len) + _Wb)
    _chunk = _pool_block_chunk(B, _per_elems, Xas.element_size())
    _bpa = B_pos_a[_pos_a]
    comp = torch.empty(B, _Fd, device=dev, dtype=Xas.dtype)
    for _b0 in range(0, B, _chunk):
        _b1 = min(_b0 + _chunk, B)
        Zam = _take_2d(Zas, g_a[_b0:_b1])
        Zbm = _take_2d(Zbs, g_b[_b0:_b1])
        _ma = mask_a[_b0:_b1]
        _mb = mask_b[_b0:_b1]
        # Plain-broadcast position add (same form pool_blocks_single already
        # uses): pad lanes are overwritten with -inf right after, and their
        # incoming grad is exactly zero (MaskedFillBackward / softmax P=0), so
        # forward values and B_pos gradients are identical to the masked-multiply
        # form — without materialising a full [chunk, max_len, c] product.
        # += on the fresh _take_2d outputs: the same elementwise kernel with the
        # same values as the out-of-place add (AddBackward saves no tensor, and
        # the only downstream consumer chain — masked_fill_/cat/softmax — keeps
        # no pre-add version).
        Zam += _bpa[None]
        Zbm += B_pos_b[b_off[_b0:_b1]]
        # the fill mask broadcasts: masked_fill_ accepts a broadcastable mask,
        # so the same booleans reach every lane (broadcast reads are exact).
        Zam.masked_fill_(~_ma[:, :, None], float('-inf'))
        Zbm.masked_fill_(~_mb[:, :, None], float('-inf'))
        WinZ = torch.cat([Zam, Zbm], 1)
        # softmax 的反向保存其输出，nan_to_num 必须出位（原地会改写保存值，
        # SoftmaxBackward 版本检查失败）；值与任何原地形式逐位相同。
        sc = torch.nan_to_num(F.softmax(WinZ, dim=1))
        # Drop the logit side before the X gathers: sc fully determines the
        # weights, so the X lanes can be fetched afterwards.  Values are
        # unchanged (the same gathers), but the peak no longer holds both the
        # logit stack and the X-lane stack at once.
        del WinZ, Zam, Zbm
        Xam = _take_2d(Xas, g_a[_b0:_b1])
        Xbm = _take_2d(Xbs, g_b[_b0:_b1])
        sc_a = sc[:, :max_len]
        sc_b = sc[:, max_len:]
        comp[_b0:_b1] = (sc_a * Xam).sum(1) + (sc_b * Xbm).sum(1)
    # Identity-order path: when `order` is the cached arange, order[last_idx]
    # returns exactly last_idx (arange[i] == i, exact integers — index
    # transport, design notes §6.1).
    return (comp, last_idx if order_is_identity else order[last_idx], B)

def pool_blocks_single(X, Z, B_pos, block_ids, n_blocks=None, monotonic=False):
    n = X.shape[0]
    dev = X.device
    B = int(n_blocks) if n_blocks is not None else _n_blocks_cached(block_ids)
    order, _ident = ((_arange_cache(block_ids.numel(), block_ids.device), True) if monotonic else _block_order(block_ids, n))
    if _ident:
        Xs, Zs = (X, Z)
    else:
        Xs = _take_rows(X, order)
        Zs = _take_rows(Z, order)
    counts, ends, starts, last_idx = _pool_parts(block_ids, B, None, n)
    _cpu = getattr(block_ids, '_pool_cpu', None)
    _geom = None
    if _cpu is not None and len(_cpu[0]) == B:
        max_len = _cpu[1]
        _gkey = ('so', int(B), int(n), int(B_pos.shape[0]))
        _gc = getattr(block_ids, '_pool_geom', None)
        _geom = None if _gc is None else _gc.get(_gkey)
    else:
        _cmin, max_len = (int(v) for v in torch.stack(torch.aminmax(counts)).tolist())
    if _geom is not None:
        g, mask, _pos_a = _geom
    else:
        # host form first — same exact-integer argument as _pool_ordered (the
        # values are identical to the device build; only the location of the
        # integer math and the transport change, §6.1/§6.2).
        _host = None
        if _POOL_PARTS_HOST and _cpu is not None and len(_cpu[0]) == B:
            _host = _pool_geom_host_single(_cpu[0], B, n, max_len, B_pos.shape[0], dev)
        if _host is not None:
            g, mask, _pos_a = _host
            _pool_geom_put(block_ids, _gkey, _host)
        else:
            # same inference_mode(False) guard as _pool_parts: the cached tensors
            # must be normal tensors even when first built during eval.
            with torch.inference_mode(False):
                pos = _range_cache(0, max_len, dev)
                _gp = starts[:, None] + pos[None, :]
                mask = _gp < ends[:, None]
                # g feeds the gathers only; the mask needs the unclamped values, so it is
                # built first and the clamp runs in place (same kernel, same integers).
                g = _gp.clamp_(max=n - 1)
                _pos_a = pos if max_len <= B_pos.shape[0] else pos.clamp(max=B_pos.shape[0] - 1)
            if _cpu is not None and len(_cpu[0]) == B:
                _pool_geom_put(block_ids, _gkey, (g, mask, _pos_a))
    # Block-chunked exactly as _pool_ordered: each block's pooled vector is a
    # function of its own lanes only, so the grouping never changes a value.
    _Fd = int(Xs.shape[1])
    # Conservative per-block element count: Xb + Zb(WinZ) + sc plus the mask.
    _per_elems = 3 * int(max_len) * _Fd + 4 * int(max_len)
    _chunk = _pool_block_chunk(B, _per_elems, Xs.element_size())
    _bpos = B_pos[_pos_a]
    comp = torch.empty(B, _Fd, device=dev, dtype=Xs.dtype)
    for _b0 in range(0, B, _chunk):
        _b1 = min(_b0 + _chunk, B)
        Xb = _take_2d(Xs, g[_b0:_b1])
        Zb = _take_2d(Zs, g[_b0:_b1])
        WinZ = Zb
        # Same plain-broadcast position add as _pool_ordered: the fill right
        # after overwrites the pad lanes with -inf, and the incoming grad at
        # those lanes is exactly zero (MaskedFillBackward / softmax P=0), so
        # the values and the B_pos gradients are identical to the
        # masked-multiply and boolean-index forms.
        # += on the fresh _take_2d output (same elementwise kernel and values as
        # the out-of-place add; see _pool_ordered).
        WinZ += _bpos[None]
        # broadcast fill mask (same argument as _pool_ordered): identical booleans
        # per lane.
        WinZ.masked_fill_(~mask[_b0:_b1, :, None], float('-inf'))
        # 同 _pool_ordered：softmax 反向保存输出，nan_to_num 出位。
        sc = torch.nan_to_num(F.softmax(WinZ, dim=1))
        comp[_b0:_b1] = (sc * Xb).sum(1)
    # same identity-order fast path as _pool_ordered: order[last_idx] over the
    # cached arange returns exactly last_idx (index transport, §6.1).
    return (comp, last_idx if _ident else order[last_idx], B)

def _rank_blocks(masked, B, ties, k=None):
    # The sort indices are consumed by the int64 gathers below and narrowed
    # to int32 on the kept slice only.  Same integers on every path (index
    # transport, design notes §6.1).
    #
    # Only the first k columns are ever consumed by _indexer_selection.  For
    # k < B, return exactly the columns the full stable sort would produce:
    # the k-th largest value is found with torch.topk, all strictly larger
    # entries are kept, and the remaining slots are filled from equal entries
    # in original column order (the stable-sort tie rule).  The selected k
    # candidates are then ordered by score with a stable argsort — the same
    # value/column order as the first k columns of the full sort.
    kk = B if k is None else min(max(int(k), 0), B)
    if kk <= 0:
        return torch.empty((masked.shape[0], 0), dtype=torch.long, device=masked.device)
    if kk >= B:
        if ties == 'earliest':
            return torch.sort(masked, dim=1, stable=True, descending=True).indices
        order = torch.sort(masked.flip(1), dim=1, stable=True, descending=True).indices
        return torch.sub(B - 1, order, out=order)
    src = masked.flip(1) if ties == 'latest' else masked
    kth = torch.topk(src, kk, dim=1, sorted=True).values[:, -1:]
    gt = src > kth
    eq = src == kth
    # gt has at most kk - 1 entries (the k-th value itself is not strictly
    # greater), so `need` is non-negative and selects exactly kk columns.
    need = (kk - gt.sum(dim=1, keepdim=True)).clamp_min(0)
    sel = gt | (eq & (eq.cumsum(dim=1) <= need))
    rank = sel.cumsum(dim=1) - 1
    n, _ = src.shape
    # Scatter the selected source columns into [n, kk] (column order); the
    # sentinel column kk absorbs the non-selected lanes and is sliced away.
    cols = torch.arange(B, device=src.device)[None, :].expand(n, B)
    buf = torch.zeros(n, kk + 1, dtype=torch.long, device=src.device)
    buf.scatter_(1, torch.where(sel, rank, kk), cols)
    cand = buf[:, :kk]
    order = torch.argsort(src.gather(1, cand), dim=1, stable=True, descending=True)
    cand = cand.gather(1, order)
    if ties == 'latest':
        cand = B - 1 - cand
    return cand

def _indexer_selection(scores, causal, k, ties='earliest', out_valid=None):
    n, B = scores.shape
    if ties not in ('earliest', 'latest'):
        raise ValueError(f'unknown tie rule {ties!r}')
    k_req = max(int(k), 0)
    if k_req <= 0:
        if out_valid is not None:
            out_valid[:] = [torch.zeros(n, 0, dtype=torch.bool, device=scores.device)]
        return torch.empty(n, 0, device=scores.device, dtype=torch.int32)
    k = min(k_req, B)
    finite = torch.isfinite(scores)
    cf = causal & finite
    # torch.where picks bit-identical values (scores or the -inf constant) —
    # the same selection masked_fill produced.
    masked = torch.where(cf, scores, float('-inf'))
    order = _rank_blocks(masked, B, ties, k=min(k, B))
    _w_keep = min(int(k), B)
    # Only the first _w_keep order columns are ever consumed (the k >= B and
    # k < B forms both reduce to usable[:, :_w_keep]), so the mask gather runs
    # on that column slice directly: gathering a column subset yields exactly
    # the corresponding columns of the full gather — the same booleans (index
    # transport, design notes §6.1).  Gathering the conjoined mask gives
    # exactly the elementwise AND of the two per-mask gathers.
    order_w = order[:, :_w_keep]
    keep = cf.gather(1, order_w)
    dest = keep.cumsum(dim=1)
    dest -= 1
    # int64 -> int32 narrowing, on the kept slice only — same integers
    # (index transport, design notes §6.1).  keep/dest are already exactly
    # _w_keep wide (keep = cf.gather(1, order_w), dest = keep.cumsum).
    order_k = order_w.to(torch.int32)
    # Sentinel-column scatter. Within a row the kept lanes hold distinct
    # cumsum ranks (0..cnt-1), so every scatter target column has exactly one
    # writer; lanes failing `keep` are redirected to a sentinel column that is
    # sliced away (its undefined winner never leaves the pad).  The kept
    # columns are therefore the same exact integers as the masked-assignment
    # form (host-side integer equivalence, design notes §6.2).
    tgt = torch.where(keep, dest, _w_keep)
    kept = torch.full((n, _w_keep + 1), -1, dtype=torch.int32, device=scores.device)
    kept.scatter_(1, tgt, order_k)
    kept = kept[:, :_w_keep]
    # last_tok from either pooling path is non-decreasing (a cumsum of
    # non-negative counts followed by the monotone (ends-1).clamp(0, n-1)),
    # so every row of `causal` is a prefix of Trues: the first-max argmax
    # returns 0 for any row with a readable block, and 0 for an all-False
    # row (ties resolve to index 0) — identically zero.
    # The padded lanes are masked downstream via the selection-valid mask
    # either way, so the zeros are the same integers the argmax produced
    # (host-side integer equivalence, design notes §6.2).
    # the invalid lanes hold exactly the -1 sentinel, so clamping the kept
    # buffer to 0 replaces the where(valid, kept, 0) selection lane-for-lane
    # (block ids are >= 0 and are never rewritten) — same integers.  The
    # validity mask is an exact function of the pre-clamp kept buffer and is
    # only consumed when the caller asks for it, so it (and the pad-block
    # constant, needed only when k exceeds the available lanes) is built only
    # on the branch that reads it.
    if out_valid is not None:
        valid_cols = kept >= 0
        if k_req > _w_keep:
            valid_cols = torch.cat([valid_cols, torch.zeros(n, k_req - _w_keep, dtype=torch.bool, device=scores.device)], dim=1)
        out_valid[:] = [valid_cols]
    kept.clamp_min_(0)
    idx_out = kept
    if k_req > _w_keep:
        pad_blk = torch.zeros(n, dtype=torch.int32, device=scores.device)
        pad = pad_blk[:, None].expand(n, k_req - _w_keep)
        idx_out = torch.cat([idx_out, pad], dim=1)
    return idx_out

_TOPK_CLAMP_WARNED = set()

def _pool_ends_host(bid):
    # 宿主侧「块结束位置（独占）」表：cumsum 的分段器 counts（_pool_cpu[0]）。
    # 池化侧 last_tok 由同一 counts 的 cumsum-1 生成（monotonic 顺序下
    # order 为恒等），且 counts 逐项 ≤ n，clamp(0, n-1) 从不改写任何值，
    # 故 ends[j]-1 与 last_tok[j] 是同一批整数（design notes §6.2 的宿主侧
    # 整数等价）。缓存挂在不可变的 block_ids 对象上（与 _pool_parts 同一
    # 约定）；缺失/形状不符时返回 None，调用方回退全宽。
    ends = getattr(bid, '_pool_ends_host', None)
    if ends is None:
        _cpu = getattr(bid, '_pool_cpu', None)
        if _cpu is None:
            return None
        try:
            ends = [int(v) for v in itertools.accumulate(_cpu[0])]
        except (TypeError, ValueError):
            return None
        try:
            bid._pool_ends_host = ends
        except (AttributeError, RuntimeError):
            pass
    return ends

def lightning_indexer(H, comp_kv, last_tok, W_DQ, W_DK, W_w, nIH, topk, return_mask=True, query_chunk=2048, random_select=False, pre_qI=None, pre_w=None, out_valid=None, causal_ends_host=None, need_soft=True):
    n = H.shape[0]
    B = comp_kv.shape[0]
    dev = H.device
    hd = W_DQ.shape[1] // nIH
    if random_select:
        qI = kI = w_idx = None
    else:
        qI = (pre_qI if pre_qI is not None else H @ W_DQ).view(n, nIH, hd).transpose(0, 1)
        kI = (comp_kv @ W_DK).view(B, nIH, hd).transpose(0, 1)
        w_idx = pre_w if pre_w is not None else F.linear(H, W_w)
    pos = _arange_cache(n, dev)
    # causal (and its negation) are exact bool functions of (pos, last_tok).
    # The fixed segmenter's pooled last_tok is ONE shared cached tensor read
    # by every row/layer/step, so the pair is cached on that tensor object —
    # the same bitwise-safe cache-on-immutable-object argument as
    # _pool_parts/_pool_geom (design notes §6.2).  Both masks are only ever
    # read downstream (masked_fill sources, no_grad selection); the dynamic
    # per-row last_tok objects simply miss and rebuild (identical values).
    # Built outside inference mode so a pair first filled during eval can
    # still serve training (masked_fill_ saves the mask for backward).
    _rkey = (int(n), int(last_tok.shape[0]))
    _rc = getattr(last_tok, '_readable_cache', None)
    if _rc is not None and _rc[0] == _rkey:
        causal, _ncausal = _rc[1:3]
    else:
        with torch.inference_mode(False):
            causal = block_readable(pos, last_tok)
            _ncausal = ~causal
        try:
            last_tok._readable_cache = (_rkey, causal, _ncausal)
        except (AttributeError, RuntimeError):
            pass
    k = min(topk, B)
    if k < int(topk):
        _wk = (int(topk), int(B))
        if _wk not in _TOPK_CLAMP_WARNED:
            _TOPK_CLAMP_WARNED.add(_wk)
            print(f'[lightning_indexer] requested topk={int(topk)} but only B={int(B)} blocks are available; effective selection is clamped to all {int(k)} blocks for this call')
    if random_select:
        _ri_key = (_dev_str(dev), 'rs_ri', int(n))
        _jc_key = (_dev_str(dev), 'rs_jc', int(B))
        _ri = _IDX_CACHE.get(_ri_key)
        if _ri is None:
            with torch.inference_mode(False):
                _ri = torch.arange(n, device=dev, dtype=torch.float64)[:, None]
            _cache_put(_IDX_CACHE, _ri_key, _ri, _MASK_CACHE_BUDGET_BYTES, _idx_cache_total)
        _jc = _IDX_CACHE.get(_jc_key)
        if _jc is None:
            with torch.inference_mode(False):
                _jc = (torch.arange(B, device=dev, dtype=torch.float64)[None, :] * 78.233).contiguous()
            _cache_put(_IDX_CACHE, _jc_key, _jc, _MASK_CACHE_BUDGET_BYTES, _idx_cache_total)
    if query_chunk is None:
        raise ValueError("lightning_indexer: query_chunk=None is not 'use the default'; pass the default (2048) explicitly or omit the argument")
    query_chunk = max(1, min(int(query_chunk), n))
    # chunk 行数上限（下方注释已论证逐 chunk 融合与全宽形式逐行逐位一致；
    # 分块边界是 (n, B, nIH, dtype) 的纯函数，跨机器确定）：每 chunk 同时
    # 物化 raw [nIH,·,B]、scores、masked、int64 order 与 keep/dest/tgt 等
    # 选择表。
    _iper = int(B) * (int(nIH) * int(H.element_size()) + 40) + 64
    if _iper > 0:
        query_chunk = max(1, min(query_chunk, (256 << 20) // _iper))
    _iscale = hd ** (-0.5)
    # 选择 + soft 尾部与打分同一逐 chunk 节奏融合执行：其中每个算子（masked
    # where/稳定排序/cumsum/scatter/logsumexp/gather/逐元素 exp）都按 query
    # 行独立——chunk 行切片上的值与全宽形式的对应行逐位一致（同 _indexer_
    # selection 内部 grid 只标记 chunk 内行号；CopySlices 反向把逐行梯度
    # 切片原样回传，与 forward 的 outs[b] = o 同一 transport 论证，§6.1）。
    # 全宽的 [n, B] scores / masked / int64 order 因此从不物化。
    _kw = max(min(int(topk), int(B)), 0)
    # `causal_ends_host` is kept in the signature for callers, but the score
    # columns are NOT narrowed to the readable prefix.  Narrowing changes the
    # GEMM shape, and a GEMM is not bitwise column-independent across shapes:
    # the prefix scores (and therefore the log-sum-exp and near-tie ordering)
    # can drift by ULPs.  The repo's comparability contract requires the same
    # inputs to produce the same bits as the un-narrowed implementation, so the
    # score/soft path stays full width; top-k selection goes through the
    # stable top-k in `_rank_blocks`.
    idx = torch.empty(n, _kw, device=dev, dtype=torch.int32)
    soft = None
    if out_valid is None and return_mask:
        out_valid = []
    _valid_parts = [] if out_valid is not None else None
    out_dtype = None
    for s in range(0, n, query_chunk):
        e = min(s + query_chunk, n)
        if random_select:
            # same elementwise op chain with in-place tails: each op is the
            # same kernel on the same values (no fusion, no reassociation —
            # note x-floor(x) is NOT frac() for negatives), so the scores are
            # bit-identical.
            z = _ri[s:e] * 12.9898 + _jc
            z.sin_()
            z.mul_(43758.5453)
            z.sub_(torch.floor(z))
            z.mul_(2.0).sub_(1.0)
            c = z.to(H.dtype)
        else:
            raw = torch.einsum('ind,ibd->inb', qI[:, s:e], kI)
            raw.mul_(_iscale)
            F.relu(raw, inplace=True)
            c = torch.einsum('inb,ni->nb', raw, w_idx[s:e])
        if out_dtype is None:
            out_dtype = c.dtype
            if need_soft:
                soft = torch.empty(n, _kw, device=c.device, dtype=out_dtype)
        _ccausal = causal[s:e]
        with torch.no_grad():
            _box = []
            _ic = _indexer_selection(c, _ccausal, k, out_valid=_box)
        idx[s:e] = _ic
        if _valid_parts is not None:
            _valid_parts.append(_box[0])
        if not need_soft:
            # need_soft=False 的调用方（无梯度评估）只消费 idx：soft 是
            # gathered_attention 直通分支（该分支在无梯度下被 _soft_live 整体
            # 跳过）的唯一下游，故整条 soft 尾部不物化（不改变 idx 的任何
            # 整数，也不改变下游任何前向 bit）。
            del c, _ic
            continue
        c.masked_fill_(_ncausal[s:e], float('-inf'))
        _lse = torch.logsumexp(c, dim=-1, keepdim=True)
        # the gather output is fresh and nothing downstream needs the pre-subtract
        # values, so the subtraction runs in place (same kernel, same values, and
        # the sub_ node records the identical broadcast gradient); exp_ is the
        # same story (its backward saves the output, as the out-of-place form).
        # nan_to_num 必须出位：exp_ 的反向保存其输出张量，原地 nan_to_num_ 会
        # 改写该保存值、使 ExpBackward 的版本检查失败（反向报错）。
        soft[s:e] = torch.nan_to_num(c.gather(1, _ic.long()).sub_(_lse).exp_())
        del c, _lse, _ic
    if _valid_parts is not None:
        out_valid[:] = [_valid_parts[0] if len(_valid_parts) == 1 else torch.cat(_valid_parts, 0)]
    m = None
    if return_mask:
        valid_cols = out_valid[0]
        with torch.no_grad():
            # Build the (n, B) mask: only the valid selections are written
            # (1.0 each; duplicate writes of the same cell are idempotent), so
            # the values are bit-identical to the sentinel form's [.., :B]
            # slice.
            _il = idx.long()
            _rows = _arange_cache(n, dev)[:, None].expand(n, _il.shape[1])
            m = torch.zeros(n, B, device=dev, dtype=out_dtype)
            m[_rows[valid_cols], _il[valid_cols]] = 1.0
            m.masked_fill_(_ncausal, 0.0)
    return (m, idx, soft)

class _SinkWiden(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, sink_logits):
        n, H, M = logits.shape
        z = logits.new_empty(n, H, M + 1)
        z[..., 1:] = logits
        z[..., 0:1] = sink_logits.view(1, -1, 1)
        return z

    @staticmethod
    def backward(ctx, gz):
        return (gz[..., 1:], gz[..., 0].sum(dim=0))

def sink_softmax(logits, sink_logits, dim=-1):
    if dim not in (-1, logits.dim() - 1):
        raise ValueError('sink_softmax only supports the last dim')
    _diff = torch.is_grad_enabled() and (getattr(logits, 'requires_grad', False) or (sink_logits is not None and getattr(sink_logits, 'requires_grad', False)))
    if _diff:
        return _sink_softmax_impl(logits, sink_logits, dim)
    with torch.no_grad():
        return _sink_softmax_impl(logits, sink_logits, dim).detach()

def _sink_softmax_impl(logits, sink_logits, dim=-1):
    if sink_logits is None:
        return torch.nan_to_num(torch.softmax(logits, dim))
    z = _SinkWiden.apply(logits, torch.nan_to_num(sink_logits))
    return torch.nan_to_num(torch.softmax(z, -1)[..., 1:])

def _sink_split_softmax(logits, sink_logits, want_sink=True):
    if sink_logits is None:
        return (sink_softmax(logits, None, -1), None)
    z = _SinkWiden.apply(logits, torch.nan_to_num(sink_logits))
    soft = torch.softmax(z, -1)[..., 1:]
    return (soft, 1.0 - soft.sum(-1) if want_sink else None)

_FO_FULL_CAP_BYTES = 64 << 20
_FO_MINL = {}

def _finfo_min(dtype):
    # torch.finfo(dtype).min is a pure function of the dtype — the same
    # constant on every call, so it is memoised per dtype (host-side, no
    # tensor values involved).
    v = _FO_MINL.get(dtype)
    if v is None:
        v = torch.finfo(dtype).min
        _FO_MINL[dtype] = v
    return v

def gathered_attention(q, k_blk, v_blk, topk_idx, last_tok, k_sw, v_sw, w, scale, q_chunk=128, soft=None, sink_logits=None, mem_budget_bytes=None, sel_valid=None, topk_distinct=False, soft_form=False):
    n = q.shape[0]
    dev = q.device
    if mem_budget_bytes is None:
        mem_budget_bytes = _attn_transient_budget(dev)
    if mem_budget_bytes is None:
        # CPU/未知设备：使用固定上限（chunk 是纯行划分，任何宽度下逐行值
        # 逐位相同）。
        mem_budget_bytes = 256 << 20
    _heads = q.shape[1] if q.dim() == 3 else 1
    _M = max(1, int(topk_idx.shape[1]) + int(w))
    # 分块行字节估算（保守上界）。
    _per_row = 4 * _M * _heads * int(q.shape[-1]) + 2 * _M * _heads
    _bytes_per_chunk_row = _per_row * q.element_size()
    _bindable = mem_budget_bytes is not None and math.isfinite(float(mem_budget_bytes))
    if _bindable and _bytes_per_chunk_row > 0:
        _max_rows = max(1, int(mem_budget_bytes) // _bytes_per_chunk_row)
        q_chunk = min(q_chunk, _max_rows)
    q_chunk = max(1, min(int(q_chunk), n))
    out = torch.empty_like(q)
    pos_all = _arange_cache(n, dev)
    win_idx, win_valid_all = _window_geometry(n, w, dev)
    _MINL = _finfo_min(k_blk.dtype)
    # The V halves are laid out contiguously once and gathered with one
    # combined row index per chunk: the selected rows of `v_all` are exactly
    # the two gather outputs concatenated in the same order, so Vset holds
    # bit-identical values.  The plain path does the same for K (the soft path
    # keeps two contiguous K gathers, see below).  Window rows are offset past
    # the block rows (index transport, §6.1).
    _nblk_rows = k_blk.shape[0]
    _has_sw = k_sw is not None and v_sw is not None
    if _has_sw:
        v_all = torch.cat([v_blk, v_sw], 0)
        # The K halves are concatenated only for the plain (soft=None,
        # soft_form=False) path; the soft path keeps its two contiguous
        # gathers, which the two logits einsums consume directly.
        k_all = torch.cat([k_blk, k_sw], 0) if (soft is None and not soft_form) else None
    else:
        k_all, v_all = (k_blk, v_blk)
    _win_off = _nblk_rows if _has_sw else 0
    # Index transport (design notes §6.1): int32 and int64 index tensors hold
    # the same exact integers, and advanced indexing / equality comparisons on
    # them return bitwise identical values.
    topk_l = topk_idx if topk_idx.dtype in (torch.int32, torch.int64) else topk_idx.long()
    # The read-everything HCA arm passes the cached arange expansion (every
    # row is exactly arange(B)): last_tok[topk_l] then gathers one identical
    # copy of last_tok per row, so the same bool table comes out of a plain
    # broadcast comparison against last_tok — same integers compared, bitwise
    # the same output (§6.1).  Identity with the cached table (not a content
    # scan) is the guard, so any other caller falls back to the gather form
    # unchanged.
    _arange_sel = (topk_l.shape[1] == last_tok.shape[0]
                   and topk_l is _IDX_CACHE.get((_dev_str(dev), 'arangeexp', int(n), int(topk_l.shape[1]))))
    sel_blk_all = (pos_all[:, None] > last_tok[None, :]) if _arange_sel else (pos_all[:, None] > last_tok[topk_l])
    # the keep mask and the [selected|window] valid mask are per-row functions of
    # the same inputs, so their full-width forms are bitwise identical to the
    # per-chunk forms; they are hoisted out of the chunk loop under
    # _FO_FULL_CAP_BYTES (the chunked fallback below covers very wide top-k
    # sweeps).
    # `topk_distinct=True` asserts every row of topk_idx holds distinct block
    # ids (the read-everything HCA arm passes an arange expansion), in which
    # case first_occurrence_mask is provably all-True — the keep mask is then
    # identical to the computed form.
    if n * topk_l.shape[1] * topk_l.shape[1] <= _FO_FULL_CAP_BYTES:
        keep_all = sel_valid
        if topk_l.shape[1] > 1 and not topk_distinct:
            _ddk_all = first_occurrence_mask(topk_l)
            keep_all = _ddk_all if keep_all is None else keep_all & _ddk_all
        if keep_all is not None:
            sel_blk_all &= keep_all
        valid_all = torch.cat([sel_blk_all, win_valid_all], dim=1)
        # full-width forms of the per-chunk negation / any-reduction: the
        # elementwise and reduction values per row are identical to the
        # per-chunk forms.
        nvalid_all = ~valid_all[:, None, :]
        anyrow_all = valid_all.any(-1) if sink_logits is None else None
    else:
        keep_all = valid_all = nvalid_all = anyrow_all = None
    _nb = int(topk_l.shape[1])
    # The chunk-invariant dispatch is computed once, above the row-chunk
    # loop: which mask/validity source each chunk reads and which of the two
    # attention forms runs are pure functions of the call-level arguments.
    # The loop body itself — and the tensor values it produces — is
    # unchanged.
    _has_all = valid_all is not None
    _has_sv = sel_valid is not None
    _dedup = (not topk_distinct) and _nb > 1
    _has_sink = sink_logits is not None
    # soft 只经直通 (soft_attn - soft_attn.detach()).add_(attn) 进入前向值：
    # soft_attn 是 _sink_split_softmax/nan_to_num 之后的输出、恒有限，x - x
    # 精确相消为 +0.0，且 0.0 加回 attn 逐位不变（attn 全元非负，无 -0.0
    # 符号翻转），故跳过该分支不改变任何前向 bit；该子图对 soft/q/K/V 的梯度
    # 也恒为 0（d(x - detach(x)) = 0），因此仅当确有相关张量需要梯度时才保留
    # 它。训练/checkpoint 重算都在 grad 开启下运行，条件与首次前向一致。
    _soft_live = soft is not None and torch.is_grad_enabled() and any(
        _t is not None and getattr(_t, 'requires_grad', False)
        for _t in (soft, q, k_blk, v_blk, k_sw, v_sw, sink_logits))
    _tk2 = _take_2d
    _einsum = torch.einsum
    if _has_all:
        # Only the keep/nvalid/anyrow slices are consumed below; the whole-row
        # sel_blk_all / valid_all tables are fully absorbed into them.
        _plan = [(None if keep_all is None else keep_all[_s:_s + q_chunk], nvalid_all[_s:_s + q_chunk], None if anyrow_all is None else anyrow_all[_s:_s + q_chunk]) for _s in range(0, n, q_chunk)]
        del sel_blk_all, valid_all
    else:
        _plan = None
    for s in range(0, n, q_chunk):
        e = s + q_chunk
        if e > n:
            e = n
        qseg = q[s:e]
        if _plan is not None:
            keep, nvalid, anyrow = _plan[s // q_chunk]
        else:
            sel_blk = sel_blk_all[s:e]
            keep = None
            if _has_sv:
                keep = sel_valid[s:e]
            if _dedup:
                _ddk = first_occurrence_mask(topk_l[s:e])
                keep = _ddk if keep is None else keep & _ddk
            if keep is not None:
                sel_blk = sel_blk & keep
            valid = torch.cat([sel_blk, win_valid_all[s:e]], dim=1)
            nvalid = ~valid[:, None, :]
            anyrow = None if _has_sink else valid.any(-1)
        # Two half-gathers (and the same for V): the block half gathers the
        # k_blk rows topk_l addresses (rows 0..n_blk-1 of the stacked tensor
        # == k_blk rows) and the window half gathers k_sw at win_idx (rows
        # n_blk+j of the stacked tensor == row j of k_sw).  index_select
        # returns exactly the same source rows, so the gathered values — and
        # hence every downstream value — are bit-identical to the
        # cat-then-gather form (index re-plumbing only, design notes §6.1).
        _sel_rows = topk_l[s:e]
        _win_local = win_idx[s:e]
        _win_rows = _win_local + _win_off if _win_off else _win_local
        # `_has_sw` is the contract for stacking window lanes: the row offset
        # may be 0 when k_blk holds no rows, and that is still a valid
        # stacked-row base.
        _rows = torch.cat([_sel_rows, _win_rows], 1) if _has_sw else _sel_rows
        # soft_form=True 在 soft=None 时仍选用软分支的硬部分解（两个半段
        # logits + cat），供调用方在跳过（无梯度下已死的）soft 尾部后保持
        # 硬分支逐位一致；它不改变任何取值，只固定走哪套既有分解。
        if soft is None and not soft_form:
            Kset = _tk2(k_all, _rows)
            logits = _einsum('qhd,qmhd->qhm', qseg, Kset)
            # K is consumed by the logits only; release it before gathering V
            # so the two [chunk, M, h, d] lane stacks never coexist.  The same
            # rows are gathered below, so every value is unchanged.
            del Kset
            if scale != 1.0:
                # x·1.0 is the IEEE identity (and grad·1.0 is bit-identical on
                # the backward Mul node), so scaling by 1.0 is a no-op.
                logits.mul_(scale)
            logits.masked_fill_(nvalid, _MINL)
            attn, _sink_unused = _sink_split_softmax(logits, sink_logits, want_sink=False)
            if not _has_sink:
                attn.mul_(anyrow[:, None, None])
            out[s:e] = _einsum('qhm,qmhd->qhd', attn, _tk2(v_all, _rows))
            continue
        # The soft branch reads its K halves as two contiguous gathers: the
        # concatenated V gather below holds exactly the same rows as two
        # separate gathers + cat, while the per-half K gathers feed the two
        # logits einsums with the same operands (index re-plumbing only,
        # design notes §6.1).  V is gathered only after the attention weights
        # are final, so its lane stack does not coexist with the K halves.
        kb = _tk2(k_blk, _sel_rows)
        kw = _tk2(k_sw, _win_local)
        blk_logits = _einsum('qhd,qmhd->qhm', qseg, kb)
        if scale != 1.0:
            blk_logits.mul_(scale)
        win_logits = _einsum('qhd,qmhd->qhm', qseg, kw)
        if scale != 1.0:
            win_logits.mul_(scale)
        logits = torch.cat([blk_logits, win_logits], -1)
        del kb, kw, blk_logits, win_logits
        if _soft_live:
            soft_g = soft[s:e]
            if keep is not None:
                soft_g = soft_g * keep
            soft_log_g = torch.log(soft_g.clamp_min(1e-12))
            soft_log_g.masked_fill_(soft_g <= 0, _MINL)
            soft_logits = logits.clone()
            soft_logits[:, :, :_nb].add_(soft_log_g[:, None, :]).clamp_min_(_MINL)
            soft_logits.masked_fill_(nvalid, _MINL)
            soft_attn, _sink_unused = _sink_split_softmax(soft_logits, sink_logits, want_sink=False)
        logits.masked_fill_(nvalid, _MINL)
        attn, _sink_unused = _sink_split_softmax(logits, sink_logits, want_sink=False)
        if _soft_live:
            attn = (soft_attn - soft_attn.detach()).add_(attn)
        if not _has_sink:
            attn.mul_(anyrow[:, None, None])
        out[s:e] = _einsum('qhm,qmhd->qhd', attn, _tk2(v_all, _rows))
    return out

def _gathered_attention_trainable(*args, **kwargs):
    # 训练时包一层非重入 checkpoint，反向按同一确定性分解重算前向。函数内
    # 无 RNG，且 mem_budget 经 _ATB_EPOCH 缓存在一次运行内不变，因此 chunk
    # 边界与每个中间值逐位相同，梯度与直调形式逐位一致（design notes §6.1
    # 的 transport 论证在「重算同一纯函数」上的形式）。
    if torch.is_grad_enabled() and args[0].requires_grad:
        return torch.utils.checkpoint.checkpoint(gathered_attention, *args, use_reentrant=False, **kwargs)
    return gathered_attention(*args, **kwargs)

def _pool_trainable(fn, *args, **kwargs):
    # 训练时包一层非重入 checkpoint，反向重算同一纯函数：池化内部无 RNG，
    # 整数几何缓存挂在不可变的 block_ids 对象上（_pool_parts / _pool_geom，
    # design notes §6.2），反向重算必命中同一几何，前向值与梯度逐位一致；
    # int 型的 n_blocks 返回值原样透传。eval 下（无梯度）直调。
    if torch.is_grad_enabled() and args[0].requires_grad:
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False, **kwargs)
    return fn(*args, **kwargs)

@dataclass
class AttnCfg:
    kind: str = 'csa'
    chunking: str = 'fixed'
    dynamic: bool = False
    block_size: int = 4
    overlap: int = 8
    index_topk: int = 32
    cos_threshold: float = 0.5
    adaptive: bool = False
    target_block_tokens: int = 16
    temperature: float = 0.1
    min_block: int = 2
    max_block: int = 128
    sliding_window: int = 128
    c_kv: int = 128
    c_index: int = 64
    n_index_heads: int = 4
    full_cosine: bool = False
    window: int = 0
    comp_lambda_mult: float = 1.0
    indexer_mode: str = 'learned'
    content_mode: str = 'pooled'
    use_sink: bool = True
    full_sink: bool = False
    fuse_mode: str = 'none'
    fuse_kernel: int = 5

class HybridAttention(nn.Module):

    def __init__(self, d_model, n_heads, d_head, cfg: AttnCfg):
        super().__init__()
        self.cfg = cfg
        self.nh, self.hd = (n_heads, d_head)
        self.kv_dim = 2 * cfg.c_kv
        self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_o = nn.Linear(n_heads * d_head, d_model, bias=False)
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        self.last_gate_mean = None
        self.need_reg = True
        self._stats = None
        self._cuts = None
        self.fuse_conv = None
        # Fused-weight cache (eval-only): the gateway is `not
        # torch.is_grad_enabled()`, so training forward passes always rebuild
        # the cat inside the autograd graph exactly as before.  Under eval the
        # concat values are pure functions of the source weights, so keying on
        # their autograd `_version` counters (bumped by every in-place
        # optimizer update) makes a stale hit impossible.
        self._wcat_cache = {}
        if cfg.kind in ('csa', 'hca') and cfg.chunking == 'cosine_learnable' and (cfg.fuse_mode == 'learned'):
            self.fuse_conv = nn.Conv1d(d_model, d_model, cfg.fuse_kernel, groups=d_model, bias=False)
            with torch.no_grad():
                self.fuse_conv.weight.zero_()
                self.fuse_conv.weight[:, :, -1] = 1.0
        if cfg.kind == 'full':
            self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
            self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
            nn.init.xavier_uniform_(self.W_k.weight)
            nn.init.xavier_uniform_(self.W_v.weight)
            self.W_aKV = None
            self.W_kvhead = None
            self.sink = nn.Parameter(torch.zeros(n_heads)) if cfg.full_sink else None
            return
        self.W_aKV = nn.Parameter(torch.empty(d_model, self.kv_dim))
        self.W_kvhead = nn.Linear(self.kv_dim, 2 * n_heads * d_head, bias=False)
        nn.init.xavier_uniform_(self.W_aKV)
        nn.init.xavier_uniform_(self.W_kvhead.weight)
        if cfg.kind not in ('csa', 'hca'):
            return
        if cfg.kind == 'hca':
            self.W_bKV = None
            self.W_bZ = None
            self.W_aZ = nn.Parameter(torch.empty(d_model, self.kv_dim))
            self.B_pos_a = nn.Parameter(torch.zeros(int(cfg.max_block) + 1, self.kv_dim))
            self.sink = nn.Parameter(torch.zeros(n_heads)) if getattr(cfg, 'use_sink', True) else None
            nn.init.xavier_uniform_(self.W_aZ)
            if cfg.chunking == 'cosine_learnable':
                Tc = max(cfg.temperature, 0.001)
                Lb = max(cfg.target_block_tokens, 2)
                self.delta_logit = nn.Parameter(torch.tensor(float(Tc * math.log(1.0 / (Lb - 1)))))
            else:
                self.delta_logit = None
            return
        self.W_bKV = nn.Parameter(torch.empty(d_model, self.kv_dim))
        self.W_aZ = nn.Parameter(torch.empty(d_model, self.kv_dim))
        self.W_bZ = nn.Parameter(torch.empty(d_model, self.kv_dim))
        self.B_pos_a = nn.Parameter(torch.zeros(int(cfg.max_block) + 1, self.kv_dim))
        self.B_pos_b = nn.Parameter(torch.zeros(int(cfg.max_block) + 1, self.kv_dim))
        self.sink = nn.Parameter(torch.zeros(n_heads)) if getattr(cfg, 'use_sink', True) else None
        self.W_DQ = nn.Parameter(torch.empty(d_model, cfg.c_index))
        self.W_DK = nn.Parameter(torch.empty(self.kv_dim, cfg.c_index))
        self.W_w = nn.Linear(d_model, cfg.n_index_heads, bias=False)
        if cfg.chunking == 'cosine_learnable':
            Tc = max(cfg.temperature, 0.001)
            Lb = max(cfg.target_block_tokens, 2)
            self.delta_logit = nn.Parameter(torch.tensor(float(Tc * math.log(1.0 / (Lb - 1)))))
        else:
            self.delta_logit = None
        for p in [self.W_bKV, self.W_aZ, self.W_bZ, self.W_DQ, self.W_DK]:
            nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.W_w.weight)

    def _fuse(self, x):
        if self.fuse_conv is None:
            return x
        k = self.cfg.fuse_kernel
        z = F.pad(x.t().unsqueeze(0), (k - 1, 0))
        return F.conv1d(z, self.fuse_conv.weight, groups=x.shape[1]).squeeze(0).t()

    def _full_batched(self, x):
        B, T, _ = x.shape
        if x.device.type == 'cuda':
            # Fused q/k/v projection (CUDA only — see _pre): each output
            # column is an independent dot product over the input dim, so the
            # sliced results are bit-identical to the three separate
            # self.W_*(x) calls (the same row/column independence the
            # pre-projection hoist relies on, design notes §6.1).
            _nhq = self.nh * self.hd
            if torch.is_grad_enabled():
                _wqkv = torch.cat([self.W_q.weight, self.W_k.weight, self.W_v.weight], 0)
            else:
                # Key on versions, dtype and device of the source weights.
                _key = (self.W_q.weight._version, self.W_k.weight._version, self.W_v.weight._version,
                        self.W_q.weight.dtype, self.W_k.weight.dtype, self.W_v.weight.dtype,
                        self.W_q.weight.device)
                _hit = self._wcat_cache.get('qkv')
                if _hit is None or _hit[0] != _key:
                    with torch.inference_mode(False):
                        _hit = (_key, torch.cat([self.W_q.weight, self.W_k.weight, self.W_v.weight], 0))
                    self._wcat_cache['qkv'] = _hit
                _wqkv = _hit[1]
            _qkv = F.linear(x, _wqkv)
            q = _qkv[..., :_nhq].reshape(B, T, self.nh, self.hd)
            k = _qkv[..., _nhq:2 * _nhq].reshape(B, T, self.nh, self.hd)
            v = _qkv[..., 2 * _nhq:].reshape(B, T, self.nh, self.hd)
            del _qkv, _wqkv
        else:
            q = self.W_q(x).view(B, T, self.nh, self.hd)
            k = self.W_k(x).view(B, T, self.nh, self.hd)
            v = self.W_v(x).view(B, T, self.nh, self.hd)
        if self.cfg.full_cosine:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            scale = 1.0
        else:
            scale = 1.0 / math.sqrt(self.hd)
        if self.sink is None:
            sdpa_kw = dict(scale=scale)
            if self.cfg.window > 0:
                sdpa_kw['attn_mask'] = causal_window_mask_f(T, self.cfg.window, x.device, q.dtype)
            else:
                sdpa_kw['is_causal'] = True
            out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), **sdpa_kw).transpose(1, 2)
            return self.W_o(out.reshape(B, T, self.nh * self.hd))
        if self.cfg.window > 0:
            mask = causal_window_mask(T, self.cfg.window, x.device)
        else:
            mask = causal_mask(T, x.device)
        sink_view = torch.nan_to_num(self.sink).view(1, self.nh, 1, 1) if self.sink is not None else None
        out = torch.empty(B, T, self.nh, self.hd, device=x.device, dtype=q.dtype)
        row_chunk = max(1, min(T, 1024))
        _budget = _attn_transient_budget(x.device)
        if _budget is not None:
            _per_row = max(1, 3 * B * self.nh * T) * q.element_size()
            row_chunk = max(1, min(row_chunk, int(_budget) // _per_row))
        # 训练时每个行块包一层非重入 checkpoint：反向按块重算同一份值（无
        # RNG、row_chunk 经 _ATB_EPOCH 在运行内不变、mask 是缓存的常量张量
        # ——块边界与中间值逐位相同，梯度与不分块形式逐位一致，同 ce_loss
        # 分块重算的论证）。
        _ck = torch.is_grad_enabled() and q.requires_grad
        for s in range(0, T, row_chunk):
            e = min(s + row_chunk, T)
            if _ck:
                out[:, s:e] = torch.utils.checkpoint.checkpoint(self._sink_attn_chunk, q[:, s:e], k, v, mask[s:e], sink_view, scale, use_reentrant=False)
            else:
                out[:, s:e] = self._sink_attn_chunk(q[:, s:e], k, v, mask[s:e], sink_view, scale)
        return self.W_o(out.reshape(B, T, self.nh * self.hd))

    def _sink_attn_chunk(self, q_c, k, v, mask_c, sink_view, scale):
        logits = torch.einsum('bnhd,bmhd->bhnm', q_c, k)
        if scale != 1.0:
            # same IEEE identity as gathered_attention (x·1.0 and its backward
            # grad·1.0 are bit-identical to no-op).
            logits.mul_(scale)
        logits.add_(mask_c)
        if sink_view is not None:
            lse = torch.logaddexp(sink_view, torch.logsumexp(logits, dim=-1, keepdim=True))
            # logsumexp 的反向保留其输入 logits，sub 必须出位（保留值与新
            # 张量上的 exp_ 原地——exp 反向保存的输出此后再无人改写）；原地
            # sub_ 会使 logsumexp 保存的张量版本失效，反向直接报错。
            attn = logits.sub(lse).exp_()
            del logits, lse
        else:
            attn = torch.softmax(logits, -1)
            del logits
        return torch.einsum('bhnm,bmhd->bnhd', attn, v)

    def _split(self, kvh):
        nh, hd = (self.nh, self.hd)
        k = kvh[..., :nh * hd].view(*kvh.shape[:-1], nh, hd)
        v = kvh[..., nh * hd:].view(*kvh.shape[:-1], nh, hd)
        return (k, v)

    def _pre(self, x, fuse_x=None):
        cfg = self.cfg
        pre = {}
        if cfg.kind == 'full':
            return None
        _want_idx = cfg.kind != 'hca' and cfg.indexer_mode != 'random'
        if x.device.type != 'cuda':
            pre['q'] = self.W_q(x)
            pre['Ca'] = x @ self.W_aKV
            pre['Cb'] = x @ self.W_bKV if self.W_bKV is not None else None
            pre['Za'] = x @ self.W_aZ if self.W_aZ is not None else None
            pre['Zb'] = x @ self.W_bZ if self.W_bZ is not None else None
            if cfg.kind != 'hca':
                if cfg.indexer_mode == 'random':
                    pre['qI'] = pre['w_idx'] = None
                else:
                    pre['qI'] = x @ self.W_DQ
                    pre['w_idx'] = F.linear(x, self.W_w.weight)
            if self.fuse_conv is not None:
                k = cfg.fuse_kernel
                fx = fuse_x.detach()
                z = F.pad(fx.transpose(1, 2), (k - 1, 0))
                pre['fused'] = F.conv1d(z, self.fuse_conv.weight, groups=fx.shape[-1]).transpose(1, 2)
            return pre
        # Batched projection (CUDA): all x @ W terms run over the concatenated
        # weight columns (row-independent output columns, so the per-slice
        # values — and the per-column grads — are bit-identical to the
        # per-projection form, design notes §6.1).
        _mats = [self.W_q.weight.t(), self.W_aKV]
        _keys = ['q', 'Ca']
        # weights are transposed below for a single F.linear (addmm) call, so
        # the kernel matches the per-projection form exactly
        if self.W_bKV is not None:
            _mats.append(self.W_bKV)
            _keys.append('Cb')
        if self.W_aZ is not None:
            _mats.append(self.W_aZ)
            _keys.append('Za')
        if self.W_bZ is not None:
            _mats.append(self.W_bZ)
            _keys.append('Zb')
        if _want_idx:
            _mats.append(self.W_DQ)
            _keys.append('qI')
        if torch.is_grad_enabled():
            _wcat = torch.cat([_m.t() for _m in _mats], dim=0)
        else:
            # Key on versions, dtype and device of the source weights.
            _key = tuple(_m._version for _m in _mats) + tuple(_m.dtype for _m in _mats) + (_mats[0].device,)
            _hit = self._wcat_cache.get('pre')
            if _hit is None or _hit[0] != _key:
                with torch.inference_mode(False):
                    _hit = (_key, torch.cat([_m.t() for _m in _mats], dim=0))
                self._wcat_cache['pre'] = _hit
            _wcat = _hit[1]
        _fused = F.linear(x, _wcat)
        _off = 0
        for _k, _m in zip(_keys, _mats):
            _w = _m.shape[1]
            pre[_k] = _fused[:, _off:_off + _w]
            _off += _w
        pre.setdefault('Cb', None)
        pre.setdefault('Za', None)
        pre.setdefault('Zb', None)
        if cfg.kind != 'hca':
            # w_idx stays on its own narrow-GEMM kernel (n_index_heads is 4:
            # folding a 4-column block into the wide fused GEMM changes the
            # kernel's accumulation tiling and hence the low-order bits, so it
            # keeps the exact per-projection form).
            if cfg.indexer_mode == 'random':
                pre['qI'] = pre['w_idx'] = None
            else:
                pre['w_idx'] = F.linear(x, self.W_w.weight)
        if self.fuse_conv is not None:
            k = cfg.fuse_kernel
            fx = fuse_x.detach()
            z = F.pad(fx.transpose(1, 2), (k - 1, 0))
            pre['fused'] = F.conv1d(z, self.fuse_conv.weight, groups=fx.shape[-1]).transpose(1, 2)
        return pre

    def _gate_signal(self, x, pre=None):
        # GPU-side half of the cosine_learnable gate, computed by the same ops as
        # the inline path in _single.  forward() calls this for every row of the
        # batch first, then moves the stacked hard decisions to the CPU; every
        # value downstream (the gate tensor itself and the per-row cut lists) is
        # identical to the per-row inline form.
        fused = pre['fused'] if pre is not None and 'fused' in pre else self._fuse(x.detach())
        sim = cosine_similarity_consecutive(fused)
        if sim.numel() == 0:
            return None
        cfg = self.cfg
        prefix_mean = torch.cumsum(sim, 0) / _range_cache(1, sim.numel() + 1, x.device)
        tau = prefix_mean + self.delta_logit
        gate = torch.sigmoid((tau - sim) / max(cfg.temperature, 0.001))
        return (gate, gate.detach() > _HALF)

    def _gate_signals(self, x, pre_all=None):
        # Batched form of the per-row `_gate_signal` calls.  Every op except the
        # prefix cumsum is pointwise or a reduction over the last dim of each
        # vector/pair, so its per-row values are independent of how many rows
        # share the launch — bit-identical to the per-row form (the same
        # row-independence the gate-decision hoist already relies on, design
        # notes §6.1).  The prefix cumsum still runs per row on the same
        # contiguous [T-1] values with the same 1-D kernel call, so its output
        # is bit-identical to the per-row form as well.  Returns (rows,
        # gate_bool_stacked): rows[b] is None or (gate_row, bool_row) exactly
        # as `_gate_signal` would have returned for row b.
        B, T, _d = x.shape
        fused = pre_all.get('fused') if pre_all is not None else None
        if fused is None:
            if self.fuse_conv is not None:
                # safe fallback identical to the per-row loop (each row fuses
                # itself, exactly like _gate_signal's inline form)
                return ([self._gate_signal(x[b], None) for b in range(B)], None)
            fused = x.detach()
        h = F.normalize(fused, dim=-1, eps=1e-08)
        sim = (h[:, 1:] * h[:, :-1]).sum(dim=-1)
        if sim.shape[-1] == 0:
            return ([None] * B, None)
        cfg = self.cfg
        # rowwise batched cumsum — same per-row values as the unbind/stack loop
        prefix_mean = torch.cumsum(sim, dim=-1)
        prefix_mean.div_(_range_cache(1, sim.shape[-1] + 1, x.device))
        tau = prefix_mean + self.delta_logit
        gate = torch.sigmoid_((tau - sim).div_(max(cfg.temperature, 0.001)))
        gate_b = gate.detach() > _HALF
        return ([(gate[b], gate_b[b]) for b in range(B)], gate_b)

    def _cut_blocks_batched(self, x):
        # Batched hoist of the per-row cosine cut computation in the
        # cosine_abs / cosine_adaptive / dynamic branches of `_single` (design
        # notes §6.1): the batched normalize + last-dim reduce is the same
        # row-independent form _gate_signals already relies on.  Every integer
        # reaching _single is identical to the per-row blocks_from_cosine form
        # (same cuts, same segmenter).
        cfg = self.cfg
        B, T, _d = x.shape
        if T <= 1:
            return None
        _h = F.normalize(x.detach(), dim=-1, eps=1e-08)
        sim = (_h[:, 1:] * _h[:, :-1]).sum(dim=-1)
        if cfg.chunking == 'cosine_adaptive' or cfg.adaptive:
            tau = causal_adaptive_threshold(sim, cfg.target_block_tokens)
        else:
            tau = cfg.cos_threshold
        cuts = (sim < tau).cpu().numpy()
        return _hoist_blocks_from_cuts(T, cuts, cfg.min_block, cfg.max_block, x.device, want_honoured=False, hon_dtype=torch.float32)

    def _single(self, x, pre=None, gate_sig=None, pre_blocks=None):
        cfg = self.cfg
        T = x.shape[0]
        assert cfg.kind in ('csa', 'hca'), 'dense layers go through _full_batched'
        scale = 1.0
        q = (pre['q'] if pre is not None else self.W_q(x)).view(T, self.nh, self.hd)
        q = F.normalize(q, dim=-1)
        Ca_raw = pre['Ca'] if pre is not None else x @ self.W_aKV
        if pre is not None:
            Cb_raw = pre['Cb']
        else:
            Cb_raw = x @ self.W_bKV if self.W_bKV is not None else None
        gate_mean = None
        if cfg.chunking == 'cosine_learnable':
            if gate_sig is not None:
                gate, gate_list = gate_sig
            else:
                fused = pre['fused'] if pre is not None and 'fused' in pre else self._fuse(x.detach())
                sim = cosine_similarity_consecutive(fused)
                if sim.numel() == 0:
                    tau = torch.tensor(0.0, device=x.device)
                    gate = torch.tensor([], device=x.device)
                else:
                    prefix_mean = torch.cumsum(sim, 0) / _range_cache(1, sim.numel() + 1, x.device)
                    tau = prefix_mean + self.delta_logit
                    gate = torch.sigmoid((tau - sim) / max(cfg.temperature, 0.001))
                gate_bool = None
                gate_list = None
                if gate.numel():
                    gate_bool = gate.detach() > _HALF
                    # .numpy() is a zero-copy view of the CPU tensor; the
                    # segmenter consumes the same booleans as the .tolist()
                    # form (design notes §6.2).
                    gate_list = gate_bool.cpu().numpy()
            if pre_blocks is not None:
                bid, nblk, _hon_dev = pre_blocks
            else:
                with torch.no_grad():
                    if gate_list is None:
                        gate_list = (gate.detach() > _HALF).cpu().numpy()
                    bid, nblk, _honoured = blocks_from_cuts(T, gate_list, cfg.min_block, cfg.max_block, x.device, return_count=True, return_honoured=True)
                _hon_dev = None
                if gate.numel() and self.need_reg:
                    _keep = [0.0] * len(gate_list)
                    for _slot in _honoured:
                        if 0 <= _slot < len(_keep):
                            _keep[_slot] = 1.0
                    _hon_dev = _h2d_async(_keep, gate.dtype, gate.device)
            if gate.numel() and self.need_reg:
                if _hon_dev.dtype != gate.dtype:
                    _hon_dev = _hon_dev.to(gate.dtype)
                _soft_honoured = gate * _hon_dev
                _shs = _soft_honoured.sum()
                gate_mean = (float(nblk - 1) + _shs - _shs.detach()) / T
            else:
                gate_mean = torch.zeros((), device=x.device) if self.need_reg else None
        elif cfg.chunking in ('cosine_abs', 'cosine_adaptive') or (cfg.dynamic and cfg.kind != 'hca'):
            if pre_blocks is not None:
                # geometry pre-computed once for the whole batch by
                # _cut_blocks_batched (forward hoist); values identical to the
                # per-row form below (design notes §6.1).
                bid, nblk, _hon_dev = pre_blocks
            else:
                sim = cosine_similarity_consecutive(x.detach())
                tau = causal_adaptive_threshold(sim, cfg.target_block_tokens) if cfg.chunking == 'cosine_adaptive' or cfg.adaptive else cfg.cos_threshold
                bid, nblk = blocks_from_cosine(x, tau, cfg.min_block, cfg.max_block, sim=sim, return_count=True)
        else:
            bid = blocks_fixed(T, cfg.block_size, x.device)
            nblk = -(-T // cfg.block_size)
        if pre is not None:
            Za, Zb = (pre['Za'], pre['Zb'])
        else:
            Za = x @ self.W_aZ
            Zb = x @ self.W_bZ if self.W_bZ is not None else None
        if cfg.kind == 'hca':
            comp_kv, last_tok, Bn = _pool_trainable(pool_blocks_single, Ca_raw, Za, self.B_pos_a, bid, n_blocks=nblk, monotonic=True)
        else:
            comp_kv, last_tok, Bn = _pool_trainable(pool_variable_blocks, Ca_raw, Cb_raw, Za, Zb, bid, self.B_pos_a, self.B_pos_b, cfg.overlap, n_blocks=nblk, monotonic=True)
        index_kv = comp_kv
        attn_kv = torch.zeros_like(comp_kv) if cfg.content_mode == 'zero' else comp_kv
        if self._stats is not None:
            # scatter_add of exact integer ones is order-independent (integer
            # addition commutes, no rounding), so the counts are the same
            # exact integers bincount returns.
            _bc = torch.zeros(int(nblk), dtype=torch.int64, device=bid.device)
            _bc.scatter_add_(0, bid if bid.dtype == torch.int64 else bid.long(), torch.ones_like(bid, dtype=torch.int64))
            self._stats.append(_bc)
            if self._cuts is not None:
                _cpu = getattr(bid, '_pool_cpu', None)
                if _cpu is not None and sum(_cpu[0]) == T:
                    # host-side integer equivalence (design notes §6.2): cut
                    # positions are the prefix sums of the same counts table
                    # that generated bid — the identical integers to the
                    # nonzero(diff)+1 form.  Consumers accept numpy rows
                    # directly (compression_report's _tcut_ix split and
                    # boundary_alignment's np.asarray).
                    self._cuts.append(np.cumsum(np.asarray(_cpu[0], dtype=np.int64))[:-1])
                else:
                    cuts = (bid[1:] != bid[:-1]).nonzero(as_tuple=True)[0] + 1
                    self._cuts.append(cuts)
        core_kv = F.normalize(attn_kv, dim=-1)
        kv_local = F.normalize(Ca_raw, dim=-1)
        if x.device.type == 'cuda':
            # Fused K/V head (CUDA only — see _pre): every output row is a
            # function of its own input row only, so the split halves are
            # bit-identical to the two self.W_kvhead(...) calls (row
            # independence, design notes §6.1).
            _Bn = int(core_kv.shape[0])
            _kvh = self.W_kvhead(torch.cat([core_kv, kv_local], 0))
            k_blk, v_blk = self._split(_kvh[:_Bn])
            k_sw, v_sw = self._split(_kvh[_Bn:])
            del _kvh
        else:
            k_blk, v_blk = self._split(self.W_kvhead(core_kv))
            k_sw, v_sw = self._split(self.W_kvhead(kv_local))
        k_blk = F.normalize(k_blk, dim=-1)
        k_sw = F.normalize(k_sw, dim=-1)
        if cfg.kind == 'hca':
            # cached int32 expansion — the same exact integers as the int64
            # arange expand narrowed to int32 (index transport, §6.1).  Every
            # row of the arange expansion holds distinct block ids, so the
            # first-occurrence dedup inside gathered_attention is provably the
            # all-True mask and is skipped (identical keep mask).
            topk_idx = _arange_exp_cache(T, Bn, x.device)
            out = _gathered_attention_trainable(q, k_blk, v_blk, topk_idx, last_tok, k_sw, v_sw, cfg.sliding_window, scale, sink_logits=self.sink if cfg.use_sink else None, topk_distinct=True)
        else:
            topk = cfg.index_topk
            _sel_valid_box = []
            # 无梯度（评估）时 gathered_attention 的 soft 直通分支整体是死
            # 的（_soft_live=False 见其注释），索引器据此不物化 soft 尾部，
            # 并以 soft_form=True 固定走软分支的硬部分解——两者前向 bit 与
            # 传入 soft 的形式完全一致。有梯度时条件为 False，走传入 soft
            # 的完整图。
            _need_soft = torch.is_grad_enabled()
            _, topk_idx, soft = lightning_indexer(x, index_kv, last_tok, self.W_DQ, self.W_DK, self.W_w.weight, cfg.n_index_heads, topk, return_mask=False, random_select=cfg.indexer_mode == 'random', pre_qI=pre['qI'] if pre is not None else None, pre_w=pre['w_idx'] if pre is not None else None, out_valid=_sel_valid_box, causal_ends_host=_pool_ends_host(bid), need_soft=_need_soft)
            # _indexer_selection 的 top-k 是无放回选择：有效槽位的块 id 逐行互不
            # 相同（order 是列排列、kept 落在互异的 cumsum 秩列上），pad 槽位的
            # 重复块 0 已被 sel_valid 置无效。故 gathered_attention 的
            # first_occurrence 去重掩码恒为全真，直接声明 topk_distinct 跳过它
            # （keep 掩码与逐位计算结果一致，下游取值不变）。
            out = _gathered_attention_trainable(q, k_blk, v_blk, topk_idx, last_tok, k_sw, v_sw, cfg.sliding_window, scale, soft=soft, sink_logits=self.sink if cfg.use_sink else None, sel_valid=_sel_valid_box[0], topk_distinct=True, soft_form=True)
        return (self.W_o(out.reshape(T, self.nh * self.hd)), gate_mean)

    def forward(self, x):
        if self.cfg.kind == 'full':
            self.last_gate_mean = None
            return self._full_batched(x)
        B, T, d = x.shape
        flat = x.reshape(B * T, d)
        pre_flat = self._pre(flat, fuse_x=x)
        pre_all = None
        if pre_flat is not None:
            pre_all = {}
            for kk, v in pre_flat.items():
                if v is None:
                    pre_all[kk] = None
                elif kk == 'fused':
                    pre_all[kk] = v
                else:
                    pre_all[kk] = v.reshape(B, T, *v.shape[1:])
        # unbind hands out the same per-row views as v[b]; the views feeding
        # _single are identical.
        if pre_all is None:
            pres = [None] * B
        else:
            _cols = {kk: (None if v is None else v.unbind(0)) for kk, v in pre_all.items()}
            pres = [{kk: (None if vv is None else vv[b]) for kk, vv in _cols.items()} for b in range(B)]
        gate_sigs = None
        pre_blocks = None
        # The batched gate-signal / block-geometry hoist below is value-identical
        # to the per-row inline form for ANY batch size (design notes §6.1), so
        # it also serves B == 1 (e.g. the seq-2048 topk sweep).
        if self.cfg.chunking == 'cosine_learnable':
            _raw, _gbool = self._gate_signals(x, pre_all)
            if all((_r is not None for _r in _raw)):
                # .numpy() is a zero-copy view of the stacked CPU tensor (the
                # base tensor stays alive through the views); the segmenter
                # consumes the same booleans.
                _gl = _gbool.cpu().numpy()
                gate_sigs = [(_r[0], _g) for _r, _g in zip(_raw, _gl)]
                # Batched block geometry for the whole batch; every value is
                # identical to the per-row form (see _hoist_blocks_from_cuts).
                pre_blocks = _hoist_blocks_from_cuts(T, _gl, self.cfg.min_block, self.cfg.max_block, x.device, want_honoured=self.need_reg, hon_dtype=_raw[0][0].dtype)
        elif self.cfg.chunking in ('cosine_abs', 'cosine_adaptive') or (self.cfg.dynamic and self.cfg.kind != 'hca'):
            # same batch-level geometry hoist for the fixed/adaptive cosine
            # paths (the cut booleans and the resulting block ids are identical
            # to the per-row form).
            pre_blocks = self._cut_blocks_batched(x)
        if _CK is not None and self.cfg.kind in ('csa', 'hca') and _CK.kernel_enabled():
            # 内核路径（CSA_KERNEL；独立变体，与 eager 统计对照，不逐位承诺）。
            # 不支持的臂在这里硬失败，绝不静默回退——否则同一面板会混入两种
            # 实现的记录。
            if self.cfg.fuse_mode != 'none' or self.cfg.indexer_mode != 'learned':
                raise RuntimeError('CSA_KERNEL 只覆盖 learned-indexer / 非 fuse 的 csa·hca 臂；'
                                   'fuse_mode 或 indexer_mode 不同的臂请 unset CSA_KERNEL 后走 eager')
            return _CK.sparse_layer_forward(self, x, pre_all, gate_sigs, pre_blocks)
        if pre_blocks is not None:
            _prefetch_pool_geometry(self, pre_blocks, T, x.device)
        # Preallocate the batch output and write each row's attention output in
        # place: the stacked buffer holds exactly the tensor torch.stack would
        # have built (a per-row copy into the same layout), and the CopySlices
        # backward hands each row the identical grad slice — value-identical to
        # the stack form (transport only, design notes §6.1).
        outs = torch.empty(B, T, d, device=x.device, dtype=x.dtype)
        gates = []
        for b in range(B):
            pre = pres[b]
            o, g = self._single(x[b], pre=pre, gate_sig=None if gate_sigs is None else gate_sigs[b], pre_blocks=None if pre_blocks is None else pre_blocks[b])
            outs[b] = o
            if g is not None:
                gates.append(g)
        if not gates:
            self.last_gate_mean = None
        else:
            gm_tot = gates[0]
            for _g in gates[1:]:
                gm_tot = gm_tot + _g
            self.last_gate_mean = gm_tot / len(gates)
        return outs

class RMSNorm(nn.Module):

    def __init__(self, d, eps=1e-06):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, self.g.shape, weight=self.g, eps=self.eps)

class MLP(nn.Module):

    def __init__(self, d, hidden, init_gen=None):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d, bias=False)
        # The shared-panel design expects identical MLP weights across variants;
        # drawing them from `init_gen` keeps that true no matter how many RNG
        # draws the attention branch consumed beforehand.
        if init_gen is not None:
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5), generator=init_gen)
                nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5), generator=init_gen)

    def forward(self, x):
        if _CKPT['on'] in ('mlp', 'block') and _CKPT_NEST[0] == 0 and torch.is_grad_enabled() and x.requires_grad:
            # 非重入 checkpoint（模块顶 CSA_CKPT 说明）：重算同一纯函数，
            # 前向值与梯度逐位一致。_CKPT_NEST > 0 时外层块已被 checkpoint
            # 覆盖，内层不再包裹。
            return torch.utils.checkpoint.checkpoint(self._fwd, x, use_reentrant=False)
        return self._fwd(x)

    def _fwd(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class Block(nn.Module):

    def __init__(self, d, n_heads, d_head, cfg: AttnCfg, mlp_ratio=4, mlp_gen=None):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = HybridAttention(d, n_heads, d_head, cfg)
        self.n2 = RMSNorm(d)
        self.mlp = MLP(d, int(round(d * mlp_ratio)), init_gen=mlp_gen)

    def forward(self, x):
        if _CKPT['on'] == 'block' and torch.is_grad_enabled() and x.requires_grad:
            if self.attn.cfg.chunking != 'cosine_learnable':
                return torch.utils.checkpoint.checkpoint(self._fwd_ckpt, x, use_reentrant=False)
            if 'cosine' not in _CKPT_WARNED:
                _CKPT_WARNED.add('cosine')
                print('[CSA_CKPT] block 级重算不覆盖 cosine_learnable 臂（gate_mean 经属性逃逸出块，初向前向的 no_grad 会剥掉它的图——v9 _V9_CKPT 的同一限制）；这些臂回退到 mlp 级，其余臂整块重算')
        return self._fwd(x)

    def _fwd(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x

    def _fwd_ckpt(self, x):
        # checkpoint 的初向前向与反向重算都经这里：置位期间内层 MLP 不再
        # 重复包裹（外层已不保存它的激活）。
        _CKPT_NEST[0] += 1
        try:
            return self._fwd(x)
        finally:
            _CKPT_NEST[0] -= 1

class SmallGPT(nn.Module):

    def __init__(self, vocab, d, n_layers, n_heads, d_head, max_seq, layer_cfgs, mlp_ratio=4):
        super().__init__()
        _rng_state = torch.get_rng_state()
        shared = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        g = torch.Generator().manual_seed(shared)
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_seq, d)
        self.head = nn.Linear(d, vocab, bias=False)
        with torch.no_grad():
            self.tok.weight.normal_(0, 0.02, generator=g)
            self.pos.weight.normal_(0, 0.02, generator=g)
            self.head.weight.normal_(0, 0.02, generator=g)
        self.blocks = nn.ModuleList()
        for i, c in enumerate(layer_cfgs):
            torch.manual_seed(shared + 104729 * (i + 1))
            mlp_g = torch.Generator().manual_seed(shared + 1299709 * (i + 1))
            self.blocks.append(self._make_block(d, n_heads, d_head, c, mlp_ratio, mlp_g))
        # Per-block init temporarily reseeds the GLOBAL torch RNG above; restore the
        # caller's state so model construction never perturbs data shuffling / dropout
        # / later model construction (which would otherwise couple the data order to
        # n_layers and break paired-experiment isolation).
        torch.set_rng_state(_rng_state)
        self.norm = RMSNorm(d)
        self.max_seq = max_seq

    def _make_block(self, d, n_heads, d_head, cfg, mlp_ratio, mlp_gen=None):
        return Block(d, n_heads, d_head, cfg, mlp_ratio, mlp_gen=mlp_gen)

    def forward(self, ids, logits_tail=None):
        x = self.trunk(ids)
        if logits_tail:
            x = x[:, -logits_tail:]
        return self.head(x)

    def trunk(self, ids):
        # Full forward minus the LM head; sets self.comp_reg exactly as forward
        # did (the reg terms are accumulated inside the single block loop — same
        # addition order as the previous two-pass form).
        T = ids.shape[1]
        x = self.tok(ids) + self.pos.weight[:T]
        reg = None
        for blk in self.blocks:
            x = blk(x)
            gm = blk.attn.last_gate_mean
            if gm is not None:
                Lb = max(blk.attn.cfg.target_block_tokens, 1)
                mult = getattr(blk.attn.cfg, 'comp_lambda_mult', 1.0)
                if isinstance(gm, (list, tuple)):
                    tot = gm[0]
                    for _g in gm[1:]:
                        tot = tot + _g
                    gm = tot / len(gm)
                term = mult * (gm * Lb - 1.0) ** 2
                reg = term if reg is None else reg + term
        if reg is None:
            reg = torch.zeros((), device=ids.device)
        self.comp_reg = reg
        return self.norm(x)

    def _ce_head_chunk(self, h_chunk, tgt_chunk, vocab):
        lg = self.head(h_chunk)
        return F.cross_entropy(lg.reshape(-1, vocab), tgt_chunk.reshape(-1), reduction='sum')

    def ce_loss(self, ids, targets, vocab, chunk=64):
        # LM-head cross-entropy computed in time chunks: the chunk sums add to
        # the same total as the full-batch form (up to fp reduction order), so
        # the scalar and its gradients are unchanged.
        # 训练时每个 chunk 包一层非重入 checkpoint：反向按 chunk 重算同一份
        # logits（同一 GEMM、无 RNG，值逐位相同）；chunk 反传的执行次序由
        # 求和链的拓扑结构决定，两种形式相同，因此梯度逐位一致。
        h = self.trunk(ids)
        n_tok = targets.numel()
        ce = None
        _ck = torch.is_grad_enabled() and h.requires_grad
        for s in range(0, ids.shape[1], chunk):
            if _ck:
                c = torch.utils.checkpoint.checkpoint(self._ce_head_chunk, h[:, s:s + chunk], targets[:, s:s + chunk], vocab, use_reentrant=False)
            else:
                c = self._ce_head_chunk(h[:, s:s + chunk], targets[:, s:s + chunk], vocab)
            ce = c if ce is None else ce + c
        return ce / n_tok
OVERLAP = 8
_WARM_KEY_RE = re.compile('.*::w(\\d+)::seed\\d+$')

def make_layer_cfgs(n_layers, variant):
    cfgs = []
    for i in range(n_layers):
        if variant == 'full':
            cfgs.append(AttnCfg(kind='full'))
        elif variant == 'full_cos':
            cfgs.append(AttnCfg(kind='full', full_cosine=True))
        elif variant == 'full_matched':
            cfgs.append(AttnCfg(kind='full'))
        elif variant == 'full_sw128':
            cfgs.append(AttnCfg(kind='full', window=128))
        elif variant == 'full_sw128_matched':
            cfgs.append(AttnCfg(kind='full', window=128))
        elif variant == 'csa_fixed':
            cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32))
        elif variant == 'csa_dynamic':
            cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1))
        elif variant == 'hybrid_fixed':
            if i % 2 == 0:
                cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32))
            else:
                cfgs.append(AttnCfg(kind='hca', dynamic=False, block_size=64))
        elif variant == 'hybrid_dynamic':
            if i % 2 == 0:
                cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1))
            else:
                cfgs.append(AttnCfg(kind='hca', dynamic=True, chunking='cosine_learnable', target_block_tokens=64, temperature=0.1, comp_lambda_mult=4.0))
        elif variant == 'full_sink':
            cfgs.append(AttnCfg(kind='full', full_sink=True))
        elif variant == 'hybrid_csa_dyn':
            if i % 2 == 0:
                cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1))
            else:
                cfgs.append(AttnCfg(kind='hca', dynamic=False, block_size=64))
        elif variant == 'hybrid_hca_dyn':
            if i % 2 == 0:
                cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32))
            else:
                cfgs.append(AttnCfg(kind='hca', dynamic=True, chunking='cosine_learnable', target_block_tokens=64, temperature=0.1, comp_lambda_mult=4.0))
        elif variant == 'csa_dyn_fuse':
            cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, fuse_mode='learned'))
        elif variant == 'hybrid_csa_dyn_fuse':
            if i % 2 == 0:
                cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, fuse_mode='learned'))
            else:
                cfgs.append(AttnCfg(kind='hca', dynamic=False, block_size=64))
        elif variant == 'csa_fix_randidx':
            cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32, indexer_mode='random'))
        elif variant == 'csa_fix_zerocont':
            cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32, content_mode='zero'))
        elif variant == 'csa_fix_nosink':
            cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32, use_sink=False))
        elif variant == 'csa_fix_topk8':
            cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=8))
        elif variant == 'csa_fix_topk64':
            cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=64))
        elif variant == 'csa_dyn_randidx':
            cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, indexer_mode='random'))
        elif variant == 'csa_dyn_zerocont':
            cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, content_mode='zero'))
        elif variant == 'csa_dyn_nosink':
            cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, use_sink=False))
        elif variant == 'csa_dyn_nocomp':
            # Loss-term ablation C: identical architecture, optimizer, LR
            # schedule, budget and data as `csa_dynamic`; only the compression
            # regularizer's multiplier is zeroed.  Keeping run_cfg's `cl`
            # unchanged is what lets `pair_reason` pair the two arms while the
            # actual loss term differs — the thing under test.
            cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, comp_lambda_mult=0.0))
        elif variant == 'csa_fixed_fullpov':
            cfgs.append(AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=128, index_topk=32))
        elif variant == 'csa_dynamic_fullpov':
            cfgs.append(AttnCfg(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=128, index_topk=32, temperature=0.1))
        else:
            raise ValueError(variant)
    return cfgs

def build_tokenizer(texts, vocab_size=8192):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    tok = Tokenizer(BPE(unk_token='<unk>'))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=True)
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=['<pad>', '<unk>', '<bos>', '<eos>'])
    tok.train_from_iterator(texts, trainer=trainer)
    return tok
PUNCT_CHARS = set('.,!?;:')

def boundary_token_ids(tok, eos_id):
    ids = set()
    for i in range(tok.get_vocab_size()):
        s = tok.decode([i]).strip()
        if s and all((ch in PUNCT_CHARS for ch in s)):
            ids.add(i)
    ids.add(eos_id)
    return ids

class _LocalTextSplit:

    def __init__(self, texts):
        self._texts = texts

    def __len__(self):
        return len(self._texts)

    def select(self, idx):
        return _LocalTextSplit([self._texts[i] for i in idx])

    def __getitem__(self, key):
        if isinstance(key, slice):
            return {'text': self._texts[key]}
        if key == 'text':
            return self._texts
        raise TypeError(f'unsupported key {key!r}')

def local_wikitext_if_available(repo_cache_dir):
    raw_txt = os.environ.get('WT103_RAW_TXT')
    candidates = []
    if raw_txt:
        candidates.append(raw_txt)
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(repo_cache_dir)), 'wt103_raw'))
    candidates.append('/root/wt103_raw')
    for cand in candidates:
        train_f = os.path.join(cand, 'wiki.train.raw')
        valid_f = os.path.join(cand, 'wiki.valid.raw')
        if os.path.exists(train_f) and os.path.exists(valid_f):
            train = _read_raw_rows(train_f)
            valid = _read_raw_rows(valid_f)
            print(f'[data] local wikitext-103 raw dump at {cand} ({len(train)} train rows, {len(valid)} val rows)')
            return {'train': _LocalTextSplit(train), 'validation': _LocalTextSplit(valid)}
    return None

def _read_raw_rows(path):
    with open(path, encoding='utf-8') as fh:
        return [ln.rstrip('\n') for ln in fh]

_IDS_FP_MEMO = {}
_IDS_FP_MEMO_CAP = 4

def _ids_fp(train_ids):
    # Pure function of the buffer bytes, memoised on (data pointer,
    # layout) with a strong ref to the array (so the keyed pointer can
    # never be recycled while the entry lives).  Same bytes in -> same
    # digest out, always.
    try:
        _key = (train_ids.__array_interface__['data'][0], train_ids.shape, train_ids.strides, train_ids.dtype.str)
    except (AttributeError, TypeError):
        _key = None
    if _key is not None:
        _hit = _IDS_FP_MEMO.get(_key)
        if _hit is not None:
            return _hit[1]
    _h = hashlib.sha256()
    _mv = np.ascontiguousarray(train_ids)
    _cs = 1 << 24
    for _off in range(0, len(_mv), _cs):
        _h.update(_mv[_off:_off + _cs].data)
    _fp = _h.hexdigest()[:16]
    if _key is not None:
        if len(_IDS_FP_MEMO) >= _IDS_FP_MEMO_CAP:
            _IDS_FP_MEMO.pop(next(iter(_IDS_FP_MEMO)))
        _IDS_FP_MEMO[_key] = (train_ids, _fp)
    return _fp

_WT103_MEMO = {}
_WT103_MEMO_CAP = 4

def load_wikitext(seq_len, n_train_tokens, vocab_size=8192, val_seqs=512, cache_dir='./wt103_cache'):
    from datasets import load_dataset
    os.makedirs(cache_dir, exist_ok=True)
    tag = f'v4_sl{seq_len}_cap{n_train_tokens}_v{vocab_size}_vs{val_seqs}'
    tr_path = os.path.join(cache_dir, f'train_ids_{tag}.npy')
    va_path = os.path.join(cache_dir, f'val_batch_{tag}.npy')
    vp_path = os.path.join(cache_dir, f'val_bnd_{tag}.npy')
    me_path = os.path.join(cache_dir, f'meta_{tag}.json')
    if all((os.path.exists(p) for p in (tr_path, va_path, vp_path, me_path))):
        # In-process memo: the served bytes are exactly what a fresh
        # load would return and the returned objects keep their identity,
        # so the identity-keyed resident-corpus cache in batch_iter
        # (_PINNED_HOST_CACHE) behaves exactly as it would without the
        # memo.  The memo key carries every file's size+mtime, so an
        # externally rebuilt cache is re-read (and re-verified) instead
        # of being served stale.  Callers never mutate the served arrays.
        _sig = tuple((os.path.getsize(_p), os.path.getmtime(_p)) for _p in (tr_path, va_path, vp_path, me_path))
        _mkey = (os.path.abspath(cache_dir), tag, _sig)
        _hit = _WT103_MEMO.get(_mkey)
        if _hit is not None:
            print('[data] serving token ids from the in-process memo (same cache files, already verified) ...')
            return _hit
        print('[data] loading cached token ids ...')
        train_ids = np.load(tr_path)
        meta = json.load(open(me_path, encoding='utf-8'))
        if meta.get('src_fp') != _ids_fp(train_ids):
            print('[data] WARNING: the cached train_ids do not match the fingerprint in the meta file — the cache is corrupt or was mixed across builds; REBUILDING it')
        else:
            val_batch = np.load(va_path)
            val_bnd = np.load(vp_path)
            _out = (train_ids, val_batch, meta['vocab'], None, val_bnd)
            if len(_WT103_MEMO) >= _WT103_MEMO_CAP:
                _WT103_MEMO.pop(next(iter(_WT103_MEMO)))
            _WT103_MEMO[_mkey] = _out
            return _out
    ds = local_wikitext_if_available(cache_dir)
    if ds is None:
        last_err = None
        for repo in ('Salesforce/wikitext', 'wikitext'):
            try:
                ds = load_dataset(repo, 'wikitext-103-raw-v1', cache_dir=cache_dir)
                break
            except Exception as e:
                last_err = e
        if ds is None:
            raise last_err if last_err is not None else RuntimeError('load_wikitext: no dataset repository could be loaded')
    if not hasattr(ds, '__getitem__') or 'train' not in ds or 'validation' not in ds:
        raise TypeError(f"load_wikitext: the loaded corpus object does not expose the ['train']/['validation'] splits (got {type(ds).__name__}); refusing to tokenise a corpus we cannot identify")
    fit_rows = min(len(ds['train']), 50000)
    fit_text = ds['train'].select(range(fit_rows))['text']
    tok = build_tokenizer(fit_text, vocab_size)
    vocab = tok.get_vocab_size()
    fit_fp = _text_fp(fit_text)
    del fit_text
    gc.collect()
    eos_id = tok.token_to_id('<eos>')

    def stream_tokenise(split, cap_tokens, label, _ds=ds, _tok=tok, _eos=eos_id):
        # Ids accumulate in bounded numpy chunks; the int64 values produced are
        # unchanged.  Rows go through the
        # boundary / cap-truncation behaviour) are identical to the
        # per-`encode` form.
        chunks = []
        cur = []
        n_have = 0

        def _flush():
            if cur:
                chunks.append(np.array(cur, dtype=np.int64))
                cur.clear()

        row_chunk = 10000
        for start in range(0, len(_ds[split]), row_chunk):
            rows = _ds[split][start:start + row_chunk]['text']
            rows = [r for r in rows if r.strip()]
            for enc in _tok.encode_batch(rows):
                cur.extend(enc.ids)
                cur.append(_eos)
            if len(cur) >= (1 << 22):
                _flush()
            n_have = sum(c.size for c in chunks) + len(cur)
            if n_have >= cap_tokens:
                break
            if start and start % 50000 == 0:
                print(f'    [{label}] {start} rows, {n_have} tokens')
        _flush()
        if not chunks:
            return np.empty(0, dtype=np.int64)
        # a single chunk is handed out directly (a copy of the same values);
        # the concatenation and the [:cap_tokens] truncation are the same
        # integers the single-list form produced.
        ids = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        return ids[:cap_tokens]
    print(f'[data] tokenising train (cap {n_train_tokens} tokens) ...')
    train_ids = stream_tokenise('train', n_train_tokens + seq_len, 'train')
    print(f'[data] train tokens = {len(train_ids)}  vocab = {vocab}')
    if len(train_ids) < n_train_tokens + seq_len:
        print(f'[data] WARNING: the train split ran dry at {len(train_ids)} tokens (< requested {n_train_tokens + seq_len}); runs configured for the advertised token budget will train on fewer tokens than advertised')
    print('[data] tokenising validation ...')
    val_ids = stream_tokenise('validation', val_seqs * seq_len + seq_len, 'val')
    n_val = min(len(val_ids) // seq_len, val_seqs)
    val_batch = val_ids[:n_val * seq_len].reshape(n_val, seq_len)
    print(f'[data] val sequences = {n_val} (seq_len={seq_len})')
    bnd_ids = sorted(boundary_token_ids(tok, eos_id))
    val_bnd = np.isin(val_batch, bnd_ids)
    print(f'[data] boundary tokens: {int(val_bnd.sum())} ({val_bnd.mean() * 100:.1f}% of val tokens)')
    for _p, _a in ((tr_path, train_ids), (va_path, val_batch), (vp_path, val_bnd)):
        _fd, _tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(_p)), prefix=os.path.basename(_p) + '.', suffix='.tmp')
        try:
            with os.fdopen(_fd, 'wb') as _fh:
                np.save(_fh, _a)
            os.replace(_tmp, _p)
        except BaseException:
            try:
                os.unlink(_tmp)
            except OSError:
                pass
            raise
    atomic_write_json(me_path, {'vocab': vocab, 'src_fp': _ids_fp(train_ids), 'fit_fp': fit_fp}, indent=0)
    try:
        # persist the exact tokenizer that produced this cache so the held-out
        # test split can be verified against it by full vocab identity (size
        # alone is not identity: different merges can share a vocab size).
        tok.save(os.path.join(cache_dir, f'tok_{tag}.json'))
    except Exception as _te:
        print(f'[data] NOTE: could not persist the train tokenizer ({type(_te).__name__}: {_te})')
    print('[data] cached token ids to disk (resume-safe)')
    del ds
    gc.collect()
    return (train_ids, val_batch, vocab, tok, val_bnd)

# ---- 独立最终 test split（wiki.test.raw，冻结后一次性评估，见 design_notes §17）
_WT103_TEST_MEMO = {}
_WT103_TEST_MEMO_CAP = 2

def _text_fp(rows):
    """sha256 of a list of text rows (fit-corpus provenance for the test cache)."""
    h = hashlib.sha256()
    for r in rows:
        h.update(r.encode('utf-8', errors='replace'))
        h.update(b'\n')
    return h.hexdigest()[:16]

def _tokenise_row_chunks(row_chunks, cap_tokens, label, tok, eos_id):
    """Standalone twin of load_wikitext's nested stream_tokenise with the exact
    same chunking / eos / cap / flush behaviour (1<<22 chunks, row-chunk 10000,
    [:cap_tokens] truncation) so the served ids obey the same discipline."""
    chunks = []
    cur = []
    n_have = 0

    def _flush():
        if cur:
            chunks.append(np.array(cur, dtype=np.int64))
            cur.clear()
    for rows in row_chunks:
        rows = [r for r in rows if r.strip()]
        for enc in tok.encode_batch(rows):
            cur.extend(enc.ids)
            cur.append(eos_id)
        if len(cur) >= (1 << 22):
            _flush()
        n_have = sum(c.size for c in chunks) + len(cur)
        if n_have >= cap_tokens:
            break
        print(f'    [{label}] {n_have} tokens so far')
    _flush()
    if not chunks:
        return np.empty(0, dtype=np.int64)
    ids = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
    return ids[:cap_tokens]

def load_wikitest(seq_len, n_seqs=256, vocab_size=8192, cache_dir='./wt103_cache'):
    """Load the WikiText-103 **test** split as (n_seqs, seq_len) int64 rows.

    只读 wiki.test.raw（本地 dump 或 HF test split），绝不触碰 train/validation，
    因此可作冻结后的独立最终评估数据（design_notes §17 的「未参与方法选择的
    数据」）。与 load_wikitext 同一套纪律：npy 缓存 + sha256 内容指纹 + 原子写；
    分词器用与主语料完全相同的拟合过程（同一份 train 前 50k 行，确定性
    BPE）重建，并把拟合语料指纹存进 meta 以供审计。缓存命中只校验自身
    指纹；train 语料被更换时应删除整个 wt103_cache/（README「Reproducing」
    一节同款约定），否则 test 缓存不会跟随重建——这是显式记录的边界，
    不是静默的失效路径。"""
    from datasets import load_dataset
    os.makedirs(cache_dir, exist_ok=True)
    tag = f'v1_sl{seq_len}_v{vocab_size}_ns{n_seqs}'
    te_path = os.path.join(cache_dir, f'test_ids_{tag}.npy')
    tb_path = os.path.join(cache_dir, f'test_bnd_{tag}.npy')
    me_path = os.path.join(cache_dir, f'test_meta_{tag}.json')
    tk_path = os.path.join(cache_dir, f'test_tok_{tag}.json')
    if all((os.path.exists(p) for p in (te_path, tb_path, me_path))):
        _sig = tuple((os.path.getsize(_p), os.path.getmtime(_p)) for _p in (te_path, tb_path, me_path))
        _mkey = (os.path.abspath(cache_dir), tag, _sig)
        _hit = _WT103_TEST_MEMO.get(_mkey)
        if _hit is not None:
            print('[data] serving TEST token ids from the in-process memo (same cache files, already verified) ...')
            return _hit
        print('[data] loading cached TEST token ids ...')
        test_ids = np.load(te_path)
        meta = json.load(open(me_path, encoding='utf-8'))
        if meta.get('src_fp') != _ids_fp(test_ids):
            print('[data] WARNING: the cached test_ids do not match the fingerprint in the test meta file — the cache is corrupt or was mixed across builds; REBUILDING it')
        else:
            test_bnd = np.load(tb_path)
            out = (test_ids, meta['vocab'], None, test_bnd)
            if len(_WT103_TEST_MEMO) >= _WT103_TEST_MEMO_CAP:
                _WT103_TEST_MEMO.pop(next(iter(_WT103_TEST_MEMO)))
            _WT103_TEST_MEMO[_mkey] = out
            return out
    # cold build: source the test rows (local dump first, HF otherwise)
    test_rows = None
    ds = None
    raw_txt = os.environ.get('WT103_RAW_TXT')
    _cands = []
    if raw_txt:
        _cands.append(raw_txt)
    _cands.append(os.path.join(os.path.dirname(os.path.abspath(cache_dir)), 'wt103_raw'))
    _cands.append('/root/wt103_raw')
    for cand in _cands:
        _f = os.path.join(cand, 'wiki.test.raw')
        if os.path.exists(_f):
            test_rows = _read_raw_rows(_f)
            print(f'[data] local wikitext-103 TEST split at {_f} ({len(test_rows)} rows)')
            break
    if test_rows is None:
        ds = None
        last_err = None
        for repo in ('Salesforce/wikitext', 'wikitext'):
            try:
                ds = load_dataset(repo, 'wikitext-103-raw-v1', cache_dir=cache_dir)
                break
            except Exception as e:
                last_err = e
        if ds is None:
            raise last_err if last_err is not None else RuntimeError('load_wikitest: no test-split source available (local dump wiki.test.raw or the HF dataset)')
        test_rows = ds['test']['text']
    # tokenizer: identical fit to load_wikitext (first <=50k TRAIN rows, deterministic BPE)
    train_fit_rows = None
    for cand in _cands:
        _f = os.path.join(cand, 'wiki.train.raw')
        if os.path.exists(_f):
            train_fit_rows = _read_raw_rows(_f)[:50000]
            break
    if train_fit_rows is None:
        if ds is None:
            for repo in ('Salesforce/wikitext', 'wikitext'):
                try:
                    ds = load_dataset(repo, 'wikitext-103-raw-v1', cache_dir=cache_dir)
                    break
                except Exception as e:
                    last_err = e
            if ds is None:
                raise last_err if last_err is not None else RuntimeError('load_wikitest: no train source to fit the shared tokenizer')
        train_fit_rows = ds['train'].select(range(min(len(ds['train']), 50000)))['text']
    tok = build_tokenizer(list(train_fit_rows), vocab_size)
    vocab = tok.get_vocab_size()
    eos_id = tok.token_to_id('<eos>')
    # vocabulary consistency with any existing train-side cache: a mismatch means
    # the two corpora builds cannot be paired — refuse rather than serve it.
    # A persisted train tokenizer is compared by FULL vocab identity (id->token
    # mapping), not merely by size: two BPE fits can share a vocab size with
    # different merges, in which case the id spaces are un-pairable.  Legacy
    # caches without a persisted tokenizer fall back to the size check.
    import glob as _glob
    _test_vocab = dict(tok.get_vocab())
    _train_tok_paths = _glob.glob(os.path.join(cache_dir, 'tok_v4_*.json'))
    for _tp in _train_tok_paths:
        try:
            from tokenizers import Tokenizer as _Tok
            _tt = _Tok.from_file(_tp)
        except Exception:
            continue
        if dict(_tt.get_vocab()) != _test_vocab:
            raise RuntimeError(f'load_wikitest: the freshly fitted test tokenizer is not identical to the cached train tokenizer ({_tp}) — same-size vocabularies can still use different merges, so the splits cannot be paired; delete wt103_cache/ and rebuild both')
    for _mp in _glob.glob(os.path.join(cache_dir, 'meta_v4_*.json')):
        try:
            _mt = json.load(open(_mp, encoding='utf-8'))
        except Exception:
            continue
        _mv = _mt.get('vocab')
        if _mv is None:
            continue
        if int(_mv) != int(vocab):
            raise RuntimeError(f'load_wikitest: the freshly fitted tokenizer (vocab={vocab}) disagrees with the cached train corpus (vocab={_mv} in {_mp}) — the splits cannot be paired; delete wt103_cache/ and rebuild both')
    print(f'[data] tokenising TEST split (cap {n_seqs * seq_len + seq_len} tokens) ...')
    _trows = list(test_rows) if not isinstance(test_rows, list) else test_rows
    ids = _tokenise_row_chunks([_trows[s:s + 10000] for s in range(0, len(_trows), 10000)], n_seqs * seq_len + seq_len, 'test', tok, eos_id)
    n_have = min(len(ids) // seq_len, n_seqs)
    if n_have < n_seqs:
        print(f'[data] WARNING: test split yields only {n_have} full rows (< {n_seqs} requested) — the final-test evaluation runs on {n_have} rows')
    test_batch = ids[:n_have * seq_len].reshape(n_have, seq_len)
    print(f'[data] TEST rows = {n_have} (seq_len={seq_len}, vocab={vocab})')
    bnd_ids = sorted(boundary_token_ids(tok, eos_id))
    test_bnd = np.isin(test_batch, bnd_ids)
    for _p, _a in ((te_path, test_batch), (tb_path, test_bnd)):
        _fd, _tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(_p)), prefix=os.path.basename(_p) + '.', suffix='.tmp')
        try:
            with os.fdopen(_fd, 'wb') as _fh:
                np.save(_fh, _a)
            os.replace(_tmp, _p)
        except BaseException:
            try:
                os.unlink(_tmp)
            except OSError:
                pass
            raise
    atomic_write_json(me_path, {'vocab': vocab, 'src_fp': _ids_fp(test_batch), 'fit_fp': _text_fp(train_fit_rows)}, indent=0)
    try:
        tok.save(tk_path)
    except Exception as _e:
        print(f'[data] NOTE: could not persist the test tokenizer ({type(_e).__name__}: {_e}) — a rebuild will re-fit it deterministically')
    print('[data] cached TEST token ids to disk (held-out split; final evaluation only)')
    return (test_batch, vocab, tok, test_bnd)

_FWEB_MEMO = {}
_FWEB_MEMO_CAP = 2

def _fineweb_disk_dtype(vocab):
    if vocab <= 65536:
        return np.uint16
    if vocab < 2 ** 31:
        return np.int32
    raise ValueError(f'vocab {vocab} exceeds the id range representable by the on-disk corpus cache')

def _disk_ids_chunk(buf, disk_dt, vocab):
    # Range check must run on the pre-cast Python ints: once the array is
    # uint16 its max is <= 65535 by construction, so checking it afterwards
    # can never fire (and an out-of-range id would already have wrapped on
    # numpy builds that silently overflow).  `_fineweb_disk_dtype` returns
    # uint16 only for vocab <= 65536, so this is a defensive guard.
    if disk_dt is np.uint16 and buf and max(buf) > 65535:
        raise ValueError(f'token id {max(buf)} exceeds the uint16 cache range (vocab={vocab}) — widen _fineweb_disk_dtype before rebuilding')
    return np.array(buf, dtype=disk_dt)

def _npy_serve_int32(path):
    # 传输层（design_notes §6.1）：磁盘上的 uint16/int32 与 served 的 int32
    # 是同一组精确整数；经 mmap 分块转换（而不是整载入再 .astype）。
    # served 的 int32 字节与直接 astype 形式逐一相同（_ids_fp 的内容指纹、
    # batch_iter 的对象身份键都不变）。
    mm = np.load(path, mmap_mode='r')
    src = np.asarray(mm)
    if src.dtype == np.int32:
        return np.array(src)
    out = np.empty(src.shape, dtype=np.int32)
    for s in range(0, src.size, 1 << 24):
        e = min(s + (1 << 24), src.size)
        out[s:e] = src[s:e]
    return out

def _fineweb_local_rows(env_value):
    """Offline FineWeb-Edu stand-in: FWEDU_LOCAL_JSONL lists files or
    directories (os.pathsep-separated; directories expand to their sorted
    *.jsonl / *.json / *.txt).  Every non-empty line is a document — either a
    JSON object with a "text" field (the FineWeb-Edu row shape) or plain text.
    Pure availability path for air-gapped kernel-dev machines; the corpus a
    panel trains on is still bound by the content fingerprint, so a local
    stand-in can never be confused with a genuine FineWeb-Edu cache."""
    import glob
    files = []
    for cand in str(env_value).split(os.pathsep):
        cand = cand.strip()
        if not cand:
            continue
        if os.path.isdir(cand):
            for pat in ('*.jsonl', '*.json', '*.txt'):
                files.extend(sorted(glob.glob(os.path.join(cand, pat))))
        elif any(ch in cand for ch in '*?['):
            files.extend(sorted(glob.glob(cand)))
        else:
            files.append(cand)
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise RuntimeError(f'FWEDU_LOCAL_JSONL is set ({env_value!r}) but resolves to no readable file — point it at FineWeb-Edu style jsonl/txt files, or unset it to stream from the hub')
    print(f'[data] local FineWeb-Edu stand-in corpus: {len(files)} file(s) ({files[0]}{" ..." if len(files) > 1 else ""})')
    for path in files:
        with open(path, encoding='utf-8') as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except ValueError:
                    yield ln
                    continue
                text = obj.get('text') if isinstance(obj, dict) else (obj if isinstance(obj, str) else None)
                if isinstance(text, str) and text.strip():
                    yield text
                elif not isinstance(obj, dict):
                    yield ln

def _fineweb_rows(source):
    """Ordered, non-empty FineWeb-Edu documents (streaming; deterministic row
    order per dataset revision).  The token cache plus its content fingerprint
    is what actually binds a panel to its corpus, so a hub-side revision change
    rebuilds a different, non-confusable corpus (different src_fp -> different
    run_cfg -> stale records are dropped, never silently reused)."""
    env_value = os.environ.get('FWEDU_LOCAL_JSONL', '')
    if env_value.strip():
        yield from _fineweb_local_rows(env_value)
        return
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceFW/fineweb-edu', name=source, split='train', streaming=True)
    n_docs = 0
    for rec in ds:
        text = rec.get('text') if isinstance(rec, dict) else None
        if text is None or not str(text).strip():
            continue
        n_docs += 1
        if n_docs % 50000 == 0:
            print(f'    [fineweb] {n_docs} docs streamed')
        yield str(text)

def load_fineweb_edu(seq_len, n_train_tokens, vocab_size=32768, val_seqs=512, cache_dir='./fineweb_cache', source='sample-10BT', fit_docs=20000):
    """B-tier corpus (docs/compute_plan.md §5): FineWeb-Edu, self-trained BPE.

    Discipline mirrors load_wikitext exactly: npy cache keyed by the full
    protocol tag, atomic writes, sha256 content fingerprint in the meta file,
    in-process memo keyed on (file size, mtime).  Differences forced by the
    scale band:
      - token ids are cached on disk at 2 bytes/token (uint16, guarded by the
        actual vocab) and served as int32; every consumer widens ids back to
        int64 per batch exactly as the int64 corpora always did, so yielded
        batches are bit-identical to the int64-storage form;
      - the corpus is built in one streaming pass (fit tokenizer on the first
        `fit_docs` documents, then train tokens up to n_train_tokens + seq_len,
        then the validation holdout: val_seqs*seq_len + seq_len tokens drawn
        from documents strictly after the train region — non-overlapping and
        deterministic per protocol tag, but not necessarily token-contiguous:
        the partial final train batch (up to one 4096-doc fetch block) is
        discarded between the two regions);
      - val_batch is int64 (eval paths feed targets straight to cross entropy).
    """
    os.makedirs(cache_dir, exist_ok=True)
    # A local stand-in corpus must never share a cache tag with the hub source:
    # otherwise a cache built from FWEDU_LOCAL_JSONL would be served on a later
    # hub run (the self-fingerprint cannot tell the two corpora apart).  The
    # marker only appears when the local path is active, so existing hub caches
    # keep their exact filenames.
    _local_env = os.environ.get('FWEDU_LOCAL_JSONL', '').strip()
    _local_tag = f'_loc{hashlib.sha1(_local_env.encode("utf-8")).hexdigest()[:8]}' if _local_env else ''
    tag = f'v1_sl{seq_len}_cap{n_train_tokens}_v{vocab_size}_vs{val_seqs}_fw{source}_fd{fit_docs}{_local_tag}'
    tr_path = os.path.join(cache_dir, f'train_ids_{tag}.npy')
    va_path = os.path.join(cache_dir, f'val_batch_{tag}.npy')
    vp_path = os.path.join(cache_dir, f'val_bnd_{tag}.npy')
    me_path = os.path.join(cache_dir, f'meta_{tag}.json')
    if all((os.path.exists(p) for p in (tr_path, va_path, vp_path, me_path))):
        _sig = tuple((os.path.getsize(p), os.path.getmtime(p)) for p in (tr_path, va_path, vp_path, me_path))
        _mkey = (os.path.abspath(cache_dir), tag, _sig)
        _hit = _FWEB_MEMO.get(_mkey)
        if _hit is not None:
            print('[data] serving fineweb token ids from the in-process memo (same cache files, already verified) ...')
            return _hit
        print('[data] loading cached fineweb token ids ...')
        meta = json.load(open(me_path, encoding='utf-8'))
        train_ids = _npy_serve_int32(tr_path)
        if meta.get('src_fp') != _ids_fp(train_ids):
            print('[data] WARNING: the cached fineweb train_ids do not match the fingerprint in the meta file — the cache is corrupt or was mixed across builds; REBUILDING it')
        else:
            val_batch = np.load(va_path).astype(np.int64)
            val_bnd = np.load(vp_path)
            _out = (train_ids, val_batch, int(meta['vocab']), None, val_bnd)
            if len(_FWEB_MEMO) >= _FWEB_MEMO_CAP:
                _FWEB_MEMO.pop(next(iter(_FWEB_MEMO)))
            _FWEB_MEMO[_mkey] = _out
            return _out
    rows = _fineweb_rows(source)
    fit_text = []
    for _i, doc in enumerate(rows):
        if _i >= fit_docs:
            break
        fit_text.append(doc)
    print(f'[data] training BPE tokenizer on {len(fit_text)} fineweb docs (vocab_size={vocab_size}) ...')
    tok = build_tokenizer(fit_text, vocab_size)
    vocab = tok.get_vocab_size()
    del fit_text
    gc.collect()
    eos_id = tok.token_to_id('<eos>')
    disk_dt = _fineweb_disk_dtype(vocab)
    cap_train = int(n_train_tokens) + int(seq_len)
    cap_val = int(val_seqs) * int(seq_len) + int(seq_len)
    raw_dir = os.path.dirname(os.path.abspath(tr_path))

    def _raw_tmp(label):
        fd, tmp = tempfile.mkstemp(dir=raw_dir, prefix=f'.fineweb_{label}_{tag}.', suffix='.raw')
        os.close(fd)
        return tmp

    raw_train = _raw_tmp('train')
    raw_val = _raw_tmp('val')
    n_train_total = 0
    n_val_total = 0
    _marks = 0

    try:
        buf = []
        print(f'[data] tokenising fineweb train (cap {n_train_tokens} tokens, source {source}) ...')
        with open(raw_train, 'wb') as fh:
            stop = False
            while not stop:
                docs = list(itertools.islice(rows, 4096))
                if not docs:
                    break
                # batched encode — the same Rust pipeline per doc as `encode`,
                # so the ids (and hence the raw staging bytes / content
                # fingerprint) are identical to the per-`encode` form; the
                # per-doc flush / cap checks below run in the original order.
                for enc in tok.encode_batch(docs):
                    buf.extend(enc.ids)
                    buf.append(eos_id)
                    if len(buf) >= (1 << 22):
                        _disk_ids_chunk(buf, disk_dt, vocab).tofile(fh)
                        n_train_total += len(buf)
                        buf = []
                        if (n_train_total >> 26) > _marks:
                            _marks = n_train_total >> 26
                            print(f'    [train] {n_train_total} tokens')
                    if n_train_total + len(buf) >= cap_train:
                        stop = True
                        break
            if buf:
                _disk_ids_chunk(buf, disk_dt, vocab).tofile(fh)
                n_train_total += len(buf)
                buf = []
        n_train = min(n_train_total, cap_train)
        print(f'[data] train tokens = {n_train}  vocab = {vocab}')
        if n_train < cap_train:
            print(f'[data] WARNING: the source ran dry at {n_train} train tokens (< requested cap {cap_train}); runs configured for the advertised token budget will train on fewer tokens than advertised')
        print('[data] tokenising fineweb validation holdout ...')
        with open(raw_val, 'wb') as fh:
            stop = False
            while not stop:
                docs = list(itertools.islice(rows, 4096))
                if not docs:
                    break
                for enc in tok.encode_batch(docs):
                    buf.extend(enc.ids)
                    buf.append(eos_id)
                    if len(buf) >= (1 << 22):
                        _disk_ids_chunk(buf, disk_dt, vocab).tofile(fh)
                        n_val_total += len(buf)
                        buf = []
                    if n_val_total + len(buf) >= cap_val:
                        stop = True
                        break
            if buf:
                _disk_ids_chunk(buf, disk_dt, vocab).tofile(fh)
                n_val_total += len(buf)
                buf = []
        n_val_tokens = min(n_val_total, cap_val)
        # 与 load_wikitext 同口径：块缓冲区含 +1 条序列的余量，钳到 val_seqs。
        n_val = min(n_val_tokens // seq_len, int(val_seqs))
        print(f'[data] val sequences = {n_val} (seq_len={seq_len}, contiguous holdout after the train region)')
        if n_val == 0:
            raise RuntimeError('load_fineweb_edu: the corpus ran dry before a single validation sequence was filled — raise n_train_tokens or provide a larger source')
        if n_train <= seq_len + 1:
            raise RuntimeError(f'load_fineweb_edu: only {n_train} train tokens available (need > seq_len+1 = {seq_len + 1}) — the source is too small for the requested cap')

        def _finalise_npy(raw_path, dst_path, count):
            # Write through a unique same-directory temp file, then replace
            # atomically.
            _fd, npy_tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(dst_path)),
                                            prefix=os.path.basename(dst_path) + '.', suffix='.tmp')
            os.close(_fd)
            try:
                mm = np.lib.format.open_memmap(npy_tmp, mode='w+', dtype=disk_dt, shape=(int(count),))
                done = 0
                with open(raw_path, 'rb') as rf:
                    while done < count:
                        take = min(1 << 22, count - done)
                        chunk = np.fromfile(rf, dtype=disk_dt, count=take)
                        if chunk.size < take:
                            raise RuntimeError(f'load_fineweb_edu: raw staging file {raw_path} is shorter than the encoded count — aborting the cache write')
                        mm[done:done + take] = chunk
                        done += take
                mm.flush()
                del mm
                os.replace(npy_tmp, dst_path)
            except BaseException:
                try:
                    os.unlink(npy_tmp)
                except OSError:
                    pass
                raise

        _finalise_npy(raw_train, tr_path, n_train)
        val_disk = np.fromfile(raw_val, dtype=disk_dt, count=n_val * seq_len)
        val_batch = val_disk.astype(np.int64).reshape(n_val, seq_len)
        bnd_ids = sorted(boundary_token_ids(tok, eos_id))
        val_bnd = np.isin(val_batch, bnd_ids)
        print(f'[data] boundary tokens: {int(val_bnd.sum())} ({val_bnd.mean() * 100:.1f}% of val tokens)')
        for _p, _a in ((va_path, val_batch), (vp_path, val_bnd)):
            _fd, _tmp = tempfile.mkstemp(dir=raw_dir, prefix=os.path.basename(_p) + '.', suffix='.tmp')
            try:
                with os.fdopen(_fd, 'wb') as _fh:
                    np.save(_fh, _a)
                os.replace(_tmp, _p)
            except BaseException:
                try:
                    os.unlink(_tmp)
                except OSError:
                    pass
                raise
        train_ids = _npy_serve_int32(tr_path)
        atomic_write_json(me_path, {'vocab': vocab, 'src_fp': _ids_fp(train_ids), 'source': source, 'fit_docs': fit_docs, 'disk_dtype': np.dtype(disk_dt).str, 'seq_len': int(seq_len), 'n_train_tokens': int(n_train_tokens), 'n_train_actual': int(n_train), 'val_seqs': int(val_seqs)}, indent=0)
        print('[data] cached fineweb token ids to disk (resume-safe)')
        _out = (train_ids, val_batch, vocab, tok, val_bnd)
        if len(_FWEB_MEMO) >= _FWEB_MEMO_CAP:
            _FWEB_MEMO.pop(next(iter(_FWEB_MEMO)))
        _FWEB_MEMO[(os.path.abspath(cache_dir), tag, tuple((os.path.getsize(p), os.path.getmtime(p)) for p in (tr_path, va_path, vp_path, me_path)))] = _out
        return _out
    finally:
        for _raw in (raw_train, raw_val):
            try:
                os.unlink(_raw)
            except OSError:
                pass

def load_run_corpus(cfg):
    """Serve the corpus tuple for a run cfg (data-source switch).

    A cfg without a 'dataset' key keeps WikiText-103 with exactly the arguments
    the drivers have always passed — behaviour is bit-identical to the former
    hard-wired load_wikitext call, so no existing panel, fingerprint or record
    is touched.  'dataset': 'fineweb' selects the B-tier corpus
    (docs/compute_plan.md §5) with its own cache directory, tokenizer width,
    source name and fit protocol."""
    name = cfg.get('dataset') or 'wikitext'
    if name == 'wikitext':
        return load_wikitext(cfg['seq_len'], cfg['n_train_tokens'], vocab_size=cfg.get('vocab_size', 8192), val_seqs=cfg.get('val_seqs', 512), cache_dir=cfg.get('cache_dir', './wt103_cache'))
    if name == 'fineweb':
        return load_fineweb_edu(cfg['seq_len'], cfg['n_train_tokens'], vocab_size=cfg.get('vocab_size', B_VOCAB), val_seqs=cfg.get('val_seqs', 512), cache_dir=cfg.get('cache_dir', './fineweb_cache'), source=cfg.get('fw_source', 'sample-10BT'), fit_docs=cfg.get('fw_fit_docs', 20000))
    raise ValueError(f"load_run_corpus: unknown dataset {name!r} (expected 'wikitext' or 'fineweb')")

def atomic_write_json(path, obj, indent=2):
    text = json.dumps(obj, indent=indent, default=float)
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=os.path.basename(path) + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

def atomic_write_text(path, text):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=os.path.basename(path) + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

def locked_json_read(path):
    """Consistent read of a shared JSON file under the same flock the writers
    use ({} when absent).  Pure read: never creates or rewrites the file."""
    try:
        import fcntl
    except ImportError:
        fcntl = None
    lock_f = None
    if fcntl is not None:
        lock_f = open(path + '.lock', 'a')
        fcntl.flock(lock_f, fcntl.LOCK_SH)
    try:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        except Exception as _e:
            print(f'[resume] FATAL: {path} exists but cannot be parsed ({type(_e).__name__}: {_e}).  Refusing to start with a silently RESET panel — move it aside to start fresh.')
            raise
    finally:
        if lock_f is not None:
            try:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
            finally:
                lock_f.close()


def locked_json_update(path, mutfn, indent=2):
    """Load-modify-write a JSON file under an exclusive flock.

    Parallel cell workers (par.py) all funnel their summary.json mutations
    through this helper, so two children finishing at the same moment can
    never lose each other's records.  mutfn(obj) mutates the loaded object in
    place (or returns a replacement); a missing file starts from {}.  The
    write itself stays atomic (atomic_write_json), so a crash mid-update
    leaves the previous content intact.
    """
    try:
        import fcntl
    except ImportError:
        fcntl = None
    lock_f = None
    if fcntl is not None:
        lock_f = open(path + '.lock', 'a')
        fcntl.flock(lock_f, fcntl.LOCK_EX)
    try:
        obj = {}
        if os.path.exists(path):
            try:
                with open(path, encoding='utf-8') as f:
                    obj = json.load(f)
            except Exception as _e:
                print(f'[resume] FATAL: {path} exists but cannot be parsed ({type(_e).__name__}: {_e}).  Refusing to overwrite it with a fresh object — move it aside to start fresh.')
                raise
        rep = mutfn(obj)
        obj = obj if rep is None else rep
        atomic_write_json(path, obj, indent=indent)
        return obj
    finally:
        if lock_f is not None:
            try:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
            finally:
                lock_f.close()


def atomic_write_csv(path, header, rows):
    import io as _io
    if header is None:
        header = []
        seen = set()
        for r in rows:
            for k in r.keys() if isinstance(r, dict) else r:
                if k not in seen:
                    seen.add(k)
                    header.append(k)
    cols = list(header)
    buf = _io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in rows:
        if isinstance(r, dict):
            w.writerow([r.get(k, '') for k in cols])
        else:
            w.writerow(list(r))
    atomic_write_text(path, buf.getvalue())

def new_experiment_id(prefix='run'):
    """A collision-resistant id for one experiment attempt.

    Uses wall-clock time + pid + OS entropy (never the Python/NumPy/Torch RNG
    streams, so taking an id cannot perturb a seeded run).  The id is written
    into the record and the append-only experiment log, making repeated
    attempts of the same (variant, seed, run_cfg) distinguishable."""
    return f'{prefix}-{time.strftime("%Y%m%dT%H%M%S", time.gmtime())}-{os.getpid()}-{os.urandom(4).hex()}'

def append_jsonl(path, record):
    """Append one JSON object to a shared JSONL log under an exclusive flock.

    Append-only tracking is deliberately separate from summary.json: summary's
    keys are the resume identity `variant::seed`, so a rerun legitimately
    overwrites an old value.  The JSONL ledger keeps every attempt (including
    failures and cap truncations), which is what makes the paper trail
    auditable after the fact.  A write failure is the caller's to report; this
    helper never swallows a partially written line (single write + flush)."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    line = json.dumps(_jsonable(record), sort_keys=True, ensure_ascii=False, default=float) + '\n'
    try:
        import fcntl
    except ImportError:
        fcntl = None
    with open(path, 'a', encoding='utf-8') as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def log_experiment_event(outdir, event):
    """Append a trace event to `outdir/experiments.jsonl` (never raises).

    Tracking is diagnostic infrastructure: a full disk or a read-only archive
    must not destroy a completed training cell.  The warning names the path so
    the missing ledger is visible rather than silent."""
    try:
        path = os.path.join(outdir, 'experiments.jsonl')
        append_jsonl(path, event)
        return path
    except Exception as _e:
        print(f'[track] WARNING: could not append to {os.path.join(outdir, "experiments.jsonl")} ({type(_e).__name__}: {_e}) — the training record itself is unaffected')
        return None

_PINNED_HOST_CACHE = {}
# 整份 pinned 的字节上限（见 batch_iter）：超出时改走普通宿主张量 + 每步
# pinned 暂存上传。
_PINNED_HOST_CAP_BYTES = 4 << 30

def _ids_narrow(train_ids):
    # Token ids are bounded by the vocab (<= 2**31 for every corpus this repo
    # produces), so the resident copy (GPU, or pinned host staging) is stored
    # int32 and widened back to int64 right after each batch gather, leaving
    # the yielded batch bit-identical.  The guard keeps any hypothetical
    # out-of-range corpus on the int64 path.
    if train_ids.dtype == np.int32:
        return train_ids
    if train_ids.size and int(train_ids.max()) < 2 ** 31:
        return train_ids.astype(np.int32)
    return train_ids

def batch_iter(train_ids, seq_len, batch_size, device, seed=0):
    rng = np.random.default_rng(seed)
    n = len(train_ids) - seq_len - 1
    if n <= 0:
        raise ValueError(f'batch_iter needs len(train_ids) > seq_len+1 to sample causal windows, got len={len(train_ids)} seq_len={seq_len}. Increase n_train_tokens or reduce seq_len.')
    train_ids = np.asarray(train_ids)
    host = None
    cached = _PINNED_HOST_CACHE.get('entry')
    if cached is not None and cached[0] is train_ids and cached[1].shape[0] == train_ids.shape[0]:
        host = cached[1]
    elif device.type == 'cuda' and train_ids.nbytes <= (512 << 20):
        try:
            host = torch.from_numpy(_ids_narrow(train_ids)).to(device)
            _PINNED_HOST_CACHE['entry'] = (train_ids, host)
        except RuntimeError:
            host = None
    if host is None:
        if device.type == 'cuda':
            try:
                # 只对不超过 _PINNED_HOST_CAP_BYTES 的语料做整份 pinned：
                # 超过上限时直接用与 numpy 共享的普通宿主张量，每步 gather 后再经
                # _h2d_async 的 pinned 暂存上传——取到的整数逐位相同（transport
                # only）。
                if train_ids.nbytes <= _PINNED_HOST_CAP_BYTES:
                    host = torch.from_numpy(_ids_narrow(train_ids)).pin_memory()
                else:
                    host = torch.from_numpy(_ids_narrow(train_ids))
                _PINNED_HOST_CACHE['entry'] = (train_ids, host)
            except RuntimeError:
                host = torch.from_numpy(train_ids)
        else:
            host = torch.from_numpy(train_ids)
    # Transport-only narrowing (design notes §6.1): every gather index is an
    # exact integer bounded by n + seq_len, so when that bound fits in int32
    # the index tensor is uploaded / consumed as int32 and the gathered ids
    # are widened back to int64 right after the gather exactly as before —
    # bit-identical batches.  The broadcast addition itself is then also done
    # in int32: every operand and the sum are < 2**31, so the int32 result
    # holds exactly the same integers as the int64 form.
    _idx_i32 = n + seq_len + 1 < 2 ** 31
    _idx_dt = np.int32 if _idx_i32 else np.int64
    cols = np.arange(seq_len + 1, dtype=_idx_dt)
    # 复用一张 [B, seq_len+1] 索引缓冲（np.add out= 逐位同值；h2d 路径经
    # _h2d_stage 的 pinned 暂存同步拷贝，下一轮的覆写发生在设备拷贝提交
    # 之后；CPU 路径 host[idx] 则立即实现）。
    idx_np = np.empty((batch_size, seq_len + 1), dtype=_idx_dt)
    while True:
        starts = rng.integers(0, n + 1, size=batch_size).astype(_idx_dt, copy=False)
        np.add(starts[:, None], cols[None, :], out=idx_np)
        idx = torch.from_numpy(idx_np)
        if host.device.type == 'cuda':
            ids = host[_h2d_async(idx, idx.dtype, host.device)]
        else:
            ids = host[idx]
        if ids.dtype != torch.int64:
            # widen back right after the gather: the yielded ids are
            # bit-identical to the int64 form.
            ids = ids.long()
        if ids.device != device:
            ids = _h2d_async(ids, ids.dtype, device)
        yield (ids[:, :-1], ids[:, 1:])

def count_params(m):
    return sum((p.numel() for p in m.parameters()))

def result_is_current(rec, code, *required_keys):
    if not isinstance(rec, dict) or not rec:
        return False
    if not all((k in rec for k in required_keys)):
        return False
    if 'ppl' in required_keys and (not ppl_is_usable(rec.get('ppl'))):
        return False
    return rec.get('_code') == code
PAIR_STEP_KEYS = ('steps', 'tokens_seen')

def pair_reason(ra, rb):
    if not isinstance(ra, dict) or not isinstance(rb, dict):
        return 'a record is not a dict'
    if ra.get('run_cfg') is None or rb.get('run_cfg') is None:
        return 'unverifiable run_cfg'
    if ra.get('run_cfg') != rb.get('run_cfg'):
        return 'different run_cfg'
    # Controlled-variable gate for the settings the run fingerprint does not
    # carry (optimizer identity / betas / eps, LR-schedule shape, grad-clip,
    # augmentation).  BOTH records must carry the descriptor or BOTH must omit
    # it (legacy-vs-legacy); a one-sided descriptor is unverifiable and is
    # refused rather than silently paired.
    _ka, _kb = (ra.get('train_knobs'), rb.get('train_knobs'))
    # Mixed presence is itself unverifiable: a record that carries the
    # descriptor and one that does not cannot be shown to share the optimizer /
    # schedule / clipping / augmentation settings, and silently pairing them is
    # exactly the uncontrolled comparison this gate exists to prevent.  Only
    # legacy-vs-legacy (both absent) keeps the historical permissive path.
    if (_ka is None) != (_kb is None):
        return 'unverifiable train_knobs (only one record carries the optimizer / lr-schedule / augmentation descriptor)'
    if _ka is not None and _ka != _kb:
        return 'different train_knobs (optimizer / lr schedule / grad-clip / augmentation)'
    bad = [k for k in PAIR_STEP_KEYS if ra.get(k) is None or rb.get(k) is None or ra.get(k) != rb.get(k)]
    if bad:
        absent = [k for k in bad if ra.get(k) is None and rb.get(k) is None]
        unknown = [k for k in bad if k not in absent and (ra.get(k) is None) != (rb.get(k) is None)]
        if unknown:
            return f'unverifiable {sorted(unknown)}'
        if absent:
            return f'unverifiable {sorted(absent)}'
        return f'disagreeing {sorted(bad)}'
    return None

def ppl_is_usable(v):
    return isinstance(v, (int, float)) and (not isinstance(v, bool)) and math.isfinite(v) and (v > 0.0)

def by_len_cells(rows, Ln, variant=None):
    keys = (Ln, int(Ln), str(int(Ln)))
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if variant is not None and r.get('variant') != variant:
            continue
        bl = r.get('by_len')
        if not isinstance(bl, dict):
            continue
        cell = None
        for k in keys:
            if k in bl:
                cell = bl[k]
                break
        if not isinstance(cell, dict):
            continue
        if ppl_is_usable(cell.get('ppl')):
            out.append(cell)
    return out

def holm_adjust(named_pvals, alpha=0.05):
    """Holm-Bonferroni step-down adjustment over a family of comparisons.

    输入 [(name, p), ...]（NaN/越界的 p 记为无法校正并原样透传，不进入排序），
    输出 {name: {'p_raw', 'p_holm', 'rank', 'reject', 'm_effective'}}。
    family 的定义必须由调用方预先声明（预注册的族用冻结清单，探索性的族
    明示「未预注册」），本函数只做数学，不做族的选择——族的选择正是
    多重比较问题的来源（design_notes §4/§17）。"""
    fam = []
    invalid = {}
    for name, p in named_pvals:
        try:
            pv = float(p)
        except (TypeError, ValueError):
            pv = float('nan')
        if (not math.isfinite(pv)) or pv < 0.0 or pv > 1.0:
            invalid[str(name)] = {'p_raw': p, 'p_holm': None, 'rank': None, 'reject': False, 'reason': 'p not in [0,1] or not finite'}
            continue
        fam.append((str(name), pv))
    out = dict(invalid)
    m = len(fam)
    order = sorted(fam, key=lambda kv: kv[1])
    prev = 0.0
    for i, (name, p) in enumerate(order):
        adj = min(1.0, (m - i) * p)
        prev = max(prev, adj)
        out[name] = {'p_raw': p, 'p_holm': float(prev), 'rank': i + 1, 'reject': bool(prev <= alpha), 'm_effective': m}
    return out

def signflip_p_floor(n):
    """双侧精确符号翻转检验在 n 个种子下的最小可达 p 值（2/2^n）；
    n<=0 返回 NaN。用于报告「未显著是分辨率上限还是数据结论」。"""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return float('nan')
    if n <= 0:
        return float('nan')
    # same value as 2.0 / 2**n, evaluated in the float exponent range.
    return math.ldexp(2.0, -n)

def finite_range(named_values):
    vals, dropped = ([], [])
    for label, m in named_values:
        if isinstance(m, (int, float)) and (not isinstance(m, bool)) and math.isfinite(m):
            vals.append(float(m))
        else:
            dropped.append(label)
    if not vals:
        return (None, None, dropped)
    return (min(vals), max(vals), dropped)

def exact_signflip_p(deltas):
    """Two-sided exact sign-flip test over paired per-seed deltas.

    H0: the per-seed difference is symmetric about 0.  For n <= 20 every one
    of the 2**n sign assignments is enumerated (deterministic, and the same
    test the driver suites report); for larger n a deterministic normal
    approximation with continuity correction is used instead, so a report
    never depends on a random draw.  Non-finite deltas are dropped (the count
    is returned).  Pure function over its input — used by the aggregate audit
    so the core panel reports the same p the per-phase suites do, rather than
    a bare mean±std a reviewer could mistake for evidence.

    有效性假设（评审二轮·意见九，必须在预注册文档同步声明）：精确符号翻转
    检验的 p 值正确性依赖「配对内交换性」——每个种子下的两臂共享除臂本身
    之外的一切随机因素（同数据顺序、同评测行、同腐坏网格；初始化 RNG 按
    种子播种，臂间参数量差异对 RNG 流的扰动属于「臂」的一部分）。种子编号
    相同本身不提供该保证；design_notes §20 记录了本仓库逐因素的控制方式
    与残余风险（训练随机性进入种间方差、文档抽样进入 doc 层方差）。"""
    d_in = [float(x) for x in deltas]
    d = np.asarray([x for x in d_in if math.isfinite(x)], dtype=float)
    n = int(d.size)
    n_dropped = len(d_in) - n
    if n == 0:
        return {'n': 0, 'p': float('nan'), 'n_dropped': n_dropped, 'method': 'none'}
    obs = abs(float(d.mean()))
    if n <= 20:
        total = 1 << n
        cnt = 0
        bitpos = np.arange(n, dtype=np.int64)[None, :]
        step = max(1, min(total, (1 << 22) // max(n, 1)))
        for off in range(0, total, step):
            bs = min(step, total - off)
            masks = np.arange(off, off + bs, dtype=np.int64)[:, None]
            signs = np.where((masks & (1 << bitpos)) != 0, 1.0, -1.0)
            cnt += int((np.abs((signs * d[None, :]).mean(1)) >= obs - 1e-12).sum())
        return {'n': n, 'p': float(cnt / total), 'n_dropped': n_dropped, 'method': 'exact'}
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    if sd <= 1e-300:
        p = 0.0 if obs > 0 else 1.0
    else:
        z = obs * math.sqrt(n) / sd
        p = math.erfc(z / math.sqrt(2.0))
    return {'n': n, 'p': float(min(1.0, p)), 'n_dropped': n_dropped, 'method': 'normal-approx'}

def paired_effect_stats(deltas, alpha=0.05, min_n=3):
    """Paired contrast summary with an explicit effect-vs-seed-noise verdict.

    Reports n, mean, sample std (ddof=1), sem, Cohen's d_z, the exact
    sign-flip p, the p-resolution floor at this n, and four booleans:

      underpowered       — n < `min_n` (the 3-seed floor the reporting rules
                           require for a key experiment);
      resolution_limited — the smallest p the exact sign-flip test can reach
                           at this n (2/2**n) is still > alpha, so NO effect,
                           however large, can ever clear the significance bar
                           at this seed count.  This is a resolution ceiling,
                           not evidence against the effect, and must not be
                           read as "no effect";
      within_seed_noise  — |mean| <= std, i.e. the seed-to-seed spread alone is
                           at least as large as the reported effect (a +0.2%
                           mean with 0.5% std is NOT a usable gain);
      significant        — p < alpha AND NOT within_seed_noise.

    Requiring both the test AND the effect-vs-spread check means a large-n
    p-value can never dress up a sub-noise effect as a real improvement, while
    `resolution_limited` stops a small-n panel from being read as a negative
    result when only the test's granularity, not the data, bound the p."""
    d = np.asarray([float(x) for x in deltas if isinstance(x, (int, float)) and (not isinstance(x, bool)) and math.isfinite(float(x))], dtype=float)
    n = int(d.size)
    if n == 0:
        return {'n': 0, 'mean': float('nan'), 'std': float('nan'), 'sem': float('nan'), 'cohens_dz': float('nan'), 'p_exact_signflip': float('nan'), 'p_floor': float('nan'), 'min_reachable_p': float('nan'), 'underpowered': True, 'resolution_limited': True, 'within_seed_noise': True, 'significant': False, 'effect_vs_noise': 'no data'}
    mean = float(d.mean())
    std = float(d.std(ddof=1)) if n > 1 else 0.0
    sem = float(std / math.sqrt(n)) if n > 1 else float('nan')
    p = exact_signflip_p(d)['p']
    if std > 1e-12:
        dz = float(mean / std)
    elif abs(mean) < 1e-300:
        dz = 0.0
    else:
        dz = float('inf') if mean > 0 else float('-inf')
    p_floor = signflip_p_floor(n)
    underpowered = n < int(min_n)
    resolution_limited = bool(math.isfinite(p_floor) and p_floor > alpha)
    within_seed_noise = abs(mean) <= std
    significant = bool(math.isfinite(p) and p < alpha and (not within_seed_noise))
    if underpowered:
        verdict = 'underpowered (n<%d)' % int(min_n)
    elif within_seed_noise:
        verdict = 'effect within seed noise (|mean|<=std)'
    elif significant:
        verdict = 'effect exceeds seed noise (p<%.3g)' % alpha
    elif resolution_limited:
        verdict = 'not decidable at this n: exact-test p floor %.4g > alpha (add seeds to resolve; this is a resolution ceiling, not a null result)' % p_floor
    else:
        verdict = 'not significant at this n (p>=%.3g)' % alpha
    return {'n': n, 'mean': mean, 'std': std, 'sem': sem, 'cohens_dz': dz, 'p_exact_signflip': float(p), 'p_floor': p_floor, 'min_reachable_p': p_floor, 'underpowered': underpowered, 'resolution_limited': resolution_limited, 'within_seed_noise': within_seed_noise, 'significant': significant, 'effect_vs_noise': verdict}

# ---- 预注册功效 / 分辨率工具（评审二轮·意见一/九） ------------------------------
# 动机：一个冻结方案如果「无论数据多强都不可能达到 p_holm < α」，它不是
# 保守，而是确认概率恒为零的构造缺陷。下面的函数把这件事变成加载清单时
# 就能机器判定的硬门槛，而不是统计阶段的一句「保守」警告。

def holm_min_attainable_p(n_seeds, m_comparisons, alpha=0.05):
    """双侧精确符号翻转 + Holm 校正族的最小可达 Holm p。

    最小原始 p = 2/2^n（全部种子同向）；Holm 第一道门乘以族大小 m，故
    最小可达 p_holm = min(1, m * 2/2^n)。若该值 > alpha，则该方案在构造上
    无法确认任何比较——这是分辨率上限，不是「保守的选择」，应当在读到
    任何测试数据之前修订（加种子 / 缩族 / 改检验），而不是事后再解释。"""
    try:
        n = int(n_seeds)
        m = int(m_comparisons)
    except (TypeError, ValueError):
        return float('nan')
    if n <= 0 or m <= 0:
        return float('nan')
    return min(1.0, m * signflip_p_floor(n))


def signflip_min_n_for_holm(m_comparisons, alpha=0.05, n_max=64):
    """Smallest seed count n for which a Holm-corrected exact sign-flip test
    can reach p_holm <= alpha EVEN IF every seed moves in the predicted
    direction.  Returns None when no n <= n_max reaches it.

    This is the planning counterpart of `holm_min_attainable_p`: it turns
    "3-5 seeds" into an explicit statement of what those seeds can and cannot
    establish.  At alpha=0.05 a single comparison needs n>=6; a family of m
    comparisons needs roughly 2**n >= 2*m/alpha.  A panel scheduled below the
    returned n is descriptive only (the exact-test floor, not the data, bounds
    the p value), so the audit labels it resolution-limited instead of reading
    it as a null result."""
    try:
        m = int(m_comparisons)
        a = float(alpha)
    except (TypeError, ValueError):
        return None
    if m <= 0 or not (0.0 < a < 1.0):
        return None
    for n in range(1, int(n_max) + 1):
        if holm_min_attainable_p(n, m, a) <= a + 1e-15:
            return n
    return None


def signflip_min_consistency(n_seeds, m_comparisons, alpha=0.05):
    """通过 Holm 第一道门所需的最小同向种子数 k_min（双侧精确检验）。

    若 k 个种子同向（n−k 反向），精确双侧 p = 2·P(Bin(n, 1/2) ≤ n−k)；
    第一道门要求 m·p < alpha。返回使 m·p(k) < alpha 的最小 k；若全部
    n 个同向仍不够，返回 None（方案不可能确认）。"""
    try:
        n = int(n_seeds)
        m = int(m_comparisons)
    except (TypeError, ValueError):
        return None
    if n <= 0 or m <= 0:
        return None
    total = 1 << n
    best = None
    for k in range(n, -1, -1):
        tail = sum(math.comb(n, i) for i in range(0, n - k + 1))
        p = 2.0 * tail / total
        if m * p < alpha:
            best = k  # keep descending: p(k) grows monotonically as k shrinks,
            # so the last k that still clears the gate is the MINIMUM k
        else:
            break
    return best


def signflip_power(n_seeds, m_comparisons, q, alpha=0.05):
    """每种子同向概率为 q 时，通过 Holm 第一道门的功效。

    P(Bin(n, q) ≥ k_min)。q 的含义：在真效应方向上，单个种子的配对 Δ
    落在预注册方向的概率（含种子间噪声）。功效对 q 敏感——预注册时必须
    报告 q 的假设来源（如 validation 面板的同向率），不得只报「数学上
    可能」。n<=0 或方案不可能确认时返回 0.0。"""
    kmin = signflip_min_consistency(n_seeds, m_comparisons, alpha)
    if kmin is None or kmin <= 0:
        return 0.0 if kmin is None else 1.0
    try:
        n = int(n_seeds)
        q = float(q)
    except (TypeError, ValueError):
        return 0.0
    if not (0.0 <= q <= 1.0) or n <= 0:
        return 0.0
    pw = sum(math.comb(n, k) * (q ** k) * ((1.0 - q) ** (n - k)) for k in range(kmin, n + 1))
    return float(min(1.0, pw))


def tost_paired(deltas, margin, conf=0.90):
    """配对 TOST 等价性检验（G0 训练级等价门槛的统计件，评审二轮·意见八）。

    「未发现显著差异 ≠ 等价」：等价性必须预设 margin 并检验 |mean| 落在
    (−margin, +margin) 内。这里用配对 t 的正态近似（n 小时偏保守声明见
    返回值 method）；返回两侧单边 p、conf 水平 CI 与 `equivalent` 判定。
    margin 必须由计划预先注册（如 kernel/eager 面板平均 PPL 相对差 <1%）；
    本函数不替调用方选择 margin。n<2 或 margin<=0 时不判定。"""
    d = np.asarray([float(x) for x in deltas if isinstance(x, (int, float)) and (not isinstance(x, bool)) and math.isfinite(float(x))], dtype=float)
    n = int(d.size)
    out = {'n': n, 'margin': float(margin), 'conf': float(conf), 'equivalent': None, 'method': 'paired-t normal-approx (n>=6 recommended; pre-registered margin required)'}
    if n < 2 or not (margin > 0):
        out['note'] = 'not decidable (n<2 or non-positive margin)'
        return out
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    out['mean'] = mean
    se = sd / math.sqrt(n)
    zc = {0.80: 1.2815515655446004, 0.90: 1.6448536269514722, 0.95: 1.959963984540054}.get(round(float(conf), 4))
    if zc is None:
        zc = 1.6448536269514722
        out['note'] = 'conf not in {0.80, 0.90, 0.95}: fell back to 0.90'
    out['ci'] = [mean - zc * se, mean + zc * se]
    if se <= 1e-300:
        p_hi = 1.0 if mean > margin else 0.0  # degenerate: mean above the upper bound cannot be rejected as 'below it'
        p_lo = 1.0 if mean < -margin else 0.0
    else:
        p_hi = 0.5 * math.erfc((margin - mean) / (math.sqrt(2.0) * se))  # H0: mean >= margin  (P(Z >= (margin-mean)/se))
        p_lo = 0.5 * math.erfc((mean + margin) / (math.sqrt(2.0) * se))  # H0: mean <= -margin  (P(Z <= (-margin-mean)/se) = 0.5*erfc((mean+margin)/(sqrt2*se)))
    out['p_tost'] = float(max(p_hi, p_lo))
    out['equivalent'] = bool(p_hi < 0.05 and p_lo < 0.05)
    out['ci_within_margin'] = bool(out['ci'][0] > -margin and out['ci'][1] < margin)
    return out

# `mt` (matched-variant set) and `mr` (mlp_match_ref) can themselves contain
# underscores (e.g. `full_rope`, `csa_dynamic`), so those two values are
# captured lazily up to the next literal key rather than by `[^_]+`.
_RUN_CFG_RE = re.compile('^steps(?P<steps>[^_]+)_sl(?P<sl>[^_]+)_bs(?P<bs>[^_]+)_nt(?P<nt>[^_]+)_lr(?P<lr>[^_]+)_wd(?P<wd>[^_]+)_d(?P<d>[^_]+)_L(?P<L>[^_]+)_H(?P<H>[^_]+)_Dh(?P<Dh>[^_]+)_v(?P<v>[^_]+)_wu(?P<wu>[^_]+)_cl(?P<cl>[^_]+)_dlm(?P<dlm>[^_]+)_mt(?P<mt>.*?)_mr(?P<mr>.*?)_ee(?P<ee>[^_]+)_es(?P<es>[^_]+)_det(?P<det>[^_]+)_df(?P<df>.+?)_cs(?P<cs>.+)$')

def run_cfg_knobs(fp):
    """Best-effort decode of a `_run_fp` string into its named training knobs.

    Used only for the audit display: it lets the report SHOW that two arms
    share optimizer / schedule / budget / data, instead of asserting it.
    `{}` means the string predates the current layout or is malformed.  The
    authoritative controlled-variable gate is still whole-string equality in
    `pair_reason` — this decoder never substitutes for it."""
    if not isinstance(fp, str):
        return {}
    m = _RUN_CFG_RE.match(fp)
    return m.groupdict() if m else {}

def variant_presence(outdir):
    out = {}
    sp = os.path.join(outdir, 'summary.json')
    if not os.path.exists(sp):
        return out
    try:
        raw = json.load(open(sp, encoding='utf-8'))
    except Exception as e:
        print(f'[stats] WARNING: cannot read {sp} ({type(e).__name__}: {e}) — variant presence cannot be established')
        return out
    kept = {}
    refused = {}
    for _k, r in raw.items():
        if not isinstance(r, dict):
            continue
        v = r.get('variant')
        if v is None:
            v = _k.split('::')[0]
        if ppl_is_usable(r.get('ppl')) and (not r.get('synthesized')) and (_as_seed_int(r.get('seed')) is not None):
            kept[v] = kept.get(v, 0) + 1
        else:
            refused[v] = refused.get(v, 0) + 1
    for v in sorted(set(kept) | set(refused)):
        if v not in refused:
            continue
        if v in kept:
            out[v] = f'present with {kept[v]} measurable record(s) (+{refused[v]} refused) — no shared usable seed'
        else:
            out[v] = f'present but ALL {refused[v]} record(s) refused as unmeasured (synthesized / error / nan ppl / no seed)'
    return out

def format_variant_presence(presence, variants):
    parts = [f'`{v}`: {presence[v]}' for v in variants if v in presence]
    if not parts:
        return None
    return '; '.join(parts) + ' — omitted rather than quoting a cross-panel delta'

def ppl_by_seed(outdir, full=False):
    _by_id = read_panel(outdir)
    out = {}
    if not _by_id:
        return out
    sp = os.path.join(outdir, 'summary.json')
    _seen_seed, _collide = ({}, {})
    for _v, _t, _s in _by_id:
        if (_v, _s) in _seen_seed and _seen_seed[_v, _s] != _t:
            _collide.setdefault(_v, set()).update({_seen_seed[_v, _s], _t})
        _seen_seed[_v, _s] = _t
    if _collide:
        for _v, _tags in sorted(_collide.items()):
            print(f'[stats] WARNING: {sp}: variant `{_v}` holds SEVERAL protocol conditions for one seed (tags {sorted(_tags)}). This reader keeps ONE record per (variant, seed) — the flat map CANNOT represent the others.  A paired contrast built from it would silently quote a single condition; use `ppl_by_seed_grouped` to read the panel per condition.')
    for (_v, _t, _s), (_k, r) in _by_id.items():
        _slot = out.setdefault(_v, {})
        if _s not in _slot:
            if full:
                _slot[_s] = r if 'protocol_tag' in r else {**r, 'protocol_tag': _t}
            else:
                _slot[_s] = r['ppl']
    return out

def _record_identity(rec, key, legacy_ok):
    seed = _as_seed_int(rec.get('seed'))
    if seed is None:
        return None
    var = rec.get('variant')
    if var is None:
        var = str(key).split('::', 1)[0]
    return (str(var), record_tag(rec, key, legacy_ok), seed)

def _as_seed_int(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, float):
        return int(v) if math.isfinite(v) and v.is_integer() else None
    try:
        return int(v)
    except (TypeError, ValueError, OverflowError):
        return None

def read_panel(outdir):
    sp = os.path.join(outdir, 'summary.json')
    if not os.path.exists(sp):
        return {}
    try:
        raw = json.load(open(sp, encoding='utf-8'))
    except Exception as e:
        print(f'[stats] WARNING: cannot read {sp} ({type(e).__name__}: {e}) — this panel is EXCLUDED')
        return {}
    _legacy_ok = legacy_warm_tags(raw)
    _by_id = {}
    for _k, r in raw.items():
        if not isinstance(r, dict) or r.get('seed') is None:
            continue
        if not ppl_is_usable(r.get('ppl')) or r.get('synthesized'):
            continue
        _id = _record_identity(r, _k, _legacy_ok)
        if _id is None:
            continue
        if _id in _by_id:
            print(f'[stats] WARNING: {sp}: two measurable records for variant={_id[0]!r} protocol={_id[1]!r} seed={_id[2]} (keys {_by_id[_id][0]!r}, {_k!r}) — the PPL is AMBIGUOUS and no averaging can resolve it; the earlier key is used, the later one is EXCLUDED')
            continue
        _by_id[_id] = (_k, r)
    return _by_id

def ppl_by_seed_grouped(outdir):
    grouped = {}
    for (_v, _t, _s), (_k, _r) in read_panel(outdir).items():
        grouped.setdefault((_v, _t), {})[_s] = _r
    return grouped

# ---- 对比实验 / 消融 / 统计可靠性的声明式审计（零 GPU）----
# 每个「创新点」按留一法（leave-one-out）写出「带模块」与「去模块」的对照
# 臂。审计对每条对照：(1) 用 `pair_reason` 验证两臂严格同 run_cfg——优化器、
# 学习率调度、权重衰减、训练步数/ token、warmup、comp 正则、模型形状、数据
# 指纹与确定性档全部一致，只有层架构（variant）不同；(2) 逐种子配对给出
# mean±std、d_z 与精确符号翻转 p；(3) 用 |mean| <= std 标出「噪声级提升」、
# 用 n<3 标出欠功效；(4) 汇总每个创新点是否至少有一条完整留一对照。
# 符号约定：Δ = 「带模块」−「去模块」，PPL 越低越好，故 Δ<0 = 模块有用。
ABLATION_DESIGN = {
    'dynamic_blocking': [
        ('csa_dynamic', 'csa_fixed', 'dynamic vs fixed blocking (CSA)'),
        ('hybrid_dynamic', 'hybrid_fixed', 'dynamic vs fixed blocking (CSA+HCA hybrid)'),
    ],
    'learned_selection': [
        ('csa_fixed', 'csa_fix_randidx', 'learned top-k vs random selection (fixed blocking)'),
        ('csa_dynamic', 'csa_dyn_randidx', 'learned top-k vs random selection (dynamic blocking)'),
    ],
    'content_pooling': [
        ('csa_fixed', 'csa_fix_zerocont', 'pooled block content vs zero content (fixed blocking)'),
        ('csa_dynamic', 'csa_dyn_zerocont', 'pooled block content vs zero content (dynamic blocking)'),
    ],
    'attention_sink': [
        ('csa_fixed', 'csa_fix_nosink', 'sink on vs off (fixed blocking)'),
        ('csa_dynamic', 'csa_dyn_nosink', 'sink on vs off (dynamic blocking)'),
        ('full_sink', 'full', 'sink on vs off (dense control)'),
    ],
    'compression_regularizer': [
        ('csa_dynamic', 'csa_dyn_nocomp', 'compression regularizer on vs off (dynamic blocking; the loss-term ablation)'),
    ],
    'local_fusion': [
        ('csa_dyn_fuse', 'csa_dynamic', 'learned local fusion on vs off (CSA)'),
        ('hybrid_csa_dyn_fuse', 'hybrid_csa_dyn', 'learned local fusion on vs off (hybrid)'),
    ],
    'hybrid_hierarchy': [
        ('hybrid_csa_dyn', 'csa_dynamic', 'CSA+HCA hybrid vs CSA-only (dynamic blocking)'),
    ],
}

# 主方法 vs 参数对齐基线的对比也属于「受控对比」：它回答的是方法本身是否
# 优于同参数量、同优化器、同预算、同数据的稠密基线，而不是只回答「消融臂
# 之间谁高谁低」。两条基线都来自 `variant_mlp_ratio` 的参数对齐（MLP 加宽
# 到与 csa_dynamic 相同），因此 `param_controlled` 是这条对比的硬读数。
BASELINE_DESIGN = {
    'compression_vs_matched_dense': [
        ('csa_dynamic', 'full_sw128_matched', 'dynamic CSA vs parameter-matched windowed dense'),
        ('csa_dynamic', 'full_matched', 'dynamic CSA vs parameter-matched full dense'),
    ],
}

def _variant_seed_map(summary):
    """{(variant, protocol, seed): rec} for measurable, non-synthesized records
    with an unambiguous seed, plus the set of (variant, protocol, seed) triples
    that hold more than one record (ambiguous → excluded, first key wins).
    This is the single reader `_print_pairs` and `methodology_audit` share."""
    _legacy_ok = legacy_warm_tags(summary)
    recs = {}
    amb = set()
    for _k, r in summary.items():
        if not (isinstance(r, dict) and ppl_is_usable(r.get('ppl')) and ('seed' in r) and (not r.get('synthesized'))):
            continue
        rk = _record_identity(r, _k, _legacy_ok)
        if rk is None or rk in amb:
            continue
        if rk in recs:
            amb.add(rk)
            recs.pop(rk, None)
            continue
        recs[rk] = r
    return recs, amb

def _paired_seed_deltas(recs, a, b):
    """Per-protocol-condition per-seed (a − b) deltas over seeds where BOTH arms
    have a measurable record AND `pair_reason` confirms identical run_cfg /
    training budget.

    Protocol conditions are matched exactly as `_print_pairs` does: equal
    non-empty tags pair together, and a tagged arm may pair against a pure-base
    ('') arm.  Two DIFFERENT non-empty conditions are never pooled — collapsing
    them (keying only on the seed) would fabricate seed counts and compare
    different measurement protocols (design_notes §3.3).

    Returns {(tag_a, tag_b): (deltas, shared_seeds, skipped)} with
    skipped = [(seed, reason)]."""
    ra = {(t, s): r for (v, t, s), r in recs.items() if v == a}
    rb = {(t, s): r for (v, t, s), r in recs.items() if v == b}
    _a_t = {t for t, _s in ra}
    _b_t = {t for t, _s in rb}
    _a_pure = _a_t == {''}
    _b_pure = _b_t == {''}
    conds = []
    if '' in _a_t and '' in _b_t:
        conds.append(('', ''))
    for tag in sorted(_a_t | _b_t):
        if not tag:
            continue
        if tag in _a_t and tag in _b_t:
            conds.append((tag, tag))
        if tag in _a_t and _b_pure:
            conds.append((tag, ''))
        if tag in _b_t and _a_pure:
            conds.append(('', tag))
    out = {}
    for ts, bs in conds:
        seeds = sorted({s for t, s in ra if t == ts} & {s for t, s in rb if t == bs})
        deltas, skipped = [], []
        for s in seeds:
            reason = pair_reason(ra[ts, s], rb[bs, s])
            if reason is not None:
                skipped.append((s, reason))
                continue
            deltas.append(float(ra[ts, s]['ppl']) - float(rb[bs, s]['ppl']))
        out[(ts, bs)] = (deltas, seeds, skipped)
    return out

def _knobs_equal_across_arms(recs, arms, key):
    """Compare a per-record descriptor (`run_cfg`, `train_knobs`, ...) across
    the arms of one component.  Returns True/False/None: True when at least two
    arms expose the key and all exposed values agree, False when any two
    disagree, None when fewer than two arms expose it (unverifiable)."""
    vals = []
    for a in arms:
        for (v, _t, _s), r in recs.items():
            if v != a:
                continue
            if key in r and r.get(key) is not None:
                vals.append(json.dumps(r.get(key), sort_keys=True, default=str))
                break
            # keep scanning this arm: a legacy/first record without the key
            # must not mask a later record that does record it.
    if len(vals) < 2:
        return None
    return len(set(vals)) == 1

def _params_of_arm(recs, arm):
    for (v, _t, _s), r in recs.items():
        if v != arm:
            continue
        p = r.get('params')
        try:
            return int(p)
        except (TypeError, ValueError):
            continue
    return None

def _param_control(recs, a, b):
    """Relative parameter-count difference between two arms, and whether it is
    small enough to call them parameter-controlled (<=1%).  None means one or
    both records do not expose `params` (unverifiable, not a mismatch)."""
    pa, pb = _params_of_arm(recs, a), _params_of_arm(recs, b)
    if pa is None or pb is None:
        return (None, None)
    rel = abs(pa - pb) / float(max(pa, pb, 1))
    return (float(rel), bool(rel <= 0.01))

def _holm_assign(entries, m_family, alpha, out_key):
    """Attach a Holm step-down adjusted p to each entry over the declared
    family.  `m_family` is the DECLARED number of contrasts, not just the ones
    with data: a missing arm must not shrink the correction family (that would
    make an incomplete ablation look better-powered than it is).  Dummy p=1
    entries fill the difference."""
    named = []
    for i, e in enumerate(entries):
        p = e.get('p_exact_signflip')
        if isinstance(p, (int, float)) and math.isfinite(float(p)):
            named.append((str(i), float(p)))
    m_eff = max(int(m_family), len(entries), len(named))
    for j in range(m_eff - len(named)):
        named.append((f'__missing_{j}', 1.0))
    h = holm_adjust(named, alpha=alpha)
    for i, e in enumerate(entries):
        rec = h.get(str(i))
        e[out_key] = rec.get('p_holm') if isinstance(rec, dict) else None

def _classify_group_status(entries, *, min_n, alpha, m_family, require_param_match=False):
    """One status per declared innovation / baseline claim.

    verified            — a controlled, parameter-controlled (for baselines)
                          contrast is significant after the family-wise Holm
                          correction AND its effect exceeds seed noise;
    measured            — a controlled contrast exists with n >= min_n, but no
                          corrected significance (the old `verified` bar, now
                          reported under an honest name);
    resolution_limited  — a controlled, non-noise contrast exists whose
                          family-corrected exact-test p floor is still above
                          alpha at this n (add seeds; never read as a null);
    underpowered        — controlled contrasts exist but every one is n < min_n;
    noise_level         — powered contrasts exist but all effects are within
                          seed noise (|mean| <= std);
    unverified          — no controlled contrast (missing arm / config mismatch).
    """
    usable = [e for e in entries if e.get('controlled') and (not require_param_match or e.get('param_controlled') is True)]
    if not usable:
        return 'unverified'
    powered = [e for e in usable if int(e.get('n') or 0) >= int(min_n)]
    if not powered:
        return 'underpowered'
    if any(e.get('significant_global') for e in usable):
        return 'verified'
    noise_free = [e for e in powered if not e.get('within_seed_noise')]
    if not noise_free:
        return 'noise_level'
    for e in noise_free:
        fl = holm_min_attainable_p(e.get('n') or 0, m_family, alpha)
        if math.isfinite(fl) and fl > alpha:
            return 'resolution_limited'
    return 'noise_level'

def _scope_design(design, present):
    """Keep only claims this panel actually tests: a component enters the
    family when at least one of its declared contrasts has BOTH arms measured.
    Once in, all of its declared contrasts keep their slots (missing arms are
    dummies), so an incomplete ablation cannot shrink its own correction."""
    out = {}
    present = set(present or ())
    for comp, contrasts in design.items():
        if any(a in present and b in present for a, b, _n in contrasts):
            out[comp] = contrasts
    return out

def methodology_seed_plan(n_seeds, design=None, baselines=None, alpha=0.05, variants=None):
    """Prospective seed budget for a planned panel.

    Returns the declared family size and the minimum n at which a
    Holm-corrected exact sign-flip test could reject at `alpha` if every seed
    moved in the predicted direction.  Drivers call this BEFORE training so a
    3-seed panel is labelled descriptive up front instead of being read after
    the fact as a failed confirmation.  Pure function; never blocks a run."""
    if design is None:
        design = ABLATION_DESIGN
        if baselines is None:
            baselines = BASELINE_DESIGN
    baselines = {} if baselines is None else baselines
    if variants is not None:
        _present = set(variants or ())
        design = _scope_design(design, _present)
        baselines = _scope_design(baselines, _present)
    m = sum(len(v) for v in design.values()) + sum(len(v) for v in baselines.values())
    req = signflip_min_n_for_holm(m, alpha)
    try:
        n = int(n_seeds)
    except (TypeError, ValueError):
        n = 0
    return {'n_seeds': n, 'declared_contrasts': int(m), 'alpha': float(alpha),
            'min_seeds_for_family_significance': req,
            'sufficient_for_family_rejection': bool(req is not None and n >= req),
            'note': 'a panel below the minimum is DESCRIPTIVE ONLY: the exact-test resolution floor, not the observed effect, bounds its p value'}

def methodology_audit(summary, design=None, *, baselines=None, alpha=0.05, min_n=3, outdir=None, write=True, verbose=True):
    """Zero-GPU audit of controlled comparison, ablation completeness and
    statistical reliability for one panel's `summary` dict.

    For every declared leave-one-out contrast (and, by default, the primary
    parameter-matched dense baselines) it recomputes the per-seed paired delta
    under a strict same-run_cfg / same-train_knobs gate, the mean±std / d_z /
    exact sign-flip p, and flags underpowered (n<min_n), noise-level
    (|mean| <= std) and resolution-limited (the exact-test p floor is above
    alpha) results.

    Multiple comparisons are corrected with Holm step-down over the declared
    family.  A missing arm does NOT shrink the family (dummy p=1 entries keep
    the declared m), so an incomplete ablation cannot buy a lower p.  A
    component is `verified` only when a controlled contrast is significant
    AFTER that correction and its effect exceeds seed noise; merely having
    n>=min_n is reported as `measured` (descriptive), never as verification.
    Per-component Holm results are also exported, because the reviewer-facing
    claim is usually per innovation, while the global correction is the
    conservative cross-component reading.

    Returns the report dict and, when `outdir` is given, writes
    `methodology_audit.json`."""
    default_design = design is None
    design = ABLATION_DESIGN if design is None else design
    recs, amb = _variant_seed_map(summary)
    present = sorted({v for (v, _t, _s) in recs})
    if baselines is None:
        # Baseline claims belong to THIS panel's family only when a matched
        # dense arm was actually scheduled.  A pure ablation panel without
        # full_matched/full_sw128_matched is not testing the baseline claim,
        # so including its (missing) slots would only inflate the correction
        # family and mislabel an untested comparison as "unverified".
        _base_arms = {x for _cs in BASELINE_DESIGN.values() for _a, _b, _n in _cs for x in (_a, _b)}
        baselines = BASELINE_DESIGN if (default_design and (_base_arms & set(present))) else {}
    try:
        _min_sig_n = signflip_min_n_for_holm(1, alpha) or 0
    except (ValueError, ZeroDivisionError):
        _min_sig_n = 0
    report = {'variants_present': present, 'n_measurable': len(recs), 'ambiguous': sorted(amb),
              'min_n': int(min_n), 'alpha': float(alpha), 'min_seeds_for_significance': _min_sig_n,
              'components': {}, 'baselines': {},
              'verified_components': [], 'measured_components': [],
              'significant_components': [], 'nominally_significant_components': [],
              'resolution_limited_components': [], 'underpowered_components': [],
              'noise_level_components': [], 'unverified_components': [],
              'verified_baselines': [], 'measured_baselines': [],
              'significant_baselines': [], 'nominally_significant_baselines': [],
              'resolution_limited_baselines': [], 'underpowered_baselines': [],
              'noise_level_baselines': [], 'unverified_baselines': [],
              'incomplete_components': [], 'controlled_variables': {},
              'component_seed_requirements': {}, 'baseline_seed_requirements': {},
              'family': {}, 'warnings': []}

    def _collect(design_group, kind):
        out = {}
        meta = {}
        for comp, contrasts in design_group.items():
            entries = []
            arms = set()
            for a, b, note in contrasts:
                arms.update((a, b))
                a_present, b_present = (a in present, b in present)
                conds = _paired_seed_deltas(recs, a, b)
                _cfg_a = next((r.get('run_cfg') for (v, _t, _s), r in recs.items() if v == a), None)
                _cfg_b = next((r.get('run_cfg') for (v, _t, _s), r in recs.items() if v == b), None)
                _rel, _param_ok = _param_control(recs, a, b)
                _contrast_measured = False
                for (ts, bs), (deltas, shared, skipped) in sorted(conds.items()):
                    st = paired_effect_stats(deltas, alpha=alpha, min_n=min_n)
                    st.update({'with': a, 'without': b, 'note': note, 'kind': kind,
                               'tag_with': ts, 'tag_without': bs,
                               'n_shared': len(shared), 'n_skipped': len(skipped),
                               'skipped_reasons': sorted({reason for _s, reason in skipped}),
                               'controlled': bool(deltas) and (not skipped),
                               'run_cfg_equal': _cfg_a is not None and _cfg_a == _cfg_b,
                               'run_cfg_knobs': run_cfg_knobs(_cfg_a) or None,
                               'param_rel_diff': _rel, 'param_controlled': _param_ok})
                    entries.append(st)
                    if deltas:
                        _contrast_measured = True
                if a_present and b_present:
                    if not _contrast_measured:
                        _why = sorted({reason for (_dl, _sh, _sk) in conds.values() for _s, reason in _sk})
                        if _why:
                            report['warnings'].append('%s: `%s` vs `%s` both measured but NO seed clears the same-run_cfg/train_knobs gate — %s' % (comp, a, b, '; '.join(_why)))
                        else:
                            report['warnings'].append('%s: `%s` vs `%s` both measured but share no common protocol condition/seed — no contrast computed' % (comp, a, b))
                elif a_present != b_present:
                    report['warnings'].append('%s: incomplete %s `%s` vs `%s` — missing `%s`' % (comp, kind, a, b, a if not a_present else b))
                else:
                    report['warnings'].append('%s: neither `%s` nor `%s` measured — no %s for this %s yet' % (comp, a, b, 'leave-one-out' if kind == 'component' else 'baseline contrast', kind))
                if kind == 'baseline' and _param_ok is False:
                    report['warnings'].append('%s: `%s` vs `%s` differ in parameter count by %.2f%% — NOT parameter-controlled, so this baseline claim is not counted as verified' % (comp, a, b, 100.0 * (_rel or 0.0)))
                elif kind == 'baseline' and _param_ok is None and a_present and b_present:
                    report['warnings'].append('%s: `%s` vs `%s` carry no usable `params` — parameter control is UNVERIFIABLE, so this baseline claim is not counted as verified' % (comp, a, b))
            out[comp] = entries
            meta[comp] = {'arms': arms}
        return out, meta

    comp_entries, comp_meta = _collect(design, 'component')
    base_entries, base_meta = _collect(baselines, 'baseline')
    # Family = claims this panel can test (a component with no measured pair
    # is not part of the correction burden).  Within a tested component, all
    # declared contrasts keep their slots, so an incomplete ablation cannot
    # buy a smaller m by omitting an arm.
    _present_set = set(present)
    _scoped_design = _scope_design(design, _present_set)
    _scoped_baselines = _scope_design(baselines, _present_set)
    declared_m = sum(len(v) for v in _scoped_design.values()) + sum(len(v) for v in _scoped_baselines.values())
    all_entries = [e for group in (comp_entries, base_entries) for entries in group.values() for e in entries]
    m_family = max(declared_m, len(all_entries))
    _holm_assign(all_entries, m_family, alpha, 'p_holm_global')
    _holm_assign(all_entries, m_family, alpha, 'p_holm_family')
    for e in all_entries:
        _ph = e.get('p_holm_global')
        e['significant_global'] = bool(_ph is not None and float(_ph) < alpha and (not e.get('within_seed_noise')) and e.get('controlled'))
        e['significant_family'] = e['significant_global']
    report['family'] = {
        'name': ('ABLATION_DESIGN+BASELINE_DESIGN' if _scoped_baselines else 'ABLATION_DESIGN') + ' (exploratory; not pre-registered)',
        'declared_contrasts': int(declared_m),
        'effective_m': int(m_family),
        'alpha': float(alpha),
        'min_seeds_for_family_significance': signflip_min_n_for_holm(m_family, alpha),
        'note': 'Holm step-down over every declared contrast; missing arms keep their slot (dummy p=1). A panel below min_seeds_for_family_significance is resolution-limited, however large the observed effect.',
    }

    def _publish(group_name, design_group, entries_by_comp, meta, kind, require_param_match):
        status_names = ('verified', 'measured', 'significant', 'nominally_significant',
                        'resolution_limited', 'underpowered', 'noise_level', 'unverified')
        lists = {s: [] for s in status_names}
        for comp, contrasts in design_group.items():
            entries = entries_by_comp.get(comp, [])
            # Per-component Holm: the family is this innovation's own declared
            # contrasts (the reviewer usually asks "is THIS module useful?").
            m_comp = max(len(contrasts), len(entries))
            _holm_assign(entries, m_comp, alpha, 'p_holm_component')
            for e in entries:
                _phc = e.get('p_holm_component')
                e['significant_component'] = bool(_phc is not None and float(_phc) < alpha and (not e.get('within_seed_noise')) and e.get('controlled'))
            status = _classify_group_status(entries, min_n=min_n, alpha=alpha, m_family=m_family,
                                            require_param_match=require_param_match)
            lists[status].append(comp)
            if status == 'verified':
                # `verified` is the corrected-significance tier; keep the
                # historical `significant_*` key pointing at the same set so
                # downstream readers do not silently lose the signal.
                lists['significant'].append(comp)
            if any(e.get('significant') for e in entries if e.get('controlled')):
                lists['nominally_significant'].append(comp)
            if any(int(e.get('n') or 0) >= int(min_n) and e.get('controlled') for e in entries):
                lists['measured'].append(comp)
            report[group_name][comp] = entries
            _arms = meta[comp]['arms']
            report['controlled_variables'][comp] = {
                'run_cfg_equal': _knobs_equal_across_arms(recs, _arms, 'run_cfg'),
                'train_knobs_equal': _knobs_equal_across_arms(recs, _arms, 'train_knobs'),
                'arms_present': sorted(a for a in _arms if a in present),
                'param_matched': all(e.get('param_controlled') is True for e in entries) if entries else None,
            }
            _req_key = 'component_seed_requirements' if kind == 'component' else 'baseline_seed_requirements'
            report[_req_key][comp] = {
                'declared_contrasts': int(len(contrasts)),
                'min_seeds_for_component_significance': signflip_min_n_for_holm(len(contrasts), alpha),
                'min_seeds_for_family_significance': signflip_min_n_for_holm(m_family, alpha),
            }
        for status in status_names:
            report[f'{status}_{group_name}'] = lists[status]
        return lists

    _publish('components', design, comp_entries, comp_meta, 'component', require_param_match=False)
    _publish('baselines', baselines, base_entries, base_meta, 'baseline', require_param_match=True)

    if write and outdir:
        try:
            os.makedirs(outdir, exist_ok=True)
            atomic_write_json(os.path.join(outdir, 'methodology_audit.json'), report)
        except Exception as e:
            print('[audit] could not write methodology_audit.json: %s: %s' % (type(e).__name__, e))
    if verbose:
        _print_methodology_audit(report)
    return report

def _print_methodology_audit(report):
    print('\n========== methodology audit: controlled comparison / ablation / statistics ==========')
    print('  variants with measurable records: ' + (', '.join(report['variants_present']) or '(none)'))
    print('  Δ = with-module − without-module (perplexity; negative = the module helps).')
    _fam = report.get('family') or {}
    print('  family: %d declared contrasts, m_eff=%s; Holm p is the family-corrected p.' % (int(_fam.get('declared_contrasts') or 0), _fam.get('effective_m')))
    if _fam.get('min_seeds_for_family_significance'):
        print('  minimum seeds for a family-corrected rejection at alpha=%g: n>=%d (below this every p is resolution-limited).' % (float(report.get('alpha', 0.05)), int(_fam['min_seeds_for_family_significance'])))
    for comp, entries in report['components'].items():
        live = [e for e in entries if e['n_shared'] > 0]
        if not live:
            continue
        _cv = report.get('controlled_variables', {}).get(comp, {})
        _kn = _cv.get('train_knobs_equal')
        _kn_s = {True: 'optimizer/schedule knobs shared', False: 'OPTIMIZER/SCHEDULE KNOBS DIFFER', None: 'optimizer/schedule knobs not recorded (legacy or absent)'}[_kn]
        print('  [%s]  %s; arms=%s' % (comp, _kn_s, ','.join(_cv.get('arms_present') or []) or 'none'))
        for e in live:
            _c = 'controlled (same run_cfg)' if e['controlled'] else ('UNCONTROLLED: ' + '; '.join(e['skipped_reasons']))
            _p = ('%.4f' % e['p_exact_signflip']) if math.isfinite(e['p_exact_signflip']) else 'n/a'
            _ph = e.get('p_holm_global')
            _phs = ('%.4f' % _ph) if isinstance(_ph, (int, float)) and math.isfinite(_ph) else 'n/a'
            _tag = '' if e.get('tag_with', '') == e.get('tag_without', '') == '' else '   [%s vs %s]' % (e.get('tag_with') or '(base)', e.get('tag_without') or '(base)')
            print('    %-22s − %-22s  n=%-2d  Δ=%+.3f ± %.3f  d_z=%+.2f  p=%s  p_Holm=%s%s' % (e['with'], e['without'], e['n'], e['mean'], e['std'], e['cohens_dz'], _p, _phs, _tag))
            print('        %s | %s' % (_c, e['effect_vs_noise']))
    if report['verified_components']:
        print('  verified innovations (controlled AND family-Holm p<alpha AND |Δ|>seed noise): ' + ', '.join(report['verified_components']))
    if report.get('measured_components'):
        print('  measured innovations (controlled contrast with n>=%d; descriptive unless also verified): ' % int(report.get('min_n', 3)) + ', '.join(report['measured_components']))
    if report.get('nominally_significant_components'):
        print('  nominally significant (raw p<alpha, before family correction): ' + ', '.join(report['nominally_significant_components']))
    if report.get('resolution_limited_components'):
        print('  resolution-limited (non-noise effect, but the family-corrected p floor > alpha at this n; need n>=%s — add seeds, do NOT read as a null): ' % str((_fam.get('min_seeds_for_family_significance') or '?')) + ', '.join(report['resolution_limited_components']))
    if report.get('noise_level_components'):
        print('  noise-level (effect within seed noise across all contrasts): ' + ', '.join(report['noise_level_components']))
    if report.get('underpowered_components'):
        print('  underpowered innovations (contrasts exist but n<%d — add seeds before claiming an effect): ' % int(report.get('min_n', 3)) + ', '.join(report['underpowered_components']))
    if report['unverified_components']:
        print('  unverified innovations (no controlled leave-one-out contrast): ' + ', '.join(report['unverified_components']))
    if report.get('baselines'):
        print('  --- primary baseline claims (parameter-matched dense control) ---')
        for comp, entries in report['baselines'].items():
            live = [e for e in entries if e['n_shared'] > 0]
            if not live:
                continue
            _pc = report.get('controlled_variables', {}).get(comp, {}).get('param_matched')
            print('  [%s]  parameter-controlled=%s' % (comp, _pc))
            for e in live:
                _p = ('%.4f' % e['p_exact_signflip']) if math.isfinite(e['p_exact_signflip']) else 'n/a'
                _ph = e.get('p_holm_global')
                _phs = ('%.4f' % _ph) if isinstance(_ph, (int, float)) and math.isfinite(_ph) else 'n/a'
                print('    %-22s − %-22s  n=%-2d  Δ=%+.3f ± %.3f  d_z=%+.2f  p=%s  p_Holm=%s  params=%s' % (e['with'], e['without'], e['n'], e['mean'], e['std'], e['cohens_dz'], _p, _phs, ('yes' if e.get('param_controlled') else ('no' if e.get('param_controlled') is False else 'n/a'))))
                print('        %s | %s' % ('controlled (same run_cfg)' if e['controlled'] else ('UNCONTROLLED: ' + '; '.join(e['skipped_reasons'])), e['effect_vs_noise']))
        if report.get('verified_baselines'):
            print('  verified baseline claims: ' + ', '.join(report['verified_baselines']))
        if report.get('nominally_significant_baselines'):
            print('  nominally significant baseline claims (raw p<alpha): ' + ', '.join(report['nominally_significant_baselines']))
        if report.get('resolution_limited_baselines'):
            print('  resolution-limited baseline claims: ' + ', '.join(report['resolution_limited_baselines']))
        if report.get('unverified_baselines'):
            print('  unverified baseline claims (missing/incomplete/param-mismatched): ' + ', '.join(report['unverified_baselines']))
    for w in report['warnings']:
        print('  [warn] ' + w)
    print('======================================================================================')

def add_paired(*_a, **_kw):
    raise NotImplementedError('use v9_supp.add_paired (exp_lib cannot import v7_supp)')

def set_seed(seed):
    """Seed every RNG that can affect a run: Python ``random``, NumPy's
    legacy global RNG, PyTorch's CPU RNG and every CUDA device, plus the
    Python hash seed (best effort for child processes).  Returns the integer
    seed so callers can record exactly what was used.  torch's deterministic
    algorithm / cuDNN switches are pinned at the same time, so two runs with
    the same seed on the same machine follow the same kernels."""
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    _attb_bump_epoch()
    torch.manual_seed(seed)
    _pin_cpu_threads()
    # Pin the backend determinism switches on CPU too: the cuDNN flags are
    # inert without CUDA, while CSA_DETERMINISTIC must still enable torch's
    # deterministic-algorithm mode on a CPU box (a CPU run is a first-class
    # reproduction target).  CUDA RNGs are seeded explicitly below.
    _pin_cuda_determinism()
    if torch.cuda.is_available():
        # manual_seed already covers all devices; the explicit current-device
        # call documents the contract and is harmless.
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # TensorFlow is not a dependency of this repo, but a future/embedded
    # TensorFlow path must not become an unseeded RNG source.  Only touch it
    # when the user has already imported it (never import TensorFlow here:
    # that would double the import cost of every CPU-only run).
    _tf = sys.modules.get('tensorflow')
    if _tf is not None:
        try:
            _tf.random.set_seed(seed)
            try:
                _tf.config.experimental.enable_op_determinism()
            except Exception:
                pass
        except Exception as _tfe:
            print(f'[determinism] could not seed the already-imported TensorFlow RNG ({type(_tfe).__name__}: {_tfe})')
    return seed
_CPU_THREADS_PINNED = None

def rng_fingerprint():
    """Digest of the *current* Python/NumPy/Torch/CUDA RNG state.

    Recorded immediately after `set_seed(seed)` in every training record.  Two
    runs with the same seed on the same stack must produce the same digest;
    a mismatch means some import/construction consumed randomness before the
    run started (and therefore that the seed is not fully controlled).  The
    digest is a diagnostic only: it never enters the resume fingerprint."""
    h = hashlib.sha256()
    h.update(repr(random.getstate()).encode('utf-8', 'replace'))
    _ns = np.random.get_state()
    h.update(str(_ns[0]).encode('ascii', 'replace'))
    h.update(np.asarray(_ns[1], dtype=np.uint64).tobytes())
    h.update(repr((_ns[2], _ns[3], _ns[4])).encode('ascii', 'replace'))
    h.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        try:
            for _st in torch.cuda.get_rng_state_all():
                h.update(_st.cpu().numpy().tobytes())
        except Exception:
            # a broken CUDA context is reported elsewhere; the CPU part of
            # the fingerprint is still a useful diagnostic
            pass
    return h.hexdigest()[:16]

def dataloader_generator(seed):
    """A torch.Generator for a future torch DataLoader (scale path).

    The current loader is `batch_iter`, which is seeded directly by the run
    seed.  This helper is for DataLoader-based code: pass it as
    ``DataLoader(..., generator=L.dataloader_generator(seed))`` and pass
    ``worker_init_fn=L.seed_worker`` so worker RNGs are derived from the same
    seed instead of the OS entropy pool."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g

def seed_worker(worker_id):
    """DataLoader worker_init_fn: seed Python/NumPy/Torch from torch's
    per-worker base seed.  PyTorch already seeds its own worker RNG before
    calling this hook; re-seeding it explicitly documents the contract and
    also covers older versions where a custom generator changes the base."""
    _seed = int(torch.initial_seed()) % (2 ** 32)
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)

def dataloader_kwargs(seed):
    """DataLoader kwargs that pin BOTH the sampler order and worker RNGs.

    Scale path: `DataLoader(ds, **L.dataloader_kwargs(seed), num_workers=n)`.
    The generator controls shuffling/sampling; worker_init_fn derives each
    worker's Python/NumPy RNG from the same seed tree instead of OS entropy.
    """
    return {'generator': dataloader_generator(seed), 'worker_init_fn': seed_worker}

def deterministic_dataloader(dataset, *, seed=0, **kwargs):
    """Thin `torch.utils.data.DataLoader` wrapper with reproducible seeding.

    Existing caller-supplied `generator` / `worker_init_fn` are respected."""
    from torch.utils.data import DataLoader
    kwargs.setdefault('generator', dataloader_generator(seed))
    kwargs.setdefault('worker_init_fn', seed_worker)
    return DataLoader(dataset, **kwargs)

def _pin_cpu_threads():
    global _CPU_THREADS_PINNED
    if _CPU_THREADS_PINNED is None:
        want = os.environ.get('CSA_CPU_THREADS', '').strip()
        try:
            n = max(1, int(want)) if want else 1
        except ValueError:
            print(f'[determinism] CSA_CPU_THREADS={want!r} is not an integer — using 1')
            n = 1
        os.environ.setdefault('OMP_NUM_THREADS', str(n))
        os.environ.setdefault('MKL_NUM_THREADS', str(n))
        _CPU_THREADS_PINNED = n
    else:
        n = _CPU_THREADS_PINNED
    try:
        if int(torch.get_num_threads()) != int(n):
            torch.set_num_threads(n)
    except Exception as e:
        print(f'[determinism] could not pin the CPU thread count to {n} ({type(e).__name__}: {e}) — a CPU run may not be reproducible')
    return n

def _pin_cuda_determinism():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    mode = os.environ.get('CSA_DETERMINISTIC', '').strip().lower()
    if mode in ('1', 'true', 'yes', 'strict'):
        try:
            torch.use_deterministic_algorithms(True)
            return 'strict'
        except Exception as e:
            print(f"[determinism] CSA_DETERMINISTIC=1 requested but the strict mode could not be enabled ({type(e).__name__}: {e}); falling back to cudnn-deterministic only.  The active level is recorded in each record's `run_cfg`.")
            return 'cudnn'
    if mode in ('warn', 'warn_only'):
        torch.use_deterministic_algorithms(True, warn_only=True)
        return 'warn'
    return 'cudnn'

def determinism_label():
    if not torch.cuda.is_available():
        return 'cpu'
    if torch.are_deterministic_algorithms_enabled():
        return 'strict' if not torch.is_deterministic_algorithms_warn_only_enabled() else 'warn'
    return 'cudnn' if torch.backends.cudnn.deterministic else 'off'

def _jsonable(v):
    """Recursively convert a config/provenance value to JSON-safe data.
    Sets are sorted so two equal configs always serialise identically."""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (set, frozenset)):
        return sorted((_jsonable(x) for x in v), key=str)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)

def _git_provenance():
    """HEAD / branch / dirty state, with a bounded dirty-file list.  Never
    raises: a source tarball without .git is still a valid experiment."""
    def _run(*args):
        try:
            p = subprocess.run(['git', *args], cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True, timeout=20)
            return p.stdout.strip() if p.returncode == 0 else None
        except Exception:
            return None
    head = _run('rev-parse', 'HEAD')
    branch = _run('rev-parse', '--abbrev-ref', 'HEAD')
    dirty = _run('status', '--porcelain')
    files = dirty.splitlines() if dirty else []
    return {'git_commit': head, 'git_branch': branch, 'git_dirty': bool(files), 'git_dirty_count': len(files), 'git_dirty_files': files[:20]}

def _gpu_names():
    try:
        return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    except Exception:
        return []

def environment_snapshot():
    """Compact software/hardware snapshot for experiment tracking."""
    import platform
    env = {
        'python': sys.version.split()[0],
        'executable': sys.executable,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
        'torch': torch.__version__,
        'numpy': np.__version__,
        'cuda_available': bool(torch.cuda.is_available()),
        'cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        'gpu_names': _gpu_names(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'determinism': determinism_label(),
        'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
        'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
        'deterministic_algorithms': bool(torch.are_deterministic_algorithms_enabled()),
        'tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32) if torch.cuda.is_available() else None,
        'tf32_cudnn': bool(torch.backends.cudnn.allow_tf32) if torch.cuda.is_available() else None,
        'cpu_threads': int(torch.get_num_threads()),
        'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
        'flags': {k: os.environ.get(k) for k in ('CSA_KERNEL', 'CSA_CKPT', 'CSA_DETERMINISTIC', 'CSA_CPU_THREADS', 'CUBLAS_WORKSPACE_CONFIG')},
    }
    for _mod, _key in (('tokenizers', 'tokenizers'), ('datasets', 'datasets')):
        try:
            env[_key] = __import__(_mod).__version__
        except Exception:
            env[_key] = None
    return env

def experiment_provenance(cfg, *, train_ids=None, val_batch=None, vocab=None):
    """Everything needed to identify a panel: code version (semantic stamp
    + git commit), environment, and the exact data bytes seen."""
    data = {'dataset': cfg.get('dataset') or 'wikitext', 'requested_train_tokens': cfg.get('n_train_tokens'), 'seq_len': cfg.get('seq_len')}
    if train_ids is not None:
        data['train_tokens'] = int(len(train_ids))
        data['train_fp'] = _ids_fp(train_ids)
    if val_batch is not None:
        _vb = np.ascontiguousarray(val_batch).reshape(-1)
        data['val_sequences'] = int(np.asarray(val_batch).shape[0])
        data['val_fp'] = _ids_fp(_vb)
    if vocab is not None:
        data['vocab_size'] = int(vocab)
    return {'code': {'code_semantics': CODE_SEMANTICS, 'code_stamp': code_stamp(), 'kernel_stamp': _kernel_stamp(), **_git_provenance()}, 'environment': environment_snapshot(), 'data': data}

def run_hyperparameters(cfg, *, seed, d, n_layers, n_heads, d_head, vocab, mlp_ratio):
    """Structured hyper-parameter record (the run_cfg string stays the
    pairing gate; this is the human/machine-readable copy)."""
    hp = {str(k): _jsonable(v) for k, v in cfg.items() if k not in ('seeds', 'variants', 'outdir')}
    hp.update({
        'seed': int(seed),
        'data_seed': int(seed),
        'd': int(d),
        'n_layers': int(n_layers),
        'n_heads': int(n_heads),
        'd_head': int(d_head),
        'vocab_size': int(vocab),
        'mlp_ratio': float(mlp_ratio),
        'dataset': cfg.get('dataset') or 'wikitext',
        'train_knobs': training_knobs(),
    })
    return hp

def write_run_manifest(outdir, cfg, run_cfg, prov, summary, sanity=None):
    """Merge one panel's provenance/config/outcome summary into
    `outdir/run_manifest.json` under the same flock used by summary.json.

    The per-record `losses` / `ppl_history` curves stay in summary.json (to
    avoid duplicating large arrays); each manifest cell points at its key.
    `sanity` (the small-batch overfit reports) is embedded when supplied, or
    read back from `outdir/overfit_sanity.json` so an auto-preflight is
    traceable without a second manual record."""
    os.makedirs(outdir, exist_ok=True)
    if sanity is None:
        _san_path = os.path.join(outdir, 'overfit_sanity.json')
        if os.path.exists(_san_path):
            try:
                with open(_san_path, encoding='utf-8') as _sf:
                    sanity = json.load(_sf)
            except Exception as _se:
                print(f'[track] WARNING: could not read {_san_path} for the run manifest ({type(_se).__name__}: {_se})')
    cells = {}
    for seed in cfg.get('seeds') or []:
        for v in cfg.get('variants') or []:
            key = f'{v}::seed{seed}'
            rec = summary.get(key)
            if not isinstance(rec, dict) or not ppl_is_usable(rec.get('ppl')) or rec.get('synthesized'):
                continue
            _losses = rec.get('losses') if isinstance(rec.get('losses'), list) else []
            _loss_nums = [float(x) for x in _losses if isinstance(x, (int, float)) and not isinstance(x, bool)]
            _ph = rec.get('ppl_history') if isinstance(rec.get('ppl_history'), list) else []
            cells[key] = {
                'variant': v,
                'seed': int(seed),
                'run_id': rec.get('run_id'),
                'started_at': rec.get('started_at'),
                'finished_at': rec.get('finished_at'),
                'ppl': rec.get('ppl'),
                'steps': rec.get('steps'),
                'tokens_seen': rec.get('tokens_seen'),
                'params': rec.get('params'),
                'train_time_s': rec.get('train_time_s'),
                'final_loss_smoothed': rec.get('final_loss_smoothed'),
                'rng_fp': rec.get('rng_fp'),
                'data_fp': rec.get('data_fp'),
                'val_fp': rec.get('val_fp'),
                'code_stamp': rec.get('_code'),
                'hyperparams': rec.get('hyperparams'),
                'curve_summary': {'n_points': len(_losses), 'first': _loss_nums[0] if _loss_nums else None, 'last': _loss_nums[-1] if _loss_nums else None, 'best': min(_loss_nums) if _loss_nums else None, 'ppl_history_points': len(_ph)},
                'curve_ref': f'summary.json#{key}.losses',
                'ppl_history_ref': f'summary.json#{key}.ppl_history',
                'sanity_ref': rec.get('sanity_ref') or ('overfit_sanity.json' if sanity else None),
            }
    entry = {
        'run_cfg': run_cfg,
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': _jsonable(cfg),
        'provenance': prov,
        'sanity': sanity,
        'experiment_log': 'experiments.jsonl',
        'cells': cells,
    }
    path = os.path.join(outdir, 'run_manifest.json')

    def _merge(m):
        if not isinstance(m, dict):
            m = {}
        m['manifest_version'] = m.get('manifest_version') or 1
        m.setdefault('panels', {})[run_cfg] = entry
        return m
    locked_json_update(path, _merge)
    return path

@torch.inference_mode()
def eval_ppl(model, val_batch, device, chunk=8, eval_rows=None, eval_seed=0):
    chunk = max(1, int(chunk))
    was_training = model.training
    model.eval()
    need_reg_layers = [blk.attn for blk in model.blocks if hasattr(blk.attn, 'need_reg')]
    need_reg_prior = [_a.need_reg for _a in need_reg_layers]
    nll = torch.zeros((), dtype=torch.float64, device=device)
    for _a in need_reg_layers:
        _a.need_reg = False
    try:
        ntok = 0
        V = None
        _sel = None
        if eval_rows is not None and int(eval_rows) > 0:
            _n_rows = int(val_batch.shape[0])
            _n_tgt = max(0, int(val_batch.shape[1]) - 1)
            _flat = _n_rows * _n_tgt
            _k = min(int(eval_rows), _flat)
            if _k < _flat:
                _g = torch.Generator().manual_seed(int(eval_seed))
                _sel = torch.randperm(_flat, generator=_g)[:_k]
                # keep the sorted host copy: per-chunk membership bounds are
                # integer range queries on a sorted table — searchsorted on the
                # host gives the same slice the boolean-mask form selected
                # (host-side integer equivalence, design notes §6.2).
                _sel_host = torch.sort(_sel).values
                _sel_sorted = _sel_host.to(device)
                print(f'[eval_ppl] scoring a uniform sample of {_k}/{_flat} target rows (eval_rows={int(eval_rows)}, eval_seed={int(eval_seed)}) — this is a SAMPLED estimate, not the full-set PPL')
            else:
                print(f'[eval_ppl] eval_rows={int(eval_rows)} covers the whole {_flat}-row target space; scoring the FULL set (this is the population statistic, not a sample)')
        _vhost = None
        if device.type == 'cuda':
            _vc = _PINNED_HOST_CACHE.get('val')
            if _vc is not None and _vc[0] is val_batch and _vc[1] == tuple(val_batch.shape):
                _vhost = _vc[2]
            else:
                try:
                    _vhost = torch.as_tensor(val_batch).pin_memory()
                    _PINNED_HOST_CACHE['val'] = (val_batch, tuple(val_batch.shape), _vhost)
                except RuntimeError:
                    _vhost = torch.as_tensor(val_batch)
        else:
            _vhost = torch.as_tensor(val_batch)
        _async = _vhost.is_pinned()
        _span = max(0, int(val_batch.shape[1]) - 1)
        for i in range(0, val_batch.shape[0], chunk):
            _lo = _hi = 0
            if _sel is not None:
                # Host-side range count on the sorted selection: chunks with
                # no sampled target contribute nothing to nll/ntok (the trunk's
                # comp_reg side effect is unread in eval and overwritten by the
                # next trunk).
                _base = i * _span
                _wide = min(chunk, int(val_batch.shape[0]) - i) * _span
                if _wide > 0:
                    _lo = int(torch.searchsorted(_sel_host, _base))
                    _hi = int(torch.searchsorted(_sel_host, _base + _wide))
                if _hi <= _lo:
                    continue
            elif _span == 0:
                continue
            ids = _vhost[i:i + chunk].to(device, non_blocking=_async)
            # Trunk + per-slice LM head; comp_reg / need_reg handling is
            # identical to model(ids) because forward() itself is exactly
            # trunk() followed by head().
            h = model.trunk(ids)
            if _sel is not None:
                _local = _sel_sorted[_lo:_hi] - _base
                _ri = torch.div(_local, _span, rounding_mode='floor')
                _ci = _local - _ri * _span
                # gather the selected hidden rows BEFORE the head: each output
                # row of the Linear is an independent function of one hidden
                # row, so head(h[sel]) == head(h)[sel]
                _rows = model.head(h[_ri, _ci, :])
                if V is None:
                    V = _rows.size(-1)
                _tgts = ids[_ri, _ci + 1]
                del _ri, _ci
                nll += F.cross_entropy(_rows, _tgts, reduction='sum').double()
                ntok += int(_tgts.numel())
                del _rows, _tgts
            else:
                _tgts = ids[:, 1:]
                for _s0 in range(0, _span, 64):
                    _sl = slice(_s0, min(_s0 + 64, _span))
                    _lg = model.head(h[:, _sl, :])
                    if V is None:
                        V = _lg.size(-1)
                    nll += F.cross_entropy(_lg.transpose(1, 2), _tgts[:, _sl], reduction='sum').double()
                    del _lg
                ntok += int(_tgts.numel())
                del _tgts
            del ids, h
    finally:
        for _a, _prev in zip(need_reg_layers, need_reg_prior):
            _a.need_reg = _prev
        model.train(was_training)
    if not ntok:
        return float('nan')
    try:
        return math.exp(nll.item() / ntok)
    except OverflowError:
        return float('inf')

def enable_block_stats(model, on):
    for blk in model.blocks:
        blk.attn._stats = [] if on else None
        blk.attn._cuts = [] if on else None

def boundary_alignment(pred_cuts, gt_mask_row, n, tol=1, provenance=None):
    gt = np.nonzero(np.asarray(gt_mask_row)[:n - 1])[0] + 1
    pred = np.asarray(pred_cuts)
    P, G = (len(pred), len(gt))
    if P == 0 or G == 0:
        if provenance is not None:
            provenance.setdefault('bnd_rand_exact', True)
        return (0.0, 0.0, 0.0, 0.0)
    hits, ghits = _greedy_match_counts(pred, gt, tol)
    prec, rec = (hits / P, ghits / G)
    f1 = 2 * prec * rec / max(prec + rec, 1e-09)
    n_pos = max(n - 1, 1)
    _approx = []
    rand_prec = _random_cut_precision(P, None, n_pos, tol, G, gt, approx_flag=_approx)
    if provenance is not None:
        provenance['bnd_rand_exact'] = provenance.get('bnd_rand_exact', True) and (not _approx)
    return (prec, rec, f1, rand_prec)

def _mask_runs(near, n_pos, tol):
    m = np.asarray(near)[1:n_pos + 1].astype(bool)
    if not m.any():
        return np.zeros(0, dtype=int)
    d = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
    starts = np.nonzero(d == 1)[0]
    ends = np.nonzero(d == -1)[0]
    lens = ends - starts
    out, ok = ([], True)
    for s, e, L in zip(starts, ends, lens):
        if L == 2 * tol + 1:
            out.append((s + e - 1) // 2)
        elif s == 0 and L == tol + 1:
            out.append(0)
        elif e == m.size and L == tol + 1:
            out.append(m.size - 1)
        elif L > 2 * tol + 1:
            ok = False
            break
        else:
            ok = False
            break
    if ok:
        _singletons = bool((lens == 1).all())
        if not (_singletons and tol == 0):
            return np.asarray(sorted(set(out)), dtype=int) + 1
        room = m.size - ends[-1]
        if len(starts) == 1 or room >= tol:
            return np.asarray(sorted(set(out)), dtype=int) + 1
    return None

def _random_cut_precision(P, near, n_pos, tol, G, gt_bounds=None, approx_flag=None):
    if P <= 0 or G <= 0 or n_pos <= 0:
        return 0.0
    if gt_bounds is None:
        if near is None:
            print('[stats] WARNING: `_random_cut_precision` got neither a `near` mask nor `gt_bounds`; the baseline is UNCOMPUTABLE for this call (returning 0.0) rather than being scored against an inflated reference set.  Pass `gt_bounds`.')
            return 0.0
        _runs = _mask_runs(near, n_pos, tol)
        if _runs is None:
            print('[stats] WARNING: `_random_cut_precision` got a `near` mask it cannot decompose into boundary neighbourhoods and no `gt_bounds`; the baseline is UNCOMPUTABLE for this call (returning 0.0) rather than being scored against an inflated reference set.  Pass `gt_bounds`.')
            return 0.0
        gt = _runs
    else:
        gt = np.asarray(gt_bounds).astype(int)
        gt = gt[(gt >= 1) & (gt <= n_pos)]
    if gt.size == 0:
        return 0.0
    gt = np.unique(gt)
    if P >= n_pos:
        return float(min(gt.size, P) / P)
    clusters, cur = ([], [int(gt[0])])
    for g in gt[1:]:
        g = int(g)
        if g - cur[-1] <= 2 * tol + 1:
            cur.append(g)
        else:
            clusters.append(cur)
            cur = [g]
    clusters.append(cur)
    expect_frac = 0.0
    for cl in clusters:
        lo, hi = (max(min(cl) - tol, 1), min(max(cl) + tol, n_pos))
        win = list(range(lo, hi + 1))
        expect_frac += _cluster_expected_matches(win, cl, tol, P, n_pos, approx_flag)
    return float(expect_frac / P)

def _log_ways_ratio(w, r, n_pos, P):
    a, b, c = (n_pos - w, P - r, P)
    if r < 0 or a < 0 or b < 0 or (c < 0) or (r > w) or (b > a):
        return float('-inf')
    return _lgamma_comb(w, r) + _lgamma_comb(a, b) - _lgamma_comb(n_pos, c)

def _lgamma_comb(n, k):
    if k < 0 or k > n:
        return float('-inf')
    if k == 0 or k == n:
        return 0.0
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
_AMP_WORK_CAP = (1 << 16) - 1
_AMP_FALLBACK_DRAWS = 8192

def _cluster_expected_matches(win, cl, tol, P, n_pos, approx_flag=None):
    w = len(win)
    if w <= 0 or P <= 0:
        return 0.0
    cl_t = tuple((int(b) for b in cl))
    A = [int(p) for p in win if any((abs(int(p) - b) <= tol for b in cl_t))]
    a = len(A)
    if a == 0:
        return 0.0
    rmax = min(P, a)
    _cl_arr = np.asarray(cl_t)
    _ord = np.argsort(_cl_arr, kind='stable')
    _pre = (_cl_arr[_ord], _ord)
    T = [0] * (rmax + 1)
    _comp_T = _component_match_tables(A, cl_t, tol, rmax, _pre)
    a_max = max((a_c for _tbl, a_c in _comp_T.values()), default=0)
    any_sampled = any((tbl == [] for tbl, _a in _comp_T.values()))
    if not any_sampled:
        T = _convolve_component_T(_comp_T, rmax)
    else:
        _work = _subset_work(a_max, rmax)
        if approx_flag is not None:
            approx_flag.append((a_max, _AMP_WORK_CAP, _work))
        print(f'[stats] WARNING: `_random_cut_precision` is APPROXIMATING: a single candidate component holds {a_max} positions and needs {_work} subset scorings, past the enumeration budget of {_AMP_WORK_CAP}, so `bnd_rand` (and hence `bnd_excess`) for this sample is a Monte-Carlo estimate ({_AMP_FALLBACK_DRAWS} draws), NOT the exact probability the other panels report.  Record it as `bnd_rand_exact: False` and do not pool it with the exact readings.')
        seed = (int(n_pos) * 1000003 + int(P) * 10007 + int(tol) * 101 + int(a) * 17 + cl_t[0]) % 2 ** 32
        rng = np.random.default_rng(seed)
        wts = []
        for r in range(1, rmax + 1):
            outside = P - r
            if outside < 0 or outside > n_pos - a:
                continue
            lw = _log_ways_ratio(a, r, n_pos, P)
            if lw == float('-inf'):
                continue
            wts.append((r, lw))
        if not wts:
            return 0.0
        nb_r = len(wts)
        order = sorted(range(nb_r), key=lambda i: -wts[i][1])
        if nb_r > _AMP_FALLBACK_DRAWS:
            # More candidate sizes than the draw budget: keep the top-weight
            # sizes, one draw each, so the actual number of draws never
            # exceeds the declared _AMP_FALLBACK_DRAWS budget.
            _keep = set(order[:_AMP_FALLBACK_DRAWS])
            budget = [1 if i in _keep else 0 for i in range(nb_r)]
        else:
            per = max(1, _AMP_FALLBACK_DRAWS // nb_r)
            budget = [per] * nb_r
            for i in range(_AMP_FALLBACK_DRAWS - per * nb_r):
                budget[order[i % nb_r]] += 1
        total = 0.0
        for i, (r, lw) in enumerate(wts):
            n_draw = budget[i]
            if n_draw <= 0:
                continue
            subs = [_sample_r_subset(rng, A, r) for _ in range(n_draw)]
            s = _score_subsets_components(subs, _comp_T, cl_t, tol, _pre)
            total += math.exp(lw) * (s / n_draw)
        return float(total)
    total = 0.0
    for r in range(0, rmax + 1):
        outside = P - r
        if outside < 0 or outside > n_pos - a:
            continue
        if not r:
            continue
        if not (comb(a, r) and comb(n_pos - a, outside)):
            continue
        mean_r = T[r] / comb(a, r)
        if mean_r == 0.0:
            continue
        total += math.exp(_log_ways_ratio(a, r, n_pos, P)) * mean_r
    return float(total)

def _match_components(A, cl_t, tol):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = (root, parent[x])
        return root

    def union(x, y):
        rx, ry = (find(x), find(y))
        if rx != ry:
            parent[rx] = ry
    if A:
        A_arr = np.asarray(A)
        for gi, b in enumerate(cl_t):
            lo = int(np.searchsorted(A_arr, b - tol, side='left'))
            hi = int(np.searchsorted(A_arr, b + tol, side='right'))
            for pi in range(lo, hi):
                union(('c', pi), ('g', gi))
    comps = {}
    for pi, p in enumerate(A):
        comps.setdefault(find(('c', pi)), ([], []))[0].append(int(p))
    for gi, b in enumerate(cl_t):
        comps.setdefault(find(('g', gi)), ([], []))[1].append(int(b))
    out = []
    for key in sorted(comps, key=repr):
        pos, bnd = comps[key]
        if pos:
            out.append((pos, bnd))
    return out

def _sorted_pre(b_t):
    if not b_t:
        z = np.zeros(0, dtype=int)
        return (z, z)
    arr = np.asarray(b_t)
    o = np.argsort(arr, kind='stable')
    return (arr[o], o)

def _subset_work(w, rmax):
    if w <= 0 or rmax <= 0:
        return 0
    r_hi = min(rmax, w)
    total = 0
    c = 1
    for r in range(0, r_hi + 1):
        if r:
            c = c * (w - r + 1) // r
        if r:
            total += c
    return total

def _component_match_tables(A, cl_t, tol, rmax, _pre):
    tables = {}
    for pos, bnd in _match_components(A, cl_t, tol):
        r_c = min(rmax, len(pos))
        b_t = tuple(bnd)
        c_pre = _sorted_pre(b_t)
        if _subset_work(len(pos), r_c) <= _AMP_WORK_CAP:
            tbl = [0] * (r_c + 1)
            for rr in range(1, r_c + 1):
                subs = list(_subsets(pos, rr))
                if subs:
                    tbl[rr] = int(sum(_greedy_match_count_batch(subs, c_pre[0], c_pre[1], tol)))
            tables[tuple(pos), b_t] = (tbl, len(pos))
        else:
            tables[tuple(pos), b_t] = ([], len(pos))
    return tables

def _convolve_component_T(tables, rmax):
    items = [(tbl, a_c) for tbl, a_c in tables.values() if tbl]
    if not items:
        return [0] * (rmax + 1)
    choice = [0.0] * (rmax + 1)
    choice[0] = 1.0
    wsum = [0.0] * (rmax + 1)
    for tbl, a_c in items:
        rc_max = min(rmax, a_c)
        term = [0.0] * (rmax + 1)
        mean = [0.0] * (rmax + 1)
        for r in range(0, rc_max + 1):
            term[r] = float(comb(a_c, r))
            if r:
                mean[r] = tbl[r] / comb(a_c, r)
        new_choice = [0.0] * (rmax + 1)
        new_wsum = [0.0] * (rmax + 1)
        for i in range(rmax + 1):
            if choice[i] == 0.0 and wsum[i] == 0.0:
                continue
            for r in range(0, rc_max + 1):
                if i + r > rmax or term[r] == 0.0:
                    continue
                new_choice[i + r] += choice[i] * term[r]
                new_wsum[i + r] += wsum[i] * term[r] + choice[i] * term[r] * mean[r]
        choice, wsum = (new_choice, new_wsum)
    return [int(round(wsum[r])) for r in range(rmax + 1)]

def _sample_r_subset(rng, A, r):
    return tuple(np.sort(rng.choice(A, size=r, replace=False)))

def _score_subsets_components(subs, tables, cl_t, tol, _pre):
    _owner_cache = getattr(_score_subsets_components, '_owner_cache', None)
    if _owner_cache is None:
        _owner_cache = _score_subsets_components._owner_cache = {}
    if len(_owner_cache) >= 64 and id(tables) not in _owner_cache:
        _owner_cache.clear()
    cached = _owner_cache.get(id(tables))
    if cached is None:
        owner = {}
        for pos, bnd in tables:
            b_t = tuple(bnd)
            c_pre = _sorted_pre(b_t)
            for p in pos:
                owner[p] = (b_t, c_pre)
        cached = (owner, tables)
        _owner_cache[id(tables)] = cached
    owner = cached[0]
    by_comp = {}
    for si, sub in enumerate(subs):
        for p in sub:
            hit = owner.get(p)
            if hit is None:
                continue
            b_t, c_pre = hit
            per_sub = by_comp.get(b_t)
            if per_sub is None:
                per_sub = by_comp[b_t] = (c_pre, {})
            pts = per_sub[1].get(si)
            if pts is None:
                per_sub[1][si] = [p]
            else:
                pts.append(p)
    total = 0
    for _b_t, (c_pre, per_sub) in by_comp.items():
        sids = sorted(per_sub)
        pieces = [tuple(sorted(per_sub[si])) for si in sids]
        for v in _greedy_match_count_batch(pieces, c_pre[0], c_pre[1], tol):
            total += v
    return total

def _subsets(items, r):
    return itertools.combinations(items, r)

def _greedy_match_counts(pred, gt, tol, _pre=None):
    pred = np.sort(np.asarray(pred))
    P, G = (len(pred), len(gt))
    if P == 0 or G == 0:
        return (0, 0)
    if _pre is not None:
        gt_sorted, order = _pre
    else:
        gt_arr = np.asarray(gt)
        order = np.argsort(gt_arr, kind='stable')
        gt_sorted = gt_arr[order]
    lo_all = np.searchsorted(gt_sorted, np.asarray(pred) - tol, side='left')
    hi_all = np.searchsorted(gt_sorted, np.asarray(pred) + tol, side='right')
    cand = []
    for pi in range(P):
        p = pred[pi]
        for j in range(int(lo_all[pi]), int(hi_all[pi])):
            gi = int(order[j])
            cand.append((abs(p - gt[gi]), pi, gi))
    cand.sort()
    used_p, used_g = (set(), set())
    for _, pi, gi in cand:
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
    return (len(used_p), len(used_g))

def _greedy_match_count_batch(subsets, gt_sorted, order, tol):
    subs = [tuple(s) for s in subsets]
    n_sub = len(subs)
    if n_sub == 0:
        return []
    counts = np.zeros(n_sub, dtype=np.int64)
    sizes = np.fromiter((len(s) for s in subs), dtype=np.int64, count=n_sub)
    if sizes.sum() == 0:
        return counts.tolist()
    flat = np.fromiter((int(p) for s in subs for p in s), dtype=np.int64, count=int(sizes.sum()))
    cut_rank = np.concatenate([np.arange(int(k), dtype=np.int64) for k in sizes]) if n_sub else np.zeros(0, dtype=np.int64)
    owner = np.repeat(np.arange(n_sub, dtype=np.int64), sizes)
    lo = np.searchsorted(gt_sorted, flat - tol, side='left')
    hi = np.searchsorted(gt_sorted, flat + tol, side='right')
    cnt = (hi - lo).astype(np.int64)
    total = int(cnt.sum())
    if total:
        row_owner = np.repeat(owner, cnt)
        starts = np.repeat(lo, cnt)
        offs = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        bnd_idx = starts + offs
        gi = order[bnd_idx]
        pi_local = np.repeat(flat, cnt)
        pi_rank = np.repeat(cut_rank, cnt)
        dist = np.abs(pi_local - gt_sorted[bnd_idx])
        order_rows = np.lexsort((gi, pi_rank, dist, row_owner))
        used_p, used_g = (set(), set())
        prev_sid = -1
        for idx in order_rows:
            sid = int(row_owner[idx])
            if sid != prev_sid:
                if prev_sid >= 0:
                    counts[prev_sid] = len(used_p)
                used_p, used_g = (set(), set())
                prev_sid = sid
            p_i = int(pi_rank[idx])
            g_i = int(gi[idx])
            if p_i not in used_p and g_i not in used_g:
                used_p.add(p_i)
                used_g.add(g_i)
        if prev_sid >= 0:
            counts[prev_sid] = len(used_p)
    return counts.tolist()

def compression_report(model, val_batch, device, n_sample=64, val_bnd=None, bnd_tol=1):
    was_training = model.training
    model.eval()
    enable_block_stats(model, True)
    need_reg_layers = [blk.attn for blk in model.blocks if hasattr(blk.attn, 'need_reg')]
    need_reg_prior = [_a.need_reg for _a in need_reg_layers]
    for _a in need_reg_layers:
        _a.need_reg = False
    try:
        return _compression_report_impl(model, val_batch, device, n_sample, val_bnd, bnd_tol)
    finally:
        for _a, _prev in zip(need_reg_layers, need_reg_prior):
            _a.need_reg = _prev
        enable_block_stats(model, False)
        model.train(was_training)

@torch.inference_mode()
def _compression_report_impl(model, val_batch, device, n_sample, val_bnd, bnd_tol):
    report = {}
    seq_len = val_batch.shape[1]
    if int(n_sample) < 1:
        raise ValueError(f'compression_report: n_sample must be >= 1, got {n_sample!r}; the report is an average over the sequences it scores, so there is nothing to measure with no sequence scored')
    n_used = min(n_sample, val_batch.shape[0])
    # inference_mode (the decorator) already covers the forward pass.
    for _r0 in range(0, n_used, 16):
        # as_tensor accepts numpy arrays AND torch tensors (eval_ppl already
        # uses it).
        ids = torch.as_tensor(np.asarray(val_batch[_r0:min(_r0 + 16, n_used)])).to(device)
        _ = model(ids, logits_tail=1)
        del ids
    # every dynamic layer's delta_logit (same scalars, same floats).
    _dl_layers = [(li, blk.attn.delta_logit) for li, blk in enumerate(model.blocks)
                  if blk.attn.cfg.dynamic and getattr(blk.attn, 'delta_logit', None) is not None]
    _dl_vals = {}
    if _dl_layers:
        _dl_vals = {li: float(v) for (li, _), v in zip(_dl_layers, torch.stack([p.detach() for _, p in _dl_layers]).tolist())}
    for li, blk in enumerate(model.blocks):
        attn = blk.attn
        st = attn._stats
        kind = attn.cfg.kind
        if st is None or kind == 'full' or len(st) == 0:
            continue
        lens = torch.cat([t if torch.is_tensor(t) else torch.as_tensor(t) for t in st]).float()
        n_blocks = sum((t.numel() for t in st)) / len(st)
        _sc = torch.stack([lens.mean(), lens.std() if lens.numel() > 1 else torch.zeros((), device=lens.device, dtype=lens.dtype), lens.min().float(), lens.max().float(), (lens <= attn.cfg.min_block).float().mean(), (lens >= attn.cfg.max_block).float().mean()])
        _vals = _sc.tolist()
        entry = {'blocks': n_blocks, 'avg_len': seq_len / max(n_blocks, 1.0), 'len_mean': _vals[0], 'len_std': _vals[1], 'len_min': int(_vals[2]), 'len_max': int(_vals[3]), 'frac_at_min': _vals[4], 'frac_at_max': _vals[5]}
        del _sc, _vals
        if attn.cfg.dynamic and getattr(attn, 'delta_logit', None) is not None:
            entry['delta'] = _dl_vals[li]
        if val_bnd is not None and attn._cuts:
            n_seq_ref = min(len(attn._cuts), val_bnd.shape[0])
            if n_seq_ref < len(attn._cuts):
                print(f'[compression_report] WARNING: {len(attn._cuts)} sequences of cuts but only {val_bnd.shape[0]} reference masks — scoring the first {n_seq_ref} only')
            ps, rs, fs, rps = ([], [], [], [])
            _cuts = attn._cuts
            _prov = {}
            # The integers transferred are exactly the per-sequence values
            # (index transport, design notes §6.1), segmented on the host.
            _tcut_ix = [si for si in range(n_seq_ref) if torch.is_tensor(_cuts[si])]
            _tcut_host = {}
            if _tcut_ix:
                _tlens = [int(_cuts[si].numel()) for si in _tcut_ix]
                _tflat = torch.cat([_cuts[si].reshape(-1) for si in _tcut_ix]).cpu().numpy()
                _toff = 0
                for si, _tln in zip(_tcut_ix, _tlens):
                    _tcut_host[si] = _tflat[_toff:_toff + _tln]
                    _toff += _tln
                del _tflat
            for si in range(n_seq_ref):
                c = _tcut_host.get(si) if si in _tcut_host else _cuts[si]
                p, r_, f_, rp = boundary_alignment(c, val_bnd[si], seq_len, tol=bnd_tol, provenance=_prov)
                ps.append(p)
                rs.append(r_)
                fs.append(f_)
                rps.append(rp)
            if n_seq_ref:
                entry['bnd_prec'] = float(np.mean(np.asarray(ps)))
                entry['bnd_rec'] = float(np.mean(np.asarray(rs)))
                entry['bnd_f1'] = float(np.mean(np.asarray(fs)))
                entry['bnd_rand'] = float(np.mean(np.asarray(rps)))
                entry['bnd_excess'] = entry['bnd_prec'] - entry['bnd_rand']
                entry['bnd_rand_exact'] = bool(_prov.get('bnd_rand_exact', True))
                if not entry['bnd_rand_exact']:
                    print(f'[compression_report] WARNING: `bnd_rand` for L{li}_{kind} is a SAMPLED estimate, not the exact expectation, so `bnd_excess` carries Monte-Carlo error — recorded as `bnd_rand_exact: False` and it must not be pooled with the exact panels.')
            else:
                print('[compression_report] WARNING: no reference sequence to score (n_seq_ref=0) — boundary metrics OMITTED rather than published as nan')
            entry['bnd_n_seq'] = n_seq_ref
        tag = 'dyn' if attn.cfg.dynamic else 'fix'
        report[f'L{li}_{kind}_{tag}'] = entry
    return report

def is_no_decay(name):
    if any((k in name for k in ('delta_logit', 'B_pos', 'W_aZ', 'W_bZ', 'sink', 'tok.weight', 'pos.weight', 'fuse_conv', 'q_norm', 'kv_norm'))):
        return True
    return name == 'g' or name.endswith('.g')

def is_delta_param(name):
    return 'delta_logit' in name

# ---- 受控变量描述符：优化器 / 调度 / 裁剪 / 数据增强 ----
# 对比实验要求「除待验证模块外，优化器、学习率策略、训练轮数、数据增强等设置
# 必须完全一致」。这些设置此前是 train_variant 内的硬编码常量，因此对所有臂
# 天然一致，但 run_cfg 指纹并不显式记录它们——一旦扩规模时为某条臂单独改优化
# 器/调度（或将来引入数据增强），两条记录仍可能通过 `pair_reason` 的同
# run_cfg 门槛而被静默配对。这里把它们提为模块常量，并把同一组常量导出成
# `training_knobs()` 描述符，随每条记录落盘；`pair_reason` 对两条都带该字段的
# 记录额外要求逐字段相等（缺字段的历史记录不受影响），审计则把该描述符打印
# 出来「证明」两臂共享优化器与调度，而不是只靠断言。
OPT_ADAMW_BETAS = (0.9, 0.999)
OPT_ADAMW_EPS = 1e-08
GRAD_CLIP_NORM = 1.0
LR_FLOOR_MULT = 0.1
LR_COSINE_AMP = 0.45

def training_knobs():
    """Canonical, JSON-safe descriptor of the optimisation / schedule /
    gradient-clipping / augmentation settings shared by every arm of a
    controlled panel.  Pure function of the module constants above, so it is
    identical across all arms unless a future scaled run deliberately diverges
    one (in which case `pair_reason` refuses to pair the records)."""
    return {
        'optimizer': 'AdamW',
        'foreach': True,
        'betas': list(OPT_ADAMW_BETAS),
        'eps': OPT_ADAMW_EPS,
        'param_groups': 'decay (wd) | no-decay (wd=0) | delta_logit (wd=0, lr x delta_lr_mult)',
        'grad_clip_norm': GRAD_CLIP_NORM,
        'lr_schedule': 'linear_warmup_then_cosine',
        'lr_floor_mult': LR_FLOOR_MULT,
        'lr_cosine_amp': LR_COSINE_AMP,
        'augmentation': 'none',
    }

def train_variant(variant, train_ids, val_batch, vocab, *, seed=0, d=256, n_layers=6, n_heads=8, d_head=32, seq_len=512, batch_size=12, steps=1500, lr=0.0003, weight_decay=0.1, warmup=50, comp_lambda=0.05, delta_lr_mult=10.0, eval_every=0, eval_subset=128, val_bnd=None, device=DEVICE, log_every=100, mlp_ratio=4, deadline_ts=None, return_model=False):
    if steps < 1:
        raise ValueError(f'train_variant needs steps >= 1, got steps={steps}')
    set_seed(seed)
    _rng_fp = rng_fingerprint()
    _attb_bump_epoch()
    if variant in PARAM_MATCHED | PARAM_MATCHED_V7 and float(mlp_ratio) == 4.0:
        raise ValueError(f'{variant} is a PARAM-MATCHED baseline but was given the default mlp_ratio=4; its MLP must be widened to match its sparse reference arm, or the comparison is not parameter-controlled. Pass mlp_ratio=variant_mlp_ratio(variant, vocab, d=..., n_layers=..., n_heads=..., d_head=..., seq_len=...).')
    cfgs = make_layer_cfgs(n_layers, variant)
    model = SmallGPT(vocab, d, n_layers, n_heads, d_head, seq_len, cfgs, mlp_ratio=mlp_ratio).to(device)
    n_param = count_params(model)
    print(f'\n[{variant} seed={seed}] params={n_param / 1000000.0:.2f}M  mlp_ratio={mlp_ratio:.2f}  layers={[c.kind for c in cfgs]}')
    decay_params = [p for n_, p in model.named_parameters() if p.requires_grad and (not is_no_decay(n_)) and (not is_delta_param(n_))]
    ndecay_params = [p for n_, p in model.named_parameters() if p.requires_grad and is_no_decay(n_) and (not is_delta_param(n_))]
    delta_params = [p for n_, p in model.named_parameters() if p.requires_grad and is_delta_param(n_)]
    # foreach=True pins the multi-tensor AdamW implementation.  Foreach applies
    # the same per-element IEEE op chain as the single-tensor form (lerp /
    # addcmul / sqrt are elementwise, so every element sees the identical
    # rounding sequence), so the update is bit-identical to the single-tensor
    # form, and on CUDA it pins the already-default foreach path.
    opt = torch.optim.AdamW([{'params': decay_params, 'weight_decay': weight_decay, 'lr_scale': 1.0}, {'params': ndecay_params, 'weight_decay': 0.0, 'lr_scale': 1.0}, {'params': delta_params, 'weight_decay': 0.0, 'lr_scale': delta_lr_mult}], lr=lr, betas=OPT_ADAMW_BETAS, eps=OPT_ADAMW_EPS, foreach=True)
    if delta_params:
        print(f'    delta_logit params: {len(delta_params)} (lr x{delta_lr_mult:g})')

    def lr_at(step):
        if warmup > 0 and step < warmup:
            return lr * (step + 1) / warmup
        t = (step - warmup) / max(steps - warmup, 1)
        return lr * (LR_FLOOR_MULT + LR_COSINE_AMP * (1.0 + math.cos(math.pi * t)))
    # lr_scale is only ever set at optimizer construction (never mutated), so
    # the per-group multiplier list is captured once here.
    _lr_scales = [g.get('lr_scale', 1.0) for g in opt.param_groups]
    bpe = batch_iter(train_ids, seq_len, batch_size, device, seed=seed)
    _eval_batch = val_batch[:eval_subset] if val_batch is not None else None
    t0 = time.time()
    losses = []
    _lbuf = []
    # deque with a fixed capacity; the window contents are identical.
    _tail = collections.deque(maxlen=50)
    ppl_hist = []
    delta_trace = {}
    model.train()
    truncated = False
    # Cyclic GC is disabled for the step loop; collections still run at a
    # fixed cadence and the original GC state is restored on every exit path.
    _gc_relief = gc.isenabled()
    if _gc_relief:
        gc.disable()
    try:
        for step in range(steps):
            if deadline_ts is not None and time.time() > deadline_ts:
                print(f'  [cap] HARD cap reached before step {step} - truncating; this run is NOT recorded (raise RUN_CAP and re-run to retry it)')
                truncated = True
                break
            base = lr_at(step)
            for g, _ls in zip(opt.param_groups, _lr_scales):
                g['lr'] = base * _ls
            x, y = next(bpe)
            ce = model.ce_loss(x, y, vocab)
            del x, y
            loss = ce + comp_lambda * model.comp_reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            _ce_d = ce.detach()
            _lbuf.append(_ce_d)
            _tail.append(_ce_d)
            if len(_lbuf) >= 512:
                losses.extend(torch.stack(_lbuf).tolist())
                _lbuf.clear()
            if (step + 1) % 1024 == 0:
                gc.collect()
            if eval_every and val_batch is not None and ((step + 1) % eval_every == 0 or step == steps - 1):
                sub_ppl = eval_ppl(model, _eval_batch, device)
                ppl_hist.append([step + 1, float(sub_ppl)])
            if log_every and (step % log_every == 0 or step == steps - 1):
                if _lbuf:
                    losses.extend(torch.stack(_lbuf).tolist())
                    _lbuf.clear()
                _lv = float(losses[-1])
                _dls = [(li, getattr(blk.attn, 'delta_logit', None)) for li, blk in enumerate(model.blocks)]
                _dls = [(li, dl) for li, dl in _dls if dl is not None]
                if _dls:
                    # (no_grad: 纯读取 Parameter 的值做日志，不需要 autograd 节点)
                    with torch.no_grad():
                        for li, _v in zip([li for li, _ in _dls], torch.stack([dl for _, dl in _dls]).tolist()):
                            delta_trace.setdefault(f'L{li}', []).append([step, float(_v)])
                print(f'  step {step:4d}  loss {_lv:.4f}  lr {opt.param_groups[0]['lr']:.2e}  ({(time.time() - t0) / max(step + 1, 1) * 1000:.0f}ms/step)')
    finally:
        if _gc_relief:
            gc.enable()
    wall = time.time() - t0
    if truncated:
        del model, opt, bpe
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return {'variant': variant, 'seed': seed, 'cap_truncated': True, 'steps_done': step, 'train_time_s': wall}
    if deadline_ts is not None and time.time() > deadline_ts:
        print('  [cap] HARD cap reached before the final evaluation - truncating; this run is NOT recorded (raise RUN_CAP and re-run to retry it)')
        del model, opt, bpe
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return {'variant': variant, 'seed': seed, 'cap_truncated': True, 'steps_done': steps, 'train_time_s': wall}
    if val_batch is None:
        ppl, stats = (float('nan'), {})
    else:
        ppl = eval_ppl(model, val_batch, device)
        stats = compression_report(model, val_batch, device, val_bnd=val_bnd)
        print(f'[{variant} seed={seed}] val PPL = {ppl:.3f}')
        print(f'    block-length diagnostics: {json.dumps(stats, default=float)}')
    if _lbuf:
        losses.extend(torch.stack(_lbuf).tolist())
        _lbuf.clear()
    losses = [float(v) for v in losses]
    result = {'variant': variant, 'seed': seed, 'ppl': ppl, 'params': n_param, 'losses': losses, 'stats': stats, 'steps': steps, 'tokens_seen': steps * batch_size * seq_len, 'train_time_s': wall, 'final_loss_smoothed': float(torch.stack(list(_tail)).mean()) if _tail else float('nan'), 'ppl_history': ppl_hist, 'delta_trace': delta_trace, 'train_knobs': training_knobs(), 'rng_fp': _rng_fp, 'data_fp': _ids_fp(train_ids)}
    if return_model:
        result['_model'] = model
    else:
        del model
    del opt, bpe
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result

@torch.no_grad()
def _lm_batch_metrics(model, x, y, vocab, chunk=64):
    """Mean cross-entropy and next-token accuracy on one fixed batch."""
    was_training = model.training
    model.eval()
    try:
        h = model.trunk(x)
        total = 0.0
        correct = 0
        n = 0
        for s in range(0, x.shape[1], chunk):
            sl = slice(s, min(s + chunk, x.shape[1]))
            lg = model.head(h[:, sl, :])
            tg = y[:, sl]
            total += float(F.cross_entropy(lg.reshape(-1, vocab), tg.reshape(-1), reduction='sum'))
            correct += int((lg.argmax(-1) == tg).sum())
            n += int(tg.numel())
        return (total / max(n, 1)), (correct / max(n, 1))
    finally:
        model.train(was_training)

def overfit_sanity_check(variant='full', *, seed=0, steps=300, lr=3e-3, d=64, n_layers=2, n_heads=4, d_head=16, seq_len=32, batch_size=2, vocab=256, train_ids=None, device=None, target_loss=0.05, target_acc=0.99, mlp_ratio=None, grad_clip=1.0):
    """Small-batch overfit sanity check (necessary, not sufficient).

    Take ONE fixed batch (sampled once from `train_ids`, or synthetic ids when
    none is supplied), disable every regulariser (weight decay = 0, the
    compression penalty = 0; the model has no dropout), and optimise only that
    batch.  A correct training path drives the LM loss towards 0 and the
    next-token accuracy towards 100%.  Failure means a serious plumbing bug
    (loss not connected to the graph, gradients not applied, wrong labels,
    etc.) — do NOT launch a full run until it passes.

    Passing does NOT prove the data pipeline / augmentation / label mapping is
    correct, so keep the intermediate-result checks as well.

    Returns a JSON-safe report; `passed` is False when any criterion fails.
    """
    if int(d) != int(n_heads) * int(d_head):
        raise ValueError(f'overfit_sanity_check: d={d} must equal n_heads*d_head={n_heads}*{d_head}')
    device = DEVICE if device is None else device
    set_seed(seed)
    _rng_fp = rng_fingerprint()
    if train_ids is not None:
        arr = np.asarray(train_ids)
        if arr.ndim != 1:
            raise ValueError('overfit_sanity_check: train_ids must be one-dimensional')
        if len(arr) < seq_len + 1:
            raise ValueError(f'overfit_sanity_check: need at least seq_len+1={seq_len + 1} tokens, got {len(arr)}')
        _rng = np.random.default_rng(int(seed))
        _hi = len(arr) - seq_len
        starts = _rng.integers(0, _hi, size=int(batch_size))
        chunk = np.stack([arr[int(s):int(s) + seq_len + 1] for s in starts]).astype(np.int64, copy=False)
        _data_fp = _ids_fp(chunk.reshape(-1))
    else:
        _rng = np.random.default_rng(int(seed))
        chunk = _rng.integers(0, max(int(vocab), 1), size=(int(batch_size), seq_len + 1), dtype=np.int64)
        _data_fp = _ids_fp(chunk.reshape(-1))
    ids_full = torch.from_numpy(chunk).to(device)
    x, y = ids_full[:, :-1], ids_full[:, 1:]
    if mlp_ratio is None:
        mlp_ratio = variant_mlp_ratio(variant, vocab, d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, seq_len=seq_len, matched=PARAM_MATCHED | PARAM_MATCHED_V7)
    cfgs = make_layer_cfgs(n_layers, variant)
    model = SmallGPT(vocab, d, n_layers, n_heads, d_head, seq_len, cfgs, mlp_ratio=mlp_ratio).to(device)
    _p_before = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=OPT_ADAMW_BETAS, eps=OPT_ADAMW_EPS, weight_decay=0.0, foreach=True)
    initial_loss, initial_acc = _lm_batch_metrics(model, x, y, vocab)
    loss_curve = []
    acc_curve = []
    grad_norm_curve = []
    grad_norm = float('nan')
    grad_finite = True
    n_grad_none = 0
    model.train()
    for step in range(int(steps)):
        ce = model.ce_loss(x, y, vocab)
        opt.zero_grad(set_to_none=True)
        ce.backward()
        _gs = []
        _none = 0
        for _n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                _none += 1
                continue
            _gs.append(p.grad.detach())
        if _gs:
            grad_norm = math.sqrt(sum((float(g.double().pow(2).sum()) for g in _gs)))
            grad_finite = all(bool(torch.isfinite(g).all()) for g in _gs)
        else:
            grad_norm = 0.0
            grad_finite = False
        n_grad_none = _none
        grad_norm_curve.append(float(grad_norm))
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        opt.step()
        _l = float(ce.detach())
        loss_curve.append(_l)
        if (step + 1) % max(1, int(steps) // 20) == 0 or step == int(steps) - 1:
            _a = _lm_batch_metrics(model, x, y, vocab)[1]
            acc_curve.append([step + 1, _a])
        if _l != _l or not math.isfinite(_l):
            break
    final_loss, final_acc = _lm_batch_metrics(model, x, y, vocab)
    _p_after = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    params_changed = not bool(torch.equal(_p_before, _p_after))
    best_loss = min(loss_curve) if loss_curve else float('nan')
    # A correct optimizer must also move the loss DOWN, not merely end below
    # the target by luck.  Keep this in `passed`: a path that starts below the
    # target and never updates (lr=0) must still fail.
    loss_decreased = bool(math.isfinite(initial_loss) and math.isfinite(final_loss) and final_loss < initial_loss)
    passed = bool(math.isfinite(final_loss) and final_loss <= float(target_loss) and final_acc >= float(target_acc) and grad_finite and params_changed and math.isfinite(best_loss) and best_loss <= float(target_loss) and loss_decreased)
    report = {
        'kind': 'small_batch_overfit_sanity',
        'variant': variant,
        'seed': int(seed),
        'steps': int(steps),
        'lr': float(lr),
        'batch_shape': [int(batch_size), int(seq_len)],
        'vocab_size': int(vocab),
        'data_source': 'train_ids' if train_ids is not None else 'synthetic',
        'data_fp': _data_fp,
        'rng_fp': _rng_fp,
        'regularization': {'weight_decay': 0.0, 'comp_lambda': 0.0, 'dropout': 0.0, 'augmentation': 'none'},
        'initial_loss': float(initial_loss),
        'initial_accuracy': float(initial_acc),
        'final_loss': float(final_loss),
        'final_accuracy': float(final_acc),
        'best_loss': float(best_loss),
        'loss_decreased': bool(loss_decreased),
        'loss_reduction': float(initial_loss - final_loss) if math.isfinite(initial_loss) and math.isfinite(final_loss) else None,
        'loss_curve': [float(v) for v in loss_curve],
        'accuracy_curve': acc_curve,
        'grad_norm_curve': grad_norm_curve,
        'grad_norm_last': float(grad_norm),
        'grad_all_finite': bool(grad_finite),
        'params_without_grad': int(n_grad_none),
        'params_changed': bool(params_changed),
        'target_loss': float(target_loss),
        'target_accuracy': float(target_acc),
    }
    report['passed'] = passed
    if passed:
        report['diagnosis'] = 'ok'
    elif not (grad_finite and params_changed):
        report['diagnosis'] = 'no/finite gradients or parameters did not update'
    elif not loss_decreased:
        report['diagnosis'] = 'loss did not decrease on the fixed batch (optimizer/step or graph is not wired through)'
    else:
        report['diagnosis'] = 'loss did not converge on a single repeated batch within the step budget'
    del model, opt, ids_full, x, y
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return report

_OVERFIT_GUARD = {}
# Reports of the checks that actually ran (the boolean guard above is the
# resume cache; this is the evidence written into the run manifest).  Keyed
# like _OVERFIT_GUARD by (variant, source).
_OVERFIT_REPORTS = {}

def overfit_reports(*, source=None):
    """Copy of the small-batch reports gathered in this process, optionally
    filtered to source 'synthetic' or 'data'.  Pure read for tracking."""
    return {k: dict(v) for k, v in _OVERFIT_REPORTS.items() if source is None or k[1] == source}

def overfit_guard(variants, *, seed=0, steps=200, seq_len=64, batch_size=2, vocab=1024, train_ids=None, force=False, target_loss=0.05, target_acc=0.99):
    """Run `overfit_sanity_check` once per variant per process and hard-stop
    on failure.  Called automatically by `run()` before the first panel and by
    the length/checkpoint phases, so every training driver inherits the check.

    `CSA_SKIP_OVERFIT=1` is the explicit escape hatch (e.g. a debugging box
    with a deliberately broken batch); it prints a loud skip line instead.
    Passing is necessary, not sufficient."""
    if os.environ.get('CSA_SKIP_OVERFIT', '').strip().lower() in ('1', 'true', 'yes', 'on'):
        print('[overfit] guard SKIPPED (CSA_SKIP_OVERFIT is set) — this run has no small-batch safety check')
        return []
    src = 'data' if train_ids is not None else 'synthetic'
    # The data cache key includes the actual token fingerprint AND the vocab
    # size: two panels in one process can run on different corpora/word pieces,
    # and a pass on corpus A must never certify corpus B.  A synthetic-keyed
    # pass remains subsumed by ANY passed data check (same model plumbing).
    _data_key = None
    if src == 'data':
        _arr = np.asarray(train_ids)
        _data_key = (int(vocab), _ids_fp(_arr if _arr.ndim == 1 else np.ascontiguousarray(_arr).reshape(-1)))

    def _known(v):
        if src == 'data':
            return (v, 'data', *_data_key) in _OVERFIT_GUARD
        return (v, 'synthetic') in _OVERFIT_GUARD or any(k[0] == v and k[1] == 'data' for k in _OVERFIT_GUARD)
    todo = [v for v in variants if force or not _known(v)]
    if not todo:
        return []
    reports = []
    for v in todo:
        r = overfit_sanity_check(v, seed=seed, steps=steps, d=64, n_layers=2, n_heads=4, d_head=16, seq_len=seq_len, batch_size=batch_size, vocab=vocab, train_ids=train_ids, target_loss=target_loss, target_acc=target_acc)
        _key = (v, 'data', *_data_key) if src == 'data' else (v, 'synthetic')
        _OVERFIT_GUARD[_key] = bool(r['passed'])
        _OVERFIT_REPORTS[_key] = r
        reports.append(r)
        print(f"[overfit] {v:<24} loss {r['initial_loss']:.3f} -> {r['final_loss']:.4f}  acc {r['final_accuracy'] * 100:.1f}%  grad_finite={r['grad_all_finite']}  params_changed={r['params_changed']}  {'PASS' if r['passed'] else 'FAIL'}")
        if not r['passed']:
            raise RuntimeError(f'small-batch overfit sanity check FAILED for `{v}` ({r["diagnosis"]}) — do NOT start or continue this run; this points at a gradient/loss/label plumbing bug.  Inspect the returned report and set CSA_SKIP_OVERFIT=1 only if you deliberately accept the risk.')
    return reports

class TimeGuard:
    BASE = (256, 6, 512, 12)

    def __init__(self, cap):
        self.total_hours = float(cap['total_hours'])
        self.margin = float(cap.get('margin', 0.93))
        self.already = float(cap.get('already_hours', 0.0))
        self.state_path = cap.get('state_path', 'run_time_state.json')
        # 内核模式使用独立的状态文件：两档速率标定不可互相污染（同一
        # state 文件会把 eager 的 sps 标定用于内核单元）。
        # 幂等：cap_spec() 导出的就是已盖章路径，par.py 子进程按它重建
        # TimeGuard 时不得再加一次后缀。
        if _kernel_stamp() and not self.state_path.endswith('.k'):
            self.state_path = self.state_path + '.k'
        self.state = self._load_state()

    def _load_state(self):
        st = {'booked_seconds': 0.0, 'norm_sps': None, 'sps_by_class': {}, 'runs': 0}
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, encoding='utf-8') as f:
                    st.update(json.load(f))
            except Exception as e:
                print(f'[cap] FATAL: {self.state_path} exists but cannot be parsed ({type(e).__name__}: {e}).  Refusing to start with a silently RESET state (booked time would read as 0) — move it aside to start fresh.')
                raise
        return st

    def _save(self):
        atomic_write_json(self.state_path, self.state)

    def cap_spec(self):
        """Plain-dict spec of this guard, enough for a child process (par.py)
        to rebuild an equivalent TimeGuard against the same shared state file."""
        return dict(total_hours=self.total_hours, margin=self.margin, already_hours=self.already, state_path=self.state_path)

    def book(self, seconds, token=None):
        """Reserve (seconds > 0) or refund (seconds < 0) projected run time
        against the shared state WITHOUT touching the run counter or the
        calibration.  Parallel cell workers pre-book their estimate before
        training and refund it after recording the actual seconds, so the
        booked total always converges to the true sum while concurrently
        launching children still see the shared headroom shrink.

        With `token`, the reservation is recorded per-booking (with the
        booking process pid) so `reconcile()` can refund reservations whose
        process died before refunding (killed child / instance shutdown) —
        without this, a killed parallel cell starves every later phase with
        a phantom reservation."""
        try:
            import fcntl
        except ImportError:
            fcntl = None
        _lock_f = None
        if fcntl is not None:
            _lock_f = open(self.state_path + '.lock', 'a')
            fcntl.flock(_lock_f, fcntl.LOCK_EX)
        try:
            self.state = self._load_state()
            if token is None:
                self.state['booked_seconds'] += float(seconds)
                if 'bookings' in self.state:
                    self.state['booked_untracked'] = float(self.state.get('booked_untracked', 0.0)) + float(seconds)
            else:
                bookings = self.state.setdefault('bookings', {})
                if 'booked_untracked' not in self.state:
                    # First tokenised booking coexisting with a pre-bookings
                    # state (e.g. the parent's data-load record_run, or any
                    # serial phase that ran before this parallel one): every
                    # second booked so far has no per-token representation,
                    # so it ALL becomes the untracked remainder.  Defaulting
                    # it to 0 would silently erase those seconds from the
                    # booked total on the next refund's recompute — the cap
                    # would then never fill in parallel mode.
                    self.state['booked_untracked'] = max(0.0, float(self.state.get('booked_seconds', 0.0)) - sum(float(b.get('seconds', 0.0)) for b in bookings.values()))
                if seconds > 0:
                    bookings[str(token)] = {'seconds': float(seconds), 'pid': int(os.getpid()), 'ts': time.time()}
                else:
                    bookings.pop(str(token), None)
                legacy = float(self.state['booked_untracked'])
                self.state['booked_seconds'] = legacy + sum(float(b.get('seconds', 0.0)) for b in bookings.values())
            self._save()
        finally:
            if _lock_f is not None:
                try:
                    fcntl.flock(_lock_f, fcntl.LOCK_UN)
                finally:
                    _lock_f.close()

    @staticmethod
    def _pid_alive(pid):
        try:
            os.kill(int(pid), 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except (OSError, AttributeError):
            return True  # 无法判定（非 posix 等）时按存活处理，绝不误退

    def reconcile(self):
        """Refund per-token bookings whose recording process is no longer
        alive (crash/kill recovery).  Old state files without a `bookings`
        map are untouched: their `booked_seconds` is preserved as
        `booked_untracked` once any tokenised booking coexists."""
        try:
            import fcntl
        except ImportError:
            fcntl = None
        _lock_f = None
        if fcntl is not None:
            _lock_f = open(self.state_path + '.lock', 'a')
            fcntl.flock(_lock_f, fcntl.LOCK_EX)
        try:
            self.state = self._load_state()
            bookings = self.state.get('bookings') or {}
            if not bookings:
                return 0.0
            total = float(self.state.get('booked_seconds', 0.0))
            tracked = sum(float(b.get('seconds', 0.0)) for b in bookings.values())
            self.state['booked_untracked'] = max(0.0, total - tracked)
            refunded = 0.0
            now = time.time()
            for tok in sorted(bookings):
                b = bookings[tok]
                est_s = float(b.get('seconds', 0.0))
                dead = not self._pid_alive(b.get('pid', -1))
                # A live child can hold a booking for at most its dispatcher
                # watchdog horizon (par.py SIGKILLs run/warm/lenphase cells at
                # max(3*est, est+1h); every other kind is capped at 12 h).  A
                # booking older than that CANNOT have a live owner: the pid
                # reading is a reuse artifact (an instance reboot recycles
                # pids — this driver's own auto-shutdown makes that the common
                # case), and without an age fallback the phantom reservation
                # would starve every later cell behind the cap forever.
                horizon = max(4.0 * est_s, 13.0 * 3600.0)
                too_old = b.get('ts') is not None and (now - float(b['ts'])) > horizon
                if dead or too_old:
                    refunded += est_s
                    bookings.pop(tok, None)
                    _why = 'dead pid' if dead else f'phantom: age {(now - float(b["ts"])) / 3600:.1f} h exceeds the watchdog horizon {horizon / 3600:.1f} h'
                    print(f'[cap] reconcile: refunded {est_s / 3600:.2f} h from booking {tok} ({_why}, pid {b.get("pid")})')
            self.state['booked_seconds'] = self.state['booked_untracked'] + sum(float(b['seconds']) for b in bookings.values())
            self._save()
            return refunded
        finally:
            if _lock_f is not None:
                try:
                    fcntl.flock(_lock_f, fcntl.LOCK_UN)
                finally:
                    _lock_f.close()

    def spent_hours(self):
        self.state = self._load_state()
        return self.already + self.state['booked_seconds'] / 3600.0

    def remaining_hours(self):
        return self.total_hours - self.spent_hours()

    def cap_hours(self):
        return self.total_hours * self.margin

    def record_run(self, seconds, steps_done, d, n_layers, seq_len, batch_size, calib_seconds=None):
        try:
            import fcntl
        except ImportError:
            fcntl = None
        _lock_f = None
        if fcntl is not None:
            _lock_f = open(self.state_path + '.lock', 'a')
            fcntl.flock(_lock_f, fcntl.LOCK_EX)
        try:
            self.state = self._load_state()
            self.state['booked_seconds'] += float(seconds)
            if 'bookings' in self.state:
                # Keep the invariant booked_seconds == booked_untracked + Σ
                # bookings: recorded actual seconds have no per-token
                # representation, so they accrue to the untracked remainder.
                # Without this the next token refund's recompute silently
                # drops every recorded second from the booked total (the run
                # -time cap then meters nothing in parallel mode).
                self.state['booked_untracked'] = float(self.state.get('booked_untracked', 0.0)) + float(seconds)
            if steps_done and steps_done > 0:
                self.state['runs'] += 1
                if calib_seconds is None or not calib_seconds > 0:
                    calib_seconds = seconds
                f = self._wallclock_factor(d, n_layers, seq_len, batch_size)
                sps = calib_seconds / steps_done
                norm = sps / f
                prev = self.state.get('norm_sps')
                self.state['norm_sps'] = norm if prev is None else 0.7 * prev + 0.3 * norm
                by_class = self.state.setdefault('sps_by_class', {})
                key = str((int(d), int(n_layers), int(seq_len), int(batch_size)))
                cprev = by_class.get(key)
                by_class[key] = sps if cprev is None else 0.7 * cprev + 0.3 * sps
            self._save()
        finally:
            if _lock_f is not None:
                try:
                    fcntl.flock(_lock_f, fcntl.LOCK_UN)
                finally:
                    _lock_f.close()

    @classmethod
    def _depth_factor(cls, n_layers):
        return n_layers / 6.0

    @classmethod
    def _compute_factor(cls, d, n_layers, seq_len, batch_size):
        return (d / 256.0) ** 2 * cls._depth_factor(n_layers) * (seq_len / 512.0) ** 1.5 * (batch_size / 12.0)

    @classmethod
    def _calib_exponent(cls):
        return 1.75

    @classmethod
    def _wallclock_factor(cls, d, n_layers, seq_len, batch_size):
        f = cls._compute_factor(d, n_layers, seq_len, batch_size)
        df = cls._depth_factor(n_layers)
        return f / df * df ** cls._calib_exponent()

    def estimate_seconds(self, steps, d=256, n_layers=6, seq_len=512, batch_size=12):
        key = str((int(d), int(n_layers), int(seq_len), int(batch_size)))
        exact = (self.state.get('sps_by_class') or {}).get(key)
        if exact is not None:
            return steps * exact * 1.15
        norm = self.state.get('norm_sps')
        if norm is None:
            norm = 0.35
        f = self._wallclock_factor(d, n_layers, seq_len, batch_size)
        return steps * norm * f * 1.15

    def note_watchdog_kill(self, elapsed_seconds, steps, d=256, n_layers=6, seq_len=512, batch_size=12):
        """Lift the calibrated per-step floor after a dispatcher-watchdog kill
        (par.py).  A cell killed at `elapsed_seconds` demonstrably needs MORE
        than that for `steps` steps, so its true sps exceeds
        elapsed/steps — yet a killed cell never records a run, so without
        this hook the calibration (and hence the watchdog bound
        max(3*est, est+1h) derived from it) would stay wrong and the same
        healthy cell would be killed at the same wall-clock point on every
        resume (uncalibrated slow hardware, CPU workers).  Lifting the floor
        to the kill time itself makes the next bound ~3.45x larger, so the
        retry window grows geometrically until a healthy cell fits; a truly
        hung cell is still killed on every attempt, just later."""
        if not steps or int(steps) <= 0 or not float(elapsed_seconds) > 0:
            return
        floor = float(elapsed_seconds) / int(steps)
        try:
            import fcntl
        except ImportError:
            fcntl = None
        _lock_f = None
        if fcntl is not None:
            _lock_f = open(self.state_path + '.lock', 'a')
            fcntl.flock(_lock_f, fcntl.LOCK_EX)
        try:
            self.state = self._load_state()
            by_class = self.state.setdefault('sps_by_class', {})
            key = str((int(d), int(n_layers), int(seq_len), int(batch_size)))
            prev = by_class.get(key)
            if prev is None or floor > float(prev):
                by_class[key] = floor
            f = self._wallclock_factor(d, n_layers, seq_len, batch_size)
            norm_floor = floor / f
            norm = self.state.get('norm_sps')
            if norm is None or norm_floor > float(norm):
                self.state['norm_sps'] = norm_floor
            self._save()
        finally:
            if _lock_f is not None:
                try:
                    fcntl.flock(_lock_f, fcntl.LOCK_UN)
                finally:
                    _lock_f.close()
        print(f'[cap] watchdog kill noted: the pace floor for (d={d}, L={n_layers}, seq={seq_len}, bs={batch_size}) is now >= {floor:.3f} s/step so the next attempt gets a wider watchdog bound')

    def can_start(self, est_seconds):
        projected = self.spent_hours() + est_seconds / 3600.0
        return projected <= self.cap_hours()

    def deadline_ts(self):
        headroom = max(self.cap_hours() - self.spent_hours(), 0.0)
        return time.time() + headroom * 3600.0

    def report(self):
        self.state = self._load_state()
        print('\n==================== usage report ====================')
        print(f'  booked GPU time : {self.state['booked_seconds'] / 3600:.2f} h ({self.state.get('runs', 0)} runs)')
        for key, sps in sorted((self.state.get('sps_by_class') or {}).items()):
            print(f'  calibrated rate : {sps:.3f} s/step  (d, L, seq, batch = {key})')
        if self.state.get('norm_sps'):
            print(f'  normalised rate : {self.state['norm_sps']:.3f} s/step (base class d=256 / 6L / seq512 / batch12)')
        print('======================================================')
PARAM_MATCHED = {'full_matched', 'full_sw128_matched'}
PARAM_MATCHED_V7 = {'full_rope', 'full_sw128_matched_rope'}

_MLP_RATIO_MEMO = {}

def variant_mlp_ratio(variant, vocab, *, d=256, n_layers=6, n_heads=8, d_head=32, seq_len=512, matched=None, ref_variant='csa_dynamic'):
    if matched is None:
        matched = PARAM_MATCHED
    if variant not in matched:
        return 4.0
    # The ratio is a pure function of these arguments (the parameter count
    # of a freshly-initialised SmallGPT depends only on shapes; the build
    # runs under torch.random.fork_rng, so skipping a repeat build leaves
    # the caller's RNG stream untouched either way) — memoise it on the
    # full argument tuple.
    _mkey = (variant, int(vocab), int(d), int(n_layers), int(n_heads), int(d_head), int(seq_len), tuple(sorted((str(_x) for _x in matched))), str(ref_variant))
    _hit = _MLP_RATIO_MEMO.get(_mkey)
    if _hit is not None:
        return _hit

    def n_params(v, ratio):
        with torch.random.fork_rng():
            cfgs = make_layer_cfgs(n_layers, v)
            m = SmallGPT(vocab, d, n_layers, n_heads, d_head, seq_len, cfgs, mlp_ratio=ratio)
            n = count_params(m)
            del m
        return n
    p_full = n_params(variant, 4)
    p_ref = n_params(ref_variant, 4)
    ratio = 4 + (p_ref - p_full) / (2.0 * d * d * n_layers)
    if ratio == 4.0:
        ratio = math.nextafter(4.0, 0.0)
    if ratio <= 1.0:
        raise RuntimeError(f'[{variant}] param-matched mlp_ratio would be {ratio:.2f} (<= 1): the reference arm {ref_variant} already has fewer/equal params — param matching is impossible for this variant/config')
    print(f'[{variant}] matched mlp_ratio = {ratio:.2f} ({p_full / 1000000.0:.2f}M -> target {p_ref / 1000000.0:.2f}M params vs {ref_variant})')
    _MLP_RATIO_MEMO[_mkey] = ratio
    return ratio

def _tag_of(rec):
    if isinstance(rec, dict) and 'protocol' in rec and (rec['protocol'] is not None):
        return str(rec['protocol'])
    if isinstance(rec, dict) and rec.get('warm_steps') is not None:
        return f'w{rec['warm_steps']}'
    return ''

def legacy_warm_tags(summary):
    _legacy = {}
    for key, rec in summary.items():
        if not isinstance(rec, dict) or 'seed' not in rec or 'ppl' not in rec:
            continue
        if not ppl_is_usable(rec.get('ppl')) or rec.get('synthesized'):
            continue
        if 'protocol' in rec and rec['protocol'] is not None or rec.get('warm_steps') is not None:
            continue
        m = _WARM_KEY_RE.match(str(key))
        _var = rec.get('variant')
        if _var is None:
            _var = str(key).split('::', 1)[0]
        _legacy.setdefault(str(_var), set()).add(m.group(1) if m else None)
    return {v: frozenset(tags) for v, tags in _legacy.items() if None not in tags}

def record_tag(rec, key, legacy_ok):
    if 'protocol' in rec and rec['protocol'] is not None or rec.get('warm_steps') is not None:
        return _tag_of(rec)
    m = _WARM_KEY_RE.match(str(key))
    if not m:
        return ''
    ok = legacy_ok
    if isinstance(ok, dict):
        _var = rec.get('variant')
        if _var is None:
            _var = str(key).split('::', 1)[0]
        ok = ok.get(str(_var))
        if ok is None:
            return ''
    if m.group(1) in ok:
        return f'w{m.group(1)}'
    return ''

def _group_params(recs):
    for r in recs:
        v = r.get('params')
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
    return None

def aggregate(summary):

    def _protocol_tag(rec, key=None):
        if key is None:
            return _tag_of(rec)
        return record_tag(rec, key, _legacy_warm_ok)
    _legacy_warm_ok = legacy_warm_tags(summary)
    per_variant = {}
    for key, rec in summary.items():
        if not isinstance(rec, dict) or 'ppl' not in rec or 'seed' not in rec:
            continue
        if not ppl_is_usable(rec.get('ppl')) or rec.get('synthesized'):
            continue
        v = rec.get('variant')
        if v is None:
            v = str(key).split('::', 1)[0]
        per_variant.setdefault((v, _protocol_tag(rec, key)), []).append(rec)
    def _seed_key(s):
        # JSON round-trips can leave the same seed as 0 or "0"; the rest of the
        # readers normalise through `_as_seed_int`.  Keying dedup/pairing on the
        # raw value would count one seed twice and fabricate an extra n.
        n = _as_seed_int(s)
        return s if n is None else n
    _dups = {}
    for k, recs in per_variant.items():
        cnt = {}
        for r in recs:
            s = _seed_key(r['seed'])
            cnt[s] = cnt.get(s, 0) + 1
        bad = sorted((s for s, c in cnt.items() if c > 1))
        if bad:
            _dups[k] = bad
    if _dups:
        raise ValueError('aggregate: these (variant, protocol) groups hold more than one record for the same seed, so their PPL is ambiguous: ' + '; '.join((f'{v}/{t or '-'}->seeds {s}' for (v, t), s in sorted(_dups.items()))) + '. Remove the stale duplicates (or give the records distinct `protocol` tags) before aggregating.  A record whose value was never MEASURED (an `error` record, or a `synthesized` reconstruction) can carry no PPL and is IGNORED here — drop it from the summary instead of re-labelling it, because a protocol tag only silences this guard, it does not make the number real.')
    _multi = {}
    for (v, tag), recs in per_variant.items():
        _multi.setdefault(v, []).append(tag)
    _multi = {v: t for v, t in _multi.items() if len(t) > 1}
    if _multi:
        print('[aggregate] WARNING: these variants hold MULTIPLE protocol groups and are reported per-group, NOT pooled across groups (pooling would fake extra seeds): ' + ', '.join((f'{v}->{t}' for v, t in sorted(_multi.items()))))

    def _cfg_fp(rec):
        _p = rec.get('params')
        try:
            _p = int(_p) if _p is not None and (not isinstance(_p, bool)) else None
        except (TypeError, ValueError):
            _p = None
        return json.dumps([rec.get('run_cfg'), _p, [rec.get(k) for k in PAIR_STEP_KEYS]], sort_keys=True, default=str)

    def _steps_ok(recs):
        if len(recs) < 2:
            return True
        return len({_cfg_fp(r) for r in recs}) == 1

    def _by_seed(recs):
        out, dup = ({}, [])
        for r in recs:
            s = _seed_key(r['seed'])
            if s in out:
                dup.append(s)
            out[s] = r
        return (out, dup)

    def _paired(recs, base_recs):
        self_map, dup_a = _by_seed(recs)
        base_map, dup_b = _by_seed(base_recs)
        assert not dup_a and (not dup_b), (sorted(set(dup_a)), sorted(set(dup_b)))
        out, skipped = ([], [])
        for s, r in self_map.items():
            b = base_map.get(s)
            if b is None:
                continue
            reason = pair_reason(r, b)
            if reason is not None:
                skipped.append((s, reason))
                continue
            out.append(r['ppl'] - b['ppl'])
        return (out, skipped)
    agg = {}
    _split_groups = []
    _group_records = {}

    def _agg_key(v, tag, fp_idx=None):
        key = v if not tag else f'{v}#{tag}'
        return key if fp_idx is None else f'{key}@cfg{fp_idx}'
    for (v, tag), recs in per_variant.items():
        if _steps_ok(recs):
            _split_groups.append(((v, tag, None), recs))
            _group_records[v, tag, None] = recs
            continue
        _subs = {}
        for r in recs:
            _subs.setdefault(_cfg_fp(r), []).append(r)
        # the `@cfg` suffix is a digest of the config fingerprint itself, so a
        # resume pass cannot reshuffle which group a published key refers to
        _ordered = sorted(_subs.items(), key=lambda kv: kv[0])
        print(f'[aggregate] WARNING: `{_agg_key(v, tag)}` holds records from {len(_ordered)} different run_cfg/load/parameter-count configurations — pooling them would average incompatible runs (or two different MODEL WIDTHS, which are not two seeds of one model) and fake extra seeds.  Reported per configuration instead.')
        for _fp, _rs in _ordered:
            _i = hashlib.sha1(_fp.encode('utf-8')).hexdigest()[:8]
            _split_groups.append(((v, tag, _i), _rs))
            _group_records[v, tag, _i] = _rs
    for (_v, _tag, _fp_idx), recs in _split_groups:
        v, tag = (_v, _tag)
        key = _agg_key(v, tag, _fp_idx)
        def _seed_sort_key(r):
            s = r.get('seed')
            if isinstance(s, (int, float)) and (not isinstance(s, bool)) and math.isfinite(float(s)):
                return (0, float(s), '')
            return (1, 0.0, str(s))
        _recs_sorted = sorted(recs, key=_seed_sort_key)
        recs = _recs_sorted
        ppls = np.array([r['ppl'] for r in recs], dtype=float)
        entry = {'variant': v, 'protocol': tag, 'n_seeds': len(recs), 'ppl_mean': float(ppls.mean()), 'ppl_std': float(ppls.std(ddof=1)) if len(ppls) > 1 else 0.0, 'ppls': [float(p) for p in ppls], 'seeds': [r.get('seed') for r in recs], 'params': _group_params(recs), 'tokens_seen': recs[0].get('tokens_seen')}
        if _fp_idx is not None:
            entry['run_cfg_class'] = _fp_idx
            entry['run_cfg'] = recs[0].get('run_cfg')
        dyn_lens, all_lens, deltas = ([], [], [])
        bf1, bex = ([], [])

        def _cell(layer, k):
            v = layer.get(k)
            return float(v) if isinstance(v, (int, float)) and (not isinstance(v, bool)) and math.isfinite(v) else None
        for r in recs:
            r_f1, r_ex, r_dl = ([], [], [])
            for lk, lv in r.get('stats', {}).items():
                if not isinstance(lv, dict):
                    continue
                lm = _cell(lv, 'len_mean')
                if lm is None:
                    continue
                all_lens.append(lm)
                if '_dyn' in lk:
                    dyn_lens.append(lm)
                    if 'bnd_f1' in lv:
                        _f1 = _cell(lv, 'bnd_f1')
                        _ex = _cell(lv, 'bnd_excess') if lv.get('bnd_rand_exact') is not False else None
                        if _f1 is not None:
                            r_f1.append(_f1)
                        if _ex is not None:
                            r_ex.append(_ex)
                dl = _cell(lv, 'delta')
                if dl is not None:
                    r_dl.append(dl)
            if r_f1:
                bf1.append(float(np.mean(r_f1)))
            if r_ex:
                bex.append(float(np.mean(r_ex)))
            if r_dl:
                # Average per record FIRST (as bf1/bex do), so records with more
                # dynamic layers do not get extra weight in the cross-seed mean.
                deltas.append(float(np.mean(r_dl)))
        if dyn_lens:
            entry['avg_dyn_block_len'] = float(np.mean(dyn_lens))
        elif all_lens:
            entry['avg_block_len'] = float(np.mean(all_lens))
        if deltas:
            entry['delta_logit_mean'] = float(np.mean(deltas))
        if bf1:
            entry['boundary_f1_dyn'] = float(np.mean(bf1))
            entry['boundary_f1_dyn_std'] = float(np.std(bf1, ddof=1)) if len(bf1) > 1 else 0.0
        if bex:
            entry['boundary_excess_dyn'] = float(np.mean(bex))
        hists = [r['ppl_history'] for r in recs if r.get('ppl_history')]
        if hists:
            per_step = {}
            for _h in hists:
                for _pt in _h:
                    try:
                        _s = int(_pt[0])
                        _v = float(_pt[1])
                    except (TypeError, ValueError, IndexError):
                        continue
                    per_step.setdefault(_s, []).append(_v)
            def _grid(_hist):
                _out = []
                for _p in _hist:
                    try:
                        _out.append(int(_p[0]))
                    except (TypeError, ValueError, IndexError):
                        continue
                return _out
            _ref_steps = _grid(hists[0])
            _off_grid = []
            for _i, _h in enumerate(hists[1:], start=1):
                _hs = _grid(_h)
                if _hs != _ref_steps:
                    _off_grid.append(_i)
            if _off_grid:
                print(f'[aggregate] {key}: seed(s) {_off_grid} of {len(hists)} evaluate at a DIFFERENT step grid than seed 0 — the seed-averaged curve is taken over the steps they SHARE (aligned on the step VALUE, not on list position), so a differently-sampled seed contributes only to the common steps.')
            steps_h = sorted(per_step)
            cols = {s: i for i, s in enumerate(steps_h)}
            mat = np.full((len(hists), len(steps_h)), float('nan'))
            for _i, _h in enumerate(hists):
                _own = {}
                for _pt in _h:
                    try:
                        _own[int(_pt[0])] = float(_pt[1])
                    except (TypeError, ValueError, IndexError):
                        continue
                for _s, _v in _own.items():
                    mat[_i, cols[_s]] = _v
            good = np.array([[ppl_is_usable(v) for v in row] for row in mat])
            n_good = good.sum(axis=0)
            sums = np.where(good, mat, 0.0).sum(axis=0)
            mean_ppl = np.where(n_good > 0, sums / np.maximum(n_good, 1), np.nan)
            n_dropped = int((~good).sum())
            if n_dropped:
                print(f'[aggregate] {key}: {n_dropped} non-finite PPL curve point(s) across {len(hists)} seed(s) excluded from the seed-averaged curve ({int((n_good == 0).sum())} step(s) had no usable seed and are omitted)')
            _keep = n_good > 0
            entry['ppl_curve'] = [[int(s), float(m)] for s, m, k in zip(steps_h, mean_ppl, _keep) if k]
        agg[key] = entry
    for key, entry in agg.items():
        recs = _group_records.get((entry['variant'], entry['protocol'], entry.get('run_cfg_class')), [])
        entry['source'] = key
        if recs:
            entry['_group_records'] = recs
        rates = [r['tokens_seen'] / r['steps'] for r in recs if r.get('tokens_seen') and r.get('steps')]
        if rates:
            entry['tokens_per_step'] = float(np.mean(rates))
    _full_groups = {(tag, fp_idx): recs for (v, tag, fp_idx), recs in _split_groups if v == 'full'}

    def _cfg_fp_noparam(rec):
        # Baseline-matching fingerprint WITHOUT params: a `full` baseline
        # legitimately differs from the sparse arm in parameter count, so a
        # params-including fingerprint could never match it (dead branch).
        # run_cfg + steps/tokens is exactly the gate `pair_reason` applies.
        return json.dumps([rec.get('run_cfg'), [rec.get(k) for k in PAIR_STEP_KEYS]], sort_keys=True, default=str)

    def _baseline_for(tag, fp_idx, base_groups, recs):
        _want = _cfg_fp_noparam(recs[0]) if recs else None
        if _want is not None:
            _cands = sorted(base_groups.items(), key=lambda kv: (kv[0][0] != tag, kv[0][1] is None))
            for (_t_c, _i_c), _grp in _cands:
                if any((_cfg_fp_noparam(_r) == _want for _r in _grp)):
                    return (_grp, _t_c)
        for _cand in ((tag, fp_idx), (tag, None), ('', None)):
            if _cand in base_groups:
                return (base_groups[_cand], _cand[0])
        _want_cfg = recs[0].get('run_cfg') if recs else None
        if _want_cfg is not None:
            _rc = [(_t_c, _grp) for (_t_c, _i_c), _grp in base_groups.items() if _t_c == tag and _grp and (_grp[0].get('run_cfg') == _want_cfg)]
            if len(_rc) == 1:
                return (_rc[0][1], _rc[0][0])
        return (None, None)
    _base_amb = []
    for (v, tag, fp_idx), recs in _split_groups:
        if v == 'full':
            continue
        key = _agg_key(v, tag, fp_idx)
        base_recs, base_tag = _baseline_for(tag, fp_idx, _full_groups, recs)
        if base_recs is None:
            print(f'[aggregate] {key}: NO `full` baseline in this panel — dPPL_vs_full not computed')
            continue
        if base_tag != tag:
            print(f'[aggregate] {key}: pairing against the `full#{base_tag or '(untagged)'}` baseline (no `full` run exists for protocol {tag or '(untagged)'})')
        elif tag and any((_t == '' for _t, _i in _full_groups)):
            _base_amb.append(f'{key}: `full#{tag}` and `full` both present')
            continue
        ds, skipped = _paired(recs, base_recs)
        if skipped:
            _why = sorted({reason for _s, reason in skipped})
            print(f'[aggregate] {key}: {len(skipped)} seed(s) NOT paired with `full` — {_why}; excluded from dPPL_vs_full')
        if ds:
            d = np.array(ds, dtype=float)
            agg[key]['dPPL_vs_full_mean'] = float(d.mean())
            agg[key]['dPPL_vs_full_std'] = float(d.std(ddof=1)) if len(d) > 1 else 0.0
            agg[key]['n_paired'] = len(d)
            agg[key]['n_paired_skipped'] = len(skipped)
            agg[key]['dPPL_vs_full_base'] = base_tag or '(untagged)'
    _sw_groups = {(tag, fp_idx): recs for (v, tag, fp_idx), recs in _split_groups if v == 'full_sw128_matched'}
    for (v, tag, fp_idx), recs in _split_groups:
        if v in ('full', 'full_sw128_matched'):
            continue
        key = _agg_key(v, tag, fp_idx)
        base_recs, base_tag = _baseline_for(tag, fp_idx, _sw_groups, recs)
        if base_recs is None:
            continue
        if base_tag != tag:
            print(f'[aggregate] {key}: pairing against the UNTAGGED `full_sw128_matched` baseline')
        elif tag and any((_t == '' for _t, _i in _sw_groups)):
            _base_amb.append(f'{key}: `full_sw128_matched#{tag}` and the untagged one both present')
            continue
        ds, skipped = _paired(recs, base_recs)
        if skipped:
            _why = sorted({reason for _s, reason in skipped})
            print(f'[aggregate] {key}: {len(skipped)} seed(s) NOT paired with `full_sw128_matched` — {_why}; excluded')
        if ds:
            d = np.array(ds, dtype=float)
            agg[key]['dPPL_vs_sw128m_mean'] = float(d.mean())
            agg[key]['dPPL_vs_sw128m_std'] = float(d.std(ddof=1)) if len(d) > 1 else 0.0
            agg[key]['n_paired_sw128m'] = len(d)
            agg[key]['n_paired_sw128m_skipped'] = len(skipped)
    if _base_amb:
        raise ValueError('aggregate: ambiguous dense baseline for ' + '; '.join(sorted(_base_amb)) + '. A same-tagged and an untagged baseline are both present, so the reported delta would depend on which one was picked. Remove or re-tag one of them before aggregating.')
    return agg
_TW = {'variant': 18, 'ppl': 16, 'd': 15, 'd2': 15, 'params': 10, 'seeds': 6, 'blklen': 7, 'bnd': 13}

def _fmt_cell(value, width, fmt, absent='—'):
    s = absent if value is None else f'{value:{fmt}}'
    return s if len(s) >= width else f'{s:>{width}s}'

def _fmt_pair(main, spread, width, main_fmt, spread_fmt, absent='—', main_cap=None, never_wider=None):
    if main is None:
        s = absent
    else:
        if never_wider is not None:
            fmt = never_wider
        elif main_cap is None:
            fmt = main_fmt
        else:
            fmt = main_cap if abs(main) >= 100000000.0 else main_fmt
        s = f'{main:{fmt}}±{spread:{spread_fmt}}'
    return s if len(s) >= width else f'{s:>{width}s}'

def _print_table(agg):
    print('\n============== validation perplexity (corpus per run cfg) ==============')
    print(f'{'variant':{_TW['variant']}s} {'PPL mean±std':>{_TW['ppl']}s} {'Δvs full':>{_TW['d']}s} {'Δvs sw128m':>{_TW['d2']}s} {'params(M)':>{_TW['params']}s} {'seeds':>{_TW['seeds']}s} {'blklen':>{_TW['blklen']}s} {'bndF1':>{_TW['bnd']}s}')
    order = ['full', 'full_matched', 'full_cos', 'full_sink', 'full_sw128', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic', 'csa_dyn_fuse', 'csa_dyn_nocomp', 'csa_fix_randidx', 'csa_fix_zerocont', 'csa_fix_nosink', 'csa_fix_topk8', 'csa_fix_topk64', 'hybrid_fixed', 'hybrid_dynamic', 'hybrid_csa_dyn', 'hybrid_hca_dyn', 'hybrid_csa_dyn_fuse']
    rows = [v for v in order if v in agg] + [v for v in agg if v not in order]
    for v in rows:
        e = agg[v]
        _n = e['n_seeds']
        ppl = _fmt_pair(e['ppl_mean'], e.get('ppl_std', 0.0), _TW['ppl'], '8.2f', '5.2f', main_cap='.2f') if _n > 1 else _fmt_cell(e['ppl_mean'], _TW['ppl'], '.2f')
        d = e.get('dPPL_vs_full_mean')
        dd = _fmt_pair(d, e.get('dPPL_vs_full_std', 0.0), _TW['d'], '+8.2f', '5.2f', main_cap='+.2f')
        d2 = e.get('dPPL_vs_sw128m_mean')
        dd2 = _fmt_pair(d2, e.get('dPPL_vs_sw128m_std', 0.0), _TW['d2'], '+8.2f', '5.2f', main_cap='+.2f')
        bl = e.get('avg_dyn_block_len', e.get('avg_block_len'))
        bls = _fmt_cell(bl, _TW['blklen'], '.1f')
        _pm = e.get('params')
        pms = _fmt_cell(_pm / 1000000.0 if _pm is not None else None, _TW['params'], '.2f')
        bf = e.get('boundary_f1_dyn')
        bfs = _fmt_pair(bf, e.get('boundary_f1_dyn_std', 0.0), _TW['bnd'], '6.3f', '5.3f', never_wider='.3f')
        try:
            _ns = _fmt_cell(int(_n), _TW['seeds'], 'd')
        except (TypeError, ValueError):
            _ns = _fmt_cell(None, _TW['seeds'], 'd')
        _cells = [(v, _TW['variant']), (ppl, _TW['ppl']), (dd, _TW['d']), (dd2, _TW['d2']), (pms, _TW['params']), (_ns, _TW['seeds']), (bls, _TW['blklen']), (bfs, _TW['bnd'])]
        print(' '.join((f'{s:>{w}s}' for s, w in _cells)))
    print('Δ vs full   = paired per-seed difference vs dense (positive = worse).')
    print('Δ vs sw128m = paired difference vs the windowed param-matched dense')
    print('              baseline = what compression adds beyond pure locality.')
    print('blklen = mean learned block length (dynamic layers); bndF1 = boundary alignment F1 mean±std over seeds (v4: n_sample=64);')
    print('        see summary.json for bnd_rand / bnd_excess: excess > 0 means the')
    print('        learned boundaries are genuinely semantic).')
    print('================================================================')

def _print_pairs(summary):
    recs, _amb = _variant_seed_map(summary)
    _pair_amb = sorted(_amb)
    _pair_skip = []
    PAIRS = [('csa_dynamic', 'csa_fixed'), ('hybrid_dynamic', 'hybrid_fixed'), ('hybrid_csa_dyn', 'hybrid_fixed'), ('hybrid_hca_dyn', 'hybrid_fixed'), ('csa_dyn_fuse', 'csa_dynamic'), ('hybrid_csa_dyn_fuse', 'hybrid_csa_dyn'), ('csa_fix_randidx', 'csa_fixed'), ('csa_fix_zerocont', 'csa_fixed'), ('csa_fix_nosink', 'csa_fixed'), ('csa_fix_topk8', 'csa_fixed'), ('csa_fix_topk64', 'csa_fixed'), ('csa_dyn_randidx', 'csa_dynamic'), ('csa_dyn_zerocont', 'csa_dynamic'), ('csa_dyn_nosink', 'csa_dynamic'), ('csa_dyn_nocomp', 'csa_dynamic'), ('full_sink', 'full')]
    print('\n========== paired per-seed deltas (a − b, positive = a worse) ==========')
    print('  (p = exact two-sided sign-flip over the paired per-seed deltas; a noise-level Δ is flagged even if p is small)')
    if _pair_amb:
        print('  [skip] a (variant, protocol, seed) triple holds more than one measured record, so its PPL is ambiguous — no delta printed for : ' + ', '.join(sorted((f'{v}/{t or '-'}/s{s}' for v, t, s in set(_pair_amb)))[:4]))
    for a, b in PAIRS:

        def _sel(v):
            return {(t, s): p for (vv, t, s), p in recs.items() if vv == v}
        ra, rb = (_sel(a), _sel(b))
        _a_t = {t for t, _s in ra}
        _b_t = {t for t, _s in rb}
        _a_pure_base = _a_t == {''}
        _b_pure_base = _b_t == {''}
        common = []
        for tag in sorted(_a_t | _b_t):
            if tag in _a_t and tag in _b_t and tag:
                seeds = sorted({s for t, s in ra if t == tag} & {s for t, s in rb if t == tag})
                common += [(tag, tag, s) for s in seeds]
            if tag and tag in _a_t and _b_pure_base:
                seeds = sorted({s for t, s in ra if t == tag} & {s for t, s in rb if t == ''})
                common += [(tag, '', s) for s in seeds]
            if tag and tag in _b_t and _a_pure_base:
                seeds = sorted({s for t, s in ra if t == ''} & {s for t, s in rb if t == tag})
                common += [('', tag, s) for s in seeds]
            if tag == '' and '' in _a_t and ('' in _b_t):
                seeds = sorted({s for t, s in ra if t == ''} & {s for t, s in rb if t == ''})
                common += [('', '', s) for s in seeds]
        _by_cond = {}
        _sk = []
        for ts, bs, s in common:
            _reason = pair_reason(ra[ts, s], rb[bs, s])
            if _reason is not None:
                _sk.append((s, _reason))
                continue
            _by_cond.setdefault((ts, bs), []).append(ra[ts, s]['ppl'] - rb[bs, s]['ppl'])
        if _sk:
            _pair_skip.append((a, b, len(_sk), len(common), sorted({r for _s, r in _sk})))
        for (ts, bs), dl in sorted(_by_cond.items()):
            st = paired_effect_stats(dl)
            _p = ('%.4f' % st['p_exact_signflip']) if math.isfinite(st['p_exact_signflip']) else 'n/a'
            _lab = '' if ts == bs == '' else f'   [{ts or '(untagged)'} vs {bs or '(untagged)'}]'
            print(f'  {a:22s} − {b:22s} = {st["mean"]:+7.2f} ± {st["std"]:5.2f}   (n={st["n"]}, p={_p}, {st["effect_vs_noise"]}){_lab}')
    if _pair_skip:
        print('  [skip] seed(s) rejected by the run_cfg/load gate (see `pair_reason`); the printed n counts only the pairs that cleared it —')
        for _a, _b, _nsk, _ncom, _why in _pair_skip:
            print(f'    {_a} − {_b}: {_nsk}/{_ncom} rejected — {_why}')
    print('=' * 78)

def _save_csv(agg, outdir):
    cols = ['variant', 'ppl_mean', 'ppl_std', 'dPPL_vs_full_mean', 'dPPL_vs_full_std', 'dPPL_vs_sw128m_mean', 'dPPL_vs_sw128m_std', 'n_seeds', 'params', 'avg_dyn_block_len', 'boundary_f1_dyn', 'boundary_excess_dyn']
    rows = [{'variant': v, **{c: e.get(c, '') for c in cols[1:]}} for v, e in agg.items()]
    atomic_write_csv(os.path.join(outdir, 'aggregate.csv'), cols, rows)

def _plot(summary, agg, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not agg:
        return
    order = ['full', 'full_matched', 'full_cos', 'full_sink', 'full_sw128', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic', 'csa_dyn_fuse', 'csa_dyn_nocomp', 'csa_fix_randidx', 'csa_fix_zerocont', 'csa_fix_nosink', 'csa_fix_topk8', 'csa_fix_topk64', 'hybrid_fixed', 'hybrid_dynamic', 'hybrid_csa_dyn', 'hybrid_hca_dyn', 'hybrid_csa_dyn_fuse']
    vs = [v for v in order if v in agg] + [v for v in agg if v not in order]
    fig, ax = plt.subplots(1, 5, figsize=(27, 4.5))
    ps = [agg[v]['ppl_mean'] for v in vs]
    es = [agg[v]['ppl_std'] for v in vs]
    ax[0].bar(range(len(vs)), ps, yerr=es, capsize=3, color='steelblue')
    ax[0].set_xticks(range(len(vs)))
    ax[0].set_xticklabels(vs, rotation=30, ha='right')
    ax[0].set_ylabel('validation perplexity (mean ± std)')
    ax[0].set_title('val PPL (corpus per run cfg)')
    ax[0].grid(alpha=0.3, axis='y')
    _unknown = [v for v in vs if agg[v].get('params') is None]
    if _unknown:
        print(f'[plot] {len(_unknown)} variant(s) report no `params` ({', '.join(_unknown)}) — their size bar is left at 0 and the size panel cannot be read for them')
    params_m = [(agg[v]['params'] or 0) / 1000000.0 for v in vs]
    ax[1].bar(range(len(vs)), params_m, color='orange')
    ax[1].set_xticks(range(len(vs)))
    ax[1].set_xticklabels(vs, rotation=30, ha='right')
    ax[1].set_ylabel('params (M)')
    ax[1].set_title('Model size')
    ax[1].grid(alpha=0.3, axis='y')
    for v in vs:
        recs = [r for r in agg[v].get('_group_records', []) if isinstance(r, dict) and 'losses' in r and (not r.get('synthesized')) and isinstance(r.get('losses'), list) and r['losses'] and all((isinstance(x, (int, float)) and math.isfinite(x) for x in r['losses']))]
        if not recs:
            continue
        _lens = sorted({len(r['losses']) for r in recs})
        if len(_lens) > 1:
            print(f'[_plot] {v}: seeds trained a DIFFERENT number of steps {_lens} — the training curve is drawn over the shared prefix of {_lens[0]} step(s) only.')
        n_pt = min((len(r['losses']) for r in recs))
        curve = np.mean([np.array(r['losses'][:n_pt], dtype=float) for r in recs], axis=0)
        k = max(1, n_pt // 50)
        _num = np.convolve(curve, np.ones(k), mode='same')
        _den = np.convolve(np.ones_like(curve), np.ones(k), mode='same')
        smooth = np.where(_den > 0, _num / _den, 0.0)
        ax[2].plot(smooth, label=v)
    ax[2].set_xlabel('step')
    ax[2].set_ylabel('train loss (seed-averaged)')
    ax[2].set_title('Training curves')
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3)
    drew = False
    for v in vs:
        cur = agg[v].get('ppl_curve')
        if not cur:
            continue
        steps_h = np.array([p[0] for p in cur], dtype=float)
        ppls_h = np.array([p[1] for p in cur], dtype=float)
        tok_per_step = agg[v].get('tokens_per_step')
        if not tok_per_step:
            print(f'[_plot] {v}: no `tokens_per_step` in the aggregate — omitted from the tokens axis (plotting steps under a tokens label would be wrong by seq_len*batch_size). Re-run `aggregate` to stamp it.')
            continue
        ax[3].plot(steps_h * tok_per_step / 1000000.0, ppls_h, marker='.', ms=4, label=v)
        drew = True
    ax[3].set_xlabel('tokens seen (M)')
    ax[3].set_ylabel('val-subset PPL')
    ax[3].set_title('PPL vs training tokens (gap dynamics)')
    if drew:
        ax[3].legend(fontsize=7)
    ax[3].grid(alpha=0.3)
    _dense_keys = [k for k in agg if k.split('#')[0].split('@')[0] == 'full']
    _dense_key = next((k for k in _dense_keys if agg[k].get('ppl_curve') and agg[k].get('tokens_per_step')), None)
    if _dense_key is None:
        print(f'[_plot] no dense `full` group with both a `ppl_curve` and a `tokens_per_step` in this aggregate (candidates: {_dense_keys or 'none'}) — the sparse-vs-dense gap panel (finding #10) is NOT drawn.')
    else:
        if _dense_key != 'full':
            print(f'[_plot] the dense baseline is published as `{_dense_key}` (the `full` group was split by `run_cfg`); the sparse-vs-dense gap is taken against it' + (f', not {_dense_keys[1:]}' if len(_dense_keys) > 1 else '') + '.')
        ref = agg[_dense_key].get('ppl_curve')
        tps0 = agg[_dense_key].get('tokens_per_step')
    if _dense_key is not None and ref and tps0:
        gx = np.array([p[0] for p in ref], dtype=float) * tps0
        gy = np.array([p[1] for p in ref], dtype=float)
        drew2 = False
        for v in vs:
            if v == _dense_key:
                continue
            if v.startswith('full@cfg'):
                continue
            cur = agg[v].get('ppl_curve')
            if not cur:
                continue
            tps = agg[v].get('tokens_per_step')
            if not tps:
                continue
            xs = np.array([p[0] for p in cur], dtype=float) * tps
            ys = np.array([p[1] for p in cur], dtype=float)
            keep = (gx >= xs[0]) & (gx <= xs[-1])
            if not keep.any():
                continue
            ax[4].plot(gx[keep] / 1000000.0, np.interp(gx[keep], xs, ys) - gy[keep], marker='.', ms=4, label=f'{v}−full')
            drew2 = True
        ax[4].axhline(0, color='k', lw=0.8)
        ax[4].set_xlabel('tokens seen (M)')
        ax[4].set_ylabel('ΔPPL vs full')
        ax[4].set_title('sparse-vs-dense gap vs tokens (finding #10)')
        if drew2:
            ax[4].legend(fontsize=7)
        ax[4].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'summary.png'), dpi=120)
    plt.close(fig)

def _read_summary_json(path):
    summary = {}
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as f:
                summary = json.load(f)
        except Exception as _e:
            print(f'[resume] FATAL: {path} exists but cannot be parsed ({type(_e).__name__}: {_e}).  Refusing to overwrite it with an empty summary — move it aside to start fresh.')
            raise
    return summary


def _run_fp(cfg, d, n_layers, n_heads, d_head, vocab, train_ids, val_batch):
    """The resume fingerprint of a run() cell.  The parent planner and every
    par.py child compute the exact same string: the child re-serves the token
    ids from the same cache files, and `_ids_fp` is content-based, so the
    fingerprint never depends on which process computed it."""
    _mg = cfg.get('matched')
    if _mg is None:
        _mkey = 'default'
    else:
        _mkey = ','.join(sorted((str(_x) for _x in _mg)))
    cfg.setdefault('warmup', 50)
    cfg.setdefault('comp_lambda', 0.05)
    return f'steps{cfg['steps']}_sl{cfg['seq_len']}_bs{cfg['batch_size']}_nt{cfg['n_train_tokens']}_lr{cfg['lr']}_wd{cfg['weight_decay']}_d{d}_L{n_layers}_H{n_heads}_Dh{d_head}_v{vocab}_wu{cfg['warmup']}_cl{cfg['comp_lambda']}_dlm{cfg.get('delta_lr_mult', 10.0)}_mt{_mkey}_mr{cfg.get('mlp_match_ref', 'csa_dynamic')}_ee{cfg.get('eval_every', 0)}_es{cfg.get('eval_subset', 128)}_det{determinism_label()}_df{_ids_fp(train_ids)}-{_ids_fp(np.ascontiguousarray(val_batch).reshape(-1))}_cs{CODE_SEMANTICS}{_kernel_stamp()}'


def _plan_run(cfg, summary_path, fp):
    """Resume-check every (seed x variant) cell up front, exactly as the old
    interleaved loop did, and return the cells that still need to run.

    Stale / synthesized / non-measurable records are dropped (under the lock)
    BEFORE any training starts, so a later failure can never leave an old
    reading in place looking like this attempt's result.  With parallel
    dispatch this check runs once in the parent, before any child spawns."""
    summary = locked_json_read(summary_path)
    cells = []
    stale_dropped = []
    for seed in cfg['seeds']:
        for v in cfg['variants']:
            key = f'{v}::seed{seed}'
            _drop_msg = None
            _drop_is_stale = False
            if key in summary and 'ppl' in summary.get(key, {}):
                old = summary[key]
                if not ppl_is_usable(old.get('ppl')):
                    _drop_msg = f"[resume] {key} holds a NON-MEASURABLE ppl ({old.get('ppl')!r}) — every statistic in this repo refuses it (`ppl_is_usable`), so counting it as 'already completed' would drop the seed from the tables silently. DROPPING it and retraining"
                elif old.get('synthesized'):
                    _drop_msg = f'[resume] {key} is a SYNTHESIZED record (reconstructed from an old aggregate.json, no weights) — DROPPING it and retraining'
                elif old.get('run_cfg') is None:
                    _drop_msg = f"[resume] {key} has no config fingerprint (predates fingerprinting) — re-running and overwriting; the STALE record is dropped before retraining, so a failed or truncated attempt cannot leave it in place looking like this attempt's result"
                    _drop_is_stale = True
                elif old.get('run_cfg') == fp:
                    print(f'[skip] {key} already completed (resume)')
                    continue
                else:
                    _drop_msg = f"[resume] {key} was run under a DIFFERENT config ({old['run_cfg']} != {fp}) — re-running and overwriting; the STALE record is dropped before retraining, so a failed or truncated attempt cannot leave it in place looking like this attempt's result"
                    _drop_is_stale = True
            if _drop_msg is not None:
                print(_drop_msg)
                summary.pop(key, None)

                def _drop_cell(s, _key=key):
                    s.pop(_key, None)
                    return s
                locked_json_update(summary_path, _drop_cell)
                if _drop_is_stale:
                    stale_dropped.append(key)
            cells.append((v, seed))
    return cells, stale_dropped


def _run_cell(cfg, v, seed, *, guard=None, ratios=None, fp=None, train_ids=None, val_batch=None, vocab=None, val_bnd=None, prebook=False, prov=None):
    """Run exactly ONE (variant, seed) training cell and merge its record into
    outdir/summary.json under the lock.

    Used by the serial loop (in-process) and by par.py children (prebook=True:
    the projected time is reserved in the shared TimeGuard before training and
    refunded after recording the actual seconds, so concurrently launching
    children still respect the shared run-time cap)."""
    d = cfg.get('d', 256)
    n_layers = cfg.get('n_layers', 6)
    n_heads = cfg.get('n_heads', 8)
    d_head = cfg.get('d_head', 32)
    outdir = cfg['outdir']
    summary_path = os.path.join(outdir, 'summary.json')
    key = f'{v}::seed{seed}'
    ratios = {} if ratios is None else ratios
    if v not in ratios:
        ratios[v] = variant_mlp_ratio(v, vocab, d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, seq_len=cfg['seq_len'], matched=cfg.get('matched'), ref_variant=cfg.get('mlp_match_ref', 'csa_dynamic'))
    out = {'status': 'ok', 'key': key, 'variant': v, 'seed': seed}
    _run_id = new_experiment_id('cell')
    _started_at = time.strftime('%Y-%m-%dT%H:%M:%S')
    _t_start = time.time()
    out['run_id'] = _run_id
    deadline_ts = None
    booked = 0.0
    _book_tok = f'run|{key}|pid{os.getpid()}'
    if guard is not None:
        est = guard.estimate_seconds(cfg['steps'], d=d, n_layers=n_layers, seq_len=cfg['seq_len'], batch_size=cfg['batch_size'])
        if prebook and est:
            guard.book(est, token=_book_tok)
            booked = est
        if not guard.can_start(0 if booked else est):
            if booked:
                guard.book(-booked, token=_book_tok)
                booked = 0.0
            print(f'[cap] SKIP {key}: projected {est / 3600:.2f} h would pass the configured cap (headroom {max(guard.total_hours - guard.spent_hours(), 0.0):.2f} h)')
            out['status'] = 'skipped_cap'
            log_experiment_event(outdir, {'kind': 'cell', 'run_id': _run_id, 'status': 'skipped_cap', 'key': key, 'variant': v, 'seed': int(seed), 'run_cfg': fp, 'code': code_stamp(), 'started_at': _started_at, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'reason': 'projected runtime would pass the configured cap'})
            return out
        deadline_ts = guard.deadline_ts()
        if booked:
            # `spent_hours()` includes this cell's own pre-booking, which is
            # the time this cell is about to use rather than time already
            # spent, so add it back here.  Other cells' bookings stay
            # subtracted (shared-headroom contract).
            deadline_ts += booked
        print(f'[cap] {key}: projected {est / 60:.0f} min, booked so far {guard.state['booked_seconds'] / 3600:.2f} h')
    t_run = time.time()
    rec = None
    try:
        try:
            rec = train_variant(v, train_ids, val_batch, vocab, seed=seed, d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, seq_len=cfg['seq_len'], batch_size=cfg['batch_size'], steps=cfg['steps'], lr=cfg['lr'], weight_decay=cfg['weight_decay'], warmup=cfg['warmup'], comp_lambda=cfg['comp_lambda'], delta_lr_mult=cfg.get('delta_lr_mult', 10.0), eval_every=cfg.get('eval_every', 0), eval_subset=cfg.get('eval_subset', 128), val_bnd=val_bnd, mlp_ratio=ratios[v], deadline_ts=deadline_ts)
            _prov = prov if prov is not None else experiment_provenance(cfg, train_ids=train_ids, val_batch=val_batch, vocab=vocab)
            _hp = run_hyperparameters(cfg, seed=seed, d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, vocab=vocab, mlp_ratio=ratios[v])

            def _apply(summary):
                if rec.get('cap_truncated'):
                    _prev_ok = ppl_is_usable((summary.get(key) or {}).get('ppl'))
                    print(f'[cap] {key} was truncated after {rec.get('steps_done')} steps — its partial record is DISCARDED (it holds no `ppl`); ' + ('keeping the previous record instead.' if _prev_ok else 'the key is left ABSENT, not written as a stub.') + ' Raise RUN_CAP and re-run to retry this cell.')
                    if not _prev_ok:
                        summary.pop(key, None)
                        out['truncated'] = True
                else:
                    rec['run_cfg'] = fp
                    rec['_code'] = f'{CODE_SEMANTICS}{_kernel_stamp()}'
                    assert fp.endswith(f'_cs{rec['_code']}'), (fp, rec['_code'])
                    rec['provenance'] = _prov
                    rec['hyperparams'] = _hp
                    rec['data_fp'] = _prov['data'].get('train_fp')
                    rec['val_fp'] = _prov['data'].get('val_fp')
                    rec['run_id'] = _run_id
                    rec['started_at'] = _started_at
                    rec['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
                    if os.path.exists(os.path.join(outdir, 'overfit_sanity.json')):
                        rec['sanity_ref'] = 'overfit_sanity.json'
                    summary[key] = rec
                return summary
            locked_json_update(summary_path, _apply)
            if out.get('truncated'):
                out['status'] = 'truncated'
                log_experiment_event(outdir, {'kind': 'cell', 'run_id': _run_id, 'status': 'truncated', 'key': key, 'variant': v, 'seed': int(seed), 'run_cfg': fp, 'code': code_stamp(), 'started_at': _started_at, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'steps_done': rec.get('steps_done'), 'reason': 'hard runtime cap reached; partial record not written'})
            elif isinstance(rec, dict) and not rec.get('cap_truncated'):
                _losses = rec.get('losses') if isinstance(rec.get('losses'), list) else []
                _ev = {'kind': 'cell', 'run_id': _run_id, 'status': 'ok', 'started_at': _started_at, 'finished_at': rec.get('finished_at'), 'elapsed_s': time.time() - _t_start, 'key': key, 'variant': v, 'seed': int(seed), 'run_cfg': fp, 'code': rec.get('_code') or code_stamp(), 'provenance': rec.get('provenance'), 'hyperparams': rec.get('hyperparams'), 'data_fp': rec.get('data_fp'), 'val_fp': rec.get('val_fp'), 'rng_fp': rec.get('rng_fp'), 'metrics': {'ppl': rec.get('ppl'), 'params': rec.get('params'), 'steps': rec.get('steps'), 'tokens_seen': rec.get('tokens_seen'), 'train_time_s': rec.get('train_time_s'), 'final_loss_smoothed': rec.get('final_loss_smoothed')}, 'curve_summary': {'n_points': len(_losses), 'first': _losses[0] if _losses else None, 'last': _losses[-1] if _losses else None, 'best': min(_losses) if _losses else None}, 'losses': _losses, 'ppl_history': rec.get('ppl_history'), 'sanity_ref': rec.get('sanity_ref')}
                log_experiment_event(outdir, _ev)
        except Exception as e:
            import traceback
            import sys as _sys
            out['status'] = 'error'
            out['error'] = f'{e}'

            def _apply_err(summary):
                if key in summary and ppl_is_usable((summary.get(key) or {}).get('ppl')):
                    print(f'[{key}] FAILED: {e} — keeping the previous MEASURED record (the error record holds no `ppl` and must not replace it); the failed attempt is still booked to the guard below.')
                    print(traceback.format_exc(), file=_sys.stderr)
                else:
                    summary[key] = {'variant': v, 'seed': seed, 'error': traceback.format_exc()}
                    print(f'[{key}] FAILED: {e}')
                return summary
            locked_json_update(summary_path, _apply_err)
            log_experiment_event(outdir, {'kind': 'cell', 'run_id': _run_id, 'status': 'error', 'started_at': _started_at, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'elapsed_s': time.time() - _t_start, 'key': key, 'variant': v, 'seed': int(seed), 'run_cfg': fp, 'code': code_stamp(), 'error': f'{e}', 'traceback': traceback.format_exc()})
    finally:
        if guard is not None:
            steps_done = 0
            calib_s = None
            if isinstance(rec, dict) and rec.get('steps_done'):
                steps_done = int(rec['steps_done'])
            elif isinstance(rec, dict) and 'ppl' in rec:
                steps_done = int(cfg['steps'])
                calib_s = rec.get('train_time_s')
            guard.record_run(time.time() - t_run, steps_done, d, n_layers, cfg['seq_len'], cfg['batch_size'], calib_seconds=calib_s)
            if booked:
                guard.book(-booked, token=_book_tok)
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return out


def run_cell_job(job):
    """Child-process entry for one `run`-kind cell (invoked by par.py).

    Re-serves the corpus from the same on-disk cache, recomputes the identical
    fingerprint, and runs the cell with pre-booked guard accounting.  The
    data-load seconds are NOT booked here: the parent already booked the
    phase's data load once, exactly like the serial path."""
    cfg = dict(job['cfg'])
    _matched = cfg.get('matched')
    if isinstance(_matched, (list, tuple)):
        cfg['matched'] = set(_matched)
    guard = TimeGuard(job['guard_cap']) if job.get('guard_cap') else None
    train_ids, val_batch, vocab, _, val_bnd = load_run_corpus(cfg)
    d = cfg.get('d', 256)
    n_layers = cfg.get('n_layers', 6)
    n_heads = cfg.get('n_heads', 8)
    d_head = cfg.get('d_head', 32)
    # pin determinism BEFORE the fingerprint: run() does the same before its
    # own fp, and `determinism_label()` is part of the fingerprint — an
    # unpinned child would stamp a different run_cfg and break resume.
    if torch.cuda.is_available():
        _pin_cuda_determinism()
    fp = _run_fp(cfg, d, n_layers, n_heads, d_head, vocab, train_ids, val_batch)
    return _run_cell(cfg, job['v'], job['seed'], guard=guard, ratios={}, fp=fp, train_ids=train_ids, val_batch=val_batch, vocab=vocab, val_bnd=val_bnd, prebook=True)


def run(cfg=None, seeds=None, guard=None, label='', par=None):
    cfg = dict(RUN) if cfg is None else dict(cfg)
    if seeds is not None:
        cfg['seeds'] = list(seeds)
    d = cfg.get('d', 256)
    n_layers = cfg.get('n_layers', 6)
    n_heads = cfg.get('n_heads', 8)
    d_head = cfg.get('d_head', 32)
    outdir = cfg['outdir']
    os.makedirs(outdir, exist_ok=True)
    # Cheap synthetic pre-flight before the (possibly slow) corpus load.
    _sanity = list(overfit_guard(cfg.get('variants') or []))
    _attb_bump_epoch()
    t_data = time.time()
    train_ids, val_batch, vocab, _, val_bnd = load_run_corpus(cfg)
    if guard is not None:
        guard.record_run(time.time() - t_data, 0, 0, 0, 0, 0)
    summary_path = os.path.join(outdir, 'summary.json')
    _read_summary_json(summary_path)
    if label:
        print(f'\n########## phase: {label} — {len(cfg['variants'])} variants x seeds {cfg['seeds']} x {cfg['steps']} steps ##########')
    # Prospective seed-budget note: a panel that contains an ablation arm is
    # checked BEFORE training so its seed count is not mistaken for a
    # confirmation after the fact.  This never blocks a run.
    try:
        _vs = set(cfg.get('variants') or [])
        _declared_pairs = [(_a, _b) for _contrasts in list(ABLATION_DESIGN.values()) + list(BASELINE_DESIGN.values()) for _a, _b, _n in _contrasts]
        if any(_a in _vs and _b in _vs for _a, _b in _declared_pairs):
            _sp = methodology_seed_plan(len(cfg['seeds']), variants=cfg.get('variants'))
            if _sp.get('min_seeds_for_family_significance') and not _sp.get('sufficient_for_family_rejection'):
                print('[seed-plan] this panel declares %d controlled contrasts; with n=%d seeds the family-corrected exact-test floor is above alpha, so results are DESCRIPTIVE ONLY. '
                      'A family-corrected rejection needs n>=%d (raise the seed list or narrow the pre-registered family before claiming significance).'
                      % (_sp['declared_contrasts'], _sp['n_seeds'], _sp['min_seeds_for_family_significance']))
    except Exception as _sp_err:
        print(f'[seed-plan] note unavailable ({type(_sp_err).__name__}: {_sp_err}) — continuing')
    if torch.cuda.is_available():
        _pin_cuda_determinism()
    fp = _run_fp(cfg, d, n_layers, n_heads, d_head, vocab, train_ids, val_batch)
    prov = experiment_provenance(cfg, train_ids=train_ids, val_batch=val_batch, vocab=vocab)
    # The real data-path check: the synthetic pass above cannot catch a broken
    # real token stream / vocabulary mapping.  It is a superset of the model
    # plumbing check, so a pass here is the stronger evidence; both reports
    # are persisted with the panel.
    try:
        _sanity += list(overfit_guard(cfg.get('variants') or [], train_ids=train_ids, vocab=vocab))
    except RuntimeError:
        raise
    if not _sanity:
        # The guard is process-cached (correctly, so repeated phases do not
        # retrain the probe).  Recover THIS variant set's evidence from the
        # cache so a fresh outdir still gets a persisted sanity record.  Data
        # reports are matched on the current corpus+vocab, never a stale one.
        _want = set(cfg.get('variants') or [])
        _cur_data = (int(vocab), _ids_fp(train_ids))
        _sanity = [dict(r) for (v, src, *rest), r in _OVERFIT_REPORTS.items() if v in _want and (src == 'synthetic' or (len(rest) >= 2 and (rest[0], rest[1]) == _cur_data))]
    if _sanity:
        atomic_write_json(os.path.join(outdir, 'overfit_sanity.json'), {
            'kind': 'small_batch_overfit_sanity_panel',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'passed': all(bool(r.get('passed')) for r in _sanity),
            'provenance': prov,
            'reports': _sanity,
        })
    cells, stale_dropped = _plan_run(cfg, summary_path, fp)
    dropped_truncations = []
    ratios = {}
    _ran_parallel = False
    if cells:
        import par as _par
        _par_ctx = par if isinstance(par, dict) else _par.resolve_ctx(par, label=label or outdir)
        if _par_ctx and _par.ctx_active(_par_ctx, len(cells)):
            _caller = sys._getframe(1).f_globals.get('__name__')
            # a '__main__' frame (e.g. run_gapfill.py driving L.run directly)
            # cannot be imported by that name in the child, and plain
            # exp_lib matches that caller's patch state exactly.
            module = _par_ctx.get('module') or (_caller if _caller and _caller != '__main__' else 'exp_lib')
            jobs = [dict(kind='run', module=module, label=label, cfg=cfg, v=v, seed=seed, guard_cap=(guard.cap_spec() if guard is not None else None), ckpt_flag=bool(_par_ctx.get('ckpt_flag')), ckpt_install=bool(_par_ctx.get('ckpt_install'))) for v, seed in cells]
            results = _par.run_jobs(jobs, _par_ctx, phase_label=label or outdir)
            for r, (v, seed) in zip(results, cells):
                if not r:
                    continue
                if r.get('truncated'):
                    dropped_truncations.append(r.get('key') or f'{v}::seed{seed}')
            _ran_parallel = True
    if not _ran_parallel:
        for v, seed in cells:
            res = _run_cell(cfg, v, seed, guard=guard, ratios=ratios, fp=fp, train_ids=train_ids, val_batch=val_batch, vocab=vocab, val_bnd=val_bnd, prov=prov)
            if res.get('truncated'):
                dropped_truncations.append(res['key'])
    summary = _read_summary_json(summary_path)
    if dropped_truncations:
        print(f'\n[cap] {len(dropped_truncations)} cell(s) were retrained, hit the cap deadline, and produced NO measurement — they are ABSENT from {summary_path} and from every table above:')
        for _k in dropped_truncations:
            print(f'    {_k}')
        print('[cap] raise RUN_CAP and re-run to fill these cells in.')
    if stale_dropped:
        _still = [k for k in stale_dropped if k not in summary]
        print(f'\n[resume] {len(stale_dropped)} cell(s) held a record from a DIFFERENT config and were re-run; {len(_still)} of them produced NO measurement this pass and are now ABSENT from {summary_path} (the stale record was removed rather than kept, so no number measured by older code can be reported under the current config):')
        for _k in _still:
            print(f'    {_k}')
        if len(_still) != len(stale_dropped):
            print(f'    ({len(stale_dropped) - len(_still)} of them succeeded and carry a fresh record.)')
    write_run_manifest(outdir, cfg, fp, prov, summary, sanity=(_sanity or None))
    _finish_panel(outdir, summary)
    return (summary, None)

def _dump_aggregate(outdir, summary):
    agg = {}
    try:
        agg = aggregate(summary)
    except Exception as e:
        print(f'[aggregate] aggregate({outdir}) failed: {type(e).__name__}: {e}')
        print('[aggregate] retrying with the unusable records (no measured `ppl`, `synthesized`, or an ambiguous same-seed duplicate) removed — the untouched records keep their exact values')
        _REQUIRED = ()
        clean = {k: r for k, r in summary.items() if isinstance(r, dict) and r.get('seed') is not None and ppl_is_usable(r.get('ppl')) and (not r.get('synthesized')) and all((r.get(f) is not None for f in _REQUIRED))}
        _legacy_ok = legacy_warm_tags(clean)
        _seen, _dups = ({}, [])
        for k, r in clean.items():
            rk = _record_identity(r, k, _legacy_ok)
            if rk is None:
                continue
            if rk in _seen:
                _dups.append((k, _seen[rk]))
                continue
            _seen[rk] = k
        for k, _keep in _dups:
            clean.pop(k, None)
        if _dups:
            print(f'[aggregate] {len(_dups)} ambiguous record(s) dropped (first key wins): ' + ', '.join((f'kept {keep!r}, dropped {k!r}' for k, keep in _dups)))
        print(f'[aggregate] {len(summary) - len(clean)} record(s) dropped in total: ' + (', '.join(sorted(set(summary) - set(clean))) or '(none)'))
        agg = aggregate(clean)
        atomic_write_json(os.path.join(outdir, 'summary.aggregatable.json'), clean)
        records_for_readers = clean
    else:
        records_for_readers = summary
    _pub = {k: {kk: vv for kk, vv in e.items() if kk != '_group_records'} if isinstance(e, dict) and '_group_records' in e else e for k, e in agg.items()}
    atomic_write_json(os.path.join(outdir, 'aggregate.json'), _pub)
    _save_csv(_pub, outdir)
    _print_table(_pub)
    _print_pairs(records_for_readers)
    try:
        methodology_audit(records_for_readers, outdir=outdir, verbose=True)
    except Exception as e:
        print(f'[audit] methodology audit failed ({type(e).__name__}: {e}) — the panel is still aggregated; the audit is advisory and never blocks a run')
    _plot(records_for_readers, agg, outdir)
    return agg

def _finish_panel(outdir, summary):
    return _dump_aggregate(outdir, summary)
