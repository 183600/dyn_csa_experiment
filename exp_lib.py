import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
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
    print(f'[setup] installing missing package(s): {', '.join(_NEEDED)}')
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *_NEEDED])
    except Exception as _pip_err:
        print(f'[setup] could not install {', '.join(_NEEDED)} ({type(_pip_err).__name__}: {_pip_err}) — continuing; a code path that actually needs them will raise ImportError')
import torch
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
import gc, math, os, re, time, json, random, csv, tempfile, itertools
from math import comb
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[setup] device = {DEVICE}   torch = {torch.__version__}')
QUICK = False
BUDGET = dict(total_yuan=140.0, price_per_hour=2.4, margin=0.93, already_spent_yuan=0.0, state_path='autodl_budget_state.json')
CODE_SEMANTICS = 'v11.114'
CKPT_CODE = CODE_SEMANTICS
RUN = dict(seq_len=512, batch_size=12, n_train_tokens=1000000 if QUICK else 8000000, steps=500 if QUICK else 1500, warmup=50, lr=0.0003, weight_decay=0.1, comp_lambda=0.05, delta_lr_mult=10.0, eval_every=250, eval_subset=128, seeds=[0] if QUICK else [0, 1, 2, 3, 4], outdir='results_lm_v3_1500', variants=['full', 'full_matched', 'full_cos', 'full_sw128', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic', 'hybrid_fixed', 'hybrid_dynamic'])
ABL_VARIANTS = ['hybrid_csa_dyn', 'hybrid_hca_dyn', 'csa_dyn_fuse', 'hybrid_csa_dyn_fuse', 'csa_fix_randidx', 'csa_fix_zerocont', 'csa_fix_nosink', 'csa_fix_topk8', 'csa_fix_topk64', 'full_sink']
RUN_ABL = dict(RUN, outdir='results_lm_v4_abl', variants=ABL_VARIANTS, seeds=[0] if QUICK else [0, 1, 2, 3, 4])
RUN_LONG = dict(RUN, n_train_tokens=2000000 if QUICK else 110000000, steps=500 if QUICK else 20000, warmup=200, eval_every=1000, seeds=[0] if QUICK else [0, 1, 2, 3], outdir='results_lm_v3_long', variants=['full', 'full_matched', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic', 'hybrid_fixed', 'hybrid_dynamic', 'hybrid_csa_dyn'])
RUN_SCALE = dict(RUN, d=384, n_layers=8, n_heads=12, d_head=32, seq_len=1024, batch_size=8, n_train_tokens=2000000 if QUICK else 40000000, steps=500 if QUICK else 4000, warmup=100, eval_every=500, seeds=[0] if QUICK else [0, 1, 2], outdir='results_lm_v5_scale', variants=['full', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic', 'hybrid_dynamic'])

def block_readable(pos, last_tok):
    return pos[:, None] > last_tok[None, :]

def cosine_similarity_consecutive(H, eps=1e-08):
    h = F.normalize(H, dim=-1, eps=eps)
    return (h[1:] * h[:-1]).sum(dim=-1)

def _segment(n, want_cut_list, min_block, max_block):
    n_cuts = len(want_cut_list)
    bids = [0] * n
    honoured = {}
    cur, cur_len = (0, 1)
    pending_cut = False
    pending_slot = -1
    for t in range(1, n):
        cur_len += 1
        ci = t - 1
        if 0 <= ci < n_cuts and want_cut_list[ci]:
            pending_cut = True
            pending_slot = ci
        may = cur_len > min_block
        must = cur_len > max_block
        if must or (pending_cut and may):
            if pending_cut and 0 <= pending_slot < n_cuts:
                honoured[pending_slot] = 1.0
            cur += 1
            cur_len = 1
            pending_cut = False
            pending_slot = -1
        bids[t] = cur
    return (bids, honoured)

def _cut_merge_mask(want_cut_list, min_block, max_block, dtype=None, device=None):
    import torch as _t
    if dtype is None:
        dtype = _t.float32
    if isinstance(want_cut_list, _t.Tensor):
        lst = want_cut_list.detach().cpu().tolist()
    else:
        lst = list(want_cut_list)
    _bids, honoured = _segment(len(lst) + 1, lst, min_block, max_block)
    keep = [0.0] * len(lst)
    for slot in honoured:
        if 0 <= slot < len(keep):
            keep[slot] = 1.0
    return _t.tensor(keep, dtype=dtype, device=device)

def blocks_from_cuts(n, want_cut_list, min_block, max_block, device):
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if n == 1:
        return torch.zeros(1, dtype=torch.long, device=device)
    bids, _honoured = _segment(n, want_cut_list, min_block, max_block)
    bids_t = torch.tensor(bids, dtype=torch.long, device=device)
    return bids_t - bids_t.min()

def blocks_from_cosine(H, tau, min_block, max_block, sim=None):
    n = H.shape[0]
    dev = H.device
    if n <= 1:
        return torch.zeros(n, dtype=torch.long, device=dev)
    if sim is None:
        sim = cosine_similarity_consecutive(H)
    cut_list = (sim < tau).cpu().tolist()
    return blocks_from_cuts(n, cut_list, min_block, max_block, dev)

def adaptive_threshold(sim, target_block_tokens):
    if sim.numel() == 0:
        return 1.0
    q = max(1.0 / max(target_block_tokens, 1), 0.001)
    return float(torch.quantile(sim.detach().float().cpu(), q).item())

def causal_adaptive_threshold(sim, target_block_tokens):
    if sim.numel() == 0:
        return torch.empty(0, dtype=sim.dtype, device=sim.device)
    q = max(1.0 / max(target_block_tokens, 1), 0.001)
    vals = [torch.quantile(sim[:k + 1].detach().float(), q) for k in range(sim.numel())]
    return torch.stack(vals).to(sim.device)

def blocks_fixed(n, block_size, device):
    return torch.arange(n, device=device) // block_size
_MASK_CACHE = {}
_CAUSAL_DONOR = -1
_CAUSAL_TRI = None

def _causal_donor(device, width):
    global _CAUSAL_TRI
    dev_s = str(device)
    held = _CAUSAL_TRI
    if held is not None and held[0] == dev_s and (held[1].shape[0] >= int(width)):
        return (held[1], dev_s)
    tri = _mask_buffer(int(width), device, 'causal')
    _PINNED_KEYS.clear()
    _PINNED_KEYS.add((dev_s, 'causal', _CAUSAL_DONOR))
    _CAUSAL_TRI = (dev_s, tri)
    _cache_put(_MASK_CACHE, (dev_s, 'causal', _CAUSAL_DONOR), tri, _MASK_CACHE_BUDGET_BYTES, _mask_cache_total)
    return (tri, dev_s)
_IDX_CACHE = {}
_IDX_CACHE_BYTES = 0
_MASK_CACHE_BUDGET_BYTES = 192 << 20

def _cache_bytes(t):
    if t is None:
        return 0
    try:
        return int(t.numel()) * int(t.element_size())
    except Exception:
        return 0

def _mask_cache_total(cache):
    return sum((_cache_bytes(v) for v in cache.values()))

def _idx_cache_total(cache):
    return sum((_cache_bytes(v) + 128 * len(k) for k, v in cache.items()))

def _entry_bytes(cache, key, total_of):
    return total_of({key: cache[key]})

def _is_pinned_key(key):
    return key in _PINNED_KEYS

def _is_pinned(cache, key):
    return _is_pinned_key(key)
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
    key = (str(device), 'arange', int(n))
    a = _IDX_CACHE.get(key)
    if a is None:
        a = torch.arange(n, device=device)
        _cache_put(_IDX_CACHE, key, a, _MASK_CACHE_BUDGET_BYTES, _idx_cache_total)
    return a

def _range_cache(lo, hi, device):
    if lo == 0:
        return _arange_cache(hi, device)
    key = (str(device), 'arange2', int(lo), int(hi))
    a = _IDX_CACHE.get(key)
    if a is None:
        a = torch.arange(lo, hi, device=device)
        _cache_put(_IDX_CACHE, key, a, _MASK_CACHE_BUDGET_BYTES, _idx_cache_total)
    return a

def _attn_chunk_key(device=None):
    try:
        budget = _attn_transient_budget(device if device is not None else DEVICE)
    except Exception:
        return 'b?'
    if budget is None:
        return 'b0'
    n = int(budget)
    if n <= 0:
        return 'b0'
    return f'b{1 << n.bit_length() - 1}'

def _attn_transient_budget(device):
    key = (str(device), _ATB_EPOCH)
    if key in _ATB_CACHE:
        return _ATB_CACHE[key]
    val = None
    try:
        if device.type == 'cuda':
            free, _total = torch.cuda.mem_get_info(device)
            val = max(64 << 20, int(free) // 4)
    except Exception:
        val = None
    _ATB_CACHE[key] = val
    return val

def tok_sel_mask(n, w, device):
    pos = torch.arange(n, device=device)
    return (pos[None, :] <= pos[:, None]) & (pos[None, :] > pos[:, None] - w)

def _tri_full(n, device, k):
    m = torch.empty((n, n), device=device)
    m.fill_(float('-inf'))
    torch.triu(m, k, out=m) if k > 0 else torch.tril(m, k, out=m)
    return m

def _mask_buffer(n, device, kind):
    if kind == 'causal':
        return _tri_full(n, device, 1)
    key = (str(device), kind, int(n))
    buf = _MASK_CACHE.get(key)
    if buf is not None:
        return buf
    if kind == 'band':
        buf = _tri_full(n, device, -1)
    else:
        buf = torch.zeros((n, n), device=device)
    _cache_put(_MASK_CACHE, key, buf, _MASK_CACHE_BUDGET_BYTES, _mask_cache_total)
    return buf

def causal_mask(T, device):
    tri, _dev_s = _causal_donor(device, T)
    if tri.shape[0] == int(T):
        return tri
    return tri[:T, :T]

def window_band(T, window, device):
    if window <= 0 or window >= T:
        return _mask_buffer(T, device, 'zero')
    key = (str(device), 'bandw', int(T), int(window))
    buf = _MASK_CACHE.get(key)
    if buf is None:
        buf = _tri_full(T, device, -int(window))
        _cache_put(_MASK_CACHE, key, buf, _MASK_CACHE_BUDGET_BYTES, _mask_cache_total)
    return buf

def causal_window_mask(T, window, device):
    if window <= 0 or window >= T:
        return causal_mask(T, device)
    key = (str(device), 'cw', int(T), int(window))
    m = _MASK_CACHE.get(key)
    if m is None:
        m = causal_mask(T, device).clone()
        m.add_(window_band(T, window, device))
        _cache_put(_MASK_CACHE, key, m, _MASK_CACHE_BUDGET_BYTES, _mask_cache_total)
    return m

def _take_rows(x, idx):
    return torch.index_select(x, 0, idx)

def _take_2d(x, idx):
    return torch.index_select(x, 0, idx.flatten()).view(idx.shape[0], idx.shape[1], *x.shape[1:])

def _block_order(block_ids, n):
    if block_ids.numel() <= 1:
        return (_arange_cache(block_ids.numel(), block_ids.device), True)
    if bool((block_ids[1:] >= block_ids[:-1]).all()):
        return (_arange_cache(block_ids.numel(), block_ids.device), True)
    return (torch.argsort(block_ids, stable=True), False)

def pool_variable_blocks(Xa, Xb, Za, Zb, block_ids, B_pos_a, B_pos_b, overlap):
    n, Fd = Xa.shape
    dev = Xa.device
    B = int(block_ids.max().item()) + 1
    order, _ident = _block_order(block_ids, n)
    if _ident:
        Xas, Xbs, Zas, Zbs = (Xa, Xb, Za, Zb)
    else:
        Xas = _take_rows(Xa, order)
        Xbs = _take_rows(Xb, order)
        Zas = _take_rows(Za, order)
        Zbs = _take_rows(Zb, order)
    return _pool_ordered(Xas, Xbs, Zas, Zbs, block_ids, B_pos_a, B_pos_b, overlap, n, Fd, dev, B, order)

def _pool_ordered(Xas, Xbs, Zas, Zbs, block_ids, B_pos_a, B_pos_b, overlap, n, Fd, dev, B, order):
    counts = torch.bincount(block_ids, minlength=B)
    ends = torch.cumsum(counts, 0)
    starts = ends - counts
    _cmin, max_len = torch.stack(list(torch.aminmax(counts))).tolist()
    pos = _range_cache(0, max_len, dev)
    _gp = starts[:, None] + pos[None, :]
    g_a = _gp.clamp(max=n - 1)
    mask_a = _gp < ends[:, None]
    Xam = _take_2d(Xas, g_a)
    Zam = _take_2d(Zas, g_a)
    ov = overlap
    end_prev = torch.cat([torch.full((1,), -1, dtype=ends.dtype, device=dev), ends[:-1]])
    starts_prev = torch.cat([torch.zeros(1, dtype=starts.dtype, device=dev), starts[:-1]])
    ov_start = torch.clamp(end_prev - ov, min=starts_prev)
    ov_len = end_prev - ov_start
    _ov_lo = _ov_hi = None
    if ov_len.numel():
        _ov_lo, _ov_hi = torch.stack(list(torch.aminmax(ov_len))).tolist()
    _W_b = max(_ov_hi if _ov_hi is not None else 0, 0)
    op = _range_cache(0, _W_b, dev)
    g_b = (ov_start[:, None] + op[None, :]).clamp(max=n - 1)
    mask_b = op[None, :] < ov_len[:, None]
    Xbm = _take_2d(Xbs, g_b)
    Zbm = _take_2d(Zbs, g_b)
    b_off = ov_start[:, None] - starts_prev[:, None] + op[None, :]
    b_off = b_off.clamp(max=B_pos_b.shape[0] - 1)
    _lim = B_pos_a.shape[0] - 1
    Zam += B_pos_a[pos.clamp(max=_lim)]
    Zbm += B_pos_b[b_off]
    Zam.masked_fill_(~mask_a[:, :, None], float('-inf'))
    Zbm.masked_fill_(~mask_b[:, :, None], float('-inf'))
    WinZ = torch.cat([Zam, Zbm], 1)
    sc = F.softmax(WinZ, dim=1)
    sc = torch.nan_to_num(sc)
    sc_a = sc[:, :max_len]
    sc_b = sc[:, max_len:]
    if _cmin < max_len:
        _pad_a = slice(_cmin, max_len)
        Xam[:, _pad_a].masked_fill_(~mask_a[:, _pad_a, None], 0.0)
    if ov:
        _pad_b = max(_ov_lo if _ov_lo is not None else 0, 0)
        if _pad_b < _W_b:
            Xbm[:, _pad_b:].masked_fill_(~mask_b[:, _pad_b:, None], 0.0)
    if 0 < _W_b < ov:
        _prod_b = Xbm.new_zeros(Xbm.shape[0], ov, Xbm.shape[2])
        _prod_b[:, :_W_b] = sc_b * Xbm
        comp = (sc_a * Xam).sum(1) + _prod_b.sum(1)
    else:
        comp = (sc_a * Xam).sum(1) + (sc_b * Xbm).sum(1)
    last_idx = (ends - 1).clamp(max=n - 1)
    return (comp, order[last_idx], B)

def pool_blocks_single(X, Z, B_pos, block_ids):
    n = X.shape[0]
    dev = X.device
    B = int(block_ids.max().item()) + 1
    order, _ident = _block_order(block_ids, n)
    if _ident:
        Xs, Zs = (X, Z)
    else:
        Xs = _take_rows(X, order)
        Zs = _take_rows(Z, order)
    counts = torch.bincount(block_ids, minlength=B)
    ends = torch.cumsum(counts, 0)
    starts = ends - counts
    _cmin, max_len = torch.stack(list(torch.aminmax(counts))).tolist()
    pos = _range_cache(0, max_len, dev)
    _gp = starts[:, None] + pos[None, :]
    g = _gp.clamp(max=n - 1)
    mask = _gp < ends[:, None]
    Xb = _take_2d(Xs, g)
    Zb = _take_2d(Zs, g)
    WinZ = Zb
    WinZ[:, :max_len] += B_pos[pos.clamp(max=B_pos.shape[0] - 1)]
    WinZ.masked_fill_(~mask[:, :, None], float('-inf'))
    sc = torch.nan_to_num(F.softmax(WinZ, dim=1))
    if _cmin < max_len:
        _pad = slice(_cmin, max_len)
        Xb[:, _pad].masked_fill_(~mask[:, _pad, None], 0.0)
    comp = (sc * Xb).sum(1)
    last_idx = (ends - 1).clamp(max=n - 1)
    return (comp, order[last_idx], B)

def _rank_blocks(masked, B, ties):
    if ties == 'earliest':
        return torch.sort(masked, dim=1, stable=True, descending=True).indices.to(torch.int32)
    order = torch.sort(masked.flip(1), dim=1, stable=True, descending=True).indices
    return torch.sub(B - 1, order, out=order).to(torch.int32)

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
    if k >= B:
        masked = scores.masked_fill(~causal | ~finite, float('-inf'))
        order = _rank_blocks(masked, B, ties)
    else:
        masked = scores.masked_fill(~causal, float('-inf'))
        masked = masked.masked_fill(~finite, float('-inf'))
        order = _rank_blocks(masked, B, ties)
    usable = causal.gather(1, order) & finite.gather(1, order)
    if k >= B:
        keep = usable
    else:
        win = min(int(k), B)
        keep = usable[:, :win]
    dest = keep.cumsum(dim=1)
    dest -= 1
    dest.clamp_(min=0)
    _w_keep = min(int(k), B)
    grid = _arange_cache(n, scores.device)[:, None].expand(n, _w_keep)
    kept = torch.full((n, _w_keep), -1, dtype=torch.int32, device=scores.device)
    _keep_w = keep[:, :_w_keep]
    kept[grid[_keep_w], dest[:, :_w_keep][_keep_w]] = order[:, :_w_keep][_keep_w]
    del grid, _keep_w
    pad_blk = causal.to(torch.int32).argmax(dim=1).to(torch.int32)
    idx_out = torch.where(kept >= 0, kept, pad_blk[:, None])
    if out_valid is not None:
        valid_cols = kept >= 0
        if k_req > _w_keep:
            valid_cols = torch.cat([valid_cols, torch.zeros(n, k_req - _w_keep, dtype=torch.bool, device=scores.device)], dim=1)
        out_valid[:] = [valid_cols]
    if k_req > _w_keep:
        pad = pad_blk[:, None].expand(n, k_req - _w_keep)
        idx_out = torch.cat([idx_out, pad], dim=1)
    return idx_out

def lightning_indexer(H, comp_kv, last_tok, W_DQ, W_DK, W_w, nIH, topk, return_mask=True, query_chunk=2048, random_select=False, pre_qI=None, pre_w=None, out_valid=None):
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
    causal = block_readable(pos, last_tok)
    k = min(topk, B)
    score_chunks = []
    if random_select:
        _ri_key = (str(dev), 'rs_ri', int(n))
        _jc_key = (str(dev), 'rs_jc', int(B))
        _ri = _IDX_CACHE.get(_ri_key)
        if _ri is None:
            _ri = torch.arange(n, device=dev, dtype=torch.float64)[:, None]
            _cache_put(_IDX_CACHE, _ri_key, _ri, _MASK_CACHE_BUDGET_BYTES, _idx_cache_total)
        _jc = _IDX_CACHE.get(_jc_key)
        if _jc is None:
            _jc = (torch.arange(B, device=dev, dtype=torch.float64)[None, :] * 78.233).contiguous()
            _cache_put(_IDX_CACHE, _jc_key, _jc, _MASK_CACHE_BUDGET_BYTES, _idx_cache_total)
        sc_dtype = H.dtype
    if query_chunk is None:
        raise ValueError("lightning_indexer: query_chunk=None is not 'use the default'; pass the default (2048) explicitly or omit the argument")
    query_chunk = max(1, min(int(query_chunk), n))
    for s in range(0, n, query_chunk):
        e = min(s + query_chunk, n)
        if random_select:
            z = torch.sin(_ri[s:e] * 12.9898 + _jc) * 43758.5453
            z = (z - torch.floor(z)) * 2.0 - 1.0
            sc = z.to(sc_dtype)
        else:
            raw = torch.einsum('ind,ibd->inb', qI[:, s:e], kI) * hd ** (-0.5)
            F.relu(raw, inplace=True)
            sc = torch.einsum('inb,ni->nb', raw, w_idx[s:e])
        score_chunks.append(sc)
    scores = torch.cat(score_chunks, dim=0)
    out_dtype = scores.dtype
    with torch.no_grad():
        idx = _indexer_selection(scores, causal, k, out_valid=out_valid)
    scores.masked_fill_(~causal, float('-inf'))
    soft = F.softmax(scores, dim=-1)
    soft = torch.nan_to_num(soft)
    del scores, score_chunks
    m = None
    if return_mask:
        with torch.no_grad():
            m = torch.zeros(n, B, device=dev, dtype=out_dtype)
            m.scatter_(1, idx.long(), 1.0)
            m.masked_fill_(~causal, 0.0)
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
    _diff = torch.is_grad_enabled() and (getattr(logits, 'requires_grad', False) or (sink_logits is not None and getattr(sink_logits, 'requires_grad', False)))
    if _diff:
        return _sink_softmax_impl(logits, sink_logits, dim)
    with torch.no_grad():
        return _sink_softmax_impl(logits, sink_logits, dim).detach()

def _sink_softmax_impl(logits, sink_logits, dim=-1):
    if sink_logits is None:
        out = torch.softmax(logits, dim)
        if torch.isnan(out).any():
            return torch.nan_to_num(out)
        return out
    z = _SinkWiden.apply(logits, sink_logits)
    if not torch.isfinite(sink_logits).all():
        return torch.nan_to_num(torch.softmax(z, -1)[..., 1:])
    return torch.softmax(z, -1)[..., 1:]

def _sink_split_softmax(logits, sink_logits, want_sink=True):
    if sink_logits is None:
        return (sink_softmax(logits, None, -1), None)
    z = _SinkWiden.apply(logits, sink_logits)
    if not torch.isfinite(sink_logits).all():
        return (torch.nan_to_num(torch.softmax(z, -1)[..., 1:]), logits.new_zeros(logits.shape[:-1]) if want_sink else None)
    soft = torch.softmax(z, -1)[..., 1:]
    return (soft, 1.0 - soft.sum(-1) if want_sink else None)

def _block_token_attn(q, k_blk, v_blk, topk_mask, last_tok, k_sw, v_sw, w, scale, soft=None, sink_logits=None, q_chunk=128, mem_budget_bytes=None):
    n = q.shape[0]
    dev = q.device
    n_blk = k_blk.shape[0]
    rel = _range_cache(0, w, dev)
    K = torch.cat([k_blk, k_sw], 0)
    V = torch.cat([v_blk, v_sw], 0)
    pos = _arange_cache(n, dev)
    causal_blk = block_readable(pos, last_tok)
    win_key = pos[:, None] - (w - 1) + rel[None, :]
    win_valid = win_key >= 0
    win_idx = win_key.clamp(min=0)
    wv_all = win_valid
    _both_idx_all = torch.cat([_arange_cache(n_blk, dev).unsqueeze(0).expand(n, n_blk).to(torch.int32), (win_idx + n_blk).to(torch.int32)], 1)
    if mem_budget_bytes is None:
        mem_budget_bytes = _attn_transient_budget(dev)
    _heads = q.shape[1] if q.dim() == 3 else 1
    _M = n_blk + int(w)
    _es = q.element_size()
    _hd = int(q.shape[-1])
    _per_row = 4 * _heads * max(1, _M) + 2 * max(1, _M) * _heads * max(1, _hd) + (max(1, _M) * 4 + max(1, _M)) / _es
    _bytes_per_chunk_row = _per_row * _es
    _MIN_CHUNK = 64
    _bindable = mem_budget_bytes is not None and math.isfinite(float(mem_budget_bytes))
    if _bindable and _bytes_per_chunk_row > 0:
        _max_rows = max(_MIN_CHUNK, int(mem_budget_bytes) // _bytes_per_chunk_row)
        _new_chunk = min(q_chunk, _max_rows)
        if _new_chunk < q_chunk:
            print(f'[_block_token_attn] q_chunk {q_chunk} -> {_new_chunk} (M={_M}, w={w}, n={n})')
            q_chunk = _new_chunk
    q_chunk = max(1, min(int(q_chunk), n))
    out = torch.empty_like(q)
    _MINL = torch.finfo(q.dtype).min
    for s in range(0, n, q_chunk):
        e = min(s + q_chunk, n)
        both_idx = _both_idx_all[s:e]
        Kset = _take_2d(K, both_idx)
        Vset = _take_2d(V, both_idx)
        _wv = wv_all[s:e]
        sel = torch.cat([(topk_mask[s:e] > 0) & causal_blk[s:e], _wv], 1)
        logits = torch.einsum('nhd,nmhd->nhm', q[s:e], Kset)
        logits.mul_(scale)
        logits.masked_fill_(~sel[:, None, :], _MINL)
        attn, _sink_unused = _sink_split_softmax(logits, sink_logits, want_sink=False)
        if soft is not None:
            blk = logits[:, :, :n_blk]
            win = logits[:, :, n_blk:]
            soft_sel = soft[s:e] * topk_mask[s:e]
            soft_log = torch.log((1 - soft_sel).clamp_min(1e-12))
            soft_log.masked_fill_(soft_sel >= 1, _MINL)
            soft_blk = blk + soft_log[:, None, :]
            soft_win = win
            blk = None
            soft_logits = torch.cat([soft_blk, soft_win], -1)
            soft_attn, _sink_unused = _sink_split_softmax(soft_logits, sink_logits, want_sink=False)
            attn = soft_attn + (attn - soft_attn.detach())
        out[s:e] = torch.einsum('nhm,nmhd->nhd', attn, Vset)
    return out

def gathered_attention(q, k_blk, v_blk, topk_idx, last_tok, k_sw, v_sw, w, scale, q_chunk=128, soft=None, sink_logits=None, mem_budget_bytes=None, sel_valid=None):
    n = q.shape[0]
    dev = q.device
    if mem_budget_bytes is None:
        mem_budget_bytes = _attn_transient_budget(dev)
    _heads = q.shape[1] if q.dim() == 3 else 1
    _per_row = max(1, int(topk_idx.shape[1]) + int(w)) * _heads * int(q.shape[-1])
    _bytes_per_chunk_row = 2 * _per_row * q.element_size()
    _MIN_CHUNK = 64
    _bindable = mem_budget_bytes is not None and math.isfinite(float(mem_budget_bytes))
    if _bindable and _bytes_per_chunk_row > 0:
        _max_rows = max(_MIN_CHUNK, int(mem_budget_bytes) // _bytes_per_chunk_row)
        _new_chunk = min(q_chunk, _max_rows)
        if _new_chunk < q_chunk:
            print(f'[gathered_attention] q_chunk {q_chunk} -> {_new_chunk} (topk={topk_idx.shape[1]}, w={w}, n={n})')
            q_chunk = _new_chunk
    q_chunk = max(1, min(int(q_chunk), n))
    out = torch.empty_like(q)
    rel = _range_cache(0, w, dev)
    pos_all = _arange_cache(n, dev)
    n_blk = k_blk.shape[0]
    k_stack = torch.cat([k_blk, k_sw], 0)
    v_stack = torch.cat([v_blk, v_sw], 0)
    win_idx = pos_all[:, None] - (w - 1) + rel[None, :]
    win_idx = win_idx.clamp(min=0)
    _both_idx_all = torch.cat([topk_idx.to(torch.int32), (win_idx + n_blk).to(torch.int32)], dim=1)
    _MINL = torch.finfo(k_blk.dtype).min
    for s in range(0, n, q_chunk):
        e = min(s + q_chunk, n)
        qseg = q[s:e]
        pos = pos_all[s:e]
        ib = topk_idx[s:e].long()
        sel_blk = torch.gather(block_readable(pos, last_tok), 1, ib)
        keep = None
        if sel_valid is not None:
            sv = sel_valid[s:e]
            k_nb = int(ib.shape[1])
            nval = sv.sum(1)
            rep = _arange_cache(k_nb, dev)[None, :] == nval[:, None]
            pad_val = ib.gather(1, nval.clamp(max=k_nb - 1)[:, None])
            pad_dup = ((ib == pad_val) & sv).any(1, keepdim=True)
            keep = sv | (rep & ~pad_dup)
            sel_blk = sel_blk & keep
        win_global = pos[:, None] - (w - 1) + rel[None, :]
        win_valid = win_global >= 0
        both_idx = _both_idx_all[s:e]
        Kset = _take_2d(k_stack, both_idx)
        Vset = _take_2d(v_stack, both_idx)
        if soft is None:
            valid = torch.cat([sel_blk, win_valid], dim=1)
            logits = torch.einsum('qhd,qmhd->qhm', qseg, Kset)
            logits.mul_(scale)
            logits.masked_fill_(~valid[:, None, :], _MINL)
            attn, _sink_unused = _sink_split_softmax(logits, sink_logits, want_sink=False)
            out[s:e] = torch.einsum('qhm,qmhd->qhd', attn, Vset)
            continue
        soft_g = soft[s:e].gather(1, ib)
        if keep is not None:
            soft_g = soft_g * keep.to(soft_g.dtype)
        _nb = int(ib.shape[1])
        kb = Kset[:, :_nb]
        kw = Kset[:, _nb:]
        soft_log_g = torch.log((1 - soft_g).clamp_min(1e-12))
        soft_log_g.masked_fill_(soft_g >= 1, _MINL)
        blk_logits = torch.einsum('qhd,qmhd->qhm', qseg, kb)
        blk_logits.mul_(scale)
        win_logits = torch.einsum('qhd,qmhd->qhm', qseg, kw)
        win_logits.mul_(scale)
        logits = torch.cat([blk_logits, win_logits], -1)
        soft_logits = torch.cat([logits[:, :, :_nb] + soft_log_g[:, None, :], win_logits], -1)
        soft_valid = torch.cat([sel_blk, win_valid], dim=1)
        valid = soft_valid
        soft_logits.masked_fill_(~soft_valid[:, None, :], _MINL)
        soft_attn, _sink_unused = _sink_split_softmax(soft_logits, sink_logits, want_sink=False)
        logits.masked_fill_(~valid[:, None, :], _MINL)
        attn, _sink_unused = _sink_split_softmax(logits, sink_logits, want_sink=False)
        attn = soft_attn + (attn - soft_attn.detach())
        out[s:e] = torch.einsum('qhm,qmhd->qhd', attn, Vset)
    return out

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
            self.sink = nn.Parameter(torch.zeros(n_heads))
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
        self.sink = nn.Parameter(torch.zeros(n_heads))
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
        q = self.W_q(x).view(B, T, self.nh, self.hd)
        k = self.W_k(x).view(B, T, self.nh, self.hd)
        v = self.W_v(x).view(B, T, self.nh, self.hd)
        if self.cfg.full_cosine:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            scale = 1.0
        else:
            scale = 1.0 / math.sqrt(self.hd)
        logits = torch.einsum('bnhd,bmhd->bhnm', q, k)
        logits.mul_(scale)
        if self.cfg.window > 0:
            mask = causal_window_mask(T, self.cfg.window, x.device)
        else:
            mask = causal_mask(T, x.device)
        if self.sink is not None:
            sculpt = self.sink.view(1, self.nh, 1, 1).expand(B, self.nh, T, 1)
            logits.add_(mask)
            attn = torch.softmax(torch.cat([sculpt, logits], -1), -1)[..., 1:]
        else:
            logits.add_(mask)
            attn = torch.softmax(logits, -1)
        out = torch.einsum('bhnm,bmhd->bnhd', attn, v)
        return self.W_o(out.reshape(B, T, self.nh * self.hd))

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
        pre['q'] = self.W_q(x)
        pre['Ca'] = x @ self.W_aKV
        pre['Cb'] = x @ self.W_bKV if self.W_bKV is not None else None
        pre['Za'] = x @ self.W_aZ if self.W_aZ is not None else None
        pre['Zb'] = x @ self.W_bZ if self.W_bZ is not None else None
        if cfg.kind != 'hca':
            pre['qI'] = x @ self.W_DQ
            pre['w_idx'] = F.linear(x, self.W_w.weight)
        if self.fuse_conv is not None:
            k = cfg.fuse_kernel
            fx = fuse_x
            z = F.pad(fx.transpose(1, 2), (k - 1, 0))
            pre['fused'] = F.conv1d(z, self.fuse_conv.weight, groups=fx.shape[-1]).transpose(1, 2)
        return pre

    def _single(self, x, pre=None):
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
            fused = pre['fused'] if pre is not None and 'fused' in pre else self._fuse(x)
            sim = cosine_similarity_consecutive(fused)
            if sim.numel() == 0:
                tau = torch.tensor(0.0, device=x.device)
                gate = torch.tensor([], device=x.device)
            else:
                prefix_mean = torch.cumsum(sim, 0) / _range_cache(1, sim.numel() + 1, x.device)
                tau = prefix_mean + self.delta_logit
                gate = torch.sigmoid((tau - sim) / max(cfg.temperature, 0.001))
            if gate.numel() and self.need_reg:
                hard_b = gate.detach() > _HALF
                hard = hard_b.to(gate.dtype)
                _hon = _cut_merge_mask(hard, cfg.min_block, cfg.max_block, dtype=gate.dtype, device=gate.device)
                _soft_honoured = gate * _hon
                gate_mean = ((hard * _hon).sum() + _soft_honoured.sum() - _soft_honoured.detach().sum()) / T
            else:
                gate_mean = torch.zeros((), device=x.device) if self.need_reg else None
            with torch.no_grad():
                gate_bool = locals().get('hard_b')
                if gate_bool is None:
                    gate_bool = gate.detach() > _HALF
                gate_list = gate_bool.cpu().tolist()
                bid = blocks_from_cuts(T, gate_list, cfg.min_block, cfg.max_block, x.device)
        elif cfg.chunking in ('cosine_abs', 'cosine_adaptive') or (cfg.dynamic and cfg.kind != 'hca'):
            sim = cosine_similarity_consecutive(x)
            tau = causal_adaptive_threshold(sim, cfg.target_block_tokens) if cfg.chunking == 'cosine_adaptive' or cfg.adaptive else cfg.cos_threshold
            bid = blocks_from_cosine(x, tau, cfg.min_block, cfg.max_block, sim=sim)
        else:
            bid = blocks_fixed(T, cfg.block_size, x.device)
        if pre is not None:
            Za, Zb = (pre['Za'], pre['Zb'])
        else:
            Za = x @ self.W_aZ
            Zb = x @ self.W_bZ if self.W_bZ is not None else None
        if cfg.kind == 'hca':
            comp_kv, last_tok, Bn = pool_blocks_single(Ca_raw, Za, self.B_pos_a, bid)
        else:
            comp_kv, last_tok, Bn = pool_variable_blocks(Ca_raw, Cb_raw, Za, Zb, bid, self.B_pos_a, self.B_pos_b, cfg.overlap)
        index_kv = comp_kv
        attn_kv = torch.zeros_like(comp_kv) if cfg.content_mode == 'zero' else comp_kv
        if self._stats is not None:
            self._stats.append(torch.bincount(bid).float().cpu())
            if self._cuts is not None:
                cuts = (bid[1:] != bid[:-1]).nonzero(as_tuple=True)[0] + 1
                self._cuts.append(cuts.cpu())
        core_kv = F.normalize(attn_kv, dim=-1)
        kv_local = F.normalize(Ca_raw, dim=-1)
        k_blk, v_blk = self._split(self.W_kvhead(core_kv))
        k_sw, v_sw = self._split(self.W_kvhead(kv_local))
        k_blk = F.normalize(k_blk, dim=-1)
        k_sw = F.normalize(k_sw, dim=-1)
        if cfg.kind == 'hca':
            topk_idx = _arange_cache(Bn, x.device).unsqueeze(0).expand(T, Bn)
            if T > 1024:
                out = gathered_attention(q, k_blk, v_blk, topk_idx, last_tok, k_sw, v_sw, cfg.sliding_window, scale, sink_logits=self.sink if cfg.use_sink else None)
            else:
                topk_mask = block_readable(_arange_cache(T, x.device), last_tok)
                out = _block_token_attn(q, k_blk, v_blk, topk_mask, last_tok, k_sw, v_sw, cfg.sliding_window, scale, sink_logits=self.sink if cfg.use_sink else None)
        else:
            topk = cfg.index_topk
            _sel_valid_box = [] if T > 1024 else None
            topk_mask, topk_idx, soft = lightning_indexer(x, index_kv, last_tok, self.W_DQ, self.W_DK, self.W_w.weight, cfg.n_index_heads, topk, return_mask=T <= 1024, random_select=cfg.indexer_mode == 'random', pre_qI=pre['qI'] if pre is not None else None, pre_w=pre['w_idx'] if pre is not None else None, out_valid=_sel_valid_box)
            if T > 1024:
                out = gathered_attention(q, k_blk, v_blk, topk_idx, last_tok, k_sw, v_sw, cfg.sliding_window, scale, soft=soft, sink_logits=self.sink if cfg.use_sink else None, sel_valid=_sel_valid_box[0])
            else:
                out = _block_token_attn(q, k_blk, v_blk, topk_mask, last_tok, k_sw, v_sw, cfg.sliding_window, scale, soft=soft, sink_logits=self.sink if cfg.use_sink else None)
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
        outs = None
        gates = []
        for b in range(B):
            pre = None if pre_all is None else {kk: v[b] if v is not None else None for kk, v in pre_all.items()}
            o, g = self._single(x[b], pre=pre)
            if outs is None:
                outs = torch.empty(B, *o.shape, device=o.device, dtype=o.dtype)
            outs[b].copy_(o)
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
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g

class MLP(nn.Module):

    def __init__(self, d, hidden):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class Block(nn.Module):

    def __init__(self, d, n_heads, d_head, cfg: AttnCfg, mlp_ratio=4):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = HybridAttention(d, n_heads, d_head, cfg)
        self.n2 = RMSNorm(d)
        self.mlp = MLP(d, int(d * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x

class SmallGPT(nn.Module):

    def __init__(self, vocab, d, n_layers, n_heads, d_head, max_seq, layer_cfgs, mlp_ratio=4):
        super().__init__()
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
            self.blocks.append(self._make_block(d, n_heads, d_head, c, mlp_ratio))
        self.norm = RMSNorm(d)
        self.max_seq = max_seq

    def _make_block(self, d, n_heads, d_head, cfg, mlp_ratio):
        return Block(d, n_heads, d_head, cfg, mlp_ratio)

    def forward(self, ids):
        T = ids.shape[1]
        x = self.tok(ids) + self.pos(_arange_cache(T, ids.device))
        for blk in self.blocks:
            x = blk(x)
        reg = None
        for blk in self.blocks:
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
        return self.head(self.norm(x))
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

def load_wikitext(seq_len, n_train_tokens, vocab_size=8192, val_seqs=512, cache_dir='./wt103_cache'):
    from datasets import load_dataset
    os.makedirs(cache_dir, exist_ok=True)
    tag = f'v3_sl{seq_len}_cap{n_train_tokens}_v{vocab_size}_vs{val_seqs}'
    tr_path = os.path.join(cache_dir, f'train_ids_{tag}.npy')
    va_path = os.path.join(cache_dir, f'val_batch_{tag}.npy')
    vp_path = os.path.join(cache_dir, f'val_bnd_{tag}.npy')
    me_path = os.path.join(cache_dir, f'meta_{tag}.json')
    if all((os.path.exists(p) for p in (tr_path, va_path, vp_path, me_path))):
        print('[data] loading cached token ids ...')
        train_ids = np.load(tr_path)
        val_batch = np.load(va_path)
        val_bnd = np.load(vp_path)
        vocab = json.load(open(me_path, encoding='utf-8'))['vocab']
        return (train_ids, val_batch, vocab, None, val_bnd)
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
    del fit_text
    gc.collect()
    eos_id = tok.token_to_id('<eos>')

    def stream_tokenise(split, cap_tokens, label, _ds=ds, _tok=tok, _eos=eos_id):
        out = []
        chunk = 10000
        for start in range(0, len(_ds[split]), chunk):
            rows = _ds[split][start:start + chunk]['text']
            for r in rows:
                if r.strip():
                    out.extend(_tok.encode(r).ids)
                    out.append(_eos)
            if len(out) >= cap_tokens:
                break
            if start and start % 50000 == 0:
                print(f'    [{label}] {start} rows, {len(out)} tokens')
        return np.array(out[:cap_tokens], dtype=np.int64)
    print(f'[data] tokenising train (cap {n_train_tokens} tokens) ...')
    train_ids = stream_tokenise('train', n_train_tokens + seq_len, 'train')
    print(f'[data] train tokens = {len(train_ids)}  vocab = {vocab}')
    print('[data] tokenising validation ...')
    val_ids = stream_tokenise('validation', val_seqs * seq_len + seq_len, 'val')
    n_val = len(val_ids) // seq_len
    val_batch = val_ids[:n_val * seq_len].reshape(n_val, seq_len)
    print(f'[data] val sequences = {n_val} (seq_len={seq_len})')
    bnd_ids = sorted(boundary_token_ids(tok, eos_id))
    val_bnd = np.isin(val_batch, bnd_ids)
    print(f'[data] boundary tokens: {int(val_bnd.sum())} ({val_bnd.mean() * 100:.1f}% of val tokens)')
    np.save(tr_path, train_ids)
    np.save(va_path, val_batch)
    np.save(vp_path, val_bnd)
    atomic_write_json(me_path, {'vocab': vocab}, indent=0)
    print('[data] cached token ids to disk (resume-safe)')
    del ds
    gc.collect()
    return (train_ids, val_batch, vocab, tok, val_bnd)

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

def batch_iter(train_ids, seq_len, batch_size, device, seed=0):
    rng = np.random.default_rng(seed)
    n = len(train_ids) - seq_len - 1
    if n <= 0:
        raise ValueError(f'batch_iter needs len(train_ids) > seq_len+1 to sample causal windows, got len={len(train_ids)} seq_len={seq_len}. Increase n_train_tokens or reduce seq_len.')
    while True:
        starts = rng.integers(0, n, size=batch_size)
        ids = np.stack([train_ids[s:s + seq_len + 1] for s in starts])
        ids = torch.from_numpy(ids).to(device)
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
PAIR_BUDGET_KEYS = ('steps', 'tokens_seen')

def pair_reason(ra, rb):
    if not isinstance(ra, dict) or not isinstance(rb, dict):
        return 'a record is not a dict'
    if ra.get('run_cfg') != rb.get('run_cfg'):
        return 'different run_cfg'
    bad = [k for k in PAIR_BUDGET_KEYS if ra.get(k) is None or rb.get(k) is None or ra.get(k) != rb.get(k)]
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
        if ppl_is_usable(r.get('ppl')) and (not r.get('synthesized')):
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
        out.setdefault(_v, {})[_s] = r if full else r['ppl']
        if full:
            r.setdefault('protocol_tag', _t)
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

def add_paired(*_a, **_kw):
    raise NotImplementedError('use v9_supp.add_paired (exp_lib cannot import v7_supp)')

def status_helper_selftest():
    cur = CODE_SEMANTICS
    assert not result_is_current({}, cur, 'ppl')
    assert not result_is_current({'ppl': 1.0}, cur, 'ppl')
    assert not result_is_current({'_code': 'v11.3', 'ppl': 1.0}, cur, 'ppl')
    assert not result_is_current({'_code': cur}, cur, 'ppl')
    assert result_is_current({'_code': cur, 'ppl': 1.0}, cur, 'ppl')
    assert not result_is_current({'_code': cur, 'ppl': float('nan')}, cur, 'ppl')
    assert not result_is_current({'_code': cur, 'ppl': float('inf')}, cur, 'ppl')
    assert not result_is_current({'_code': cur, 'ppl': 0.0}, cur, 'ppl')
    assert not result_is_current({'_code': cur, 'ppl': -1.0}, cur, 'ppl')
    assert result_is_current({'_code': cur, 'by_len': {}}, cur, 'by_len')
    assert not result_is_current({'_code': 'v11.4', 'ppl': 1.0}, cur, 'ppl') or cur == 'v11.4'
    return True

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    _attb_bump_epoch()
    torch.manual_seed(seed)
    _pin_cpu_threads()
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        _pin_cuda_determinism()
_CPU_THREADS_PINNED = None

def _pin_cpu_threads():
    global _CPU_THREADS_PINNED
    if _CPU_THREADS_PINNED is None:
        want = os.environ.get('CSA_CPU_THREADS', '').strip()
        try:
            n = max(1, int(want)) if want else 1
        except ValueError:
            print(f'[determinism] CSA_CPU_THREADS={want!r} is not an integer — using 1')
            n = 1
        os.environ['OMP_NUM_THREADS'] = str(n)
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

@torch.no_grad()
def eval_ppl(model, val_batch, device, chunk=64, eval_rows=None, eval_seed=0):
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
                _sel_sorted = torch.sort(_sel).values
                print(f'[eval_ppl] scoring a uniform sample of {_k}/{_flat} target rows (eval_rows={int(eval_rows)}, eval_seed={int(eval_seed)}) — this is a SAMPLED estimate, not the full-set PPL')
            else:
                print(f'[eval_ppl] eval_rows={int(eval_rows)} covers the whole {_flat}-row target space; scoring the FULL set (this is the population statistic, not a sample)')
        for i in range(0, val_batch.shape[0], chunk):
            ids = torch.from_numpy(val_batch[i:i + chunk]).to(device)
            logits = model(ids)
            if V is None:
                V = logits.size(-1)
            _span = max(0, ids.shape[1] - 1)
            if _span == 0:
                del ids, logits
                continue
            if _sel is not None:
                _base = i * _span
                _wide = int(ids.shape[0]) * _span
                _m = (_sel_sorted >= _base) & (_sel_sorted < _base + _wide)
                _local = _sel_sorted[_m] - _base
                if _local.numel() == 0:
                    del ids, logits
                    continue
                _local = _local.to(logits.device)
                _ri = torch.div(_local, _span, rounding_mode='floor')
                _ci = _local - _ri * _span
                _rows = logits[_ri, _ci, :]
                _tgts = ids[_ri, _ci + 1]
                del _ri, _ci
            else:
                _rows = logits[:, :_span, :].reshape(-1, logits.size(-1))
                _tgts = ids[:, 1:].reshape(-1)
            del ids, logits
            g = F.log_softmax(_rows, dim=-1).gather(1, _tgts[:, None]).squeeze(1)
            del _rows, _tgts
            nll -= g.double().sum()
            ntok += int(g.numel())
            del g
    finally:
        for _a, _prev in zip(need_reg_layers, need_reg_prior):
            _a.need_reg = _prev
        model.train(was_training)
    return math.exp(nll.item() / max(ntok, 1))

def enable_block_stats(model, on):
    for blk in model.blocks:
        blk.attn._stats = [] if on else None
        blk.attn._cuts = [] if on else None

def boundary_alignment(pred_cuts, gt_mask_row, n, tol=1, provenance=None):
    gt = np.nonzero(gt_mask_row[:n - 1])[0] + 1
    pred = np.asarray(pred_cuts)
    P, G = (len(pred), len(gt))
    if P == 0 or G == 0:
        if provenance is not None:
            provenance['bnd_rand_exact'] = True
        return (0.0, 0.0, 0.0, 0.0)
    hits, ghits = _greedy_match_counts(pred, gt, tol)
    prec, rec = (hits / P, ghits / G)
    f1 = 2 * prec * rec / max(prec + rec, 1e-09)
    near = np.zeros(n + 1, dtype=bool)
    for g in gt:
        near[max(g - tol, 1):g + tol + 1] = True
    n_pos = max(n - 1, 1)
    _approx = []
    rand_prec = _random_cut_precision(P, near, n_pos, tol, G, gt, approx_flag=_approx)
    if provenance is not None:
        provenance['bnd_rand_exact'] = not _approx
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
            step = 2 * tol + 1
            k = -(-L // step)
            span = (k - 1) * step
            off = (L - 1 - span) // 2
            if s == 0:
                off = 0
            elif e == m.size:
                off = L - 1 - span
            out.extend((s + off + i * step for i in range(k)))
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
    return np.nonzero(m)[0].astype(int) + 1

def _random_cut_precision(P, near, n_pos, tol, G, gt_bounds=None, approx_flag=None):
    if P <= 0 or G <= 0 or n_pos <= 0:
        return 0.0
    if gt_bounds is None:
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
    if P >= n_pos:
        return float(min(gt.size, P) / P)
    gt = np.unique(gt)
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
_AMP_CAP = 16
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
        per = max(1, _AMP_FALLBACK_DRAWS // nb_r)
        order = sorted(range(nb_r), key=lambda i: -wts[i][1])
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

def _score_subset_components(sub, tables, cl_t, tol, _pre):
    owner = {}
    for pos, bnd in tables:
        b_t = tuple(bnd)
        c_pre = _sorted_pre(b_t)
        for p in pos:
            owner[p] = (b_t, c_pre)
    by_comp = {}
    for p in sub:
        hit = owner.get(p)
        if hit is None:
            continue
        by_comp.setdefault(hit[0], (hit[1], []))[1].append(p)
    total = 0
    for b_t, (c_pre, pts) in by_comp.items():
        total += _match_count(tuple(sorted(pts)), b_t, tol, c_pre)
    return total

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

def _match_count(cuts, bounds, tol, _pre=None):
    return _greedy_match_counts(cuts, bounds, tol, _pre=_pre)[0]

def _greedy_match_counts(pred, gt, tol, _pre=None):
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
    try:
        return _compression_report_impl(model, val_batch, device, n_sample, val_bnd, bnd_tol)
    finally:
        enable_block_stats(model, False)
        model.train(was_training)

@torch.no_grad()
def _compression_report_impl(model, val_batch, device, n_sample, val_bnd, bnd_tol):
    report = {}
    seq_len = val_batch.shape[1]
    if int(n_sample) < 1:
        raise ValueError(f'compression_report: n_sample must be >= 1, got {n_sample!r}; the report is an average over the sequences it scores, so there is nothing to measure with no sequence scored')
    n_used = min(n_sample, val_batch.shape[0])
    with torch.no_grad():
        ids = torch.from_numpy(val_batch[:n_used]).to(device)
        _ = model(ids)
        del ids
    for li, blk in enumerate(model.blocks):
        attn = blk.attn
        st = attn._stats
        kind = attn.cfg.kind
        if st is None or kind == 'full' or len(st) == 0:
            continue
        lens = torch.cat(list(st)).float()
        n_blocks = sum((t.numel() for t in st)) / len(st)
        entry = {'blocks': n_blocks, 'avg_len': seq_len / max(n_blocks, 1.0), 'len_mean': float(lens.mean()), 'len_std': float(lens.std()) if lens.numel() > 1 else 0.0, 'len_min': int(lens.min()), 'len_max': int(lens.max()), 'frac_at_min': float((lens <= attn.cfg.min_block).float().mean()), 'frac_at_max': float((lens >= attn.cfg.max_block).float().mean())}
        if attn.cfg.dynamic and getattr(attn, 'delta_logit', None) is not None:
            entry['delta'] = float(attn.delta_logit.item())
        if val_bnd is not None and attn._cuts:
            n_seq_ref = min(len(attn._cuts), val_bnd.shape[0])
            if n_seq_ref < len(attn._cuts):
                print(f'[compression_report] WARNING: {len(attn._cuts)} sequences of cuts but only {val_bnd.shape[0]} reference masks — scoring the first {n_seq_ref} only')
            ps, rs, fs, rps = ([], [], [], [])
            _cuts = attn._cuts
            _prov = {}
            for si in range(n_seq_ref):
                c = _cuts[si]
                if hasattr(c, 'numpy'):
                    c = c.numpy()
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
    return any((k in name for k in ('delta_logit', 'B_pos', 'W_aZ', 'W_bZ', 'sink', 'tok.weight', 'pos.weight', '.g', 'fuse_conv', 'q_norm', 'kv_norm')))

def is_delta_param(name):
    return 'delta_logit' in name

def train_variant(variant, train_ids, val_batch, vocab, *, seed=0, d=256, n_layers=6, n_heads=8, d_head=32, seq_len=512, batch_size=12, steps=1500, lr=0.0003, weight_decay=0.1, warmup=50, comp_lambda=0.05, delta_lr_mult=10.0, eval_every=0, eval_subset=128, val_bnd=None, device=DEVICE, log_every=100, mlp_ratio=4, deadline_ts=None, return_model=False):
    set_seed(seed)
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
    opt = torch.optim.AdamW([{'params': decay_params, 'weight_decay': weight_decay, 'lr_scale': 1.0}, {'params': ndecay_params, 'weight_decay': 0.0, 'lr_scale': 1.0}, {'params': delta_params, 'weight_decay': 0.0, 'lr_scale': delta_lr_mult}], lr=lr)
    if delta_params:
        print(f'    delta_logit params: {len(delta_params)} (lr x{delta_lr_mult:g})')

    def lr_at(step):
        if warmup > 0 and step < warmup:
            return lr * (step + 1) / warmup
        t = (step - warmup) / max(steps - warmup, 1)
        return lr * (0.1 + 0.45 * (1.0 + math.cos(math.pi * t)))
    bpe = batch_iter(train_ids, seq_len, batch_size, device, seed=seed)
    t0 = time.time()
    losses = []
    ppl_hist = []
    delta_trace = {}
    model.train()
    truncated = False
    for step in range(steps):
        if deadline_ts is not None and time.time() > deadline_ts:
            print(f'  [budget] HARD cap reached before step {step} - truncating; this run is NOT recorded (raise BUDGET and re-run to retry it)')
            truncated = True
            break
        base = lr_at(step)
        for g in opt.param_groups:
            g['lr'] = base * g.get('lr_scale', 1.0)
        x, y = next(bpe)
        logits = model(x)
        ce = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        loss = ce + comp_lambda * model.comp_reg
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
        _lv = losses[-1]
        if eval_every and val_batch is not None and ((step + 1) % eval_every == 0 or step == steps - 1):
            sub_ppl = eval_ppl(model, val_batch[:eval_subset], device)
            ppl_hist.append([step + 1, float(sub_ppl)])
        if log_every and (step % log_every == 0 or step == steps - 1):
            for li, blk in enumerate(model.blocks):
                dl = getattr(blk.attn, 'delta_logit', None)
                if dl is not None:
                    delta_trace.setdefault(f'L{li}', []).append([step, float(dl.item())])
            print(f'  step {step:4d}  loss {_lv:.4f}  lr {opt.param_groups[0]['lr']:.2e}  ({(time.time() - t0) / max(step + 1, 1) * 1000:.0f}ms/step)')
    wall = time.time() - t0
    if truncated:
        del model, opt, bpe
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return {'variant': variant, 'seed': seed, 'budget_truncated': True, 'steps_done': step, 'train_time_s': wall}
    if deadline_ts is not None and time.time() > deadline_ts:
        print('  [budget] HARD cap reached before the final evaluation - truncating; this run is NOT recorded (raise BUDGET and re-run to retry it)')
        del model, opt, bpe
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return {'variant': variant, 'seed': seed, 'budget_truncated': True, 'steps_done': steps, 'train_time_s': wall}
    if val_batch is None:
        ppl, stats = (float('nan'), {})
    else:
        ppl = eval_ppl(model, val_batch, device)
        stats = compression_report(model, val_batch, device, val_bnd=val_bnd)
        print(f'[{variant} seed={seed}] val PPL = {ppl:.3f}')
        print(f'    block-length diagnostics: {json.dumps(stats, default=float)}')
    losses = [float(v) for v in losses]
    result = {'variant': variant, 'seed': seed, 'ppl': ppl, 'params': n_param, 'losses': losses, 'stats': stats, 'steps': steps, 'tokens_seen': steps * batch_size * seq_len, 'train_time_s': wall, 'final_loss_smoothed': float(np.mean(losses[-50:])), 'ppl_history': ppl_hist, 'delta_trace': delta_trace}
    if return_model:
        result['_model'] = model
    else:
        del model
    del opt, bpe
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result

class CostGuard:
    BASE = (256, 6, 512, 12)

    def __init__(self, budget):
        self.total_yuan = float(budget['total_yuan'])
        self.price = float(budget['price_per_hour'])
        self.margin = float(budget.get('margin', 0.93))
        self.already = float(budget.get('already_spent_yuan', 0.0))
        self.state_path = budget.get('state_path', 'autodl_budget_state.json')
        self.state = {'booked_seconds': 0.0, 'norm_sps': None, 'sps_by_class': {}, 'runs': 0}
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, encoding='utf-8') as f:
                    self.state.update(json.load(f))
            except Exception as e:
                print(f'[budget] FATAL: {self.state_path} exists but cannot be parsed ({type(e).__name__}: {e}).  Refusing to start with a silently RESET ledger (spent would read as 0) — move it aside to start fresh.')
                raise

    def _save(self):
        atomic_write_json(self.state_path, self.state)

    def spent_yuan(self):
        return self.already + self.state['booked_seconds'] / 3600.0 * self.price

    def remaining_yuan(self):
        return self.total_yuan - self.spent_yuan()

    def cap_yuan(self):
        return self.total_yuan * self.margin

    def record_run(self, seconds, steps_done, d, n_layers, seq_len, batch_size):
        self.state['booked_seconds'] += float(seconds)
        self.state['runs'] += 1
        if steps_done and steps_done > 0:
            f = self._wallclock_factor(d, n_layers, seq_len, batch_size)
            sps = seconds / steps_done
            norm = sps / f
            prev = self.state.get('norm_sps')
            self.state['norm_sps'] = norm if prev is None else 0.7 * prev + 0.3 * norm
            by_class = self.state.setdefault('sps_by_class', {})
            key = str((int(d), int(n_layers), int(seq_len), int(batch_size)))
            cprev = by_class.get(key)
            by_class[key] = sps if cprev is None else 0.7 * cprev + 0.3 * sps
        self._save()

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

    def can_start(self, est_seconds):
        projected = self.spent_yuan() + est_seconds / 3600.0 * self.price
        return projected <= self.cap_yuan()

    def deadline_ts(self):
        headroom = max(self.cap_yuan() - self.spent_yuan(), 0.0)
        return time.time() + headroom / self.price * 3600.0

    def report(self):
        print('\n==================== AutoDL budget report ====================')
        print(f'  booked GPU time : {self.state['booked_seconds'] / 3600:.2f} h ({self.state.get('runs', 0)} runs)')
        print(f'  cost so far     : ¥{self.spent_yuan():.2f} of ¥{self.total_yuan:.2f} (planning cap ¥{self.cap_yuan():.2f})')
        for key, sps in sorted((self.state.get('sps_by_class') or {}).items()):
            print(f'  calibrated speed: {sps:.3f} s/step  (d, L, seq, batch = {key})')
        if self.state.get('norm_sps'):
            print(f'  normalised speed: {self.state['norm_sps']:.3f} s/step (base class d=256 / 6L / seq512 / batch12)')
        print('  REMINDER: AutoDL bills the instance while it is ON — shut it')
        print('  down (关机) now that the run is done; only the data disk')
        print('  keeps billing after shutdown.')
        print('================================================================')
PARAM_MATCHED = {'full_matched', 'full_sw128_matched'}
PARAM_MATCHED_V7 = {'full_rope', 'full_sw128_matched_rope'}

def variant_mlp_ratio(variant, vocab, *, d=256, n_layers=6, n_heads=8, d_head=32, seq_len=512, matched=None, ref_variant='csa_dynamic'):
    if matched is None:
        matched = PARAM_MATCHED
    if variant not in matched:
        return 4.0

    def n_params(v, ratio):
        cfgs = make_layer_cfgs(n_layers, v)
        m = SmallGPT(vocab, d, n_layers, n_heads, d_head, seq_len, cfgs, mlp_ratio=ratio)
        n = count_params(m)
        del m
        return n
    p_full = n_params(variant, 4)
    p_ref = n_params(ref_variant, 4)
    ratio = 4 + (p_ref - p_full) / (2.0 * d * d * n_layers)
    print(f'[{variant}] matched mlp_ratio = {ratio:.2f} ({p_full / 1000000.0:.2f}M -> target {p_ref / 1000000.0:.2f}M params vs {ref_variant})')
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
        return m.group(1)
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
        v = rec.get('variant', key)
        per_variant.setdefault((v, _protocol_tag(rec, key)), []).append(rec)
    _dups = {}
    for k, recs in per_variant.items():
        cnt = {}
        for r in recs:
            s = r['seed']
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
        return json.dumps([rec.get('run_cfg'), _p, [rec.get(k) for k in PAIR_BUDGET_KEYS]], sort_keys=True, default=str)

    def _budget_ok(recs):
        if len(recs) < 2:
            return True
        ref = recs[0]

        def _pnum(r):
            v = r.get('params')
            if v is None or isinstance(v, bool):
                return None
            try:
                return int(v)
            except (TypeError, ValueError):
                return None
        _ref_p = _pnum(ref)
        for r in recs[1:]:
            if ref.get('run_cfg') != r.get('run_cfg'):
                return False
            _r_p = _pnum(r)
            if _ref_p is not None and _r_p is not None and (_ref_p != _r_p):
                return False
            for k in PAIR_BUDGET_KEYS:
                a, b = (ref.get(k), r.get(k))
                if (a is None) != (b is None) or (a is not None and a != b):
                    return False
        return True

    def _by_seed(recs):
        out, dup = ({}, [])
        for r in recs:
            s = r['seed']
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
        if _budget_ok(recs):
            _split_groups.append(((v, tag, None), recs))
            _group_records[v, tag, None] = recs
            continue
        _subs = {}
        for r in recs:
            _subs.setdefault(_cfg_fp(r), []).append(r)
        _ordered = sorted(_subs.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        print(f'[aggregate] WARNING: `{_agg_key(v, tag)}` holds records from {len(_ordered)} different run_cfg/budget/parameter-count configurations — pooling them would average incompatible runs (or two different MODEL WIDTHS, which are not two seeds of one model) and fake extra seeds.  Reported per configuration instead.')
        for _i, (_fp, _rs) in enumerate(_ordered):
            _split_groups.append(((v, tag, _i), _rs))
            _group_records[v, tag, _i] = _rs
    for (_v, _tag, _fp_idx), recs in _split_groups:
        v, tag = (_v, _tag)
        key = _agg_key(v, tag, _fp_idx)
        _recs_sorted = sorted(recs, key=lambda r: r.get('seed'))
        recs = _recs_sorted
        ppls = np.array([r['ppl'] for r in recs], dtype=float)
        entry = {'variant': v, 'protocol': tag, 'n_seeds': len(recs), 'ppl_mean': float(ppls.mean()), 'ppl_std': float(ppls.std(ddof=1)) if len(ppls) > 1 else 0.0, 'ppls': [float(p) for p in ppls], 'seeds': [r.get('seed') for r in recs], 'params': _group_params(recs), 'tokens_seen': recs[0].get('tokens_seen')}
        if _fp_idx is not None:
            entry['run_cfg_class'] = _fp_idx
        dyn_lens, all_lens, deltas = ([], [], [])
        bf1, bex = ([], [])

        def _cell(layer, k):
            v = layer.get(k)
            return float(v) if isinstance(v, (int, float)) and (not isinstance(v, bool)) and math.isfinite(v) else None
        for r in recs:
            r_f1, r_ex = ([], [])
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
                        _f1, _ex = (_cell(lv, 'bnd_f1'), _cell(lv, 'bnd_excess'))
                        if _f1 is not None and _ex is not None:
                            r_f1.append(_f1)
                            r_ex.append(_ex)
                dl = _cell(lv, 'delta')
                if dl is not None:
                    deltas.append(dl)
            if r_f1:
                bf1.append(float(np.mean(r_f1)))
                bex.append(float(np.mean(r_ex)))
        if dyn_lens:
            entry['avg_dyn_block_len'] = float(np.mean(dyn_lens))
        elif all_lens:
            entry['avg_block_len'] = float(np.mean(all_lens))
        if deltas:
            entry['delta_logit_mean'] = float(np.mean(deltas))
        if bf1:
            entry['boundary_f1_dyn'] = float(np.mean(bf1))
            entry['boundary_f1_dyn_std'] = float(np.std(bf1, ddof=1)) if len(bf1) > 1 else 0.0
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
            _ref_steps = [int(p[0]) for p in hists[0]]
            _off_grid = []
            for _i, _h in enumerate(hists[1:], start=1):
                _hs = [int(p[0]) for p in _h]
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

    def _baseline_for(tag, fp_idx, base_groups, recs):
        _want = _cfg_fp(recs[0]) if recs else None
        if _want is not None:
            _cands = sorted(base_groups.items(), key=lambda kv: (kv[0][0] != tag, kv[0][1] is None, kv[0][1] if kv[0][1] is not None else -1))
            for (_t_c, _i_c), _grp in _cands:
                if _grp and _cfg_fp(_grp[0]) == _want:
                    return (_grp, _t_c)
        for _cand in ((tag, fp_idx), (tag, None), ('', None)):
            if _cand in base_groups:
                return (base_groups[_cand], _cand[0])
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
            print(f'[aggregate] {key}: pairing against the UNTAGGED `full` baseline (no `full` run exists for protocol {tag or '(untagged)'})')
        elif tag and ('', None) in _full_groups:
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
        elif tag and ('', None) in _sw_groups:
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
    print('\n============== WikiText-103 validation perplexity ==============')
    print(f'{'variant':{_TW['variant']}s} {'PPL mean±std':>{_TW['ppl']}s} {'Δvs full':>{_TW['d']}s} {'Δvs sw128m':>{_TW['d2']}s} {'params(M)':>{_TW['params']}s} {'seeds':>{_TW['seeds']}s} {'blklen':>{_TW['blklen']}s} {'bndF1':>{_TW['bnd']}s}')
    order = ['full', 'full_matched', 'full_cos', 'full_sink', 'full_sw128', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic', 'csa_dyn_fuse', 'csa_fix_randidx', 'csa_fix_zerocont', 'csa_fix_nosink', 'csa_fix_topk8', 'csa_fix_topk64', 'hybrid_fixed', 'hybrid_dynamic', 'hybrid_csa_dyn', 'hybrid_hca_dyn', 'hybrid_csa_dyn_fuse']
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
    recs = {}
    _pair_amb = []
    _legacy_ok = legacy_warm_tags(summary)
    for _k, r in summary.items():
        if not (isinstance(r, dict) and ppl_is_usable(r.get('ppl')) and ('seed' in r) and (not r.get('synthesized'))):
            continue
        rk = _record_identity(r, _k, _legacy_ok)
        if rk is None:
            continue
        if rk in recs:
            _pair_amb.append(rk)
            continue
        recs[rk] = r
    _pair_skip = []
    PAIRS = [('csa_dynamic', 'csa_fixed'), ('hybrid_dynamic', 'hybrid_fixed'), ('hybrid_csa_dyn', 'hybrid_fixed'), ('hybrid_hca_dyn', 'hybrid_fixed'), ('csa_dyn_fuse', 'csa_dynamic'), ('hybrid_csa_dyn_fuse', 'hybrid_csa_dyn'), ('csa_fix_randidx', 'csa_fixed'), ('csa_fix_zerocont', 'csa_fixed'), ('csa_fix_nosink', 'csa_fixed'), ('csa_fix_topk8', 'csa_fixed'), ('csa_fix_topk64', 'csa_fixed'), ('full_sink', 'full')]
    print('\n========== paired per-seed deltas (a − b, positive = a worse) ==========')
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
        dl, _sk = ([], [])
        for ts, bs, s in common:
            _reason = pair_reason(ra[ts, s], rb[bs, s])
            if _reason is not None:
                _sk.append((s, _reason))
                continue
            dl.append(ra[ts, s]['ppl'] - rb[bs, s]['ppl'])
        if _sk:
            _pair_skip.append((a, b, len(_sk), len(common), sorted({r for _s, r in _sk})))
        if dl:
            d = np.array(dl, dtype=float)
            std = float(d.std(ddof=1)) if len(d) > 1 else 0.0
            print(f'  {a:22s} − {b:22s} = {d.mean():+7.2f} ± {std:5.2f}   (n={len(d)})')
    if _pair_skip:
        print('  [skip] seed(s) rejected by the run_cfg/budget gate (see `pair_reason`); the printed n counts only the pairs that cleared it —')
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
    order = ['full', 'full_matched', 'full_cos', 'full_sink', 'full_sw128', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic', 'csa_dyn_fuse', 'csa_fix_randidx', 'csa_fix_zerocont', 'csa_fix_nosink', 'csa_fix_topk8', 'csa_fix_topk64', 'hybrid_fixed', 'hybrid_dynamic', 'hybrid_csa_dyn', 'hybrid_hca_dyn', 'hybrid_csa_dyn_fuse']
    vs = [v for v in order if v in agg] + [v for v in agg if v not in order]
    fig, ax = plt.subplots(1, 5, figsize=(27, 4.5))
    ps = [agg[v]['ppl_mean'] for v in vs]
    es = [agg[v]['ppl_std'] for v in vs]
    ax[0].bar(range(len(vs)), ps, yerr=es, capsize=3, color='steelblue')
    ax[0].set_xticks(range(len(vs)))
    ax[0].set_xticklabels(vs, rotation=30, ha='right')
    ax[0].set_ylabel('validation perplexity (mean ± std)')
    ax[0].set_title('WikiText-103 PPL')
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
        smooth = np.convolve(curve, np.ones(k) / k, mode='valid')
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
    _dense_keys = [k for k in agg if k == 'full' or k.startswith('full@cfg')]
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

def run(cfg=None, seeds=None, guard=None, label=''):
    cfg = dict(RUN) if cfg is None else dict(cfg)
    if seeds is not None:
        cfg['seeds'] = list(seeds)
    _matched = cfg.get('matched')
    d = cfg.get('d', 256)
    n_layers = cfg.get('n_layers', 6)
    n_heads = cfg.get('n_heads', 8)
    d_head = cfg.get('d_head', 32)
    outdir = cfg['outdir']
    os.makedirs(outdir, exist_ok=True)
    _attb_bump_epoch()
    t_data = time.time()
    train_ids, val_batch, vocab, _, val_bnd = load_wikitext(cfg['seq_len'], cfg['n_train_tokens'])
    if guard is not None:
        guard.record_run(time.time() - t_data, 0, 0, 0, 0, 0)
    summary_path = os.path.join(outdir, 'summary.json')
    summary = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, encoding='utf-8') as f:
                summary = json.load(f)
        except Exception as _e:
            print(f'[resume] FATAL: {summary_path} exists but cannot be parsed ({type(_e).__name__}: {_e}).  Refusing to overwrite it with an empty summary — move it aside to start fresh.')
            raise
    if label:
        print(f'\n########## v5 phase: {label} — {len(cfg['variants'])} variants x seeds {cfg['seeds']} x {cfg['steps']} steps ##########')
    _mg = cfg.get('matched')
    if _mg is None:
        _mkey = 'default'
    else:
        _mkey = ','.join(sorted((str(_x) for _x in _mg)))
    fp = f'steps{cfg['steps']}_sl{cfg['seq_len']}_bs{cfg['batch_size']}_nt{cfg['n_train_tokens']}_lr{cfg['lr']}_wd{cfg['weight_decay']}_d{d}_L{n_layers}_H{n_heads}_Dh{d_head}_wu{cfg.get('warmup', 50)}_cl{cfg.get('comp_lambda', 0.05)}_dlm{cfg.get('delta_lr_mult', 10.0)}_mt{_mkey}_det{determinism_label()}_cs{CODE_SEMANTICS}'
    ratios = {}
    dropped_truncations = []
    stale_dropped = []
    for seed in cfg['seeds']:
        for v in cfg['variants']:
            key = f'{v}::seed{seed}'
            if key in summary and 'ppl' in summary.get(key, {}):
                old = summary[key]
                if not ppl_is_usable(old.get('ppl')):
                    print(f"[resume] {key} holds a NON-MEASURABLE ppl ({old.get('ppl')!r}) — every statistic in this repo refuses it (`ppl_is_usable`), so counting it as 'already completed' would drop the seed from the tables silently. DROPPING it and retraining")
                    del summary[key]
                elif old.get('synthesized'):
                    print(f'[resume] {key} is a SYNTHESIZED record (reconstructed from an old aggregate.json, no weights) — DROPPING it and retraining')
                    del summary[key]
                elif old.get('run_cfg') is None:
                    print(f"[resume] {key} has no config fingerprint (predates fingerprinting) — re-running and overwriting; the STALE record is dropped now, so a failed or truncated retrain cannot leave it in place looking like this attempt's result")
                    stale_dropped.append(key)
                    del summary[key]
                elif old.get('run_cfg') == fp:
                    print(f'[skip] {key} already completed (resume)')
                    continue
                else:
                    print(f"[resume] {key} was run under a DIFFERENT config ({old['run_cfg']} != {fp}) — re-running and overwriting; the STALE record is dropped now, so a failed or truncated retrain cannot leave it in place looking like this attempt's result")
                    stale_dropped.append(key)
                    del summary[key]
            if v not in ratios:
                ratios[v] = variant_mlp_ratio(v, vocab, d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, seq_len=cfg['seq_len'], matched=_matched)
            deadline_ts = None
            if guard is not None:
                est = guard.estimate_seconds(cfg['steps'], d=d, n_layers=n_layers, seq_len=cfg['seq_len'], batch_size=cfg['batch_size'])
                if not guard.can_start(est):
                    print(f'[budget] SKIP {key}: projected ¥{est / 3600 * guard.price:.2f} would pass cap ¥{guard.cap_yuan():.2f} (spent ¥{guard.spent_yuan():.2f})')
                    continue
                deadline_ts = guard.deadline_ts()
                print(f'[budget] {key}: projected {est / 60:.0f} min (¥{est / 3600 * guard.price:.2f}), spent so far ¥{guard.spent_yuan():.2f}')
            t_run = time.time()
            rec = None
            try:
                rec = train_variant(v, train_ids, val_batch, vocab, seed=seed, d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, seq_len=cfg['seq_len'], batch_size=cfg['batch_size'], steps=cfg['steps'], lr=cfg['lr'], weight_decay=cfg['weight_decay'], warmup=cfg['warmup'], comp_lambda=cfg['comp_lambda'], delta_lr_mult=cfg.get('delta_lr_mult', 10.0), eval_every=cfg.get('eval_every', 0), eval_subset=cfg.get('eval_subset', 128), val_bnd=val_bnd, mlp_ratio=ratios[v], deadline_ts=deadline_ts)
                if rec.get('budget_truncated'):
                    print(f'[budget] {key} was truncated after {rec.get('steps_done')} steps — its partial record is DISCARDED (it holds no `ppl`); ' + ('keeping the previous record instead.' if key in summary else 'the key is left ABSENT, not written as a stub.') + ' Raise BUDGET and re-run to retry this cell.')
                    if key not in summary:
                        dropped_truncations.append(key)
                else:
                    rec['run_cfg'] = fp
                    rec['_code'] = CODE_SEMANTICS
                    assert fp.endswith(f'_cs{rec['_code']}'), (fp, rec['_code'])
                    summary[key] = rec
            except Exception as e:
                import traceback
                import sys as _sys
                if key in summary and 'ppl' in (summary.get(key) or {}):
                    print(f'[{key}] FAILED: {e} — keeping the previous MEASURED record (the error record holds no `ppl` and must not replace it); the failed attempt is still billed to the guard below.')
                    print(traceback.format_exc(), file=_sys.stderr)
                else:
                    summary[key] = {'variant': v, 'seed': seed, 'error': traceback.format_exc()}
                    print(f'[{key}] FAILED: {e}')
            if guard is not None:
                steps_done = 0
                if isinstance(rec, dict) and rec.get('steps_done'):
                    steps_done = int(rec['steps_done'])
                elif isinstance(rec, dict) and 'ppl' in rec:
                    steps_done = int(cfg['steps'])
                guard.record_run(time.time() - t_run, steps_done, d, n_layers, cfg['seq_len'], cfg['batch_size'])
            atomic_write_json(summary_path, summary)
            gc.collect()
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
    if dropped_truncations:
        print(f'\n[budget] {len(dropped_truncations)} cell(s) were retrained, hit the budget deadline, and produced NO measurement — they are ABSENT from {summary_path} and from every table above:')
        for _k in dropped_truncations:
            print(f'    {_k}')
        print('[budget] raise BUDGET and re-run to fill these cells in.')
    if stale_dropped:
        _still = [k for k in stale_dropped if k not in summary]
        print(f'\n[resume] {len(stale_dropped)} cell(s) held a record from a DIFFERENT config and were re-run; {len(_still)} of them produced NO measurement this pass and are now ABSENT from {summary_path} (the stale record was removed rather than kept, so no number measured by older code can be reported under the current config):')
        for _k in _still:
            print(f'    {_k}')
        if len(_still) != len(stale_dropped):
            print(f'    ({len(stale_dropped) - len(_still)} of them succeeded and carry a fresh record.)')
    _finish_panel(outdir, summary)
    return (summary, None)

def _dump_aggregate(outdir, summary):
    agg = {}
    try:
        agg = aggregate(summary)
    except Exception as e:
        print(f'[aggregate] aggregate({outdir}) failed: {type(e).__name__}: {e}')
        print('[aggregate] retrying with the unusable records (no measured `ppl`, `synthesized`, an ambiguous same-seed duplicate, or a record missing a key `aggregate` indexes unconditionally) removed — the untouched records keep their exact values')
        _REQUIRED = ()
        clean = {k: r for k, r in summary.items() if isinstance(r, dict) and ppl_is_usable(r.get('ppl')) and (not r.get('synthesized')) and all((r.get(f) is not None for f in _REQUIRED))}
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
    _plot(records_for_readers, agg, outdir)
    return agg

def _finish_panel(outdir, summary):
    return _dump_aggregate(outdir, summary)
