#!/usr/bin/env python3
import collections
import dataclasses
import gc
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except (ValueError, OSError):
            pass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
_REAL_CHECK_CALL = subprocess.check_call

def _soft_check_call(*a, **k):
    try:
        return _REAL_CHECK_CALL(*a, **k)
    except Exception as e:
        print(f'[v7] pip install skipped ({type(e).__name__}: {e})')
        return 0
REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
sys.path.insert(0, REPO)
subprocess.check_call = _soft_check_call
try:
    import exp_lib as L
finally:
    subprocess.check_call = _REAL_CHECK_CALL
DEVICE = L.DEVICE
OVERLAP = L.OVERLAP
RUN_CAP_V7 = dict(L.RUN_CAP)
RUN_CAP_V7.update(total_hours=float(os.environ.get('V7_CAP_HOURS', 44.583333333333336)), state_path='run_time_state_v7.json')
RUN_CAP_V7['already_hours'] = 0.0
_ROPE_CS_CACHE = {}
_ROPE_INV_CACHE = {}

def rope_inv_freq(half, device, base=10000.0):
    key = (float(base), int(half), str(device))
    inv = _ROPE_INV_CACHE.get(key)
    if inv is None:
        with torch.inference_mode(False):
            inv = base ** (-torch.arange(0, half, device=device, dtype=torch.float64) / half)
        _ROPE_INV_CACHE[key] = inv
    return inv

def rope_cos_sin(head_dim, rope_dim, positions, device, base=10000.0):
    cacheable = isinstance(positions, torch.Tensor) and positions.dtype in (torch.int64, torch.int32) and (positions.numel() > 0)
    if cacheable:
        cacheable = positions is L._arange_cache(positions.numel(), positions.device)
    if cacheable:
        key = (float(base), int(rope_dim), int(head_dim), str(device), int(positions.numel()), str(positions.dtype))
        hit = _ROPE_CS_CACHE.get(key)
        if hit is not None:
            return hit
    half = rope_dim // 2
    inv = rope_inv_freq(half, device, base)
    with torch.inference_mode(False):
        ang = positions.to(device).double()[:, None] * inv[None, :]
        out = (torch.cos(ang).to(torch.float32), torch.sin(ang).to(torch.float32))
    if cacheable and len(_ROPE_CS_CACHE) < 64:
        _ROPE_CS_CACHE[key] = out
    return out

def apply_rope(x, cos, sin, rope_dim):
    half = rope_dim // 2
    if rope_dim > x.shape[-1]:
        # Over-long rope_dim: rotate the available half of the row and keep
        # an empty passthrough tail.  The slice-write path below cannot
        # express that case (the second write would exceed the output width),
        # so use the general cat form.
        xr = x[..., :rope_dim]
        xp = x[..., rope_dim:]
        x1, x2 = (xr[..., :half], xr[..., half:])
        rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)
        return torch.cat([rot, xp], -1)
    x1 = x[..., :half]
    x2 = x[..., half:rope_dim]
    # same elementwise expressions, written straight into one output buffer:
    # the two cats only ever copied these values, so the result is
    # bit-identical.
    out = torch.empty_like(x)
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:rope_dim] = x1 * sin + x2 * cos
    if rope_dim < x.shape[-1]:
        out[..., rope_dim:] = x[..., rope_dim:]
    return out

@dataclasses.dataclass
class AttnCfgRope(L.AttnCfg):
    rope: bool = False
    rope_dim: int = 16
    qk_norm: bool = True

def gather_rows(x, idx):
    lead = x.shape[1:]
    flat = x.reshape(x.shape[0], -1)
    picked = flat.index_select(0, idx.reshape(-1))
    return picked.reshape(idx.shape[0], idx.shape[1], *lead)

def _atomic_torch_save(obj, path):
    # Write through a unique same-directory temp file, then replace atomically.
    d = os.path.dirname(os.path.abspath(path)) or '.'
    os.makedirs(d, exist_ok=True)
    _fd, _tmp = tempfile.mkstemp(dir=d, prefix=os.path.basename(path) + '.', suffix='.tmp')
    try:
        with os.fdopen(_fd, 'wb') as _fh:
            torch.save(obj, _fh)
            _fh.flush()
            os.fsync(_fh.fileno())
        os.replace(_tmp, path)
    except BaseException:
        try:
            os.unlink(_tmp)
        except OSError:
            pass
        raise

def _load_ckpt(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    except (RuntimeError, TypeError, ValueError):
        return torch.load(path, map_location='cpu', weights_only=False)

class HybridAttentionRoPE(L.HybridAttention):

    def __init__(self, d_model, n_heads, d_head, cfg):
        super().__init__(d_model, n_heads, d_head, cfg)
        if getattr(cfg, 'qk_norm', False):
            self.q_norm = L.RMSNorm(d_head)
            self.kv_norm = L.RMSNorm(d_head if cfg.kind == 'full' else 2 * cfg.c_kv)
        else:
            self.q_norm = self.kv_norm = None

    def _full_batched(self, x):
        if not getattr(self.cfg, 'rope', False):
            return super()._full_batched(x)
        B, T, _ = x.shape
        cfg = self.cfg
        # Eval-only fused q/k/v projection (same cache contract as
        # exp_lib.HybridAttention._full_batched).  Training uses the three
        # separate projections so no concatenated weight is rebuilt per step.
        _nhq = self.nh * self.hd
        if x.device.type == 'cuda' and not torch.is_grad_enabled():
            # eval-only weight-concat cache, keyed on the source weights'
            # autograd `_version` counters, dtype and device, exactly like
            # exp_lib's `_wcat_cache`.
            _key = (self.W_q.weight._version, self.W_k.weight._version, self.W_v.weight._version,
                    self.W_q.weight.dtype, self.W_k.weight.dtype, self.W_v.weight.dtype,
                    self.W_q.weight.device)
            _hit = self._wcat_cache.get('qkv')
            if _hit is None or _hit[0] != _key:
                with torch.inference_mode(False):
                    _hit = (_key, torch.cat([self.W_q.weight, self.W_k.weight, self.W_v.weight], 0))
                self._wcat_cache['qkv'] = _hit
            _qkv = F.linear(x, _hit[1])
            q = _qkv[..., :_nhq].reshape(B, T, self.nh, self.hd)
            k = _qkv[..., _nhq:2 * _nhq].reshape(B, T, self.nh, self.hd)
            v = _qkv[..., 2 * _nhq:].reshape(B, T, self.nh, self.hd)
            del _qkv
        else:
            q = self.W_q(x).view(B, T, self.nh, self.hd)
            k = self.W_k(x).view(B, T, self.nh, self.hd)
            v = self.W_v(x).view(B, T, self.nh, self.hd)
        if cfg.qk_norm:
            q = self.q_norm(q)
            k = self.kv_norm(k)
        rd = min(cfg.rope_dim, self.hd)
        cos, sin = rope_cos_sin(self.hd, rd, L._arange_cache(T, x.device), x.device)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        q = apply_rope(q, cos, sin, rd)
        k = apply_rope(k, cos, sin, rd)
        if cfg.full_cosine:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            scale = 1.0
        else:
            scale = 1.0 / math.sqrt(self.hd)
        # The sink branch builds a full cached mask within the row cap, or the
        # matching row slice per chunk beyond it.
        _use_full_mask = T * T * 4 <= L._DENSE_MASK_ROW_CAP_BYTES
        if self.sink is None:
            if 0 < cfg.window < T:
                out = L._window_attn_local(q, k, v, cfg.window, scale)
                return self.W_o(out.reshape(B, T, self.nh * self.hd))
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=True, scale=scale).transpose(1, 2)
            return self.W_o(out.reshape(B, T, self.nh * self.hd))
        if _use_full_mask:
            if cfg.window > 0:
                mask = L.causal_window_mask(T, cfg.window, x.device)
            else:
                mask = L.causal_mask(T, x.device)
        else:
            mask = None
        sink_view = torch.nan_to_num(self.sink).view(1, self.nh, 1, 1) if self.sink is not None else None
        row_chunk = L._full_attn_row_chunk(x.device, B, self.nh, T, q.element_size())
        # Reuse the base-class row loop / whole-loop checkpoint (same chunk
        # partition, same masked-lane fill and softmax, values unchanged).
        if torch.is_grad_enabled() and q.requires_grad:
            out = torch.utils.checkpoint.checkpoint(
                self._sink_attn_loop, q, k, v, mask, _use_full_mask, sink_view,
                scale, row_chunk, use_reentrant=False, preserve_rng_state=False)
        else:
            out = self._sink_attn_loop(q, k, v, mask, _use_full_mask, sink_view, scale, row_chunk)
        return self.W_o(out.reshape(B, T, self.nh * self.hd))

    def _single_rope(self, x, pre=None, gate_sig=None, pre_blocks=None):
        cfg = self.cfg
        T = x.shape[0]
        nh, hd = (self.nh, self.hd)
        scale = 1.0
        if not getattr(cfg, 'rope', False):
            return L.HybridAttention._single(self, x, pre)
        rd = min(cfg.rope_dim, hd)
        q = F.normalize((pre['q'] if pre is not None else self.W_q(x)).view(T, nh, hd), dim=-1)
        Ca_raw = pre['Ca'] if pre is not None else x @ self.W_aKV
        if pre is not None:
            Cb_raw = pre['Cb']
        else:
            Cb_raw = x @ self.W_bKV if self.W_bKV is not None else None
        gate_mean = None
        if cfg.chunking == 'cosine_learnable':
            if gate_sig is not None:
                gate, gl = gate_sig
            else:
                fused = pre['fused'] if pre is not None and 'fused' in pre else self._fuse(x.detach())
                sim = L.cosine_similarity_consecutive(fused)
                if sim.numel() == 0:
                    gate = None
                else:
                    prefix_mean = torch.cumsum(sim, 0) / L._range_cache(1, sim.numel() + 1, x.device)
                    tau = prefix_mean + self.delta_logit
                    gate = torch.sigmoid((tau - sim) / max(cfg.temperature, 0.001))
                    hard_b = gate.detach() > L._HALF
                    # zero-copy numpy view — same booleans as the .tolist() form
                    gl = hard_b.cpu().numpy()
            if gate is None:
                bid = torch.zeros(T, dtype=torch.long, device=x.device)
                nblk = 1
                gate_mean = torch.zeros((), device=x.device) if self.need_reg else None
            elif pre_blocks is not None:
                bid, nblk, _hon = pre_blocks
                if self.need_reg:
                    if _hon.dtype != gate.dtype:
                        _hon = _hon.to(gate.dtype)
                    _soft_hon = gate * _hon
                    _shs = _soft_hon.sum()
                    gate_mean = (float(nblk - 1) + _shs - _shs.detach()) / T
            else:
                with torch.no_grad():
                    bid, nblk, _hon_d = L.blocks_from_cuts(T, gl, cfg.min_block, cfg.max_block, x.device, return_count=True, return_honoured=True)
                if self.need_reg:
                    # gate has length T-1 (consecutive-pair cosine similarities),
                    # so the honoured mask must too (cf. exp_lib._single_rope).
                    _hon = torch.zeros(T - 1, dtype=gate.dtype, device=gate.device)
                    _hon[torch.as_tensor(list(_hon_d), dtype=torch.long, device=gate.device)] = 1.0
                    _soft_hon = gate * _hon
                    _shs = _soft_hon.sum()
                    gate_mean = (float(nblk - 1) + _shs - _shs.detach()) / T
        elif cfg.chunking in ('cosine_abs', 'cosine_adaptive') or (cfg.dynamic and cfg.kind != 'hca'):
            if pre_blocks is not None:
                # geometry pre-computed once for the whole batch by the
                # inherited _cut_blocks_batched (forward hoist); values
                # identical to the per-row form below (design notes §6.1).
                bid, nblk, _hon = pre_blocks
            else:
                sim = L.cosine_similarity_consecutive(x.detach())
                tau = L.causal_adaptive_threshold(sim, cfg.target_block_tokens) if cfg.chunking == 'cosine_adaptive' or cfg.adaptive else cfg.cos_threshold
                bid, nblk = L.blocks_from_cosine(x, tau, cfg.min_block, cfg.max_block, sim=sim, return_count=True)
        else:
            bid = L.blocks_fixed(T, cfg.block_size, x.device)
            nblk = -(-T // cfg.block_size)
        if pre is not None:
            Za, Zb = (pre['Za'], pre['Zb'])
        else:
            Za = x @ self.W_aZ
            Zb = x @ self.W_bZ if self.W_bZ is not None else None
        if cfg.kind == 'hca':
            comp_kv, last_tok, Bn = L._pool_trainable(L.pool_blocks_single, Ca_raw, Za, self.B_pos_a, bid, n_blocks=nblk, monotonic=True)
        else:
            comp_kv, last_tok, Bn = L._pool_trainable(L.pool_variable_blocks, Ca_raw, Cb_raw, Za, Zb, bid, self.B_pos_a, self.B_pos_b, cfg.overlap, n_blocks=nblk, monotonic=True)
        index_kv = comp_kv
        attn_kv = torch.zeros_like(comp_kv) if cfg.content_mode == 'zero' else comp_kv
        if self._stats is not None:
            self._stats.append(torch.bincount(bid))
            if self._cuts is not None:
                cuts = (bid[1:] != bid[:-1]).nonzero(as_tuple=True)[0] + 1
                self._cuts.append(cuts.cpu() if cuts.device.type != 'cpu' else cuts)
        comp_n = self.kv_norm(attn_kv) if cfg.qk_norm else attn_kv
        qn = self.q_norm(q) if cfg.qk_norm else q
        sw_n = self.kv_norm(Ca_raw) if cfg.qk_norm else Ca_raw
        k_blk, v_blk = self._split(self.W_kvhead(F.normalize(comp_n, dim=-1)))
        k_sw, v_sw = self._split(self.W_kvhead(F.normalize(sw_n, dim=-1)))
        k_blk = F.normalize(k_blk, dim=-1)
        k_sw = F.normalize(k_sw, dim=-1)
        soft = None
        sel_valid = None
        if cfg.kind == 'hca':
            # cached int32 expansion, exactly as in exp_lib._single's hca arm
            # (index transport, design notes §6.1): the same integers as the
            # int64 arange expand narrowed to int32.
            topk_idx = L._arange_exp_cache(T, Bn, x.device)
        else:
            _sel_valid_box = []
            # Eval does not need the soft prior tensor: the straight-through
            # S path is dead without gradients, so its whole tail is skipped
            # (same contract as exp_lib._single).
            _need_soft = torch.is_grad_enabled()
            _, topk_idx, soft = L.lightning_indexer(x, index_kv, last_tok, self.W_DQ, self.W_DK, self.W_w.weight, cfg.n_index_heads, cfg.index_topk, return_mask=False, random_select=cfg.indexer_mode == 'random', pre_qI=pre['qI'] if pre is not None else None, pre_w=pre['w_idx'] if pre is not None else None, out_valid=_sel_valid_box, need_soft=_need_soft)
            sel_valid = _sel_valid_box[0]
        sink = self.sink if cfg.use_sink else None
        # Both branches hold distinct block ids per row: the HCA arm passes the
        # arange expansion, the csa arm's no-replacement indexer selection fills
        # the valid slots with distinct columns and only the tail pads repeat
        # block 0 (already invalidated by sel_valid) — the keep mask is
        # identical to the computed form.
        out = self._rope_attn_trainable(qn, k_blk, v_blk, topk_idx, last_tok, k_sw, v_sw, cfg.sliding_window, scale, sink, rd, soft=soft, sel_valid=sel_valid, topk_distinct=True)
        return (self.W_o(out.reshape(T, nh * hd)), gate_mean)

    def _rope_attn_trainable(self, *args, **kwargs):
        # Same non-reentrant recompute contract as exp_lib._gathered_attention
        # _trainable: _rope_attn is a pure function of its inputs with no RNG,
        # so backward recomputes the same chunk decomposition and gradients are
        # unchanged.
        q = args[0]
        if torch.is_grad_enabled() and q.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                self._rope_attn, *args, use_reentrant=False,
                preserve_rng_state=False, **kwargs)
        return self._rope_attn(*args, **kwargs)

    def _rope_attn(self, q, k_blk, v_blk, topk_idx, last_tok, k_sw, v_sw, w, scale, sink, rope_dim, q_chunk=128, soft=None, rope_base=10000.0, mem_budget_bytes=None, sel_valid=None, topk_distinct=False):
        T, nh, hd = q.shape
        dev = q.device
        if mem_budget_bytes is None:
            mem_budget_bytes = L._attn_transient_budget(dev)
        _es = q.element_size()
        _m = max(1, int(topk_idx.shape[1]) + int(w))
        _bytes_per_chunk_row = 2 * _m * nh * hd * _es + 4 * _m * nh * _es
        _bindable = mem_budget_bytes is not None and math.isfinite(float(mem_budget_bytes))
        if _bindable and _bytes_per_chunk_row > 0:
            _max_rows = max(1, int(mem_budget_bytes) // _bytes_per_chunk_row)
            q_chunk = min(q_chunk, _max_rows)
        cos, sin = rope_cos_sin(hd, rope_dim, L._arange_cache(T, dev), dev, rope_base)
        # RoPE is applied to the query chunk inside the loop: every rotated
        # position depends only on the same position of q, so the per-chunk
        # rotation is bit-identical (and each q position is rotated by exactly
        # one chunk, so q.grad is the same sum of disjoint contributions).
        _lt = last_tok.long()
        bc, bs = (cos[_lt], sin[_lt])
        k_blk_r = apply_rope(k_blk, bc[:, None, :], bs[:, None, :], rope_dim)
        k_sw_r = apply_rope(k_sw, cos[:, None, :], sin[:, None, :], rope_dim)
        out = torch.empty_like(q)
        _wi_all, _wvalid_all = L._window_geometry(T, w, dev)
        pos_all = L._arange_cache(T, dev)
        q_chunk = max(1, min(int(q_chunk), T))
        _MINL = L._finfo_min(k_blk.dtype)
        # same int32 index transport as exp_lib.gathered_attention (§6.1):
        # identical integers, identical gathered values.
        topk_l = topk_idx if topk_idx.dtype in (torch.int32, torch.int64) else topk_idx.long()
        # Same index comparison as exp_lib.gathered_attention: with the cached
        # arange expansion every row of last_tok[topk_l] is identical, so a
        # broadcast comparison gives the same booleans.
        _arange_sel = (topk_l.shape[1] == last_tok.shape[0]
                       and topk_l is L._IDX_CACHE.get((L._dev_str(dev), 'arangeexp', int(T), int(topk_l.shape[1]))))
        _sel_all = None
        # The K/V halves are stacked and gathered with one combined row
        # index per chunk, exactly as exp_lib.gathered_attention does: the
        # selected rows are the two separate gather outputs concatenated in the
        # same order, so the values are bit-identical.
        _nblk_rows = k_blk_r.shape[0]
        _has_sw = k_sw_r is not None and v_sw is not None
        if _has_sw:
            k_all = torch.cat([k_blk_r, k_sw_r], 0)
            v_all = torch.cat([v_blk, v_sw], 0)
        else:
            k_all, v_all = (k_blk_r, v_blk)
        _win_off = _nblk_rows if _has_sw else 0
        _win_off_tab = L._window_block_offset(T, w, _nblk_rows, dev) if _win_off else None
        # see exp_lib.gathered_attention: these per-row masks are hoisted under
        # L._FO_FULL_CAP_BYTES; the hoisted values are bitwise identical to the
        # per-chunk forms.  `topk_distinct=True` (the read-everything HCA arm's
        # arange expansion) makes first_occurrence_mask provably all-True — the
        # keep mask is then identical to the computed form.
        _dedup = (not topk_distinct) and topk_l.shape[1] > 1
        if L._attn_full_width_ok(T, topk_l.shape[1], w, _dedup):
            _sel_all = (pos_all[:, None] > last_tok[None, :]) if _arange_sel else (pos_all[:, None] > last_tok[topk_l])
            _keep_all = sel_valid
            if _dedup:
                _ddk_all = L.first_occurrence_mask(topk_l)
                _keep_all = _ddk_all if _keep_all is None else _keep_all & _ddk_all
            if _keep_all is not None:
                _sel_all &= _keep_all
            _valid_all = torch.cat([_sel_all, _wvalid_all], 1)
            # same hoisted forms as exp_lib.gathered_attention: the negation /
            # any-reduction values per row are identical to the per-chunk
            # forms.
            _nvalid_all = ~_valid_all[:, None, :]
            _anyrow_all = _valid_all.any(-1) if sink is None else None
        else:
            _keep_all = _valid_all = _nvalid_all = _anyrow_all = None
        _nb = int(topk_l.shape[1])
        for s in range(0, T, q_chunk):
            e = min(s + q_chunk, T)
            ib = topk_l[s:e]
            if _valid_all is not None:
                sel = _sel_all[s:e]
                _keep = None if _keep_all is None else _keep_all[s:e]
                valid = _valid_all[s:e]
                _nvalid = _nvalid_all[s:e]
                _anyrow = None if _anyrow_all is None else _anyrow_all[s:e]
            else:
                # Row-chunked construction instead of a full [T, topk] mask
                # when the full-width budget is exceeded.
                sel = ((pos_all[s:e, None] > last_tok[None, :]) if _arange_sel
                       else (pos_all[s:e, None] > last_tok[topk_l[s:e]]))
                _keep = sel_valid[s:e] if sel_valid is not None else None
                if ib.shape[1] > 1 and not topk_distinct:
                    _ddk = L.first_occurrence_mask(ib)
                    _keep = _ddk if _keep is None else _keep & _ddk
                if _keep is not None:
                    sel = sel & _keep
                valid = torch.cat([sel, _wvalid_all[s:e]], 1)
                _nvalid = ~valid[:, None, :]
                _anyrow = valid.any(-1) if sink is None else None
            # ib selects rows of k_blk_r / v_blk and _wi_all selects rows of
            # k_sw_r / v_sw — exactly the rows the cat-then-gather form
            # addressed through the +n_blk offsets, so every gathered value is
            # bit-identical (index re-plumbing only, §6.1).
            _iw = _wi_all[s:e]
            _iw_rows = _win_off_tab[s:e] if _win_off_tab is not None else _iw
            _rows = torch.cat([ib, _iw_rows], 1) if _has_sw else ib
            Kset = L._take_2d(k_all, _rows)
            qc = apply_rope(q[s:e], cos[s:e, None, :], sin[s:e, None, :], rope_dim)
            raw_logits = torch.einsum('qhd,qmhd->qhm', qc, Kset)
            del Kset
            if scale != 1.0:
                # x·1.0 is the IEEE identity (and its backward Mul node maps
                # grad·1.0 identically), so scaling by 1.0 is a no-op — same
                # values as exp_lib's gathered_attention.
                raw_logits.mul_(scale)
            if soft is None:
                logits = raw_logits.masked_fill_(_nvalid, _MINL)
                attn, _sink_unused = L._sink_split_softmax(logits, sink, want_sink=False)
                if sink is None:
                    attn.mul_(_anyrow[:, None, None])
            if soft is not None:
                sv = soft[s:e]
                if _keep is not None:
                    sv = sv * _keep
                soft_log = torch.log(sv.clamp_min(1e-12))
                soft_log.masked_fill_(sv <= 0, _MINL)
                # The clone is taken before the masked_fill_ below, so it
                # carries the unmasked logits.
                soft_logits = raw_logits.clone()
                soft_logits[:, :, :_nb].add_(soft_log[:, None, :]).clamp_min_(_MINL)
                logits = raw_logits.masked_fill_(_nvalid, _MINL)
                attn, _sink_unused = L._sink_split_softmax(logits, sink, want_sink=False)
                soft_logits.masked_fill_(_nvalid, L._finfo_min(soft_logits.dtype))
                soft_attn, _sink_unused = L._sink_split_softmax(soft_logits, sink, want_sink=False)
                attn = (soft_attn - soft_attn.detach()).add_(attn)
                if sink is None:
                    attn.mul_(_anyrow[:, None, None])
            out[s:e] = torch.einsum('qhm,qmhd->qhd', attn, L._take_2d(v_all, _rows))
        return out
    def _dense_warmup_forward(self, x):
        B, T, _ = x.shape
        nh, hd = (self.nh, self.hd)
        scale = 1.0
        use_rope = isinstance(self.cfg, AttnCfgRope) and self.cfg.rope
        rd = min(getattr(self.cfg, 'rope_dim', 0), hd)
        win = getattr(self, '_dense_max_T', None)
        _use_full_mask = T * T * 4 <= L._DENSE_MASK_ROW_CAP_BYTES
        if _use_full_mask:
            mask = L.causal_mask(T, x.device)
            if win is not None and win < T:
                mask = L.causal_window_mask(T, win, x.device)
        else:
            mask = None
        Ca = x @ self.W_aKV
        q = F.normalize(self.W_q(x).view(B, T, nh, hd), dim=-1)
        norm_kv = getattr(self.cfg, 'qk_norm', False)
        _Ca_int = self.kv_norm(Ca) if norm_kv else Ca
        _Ca_int = F.normalize(_Ca_int, dim=-1)
        k, v = self._split(self.W_kvhead(_Ca_int))
        k = F.normalize(k, dim=-1)
        if norm_kv:
            q = self.q_norm(q)
        if use_rope:
            cos, sin = rope_cos_sin(hd, rd, L._arange_cache(T, x.device), x.device)
            q = apply_rope(q, cos[:, None, :], sin[:, None, :], rd)
            k = apply_rope(k, cos[:, None, :], sin[:, None, :], rd)
        sink = self.sink if self.cfg.use_sink and self.sink is not None else None
        if sink is None and win is None:
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=True, scale=scale).transpose(1, 2)
            return self.W_o(out.reshape(B, T, nh * hd))
        row_chunk = L._full_attn_row_chunk(x.device, B, nh, T, q.element_size())
        # Same whole-loop activation checkpoint as exp_lib._full_batched's
        # sink/window path: the row loop is a pure, RNG-free function of its
        # inputs, so backward recomputes the identical chunks.
        if torch.is_grad_enabled() and q.requires_grad:
            out = torch.utils.checkpoint.checkpoint(
                self._dense_warmup_loop, q, k, v, mask, _use_full_mask, sink, win,
                scale, row_chunk, use_reentrant=False, preserve_rng_state=False)
        else:
            out = self._dense_warmup_loop(q, k, v, mask, _use_full_mask, sink, win, scale, row_chunk)
        return self.W_o(out.reshape(B, T, nh * hd))

    def _dense_warmup_loop(self, q, k, v, mask, use_full_mask, sink, win, scale, row_chunk):
        B, T, nh, hd = q.shape
        out = torch.empty(B, T, nh, hd, device=q.device, dtype=q.dtype)
        for s in range(0, T, row_chunk):
            e = min(s + row_chunk, T)
            logits = torch.einsum('bthd,bshd->bths', q[:, s:e], k)
            if scale != 1.0:
                logits.mul_(scale)
            _mask_c = mask[s:e] if use_full_mask else L._dense_causal_window_rows(T, win if win is not None else 0, s, e, q.device)
            # bool masked-lane table -> fill: bit-identical logits to the
            # additive 0/-inf form (same lanes get -inf; x + 0.0 == x except
            # the sign of zero, whose exp is 1.0 either way).
            logits.masked_fill_(_mask_c[:, None, :], float('-inf'))
            zflat = logits.reshape(B * (e - s), nh, T)
            if sink is None:
                attn = L.sink_softmax(zflat, sink)
            else:
                attn, _sink_unused = L._sink_split_softmax(zflat, sink, want_sink=False)
            out[:, s:e] = torch.einsum('bths,bshd->bthd', attn.view(B, e - s, nh, T), v)
        return out

    def forward(self, x):
        B, T, d = x.shape
        if B == 0 or T == 0:
            # Same empty-input contract as exp_lib.HybridAttention.forward.
            self.last_gate_mean = None
            return x.new_empty((B, T, d))
        if getattr(self, '_dense_warmup', False) and self.cfg.kind in ('csa', 'hca'):
            self.last_gate_mean = None
            return self._dense_warmup_forward(x)
        if getattr(self.cfg, 'rope', False) and self.cfg.kind in ('csa', 'hca'):
            self.last_gate_mean = None
            B, T, d = x.shape
            flat = x.reshape(B * T, d)
            pre_flat = self._pre(flat, fuse_x=x)
            # The per-row path reads only its own [T, ...] slice of the fused
            # projection; _gate_signals needs only the fused table.
            pre_all = None
            pres = None
            if pre_flat is not None:
                pre_all = {'fused': pre_flat.get('fused')}
                pres = []
                for b in range(B):
                    _row = {}
                    for kk, v in pre_flat.items():
                        if v is None:
                            _row[kk] = None
                        elif kk == 'fused':
                            _row[kk] = v[b]
                        else:
                            _sl = v[b * T:(b + 1) * T]
                            # qI feeds the indexer GEMM, which expects a
                            # contiguous operand.
                            _row[kk] = _sl.contiguous() if kk == 'qI' else _sl
                    pres.append(_row)
            if pres is None:
                pres = [None] * B
            gate_sigs = None
            pre_blocks = None
            # same value-identical hoist as exp_lib.HybridAttention.forward,
            # for any batch size (design notes §6.1): the inherited batched
            # _gate_signals is bit-identical to the per-row _gate_signal calls
            if self.cfg.chunking == 'cosine_learnable':
                _raw, _gbool = self._gate_signals(x, pre_all)
                if all((_r is not None for _r in _raw)):
                    _gl = _gbool.cpu().numpy()
                    gate_sigs = [(_r[0], _g) for _r, _g in zip(_raw, _gl)]
                    # same batched block geometry as exp_lib.HybridAttention.forward;
                    # every value identical to the per-row form
                    pre_blocks = L._hoist_blocks_from_cuts(T, _gl, self.cfg.min_block, self.cfg.max_block, x.device, want_honoured=self.need_reg, hon_dtype=_raw[0][0].dtype)
            elif self.cfg.chunking in ('cosine_abs', 'cosine_adaptive') or (self.cfg.dynamic and self.cfg.kind != 'hca'):
                # same batch-level geometry hoist as exp_lib.HybridAttention.forward:
                # values identical.
                pre_blocks = self._cut_blocks_batched(x)
            # Write each row's attention output into the batch output — the
            # same transport-only, value-identical form as
            # exp_lib.HybridAttention.forward (design notes §6.1).
            outs = torch.empty(B, T, x.shape[-1], device=x.device, dtype=x.dtype)
            gates = []
            for b in range(B):
                pre = pres[b]
                o, g = self._single_rope(x[b], pre=pre, gate_sig=None if gate_sigs is None else gate_sigs[b], pre_blocks=None if pre_blocks is None else pre_blocks[b])
                outs[b] = o
                if g is not None:
                    gates.append(g)
            if not gates:
                self.last_gate_mean = None
            else:
                _gm_tot = gates[0]
                for _g in gates[1:]:
                    _gm_tot = _gm_tot + _g
                self.last_gate_mean = _gm_tot / len(gates)
            return outs
        return super().forward(x)

class BlockRoPE(nn.Module):

    def __init__(self, d, n_heads, d_head, cfg, mlp_ratio=4, mlp_gen=None):
        super().__init__()
        self.n1 = L.RMSNorm(d)
        self.attn = HybridAttentionRoPE(d, n_heads, d_head, cfg)
        self.n2 = L.RMSNorm(d)
        self.mlp = L.MLP(d, int(round(d * mlp_ratio)), init_gen=mlp_gen)

    def forward(self, x):
        # Same activation-checkpoint policy as exp_lib.Block: the recomputed
        # forward is the identical pure function, so values/gradients are
        # unchanged; cosine_learnable arms keep their gate side channel and
        # fall back to the MLP-level checkpoint.
        if L._CKPT['on'] == 'block' and torch.is_grad_enabled() and x.requires_grad:
            if getattr(self.attn.cfg, 'chunking', '') != 'cosine_learnable':
                return torch.utils.checkpoint.checkpoint(
                    self._fwd_ckpt, x, use_reentrant=False, preserve_rng_state=False)
        return self._fwd(x)

    def _fwd(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x

    def _fwd_ckpt(self, x):
        # Suppress the inner MLP checkpoint while the outer block recomputes.
        L._CKPT_NEST[0] += 1
        try:
            return self._fwd(x)
        finally:
            L._CKPT_NEST[0] -= 1

class SmallGPTRoPE(L.__dict__.get('_v7_orig_smallgpt', L.SmallGPT)):

    def __init__(self, vocab, d, n_layers, n_heads, d_head, max_seq, layer_cfgs, mlp_ratio=4):
        super().__init__(vocab, d, n_layers, n_heads, d_head, max_seq, layer_cfgs, mlp_ratio)
        self.use_abs_pe = not any((getattr(c, 'rope', False) for c in layer_cfgs))
        if not self.use_abs_pe:
            del self.pos

    def _make_block(self, d, n_heads, d_head, cfg, mlp_ratio, mlp_gen=None):
        if getattr(cfg, 'rope', False):
            return BlockRoPE(d, n_heads, d_head, cfg, mlp_ratio, mlp_gen=mlp_gen)
        return L.Block(d, n_heads, d_head, cfg, mlp_ratio, mlp_gen=mlp_gen)

    def forward(self, ids, logits_tail=None):
        x = self.trunk(ids)
        if logits_tail:
            x = x[:, -logits_tail:]
        return self.head(x)

    def trunk(self, ids):
        # RoPE embedding form of exp_lib.SmallGPT.trunk — trunk is dispatched
        # here by the inherited ce_loss; comp_reg terms accumulate in the same
        # order as the full forward (zeros init + per-block adds).
        T = ids.shape[1]
        x = self.tok(ids)
        if getattr(self, 'use_abs_pe', True):
            x = x + self.pos(L._arange_cache(T, ids.device))
        reg = torch.zeros((), device=ids.device)
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
                reg = reg + mult * (gm * Lb - 1.0) ** 2
        self.comp_reg = reg
        return self.norm(x)

def make_layer_cfgs_v7(n_layers, variant):
    if variant == 'csa_fixed_rope':
        return [AttnCfgRope(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32, rope=True, rope_dim=16, qk_norm=True) for _ in range(n_layers)]
    if variant == 'csa_fixed_rope_randidx':
        # 从头训练的随机选择对照（评审点 2）：与 csa_fixed_rope 逐字段一致，
        # 仅 indexer_mode='random'。它回答的是「训练充分的随机选择」而非
        # 「测试时把已训练模型的选择器换成随机」——后者（v10 P3MP 的
        # randidx 臂）测的是依赖性，本变体测的是选择方式本身的优劣。
        return [AttnCfgRope(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32, indexer_mode='random', rope=True, rope_dim=16, qk_norm=True) for _ in range(n_layers)]
    if variant == 'csa_fixed_rope_allblocks':
        # 从头训练的全块读取对照（评审点 2 + 5）：与 csa_fixed_rope 同一
        # c_kv 压缩 / qk_norm / RoPE / sink / 窗口投影，仅 index_topk 覆盖
        # 全部块（lightning_indexer 内部 k=min(topk, B)）。它同时是
        # 「逐项对齐的 dense」——与 full_rope 的差别只剩 KV 压缩投影与
        # 分块读取方式，不再是投影/归一化/编码的混合差异。
        return [AttnCfgRope(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=1000000, rope=True, rope_dim=16, qk_norm=True) for _ in range(n_layers)]
    if variant == 'csa_dynamic_rope':
        return [AttnCfgRope(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, rope=True, rope_dim=16, qk_norm=True) for _ in range(n_layers)]
    if variant == 'hybrid_fixed_rope':
        out = []
        for i in range(n_layers):
            if i % 2 == 0:
                out.append(AttnCfgRope(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32, rope=True, rope_dim=16, qk_norm=True))
            else:
                out.append(AttnCfgRope(kind='hca', dynamic=False, block_size=64, rope=True, rope_dim=16, qk_norm=True))
        return out
    if variant == 'full_rope':
        return [AttnCfgRope(kind='full', rope=True, rope_dim=16, qk_norm=True) for _ in range(n_layers)]
    if variant == 'full_sw128_matched_rope':
        return [AttnCfgRope(kind='full', window=128, rope=True, rope_dim=16, qk_norm=True) for _ in range(n_layers)]
    if variant == 'csa_fix_m1':
        return [L.AttnCfg(kind='csa', dynamic=False, block_size=1, overlap=0, index_topk=32) for _ in range(n_layers)]
    if variant.startswith('csa_fixed_topk'):
        k = int(variant.replace('csa_fixed_topk', ''))
        return [L.AttnCfg(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=k) for _ in range(n_layers)]
    return L.__dict__['_orig_make_layer_cfgs'](n_layers, variant)
_PATCHES_INSTALLED = False

def install_patches():
    global _PATCHES_INSTALLED
    if _PATCHES_INSTALLED:
        L.make_layer_cfgs = make_layer_cfgs_v7
        L.SmallGPT = L.__dict__['_v7_smallgpt_factory']
        L.HybridAttention.forward = L.__dict__['_v7_forward_warmup_aware']
        return
    if '_orig_make_layer_cfgs' not in L.__dict__:
        L.__dict__['_orig_make_layer_cfgs'] = L.make_layer_cfgs
    # Fall back to SmallGPTRoPE's actual base (the class it was defined
    # against at import time) rather than the current L.SmallGPT: if another
    # patcher replaced L.SmallGPT before this install, capturing it here while
    # SmallGPTRoPE keeps the older base would make the two factory branches
    # inherit from different class chains.
    _orig_smallgpt = L.__dict__.get('_v7_orig_smallgpt', SmallGPTRoPE.__mro__[1])
    L.__dict__['_v7_orig_smallgpt'] = _orig_smallgpt

    def smallgpt_factory(vocab, d, n_layers, n_heads, d_head, max_seq, layer_cfgs, mlp_ratio=4):
        if any((getattr(c, 'rope', False) for c in layer_cfgs)):
            return SmallGPTRoPE(vocab, d, n_layers, n_heads, d_head, max_seq, layer_cfgs, mlp_ratio)
        return _orig_smallgpt(vocab, d, n_layers, n_heads, d_head, max_seq, layer_cfgs, mlp_ratio)
    _orig_forward = L.HybridAttention.forward

    def forward_warmup_aware(self, x):
        if x.shape[0] == 0 or x.shape[1] == 0:
            self.last_gate_mean = None
            return x.new_empty(x.shape)
        if getattr(self, '_dense_warmup', False) and self.cfg.kind in ('csa', 'hca'):
            self.last_gate_mean = None
            return HybridAttentionRoPE._dense_warmup_forward(self, x)
        return _orig_forward(self, x)
    L.__dict__['_v7_smallgpt_factory'] = smallgpt_factory
    L.__dict__['_v7_forward_warmup_aware'] = forward_warmup_aware
    L.HybridAttention.forward = forward_warmup_aware
    L.make_layer_cfgs = make_layer_cfgs_v7
    L.SmallGPT = smallgpt_factory
    _PATCHES_INSTALLED = True
install_patches()
# 代码语义戳必须以 exp_lib.code_stamp() 为准（其 docstring 的明确要求）：
# eager 模式保持既有值；内核模式（CSA_KERNEL）下追加 -tk1 后缀
# （design_notes §3.1）。
CKPT_CODE = L.code_stamp()
ROPE_VARIANTS = ['full_rope', 'csa_fixed_rope', 'hybrid_fixed_rope', 'csa_dynamic_rope', 'csa_fixed']
PARAM_MATCHED_V7 = {'full_rope', 'full_sw128_matched_rope'}

def train_warmup(variant, train_ids, val_batch, vocab, *, seed=0, d=256, n_layers=6, n_heads=8, d_head=32, seq_len=512, batch_size=12, steps=20000, warm_steps=0, lr=0.0003, weight_decay=0.1, warmup=200, comp_lambda=0.05, delta_lr_mult=10.0, eval_every=1000, eval_subset=128, val_bnd=None, device=DEVICE, log_every=1000, mlp_ratio=4, deadline_ts=None):
    L.set_seed(seed)
    _rng_fp = L.rng_fingerprint()
    L._attb_bump_epoch()
    if variant in set(L.PARAM_MATCHED) | set(PARAM_MATCHED_V7) and float(mlp_ratio) == 4.0:
        raise ValueError(f'{variant} is a PARAM-MATCHED baseline but was given the default mlp_ratio=4; its MLP must be widened to match its sparse reference arm, or the comparison is not parameter-controlled. Pass mlp_ratio=L.variant_mlp_ratio(...).')
    if steps < 1:
        raise ValueError(f'steps={steps}: nothing to train')
    if warm_steps >= steps:
        raise ValueError(f'warm_steps={warm_steps} >= steps={steps}: the dense->sparse switch never fires, so the recorded numbers would come from a dense-protocol model')
    cfgs = L.make_layer_cfgs(n_layers, variant)
    if warm_steps > 0 and any((getattr(c, 'kind', None) != 'full' and (not getattr(c, 'use_sink', True)) for c in cfgs)):
        raise ValueError(f'{variant} disables the attention sink on its sparse layers; the dense-warmup forward is only validated against sink-enabled layers, so running it under warmup would mix two protocols')
    model = L.SmallGPT(vocab, d, n_layers, n_heads, d_head, seq_len, cfgs, mlp_ratio=mlp_ratio).to(device)
    n_param = L.count_params(model)
    print(f'\n[{variant} seed={seed}] WARMUP warm_steps={warm_steps} total={steps}  params={n_param / 1000000.0:.2f}M  mlp_ratio={mlp_ratio:.2f}')
    is_no_decay = L.is_no_decay
    is_delta = L.is_delta_param
    decay = [p for n_, p in model.named_parameters() if p.requires_grad and (not is_no_decay(n_)) and (not is_delta(n_))]
    ndecay = [p for n_, p in model.named_parameters() if p.requires_grad and is_no_decay(n_) and (not is_delta(n_))]
    dpar = [p for n_, p in model.named_parameters() if p.requires_grad and is_delta(n_)]
    # Optimizer / betas / eps come from the SAME module constants every other
    # arm trains under (L.OPT_ADAMW_BETAS / L.OPT_ADAMW_EPS).  The record's
    # `train_knobs` descriptor (checked field-by-field by `pair_reason`)
    # makes the shared settings an auditable fact instead of an assumption.
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': weight_decay, 'lr_scale': 1.0}, {'params': ndecay, 'weight_decay': 0.0, 'lr_scale': 1.0}, {'params': dpar, 'weight_decay': 0.0, 'lr_scale': delta_lr_mult}], lr=lr, betas=L.OPT_ADAMW_BETAS, eps=L.OPT_ADAMW_EPS, foreach=True)

    def lr_at(step):
        if warmup > 0 and step < warmup:
            return lr * (step + 1) / warmup
        t = (step - warmup) / max(steps - warmup, 1)
        return lr * (L.LR_FLOOR_MULT + L.LR_COSINE_AMP * (1.0 + math.cos(math.pi * t)))
    bpe = L.batch_iter(train_ids, seq_len, batch_size, device, seed=seed)
    _eval_batch = val_batch[:eval_subset]
    warm_on = warm_steps > 0
    for _blk in model.blocks:
        _blk.attn._dense_warmup = warm_on
    t0 = time.time()
    losses, ppl_hist, warm_hist, switch_ppl = ([], [], [], None)
    _lbuf = []
    _ce_baseline = None
    # fixed-capacity window: the surviving window contents (and hence
    # final_loss_smoothed) are identical to the list form.
    _tail = collections.deque(maxlen=50)
    model.train()
    x = y = logits = ce = loss = None
    # same cyclic-GC handling as exp_lib.train_variant: collections run at a
    # fixed cadence and the original GC state is restored on every exit path.
    _gc_relief = gc.isenabled()
    if _gc_relief:
        gc.disable()
    # Same CUDA checkpoint guard as exp_lib.train_variant: when activation
    # checkpointing is selected, recompute the identical pure forward during
    # backward.  Values/gradients are unchanged; the explicit CSA_CKPT
    # setting still wins.
    _ckpt_saved = L._CKPT['on']
    _auto_ckpt, _ = L._auto_ckpt_mode(device, n_param, n_layers, d, mlp_ratio, batch_size, seq_len, cfgs)
    if _auto_ckpt:
        L._CKPT['on'] = _auto_ckpt
    try:
        for step in range(steps):
            if deadline_ts is not None and time.time() > deadline_ts:
                wall = time.time() - t0
                print(f'[{variant} seed={seed} warm={warm_steps}] time-cap deadline reached after {step} steps ({wall / 60:.1f} min) — stopping; this cell produced NO measurement and will be retried.')
                del model, opt, bpe, decay, ndecay, dpar, x, y, logits, ce, loss
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                return {'cap_truncated': True, 'steps_done': step, 'train_time_s': wall}
            if warm_steps > 0 and step == warm_steps:
                for _blk in model.blocks:
                    _blk.attn._dense_warmup = False
                switch_ppl = float(L.eval_ppl(model, _eval_batch, device))
                print(f'  [warmup] step {step}: dense -> SPARSE (h=val PPL {switch_ppl:.2f})')
            base = lr_at(step)
            for g in opt.param_groups:
                g['lr'] = base * g.get('lr_scale', 1.0)
            x, y = next(bpe)
            logits = None
            ce = model.ce_loss(x, y, vocab)
            _blp = getattr(model, 'boundary_logp', None)
            _ce_d = None
            if _blp is not None:
                # REINFORCE baseline kept on the device: no .item() before
                # backward.  The baseline is updated after the optimizer step,
                # matching exp_lib's post-backward policy.
                _ce_d = ce.detach()
                if _ce_baseline is None:
                    _ce_baseline = _ce_d.new_zeros(())
                _bdec = max(int(getattr(model, 'boundary_decisions', 0) or 0), 1)
                loss = ce + comp_lambda * model.comp_reg + L.BOUNDARY_PG_COEF * (-(_ce_d - _ce_baseline) * _blp / _bdec)
            else:
                loss = ce + comp_lambda * model.comp_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), L.GRAD_CLIP_NORM)
            opt.step()
            # No gradient is needed after the update.
            opt.zero_grad(set_to_none=True)
            if _ce_d is not None:
                _ce_baseline.mul_(0.99).add_(_ce_d, alpha=0.01)
            _lbuf.append(ce.detach())
            _tail.append(ce.detach())
            # Drop the step's graph references before the next forward;
            # trunk() rebuilds comp_reg each call.
            loss = ce = None
            model.comp_reg = None
            if len(_lbuf) >= 512:
                losses.extend(torch.stack(_lbuf).tolist())
                _lbuf.clear()
            if (step + 1) % 1024 == 0:
                gc.collect()
            if eval_every and ((step + 1) % eval_every == 0 or step == steps - 1):
                _pv = float(L.eval_ppl(model, _eval_batch, device))
                (warm_hist if getattr(model.blocks[0].attn, '_dense_warmup', False) else ppl_hist).append([step + 1, _pv])
            if log_every and (step % log_every == 0 or step == steps - 1):
                if _lbuf:
                    losses.extend(torch.stack(_lbuf).tolist())
                    _lbuf.clear()
                print(f'  step {step:5d}  loss {float(losses[-1]):.4f}  lr {opt.param_groups[0]['lr']:.2e}  dense={getattr(model.blocks[0].attn, '_dense_warmup', False)}  ({(time.time() - t0) / max(step + 1, 1) * 1000:.0f}ms/step)')
    finally:
        L._CKPT['on'] = _ckpt_saved
        if _gc_relief:
            gc.enable()
    wall = time.time() - t0
    if deadline_ts is not None and time.time() > deadline_ts:
        print(f'[{variant} seed={seed} warm={warm_steps}] time-cap deadline reached before the final evaluation ({wall / 60:.1f} min) — truncating; this cell produced NO measurement and will be retried.')
        del model, opt, bpe, decay, ndecay, dpar, x, y, logits, ce, loss
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return {'cap_truncated': True, 'steps_done': steps, 'train_time_s': wall}
    del x, y, logits, ce, loss
    _eval_stats = L.eval_ppl(model, val_batch, device, return_stats=True)
    ppl = float(_eval_stats['ppl'])
    stats = L.compression_report(model, val_batch, device, val_bnd=val_bnd)
    print(f'[{variant} seed={seed} warm={warm_steps}] val PPL = {ppl:.3f}  ({wall / 60:.1f} min)')
    if _lbuf:
        losses.extend(torch.stack(_lbuf).tolist())
        _lbuf.clear()
    losses = [float(v) for v in losses]
    res = {'variant': variant, 'seed': seed, 'ppl': ppl, 'params': n_param, 'losses': losses, 'stats': stats, 'steps': steps, 'tokens_seen': steps * batch_size * seq_len, 'train_time_s': wall, 'warm_steps': warm_steps, 'ppl_at_switch': switch_ppl, 'final_loss_smoothed': float(torch.stack(list(_tail)).mean()) if _tail else float('nan'), 'ppl_history': ppl_hist, 'ppl_history_dense_prefix': warm_hist, 'delta_trace': {}, 'train_knobs': L.training_knobs(), 'rng_fp': _rng_fp, 'data_fp': L._ids_fp(train_ids),
           'nll_sum': _eval_stats.get('nll_sum'), 'n_tokens': _eval_stats.get('n_tokens'),
           'nll_per_row': _eval_stats.get('nll_per_row'), 'n_tokens_per_row': _eval_stats.get('n_tokens_per_row'),
           'eval_rows': _eval_stats.get('eval_rows'), 'eval_seed': _eval_stats.get('eval_seed'),
           'eval_subset': _eval_stats.get('eval_subset'),
           'eval_fp': L._ids_fp(np.ascontiguousarray(val_batch).reshape(-1)),
           'param_breakdown': L.param_breakdown(model),
           'kv_cache_elements_per_token': L.kv_cache_elements_per_token(model),
           'train_objective': L.train_objective(model, comp_lambda=comp_lambda, delta_lr_mult=delta_lr_mult)}
    del model, opt, bpe, decay, ndecay, dpar
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return res

def _warm_fp(cfg, vocab, train_ids, val_batch):
    """Resume fingerprint of a warmup cell — identical for the parent planner
    and for a par.py child recomputing it from the same cached corpus."""
    d = cfg.get('d', 256)
    n_layers = cfg.get('n_layers', 6)
    n_heads = cfg.get('n_heads', 8)
    d_head = cfg.get('d_head', 32)
    _wu = cfg['warmup']
    _mkey = ','.join(sorted((str(_x) for _x in (set(PARAM_MATCHED_V7) | {'full_matched', 'full_sw128_matched'}))))
    return f'steps{cfg['steps']}_sl{cfg['seq_len']}_bs{cfg['batch_size']}_nt{cfg['n_train_tokens']}_lr{cfg['lr']}_wd{cfg['weight_decay']}_wu{_wu}_cl{cfg.get('comp_lambda', 0.05)}_dlm{cfg.get('delta_lr_mult', 10.0)}_d{d}_L{n_layers}_H{n_heads}_Dh{d_head}_v{vocab}_mt{_mkey}_mr{cfg.get('mlp_match_ref', 'csa_dynamic')}_ee{cfg.get('eval_every', 0)}_es{cfg.get('eval_subset', 128)}_det{L.determinism_label()}_df{L._ids_fp(train_ids)}-{L._ids_fp(np.ascontiguousarray(val_batch).reshape(-1))}_cs{CKPT_CODE}'


def run_warmup_cell(cfg, v, warm, seed, guard=None, ratios=None, fp=None, train_ids=None, val_batch=None, vocab=None, val_bnd=None, prebook=False):
    """Run exactly ONE (variant, warm-steps, seed) warmup cell with a locked
    summary merge.  Shared by the serial loop and by par.py children."""
    cfg = dict(cfg)
    outdir = cfg['outdir']
    d = cfg.get('d', 256)
    n_layers = cfg.get('n_layers', 6)
    n_heads = cfg.get('n_heads', 8)
    d_head = cfg.get('d_head', 32)
    spath = os.path.join(outdir, 'summary.json')
    key = f'{v}::w{warm}::seed{seed}'
    out = {'status': 'ok', 'key': key, 'variant': v, 'seed': seed, 'warm': warm}
    ratios = {} if ratios is None else ratios
    if v not in ratios:
        ratios[v] = L.variant_mlp_ratio(v, vocab, d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, seq_len=cfg['seq_len'], matched=set(PARAM_MATCHED_V7) | {'full_matched', 'full_sw128_matched'}, ref_variant=cfg.get('mlp_match_ref', 'csa_dynamic'))
    _cur = L.locked_json_read(spath).get(key)
    _drop_msg = None
    if isinstance(_cur, dict) and ((_cur.get('synthesized')) or not L.result_is_current(_cur, CKPT_CODE, 'ppl')):
        _drop_msg = f'[resume] {key} holds a record that is not current under this code semantics (stale code stamp, missing ppl, or synthesized) — DROPPING it before retraining, so a failed or truncated attempt cannot leave the old reading in place looking like a fresh result'
        _cur = None
    if L.result_is_current(_cur, CKPT_CODE, 'ppl'):
        if _cur.get('run_cfg') == fp:
            print(f'[skip] {key} already completed (resume)')
            out['status'] = 'skipped_resume'
            return out
        _drop_msg = f'[resume] {key} carries no matching config fingerprint (stored {_cur.get("run_cfg")!r}) — re-running and overwriting so a config change can never be mistaken for a fresh result'
    if _drop_msg is not None:
        print(_drop_msg)

        def _drop_cell(summary, _key=key):
            summary.pop(_key, None)
            return summary
        L.locked_json_update(spath, _drop_cell)
        out['stale_dropped'] = True
    booked = 0.0
    _book_tok = f'warm|{key}|pid{os.getpid()}'
    if guard is not None:
        est = guard.estimate_seconds(cfg['steps'], d=d, n_layers=n_layers, seq_len=cfg['seq_len'], batch_size=cfg['batch_size'])
        if prebook and est:
            guard.book(est, token=_book_tok)
            booked = est
        if not guard.can_start(0 if booked else est):
            if booked:
                guard.book(-booked, token=_book_tok)
            print(f'[cap] SKIP {key}: projected {est / 3600:.2f} h would pass the configured cap')
            out['status'] = 'skipped_cap'
            return out
    t_run = time.time()
    _run_id = L.new_experiment_id('warm')
    _started_at = time.strftime('%Y-%m-%dT%H:%M:%S')
    _deadline = None
    if guard is not None:
        _deadline = guard.deadline_ts()
        if booked:
            # the guard's spent total already contains this cell's own
            # pre-booking; add it back so the cell is not truncated by the
            # estimate of the time it is about to use (see exp_lib._run_cell).
            _deadline += booked
    rec = None
    try:
        try:
            rec = train_warmup(v, train_ids, val_batch, vocab, seed=seed, warm_steps=warm, d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, seq_len=cfg['seq_len'], batch_size=cfg['batch_size'], steps=cfg['steps'], lr=cfg['lr'], weight_decay=cfg['weight_decay'], warmup=cfg['warmup'], comp_lambda=cfg['comp_lambda'], delta_lr_mult=cfg.get('delta_lr_mult', 10.0), eval_every=cfg.get('eval_every', 0), eval_subset=cfg.get('eval_subset', 128), val_bnd=val_bnd, mlp_ratio=ratios[v], deadline_ts=_deadline)
            _prov = L.experiment_provenance(cfg, train_ids=train_ids, val_batch=val_batch, vocab=vocab)
            _hp = L.run_hyperparameters(cfg, seed=seed, d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, vocab=vocab, mlp_ratio=ratios[v])
            _hp['warm_steps'] = int(warm)

            def _apply(summary):
                if rec.get('cap_truncated'):
                    print(f'[cap] {key} was truncated after {rec.get('steps_done')} steps — its partial record is DISCARDED (it holds no `ppl`) and the key is left ABSENT. Raise the cap and re-run to retry this cell.')
                    summary.pop(key, None)
                    out['truncated'] = True
                else:
                    summary[key] = rec
                    summary[key]['_code'] = CKPT_CODE
                    summary[key]['run_cfg'] = fp
                    summary[key]['provenance'] = _prov
                    # Same top-level traceability fields the exp_lib record
                    # carries: run_manifest integrity counters and any reader
                    # that audits a record without unwrapping `provenance`
                    # need them at record level, not only nested.
                    summary[key]['data_version'] = (_prov.get('data') or {}).get('data_version')
                    summary[key]['val_fp'] = (_prov.get('data') or {}).get('val_fp')
                    summary[key]['hyperparams'] = _hp
                    summary[key]['run_id'] = _run_id
                    summary[key]['started_at'] = _started_at
                    summary[key]['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
                    if os.path.exists(os.path.join(outdir, 'overfit_sanity.json')):
                        summary[key]['sanity_ref'] = 'overfit_sanity.json'
                return summary
            L.locked_json_update(spath, _apply)
            if out.get('truncated'):
                out['status'] = 'truncated'
                L.log_experiment_event(outdir, {'kind': 'cell', 'tracking_schema': 2, 'run_id': _run_id, 'status': 'truncated', 'started_at': _started_at, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'key': key, 'variant': v, 'seed': int(seed), 'warm_steps': int(warm), 'run_cfg': fp, 'code': CKPT_CODE, 'reason': 'hard runtime cap reached; partial record not written'})
            elif isinstance(rec, dict) and not rec.get('cap_truncated'):
                _losses = rec.get('losses') if isinstance(rec.get('losses'), list) else []
                L.log_experiment_event(outdir, {'kind': 'cell', 'tracking_schema': 2, 'run_id': _run_id, 'status': 'ok', 'started_at': _started_at, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'elapsed_s': time.time() - t_run, 'key': key, 'variant': v, 'seed': int(seed), 'warm_steps': int(warm), 'run_cfg': fp, 'code': CKPT_CODE, 'provenance': _prov, 'hyperparams': _hp, 'data_fp': rec.get('data_fp'), 'data_version': (_prov.get('data') or {}).get('data_version'), 'val_fp': (_prov.get('data') or {}).get('val_fp'), 'rng_fp': rec.get('rng_fp'), 'sanity_ref': ('overfit_sanity.json' if os.path.exists(os.path.join(outdir, 'overfit_sanity.json')) else None), 'metrics': {'ppl': rec.get('ppl'), 'ppl_at_switch': rec.get('ppl_at_switch'), 'params': rec.get('params'), 'steps': rec.get('steps'), 'tokens_seen': rec.get('tokens_seen'), 'train_time_s': rec.get('train_time_s'), 'final_loss_smoothed': rec.get('final_loss_smoothed')}, 'curve_summary': {'n_points': len(_losses), 'first': _losses[0] if _losses else None, 'last': _losses[-1] if _losses else None, 'best': min(_losses) if _losses else None}, 'losses': _losses, 'ppl_history': rec.get('ppl_history'), 'ppl_history_dense_prefix': rec.get('ppl_history_dense_prefix')})
        except Exception as e:
            _err = e
            _err_tb = traceback.format_exc()
            out['status'] = 'error'
            out['error'] = f'{_err}'

            def _apply_err(summary):
                if L.result_is_current(summary.get(key), CKPT_CODE, 'ppl'):
                    print(f'[{key}] FAILED: {_err} — keeping the previous MEASURED record (current under this code semantics; the error record holds no `ppl` and must not replace it)')
                    print(_err_tb)
                else:
                    summary[key] = {'variant': v, 'seed': seed, 'warm_steps': warm, 'error': _err_tb}
                    print(f'[{key}] FAILED: {_err}')
                return summary
            L.locked_json_update(spath, _apply_err)
            L.log_experiment_event(outdir, {'kind': 'cell', 'tracking_schema': 2, 'run_id': _run_id, 'status': 'error', 'started_at': _started_at, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'elapsed_s': time.time() - t_run, 'key': key, 'variant': v, 'seed': int(seed), 'warm_steps': int(warm), 'run_cfg': fp, 'code': CKPT_CODE, 'error': f'{_err}', 'traceback': _err_tb})
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


def run_warmup_cell_job(job):
    """Child-process entry for one warmup cell (invoked by par.py)."""
    cfg = dict(job['cfg'])
    guard = L.TimeGuard(job['guard_cap']) if job.get('guard_cap') else None
    L.set_seed(int(job['seed']))
    train_ids, val_batch, vocab, _, val_bnd = L.load_wikitext(cfg['seq_len'], cfg['n_train_tokens'])
    # pin determinism BEFORE the fingerprint (run_warmup does the same in the
    # parent; `_warm_fp` embeds `determinism_label()`)
    if torch.cuda.is_available():
        L._pin_cuda_determinism()
    fp = _warm_fp(cfg, vocab, train_ids, val_batch)
    return run_warmup_cell(cfg, job['v'], job['extra']['warm'], job['seed'], guard=guard, ratios={}, fp=fp, train_ids=train_ids, val_batch=val_batch, vocab=vocab, val_bnd=val_bnd, prebook=True)


def run_warmup(cfg, seeds, guard=None, label='', warm_grid=(0, 5000, 10000), variants=('csa_fixed',), par=None):
    cfg = dict(cfg)
    outdir = cfg['outdir']
    os.makedirs(outdir, exist_ok=True)
    d = cfg.get('d', 256)
    n_layers = cfg.get('n_layers', 6)
    n_heads = cfg.get('n_heads', 8)
    d_head = cfg.get('d_head', 32)
    t_data = time.time()
    L.set_seed(int((list(seeds) or [0])[0]))
    train_ids, val_batch, vocab, _, val_bnd = L.load_wikitext(cfg['seq_len'], cfg['n_train_tokens'])
    if guard is not None:
        guard.record_run(time.time() - t_data, 0, 0, 0, 0, 0)
    spath = os.path.join(outdir, 'summary.json')
    L.locked_json_read(spath)
    # Warmup training uses the real corpus through its own train_warmup loop,
    # so the generic run() pre-flight never sees it.  Check the actual token
    # stream before launching the 20k-step P0W cells.
    L.overfit_guard(list(variants), train_ids=train_ids, vocab=vocab)
    _panel_started = time.strftime('%Y-%m-%dT%H:%M:%S')
    _panel_prov = L.experiment_provenance(cfg, train_ids=train_ids, val_batch=val_batch, vocab=vocab)
    L.log_experiment_event(outdir, {
        'kind': 'panel_start', 'tracking_schema': 2, 'label': label or outdir,
        'phase': 'warmup', 'started_at': _panel_started,
        'config': L._jsonable(cfg), 'seeds': list(seeds),
        'variants': list(variants), 'warm_grid': list(warm_grid),
        'code': CKPT_CODE, 'provenance': _panel_prov,
    })
    print(f'\n########## v7 phase: {label} — {len(variants)} variants x warm {warm_grid} x seeds {seeds} x {cfg['steps']} steps ##########')
    ratios = {}
    for v in variants:
        if v not in ratios:
            ratios[v] = L.variant_mlp_ratio(v, vocab, d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, seq_len=cfg['seq_len'], matched=set(PARAM_MATCHED_V7) | {'full_matched', 'full_sw128_matched'}, ref_variant=cfg.get('mlp_match_ref', 'csa_dynamic'))
    if torch.cuda.is_available():
        L._pin_cuda_determinism()
    _fp = _warm_fp(cfg, vocab, train_ids, val_batch)
    cells = [(v, warm, seed) for seed in seeds for warm in warm_grid for v in variants]
    stale_dropped = []
    dropped_truncations = []
    _ran_parallel = False
    if len(cells) > 1:
        import par as _par
        _par_ctx = par if isinstance(par, dict) else _par.resolve_ctx(par, label=label or outdir)
        if _par_ctx and _par.ctx_active(_par_ctx, len(cells)):
            module = _par_ctx.get('module') or (_caller if (_caller := sys._getframe(1).f_globals.get('__name__')) and _caller != '__main__' else 'v7_supp')
            jobs = [dict(kind='warm', module=module, label=label, cfg=cfg, v=v, seed=seed, extra={'warm': warm}, guard_cap=(guard.cap_spec() if guard is not None else None)) for v, warm, seed in cells]
            results = _par.run_jobs(jobs, _par_ctx, phase_label=label or outdir)
            for r, (v, warm, seed) in zip(results, cells):
                if not r:
                    continue
                if r.get('truncated'):
                    dropped_truncations.append(r.get('key') or f'{v}::w{warm}::seed{seed}')
                if r.get('stale_dropped'):
                    stale_dropped.append(r.get('key') or f'{v}::w{warm}::seed{seed}')
            _ran_parallel = True
    if not _ran_parallel:
        for v, warm, seed in cells:
            res = run_warmup_cell(cfg, v, warm, seed, guard=guard, ratios=ratios, fp=_fp, train_ids=train_ids, val_batch=val_batch, vocab=vocab, val_bnd=val_bnd)
            if res.get('truncated'):
                dropped_truncations.append(res['key'])
            if res.get('stale_dropped'):
                stale_dropped.append(res['key'])
    summary = L.locked_json_read(spath)
    if stale_dropped:
        _still = [k for k in stale_dropped if k not in summary]
        print(f'\n[resume] {len(stale_dropped)} warmup cell(s) held a non-current record and were dropped before retraining; {len(_still)} of them produced NO measurement this pass (cap gate / truncation / failure) and are ABSENT from {spath}:')
        for _k in _still:
            print(f'    {_k}')
        if len(_still) != len(stale_dropped):
            print(f'    ({len(stale_dropped) - len(_still)} of them succeeded and carry a fresh record.)')
    _measured = sum(1 for _r in summary.values()
                    if isinstance(_r, dict) and L.ppl_is_usable(_r.get('ppl'))
                    and not _r.get('synthesized'))
    L.log_experiment_event(outdir, {
        'kind': 'panel_finish', 'tracking_schema': 2, 'label': label or outdir,
        'phase': 'warmup', 'started_at': _panel_started,
        'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'n_cells_planned': len(cells), 'n_cells_measured': _measured,
        'n_cells_missing': max(0, len(cells) - _measured),
        'n_stale_dropped': len(stale_dropped),
        'n_truncated': len(dropped_truncations),
        'summary_ref': os.path.basename(spath), 'code': CKPT_CODE,
        'provenance': _panel_prov,
    })
    return summary

def build_niah_batch(n_seq, seq_len, n_pairs=4, vocab=8192, seed=0):
    rng = np.random.default_rng(seed)
    key_hi = max(2 + (vocab // 2 - 2) // 4, 2 + 2 * n_pairs)
    ids = rng.integers(key_hi, vocab // 2, size=(n_seq, seq_len)).astype(np.int64)
    tgt = np.full((n_seq, seq_len - 1), -100, dtype=np.int64)
    dist = np.full((n_seq, seq_len - 1), -1, dtype=np.int64)
    kdist = np.full((n_seq, seq_len - 1), -1, dtype=np.int64)
    tail = 2 * n_pairs
    hay = seq_len - tail
    band = max(hay // n_pairs, 2)
    for i in range(n_seq):
        keys = rng.choice(np.arange(2, key_hi), size=n_pairs, replace=False)
        vals = rng.choice(np.arange(vocab // 2, vocab), size=n_pairs, replace=False)
        key_to_val, val_pos = ({}, {})
        for j in range(n_pairs):
            start = j * band
            kp = start
            vp = min(start + band // 2, start + band - 1, hay - 1)
            vp = max(vp, kp + 1)
            assert kp < vp < hay, 'band too small for a (key,value) couple'
            ids[i, kp] = keys[j]
            ids[i, vp] = vals[j]
            key_to_val[int(keys[j])] = int(vals[j])
            val_pos[int(keys[j])] = int(vp)
        order = rng.permutation(n_pairs)
        for j, pj in enumerate(order):
            kp = hay + 2 * j
            ap = kp + 1
            ids[i, kp] = keys[pj]
            ids[i, ap] = rng.integers(key_hi, vocab // 2)
            tgt[i, kp] = key_to_val[int(keys[pj])]
            dist[i, kp] = kp - val_pos[int(keys[pj])]
            kdist[i, kp] = kp - pj * band
    return (ids, tgt, dist, kdist)

@torch.inference_mode()
def eval_niah(model, seq_len, device=DEVICE, n_seq=64, n_pairs=4, vocab=8192, seed=1234, chunk=16):
    was = model.training
    model.eval()
    try:
        max_pos = getattr(model, 'max_seq', seq_len)
        ids, tgt, dist, kdist = build_niah_batch(n_seq, seq_len, n_pairs, vocab, seed)
        correct = np.zeros_like(tgt, dtype=bool)
        span = min(seq_len, max_pos) if getattr(model, 'use_abs_pe', True) else seq_len
        tail = 2 * n_pairs
        # targets uploaded once and the per-chunk comparison done on-device:
        # only the [n_seq, tail-1] bool strip crosses the bus at the end.
        tgt_dev = L._h2d_async(tgt, torch.int64, device)
        correct_dev = torch.zeros(n_seq, tail - 1, dtype=torch.bool, device=device)
        i = 0
        while i < n_seq:
            j = min(i + chunk, n_seq)
            # same bit-exact staged upload as batch_iter (design notes §6.1)
            x = L._h2d_async(ids[i:j], torch.int64, device)
            trunc = getattr(model, 'use_abs_pe', True) and seq_len > max_pos
            try:
                logits = model(x[:, -span:].clamp(0, vocab - 1) if trunc else x, logits_tail=tail)
            except torch.cuda.OutOfMemoryError:
                del x
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                if chunk <= 1:
                    raise
                chunk = max(1, chunk // 2)
                print(f'[niah] eval device-memory exhaustion; retrying with chunk={chunk}')
                continue
            _t = tgt_dev[i:j, -(tail - 1):]
            correct_dev[i:j] = (logits[:, :-1].argmax(-1) == _t) & (_t >= 0)
            del x, logits
            i = j
        correct[:, -(tail - 1):] = correct_dev.cpu().numpy()
        del tgt_dev, correct_dev
        scored_cols = tail - 1
        m = np.zeros_like(tgt, dtype=bool)
        m[:, -scored_cols:] = tgt[:, -scored_cols:] >= 0
        if m.sum() == 0:
            return {'n': 0, 'by_dist': {}}
        d = dist[m]
        c = correct[m]
        if getattr(model, 'use_abs_pe', True) and seq_len > max_pos:
            _cut = seq_len - span
            _cols = np.nonzero(m)[1]
            _in_ctx = ((_cols - d) >= _cut) & ((_cols - kdist[m]) >= _cut)
            d = d[_in_ctx]
            c = c[_in_ctx]
            if d.size == 0:
                return {'n': 0, 'by_dist': {}}
        buckets = [(0, 128), (128, 512), (512, 2048), (2048, 8192), (8192, 10 ** 9)]
        by = {}
        for lo, hi in buckets:
            sel = (d >= lo) & (d < hi)
            if sel.sum():
                by[f'{lo}-{(hi if hi < 10 ** 9 else 'inf')}'] = {'acc': float(c[sel].mean()), 'n': int(sel.sum())}
        _trunc = bool(getattr(model, 'use_abs_pe', True) and seq_len > max_pos)
        return {'acc': float(c.mean()), 'n': int(c.size), 'by_dist': by, 'eval_span': int(span), 'train_pos': int(max_pos), 'truncated': _trunc, 'mid_pos': int(seq_len - span) if _trunc else 0}
    finally:
        model.train(was)

def exact_sign_permutation(deltas):
    d = np.asarray(deltas, dtype=float)
    d_in = d
    d = d[np.isfinite(d)]
    n = len(d)
    n_dropped = int(len(d_in) - n)

    def _extras(res, d, n):
        """效应量与逐种子差值（评审点 4）：报告必须给出逐种子 Δ 与
        d_z（配对差均值/样本标准差），并把该 n 下的 p 值分辨率下限
        一并写出，避免把分辨率上限误读为数据结论。纯增量字段。"""
        res['per_seed'] = [float(x) for x in d.tolist()]
        res['p_floor'] = math.ldexp(2.0, -int(n)) if n > 0 else float('nan')
        _sd = res.get('std', 0.0)
        _mu = res.get('mean', float('nan'))
        if (not isinstance(_sd, (int, float))) or (not math.isfinite(_mu)):
            res['cohens_dz'] = float('nan')
        elif _sd > 1e-12:
            res['cohens_dz'] = float(_mu / _sd)
        elif abs(_mu) < 1e-300:
            res['cohens_dz'] = 0.0
        else:
            res['cohens_dz'] = float('inf') if _mu > 0 else float('-inf')
        res['test'] = 'exact two-sided sign-flip over paired per-seed deltas'
        return res
    if n == 0:
        return _extras({'n': 0, 'mean': float('nan'), 'std': float('nan'), 'p_exact_signflip': float('nan'), 'n_flips': 0, 'n_dropped': n_dropped}, d, n)
    obs = abs(d.mean())
    if n > 20:
        rng = np.random.default_rng(0)
        B = 200000
        cnt = 1
        step = max(1, min(B, (1 << 24) // max(n, 1)))
        for off in range(0, B, step):
            bs = min(step, B - off)
            signs = np.where(rng.random((bs, n)) < 0.5, 1.0, -1.0)
            cnt += int((np.abs((signs * d[None, :]).mean(1)) >= obs - 1e-12).sum())
        p = float(cnt / (B + 1))
        return _extras({'n': n, 'mean': float(d.mean()), 'std': float(d.std(ddof=1)) if n > 1 else 0.0, 'p_exact_signflip': p, 'n_flips': B, 'n_dropped': n_dropped}, d, n)
    cnt = 0
    total = 1 << n
    bitpos = np.arange(n, dtype=np.int64)[None, :]
    step = max(1, min(total, (1 << 24) // max(n, 1)))
    for off in range(0, total, step):
        bs = min(step, total - off)
        masks = np.arange(off, off + bs, dtype=np.int64)[:, None]
        flips = np.where((masks & (1 << bitpos)) != 0, 1.0, -1.0)
        cnt += int((np.abs((flips * d[None, :]).mean(1)) >= obs - 1e-12).sum())
    p = float(cnt / total)
    return _extras({'n': n, 'mean': float(d.mean()), 'std': float(d.std(ddof=1)) if n > 1 else 0.0, 'p_exact_signflip': p, 'n_flips': int(total), 'n_dropped': n_dropped}, d, n)

def flops_analysis(outdir='analysis_v7', seq_lens=None):
    os.makedirs(outdir, exist_ok=True)
    seq_lens = seq_lens or [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    d, H, Dh, L_layers = (256, 8, 32, 6)
    m, k, w, mph = (4, 32, 128, 64)
    nIH, cI = (4, 64)
    rows = []
    for n in seq_lens:
        B = max(n // m, 1)
        Bh = max(n // mph, 1)
        dense = 2 * (2 * H * Dh * n)
        idx = 2 * cI * B + 2 * nIH * B
        core = 2 * (2 * H * Dh * min(k, B) + 2 * H * Dh * min(w, n))
        csa = idx + core
        hca = 2 * (2 * H * Dh * min(Bh, n) + 2 * H * Dh * min(w, n))
        hybrid = 0.5 * csa + 0.5 * hca
        kv_dense = 2 * H * Dh * n
        kv_csa = 2 * H * Dh * (B + min(w, n))
        kv_hca = 2 * H * Dh * (Bh + min(w, n))
        kv_hybrid = 0.5 * (kv_csa + kv_hca)
        rows.append({'seq_len': n, 'sel_ratio': min(k, B) / B if B else 1.0, 'dense_flops': dense, 'csa_flops': csa, 'hca_flops': hca, 'hybrid_flops': hybrid, 'csa_over_dense': csa / dense, 'hybrid_over_dense': hybrid / dense, 'kv_dense': kv_dense, 'kv_csa': kv_csa, 'kv_hca': kv_hca, 'kv_hybrid': kv_hybrid, 'csa_kv_over_dense': kv_csa / kv_dense, 'hybrid_kv_over_dense': kv_hybrid / kv_dense})
    cross = next((r['seq_len'] for r in rows if r['csa_over_dense'] < 1.0), None)
    if cross is None:
        print('[flops] NOTE: under this accounting CSA is NOT cheaper than dense at any tested length — no crossover exists in this config; reporting None.')
    # The analytic crossover is a quotable claim (it reaches REPORT_v7.md and
    # the README table), so the artifact carries the code/git provenance and a
    # ledger entry like every measured panel — an unstamped json could not be
    # told apart from one regenerated after a formula change.
    L.stamped_analysis('flops_analytic', {'config': {'d': d, 'heads': H, 'd_head': Dh, 'layers': L_layers, 'm': m, 'topk': k, 'window': w, 'm_hca': mph}, 'crossover_seq_len_csa_beats_dense': cross, 'rows': rows}, outdir=outdir, name='flops_analytic.json')
    L.atomic_write_csv(os.path.join(outdir, 'flops_analytic.csv'), None, rows)
    _flops_plot(rows, outdir)
    print(f'[flops] crossover (CSA cheaper than dense) at seq = {cross}')
    print(f'[flops] wrote {outdir}/flops_analytic.json|.csv|.png')
    return rows

def _flops_plot(rows, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = [r['seq_len'] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].plot(n, [r['dense_flops'] / 1000000.0 for r in rows], 'o-', label='dense')
    ax[0].plot(n, [r['csa_flops'] / 1000000.0 for r in rows], 'o-', label='CSA')
    ax[0].plot(n, [r['hybrid_flops'] / 1000000.0 for r in rows], 'o-', label='hybrid')
    ax[0].plot(n, [r['hca_flops'] / 1000000.0 for r in rows], 'o-', label='HCA')
    ax[0].set_xscale('log', base=2)
    ax[0].set_yscale('log')
    ax[0].set_xlabel('sequence length')
    ax[0].set_ylabel('attn FLOPs/token (M)')
    ax[0].set_title('analytic per-token attention FLOPs')
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot(n, [r['csa_over_dense'] for r in rows], 'o-', label='CSA/dense')
    ax[1].plot(n, [r['hybrid_over_dense'] for r in rows], 'o-', label='hybrid/dense')
    ax[1].axhline(1.0, color='k', lw=0.8, ls='--')
    ax[1].set_xscale('log', base=2)
    ax[1].set_yscale('log')
    ax[1].set_xlabel('sequence length')
    ax[1].set_ylabel('ratio vs dense')
    ax[1].set_title('crossover')
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    ax[2].plot(n, [r['kv_dense'] / 1000000.0 for r in rows], 'o-', label='dense')
    ax[2].plot(n, [r['kv_csa'] / 1000000.0 for r in rows], 'o-', label='CSA')
    ax[2].plot(n, [r['kv_hybrid'] / 1000000.0 for r in rows], 'o-', label='hybrid')
    ax[2].set_xscale('log', base=2)
    ax[2].set_yscale('log')
    ax[2].set_xlabel('sequence length')
    ax[2].set_ylabel('KV cache (M elem/token)')
    ax[2].set_title('KV cache per token')
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'flops_analytic.png'), dpi=120)
    plt.close(fig)

def bootstrap_report(outdirs, out='analysis_v7/stats.json'):
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    recs = {}
    for od in outdirs:
        sp = os.path.join(od, 'summary.json')
        if not os.path.exists(sp):
            continue
        try:
            s = json.load(open(sp, encoding='utf-8'))
        except Exception as e:
            print(f'[stats] WARNING: cannot read {sp} ({type(e).__name__}: {e}) — its panel is EXCLUDED from these statistics')
            continue
        _legacy_ok = L.legacy_warm_tags(s)
        for k, r in s.items():
            if isinstance(r, dict) and 'seed' in r and (not r.get('synthesized')) and L.ppl_is_usable(r.get('ppl')):
                tag = f'{od.split('/')[-1]}::{r.get('variant')}'
                if r.get('warm_steps') is not None:
                    tag += f'::w{r['warm_steps']}'
                else:
                    _wt = L.record_tag(r, k, _legacy_ok)
                    if _wt:
                        tag += f'::{_wt}'
                _slot = recs.setdefault(tag, {})
                _old = _slot.get(r['seed'])
                if _old is not None:
                    if _old.get('_dup'):
                        continue
                    _new_cur = r.get('_code') == CKPT_CODE
                    _old_cur = _old.get('_code') == CKPT_CODE
                    if _new_cur and not _old_cur:
                        print(f'[stats] WARNING: {tag} seed {r['seed']} holds TWO measurable records — the PPL is ambiguous; the one stamped with the current code semantics is kept and the duplicate is dropped')
                        _slot[r['seed']] = r
                    elif _new_cur == _old_cur:
                        print(f'[stats] WARNING: {tag} seed {r['seed']} holds TWO measurable records with identical stamp status — the seed is EXCLUDED from every comparison rather than quoting an arbitrarily picked PPL')
                        _dup = dict(_old)
                        _dup['_dup'] = True
                        _slot[r['seed']] = _dup
                    continue
                _slot[r['seed']] = r
    pairs = [('warmup w=5000 vs scratch', 'results_lm_v7_warmup::csa_fixed::w5000', 'results_lm_v7_warmup::csa_fixed::w0'), ('warmup w=10000 vs scratch', 'results_lm_v7_warmup::csa_fixed::w10000', 'results_lm_v7_warmup::csa_fixed::w0'), ('CSA+RoPE vs CSA absPE', 'results_lm_v7_rope::csa_fixed_rope', 'results_lm_v7_rope::csa_fixed'), ('CSA+RoPE vs dense+RoPE', 'results_lm_v7_rope::csa_fixed_rope', 'results_lm_v7_rope::full_rope'), ('hybrid+RoPE vs dense+RoPE', 'results_lm_v7_rope::hybrid_fixed_rope', 'results_lm_v7_rope::full_rope')]
    per_variant = {}
    for k, v in recs.items():
        _m = {s: r for s, r in v.items() if not r.get('_dup')}
        if not _m:
            continue
        _groups = {}
        for s, r in _m.items():
            _groups.setdefault(r.get('run_cfg'), []).append(s)
        if len(_groups) > 1:
            _cur_sfx = f'_cs{CKPT_CODE}'
            _ranked = sorted(_groups.items(), key=lambda kv: (not (isinstance(kv[0], str) and kv[0].endswith(_cur_sfx)), -len(kv[1]), str(kv[0])))
            _keep_seeds = _ranked[0][1]
            _dropped = [s for _g, _ss in _ranked[1:] for s in _ss]
            print(f'[stats] WARNING: {k}: seed(s) {_dropped} were measured under a DIFFERENT run_cfg group than seed(s) {_keep_seeds} — the per-variant mean pools only one configuration group ({len(_keep_seeds)} seed(s), current code stamp preferred); pooling across configurations is not allowed')
            _m = {s: _m[s] for s in _keep_seeds}
        per_variant[k] = {'ppls': {s: r['ppl'] for s, r in _m.items()}, 'mean': float(np.mean([r['ppl'] for r in _m.values()]))}
    out_d = {'per_variant': per_variant, 'comparisons': {}}
    for name, a, b in pairs:
        if a not in recs or b not in recs:
            _missing = [t for t in (a, b) if t not in recs]
            print(f'[stats] {name}: comparison OMITTED — panel tag(s) {_missing} absent from the loaded summaries (not measured under this naming, or the directory was renamed); not quoting a cross-panel delta')
            out_d['comparisons'][name] = {'omitted': f'panel tag(s) {_missing} absent', 'n': 0}
            continue
        common = sorted(set(recs[a]) & set(recs[b]))
        dl, skipped, unstamped = ([], [], 0)
        mismatch_fields = set()
        for s in common:
            ra, rb = (recs[a][s], recs[b][s])
            if ra.get('_dup') or rb.get('_dup'):
                mismatch_fields.add('an ambiguous duplicate measurement on one side')
                skipped.append(s)
                continue
            if ra.get('run_cfg') is None:
                unstamped += 1
            if rb.get('run_cfg') is None:
                unstamped += 1
            reason = L.pair_reason(ra, rb)
            if reason is not None:
                if reason.startswith('disagreeing') or reason.startswith('unverifiable'):
                    mismatch_fields.add(reason)
                skipped.append(s)
                continue
            dl.append(ra['ppl'] - rb['ppl'])
        if skipped:
            why = '; '.join(sorted(mismatch_fields)) if mismatch_fields else 'different run_cfg'
            print(f'[stats] {name}: {len(skipped)} seed(s) NOT paired — the two sides have {why}; excluded from the test')
        if not dl:
            print(f'[stats] {name}: NO usable pair after the run_cfg/steps checks — comparison omitted rather than quoting a cross-configuration delta')
            out_d['comparisons'][name] = {'omitted': 'no usable pair after the run_cfg/steps checks', 'n': 0, 'n_skipped_config_mismatch': len(skipped), 'n_unstamped': unstamped}
            continue
        res = exact_sign_permutation(dl)
        res['n_skipped_config_mismatch'] = len(skipped)
        res['n_unstamped'] = unstamped
        out_d['comparisons'][name] = res
    # Family-wise correction over the report's five pre-listed comparisons
    # (review 3 issue 3).  This is a guard against multiple-comparison
    # over-reading; the panel remains EXPLORATORY because its comparisons and
    # seeds were chosen after seeing validation results (design_notes §17).
    _pairs = [(k, float(v.get('p_exact_signflip'))) for k, v in out_d['comparisons'].items()
              if isinstance(v, dict) and isinstance(v.get('p_exact_signflip'), (int, float))
              and math.isfinite(float(v['p_exact_signflip']))]
    _h = L.holm_adjust(_pairs) if _pairs else {}
    for _k, _v in out_d['comparisons'].items():
        if isinstance(_v, dict):
            _v['p_holm'] = (_h.get(_k) or {}).get('p_holm') if _k in _h else None
    out_d['family'] = {'declared': len(out_d['comparisons']), 'tested': len(_pairs), 'alpha': 0.05,
                       'correction': 'holm step-down', 'evidence_level': 'exploratory',
                       'note': 'v7 panels were selected on the validation split; Holm limits over-reading but does not make them confirmatory'}
    out_d = L.stamped_analysis('v7_bootstrap_stats', out_d, outdir=(os.path.dirname(out) or '.'), name=os.path.basename(out))
    print(f'[stats] wrote {out}')
    for k, v in out_d['comparisons'].items():
        if 'omitted' in v:
            print(f'  {k:34s} omitted — {v['omitted']}')
            continue
        warn = f'  [{v['n_skipped_config_mismatch']} seed(s) dropped: config mismatch]' if v.get('n_skipped_config_mismatch') else ''
        _ph = v.get('p_holm')
        _pht = ('%.3f' % _ph) if isinstance(_ph, (int, float)) and math.isfinite(float(_ph)) else 'n/a'
        print(f'  {k:34s} Δ={v['mean']:+7.2f} ± {v.get('std', 0):5.2f}  p(sign-flip)={v['p_exact_signflip']:.3f}  p_Holm={_pht}  n={v['n']}{warn}')
    return out_d

def git_push(msg):
    if os.environ.get('V7_NO_PUSH'):
        print(f'[git] push skipped (V7_NO_PUSH): {msg}')
        return True
    run = lambda *a: subprocess.run(a, cwd=REPO, capture_output=True, text=True, encoding='utf-8', errors='replace')
    ra = run('git', 'add', '-A')
    if ra.returncode != 0:
        print(f'[git] add FAILED: {(ra.stdout + ra.stderr)[-300:]} — NOT pushing (a commit could miss unstaged work); leaving the tree for inspection')
        return False
    r = run('git', 'commit', '-m', msg)
    committed = r.returncode == 0
    if not committed and 'nothing to commit' not in r.stdout + r.stderr:
        print(f'[git] commit problem: {(r.stdout + r.stderr)[-300:]}')
        if run('git', 'status', '--porcelain').stdout.strip():
            print('[git] the commit failed while changes are still uncommitted — NOT pushing (a push would publish stale history and report success); leaving the tree for inspection')
            return False
    _branch = run('git', 'rev-parse', '--abbrev-ref', 'HEAD').stdout.strip()
    if _branch != 'HEAD':
        rb = run('git', 'pull', '--rebase', '--autostash', 'origin', _branch)
        if committed and rb.returncode != 0:
            print(f'[git] rebase onto origin FAILED: {(rb.stdout + rb.stderr)[-300:]}')
            run('git', 'rebase', '--abort')
    r = run('git', 'push', 'origin', 'HEAD')
    ok = r.returncode == 0
    if not ok:
        print(f'[git] push FAILED: {(r.stdout + r.stderr)[-300:]}')
        if committed:
            print('[git] WARNING: the commit exists only on this instance — the work is NOT on the remote.  Resolve and re-push before shutting the instance down.')
    else:
        print(f'[git] push OK: {msg}')
    return ok

def schedule_shutdown(delay_s=120):
    if os.environ.get('V7_NO_SHUTDOWN'):
        print('[v7] shutdown suppressed (V7_NO_SHUTDOWN)')
        return
    subprocess.Popen(['bash', '-c', f'sleep {delay_s}; shutdown -h now'], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f'[v7] instance shuts down in {delay_s}s.')
LONG_EXT = dict(L.RUN_LONG, steps=40000, n_train_tokens=110000000, eval_every=2000, outdir='results_lm_v7_long40')
PHASES = [('P0R', 'plain', dict(cfg=dict(L.RUN, outdir='results_lm_v7_rope', variants=ROPE_VARIANTS, steps=1500, matched=set(PARAM_MATCHED_V7), mlp_match_ref='csa_fixed_rope'), seeds=[0, 1, 2])), ('P0W', 'warm', dict(cfg=dict(L.RUN_LONG, outdir='results_lm_v7_warmup', variants=['csa_fixed']), seeds=[0, 1, 2], warm_grid=(0, 5000, 10000))), ('P1L', 'len', dict(outdir='results_len', vocab=8192, steps=3000, train_len=512, eval_lens=[512, 1024, 2048, 4096], n_seq=16, variants=['full', 'csa_fixed', 'full_rope', 'csa_fixed_rope'])), ('P2S', 'plain', dict(cfg=dict(L.RUN_SCALE, outdir='results_lm_v5_scale', variants=['full_matched', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic']), seeds=[2, 3])), ('P1T', 'plain', dict(cfg=dict(L.RUN, outdir='results_lm_v7_seq2k', seq_len=2048, batch_size=3, steps=1500, variants=['csa_fixed_topk8', 'csa_fixed_topk32', 'csa_fixed_topk128', 'csa_fixed_topk512', 'csa_fix_m1']), seeds=[0, 1])), ('P0E', 'plain', dict(cfg=dict(LONG_EXT, variants=['csa_fixed', 'full']), seeds=[0, 1]))]

def run_phase(name, guard, only=None, par=None):
    import par as _par
    _ctx = _par.resolve_ctx(par, label=f'v7 {name}')
    for pname, kind, payload in PHASES:
        if pname != name:
            continue
        if isinstance(_ctx, dict):
            _ctx = dict(_ctx, module='v7_supp')
        if kind == 'plain':
            s, a = L.run(payload['cfg'], seeds=payload['seeds'], guard=guard, label=f'v7 {pname}', par=_ctx)
            return s
        if kind == 'warm':
            return run_warmup(payload['cfg'], seeds=payload['seeds'], guard=guard, label=f'v7 {pname}', warm_grid=payload['warm_grid'], variants=tuple(payload['cfg'].get('variants', ('csa_fixed',))), par=_ctx)
        if kind == 'niah':
            if isinstance(_ctx, dict):
                print('[v7] NOTE: the NIAH phase has no task-level parallel path, running it serially (every other phase kind parallelises)')
            return run_niah_phase(payload, guard=guard, label=f'v7 {pname}')
        if kind == 'len':
            return run_lenphase(payload, guard, label=f'v7 {pname}', par=_ctx)
    raise SystemExit(f'unknown phase {name}')

def run_niah_phase(payload, guard=None, label=''):
    outdir = payload['outdir']
    os.makedirs(outdir, exist_ok=True)
    ckpt_dir = os.path.join(outdir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    seq_lens = payload['seq_lens']
    variants = payload['variants']
    seeds = list(payload.get('seeds', [0, 1, 2]))
    vocab = payload.get('vocab', 8192)
    n_steps = payload.get('steps', 4000)
    # This phase has its own training loop (synthetic NIAH task), so the
    # generic run() pre-flight does not cover it.  Check the model/loss/
    # gradient plumbing before the first checkpoint is written.
    L.overfit_guard(list(variants), vocab=int(vocab))
    _panel_started = time.strftime('%Y-%m-%dT%H:%M:%S')
    _panel_prov = L.experiment_provenance({'dataset': 'synthetic_niah', 'n_train_tokens': 0, 'seq_len': 512})
    L.log_experiment_event(outdir, {
        'kind': 'panel_start', 'tracking_schema': 2, 'label': label or outdir,
        'phase': 'niah', 'started_at': _panel_started,
        'config': L._jsonable(payload), 'seeds': seeds, 'variants': list(variants),
        'seq_lens': list(seq_lens), 'code': CKPT_CODE, 'provenance': _panel_prov,
    })
    spath = os.path.join(outdir, 'summary.json')
    summary = {}
    if os.path.exists(spath):
        try:
            summary = json.load(open(spath, encoding='utf-8'))
        except Exception as _e:
            print(f'[resume] FATAL: {spath} exists but cannot be parsed ({type(_e).__name__}: {_e}).  Refusing to overwrite it with an empty summary — move it aside to start fresh.')
            raise
    _mr_ref = 'csa_fixed_rope'
    for v in variants:
        mr = 4.0
        if v in PARAM_MATCHED_V7:
            mr = L.variant_mlp_ratio(v, vocab, d=256, n_layers=6, n_heads=8, d_head=32, seq_len=512, matched=set(PARAM_MATCHED_V7), ref_variant=_mr_ref)
        _recipe = f'niah_steps{n_steps}_v{vocab}_bs12_sl512_lr0.0003_mr{mr:.8g}-{_mr_ref}_reuse4'
        for seed in seeds:
            ck = os.path.join(ckpt_dir, f'{v}_seed{seed}.pt')
            stale = False
            _ckp_pre = None
            if os.path.exists(ck):
                try:
                    _meta = _load_ckpt(ck)
                except Exception as _cke:
                    print(f'[niah] {v} s{seed}: checkpoint unreadable ({type(_cke).__name__}: {_cke}) — RETRAINING')
                    stale = True
                else:
                    stale = _meta.get('code') != CKPT_CODE
                    if stale:
                        print(f'[niah] {v} s{seed}: checkpoint predates CKPT_CODE={CKPT_CODE} (code={_meta.get('code')!r}) — RETRAINING')
                    elif _meta.get('recipe') != _recipe:
                        stale = True
                        print(f'[niah] {v} s{seed}: checkpoint recipe mismatch (stored {_meta.get('recipe')!r}) — RETRAINING')
                    else:
                        _ckp_pre = _meta
                    del _meta
            if stale or not os.path.exists(ck):
                _ev_pref = f'{v}::seed{seed}::'
                for _k in [_k for _k in summary if _k.startswith(_ev_pref)]:
                    del summary[_k]

                def _drop_niah_evals(s, _pref=_ev_pref):
                    for _k in [_k for _k in s if _k.startswith(_pref)]:
                        del s[_k]
                    return s
                L.locked_json_update(spath, _drop_niah_evals)
                if guard is not None:
                    est = guard.estimate_seconds(n_steps, d=256, n_layers=6, seq_len=512, batch_size=12)
                    if not guard.can_start(est):
                        print(f'[cap] SKIP niah train {v} s{seed}')
                        continue
                t0 = time.time()
                cfgs = L.make_layer_cfgs(6, v)
                # 随机源在建模前一刻定界：同 seed 下各臂的共享组件（tok/head/MLP）必须同构
                L.set_seed(seed)
                _niah_rng_fp = L.rng_fingerprint()
                _niah_losses = []
                _niah_buf = []
                def _niah_flush():
                    if _niah_buf:
                        _niah_losses.extend(torch.stack(_niah_buf).tolist())
                        _niah_buf.clear()
                model = L.SmallGPT(vocab, 256, 6, 8, 32, 512, cfgs, mlp_ratio=mr).to(DEVICE)
                opt = torch.optim.AdamW(model.parameters(), lr=0.0003, foreach=True)
                model.train()
                cache = {'ids': None, 'tgt': None, 'age': 10 ** 9, 'epoch': 0}

                def _batch(cache=cache, _seed=seed, _vocab=vocab):
                    if cache['age'] >= 4 or cache['ids'] is None:
                        eps = _seed * 100003 + cache['epoch'] * 7919
                        ids, tgt, _, _kd = build_niah_batch(12, 512, 4, _vocab, seed=eps)
                        # staged upload — the same bit-exact transport as
                        # batch_iter's staging path (§6.1).
                        cache['ids'] = L._h2d_async(ids, torch.int64, DEVICE)
                        cache['tgt'] = L._h2d_async(tgt, torch.int64, DEVICE)
                        cache['age'] = 0
                        cache['epoch'] += 1
                    cache['age'] += 1
                    return (cache['ids'], cache['tgt'])
                _deadline = guard.deadline_ts() if guard is not None else None
                niah_truncated = False
                # same cyclic-GC handling as train_warmup; the original GC
                # state is restored on exit.
                _gc_relief = gc.isenabled()
                if _gc_relief:
                    gc.disable()
                try:
                    for step in range(n_steps):
                        if _deadline is not None and time.time() > _deadline:
                            wall = time.time() - t0
                            print(f'[cap] niah train {v} s{seed} hit the deadline after {step}/{n_steps} steps ({wall / 60:.1f} min) — the cell is abandoned WITHOUT a checkpoint, so it retrains from scratch on the next pass rather than being probed from a half-trained model.')
                            niah_truncated = True
                            break
                        x, y = _batch()
                        # Chunked LM head + cross-entropy (the inherited
                        # ce_loss path).  The trunk still sees the full T-token
                        # input (block geometry is T-dependent); ce_loss scores
                        # the first T-1 hidden rows against y and averages over
                        # the valid NIAH targets only (targets use -100).
                        ce = model.ce_loss(x, y, vocab, ignore_index=-100)
                        loss = ce + 0.05 * model.comp_reg
                        # Keep the detached scalar on the device and move a
                        # whole block at the cadence the other loops use.
                        _niah_buf.append(ce.detach())
                        if len(_niah_buf) >= 512:
                            _niah_flush()
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                        if (step + 1) % 1024 == 0:
                            gc.collect()
                        if step % 500 == 0 or step == n_steps - 1:
                            print(f'  [niah-train] {v:18s} s{seed} step {step:5d} loss {loss.item():.4f}', flush=True)
                finally:
                    if _gc_relief:
                        gc.enable()
                _niah_flush()
                if niah_truncated:
                    del model, opt
                    gc.collect()
                    torch.cuda.empty_cache()
                    if guard is not None:
                        guard.record_run(time.time() - t0, step, 256, 6, 512, 12)
                    continue
                _n_params = L.count_params(model)
                _atomic_torch_save({'cfg': cfgs, 'mlp_ratio': mr, 'vocab': vocab, 'd': 256, 'n_layers': 6, 'n_heads': 8, 'd_head': 32, 'code': CKPT_CODE, 'recipe': _recipe, 'sd': model.state_dict(), 'params': _n_params, 'rng_fp': _niah_rng_fp, 'losses': _niah_losses, 'steps': n_steps}, ck)
                if guard is not None:
                    guard.record_run(time.time() - t0, n_steps, 256, 6, 512, 12)
                L.log_experiment_event(outdir, {'kind': 'cell', 'tracking_schema': 2, 'status': 'ok', 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(t0)), 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'elapsed_s': time.time() - t0, 'key': f'{v}::seed{seed}::niah_train', 'variant': v, 'seed': int(seed), 'run_cfg': _recipe, 'code': CKPT_CODE, 'rng_fp': _niah_rng_fp, 'hyperparams': {'steps': int(n_steps), 'seq_len': 512, 'batch_size': 12, 'lr': 0.0003, 'weight_decay': 0.0, 'comp_lambda': 0.05, 'mlp_ratio': float(mr)}, 'metrics': {'params': _n_params, 'steps': int(n_steps), 'final_loss': _niah_losses[-1] if _niah_losses else None, 'train_time_s': time.time() - t0}, 'curve_summary': {'n_points': len(_niah_losses), 'first': _niah_losses[0] if _niah_losses else None, 'last': _niah_losses[-1] if _niah_losses else None, 'best': min(_niah_losses) if _niah_losses else None}, 'losses': _niah_losses})
                del model, opt
                gc.collect()
                torch.cuda.empty_cache()
            try:
                del _batch, cache
            except UnboundLocalError:
                pass
            _ckp = _ckp_pre if _ckp_pre is not None else _load_ckpt(ck)
            model = L.SmallGPT(_ckp.get('vocab', vocab), 256, 6, 8, 32, 512, _ckp['cfg'], mlp_ratio=_ckp['mlp_ratio']).to(DEVICE)
            model.load_state_dict(_ckp['sd'])
            for Ln in seq_lens:
                key = f'{v}::seed{seed}::len{Ln}'
                if L.result_is_current(summary.get(key), CKPT_CODE, 'acc'):
                    continue
                if not cuda_healthy():
                    print(f'  [niah] CUDA context poisoned; skipping len={Ln}')
                    summary[key] = {'variant': v, 'seed': seed, 'seq_len': Ln, 'error': 'cuda_context_poisoned_before_eval'}

                    def _put_niah_err(s, _k=key, _r=summary[key]):
                        s[_k] = _r
                        return s
                    L.locked_json_update(spath, _put_niah_err)
                    # the failure is a measurement outcome too — it must reach
                    # the ledger or the panel counts a cell that summary.json
                    # shows as failed but experiments.jsonl never saw at all
                    L.log_experiment_event(outdir, {'kind': 'eval', 'tracking_schema': 2, 'status': 'error', 'key': key, 'variant': v, 'seed': int(seed), 'seq_len': int(Ln), 'code': CKPT_CODE, 'error': 'cuda_context_poisoned_before_eval'})
                    break
                t0 = time.time()
                try:
                    r = eval_niah(model, Ln, DEVICE, vocab=int(_ckp.get('vocab', vocab)))
                except Exception as e:
                    r = {'error': f'{type(e).__name__}: {e}'}
                if guard is not None:
                    guard.record_run(time.time() - t0, 0, 0, 0, 0, 0)
                r.update({'variant': v, 'seed': seed, 'seq_len': Ln, 'params': _ckp['params'], 'probe_s': time.time() - t0, '_code': CKPT_CODE})
                summary[key] = r
                print(f'  [niah] {v:18s} s{seed} len={Ln:5d}  acc={r.get('acc')}  by_dist={r.get('by_dist')}')

                def _put_niah(s, _k=key, _r=r):
                    s[_k] = _r
                    return s
                L.locked_json_update(spath, _put_niah)
                # Every eval record that lands in summary.json must also reach
                # the ledger: without this entry a NIAH accuracy would be a
                # summary.json number with no trace of which code/weights
                # produced it.
                L.log_experiment_event(outdir, {'kind': 'eval', 'tracking_schema': 2, 'status': 'ok', 'key': key, 'variant': v, 'seed': int(seed), 'seq_len': int(Ln), 'code': CKPT_CODE, 'recipe': _ckp.get('recipe'), 'metrics': {'acc': r.get('acc'), 'n': r.get('n'), 'eval_span': r.get('eval_span'), 'params': r.get('params'), 'probe_s': r.get('probe_s')}})
            del model, _ckp
            gc.collect()
            torch.cuda.empty_cache()
    _measured = len({(r.get('variant'), r.get('seed')) for r in summary.values()
                     if isinstance(r, dict) and isinstance(r.get('acc'), (int, float))
                     and not isinstance(r.get('acc'), bool) and math.isfinite(float(r['acc']))})
    L.log_experiment_event(outdir, {
        'kind': 'panel_finish', 'tracking_schema': 2, 'label': label or outdir,
        'phase': 'niah', 'started_at': _panel_started,
        'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'n_cells_planned': len(variants) * len(seeds),
        'n_cells_measured': _measured,
        'n_cells_missing': max(0, len(variants) * len(seeds) - _measured),
        'summary_ref': os.path.basename(spath), 'code': CKPT_CODE,
        'provenance': _panel_prov,
    })
    _niah_plot(summary, outdir, seq_lens)
    return summary

@torch.inference_mode()
def eval_length_gen(model, val_ids, eval_lens, device=DEVICE, n_seq=16, max_pos=None):
    was = model.training
    model.eval()
    out = {}
    _within = [int(Ln) for Ln in eval_lens if max_pos is None or int(Ln) <= max_pos]
    _trunc_lens = [int(Ln) for Ln in eval_lens if max_pos is not None and int(Ln) > max_pos]
    # The compression regulariser is a training side channel: disabling it for
    # scoring skips the gate-honour bookkeeping without touching any logit
    # (the block geometry is still built), exactly as eval_ppl does.
    _reg_layers = [blk.attn for blk in model.blocks if hasattr(blk.attn, 'need_reg')]
    _reg_prior = [_a.need_reg for _a in _reg_layers]
    for _a in _reg_layers:
        _a.need_reg = False
    try:
        # LM-head logits are computed a time slice at a time: each output row
        # is an independent function of its hidden row, so per-token NLLs are
        # bit-identical to the full-table form.
        def _head_ce_none(_h, _tgt, _rows, _span):
            _nll = torch.empty((_rows, _span), dtype=torch.float32, device=device)
            _budget = 128 << 20
            _hchunk = max(1, min(_span, _budget // max(_rows * int(model.head.out_features) * 4, 1)))
            _hchunk = min(_hchunk, 256)
            for _s in range(0, _span, _hchunk):
                _e = min(_s + _hchunk, _span)
                _lg = model.head(_h[:, _s:_e, :])
                _ce = F.cross_entropy(_lg.transpose(1, 2), _tgt[:, _s:_e], reduction='none')
                _nll[:, _s:_e] = _ce
                del _lg, _ce
            return _nll
        if _within:
            _Lmax = max(_within)
            ids = np.asarray(val_ids[:n_seq, :_Lmax], dtype=np.int64)
            # same bit-exact staged upload as batch_iter (design notes §6.1)
            x = L._h2d_async(ids, torch.int64, device)
            nll_cols = torch.zeros((int(ids.shape[0]), _Lmax - 1), dtype=torch.float64, device=device)
            _r, _chunk = (0, int(ids.shape[0]))
            while _r < int(ids.shape[0]):
                _sub = x[_r:_r + _chunk]
                try:
                    _h = model.trunk(_sub)
                except RuntimeError as _oe:
                    if 'out of memory' in str(_oe).lower() and _chunk > 1:
                        _chunk = max(1, _chunk // 2)
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue
                    raise
                nll_cols[_r:_r + _chunk] = _head_ce_none(_h, _sub[:, 1:], int(_sub.shape[0]), _Lmax - 1).double()
                del _h
                _r += _chunk
            _cum = nll_cols.cumsum(dim=1)
            for Ln in _within:
                _t = float(_cum[:, Ln - 2].sum()) if Ln >= 2 else 0.0
                _ntok = int(ids.shape[0]) * (Ln - 1)
                ppl = math.exp(_t / max(_ntok, 1))
                out[Ln] = {'ppl': float(ppl), 'n_tok': _ntok, 'n_tokens': _ntok, 'truncated': False, 'eval_span': Ln,
                           'nll_sum': float(_t),
                           'nll_per_seq': [float(v) for v in _cum[:, Ln - 2].tolist()],
                           'n_seq': int(ids.shape[0])}
                # per-document rows (review-2 issue 9b): the pooled ppl above is
                # UNCHANGED (same reduction); these extra rows only make the
                # document-sampling uncertainty visible and stratifiable
                # (doc-level deltas vs seed-level variance).  Rows are the
                # panel's fixed eval rows, so the list order is stable.
                if Ln >= 2:
                    out[Ln]['ppl_per_seq'] = [float(math.exp(float(v) / (Ln - 1))) for v in _cum[:, Ln - 2].tolist()]
                    out[Ln]['n_seq'] = int(ids.shape[0])
            del x, nll_cols, _cum
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        _trunc_cache = {}
        _val_width = int(np.asarray(val_ids).shape[1])
        for Ln in _trunc_lens:
            Ln = int(Ln)
            trunc = max_pos is not None and Ln > max_pos
            # Identify the actual token slice, not just its length: for
            # Ln > max_pos the evaluation uses the LAST max_pos tokens of the
            # Ln-prefix, so two lengths can share a span but different rows.
            # Only genuinely identical (start, end) slices are reused.
            _end = min(Ln, _val_width)
            _start = max(0, _end - int(max_pos)) if trunc else 0
            _key = (_start, _end)
            _cached = _trunc_cache.get(_key)
            if _cached is not None:
                if 'error' in _cached:
                    out[Ln] = {'error': _cached['error']}
                else:
                    out[Ln] = {'ppl': _cached['ppl'], 'n_tok': _cached['n_tok'], 'truncated': trunc, 'eval_span': _cached['eval_span']}
                continue
            ids = np.asarray(val_ids[:n_seq, :Ln], dtype=np.int64)
            if max_pos is not None and Ln > max_pos:
                ids = ids[:, -max_pos:]
            try:
                x = L._h2d_async(ids, torch.int64, device)
                nll, ntok = (torch.zeros((), dtype=torch.float64, device=device), 0)
                _r, _chunk = (0, int(ids.shape[0]))
                while _r < int(ids.shape[0]):
                    _sub = x[_r:_r + _chunk]
                    try:
                        _h = model.trunk(_sub)
                    except RuntimeError as _oe:
                        if 'out of memory' in str(_oe).lower() and _chunk > 1:
                            _chunk = max(1, _chunk // 2)
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()
                            continue
                        raise
                    # Keep the per-token losses in a small float32 table and
                    # sum once, so the reduction order matches the
                    # full-logits cross_entropy(reduction='sum').
                    _nll_rows = _head_ce_none(_h, _sub[:, 1:], int(_sub.shape[0]), int(_sub.shape[1]) - 1)
                    nll += _nll_rows.sum().double()
                    ntok += int(_sub[:, 1:].numel())
                    del _h, _nll_rows
                    _r += _chunk
                ppl = math.exp(float(nll) / max(ntok, 1))
                _ev_span = int(x.shape[1])
                out[Ln] = {'ppl': float(ppl), 'n_tok': ntok, 'truncated': trunc, 'eval_span': _ev_span}
                _trunc_cache[_key] = {'ppl': float(ppl), 'n_tok': ntok, 'eval_span': _ev_span}
                del x
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            except Exception as e:
                _err = f'{type(e).__name__}: {e}'
                out[Ln] = {'error': _err}
                _trunc_cache[_key] = {'error': _err}
    finally:
        for _a, _prev in zip(_reg_layers, _reg_prior):
            _a.need_reg = _prev
        model.train(was)
    return out

def run_lenphase_cell(payload, v, seed, guard=None, prebook=False, val_ids=None):
    """Run exactly ONE (variant, seed) cell of a length-extrapolation phase:
    optional (re)train -> checkpoint -> by-length eval -> locked summary merge.

    Shared by the serial loop (in-process) and by par.py children.  Under
    parallel dispatch the projected train time is pre-booked into the shared
    TimeGuard and refunded after the actual seconds are recorded."""
    outdir = payload['outdir']
    ckpt_dir = os.path.join(outdir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    eval_lens = payload['eval_lens']
    n_seq = int(payload.get('n_seq', 16))
    steps = payload.get('steps', 3000)
    train_len = payload.get('train_len', 512)
    vocab = payload.get('vocab', 8192)
    batch_size = payload.get('batch_size', 12)
    spath = os.path.join(outdir, 'summary.json')
    out = {'status': 'ok', 'key': f'{v}::seed{seed}', 'variant': v, 'seed': seed}
    _lp_run_id = L.new_experiment_id('p1l')
    _lp_started = time.strftime('%Y-%m-%dT%H:%M:%S')
    _lp_t0 = time.time()
    _prov_cfg = {'dataset': 'wikitext', 'n_train_tokens': 8000000, 'seq_len': train_len}
    _tr = None
    # Seed before serving any data: a cache built in this child process must
    # not draw from the OS entropy pool.  Training re-seeds in train_variant.
    L.set_seed(seed)
    need = max(eval_lens) + 1
    if val_ids is None:
        val_ids = L.load_wikitext(max(train_len, need), 4000000)[1]
    mr = 4.0
    if v in PARAM_MATCHED_V7:
        mr = L.variant_mlp_ratio(v, vocab, d=256, n_layers=6, n_heads=8, d_head=32, seq_len=train_len, matched=set(PARAM_MATCHED_V7), ref_variant='csa_fixed_rope')
    _recipe = f'p1l_steps{steps}_tl{train_len}_v{vocab}_bs{batch_size}_lr0.0003_nt8000000_wu50_wd0.1_cl0.05_dlm10.0_mr{mr:.8g}-csa_fixed_rope_ns{n_seq}_df{L._ids_fp(np.ascontiguousarray(val_ids).reshape(-1))}'
    ck = os.path.join(ckpt_dir, f'{v}_seed{seed}.pt')
    stale = False
    _ckp_pre = None
    if os.path.exists(ck):
        try:
            _meta = _load_ckpt(ck)
        except Exception as _cke:
            print(f'[p1l] {v} s{seed}: checkpoint unreadable ({type(_cke).__name__}: {_cke}) — RETRAINING')
            stale = True
        else:
            stale = _meta.get('code') != CKPT_CODE
            if stale:
                print(f'[p1l] {v} s{seed}: checkpoint predates CKPT_CODE={CKPT_CODE} (code={_meta.get('code')!r}) — RETRAINING')
            elif _meta.get('recipe') != _recipe:
                stale = True
                print(f'[p1l] {v} s{seed}: checkpoint recipe mismatch (stored {_meta.get('recipe')!r}) — RETRAINING')
            else:
                _ckp_pre = _meta
            del _meta
    if stale or not os.path.exists(ck):
        key0 = f'{v}::seed{seed}'

        def _drop_eval(summary, _key0=key0):
            if summary.get(_key0) is not None:
                print(f'[p1l] {v} s{seed}: weights are being retrained — dropping the cached eval, which belongs to the previous weights')
                summary.pop(_key0, None)
            return summary
        L.locked_json_update(spath, _drop_eval)
        booked = 0.0
        _book_tok = f'lenphase|{v}::seed{seed}|pid{os.getpid()}'
        if guard is not None:
            est = guard.estimate_seconds(steps, d=256, n_layers=6, seq_len=train_len, batch_size=batch_size)
            if prebook and est:
                guard.book(est, token=_book_tok)
                booked = est
            if not guard.can_start(0 if booked else est):
                if booked:
                    guard.book(-booked, token=_book_tok)
                print(f'[cap] SKIP lenphase train {v} s{seed}')
                out['status'] = 'skipped_cap'
                return out
        t0 = time.time()
        _deadline = guard.deadline_ts() if guard is not None else None
        if _deadline is not None and booked:
            # subtract this cell's own pre-booking from the elapsed side (see
            # exp_lib._run_cell): the reservation is future time for this
            # cell, not time already spent.
            _deadline += booked
        L.set_seed(seed)
        cfgs = L.make_layer_cfgs(6, v)
        train_ids = L.load_wikitext(train_len, 8000000)[0]
        try:
            # The parent ran only the SYNTHETIC pre-flight.  The real token
            # stream can still be broken (vocabulary mapping, cache mixing,
            # label shift downstream), so run the same small-batch overfit
            # probe on the actual corpus before starting the cell.  The probe
            # is RNG-preserving, so it cannot perturb this cell.
            L.overfit_guard([v], train_ids=train_ids, vocab=vocab, steps=150,
                            seq_len=min(64, int(train_len)), batch_size=2)
            _tr = L.train_variant(v, train_ids, None, vocab, seed=seed, d=256, n_layers=6, n_heads=8, d_head=32, seq_len=train_len, batch_size=batch_size, steps=steps, lr=0.0003, weight_decay=0.1, warmup=50, comp_lambda=0.05, delta_lr_mult=10.0, eval_every=0, mlp_ratio=mr, device=DEVICE, log_every=500, deadline_ts=_deadline, return_model=True)
        except Exception:
            # A raised train_variant leaves no record_run behind, so the
            # pre-booked estimate would stay reserved forever (serial mode:
            # this process is alive, reconcile cannot refund our pid).
            if guard is not None:
                guard.record_run(time.time() - t0, 0, 256, 6, train_len, batch_size)
                if booked:
                    guard.book(-booked, token=_book_tok)
            raise
        if _tr.get('cap_truncated'):
            print(f'  [p1l-train] {v:18s} s{seed} TRUNCATED by the cap — not saved, will retry')
            _sd_tr = _tr.get('steps_done')
            if guard is not None:
                guard.record_run(time.time() - t0, _sd_tr if _sd_tr else 0, 256, 6, train_len, batch_size)
                if booked:
                    guard.book(-booked, token=_book_tok)
            out['status'] = 'truncated'
            out['truncated'] = True
            del _tr
            gc.collect()
            torch.cuda.empty_cache()
            return out
        model = _tr.pop('_model')
        _atomic_torch_save({'cfg': cfgs, 'mlp_ratio': mr, 'vocab': vocab, 'd': 256, 'n_layers': 6, 'n_heads': 8, 'd_head': 32, 'train_len': train_len, 'max_seq': train_len, 'code': CKPT_CODE, 'recipe': _recipe, 'sd': model.state_dict(), 'final_ppl': _tr.get('ppl'), 'params': L.count_params(model), 'rng_fp': _tr.get('rng_fp'), 'losses': _tr.get('losses'), 'ppl_history': _tr.get('ppl_history'), 'steps': steps}, ck)
        if guard is not None:
            guard.record_run(time.time() - t0, steps, 256, 6, train_len, batch_size, calib_seconds=_tr.get('train_time_s'))
            if booked:
                guard.book(-booked, token=_book_tok)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    key = f'{v}::seed{seed}'
    _cur = L.locked_json_read(spath).get(key)
    _need_lens = list(eval_lens)
    if L.result_is_current(_cur, CKPT_CODE, 'by_len') and _cur.get('recipe') == _recipe:
        _need_lens = [Ln for Ln in eval_lens if not L.by_len_cells([_cur], Ln)]
        if not _need_lens:
            return out
        print(f'[resume] {key}: by_len record is stamped current but at least one eval length holds no usable measurement (error cell) — re-evaluating the missing lengths')
    elif L.result_is_current(_cur, CKPT_CODE, 'by_len'):
        # A PPL measured on a different document count / recipe is a different
        # quantity.  Re-evaluate all lengths instead of merging 8-doc and
        # 16-doc cells under one key.
        print(f'[resume] {key}: by_len record was measured under a DIFFERENT recipe ({_cur.get("recipe")!r} != {_recipe!r}) — re-evaluating all lengths')
    _ckp = _ckp_pre if _ckp_pre is not None else _load_ckpt(ck)
    model = L.SmallGPT(_ckp.get('vocab', vocab), 256, 6, 8, 32, _ckp.get('train_len', train_len), _ckp['cfg'], mlp_ratio=_ckp['mlp_ratio']).to(DEVICE)
    model.load_state_dict(_ckp['sd'])
    mp = None if not getattr(model, 'use_abs_pe', True) else _ckp.get('max_seq', train_len)
    _t_ev = time.time()
    r = eval_length_gen(model, val_ids, _need_lens, DEVICE, max_pos=mp, n_seq=n_seq)
    _ckp_params = _ckp['params']
    if guard is not None:
        guard.record_run(time.time() - _t_ev, 0, 0, 0, 0, 0)

    def _merge(summary):
        _merged = {}
        _cur_now = summary.get(key)
        for _bk, _bc in (((_cur_now or {}).get('by_len') or {}).items() if _need_lens != list(eval_lens) else []):
            try:
                _merged[int(_bk)] = _bc
            except (TypeError, ValueError):
                continue
        _merged.update(r)
        _rec = {'variant': v, 'seed': seed, 'params': _ckp_params, 'max_pos': mp, 'by_len': _merged, '_code': CKPT_CODE, 'recipe': _recipe, 'eval_n_seq': n_seq}
        try:
            _prov = L.experiment_provenance(_prov_cfg, train_ids=(train_ids if _tr is not None else None), val_batch=val_ids, vocab=vocab)
            _rec['provenance'] = _prov
            _rec['hyperparams'] = L.run_hyperparameters({'steps': steps, 'n_train_tokens': _prov_cfg['n_train_tokens'], 'seq_len': train_len, 'batch_size': batch_size, 'lr': 0.0003, 'weight_decay': 0.1, 'warmup': 50, 'comp_lambda': 0.05, 'delta_lr_mult': 10.0}, seed=seed, d=256, n_layers=6, n_heads=8, d_head=32, vocab=vocab, mlp_ratio=mr)
            _rec['data_fp'] = _prov['data'].get('train_fp') or _prov['data'].get('val_fp')
            _rec['val_fp'] = _prov['data'].get('val_fp')
            if _tr is not None:
                _rec['rng_fp'] = _tr.get('rng_fp')
                _rec['losses'] = _tr.get('losses')
                _rec['ppl_history'] = _tr.get('ppl_history')
                _rec['train_knobs'] = _tr.get('train_knobs')
        except Exception as _pe:
            print(f'[p1l] WARNING: could not attach provenance to {key} ({type(_pe).__name__}: {_pe})')
        summary[key] = _rec
        print(f'  [p1l] {v:18s} s{seed} ' + '  '.join((f'L{k}={vv.get('ppl', float('nan')):.2f}' for k, vv in sorted(_merged.items()))))
        return summary
    L.locked_json_update(spath, _merge)
    _ev_rec = L.locked_json_read(spath).get(key) or {}
    L.log_experiment_event(outdir, {'kind': 'cell', 'tracking_schema': 2, 'run_id': _lp_run_id, 'status': 'ok', 'started_at': _lp_started, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'elapsed_s': time.time() - _lp_t0, 'key': key, 'variant': v, 'seed': int(seed), 'run_cfg': _recipe, 'code': CKPT_CODE, 'provenance': _ev_rec.get('provenance'), 'hyperparams': _ev_rec.get('hyperparams'), 'data_fp': _ev_rec.get('data_fp'), 'val_fp': _ev_rec.get('val_fp'), 'rng_fp': _ev_rec.get('rng_fp'), 'metrics': {'params': _ev_rec.get('params'), 'by_len': {str(k): (v2 or {}).get('ppl') for k, v2 in (_ev_rec.get('by_len') or {}).items()}, 'train_time_s': (_tr or {}).get('train_time_s')}, 'losses': _ev_rec.get('losses'), 'ppl_history': _ev_rec.get('ppl_history')})
    del model, _ckp
    gc.collect()
    torch.cuda.empty_cache()
    return out


def run_lenphase_cell_job(job):
    """Child-process entry for one lenphase cell (invoked by par.py)."""
    guard = L.TimeGuard(job['guard_cap']) if job.get('guard_cap') else None
    return run_lenphase_cell(job['payload'], job['v'], job['seed'], guard=guard, prebook=True)


def run_lenphase(payload, guard=None, label='', par=None):
    outdir = payload['outdir']
    os.makedirs(outdir, exist_ok=True)
    ckpt_dir = os.path.join(outdir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    variants = payload['variants']
    L.overfit_guard(list(variants))
    eval_lens = payload['eval_lens']
    seeds = list(payload.get('seeds', [0, 1, 2]))
    steps = payload.get('steps', 3000)
    train_len = payload.get('train_len', 512)
    vocab = payload.get('vocab', 8192)
    spath = os.path.join(outdir, 'summary.json')
    _read = L.locked_json_read(spath)
    need = max(eval_lens) + 1
    _t_data = time.time()
    L.set_seed(int((seeds or [0])[0]))
    _, val_ids, _, _, _ = L.load_wikitext(max(train_len, need), 4000000)
    if guard is not None:
        guard.record_run(time.time() - _t_data, 0, 0, 0, 0, 0)
    print(f'[p1l] val slice {val_ids.shape}, eval_lens={eval_lens}, seeds={seeds}')
    _panel_started = time.strftime('%Y-%m-%dT%H:%M:%S')
    _panel_prov = L.experiment_provenance({'dataset': 'wikitext', 'n_train_tokens': 8000000, 'seq_len': train_len}, val_batch=val_ids, vocab=vocab)
    L.log_experiment_event(outdir, {
        'kind': 'panel_start', 'tracking_schema': 2, 'label': label or outdir,
        'phase': 'lenphase', 'started_at': _panel_started,
        'config': L._jsonable(payload), 'seeds': seeds, 'variants': list(variants),
        'eval_lens': list(eval_lens), 'code': CKPT_CODE,
        'provenance': _panel_prov,
    })
    _ran_parallel = False
    cells = [(v, seed) for v in variants for seed in seeds]
    if len(cells) > 1:
        import par as _par
        _par_ctx = par if isinstance(par, dict) else _par.resolve_ctx(par, label=label or outdir)
        if _par_ctx and _par.ctx_active(_par_ctx, len(cells)):
            module = _par_ctx.get('module') or (_caller if (_caller := sys._getframe(1).f_globals.get('__name__')) and _caller != '__main__' else 'v7_supp')
            L.load_wikitext(train_len, 8000000)
            jobs = [dict(kind='lenphase', module=module, label=label, payload=payload, v=v, seed=seed, guard_cap=(guard.cap_spec() if guard is not None else None)) for v, seed in cells]
            _par.run_jobs(jobs, _par_ctx, phase_label=label or outdir)
            _ran_parallel = True
    if not _ran_parallel:
        for v, seed in cells:
            run_lenphase_cell(payload, v, seed, guard=guard, val_ids=val_ids)
    summary = L.locked_json_read(spath)
    _measured = 0
    for _r in summary.values():
        if not isinstance(_r, dict) or not isinstance(_r.get('by_len'), dict):
            continue
        if any(isinstance(_c, dict) and L.ppl_is_usable(_c.get('ppl')) for _c in _r['by_len'].values()):
            _measured += 1
    L.log_experiment_event(outdir, {
        'kind': 'panel_finish', 'tracking_schema': 2, 'label': label or outdir,
        'phase': 'lenphase', 'started_at': _panel_started,
        'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'n_cells_planned': len(cells), 'n_cells_measured': _measured,
        'n_cells_missing': max(0, len(cells) - _measured),
        'summary_ref': os.path.basename(spath),
        'provenance': _panel_prov, 'code': CKPT_CODE,
    })
    _lenplot(summary, eval_lens, outdir)
    return summary

def _lenplot(summary, eval_lens, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    vs = sorted({r['variant'] for r in summary.values() if 'by_len' in r})
    for v in vs:
        per, trunc_flags = ([], [])
        for Ln in eval_lens:
            recs = L.by_len_cells(list(summary.values()), Ln, variant=v)
            ok = [x for x in recs if not x.get('truncated')]
            use, is_tr = (ok, False) if ok else (recs, bool(recs))
            vals = [x['ppl'] for x in use]
            per.append(float(np.mean(vals)) if vals else np.nan)
            trunc_flags.append(is_tr)
        full = [np.nan if t else p for p, t in zip(per, trunc_flags)]
        trunc = [np.nan if not t else p for p, t in zip(per, trunc_flags)]
        ax[0].plot(eval_lens, full, 'o-', label=v)
        ax[0].plot(eval_lens, trunc, 'o--', mfc='none', alpha=0.75)
        base = per[0] if per and (not np.isnan(per[0])) else np.nan
        base_trunc = bool(trunc_flags[0]) if trunc_flags else False

        def _rel(p, t, base=base):
            if not base or np.isnan(p):
                return np.nan
            return p / base
        rel_full = [np.nan if t else _rel(p, t) for p, t in zip(per, trunc_flags)]
        rel_trunc = [_rel(p, t) if t else np.nan for p, t in zip(per, trunc_flags)]
        lab = f'{v} (ratio base is TRUNCATED — not a true L{int(eval_lens[0])} score)' if base_trunc else v
        ax[1].plot(eval_lens, rel_full, 'o-', label=lab)
        ax[1].plot(eval_lens, rel_trunc, 'o--', mfc='none', alpha=0.75)
    ax[0].text(0.02, 0.02, 'dashed/hollow = position-limited (abs-PE): evaluated on a\nSHORTER span than the x-axis length (not a long-context score)', transform=ax[0].transAxes, fontsize=7, va='bottom')
    for a, ttl in ((ax[0], 'wikitext PPL vs eval length (train 512)'), (ax[1], 'PPL relative to eval@512')):
        a.set_xscale('log', base=2)
        a.set_xticks(eval_lens)
        a.set_xticklabels([str(s) for s in eval_lens])
        a.set_xlabel('evaluation sequence length')
        a.set_title(ttl)
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    ax[0].set_ylabel('perplexity')
    ax[1].set_ylabel('ratio vs eval@512')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'length_gen.png'), dpi=120)
    plt.close(fig)
    print(f'[p1l] wrote {outdir}/length_gen.png')

def _niah_plot(summary, outdir, seq_lens):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    vs = sorted({r['variant'] for r in summary.values() if 'acc' in r})
    if not vs:
        return
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for v in vs:
        ys, xs = ([], [])
        for Ln in seq_lens:
            a = [r['acc'] for r in summary.values() if r.get('variant') == v and r.get('seq_len') == Ln and ('acc' in r)]
            if a:
                xs.append(Ln)
                ys.append(float(np.mean(a)))
        if xs:
            ax[0].plot(xs, ys, 'o-', label=v)
    ax[0].set_xscale('log', base=2)
    ax[0].set_xticks(seq_lens)
    ax[0].set_xticklabels([str(s) for s in seq_lens])
    ax[0].set_xlabel('evaluation sequence length')
    ax[0].set_ylabel('NIAH accuracy')
    ax[0].set_title('retrieval accuracy vs context length (train seq 512)')
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    Ln = max(seq_lens)

    def _bucket_key(k):
        _lo = re.match('\\s*(-?\\d+)', str(k))
        _hi = re.match('\\s*-?\\d+\\s*-\\s*(-?\\d+|inf)', str(k))
        return (int(_lo.group(1)) if _lo else 0, 1e+18 if _hi and (not _hi.group(1).isdigit()) else int(_hi.group(1)) if _hi else 0)
    buckets = sorted({k for r in summary.values() if r.get('seq_len') == Ln for k in r.get('by_dist', {})}, key=_bucket_key)
    for v in vs:
        rs = [r for r in summary.values() if r.get('variant') == v and r.get('seq_len') == Ln and r.get('by_dist')]
        if not rs:
            continue
        ys = []
        for b in buckets:
            vals = [r['by_dist'][b]['acc'] for r in rs if b in r['by_dist']]
            ys.append(float(np.mean(vals)) if vals else np.nan)
        ax[1].plot(range(len(buckets)), ys, 'o-', label=v)
    ax[1].set_xticks(range(len(buckets)))
    ax[1].set_xticklabels(buckets, rotation=20, fontsize=8)
    ax[1].set_ylim(-0.02, 1.02)
    ax[1].set_xlabel('needle distance bucket (tokens)')
    ax[1].set_ylabel('accuracy')
    ax[1].set_title(f'accuracy by distance at seq {Ln}')
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'niah.png'), dpi=120)
    plt.close(fig)
    print(f'[niah] wrote {outdir}/niah.png')

def cuda_healthy():
    if DEVICE.type != 'cuda':
        return True
    try:
        a = torch.zeros(8, device=DEVICE)
        b = (a + 1).sum().item()
        return b == 8.0
    except Exception:
        return False

def run_full():
    guard = L.TimeGuard(RUN_CAP_V7)
    guard.report()
    print(f'[v7] headroom {max(guard.remaining_hours(), 0.0):.2f} h left (booked {guard.state['booked_seconds'] / 3600:.2f} h)')
    git_push('v7: supplementary experiment library (RoPE+QK-norm, dense->sparse warmup, NIAH probe, topk sweep, analytic FLOPs)')
    import par
    _ctx = par.resolve_ctx(label='v7')
    all_ok = True
    for pname, kind, payload in PHASES:
        rem = guard.remaining_hours()
        if rem < 0.4166666666666667:
            print(f'[v7] stopping before {pname}: headroom exhausted ({rem:.2f} h)')
            break
        if not cuda_healthy():
            print(f'[v7] CUDA context poisoned before {pname} -- restart the process to continue (resume will skip completed runs).  Aborting phase loop.')
            all_ok = False
            break
        print(f'\n===== v7 phase {pname} =====')
        try:
            run_phase(pname, guard, par=_ctx)
        except Exception:
            traceback.print_exc()
            all_ok = False
        all_ok &= git_push(f'v7: phase {pname} results')
    try:
        flops_analysis()
        bootstrap_report(['results_lm_v7_warmup', 'results_lm_v7_rope', 'results_lm_v7_long40', 'results_lm_v7_seq2k', 'results_lm_v3_1500'])
    except Exception:
        traceback.print_exc()
        all_ok = False
    all_ok &= git_push('v7: analytic FLOPs/KV-cache + paired sign-flip stats')
    guard.report()
    print('\n[v7] ALL PHASES DONE.')
    schedule_shutdown(120 if all_ok else 2400)

def run_smoke():
    print('[smoke] 1) RoPE variant forward/backward')
    L.set_seed(0)
    for v in ['full_rope', 'csa_fixed_rope', 'hybrid_fixed_rope']:
        cfgs = L.make_layer_cfgs(2, v)
        m = L.SmallGPT(8192, 256, 2, 8, 32, 128, cfgs).to(DEVICE)
        x = torch.randint(0, 8192, (2, 128), device=DEVICE)
        out = m(x)
        loss = F.cross_entropy(out.reshape(-1, 8192), x.reshape(-1)) + 0.05 * m.comp_reg
        loss.backward()
        print(f'  {v:20s} out={tuple(out.shape)} loss={loss.item():.3f} use_abs_pe={getattr(m, 'use_abs_pe', True)}')
    print('[smoke] 1b) dense-warmup path vs an independent windowed reference')
    for v in ('csa_fixed', 'hybrid_fixed'):
        L.set_seed(0)
        cfgs = L.make_layer_cfgs(2, v)
        m = L.SmallGPT(8192, 256, 2, 8, 32, 32, cfgs).to(DEVICE)
        a = m.blocks[0].attn
        x = torch.randn(1, 32, 256, device=DEVICE)
        a._dense_warmup, a._dense_max_T = (True, None)
        full_ref = a(x)
        a._dense_max_T = 32
        ref = a(x)
        a._dense_max_T = 16
        win = a(x)
        print(f'  {v:14s} window>=T == dense: {bool(torch.equal(full_ref, ref))}  w16 differs from dense: {not bool(torch.allclose(win, ref))}')
        check = torch.equal(full_ref, ref) and (not torch.allclose(win, ref, atol=1e-06))
        a._dense_warmup = a._dense_max_T = None
        if not check:
            raise AssertionError(f'dense-warmup truncation broken for {v}')
    print('[smoke] 2) dense-warmup toggle + 200-step transfer')
    train_ids, val_batch, vocab, _, vb = L.load_wikitext(512, 1000000)
    for warm in (0, 100):
        r = train_warmup('csa_fixed', train_ids, val_batch, vocab, seed=0, steps=200, warm_steps=warm, eval_every=100, eval_subset=16, log_every=100, val_bnd=vb)
        print(f'  warm={warm:4d} ppl={r['ppl']:.2f} switch={r['ppl_at_switch']}')
    print('[smoke] 3) NIAH probe')
    cfgs = L.make_layer_cfgs(2, 'full_rope')
    m = L.SmallGPT(vocab, 256, 2, 8, 32, 512, cfgs).to(DEVICE)
    print('  niah:', eval_niah(m, 512, DEVICE, n_seq=8))
    print('[smoke] 4) analytic FLOPs + stats')
    flops_analysis(outdir='analysis_v7_smoke')
    bootstrap_report(['results_lm_v3_1500'], out='analysis_v7_smoke/stats.json')
    print('\n[smoke] PASSED')
if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'full'
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    _MODES = ('full', 'smoke', 'analysis', 'phase')
    if mode in ('-h', '--help', 'help'):
        print(__doc__ or f"[v7] modes: {_MODES} (no argument means 'full')")
        raise SystemExit(0)
    if mode not in _MODES:
        print(f"[v7] unknown mode {mode!r}; expected one of {_MODES} (no argument means 'full').  Refusing to start a run.")
        raise SystemExit(2)
    if mode == 'phase' and len(sys.argv) < 3:
        print(f"[v7] mode 'phase' needs a phase name, e.g. `python v7_supp.py phase P0E`.  Available: {[p[0] for p in PHASES]}")
        raise SystemExit(2)
    print(f'[v7] mode={mode} repo={REPO} device={DEVICE} ({(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')})')
    try:
        if mode == 'smoke':
            run_smoke()
        elif mode == 'analysis':
            flops_analysis()
            bootstrap_report(['results_lm_v7_warmup', 'results_lm_v7_rope', 'results_lm_v7_long40', 'results_lm_v7_seq2k', 'results_lm_v3_1500'])
        elif mode == 'phase':
            run_phase(sys.argv[2], L.TimeGuard(RUN_CAP_V7))
            git_push(f'v7: phase {sys.argv[2]} results')
        else:
            run_full()
    except Exception:
        traceback.print_exc()
        try:
            git_push('v7: PARTIAL — crashed, see log (re-run resumes)')
        except Exception:
            traceback.print_exc()
        if mode == 'full':
            schedule_shutdown(2400)
        sys.exit(1)
