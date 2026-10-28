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
    # Only rotate complete pairs that actually exist in the row.  An
    # over-long rope_dim must not widen the result, and an odd rope_dim must
    # leave the unpaired middle lane untouched instead of broadcasting it into
    # a different width.
    width = int(x.shape[-1])
    rot_dim = max(0, min(int(rope_dim), width))
    half = rot_dim // 2
    if half <= 0:
        return x.clone()
    cos_h = cos[..., :half] if cos.shape[-1] > half else cos
    sin_h = sin[..., :half] if sin.shape[-1] > half else sin
    x1 = x[..., :half]
    x2 = x[..., half:2 * half]
    # Same elementwise expressions as the previous slice-write path; the
    # prefix is overwritten and every lane from 2*half onward (including the
    # unpaired odd lane and any tail beyond rope_dim) passes through.
    out = x.clone()
    out[..., :half] = x1 * cos_h - x2 * sin_h
    out[..., half:2 * half] = x1 * sin_h + x2 * cos_h
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
        _use_full_mask = L._dense_mask_fits(T)
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
        sink_view = L._safe_nan_to_num(self.sink, where='sink_param_v7').view(1, self.nh, 1, 1) if self.sink is not None else None
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
                sim = L.chunk_similarity(x, cfg, fused=fused)
                if sim.numel() == 0:
                    gate = None
                else:
                    prefix_mean = torch.cumsum(sim, 0) / L._range_cache(1, sim.numel() + 1, x.device)
                    tau = prefix_mean + self.delta_logit
                    gate = torch.sigmoid((tau - sim) / max(cfg.temperature, 0.001))
                    hard_b = gate.detach() > L._HALF
                    # numpy view — same booleans as the .tolist() form
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
                _fused = self._fuse(x.detach()) if L.chunk_similarity_source(cfg) == 'fused' else None
                sim = L.chunk_similarity(x.detach(), cfg, fused=_fused)
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
                self._cuts.append(cuts)
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
            _, topk_idx, soft = L._indexer_trainable(x, index_kv, last_tok, self.W_DQ, self.W_DK, self.W_w.weight, cfg.n_index_heads, cfg.index_topk, random_select=cfg.indexer_mode == 'random', pre_qI=pre['qI'] if pre is not None else None, pre_w=pre['w_idx'] if pre is not None else None, out_valid=_sel_valid_box, need_soft=_need_soft, random_salt=getattr(cfg, 'random_salt', 0))
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
        # The window half is gathered and rotated per query chunk below.
        # RoPE is position-wise, so rotating the gathered rows gives
        # exactly the same values as rotating the table and then gathering.
        out = torch.empty_like(q)
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
        # The K/V halves are gathered separately per chunk and concatenated
        # in the same order, so every gathered value is bit-identical.
        _has_sw = k_sw is not None and v_sw is not None
        # see exp_lib.gathered_attention: these per-row masks are hoisted under
        # L._FO_FULL_CAP_BYTES; the hoisted values are bitwise identical to the
        # per-chunk forms.  `topk_distinct=True` (the read-everything HCA arm's
        # arange expansion) makes first_occurrence_mask provably all-True — the
        # keep mask is then identical to the computed form.
        _dedup = (not topk_distinct) and topk_l.shape[1] > 1
        _full_ok = L._attn_full_width_ok(T, topk_l.shape[1], w, _dedup)
        if _full_ok:
            _wi_all, _wvalid_all = L._window_geometry(T, w, dev)
        else:
            _wi_all = _wvalid_all = None
        if _full_ok:
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
                _iw = _wi_all[s:e]
            else:
                # Row-chunked construction when the full-width mask is not
                # built.
                sel = ((pos_all[s:e, None] > last_tok[None, :]) if _arange_sel
                       else (pos_all[s:e, None] > last_tok[topk_l[s:e]]))
                _keep = sel_valid[s:e] if sel_valid is not None else None
                if ib.shape[1] > 1 and not topk_distinct:
                    _ddk = L.first_occurrence_mask(ib)
                    _keep = _ddk if _keep is None else _keep & _ddk
                if _keep is not None:
                    sel = sel & _keep
                # Build only this chunk's window rows; the full O(T*w) table
                # is not materialised when the mask hoist is skipped.
                _iw, _wvalid = L._window_geometry_rows(T, w, dev, s, e)
                valid = torch.cat([sel, _wvalid], 1)
                _nvalid = ~valid[:, None, :]
                _anyrow = valid.any(-1) if sink is None else None
            # ib selects rows of k_blk_r / v_blk and _iw selects rows of
            # k_sw_r / v_sw — exactly the rows the cat-then-gather form
            # addressed through the +n_blk offsets, so every gathered value is
            # bit-identical (index re-plumbing only, §6.1).
            _Kb = L._take_2d(k_blk_r, ib)
            if _has_sw:
                _Kw = L._take_2d(k_sw, _iw)
                # `_Kw` is [q, m, nh, hd]; the position table must gain a
                # singleton HEAD axis (`[:, :, None, :]`), not a singleton
                # COLUMN axis.  The latter only broadcasts when the number of
                # gathered lanes happens to equal the head count.
                _Kw = apply_rope(_Kw, cos[_iw][:, :, None, :], sin[_iw][:, :, None, :], rope_dim)
                Kset = torch.cat([_Kb, _Kw], 1)
                del _Kb, _Kw
            else:
                Kset = _Kb
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
            _V = (torch.cat([L._take_2d(v_blk, ib), L._take_2d(v_sw, _iw)], 1)
                  if _has_sw else L._take_2d(v_blk, ib))
            out[s:e] = torch.einsum('qhm,qmhd->qhd', attn, _V)
            del _V
        return out
    def _dense_warmup_forward(self, x):
        B, T, _ = x.shape
        nh, hd = (self.nh, self.hd)
        scale = 1.0
        use_rope = isinstance(self.cfg, AttnCfgRope) and self.cfg.rope
        rd = min(getattr(self.cfg, 'rope_dim', 0), hd)
        win = getattr(self, '_dense_max_T', None)
        _use_full_mask = L._dense_mask_fits(T)
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
                    if _gbool is not None and L._device_segment_enabled(x.device):
                        gate_sigs = [(_r[0], _g) for _r, _g in zip(_raw, _gbool.unbind(0))]
                        pre_blocks = L._device_blocks_from_gate_bool(
                            T, _gbool, self.cfg.min_block, self.cfg.max_block, x.device,
                            want_honoured=self.need_reg, hon_dtype=_raw[0][0].dtype)
                    else:
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

    def forward(self, ids, logits_tail=None, targets=None, vocab=None):
        if targets is not None:
            if vocab is None:
                raise ValueError('forward(targets=...) requires vocab')
            return self.ce_loss(ids, targets, vocab)
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
            _max_pos = int(self.pos.weight.shape[0])
            if T > _max_pos:
                raise RuntimeError(
                    f'absolute position table covers {_max_pos} tokens but the input has {T}; '
                    'use a RoPE arm or pass an explicit max_pos (recorded as truncated=True)')
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

def _indexer_salt(layer):
    # Per-(seed, layer) salt for the seeded-random control.  The legacy
    # `csa_fixed_rope_randidx` keeps salt 0 and therefore its exact
    # fixed-hash behaviour; only the new arm draws a different pattern.
    seed = int(getattr(L, '_LAST_SEED', 0) or 0)
    return seed * 1000003 + int(layer) + 1

def make_layer_cfgs_v7(n_layers, variant):
    if variant == 'csa_fixed_rope':
        return [AttnCfgRope(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32, rope=True, rope_dim=16, qk_norm=True) for _ in range(n_layers)]
    if variant == 'csa_fixed_rope_randidx':
        # 从头训练的随机选择对照：与 csa_fixed_rope 逐字段一致，
        # 仅 indexer_mode='random'。它回答的是「训练充分的随机选择」而非
        # 「测试时把已训练模型的选择器换成随机」——后者（v10 P3MP 的
        # randidx 臂）测的是依赖性，本变体测的是选择方式本身的优劣。
        return [AttnCfgRope(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32, indexer_mode='random', rope=True, rope_dim=16, qk_norm=True) for _ in range(n_layers)]
    if variant == 'csa_fixed_rope_randidx_seeded':
        # Equal-read-budget random control: same architecture/top-k/params as
        # csa_fixed_rope, but the selected blocks are a per-(seed, layer)
        # pseudo-random draw instead of one fixed hash shared by all seeds.
        return [AttnCfgRope(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32, indexer_mode='random', random_salt=_indexer_salt(i), rope=True, rope_dim=16, qk_norm=True) for i in range(n_layers)]
    if variant == 'full_rope_sink':
        # Sink-matched dense reference: the only difference from full_rope is
        # the learned per-head sink that the csa arms already have.
        return [AttnCfgRope(kind='full', rope=True, rope_dim=16, qk_norm=True, full_sink=True) for _ in range(n_layers)]
    if variant == 'csa_fixed_rope_nosink':
        # Sink-ablated csa cell of the 2x2 (csa/dense x sink/no-sink).
        return [AttnCfgRope(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32, rope=True, rope_dim=16, qk_norm=True, use_sink=False) for _ in range(n_layers)]
    if variant == 'csa_fixed_rope_allblocks':
        # 从头训练的全块读取对照：与 csa_fixed_rope 同一
        # c_kv 压缩 / qk_norm / RoPE / sink / 窗口投影，仅 index_topk 覆盖
        # 全部块（lightning_indexer 内部 k=min(topk, B)）。它同时是
        # 「逐项对齐的 dense」——与 full_rope 的差别只剩 KV 压缩投影与
        # 分块读取方式，不再是投影/归一化/编码的混合差异。
        return [AttnCfgRope(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=1000000, rope=True, rope_dim=16, qk_norm=True) for _ in range(n_layers)]
    if variant == 'csa_dynamic_rope':
        return [AttnCfgRope(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, rope=True, rope_dim=16, qk_norm=True) for _ in range(n_layers)]
    if variant == 'csa_dynamic_rope_randidx_seeded':
        # Dynamic-blocking selection control: the cut policy and the read
        # budget are identical to `csa_dynamic_rope`; only the top-k block
        # selection is a per-(seed, layer) pseudo-random draw.  Fixed-vs-
        # dynamic under BOTH random selectors therefore isolates the boundary
        # (segmentation) mechanism from learned selection.
        return [AttnCfgRope(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, indexer_mode='random', random_salt=_indexer_salt(i), rope=True, rope_dim=16, qk_norm=True) for i in range(n_layers)]
    if variant == 'csa_dynamic_rope_task':
        # Task-learned boundary policy, rope-consistent with the other dynamic
        # arms.  The non-rope `csa_dynamic_task` must never be compared with
        # rope arms: positional encoding would be a second factor.
        return [AttnCfgRope(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, boundary_policy='reinforce', rope=True, rope_dim=16, qk_norm=True) for _ in range(n_layers)]
    if variant == 'hybrid_dynamic_rope':
        # HCA reads every closed block, so this arm is an information-rich
        # system-level comparison, not an equal-read-budget control.
        out = []
        for i in range(n_layers):
            if i % 2 == 0:
                out.append(AttnCfgRope(kind='csa', dynamic=True, chunking='cosine_learnable', target_block_tokens=4, overlap=OVERLAP, index_topk=32, temperature=0.1, rope=True, rope_dim=16, qk_norm=True))
            else:
                out.append(AttnCfgRope(kind='hca', dynamic=True, chunking='cosine_learnable', target_block_tokens=64, block_size=64, temperature=0.1, comp_lambda_mult=4.0, rope=True, rope_dim=16, qk_norm=True))
        return out
    if variant.startswith('csa_fixed_rope_topk'):
        # Read-budget-matched fixed control for the dynamic study.  The
        # concrete k is frozen only after the realized mean dynamic block
        # length is measured on validation (see confirmation_frozen_dynamic).
        _k = int(variant.replace('csa_fixed_rope_topk', ''))
        if _k < 1:
            raise ValueError(f'{variant}: top-k must be >= 1, got {_k}')
        return [AttnCfgRope(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=_k, rope=True, rope_dim=16, qk_norm=True) for _ in range(n_layers)]
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


def _v7_knobs(device=None, batch_size=None):
    """v7's own loops are pure fp32 (no autocast); record that truth rather
    than the CUDA default AMP mode `training_knobs()` would infer."""
    return L.training_knobs(amp_dtype=None,
                            eager_batch=L._eager_batch_active(device, batch_size))


def _v7_knobs_fp(device=None, batch_size=None):
    return L.training_knobs_fingerprint(
        amp_dtype=None, eager_batch=L._eager_batch_active(device, batch_size))


ROPE_VARIANTS = ['full_rope', 'csa_fixed_rope', 'hybrid_fixed_rope', 'csa_dynamic_rope', 'csa_fixed']
PARAM_MATCHED_V7 = {'full_rope', 'full_sw128_matched_rope', 'full_rope_sink'}
ROPE_CONTRAST_FACTORS = {
    'CSA+RoPE vs CSA absPE': {
        'a': 'csa_fixed_rope', 'b': 'csa_fixed',
        'single_factor': True,
        'factors': ['qk_norm', 'rope', 'rope_dim'],
        'factors_note': ('one positional-encoding family: RoPE on/off, its dimension, '
                         'and the QK normalization used by the RoPE arms'),
    },
    'CSA+RoPE vs dense+RoPE': {
        'a': 'csa_fixed_rope', 'b': 'full_rope',
        'single_factor': False,
        'factors': ['kind'],
        'factors_note': 'attention-operator swap: architecture-level, not a single knob',
    },
    'hybrid+RoPE vs dense+RoPE': {
        'a': 'hybrid_fixed_rope', 'b': 'full_rope',
        'single_factor': False,
        'factors': ['block_size', 'kind'],
        'factors_note': ('attention-operator swap plus the HCA block geometry: '
                         'architecture-level, not a single knob'),
    },
    'v7 scratch: seeded-random selection vs learned': {
        'a': 'csa_fixed_rope_randidx_seeded', 'b': 'csa_fixed_rope',
        'single_factor': True,
        'factors': ['indexer_mode', 'random_salt'],
        'factors_note': 'single-factor selection control at the same top-k; the salt defines the per-seed/layer draw',
    },
    'v7 scratch: all-block reading vs learned': {
        'a': 'csa_fixed_rope_allblocks', 'b': 'csa_fixed_rope',
        'single_factor': True,
        'factors': ['index_topk'],
        'factors_note': 'single-factor read-budget intervention (top-k -> all blocks)',
    },
    'v7 scratch: no-sink vs learned': {
        'a': 'csa_fixed_rope_nosink', 'b': 'csa_fixed_rope',
        'single_factor': True,
        'factors': ['use_sink'],
        'factors_note': 'single-factor attention-sink ablation',
    },
    'v7 scratch: sink-matched dense vs dense': {
        'a': 'full_rope_sink', 'b': 'full_rope',
        'single_factor': True,
        'factors': ['full_sink'],
        'factors_note': 'single-factor dense attention-sink ablation',
    },
}

def attach_contrast_factor_metadata(comparisons, declarations, factor_report):
    _pairs = factor_report.get('pairs') if isinstance(factor_report, dict) else None
    _pairs = _pairs if isinstance(_pairs, dict) else {}
    for _name, _rec in (comparisons or {}).items():
        _decl = (declarations or {}).get(_name)
        if not isinstance(_decl, dict) or not isinstance(_rec, dict):
            continue
        _rec.update(_decl)
        _pair = _pairs.get(_name)
        _verified = isinstance(_pair, dict) and _pair.get('verified') is True
        _rec['factor_declaration_verified'] = _verified
        if not _verified and _rec.get('single_factor') is True:
            _rec['declared_single_factor'] = True
            _rec['single_factor'] = False
    return comparisons

def train_warmup(variant, train_ids, val_batch, vocab, *, seed=0, d=256, n_layers=6, n_heads=8, d_head=32, seq_len=512, batch_size=12, steps=20000, warm_steps=0, lr=0.0003, weight_decay=0.1, warmup=200, comp_lambda=0.05, delta_lr_mult=10.0, eval_every=1000, eval_subset=128, val_bnd=None, device=DEVICE, log_every=1000, mlp_ratio=4, deadline_ts=None, save_ckpt=None, ckpt_dir=None, measure_eff=True, diag_every=None):
    L.set_seed(seed)
    L.nan_guard_state(reset=True)
    _data_audit = L.audit_token_data(train_ids, val_batch, vocab)
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
    _save_mode = str(save_ckpt or '').strip().lower()
    if _save_mode not in ('', 'none', 'off', '0', 'final'):
        print(f'[ckpt] train_warmup ignores unknown save_ckpt={save_ckpt!r} (use final/none)')
        _save_mode = ''
    if _save_mode in ('none', 'off', '0'):
        _save_mode = ''
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
    # Health/NaN monitoring mirrors exp_lib.train_variant: a fixed training
    # batch (own RNG stream, never perturbing `bpe`) tracks train CE on the
    # eval grid so the longest runs in the study get a curve_health verdict
    # instead of only a final number.
    _fixed_eval = None
    if eval_every and train_ids is not None and int(eval_every) > 0:
        try:
            _fixed_eval = L._fixed_train_eval_batch(train_ids, seq_len, batch_size, device, seed)
        except Exception as _fe:
            print(f'[health] WARNING: could not build the fixed train-eval batch ({type(_fe).__name__}: {_fe}); training CE will not be tracked')
            _fixed_eval = None
    _diag_stride = max(1, int(diag_every) if diag_every else max(1, int(steps) // 1000))
    _token_check_every = max(1, int(steps) // 100)
    _nan_events = []
    _clip_steps = 0
    _skipped_steps = 0
    _nonfinite_loss_steps = 0
    _nonfinite_grad_steps = 0
    _first_bad_step = None
    _health_history = []
    _best_val_ppl = None
    _best_step = None
    _last_subset_ppl = None
    _last_train_ce = None
    _gn_hist = []
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
    _auto_ckpt, _ = L._auto_ckpt_mode(device, n_param, n_layers, d, mlp_ratio, batch_size, seq_len, cfgs, n_heads=n_heads, d_head=d_head)
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
            if step == 0 or step % _token_check_every == 0 or _first_bad_step is not None:
                try:
                    L._check_token_batch(x, y, vocab, step=step + 1)
                except (TypeError, ValueError, OverflowError) as _be:
                    raise RuntimeError(f'[data] bad training batch at step {step + 1}: {_be}') from _be
                if step == 0:
                    try:
                        _shift_ok = bool(x.shape == y.shape and x.shape[1] > 0
                                         and torch.equal(y[:, :-1], x[:, 1:]))
                    except Exception:
                        _shift_ok = False
                    _data_audit['label_shift'] = {
                        'checked': True, 'ok': _shift_ok,
                        'n_rows': int(x.shape[0]), 'n_tokens': int(x.numel())}
                    if not _shift_ok:
                        raise RuntimeError(
                            '[data] training inputs/targets are not a one-token causal shift; '
                            'investigate the loader before training further')
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
            _loss_item = float(loss.detach().item())
            _gn_pre = None
            _step_ok = True
            if not math.isfinite(_loss_item):
                # Same contract as exp_lib.train_variant: a non-finite loss
                # must never reach the parameters; the step is skipped and the
                # event is recorded for the opt_diag_summary audit trail.
                _step_ok = False
                _skipped_steps += 1
                _nonfinite_loss_steps += 1
                if _first_bad_step is None:
                    _first_bad_step = step + 1
                _nan_events.append({'step': step + 1, 'kind': 'loss', 'loss': _loss_item,
                                    'ce': float(ce.detach().item()), 'lr': base})
                print(f'  [nan] non-finite loss at step {step + 1}; skipping the optimizer update')
                opt.zero_grad(set_to_none=True)
            else:
                loss.backward()
                _gn = torch.nn.utils.clip_grad_norm_(model.parameters(), L.GRAD_CLIP_NORM)
                _gn_pre = float(_gn.detach().item()) if torch.is_tensor(_gn) else float(_gn)
                if not math.isfinite(_gn_pre):
                    _step_ok = False
                    _skipped_steps += 1
                    _nonfinite_grad_steps += 1
                    if _first_bad_step is None:
                        _first_bad_step = step + 1
                    _nan_events.append({'step': step + 1, 'kind': 'grad', 'grad_norm': _gn_pre,
                                        'loss': _loss_item, 'lr': base})
                    print(f'  [nan] non-finite gradient norm at step {step + 1}; zeroing grads and skipping the update')
                    opt.zero_grad(set_to_none=True)
                else:
                    _clip_steps += int(_gn_pre > L.GRAD_CLIP_NORM)
                    if step % _diag_stride == 0 or step == steps - 1:
                        _gn_hist.append(_gn_pre)
                    opt.step()
                    # No gradient is needed after the update.
                    opt.zero_grad(set_to_none=True)
            if _step_ok and (step == 0 or _first_bad_step is not None or (step + 1) % 1024 == 0) and not L._params_all_finite(model):
                _nan_events.append({'step': step + 1, 'kind': 'param', 'loss': _loss_item,
                                    'grad_norm': _gn_pre, 'lr': base})
                raise FloatingPointError(f'non-finite parameter values after step {step + 1}; aborting instead of training on corrupted weights')
            if _ce_d is not None:
                _ce_baseline.mul_(0.99).add_(_ce_d, alpha=0.01)
            _lbuf.append(ce.detach())
            _tail.append(ce.detach())
            loss = ce = None
            model.comp_reg = None
            if len(_lbuf) >= 512:
                losses.extend(torch.stack(_lbuf).tolist())
                _lbuf.clear()
            if (step + 1) % 1024 == 0:
                gc.collect()
            if eval_every and ((step + 1) % eval_every == 0 or step == steps - 1):
                _pv = float(L.eval_ppl(model, _eval_batch, device))
                _dense_now = getattr(model.blocks[0].attn, '_dense_warmup', False)
                (warm_hist if _dense_now else ppl_hist).append([step + 1, _pv])
                if not _dense_now:
                    # Sparse-phase points only: the dense prefix is a
                    # different protocol, so mixing it into the health
                    # classification would compare across a protocol switch.
                    _train_ce = None
                    if _fixed_eval is not None:
                        try:
                            _train_ce, _ = L._lm_batch_metrics(model, _fixed_eval[0], _fixed_eval[1], vocab)
                        except Exception as _te:
                            print(f'[health] WARNING: fixed train-CE evaluation failed ({type(_te).__name__}: {_te}); train CE stays unmonitored at this point')
                            _train_ce = None
                    # The `*_final` fields describe the last measurement,
                    # including a non-finite one; keeping the previous finite
                    # value would misreport a diverged final eval as healthy.
                    _last_train_ce = _train_ce
                    _val_ce = math.log(_pv) if L._is_finite_number(_pv) and _pv > 0 else float('nan')
                    _health_history.append({'step': step + 1, 'val_ppl': _pv, 'val_ce': _val_ce,
                                            'train_ce': _train_ce,
                                            'gap': (_val_ce - _train_ce) if (math.isfinite(_val_ce) and L._is_finite_number(_train_ce)) else None})
                    _last_subset_ppl = _pv
                    if L._is_finite_number(_pv) and _pv > 0 and (_best_val_ppl is None or _pv < _best_val_ppl):
                        _best_val_ppl = _pv
                        _best_step = step + 1
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
    _eval_stats = L.eval_ppl(model, val_batch, device, return_stats=True,
                             collect_block_stats=True, val_bnd=val_bnd)
    ppl = float(_eval_stats['ppl'])
    stats = _eval_stats.get('compression')
    if stats is None:
        stats = L.compression_report(model, val_batch, device, val_bnd=val_bnd)
    print(f'[{variant} seed={seed} warm={warm_steps}] val PPL = {ppl:.3f}  ({wall / 60:.1f} min)')
    if _lbuf:
        losses.extend(torch.stack(_lbuf).tolist())
        _lbuf.clear()
    losses = [float(v) for v in losses]
    if _health_history:
        _health = L.classify_training_health(_health_history)
    else:
        _health = {'status': 'unmonitored', 'reason': 'no sparse-phase validation eval points (eval_every=0 or all points fell in the dense prefix)',
                   'n_points': 0, 'best_step': None, 'best_val_ppl': None,
                   'final_val_ppl': None, 'final_minus_best': None}
    _grad_vals = [float(v) for v in _gn_hist if L._is_finite_number(v)]
    _opt_summary = {
        'n_steps': int(steps), 'n_diag_points': len(_gn_hist),
        'grad_norm_mean': (sum(_grad_vals) / len(_grad_vals)) if _grad_vals else None,
        'grad_norm_max': max(_grad_vals) if _grad_vals else None,
        'clip_steps': int(_clip_steps),
        'clip_fraction': (float(_clip_steps) / max(1, int(steps))),
        'skipped_updates': int(_skipped_steps),
        'nonfinite_loss_steps': int(_nonfinite_loss_steps),
        'nonfinite_grad_steps': int(_nonfinite_grad_steps),
        'first_bad_step': _first_bad_step,
        'diag_stride': int(_diag_stride),
        'nan_guard': L.nan_guard_state(),
        'events': _nan_events[:20],
    }
    if _health.get('status') != 'healthy' and _health.get('status') != 'unmonitored':
        print(f'[health] WARNING: {variant} seed={seed} curve_health={_health.get("status")} ({_health.get("reason")})')
    _instability = L.diagnose_training_instability(
        {'opt_diag_summary': _opt_summary, 'amp_dtype': None,
         'train_knobs': _v7_knobs(device, batch_size),
         'curve_health': _health, 'health_history': _health_history,
         'data_audit': _data_audit})
    _checkpoints = {}
    if _save_mode and ckpt_dir:
        try:
            os.makedirs(ckpt_dir, exist_ok=True)
            _ck = os.path.join(ckpt_dir, f'{variant}_seed{seed}_final.pt')
            _atomic_torch_save({'cfg': cfgs, 'mlp_ratio': mlp_ratio, 'vocab': vocab, 'd': d,
                                'n_layers': n_layers, 'n_heads': n_heads, 'd_head': d_head,
                                'train_len': seq_len, 'max_seq': seq_len, 'code': CKPT_CODE,
                                'variant': variant, 'seed': int(seed), 'step': int(steps),
                                'sd': model.state_dict(),
                                'val_ppl': (float(ppl) if L._is_finite_number(ppl) else None),
                                'curve_health': _health,
                                'health_history': _health_history,
                                'opt_diag_summary': _opt_summary,
                                'instability_diagnosis': _instability,
                                'best_step': _best_step, 'best_val_ppl': _best_val_ppl,
                                'train_ce_fixed_final': _last_train_ce,
                                'warm_steps': int(warm_steps), 'params': int(n_param),
                                'rng_fp': _rng_fp, 'train_fp': L._ids_fp(train_ids),
                                'val_fp': L._ids_fp(np.ascontiguousarray(np.asarray(val_batch)).reshape(-1)),
                                'losses': losses, 'steps': int(steps)}, _ck)
            _checkpoints['final'] = {'path': _ck, 'step': int(steps)}
            print(f'  [ckpt] saved final checkpoint {_ck}')
        except Exception as _se:
            print(f'[ckpt] WARNING: could not save the final checkpoint ({type(_se).__name__}: {_se}); the run continues, but this weight is unavailable for post-hoc analysis')
    _efficiency = None
    if measure_eff:
        try:
            _eff_ids = np.ascontiguousarray(np.asarray(_eval_batch)[:min(4, int(np.asarray(_eval_batch).shape[0]))]) if _eval_batch is not None else None
            if _eff_ids is not None and len(_eff_ids):
                _efficiency = L.measure_efficiency(model, _eff_ids, device, warmup=1, iters=5, decode_tokens=4)
        except Exception as _ee:
            print(f'[efficiency] WARNING: measurement failed ({type(_ee).__name__}: {_ee})')
            _efficiency = {'status': 'error', 'error': f'{type(_ee).__name__}: {_ee}'}
    res = {'variant': variant, 'seed': seed, 'ppl': ppl, 'params': n_param, 'losses': losses, 'stats': stats, 'steps': steps, 'tokens_seen': steps * batch_size * seq_len, 'train_time_s': wall, 'warm_steps': warm_steps, 'ppl_at_switch': switch_ppl, 'final_loss_smoothed': float(torch.stack(list(_tail)).mean()) if _tail else float('nan'), 'ppl_history': ppl_hist, 'ppl_history_dense_prefix': warm_hist, 'delta_trace': {}, 'train_knobs': _v7_knobs(device, batch_size), 'rng_fp': _rng_fp, 'data_fp': L._ids_fp(train_ids), 'data_audit': _data_audit,
           'nll_sum': _eval_stats.get('nll_sum'), 'n_tokens': _eval_stats.get('n_tokens'),
           'nll_per_row': _eval_stats.get('nll_per_row'), 'n_tokens_per_row': _eval_stats.get('n_tokens_per_row'),
           'eval_rows': _eval_stats.get('eval_rows'), 'eval_seed': _eval_stats.get('eval_seed'),
           'eval_subset': _eval_stats.get('eval_subset'),
           'eval_fp': L._ids_fp(np.ascontiguousarray(np.asarray(val_batch)).reshape(-1)),
           'param_breakdown': L.param_breakdown(model),
           'kv_cache_elements_per_token': L.kv_cache_elements_per_token(model),
           'train_objective': L.train_objective(model, comp_lambda=comp_lambda, delta_lr_mult=delta_lr_mult),
           'curve_health': _health, 'health_history': _health_history,
           'opt_diag_summary': _opt_summary,
           'instability_diagnosis': _instability,
           'best_step': _best_step, 'best_val_ppl': _best_val_ppl,
           'best_val_ppl_subset': _best_val_ppl,
           'final_subset_val_ppl': _last_subset_ppl,
           'final_full_val_ppl': (float(ppl) if L._is_finite_number(ppl) else None),
           'final_minus_best': (float(_last_subset_ppl) - float(_best_val_ppl)) if (L._is_finite_number(_last_subset_ppl) and L._is_finite_number(_best_val_ppl)) else None,
           'train_ce_fixed_final': _last_train_ce,
           'train_val_gap_final': _health.get('gap_last'),
           'eval_subset_rows': int(np.asarray(_eval_batch).shape[0]) if _eval_batch is not None else 0,
           'nonfinite_loss_steps': int(_nonfinite_loss_steps),
           'nonfinite_grad_steps': int(_nonfinite_grad_steps),
           'efficiency': _efficiency, 'checkpoints': _checkpoints}
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
    return f'warm_v2_steps{cfg['steps']}_sl{cfg['seq_len']}_bs{cfg['batch_size']}_nt{cfg['n_train_tokens']}_lr{cfg['lr']}_wd{cfg['weight_decay']}_wu{_wu}_cl{cfg.get('comp_lambda', 0.05)}_dlm{cfg.get('delta_lr_mult', 10.0)}_gc{L.training_knobs().get('grad_clip_norm')}_tk{_v7_knobs_fp(DEVICE, cfg['batch_size'])}_d{d}_L{n_layers}_H{n_heads}_Dh{d_head}_v{vocab}_mt{_mkey}_mr{cfg.get('mlp_match_ref', 'csa_dynamic')}_ee{cfg.get('eval_every', 0)}_es{cfg.get('eval_subset', 128)}_det{L.determinism_label()}_df{L._ids_fp(train_ids)}-{L._ids_fp(np.ascontiguousarray(val_batch).reshape(-1))}_cs{CKPT_CODE}'


def _phase_cell_record(v, seed, rec, *, run_cfg, metric='ppl', key=None):
    """Normalised manifest cell for a custom phase (warmup/NIAH/lenphase)."""
    _losses = rec.get('losses') if isinstance(rec.get('losses'), list) else []
    _loss_nums = [float(x) for x in _losses if isinstance(x, (int, float)) and not isinstance(x, bool)]
    _curve = {
        'n_points': len(_loss_nums),
        'first': _loss_nums[0] if _loss_nums else rec.get('curve_first'),
        'last': _loss_nums[-1] if _loss_nums else rec.get('curve_last'),
        'best': min(_loss_nums) if _loss_nums else rec.get('curve_best'),
    }
    if metric == 'ppl':
        _metric = rec.get('ppl')
    elif metric == 'acc':
        _metric = rec.get('acc')
    elif metric == 'by_len':
        _vals = [float(x.get('ppl')) for x in (rec.get('by_len') or {}).values()
                 if isinstance(x, dict) and L.ppl_is_usable(x.get('ppl'))]
        _metric = (float(np.mean(_vals)) if _vals else None)
    else:
        _metric = None
    return {
        'variant': v,
        'seed': int(seed),
        'key': key,
        'run_id': rec.get('run_id'),
        'started_at': rec.get('started_at'),
        'finished_at': rec.get('finished_at'),
        'ppl': rec.get('ppl'),
        'accuracy': rec.get('acc'),
        'metric': metric,
        'metric_value': _metric,
        'steps': rec.get('steps'),
        'tokens_seen': rec.get('tokens_seen'),
        'params': rec.get('params'),
        'train_time_s': rec.get('train_time_s'),
        'final_loss_smoothed': rec.get('final_loss_smoothed'),
        'rng_fp': rec.get('rng_fp'),
        'data_fp': rec.get('data_fp'),
        'val_fp': rec.get('val_fp'),
        'data_version': rec.get('data_version') or (rec.get('provenance') or {}).get('data', {}).get('data_version'),
        'code_stamp': rec.get('_code') or CKPT_CODE,
        'run_cfg': rec.get('run_cfg') or run_cfg,
        'hyperparams': rec.get('hyperparams'),
        'curve_summary': _curve,
        'curve_ref': rec.get('curve_ref') or (f'summary.json#{key}.losses' if key else None),
        'sanity_ref': rec.get('sanity_ref'),
    }

def _write_phase_manifest(outdir, cfg, run_cfg, prov, *, phase, planned_cells, cell_records=None, status='running', started_at=None, data_contract=None):
    """Write/refresh the same crash-safe run_manifest.json used by run()."""
    try:
        return L.write_run_manifest(
            outdir, cfg, run_cfg, prov, {},
            status=status, planned_cells=list(planned_cells),
            started_at=started_at, cell_records=cell_records,
            data_contract=data_contract)
    except Exception as _e:
        print(f'[track] WARNING: could not write run_manifest.json for {phase} ({type(_e).__name__}: {_e})')
        return None
