#!/usr/bin/env python3
import dataclasses
import gc
import json
import math
import os
import re
import subprocess
import sys
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
BUDGET_V7 = dict(L.BUDGET)
BUDGET_V7.update(total_yuan=float(os.environ.get('V7_BUDGET_YUAN', 107.0)), price_per_hour=float(os.environ.get('V7_PRICE_PER_HOUR', 2.4)), state_path='autodl_budget_state_v7.json')
BUDGET_V7['already_spent_yuan'] = 0.0
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
    xr = x[..., :rope_dim]
    xp = x[..., rope_dim:]
    x1, x2 = (xr[..., :half], xr[..., half:])
    rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)
    return torch.cat([rot, xp], -1)

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
    _tmp = path + '.tmp'
    torch.save(obj, _tmp)
    os.replace(_tmp, path)

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
        if self.sink is None:
            sdpa_kw = dict(scale=scale)
            if cfg.window > 0:
                sdpa_kw['attn_mask'] = L.causal_window_mask(T, cfg.window, x.device)
            else:
                sdpa_kw['is_causal'] = True
            out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), **sdpa_kw).transpose(1, 2)
            return self.W_o(out.reshape(B, T, self.nh * self.hd))
        logits = torch.einsum('bnhd,bmhd->bhnm', q, k)
        logits.mul_(scale)
        mask = L.causal_mask(T, x.device)
        if cfg.window > 0:
            mask = L.causal_window_mask(T, cfg.window, x.device)
        logits.add_(mask)
        if self.sink is not None:
            sink_view = torch.nan_to_num(self.sink).view(1, self.nh, 1, 1)
            lse = torch.logaddexp(sink_view, torch.logsumexp(logits, dim=-1, keepdim=True))
            attn = torch.exp(logits - lse)
        else:
            attn = torch.softmax(logits, -1)
        out = torch.einsum('bhnm,bmhd->bnhd', attn, v)
        return self.W_o(out.reshape(B, T, self.nh * self.hd))

    def _single_rope(self, x, pre=None):
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
            fused = pre['fused'] if pre is not None and 'fused' in pre else self._fuse(x)
            sim = L.cosine_similarity_consecutive(fused)
            if sim.numel() == 0:
                bid = torch.zeros(T, dtype=torch.long, device=x.device)
                nblk = 1
                gate_mean = torch.zeros((), device=x.device) if self.need_reg else None
            else:
                prefix_mean = torch.cumsum(sim, 0) / L._range_cache(1, sim.numel() + 1, x.device)
                tau = prefix_mean + self.delta_logit
                gate = torch.sigmoid((tau - sim) / max(cfg.temperature, 0.001))
                hard_b = gate.detach() > L._HALF
                gl = hard_b.cpu().tolist()
                with torch.no_grad():
                    bid, nblk, _hon_d = L.blocks_from_cuts(T, gl, cfg.min_block, cfg.max_block, x.device, return_count=True, return_honoured=True)
                if self.need_reg:
                    _keep = [0.0] * len(gl)
                    for _slot in _hon_d:
                        if 0 <= _slot < len(_keep):
                            _keep[_slot] = 1.0
                    _hon = torch.tensor(_keep, dtype=gate.dtype, device=gate.device)
                    _soft_hon = gate * _hon
                    gate_mean = (float(nblk - 1) + _soft_hon.sum() - _soft_hon.detach().sum()) / T
        elif cfg.chunking in ('cosine_abs', 'cosine_adaptive') or (cfg.dynamic and cfg.kind != 'hca'):
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
            comp_kv, last_tok, Bn = L.pool_blocks_single(Ca_raw, Za, self.B_pos_a, bid, n_blocks=nblk, monotonic=True)
        else:
            comp_kv, last_tok, Bn = L.pool_variable_blocks(Ca_raw, Cb_raw, Za, Zb, bid, self.B_pos_a, self.B_pos_b, cfg.overlap, n_blocks=nblk, monotonic=True)
        index_kv = comp_kv
        attn_kv = torch.zeros_like(comp_kv) if cfg.content_mode == 'zero' else comp_kv
        if self._stats is not None:
            self._stats.append(torch.bincount(bid).float().cpu())
            if self._cuts is not None:
                cuts = (bid[1:] != bid[:-1]).nonzero(as_tuple=True)[0] + 1
                self._cuts.append(cuts.cpu())
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
            topk_idx = L._arange_cache(Bn, x.device).unsqueeze(0).expand(T, Bn)
        else:
            _sel_valid_box = []
            _, topk_idx, soft = L.lightning_indexer(x, index_kv, last_tok, self.W_DQ, self.W_DK, self.W_w.weight, cfg.n_index_heads, cfg.index_topk, return_mask=False, random_select=cfg.indexer_mode == 'random', pre_qI=pre['qI'] if pre is not None else None, pre_w=pre['w_idx'] if pre is not None else None, out_valid=_sel_valid_box)
            sel_valid = _sel_valid_box[0]
        sink = self.sink if cfg.use_sink else None
        out = self._rope_attn(qn, k_blk, v_blk, topk_idx, last_tok, k_sw, v_sw, cfg.sliding_window, scale, sink, rd, soft=soft, sel_valid=sel_valid)
        return (self.W_o(out.reshape(T, nh * hd)), gate_mean)

    def _rope_attn(self, q, k_blk, v_blk, topk_idx, last_tok, k_sw, v_sw, w, scale, sink, rope_dim, q_chunk=128, soft=None, rope_base=10000.0, mem_budget_bytes=None, sel_valid=None):
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
        qr = apply_rope(q, cos[:, None, :], sin[:, None, :], rope_dim)
        _lt = last_tok.long()
        bc, bs = (cos[_lt], sin[_lt])
        k_blk_r = apply_rope(k_blk, bc[:, None, :], bs[:, None, :], rope_dim)
        k_sw_r = apply_rope(k_sw, cos[:, None, :], sin[:, None, :], rope_dim)
        out = torch.empty_like(q)
        _wi_all, _wvalid_all = L._window_geometry(T, w, dev)
        pos_all = L._arange_cache(T, dev)
        q_chunk = max(1, min(int(q_chunk), T))
        _n_blk = k_blk_r.shape[0]
        _k_stack = torch.cat([k_blk_r, k_sw_r], 0)
        _v_stack = torch.cat([v_blk, v_sw], 0)
        for s in range(0, T, q_chunk):
            e = min(s + q_chunk, T)
            pos = pos_all[s:e]
            ib = topk_idx[s:e].long()
            sel = pos[:, None] > last_tok[ib]
            _keep = sel_valid[s:e] if sel_valid is not None else None
            if ib.shape[1] > 1:
                _ddk = L.first_occurrence_mask(ib)
                _keep = _ddk if _keep is None else _keep & _ddk
            if _keep is not None:
                sel = sel & _keep
            wvalid = _wvalid_all[s:e]
            wi = _wi_all[s:e]
            both = torch.cat([ib, wi + _n_blk], 1).to(torch.int32)
            Kset = L._take_2d(_k_stack, both)
            Vset = L._take_2d(_v_stack, both)
            valid = torch.cat([sel, wvalid], 1)
            raw_logits = torch.einsum('qhd,qmhd->qhm', qr[s:e], Kset) * scale
            if soft is None:
                logits = raw_logits.masked_fill_(~valid[:, None, :], torch.finfo(raw_logits.dtype).min)
                attn, _sink_unused = L._sink_split_softmax(logits, sink, want_sink=False)
                if sink is None:
                    attn = attn * valid.any(-1)[:, None, None].to(attn.dtype)
            if soft is not None:
                logits = raw_logits.masked_fill(~valid[:, None, :], torch.finfo(raw_logits.dtype).min)
                attn, _sink_unused = L._sink_split_softmax(logits, sink, want_sink=False)
                nb = ib.shape[1]
                sv = torch.gather(soft[s:e], 1, ib)
                if _keep is not None:
                    sv = sv * _keep.to(sv.dtype)
                soft_log = torch.log(sv.clamp_min(1e-12))
                soft_log.masked_fill_(sv <= 0, torch.finfo(sv.dtype).min)
                soft_logits = torch.cat([raw_logits[:, :, :nb] + soft_log[:, None, :], raw_logits[:, :, nb:]], -1)
                soft_logits = soft_logits.masked_fill(~valid[:, None, :], torch.finfo(soft_logits.dtype).min)
                soft_attn, _sink_unused = L._sink_split_softmax(soft_logits, sink, want_sink=False)
                attn = attn + (soft_attn - soft_attn.detach())
                if sink is None:
                    attn = attn * valid.any(-1)[:, None, None].to(attn.dtype)
            out[s:e] = torch.einsum('qhm,qmhd->qhd', attn, Vset)
        return out
    def _dense_warmup_forward(self, x):
        B, T, _ = x.shape
        nh, hd = (self.nh, self.hd)
        scale = 1.0
        use_rope = isinstance(self.cfg, AttnCfgRope) and self.cfg.rope
        rd = min(getattr(self.cfg, 'rope_dim', 0), hd)
        mask = L.causal_mask(T, x.device)
        win = getattr(self, '_dense_max_T', None)
        if win is not None and win < T:
            mask = L.causal_window_mask(T, win, x.device)
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
        out = torch.empty(B, T, nh, hd, device=x.device, dtype=q.dtype)
        row_chunk = max(1, min(T, 1024))
        _budget = L._attn_transient_budget(x.device)
        if _budget is not None:
            _per_row = max(1, 3 * B * nh * T) * q.element_size()
            row_chunk = max(1, min(row_chunk, int(_budget) // _per_row))
        for s in range(0, T, row_chunk):
            e = min(s + row_chunk, T)
            logits = torch.einsum('bthd,bshd->bths', q[:, s:e], k)
            logits.mul_(scale)
            logits.add_(mask[s:e][:, None, :])
            zflat = logits.reshape(B * (e - s), nh, T)
            if sink is None:
                attn = L.sink_softmax(zflat, sink)
            else:
                attn, _sink_unused = L._sink_split_softmax(zflat, sink, want_sink=False)
            out[:, s:e] = torch.einsum('bths,bshd->bthd', attn.view(B, e - s, nh, T), v)
        return self.W_o(out.reshape(B, T, nh * hd))

    def forward(self, x):
        if getattr(self, '_dense_warmup', False) and self.cfg.kind in ('csa', 'hca'):
            self.last_gate_mean = None
            return self._dense_warmup_forward(x)
        if getattr(self.cfg, 'rope', False) and self.cfg.kind in ('csa', 'hca'):
            self.last_gate_mean = None
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
            outs = []
            gates = []
            for b in range(B):
                pre = None if pre_all is None else {kk: v[b] if v is not None else None for kk, v in pre_all.items()}
                o, g = self._single_rope(x[b], pre=pre)
                outs.append(o)
                if g is not None:
                    gates.append(g)
            outs = torch.stack(outs, 0)
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
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x

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
        T = ids.shape[1]
        x = self.tok(ids)
        if getattr(self, 'use_abs_pe', True):
            x = x + self.pos(L._arange_cache(T, ids.device))
        for blk in self.blocks:
            x = blk(x)
        reg = torch.zeros((), device=ids.device)
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
                reg = reg + mult * (gm * Lb - 1.0) ** 2
        self.comp_reg = reg
        if logits_tail:
            x = x[:, -logits_tail:]
        return self.head(self.norm(x))

def make_layer_cfgs_v7(n_layers, variant):
    if variant == 'csa_fixed_rope':
        return [AttnCfgRope(kind='csa', dynamic=False, block_size=4, overlap=OVERLAP, index_topk=32, rope=True, rope_dim=16, qk_norm=True) for _ in range(n_layers)]
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
    _orig_smallgpt = L.__dict__.get('_v7_orig_smallgpt', L.SmallGPT)
    L.__dict__['_v7_orig_smallgpt'] = _orig_smallgpt

    def smallgpt_factory(vocab, d, n_layers, n_heads, d_head, max_seq, layer_cfgs, mlp_ratio=4):
        if any((getattr(c, 'rope', False) for c in layer_cfgs)):
            return SmallGPTRoPE(vocab, d, n_layers, n_heads, d_head, max_seq, layer_cfgs, mlp_ratio)
        return _orig_smallgpt(vocab, d, n_layers, n_heads, d_head, max_seq, layer_cfgs, mlp_ratio)
    _orig_forward = L.HybridAttention.forward

    def forward_warmup_aware(self, x):
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
CKPT_CODE = L.CKPT_CODE
ROPE_VARIANTS = ['full_rope', 'csa_fixed_rope', 'hybrid_fixed_rope', 'csa_dynamic_rope', 'csa_fixed']
PARAM_MATCHED_V7 = {'full_rope', 'full_sw128_matched_rope'}

def train_warmup(variant, train_ids, val_batch, vocab, *, seed=0, d=256, n_layers=6, n_heads=8, d_head=32, seq_len=512, batch_size=12, steps=20000, warm_steps=0, lr=0.0003, weight_decay=0.1, warmup=200, comp_lambda=0.05, delta_lr_mult=10.0, eval_every=1000, eval_subset=128, val_bnd=None, device=DEVICE, log_every=1000, mlp_ratio=4, deadline_ts=None):
    L.set_seed(seed)
    L._attb_bump_epoch()
    if variant in set(L.PARAM_MATCHED) | set(PARAM_MATCHED_V7) and float(mlp_ratio) == 4.0:
        raise ValueError(f'{variant} is a PARAM-MATCHED baseline but was given the default mlp_ratio=4; its MLP must be widened to match its sparse reference arm, or the comparison is not parameter-controlled. Pass mlp_ratio=L.variant_mlp_ratio(...).')
    if steps < 1:
        raise ValueError(f'steps={steps}: nothing to train')
    if warm_steps >= steps:
        raise ValueError(f'warm_steps={warm_steps} >= steps={steps}: the dense->sparse switch never fires, so the recorded numbers would come from a dense-protocol model')
    cfgs = L.make_layer_cfgs(n_layers, variant)
    model = L.SmallGPT(vocab, d, n_layers, n_heads, d_head, seq_len, cfgs, mlp_ratio=mlp_ratio).to(device)
    n_param = L.count_params(model)
    print(f'\n[{variant} seed={seed}] WARMUP warm_steps={warm_steps} total={steps}  params={n_param / 1000000.0:.2f}M  mlp_ratio={mlp_ratio:.2f}')
    is_no_decay = L.is_no_decay
    is_delta = L.is_delta_param
    decay = [p for n_, p in model.named_parameters() if p.requires_grad and (not is_no_decay(n_)) and (not is_delta(n_))]
    ndecay = [p for n_, p in model.named_parameters() if p.requires_grad and is_no_decay(n_) and (not is_delta(n_))]
    dpar = [p for n_, p in model.named_parameters() if p.requires_grad and is_delta(n_)]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': weight_decay, 'lr_scale': 1.0}, {'params': ndecay, 'weight_decay': 0.0, 'lr_scale': 1.0}, {'params': dpar, 'weight_decay': 0.0, 'lr_scale': delta_lr_mult}], lr=lr)

    def lr_at(step):
        if warmup > 0 and step < warmup:
            return lr * (step + 1) / warmup
        t = (step - warmup) / max(steps - warmup, 1)
        return lr * (0.1 + 0.45 * (1.0 + math.cos(math.pi * t)))
    bpe = L.batch_iter(train_ids, seq_len, batch_size, device, seed=seed)
    _eval_batch = val_batch[:eval_subset]
    warm_on = warm_steps > 0
    for _blk in model.blocks:
        _blk.attn._dense_warmup = warm_on
    t0 = time.time()
    losses, ppl_hist, warm_hist, switch_ppl = ([], [], [], None)
    _lbuf = []
    _tail = []
    model.train()
    x = y = logits = ce = loss = None
    for step in range(steps):
        if deadline_ts is not None and time.time() > deadline_ts:
            wall = time.time() - t0
            print(f'[{variant} seed={seed} warm={warm_steps}] BUDGET deadline reached after {step} steps ({wall / 60:.1f} min) — stopping; this cell produced NO measurement and will be retried.')
            del model, opt, bpe, decay, ndecay, dpar, x, y, logits, ce, loss
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            return {'budget_truncated': True, 'steps_done': step, 'train_time_s': wall}
        if warm_steps > 0 and step == warm_steps:
            for _blk in model.blocks:
                _blk.attn._dense_warmup = False
            switch_ppl = float(L.eval_ppl(model, _eval_batch, device))
            print(f'  [warmup] step {step}: dense -> SPARSE (h=val PPL {switch_ppl:.2f})')
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
        _lbuf.append(ce.detach())
        _tail.append(ce.detach())
        if len(_tail) > 50:
            del _tail[:-50]
        if len(_lbuf) >= 512:
            losses.extend(torch.stack(_lbuf).tolist())
            _lbuf.clear()
        if eval_every and ((step + 1) % eval_every == 0 or step == steps - 1):
            _pv = float(L.eval_ppl(model, _eval_batch, device))
            (warm_hist if getattr(model.blocks[0].attn, '_dense_warmup', False) else ppl_hist).append([step + 1, _pv])
        if log_every and (step % log_every == 0 or step == steps - 1):
            if _lbuf:
                losses.extend(torch.stack(_lbuf).tolist())
                _lbuf.clear()
            print(f'  step {step:5d}  loss {float(losses[-1]):.4f}  lr {opt.param_groups[0]['lr']:.2e}  dense={getattr(model.blocks[0].attn, '_dense_warmup', False)}  ({(time.time() - t0) / max(step + 1, 1) * 1000:.0f}ms/step)')
    wall = time.time() - t0
    if deadline_ts is not None and time.time() > deadline_ts:
        print(f'[{variant} seed={seed} warm={warm_steps}] BUDGET deadline reached before the final evaluation ({wall / 60:.1f} min) — truncating; this cell produced NO measurement and will be retried.')
        del model, opt, bpe, decay, ndecay, dpar, x, y, logits, ce, loss
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return {'budget_truncated': True, 'steps_done': steps, 'train_time_s': wall}
    del x, y, logits, ce, loss
    ppl = L.eval_ppl(model, val_batch, device)
    stats = L.compression_report(model, val_batch, device, val_bnd=val_bnd)
    print(f'[{variant} seed={seed} warm={warm_steps}] val PPL = {ppl:.3f}  ({wall / 60:.1f} min)')
    if _lbuf:
        losses.extend(torch.stack(_lbuf).tolist())
        _lbuf.clear()
    losses = [float(v) for v in losses]
    res = {'variant': variant, 'seed': seed, 'ppl': ppl, 'params': n_param, 'losses': losses, 'stats': stats, 'steps': steps, 'tokens_seen': steps * batch_size * seq_len, 'train_time_s': wall, 'warm_steps': warm_steps, 'ppl_at_switch': switch_ppl, 'final_loss_smoothed': float(torch.stack(_tail[-50:]).mean()) if _tail else float('nan'), 'ppl_history': ppl_hist, 'ppl_history_dense_prefix': warm_hist, 'delta_trace': {}}
    del model, opt, bpe, decay, ndecay, dpar
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return res

def run_warmup(cfg, seeds, guard=None, label='', warm_grid=(0, 5000, 10000), variants=('csa_fixed',)):
    cfg = dict(cfg)
    outdir = cfg['outdir']
    os.makedirs(outdir, exist_ok=True)
    d = cfg.get('d', 256)
    n_layers = cfg.get('n_layers', 6)
    n_heads = cfg.get('n_heads', 8)
    d_head = cfg.get('d_head', 32)
    t_data = time.time()
    train_ids, val_batch, vocab, _, val_bnd = L.load_wikitext(cfg['seq_len'], cfg['n_train_tokens'])
    if guard is not None:
        guard.record_run(time.time() - t_data, 0, 0, 0, 0, 0)
    spath = os.path.join(outdir, 'summary.json')
    summary = {}
    if os.path.exists(spath):
        try:
            summary = json.load(open(spath, encoding='utf-8'))
        except Exception as _e:
            print(f'[resume] FATAL: {spath} exists but cannot be parsed ({type(_e).__name__}: {_e}).  Refusing to overwrite it with an empty summary — move it aside to start fresh.')
            raise
    print(f'\n########## v7 phase: {label} — {len(variants)} variants x warm {warm_grid} x seeds {seeds} x {cfg['steps']} steps ##########')
    ratios = {}
    for v in variants:
        if v not in ratios:
            ratios[v] = L.variant_mlp_ratio(v, vocab, d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, seq_len=cfg['seq_len'], matched=set(PARAM_MATCHED_V7) | {'full_matched', 'full_sw128_matched'}, ref_variant=cfg.get('mlp_match_ref', 'csa_dynamic'))
    _wu = cfg['warmup']
    _mkey = ','.join(sorted((str(_x) for _x in (set(PARAM_MATCHED_V7) | {'full_matched', 'full_sw128_matched'}))))
    _fp = f'steps{cfg['steps']}_sl{cfg['seq_len']}_bs{cfg['batch_size']}_nt{cfg['n_train_tokens']}_lr{cfg['lr']}_wd{cfg['weight_decay']}_wu{_wu}_cl{cfg.get('comp_lambda', 0.05)}_dlm{cfg.get('delta_lr_mult', 10.0)}_d{d}_L{n_layers}_H{n_heads}_Dh{d_head}_v{vocab}_mt{_mkey}_mr{cfg.get('mlp_match_ref', 'csa_dynamic')}_ee{cfg.get('eval_every', 0)}_es{cfg.get('eval_subset', 128)}_det{L.determinism_label()}_cs{CKPT_CODE}'
    stale_dropped = []
    for seed in seeds:
        for warm in warm_grid:
            for v in variants:
                key = f'{v}::w{warm}::seed{seed}'
                _cur = summary.get(key)
                _drop_msg = None
                if isinstance(_cur, dict) and ((_cur.get('synthesized')) or not L.result_is_current(_cur, CKPT_CODE, 'ppl')):
                    _drop_msg = f'[resume] {key} holds a record that is not current under this code semantics (stale code stamp, missing ppl, or synthesized) — DROPPING it before retraining, so a failed or truncated attempt cannot leave the old reading in place looking like a fresh result'
                    _cur = None
                if L.result_is_current(_cur, CKPT_CODE, 'ppl'):
                    if _cur.get('run_cfg') == _fp:
                        print(f'[skip] {key} already completed (resume)')
                        continue
                    _drop_msg = f'[resume] {key} carries no matching config fingerprint (stored {_cur.get('run_cfg')!r}) — re-running and overwriting so a config change can never be mistaken for a fresh result'
                if _drop_msg is not None:
                    print(_drop_msg)
                    summary.pop(key, None)
                    stale_dropped.append(key)
                    L.atomic_write_json(spath, summary, indent=2)
                if guard is not None:
                    est = guard.estimate_seconds(cfg['steps'], d=d, n_layers=n_layers, seq_len=cfg['seq_len'], batch_size=cfg['batch_size'])
                    if not guard.can_start(est):
                        print(f'[budget] SKIP {key}: projected ¥{est / 3600 * guard.price:.2f} would pass cap ¥{guard.cap_yuan():.2f} (spent ¥{guard.spent_yuan():.2f})')
                        continue
                t_run = time.time()
                _deadline = None
                if guard is not None:
                    _deadline = guard.deadline_ts()
                rec = None
                try:
                    rec = train_warmup(v, train_ids, val_batch, vocab, seed=seed, warm_steps=warm, d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, seq_len=cfg['seq_len'], batch_size=cfg['batch_size'], steps=cfg['steps'], lr=cfg['lr'], weight_decay=cfg['weight_decay'], warmup=_wu, comp_lambda=cfg['comp_lambda'], delta_lr_mult=cfg.get('delta_lr_mult', 10.0), eval_every=cfg.get('eval_every', 0), eval_subset=cfg.get('eval_subset', 128), val_bnd=val_bnd, mlp_ratio=ratios[v], deadline_ts=_deadline)
                    if rec.get('budget_truncated'):
                        print(f'[budget] {key} was truncated after {rec.get('steps_done')} steps — its partial record is DISCARDED (it holds no `ppl`) and the key is left ABSENT. Raise the budget and re-run to retry this cell.')
                    else:
                        summary[key] = rec
                        summary[key]['_code'] = CKPT_CODE
                        summary[key]['run_cfg'] = _fp
                except Exception as e:
                    if L.result_is_current(summary.get(key), CKPT_CODE, 'ppl'):
                        print(f'[{key}] FAILED: {e} — keeping the previous MEASURED record (current under this code semantics; the error record holds no `ppl` and must not replace it)')
                    else:
                        summary[key] = {'variant': v, 'seed': seed, 'warm_steps': warm, 'error': traceback.format_exc()}
                        print(f'[{key}] FAILED: {e}')
                if guard is not None:
                    steps_done = 0
                    calib_s = None
                    if isinstance(rec, dict) and rec.get('steps_done'):
                        steps_done = int(rec['steps_done'])
                    elif isinstance(rec, dict) and 'ppl' in rec:
                        steps_done = int(cfg['steps'])
                        calib_s = rec.get('train_time_s')
                    guard.record_run(time.time() - t_run, steps_done, d, n_layers, cfg['seq_len'], cfg['batch_size'], calib_seconds=calib_s)
                L.atomic_write_json(spath, summary, indent=2)
                gc.collect()
                if DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()
    if stale_dropped:
        _still = [k for k in stale_dropped if k not in summary]
        print(f'\n[resume] {len(stale_dropped)} warmup cell(s) held a non-current record and were dropped before retraining; {len(_still)} of them produced NO measurement this pass (budget gate / truncation / failure) and are ABSENT from {spath}:')
        for _k in _still:
            print(f'    {_k}')
        if len(_still) != len(stale_dropped):
            print(f'    ({len(stale_dropped) - len(_still)} of them succeeded and carry a fresh record.)')
    return summary

def build_niah_batch(n_seq, seq_len, n_pairs=4, vocab=8192, seed=0):
    rng = np.random.default_rng(seed)
    key_hi = max(2 + (vocab // 2 - 2) // 4, 2 + 2 * n_pairs)
    ids = rng.integers(key_hi, vocab // 2, size=(n_seq, seq_len)).astype(np.int64)
    tgt = np.full((n_seq, seq_len - 1), -100, dtype=np.int64)
    dist = np.full((n_seq, seq_len - 1), -1, dtype=np.int64)
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
    return (ids, tgt, dist)

@torch.inference_mode()
def eval_niah(model, seq_len, device=DEVICE, n_seq=64, n_pairs=4, vocab=8192, seed=1234, chunk=16):
    was = model.training
    model.eval()
    try:
        max_pos = getattr(model, 'max_seq', seq_len)
        ids, tgt, dist = build_niah_batch(n_seq, seq_len, n_pairs, vocab, seed)
        correct = np.zeros_like(tgt, dtype=bool)
        span = min(seq_len, max_pos) if getattr(model, 'use_abs_pe', True) else seq_len
        tail = 2 * n_pairs
        i = 0
        while i < n_seq:
            j = min(i + chunk, n_seq)
            x = torch.from_numpy(ids[i:j]).to(device)
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
                print(f'[niah] eval OOM; retrying with chunk={chunk}')
                continue
            pred = logits[:, :-1].argmax(-1).cpu().numpy()
            t = tgt[i:j, -(tail - 1):]
            correct[i:j, -(tail - 1):] = (pred == t) & (t >= 0)
            del x, logits
            i = j
        scored_cols = tail - 1
        m = np.zeros_like(tgt, dtype=bool)
        m[:, -scored_cols:] = tgt[:, -scored_cols:] >= 0
        if m.sum() == 0:
            return {'acc': 0.0, 'n': 0, 'by_dist': {}}
        d = dist[m]
        c = correct[m]
        if getattr(model, 'use_abs_pe', True) and seq_len > max_pos:
            _cut = seq_len - span
            _cols = np.nonzero(m)[1]
            _in_ctx = (_cols - d) >= _cut
            d = d[_in_ctx]
            c = c[_in_ctx]
            if d.size == 0:
                return {'acc': 0.0, 'n': 0, 'by_dist': {}}
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
    if n == 0:
        return {'n': 0, 'mean': float('nan'), 'std': float('nan'), 'p_exact_signflip': float('nan'), 'n_flips': 0, 'n_dropped': n_dropped}
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
        return {'n': n, 'mean': float(d.mean()), 'std': float(d.std(ddof=1)) if n > 1 else 0.0, 'p_exact_signflip': p, 'n_flips': B, 'n_dropped': n_dropped}
    cnt = 0
    total = 1 << n
    bitpos = np.arange(n, dtype=np.int64)[None, :]
    step = max(1, min(total, (1 << 24) // max(n, 1)))
    for off in range(0, total, step):
        bs = min(step, total - off)
        masks = np.arange(off, off + bs, dtype=np.int64)[:, None]
        flips = np.where(masks & 1 << bitpos != 0, 1.0, -1.0)
        cnt += int((np.abs((flips * d[None, :]).mean(1)) >= obs - 1e-12).sum())
    p = float(cnt / total)
    return {'n': n, 'mean': float(d.mean()), 'std': float(d.std(ddof=1)) if n > 1 else 0.0, 'p_exact_signflip': p, 'n_flips': int(total), 'n_dropped': n_dropped}

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
    L.atomic_write_json(os.path.join(outdir, 'flops_analytic.json'), {'config': {'d': d, 'heads': H, 'd_head': Dh, 'layers': L_layers, 'm': m, 'topk': k, 'window': w, 'm_hca': mph}, 'crossover_seq_len_csa_beats_dense': cross, 'rows': rows})
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
            print(f'[stats] {name}: NO usable pair after the run_cfg/budget checks — comparison omitted rather than quoting a cross-configuration delta')
            continue
        res = exact_sign_permutation(dl)
        res['n_skipped_config_mismatch'] = len(skipped)
        res['n_unstamped'] = unstamped
        out_d['comparisons'][name] = res
    L.atomic_write_json(out, out_d)
    print(f'[stats] wrote {out}')
    for k, v in out_d['comparisons'].items():
        if 'omitted' in v:
            print(f'  {k:34s} omitted — {v['omitted']}')
            continue
        warn = f'  [{v['n_skipped_config_mismatch']} seed(s) dropped: config mismatch]' if v.get('n_skipped_config_mismatch') else ''
        print(f'  {k:34s} Δ={v['mean']:+7.2f} ± {v.get('std', 0):5.2f}  p(sign-flip)={v['p_exact_signflip']:.3f}  n={v['n']}{warn}')
    return out_d

def git_push(msg):
    if os.environ.get('V7_NO_PUSH'):
        print(f'[git] push skipped (V7_NO_PUSH): {msg}')
        return True
    run = lambda *a: subprocess.run(a, cwd=REPO, capture_output=True, text=True, encoding='utf-8', errors='replace')
    run('git', 'add', '-A')
    r = run('git', 'commit', '-m', msg)
    committed = r.returncode == 0
    if not committed and 'nothing to commit' not in r.stdout + r.stderr:
        print(f'[git] commit FAILED (will still try to push): {(r.stdout + r.stderr)[-300:]}')
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
    print(f'[v7] AutoDL instance shuts down in {delay_s}s (only the data disk bills afterwards).')
LONG_EXT = dict(L.RUN_LONG, steps=40000, n_train_tokens=110000000, eval_every=2000, outdir='results_lm_v7_long40')
PHASES = [('P0R', 'plain', dict(cfg=dict(L.RUN, outdir='results_lm_v7_rope', variants=ROPE_VARIANTS, steps=1500, matched=set(PARAM_MATCHED_V7), mlp_match_ref='csa_fixed_rope'), seeds=[0, 1, 2]), 2.6), ('P0W', 'warm', dict(cfg=dict(L.RUN_LONG, outdir='results_lm_v7_warmup', variants=['csa_fixed']), seeds=[0, 1], warm_grid=(0, 5000, 10000)), 3.4), ('P1L', 'len', dict(outdir='results_len', vocab=8192, steps=3000, train_len=512, eval_lens=[512, 1024, 2048, 4096], variants=['full', 'csa_fixed', 'full_rope', 'csa_fixed_rope']), 2.5), ('P2S', 'plain', dict(cfg=dict(L.RUN_SCALE, outdir='results_lm_v5_scale', variants=['full_matched', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic']), seeds=[2, 3]), 4.5), ('P1T', 'plain', dict(cfg=dict(L.RUN, outdir='results_lm_v7_seq2k', seq_len=2048, batch_size=3, steps=1500, variants=['csa_fixed_topk8', 'csa_fixed_topk32', 'csa_fixed_topk128', 'csa_fixed_topk512', 'csa_fix_m1']), seeds=[0, 1]), 6.0), ('P0E', 'plain', dict(cfg=dict(LONG_EXT, variants=['csa_fixed', 'full']), seeds=[0, 1]), 17.0)]

def run_phase(name, guard, only=None):
    for pname, kind, payload, _h in PHASES:
        if pname != name:
            continue
        if kind == 'plain':
            s, a = L.run(payload['cfg'], seeds=payload['seeds'], guard=guard, label=f'v7 {pname}')
            return s
        if kind == 'warm':
            return run_warmup(payload['cfg'], seeds=payload['seeds'], guard=guard, label=f'v7 {pname}', warm_grid=payload['warm_grid'], variants=tuple(payload['cfg'].get('variants', ('csa_fixed',))))
        if kind == 'niah':
            return run_niah_phase(payload, guard, label=f'v7 {pname}')
        if kind == 'len':
            return run_lenphase(payload, guard, label=f'v7 {pname}')
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
    spath = os.path.join(outdir, 'summary.json')
    summary = {}
    if os.path.exists(spath):
        try:
            summary = json.load(open(spath, encoding='utf-8'))
        except Exception as _e:
            print(f'[resume] FATAL: {spath} exists but cannot be parsed ({type(_e).__name__}: {_e}).  Refusing to overwrite it with an empty summary — move it aside to start fresh.')
            raise
    _mr_ref = 'csa_fixed_rope'
    _recipe = f'niah_steps{n_steps}_v{vocab}_bs12_sl512_lr0.0003_mr-{_mr_ref}_reuse4'
    for v in variants:
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
                L.atomic_write_json(spath, summary, indent=2)
                if guard is not None:
                    est = guard.estimate_seconds(n_steps, d=256, n_layers=6, seq_len=512, batch_size=12)
                    if not guard.can_start(est):
                        print(f'[budget] SKIP niah train {v} s{seed}')
                        continue
                t0 = time.time()
                cfgs = L.make_layer_cfgs(6, v)
                mr = 4.0
                if v in PARAM_MATCHED_V7:
                    mr = L.variant_mlp_ratio(v, vocab, d=256, n_layers=6, n_heads=8, d_head=32, seq_len=512, matched=set(PARAM_MATCHED_V7), ref_variant=_mr_ref)
                # 随机源在建模前一刻定界：同 seed 下各臂的共享组件（tok/head/MLP）必须同构
                L.set_seed(seed)
                model = L.SmallGPT(vocab, 256, 6, 8, 32, 512, cfgs, mlp_ratio=mr).to(DEVICE)
                opt = torch.optim.AdamW(model.parameters(), lr=0.0003)
                model.train()
                cache = {'ids': None, 'tgt': None, 'age': 10 ** 9, 'epoch': 0}

                def _batch(cache=cache, _seed=seed, _vocab=vocab):
                    if cache['age'] >= 4 or cache['ids'] is None:
                        eps = _seed * 100003 + cache['epoch'] * 7919
                        ids, tgt, _ = build_niah_batch(12, 512, 4, _vocab, seed=eps)
                        cache['ids'] = torch.from_numpy(ids).to(DEVICE)
                        cache['tgt'] = torch.from_numpy(tgt).to(DEVICE)
                        cache['age'] = 0
                        cache['epoch'] += 1
                    cache['age'] += 1
                    return (cache['ids'], cache['tgt'])
                _deadline = guard.deadline_ts() if guard is not None else None
                niah_truncated = False
                for step in range(n_steps):
                    if _deadline is not None and time.time() > _deadline:
                        wall = time.time() - t0
                        print(f'[budget] niah train {v} s{seed} hit the deadline after {step}/{n_steps} steps ({wall / 60:.1f} min) — the cell is abandoned WITHOUT a checkpoint, so it retrains from scratch on the next pass rather than being probed from a half-trained model.')
                        niah_truncated = True
                        break
                    x, y = _batch()
                    lg = model(x)
                    ce = F.cross_entropy(lg[:, :-1].reshape(-1, vocab), y.reshape(-1), ignore_index=-100)
                    loss = ce + 0.05 * model.comp_reg
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    if step % 500 == 0 or step == n_steps - 1:
                        print(f'  [niah-train] {v:18s} s{seed} step {step:5d} loss {loss.item():.4f}', flush=True)
                if niah_truncated:
                    del model, opt
                    gc.collect()
                    torch.cuda.empty_cache()
                    if guard is not None:
                        guard.record_run(time.time() - t0, step, 256, 6, 512, 12)
                    continue
                _atomic_torch_save({'cfg': cfgs, 'mlp_ratio': mr, 'vocab': vocab, 'd': 256, 'n_layers': 6, 'n_heads': 8, 'd_head': 32, 'code': CKPT_CODE, 'recipe': _recipe, 'sd': model.state_dict(), 'params': L.count_params(model)}, ck)
                if guard is not None:
                    guard.record_run(time.time() - t0, n_steps, 256, 6, 512, 12)
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
                    L.atomic_write_json(spath, summary, indent=2)
                    break
                t0 = time.time()
                try:
                    r = eval_niah(model, Ln, DEVICE, vocab=int(_ckp.get('vocab', vocab)))
                except Exception as e:
                    r = {'error': f'{type(e).__name__}: {e}'}
                r.update({'variant': v, 'seed': seed, 'seq_len': Ln, 'params': _ckp['params'], 'probe_s': time.time() - t0, '_code': CKPT_CODE})
                summary[key] = r
                print(f'  [niah] {v:18s} s{seed} len={Ln:5d}  acc={r.get('acc')}  by_dist={r.get('by_dist')}')
                L.atomic_write_json(spath, summary, indent=2)
            del model, _ckp
            gc.collect()
            torch.cuda.empty_cache()
    _niah_plot(summary, outdir, seq_lens)
    return summary

@torch.inference_mode()
def eval_length_gen(model, val_ids, eval_lens, device=DEVICE, n_seq=8, max_pos=None):
    was = model.training
    model.eval()
    out = {}
    try:
        for Ln in eval_lens:
            trunc = False
            ids = np.asarray(val_ids[:n_seq, :Ln], dtype=np.int64)
            if max_pos is not None and Ln > max_pos:
                ids = ids[:, -max_pos:]
                trunc = True
            try:
                x = torch.from_numpy(ids).to(device)
                nll, ntok = (torch.zeros((), dtype=torch.float64, device=device), 0)
                _r, _chunk = (0, int(ids.shape[0]))
                while _r < int(ids.shape[0]):
                    _sub = x[_r:_r + _chunk]
                    try:
                        logits = model(_sub)
                    except RuntimeError as _oe:
                        if 'out of memory' in str(_oe).lower() and _chunk > 1:
                            _chunk = max(1, _chunk // 2)
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()
                            continue
                        raise
                    nll += F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), _sub[:, 1:].reshape(-1), reduction='sum').double()
                    ntok += int(_sub[:, 1:].numel())
                    del logits
                    _r += _chunk
                ppl = math.exp(float(nll) / max(ntok, 1))
                out[int(Ln)] = {'ppl': float(ppl), 'n_tok': ntok, 'truncated': trunc, 'eval_span': int(x.shape[1])}
            except Exception as e:
                out[int(Ln)] = {'error': f'{type(e).__name__}: {e}'}
    finally:
        model.train(was)
    return out

def run_lenphase(payload, guard=None, label=''):
    outdir = payload['outdir']
    os.makedirs(outdir, exist_ok=True)
    ckpt_dir = os.path.join(outdir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    variants = payload['variants']
    eval_lens = payload['eval_lens']
    seeds = list(payload.get('seeds', [0, 1, 2]))
    steps = payload.get('steps', 3000)
    train_len = payload.get('train_len', 512)
    vocab = payload.get('vocab', 8192)
    spath = os.path.join(outdir, 'summary.json')
    summary = {}
    if os.path.exists(spath):
        try:
            summary = json.load(open(spath, encoding='utf-8'))
        except Exception as _e:
            print(f'[resume] FATAL: {spath} exists but cannot be parsed ({type(_e).__name__}: {_e}).  Refusing to overwrite it with an empty summary — move it aside to start fresh.')
            raise
    need = max(eval_lens) + 1
    _t_data = time.time()
    _, val_ids, _, _, _ = L.load_wikitext(max(train_len, need), 4000000)
    if guard is not None:
        guard.record_run(time.time() - _t_data, 0, 0, 0, 0, 0)
    print(f'[p1l] val slice {val_ids.shape}, eval_lens={eval_lens}, seeds={seeds}')
    _train_cache = [None]
    _mr_ref = 'csa_fixed_rope'
    _recipe = f'p1l_steps{steps}_tl{train_len}_v{vocab}_bs12_lr0.0003_nt8000000_wu50_wd0.1_cl0.05_dlm10.0_mr-{_mr_ref}'
    for v in variants:
        for seed in seeds:
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
                if summary.get(key0) is not None:
                    print(f'[p1l] {v} s{seed}: weights are being retrained — dropping the cached eval, which belongs to the previous weights')
                    del summary[key0]
                    L.atomic_write_json(spath, summary, indent=2)
                if guard is not None:
                    est = guard.estimate_seconds(steps, d=256, n_layers=6, seq_len=train_len, batch_size=12)
                    if not guard.can_start(est):
                        print(f'[budget] SKIP lenphase train {v} s{seed}')
                        continue
                t0 = time.time()
                L.set_seed(seed)
                cfgs = L.make_layer_cfgs(6, v)
                mr = 4.0
                if v in PARAM_MATCHED_V7:
                    mr = L.variant_mlp_ratio(v, vocab, d=256, n_layers=6, n_heads=8, d_head=32, seq_len=train_len, matched=set(PARAM_MATCHED_V7), ref_variant=_mr_ref)
                if _train_cache[0] is None:
                    _train_cache[0] = L.load_wikitext(train_len, 8000000)[0]
                train_ids = _train_cache[0]
                _tr = L.train_variant(v, train_ids, None, vocab, seed=seed, d=256, n_layers=6, n_heads=8, d_head=32, seq_len=train_len, batch_size=12, steps=steps, lr=0.0003, weight_decay=0.1, warmup=50, comp_lambda=0.05, delta_lr_mult=10.0, eval_every=0, mlp_ratio=mr, device=DEVICE, log_every=500, deadline_ts=guard.deadline_ts() if guard is not None else None, return_model=True)
                if _tr.get('budget_truncated'):
                    print(f'  [p1l-train] {v:18s} s{seed} TRUNCATED by budget — not saved, will retry')
                    _sd_tr = _tr.get('steps_done')
                    if guard is not None:
                        guard.record_run(time.time() - t0, _sd_tr if _sd_tr else 0, 256, 6, train_len, 12)
                    del _tr
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                model = _tr.pop('_model')
                _atomic_torch_save({'cfg': cfgs, 'mlp_ratio': mr, 'vocab': vocab, 'd': 256, 'n_layers': 6, 'n_heads': 8, 'd_head': 32, 'train_len': train_len, 'max_seq': train_len, 'code': CKPT_CODE, 'recipe': _recipe, 'sd': model.state_dict(), 'final_ppl': _tr.get('ppl'), 'params': L.count_params(model)}, ck)
                if guard is not None:
                    guard.record_run(time.time() - t0, steps, 256, 6, train_len, 12, calib_seconds=_tr.get('train_time_s'))
                del model
                gc.collect()
                torch.cuda.empty_cache()
            key = f'{v}::seed{seed}'
            _cur = summary.get(key)
            _need_lens = list(eval_lens)
            if L.result_is_current(_cur, CKPT_CODE, 'by_len'):
                _need_lens = [Ln for Ln in eval_lens if not L.by_len_cells([_cur], Ln)]
                if not _need_lens:
                    continue
                print(f'[resume] {key}: by_len record is stamped current but at least one eval length holds no usable measurement (error cell) — re-evaluating the missing lengths')
            _ckp = _ckp_pre if _ckp_pre is not None else _load_ckpt(ck)
            model = L.SmallGPT(_ckp.get('vocab', vocab), 256, 6, 8, 32, _ckp.get('train_len', train_len), _ckp['cfg'], mlp_ratio=_ckp['mlp_ratio']).to(DEVICE)
            model.load_state_dict(_ckp['sd'])
            mp = None if not getattr(model, 'use_abs_pe', True) else _ckp.get('max_seq', train_len)
            _t_ev = time.time()
            r = eval_length_gen(model, val_ids, _need_lens, DEVICE, max_pos=mp)
            if guard is not None:
                guard.record_run(time.time() - _t_ev, 0, 0, 0, 0, 0)
            _merged = {}
            for _bk, _bc in (((_cur or {}).get('by_len') or {}).items() if _need_lens != list(eval_lens) else []):
                try:
                    _merged[int(_bk)] = _bc
                except (TypeError, ValueError):
                    continue
            _merged.update(r)
            summary[key] = {'variant': v, 'seed': seed, 'params': _ckp['params'], 'max_pos': mp, 'by_len': _merged, '_code': CKPT_CODE}
            print(f'  [p1l] {v:18s} s{seed} ' + '  '.join((f'L{k}={vv.get('ppl', float('nan')):.2f}' for k, vv in sorted(_merged.items()))))
            L.atomic_write_json(spath, summary, indent=2)
            del model, _ckp
            gc.collect()
            torch.cuda.empty_cache()
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
    guard = L.CostGuard(BUDGET_V7)
    guard.report()
    print(f'[v7] remaining ¥{guard.remaining_yuan():.2f} (cap ¥{guard.cap_yuan():.2f} @ ¥{guard.price:.2f}/h)')
    git_push('v7: supplementary experiment library (RoPE+QK-norm, dense->sparse warmup, NIAH probe, topk sweep, analytic FLOPs)')
    all_ok = True
    for pname, kind, payload, est_h in PHASES:
        rem = guard.remaining_yuan()
        if rem < 1.0:
            print(f'[v7] stopping before {pname}: ¥{rem:.2f} left')
            break
        if not cuda_healthy():
            print(f'[v7] CUDA context poisoned before {pname} -- restart the process to continue (resume will skip completed runs).  Aborting phase loop.')
            all_ok = False
            break
        print(f'\n===== v7 phase {pname} (~{est_h} h est, ¥{rem:.2f} left) =====')
        try:
            run_phase(pname, guard)
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
            run_phase(sys.argv[2], L.CostGuard(BUDGET_V7))
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
