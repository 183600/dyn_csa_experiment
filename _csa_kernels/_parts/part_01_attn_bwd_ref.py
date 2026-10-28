

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
      dλ_l        = dzS_l（块泳道）；dsoft_l = dλ_l · [soft > 1e-12] / soft
      dsink_h     = −Σ_{r,t} (P_sink^A·GA + P_sink^S·GS)
    """
    R, T, nh, hd = q.shape
    w = int(w)
    k_sw, v_sw = _attn_window_pair(k_sw, v_sw, q, w)
    nb = topk_idx.shape[-1]
    Bm = k_blk.shape[1]
    dev = q.device
    dtype = q.dtype
    acc = torch.promote_types(dtype, torch.float32)
    MINL = _finfo_min(acc)
    M = nb + w
    t_idx = torch.arange(T, device=dev)[None, :, None]
    if valid is None:
        valid = torch.ones_like(topk_idx, dtype=torch.bool)
    win_raw = (t_idx - (w - 1) + torch.arange(w, device=dev)[None, None, :]).expand(R, T, w)
    # q/K/V are deliberately computed in float32 above; the sink logits
    # must use the same work dtype or torch.cat promotes to the input dtype
    # and the softmax/einsum then mixes (for fp16/bf16/fp64 input) dtypes.
    sink_v = torch.nan_to_num(sink).to(acc) if sink is not None else None
    dq = torch.zeros_like(q) if need_dq else None
    dsoft = torch.zeros_like(soft) if (need_dsoft and soft is not None) else None
    dsink = torch.zeros_like(sink) if (need_dsink and sink is not None) else None
    r_idx = torch.arange(R, device=dev)[:, None, None]
    tc = _attn_t_chunk(R, M, nh, hd)
    q_heads = [q[:, :, h, :] for h in range(nh)]
    dO_heads = [dO[:, :, h, :] for h in range(nh)]
    kb_heads = [k_blk[:, :, h, :] for h in range(nh)]
    kw_heads = [k_sw[:, :, h, :] for h in range(nh)]
    vb_heads = [v_blk[:, :, h, :] for h in range(nh)]
    vw_heads = [v_sw[:, :, h, :] for h in range(nh)]
    lt_l = last_tok.to(torch.int64)
    try:
        _slab_budget = max(1, int(os.environ.get('CSA_ATTN_GRAD_SLAB_MB', '64'))) << 20
    except (TypeError, ValueError):
        _slab_budget = 64 << 20
    _acc_bytes = int(torch.empty((), dtype=acc).element_size())
    _has_win = int(w) > 0
    _per_head_bytes = _acc_bytes * hd * (
        R * Bm * (int(bool(need_dkb)) + int(bool(need_dvb)))
        + (R * T * (int(bool(need_dksw)) + int(bool(need_dvsw)))
           if _has_win else 0))
    group = max(1, min(int(nh), int(_slab_budget) // max(1, int(_per_head_bytes))))
    dk_blk = torch.empty((R, Bm, nh, hd), device=dev, dtype=dtype) if need_dkb else None
    dv_blk = torch.empty((R, Bm, nh, hd), device=dev, dtype=dtype) if need_dvb else None
    # A zero-width window contributes no lanes and no gradient, but autograd
    # still requires the returned gradient to match the input tensor's shape.
    # `zeros_like` is therefore the exact dgrad for w == 0 (and is also the
    # correct empty gradient for a caller-supplied [R, 0, nh, hd] table).
    # For a real window every row is written by the head loop below, so an
    # uninitialised buffer avoids zero-filling an [R, T, nh, hd] tensor at
    # long-context scale.
    dk_sw = ((torch.empty_like(k_sw) if _has_win else torch.zeros_like(k_sw))
             if need_dksw else None)
    dv_sw = ((torch.empty_like(v_sw) if _has_win else torch.zeros_like(v_sw))
             if need_dvsw else None)
    dk_slab = torch.zeros((group, R * Bm, hd), device=dev, dtype=acc) if need_dkb else None
    dv_slab = torch.zeros((group, R * Bm, hd), device=dev, dtype=acc) if need_dvb else None
    dksw_slab = torch.zeros((group, R * T, hd), device=dev, dtype=acc) if (_has_win and need_dksw) else None
    dvsw_slab = torch.zeros((group, R * T, hd), device=dev, dtype=acc) if (_has_win and need_dvsw) else None
    dk_scratch = torch.zeros((R * Bm, hd), device=dev, dtype=acc) if (need_dkb or need_dvb) else None
    dksw_scratch = torch.zeros((R * T, hd), device=dev, dtype=acc) if (_has_win and (need_dksw or need_dvsw)) else None

    for h0 in range(0, nh, group):
        h1 = min(h0 + group, nh)
        # The final block can be narrower than `group` (nh % group != 0);
        # the slab always has `group` rows, so only the live prefix is viewed.
        _g = h1 - h0
        for _slab in (dk_slab, dv_slab, dksw_slab, dvsw_slab):
            if _slab is not None:
                _slab.zero_()
        for t0 in range(0, T, tc):
            t1 = min(t0 + tc, T)
            src32 = topk_idx[:, t0:t1]
            src64 = src32.to(torch.int64)
            lt_sel = lt_l.gather(1, src64.reshape(R, -1)).view(R, t1 - t0, nb)
            sel_ok = valid[:, t0:t1] & (t_idx[:, t0:t1, :] > lt_sel)
            src_win = win_raw[:, t0:t1, :].clamp(min=0)
            win_ok = win_raw[:, t0:t1, :] >= 0
            lane_ok = torch.cat([sel_ok, win_ok], dim=-1)
            anyrow = (sel_ok.any(dim=-1) | win_ok.any(dim=-1)) if sink_v is None else None
            soft_c = soft[:, t0:t1, :].to(acc) if soft is not None else None
            seg_b_c = None
            if need_dkb or need_dvb:
                kb_keys = r_idx.expand(R, t1 - t0, nb) * Bm + src64
                seg_b_c = _seg_structure(kb_keys.reshape(-1), R * Bm)
                del kb_keys
            seg_w_c = None
            if _has_win and (need_dksw or need_dvsw):
                kw_keys = r_idx.expand(R, t1 - t0, w) * T + src_win
                seg_w_c = _seg_structure(kw_keys.reshape(-1), R * T)
                del kw_keys
            for h in range(h0, h1):
                hi = h - h0
                qh = q_heads[h][:, t0:t1].to(acc)
                dOh = dO_heads[h][:, t0:t1].to(acc)
                Kb = _gather_rows(kb_heads[h], src64).to(acc)
                Kw = _gather_rows(kw_heads[h], src_win).to(acc)
                Vb = _gather_rows(vb_heads[h], src64).to(acc)
                Vw = _gather_rows(vw_heads[h], src_win).to(acc)
                Kl = torch.cat([Kb, Kw], dim=-2)
                Vl = torch.cat([Vb, Vw], dim=-2)
                zA = torch.cat([torch.einsum('rtd,rtmd->rtm', qh, Kb), torch.einsum('rtd,rtmd->rtm', qh, Kw)], dim=-1)
                if scale != 1.0:
                    # same IEEE identity as the forward ref: in-place on the fresh
                    # cat output, same values as the out-of-place form.
                    zA.mul_(scale)
                zA.masked_fill_(~lane_ok, MINL)
                g = torch.einsum('rtd,rtmd->rtm', dOh, Vl)
                gh = g * (anyrow[:, :, None] if anyrow is not None else 1.0)
                P_S = GS = dzS = P_sinkS = P_sinkA = None
                s_ok = None
                if soft_c is not None:
                    # S 路径（soft 先验）与 sink 无关——eager 的直通估计器在
                    # use_sink=False 时同样有 S 路梯度。
                    lam = torch.log(soft_c.clamp_min(1e-12))
                    lam = lam.masked_fill(soft_c <= 0, MINL)
                    # Keep the pre-clamp value: eager's clamp_min(_MINL) has a
                    # zero derivative on saturated lanes (including the invalid
                    # lanes that masked_fill placed exactly at _MINL).  Applying
                    # dzS there would add a uniform-softmax term eager never has.
                    zS_pre = zA.clone()
                    zS_pre[:, :, :nb] = zA[:, :, :nb] + lam
                    s_ok = lane_ok & (zS_pre > MINL)
                    zS = zS_pre.clamp_min(MINL)
                if sink_v is None:
                    P_A = torch.softmax(zA, dim=-1)
                    P_A_f = P_A * (anyrow[:, :, None] if anyrow is not None else 1.0)
                    GA = (P_A * gh).sum(dim=-1)
                    dzA = P_A * (gh - GA[:, :, None])
                    if soft_c is not None:
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
                    if soft_c is not None:
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
                if dzS is not None:
                    dzS = torch.where(s_ok, dzS, torch.zeros_like(dzS))
                dz = dzA if dzS is None else dzA + dzS
                if dq is not None:
                    _dq = torch.einsum('rtm,rtmd->rtd', dz, Kl)
                    if scale != 1.0:
                        _dq.mul_(scale)
                    dq[:, t0:t1, h, :] += _dq.to(dtype)
                if dsoft is not None and dzS is not None:
                    dl = dzS[:, :, :nb]
                    sf = soft_c
                    # log(clamp_min(soft, 1e-12)) has a zero derivative at
                    # the clamp boundary, so the strict interior is the correct
                    # support (>= would emit dl/1e-12 at exactly 1e-12).
                    dsoft[:, t0:t1, :] += torch.where(sf > 1e-12, dl / sf.clamp_min(1e-12), torch.zeros_like(dl)).to(dtype)
                if need_dkb or (need_dksw and _has_win):
                    dkl = torch.einsum('rtm,rtd->rtmd', dz, qh)
                    if scale != 1.0:
                        dkl.mul_(scale)
                    if need_dkb:
                        dk_scratch.zero_()
                        _seg_apply(seg_b_c, dkl[:, :, :nb].reshape(-1, hd), dk_scratch)
                        dk_slab[hi].add_(dk_scratch)
                    if need_dksw and _has_win:
                        dksw_scratch.zero_()
                        _seg_apply(seg_w_c, dkl[:, :, nb:].reshape(-1, hd), dksw_scratch)
                        dksw_slab[hi].add_(dksw_scratch)
                    del dkl
                if need_dvb or (need_dvsw and _has_win):
                    dvl = torch.einsum('rtm,rtd->rtmd', P_A_f, dOh)
                    if need_dvb:
                        dk_scratch.zero_()
                        _seg_apply(seg_b_c, dvl[:, :, :nb].reshape(-1, hd), dk_scratch)
                        dv_slab[hi].add_(dk_scratch)
                    if need_dvsw and _has_win:
                        dksw_scratch.zero_()
                        _seg_apply(seg_w_c, dvl[:, :, nb:].reshape(-1, hd), dksw_scratch)
                        dvsw_slab[hi].add_(dksw_scratch)
                    del dvl
                del (qh, dOh, Kb, Kw, Vb, Vw, Kl, Vl, zA, g, gh, P_S, GS, dzS,
                     P_sinkS, P_sinkA, P_A, P_A_f, GA, dzA, dz)
            del (src32, src64, lt_sel, sel_ok, src_win, win_ok, lane_ok, anyrow,
                 soft_c, seg_b_c, seg_w_c)
        if dk_blk is not None:
            dk_blk[:, :, h0:h1, :] = dk_slab[:_g].view(_g, R, Bm, hd).permute(1, 2, 0, 3).to(dtype)
        if dv_blk is not None:
            dv_blk[:, :, h0:h1, :] = dv_slab[:_g].view(_g, R, Bm, hd).permute(1, 2, 0, 3).to(dtype)
        if dk_sw is not None and _has_win:
            dk_sw[:, :, h0:h1, :] = dksw_slab[:_g].view(_g, R, T, hd).permute(1, 2, 0, 3).to(dtype)
        if dv_sw is not None and _has_win:
            dv_sw[:, :, h0:h1, :] = dvsw_slab[:_g].view(_g, R, T, hd).permute(1, 2, 0, 3).to(dtype)
    if dsink is not None:
        # eager sanitises the sink before its softmax; nan_to_num's backward
        # is zero at the replaced elements, so the kernel must not route a
        # gradient back into a non-finite sink entry.
        dsink = torch.where(torch.isfinite(sink), dsink, torch.zeros_like(dsink))
    return (dq, dk_blk, dv_blk, dk_sw, dv_sw, dsoft, dsink)


# ============================================================================
# 算子 3 的可微封装 + 公共入口
# ============================================================================

def _attn_impl_for_shape(impl, q, w):
    if impl != 'triton':
        return impl
    if q.dtype == torch.float64:
        # The Triton attention kernels cast q/k/v to fp32; the reference twin
        # keeps fp64.  Route fp64 to the reference instead of narrowing it
        # (which can overflow a finite value or change a near-tie result).
        _warn_shape_fallback(
            'attention', 'fp64 inputs require the torch-reference backend '
                         '(the Triton kernels compute in fp32)')
        return 'ref'
    hd = int(q.shape[-1])
    if int(w) < 1 or hd < 16 or (hd & (hd - 1)) != 0:
        _warn_shape_fallback('attention', f'w={int(w)}, hd={hd} require w >= 1 and a power-of-two hd >= 16')
        return 'ref'
    return impl


class FusedSparseAttention(torch.autograd.Function):
    """稀疏注意力的可微封装：前向 = A 路径（与 eager 的直通估计器前向值
    严格一致）；反向按 _attn_bwd_ref 的 A+S 双路分解（triton 后端为同公式
    的四内核版：统计 / dq / dk-dv / dsoft）。"""

    @staticmethod
    def forward(ctx, impl, q, k_blk, v_blk, k_sw, v_sw, topk_idx, valid, last_tok, soft, sink, w, scale):
        if impl == 'triton' and any(
                isinstance(t, torch.Tensor) and t.dtype == torch.float64
                for t in (q, k_blk, v_blk, k_sw, v_sw)):
            _warn_shape_fallback(
                'attention', 'fp64 inputs require the torch-reference backend '
                             '(the Triton kernels compute in fp32)')
            impl = 'ref'
        else:
            impl = _attn_impl_for_shape(impl, q, w)
        # Accept the eager API's block-only form (w=0, k_sw=v_sw=None) and
        # refuse a non-zero window without its K/V tables.  Normalising once
        # here also makes the saved tensors and the returned dgrad shapes
        # agree with the caller's zero-width window.
        k_sw, v_sw = _attn_window_pair(k_sw, v_sw, q, w)
        # HCA's read-everything arm passes the cached stride-0 arange table:
        # every row is exactly arange(nb), so the triton kernels derive the
        # block id from the lane index.
        idx_arange = bool(getattr(topk_idx, '_idx_arange', False))
        if impl == 'triton':
            out = _attn_fwd_triton(q, k_blk, v_blk, k_sw, v_sw, topk_idx, valid, last_tok, soft, sink, w, scale, idx_arange)
        else:
            out = _attn_fwd_ref(q, k_blk, v_blk, k_sw, v_sw, topk_idx, valid, last_tok, soft, sink, w, scale)
        ctx.save_for_backward(q, k_blk, v_blk, k_sw, v_sw, topk_idx, valid, last_tok, soft, sink)
        # 反向必须跑与前向同一个后端：前向拿到的是调用方解析好的 impl
        # （gathered_attention_kernel 经 _op_impl('ATTN')）。
        ctx.impl = impl
        ctx.idx_arange = bool(idx_arange and impl == 'triton')
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
                                     ctx.w, ctx.scale, dO, ctx.needs_input_grad, ctx.idx_arange)
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
        r = (rb // Bm).to(tl.int64)
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
        tl.store(comp_ptr + (r * Bm + b).to(tl.int64) * C + cs, acc.to(comp_ptr.dtype.element_ty), mask=cm)

    ns.pool_fwd = pool_fwd

    # ---------- 算子 2：索引打分 ----------
    # T0/TF：打分按 query 行分块启动；T0 是本块的全局行
    # 偏移，TF 是完整序列长（qI/w 的行 stride），scores/输出按块内 T 寻址。
    # T0=0、TF=T 时寻址与不分块形式完全一致（同一批全局行、同一批值）。
    # HDP：点积泳道宽（>= 16 的 2 幂）；hd < HDP 时 d 泳道零填充——零乘积
    # 向点积累加恰好贡献 +0.0，分数逐位不变；hd 无需是 2 的幂或 >= 16。
    @triton.jit
    def idx_score(qI_ptr, kI_ptr, w_ptr, scores_ptr, T0, TF, T, Bm, ISCALE,
                  N_IH: tl.constexpr, HD: tl.constexpr, HDP: tl.constexpr, BT: tl.constexpr, BB: tl.constexpr,
                  DOT: tl.constexpr):
        pid = tl.program_id(0)
        n_t = tl.cdiv(T, BT)
        pid_t = pid % n_t
        r = (pid // n_t).to(tl.int64)
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
    # where/nan 守卫清零——两种情形都与全宽 extract 归约同值。
    @triton.jit
    def idx_topk(scores_ptr, last_tok_ptr, idx_ptr, valid_ptr, soft_ptr, T0, TF, T, Bm,
                 K: tl.constexpr, BB: tl.constexpr, HAS_SOFT: tl.constexpr):
        pid = tl.program_id(0)
        r = (pid // T).to(tl.int64)
        t = pid % T
        b = tl.arange(0, BB).to(tl.int64)
        bm = b < Bm
        rt = r * T + t
        tg = (T0 + t).to(tl.int64)
        rt_out = r * TF + tg
        s = tl.load(scores_ptr + rt * Bm + b, mask=bm, other=float('-inf')).to(tl.float32)
        lt = tl.load(last_tok_ptr + r * Bm + b, mask=bm, other=TF - 1).to(tl.int64)
        causal = (tg > lt) & bm
        finite = (s - s) == 0.0
        if HAS_SOFT:
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
            tl.store(idx_ptr + rt_out * K + j, use.to(tl.int32))
            tl.store(valid_ptr + rt_out * K + j, has.to(tl.int8))
            if HAS_SOFT:
                sv = tl.exp(m - lse)
                sv = tl.where(has, sv, 0.0)
                sv = tl.where(sv == sv, sv, 0.0)
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


def _attn_fwd_triton(q, k_blk, v_blk, k_sw, v_sw, topk_idx, valid, last_tok, soft, sink, w, scale, idx_arange=False):
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
    # The arange table is a stride-0 expand view; the kernels read the lane
    # index.
    topk_idx = topk_idx if idx_arange else topk_idx.contiguous()
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
                     IDX_ARANGE=bool(idx_arange),
                     DOT=_dot_prec(), num_warps=4)
    return out


def _attn_bwd_triton(q, k_blk, v_blk, k_sw, v_sw, topk_idx, valid, last_tok, soft, sink, w, scale, dO, needs, idx_arange=False):
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
    topk_idx = topk_idx if idx_arange else topk_idx.contiguous()
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
        IDX_ARANGE=bool(idx_arange),
        DOT=_dot_prec(), num_warps=4)
    dq = None
    if need_dq:
        dq = torch.empty_like(q)
        K.attn_bwd_dq[(R * nh * triton.cdiv(T, BT),)](
            q, k_blk, v_blk, k_sw, v_sw, topk_idx, valid_c, last_tok, soft_c, sink_arg, dO,
            mA, ZA, mS, ZS, GA, GS, dq, T, Bm, int(w), float(scale),
            NB=nb, NH=nh, HD=hd, BT=BT, BW=BW,
            HAS_VALID=valid is not None, HAS_SOFT=soft is not None, HAS_SINK=sink is not None,
            IDX_ARANGE=bool(idx_arange),
            DOT=_dot_prec(), num_warps=4)
    dk_blk = dv_blk = dk_sw = dv_sw = dsoft = None
    if need_dkb or need_dvb:
        # attn_bwd_dkdv_blocks 对网格内每个 (r,b,h) 无条件 store（NEED_DK/
        # NEED_DV 与这里的分配条件一致）。
        dk_blk = torch.empty((R, Bm, nh, hd), device=dev, dtype=dtype) if need_dkb else q
        dv_blk = torch.empty((R, Bm, nh, hd), device=dev, dtype=dtype) if need_dvb else q
        if idx_arange:
            # Every row is arange(nb): the stable block-major order is
            # implicit (lane = t*nb + b).  The kernel derives the segment
            # bounds from b and T directly.
            order = offs = q
        else:
            flat = topk_idx.reshape(R, T * nb)
            # 各行互相独立，稳定序逐位不变；order 保留 int64：lane 值域是
            # [0, T·nb)，T·nb ≥ 2³¹ 的超大配置下窄化到 int32 会翻负、内核
            # 再按 lane//NB 寻址即越界。
            order = torch.argsort(flat, dim=1, stable=True)
            # 计数是整数（块 id 非负且 < Bm），scatter_add_ 与逐行 bincount
            # 给出同一组精确整数。lane 值域 [0, T·NB) 只在确实越过 int32
            # 时才用 int64；常规规模下 order/计数/偏移表全部用 int32。
            _narrow = int(T) * int(nb) < (1 << 31)
            _idt = torch.int32 if _narrow else torch.int64
            if _narrow:
                order = order.to(torch.int32)
            flat_l = flat.to(_idt)
            cnt = torch.zeros((R, Bm), device=dev, dtype=_idt)
            cnt.scatter_add_(1, flat_l, torch.ones_like(flat_l))
            del flat_l
            offs = torch.zeros((R, Bm + 1), device=dev, dtype=_idt)
            offs[:, 1:] = cnt.cumsum(dim=1)
            del flat, cnt
        K.attn_bwd_dkdv_blocks[(R * Bm * nh,)](
            q, k_blk, v_blk, topk_idx, valid_c, last_tok, soft_c, sink_arg, dO,
            mA, ZA, mS, ZS, GA, GS,
            dk_blk, dv_blk, order, offs, T, Bm, nb, float(scale),
            NH=nh, HD=hd, HAS_VALID=valid is not None, HAS_SOFT=soft is not None,
            HAS_SINK=sink is not None,
            NEED_DK=need_dkb, NEED_DV=need_dvb, IDX_ARANGE=bool(idx_arange),
            num_warps=1)
    if need_dsoft:
        # dsoft 是跨头求和（ref 反向在头循环里 +=），(r,b,h) 网格的
        # dkdv_blocks 表达不了——由独立内核完成：每个 (t,j) 泳道恰好一个
        # 写者、头按固定顺序寄存器累加，每个 (t,j) 都被无条件 store。
        dsoft = torch.empty((R, T, nb), device=dev, dtype=dtype)
        K.attn_bwd_dsoft[(R * triton.cdiv(T, BT),)](
            q, k_blk, v_blk, topk_idx, valid_c, last_tok, soft_c, dO,
            mS, ZS, GS, dsoft, T, Bm, float(scale),
            NB=nb, NH=nh, HD=hd, BT=BT, HAS_VALID=valid is not None,
            IDX_ARANGE=bool(idx_arange), num_warps=4)
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
        finite = torch.isfinite(sink)
        dsink = torch.where(finite, acc.to(dtype), torch.zeros_like(acc, dtype=dtype))
    return (dq, dk_blk, dv_blk, dk_sw, dv_sw, dsoft, dsink)


def _attach_attn_kernels(ns):
    import triton
    import triton.language as tl

    @triton.jit
    def attn_fwd(q_ptr, kb_ptr, vb_ptr, ksw_ptr, vsw_ptr, idx_ptr, valid_ptr, lt_ptr, soft_ptr, sink_ptr, out_ptr,
                 T, Bm, W, SCALE,
                 NB, NH: tl.constexpr, HD: tl.constexpr, BT: tl.constexpr, BW: tl.constexpr,
                 HAS_VALID: tl.constexpr, HAS_SINK: tl.constexpr, IDX_ARANGE: tl.constexpr, DOT: tl.constexpr):
        pid = tl.program_id(0)
        n_t = tl.cdiv(T, BT)
        pid_t = pid % n_t
        rh = pid // n_t
        h = rh % NH
        r = (rh // NH).to(tl.int64)
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
        for j in range(NB):
            if IDX_ARANGE:
                # Each row is exactly arange(NB): the block id is the
                # lane index, so no dense index table is read.
                src = tl.zeros((BT,), tl.int64) + j
            else:
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
                       NB, NH: tl.constexpr, HD: tl.constexpr, BT: tl.constexpr, BW: tl.constexpr,
                       HAS_VALID: tl.constexpr, HAS_SOFT: tl.constexpr, HAS_SINK: tl.constexpr,
                       IDX_ARANGE: tl.constexpr, DOT: tl.constexpr):
        pid = tl.program_id(0)
        n_t = tl.cdiv(T, BT)
        pid_t = pid % n_t
        rh = pid // n_t
        h = rh % NH
        r = (rh // NH).to(tl.int64)
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
        for j in range(NB):
            if IDX_ARANGE:
                # Each row is exactly arange(NB): the block id is the
                # lane index, so no dense index table is read.
                src = tl.zeros((BT,), tl.int64) + j
            else:
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
                    NB, NH: tl.constexpr, HD: tl.constexpr, BT: tl.constexpr, BW: tl.constexpr,
                    HAS_VALID: tl.constexpr, HAS_SOFT: tl.constexpr, HAS_SINK: tl.constexpr,
                    IDX_ARANGE: tl.constexpr, DOT: tl.constexpr):
        pid = tl.program_id(0)
        n_t = tl.cdiv(T, BT)
        pid_t = pid % n_t
        rh = pid // n_t
        h = rh % NH
        r = (rh // NH).to(tl.int64)
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
        for j in range(NB):
            if IDX_ARANGE:
                # Each row is exactly arange(NB): the block id is the
                # lane index, so no dense index table is read.
                src = tl.zeros((BT,), tl.int64) + j
            else:
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
                zS_pre = zj + lam
                zS = tl.maximum(zS_pre, -3.402823466e38)
                zS = tl.where(ok, zS, float('-inf'))
                eS = tl.exp(zS - mS)
                # clamp_min has a zero derivative on saturated lanes; drop the
                # S-path term there (matches the eager autograd graph).
                dzS = tl.where(ok & (zS_pre > -3.402823466e38),
                               eS / ZSs * (g - GS), 0.0)
                dz += dzS
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
                             NEED_DK: tl.constexpr, NEED_DV: tl.constexpr, IDX_ARANGE: tl.constexpr):
        pid = tl.program_id(0)
        h = pid % NH
        rb = pid // NH
        b = rb % Bm
        r = (rb // Bm).to(tl.int64)
        if IDX_ARANGE:
            # Every token selects all blocks: block b's lanes are exactly one
            # per token, so the segment is [b*T, (b+1)*T) and lane
            # k -> (t = k - b*T, j = b).  No order/offset tables are read.
            seg0 = b.to(tl.int64) * T
            seg1 = seg0 + T
        else:
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
            if IDX_ARANGE:
                tt = (k - seg0).to(tl.int64)
                jj = b
            else:
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
                    zS_pre = z + lam
                    zS = tl.maximum(zS_pre, -3.402823466e38)
                    PS = tl.exp(zS - mS) / ZSs
                    dzS = PS * (g - GS)
                    dzS = tl.where(zS_pre > -3.402823466e38, dzS, 0.0)
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
                       NB, NH: tl.constexpr, HD: tl.constexpr, BT: tl.constexpr,
                       HAS_VALID: tl.constexpr, IDX_ARANGE: tl.constexpr):
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
        r = (pid // n_t).to(tl.int64)
        t0 = pid_t * BT
        t = t0 + tl.arange(0, BT)
        tm = t < T
        ti = t.to(tl.int64)
        d = tl.arange(0, HD)
        for j in range(NB):
            if IDX_ARANGE:
                # Each row is exactly arange(NB): the block id is the
                # lane index, so no dense index table is read.
                src = tl.zeros((BT,), tl.int64) + j
            else:
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
                zS_pre = z + lam
                zS = tl.maximum(zS_pre, -3.402823466e38)
                zS = tl.where(ok, zS, float('-inf'))
                PS = tl.exp(zS - mS) / ZSs
                g = tl.sum(dO_v * v_v, 1)
                dzS = PS * (g - GS)
                # Strict interior for log(clamp_min(soft, 1e-12)); its
                # derivative is zero at the boundary, and the outer
                # clamp_min(_MINL) also kills saturated lanes.
                ds_h = tl.where(ok & (zS_pre > -3.402823466e38) & (sf > 1e-12),
                                dzS / tl.maximum(sf, 1e-12), 0.0)
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
        r = (rp // T).to(tl.int64)
        p = (rp % T).to(tl.int64)
        d = tl.arange(0, HD)
        k_v = tl.load(ksw_ptr + ((r * T + p) * NH + h) * HD + d).to(tl.float32)
        v_v = tl.load(vsw_ptr + ((r * T + p) * NH + h) * HD + d).to(tl.float32)
        dk_acc = tl.zeros((HD,), tl.float32)
        dv_acc = tl.zeros((HD,), tl.float32)
        t_hi = p + W
        if t_hi > T:
            t_hi = T
        # tt 沿 [p, p+W) 连续——按 [BW, HD] 分片载入：
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
    if any(t is not None and t.dtype == torch.float64
           for t in (Xa, Xb, Za, Zb)):
        # The Triton pool loads every lane in fp32, while the shared backward
        # is written in fp64 for fp64 inputs.  Returning the fp64 reference
        # keeps forward/backward the same function and avoids fp32 overflow.
        _warn_shape_fallback(
            'pool', 'fp64 inputs require the torch-reference backend '
                    '(the Triton kernel computes in fp32)')
        return _pool_fwd_ref(Xa, Xb, Za, Zb, geom, B_pos_a, B_pos_b)
    K = _kernels()
    triton = K.triton
    R, T, C = Xa.shape
    Bm = geom.Bm
    L, W = geom.max_len, geom.W_b
    has_b = Xb is not None and W > 0
    if geom.starts.dtype != torch.int32:
        # The Triton kernel addresses lanes with int32 offsets.  A geometry
        # wider than that cannot be represented by this kernel; fail closed
        # instead of letting a truncated pointer target another block.  The
        # torch-reference/eager paths accept the same int64 geometry.
        raise RuntimeError(
            'Triton pool kernel requires int32 geometry; this sequence '
            'exceeds the int32 lane range — use the eager/reference path')
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


_IDX_SCORES_CAP_BYTES = 128 << 20


def _indexer_topk_chunked(scores, last_tok, t0, T, Bm, k_out, need_soft, out_dtype):
    """Exact stable top-k for a score tile when Bm exceeds the lane limit.

    `scores` is the already-materialised [R, tc, Bm] tile from idx_score.  It
    is walked in column chunks; each chunk contributes its own stable top-k,
    and the merged candidate table is stable-sorted again.  Candidate order is
    chunk-major / column-minor, i.e. exactly ascending global block id, so the
    stable descending sort reproduces the full-width tie rule (earliest
    block wins).
    """
    R, tc = int(scores.shape[0]), int(scores.shape[1])
    dev = scores.device
    k_out = max(0, min(int(k_out), int(Bm)))
    # Chunk width: keep one candidate tile around a few megabytes.
    bc = max(64, min(int(Bm), (4 << 20) // max(1, R * tc * 4)))
    b = torch.arange(int(Bm), device=dev, dtype=torch.long)
    lt = last_tok[:, :Bm].to(torch.int64)
    tp = torch.arange(int(t0), int(t0) + tc, device=dev, dtype=torch.int64)[None, :, None]
    cand_v, cand_i = [], []
    m_run = None
    z_run = None
    for b0 in range(0, int(Bm), bc):
        b1 = min(b0 + bc, int(Bm))
        sc = scores[:, :, b0:b1]
        causal = tp > lt[:, None, b0:b1]
        masked = torch.where(causal, sc, float('-inf'))
        if need_soft:
            cm = masked.max(dim=-1).values
            if m_run is None:
                m_run = cm
                cm_safe = torch.where(cm == float('-inf'), torch.zeros_like(cm), cm)
                z_run = torch.exp(masked - cm_safe[..., None]).sum(dim=-1)
            else:
                mn = torch.maximum(m_run, cm)
                mn_safe = torch.where(mn == float('-inf'), torch.zeros_like(mn), mn)
                z_run = (z_run * torch.exp(m_run - mn_safe)
                         + torch.exp(masked - mn_safe[..., None]).sum(dim=-1))
                m_run = mn
        work = torch.where(causal & torch.isfinite(sc), sc, float('-inf'))
        val, ind = _stable_topk_desc(work, min(k_out, b1 - b0))
        cand_v.append(val)
        cand_i.append(ind + b0)
    if not cand_v:
        idx = torch.zeros((R, tc, k_out), device=dev, dtype=torch.int32)
        valid = torch.zeros((R, tc, k_out), device=dev, dtype=torch.bool)
        soft = torch.zeros((R, tc, k_out), device=dev, dtype=out_dtype) if need_soft else None
        return idx, valid, soft
    cv = torch.cat(cand_v, dim=-1)
    ci = torch.cat(cand_i, dim=-1)
    # The concatenated per-chunk candidates are only locally sorted, so a
    # final stable merge is always required (including k_out == Bm, where the
    # full selection must come back in descending order).
    top_val, top_pos = _stable_topk_desc(cv, k_out)
    top_idx = ci.gather(-1, top_pos)
    valid = top_val > float('-inf')
    idx = torch.where(valid, top_idx, torch.zeros((), device=dev, dtype=torch.long)).to(torch.int32)
    soft = None
    if need_soft:
        lse = m_run + torch.log(z_run)
        sv = torch.exp(top_val - lse[..., None])
        sv = torch.where(valid & torch.isfinite(lse)[..., None], sv, torch.zeros_like(sv))
        soft = torch.where(sv == sv, sv, torch.zeros_like(sv)).to(out_dtype)
    return idx, valid, soft


def _indexer_fwd_triton(qI, kI, w_idx, last_tok, topk, need_soft=True):
    K = _kernels()
    triton = K.triton
    R, T, nIH, hd = qI.shape
    Bm = kI.shape[1]
    qI = qI.contiguous()
    kI = kI.contiguous()
    w_idx = w_idx.contiguous()
    last_tok = last_tok.contiguous()
    # 打分泳道宽补到 >= 16 的 2 幂（零填充泳道对点积恰好贡献 +0.0，分数
    # 逐位不变）；hd 无需是 2 的幂或 >= 16。
    HDP = max(_next_pow2(hd), 16)
    k_out = max(0, min(int(topk), Bm))
    BB2 = _next_pow2(Bm)
    idx = torch.empty((R, T, k_out), device=qI.device, dtype=torch.int32)
    valid_i8 = torch.empty((R, T, k_out), device=qI.device, dtype=torch.int8)
    # The soft prior is training-only (its straight-through path is dead
    # without gradients), so eval does not build it.  The dummy pointer is
    # never dereferenced when HAS_SOFT is false.
    soft = torch.empty((R, T, k_out), device=qI.device, dtype=qI.dtype) if need_soft else None
    soft_arg = soft if soft is not None else idx
    BT, BB = 64, 64
    T_c = max(1, min(T, _IDX_SCORES_CAP_BYTES // max(R * Bm * 4, 1)))
    scores = torch.empty((R, T_c, Bm), device=qI.device, dtype=torch.float32)
    for t0 in range(0, T, T_c):
        tc = min(T_c, T - t0)
        K.idx_score[(R * triton.cdiv(tc, BT),)](qI, kI, w_idx, scores, t0, T, tc, Bm, hd ** -0.5,
                                                N_IH=nIH, HD=hd, HDP=HDP, BT=BT, BB=BB,
                                                DOT=_dot_prec(), num_warps=4)
        if Bm <= 8192:
            K.idx_topk[(R * tc,)](scores, last_tok, idx, valid_i8, soft_arg, t0, T, tc, Bm,
                                  K=k_out, BB=BB2, HAS_SOFT=bool(need_soft),
                                  num_warps=8 if BB2 >= 1024 else 4)
        else:
            # idx_score stores each tile compactly with row stride ``tc``
            # (the same layout idx_topk consumes).  The final chunk may be
            # shorter than the allocated [R, T_c, Bm] tile, so slicing
            # ``scores[:, :tc]`` would keep the allocation's T_c row stride
            # and read the wrong rows for r >= 2; view the filled compact
            # prefix instead.  Passing only the filled rows also keeps the
            # helper's result assignable into idx[:, t0:t0+tc].
            _tile = (scores if tc == T_c
                     else scores.reshape(-1, Bm)[:R * tc].view(R, tc, Bm))
            _i, _v, _s = _indexer_topk_chunked(
                _tile, last_tok, t0, T, Bm, k_out, bool(need_soft), qI.dtype)
            idx[:, t0:t0 + tc] = _i
            valid_i8[:, t0:t0 + tc] = _v.to(torch.int8)
            if soft is not None:
                soft[:, t0:t0 + tc] = _s
    return idx, valid_i8.view(torch.bool), soft
