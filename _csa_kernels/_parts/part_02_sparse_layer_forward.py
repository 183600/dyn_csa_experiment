

# ============================================================================
# HybridAttention 的稀疏内核路径（exp_lib.forward 的批量分发放到这里，
# eager 路径零改动）
# ============================================================================

def sparse_layer_forward(attn, x, pre_all, gate_sigs, pre_blocks, impl=None):
    """csa/hca 层的批量化内核前向（对应 eager 的 forward 稀疏分支 + _single）。

    输入：attn = HybridAttention 实例；x [B, T, d]；pre_all/gate_sigs/pre_blocks
    = exp_lib.HybridAttention.forward 已算好的批量门控信号 / 块几何。
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
    # 逐算子解析后端：显式 impl= 强制全部三算子；
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
        if pre_blocks is None and int(T) <= 1:
            # A length-1 sequence has no consecutive-pair similarity, so the
            # eager forward hoist yields no gate row and leaves the geometry
            # to the segmenter (the single token is one whole block).  Mirror
            # that fallback here: the kernel path must not fail on the same
            # input the eager path accepts.
            import exp_lib as _L
            _bid, _nb, _hon = _L.blocks_from_cuts(
                int(T), [], cfg.min_block, cfg.max_block, dev,
                return_count=True, return_honoured=True)
            pre_blocks = [(_bid, _nb, _hon) for _ in range(int(B))]
            if gate_sigs is None:
                # The gate vectors are zero-width at T=1; keep the shape the
                # per-row path would feed to the segmentation statistics.
                _gate = x.new_empty((int(B), 0))
                _gate_bool = torch.empty((int(B), 0), dtype=torch.bool, device=dev)
                gate_sigs = [(_gate[_b], _gate_bool[_b]) for _b in range(int(B))]
        if pre_blocks is None or gate_sigs is None:
            raise RuntimeError('cosine_learnable 层需要 forward 预计算的门控信号与块几何')
        bids = [pb[0] for pb in pre_blocks]
        nblks = [pb[1] for pb in pre_blocks]
        hons = [pb[2] for pb in pre_blocks]
        # Write the rows directly into the int32 device table: the exact same
        # block ids the int64 stack-then-cast produced.
        bid_full = torch.empty((B, T), dtype=torch.int32, device=dev)
        for _b, _bd in enumerate(bids):
            bid_full[_b].copy_(_bd)
        counts_rows = [getattr(bd, '_pool_cpu', None) for bd in bids]
        if any(c is None for c in counts_rows):
            raise RuntimeError('块几何缺少 _pool_cpu counts——分段器出口被绕过')
        counts_rows = [c[0] for c in counts_rows]
    elif pre_blocks is not None:
        bids = [pb[0] for pb in pre_blocks]
        nblks = [pb[1] for pb in pre_blocks]
        hons = None
        bid_full = torch.empty((B, T), dtype=torch.int32, device=dev)
        for _b, _bd in enumerate(bids):
            bid_full[_b].copy_(_bd)
        counts_rows = [getattr(bd, '_pool_cpu', None) for bd in bids]
        if any(c is None for c in counts_rows):
            raise RuntimeError('块几何缺少 _pool_cpu counts——分段器出口被绕过')
        counts_rows = [c[0] for c in counts_rows]
    else:
        base = _exp_blocks_fixed(T, cfg.block_size, dev)
        counts = getattr(base, '_pool_cpu', None)
        if counts is None:
            raise RuntimeError('blocks_fixed 未携带 _pool_cpu——分段器出口被绕过')
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
            # Eager `_single` also feeds the forced/natural cut counters from
            # the same boolean cut row; the kernel path used to omit them.
            if gate_sigs is not None:
                attn._stats.add_segmentation(
                    bid_full[_b], gate_sigs[_b][1], cfg.max_block)
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
    # Same value as normalizing the zero table (0/eps == 0) for the
    # zero-content ablation.
    core_kv = (torch.zeros_like(comp) if cfg.content_mode == 'zero'
               else F.normalize(comp, dim=-1))
    kv_local = F.normalize(Ca, dim=-1)
    Bm = geom.Bm
    kv = core_kv.shape[-1]
    _kvh = attn.W_kvhead(torch.cat([core_kv.reshape(B * Bm, kv), kv_local.reshape(B * T, kv)], 0))
    _rows = B * Bm + B * T
    _nhq = attn.nh * attn.hd
    # One row-wise normalize over the fused K columns: each row's norm/divide
    # is independent, so the values are identical to the two per-half
    # normalizations (the same form exp_lib._single uses).  V needs no
    # normalization.
    _k_all = F.normalize(_kvh[:, :_nhq].view(_rows, attn.nh, attn.hd), dim=-1)
    _v_all = _kvh[:, _nhq:].view(_rows, attn.nh, attn.hd)
    k_blk = _k_all[:B * Bm].view(B, Bm, attn.nh, attn.hd)
    k_sw = _k_all[B * Bm:].view(B, T, attn.nh, attn.hd)
    v_blk = _v_all[:B * Bm].view(B, Bm, attn.nh, attn.hd)
    v_sw = _v_all[B * Bm:].view(B, T, attn.nh, attn.hd)
    del _kvh, _k_all, _v_all
    if cfg.kind == 'hca':
        # Always the stride-0 expand view: the ref backend reads it directly,
        # and the triton attention kernels derive the block id from the lane
        # index.
        topk_idx = _full_arange_cached(B, T, Bm, dev, contiguous=False)
        valid = None
        soft = None
    else:
        nIH = cfg.n_index_heads
        hd = cfg.c_index // nIH
        qI = pre_all['qI'].view(B, T, nIH, hd)
        w_idx = pre_all['w_idx']
        kI = (comp.reshape(B * Bm, kv) @ attn.W_DK).view(B, Bm, nIH, hd)
        topk_idx, valid, soft = lightning_indexer_kernel(qI, kI, w_idx, last_tok, cfg.index_topk, impl=impl_idx, need_soft=torch.is_grad_enabled())
    sink = attn.sink if cfg.use_sink else None
    out = gathered_attention_kernel(q, k_blk, v_blk, topk_idx, last_tok, k_sw, v_sw,
                                    cfg.sliding_window, 1.0,
                                    soft=soft, sink_logits=sink, sel_valid=valid, impl=impl_attn)
    outs = attn.W_o(out.reshape(B, T, attn.nh * attn.hd))
    attn.last_gate_mean = _gate_mean_batched(attn, cfg, T, B, gate_sigs, hons, nblks)
    # REINFORCE boundary policy: the batched/kernel path used to leave the
    # score-function log-probability unset, silently dropping the ONLY
    # boundary-training term for csa_dynamic_task.  Sum the per-row values
    # (the row loop overwrites them, which is why B=1 hid the loss).
    if gate_sigs is not None and gate_sigs and len(gate_sigs[0]) > 2:
        attn.last_boundary_logp = torch.stack([_gs[2] for _gs in gate_sigs]).sum()
        attn.last_boundary_decisions = int(
            sum(int(_gs[1].numel()) for _gs in gate_sigs))
    else:
        attn.last_boundary_logp = None
        attn.last_boundary_decisions = 0
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
    import exp_lib as L
    nblk_t = L._h2d_async([float(nb - 1) for nb in nblks], gate_stack.dtype, gate_stack.device)
    shs = (gate_stack * hon_stack).sum(dim=1)
    per_row = (nblk_t + shs - shs.detach()) / T
    gm = per_row.sum() / B
    return gm


def _exp_blocks_fixed(n, block_size, device):
    import exp_lib as L
    return L.blocks_fixed(n, block_size, device)


# ============================================================================
# G0 自检：算子级 kernel/eager 数值对照 + 模型级前反向对照
# （驱动入口：python explib.m0.py kernel）
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


def _rel(a, b, pad_mask=None, floor_frac=1e-4):
    """稳健相对误差：分母 = |b| + floor_frac·max|b|，再取逐元素最大值
    （pad 位置可屏蔽）。floor_frac 必须按被比较数据的浮点精度选取：
    纯舍入噪声 atol ≈ k·eps·max|b|，度量在近零条目上放大 k·eps/floor_frac
    倍；floor_frac 远小于 eps 会让噪声淹没一切真实差异（fp32 eps≈1.2e-7，
    1e-4 的下限把 1-ULP 噪声放大到 ~1e-3 量级）。"""
    af = a.detach().to(torch.float64)
    bf = b.detach().to(torch.float64)
    if pad_mask is not None:
        af = af[pad_mask]
        bf = bf[pad_mask]
    if af.numel() == 0:
        return 0.0
    denom = bf.abs() + floor_frac * float(bf.abs().max())
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
    device = torch.device('cpu') if device is None else torch.device(device)
    gen = torch.Generator().manual_seed(seed + 17)
    R, T = 3, 96
    min_block, max_block, overlap = 4, 8, 8
    kv, c_index, nIH = 32, 32, 4
    hd = c_index // nIH
    nh, d_head, d_model = 4, 16, 48
    topk, win = 8, 16
    results = {}
    _ffloor = 1e-2 if dtype == torch.float32 else 1.0

    def mrel(x, y, pad=None):
        return _rel(x, y, pad, floor_frac=_ffloor)

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
        results[f'pool.{nm}'] = mrel(ker[i], ref[i])
    if verbose:
        print(f'[selfcheck:{impl}:{str(dtype)}] pool  ' + '  '.join(f'{k.split(".")[-1]}={v:.2e}' for k, v in results.items() if k.startswith('pool.')))
    # Multi-chunk check: shrinking _POOL_CHUNK_BYTES exercises the b0>0
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
        results[f'pool_mc.{nm}'] = mrel(ker2[i], ref[i])
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
    results['idxer.soft'] = mrel(ker_i[1] * valid, ref_i[1] * valid)
    for i, nm in enumerate(['gH', 'gWDQ', 'gWDK', 'gWw']):
        results[f'idxer.{nm}'] = mrel(ker_i[2 + i], ref_i[2 + i])
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
        results[f'attn.{nm}'] = mrel(ker_a[i], ref_a[i])
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
            results[f'attn_{cfg_tag}.{nm}'] = mrel(ker_d[i], ref_d[i])
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
    else:
        device = torch.device(device)
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
            worst = max(worst, _rel(a, b, floor_frac=1e-2))
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
    """稀疏注意力算子的微基准。

    计数口径：前向 = 4·R·T·(topk+win)·nh·hd（QK^T 与 PV 各 2·hd/泳道），
    反向按 2.5× 前向估计（统计重算 + dq + dk/dv 三遍）。有效速率 =
    (前向 + 2.5×前向) / 实测 (前向+反向) 墙钟——与稠密臂的口径可比
    （均按注意力项计数 / 墙钟）。
    """
    device = torch.device(device)
    # Device-matched generator: on CPU this matches torch.Generator(), while a
    # CPU generator handed to CUDA factory calls (or Tensor.normal_) is a
    # device mismatch.  Pinning every draw also makes repeated invocations
    # reproducible instead of dependent on ambient RNG state.
    gen = torch.Generator(device=device).manual_seed(1234)
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
        g = torch.empty_like(out)
        g.normal_(0.0, 1.0, generator=gen)
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
