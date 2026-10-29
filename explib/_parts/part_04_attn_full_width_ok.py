
def _attn_full_width_ok(n, k, w, need_dedup):
    # Bytes of the hoisted tables: the [n, k+w] valid mask, its negated twin
    # nvalid_all, the per-row any-reduction, the optional dedup/keep masks, and
    # the int64 [n, k+w] rows table built by the same branch.  Counting only
    # one boolean table admitted 2-4x the intended cap at the threshold.
    #
    # The window branch additionally materialises the cached full geometry
    # [n, w] (int32 indices + bool validity, 5 bytes/lane) and, when the row
    # table is hoisted, its int32 block-offset twin (4 bytes/lane).  Leaving
    # those out let a shape pass this 64 MiB gate while the window tables alone
    # added up to ~2x the cap at long context.  Keep the accounting complete so
    # the chunked fallback (value-identical) is chosen before the allocation.
    _m = int(k) + int(w)
    _bytes = int(n) * (2 * _m * 1 + 8 + (2 * int(k) if need_dedup else 0)
                       + 8 * _m)
    if int(w) > 0:
        _bytes += int(n) * int(w) * (4 + 1 + 4)
    return _bytes <= _FO_FULL_CAP_BYTES

def _finfo_min(dtype):
    # torch.finfo(dtype).min is a pure function of the dtype — the same
    # constant on every call, so it is memoised per dtype (host-side, no
    # tensor values involved).
    v = _FO_MINL.get(dtype)
    if v is None:
        v = torch.finfo(dtype).min
        _FO_MINL[dtype] = v
    return v

def gathered_attention(q, k_blk, v_blk, topk_idx, last_tok, k_sw, v_sw, w, scale, q_chunk=128, soft=None, sink_logits=None, mem_budget_bytes=None, sel_valid=None, topk_distinct=False, soft_form=False, v_stacked=None, k_stacked=None):
    if (k_sw is None) != (v_sw is None):
        raise ValueError('gathered_attention needs BOTH k_sw and v_sw (or neither)')
    if k_sw is None and int(w) != 0:
        # Without window tensors the mask/row tables would be built for `w`
        # lanes that have no K/V source, surfacing later as a shape error.
        raise ValueError(f'gathered_attention: w={w} requires k_sw/v_sw; pass w=0 for block-only attention')
    n = q.shape[0]
    dev = q.device
    if mem_budget_bytes is None:
        mem_budget_bytes = _attn_transient_budget(dev)
    if mem_budget_bytes is None:
        mem_budget_bytes = 256 << 20
    _heads = q.shape[1] if q.dim() == 3 else 1
    _M = max(1, int(topk_idx.shape[1]) + int(w))
    _per_row = 4 * _M * _heads * int(q.shape[-1]) + 2 * _M * _heads
    _bytes_per_chunk_row = _per_row * q.element_size()
    _bindable = mem_budget_bytes is not None and math.isfinite(float(mem_budget_bytes))
    if _bindable and _bytes_per_chunk_row > 0:
        _max_rows = max(1, int(mem_budget_bytes) // _bytes_per_chunk_row)
        q_chunk = min(q_chunk, _max_rows)
    q_chunk = max(1, min(int(q_chunk), n))
    out = torch.empty_like(q)
    pos_all = _arange_cache(n, dev)
    # Only the full-width branch needs the O(n*w) geometry table.  At long
    # context (or a wide top-k sweep) the chunked branch serves every row it
    # reads from `_window_geometry_rows`, so the cached full table would be a
    # pure O(n*w) allocation with no consumer.
    _nb = int(topk_idx.shape[1])
    _dedup = (not topk_distinct) and _nb > 1
    _full_width = _attn_full_width_ok(n, _nb, w, _dedup)
    if _full_width:
        win_idx, win_valid_all = _window_geometry(n, w, dev)
    else:
        win_idx = win_valid_all = None
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
        # `v_stacked` lets a caller that already owns a row-contiguous
        # [blocks | window] V view (the fused KV head in `_single`) hand it in
        # directly: gathering its rows yields exactly the rows of
        # cat([v_blk, v_sw]) in the same order, so the values are bit-identical.
        v_all = v_stacked if v_stacked is not None else torch.cat([v_blk, v_sw], 0)
        # `k_stacked` is the analogous row-contiguous [blocks | window] K
        # table (the normalized K columns of the fused KV head in `_single`).
        # The soft path can gather both halves from it with the window-offset
        # row index.
        if k_stacked is not None:
            k_all = k_stacked
        else:
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
                   and topk_l is _cache_get(_IDX_CACHE, (_dev_str(dev), 'arangeexp', int(n), int(topk_l.shape[1]))))
    # the keep mask and the [selected|window] valid mask are per-row functions of
    # the same inputs, so their full-width forms are bitwise identical to the
    # per-chunk forms; they are hoisted out of the chunk loop under
    # _FO_FULL_CAP_BYTES (the chunked fallback below covers very wide top-k
    # sweeps).  The full [n, k] block table itself is built ONLY in the
    # full-width branch: building it before the size check would defeat the
    # chunked fallback at long context (HCA has B ~ T / block_size).
    # `topk_distinct=True` asserts every row of topk_idx holds distinct block
    # ids (the read-everything HCA arm passes an arange expansion), in which
    # case first_occurrence_mask is provably all-True — the keep mask is then
    # identical to the computed form.
    if _full_width:
        # Build the [selected | window] table in place: the block half is
        # written straight into its slice, with no separate sel table and cat
        # copy needed; the result is the same boolean table, lane for lane.
        valid_all = win_valid_all.new_empty((n, _nb + int(w)))
        valid_all[:, :_nb] = (pos_all[:, None] > last_tok[None, :]) if _arange_sel else (pos_all[:, None] > last_tok[topk_l])
        keep_all = sel_valid
        if _dedup:
            _ddk_all = _first_occurrence_of_valid(topk_l, sel_valid)
            keep_all = _ddk_all if keep_all is None else keep_all & _ddk_all
        if keep_all is not None:
            valid_all[:, :_nb] &= keep_all
        valid_all[:, _nb:] = win_valid_all
        # full-width forms of the per-chunk negation / any-reduction: the
        # elementwise and reduction values per row are identical to the
        # per-chunk forms.
        nvalid_all = ~valid_all[:, None, :]
        anyrow_all = valid_all.any(-1) if sink_logits is None else None
    else:
        keep_all = valid_all = nvalid_all = anyrow_all = None
    # The chunk-invariant dispatch is computed once, above the row-chunk
    # loop: which mask/validity source each chunk reads and which of the two
    # attention forms runs are pure functions of the call-level arguments.
    # The loop body itself — and the tensor values it produces — is
    # unchanged.
    _has_all = valid_all is not None
    _has_sv = sel_valid is not None
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
        # valid_all table is fully absorbed into them.
        _plan = [(None if keep_all is None else keep_all[_s:_s + q_chunk], nvalid_all[_s:_s + q_chunk], None if anyrow_all is None else anyrow_all[_s:_s + q_chunk]) for _s in range(0, n, q_chunk)]
        del valid_all
    else:
        _plan = None
    # The full offset table is also O(n*w) and is only consumed by the
    # full-width branch.  The chunked branch adds the same block offset to its
    # own row slice in `_window_geometry_rows`, so it never allocates this
    # table.
    _win_off_tab = (_window_block_offset(n, w, _nblk_rows, dev)
                    if (_win_off and _has_all) else None)
    # Full-width row-table hoist: when the hoisted [n, nb + w] masks are live
    # (`_has_all`) and the int64 rows table fits the same byte ceiling the
    # mask hoist uses, the concatenated [selected | window] row table is
    # built once and each chunk reads its own rows as a slice — a slice of
    # the concatenation holds exactly the rows the per-chunk concatenation
    # builds (same operands, same cat semantics, same integers; index
    # transport, design notes §6.1).  Above the cap the per-chunk cat form
    # below is value-identical.
    _rows_full = None
    if _has_all and _has_sw:
        _r_elems = int(n) * (int(_nb) + int(w))
        if _r_elems * 8 <= _FO_FULL_CAP_BYTES:
            _rows_full = torch.cat([topk_l, (_win_off_tab if _win_off_tab is not None else win_idx)], 1)
    for s in range(0, n, q_chunk):
        e = s + q_chunk
        if e > n:
            e = n
        qseg = q[s:e]
        if _plan is not None:
            keep, nvalid, anyrow = _plan[s // q_chunk]
            _win_local = win_idx[s:e]
            _win_rows = _win_off_tab[s:e] if _win_off_tab is not None else _win_local
        else:
            # per-chunk form of the full-width table (same integers, same
            # booleans).
            _lt_rows = last_tok[None, :] if _arange_sel else last_tok[topk_l[s:e]]
            sel_blk = pos_all[s:e, None] > _lt_rows
            keep = None
            if _has_sv:
                keep = sel_valid[s:e]
            if _dedup:
                _ddk = _first_occurrence_of_valid(
                    topk_l[s:e], sel_valid[s:e] if _has_sv else None)
                keep = _ddk if keep is None else keep & _ddk
            if keep is not None:
                sel_blk = sel_blk & keep
            # Build only the [rows, w] window geometry this chunk reads: the
            # full [n, w] table/offset table is not materialised here.  The
            # local indices address k_sw/v_sw; the offset rows address the
            # concatenated [blocks | window] table.
            _win_local, _win_valid = _window_geometry_rows(n, w, dev, s, e)
            _win_rows = ((_win_local + _win_off).to(torch.int32)
                         if _win_off else _win_local)
            valid = torch.cat([sel_blk, _win_valid], dim=1)
            nvalid = ~valid[:, None, :]
            anyrow = None if _has_sink else valid.any(-1)
            del sel_blk, valid
        # Two half-gathers (and the same for V): the block half gathers the
        # k_blk rows topk_l addresses (rows 0..n_blk-1 of the stacked tensor
        # == k_blk rows) and the window half gathers k_sw at win_idx (rows
        # n_blk+j of the stacked tensor == row j of k_sw).  index_select
        # returns exactly the same source rows, so the gathered values — and
        # hence every downstream value — are bit-identical to the
        # cat-then-gather form (index re-plumbing only, design notes §6.1).
        _sel_rows = topk_l[s:e]
        # `_has_sw` is the contract for stacking window lanes: the row offset
        # may be 0 when k_blk holds no rows, and that is still a valid
        # stacked-row base.
        _rows = (_rows_full[s:e] if _rows_full is not None else
                 (torch.cat([_sel_rows, _win_rows], 1) if _has_sw else _sel_rows))
        # soft_form=True 在 soft=None 时仍选用软分支的硬部分解（两个半段
        # logits + cat），供调用方在跳过（无梯度下已死的）soft 尾部后保持
        # 硬分支逐位一致；它不改变任何取值，只固定走哪套既有分解。
        if soft is None and not soft_form:
            Kset = _tk2(k_all, _rows)
            logits = _einsum('qhd,qmhd->qhm', qseg, Kset)
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
            del logits, attn, nvalid, anyrow, keep, _sel_rows, _win_local, _win_rows, _rows, qseg
            continue
        # The soft branch reads its K halves as two contiguous gathers: the
        # concatenated V gather below holds exactly the same rows as two
        # separate gathers + cat, while the per-half K gathers feed the two
        # logits einsums with the same operands (index re-plumbing only,
        # design notes §6.1).
        if k_all is not None:
            # The combined table holds the same rows as the two halves:
            # block rows at the front and window row j at n_blk + j.
            kb = _tk2(k_all, _sel_rows)
            kw = _tk2(k_all, _win_rows)
        else:
            kb = _tk2(k_blk, _sel_rows)
            kw = _tk2(k_sw, _win_local)
        if _soft_live:
            # The live-soft path keeps the cat form: its clone is fed to the
            # straight-through branch, and a slice-assigned CopySlices table
            # does not reproduce that graph's gradients exactly.
            blk_logits = _einsum('qhd,qmhd->qhm', qseg, kb)
            if scale != 1.0:
                blk_logits.mul_(scale)
            win_logits = _einsum('qhd,qmhd->qhm', qseg, kw)
            if scale != 1.0:
                win_logits.mul_(scale)
            logits = torch.cat([blk_logits, win_logits], -1)
            del blk_logits, win_logits
        else:
            # Write the two independent logits halves into one preallocated
            # [q, h, nb + w] table instead of materialising both halves and a
            # concatenated copy.  The slice assignment feeds the same einsum
            # values to the same lanes.
            _w0 = int(_win_local.shape[1])
            logits = qseg.new_empty((qseg.shape[0], qseg.shape[1], _nb + _w0))
            logits[:, :, :_nb] = _einsum('qhd,qmhd->qhm', qseg, kb)
            logits[:, :, _nb:] = _einsum('qhd,qmhd->qhm', qseg, kw)
            if scale != 1.0:
                logits.mul_(scale)
        del kb, kw
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
            # soft_logits is dead once soft_attn is built.
            del soft_logits, soft_log_g
        logits.masked_fill_(nvalid, _MINL)
        attn, _sink_unused = _sink_split_softmax(logits, sink_logits, want_sink=False)
        if _soft_live:
            attn = (soft_attn - soft_attn.detach()).add_(attn)
        if not _has_sink:
            attn.mul_(anyrow[:, None, None])
        out[s:e] = _einsum('qhm,qmhd->qhd', attn, _tk2(v_all, _rows))
        # `soft_attn` only exists on the live-soft path.
        del logits, attn, nvalid, anyrow, keep, _sel_rows, _win_local, _win_rows, _rows, qseg
        if _soft_live:
            del soft_attn
    return out


def gathered_attention_open(q, k_blk, v_blk, topk_idx, sel_valid, last_tok,
                            k_sw, v_sw, w, scale, *, prefix_geo,
                            prefix_kind='summary', prefix_k=None, prefix_v=None,
                            soft=None, sink_logits=None, q_chunk=128,
                            mem_budget_bytes=None, return_prefix_mass=False):
    """Gathered attention with per-query prefix candidates.

    `topk_idx` holds combined candidate ids (closed blocks `0..B-1`, prefix
    candidates `B..`).  Closed rows gather from `k_blk`; prefix rows gather
    from the per-query summary (`prefix_k`, one row per query) or from the
    local token table (`k_sw`, for length-1 pooled singletons).  The local
    window is concatenated exactly as in `gathered_attention`; when the
    scope is `full_prefix`, a singleton that is also in the window is a
    deliberate duplicate information path, not a deduplicated read.

    `sel_valid` already encodes the strict closed-block gate and the causal
    prefix scope, so no future position can be read through the candidate
    half.  Returns `(out, prefix_mass)`; `prefix_mass` is the mean-over-heads
    attention mass assigned to prefix candidate slots per query (zero when
    not requested).
    """
    n = int(q.shape[0])
    B = int(k_blk.shape[0])
    K = int(topk_idx.shape[1])
    dev = q.device
    nh, hd = int(q.shape[1]), int(q.shape[2])
    if mem_budget_bytes is None:
        mem_budget_bytes = _attn_transient_budget(dev)
    if mem_budget_bytes is None:
        mem_budget_bytes = 256 << 20
    M = max(1, K + int(w))
    per_row = 4 * M * nh * hd + 2 * M * nh
    q_chunk = max(1, min(int(q_chunk), max(1, int(mem_budget_bytes) // max(1, per_row * q.element_size()))))
    q_chunk = max(1, min(q_chunk, n))
    _MINL = _finfo_min(k_blk.dtype)
    out = torch.empty_like(q)
    mass = None
    if return_prefix_mass:
        mass = q.new_zeros((n,))
    for s in range(0, n, q_chunk):
        e = min(s + q_chunk, n)
        qseg = q[s:e]
        ids = topk_idx[s:e].long()
        valid = sel_valid[s:e]
        is_pref = ids >= B
        if B > 0:
            _safe_ids = ids.clamp(0, B - 1)
            kb = k_blk[_safe_ids]
            vb = v_blk[_safe_ids]
        else:
            kb = q.new_zeros((e - s, K, nh, hd))
            vb = q.new_zeros((e - s, K, nh, hd))
        if prefix_kind == 'summary':
            pk = prefix_k[s:e, None].expand(e - s, K, nh, hd)
            pv = prefix_v[s:e, None].expand(e - s, K, nh, hd)
        elif prefix_kind == 'singletons':
            P = int(prefix_geo['token_idx'].shape[1])
            if P <= 0:
                # No represented outside-window token for any query: the
                # candidate set is exactly the closed set.  `is_pref` is
                # all-False, so these placeholders are never read.
                pk = kb
                pv = vb
            else:
                pid = (ids - B).clamp(0, max(P - 1, 0))
                u = prefix_geo['token_idx'][s:e].gather(1, pid)
                pk = k_sw[u]
                pv = v_sw[u]
        else:
            raise ValueError(f'unknown prefix_kind {prefix_kind!r}')
        Kc = torch.where(is_pref[:, :, None, None], pk, kb)
        Vc = torch.where(is_pref[:, :, None, None], pv, vb)
        # Per-chunk window geometry: this path has no full-width mask hoist,
        # so there is no reason to materialise the O(n*w) table.
        _win_local, _win_valid = _window_geometry_rows(n, int(w), dev, s, e)
        kw = k_sw[_win_local]
        vw = v_sw[_win_local]
        Kall = torch.cat([Kc, kw], 1)
        Vall = torch.cat([Vc, vw], 1)
        valid_all = torch.cat([valid, _win_valid], 1)
        logits = torch.einsum('qhd,qmhd->qhm', qseg, Kall)
        if scale != 1.0:
            logits.mul_(scale)
        # Straight-through indexer prior: its forward contribution cancels
        # exactly, so a no-grad/forward-only pass can skip it without changing
        # any output value (the same argument the closed path uses).
        _soft_live = (soft is not None and torch.is_grad_enabled())
        attn_soft = None
        if _soft_live:
            soft_g = soft[s:e]
            s_log = torch.log(soft_g.clamp_min(1e-12))
            s_log.masked_fill_(~valid, _MINL)
            logits_soft = logits.clone()
            logits_soft[:, :, :K].add_(s_log[:, None, :])
            logits_soft.masked_fill_(~valid_all[:, None, :], _MINL)
            attn_soft, _ = _sink_split_softmax(logits_soft, sink_logits, want_sink=False)
            del logits_soft, s_log
        logits.masked_fill_(~valid_all[:, None, :], _MINL)
        attn, _ = _sink_split_softmax(logits, sink_logits, want_sink=False)
        if _soft_live:
            attn = (attn_soft - attn_soft.detach()).add_(attn)
            del attn_soft
        if sink_logits is None:
            anyrow = valid_all.any(-1)
            attn.mul_(anyrow[:, None, None])
        if return_prefix_mass:
            _pm = (attn[:, :, :K] * is_pref[:, None, :].to(attn.dtype)).sum(-1).mean(-1)
            mass[s:e] = _pm
        out[s:e] = torch.einsum('qhm,qmhd->qhd', attn, Vall)
        del Kall, Vall, Kc, Vc, kw, vw, kb, vb, pk, pv, logits, attn, valid_all
    return (out, mass) if return_prefix_mass else (out, None)


def _gathered_attention_open_trainable(*args, **kwargs):
    """Checkpoint wrapper for the open-block attention implementation."""
    if torch.is_grad_enabled() and args[0].requires_grad:
        q = args[0]
        if q.device.type != 'cuda':
            _M = int(args[3].shape[1]) + int(args[8])
            _saved = 2 * int(q.numel()) * max(1, _M) * int(q.element_size()) * int(_CPU_ACTIVE_BATCH[0])
            if _cpu_saved_ok(_saved, _CPU_GATHER_CKPT_BYTES):
                return gathered_attention_open(*args, **kwargs)
        return torch.utils.checkpoint.checkpoint(
            gathered_attention_open, *args, use_reentrant=False,
            preserve_rng_state=False, **kwargs)
    return gathered_attention_open(*args, **kwargs)


_CPU_ACTIVE_BATCH = [1]
_CPU_GATHER_CKPT_BYTES = 512 << 20
_CPU_POOL_CKPT_BYTES = 64 << 20
_CPU_CE_CKPT_BYTES = 128 << 20

def _cpu_saved_ok(saved_bytes, limit):
    return torch.is_grad_enabled() and saved_bytes < int(limit)

def _gathered_attention_trainable(*args, **kwargs):
    # 训练时包一层非重入 checkpoint，反向按同一确定性分解重算前向。函数内
    # 无 RNG，因此 chunk 边界与每个中间值逐位相同，梯度与直调形式逐位一致。
    # 重算不消耗也不改写任何随机流。
    if torch.is_grad_enabled() and args[0].requires_grad:
        q = args[0]
        if q.device.type != 'cuda':
            _M = int(args[3].shape[1]) + int(args[7])
            _saved = 2 * int(q.numel()) * max(1, _M) * int(q.element_size()) * int(_CPU_ACTIVE_BATCH[0])
            if _cpu_saved_ok(_saved, _CPU_GATHER_CKPT_BYTES):
                return gathered_attention(*args, **kwargs)
        return torch.utils.checkpoint.checkpoint(gathered_attention, *args, use_reentrant=False, preserve_rng_state=False, **kwargs)
    return gathered_attention(*args, **kwargs)

def _pool_trainable(fn, *args, **kwargs):
    # 训练时包一层非重入 checkpoint，反向重算同一纯函数：池化内部无 RNG，
    # 整数几何缓存挂在不可变的 block_ids 对象上（_pool_parts / _pool_geom，
    # design notes §6.2），反向重算必命中同一几何，前向值与梯度逐位一致；
    # int 型的 n_blocks 返回值原样透传。eval 下（无梯度）直调。池化同样
    # 无 RNG，所以 checkpoint 不需要保存/恢复随机流。
    if torch.is_grad_enabled() and args[0].requires_grad:
        x = args[0]
        if x.device.type != 'cuda':
            _saved = 16 * int(x.numel()) * int(x.element_size()) * int(_CPU_ACTIVE_BATCH[0])
            if _cpu_saved_ok(_saved, _CPU_POOL_CKPT_BYTES):
                return fn(*args, **kwargs)
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False, **kwargs)
    return fn(*args, **kwargs)

OPEN_BLOCK_MODES = ('closed', 'prefix_summary', 'prefix_singletons')
OPEN_BLOCK_SCOPES = ('full_prefix', 'outside_window')
SELECTION_BUDGETS = ('shared_topk', 'all_closed_plus_prefix')
POOL_POSITION_POLICIES = ('truncate',)
CHUNK_SIGNAL_SOURCES = ('auto', 'context', 'fused')
# Content controls for the compression ablation:
#   pooled — the block's own pooled content (the treatment);
#   zero   — no content at all (presence control);
#   shift  — the SAME pooled content, moved back one block (alignment
#            control).  `shift` keeps the tensor shape, parameter count,
#            read budget and (after normalisation) the row norm of `pooled`,
#            but destroys the correspondence between a block's content and
#            its own token span.  It therefore separates "any non-zero
#            content helps" from "block-aligned content carries information".
CONTENT_MODES = ('pooled', 'zero', 'shift')

# Sink controls: `learned` is the treatment (a trainable per-head logit).
# `random` keeps the SAME parameter shape/count but replaces the learned value
# with a fixed pseudo-random vector and freezes it.  It therefore separates "a
# learned sink absorbs off-context mass" from "the extra sink
# parameter/capacity helps".  The vector is drawn from a LOCAL generator seeded
# by a hash of the layer's initialized W_q: it consumes no global RNG, so the
# two arms keep the same initialization for every other tensor (pair_reason's
# init_core_fp and the per-layer W_q stay matched), while the placebo values
# are not aligned with the layer's query directions.
SINK_MODES = ('learned', 'random')

# Fusion controls: `none` is identity fusion (no convolution), `learned` is the
# shipped trainable depth-wise convolution, and `random` keeps the SAME
# parameter shape/count but replaces the learned weight with a fixed
# pseudo-random tensor and freezes it.  The random placebo is the capacity-
# matched control for "the learned fusion matters"; it consumes no global RNG
# so the treatment and placebo still start from the same weights everywhere
# else.  The learned arm's Conv1d is also constructed while the global torch RNG
# is saved/restored: otherwise merely *having* the extra convolution would
# shift the initialization of every later block parameter, making the on/off
# contrast a joint test of fusion AND a different initialization.
FUSE_MODES = ('none', 'learned', 'random')


def _frozen_random_fuse_weight(w_q, d_model, kernel, dtype):
    """A deterministic, non-learned depth-wise fusion kernel.

    The seed is a hash of the layer's initialized W_q plus a fuse-specific
    domain separator.  It therefore draws from a LOCAL generator (no global
    RNG consumption) and is independent of the learned projection, while the
    shape and parameter count match the treatment exactly.
    """
    out_channels = int(d_model)
    kernel = int(kernel)
    shape = (out_channels, 1, kernel)
    try:
        _bytes = w_q.detach().to('cpu').float().contiguous().numpy().tobytes()
        _seed = int.from_bytes(
            hashlib.sha256(_bytes + b'fuse').digest()[:8], 'big')
        _gen = torch.Generator(device='cpu')
        _gen.manual_seed(_seed)
        w = torch.randn(shape, generator=_gen, dtype=torch.float32)
    except Exception:
        w = torch.empty(shape, dtype=torch.float32)
    if not torch.isfinite(w).all() or float(w.norm()) == 0.0:
        w = torch.linspace(-1.0, 1.0, out_channels * kernel,
                           dtype=torch.float32).reshape(shape)
    # Identity fusion has per-channel kernel norm 1; scale the random kernel
    # the same way so it is a magnitude-matched placebo, not a larger/smaller
    # intervention.
    w = w / math.sqrt(max(kernel, 1))
    return w.to(dtype=dtype)


def _shift_block_content(table):
    """Move each row of a block-content table back by one block.

    Row 0 has no predecessor and is zeroed (the same value as the zero-content
    control for that one block); every other row is the previous row.  The
    transform is strictly causal (row i depends only on rows < i), uses no
    RNG and consumes no gradients into the current row, so prefix invariance
    and deterministic replay are preserved.  The result has the same shape,
    dtype and per-row norm as the input after the caller's normalisation.
    """
    if not torch.is_tensor(table) or table.ndim != 2:
        raise ValueError(f'_shift_block_content expects a 2-D tensor, got {type(table).__name__}')
    if table.shape[0] <= 1:
        return torch.zeros_like(table)
    return torch.cat([table.new_zeros((1, table.shape[1])), table[:-1]], dim=0)

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
    # `max_block=None` means "dynamic boundaries only": no forced cut is
    # introduced by a maximum length.  The segmenters convert it to an
    # explicit safe bound (the current sequence length) at their entry
    # points; it is never a hidden large constant.
    max_block: int | None = 128
    # Position-embedding table capacity is deliberately independent of
    # `max_block`: every arm of the open-block panel can share the same
    # parameters while the maximum block length varies.  Out-of-range
    # positions are handled by `pool_position_policy` (currently `truncate`).
    pool_position_capacity: int = 129
    pool_position_policy: str = 'truncate'
    # Open-block reading rule.  `closed` is the historical behaviour: only
    # blocks whose end is strictly before the query are readable.  The two
    # prefix modes add a causal representation of the current (still open)
    # block to the candidate set without changing the dynamic cut table.
    open_block_mode: str = 'closed'
    open_block_scope: str = 'full_prefix'
    selection_budget: str = 'shared_topk'
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
    sink_mode: str = 'learned'
    fuse_mode: str = 'none'
    fuse_kernel: int = 5
    # 'threshold' uses hard-cut decisions.  'reinforce'
    # samples the cut decisions and adds a score-function term for the LM loss,
    # so the task objective can actually learn the boundary policy instead of
    # only the compression-rate regularizer.
    boundary_policy: str = 'threshold'
    # For indexer_mode='random': a per-(seed, layer) additive salt.  The
    # legacy fixed hash corresponds to salt 0 (bit-identical path).  A
    # non-zero salt makes the 'random' control a draw per seed/layer, so
    # the paired sign-flip sees a random variable rather than one fixed
    # arbitrary pattern shared by every seed.
    random_salt: int = 0
    # Which representation feeds the chunk-boundary similarity signal.
    # 'auto' keeps the historical mapping (learned boundaries use the fused
    # local projection, fixed/adaptive cosine uses the layer input).  Setting
    # it explicitly lets a panel FORCE the same signal for every arm, so a
    # preprocessing/fusion difference cannot masquerade as a chunking-policy
    # difference.  The effective source is recorded by chunking_protocol().
    cut_signal_source: str = 'auto'

    def __post_init__(self):
        if self.cut_signal_source not in CHUNK_SIGNAL_SOURCES:
            raise ValueError(
                f'unknown cut_signal_source {self.cut_signal_source!r}; expected '
                f'one of {CHUNK_SIGNAL_SOURCES}')
        if self.content_mode not in CONTENT_MODES:
            raise ValueError(
                f'unknown content_mode {self.content_mode!r}; expected one of '
                f'{CONTENT_MODES}')
        if self.sink_mode not in SINK_MODES:
            raise ValueError(
                f'unknown sink_mode {self.sink_mode!r}; expected one of '
                f'{SINK_MODES}')
        _sink_active = (self.full_sink if self.kind == 'full'
                        else self.use_sink)
        if self.sink_mode == 'random' and not _sink_active:
            raise ValueError(
                'sink_mode="random" requires the sink to be active for this '
                'layer kind (use_sink for csa/hca, full_sink for full); the '
                'absence control must use sink_mode="learned" with the sink '
                'disabled')
        if self.fuse_mode not in FUSE_MODES:
            raise ValueError(
                f'unknown fuse_mode {self.fuse_mode!r}; expected one of '
                f'{FUSE_MODES}')
        _fuse_active = (self.kind in ('csa', 'hca')
                        and self.chunking == 'cosine_learnable')
        if self.fuse_mode == 'random' and not _fuse_active:
            raise ValueError(
                'fuse_mode="random" requires a csa/hca layer with '
                'chunking="cosine_learnable"; a layer that cannot fuse must '
                'keep fuse_mode="none" (or the historical inert "learned")')
        if self.open_block_mode not in OPEN_BLOCK_MODES:
            raise ValueError(f'unknown open_block_mode {self.open_block_mode!r}; expected one of {OPEN_BLOCK_MODES}')
        if self.open_block_scope not in OPEN_BLOCK_SCOPES:
            raise ValueError(f'unknown open_block_scope {self.open_block_scope!r}; expected one of {OPEN_BLOCK_SCOPES}')
        if self.selection_budget not in SELECTION_BUDGETS:
            raise ValueError(f'unknown selection_budget {self.selection_budget!r}; expected one of {SELECTION_BUDGETS}')
        if self.pool_position_policy not in POOL_POSITION_POLICIES:
            raise ValueError(f'unknown pool_position_policy {self.pool_position_policy!r}; expected one of {POOL_POSITION_POLICIES}')
        if int(self.pool_position_capacity) < 1:
            raise ValueError(f'pool_position_capacity must be >= 1, got {self.pool_position_capacity!r}')
        # Non-negative/positive structural bounds.  A negative window or
        # overlap is not a meaningful configuration; reject it here rather
        # than failing later inside torch.arange/index maths.
        if int(self.min_block) < 1:
            raise ValueError(f'min_block must be >= 1, got {self.min_block!r}')
        if int(self.block_size) < 1:
            raise ValueError(f'block_size must be >= 1, got {self.block_size!r}')
        if int(self.target_block_tokens) < 1:
            raise ValueError(f'target_block_tokens must be >= 1, got {self.target_block_tokens!r}')
        if int(self.overlap) < 0:
            raise ValueError(f'overlap must be >= 0, got {self.overlap!r}')
        if int(self.sliding_window) < 0:
            raise ValueError(f'sliding_window must be >= 0, got {self.sliding_window!r}')
        if int(self.window) < 0:
            raise ValueError(f'window must be >= 0, got {self.window!r}')
        if int(self.index_topk) < 0:
            raise ValueError(f'index_topk must be >= 0, got {self.index_topk!r}')
        if int(self.c_kv) < 0 or int(self.c_index) < 0:
            raise ValueError(f'c_kv/c_index must be >= 0, got {self.c_kv!r}/{self.c_index!r}')
        if int(self.n_index_heads) < 1:
            raise ValueError(f'n_index_heads must be >= 1, got {self.n_index_heads!r}')
        if self.max_block is not None:
            if int(self.max_block) < 1:
                raise ValueError(f'max_block must be None or >= 1, got {self.max_block!r}')
            if int(self.max_block) < int(self.min_block):
                raise ValueError(f'max_block={self.max_block} is smaller than min_block={self.min_block}')
        if self.open_block_mode != 'closed':
            if self.kind not in ('csa', 'hca'):
                raise ValueError(f'open_block_mode={self.open_block_mode!r} is only defined for csa/hca layers, not {self.kind!r}')
            if self.kind == 'hca' and self.selection_budget == 'shared_topk':
                # HCA reads every closed block (no top-k), so a "shared top-k"
                # budget is not the rule it implements.  Make the extra-read
                # budget explicit rather than silently pretending the two
                # arms have the same number of slots.
                raise ValueError('open_block_mode on an hca layer requires selection_budget="all_closed_plus_prefix": HCA reads every closed block and has no top-k budget to share; an extra prefix candidate is an ADDITIONAL read slot')
            if self.kind == 'csa' and self.selection_budget != 'shared_topk':
                raise ValueError(f'csa open-block layers implement selection_budget="shared_topk"; got {self.selection_budget!r}')

    @property
    def max_block_tokens(self):
        """Explicit experiment-facing alias for `max_block`."""
        return self.max_block

    def max_block_bound(self, seq_len=None):
        """Explicit safe maximum block length for the segmenters.

        ``None`` is not passed into integer arithmetic: at a call site with a
        known sequence length it becomes that length (which can never force a
        cut), so "no maximum" is a real configuration rather than a hidden
        fixed constant.
        """
        if self.max_block is not None:
            return int(self.max_block)
        if seq_len is None:
            raise ValueError('max_block=None needs the sequence length to derive its explicit safe bound')
        return int(seq_len)

    def position_capacity(self):
        return int(self.pool_position_capacity)

    def position_policy(self):
        return str(self.pool_position_policy)

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
        self.last_boundary_logp = None
        self.last_boundary_decisions = 0
        self.need_reg = True
        self._stats = None
        self._prefix_stats = None
        self._cuts = None
        self.fuse_conv = None
        # Fused-weight cache (eval-only): the gateway is `not
        # torch.is_grad_enabled()`, so training forward passes always rebuild
        # the cat inside the autograd graph exactly as before.  Under eval the
        # concat values are pure functions of the source weights, so keying on
        # their autograd `_version` counters (bumped by every in-place
        # optimizer update) makes a stale hit impossible.
        self._wcat_cache = {}
        if (cfg.kind in ('csa', 'hca')
                and cfg.chunking == 'cosine_learnable'
                and cfg.fuse_mode in ('learned', 'random')):
            # `nn.Conv1d` initializes its weight from the global torch RNG.
            # Creating it here would shift every later block parameter's draw
            # relative to the no-fuse arm, so the fusion on/off contrast would
            # also change the initialisation of W_aKV/W_bKV/... .  Save and
            # restore the stream around construction; fusion then differs from
            # identity ONLY in the fuse_conv parameter, for both the learned
            # treatment and the frozen-random placebo.
            _rng_state = torch.get_rng_state()
            try:
                self.fuse_conv = nn.Conv1d(
                    d_model, d_model, cfg.fuse_kernel, groups=d_model,
                    bias=False)
            finally:
                torch.set_rng_state(_rng_state)
            with torch.no_grad():
                if cfg.fuse_mode == 'learned':
                    self.fuse_conv.weight.zero_()
                    self.fuse_conv.weight[:, :, -1] = 1.0
                else:
                    _w = _frozen_random_fuse_weight(
                        self.W_q.weight, d_model, cfg.fuse_kernel,
                        self.fuse_conv.weight.dtype)
                    self.fuse_conv.weight.copy_(_w)
                    self.fuse_conv.weight.requires_grad_(False)
        if cfg.kind == 'full':
            self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)
            self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)
            nn.init.xavier_uniform_(self.W_k.weight)
            nn.init.xavier_uniform_(self.W_v.weight)
            self.W_aKV = None
            self.W_kvhead = None
            self.sink = self._make_sink(n_heads)
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
            self.B_pos_a = nn.Parameter(torch.zeros(int(cfg.position_capacity()), self.kv_dim))
            self.sink = self._make_sink(n_heads)
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
        self.B_pos_a = nn.Parameter(torch.zeros(int(cfg.position_capacity()), self.kv_dim))
        self.B_pos_b = nn.Parameter(torch.zeros(int(cfg.position_capacity()), self.kv_dim))
        self.sink = self._make_sink(n_heads)
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

    def _make_sink(self, n_heads):
        """Learned or frozen-random sink parameter with identical shape.

        The random placebo is drawn from a local generator whose seed is a
        hash of the layer's initialized W_q.  It therefore consumes no global
        RNG (constructing the placebo does not shift the initialization of any
        later tensor) and its values are not aligned with the layer's query
        directions.  That is what makes it a capacity-matched *placebo* rather
        than a different model.
        """
        active = self.cfg.full_sink if self.cfg.kind == 'full' else getattr(self.cfg, 'use_sink', True)
        if not active:
            return None
        if self.cfg.sink_mode == 'learned':
            return nn.Parameter(torch.zeros(n_heads))
        with torch.no_grad():
            _w = self.W_q.weight.detach()
            _bytes = _w.to('cpu').float().contiguous().numpy().tobytes()
            _seed = int.from_bytes(hashlib.sha256(_bytes).digest()[:8], 'big')
            _gen = torch.Generator(device='cpu')
            _gen.manual_seed(_seed)
            _base = torch.randn(int(n_heads), generator=_gen, dtype=torch.float32)
            if not torch.isfinite(_base).all() or float(_base.norm()) == 0.0:
                _base = torch.linspace(-1.0, 1.0, int(n_heads))
            _base = _base / _base.norm() * math.sqrt(int(n_heads))
            _base = _base.to(dtype=_w.dtype)
        return nn.Parameter(_base, requires_grad=False)

    def _fuse(self, x):
        if self.fuse_conv is None:
            return x
        k = self.cfg.fuse_kernel
        z = F.pad(x.t().unsqueeze(0), (k - 1, 0))
        return F.conv1d(z, self.fuse_conv.weight, groups=x.shape[1]).squeeze(0).t()

    def _full_batched(self, x):
        B, T, _ = x.shape
        _nhq = self.nh * self.hd
        if x.device.type == 'cuda' and not torch.is_grad_enabled():
            # Key on versions, dtype and device of the source weights.
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
        if self.cfg.full_cosine:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            scale = 1.0
        else:
            scale = 1.0 / math.sqrt(self.hd)
        # The sink branch builds a full cached bool mask within the row cap,
        # or the matching row slice per chunk beyond it.  The cached table is
        # bool (one byte per element), not a 4-byte additively-typed mask.
        _use_full_mask = _dense_mask_fits(T)
        if self.sink is None:
            if 0 < self.cfg.window < T:
                out = _window_attn_local(q, k, v, self.cfg.window, scale)
                return self.W_o(out.reshape(B, T, self.nh * self.hd))
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=True, scale=scale).transpose(1, 2)
            return self.W_o(out.reshape(B, T, self.nh * self.hd))
        if _use_full_mask:
            if self.cfg.window > 0:
                mask = causal_window_mask(T, self.cfg.window, x.device)
            else:
                mask = causal_mask(T, x.device)
        else:
            mask = None
        sink_view = _safe_nan_to_num(self.sink, where='sink_param').view(1, self.nh, 1, 1) if self.sink is not None else None
        row_chunk = _full_attn_row_chunk(x.device, B, self.nh, T, q.element_size())
        # 训练时整段注意力循环包一层非重入 checkpoint：反向按同一份纯行划分
        # 重算，值/梯度不变。row_chunk 在运行内经 _ATB_EPOCH 不变，mask 是
        # 缓存的常量张量，因此重算的块边界与中间值逐位相同（同 ce_loss
        # 分块重算的论证）。
        if torch.is_grad_enabled() and q.requires_grad:
            out = torch.utils.checkpoint.checkpoint(
                self._sink_attn_loop, q, k, v, mask, _use_full_mask, sink_view,
                scale, row_chunk, use_reentrant=False, preserve_rng_state=False)
        else:
            out = self._sink_attn_loop(q, k, v, mask, _use_full_mask, sink_view, scale, row_chunk)
        return self.W_o(out.reshape(B, T, self.nh * self.hd))

    def _sink_attn_loop(self, q, k, v, mask, use_full_mask, sink_view, scale, row_chunk):
        B, T = q.shape[0], q.shape[1]
        out = torch.empty(B, T, self.nh, self.hd, device=q.device, dtype=q.dtype)
        for s in range(0, T, row_chunk):
            e = min(s + row_chunk, T)
            # A tuple is the row-mask recipe, not a tensor: the checkpointed
            # recompute rebuilds the identical slice.
            _mask_c = mask[s:e] if use_full_mask else (s, e, T, self.cfg.window)
            out[:, s:e] = self._sink_attn_chunk(q[:, s:e], k, v, _mask_c, sink_view, scale)
        return out

    def _sink_attn_chunk(self, q_c, k, v, mask_c, sink_view, scale):
        if isinstance(mask_c, tuple):
            s, e, T, window = mask_c
            mask_c = _dense_causal_window_rows(T, window, s, e, q_c.device)
        logits = torch.einsum('bnhd,bmhd->bhnm', q_c, k)
        if scale != 1.0:
            # same IEEE identity as gathered_attention (x·1.0 and its backward
            # grad·1.0 are bit-identical to no-op).
            logits.mul_(scale)
        # The mask is the boolean masked-lane table; the fill writes exactly
        # the -inf the additive 0/-inf form added.  x + 0.0 == x except
        # -0.0 -> +0.0 (exp is 1.0 either way, so the softmax/logsumexp row
        # reduction sees the same finite lane values in the same order and
        # every downstream value is bit-identical), and the fill's backward
        # zeroes exactly the lanes the softmax backward already scaled by
        # p_i == 0 in the additive form — unmasked lanes pass the identical
        # grad through in both forms.
        logits.masked_fill_(mask_c, float('-inf'))
        if sink_view is not None:
            lse = torch.logaddexp(sink_view, torch.logsumexp(logits, dim=-1, keepdim=True))
            # logsumexp 的反向保留其输入 logits，sub 必须出位（保留值与新
            # 张量上的 exp_ 原地——exp 反向保存的输出此后再无人改写）；原地
            # sub_ 会使 logsumexp 保存的张量版本失效，反向直接报错。
            # 推理（无梯度）时没有保存值，logits 也不再被读取，因此可以直接
            # 原地归一化，省掉一张与 logits 同尺寸的临时表。
            if torch.is_grad_enabled():
                attn = logits.sub(lse).exp_()
            else:
                attn = logits.sub_(lse)
                attn.exp_()
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
        # Training never builds the concatenated weight: it is only a cached
        # constant on the eval path.  The separate projections produce the
        # same independent output columns without the per-step cat.
        if x.device.type != 'cuda' or torch.is_grad_enabled():
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
        sim = chunk_similarity(x, self.cfg, fused=fused)
        if sim.numel() == 0:
            return None
        cfg = self.cfg
        # In-place tails over the fresh cumsum output: same op order
        # (divide -> add -> subtract -> divide) and therefore the same values.
        prefix_mean = torch.cumsum(sim, 0)
        prefix_mean.div_(_range_cache(1, sim.numel() + 1, x.device))
        prefix_mean.add_(self.delta_logit)
        prefix_mean.sub_(sim)
        prefix_mean.div_(max(cfg.temperature, 0.001))
        if getattr(cfg, 'boundary_policy', 'threshold') == 'reinforce' and torch.is_grad_enabled():
            logits = prefix_mean
            gate = torch.sigmoid(logits)
            gate_bool = torch.bernoulli(gate).bool()
            logp = -F.binary_cross_entropy_with_logits(logits, gate_bool.to(gate.dtype), reduction='sum')
            return (gate, gate_bool, logp)
        gate = torch.sigmoid_(prefix_mean)
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
        cfg = self.cfg
        fused = pre_all.get('fused') if pre_all is not None else None
        if fused is None:
            if self.fuse_conv is not None:
                # safe fallback identical to the per-row loop (each row fuses
                # itself, exactly like _gate_signal's inline form)
                return ([self._gate_signal(x[b], None) for b in range(B)], None)
            fused = x.detach()
        sim = chunk_similarity(x, cfg, fused=fused, eps=1e-08)
        if sim.shape[-1] == 0:
            return ([None] * B, None)
        # rowwise batched cumsum — same per-row values as the unbind/stack loop.
        # The fresh cumsum output carries the divide/add/subtract/divide tails
        # in place.
        prefix_mean = torch.cumsum(sim, dim=-1)
        prefix_mean.div_(_range_cache(1, sim.shape[-1] + 1, x.device))
        prefix_mean.add_(self.delta_logit)
        prefix_mean.sub_(sim)
        prefix_mean.div_(max(cfg.temperature, 0.001))
        if getattr(cfg, 'boundary_policy', 'threshold') == 'reinforce' and torch.is_grad_enabled():
            logits = prefix_mean
            gate = torch.sigmoid(logits)
            gate_b = torch.bernoulli(gate).bool()
            logp = -F.binary_cross_entropy_with_logits(logits, gate_b.to(gate.dtype), reduction='none').sum(dim=-1)
            return ([(gate[b], gate_b[b], logp[b]) for b in range(B)], gate_b, logp)
        gate = torch.sigmoid_(prefix_mean)
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
        _fused = self._fuse(x.detach()) if chunk_similarity_source(cfg) == 'fused' else None
        sim = chunk_similarity(x.detach(), cfg, fused=_fused, eps=1e-08)
        if cfg.chunking == 'cosine_adaptive' or cfg.adaptive:
            tau = causal_adaptive_threshold(sim, cfg.target_block_tokens)
        else:
            tau = cfg.cos_threshold
        cuts = sim < tau
        if _device_segment_enabled(x.device):
            return _device_blocks_from_cuts(T, cuts, cfg.min_block, cfg.max_block, x.device, want_honoured=False, hon_dtype=torch.float32)
        cuts = cuts.cpu().numpy()
        return _hoist_blocks_from_cuts(T, cuts, cfg.min_block, cfg.max_block, x.device, want_honoured=False, hon_dtype=torch.float32)

    def _single(self, x, pre=None, gate_sig=None, pre_blocks=None):
        cfg = self.cfg
        self.last_boundary_logp = None
        self.last_boundary_decisions = 0
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
        _seg_cuts = None
        if cfg.chunking == 'cosine_learnable':
            if gate_sig is not None:
                gate, gate_list = gate_sig[0], gate_sig[1]
                if len(gate_sig) > 2:
                    self.last_boundary_logp = gate_sig[2]
                    self.last_boundary_decisions = int(gate.numel())
            else:
                fused = pre['fused'] if pre is not None and 'fused' in pre else self._fuse(x.detach())
                sim = chunk_similarity(x, cfg, fused=fused)
                if sim.numel() == 0:
                    gate = torch.tensor([], device=x.device)
                else:
                    # same in-place tails as _gate_signal (same values)
                    prefix_mean = torch.cumsum(sim, 0)
                    prefix_mean.div_(_range_cache(1, sim.numel() + 1, x.device))
                    prefix_mean.add_(self.delta_logit)
                    prefix_mean.sub_(sim)
                    prefix_mean.div_(max(cfg.temperature, 0.001))
                    gate = torch.sigmoid_(prefix_mean)
                gate_bool = None
                gate_list = None
                if gate.numel():
                    if getattr(cfg, 'boundary_policy', 'threshold') == 'reinforce' and torch.is_grad_enabled():
                        # score-function path: the hard cuts are sampled, and
                        # the log-probability is returned so the LM loss can
                        # train the boundary policy (not just comp_reg).
                        _logits = torch.logit(gate.clamp(1e-7, 1 - 1e-7))
                        gate_bool = torch.bernoulli(gate).bool()
                        self.last_boundary_logp = -F.binary_cross_entropy_with_logits(
                            _logits, gate_bool.to(gate.dtype), reduction='sum')
                        self.last_boundary_decisions = int(gate.numel())
                    else:
                        gate_bool = gate.detach() > _HALF
                    # On the device path the soft-honoured mask is already
                    # attached to pre_blocks, so the boolean cut row can stay
                    # on the GPU (it is not consumed by the host segmenter).
                    gate_list = gate_bool if _device_segment_enabled(x.device) else gate_bool.cpu().numpy()
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
            _seg_cuts = gate_list
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
                _fused = self._fuse(x.detach()) if chunk_similarity_source(cfg) == 'fused' else None
                sim = chunk_similarity(x.detach(), cfg, fused=_fused)
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
        # Open-block (current-prefix) representations.  These are computed
        # from x_{<=t} only and never modify `bid`; the closed block table
        # above is unchanged.  The candidate is a temporary representation of
        # the current block for THIS query, not a future block summary.
        _open_mode = cfg.open_block_mode
        if cfg.content_mode == 'shift' and _open_mode != 'closed':
            raise RuntimeError(
                'content_mode="shift" is defined only for closed blocks; the '
                'open-block prefix representations have a different candidate '
                'geometry, so shifting their content would not be a controlled '
                'content-alignment manipulation')
        _prefix_geo = None
        _prefix_comp = None
        _prefix_kind = None
        if _open_mode != 'closed':
            _scope = cfg.open_block_scope
            _prefix_geo = _open_prefix_geometry(
                bid, n_blocks=int(nblk), scope=_scope,
                window=cfg.sliding_window,
                candidate_table=(_open_mode != 'prefix_summary'))
            if _open_mode == 'prefix_summary':
                _prefix_comp = causal_prefix_pool(
                    Ca_raw, (Cb_raw if cfg.kind == 'csa' else None), Za,
                    (Zb if cfg.kind == 'csa' else None), bid,
                    self.B_pos_a, (self.B_pos_b if cfg.kind == 'csa' else None),
                    cfg.overlap, n_blocks=int(nblk), scope=_scope,
                    window=cfg.sliding_window,
                    use_overlap=(cfg.kind == 'csa' and _scope == 'full_prefix'))
                _prefix_kind = 'summary'
            elif _open_mode == 'prefix_singletons':
                # A length-1 block through the same pooling path: with one A
                # lane the softmax is exactly 1 and no position bias survives,
                # so the pooled value is Ca_raw[u] (a projection of the token,
                # NOT the raw token K/V).  The K/V table is the existing
                # local-token table; the candidate ids still go through the
                # same indexer/scoring path as every other candidate.
                _prefix_kind = 'singletons'
            else:
                raise ValueError(f'unknown open_block_mode {_open_mode!r}')
        # The indexer keeps the block's true content; only the attention
        # K/V source is shifted, so the ablation isolates content alignment
        # in the read path rather than simultaneously blinding the selector.
        _attn_comp_kv = (_shift_block_content(comp_kv)
                         if cfg.content_mode == 'shift' else comp_kv)
        index_kv = comp_kv
        if self._stats is not None:
            # scatter_add of exact integer ones is order-independent (integer
            # addition commutes, no rounding).  bincount is deterministic
            # (no atomics) and returns the same exact integer counts.
            _bc = torch.bincount(bid.reshape(-1).long(), minlength=int(nblk))[:int(nblk)]
            self._stats.append(_bc)
            self._stats.add_segmentation(bid, _seg_cuts, cfg.max_block)
            if self._cuts is not None:
                _cpu = getattr(bid, '_pool_cpu', None)
                _cpu_counts = (np.asarray(_cpu[0], dtype=np.int64)
                               if _cpu is not None else None)
                # `sum(numpy_row)` walks the row element by element in Python
                # and is O(blocks) per layer per step; the vectorised sum is
                # the same exact integer total and keeps the long-sequence
                # block_size=1 arm linear in C rather than Python.
                if (_cpu_counts is not None
                        and int(_cpu_counts.sum()) == T):
                    # host-side integer equivalence (design notes §6.2): cut
                    # positions are the prefix sums of the same counts table
                    # that generated bid — the identical integers to the
                    # nonzero(diff)+1 form.  Consumers accept numpy rows
                    # directly (compression_report's _tcut_ix split and
                    # boundary_alignment's np.asarray).
                    self._cuts.append(np.cumsum(_cpu_counts)[:-1])
                else:
                    cuts = (bid[1:] != bid[:-1]).nonzero(as_tuple=True)[0] + 1
                    # Keep the device tensor for compression_report.
                    self._cuts.append(cuts)
        # Normalizing an all-zero content table is exactly zero (0/eps == 0),
        # so the zero-content ablation keeps the same values.
        core_kv = (torch.zeros_like(comp_kv) if cfg.content_mode == 'zero'
                   else F.normalize(_attn_comp_kv, dim=-1))
        kv_local = F.normalize(Ca_raw, dim=-1)
        if x.device.type == 'cuda':
            # Fused K/V head (CUDA only — see _pre): every output row is a
            # function of its own input row only, so the split halves are
            # bit-identical to the two self.W_kvhead(...) calls (row
            # independence, design notes §6.1).
            _Bn = int(core_kv.shape[0])
            _kvh = self.W_kvhead(torch.cat([core_kv, kv_local], 0))
            # Row-contiguous [blocks | window] view of the V columns: the same
            # values as cat([v_blk, v_sw]).  The views keep `_kvh` alive exactly
            # as v_blk/v_sw already did.
            v_stacked = _kvh[:, self.nh * self.hd:].view(_Bn + T, self.nh, self.hd)
            v_blk = v_stacked[:_Bn]
            v_sw = v_stacked[_Bn:]
            # Normalize both K halves over the fused K columns: each row's
            # norm/divide is independent, so the values are the same as the two
            # per-half normalizations.  The result is one contiguous
            # [blocks | window] table.  `_kvh` stays alive through v_stacked.
            k_stacked = F.normalize(_kvh[:, :self.nh * self.hd].view(_Bn + T, self.nh, self.hd), dim=-1)
            k_blk = k_stacked[:_Bn]
            k_sw = k_stacked[_Bn:]
            del _kvh
        else:
            k_blk, v_blk = self._split(self.W_kvhead(core_kv))
            k_sw, v_sw = self._split(self.W_kvhead(kv_local))
            v_stacked = None
            k_stacked = None
            k_blk = F.normalize(k_blk, dim=-1)
            k_sw = F.normalize(k_sw, dim=-1)
        _prefix_k = _prefix_v = None
        if _prefix_comp is not None:
            _pcore = (torch.zeros_like(_prefix_comp) if cfg.content_mode == 'zero'
                      else F.normalize(_prefix_comp, dim=-1))
            _pk, _pv = self._split(self.W_kvhead(_pcore))
            _prefix_k = F.normalize(_pk, dim=-1)
            _prefix_v = _pv
        _prefix_mass = None
        if cfg.kind == 'hca':
            if _open_mode == 'closed':
                # cached int32 expansion — the same exact integers as the int64
                # arange expand narrowed to int32 (index transport, §6.1).  Every
                # row of the arange expansion holds distinct block ids, so the
                # first-occurrence dedup inside gathered_attention is provably the
                # all-True mask and is skipped (identical keep mask).
                topk_idx = _arange_exp_cache(T, Bn, x.device)
                out = _gathered_attention_trainable(q, k_blk, v_blk, topk_idx, last_tok, k_sw, v_sw, cfg.sliding_window, scale, sink_logits=self.sink if cfg.use_sink else None, topk_distinct=True, v_stacked=v_stacked, k_stacked=k_stacked)
            else:
                # HCA has no top-k: every closed block is read.  Adding a
                # prefix candidate is an ADDITIONAL read slot; the config
                # validation refuses to label this `shared_topk`.  The main
                # open-block panel uses CSA layers so the three arms share a
                # fixed top-k budget.
                _P = 1 if _prefix_kind == 'summary' else int(_prefix_geo['token_idx'].shape[1])
                _tot = int(Bn) + int(_P)
                topk_idx = _arange_exp_cache(T, _tot, x.device)
                # The closed half still follows the strict closed-block gate
                # (`pos > last_tok`): HCA reads every *finished* block, not
                # the still-open current block whose pooled K/V contains
                # future tokens.  The prefix half obeys its own causal mask:
                # summary rows with an empty represented prefix and singleton
                # columns at j >= plen must never be read (the singleton
                # table pads by clamping to the last token).  Marking these
                # slots valid would let a HCA arm attend to future tokens, so
                # use exactly the gate the CSA path applies.
                _pos = _arange_cache(T, x.device)
                _closed_valid = _pos[:, None] > last_tok[None, :]
                if _prefix_kind == 'summary':
                    _pref_valid = (_prefix_geo['plen'][:, None] > 0)
                else:
                    _pref_valid = _prefix_geo['valid']
                _sel_valid = torch.cat([_closed_valid, _pref_valid], 1)
                out, _prefix_mass = _gathered_attention_open_trainable(
                    q, k_blk, v_blk, topk_idx, _sel_valid, last_tok, k_sw, v_sw,
                    cfg.sliding_window, scale, prefix_geo=_prefix_geo,
                    prefix_kind=_prefix_kind, prefix_k=_prefix_k, prefix_v=_prefix_v,
                    sink_logits=self.sink if cfg.use_sink else None,
                    return_prefix_mass=(self._prefix_stats is not None))
        else:
            topk = cfg.index_topk
            _need_soft = torch.is_grad_enabled()
            if _open_mode == 'closed':
                _sel_valid_box = []
                # 无梯度（评估）时 gathered_attention 的 soft 直通分支整体是死
                # 的（_soft_live=False 见其注释），索引器据此不物化 soft 尾部，
                # 并以 soft_form=True 固定走软分支的硬部分解——两者前向 bit 与
                # 传入 soft 的形式完全一致。有梯度时条件为 False，走传入 soft
                # 的完整图。
                _, topk_idx, soft = _indexer_trainable(x, index_kv, last_tok, self.W_DQ, self.W_DK, self.W_w.weight, cfg.n_index_heads, topk, random_select=cfg.indexer_mode == 'random', pre_qI=pre['qI'] if pre is not None else None, pre_w=pre['w_idx'] if pre is not None else None, out_valid=_sel_valid_box, causal_ends_host=None, need_soft=_need_soft, random_salt=getattr(cfg, 'random_salt', 0))
                # _indexer_selection 的 top-k 是无放回选择：有效槽位的块 id 逐行互不
                # 相同（order 是列排列、kept 落在互异的 cumsum 秩列上），pad 槽位的
                # 重复块 0 已被 sel_valid 置无效。故 gathered_attention 的
                # first_occurrence 去重掩码恒为全真，直接声明 topk_distinct 跳过它
                # （keep 掩码与逐位计算结果一致，下游取值不变）。
                out = _gathered_attention_trainable(q, k_blk, v_blk, topk_idx, last_tok, k_sw, v_sw, cfg.sliding_window, scale, soft=soft, sink_logits=self.sink if cfg.use_sink else None, sel_valid=_sel_valid_box[0], topk_distinct=True, soft_form=True, v_stacked=v_stacked, k_stacked=k_stacked)
            else:
                if cfg.indexer_mode != 'learned':
                    raise RuntimeError('open_block_mode requires the learned indexer: the random indexer has no comparable candidate score for the prefix path')
                topk_idx, _sel_valid, soft = lightning_indexer_open(
                    x, index_kv, last_tok, self.W_DQ, self.W_DK,
                    self.W_w.weight, cfg.n_index_heads, topk,
                    prefix_geo=_prefix_geo, prefix_kv=_prefix_comp,
                    prefix_token_kv=Ca_raw, prefix_kind=_prefix_kind,
                    pre_qI=pre['qI'] if pre is not None else None,
                    pre_w=pre['w_idx'] if pre is not None else None,
                    need_soft=_need_soft)
                out, _prefix_mass = _gathered_attention_open_trainable(
                    q, k_blk, v_blk, topk_idx, _sel_valid, last_tok, k_sw, v_sw,
                    cfg.sliding_window, scale, prefix_geo=_prefix_geo,
                    prefix_kind=_prefix_kind, prefix_k=_prefix_k, prefix_v=_prefix_v,
                    soft=soft, sink_logits=self.sink if cfg.use_sink else None,
                    return_prefix_mass=(self._prefix_stats is not None))
        if _open_mode != 'closed' and self._prefix_stats is not None:
            self._prefix_stats.append_from_layer(
                _prefix_geo, T, topk_idx, _sel_valid, self.cfg,
                prefix_mass=_prefix_mass, closed_blocks=int(Bn))
        return (self.W_o(out.reshape(T, self.nh * self.hd)), gate_mean)

    def forward(self, x):
        B, T, d = x.shape
        if B == 0 or T == 0:
            # No tokens (or no rows): every attention kind returns the same
            # empty [B, T, d] result.  The pooling/indexer paths are defined
            # for empty input, but segmenter-derived tensors have no extrema
            # to read, so short-circuit before touching them.
            self.last_gate_mean = None
            return x.new_empty((B, T, d))
        if self.cfg.kind == 'full':
            self.last_gate_mean = None
            return self._full_batched(x)
        flat = x.reshape(B * T, d)
        pre_flat = self._pre(flat, fuse_x=x)
        _open_mode = getattr(self.cfg, 'open_block_mode', 'closed')
        _kernel_path = _CK is not None and self.cfg.kind in ('csa', 'hca') and _CK.kernel_enabled()
        if _kernel_path and _open_mode != 'closed':
            raise RuntimeError(
                'CSA_KERNEL does not implement open_block_mode != "closed"; '
                'the prefix candidate construction and its mask are only '
                'implemented in the eager reference path.  Unset CSA_KERNEL '
                'for open-block arms rather than silently running a different '
                'read rule.')
        if _kernel_path and getattr(self.cfg, 'max_block', 128) is None:
            raise RuntimeError(
                'CSA_KERNEL requires an explicit max_block; the no-cap '
                'configuration is supported only by the eager segmenter. '
                'Unset CSA_KERNEL for the max_block=None arms.')
        if _kernel_path and (int(getattr(self.cfg, 'max_block', 128)) != 128
                             or int(getattr(self.cfg, 'position_capacity', lambda: 129)()) != 129):
            raise RuntimeError(
                'CSA_KERNEL was written for the default (max_block=128, '
                'pool_position_capacity=129) geometry; its pool-geometry '
                'cache keys on max_block and would not carry the decoupled '
                'position capacity.  Unset CSA_KERNEL for the open-block '
                'panel (all arms use the eager row path).')
        # Batched eager path: the reference twin consumes [B, T, ...] and
        # issues batch GEMMs.  It is the default for B > 1 on CUDA (values
        # follow the declared reference-backend tolerance, not the eager
        # bitwise contract); CSA_EAGER_BATCH=0 restores the row loop.
        # Open-block arms force the row path: the twin's per-query prefix
        # construction has not been mirrored there, and mixing implementations
        # inside one panel would confound the read rule with the backend.
        _eager_batch = (_CK is not None and not _kernel_path
                        and x.device.type == 'cuda' and B > 1
                        and self.cfg.kind in ('csa', 'hca')
                        and self.cfg.fuse_mode == 'none'
                        and self.cfg.indexer_mode == 'learned'
                        and _open_mode == 'closed'
                        and int(getattr(self.cfg, 'max_block', 128) or 0) == 128
                        and int(getattr(self.cfg, 'position_capacity', lambda: 129)()) == 129
                        and _eager_batch_active(x.device, B))
        _batch_path = _kernel_path or _eager_batch
        pre_all = None
        pres = None
        if pre_flat is not None:
            if _batch_path:
                # The kernel / batched twin consumes the whole [B, T, ...]
                # batch and returns before the per-row loop, so the per-row
                # projection views are not built on this path.
                pre_all = {}
                for kk, v in pre_flat.items():
                    if v is None:
                        pre_all[kk] = None
                    elif kk == 'fused':
                        pre_all[kk] = v
                    else:
                        pre_all[kk] = v.reshape(B, T, *v.shape[1:])
            else:
                # The eager row path reads only its own [T, ...] slice of the
                # fused projection; each row is sliced from the flat output.
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
        gate_sigs = None
        pre_blocks = None
        # The batched gate-signal / block-geometry hoist below is value-identical
        # to the per-row inline form for ANY batch size (design notes §6.1), so
        # it also serves B == 1 (e.g. the seq-2048 topk sweep).
        if self.cfg.chunking == 'cosine_learnable':
            _gs = self._gate_signals(x, pre_all)
            _raw, _gbool = _gs[0], _gs[1]
            if all((_r is not None for _r in _raw)):
                if _gbool is not None and _device_segment_enabled(x.device):
                    # Device-side segmentation: the gate decisions and the
                    # resulting block ids stay on the GPU.
                    gate_sigs = [(tuple(_r) if len(_r) == 3 else (_r[0], _g))
                                 for _r, _g in zip(_raw, _gbool.unbind(0))]
                    pre_blocks = _device_blocks_from_gate_bool(
                        T, _gbool, self.cfg.min_block, self.cfg.max_block, x.device,
                        want_honoured=self.need_reg, hon_dtype=_raw[0][0].dtype)
                else:
                    # .numpy() is a view of the stacked CPU tensor (the
                    # base tensor stays alive through the views); the segmenter
                    # consumes the same booleans.
                    _gl = _gbool.cpu().numpy()
                    gate_sigs = [(tuple(_r) if len(_r) == 3 else (_r[0], _g)) for _r, _g in zip(_raw, _gl)]
                    # Batched block geometry for the whole batch; every value is
                    # identical to the per-row form (see _hoist_blocks_from_cuts).
                    pre_blocks = _hoist_blocks_from_cuts(T, _gl, self.cfg.min_block, self.cfg.max_block, x.device, want_honoured=self.need_reg, hon_dtype=_raw[0][0].dtype)
        elif self.cfg.chunking in ('cosine_abs', 'cosine_adaptive') or (self.cfg.dynamic and self.cfg.kind != 'hca'):
            # same batch-level geometry hoist for the fixed/adaptive cosine
            # paths (the cut booleans and the resulting block ids are identical
            # to the per-row form).
            pre_blocks = self._cut_blocks_batched(x)
        if _eager_batch:
            return _CK.sparse_layer_forward(self, x, pre_all, gate_sigs, pre_blocks, impl='ref')
        if _kernel_path:
            # 内核路径（CSA_KERNEL；独立变体，与 eager 统计对照，不逐位承诺）。
            # 不支持的臂在这里硬失败，绝不静默回退——否则同一面板会混入两种
            # 实现的记录。
            if self.cfg.fuse_mode != 'none' or self.cfg.indexer_mode != 'learned':
                raise RuntimeError('CSA_KERNEL 只覆盖 learned-indexer / 非 fuse 的 csa·hca 臂；'
                                   'fuse_mode 或 indexer_mode 不同的臂请 unset CSA_KERNEL 后走 eager')
            return _CK.sparse_layer_forward(self, x, pre_all, gate_sigs, pre_blocks)
        if pres is None:
            pres = [None] * B
        if pre_blocks is not None:
            _prefetch_pool_geometry(self, pre_blocks, T, x.device)
        # Write each row's attention output into the batch output: the stacked
        # buffer holds exactly the tensor torch.stack would have built (a
        # per-row copy into the same layout), and the CopySlices backward hands
        # each row the identical grad slice — value-identical to the stack form
        # (transport only, design notes §6.1).
        outs = torch.empty(B, T, d, device=x.device, dtype=x.dtype)
        gates = []
        _prev_batch = _CPU_ACTIVE_BATCH[0]
        _CPU_ACTIVE_BATCH[0] = int(B)
        _blp = None
        _bdec = 0
        try:
            for b in range(B):
                pre = pres[b]
                o, g = self._single(x[b], pre=pre, gate_sig=None if gate_sigs is None else gate_sigs[b], pre_blocks=None if pre_blocks is None else pre_blocks[b])
                outs[b] = o
                if g is not None:
                    gates.append(g)
                # `_single` resets these per row; the row loop used to leave
                # only the LAST row's REINFORCE log-probability, so a batch
                # trained on a fraction of its boundary decisions.  Accumulate
                # all rows (the batched/kernel path sums them too).
                _lp = getattr(self, 'last_boundary_logp', None)
                if _lp is not None:
                    _blp = _lp if _blp is None else _blp + _lp
                    _bdec += int(getattr(self, 'last_boundary_decisions', 0) or 0)
        finally:
            _CPU_ACTIVE_BATCH[0] = _prev_batch
        self.last_boundary_logp = _blp
        self.last_boundary_decisions = int(_bdec)
        if not gates:
            self.last_gate_mean = None
        else:
            gm_tot = gates[0]
            for _g in gates[1:]:
                gm_tot = gm_tot + _g
            self.last_gate_mean = gm_tot / len(gates)
        return outs
