
def _pool_geom_flat_single(counts_l, B, n, max_len, p_rows, pad_to=1):
    # Single-block-form twin of the variable-block host builder behind
    # pool_blocks_single (the same g/mask/pos_a flats and exact-integer host
    # build, §6.2).  Returns (flat_i, mask_np, n_g, max_len) or None.
    try:
        B = int(B)
        n = int(n)
        max_len = int(max_len)
        p_rows = int(p_rows)
        _raw_counts = np.asarray(counts_l)
        if _raw_counts.shape != (B,) or n <= 0 or max_len <= 0 or p_rows <= 0:
            return None
        _cmax = int(_raw_counts.max()) if _raw_counts.size else 0
        _dt = _pool_host_idx_dtype(n, max_len, _cmax, p_rows)
        counts_np = _raw_counts.astype(_dt, copy=False)
        _ck = ('s', counts_np.dtype.str, counts_np.tobytes(), B, n, max_len, p_rows, int(pad_to)) if B <= 1024 else None
        _hit = _pool_geom_content_get(_ck)
        if _hit is not None:
            return _hit
        ends_np = np.cumsum(counts_np, dtype=_dt)
        starts_np = ends_np - counts_np
        pos = _np_range(max_len).astype(_dt, copy=False)
        n_g = B * max_len
        _pi = (-(n_g + max_len)) % int(pad_to) if int(pad_to) > 1 else 0
        _pb = (-n_g) % int(pad_to) if int(pad_to) > 1 else 0
        flat_i = np.empty(n_g + max_len + _pi, dtype=_dt)
        mask_np = np.empty(n_g + _pb, dtype=np.bool_)
        gp = flat_i[0:n_g].reshape(B, max_len)
        np.add(starts_np[:, None], pos[None, :], out=gp)
        np.less(gp, ends_np[:, None], out=mask_np[0:n_g].reshape(B, max_len))
        np.minimum(gp, n - 1, out=gp)
        pos_a = flat_i[n_g:n_g + max_len]
        if max_len <= p_rows:
            pos_a[:] = pos
        else:
            np.minimum(pos, p_rows - 1, out=pos_a)
        return _pool_geom_content_put(_ck, (flat_i, mask_np, n_g, max_len))
    except (RuntimeError, TypeError, ValueError, OverflowError, MemoryError, AttributeError):
        return None

def _pool_geom_host_single(counts_l, B, n, max_len, p_rows, dev):
    # Single-block-form twin of _pool_geom_host_var (pool_blocks_single's
    # g/mask/_pos_a): same exact-integer host build (§6.2), same staged
    # transport (§6.1) with bucket-padded flats (bounded pool shape keys,
    # unread pad tails), same None-means-device-fallback contract.
    _built = _pool_geom_flat_single(counts_l, B, n, max_len, p_rows, _POOL_GEOM_BUCKET)
    if _built is None:
        return None
    try:
        B = int(B)
        flat_i, mask_np, n_g, max_len = _built
        # Both flats are already bucket-padded.
        _idt = torch.int64 if flat_i.dtype == np.int64 else torch.int32
        buf = _h2d_async(flat_i, _idt, dev)
        mask = _h2d_async(mask_np, torch.bool, dev)[:n_g].view(B, max_len)
        return (buf[0:n_g].view(B, max_len), mask, buf[n_g:n_g + max_len])
    except (RuntimeError, TypeError, ValueError, OverflowError, MemoryError, AttributeError):
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
        _pkey = (nblk, ov if _csa else -1, int(n))
        _gkey = (('vo', nblk, ov, int(n), pa_rows, pb_rows) if _csa
                 else ('so', nblk, int(n), pa_rows))
        _pc = getattr(bid, '_pool_gpu', None)
        _gc = getattr(bid, '_pool_geom', None)
        if _pc is not None and _gc is not None and _pkey in _pc and _gkey in _gc:
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
    _tot_i = 0
    _tot_b = 0
    for _r in rows:
        if _r is None:
            continue
        _tot_i += len(_r[2][0]) + len(_r[3][0])
        _tot_b += len(_r[3][1].ravel())
    _idt = np.int64 if any(
        _r is not None and (
            _r[2][0].dtype == np.dtype(np.int64)
            or _r[3][0].dtype == np.dtype(np.int64))
        for _r in rows) else np.int32
    flat_i = np.empty(_tot_i + (-_tot_i) % _POOL_GEOM_BUCKET, dtype=_idt)
    flat_b = np.empty(_tot_b + (-_tot_b) % _POOL_GEOM_BUCKET, dtype=np.bool_)
    _oi = 0
    _ob = 0
    for _r in rows:
        if _r is None:
            continue
        _pflat = _r[2][0]
        _gflat = _r[3][0]
        _bflat = _r[3][1].ravel()
        _n = len(_pflat)
        flat_i[_oi:_oi + _n] = _pflat
        _oi += _n
        _n = len(_gflat)
        flat_i[_oi:_oi + _n] = _gflat
        _oi += _n
        _n = len(_bflat)
        flat_b[_ob:_ob + _n] = _bflat
        _ob += _n
    buf_i = _h2d_async(flat_i, torch.int64 if _idt == np.dtype(np.int64) else torch.int32, dev)
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
    _identity = bool(order_is_identity)
    _cpu = getattr(block_ids, '_pool_cpu', None)
    if _cpu is not None and len(_cpu[0]) == B:
        max_len = _cpu[1]
        _ov_hi = min(ov, _cpu[2]) if B > 1 else -1
        _gkey = ('vo', int(B), int(ov), int(n), int(B_pos_a.shape[0]), int(B_pos_b.shape[0]))
        _geom = getattr(block_ids, '_pool_geom', None)
        # The cached geometry stores bias rows as within-sorted-block ranks,
        # which is only the token's within-block offset when the block ids are
        # in their original (monotone) order.  A sorted non-monotone layout
        # needs the per-block bias table built below.
        _geom = (None if (_geom is None or not _identity)
                 else _geom.get(_gkey))
    elif ov_len.numel():
        # Only the two maxima are consumed below; fold them into one stacked
        # read and discard the minima.
        max_len, _ov_hi = (int(v) for v in torch.stack([counts.max(), ov_len.max()]).tolist())
        _geom = None
    else:
        max_len = int(counts.max())
        _ov_hi = None
        _geom = None
    if _geom is not None:
        g_a, mask_a, g_b, mask_b, b_off, _pos_a = _geom
    else:
        # Host form first (the same exact-integer build _pool_parts_host makes
        # for the parts — design notes §6.2): the host form produces values
        # identical to the on-device build below.  None means fall back to the
        # on-device form, whose values are identical.
        _host = None
        if _identity and _POOL_PARTS_HOST and _cpu is not None and len(_cpu[0]) == B:
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
            if _identity and _cpu is not None and len(_cpu[0]) == B:
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
    # Within the embedding table the gather index is exactly arange(max_len)
    # only when the row order is the original token order.  With a sorted
    # non-monotone layout every block's bias index must be recomputed from the
    # block's ORIGINAL token positions: the sorted lane rank is not
    # `token_position - block_start` when a block's tokens are non-contiguous.
    _bias_a = _bias_b = None
    if not _identity:
        _orig = order
        _ja = _range_cache(0, int(g_a.shape[1]), dev)
        _rank_a = (starts[:, None] + _ja[None, :]).clamp(max=n - 1)
        _start_a = _orig[starts.clamp(max=n - 1)][:, None]
        _bias_a = (_orig[_rank_a] - _start_a).clamp(
            min=0, max=B_pos_a.shape[0] - 1)
        if _Wb > 0:
            _jb = _range_cache(0, _Wb, dev)
            _rank_b = (ov_start[:, None] + _jb[None, :]).clamp(max=n - 1)
            _start_b = _orig[starts_prev.clamp(max=n - 1)][:, None]
            _bias_b = (_orig[_rank_b] - _start_b).clamp(
                min=0, max=B_pos_b.shape[0] - 1)
    _bpa = B_pos_a[:max_len] if max_len <= B_pos_a.shape[0] else B_pos_a[_pos_a]
    # Every real block carries at least one token (the segmenters only emit
    # positive counts), so the A lanes always contain a finite logit and the
    # softmax cannot produce NaN/Inf.  When the block-ids object exposes its
    # host counts we can prove that here, so the defensive nan_to_num is
    # unnecessary; the fallback keeps the defensive form for externally built
    # tables.
    _clean = _pool_counts_clean(block_ids, B)
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
        # form.
        # += on the fresh _take_2d outputs: the same elementwise kernel with the
        # same values as the out-of-place add (AddBackward saves no tensor, and
        # the only downstream consumer chain — masked_fill_/cat/softmax — keeps
        # no pre-add version).
        if _bias_a is None:
            Zam += _bpa[None]
        else:
            Zam += B_pos_a[_bias_a[_b0:_b1]]
        if _bias_b is None:
            Zbm += B_pos_b[b_off[_b0:_b1]]
        else:
            Zbm += B_pos_b[_bias_b[_b0:_b1]]
        # the fill mask broadcasts: masked_fill_ accepts a broadcastable mask,
        # so the same booleans reach every lane (broadcast reads are exact).
        Zam.masked_fill_(~_ma[:, :, None], float('-inf'))
        Zbm.masked_fill_(~_mb[:, :, None], float('-inf'))
        WinZ = torch.cat([Zam, Zbm], 1)
        # softmax 的反向保存其输出，nan_to_num 必须出位（原地会改写保存值，
        # SoftmaxBackward 版本检查失败）；值与任何原地形式逐位相同。
        # The safe-count path uses the finite softmax directly.
        sc = F.softmax(WinZ, dim=1) if _clean else _safe_nan_to_num(F.softmax(WinZ, dim=1), where='block_pool')
        del WinZ, Zam, Zbm
        Xam = _take_2d(Xas, g_a[_b0:_b1])
        Xbm = _take_2d(Xbs, g_b[_b0:_b1])
        sc_a = sc[:, :max_len]
        sc_b = sc[:, max_len:]
        part = (sc_a * Xam).sum(1)
        # Release the A-lane gather before the B-lane product; the two partial
        # sums are added in the original order.
        del Xam
        part = part + (sc_b * Xbm).sum(1)
        comp[_b0:_b1] = part
        del sc, sc_a, sc_b, Xbm, _ma, _mb, part
    # Identity-order path: when `order` is the cached arange, order[last_idx]
    # returns exactly last_idx (arange[i] == i, exact integers — index
    # transport, design notes §6.1).
    return (comp, last_idx if order_is_identity else order[last_idx], B)

def pool_blocks_single(X, Z, B_pos, block_ids, n_blocks=None, monotonic=False):
    n = X.shape[0]
    dev = X.device
    if n == 0:
        # Same empty-sequence contract as pool_variable_blocks.
        return (X[:0], torch.empty(0, dtype=torch.long, device=dev), 0)
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
        # The cached bias rows are within-sorted-block ranks; only an identity
        # row order makes them the token's within-block offsets.
        _geom = (None if (_gc is None or not _ident)
                 else _gc.get(_gkey))
    else:
        max_len = int(counts.max())
    if _geom is not None:
        g, mask, _pos_a = _geom
    else:
        # host form first — same exact-integer argument as _pool_ordered (the
        # values are identical to the device build; only the location of the
        # integer math and the transport change, §6.1/§6.2).
        _host = None
        if _ident and _POOL_PARTS_HOST and _cpu is not None and len(_cpu[0]) == B:
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
            if _ident and _cpu is not None and len(_cpu[0]) == B:
                _pool_geom_put(block_ids, _gkey, (g, mask, _pos_a))
    # Block-chunked exactly as _pool_ordered: each block's pooled vector is a
    # function of its own lanes only, so the grouping never changes a value.
    _Fd = int(Xs.shape[1])
    # Conservative per-block element count: Xb + Zb(WinZ) + sc plus the mask.
    _per_elems = 3 * int(max_len) * _Fd + 4 * int(max_len)
    _chunk = _pool_block_chunk(B, _per_elems, Xs.element_size())
    # Same slice-view form as _pool_ordered (index == arange(max_len)) only for
    # an identity row order.  A sorted non-monotone layout needs each block's
    # bias index recomputed from the ORIGINAL token positions.
    _bias = None
    if not _ident:
        _orig = order
        _j = _range_cache(0, int(g.shape[1]), dev)
        _rank = (starts[:, None] + _j[None, :]).clamp(max=n - 1)
        _start = _orig[starts.clamp(max=n - 1)][:, None]
        _bias = (_orig[_rank] - _start).clamp(
            min=0, max=B_pos.shape[0] - 1)
    _bpos = B_pos[:max_len] if max_len <= B_pos.shape[0] else B_pos[_pos_a]
    _clean = _pool_counts_clean(block_ids, B)
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
        if _bias is None:
            WinZ += _bpos[None]
        else:
            WinZ += B_pos[_bias[_b0:_b1]]
        # broadcast fill mask (same argument as _pool_ordered): identical booleans
        # per lane.
        WinZ.masked_fill_(~mask[_b0:_b1, :, None], float('-inf'))
        # 同 _pool_ordered：softmax 反向保存输出，nan_to_num 出位。
        sc = F.softmax(WinZ, dim=1) if _clean else _safe_nan_to_num(F.softmax(WinZ, dim=1), where='block_pool')
        comp[_b0:_b1] = (sc * Xb).sum(1)
        del WinZ, Xb, Zb, sc
    # same identity-order path as _pool_ordered: order[last_idx] over the
    # cached arange returns exactly last_idx (index transport, §6.1).
    return (comp, last_idx if _ident else order[last_idx], B)

# ---------------------------------------------------------------------------
# Causal open-block (prefix) representations
# ---------------------------------------------------------------------------
# The prefix of the CURRENT (still open) block is a per-query temporary
# representation: P_t = [s_t, t].  It must never change the dynamic cut table,
# and every input to it is causal.  In particular we never use the block's
# final length/end or a position bias computed from the block tail: the block
# geometry is read from the segmenter's per-token ids, and the only positions
# used are `u - s_t` for u <= t.
#
# `causal_prefix_pool` is the optimized implementation.  It is mathematically
# the same softmax mixture as the repository's block pooling, but it computes
# every prefix in one pass with a cumulative sum:
#
#   comp_t = (sum_{u in P_t} e_u X_u) / (sum_{u in P_t} e_u),  e_u = exp(z_u)
#
# where z_u = Z_u + B_pos_a[u - s_t] (plus the same previous-block overlap
# lanes the closed block pooling uses, for the full-prefix scope).  The
# cumulative sums are taken within a block by subtracting the cumulative sum
# at the prefix start, so no future token enters any row.  A per-query
# reference implementation lives next to it and the test suite checks forward
# values and gradients against it.

def _open_block_geometry(bid, n_blocks=None):
    """Block starts/ends and the per-token in-block offset.

    `bid` is the segmenter's monotone, contiguous block-id row.  The returned
    integers are exact functions of that row; no block-tail quantity is read.
    The geometry is therefore cached on the immutable block-id row: every
    open-block layer reads the same bid object, so the bincount/cumsum/gather
    work is shared instead of rebuilt per layer.
    """
    if bid.dim() != 1:
        raise ValueError(f'_open_block_geometry expects a 1-D block-id row, got {tuple(bid.shape)}')
    _key = -1 if n_blocks is None else int(n_blocks)
    _cache = getattr(bid, '_open_geo_cache', None)
    if _cache is not None:
        _hit = _cache.get(_key)
        if _hit is not None:
            return _hit
    n = int(bid.numel())
    if n == 0:
        z = bid.new_empty((0,))
        geo = {'n': 0, 'n_blocks': 0, 'counts': z, 'starts': z, 'ends': z,
               'start_of_token': z, 'offset': z, 'block_of_token': z}
    else:
        B = int(n_blocks) if n_blocks is not None else int(bid.max().item()) + 1
        bid_long = bid.long()
        counts = torch.bincount(bid.reshape(-1).long(), minlength=B)[:B]
        ends = torch.cumsum(counts, 0)
        starts = ends - counts
        start_of_token = starts[bid_long]
        offset = _arange_cache(n, bid.device) - start_of_token
        geo = {'n': n, 'n_blocks': B, 'counts': counts, 'starts': starts,
               'ends': ends, 'start_of_token': start_of_token, 'offset': offset,
               'block_of_token': bid_long}
    try:
        if _cache is None:
            _cache = {}
            bid._open_geo_cache = _cache
        if len(_cache) < 8:
            _cache[_key] = geo
    except (AttributeError, RuntimeError):
        pass
    return geo


def _open_prefix_geometry(bid, n_blocks=None, scope='full_prefix', window=128,
                          window_start=None, candidate_table=True):
    """Per-query prefix start/length and the candidate token table.

    Returns a dict with:
      start_of_token, block_of_token, offset  — block geometry
      u_start  [T]  first token of the represented prefix (always s_t; the
                    scope changes where it ENDS, not where it starts)
      plen     [T]  number of represented tokens (may be 0 outside-window)
      token_idx [T, P] candidate token ids for `prefix_singletons` (P = max
                    plen; padded rows are clamped and masked by plen).  With
                    `candidate_table=False` (the summary path) this is a
                    single per-query row, not an O(T^2) table.
      valid    [T, P] candidate validity (j < plen, or plen>0 for summaries)
    """
    _ckey = None
    if window_start is None:
        _ckey = (int(n_blocks) if n_blocks is not None else -1,
                 str(scope), int(window), bool(candidate_table))
        _pcache = getattr(bid, '_open_prefix_cache', None)
        if _pcache is not None:
            _hit = _pcache.get(_ckey)
            if _hit is not None:
                return _hit
    geo = _open_block_geometry(bid, n_blocks=n_blocks)
    # Drop the private cache scalars attached to the shared geometry dict.
    _base_geo = {k: v for k, v in geo.items()
                 if not (isinstance(k, str) and k.startswith('_'))}
    n, B = int(geo['n']), int(geo['n_blocks'])
    dev = bid.device
    if n == 0:
        z = bid.new_empty((0,))
        out = {'u_start': z, 'plen': z, 'token_idx': z[:, None],
               'valid': torch.zeros((0, 1), dtype=torch.bool, device=dev),
               **_base_geo}
        return out
    t = _arange_cache(n, dev)
    s = geo['start_of_token']
    if str(scope) == 'full_prefix':
        u_start = s
        plen = (t - s + 1).clamp_min(0)
        win_start_t = None
    elif str(scope) == 'outside_window':
        if window_start is None:
            w = int(window)
            # W_t = [max(0, t-w+1), t]; w <= 0 means no local window, so the
            # whole prefix is outside it (window start t+1).
            win_start_t = ((t - w + 1) if w > 0 else (t + 1)).clamp_min(0)
        else:
            win_start_t = window_start.to(dev).long()
        # U_t = P_t \ W_t = [s_t, min(t, window_start-1)].  The represented
        # prefix ALWAYS starts at the block start; what changes is where it
        # ends (immediately before the local window), never a suffix inside
        # the window.
        u_start = s
        pend = torch.minimum(t, win_start_t - 1)
        plen = (pend - s + 1).clamp_min(0)
    else:
        raise ValueError(f'unknown open_block_scope {scope!r}')
    # `prefix_summary` needs only ONE candidate per query; only the singleton
    # table is [T, P].  Never materialise an O(T^2) table for a summary just
    # because a no-cap block is long.  The maximum prefix length is read from
    # the device only when the singleton table actually needs its width, so
    # the summary path performs no GPU->CPU synchronisation.
    if not candidate_table:
        P = 1
    else:
        P = int(plen.max().item()) if n else 0
        # A full-prefix singleton table is [T, P]; for a pathological no-cap
        # block this can be O(T^2).  Fail explicitly instead of silently
        # exhausting GPU memory; the panel reports candidate growth and can
        # cap/evaluate in shorter slices.
        if n * max(P, 1) > (16 << 20):
            raise RuntimeError(
                f'open-block prefix candidate table would be {n}x{P} '
                f'({n * max(P, 1)} elements); reduce max_block_tokens, use '
                'prefix_summary instead of prefix_singletons, or evaluate in '
                'shorter sequence slices')
    if P <= 0:
        token_idx = torch.zeros((n, 0), dtype=torch.long, device=dev)
        valid = torch.zeros((n, 0), dtype=torch.bool, device=dev)
    elif not candidate_table:
        token_idx = u_start[:, None].clamp(max=max(n - 1, 0))
        valid = (plen > 0)[:, None]
    else:
        j = _arange_cache(P, dev)[None, :]
        token_idx = (u_start[:, None] + j).clamp(max=n - 1)
        valid = j < plen[:, None]
    out = {'u_start': u_start, 'plen': plen, 'token_idx': token_idx,
           'valid': valid, 'window_start': win_start_t, **_base_geo}
    if _ckey is not None:
        try:
            _pcache = getattr(bid, '_open_prefix_cache', None)
            if _pcache is None:
                _pcache = {}
                bid._open_prefix_cache = _pcache
            if len(_pcache) < 8:
                _pcache[_ckey] = out
        except (AttributeError, RuntimeError):
            pass
    return out


def _prefix_overlap_values(geo, Xb, Zb, B_pos_b, overlap):
    """Raw previous-block overlap lanes for every prefix summary.

    Returns ``(x_ov, z_ov, mask, u)`` or ``None`` when there is no overlap.
    Keeping the raw logits separate lets the caller choose one numerical
    shift for both the A lanes and these lanes before exponentiating.
    """
    B = int(geo['n_blocks'])
    dev = geo['starts'].device
    Fd = int(Zb.shape[1]) if Zb is not None else (int(Xb.shape[1]) if Xb is not None else 0)
    if int(overlap) <= 0 or B <= 1 or Xb is None or Zb is None:
        return None
    starts, ends = geo['starts'], geo['ends']
    prev_start = starts[:-1]
    prev_end = ends[:-1]
    ov_start_prev = (prev_end - int(overlap)).clamp_min(prev_start)
    ov_len_prev = (prev_end - ov_start_prev).clamp_min(0)
    # block 0 has no previous block; the other rows take previous-block values
    ov_start = torch.cat([starts[:1], ov_start_prev], 0)
    ov_len = torch.cat([torch.zeros_like(ov_len_prev[:1]), ov_len_prev], 0)
    prev_start_full = torch.cat([starts[:1], prev_start], 0)
    # The overlap-lane width is a pure function of (geo, overlap): cache it on
    # the shared geometry dict so the GPU->CPU maximum is read once per
    # forward instead of once per layer.
    _wkey = '_ov_w_' + str(int(overlap))
    Wb = geo.get(_wkey)
    if Wb is None:
        Wb = int(ov_len.max().item()) if B else 0
        geo[_wkey] = Wb
    if Wb <= 0:
        return None
    j = _arange_cache(Wb, dev)[None, :]
    mask = j < ov_len[:, None]
    u = (ov_start[:, None] + j).clamp(max=int(geo['n']) - 1)
    b_off = (ov_start[:, None] - prev_start_full[:, None] + j).clamp(
        min=0, max=int(B_pos_b.shape[0]) - 1)
    z_ov = Zb[u] + B_pos_b[b_off]
    x_ov = Xb[u]
    return x_ov, z_ov, mask, u


def _prefix_overlap_terms(geo, Xb, Zb, B_pos_b, overlap, shift, use_overlap,
                          values=None):
    """Previous-block overlap lanes for every prefix summary.

    The overlap convention is exactly the closed block pooling's: block b>0
    carries the last `overlap` tokens of block b-1, whose position bias index
    is `u - start_{b-1}`.  These tokens are all < s_b <= t, hence causal.
    The lanes are constant across the prefixes of one block, so they can be
    folded once into the per-block initial numerator/denominator.

    `shift` is a single per-block offset chosen to keep every exponent in
    range.  It cancels in the normalized mixture, so it changes no value.
    """
    B = int(geo['n_blocks'])
    dev = geo['starts'].device
    Fd = int(Zb.shape[1]) if Zb is not None else (int(Xb.shape[1]) if Xb is not None else 0)
    if (not use_overlap) or int(overlap) <= 0 or B <= 1 or Xb is None or Zb is None:
        z = geo['starts'].new_zeros((B, Fd), dtype=torch.float32)
        return z, z.clone(), torch.zeros((B, 0), dtype=torch.long, device=dev)
    if values is None:
        values = _prefix_overlap_values(geo, Xb, Zb, B_pos_b, overlap)
    if values is None:
        z = geo['starts'].new_zeros((B, Fd), dtype=torch.float32)
        return z, z.clone(), torch.zeros((B, 0), dtype=torch.long, device=dev)
    x_ov, z_ov, mask, u = values
    _dt = shift.dtype if shift.is_floating_point() else torch.float32
    # Mask before exp: a padded lane is clamped to a real token id, but its
    # logit may be far above the shift.  exp() must never see it.
    _z_eff = z_ov.to(_dt).masked_fill(~mask[:, :, None], float('-inf'))
    e = torch.exp(_z_eff - shift[:, None, :])
    e = e * mask[:, :, None].to(e.dtype)
    den = e.sum(1)
    num = (e * x_ov.to(_dt)).sum(1)
    return den, num, u


def causal_prefix_pool(Xa, Xb, Za, Zb, bid, B_pos_a, B_pos_b, overlap,
                       *, n_blocks=None, scope='full_prefix', window=128,
                       window_start=None, use_overlap=True):
    """Optimized all-query causal prefix pooling (see block comment above).

    Xa/Za are the A-lane content/position projections; Xb/Zb are the
    previous-block overlap content/position projections (None for HCA).
    Returns `comp` [T, Fd] whose row t is the pooling of [u_start_t, t] with
    the same weights and position biases as the closed block pooling.
    """
    if Xa.dim() != 2:
        raise ValueError(f'causal_prefix_pool expects Xa [T, Fd], got {tuple(Xa.shape)}')
    geo = _open_block_geometry(bid, n_blocks=n_blocks)
    T = int(geo['n'])
    Fd = int(Xa.shape[1])
    if T == 0:
        return Xa[:0]
    pg = _open_prefix_geometry(bid, n_blocks=int(geo['n_blocks']), scope=scope,
                               window=window, window_start=window_start,
                               candidate_table=False)
    cap = int(B_pos_a.shape[0])
    offset = geo['offset'].clamp(max=cap - 1)
    z_a = Za + B_pos_a[offset]
    # Overlap lanes are constant for every prefix inside a block, so build
    # their raw values once.  outside_window excludes them by definition.
    _ov_vals = None
    _want_ov = (use_overlap and str(scope) != 'outside_window'
                and Zb is not None and Xb is not None)
    if _want_ov:
        _ov_vals = _prefix_overlap_values(geo, Xb, Zb, B_pos_b, overlap)

    # Stable, causal segmented softmax prefix scan.  Each feature column is
    # an independent scalar softmax over the block prefix [s_b, t].  A
    # per-token (max, sum, weighted-sum) monoid is combined in log-space:
    # the shift is the maximum over the *prefix* itself, so no future token
    # can change any row and exp() never overflows.  Using a prefix-wide (not
    # block-wide) maximum also keeps an early prefix from underflowing when a
    # later token in the same block has a much larger logit.  The shifts are
    # detached constants, so the gradients are exactly those of the
    # normalized softmax.
    _acc = torch.float64
    _bt = geo['block_of_token']
    _m = z_a.detach().to(_acc)
    _e = torch.exp(z_a.to(_acc) - _m)
    _s = _e
    _n = _e * Xa.to(_acc)
    # Overlap lanes are constant for every prefix inside a block, so fold
    # them into that block's first token before the scan.  They are previous
    # block tokens (strictly past) and use the same monoid.
    if _want_ov and _ov_vals is not None:
        _x_ov, _z_ov, _ov_mask, _ = _ov_vals
        _z_ov64 = _z_ov.to(_acc)
        _z_ov_eff = _z_ov64.detach().masked_fill(~_ov_mask[:, :, None], float('-inf'))
        _m_ov = _z_ov_eff.amax(1)
        _has_ov = torch.isfinite(_m_ov).any(1)
        _m_ov_safe = torch.where(torch.isfinite(_m_ov), _m_ov, torch.zeros_like(_m_ov))
        _e_ov = torch.exp(_z_ov64.masked_fill(~_ov_mask[:, :, None], float('-inf'))
                          - _m_ov_safe[:, None, :])
        _s_ov = _e_ov.sum(1)
        _n_ov = (_e_ov * _x_ov.to(_acc)).sum(1)
        _is_start = torch.zeros(T, dtype=torch.bool, device=Xa.device)
        # An oversized n_blocks argument leaves trailing EMPTY blocks whose
        # start is n (one past the last token).  Those rows own no token, so
        # only real starts may be scattered into the [T] row table; indexing
        # with n would raise instead of ignoring the padding.
        _real_starts = geo['starts'][geo['starts'] < T]
        _is_start[_real_starts] = True
        _use_ov = _is_start & _has_ov[_bt]
        _m_ov_t = _m_ov_safe[_bt]
        _new_m = torch.maximum(_m, _m_ov_t)
        _l = torch.exp(_m - _new_m)
        _r = torch.exp(_m_ov_t - _new_m)
        _s = torch.where(_use_ov[:, None], _s * _l + _s_ov[_bt] * _r, _s)
        _n = torch.where(_use_ov[:, None], _n * _l + _n_ov[_bt] * _r, _n)
        _m = torch.where(_use_ov[:, None], _new_m, _m)
    # Hillis-Steele doubling scan, reset at every block boundary.  The monoid
    # is commutative, so combining the earlier and later half in either order
    # gives the same prefix statistics.
    _off = 1
    while _off < T:
        _same = (_bt[_off:] == _bt[:-_off])[:, None]
        _mL, _mR = _m[:-_off], _m[_off:]
        _new_m = torch.maximum(_mL, _mR)
        _l = torch.exp(_mL - _new_m)
        _r = torch.exp(_mR - _new_m)
        _new_s = _s[:-_off] * _l + _s[_off:] * _r
        _new_n = _n[:-_off] * _l + _n[_off:] * _r
        _m = torch.cat([_m[:_off], torch.where(_same, _new_m, _m[_off:])], 0)
        _s = torch.cat([_s[:_off], torch.where(_same, _new_s, _s[_off:])], 0)
        _n = torch.cat([_n[:_off], torch.where(_same, _new_n, _n[_off:])], 0)
        _off <<= 1

    # Each scope ends at a different token (t for full_prefix, immediately
    # before W_t for outside_window); read that row of the scanned prefix.
    end_idx = (pg['u_start'] + pg['plen'] - 1).clamp(0, T - 1)
    s_end = _s.index_select(0, end_idx)
    n_end = _n.index_select(0, end_idx)
    # A prefix can be empty only in the outside_window scope; those rows are
    # invalid candidates and their value is never read by the attention path.
    comp = torch.where((pg['plen'] > 0)[:, None],
                       n_end / s_end.clamp_min(1e-300),
                       torch.zeros((), dtype=n_end.dtype, device=n_end.device))
    return comp.to(Xa.dtype)


def causal_prefix_pool_reference(Xa, Xb, Za, Zb, bid, B_pos_a, B_pos_b, overlap,
                                 *, n_blocks=None, scope='full_prefix', window=128,
                                 window_start=None):
    """Per-query, numerically stable reference for `causal_prefix_pool`.

    This is deliberately slow and loops over queries; it is the oracle the
    optimized cumulative-sum form is tested against (forward and gradients).
    """
    geo = _open_block_geometry(bid, n_blocks=n_blocks)
    T = int(geo['n'])
    Fd = int(Xa.shape[1])
    out = Xa.new_zeros((T, Fd))
    if T == 0:
        return out
    pg = _open_prefix_geometry(bid, n_blocks=int(geo['n_blocks']), scope=scope,
                               window=window, window_start=window_start,
                               candidate_table=False)
    cap_a = int(B_pos_a.shape[0])
    cap_b = int(B_pos_b.shape[0])
    starts = geo['starts'].tolist()
    counts = geo['counts'].tolist()
    # The optimized causal_prefix_pool accumulates in float64.  A reference
    # oracle that unconditionally casts to float32 is only fp32-accurate even
    # for fp64 inputs, so it cannot validate the optimized path at fp64 (or
    # even exactly reproduce its fp32 accumulation).
    _acc = torch.float64
    for tt in range(T):
        u_start = int(pg['u_start'][tt].item())
        if int(pg['plen'][tt].item()) <= 0:
            continue
        idx = list(range(u_start, u_start + int(pg['plen'][tt].item())))
        zs = [Za[u].to(_acc) + B_pos_a[min(u - int(geo['start_of_token'][tt].item()), cap_a - 1)].to(_acc) for u in idx]
        xs = [Xa[u].to(_acc) for u in idx]
        b = int(geo['block_of_token'][tt].item())
        if (str(scope) == 'full_prefix' and int(overlap) > 0 and b > 0
                and Xb is not None and Zb is not None):
            ps, pend = int(starts[b - 1]), int(starts[b - 1]) + int(counts[b - 1])
            ovs = max(ps, pend - int(overlap))
            for u in range(ovs, pend):
                zs.append(Zb[u].to(_acc) + B_pos_b[min(u - ps, cap_b - 1)].to(_acc))
                xs.append(Xb[u].to(_acc))
        zz = torch.stack(zs, 0)
        xx = torch.stack(xs, 0)
        w = torch.softmax(zz, dim=0)
        out[tt] = (w * xx).sum(0).to(Xa.dtype)
    return out


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
        # argsort returns exactly the same stable permutation as sort(...).indices.
        if ties == 'earliest':
            return torch.argsort(masked, dim=1, stable=True, descending=True)
        order = torch.argsort(masked.flip(1), dim=1, stable=True, descending=True)
        return torch.sub(B - 1, order, out=order)
    src = masked.flip(1) if ties == 'latest' else masked
    n, _ = src.shape
    # Keep only the k-th value.  Either branch produces the same value
    # (k-th largest == (B-kk+1)-th smallest).
    if kk * 4 >= B:
        kth = torch.kthvalue(src, B - kk + 1, dim=1).values.unsqueeze(1)
    else:
        kth = torch.topk(src, kk, dim=1, sorted=True).values[:, -1].unsqueeze(1).clone()
    # gt has at most kk - 1 entries (the k-th value itself is not strictly
    # greater), so `need` is non-negative and selects exactly kk columns.
    # Prefix ranks use int32 (B < 2^31): the same exact integers as the int64
    # form; the non-selected lanes are filled with the sentinel column in
    # place.  The scatter index below is widened to int64 because torch < 2.14
    # requires an int64 index for scatter_ (the values are unchanged).
    gt = src > kth
    eq = src == kth
    need = (kk - gt.sum(dim=1)).clamp_min(0).to(torch.int32)
    eq_cum = torch.cumsum(eq, dim=1, dtype=torch.int32)
    sel = gt | (eq & (eq_cum <= need[:, None]))
    del eq_cum, gt, eq, need
    rank = torch.cumsum(sel, dim=1, dtype=torch.int32)
    rank -= 1
    rank.masked_fill_(~sel, kk)
    # Scatter the selected source columns into [n, kk] (column order); the
    # sentinel column kk absorbs the non-selected lanes and is sliced away.
    # The base arange is a read-only constant of (B, device) — the same exact
    # integers a fresh torch.arange returns every call — so it rides the
    # shared index-table cache instead of being rebuilt per chunk.
    cols = _i32_range_cache(B, src.device)[None, :].expand(n, B)
    buf = torch.empty(n, kk + 1, dtype=torch.int32, device=src.device)
    # torch < 2.14 requires an int64 scatter index; `.long()` on the index only
    # widens the dtype and preserves every integer value.  `cols` stays int32
    # because scatter_ also requires src to match the destination dtype.
    buf.scatter_(1, rank.long(), cols)
    del rank, sel, cols
    cand = buf[:, :kk].long()
    del buf, kth
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
    # Boolean finite-and-causal table.  `torch.isfinite` already returns a
    # fresh tensor, so the conjunction runs in place; `causal` is read-only.
    cf = torch.isfinite(scores)
    cf.logical_and_(causal)
    # torch.where picks bit-identical values (scores or the -inf constant) —
    # the same selection masked_fill produced.
    masked = torch.where(cf, scores, float('-inf'))
    order = _rank_blocks(masked, B, ties, k=min(k, B))
    del masked
    _w_keep = min(int(k), B)
    # Only the first _w_keep order columns are ever consumed (the k >= B and
    # k < B forms both reduce to usable[:, :_w_keep]), so the mask gather runs
    # on that column slice directly: gathering a column subset yields exactly
    # the corresponding columns of the full gather — the same booleans (index
    # transport, design notes §6.1).  Gathering the conjoined mask gives
    # exactly the elementwise AND of the two per-mask gathers.
    order_w = order[:, :_w_keep]
    keep = cf.gather(1, order_w)
    # Integer prefix ranks fit in int32 (k <= B < 2**31): the same exact
    # integers as the default int64 cumsum.  The scatter target is nevertheless
    # cast to int64 below: torch < 2.14 requires an int64 index for scatter_,
    # and `.long()` preserves every integer value.
    dest = keep.cumsum(dim=1, dtype=torch.int32)
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
    kept.scatter_(1, tgt.long(), order_k)
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
# Above this many elements the per-query-chunk form is used on last_tok
# (value-identical; the dynamic arms have B ~ T / block_size).
_READABLE_CACHE_MAX_ELEMS = 4 << 20

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
            _cnt = _cpu[0]
            if isinstance(_cnt, np.ndarray):
                # Large cut tables carry a NumPy counts row; the C cumsum +
                # tolist produces the same Python integers as accumulate.
                ends = np.cumsum(_cnt, dtype=np.int64).tolist()
            else:
                ends = [int(v) for v in itertools.accumulate(_cnt)]
        except (TypeError, ValueError, OverflowError):
            return None
        try:
            bid._pool_ends_host = ends
        except (AttributeError, RuntimeError):
            pass
    return ends

def lightning_indexer(H, comp_kv, last_tok, W_DQ, W_DK, W_w, nIH, topk, return_mask=True, query_chunk=2048, random_select=False, pre_qI=None, pre_w=None, out_valid=None, causal_ends_host=None, need_soft=True, random_salt=0):
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
        # The no-soft form is only safe when no differentiable path reaches
        # the indexer (the S prior is the only route for its gradients).  A
        # caller that asks for no-soft while the indexer tensors require grad
        # would otherwise detach it, so re-enable the prior here.
        if not need_soft and any((_t is not None and _t.requires_grad for _t in (qI, kI, w_idx))):
            need_soft = True
    pos = _arange_cache(n, dev)
    # causal (and its negation) are exact bool functions of (pos, last_tok).
    # The fixed segmenter's pooled last_tok is ONE shared cached tensor read
    # by every row/layer/step, so a SMALL pair is cached on that tensor object
    # — the same bitwise-safe cache-on-immutable-object argument as
    # _pool_parts/_pool_geom (design notes §6.2).  The table is [T, B] with
    # B ~ T / block_size for the dynamic arms.  Above the element cap we build
    # the same mask per query chunk below (value-identical slices) and cache
    # nothing.
    # Built outside inference mode so a pair first filled during eval can
    # still serve training (masked_fill_ saves the mask for backward).
    _rkey = (int(n), int(last_tok.shape[0]))
    _rc = getattr(last_tok, '_readable_cache', None)
    _causal_full = _ncausal_full = None
    if _rc is not None and _rc[0] == _rkey:
        _causal_full, _ncausal_full = _rc[1:3]
    elif int(n) * int(last_tok.shape[0]) <= _READABLE_CACHE_MAX_ELEMS:
        with torch.inference_mode(False):
            _causal_full = block_readable(pos, last_tok)
            _ncausal_full = ~_causal_full
        try:
            last_tok._readable_cache = (_rkey, _causal_full, _ncausal_full)
        except (AttributeError, RuntimeError):
            pass

    def _causal_of(_s, _e):
        if _causal_full is not None:
            return _causal_full[_s:_e], _ncausal_full[_s:_e]
        with torch.inference_mode(False):
            _c = block_readable(pos[_s:_e], last_tok)
        return _c, ~_c
    k = min(topk, B)
    if k < int(topk):
        _wk = (int(topk), int(B))
        if _wk not in _TOPK_CLAMP_WARNED:
            _TOPK_CLAMP_WARNED.add(_wk)
            print(f'[lightning_indexer] requested topk={int(topk)} but only B={int(B)} blocks are available; effective selection is clamped to all {int(k)} blocks for this call')
    if random_select:
        _ri_key = (_dev_str(dev), 'rs_ri', int(n))
        _jc_key = (_dev_str(dev), 'rs_jc', int(B))
        _ri = _cache_get(_IDX_CACHE, _ri_key)
        if _ri is None:
            with torch.inference_mode(False):
                _ri = torch.arange(n, device=dev, dtype=torch.float64)[:, None]
            _cache_put(_IDX_CACHE, _ri_key, _ri, _MASK_CACHE_BUDGET_BYTES, _idx_cache_total)
        _jc = _cache_get(_IDX_CACHE, _jc_key)
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
    if n == 0:
        # Empty sequence: return the same shapes the non-empty path would,
        # including the valid-cols flag when requested.
        idx = torch.empty(0, _kw, device=dev, dtype=torch.int32)
        soft = torch.empty(0, _kw, device=dev, dtype=H.dtype) if need_soft else None
        if out_valid is not None:
            out_valid[:] = [torch.zeros(0, _kw, device=dev, dtype=torch.bool)]
        m = torch.zeros(0, B, device=dev, dtype=H.dtype) if return_mask else None
        return (m, idx, soft)
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
            if random_salt:
                z.add_(float(random_salt))
            z.sin_()
            z.mul_(43758.5453)
            z.sub_(torch.floor(z))
            z.mul_(2.0).sub_(1.0)
            c = z.to(H.dtype)
            del z
        else:
            raw = torch.einsum('ind,ibd->inb', qI[:, s:e], kI)
            raw.mul_(_iscale)
            F.relu(raw, inplace=True)
            c = torch.einsum('inb,ni->nb', raw, w_idx[s:e])
            # `raw` is dead once the head-weighted score table exists.
            del raw
        if out_dtype is None:
            out_dtype = c.dtype
            if need_soft:
                soft = torch.empty(n, _kw, device=c.device, dtype=out_dtype)
        _ccausal, _nccausal = _causal_of(s, e)
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
            del c, _ic, _ccausal, _nccausal
            continue
        c.masked_fill_(_nccausal, float('-inf'))
        _lse = torch.logsumexp(c, dim=-1, keepdim=True)
        # the gather output is fresh and nothing downstream needs the pre-subtract
        # values, so the subtraction runs in place (same kernel, same values, and
        # the sub_ node records the identical broadcast gradient); exp_ is the
        # same story (its backward saves the output, as the out-of-place form).
        # nan_to_num 必须出位：exp_ 的反向保存其输出张量，原地 nan_to_num_ 会
        # 改写该保存值、使 ExpBackward 的版本检查失败（反向报错）。
        _soft_g = _safe_nan_to_num(c.gather(1, _ic.long()).sub_(_lse).exp_(), where='indexer_softmax')
        # Padded/invalid selection lanes carry block id 0 as a transport pad,
        # not as a selected block.  The kernel reference and the open-prefix
        # twin already zero those lanes; leaving exp(score) there gives a
        # non-zero prior to callers that consume soft without sel_valid.
        _soft_g = torch.where(_box[0], _soft_g, torch.zeros((), dtype=_soft_g.dtype, device=_soft_g.device))
        soft[s:e] = _soft_g
        del c, _lse, _ic, _ccausal, _nccausal, _soft_g
    if _valid_parts is not None:
        out_valid[:] = [_valid_parts[0] if len(_valid_parts) == 1 else torch.cat(_valid_parts, 0)]
    m = None
    if return_mask:
        valid_cols = out_valid[0]
        with torch.no_grad():
            # Build the (n, B) mask chunk-by-chunk: only the valid selections
            # are written (1.0 each; duplicate writes of the same cell are
            # idempotent), and each chunk is masked with the same causal slice
            # the full-table form would read.  The (n, B) result is the output
            # itself.
            _il = idx.long()
            _ar = _arange_cache(n, dev)
            m = torch.zeros(n, B, device=dev, dtype=out_dtype)
            for _s in range(0, n, query_chunk):
                _e = min(_s + query_chunk, n)
                _vc = valid_cols[_s:_e]
                _rows = _ar[_s:_e, None].expand(_e - _s, _il.shape[1])
                m[_rows[_vc], _il[_s:_e][_vc]] = 1.0
                _, _ncc = _causal_of(_s, _e)
                m[_s:_e].masked_fill_(_ncc, 0.0)
    return (m, idx, soft)


def lightning_indexer_open(H, comp_kv, last_tok, W_DQ, W_DK, W_w, nIH, topk, *,
                           prefix_geo, prefix_kv=None, prefix_token_kv=None,
                           prefix_kind='summary', query_chunk=2048,
                           pre_qI=None, pre_w=None, need_soft=True):
    """Indexer selection over `closed blocks ∪ current-prefix candidates`.

    Candidate ids are combined: `0..B-1` are the closed blocks, `B..` are the
    current-prefix candidates.  Every candidate uses the same q/k/w scoring
    path; the prefix columns are not given a separate scorer.  `last_tok`
    retains the strict closed-block rule (`pos > last_tok`), and the prefix
    columns carry their own causal validity mask, so this function never
    widens the closed read gate.

    Returns `(idx, sel_valid, soft)`:
      idx        [T, topk] combined candidate ids (padding marked invalid)
      sel_valid  [T, topk] bool
      soft       [T, topk] indexer softmax prior over the selected candidates
                 (or None when `need_soft=False`)
    """
    if prefix_kind not in ('summary', 'singletons'):
        raise ValueError(f'unknown prefix_kind {prefix_kind!r}')
    n = int(H.shape[0])
    B = int(comp_kv.shape[0])
    dev = H.device
    if prefix_geo is None:
        raise ValueError('lightning_indexer_open requires prefix_geo')
    if prefix_kind == 'summary':
        P = 1
        if prefix_kv is None:
            raise ValueError('prefix_summary needs prefix_kv')
        pref_valid = (prefix_geo['plen'][:, None] > 0)
    else:
        if prefix_token_kv is None:
            raise ValueError('prefix_singletons needs prefix_token_kv')
        P = int(prefix_geo['token_idx'].shape[1])
        pref_valid = prefix_geo['valid']
    hd = W_DQ.shape[1] // nIH
    iscale = hd ** (-0.5)
    if pre_qI is not None:
        qI = pre_qI.view(n, nIH, hd).transpose(0, 1)
    else:
        qI = (H @ W_DQ).view(n, nIH, hd).transpose(0, 1)
    kI = (comp_kv @ W_DK).view(B, nIH, hd).transpose(0, 1)
    if pre_w is not None:
        w_idx = pre_w
    else:
        w_idx = F.linear(H, W_w)
    if prefix_kind == 'summary':
        kIp = (prefix_kv @ W_DK).view(n, nIH, hd)
    else:
        kIt = (prefix_token_kv @ W_DK).view(n, nIH, hd)
    topk = max(int(topk), 0)
    q_chunk = max(1, min(int(query_chunk), max(n, 1)))
    # The closed-block scores are [C, nIH, B] (plus [C, B] reductions); with
    # P=1 the prefix cap below is huge, so a long-context open-block panel
    # could materialise an O(C*B) table.  Mirror the closed indexer's byte
    # cap, then apply the singleton-prefix cap independently.
    if B > 0:
        _per_blocks = max(1, int(B) * (int(nIH) * int(H.element_size()) + 40)
                          + 64)
        q_chunk = max(1, min(q_chunk, (256 << 20) // _per_blocks))
    # The singleton gather is [C, P, nIH, hd]; cap it so long contexts use
    # smaller query chunks instead of materialising a huge candidate table.
    if P > 0:
        _per = max(1, int(nIH) * int(P) * int(hd))
        q_chunk = max(1, min(q_chunk, (16 << 20) // _per))
    idx = torch.empty((n, topk), dtype=torch.int32, device=dev)
    soft = None
    _valid_parts = []
    if n == 0:
        sel_valid = torch.zeros((0, topk), dtype=torch.bool, device=dev)
        if need_soft:
            soft = torch.zeros((0, topk), dtype=H.dtype, device=dev)
        return idx, sel_valid, soft
    _pos = _arange_cache(n, dev)
    for s in range(0, n, q_chunk):
        e = min(s + q_chunk, n)
        q = qI[:, s:e].permute(1, 0, 2)                      # [C, nIH, hd]
        raw = torch.einsum('cid,ibd->cib', q, kI)
        raw.mul_(iscale)
        F.relu(raw, inplace=True)
        c = torch.einsum('cib,ci->cb', raw, w_idx[s:e])       # [C, B]
        if P <= 0:
            scores = c
            valid = (_pos[s:e, None] > last_tok[None, :])
        elif prefix_kind == 'summary':
            kp = kIp[s:e]                                    # [C, nIH, hd]
            rawp = (q * kp).sum(-1)
            rawp.mul_(iscale)
            rawp.relu_()
            p = (rawp * w_idx[s:e]).sum(-1, keepdim=True)     # [C, 1]
            scores = torch.cat([c, p], 1)
            valid = torch.cat([_pos[s:e, None] > last_tok[None, :], pref_valid[s:e]], 1)
        else:
            u = prefix_geo['token_idx'][s:e]                  # [C, P]
            kg = kIt[u]                                      # [C, P, nIH, hd]
            kg = kg.permute(0, 2, 1, 3)                       # [C, nIH, P, hd]
            rawp = torch.einsum('cid,cipd->cip', q, kg)
            rawp.mul_(iscale)
            F.relu(rawp, inplace=True)
            p = torch.einsum('cip,ci->cp', rawp, w_idx[s:e])  # [C, P]
            scores = torch.cat([c, p], 1)
            valid = torch.cat([_pos[s:e, None] > last_tok[None, :], pref_valid[s:e]], 1)
        _box = []
        with torch.no_grad():
            _ic = _indexer_selection(scores, valid, topk, out_valid=_box)
        idx[s:e] = _ic
        _valid_parts.append(_box[0])
        if need_soft:
            if soft is None:
                soft = torch.empty((n, topk), dtype=scores.dtype, device=dev)
            masked = torch.where(valid, scores, torch.full((), float('-inf'), dtype=scores.dtype, device=dev))
            if scores.shape[1] > 0:
                lse = torch.logsumexp(masked, dim=-1, keepdim=True)
                g = masked.gather(1, _ic.long()).sub_(lse).exp_()
                g = torch.where(_box[0], g, torch.zeros((), dtype=g.dtype, device=dev))
                g = _safe_nan_to_num(g, where='indexer_softmax_open')
            else:
                g = torch.zeros_like(_ic, dtype=scores.dtype)
            soft[s:e] = g
        del scores, valid, c, raw
    sel_valid = _valid_parts[0] if len(_valid_parts) == 1 else torch.cat(_valid_parts, 0)
    if topk <= 0:
        sel_valid = torch.zeros((n, 0), dtype=torch.bool, device=dev)
        soft = torch.zeros((n, 0), dtype=H.dtype, device=dev) if need_soft else None
    return idx, sel_valid, soft


def _indexer_trainable(H, comp_kv, last_tok, W_DQ, W_DK, W_w, nIH, topk, *,
                       random_select=False, pre_qI=None, pre_w=None,
                       out_valid=None, causal_ends_host=None, need_soft=True,
                       random_salt=0):
    """Training wrapper around `lightning_indexer` with activation recompute.

    The indexer builds `raw`/`c`/`soft` tables of shape [T, B_blocks, nIH]
    (plus the [T, topk] prior) that stay alive until backward.  Recomputing
    the whole pure function in backward keeps only its inputs, matching the
    FusedIndexer save policy.  Values and gradients are unchanged (no RNG in the learned path
    unless `random_select`, which is a fixed function of the row/column ids).
    """
    def _run(_h, _ck, _lt):
        _m, _idx, _soft = lightning_indexer(
            _h, _ck, _lt, W_DQ, W_DK, W_w, nIH, topk, return_mask=False,
            random_select=random_select, pre_qI=pre_qI, pre_w=pre_w,
            out_valid=out_valid, causal_ends_host=causal_ends_host,
            need_soft=need_soft, random_salt=random_salt)
        return _m, _idx, _soft
    # If an outer block/MLP checkpoint already owns this forward, do not nest
    # a second checkpoint (the outer recompute covers the indexer too).
    _grad = _CKPT_NEST[0] == 0 and torch.is_grad_enabled() and (
        H.requires_grad or comp_kv.requires_grad
        or (pre_qI is not None and pre_qI.requires_grad)
        or (pre_w is not None and pre_w.requires_grad))
    if _grad:
        _saved = int(H.shape[0]) * int(comp_kv.shape[0]) * (int(nIH) + 4) * int(H.element_size())
        if _saved < (32 << 20):
            _grad = False
    if _grad:
        return torch.utils.checkpoint.checkpoint(
            _run, H, comp_kv, last_tok, use_reentrant=False,
            preserve_rng_state=False)
    return _run(H, comp_kv, last_tok)


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

def _sink_log_normalizer(logits, sink_logits):
    return torch.logaddexp(
        sink_logits.view(1, -1, 1),
        torch.logsumexp(logits, dim=-1, keepdim=True))


def _sink_softmax_impl(logits, sink_logits, dim=-1):
    if sink_logits is None:
        return _safe_nan_to_num(torch.softmax(logits, dim), where='sink_softmax')
    sink_logits = _safe_nan_to_num(sink_logits, where='sink_logit')
    # Under autocast the module sink parameter stays fp32 while q/k/v are
    # fp16/bf16.  logaddexp promotes the result to fp32, so `(logits - lse)`
    # and the final einsum with a half-precision V mixed dtypes and raised.
    # Match the kernel reference, which casts the sink to the attention work
    # dtype before combining.
    if sink_logits.dtype != logits.dtype:
        sink_logits = sink_logits.to(dtype=logits.dtype)
    lse = _sink_log_normalizer(logits, sink_logits)
    return _safe_nan_to_num((logits - lse).exp(), where='sink_softmax')

def _sink_split_softmax(logits, sink_logits, want_sink=True):
    if sink_logits is None:
        return (sink_softmax(logits, None, -1), None)
    sink_logits = _safe_nan_to_num(sink_logits, where='sink_split')
    if sink_logits.dtype != logits.dtype:
        # See _sink_softmax_impl: a mixed sink dtype silently promoted the
        # attention table and broke the V einsum on fp16/bf16 activations.
        sink_logits = sink_logits.to(dtype=logits.dtype)
    lse = _sink_log_normalizer(logits, sink_logits)
    soft = (logits - lse).exp()
    if not want_sink:
        return (soft, None)
    return (soft, (sink_logits.view(1, -1, 1) - lse).exp())

_FO_FULL_CAP_BYTES = 64 << 20
_FO_MINL = {}
