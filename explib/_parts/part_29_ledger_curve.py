

def _ledger_curve(values, max_points=_LEDGER_CURVE_MAX_POINTS):
    """Bound the curves embedded in the append-only JSONL ledger.

    The full curve is already in the per-cell sidecar / summary; a 200k-point
    array in every ledger line grows the trace quadratically with panel size.
    Small curves stay byte-identical for readers of old ledgers; a long curve
    is spike-preserving downsampled and annotated with its true length.
    """
    vals = list(values) if isinstance(values, (list, tuple)) else []
    n = len(vals)
    if n <= int(max_points):
        return vals, n, False
    return _reduce_curve_values(vals, int(max_points)), n, True


def _ledger_curve_points(points, max_points=_LEDGER_CURVE_MAX_POINTS):
    pts = list(points) if isinstance(points, (list, tuple)) else []
    n = len(pts)
    if n <= int(max_points):
        return pts, n, False
    return _reduce_curve_points(pts, int(max_points)), n, True


def _curve_append_point(dst, point):
    dst.append(point)
    if len(dst) > 2 * _CURVE_MAX_POINTS:
        dst[:] = _reduce_curve_points(dst, _CURVE_MAX_POINTS)


def distributed_info():
    """Return ``(rank, world_size, active)`` for a torchrun/DDP launch.

    Initialises the default process group on first use when ``WORLD_SIZE > 1``
    and no group exists yet.  Single-process runs return ``(0, 1, False)`` and
    never touch torch.distributed, so the historical path is unchanged."""
    if not torch.distributed.is_available():
        return (0, 1, False)
    if torch.distributed.is_initialized():
        return (torch.distributed.get_rank(), torch.distributed.get_world_size(), torch.distributed.get_world_size() > 1)
    try:
        world = int(os.environ.get('WORLD_SIZE', '1') or 1)
    except ValueError:
        world = 1
    if world <= 1:
        return (0, 1, False)
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    torch.distributed.init_process_group(backend=backend)
    return (torch.distributed.get_rank(), torch.distributed.get_world_size(), True)


def is_main_process():
    return distributed_info()[0] == 0


def _all_ranks_agree(local_ok, distributed, device):
    """Rank-agreed boolean for a branch that runs DDP collectives.

    A per-rank condition (finite loss, finite gradient norm) must be reduced
    before choosing whether to enter backward/step: if one rank skips and a
    peer enters the DDP gradient collective, the job deadlocks.  Outside a
    multi-process launch this is exactly the caller's local boolean.
    """
    if not distributed:
        return bool(local_ok)
    flag = torch.tensor([1 if bool(local_ok) else 0], dtype=torch.int32,
                        device=device)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    return bool(int(flag.item()))


def _amp_requested():
    """Mixed-precision mode for the training path.

    CUDA training defaults to bf16 autocast with fp32 master parameters;
    ``CSA_AMP=fp16`` additionally uses a GradScaler, and ``CSA_AMP=0``
    (or false/no/off) opts back into the pure-fp32 path.  The chosen dtype is
    recorded in each result so fp32 and autocast runs never mix silently."""
    v = os.environ.get('CSA_AMP', '').strip().lower()
    if v in ('0', 'false', 'no', 'off'):
        return None
    if v in ('', '1', 'true', 'yes', 'on', 'bf16', 'bfloat16'):
        return 'bf16'
    if v in ('fp16', 'float16', 'half'):
        return 'fp16'
    print(f'[amp] CSA_AMP={v!r} is not recognised (use bf16/fp16/0) — staying in fp32 instead of silently changing the numerical path')
    return None


def _active_amp(device=None):
    """The autocast dtype the training path uses on ``device``.

    The fp32 path is the historical, bitwise-stable one; AMP only applies on
    CUDA.  Keeping this helper separate lets `training_knobs()` record the
    concrete choice on a CPU analysis box (None) without pretending that a
    GPU job ran in fp32."""
    if device is None:
        device = DEVICE
    if getattr(device, 'type', str(device)) != 'cuda':
        return None
    _amp = _amp_requested()
    # bf16 is requested unconditionally on CUDA, but pre-Ampere
    # GPUs (V100/T4) cannot run it.  Failing here with a clear message beats
    # training under a silently different numerical path and then having the
    # frozen confirmation binding reject or mis-certify every record.
    if _amp == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('this GPU does not support bf16 (pre-Ampere); set CSA_AMP=fp16 to train under the fp16 autocast path, or move the frozen run to an Ampere-or-newer GPU — do not run the frozen confirmation in fp32/bf16-mismatch mode')
    return _amp


def _normalise_amp_mode(v):
    """Canonical ``None`` / ``'bf16'`` / ``'fp16'`` spelling of an AMP mode."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ('', 'none', 'fp32', 'float32', 'off', '0', 'false', 'no'):
        return None
    if s in ('bf16', 'bfloat16'):
        return 'bf16'
    if s in ('fp16', 'float16', 'half'):
        return 'fp16'
    raise ValueError(f"unknown amp_dtype {v!r} (use bf16/fp16/fp32 or auto)")


def _normalise_eager_batch(v):
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ('', 'auto'):
            return None
        if s in ('1', 'true', 'yes', 'on'):
            return True
        if s in ('0', 'false', 'no', 'off'):
            return False
        raise ValueError(f"unknown eager_batch {v!r} (use true/false or auto)")
    if isinstance(v, numbers.Integral):
        if int(v) in (0, 1):
            return bool(int(v))
    elif isinstance(v, numbers.Real):
        f = float(v)
        if math.isfinite(f) and f in (0.0, 1.0):
            return bool(int(f))
    # A typo such as 'flase', NaN or 1.5 must not silently select the batched
    # backend; make the caller state the mode explicitly.
    raise ValueError(f"unknown eager_batch {v!r} (use true/false or auto)")


def _resolve_impl_modes(cfg=None, *, device=None):
    """Resolve ``(amp_dtype, eager_batch)`` for one concrete run.

    Explicit values in ``cfg`` always win, so a planner / report process and a
    par.py child can recompute the same resume fingerprint from a persisted
    manifest without probing their own CUDA state.  Only when the config says
    ``auto`` (or omits the key) AND a concrete ``device`` is supplied is the
    mode inferred locally.  With neither, the machine-independent canonical
    default (fp32 / row-by-row) is used.

    A frozen config naming a mode the device cannot provide is an error, not a
    silent downgrade: the record would otherwise claim an implementation it
    did not run.
    """
    cfg = {} if cfg is None else cfg
    amp_raw = cfg.get('amp_dtype', 'auto')
    eb_raw = cfg.get('eager_batch', 'auto')
    amp_auto = isinstance(amp_raw, str) and amp_raw.strip().lower() == 'auto'
    eb_auto = _normalise_eager_batch(eb_raw) is None

    if amp_auto:
        amp = _active_amp(device) if device is not None else None
    else:
        amp = _normalise_amp_mode(amp_raw)
        if amp is not None and device is not None and getattr(device, 'type', str(device)) != 'cuda':
            raise RuntimeError(f'run config requires amp_dtype={amp!r} but device={device} cannot provide CUDA autocast; pass amp_dtype=fp32 or run on CUDA instead of silently changing the numerical path')
    if eb_auto:
        eb = _eager_batch_active(device, cfg.get('batch_size')) if device is not None else False
    else:
        eb = bool(_normalise_eager_batch(eb_raw))
        if eb and device is not None and not _eager_batch_active(device, cfg.get('batch_size')):
            raise RuntimeError(f'run config requires eager_batch={eb!r} but device={device}/backend cannot provide the batched CUDA twin; pass eager_batch=false or fix the runtime instead of silently changing the implementation')
    return amp, bool(eb)


def _is_finite_number(v):
    return _finite_scalar(v) is not None


def data_seed_for_rank(seed, rank):
    """Domain-safe per-rank data seed for DDP.

    Rank 0 keeps the single-process stream unchanged.  Other ranks get a
    deterministic SplitMix64 derivation instead of ``seed + rank``, which can
    overflow the NumPy/PyTorch seed domain and can silently collide with a
    different declared seed (seed=0/rank=1 vs seed=1/rank=0).
    """
    seed = _seed_value(seed)
    rank = int(rank)
    if rank < 0:
        raise ValueError(f'rank must be non-negative, got {rank}')
    if rank == 0:
        return seed
    mixed = _splitmix64_u64((int(seed) << 32) ^ (rank & 0xFFFFFFFF))
    return int(mixed % (MAX_SEED + 1))

def _fixed_train_eval_batch(train_ids, seq_len, batch_size, device, seed):
    """One deterministic batch used ONLY to measure training CE on a fixed
    distribution.  It has its own RNG stream, so it cannot perturb the training
    batch order."""
    gen = batch_iter(train_ids, seq_len, batch_size, device, seed=int(seed) ^ 0x5F3759DF)
    try:
        return next(gen)
    finally:
        try:
            gen.close()
        except Exception:
            pass


def _audit_sample_flat(arr, max_tokens):
    """Deterministic, bounded sample of a flat token stream for data audits.

    A scale-up corpus can hold billions of ids; scanning all of it on every
    training cell would make the guard itself a cost centre.  When the array is
    larger than ``max_tokens`` we read equally spaced windows (the same strategy
    as ``audit_corpus_split``) and report that the check was sampled.
    """
    flat = np.asarray(arr).reshape(-1)
    n = int(flat.size)
    _max = _exact_count(max_tokens, minimum=1)
    if _max is None:
        raise ValueError(
            f'_audit_sample_flat needs a positive integer max_tokens, got {max_tokens!r}')
    max_tokens = int(_max)
    if n <= max_tokens:
        return flat, False
    # The budget can be smaller than the historical 16 windows.  Shrinking the
    # window count (and keeping width >= 1) makes the bound hold even for
    # max_tokens=1; the old fixed 16 windows returned 16 elements for a
    # one-token budget, so the audit's stated cost was exceeded 16x.
    n_windows = min(16, max_tokens)
    width = max(1, max_tokens // n_windows)
    starts = np.linspace(0, max(0, n - width), num=n_windows, dtype=np.int64)
    parts = [flat[int(s):int(s) + width] for s in starts]
    return np.concatenate(parts) if parts else flat[:0], True


def audit_token_data(train_ids, val_batch, vocab, *, max_sample_tokens=2_000_000):
    """Pre-training data contract for the NaN/Inf triage path.

    The prompt's first NaN suspect is the data, so the trainer records an
    explicit, machine-checkable answer instead of a free-text "data checked"
    claim.  Integer token caches cannot contain NaN/Inf, but a float feature
    cache can; both are checked here.  Out-of-vocabulary or negative ids and
    non-finite float values abort the run before a model is built.  The
    training stream is sampled deterministically when it is large; the
    validation split is scanned in full because it is bounded by ``val_seqs``.
    """
    _max = _exact_count(max_sample_tokens, minimum=1)
    if _max is None:
        raise ValueError(f'audit_token_data needs a positive integer max_sample_tokens, got {max_sample_tokens!r}')
    max_sample_tokens = int(_max)
    vocab = int(vocab)
    out = {'kind': 'token_data_audit', 'status': 'ok', 'vocab': vocab,
           'max_sample_tokens': max_sample_tokens, 'sections': {}}

    def _check(arr, name, *, sample):
        arr = np.asarray(arr)
        if arr.size == 0:
            raise ValueError(f'{name}: token array is empty')
        flat = arr.reshape(-1)
        values, sampled = _audit_sample_flat(flat, max_sample_tokens) if sample else (flat, False)
        is_bool = (np.dtype(arr.dtype).kind == 'b')
        is_float = np.dtype(arr.dtype).kind in 'fc'
        n_nonfinite = 0
        if is_float:
            try:
                n_nonfinite = int(values.size - np.isfinite(values).sum())
            except (TypeError, ValueError, OverflowError):
                n_nonfinite = int(values.size)
        if n_nonfinite:
            raise ValueError(f'{name}: {n_nonfinite} non-finite (NaN/Inf) value(s) in the checked '
                             f'{values.size:,} token(s); fix the data before training')
        if is_float or is_bool or not np.issubdtype(arr.dtype, np.integer):
            raise TypeError(f'{name}: expected integer token ids, got dtype {arr.dtype}'
                            + (' (all finite, but token ids must be integral)' if is_float and not n_nonfinite else ''))
        try:
            lo = int(values.min())
            hi = int(values.max())
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'{name}: could not read the token-id range') from None
        n_out = int(np.count_nonzero((values < 0) | (values >= vocab)))
        if n_out:
            raise ValueError(f'{name}: {n_out} token id(s) outside vocab [0, {vocab}) '
                             f'(checked range [{lo}, {hi}])')
        return {'dtype': np.dtype(arr.dtype).str, 'n_tokens': int(flat.size),
                'n_checked': int(values.size), 'sampled': bool(sampled),
                'n_nonfinite': int(n_nonfinite), 'n_out_of_range': int(n_out),
                'min_id': lo, 'max_id': hi}

    out['sections']['train'] = _check(train_ids, 'training ids', sample=True)
    if val_batch is not None:
        out['sections']['validation'] = _check(val_batch, 'validation ids', sample=False)
    else:
        out['sections']['validation'] = None
    out['checks'] = {
        'integer_ids': True,
        'range_ok': True,
        'finite_ok': True,
        'train_sampled': bool(out['sections']['train']['sampled']),
        'validation_scanned_in_full': val_batch is not None,
    }
    return out


def _check_token_batch(x, y, vocab, *, step=None):
    """Fail fast on a corrupted batch before it reaches the model."""
    for name, t in (('inputs', x), ('targets', y)):
        if not torch.is_tensor(t):
            raise TypeError(f'batch {name} is not a tensor (step={step})')
        if t.ndim != 2 or t.numel() == 0:
            raise ValueError(f'batch {name} must be a non-empty 2-D tensor, got shape {tuple(t.shape)} (step={step})')
        if t.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError(f'batch {name} must contain integer token ids, got {t.dtype} (step={step})')
    if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
        raise ValueError(f'batch inputs/targets shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)} (step={step})')
    try:
        lo = int(x.min().item())
        hi = int(x.max().item())
    except Exception:
        return
    if lo < 0 or hi >= int(vocab):
        raise ValueError(f'batch token id range [{lo}, {hi}] outside vocab [0, {int(vocab)}) at step={step}')
    if bool((y < 0).any().item()) or int(y.max().item()) >= int(vocab):
        raise ValueError(f'batch target ids outside vocab [0, {int(vocab)}) at step={step}')


_PARAM_FINITE_CHECK = os.environ.get('CSA_PARAM_FINITE_CHECK', 'sample').strip().lower()


def _params_all_finite(model):
    flags = [torch.isfinite(p).all() for p in model.parameters() if p.numel()]
    if not flags:
        return True
    return bool(torch.stack(flags).all().item())


class _HashingWriter:
    """File proxy that hashes bytes as `torch.save` writes them.

    Hashing the same bytes on the way out avoids a second full-file read for
    the checkpoint digest."""

    __slots__ = ('_f', '_h')

    def __init__(self, f, h):
        self._f = f
        self._h = h

    def write(self, data):
        self._h.update(data)
        return self._f.write(data)

    def flush(self):
        return self._f.flush()

    def __getattr__(self, name):
        return getattr(self._f, name)


def _atomic_torch_save(obj, path, hash_out=None):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=os.path.basename(path) + '.', suffix='.tmp')
    os.close(fd)
    h = hashlib.sha256() if hash_out is not None else None
    try:
        if h is None:
            torch.save(obj, tmp)
        else:
            with open(tmp, 'wb') as f:
                torch.save(obj, _HashingWriter(f, h))
        os.replace(tmp, path)
        if hash_out is not None:
            hash_out.append(h.hexdigest())
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _checkpoint_payload(model, *, variant, seed, step, val_ppl, train_ce, health, extra=None):
    # Keep the tensors where they live; torch.save streams each storage to the
    # file.  Loaders already use map_location='cpu' (error_analysis / explib.rope).
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    payload = {'format': 'csa-analysis-ckpt-v1', 'variant': variant, 'seed': int(seed),
               'step': int(step), 'val_ppl': val_ppl, 'train_ce_fixed': train_ce,
               # The checkpoint must name the implementation that wrote it.
               # CKPT_CODE is the base semantics and omits the kernel suffix,
               # so a kernel-mode checkpoint was labelled eager.
               'curve_health': health, 'code': code_stamp(), 'sd': sd}
    if extra:
        payload.update(extra)
    return payload


def _save_analysis_checkpoint(path, model, meta):
    _sha = []
    _atomic_torch_save(_checkpoint_payload(model, **meta), path, hash_out=_sha)
    try:
        size = os.path.getsize(path)
    except OSError:
        size = None
    return {'path': path, 'bytes': size, 'sha256': (_sha[0] if _sha else None),
            'step': int(meta.get('step') or 0)}


def _percentile(vals, q):
    xs = []
    for v in vals:
        try:
            fv = float(v)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(fv):
            xs.append(fv)
    xs.sort()
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


@torch.inference_mode()
def measure_efficiency(model, ids, device, warmup=1, iters=5, decode_tokens=8):
    """Measure prefill / prefix-prefill timing and allocator peaks.

    There is no KV cache in the model, so the decode number is explicitly a
    PREFIX-PREFILL measurement (it re-runs the whole prefix), not incremental
    decode.  Results are reported as p50/p95 wall-clock plus tokens/s and
    allocator peaks."""
    was = model.training
    model.eval()
    ids = torch.as_tensor(ids)
    if ids.ndim == 1:
        ids = ids[None, :]
    if ids.ndim != 2 or int(ids.shape[0]) < 1 or int(ids.shape[1]) < 1:
        raise ValueError(
            f'measure_efficiency needs a non-empty 1-D/2-D token table, got '
            f'shape {tuple(ids.shape)}')
    _warm = _exact_count(warmup, minimum=0)
    _iters = _exact_count(iters, minimum=1)
    _dec = _exact_count(decode_tokens, minimum=1)
    if _warm is None:
        raise ValueError(f'warmup must be a non-negative integer, got {warmup!r}')
    if _iters is None:
        raise ValueError(f'iters must be a positive integer, got {iters!r}')
    if _dec is None:
        raise ValueError(f'decode_tokens must be a positive integer, got {decode_tokens!r}')
    warmup, iters, decode_tokens = int(_warm), int(_iters), int(_dec)
    ids = ids.to(device)
    try:
        for _ in range(max(0, int(warmup))):
            model(ids)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        pre = []
        for _ in range(max(1, int(iters))):
            t0 = time.perf_counter()
            model(ids)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            pre.append((time.perf_counter() - t0) * 1000.0)
        n_dec = min(max(1, int(decode_tokens)), max(1, int(ids.shape[1]) - 1))
        dec = []
        for t in range(max(1, int(ids.shape[1]) - n_dec), int(ids.shape[1]) + 1):
            t0 = time.perf_counter()
            model(ids[:, :t])
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            dec.append((time.perf_counter() - t0) * 1000.0)
        out = {'status': 'ok', 'n_tokens': int(ids.shape[1]), 'batch': int(ids.shape[0]),
               'device': str(device), 'torch': torch.__version__,
               'prefill_ms_p50': _percentile(pre, 0.50), 'prefill_ms_p95': _percentile(pre, 0.95),
               'prefix_prefill_ms_p50': _percentile(dec, 0.50),
               'prefix_prefill_ms_p95': _percentile(dec, 0.95),
               'tokens_per_s_prefill': (1000.0 * int(ids.shape[0]) * int(ids.shape[1]) / _percentile(pre, 0.50)) if _percentile(pre, 0.50) else None,
               'decode_has_kv_cache': False,
               'decode_metric': 'prefix_prefill (no KV cache; re-runs the whole prefix)'}
        if device.type == 'cuda':
            out['peak_mem_allocated_mb'] = torch.cuda.max_memory_allocated(device) / (1 << 20)
            out['peak_mem_reserved_mb'] = torch.cuda.max_memory_reserved(device) / (1 << 20)
        return out
    finally:
        model.train(was)


@torch.inference_mode()
def measure_efficiency_sweep(model, ids, device, *, lengths=(512, 2048), batches=(1,),
                             warmup=1, iters=3):
    """Batch x sequence-length latency/throughput sweep on ONE checkpoint.

    `measure_efficiency` answers "how fast is this one input"; a claim that an
    architecture is cheaper needs the shape dependence too (batch 1 vs 8,
    short vs long).  CUDA timings use events (so kernel launch is included but
    host-side jitter is not); CPU timings use perf_counter.  The output records
    the device, torch/triton versions and TF32 state so two rows are only
    compared when they were measured under the same conditions.

    There is still no KV cache in this model: the sweep measures full-prefix
    prefill, and `decode_has_kv_cache=False` is recorded rather than pretending
    this is incremental decoding.
    """
    was = model.training
    model.eval()
    arr = torch.as_tensor(ids)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or int(arr.shape[0]) < 1 or int(arr.shape[1]) < 1:
        raise ValueError(
            f'measure_efficiency_sweep needs a non-empty 1-D/2-D token table, '
            f'got shape {tuple(arr.shape)}')
    try:
        _lens_raw = list(lengths)
    except TypeError:
        raise ValueError(f'lengths must be an iterable of positive integers, got {lengths!r}') from None
    if not _lens_raw:
        raise ValueError('lengths must contain at least one sequence length')
    _lens = []
    for _x in _lens_raw:
        _n = _exact_count(_x, minimum=1)
        if _n is None:
            raise ValueError(f'lengths must contain positive integers, got {_x!r}')
        _lens.append(int(_n))
    try:
        _batches_raw = list(batches)
    except TypeError:
        raise ValueError(f'batches must be an iterable of positive integers, got {batches!r}') from None
    if not _batches_raw:
        raise ValueError('batches must contain at least one batch size')
    _batches = []
    for _x in _batches_raw:
        _n = _exact_count(_x, minimum=1)
        if _n is None:
            raise ValueError(f'batches must contain positive integers, got {_x!r}')
        _batches.append(int(_n))
    _warm = _exact_count(warmup, minimum=0)
    _iters = _exact_count(iters, minimum=1)
    if _warm is None:
        raise ValueError(f'warmup must be a non-negative integer, got {warmup!r}')
    if _iters is None:
        raise ValueError(f'iters must be a positive integer, got {iters!r}')
    warmup, iters = int(_warm), int(_iters)
    arr = arr.to(device)
    max_len = int(min(arr.shape[1], max(_lens)))
    if not any(0 < int(x) <= max_len for x in _lens):
        raise ValueError(
            f'none of lengths={_lens} fits the token table width '
            f'{int(arr.shape[1])}; refusing to return an empty sweep as success')
    _skipped_lens = sorted({int(x) for x in _lens
                            if not (0 < int(x) <= max_len)})
    out = {'status': 'ok', 'device': str(device), 'torch': torch.__version__,
           'triton': None, 'tf32': None, 'rows': [],
           'requested_seq_lens': sorted({int(x) for x in _lens}),
           'skipped_seq_lens': _skipped_lens,
           'decode_has_kv_cache': False,
           'decode_metric': 'prefix_prefill (no KV cache)',
           'measurement': ('cuda events' if device.type == 'cuda' else 'cpu perf_counter')}
    if _skipped_lens:
        # A requested length wider than the token table is not measured.  Keep
        # it visible instead of returning a shorter grid that still claims
        # status=ok (the all-lengths-invalid case already raises).
        out['skipped_seq_lens_reason'] = (
            f'requested length(s) {_skipped_lens} exceed the token table width '
            f'{int(arr.shape[1])}; they were not measured')
    try:
        import triton  # noqa: F401
        out['triton'] = getattr(triton, '__version__', None)
    except Exception:
        out['triton'] = None
    try:
        out['tf32'] = bool(torch.backends.cuda.matmul.allow_tf32) if device.type == 'cuda' else None
    except Exception:
        pass
    try:
        for bs in _batches:
            for L in [int(x) for x in _lens if 0 < int(x) <= max_len]:
                base = arr[:, :L]
                if base.shape[0] < bs:
                    reps = (bs + base.shape[0] - 1) // base.shape[0]
                    batch = base.repeat(reps, 1)[:bs]
                else:
                    batch = base[:bs]
                for _ in range(max(0, int(warmup))):
                    model(batch)
                times = []
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                for _ in range(max(1, int(iters))):
                    if device.type == 'cuda':
                        _s = torch.cuda.Event(enable_timing=True)
                        _e = torch.cuda.Event(enable_timing=True)
                        _s.record()
                        model(batch)
                        _e.record()
                        torch.cuda.synchronize(device)
                        times.append(float(_s.elapsed_time(_e)))
                    else:
                        _t0 = time.perf_counter()
                        model(batch)
                        times.append((time.perf_counter() - _t0) * 1000.0)
                p50 = _percentile(times, 0.50)
                row = {'batch': bs, 'seq_len': L,
                       'prefill_ms_p50': p50, 'prefill_ms_p95': _percentile(times, 0.95),
                       'tokens_per_s': (1000.0 * bs * L / p50) if p50 else None}
                if device.type == 'cuda':
                    row['peak_mem_allocated_mb'] = torch.cuda.max_memory_allocated(device) / (1 << 20)
                    row['peak_mem_reserved_mb'] = torch.cuda.max_memory_reserved(device) / (1 << 20)
                out['rows'].append(row)
    finally:
        model.train(was)
    out['measured_seq_lens'] = sorted({int(r['seq_len']) for r in out['rows']})
    return out


def train_variant(variant, train_ids, val_batch, vocab, *, seed=0, d=256, n_layers=6, n_heads=8, d_head=32, seq_len=512, batch_size=12, steps=1500, lr=0.0003, weight_decay=0.1, warmup=50, comp_lambda=0.05, delta_lr_mult=10.0, eval_every=0, eval_subset=128, val_bnd=None, device=DEVICE, log_every=100, mlp_ratio=4, deadline_ts=None, return_model=False, distributed=False, save_ckpt=None, ckpt_dir=None, diag_every=None, measure_eff=False, measure_eff_sweep=False, amp_dtype='auto', eager_batch='auto', early_stop_patience=None, early_stop_min_delta=0.0):
    if steps < 1:
        raise ValueError(f'train_variant needs steps >= 1, got steps={steps}')
    if early_stop_patience is None:
        _es_patience = None
        _es_min_delta = 0.0
    else:
        if (isinstance(early_stop_patience, (bool, np.bool_))
                or not isinstance(early_stop_patience, (int, np.integer))
                or int(early_stop_patience) < 1):
            raise ValueError(f'early_stop_patience must be a positive integer or None, got {early_stop_patience!r}')
        if isinstance(early_stop_min_delta, (bool, np.bool_)):
            raise ValueError(f'early_stop_min_delta must be a finite non-negative number, got {early_stop_min_delta!r}')
        try:
            _es_min_delta = float(early_stop_min_delta)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'early_stop_min_delta must be a finite non-negative number, got {early_stop_min_delta!r}') from None
        if not math.isfinite(_es_min_delta) or _es_min_delta < 0.0:
            raise ValueError(f'early_stop_min_delta must be a finite non-negative number, got {early_stop_min_delta!r}')
        if not (eval_every and int(eval_every) > 0):
            raise ValueError('early_stop_patience needs eval_every > 0 so the validation curve is measured')
        _es_patience = int(early_stop_patience)
    seed = set_seed(seed)
    # Data is the first suspect on the NaN/Inf triage path.  Record an actual
    # contract check (range/dtype/finiteness) rather than a free-text claim.
    _data_audit = audit_token_data(train_ids, val_batch, vocab)
    _rng_fp = rng_fingerprint()
    _exec_fp = execution_fingerprint()
    _env_fp_v2 = environment_fingerprint_v2()
    _attb_bump_epoch()
    _amp, _ = _resolve_impl_modes({'amp_dtype': amp_dtype, 'eager_batch': False}, device=device)
    _amp_dtype = torch.bfloat16 if _amp == 'bf16' else (torch.float16 if _amp == 'fp16' else None)
    _scaler = torch.amp.GradScaler('cuda', enabled=(_amp == 'fp16')) if _amp is not None else None
    if _amp is not None:
        print(f'[amp] training autocast dtype={_amp} grad_scaler={_amp == "fp16"} (fp32 master weights)')
    _save_mode = str(save_ckpt or '').strip().lower()
    if _save_mode not in ('', 'none', 'off', '0', 'best', 'final', 'both'):
        print(f'[ckpt] ignoring unknown save_ckpt={save_ckpt!r} (use best/final/both/none)')
        _save_mode = ''
    if _save_mode in ('none', 'off', '0'):
        _save_mode = ''
    _nan_events = []
    _rank, _world, _dist = (0, 1, False)
    if distributed:
        _rank, _world, _dist = distributed_info()
        if not _dist:
            raise RuntimeError('train_variant(distributed=True) needs a torchrun WORLD_SIZE > 1 launch')
    nan_guard_state(reset=True)
    if variant in PARAM_MATCHED | PARAM_MATCHED_V7 and float(mlp_ratio) == 4.0:
        raise ValueError(f'{variant} is a PARAM-MATCHED baseline but was given the default mlp_ratio=4; its MLP must be widened to match its sparse reference arm, or the comparison is not parameter-controlled. Pass mlp_ratio=variant_mlp_ratio(variant, vocab, d=..., n_layers=..., n_heads=..., d_head=..., seq_len=...).')
    cfgs = make_layer_cfgs(n_layers, variant)
    # Validate an explicitly requested eager mode against this environment before
    # claiming it in the record; the concrete per-variant choice is still
    # `_uses_batched_eager` (a variant with no CSA/HCA layer never takes the
    # batched twin even when the backend is available).
    _resolve_impl_modes({'amp_dtype': amp_dtype, 'eager_batch': eager_batch,
                         'batch_size': batch_size}, device=device)
    _eager_batch_used = _uses_batched_eager(cfgs, device, batch_size)
    model = SmallGPT(vocab, d, n_layers, n_heads, d_head, seq_len, cfgs, mlp_ratio=mlp_ratio).to(device)
    _inner_model = model
    # Matched-initialization evidence for paired seeds: the shared trunk is
    # built before the variant-specific blocks, so two arms at the same seed
    # must hash identically.  This is what turns "paired by seed" into an
    # actual check rather than an assumption.
    _init_core_fp = init_core_fingerprint(_inner_model)
    if _dist and _world > 1:
        _dev_ids = [device] if device.type == 'cuda' else None
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=_dev_ids,
            output_device=(_dev_ids[0] if _dev_ids else None),
            gradient_as_bucket_view=True)
    n_param = count_params(model)
    print(f'\n[{variant} seed={seed}] params={n_param / 1000000.0:.2f}M  mlp_ratio={mlp_ratio:.2f}  layers={[c.kind for c in cfgs]}')
    # foreach=True pins the multi-tensor AdamW implementation.  Foreach applies
    # the same per-element IEEE op chain as the single-tensor form (lerp /
    # addcmul / sqrt are elementwise, so every element sees the identical
    # rounding sequence), so the update is bit-identical to the single-tensor
    # form, and on CUDA it pins the already-default foreach path.
    opt = build_optimizer(_inner_model, lr=lr, weight_decay=weight_decay,
                          delta_lr_mult=delta_lr_mult)
    _, _, delta_params = optimizer_param_groups(_inner_model)
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
    # In DDP each rank consumes a different data shard; the global batch is
    # batch_size * world_size (recorded below).  Single-process runs keep the
    # same seed exactly.
    _data_seed = data_seed_for_rank(seed, _rank) if _dist else seed
    bpe = batch_iter(train_ids, seq_len, batch_size, device, seed=_data_seed)
    _eval_batch = val_batch[:eval_subset] if val_batch is not None else None
    _eval_subset_fp = _ids_fp(np.ascontiguousarray(np.asarray(_eval_batch)).reshape(-1)) if _eval_batch is not None else None
    _fixed_eval = None
    if eval_every and train_ids is not None and int(eval_every) > 0:
        try:
            _fixed_eval = _fixed_train_eval_batch(train_ids, seq_len, batch_size, device, seed)
        except Exception as _fe:
            print(f'[health] WARNING: could not build the fixed train-eval batch ({type(_fe).__name__}: {_fe}); training CE will not be tracked')
            _fixed_eval = None
    _diag_stride = max(1, int(diag_every) if diag_every else max(1, int(steps) // 1000))
    # Lightweight token-range/shape checks stay at roughly every 1% of the run
    # (plus step 0 and any post-anomaly window), not once per 1024 steps: a
    # single corrupted window can poison all later checkpoints.
    _token_check_every = max(1, int(steps) // 100)
    _diag = {'step': [], 'ce': [], 'total_loss': [], 'comp_reg': [], 'boundary_pg': [],
             'grad_norm_pre_clip': [], 'update_norm': [], 'lr': [], 'clipped': [],
             'update_skipped': [], 'amp_scale': []}
    _diag_events = {}
    _clip_steps = 0
    _skipped_steps = 0
    _nonfinite_loss_steps = 0
    _nonfinite_grad_steps = 0
    _first_bad_step = None
    _scaler_overflows = 0
    _scaler_scale_min = None
    _best_val_ppl = None
    _best_step = None
    _last_subset_ppl = None
    _checkpoints = {}
    _health_history = []
    _last_train_ce = None
    _early_stop = (None if _es_patience is None else
                   {'patience': _es_patience, 'min_delta': _es_min_delta,
                    'stopped': False, 'stopped_step': None,
                    'planned_steps': int(steps), 'best_val_ppl': None,
                    'best_step': None, 'reason': 'patience not reached'})
    if device.type == 'cuda':
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass
    t0 = time.time()
    losses = []
    _lbuf = []
    _loss_stage = None
    _loss_ev = None
    if device.type == 'cuda':
        try:
            _loss_stage = torch.empty(512, dtype=torch.float32, pin_memory=True)
            _loss_ev = torch.cuda.Event()
        except Exception:
            _loss_stage = None
            _loss_ev = None

    def _flush_loss_buffer():
        if not _lbuf:
            return
        if _loss_stage is not None and len(_lbuf) <= _loss_stage.numel():
            _staged = _loss_stage[:len(_lbuf)]
            _staged.copy_(torch.stack(_lbuf), non_blocking=True)
            _loss_ev.record()
            _loss_ev.synchronize()
            _curve_extend(losses, _staged.tolist())
        else:
            _curve_extend(losses, torch.stack(_lbuf).tolist())
        _lbuf.clear()
    ppl_hist = []
    _ce_baseline = 0.0
    delta_trace = {}
    model.train()
    truncated = False
    # Cyclic GC is disabled for the step loop; collections still run at a
    # fixed cadence and the original GC state is restored on every exit path.
    _gc_relief = gc.isenabled()
    _ckpt_saved = _CKPT['on']
    _auto_ckpt, _auto_need = _auto_ckpt_mode(device, n_param, n_layers, d, mlp_ratio, batch_size, seq_len, cfgs, n_heads=n_heads, d_head=d_head)
    if _auto_ckpt:
        _CKPT['on'] = _auto_ckpt
        try:
            _free_now, _ = torch.cuda.mem_get_info(device)
            print(f'[ckpt] auto mode={_auto_ckpt} estimated_working_set={_auto_need / (1 << 30):.2f}GiB free_before={_free_now / (1 << 30):.2f}GiB (includes indexer/attention workspace)')
        except Exception:
            print(f'[ckpt] auto mode={_auto_ckpt} estimated_working_set={(_auto_need or 0) / (1 << 30):.2f}GiB')
    _ckpt_effective = _CKPT['on']

    def _do_save(tag, step_no, val_ppl, train_ce):
        if not _save_mode or not ckpt_dir:
            return
        if tag == 'best' and _save_mode not in ('best', 'both'):
            return
        if tag == 'final' and _save_mode not in ('final', 'both'):
            return
        try:
            path = os.path.join(ckpt_dir, f'{variant}_seed{seed}_{tag}.pt')
            _h = classify_training_health(_health_history)
            _checkpoints[tag] = _save_analysis_checkpoint(path, _inner_model, {
                'variant': variant, 'seed': seed, 'step': step_no,
                'val_ppl': val_ppl, 'train_ce': train_ce, 'health': _h,
                'extra': {'train_fp': _ids_fp(train_ids),
                          'eval_fp': _ids_fp(np.ascontiguousarray(np.asarray(val_batch)).reshape(-1)) if val_batch is not None else None,
                          'rng_fp': _rng_fp, 'params': int(n_param),
                          # The matched-initialization digest travels with the
                          # weights, not only with the training summary: the
                          # confirmation evaluator must be able to verify the
                          # pair without trusting a separately persisted
                          # summary file.
                          'init_core_fp': _init_core_fp,
                          'open_block_config': open_block_config(_inner_model),
                          # Enough geometry to rebuild the model for
                          # post-hoc error analysis without guessing defaults.
                          'cfg': cfgs, 'vocab': int(vocab), 'd': int(d),
                          'n_layers': int(n_layers), 'n_heads': int(n_heads),
                          'd_head': int(d_head), 'train_len': int(seq_len),
                          'max_seq': int(seq_len), 'mlp_ratio': float(mlp_ratio)}})
            print(f'  [ckpt] saved {tag} checkpoint {path} (step={step_no}, val_ppl={val_ppl})')
        except Exception as _se:
            print(f'[ckpt] WARNING: could not save {tag} checkpoint ({type(_se).__name__}: {_se}); the run continues, but this weight is unavailable for post-hoc analysis')

    if _gc_relief:
        gc.disable()
    try:
        for step in range(steps):
            if deadline_ts is not None and time.time() > deadline_ts:
                print(f'  [cap] HARD cap reached before step {step} - truncating; this run is NOT recorded (raise RUN_CAP and re-run to retry it)')
                truncated = True
                break
            if device.type == 'cuda' and step and step % 256 == 0:
                _attb_bump_epoch()
            base = lr_at(step)
            for g, _ls in zip(opt.param_groups, _lr_scales):
                g['lr'] = base * _ls
            x, y = next(bpe)
            # Corruption checks are a startup / anomaly guard: first batch,
            # roughly every 1% of the run, and again after any non-finite event.
            if step == 0 or step % _token_check_every == 0 or _first_bad_step is not None:
                try:
                    _check_token_batch(x, y, vocab, step=step + 1)
                except (TypeError, ValueError, OverflowError) as _be:
                    raise RuntimeError(f'[data] bad training batch at step {step + 1}: {_be}') from _be
                if step == 0:
                    # A wrong input/target shift leaves every token in range but
                    # trains the model to predict its own input.  It is a data
                    # pipeline fault, not an optimisation problem.
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
            _comp_item = float('nan')
            _pg_item = 0.0
            _loss_item = float('nan')
            _gn_pre = None
            _gn_post = None
            _update_norm = None
            _is_diag_step = (step % _diag_stride == 0 or step == steps - 1)
            _step_ok = True
            with torch.autocast(device_type='cuda', dtype=(_amp_dtype or torch.bfloat16),
                                enabled=(_amp is not None)):
                # Go through the (possibly DDP-wrapped) module's forward so
                # DDP can install its gradient-sync hooks; custom attributes
                # and methods always live on `_inner_model` because
                # DistributedDataParallel does not proxy them.
                ce = model(x, targets=y, vocab=vocab)
                del x, y
                _comp = getattr(_inner_model, 'comp_reg', None)
                _blp = getattr(_inner_model, 'boundary_logp', None)
                _ce_item = None
                _comp_item = None
                if _blp is not None:
                    # REINFORCE needs the current CE scalar to form its
                    # advantage; this arm is the only one that reads it per
                    # step.
                    _ce_item = float(ce.detach().item())
                    _comp_item = float(_comp.detach().item()) if _comp is not None else 0.0
                    _adv = _ce_item - float(_ce_baseline)
                    _bdec = max(int(getattr(_inner_model, 'boundary_decisions', 0) or 0), 1)
                    _pg = -_adv * _blp / _bdec
                    _pg_item = float(_pg.detach().item())
                    loss = ce + comp_lambda * _inner_model.comp_reg + BOUNDARY_PG_COEF * _pg
                    _ce_baseline = 0.99 * float(_ce_baseline) + 0.01 * _ce_item
                else:
                    loss = ce + comp_lambda * _inner_model.comp_reg
                _loss_item = float(loss.detach().item())
                # CE / comp_reg are diagnostics on the ordinary path; read
                # them only on logging/diagnostic steps (or after a bad loss).
                _need_scalars = (_is_diag_step or not math.isfinite(_loss_item)
                                 or (log_every and (step % log_every == 0 or step == steps - 1)))
                if _need_scalars:
                    if _ce_item is None:
                        _ce_item = float(ce.detach().item())
                    if _comp_item is None:
                        _comp_item = float(_comp.detach().item()) if _comp is not None else 0.0
                else:
                    _ce_item = float('nan')
                    _comp_item = 0.0
            _loss_finite_local = bool(math.isfinite(_loss_item))
            # Branch on the rank-agreed value, not on this rank's local loss:
            # a rank whose own shard happened to be finite used to enter
            # backward while a peer skipped it, deadlocking DDP.
            _loss_ok = _all_ranks_agree(_loss_finite_local,
                                        _dist and _world > 1, device)
            _step_ok = _loss_ok
            if not _loss_ok:
                _skipped_steps += 1
                _nonfinite_loss_steps += 1
                if _first_bad_step is None:
                    _first_bad_step = step + 1
                _nan_events.append({'step': step + 1, 'kind': 'loss', 'loss': _loss_item,
                                    'ce': _ce_item, 'comp_reg': _comp_item,
                                    'boundary_pg': _pg_item, 'lr': base,
                                    'rank_local_finite': _loss_finite_local,
                                    'rank': int(_rank)})
                _where = ('this rank' if not _loss_finite_local
                          else f'another rank (rank {int(_rank)} loss is finite)')
                print(f'  [nan] non-finite loss at step {step + 1} ({_where}; ce={_ce_item}, comp={_comp_item}, pg={_pg_item}); skipping the optimizer update on every rank')
                opt.zero_grad(set_to_none=True)
                if _scaler is not None:
                    # GradScaler.update() consumes/reset the found-inf state.
                    # Leaving it pending makes the next unscale_/step keep
                    # skipping and blocks the scale reduction (they are
                    # paired even when backward was skipped for a bad loss).
                    _scaler.update()
            elif _scaler is not None:
                _scale_before = float(_scaler.get_scale())
                _scaler.scale(loss).backward()
                _scaler.unscale_(opt)
                _gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                _gn_pre = float(_gn.detach().item()) if torch.is_tensor(_gn) else float(_gn)
                _grad_ok = _all_ranks_agree(math.isfinite(_gn_pre),
                                            _dist and _world > 1, device)
                if not _grad_ok:
                    _step_ok = False
                    _skipped_steps += 1
                    _nonfinite_grad_steps += 1
                    if _first_bad_step is None:
                        _first_bad_step = step + 1
                    _nan_events.append({'step': step + 1, 'kind': 'grad', 'grad_norm': _gn_pre,
                                        'loss': _loss_item, 'ce': _ce_item, 'amp_scale': _scale_before,
                                        'lr': base})
                    print(f'  [nan] non-finite gradient norm at step {step + 1}; zeroing grads and skipping the update')
                    opt.zero_grad(set_to_none=True)
                    # unscale_() has marked found_inf; update() is what
                    # consumes that state and lowers the scale.  Skipping it
                    # leaves the scale too high and the overflow state set.
                    _scaler.update()
                else:
                    _gn_post = min(_gn_pre, float(GRAD_CLIP_NORM))
                    _clip_steps += int(_gn_pre > GRAD_CLIP_NORM)
                    _ub = _param_change_sample(_inner_model, max_per_tensor=256).clone() if _is_diag_step else None
                    _scaler.step(opt)
                    _scaler.update()
                    if _ub is not None:
                        try:
                            _update_norm = float((_param_change_sample(_inner_model, max_per_tensor=256) - _ub).norm().item())
                        except Exception:
                            _update_norm = None
                _scale_after = float(_scaler.get_scale())
                _scaler_scale_min = _scale_after if _scaler_scale_min is None else min(_scaler_scale_min, _scale_after)
                if _scale_after < _scale_before:
                    _scaler_overflows += 1
            else:
                loss.backward()
                _gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                _gn_pre = float(_gn.detach().item()) if torch.is_tensor(_gn) else float(_gn)
                _grad_ok = _all_ranks_agree(math.isfinite(_gn_pre),
                                            _dist and _world > 1, device)
                if not _grad_ok:
                    _step_ok = False
                    _skipped_steps += 1
                    _nonfinite_grad_steps += 1
                    if _first_bad_step is None:
                        _first_bad_step = step + 1
                    _nan_events.append({'step': step + 1, 'kind': 'grad', 'grad_norm': _gn_pre,
                                        'loss': _loss_item, 'ce': _ce_item, 'lr': base})
                    print(f'  [nan] non-finite gradient norm at step {step + 1}; zeroing grads and skipping the update')
                    opt.zero_grad(set_to_none=True)
                else:
                    _gn_post = min(_gn_pre, float(GRAD_CLIP_NORM))
                    _clip_steps += int(_gn_pre > GRAD_CLIP_NORM)
                    _ub = _param_change_sample(_inner_model, max_per_tensor=256).clone() if _is_diag_step else None
                    opt.step()
                    if _ub is not None:
                        try:
                            _update_norm = float((_param_change_sample(_inner_model, max_per_tensor=256) - _ub).norm().item())
                        except Exception:
                            _update_norm = None
            # The default is a low-frequency sample (step 0, after anomalies,
            # and every 1024 steps) so a corrupted update cannot run for an
            # entire panel unnoticed; CSA_PARAM_FINITE_CHECK=full scans every
            # 256 steps and =off disables it.
            _param_full = _PARAM_FINITE_CHECK in ('always', 'full', '1', 'true', 'on')
            _param_sample = _PARAM_FINITE_CHECK in ('sample', 'auto')
            _param_check_due = (
                (_param_full and (step % 256 == 0 or _first_bad_step is not None))
                or (_param_sample and (step == 0 or _first_bad_step is not None
                                       or (step + 1) % 1024 == 0)))
            if (_step_ok and _param_check_due
                    and not _params_all_finite(_inner_model)):
                _nan_events.append({'step': step + 1, 'kind': 'param', 'loss': _loss_item,
                                    'grad_norm': _gn_pre, 'lr': base})
                raise FloatingPointError(f'non-finite parameter values after step {step + 1}; aborting instead of training on corrupted weights')
            # Gradients are not needed after the update.
            opt.zero_grad(set_to_none=True)
            _ce_d = ce.detach()
            _lbuf.append(_ce_d)
            # The loss buffer stores the detached CE tensor; `final_loss_smoothed`
            # is derived from the full recorded curve at the end (not from
            # sparse scalar reads, which are NaN on most steps).
            # Drop this step's graph/scalar references before the next forward.
            del loss, ce
            _inner_model.comp_reg = None
            if len(_lbuf) >= 512:
                _flush_loss_buffer()
            if step % _diag_stride == 0 or step == steps - 1 or not _step_ok:
                _diag_events[step + 1] = {'step': step + 1, 'ce': _ce_item, 'total_loss': _loss_item,
                                          'comp_reg': _comp_item, 'boundary_pg': _pg_item,
                                          'grad_norm_pre_clip': _gn_pre, 'update_norm': _update_norm,
                                          'lr': base,
                                          'clipped': bool(_gn_pre is not None and math.isfinite(_gn_pre) and _gn_pre > GRAD_CLIP_NORM),
                                          'update_skipped': (not _step_ok),
                                          'amp_scale': (float(_scaler.get_scale()) if _scaler is not None else None)}
                for _k in _diag:
                    _diag[_k].append(_diag_events[step + 1].get(_k))
            if (step + 1) % 1024 == 0:
                gc.collect()
            if eval_every and ((step + 1) % eval_every == 0 or step == steps - 1):
                if val_batch is not None:
                    _sub_stats = eval_ppl(_inner_model, _eval_batch, device, return_stats=True)
                    sub_ppl = float(_sub_stats.get('ppl', float('nan')))
                    ppl_hist.append([step + 1, sub_ppl])
                    _ci = _nll_doc_ci(_sub_stats.get('nll_per_row'), _sub_stats.get('n_tokens_per_row'))
                else:
                    sub_ppl = float('nan')
                    _ci = {'ppl': None, 'doc_ce_sem': None, 'ppl_ci95_low': None,
                           'ppl_ci95_high': None, 'n_docs': 0}
                _train_ce = None
                if _fixed_eval is not None:
                    try:
                        _train_ce, _train_acc = _lm_batch_metrics(_inner_model, _fixed_eval[0], _fixed_eval[1], vocab)
                    except Exception as _te:
                        print(f'[health] WARNING: fixed train-CE evaluation failed ({type(_te).__name__}: {_te}); train CE stays unmonitored at this point')
                        _train_ce = None
                # None = not measured (infra missing/failed); NaN = measured
                # and non-finite.  classify_training_health counts only the
                # latter as a non-finite curve point.  `*_final` fields must
                # describe the LAST measurement, not the last finite one: a
                # run whose final eval is non-finite must not report an
                # earlier step's PPL as its final value.
                _last_train_ce = _train_ce
                _val_ce = math.log(sub_ppl) if (_is_finite_number(sub_ppl) and sub_ppl > 0) else float('nan')
                _entry = {'step': step + 1, 'val_ppl': sub_ppl, 'val_ce': _val_ce,
                          'train_ce': _train_ce,
                          'gap': (_val_ce - _train_ce) if (math.isfinite(_val_ce) and _is_finite_number(_train_ce)) else None,
                          'val_ci95_low': _ci.get('ppl_ci95_low'),
                          'val_ci95_high': _ci.get('ppl_ci95_high'),
                          'val_doc_ce_sem': _ci.get('doc_ce_sem'),
                          'val_n_docs': _ci.get('n_docs')}
                _health_history.append(_entry)
                _last_subset_ppl = sub_ppl
                if _is_finite_number(sub_ppl) and sub_ppl > 0 and (_best_val_ppl is None or sub_ppl < _best_val_ppl):
                    _best_val_ppl = sub_ppl
                    _best_step = step + 1
                    _do_save('best', step + 1, sub_ppl, _last_train_ce)
                if _early_stop is not None and val_batch is not None:
                    _es = early_stop_decision(_health_history, patience=_es_patience,
                                              min_delta=_es_min_delta)
                    _early_stop.update({'best_val_ppl': _es.get('best_val_ppl'),
                                        'best_step': _es.get('best_step'),
                                        'stale_evals': _es.get('stale_evals')})
                    if _es.get('should_stop'):
                        _early_stop.update({'stopped': True, 'stopped_step': step + 1,
                                            'reason': (f'validation PPL did not improve by >{_es_min_delta:g} '
                                                       f'for {_es_patience} consecutive eval(s)')})
                        print(f'  [early-stop] stopped at step {step + 1}/{steps}: {_early_stop["reason"]}; '
                              f'best val PPL {_best_val_ppl} at step {_best_step}')
                        break
            if log_every and (step % log_every == 0 or step == steps - 1):
                if _lbuf:
                    _flush_loss_buffer()
                _lv = float(losses[-1])
                _dls = [(li, getattr(blk.attn, 'delta_logit', None)) for li, blk in enumerate(_inner_model.blocks)]
                _dls = [(li, dl) for li, dl in _dls if dl is not None]
                if _dls:
                    # (no_grad: 纯读取 Parameter 的值做日志，不需要 autograd 节点)
                    with torch.no_grad():
                        for li, _v in zip([li for li, _ in _dls], torch.stack([dl for _, dl in _dls]).tolist()):
                            _curve_append_point(delta_trace.setdefault(f'L{li}', []), [step, float(_v)])
                _gn_txt = f'{_gn_pre:.3g}' if (_gn_pre is not None and math.isfinite(_gn_pre)) else 'n/a'
                print(f'  step {step:4d}  loss {_lv:.4f}  ce {_ce_item:.4f}  comp {_comp_item:.3g}  grad {_gn_txt}  lr {opt.param_groups[0]['lr']:.2e}  ({(time.time() - t0) / max(step + 1, 1) * 1000:.0f}ms/step)')
    except BaseException:
        # Best-effort distributed cleanup: if one rank fails mid-step the
        # others can otherwise block forever in the next collective/barrier.
        if _dist and _world > 1:
            try:
                torch.distributed.destroy_process_group()
            except Exception:
                pass
        raise
    finally:
        _CKPT['on'] = _ckpt_saved
        if _gc_relief:
            gc.enable()
    wall = time.time() - t0
    _steps_completed = min(int(steps), int(step) + 1)
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
        return {'variant': variant, 'seed': seed, 'cap_truncated': True, 'steps_done': _steps_completed, 'train_time_s': wall}
    if _dist and _world > 1 and _rank != 0:
        # Non-zero ranks have no file-writing role; wait for rank 0's final
        # evaluation so the whole job exits together.
        torch.distributed.barrier()
        _res = {'variant': variant, 'seed': seed, 'ppl': float('nan'),
                'params': n_param, 'steps': steps, 'steps_completed': _steps_completed,
                'train_time_s': wall,
                'distributed_rank': _rank, 'distributed_world_size': _world,
                'tokens_seen': _steps_completed * batch_size * seq_len * _world}
        if _lbuf:
            _flush_loss_buffer()
        _res['losses'] = [float(v) for v in losses]
        if return_model:
            _res['model'] = _inner_model
        return _res
    if val_batch is None:
        ppl, stats = (float('nan'), {})
        _eval_stats = {'ppl': float('nan'), 'nll_sum': float('nan'), 'n_tokens': 0,
                       'nll_per_row': [], 'n_tokens_per_row': [], 'eval_rows': None,
                       'eval_seed': 0, 'eval_subset': 0}
    else:
        # Final measurement uses the FULL fixed validation batch;
        # `eval_subset` only controls the per-eval-step history.
        # Keeping this distinction byte-compatible matters: `_run_fp` pins the
        # full val fingerprint, and changing the final row count without a
        # CODE_SEMANTICS bump would silently mix two PPL definitions on resume.
        _eval_stats = eval_ppl(_inner_model, val_batch, device, return_stats=True,
                               collect_block_stats=True, val_bnd=val_bnd)
        ppl = float(_eval_stats['ppl'])
        # Reuse the statistics collected during the PPL forward.
        stats = _eval_stats.get('compression')
        if stats is None:
            stats = compression_report(_inner_model, val_batch, device, val_bnd=val_bnd)
        print(f'[{variant} seed={seed}] val PPL = {ppl:.3f}  (n_tok={_eval_stats["n_tokens"]}, nll={_eval_stats["nll_sum"]:.6f})')
        print(f'    block-length diagnostics: {json.dumps(stats, default=float)}')
    if _lbuf:
        _flush_loss_buffer()
    losses = [float(v) for v in losses]
    _curve_summary = _finite_curve(losses)
    # True final-window mean: the last 50 recorded training steps, not the
    # last 50 diagnostic scalar reads.  The latter are NaN on most steps
    # (reading CE every step would add a GPU sync), so the old value could be
    # an average of points spread over a large fraction of the run.
    _tail_losses = [v for v in losses[-50:] if math.isfinite(v)]
    _final_loss_smoothed = (sum(_tail_losses) / len(_tail_losses)) if _tail_losses else None
    _train_health = classify_train_loss_curve(losses)
    if _health_history:
        _health = classify_training_health(_health_history)
    else:
        _train_status = _train_health.get('status')
        _health = {'status': ('nonfinite' if _train_status == 'nonfinite' else 'unmonitored'),
                   'reason': ('no validation eval points (eval_every=0 or no val_batch); '
                              f'training-loss curve status={_train_status} ({_train_health.get("reason")})'),
                   'n_points': 0, 'best_step': None, 'best_val_ppl': None,
                   'final_val_ppl': None, 'final_minus_best': None,
                   'train_curve_status': _train_status}
    _grad_vals = [float(v) for v in _diag.get('grad_norm_pre_clip', []) if _is_finite_number(v)]
    _upd_vals = [float(v) for v in _diag.get('update_norm', []) if _is_finite_number(v)]
    _opt_summary = {
        'n_steps': int(steps), 'n_steps_completed': int(_steps_completed),
        'n_diag_points': len(_diag.get('step', [])),
        'grad_norm_mean': (sum(_grad_vals) / len(_grad_vals)) if _grad_vals else None,
        'grad_norm_max': max(_grad_vals) if _grad_vals else None,
        'update_norm_mean': (sum(_upd_vals) / len(_upd_vals)) if _upd_vals else None,
        'update_norm_max': max(_upd_vals) if _upd_vals else None,
        'clip_steps': int(_clip_steps),
        'clip_fraction': (float(_clip_steps) / max(1, int(_steps_completed))),
        'skipped_updates': int(_skipped_steps),
        'nonfinite_loss_steps': int(_nonfinite_loss_steps),
        'nonfinite_grad_steps': int(_nonfinite_grad_steps),
        'first_bad_step': _first_bad_step,
        'amp_scaler_overflows': int(_scaler_overflows),
        'amp_scaler_scale_min': _scaler_scale_min,
        'diag_stride': int(_diag_stride),
        'nan_guard': nan_guard_state(),
        'events': _nan_events[:20],
    }
    _instability = diagnose_training_instability(
        {'opt_diag_summary': _opt_summary, 'amp_dtype': _amp,
         'train_knobs': training_knobs(amp_dtype=_amp, eager_batch=_eager_batch_used),
         'curve_health': _health, 'health_history': _health_history,
         'train_curve_health': _train_health, 'losses': losses,
         'data_audit': _data_audit})
    _peak = {}
    if device.type == 'cuda':
        try:
            _peak_alloc = int(torch.cuda.max_memory_allocated(device))
            _peak = {'max_memory_allocated_mb': _peak_alloc / (1 << 20),
                     'max_memory_reserved_mb': torch.cuda.max_memory_reserved(device) / (1 << 20)}
            _gk = _ckpt_geom_key(n_param, n_layers, d, mlp_ratio, batch_size, seq_len, cfgs, n_heads=n_heads, d_head=d_head)
            _CKPT_PEAK_MEMO[_gk] = max(int(_CKPT_PEAK_MEMO.get(_gk, 0)), _peak_alloc)
        except Exception:
            _peak = {}
    if _save_mode and ckpt_dir:
        _do_save('final', int(_steps_completed), float(ppl) if _is_finite_number(ppl) else None, _last_train_ce)
    _efficiency = None
    _efficiency_sweep = None
    if measure_eff:
        try:
            if _eval_batch is not None:
                _eff_ids = np.ascontiguousarray(np.asarray(_eval_batch)[:min(4, len(_eval_batch))])
            elif _fixed_eval is not None:
                _eff_ids = _fixed_eval[0].detach().cpu().numpy()[:min(4, int(_fixed_eval[0].shape[0]))]
            else:
                _eff_ids = None
            if _eff_ids is not None and len(_eff_ids):
                _efficiency = measure_efficiency(_inner_model, _eff_ids, device, warmup=1, iters=5, decode_tokens=4)
                if measure_eff_sweep:
                    # A single (batch, length) point cannot support an
                    # efficiency claim at a new scale.  Two lengths at batch 1
                    # keep the extra cost small while exposing the shape
                    # dependence (the full sweep API supports larger grids).
                    _sweep_lens = sorted({min(512, int(_eff_ids.shape[1])), int(_eff_ids.shape[1])})
                    _sweep_lens = [n for n in _sweep_lens if 0 < n <= int(_eff_ids.shape[1])]
                    _efficiency_sweep = measure_efficiency_sweep(
                        _inner_model, _eff_ids, device,
                        lengths=tuple(_sweep_lens or [int(_eff_ids.shape[1])]),
                        batches=(1,), warmup=1, iters=3)
        except Exception as _ee:
            print(f'[efficiency] WARNING: measurement failed ({type(_ee).__name__}: {_ee})')
            _efficiency = {'status': 'error', 'error': f'{type(_ee).__name__}: {_ee}'}
            _efficiency_sweep = {'status': 'error', 'error': f'{type(_ee).__name__}: {_ee}'}
    # `tokens_seen` counts DRAWS, not unique tokens: training windows are
    # sampled with replacement and overlap heavily, so a 3000-step run over an
    # 8M-token cache does not consume 18.4M distinct tokens.  Record an
    # explicit coverage estimate instead of letting readers infer uniqueness.
    if _early_stop is not None:
        _early_stop['steps_completed'] = int(_steps_completed)
        if not _early_stop.get('stopped'):
            _early_stop['reason'] = 'patience not reached; ran to the planned step'
    _draws = int(_steps_completed) * int(batch_size) * int(_world)
    _n_train = int(len(train_ids)) if hasattr(train_ids, '__len__') else 0
    _uniq_frac = None
    if _n_train > 0:
        _p = min(float(seq_len) / float(_n_train), 1.0)
        _uniq_frac = (1.0 if _p >= 1.0 else float(1.0 - math.exp(_draws * math.log1p(-_p))))
    result = {'variant': variant, 'seed': seed, 'data_seed': int(_data_seed), 'exec_fp': _exec_fp, 'env_fp_v2': _env_fp_v2, 'ppl': ppl, 'params': n_param, 'losses': losses, 'stats': stats, 'steps': steps, 'steps_completed': int(_steps_completed), 'early_stop': _early_stop, 'tokens_seen': _steps_completed * batch_size * seq_len * _world,
              'tokens_seen_unique_estimate': (int(round(_n_train * _uniq_frac)) if _uniq_frac is not None else None),
              'tokens_seen_unique_frac_estimate': _uniq_frac,
              'token_accounting_note': 'tokens_seen counts window draws; unique-token coverage is an estimate under independent uniform window draws, not a measurement',
              'data_audit': _data_audit,
              'train_time_s': wall, 'train_tokens_per_s': ((_steps_completed * batch_size * seq_len * _world) / wall if wall and wall > 0 else None), 'distributed_rank': _rank, 'distributed_world_size': _world, 'final_loss_smoothed': _final_loss_smoothed, 'final_loss_smoothed_window': 50, 'final_loss_smoothed_n': len(_tail_losses), 'ppl_history': ppl_hist, 'delta_trace': delta_trace, 'train_knobs': training_knobs(amp_dtype=_amp, eager_batch=_eager_batch_used), 'amp_dtype': _amp, 'amp_grad_scaler': bool(_scaler is not None), 'eager_batch': _eager_batch_used, 'ckpt_mode': _ckpt_effective, 'ckpt_estimated_bytes': _auto_need, 'rng_fp': _rng_fp, 'init_core_fp': _init_core_fp, 'data_fp': _ids_fp(train_ids),
              # The token-level evidence behind the reported PPL.
              # `nll_sum / n_tokens` is the pooled-NLL estimator; the per-row
              # rows let a report bootstrap DOCUMENT uncertainty instead of
              # treating the seed as the only source of error.
              'nll_sum': _eval_stats.get('nll_sum'), 'n_tokens': _eval_stats.get('n_tokens'),
              'nll_per_row': _eval_stats.get('nll_per_row'), 'n_tokens_per_row': _eval_stats.get('n_tokens_per_row'),
              'eval_rows': _eval_stats.get('eval_rows'), 'eval_seed': _eval_stats.get('eval_seed'),
              'eval_subset': _eval_stats.get('eval_subset'),
              'eval_subset_fp': _eval_subset_fp, 'eval_subset_rows': len(_eval_batch) if _eval_batch is not None else 0,
              'eval_fp': _ids_fp(np.ascontiguousarray(val_batch).reshape(-1)) if val_batch is not None else None,
              'param_breakdown': param_breakdown(_inner_model),
              'kv_cache_elements_per_token': kv_cache_elements_per_token(_inner_model),
              'train_objective': train_objective(_inner_model, comp_lambda=comp_lambda, delta_lr_mult=delta_lr_mult),
              'chunking_protocol': model_chunking_protocols(_inner_model),
              'open_block_config': open_block_config(_inner_model),
              'curve_summary': _curve_summary,
              'curve_health': _health, 'health_history': _health_history,
              'train_curve_health': _train_health,
              'train_ce_fixed_final': _last_train_ce,
              'train_val_gap_final': _health.get('gap_last'),
              'best_step': _best_step, 'best_val_ppl': _best_val_ppl,
              'best_val_ppl_subset': _best_val_ppl,
              'final_subset_val_ppl': _last_subset_ppl,
              'final_full_val_ppl': (float(ppl) if _is_finite_number(ppl) else None),
              'final_minus_best': (float(_last_subset_ppl - _best_val_ppl)
                                   if (_is_finite_number(_last_subset_ppl) and _best_val_ppl is not None) else None),
              'opt_diag': _diag, 'opt_diag_summary': _opt_summary,
              'instability_diagnosis': _instability,
              'peak_memory': _peak, 'efficiency': _efficiency,
              'efficiency_sweep': _efficiency_sweep,
              'checkpoints': _checkpoints}
    if return_model:
        result['_model'] = _inner_model
    else:
        del model
    del opt, bpe
    gc.collect()
    if _dist and _world > 1:
        torch.distributed.barrier()
    return result

@torch.no_grad()
def _lm_batch_metrics(model, x, y, vocab, chunk=64, ignore_index=None):
    """Mean cross-entropy and next-token accuracy on one fixed batch.

    With ``ignore_index`` set, both the loss and the accuracy are reduced over
    the valid targets only (the same denominator the caller's training loss
    uses).  The historical non-masked path keeps its exact reduction order.
    """
    was_training = model.training
    model.eval()
    try:
        h = model.trunk(x)
        dev = h.device
        total = torch.zeros((), dtype=torch.float64, device=dev)
        correct = torch.zeros((), dtype=torch.long, device=dev)
        n = 0
        n_cols = min(int(x.shape[1]), int(y.shape[1]))
        for s in range(0, n_cols, chunk):
            sl = slice(s, min(s + chunk, n_cols))
            lg = model.head(h[:, sl, :])
            tg = y[:, sl]
            if ignore_index is None:
                total += F.cross_entropy(lg.reshape(-1, vocab), tg.reshape(-1), reduction='sum').double()
                correct += (lg.argmax(-1) == tg).sum()
                n += int(tg.numel())
            else:
                valid = tg != int(ignore_index)
                if bool(valid.any()):
                    lgf = lg.reshape(-1, vocab)[valid.reshape(-1)]
                    tgf = tg.reshape(-1)[valid.reshape(-1)]
                    total += F.cross_entropy(lgf, tgf, reduction='sum').double()
                    correct += (lg.argmax(-1)[valid] == tg[valid]).sum()
                    n += int(valid.sum())
        if n == 0:
            # Every target was masked: the batch has no measurement, not a
            # perfect zero loss.
            return float('nan'), float('nan')
        return (float(total.item()) / n), (float(correct.item()) / n)
    finally:
        model.train(was_training)

@torch.inference_mode()
def classify_token_error(*, ce, entropy, margin, rare_target, target_id, topk_ids, vocab):
    """Heuristic triage for one bad case.

    The labels are deliberately phrased as review prompts, not verdicts:
    ``annotation_or_rare_target`` still needs a human to check the annotation,
    ``ambiguous_context`` means the model distribution is close to uniform,
    and ``confident_model_blind_spot`` means low-entropy, high-margin and high
    CE (the model is wrong while sure).  ``target_outside_topk`` is a supporting
    fact, not a cause by itself.
    """
    try:
        ce = float(ce)
        entropy = float(entropy)
        margin = float(margin)
        vocab = int(vocab)
    except (TypeError, ValueError, OverflowError):
        return {'primary': 'unmeasured', 'causes': [], 'target_in_topk': None}
    if not math.isfinite(ce):
        return {'primary': 'unmeasured', 'causes': [], 'target_in_topk': None}
    tids = []
    for t in (topk_ids or []):
        try:
            tids.append(int(t))
        except (TypeError, ValueError, OverflowError):
            continue
    try:
        target_id = int(target_id)
    except (TypeError, ValueError, OverflowError):
        target_id = None
    in_topk = (target_id in tids) if (tids and target_id is not None) else None
    uniform = math.log(vocab) if vocab > 1 else 0.0
    causes = []
    if bool(rare_target):
        causes.append('annotation_or_rare_target')
    if math.isfinite(entropy) and (entropy >= 0.8 * uniform
                                   or (math.isfinite(margin) and margin <= 0.5)):
        causes.append('ambiguous_context')
    if (math.isfinite(entropy) and math.isfinite(margin)
            and entropy <= 0.5 * uniform and margin >= 2.0):
        causes.append('confident_model_blind_spot')
    if in_topk is False:
        causes.append('target_outside_topk')
    if not causes:
        causes.append('mixed_unclear')
    priority = ('annotation_or_rare_target', 'ambiguous_context',
                'confident_model_blind_spot', 'target_outside_topk', 'mixed_unclear')
    primary = next((c for c in priority if c in causes), 'mixed_unclear')
    return {'primary': primary, 'causes': causes, 'target_in_topk': in_topk}
