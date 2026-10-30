

def _overfit_sanity_check_impl(variant='full', *, seed=0, steps=300, lr=3e-3, d=64, n_layers=2, n_heads=4, d_head=16, seq_len=32, batch_size=2, vocab=256, train_ids=None, batch=None, ignore_index=None, device=None, target_loss=0.05, target_acc=0.99, mlp_ratio=None, grad_clip=1.0):
    """Small-batch overfit sanity check (necessary, not sufficient).

    Take ONE fixed batch (the first batch from the real ``batch_iter``
    training loader when ``train_ids`` is given, passed explicitly as
    ``batch=(inputs, targets)``, or synthetic ids when neither is supplied),
    disable every regulariser (weight decay = 0, the compression penalty = 0;
    the model has no dropout), and optimise only that batch.  A correct
    training path drives the LM loss towards 0 and the next-token accuracy
    towards 100%.  Failure means a serious plumbing fault (loss not connected
    to the graph, gradients not applied, wrong labels, etc.) — do NOT launch a
    full run until it passes.

    ``batch`` exists for task-specific training loops (NIAH, masked next-token
    targets): the probe then exercises the actual input/label tensors and the
    ``ignore_index`` convention of that loop, not merely random ids.

    Passing does NOT prove the data pipeline / augmentation / label mapping is
    correct, so keep the intermediate-result checks as well.

    Returns a JSON-safe report; `passed` is False when any criterion fails.
    """
    if int(d) != int(n_heads) * int(d_head):
        raise ValueError(f'overfit_sanity_check: d={d} must equal n_heads*d_head={n_heads}*{d_head}')
    if batch is not None and train_ids is not None:
        raise ValueError('overfit_sanity_check: pass either train_ids or batch, not both')
    _steps = _exact_count(steps, minimum=1)
    if _steps is None:
        raise ValueError(f'overfit_sanity_check: steps must be a positive integer, got {steps!r}')
    _seq = _exact_count(seq_len, minimum=1)
    if _seq is None:
        raise ValueError(f'overfit_sanity_check: seq_len must be a positive integer, got {seq_len!r}')
    _bs = _exact_count(batch_size, minimum=1)
    if _bs is None:
        raise ValueError(f'overfit_sanity_check: batch_size must be a positive integer, got {batch_size!r}')
    _vocab = _exact_count(vocab, minimum=1)
    if _vocab is None:
        raise ValueError(f'overfit_sanity_check: vocab must be a positive integer, got {vocab!r}')
    steps, seq_len, batch_size, vocab = int(_steps), int(_seq), int(_bs), int(_vocab)
    if ignore_index is not None and (isinstance(ignore_index, (bool, np.bool_))
                                     or not isinstance(ignore_index, numbers.Integral)):
        raise ValueError(f'overfit_sanity_check: ignore_index must be an integer or None, got {ignore_index!r}')
    device = DEVICE if device is None else (device if isinstance(device, torch.device) else torch.device(device))
    seed = set_seed(seed)
    _rng_fp = rng_fingerprint()
    _label_shift_ok = None
    _batch_contract = None
    _loader_mode = None
    _ign = None if ignore_index is None else int(ignore_index)
    ids_full = None
    if batch is not None:
        x_np, y_np = _batch_arrays(batch, vocab=vocab)
        batch_size, seq_len = int(x_np.shape[0]), int(x_np.shape[1])
        _data_fp = _ids_fp(x_np.reshape(-1)) + ':' + _ids_fp(y_np.reshape(-1))
        _data_source = 'batch'
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        _loader_mode = 'explicit_batch'
        if ignore_index is None:
            _ign = -100
        _batch_contract = bool(x.shape[0] == y.shape[0] and y.shape[1] <= x.shape[1])
        # Explicit batches may be task-specific (NIAH/masked targets), so the
        # next-token shift is REPORTED but not required here.
        _label_shift_ok = bool(y.shape == x.shape and y.shape[1] > 1
                               and torch.equal(y[:, :-1], x[:, 1:]))
    elif train_ids is not None:
        arr = np.asarray(train_ids)
        if arr.ndim != 1:
            raise ValueError('overfit_sanity_check: train_ids must be one-dimensional')
        if len(arr) < seq_len + 1:
            raise ValueError(f'overfit_sanity_check: need at least seq_len+1={seq_len + 1} tokens, got {len(arr)}')
        _loader = batch_iter(arr, int(seq_len), int(batch_size), device, seed=int(seed))
        try:
            x, y = next(_loader)
        finally:
            del _loader
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        _data_fp = _ids_fp(x_np.reshape(-1)) + ':' + _ids_fp(y_np.reshape(-1))
        _data_source = 'train_ids'
        _loader_mode = 'batch_iter'
        _label_shift_ok = bool(y.shape == x.shape and y.shape[1] > 0
                               and torch.equal(y[:, :-1], x[:, 1:]))
        _batch_contract = bool(x.shape[0] == y.shape[0] and y.shape[1] == x.shape[1])
    else:
        _rng = np.random.default_rng(int(seed))
        chunk = _rng.integers(0, max(int(vocab), 1), size=(int(batch_size), seq_len + 1), dtype=np.int64)
        _data_fp = _ids_fp(chunk.reshape(-1))
        _data_source = 'synthetic'
        _loader_mode = 'synthetic'
        ids_full = torch.from_numpy(chunk).to(device)
        x, y = ids_full[:, :-1], ids_full[:, 1:]
        _label_shift_ok = bool(y.shape == x.shape and y.shape[1] > 0
                               and torch.equal(y[:, :-1], x[:, 1:]))
    # The synthetic/repository loader paths are next-token LM by contract; the
    # explicit task batches above are not.  Only the contract paths make the
    # label shift and near-chance initial loss PASS criteria.
    _label_shift_required = _data_source in ('synthetic', 'train_ids')
    _initial_chance_required = _label_shift_required
    if mlp_ratio is None:
        mlp_ratio = variant_mlp_ratio(variant, vocab, d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, seq_len=seq_len, matched=PARAM_MATCHED | PARAM_MATCHED_V7)
    cfgs = make_layer_cfgs(n_layers, variant)
    model = SmallGPT(vocab, d, n_layers, n_heads, d_head, seq_len, cfgs, mlp_ratio=mlp_ratio).to(device)
    _p_before = _param_change_sample(model).clone()
    opt = build_optimizer(model, lr=lr, weight_decay=0.0, delta_lr_mult=1.0)

    _loss_path = 'forward' if _ign in (None, -100) else 'ce_loss'

    def _loss_consistency():
        with torch.no_grad():
            if _loss_path == 'forward':
                _trained = float(model(x, targets=y, vocab=vocab).detach())
            else:
                _trained = float(model.ce_loss(x, y, vocab, ignore_index=_ign).detach())
            _measured = float(_lm_batch_metrics(model, x, y, vocab, ignore_index=_ign)[0])
        if not (math.isfinite(_trained) and math.isfinite(_measured)):
            return {'trained_loss': _trained, 'measured_ce': _measured,
                    'abs_diff': None, 'rel_diff': None,
                    'tolerance': None, 'passed': False}
        _abs = abs(_trained - _measured)
        _rel = _abs / max(abs(_measured), 1e-12)
        _tol = max(1e-5, 1e-4 * abs(_measured))
        return {'trained_loss': _trained, 'measured_ce': _measured,
                'abs_diff': _abs, 'rel_diff': _rel,
                'tolerance': _tol, 'passed': bool(_abs <= _tol)}

    initial_loss, initial_acc = _lm_batch_metrics(model, x, y, vocab, ignore_index=_ign)
    _initial_loss_consistency = _loss_consistency()
    loss_curve = []
    acc_curve = []
    grad_norm_curve = []
    _nonfinite_step = None
    grad_norm = float('nan')
    grad_finite = True
    n_grad_none = 0
    _none_names = []
    _n_trainable = 0
    _n_with_grad = 0
    model.train()
    for step in range(int(steps)):
        if _loss_path == 'forward':
            ce = model(x, targets=y, vocab=vocab)
        else:
            ce = model.ce_loss(x, y, vocab, ignore_index=_ign)
        opt.zero_grad(set_to_none=True)
        ce.backward()
        _gs = []
        _none = 0
        _none_names = []
        _n_trainable = 0
        for _n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            _n_trainable += 1
            if p.grad is None:
                _none += 1
                if len(_none_names) < 50:
                    _none_names.append(_n)
                continue
            _gs.append(p.grad.detach())
        if _gs:
            try:
                _norms = torch.stack(
                    [v.to(torch.float64) for v in torch._foreach_norm(_gs)]
                ).cpu().tolist()
            except Exception:
                _norms = [float(torch.linalg.vector_norm(g)) for g in _gs]
            grad_finite = all(math.isfinite(_x) for _x in _norms)
            grad_norm = math.sqrt(math.fsum(_x * _x for _x in _norms)) if grad_finite else float('inf')
        else:
            grad_norm = 0.0
            grad_finite = False
        n_grad_none = _none
        _n_with_grad = len(_gs)
        grad_norm_curve.append(float(grad_norm))
        _l = float(ce.detach())
        loss_curve.append(_l)
        # Check BEFORE the optimizer update: a non-finite loss/gradient must
        # never be written into the parameters.
        if (not math.isfinite(_l)) or (not grad_finite):
            _nonfinite_step = step + 1
            break
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        opt.step()
        if (step + 1) % max(1, int(steps) // 20) == 0 or step == int(steps) - 1:
            _a = _lm_batch_metrics(model, x, y, vocab, ignore_index=_ign)[1]
            acc_curve.append([step + 1, _a])
    final_loss, final_acc = _lm_batch_metrics(model, x, y, vocab, ignore_index=_ign)
    _final_loss_consistency = _loss_consistency()
    _loss_consistency_ok = bool(_initial_loss_consistency['passed'] and _final_loss_consistency['passed'])
    _p_after = _param_change_sample(model)
    params_changed = not bool(torch.equal(_p_before, _p_after))
    _finite_losses = [v for v in loss_curve if math.isfinite(float(v))]
    best_loss = min(_finite_losses) if _finite_losses else float('nan')
    # A correct optimizer must also move the loss DOWN, not merely end below
    # the target by luck.  Keep this in `passed`: a path that starts below the
    # target and never updates (lr=0) must still fail.
    loss_decreased = bool(math.isfinite(initial_loss) and math.isfinite(final_loss) and final_loss < initial_loss)
    _chance = math.log(max(int(vocab), 2))
    _initial_near_chance = bool(math.isfinite(initial_loss) and abs(float(initial_loss) - _chance) <= max(0.5 * _chance, 1.0))
    _finite_curve = all(math.isfinite(float(v)) for v in loss_curve)
    # A wrong label shift or an initial loss nowhere near log(vocab) is a data
    # plumbing fault even if the model can still memorise the fixed batch, so
    # these diagnostics are part of the verdict for the next-token paths (the
    # explicit task-batch path keeps them as diagnostics only).  The batch
    # contract must never be False; None means "not applicable".
    _passed = bool(
        math.isfinite(final_loss) and final_loss <= float(target_loss)
        and final_acc >= float(target_acc) and grad_finite and params_changed
        and math.isfinite(best_loss) and best_loss <= float(target_loss)
        and loss_decreased and _finite_curve and _nonfinite_step is None
        and _batch_contract is not False
        and (not _label_shift_required or bool(_label_shift_ok))
        and (not _initial_chance_required or _initial_near_chance)
        and _loss_consistency_ok)
    passed = _passed
    # Intermediate-result evidence kept alongside the pass/fail bit: a sanity
    # check that only reports "PASS" hides WHICH part of the pipeline was
    # exercised.
    report = {
        'kind': 'small_batch_overfit_sanity',
        'variant': variant,
        'seed': int(seed),
        'code': code_stamp(),
        'determinism': determinism_label(),
        'device': str(device),
        'steps': int(steps),
        'lr': float(lr),
        'batch_shape': [int(batch_size), int(seq_len)],
        'vocab_size': int(vocab),
        'architecture': {'d': int(d), 'n_layers': int(n_layers), 'n_heads': int(n_heads), 'd_head': int(d_head)},
        'mlp_ratio': float(mlp_ratio),
        'ignore_index': _ign,
        'label_contract': {
            'source': _data_source,
            'ignore_index': _ign,
            'n_scored_targets': int((y != _ign).sum()) if _ign is not None else int(y.numel()),
            'n_targets': int(y.numel()),
        },
        'data_source': _data_source,
        'loader_mode': _loader_mode,
        'loss_path': _loss_path,
        'data_fp': _data_fp,
        'rng_fp': _rng_fp,
        'regularization': {'weight_decay': 0.0, 'comp_lambda': 0.0, 'dropout': 0.0, 'augmentation': 'none'},
        'initial_loss': float(initial_loss),
        'initial_accuracy': float(initial_acc),
        'loss_contract': {'initial': _initial_loss_consistency,
                          'final': _final_loss_consistency},
        'final_loss': float(final_loss),
        'final_accuracy': float(final_acc),
        'best_loss': float(best_loss),
        'loss_decreased': bool(loss_decreased),
        'loss_reduction': float(initial_loss - final_loss) if math.isfinite(initial_loss) and math.isfinite(final_loss) else None,
        'loss_curve': [float(v) for v in loss_curve],
        'accuracy_curve': acc_curve,
        'grad_norm_curve': grad_norm_curve,
        'grad_norm_last': float(grad_norm),
        'nonfinite_step': _nonfinite_step,
        'grad_all_finite': bool(grad_finite),
        'params_without_grad': int(n_grad_none),
        'params_without_grad_names': list(_none_names),
        'trainable_params': int(_n_trainable),
        'params_with_grad': int(_n_with_grad),
        'grad_coverage': (float(_n_with_grad) / float(_n_trainable) if _n_trainable else 0.0),
        'params_changed': bool(params_changed),
        'batch_reused': True,
        'target_loss': float(target_loss),
        'target_accuracy': float(target_acc),
        'expected_initial_loss_log_vocab': float(_chance),
        'label_shift_required': bool(_label_shift_required),
        'pass_criteria': {
            'finite_final_loss_below_target': bool(math.isfinite(final_loss) and final_loss <= float(target_loss)),
            'accuracy_above_target': bool(final_acc >= float(target_acc)),
            'gradients_finite': bool(grad_finite),
            'parameters_changed': bool(params_changed),
            'best_loss_below_target': bool(math.isfinite(best_loss) and best_loss <= float(target_loss)),
            'loss_decreased': bool(loss_decreased),
            'loss_curve_all_finite': bool(_finite_curve and _nonfinite_step is None),
            'batch_contract_not_failed': bool(_batch_contract is not False),
            'labels_are_next_token_shift': (bool(_label_shift_ok) if _label_shift_required else None),
            'initial_loss_near_log_vocab': (bool(_initial_near_chance) if _initial_chance_required else None),
            'loss_reduction_matches_mean_ce': bool(_loss_consistency_ok),
        },
        'intermediate_checks': {
            'labels_are_next_token_shift': _label_shift_ok,
            'batch_contract_ok': _batch_contract,
            'initial_loss_near_log_vocab': _initial_near_chance,
            'loss_curve_all_finite': bool(_finite_curve),
            'grad_all_finite': bool(grad_finite),
            'params_changed': bool(params_changed),
            'loss_decreased': bool(loss_decreased),
            'grad_coverage': (float(_n_with_grad) / float(_n_trainable) if _n_trainable else 0.0),
            'loss_reduction_matches_mean_ce': bool(_loss_consistency_ok),
        },
    }
    report['passed'] = passed
    report['failed_criteria'] = [k for k, ok in report['pass_criteria'].items() if ok is False]
    if passed:
        report['diagnosis'] = 'ok'
    elif not (grad_finite and params_changed):
        report['diagnosis'] = 'no/finite gradients or parameters did not update'
    elif not loss_decreased:
        report['diagnosis'] = 'loss did not decrease on the fixed batch (optimizer/step or graph is not wired through)'
    elif _label_shift_required and not _label_shift_ok:
        report['diagnosis'] = 'labels are not the next-token shift of the inputs (label construction/order bug)'
    elif _initial_chance_required and not _initial_near_chance:
        report['diagnosis'] = 'initial loss is far from log(vocab) (wrong label map, vocabulary mismatch or broken initialisation)'
    elif not _loss_consistency_ok:
        report['diagnosis'] = 'the reported training loss does not match the mean cross-entropy over the scored targets (loss reduction/denominator bug)'
    elif _batch_contract is False:
        report['diagnosis'] = 'the explicit batch violates the shape/label contract'
    elif not (_finite_curve and _nonfinite_step is None):
        report['diagnosis'] = 'the loss/gradient curve became non-finite before the target was reached'
    else:
        report['diagnosis'] = 'loss did not converge on a single repeated batch within the step limit'
    del model, opt, ids_full, x, y
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return report

def overfit_sanity_check(variant='full', *, seed=0, steps=300, lr=3e-3, d=64, n_layers=2, n_heads=4, d_head=16, seq_len=32, batch_size=2, vocab=256, train_ids=None, batch=None, ignore_index=None, device=None, target_loss=0.05, target_acc=0.99, mlp_ratio=None, grad_clip=1.0):
    """Side-effect-free wrapper around the small-batch overfit probe.

    The probe deliberately reseeds the process to ``seed`` and then trains a
    tiny model.  Saving/restoring every RNG stream around it keeps the probe
    from changing the stream seen by the real training cell that follows, so
    the diagnostic itself can never alter a recorded loss curve.  The report
    additionally records the before/after RNG fingerprint as evidence that
    the preservation contract held.
    """
    _fp_before = rng_fingerprint()
    with _preserve_rng_state():
        report = _overfit_sanity_check_impl(variant, seed=seed, steps=steps, lr=lr, d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, seq_len=seq_len, batch_size=batch_size, vocab=vocab, train_ids=train_ids, batch=batch, ignore_index=ignore_index, device=device, target_loss=target_loss, target_acc=target_acc, mlp_ratio=mlp_ratio, grad_clip=grad_clip)
    _fp_after = rng_fingerprint()
    report['rng_fp_before'] = _fp_before
    report['rng_fp_after'] = _fp_after
    report['rng_state_preserved'] = bool(_fp_before == _fp_after)
    report.setdefault('pass_criteria', {})['rng_state_preserved'] = bool(report['rng_state_preserved'])
    if not report['rng_state_preserved']:
        # A diagnostic that mutates the process RNG streams is itself a
        # reproducibility fault: the training cell after it would see a
        # different data order than the same run without the probe.
        report['passed'] = False
        report['failed_criteria'] = list(report.get('failed_criteria') or []) + ['rng_state_preserved']
        report['diagnosis'] = 'the overfit probe did not restore the process RNG state'
    return report

_OVERFIT_GUARD = {}
# Reports of the checks that actually ran (the boolean guard above is the
# resume cache; this is the evidence written into the run manifest).  Keyed
# like _OVERFIT_GUARD by (variant, source, ...).
_OVERFIT_REPORTS = {}

_OVERFIT_ARCH = {'d': 64, 'n_layers': 2, 'n_heads': 4, 'd_head': 16}

def _probe_arch(arch=None):
    """Effective probe architecture: the tiny default, overridden by the
    caller-supplied keys (a scaling driver can probe the architecture family
    it is about to train — d/n_heads/d_head geometry mismatches that a 4-head
    probe can never see are caught before the panel starts)."""
    _arch = dict(_OVERFIT_ARCH)
    if arch:
        for _k in _arch:
            if arch.get(_k) is not None:
                _arch[_k] = int(arch[_k])
    return _arch

def _overfit_probe_config(*, seed, steps, seq_len, batch_size, vocab, target_loss, target_acc, arch=None, mlp_ratio=None, ignore_index=None, source='synthetic', data_fp=None, device=None):
    """Exact probe configuration that a cached PASS is allowed to certify.

    A pass only proves that THIS tiny run can overfit (same step count, batch
    size, sequence length, targets, architecture, MLP width, ignore-index
    convention, data batch and code semantics).  Without these fields in the
    cache key a lax probe (e.g. one step, target_loss=100) would silently
    satisfy a later strict probe, turning the safety gate into a formality.
    The code stamp is included because changing the trainer/loss must
    invalidate an old pass even when the numeric knobs look identical.  The
    data source/fingerprint is kept separate (`_overfit_probe_semantics`) so a
    synthetic pass can still be subsumed by a real-data pass of the same
    model/probe geometry.
    """
    cfg = {
        'schema': 2,
        'seed': int(seed),
        'steps': int(steps),
        'seq_len': int(seq_len),
        'batch_size': int(batch_size),
        'vocab': int(vocab),
        'target_loss': float(target_loss),
        'target_accuracy': float(target_acc),
        'architecture': _probe_arch(arch),
        'mlp_ratio': (None if mlp_ratio is None else float(mlp_ratio)),
        'ignore_index': (None if ignore_index is None else int(ignore_index)),
        'source': str(source),
        'data_fp': (None if data_fp is None else str(data_fp)),
        # The probe certifies THIS execution target.  A CPU pass cannot attest
        # that the CUDA kernel path applies gradients; including the device and
        # the environment class in the cache key prevents a cached pass from
        # crossing hardware/software boundaries.
        'device': str(DEVICE if device is None else device),
        'determinism': determinism_label(),
        'exec_fp': execution_fingerprint(),
        'code': code_stamp(),
    }
    return cfg

def _canonical_json_digest(obj):
    """Short, stable digest of a JSON-serialisable probe key."""
    blob = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]

def _overfit_probe_digest(cfg):
    return _canonical_json_digest(cfg)

def _overfit_probe_semantics(cfg):
    """Digest of everything except WHERE the batch came from.

    A passed real-data probe of the same architecture/MLP/step/target
    configuration subsumes the synthetic probe; a different corpus does not
    invalidate the model-plumbing certificate, but the data certificate itself
    is still keyed by the full config (source + fingerprint)."""
    sem = {k: v for k, v in (cfg or {}).items() if k not in ('source', 'data_fp')}
    return _canonical_json_digest(sem)

def probe_seq_len(seq_len, *, min_blocks=128, cap=256):
    """Sequence length for the pre-flight probe.

    A 64-token probe never exercises the real block geometry (max_block=128,
    HCA block_size=64).  Use the panel's own length up to ``cap`` so at least
    one multi-block schedule is covered without running the full-scale probe;
    short panels keep their own (already short) length.  ``cap`` can be
    overridden with ``CSA_PROBE_SEQ``.
    """
    raw = os.environ.get('CSA_PROBE_SEQ', '').strip()
    if raw:
        try:
            return max(2, int(raw))
        except ValueError:
            print(f'[overfit] CSA_PROBE_SEQ={raw!r} is not an integer — using the automatic probe length')
    s = max(2, int(seq_len))
    return min(s, int(cap)) if s >= int(min_blocks) else s

def overfit_reports(*, source=None):
    """Copy of the small-batch reports gathered in this process, optionally
    filtered to source 'synthetic' or 'data'.  Pure read for tracking."""
    return {k: dict(v) for k, v in _OVERFIT_REPORTS.items() if source is None or k[1] == source}

def overfit_reports_for(variants, *, source=None, probe_semantics=None, probe_key=None, data_fp=None):
    """Exact-match evidence recovery for a panel.

    `run()` must persist only the probes that certify the CURRENT config: a
    report written under different or looser settings must not be attached to a
    new panel.  Matching is by probe semantics
    (architecture/MLP/steps/targets/ignore-index), optional source and data
    fingerprint, and optionally the full probe digest."""
    _want = set(variants)
    out = []
    for (_v, _src, *rest), _r in _OVERFIT_REPORTS.items():
        if _v not in _want:
            continue
        if source is not None and _src != source:
            continue
        if probe_semantics is not None and _r.get('guard_probe_semantics') != probe_semantics:
            continue
        if probe_key is not None and _r.get('guard_probe_key') != probe_key:
            continue
        if data_fp is not None and (_v, _src, *rest) and len(rest) >= 2 and rest[1] != data_fp:
            continue
        out.append(dict(_r))
    return out

def _overfit_ratio_for(mlp_ratio, variant):
    if isinstance(mlp_ratio, dict):
        return mlp_ratio.get(variant)
    return mlp_ratio

def _production_loop_sanity_check_impl(variant='full', *, seed=0, steps=150, lr=1e-2,
                                      d=32, n_layers=1, n_heads=4, d_head=8,
                                      seq_len=16, batch_size=2, vocab=64,
                                      train_ids=None, device=None,
                                      target_loss=0.05, target_acc=0.99,
                                      mlp_ratio=None):
    """Overfit ONE fixed window through the REAL `train_variant` loop.

    The hand-rolled small-batch probe proves the model/loss/label plumbing.
    This companion check proves the production trainer (optimizer groups,
    scheduler, clipping, loss assembly, PPL/eval path) actually applies the
    updates: a bug in `train_variant` cannot hide behind a separately written
    probe loop.
    """
    if int(d) != int(n_heads) * int(d_head):
        raise ValueError(f'production_loop_sanity_check: d={d} must equal n_heads*d_head={n_heads}*{d_head}')
    _steps = _exact_count(steps, minimum=1)
    if _steps is None:
        raise ValueError(f'production_loop_sanity_check: steps must be a positive integer, got {steps!r}')
    _seq = _exact_count(seq_len, minimum=1)
    if _seq is None:
        raise ValueError(f'production_loop_sanity_check: seq_len must be a positive integer, got {seq_len!r}')
    _bs = _exact_count(batch_size, minimum=1)
    if _bs is None:
        raise ValueError(f'production_loop_sanity_check: batch_size must be a positive integer, got {batch_size!r}')
    _vocab = _exact_count(vocab, minimum=1)
    if _vocab is None:
        raise ValueError(f'production_loop_sanity_check: vocab must be a positive integer, got {vocab!r}')
    steps, seq_len, batch_size, vocab = int(_steps), int(_seq), int(_bs), int(_vocab)
    device = DEVICE if device is None else (device if isinstance(device, torch.device) else torch.device(device))
    seed = set_seed(seed)
    rng = np.random.default_rng((seed + 7919) % (MAX_SEED + 1))
    if train_ids is None:
        window = rng.integers(0, vocab, size=seq_len + 1, dtype=np.int64)
        source = 'synthetic'
    else:
        arr = np.asarray(train_ids)
        if arr.ndim != 1:
            raise ValueError('production_loop_sanity_check: train_ids must be one-dimensional')
        if arr.size < seq_len + 1:
            raise ValueError(f'production_loop_sanity_check: need at least seq_len+1={seq_len + 1} tokens, got {arr.size}')
        start = int(rng.integers(0, int(arr.size) - int(seq_len)))
        window = np.ascontiguousarray(arr[start:start + seq_len + 1], dtype=np.int64)
        source = 'train_ids'
    if mlp_ratio is None:
        mlp_ratio = variant_mlp_ratio(variant, vocab, d=d, n_layers=n_layers,
                                      n_heads=n_heads, d_head=d_head,
                                      seq_len=seq_len,
                                      matched=PARAM_MATCHED | PARAM_MATCHED_V7)
    val = np.tile(window[None, :seq_len], (batch_size, 1)).astype(np.int64, copy=False)
    rec = train_variant(variant, window, val, vocab, seed=seed, d=d,
                        n_layers=n_layers, n_heads=n_heads, d_head=d_head,
                        seq_len=seq_len, batch_size=batch_size, steps=steps,
                        lr=lr, weight_decay=0.0, warmup=0, comp_lambda=0.0,
                        delta_lr_mult=1.0, eval_every=0, eval_subset=0,
                        val_bnd=None, device=device, log_every=0,
                        diag_every=1, measure_eff=False, amp_dtype='fp32',
                        eager_batch=False, return_model=True,
                        save_ckpt='none', ckpt_dir=None,
                        early_stop_patience=None)
    model = rec.pop('_model', None)
    try:
        if rec.get('cap_truncated'):
            final_loss = final_acc = float('nan')
            losses = []
            opt = {}
        else:
            losses = [float(v) for v in (rec.get('losses') or [])]
            x = torch.from_numpy(np.tile(window[:-1], (batch_size, 1))).to(device)
            y = torch.from_numpy(np.tile(window[1:], (batch_size, 1))).to(device)
            try:
                final_loss, final_acc = _lm_batch_metrics(model, x, y, vocab)
            finally:
                del x, y
            opt = rec.get('opt_diag_summary') if isinstance(rec.get('opt_diag_summary'), dict) else {}
    finally:
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    finite = all(math.isfinite(v) for v in losses)
    best_loss = min(losses) if losses else float('nan')
    initial_loss = losses[0] if losses else float('nan')
    loss_decreased = bool(math.isfinite(initial_loss) and math.isfinite(final_loss)
                          and final_loss < initial_loss)
    skipped = int(opt.get('skipped_updates') or 0)
    nonfinite_loss = int(opt.get('nonfinite_loss_steps') or 0)
    nonfinite_grad = int(opt.get('nonfinite_grad_steps') or 0)
    grad_max = opt.get('grad_norm_max')
    update_max = opt.get('update_norm_max')
    grad_ok = (grad_max is not None and math.isfinite(float(grad_max)) and float(grad_max) > 0.0)
    params_changed = bool(update_max is not None and math.isfinite(float(update_max))
                          and float(update_max) > 0.0)
    criteria = {
        'all_steps_completed': bool(not rec.get('cap_truncated')
                                    and int(rec.get('steps_completed') or 0) == steps),
        'no_skipped_updates': bool(skipped == 0),
        'no_nonfinite_loss_steps': bool(nonfinite_loss == 0),
        'no_nonfinite_grad_steps': bool(nonfinite_grad == 0),
        'gradients_finite_and_nonzero': bool(grad_ok),
        'parameters_changed': bool(params_changed),
        'loss_curve_all_finite': bool(finite),
        'loss_decreased': bool(loss_decreased),
        'final_loss_below_target': bool(math.isfinite(final_loss) and final_loss <= float(target_loss)),
        'accuracy_above_target': bool(math.isfinite(final_acc) and final_acc >= float(target_acc)),
    }
    passed = all(criteria.values())
    report = {
        'kind': 'production_loop_overfit_sanity',
        'variant': variant,
        'seed': int(seed),
        'code': code_stamp(),
        'device': str(device),
        'steps': steps,
        'lr': float(lr),
        'seq_len': seq_len,
        'batch_size': batch_size,
        'vocab_size': vocab,
        'architecture': {'d': int(d), 'n_layers': int(n_layers),
                         'n_heads': int(n_heads), 'd_head': int(d_head)},
        'mlp_ratio': float(mlp_ratio),
        'data_source': source,
        'data_fp': _ids_fp(window),
        'optimizer': 'build_optimizer/AdamW',
        'losses': losses,
        'initial_loss': float(initial_loss),
        'final_loss': float(final_loss),
        'best_loss': float(best_loss),
        'final_accuracy': float(final_acc),
        'loss_decreased': bool(loss_decreased),
        'grad_norm_max': grad_max,
        'update_norm_max': update_max,
        'params_changed': bool(params_changed),
        'skipped_updates': skipped,
        'nonfinite_loss_steps': nonfinite_loss,
        'nonfinite_grad_steps': nonfinite_grad,
        'target_loss': float(target_loss),
        'target_accuracy': float(target_acc),
        'pass_criteria': criteria,
        'passed': passed,
        'failed_criteria': [k for k, ok in criteria.items() if not ok],
    }
    report['diagnosis'] = ('ok' if passed else
                           'the production training loop did not overfit the fixed window '
                           '(see failed_criteria)')
    return report

def production_loop_sanity_check(variant='full', *, seed=0, **kwargs):
    """RNG-preserving wrapper around the production-loop overfit probe."""
    _fp_before = rng_fingerprint()
    with _preserve_rng_state():
        report = _production_loop_sanity_check_impl(variant, seed=seed, **kwargs)
    _fp_after = rng_fingerprint()
    report['rng_fp_before'] = _fp_before
    report['rng_fp_after'] = _fp_after
    report['rng_state_preserved'] = bool(_fp_before == _fp_after)
    report['pass_criteria']['rng_state_preserved'] = bool(report['rng_state_preserved'])
    if not report['rng_state_preserved']:
        report['passed'] = False
        report['failed_criteria'] = list(report.get('failed_criteria') or []) + ['rng_state_preserved']
        report['diagnosis'] = 'the production-loop probe did not restore the process RNG state'
    return report

def overfit_guard(variants, *, seed=0, steps=200, seq_len=64, batch_size=2, vocab=1024, train_ids=None, batch=None, ignore_index=None, mlp_ratio=None, force=False, target_loss=0.05, target_acc=0.99, arch=None, device=None):
    """Run `overfit_sanity_check` once per variant per process and hard-stop
    on failure.  Called automatically by `run()` before the first panel and by
    the length/checkpoint phases, so every training driver inherits the check.

    The in-process cache is keyed by the FULL probe configuration (seed, step
    count, shape, targets, architecture, MLP width, ignore-index convention,
    data source/fingerprint, code) as well as the corpus bytes, so a lenient
    probe can never certify a stricter one.  `arch` optionally overrides the
    probe architecture (d/n_layers/n_heads/d_head): pass the panel's own
    geometry on the scale ladder so the probe exercises the same head/layer
    shapes the real run will use; unset it for the tiny default probe.
    `mlp_ratio` accepts a scalar or a per-variant mapping.  `batch` accepts a
    task-specific ``(inputs, targets)`` pair (NIAH/masked next-token targets)
    and `ignore_index` carries that loop's label convention; passing either
    `batch` or `train_ids` runs the data-path probe.  `CSA_SKIP_OVERFIT=1` is
    the explicit escape hatch (e.g. a diagnostic box with a deliberately broken
    batch); it prints a loud skip line instead.  Passing is necessary, not
    sufficient."""
    if os.environ.get('CSA_SKIP_OVERFIT', '').strip().lower() in ('1', 'true', 'yes', 'on'):
        print('[overfit] guard SKIPPED (CSA_SKIP_OVERFIT is set) — this run has no small-batch safety check')
        return []
    if batch is not None and train_ids is not None:
        raise ValueError('overfit_guard: pass either train_ids or batch, not both')
    seed = _seed_value(seed)
    variants = list(variants)
    _arch = _probe_arch(arch)
    _ign = None
    _data_fp = None
    if batch is not None:
        src = 'batch'
        _x, _y = _batch_arrays(batch, vocab=vocab)
        seq_len, batch_size = int(_x.shape[1]), int(_x.shape[0])
        _data_fp = _ids_fp(_x.reshape(-1)) + ':' + _ids_fp(_y.reshape(-1))
        _ign = -100 if ignore_index is None else int(ignore_index)
    elif train_ids is not None:
        src = 'data'
        _arr = np.asarray(train_ids)
        _data_fp = _ids_fp(_arr if _arr.ndim == 1 else np.ascontiguousarray(_arr).reshape(-1))
        _ign = None if ignore_index is None else int(ignore_index)
    else:
        src = 'synthetic'
        _ign = None if ignore_index is None else int(ignore_index)
    if _arch != _OVERFIT_ARCH:
        print(f"[overfit] probe architecture override: d={_arch['d']} n_layers={_arch['n_layers']} n_heads={_arch['n_heads']} d_head={_arch['d_head']} (default tiny probe is d={_OVERFIT_ARCH['d']}/{_OVERFIT_ARCH['n_layers']}L/{_OVERFIT_ARCH['n_heads']}H)" )
    _probe_cfgs = {}
    _cfg_keys = {}
    _semantics = {}
    for v in variants:
        _pc = _overfit_probe_config(seed=seed, steps=steps, seq_len=seq_len, batch_size=batch_size, vocab=vocab, target_loss=target_loss, target_acc=target_acc, arch=_arch, mlp_ratio=_overfit_ratio_for(mlp_ratio, v), ignore_index=_ign, source=src, data_fp=_data_fp, device=device)
        _probe_cfgs[v] = _pc
        _cfg_keys[v] = _overfit_probe_digest(_pc)
        _semantics[v] = _overfit_probe_semantics(_pc)

    def _key(v):
        if src in ('data', 'batch'):
            return (v, src, int(vocab), _data_fp, _cfg_keys[v])
        return (v, 'synthetic', _cfg_keys[v])

    def _known(v):
        # Only a PASSED check may certify a variant.  A cached failure must be
        # retried (or re-raised) on the next call: treating any present key as
        # "already checked" would let a caller that caught the first failure
        # bypass the hard stop on a later phase in the same process.
        if _OVERFIT_GUARD.get(_key(v)) is True:
            return True
        if src == 'synthetic':
            # A passed data/batch probe of the same model/probe geometry
            # subsumes the synthetic plumbing check.
            for _k in list(_OVERFIT_GUARD):
                if (_k[0] == v and _k[1] in ('data', 'batch')
                        and _OVERFIT_GUARD.get(_k) is True
                        and (_OVERFIT_REPORTS.get(_k) or {}).get('guard_probe_semantics') == _semantics[v]):
                    return True
        return False

    todo = [v for v in variants if force or not _known(v)]
    if not todo:
        return []
    reports = []
    for v in todo:
        r = overfit_sanity_check(v, seed=seed, steps=steps, d=_arch['d'], n_layers=_arch['n_layers'], n_heads=_arch['n_heads'], d_head=_arch['d_head'], seq_len=seq_len, batch_size=batch_size, vocab=vocab, train_ids=(train_ids if src == 'data' else None), batch=(batch if src == 'batch' else None), ignore_index=_ign, mlp_ratio=_overfit_ratio_for(mlp_ratio, v), target_loss=target_loss, target_acc=target_acc, device=device)
        r['guard_probe_config'] = _probe_cfgs[v]
        r['guard_probe_key'] = _cfg_keys[v]
        r['guard_probe_semantics'] = _semantics[v]
        _key_v = _key(v)
        _OVERFIT_GUARD[_key_v] = bool(r['passed'])
        _OVERFIT_REPORTS[_key_v] = r
        reports.append(r)
        print(f"[overfit] {v:<24} loss {r['initial_loss']:.3f} -> {r['final_loss']:.4f}  acc {r['final_accuracy'] * 100:.1f}%  grad_finite={r['grad_all_finite']}  params_changed={r['params_changed']}  {'PASS' if r['passed'] else 'FAIL'}")
        if not r['passed']:
            raise RuntimeError(f'small-batch overfit sanity check FAILED for `{v}` ({r["diagnosis"]}) — do NOT start or continue this run; this points at a gradient/loss/label plumbing fault.  Inspect the returned report and set CSA_SKIP_OVERFIT=1 only if you deliberately accept the risk.')
    return reports

_REPRO_GUARD = {}
_REPRO_REPORTS = {}

_REPRO_ARCH = {'d': 32, 'n_layers': 1, 'n_heads': 4, 'd_head': 8}

def _repro_probe_arch(arch=None):
    got = dict(_REPRO_ARCH)
    if arch:
        for _k in got:
            if arch.get(_k) is not None:
                got[_k] = int(arch[_k])
    return got

def _repro_batch(seed, *, batch_size, seq_len, vocab, train_ids=None):
    """Fixed probe batch + its identity.

    With ``train_ids`` the batch is a deterministic slice of the REAL token
    stream (the same corpus the panel trains on); otherwise it is synthetic.
    The generator is separate from the model's PyTorch RNG stream."""
    _rng = np.random.default_rng((_seed_value(seed) + 7919) % (MAX_SEED + 1))
    if train_ids is not None:
        arr = np.asarray(train_ids).reshape(-1)
        need = int(seq_len) + 1
        if arr.size < need:
            raise ValueError(f'reproducibility probe: need at least seq_len+1={need} tokens, got {arr.size}')
        starts = _rng.integers(0, int(arr.size) - int(seq_len), size=int(batch_size))
        chunk = np.stack([arr[int(s):int(s) + int(seq_len) + 1] for s in starts]).astype(np.int64, copy=False)
        return chunk[:, :-1], chunk[:, 1:], _ids_fp(chunk.reshape(-1)), 'train_ids'
    chunk = _rng.integers(0, max(int(vocab), 1), size=(int(batch_size), int(seq_len) + 1), dtype=np.int64)
    return chunk[:, :-1], chunk[:, 1:], _ids_fp(chunk.reshape(-1)), 'synthetic'

def _repro_tiny_losses(variant, seed, *, steps=3, d=32, n_layers=1, n_heads=4, d_head=8, seq_len=16, batch_size=2, vocab=64, lr=0.003, train_ids=None, mlp_ratio=None, device=None):
    """Run one very short, fully seeded training stream and return its losses.

    Uses a fixed batch (sampled from its own NumPy generator, so the data
    stream cannot consume the model's PyTorch RNG) and disables weight decay.
    Two calls with the same ``seed`` must agree bit-for-bit on the same stack;
    that is the property the pre-flight reproducibility guard checks.  The
    architecture/MLP/data are caller-selectable so a scale panel can repeat
    its REAL geometry instead of a tiny proxy."""
    device = DEVICE if device is None else (device if isinstance(device, torch.device) else torch.device(device))
    seed = set_seed(seed)
    x_np, y_np, _data_fp, _src = _repro_batch(seed, batch_size=batch_size, seq_len=seq_len, vocab=vocab, train_ids=train_ids)
    x = torch.from_numpy(np.ascontiguousarray(x_np)).to(device)
    y = torch.from_numpy(np.ascontiguousarray(y_np)).to(device)
    if mlp_ratio is None:
        mlp_ratio = variant_mlp_ratio(variant, int(vocab), d=int(d), n_layers=int(n_layers), n_heads=int(n_heads), d_head=int(d_head), seq_len=int(seq_len), matched=PARAM_MATCHED | PARAM_MATCHED_V7)
    cfgs = make_layer_cfgs(int(n_layers), variant)
    model = SmallGPT(int(vocab), int(d), int(n_layers), int(n_heads), int(d_head), int(seq_len), cfgs, mlp_ratio=mlp_ratio).to(device)
    opt = build_optimizer(model, lr=lr, weight_decay=0.0, delta_lr_mult=1.0)
    model.train()
    losses = []
    try:
        for _ in range(int(steps)):
            ce = model.ce_loss(x, y, int(vocab))
            opt.zero_grad(set_to_none=True)
            ce.backward()
            opt.step()
            losses.append(float(ce.detach()))
    finally:
        del model, opt, x, y, x_np, y_np
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return losses

def _repro_probe_config(*, seed, steps, tol, arch, seq_len, batch_size, vocab, mlp_ratio, data_fp, train_ids, lr=0.003, device=None):
    return {
        'schema': 3,
        'seed': int(seed),
        'steps': int(steps),
        'tol': float(tol),
        'lr': float(lr),
        'architecture': _repro_probe_arch(arch),
        'seq_len': int(seq_len),
        'batch_size': int(batch_size),
        'vocab': int(vocab),
        'mlp_ratio': (None if mlp_ratio is None else float(mlp_ratio)),
        'data_source': ('train_ids' if train_ids is not None else 'synthetic'),
        'data_fp': (None if data_fp is None else str(data_fp)),
        'device': str(DEVICE if device is None else device),
        'determinism': determinism_label(),
        'code': code_stamp(),
    }

def _repro_probe_digest(cfg):
    return _canonical_json_digest(cfg)

def reproducibility_check(variant='full', *, seed=0, steps=3, tol=1e-06, device=None, arch=None, train_ids=None, vocab=None, mlp_ratio=None, seq_len=None, batch_size=2, lr=0.003):
    """Two identical short runs + one different-seed run, compared.

    This is the automated version of "same seed, same result on this box":
    the same seed must reproduce the complete short loss curve (exactly on
    CPU; within ``tol`` on a GPU whose kernels are not bitwise deterministic),
    while a different seed must actually change it.  `arch`/`train_ids` let a
    scale panel repeat its REAL architecture on a small fixed slice of the REAL
    corpus rather than a tiny synthetic proxy.  The report is persisted with
    the panel, so a reproducibility problem is visible instead of being an
    assumption.  The check never changes the RNG state of the following
    training cell."""
    device = DEVICE if device is None else (device if isinstance(device, torch.device) else torch.device(device))
    seed = _seed_value(seed)
    _steps = _exact_count(steps, minimum=1)
    if _steps is None:
        raise ValueError(f'reproducibility_check: steps must be a positive integer, got {steps!r}')
    steps = int(_steps)
    _bs = _exact_count(batch_size, minimum=1)
    if _bs is None:
        raise ValueError(f'reproducibility_check: batch_size must be a positive integer, got {batch_size!r}')
    batch_size = int(_bs)
    try:
        _lr = float(lr)
        _tol = float(tol)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'reproducibility_check: lr/tol must be finite numbers, got lr={lr!r} tol={tol!r}') from None
    if not math.isfinite(_lr) or not math.isfinite(_tol) or _tol < 0:
        raise ValueError(f'reproducibility_check: lr/tol must be finite (tol >= 0), got lr={lr!r} tol={tol!r}')
    lr, tol = _lr, _tol
    _arch = _repro_probe_arch(arch)
    if vocab is None:
        vocab = (int(np.max(train_ids)) + 1) if train_ids is not None else 64
    if seq_len is None:
        seq_len = 16
    _seq = _exact_count(seq_len, minimum=1)
    if _seq is None:
        raise ValueError(f'reproducibility_check: seq_len must be a positive integer, got {seq_len!r}')
    seq_len = int(_seq)
    # Resolve the data identity once, using the SAME generator the losses use.
    _x, _y, _data_fp, _src = _repro_batch(seed, batch_size=batch_size, seq_len=seq_len, vocab=vocab, train_ids=train_ids)
    _fp_before = rng_fingerprint()
    _kw = dict(steps=steps, d=_arch['d'], n_layers=_arch['n_layers'], n_heads=_arch['n_heads'], d_head=_arch['d_head'], seq_len=seq_len, batch_size=batch_size, vocab=int(vocab), lr=lr, train_ids=train_ids, mlp_ratio=mlp_ratio, device=device)
    with _preserve_rng_state():
        a = _repro_tiny_losses(variant, seed, **_kw)
        b = _repro_tiny_losses(variant, seed, **_kw)
        c = _repro_tiny_losses(variant, _other_seed(seed), **_kw)
    _fp_after = rng_fingerprint()
    _rng_preserved = bool(_fp_before == _fp_after)
    _n = min(len(a), len(b))
    max_diff = max((abs(a[i] - b[i]) for i in range(_n)), default=float('inf'))
    different_seed_diff = max((abs(a[i] - c[i]) for i in range(min(len(a), len(c)))), default=0.0)
    finite = all(math.isfinite(v) for v in a + b + c)
    bitwise = bool(a == b)
    passed = bool(finite and max_diff <= float(tol) and different_seed_diff > 0.0
                  and _rng_preserved)
    r = {
        'kind': 'duplicate_run_reproducibility',
        'variant': variant,
        'seed': int(seed),
        'code': code_stamp(),
        'steps': int(steps),
        'tol': float(tol),
        'lr': float(lr),
        'device': str(device),
        'determinism': determinism_label(),
        'architecture': _arch,
        'mlp_ratio': (None if mlp_ratio is None else float(mlp_ratio)),
        'seq_len': int(seq_len),
        'batch_size': int(batch_size),
        'data_source': _src,
        'data_fp': _data_fp,
        'same_seed_losses_a': a,
        'same_seed_losses_b': b,
        'different_seed_losses': c,
        'max_abs_diff_same_seed': float(max_diff),
        'bitwise_identical': bitwise,
        'different_seed_changes_curve': bool(different_seed_diff > 0.0),
        'different_seed_max_abs_diff': float(different_seed_diff),
        'all_finite': bool(finite),
        'rng_fp_before': _fp_before,
        'rng_fp_after': _fp_after,
        'rng_state_preserved': _rng_preserved,
        'passed': passed,
    }
    if not passed:
        if not finite:
            r['diagnosis'] = 'non-finite loss in a tiny seeded run'
        elif max_diff > float(tol):
            r['diagnosis'] = 'same seed did not reproduce the short loss curve on this machine/stack'
        elif not _rng_preserved:
            r['diagnosis'] = 'the reproducibility probe did not restore the process RNG state'
        else:
            r['diagnosis'] = 'a different seed produced the identical loss curve (seed is not wired into the run)'
    else:
        r['diagnosis'] = 'ok'
    return r

def _repro_guard_base(*, seed=0, steps=3, tol=1e-06, arch=None, vocab=None, mlp_ratio=None, seq_len=None, batch_size=2, train_ids=None, data_fp=None, lr=0.003, device=None):
    """Exact cache key shared by the guard and run()'s evidence recovery.

    Strictly a description of the probe that produced a PASS: changing the
    tolerance, step count, architecture, MLP width, sequence length, vocab,
    device, determinism level, data stream or code invalidates the cached
    certificate.  A pass under a loose tolerance must never be reused for a
    stricter request."""
    _fp = data_fp
    if _fp is None and train_ids is not None:
        _arr = np.asarray(train_ids)
        _fp = _ids_fp(_arr if _arr.ndim == 1 else np.ascontiguousarray(_arr).reshape(-1))
    return _repro_probe_digest(_repro_probe_config(seed=seed, steps=steps, tol=tol, arch=arch, seq_len=(16 if seq_len is None else seq_len), batch_size=batch_size, vocab=(64 if vocab is None else vocab), mlp_ratio=mlp_ratio, data_fp=_fp, train_ids=train_ids, lr=lr, device=device))

def reproducibility_guard(variants, *, seed=0, steps=3, tol=1e-06, force=False, arch=None, train_ids=None, vocab=None, mlp_ratio=None, seq_len=None, batch_size=2, seeds=None, strict=None, lr=0.003, device=None):
    """Run ``reproducibility_check`` once per (variant, seed, probe config) per process.

    Mirrors ``overfit_guard``: results are cached as evidence, a skip env var
    is explicit and loud, and a failure is a warning unless the caller sets
    ``CSA_REQUIRE_REPRO=1`` (or passes ``strict=True``).  The cache key
    includes the full probe configuration -- tolerance, code stamp, seed,
    device, architecture, MLP width, sequence length, vocabulary and data
    fingerprint -- so a loose/synthetic probe cannot certify a real one.
    ``seeds`` lets a panel check its first and last seed; ``arch`` and
    ``train_ids`` make it exercise the real architecture on a small slice of
    the real corpus.  Returns the reports that ran in this call."""
    if os.environ.get('CSA_SKIP_REPRO', '').strip().lower() in ('1', 'true', 'yes', 'on'):
        print('[repro] guard SKIPPED (CSA_SKIP_REPRO is set) — this run has no same-machine seed check')
        return []
    _want_strict = (os.environ.get('CSA_REQUIRE_REPRO', '').strip().lower() in ('1', 'true', 'yes', 'on')) if strict is None else bool(strict)
    _device = DEVICE if device is None else device
    _arch = _repro_probe_arch(arch)
    if vocab is None:
        vocab = (int(np.max(train_ids)) + 1) if train_ids is not None else 64
    if seq_len is None:
        seq_len = 16
    _raw_seeds = list(seeds if seeds is not None else [seed])
    _parsed_seeds = []
    for _raw in _raw_seeds:
        try:
            _parsed_seeds.append(_seed_value(_raw))
        except ValueError as _se:
            raise ValueError(f'reproducibility_guard seeds must be integers in [0, {MAX_SEED}], got {_raw_seeds!r}') from _se
    _seed_list = list(dict.fromkeys(_parsed_seeds))
    _full_data_fp = None
    if train_ids is not None:
        _arr = np.asarray(train_ids)
        _full_data_fp = _ids_fp(_arr if _arr.ndim == 1 else np.ascontiguousarray(_arr).reshape(-1))
    _base_cache = {}

    def _base_for(v, sd):
        key = (v, sd)
        if key not in _base_cache:
            _base_cache[key] = _repro_guard_base(seed=sd, steps=steps, tol=tol, arch=_arch, vocab=vocab, mlp_ratio=_overfit_ratio_for(mlp_ratio, v), seq_len=seq_len, batch_size=batch_size, train_ids=train_ids, data_fp=_full_data_fp, lr=lr, device=_device)
        return _base_cache[key]

    # As in overfit_guard, a cached FAILURE is not a certificate: retry it so a
    # later phase cannot silently inherit the failed status as "already done".
    todo = [(v, sd) for v in variants for sd in _seed_list if force or _REPRO_GUARD.get((v, _base_for(v, sd))) is not True]
    reports = []
    for v, sd in todo:
        r = reproducibility_check(v, seed=sd, steps=steps, tol=tol, arch=_arch, train_ids=train_ids, vocab=vocab, mlp_ratio=_overfit_ratio_for(mlp_ratio, v), seq_len=seq_len, batch_size=batch_size, lr=lr, device=_device)
        _base = _base_for(v, sd)
        r['guard_probe_key'] = _base
        r['guard_probe_config'] = _repro_probe_config(seed=sd, steps=steps, tol=tol, arch=_arch, seq_len=seq_len, batch_size=batch_size, vocab=vocab, mlp_ratio=_overfit_ratio_for(mlp_ratio, v), data_fp=r.get('data_fp'), train_ids=train_ids, lr=lr, device=_device)
        key = (v, _base)
        _REPRO_GUARD[key] = bool(r['passed'])
        _REPRO_REPORTS[key] = r
        reports.append(r)
        print(f"[repro] {v:<24} seed={sd} same-seed max|Δ|={r['max_abs_diff_same_seed']:.3e}  different-seed Δ={r['different_seed_max_abs_diff']:.3e}  bitwise={r['bitwise_identical']}  {'PASS' if r['passed'] else 'FAIL'}")
        if not r['passed']:
            _msg = f"same-machine reproducibility check FAILED for `{v}` seed={sd} ({r['diagnosis']})"
            if _want_strict:
                raise RuntimeError(_msg + ' — strict reproducibility is required for this run. Correct the seed/kernel nondeterminism or explicitly opt out (CSA_REQUIRE_REPRO=0).')
            print(f'[repro] WARNING: {_msg}; the panel will record the failure (set CSA_REQUIRE_REPRO=1 to make it fatal, CSA_SKIP_REPRO=1 to skip)')
    return reports

def reproducibility_reports_for(variants, *, probe_key=None, data_fp=None):
    """Exact-match recovery of reproducibility evidence for the current panel."""
    _want = set(variants)
    out = []
    for _k, _r in _REPRO_REPORTS.items():
        if _k[0] not in _want:
            continue
        if probe_key is not None and _k[1] != probe_key:
            continue
        if data_fp is not None and _r.get('data_fp') != data_fp:
            continue
        out.append(dict(_r))
    return out

def reproducibility_reports():
    """Copy of the reproducibility reports gathered in this process."""
    return {k: dict(v) for k, v in _REPRO_REPORTS.items()}

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

    @staticmethod
    def _class_key(d, n_layers, seq_len, batch_size, variant=None):
        """Calibration key for one (geometry, variant) class.

        Geometry alone is too coarse on the B-tier ladder: a dense full-attention
        arm and a sparse CSA arm with the same shapes can differ by a large
        factor, so one shared rate would systematically under- or over-book
        cells.  `variant=None` keeps the historical geometry-only key used by
        the v7-v12 drivers, so existing state files stay valid."""
        base = str((int(d), int(n_layers), int(seq_len), int(batch_size)))
        return base if variant is None else f'{base}|{variant}'

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
                    # it to 0 would erase those seconds from the
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

    def record_run(self, seconds, steps_done, d, n_layers, seq_len, batch_size, calib_seconds=None, variant=None):
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
                key = self._class_key(d, n_layers, seq_len, batch_size, variant)
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

    def estimate_seconds(self, steps, d=256, n_layers=6, seq_len=512, batch_size=12, variant=None):
        by_class = self.state.get('sps_by_class') or {}
        key = self._class_key(d, n_layers, seq_len, batch_size, variant)
        exact = by_class.get(key)
        if exact is None and variant is not None:
            exact = by_class.get(self._class_key(d, n_layers, seq_len, batch_size, None))
        if exact is not None:
            return steps * exact * 1.15
        norm = self.state.get('norm_sps')
        if norm is None:
            norm = 0.35
        f = self._wallclock_factor(d, n_layers, seq_len, batch_size)
        return steps * norm * f * 1.15

    def note_watchdog_kill(self, elapsed_seconds, steps, d=256, n_layers=6, seq_len=512, batch_size=12, variant=None):
        """Lift the calibrated per-step floor after a dispatcher-watchdog kill
        (par.py).  A cell killed at `elapsed_seconds` demonstrably needs MORE
        than that for `steps` steps, so its true sps exceeds
        elapsed/steps — yet a killed cell never records a run, so without
        this hook the calibration (and hence the watchdog bound
        max(3*est, est+1h) derived from it) would stay wrong and the same
        healthy cell would be killed at the same wall-clock point on every
        resume (uncalibrated hardware, CPU workers).  Lifting the floor
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
            key = self._class_key(d, n_layers, seq_len, batch_size, variant)
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
        v = _as_seed_int(r.get('params'))
        if v is not None and v > 0:
            return v
    return None
