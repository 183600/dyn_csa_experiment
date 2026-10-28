
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
        val_ids = L.load_wikitext(max(train_len, need), 8000000)[1]
    # The p1l recipe must describe the implementation that ACTUALLY
    # runs and the training corpus it actually reads.  Both are derived
    # explicitly before training and carried into the checkpoint/record.
    train_ids = L.load_wikitext(train_len, 8000000)[0]
    _train_fp = L._ids_fp(train_ids)
    _val_fp = L._ids_fp(val_ids)
    _diag_eval_every = int(payload.get('diag_eval_every', 0) or 0)
    _amp_mode, _ = L._resolve_impl_modes(
        {'amp_dtype': payload.get('amp_dtype', 'auto'),
         'batch_size': batch_size}, device=L.DEVICE)
    # The ACTUAL per-arm dispatch, not the backend capability: every frozen
    # rope arm takes the per-row `_single_rope` loop even when the CUDA batched
    # twin exists.  Using the backend default here wrote eager_batch=True into
    # the recipe while train_variant recorded False (and vice versa).
    _plan_cfgs = L.make_layer_cfgs(6, v)
    _eager_mode = L._uses_batched_eager(_plan_cfgs, L.DEVICE, batch_size)
    _tk_fp = L.training_knobs_fingerprint(amp_dtype=_amp_mode, eager_batch=_eager_mode)
    mr = 4.0
    if v in PARAM_MATCHED_V7:
        mr = L.variant_mlp_ratio(v, vocab, d=256, n_layers=6, n_heads=8, d_head=32, seq_len=train_len, matched=set(PARAM_MATCHED_V7), ref_variant='csa_fixed_rope')
    _recipe = (f'p1l_steps{steps}_tl{train_len}_v{vocab}_bs{batch_size}_lr0.0003_nt8000000_wu50_wd0.1_'
               f'cl0.05_dlm10.0_mr{mr:.8g}-csa_fixed_rope_ns{n_seq}_'
               f'df{_val_fp}'
               f'_gc{L.GRAD_CLIP_NORM}_b{L.OPT_ADAMW_BETAS[0]},{L.OPT_ADAMW_BETAS[1]}'
               f'_eps{L.OPT_ADAMW_EPS:g}_tk{_tk_fp}'
               f'_amp{_amp_mode or "fp32"}_eb{int(bool(_eager_mode))}'
               f'_det{L.determinism_label()}_trfp{_train_fp}')
    # A proper arm-independent run fingerprint for `pair_reason`: the p1l recipe
    # above intentionally carries the per-arm `mr`, so using it as `run_cfg`
    # made every frozen pair fail the "different run_cfg" check.
    _run_cfg = L._run_fp({'steps': steps, 'seq_len': train_len, 'batch_size': batch_size,
                          'n_train_tokens': 8000000, 'lr': 0.0003, 'weight_decay': 0.1,
                          'warmup': 50, 'comp_lambda': 0.05, 'delta_lr_mult': 10.0,
                          'eval_every': _diag_eval_every,
                          'eval_subset': (8 if _diag_eval_every else 128),
                          'amp_dtype': _amp_mode, 'eager_batch': _eager_mode,
                          'mlp_match_ref': 'csa_fixed_rope'},
                         256, 6, 8, 32, vocab, train_ids, val_ids, device=L.DEVICE)
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
            elif _diag_eval_every > 0 and (not isinstance(_meta.get('curve_health'), dict)
                                           or str((_meta.get('curve_health') or {}).get('status')) in ('unmonitored', 'insufficient_data')):
                stale = True
                print(f'[p1l] {v} s{seed}: checkpoint has no usable training-curve health record (status={((_meta.get("curve_health") or {}).get("status") if isinstance(_meta.get("curve_health"), dict) else None)!r}) — RETRAINING so the pre-test health gate has evidence (this does not change the frozen recipe)')
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
            _deadline += booked
        L.set_seed(seed)
        cfgs = _plan_cfgs
        # Training-health diagnostics only: a fixed validation slice is scored
        # on the eval grid so P5C can refuse the one-shot test access on a run
        # that is overfitting/stalled/non-finite.  It never selects a
        # checkpoint or changes the frozen arms/seeds/endpoint.
        _diag_val = None
        if _diag_eval_every:
            _va = np.asarray(val_ids)
            if _va.ndim == 2 and _va.shape[1] >= train_len:
                _diag_val = np.ascontiguousarray(_va[:min(8, int(_va.shape[0])), :int(train_len)], dtype=np.int64)
        try:
            # The parent ran only the SYNTHETIC pre-flight.  The real token
            # stream can still be broken (vocabulary mapping, cache mixing,
            # label shift downstream), so run the same small-batch overfit
            # probe on the actual corpus before starting the cell.  The probe
            # is RNG-preserving, so it cannot perturb this cell.
            try:
                L.overfit_guard([v], train_ids=train_ids, vocab=vocab, steps=150,
                                seq_len=L.probe_seq_len(train_len), batch_size=2,
                                mlp_ratio=mr,
                                arch={'d': 256, 'n_layers': 6, 'n_heads': 8, 'd_head': 32})
            except RuntimeError as _fe:
                _p1l_prov = L.experiment_provenance(_prov_cfg, train_ids=train_ids, val_batch=val_ids, vocab=vocab)
                _pfp = L.preflight_fingerprint(payload, stage='lenphase_overfit', vocab=int(vocab), data_fp=L._ids_fp(train_ids), probe_cfg={'seq_len': L.probe_seq_len(train_len), 'arch': {'d': 256, 'n_layers': 6, 'n_heads': 8, 'd_head': 32}})
                L.atomic_write_json(os.path.join(outdir, 'overfit_sanity.json'), {
                    'kind': 'small_batch_overfit_sanity_panel', 'stage': 'lenphase_preflight',
                    'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'run_cfg': _pfp,
                    'passed': False, 'error': f'{_fe}', 'provenance': _p1l_prov,
                    'reports': [dict(r) for k, r in L._OVERFIT_REPORTS.items() if k[0] == v],
                })
                L.log_experiment_event(outdir, {'kind': 'preflight', 'tracking_schema': 2, 'status': 'failed', 'stage': 'lenphase_overfit', 'run_cfg': _pfp, 'code': CKPT_CODE, 'provenance': _p1l_prov, 'variant': v, 'seed': int(seed), 'error': f'{_fe}', 'sanity': {'passed': False}})
                raise
            if _diag_eval_every and _diag_val is not None:
                print(f'[p1l] {v} s{seed}: diagnostic validation slice {_diag_val.shape}, eval_every={_diag_eval_every} (health monitoring only; not used for selection)')
            _tr = L.train_variant(v, train_ids, _diag_val, vocab, seed=seed, d=256, n_layers=6, n_heads=8, d_head=32, seq_len=train_len, batch_size=batch_size, steps=steps, lr=0.0003, weight_decay=0.1, warmup=50, comp_lambda=0.05, delta_lr_mult=10.0, eval_every=_diag_eval_every, eval_subset=(int(_diag_val.shape[0]) if _diag_val is not None else 0), mlp_ratio=mr, device=DEVICE, log_every=500, deadline_ts=_deadline, return_model=True, measure_eff=True, amp_dtype=_amp_mode, eager_batch=_eager_mode)
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
        # Bind the file to the arm/seed and to the exact weights it contains.
        # The recipe deliberately omits the arm name (two frozen controls can
        # share the same recipe and parameter count), so without these fields a
        # copied checkpoint from another arm/seed passed every P5T identity
        # check and could enter the confirmation panel as the wrong weights.
        _sd = model.state_dict()
        _sd_sha = L.state_dict_sha256(_sd)
        _atomic_torch_save({'cfg': cfgs, 'mlp_ratio': mr, 'vocab': vocab, 'd': 256, 'n_layers': 6, 'n_heads': 8, 'd_head': 32, 'train_len': train_len, 'max_seq': train_len, 'code': CKPT_CODE, 'variant': v, 'seed': int(seed), 'state_dict_sha256': _sd_sha, 'recipe': _recipe, 'run_cfg': _run_cfg, 'sd': _sd, 'final_ppl': _tr.get('ppl'), 'params': L.count_params(model), 'rng_fp': _tr.get('rng_fp'), 'train_fp': _tr.get('data_fp'), 'val_fp': _tr.get('eval_fp'), 'losses': _tr.get('losses'), 'ppl_history': _tr.get('ppl_history'), 'curve_health': _tr.get('curve_health'), 'health_history': _tr.get('health_history'), 'opt_diag_summary': _tr.get('opt_diag_summary'), 'best_step': _tr.get('best_step'), 'best_val_ppl': _tr.get('best_val_ppl'), 'train_ce_fixed_final': _tr.get('train_ce_fixed_final'), 'train_knobs': _tr.get('train_knobs'), 'train_objective': _tr.get('train_objective'), 'param_breakdown': _tr.get('param_breakdown'), 'n_seq': n_seq, 'steps': steps}, ck)
        if guard is not None:
            guard.record_run(time.time() - t0, steps, 256, 6, train_len, batch_size, calib_seconds=_tr.get('train_time_s'))
            if booked:
                guard.book(-booked, token=_book_tok)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    key = f'{v}::seed{seed}'
    _cur = L.locked_json_read(spath).get(key)
    if isinstance(_cur, dict) and _cur.get('synthesized'):
        # A reconstructed aggregate is not an independent measurement.  The
        # warmup path already drops it; this path must not let it satisfy
        # resume and mark the length panel complete.
        print(f'[resume] {key}: existing by_len record is a SYNTHESIZED reconstruction — re-evaluating it from the checkpoint')
        _cur = None
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
    _ckp_train_fp = _ckp.get('train_fp')
    if guard is not None:
        guard.record_run(time.time() - _t_ev, 0, 0, 0, 0, 0)

    def _merge(summary):
        _merged = {}
        _cur_now = summary.get(key)
        for _bk, _bc in (((_cur_now or {}).get('by_len') or {}).items() if _need_lens != list(eval_lens) else []):
            try:
                _merged[int(_bk)] = _bc
            except (TypeError, ValueError, OverflowError):
                continue
        _merged.update(r)
        _rec = {'variant': v, 'seed': seed, 'params': _ckp_params, 'max_pos': mp, 'by_len': _merged, '_code': CKPT_CODE, 'recipe': _recipe, 'run_cfg': _run_cfg, 'eval_n_seq': n_seq, 'n_seq': n_seq}
        _prev_resume = _cur_now if isinstance(_cur_now, dict) else {}
        # The training record carries the same weight identity as the
        # checkpoint it produced, so P5T can detect a record swapped between
        # arms/seeds even when both share a recipe and parameter count.
        _rec['state_dict_sha256'] = (_ckp.get('state_dict_sha256')
                                     or _prev_resume.get('state_dict_sha256'))
        # `pair_reason` requires steps/tokens_seen to be present and equal;
        # both are recorded here so a frozen pair is not "unverifiable
        # steps/tokens_seen".
        _rec['steps'] = int((_tr if isinstance(_tr, dict) else _prev_resume).get('steps', steps))
        _rec['tokens_seen'] = int((_tr if isinstance(_tr, dict) else _prev_resume).get(
            'tokens_seen', int(steps) * int(batch_size) * int(train_len)))
        # Fail closed OUTSIDE the provenance try/except below: a validation
        # fingerprint is NOT a training fingerprint.  Writing one here would
        # let a later confirmation round treat the run as trained on the eval
        # slice; retraining (which records the real corpus) is the only honest
        # recovery.
        _resume_fp = (_ckp_train_fp or _prev_resume.get('data_fp')) if _tr is None else None
        if _tr is None and (not isinstance(_resume_fp, str) or not _resume_fp):
            raise RuntimeError(
                f'[p1l] REFUSE resume {key}: neither the checkpoint nor the existing record carries a training-corpus fingerprint (data_fp); re-run training / regenerate the checkpoint instead of relabelling the validation slice as the training corpus')
        try:
            _prov = L.experiment_provenance(_prov_cfg, train_ids=(train_ids if _tr is not None else None), val_batch=val_ids, vocab=vocab)
            _rec['provenance'] = _prov
            _known_knobs = ((_tr or {}).get('train_knobs') if _tr is not None else _prev_resume.get('train_knobs'))
            _known_knobs = _known_knobs if isinstance(_known_knobs, dict) else {}
            _rec['hyperparams'] = L.run_hyperparameters({'steps': steps, 'n_train_tokens': _prov_cfg['n_train_tokens'], 'seq_len': train_len, 'batch_size': batch_size, 'lr': 0.0003, 'weight_decay': 0.1, 'warmup': 50, 'comp_lambda': 0.05, 'delta_lr_mult': 10.0}, seed=seed, d=256, n_layers=6, n_heads=8, d_head=32, vocab=vocab, mlp_ratio=mr, amp_dtype=_known_knobs.get('amp_dtype', _amp_mode), eager_batch=_known_knobs.get('eager_batch', _eager_mode))
            if _tr is not None:
                _rec['data_fp'] = _prov['data'].get('train_fp') or _prov['data'].get('val_fp')
                _rec['rng_fp'] = _tr.get('rng_fp')
                _rec['losses'] = _tr.get('losses')
                _rec['ppl_history'] = _tr.get('ppl_history')
                _rec['train_knobs'] = _tr.get('train_knobs')
                _rec['train_objective'] = _tr.get('train_objective')
                _rec['param_breakdown'] = _tr.get('param_breakdown')
            else:
                # A partial resume re-evaluates only missing lengths and has no
                # fresh training record.  Preserve the corpus/optimizer
                # provenance of the existing record (the fail-closed check
                # above already guaranteed a real training fingerprint).
                _rec['data_fp'] = _resume_fp
                for _kk in ('rng_fp', 'losses', 'ppl_history', 'train_knobs', 'train_objective', 'param_breakdown'):
                    if _prev_resume.get(_kk) is not None:
                        _rec[_kk] = _prev_resume[_kk]
            _rec['val_fp'] = _prov['data'].get('val_fp')
            _rec['data_version'] = _prov['data'].get('data_version')
        except Exception as _pe:
            print(f'[p1l] WARNING: could not attach provenance to {key} ({type(_pe).__name__}: {_pe})')
        # Never fabricate a train_knobs descriptor on resume.  The
        # weights may have been trained under a different numeric path, which
        # would make the frozen implementation binding certify a false
        # descriptor.
        if not isinstance(_rec.get('train_knobs'), dict):
            raise RuntimeError(f'[p1l] REFUSE resume {key}: the record/checkpoint carries no real train_knobs descriptor; retrain instead of fabricating an implementation claim')
        _rec.setdefault('train_objective', _prev_resume.get('train_objective'))
        _rec.setdefault('param_breakdown', _prev_resume.get('param_breakdown'))
        summary[key] = _rec
        print(f'  [p1l] {v:18s} s{seed} ' + '  '.join((f'L{k}={vv.get('ppl', float('nan')):.2f}' for k, vv in sorted(_merged.items()))))
        return summary
    L.locked_json_update(spath, _merge)
    _ev_rec = L.locked_json_read(spath).get(key) or {}
    L.log_experiment_event(outdir, {'kind': 'cell', 'tracking_schema': 2, 'run_id': _lp_run_id, 'status': 'ok', 'started_at': _lp_started, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'elapsed_s': time.time() - _lp_t0, 'key': key, 'variant': v, 'seed': int(seed), 'run_cfg': _run_cfg, 'code': CKPT_CODE, 'provenance': _ev_rec.get('provenance'), 'hyperparams': _ev_rec.get('hyperparams'), 'data_fp': _ev_rec.get('data_fp'), 'val_fp': _ev_rec.get('val_fp'), 'rng_fp': _ev_rec.get('rng_fp'), 'train_knobs': _ev_rec.get('train_knobs'), 'metrics': {'params': _ev_rec.get('params'), 'by_len': {str(k): (v2 or {}).get('ppl') for k, v2 in (_ev_rec.get('by_len') or {}).items()}, 'train_time_s': (_tr or {}).get('train_time_s')}, 'losses': _ev_rec.get('losses'), 'ppl_history': _ev_rec.get('ppl_history')})
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
    variants = list(payload['variants'])
    eval_lens = payload['eval_lens']
    seeds = list(payload.get('seeds', [0, 1, 2]))
    steps = payload.get('steps', 3000)
    train_len = payload.get('train_len', 512)
    vocab = payload.get('vocab', 8192)
    _p1l_ratios = {}
    for _v in variants:
        try:
            _p1l_ratios[_v] = (L.variant_mlp_ratio(_v, int(vocab), d=256, n_layers=6, n_heads=8, d_head=32, seq_len=int(train_len), matched=set(PARAM_MATCHED_V7), ref_variant='csa_fixed_rope') if _v in PARAM_MATCHED_V7 else 4.0)
        except Exception:
            _p1l_ratios[_v] = None
    L.overfit_guard(list(variants), vocab=int(vocab), seq_len=L.probe_seq_len(train_len),
                    mlp_ratio=_p1l_ratios,
                    arch={'d': 256, 'n_layers': 6, 'n_heads': 8, 'd_head': 32})
    spath = os.path.join(outdir, 'summary.json')
    _read = L.locked_json_read(spath)
    need = max(eval_lens) + 1
    _t_data = time.time()
    L.set_seed(int((seeds or [0])[0]))
    _, val_ids, _, _, _ = L.load_wikitext(max(train_len, need), 8000000)
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
    _len_panel_cfg = L.preflight_fingerprint({'payload': payload}, stage='lenphase_panel', vocab=int(vocab))
    # Block by seed, randomize the arm order inside each seed, and rotate the
    # starting position across seeds.  The old arm-major order (all seeds of
    # arm A, then all seeds of arm B) made arm and calendar time perfectly
    # confounded; this plan trains both arms of a pair close together and
    # records the design for the zero-GPU audit.
    _all_len_cells = [(v, seed) for v in variants for seed in seeds]
    _run_order_plan = L.build_run_order_plan(
        _all_len_cells, block_key=lambda c: int(c[1]),
        seed=int((seeds or [0])[0]), block_name='seed')
    L.write_run_order(outdir, _run_order_plan)
    _planned_len = list(_run_order_plan['planned_cells'])
    _order_rank = {lab: i for i, lab in enumerate(_planned_len)}
    cells = sorted(_all_len_cells,
                   key=lambda c: _order_rank.get(f'{c[0]}::seed{c[1]}', len(_order_rank)))
    _write_phase_manifest(outdir, {'seeds': seeds, 'variants': list(variants), 'seq_len': train_len},
                          _len_panel_cfg, _panel_prov, phase='lenphase',
                          planned_cells=_planned_len, status='running', started_at=_panel_started)
    _ran_parallel = False
    if len(cells) > 1:
        import par as _par
        _par_ctx = par if isinstance(par, dict) else _par.resolve_ctx(par, label=label or outdir)
        if _par_ctx and _par.ctx_active(_par_ctx, len(cells)):
            module = _par_ctx.get('module') or (_caller if (_caller := sys._getframe(1).f_globals.get('__name__')) and _caller != '__main__' else 'explib.rope')
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
    _len_cells = {}
    for _k in _planned_len:
        _r = summary.get(_k)
        if not isinstance(_r, dict) or not isinstance(_r.get('by_len'), dict):
            continue
        if not any(isinstance(_c, dict) and L.ppl_is_usable(_c.get('ppl')) for _c in _r['by_len'].values()):
            continue
        try:
            _v, _sd = _k.split('::')
            _len_cells[_k] = _phase_cell_record(_v, int(_sd.replace('seed', '')), _r,
                                                run_cfg=_r.get('run_cfg'), metric='by_len', key=_k)
            _len_cells[_k]['eval_lens'] = list(eval_lens)
        except Exception:
            continue
    _missing_len = [k for k in _planned_len if k not in _len_cells]
    _len_status = 'complete' if not _missing_len else 'incomplete'
    _write_phase_manifest(outdir, {'seeds': seeds, 'variants': list(variants), 'seq_len': train_len},
                          _len_panel_cfg, _panel_prov, phase='lenphase',
                          planned_cells=_planned_len, cell_records=_len_cells,
                          status=_len_status, started_at=_panel_started)
    L.log_experiment_event(outdir, {
        'kind': 'panel_finish', 'tracking_schema': 2, 'label': label or outdir,
        'phase': 'lenphase', 'started_at': _panel_started,
        'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'n_cells_planned': len(cells), 'n_cells_measured': len(_len_cells),
        'n_cells_missing': len(_missing_len),
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
            ok = [x for x in recs if isinstance(x.get('truncated'), (bool, np.bool_)) and not bool(x.get('truncated'))]
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
def _run_cli():
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
        print(f"[v7] mode 'phase' needs a phase name, e.g. `python -m explib.rope phase P0E`.  Available: {[p[0] for p in PHASES]}")
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
