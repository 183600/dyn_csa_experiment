
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
            _deadline += booked
    rec = None
    try:
        try:
            rec = train_warmup(v, train_ids, val_batch, vocab, seed=seed, warm_steps=warm, d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, seq_len=cfg['seq_len'], batch_size=cfg['batch_size'], steps=cfg['steps'], lr=cfg['lr'], weight_decay=cfg['weight_decay'], warmup=cfg['warmup'], comp_lambda=cfg['comp_lambda'], delta_lr_mult=cfg.get('delta_lr_mult', 10.0), eval_every=cfg.get('eval_every', 0), eval_subset=cfg.get('eval_subset', 128), val_bnd=val_bnd, mlp_ratio=ratios[v], deadline_ts=_deadline)
            _prov = L.experiment_provenance(cfg, train_ids=train_ids, val_batch=val_batch, vocab=vocab)
            _hp = L.run_hyperparameters(cfg, seed=seed, d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, vocab=vocab, mlp_ratio=ratios[v], amp_dtype=rec.get('amp_dtype', 'auto'), eager_batch=rec.get('eager_batch', 'auto'))
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
    ratios = {}
    for v in variants:
        if v not in ratios:
            ratios[v] = L.variant_mlp_ratio(v, vocab, d=d, n_layers=n_layers, n_heads=n_heads, d_head=d_head, seq_len=cfg['seq_len'], matched=set(PARAM_MATCHED_V7) | {'full_matched', 'full_sw128_matched'}, ref_variant=cfg.get('mlp_match_ref', 'csa_dynamic'))
    # Warmup training uses the real corpus through its own train_warmup loop,
    # so the generic run() pre-flight never sees it.  Check the actual token
    # stream with the panel's real architecture/MLP and a block-length
    # sequence before launching the 20k-step P0W cells.
    _probe_seq = L.probe_seq_len(cfg['seq_len'])
    try:
        L.overfit_guard(list(variants), train_ids=train_ids, vocab=vocab,
                        arch={'d': d, 'n_layers': n_layers, 'n_heads': n_heads, 'd_head': d_head},
                        mlp_ratio=ratios, seq_len=_probe_seq)
    except RuntimeError as _fe:
        _prov0 = L.experiment_provenance(cfg, train_ids=train_ids, val_batch=val_batch, vocab=vocab)
        _reps0 = [dict(r) for k, r in L._OVERFIT_REPORTS.items() if k[0] in set(variants)]
        _pfp = L.preflight_fingerprint(cfg, stage='warmup_overfit', vocab=vocab, data_fp=L._ids_fp(train_ids), probe_cfg={'_probe_seq': _probe_seq, 'arch': {'d': d, 'n_layers': n_layers, 'n_heads': n_heads, 'd_head': d_head}})
        L.atomic_write_json(os.path.join(outdir, 'overfit_sanity.json'), {
            'kind': 'small_batch_overfit_sanity_panel', 'stage': 'warmup_preflight',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'run_cfg': _pfp,
            'label': label or outdir, 'passed': False, 'error': f'{_fe}',
            'provenance': _prov0, 'reports': _reps0,
        })
        raise
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
    if torch.cuda.is_available():
        L._pin_cuda_determinism()
    _fp = _warm_fp(cfg, vocab, train_ids, val_batch)
    _data_contract = L.batch_contract_check(
        train_ids, cfg['seq_len'], cfg['batch_size'], L.DEVICE,
        seed=int(list(seeds)[0] if seeds else 0), vocab=vocab)
    _data_contract.update({'run_cfg': _fp, 'label': label or outdir,
                           'provenance': _panel_prov})
    L.atomic_write_json(os.path.join(outdir, 'data_contract.json'), _data_contract)
    L.log_experiment_event(outdir, {
        'kind': 'preflight', 'tracking_schema': 2,
        'status': ('ok' if _data_contract.get('passed') else 'failed'),
        'stage': 'batch_iterator_contract', 'run_cfg': _fp,
        'code': CKPT_CODE, 'provenance': _panel_prov,
        'data_contract': {k: _data_contract.get(k) for k in
                          ('passed', 'diagnosis', 'seq_len', 'batch_size', 'seed', 'n_batches')},
    })
    if not _data_contract.get('passed'):
        raise RuntimeError(
            'batch iterator contract check FAILED '
            f"({_data_contract.get('diagnosis')}) — the warmup panel's actual "
            'loader is not producing shifted, in-vocabulary, repeatable batches. '
            'See data_contract.json.')
    _planned = [f'{v}::w{warm}::seed{seed}' for seed in seeds for warm in warm_grid for v in variants]
    _write_phase_manifest(outdir, cfg, _fp, _panel_prov, phase='warmup', planned_cells=_planned, status='running', started_at=_panel_started, data_contract=_data_contract)
    cells = [(v, warm, seed) for seed in seeds for warm in warm_grid for v in variants]
    stale_dropped = []
    dropped_truncations = []
    _ran_parallel = False
    if len(cells) > 1:
        import par as _par
        _par_ctx = par if isinstance(par, dict) else _par.resolve_ctx(par, label=label or outdir)
        if _par_ctx and _par.ctx_active(_par_ctx, len(cells)):
            module = _par_ctx.get('module') or (_caller if (_caller := sys._getframe(1).f_globals.get('__name__')) and _caller != '__main__' else 'explib.rope')
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
    _cell_recs = {}
    for _k in _planned:
        _r = summary.get(_k)
        if not isinstance(_r, dict) or not L.ppl_is_usable(_r.get('ppl')) or _r.get('synthesized'):
            continue
        try:
            _v, _w, _sd = _k.split('::')
            _cell_recs[_k] = _phase_cell_record(_v, int(_sd.replace('seed', '')), _r, run_cfg=_fp, metric='ppl', key=_k)
            _cell_recs[_k]['warm_steps'] = int(_w.replace('w', ''))
        except Exception:
            continue
    _missing_warm = [k for k in _planned if k not in _cell_recs]
    _warm_status = 'complete' if not _missing_warm else 'incomplete'
    _write_phase_manifest(outdir, cfg, _fp, _panel_prov, phase='warmup', planned_cells=_planned, cell_records=_cell_recs, status=_warm_status, started_at=_panel_started, data_contract=_data_contract)
    L.log_experiment_event(outdir, {
        'kind': 'panel_finish', 'tracking_schema': 2, 'label': label or outdir,
        'phase': 'warmup', 'started_at': _panel_started,
        'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'n_cells_planned': len(cells), 'n_cells_measured': len(_cell_recs),
        'n_cells_missing': len(_missing_warm),
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
        """效应量与逐种子差值：报告必须给出逐种子 Δ 与
        d_z（配对差均值/样本标准差），并把该 n 下的 p 值分辨率下限
        一并写出，避免把分辨率上限误读为数据结论。纯增量字段。"""
        res['per_seed'] = [float(x) for x in d.tolist()]
        # Zero deltas are sign-neutral: they do not change p, but they lower
        # the attainable resolution from 2/2**n to 2/2**(#nonzero).
        _n_nonzero = int(np.count_nonzero(d))
        res['n_nonzero'] = _n_nonzero
        res['p_floor'] = (float('nan') if n <= 0 else
                          (1.0 if _n_nonzero == 0 else
                           math.ldexp(2.0, -_n_nonzero)))
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
        _m = str(res.get('method', 'exact'))
        if _m.startswith('exact'):
            res['test'] = 'exact two-sided sign-flip over paired per-seed deltas'
        elif _m.startswith('saddlepoint'):
            res['test'] = ('deterministic two-sided sign-flip saddlepoint tail (Lugannani-Rice, '
                           'null variance sum(d_i**2)); exact enumeration/MITM only for n<=40')
        else:
            res['test'] = 'deterministic two-sided sign-flip approximation over paired per-seed deltas'
        return res
    if n == 0:
        return _extras({'n': 0, 'mean': float('nan'), 'std': float('nan'), 'p_exact_signflip': float('nan'), 'n_flips': 0, 'n_dropped': n_dropped}, d, n)
    obs = abs(d.mean())
    if n > 20:
        # exp_lib keeps the test exact via meet-in-the-middle through n=40
        # (and uses a deterministic saddlepoint tail, never a random
        # Monte-Carlo draw, beyond that).  Do not label a sampled p-value as
        # an exact sign-flip p.
        _r = L.exact_signflip_p(d)
        return _extras({'n': n, 'mean': float(d.mean()),
                        'std': float(d.std(ddof=1)) if n > 1 else 0.0,
                        'p_exact_signflip': float(_r['p']),
                        'n_flips': (int(1 << n) if _r.get('method', '').startswith('exact') else 0),
                        'method': _r.get('method'),
                        'n_dropped': n_dropped}, d, n)
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
            _pa, _pb = L._record_primary_ppl(ra), L._record_primary_ppl(rb)
            if _pa is None or _pb is None:
                mismatch_fields.add('primary estimand missing (nested record without holdout PPL/NLL accounting)')
                skipped.append(s)
                continue
            dl.append(float(_pa) - float(_pb))
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
    _factor_rep = L.sweep_contrast_factor_audit(
        ROPE_CONTRAST_FACTORS, factory=make_layer_cfgs_v7, verbose=False)
    attach_contrast_factor_metadata(
        out_d['comparisons'], ROPE_CONTRAST_FACTORS, _factor_rep)
    # Family-wise correction over the report's five pre-listed comparisons
    # This is a guard against multiple-comparison
    # over-reading; the panel remains EXPLORATORY because its comparisons and
    # seeds were chosen after seeing validation results (design_notes §17).
    # A declared-but-unmeasurable comparison keeps a p=1 slot: dropping it
    # from the family would make the remaining p-values look better than the
    # pre-listed family allows (the same rule attach_holm enforces).
    _pairs = []
    _tested = 0
    for k, v in out_d['comparisons'].items():
        pv = v.get('p_exact_signflip') if isinstance(v, dict) and 'omitted' not in v else None
        try:
            pv = float(pv)
        except (TypeError, ValueError, OverflowError):
            pv = float('nan')
        if math.isfinite(pv) and 0.0 <= pv <= 1.0:
            _pairs.append((k, pv))
            _tested += 1
        else:
            _pairs.append((k, 1.0))
    _h = L.holm_adjust(_pairs) if _pairs else {}
    for _k, _v in out_d['comparisons'].items():
        if isinstance(_v, dict):
            _v['p_holm'] = (_h.get(_k) or {}).get('p_holm') if _k in _h else None
    out_d['family'] = {'declared': len(out_d['comparisons']), 'tested': _tested, 'alpha': 0.05,
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
PHASES = [('P0R', 'plain', dict(cfg=dict(L.RUN, outdir='results_lm_v7_rope', variants=ROPE_VARIANTS, steps=1500, matched=set(PARAM_MATCHED_V7), mlp_match_ref='csa_fixed_rope'), seeds=[0, 1, 2])), ('P0W', 'warm', dict(cfg=dict(L.RUN_LONG, outdir='results_lm_v7_warmup', variants=['csa_fixed']), seeds=[0, 1, 2], warm_grid=(0, 5000, 10000))), ('P1L', 'len', dict(outdir='results_len', vocab=8192, steps=3000, train_len=512, eval_lens=[512, 1024, 2048, 4096], n_seq=16, variants=['full', 'csa_fixed', 'full_rope', 'csa_fixed_rope'])), ('P2S', 'plain', dict(cfg=dict(L.RUN_SCALE, outdir='results_lm_v5_scale', variants=['full_matched', 'full_sw128_matched', 'csa_fixed', 'csa_dynamic']), seeds=[2, 3])), ('P1T', 'plain', dict(cfg=dict(L.RUN, outdir='results_lm_v7_seq2k', seq_len=2048, batch_size=3, steps=1500, variants=['csa_fixed_topk8', 'csa_fixed_topk32', 'csa_fixed_topk128', 'csa_fixed_topk512', 'csa_fix_m1']), seeds=[0, 1, 2])), ('P0E', 'plain', dict(cfg=dict(LONG_EXT, variants=['csa_fixed', 'full']), seeds=[0, 1, 2]))]

def run_phase(name, guard, only=None, par=None):
    import par as _par
    _ctx = _par.resolve_ctx(par, label=f'v7 {name}')
    for pname, kind, payload in PHASES:
        if pname != name:
            continue
        if isinstance(_ctx, dict):
            _ctx = dict(_ctx, module='explib.rope')
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
    # generic run() pre-flight does not cover it.  The ACTUAL task batch
    # (same key/value construction, padding and ignore_index=-100 convention)
    # is checked below before each seed's first checkpoint is written.
    _panel_started = time.strftime('%Y-%m-%dT%H:%M:%S')
    _panel_prov = L.experiment_provenance({'dataset': 'synthetic_niah', 'n_train_tokens': 0, 'seq_len': 512, 'data_extra': {'generator': 'build_niah_batch', 'n_pairs': 4, 'ignore_index': -100}})
    L.log_experiment_event(outdir, {
        'kind': 'panel_start', 'tracking_schema': 2, 'label': label or outdir,
        'phase': 'niah', 'started_at': _panel_started,
        'config': L._jsonable(payload), 'seeds': seeds, 'variants': list(variants),
        'seq_lens': list(seq_lens), 'code': CKPT_CODE, 'provenance': _panel_prov,
    })
    _niah_panel_cfg = L.preflight_fingerprint({'payload': payload}, stage='niah_panel', vocab=int(vocab))
    _planned_niah = [f'{v}::seed{seed}' for v in variants for seed in seeds]
    _write_phase_manifest(outdir, {'seeds': seeds, 'variants': list(variants), 'seq_len': 512},
                          _niah_panel_cfg, _panel_prov, phase='niah',
                          planned_cells=_planned_niah, status='running', started_at=_panel_started)
    _recipe_by_v = {}
    _prov_by_seed = {}
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
        _opt_sig = (f'wd0.0_b{L.OPT_ADAMW_BETAS[0]},{L.OPT_ADAMW_BETAS[1]}'
                    f'_eps{L.OPT_ADAMW_EPS:g}_gc{L.GRAD_CLIP_NORM}_cl0.05'
                    f'_tk{_v7_knobs_fp(L.DEVICE, 12)}')
        _recipe = f'niah_v2_steps{n_steps}_v{vocab}_bs12_sl512_lr0.0003_mr{mr:.8g}-{_mr_ref}_{_opt_sig}_reuse4'
        _recipe_by_v[v] = _recipe
        for seed in seeds:
            _niah_batch_seed = int(seed) * 100003
            _g_ids, _g_tgt, _g_dist, _g_kdist = build_niah_batch(12, 512, 4, int(vocab), seed=_niah_batch_seed)
            _niah_batch = (_g_ids, _g_tgt)
            _niah_batch_fp = L._ids_fp(_g_ids.reshape(-1)) + ':' + L._ids_fp(_g_tgt.reshape(-1))
            _niah_data_recipe = {
                'generator': 'build_niah_batch', 'n_seq': 12, 'seq_len': 512,
                'n_pairs': 4, 'vocab': int(vocab), 'seed': _niah_batch_seed,
                'ignore_index': -100, 'n_scored_targets': int((_g_tgt >= 0).sum()),
                'batch_fp': _niah_batch_fp,
            }
            _niah_prov = L.experiment_provenance({
                'dataset': 'synthetic_niah', 'n_train_tokens': 0, 'seq_len': 512,
                'data_extra': _niah_data_recipe,
            }, vocab=int(vocab))
            _niah_knobs = _v7_knobs(L.DEVICE, 12)
            _prov_by_seed[int(seed)] = _niah_prov
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
                # Small-batch overfit on the EXACT task batch (masked -100
                # labels, query/answer positions), not a random next-token
                # stream: a wrong NIAH label map must be caught here.
                try:
                    L.overfit_guard([v], batch=_niah_batch, ignore_index=-100,
                                    vocab=int(vocab), steps=300, seq_len=512,
                                    batch_size=12, mlp_ratio=mr,
                                    arch={'d': 256, 'n_layers': 6, 'n_heads': 8, 'd_head': 32})
                except RuntimeError as _fe:
                    _pfp = L.preflight_fingerprint({'payload': payload}, stage='niah_overfit', vocab=int(vocab), data_fp=_niah_batch_fp, probe_cfg=_niah_data_recipe)
                    L.atomic_write_json(os.path.join(outdir, 'overfit_sanity.json'), {
                        'kind': 'small_batch_overfit_sanity_panel', 'stage': 'niah_preflight',
                        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'run_cfg': _pfp,
                        'passed': False, 'error': f'{_fe}', 'provenance': _niah_prov,
                        'reports': [dict(r) for k, r in L._OVERFIT_REPORTS.items() if k[0] == v],
                    })
                    L.log_experiment_event(outdir, {'kind': 'preflight', 'tracking_schema': 2, 'status': 'failed', 'stage': 'niah_overfit', 'run_cfg': _pfp, 'code': CKPT_CODE, 'provenance': _niah_prov, 'variant': v, 'seed': int(seed), 'error': f'{_fe}', 'sanity': {'passed': False}})
                    raise
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
                opt = torch.optim.AdamW(model.parameters(), lr=0.0003, betas=L.OPT_ADAMW_BETAS, eps=L.OPT_ADAMW_EPS, weight_decay=0.0, foreach=True)
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
                _atomic_torch_save({'cfg': cfgs, 'mlp_ratio': mr, 'vocab': vocab, 'd': 256, 'n_layers': 6, 'n_heads': 8, 'd_head': 32, 'code': CKPT_CODE, 'recipe': _recipe, 'sd': model.state_dict(), 'params': _n_params, 'rng_fp': _niah_rng_fp, 'losses': _niah_losses, 'steps': n_steps, 'provenance': _niah_prov, 'data_fp': _niah_batch_fp, 'data_version': (_niah_prov.get('data') or {}).get('data_version'), 'train_knobs': _niah_knobs, 'data_recipe': _niah_data_recipe}, ck)
                if guard is not None:
                    guard.record_run(time.time() - t0, n_steps, 256, 6, 512, 12)
                _niah_hp = {
                    'steps': int(n_steps), 'seq_len': 512, 'batch_size': 12,
                    'lr': 0.0003, 'weight_decay': 0.0,
                    'betas': list(L.OPT_ADAMW_BETAS), 'eps': L.OPT_ADAMW_EPS,
                    'grad_clip_norm': L.GRAD_CLIP_NORM, 'comp_lambda': 0.05,
                    'mlp_ratio': float(mr), 'optimizer': 'AdamW',
                    'train_knobs': _niah_knobs,
                    'data_recipe': _niah_data_recipe,
                }
                L.log_experiment_event(outdir, {
                    'kind': 'cell', 'tracking_schema': 2, 'status': 'ok',
                    'started_at': time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(t0)),
                    'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    'elapsed_s': time.time() - t0, 'key': f'{v}::seed{seed}::niah_train',
                    'variant': v, 'seed': int(seed), 'run_cfg': _recipe,
                    'code': CKPT_CODE, 'rng_fp': _niah_rng_fp,
                    'provenance': _niah_prov,
                    'data_fp': _niah_batch_fp,
                    'data_version': (_niah_prov.get('data') or {}).get('data_version'),
                    'train_knobs': _niah_knobs, 'data_recipe': _niah_data_recipe,
                    'hyperparams': _niah_hp,
                    'metrics': {'params': _n_params, 'steps': int(n_steps),
                                'final_loss': _niah_losses[-1] if _niah_losses else None,
                                'train_time_s': time.time() - t0},
                    'curve_summary': {'n_points': len(_niah_losses),
                                      'first': _niah_losses[0] if _niah_losses else None,
                                      'last': _niah_losses[-1] if _niah_losses else None,
                                      'best': min(_niah_losses) if _niah_losses else None},
                    'losses': _niah_losses,
                })
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
                _niah_eval_prov = _ckp.get('provenance') or _niah_prov
                _niah_eval_knobs = _ckp.get('train_knobs') or _niah_knobs
                r.update({'variant': v, 'seed': seed, 'seq_len': Ln,
                          'params': _ckp['params'], 'probe_s': time.time() - t0,
                          '_code': CKPT_CODE, 'run_cfg': _recipe,
                          'provenance': _niah_eval_prov,
                          'data_fp': _ckp.get('data_fp') or _niah_batch_fp,
                          'data_version': _ckp.get('data_version') or (_niah_eval_prov.get('data') or {}).get('data_version'),
                          'train_knobs': _niah_eval_knobs,
                          'data_recipe': _ckp.get('data_recipe') or _niah_data_recipe})
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
                L.log_experiment_event(outdir, {'kind': 'eval', 'tracking_schema': 2, 'status': 'ok', 'key': key, 'variant': v, 'seed': int(seed), 'seq_len': int(Ln), 'code': CKPT_CODE, 'recipe': _ckp.get('recipe'), 'run_cfg': _recipe, 'provenance': _niah_eval_prov, 'data_fp': r.get('data_fp'), 'data_version': r.get('data_version'), 'train_knobs': _niah_eval_knobs, 'metrics': {'acc': r.get('acc'), 'n': r.get('n'), 'eval_span': r.get('eval_span'), 'params': r.get('params'), 'probe_s': r.get('probe_s')}})
            del model, _ckp
            gc.collect()
            torch.cuda.empty_cache()
    _measured = len({(r.get('variant'), r.get('seed')) for r in summary.values()
                     if isinstance(r, dict) and isinstance(r.get('acc'), (int, float))
                     and not isinstance(r.get('acc'), bool) and math.isfinite(float(r['acc']))})
    _niah_cells = {}
    for _v in variants:
        for _sd in seeds:
            _evs = [summary.get(f'{_v}::seed{_sd}::len{Ln}') for Ln in seq_lens]
            _evs = [x for x in _evs if isinstance(x, dict) and isinstance(x.get('acc'), (int, float))
                    and not isinstance(x.get('acc'), bool) and math.isfinite(float(x['acc']))]
            if not _evs:
                continue
            _base = dict(_evs[0])
            _accs = [float(x['acc']) for x in _evs]
            _base['acc'] = float(np.mean(_accs))
            _base['run_cfg'] = _recipe_by_v.get(_v)
            _ck = _phase_cell_record(_v, int(_sd), _base, run_cfg=_recipe_by_v.get(_v), metric='acc', key=f'{_v}::seed{_sd}')
            _ck['by_len_acc'] = {str(int(x.get('seq_len'))): x.get('acc') for x in _evs}
            _ck['curve_ref'] = f'experiments.jsonl#{_v}::seed{_sd}::niah_train.losses'
            _niah_cells[f'{_v}::seed{_sd}'] = _ck
    _write_phase_manifest(outdir, {'seeds': seeds, 'variants': list(variants), 'seq_len': 512},
                          _niah_panel_cfg, _panel_prov, phase='niah',
                          planned_cells=_planned_niah, cell_records=_niah_cells,
                          status='complete', started_at=_panel_started)
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
def eval_length_gen(model, val_ids, eval_lens, device=DEVICE, n_seq=16, max_pos=None,
                    doc_ids=None, article_ids=None):
    """By-length evaluation with optional document/article cluster mappings.

    ``doc_ids`` is an optional per-token array with the SAME shape as
    ``val_ids``.  Fixed evaluation rows are arbitrary windows: a document can
    span two rows and a row can merge two documents, so ``ppl_per_seq`` is a
    per-ROW statistic.  When the document mapping is supplied, each cell also
    carries per-DOCUMENT NLL/token counts/PPL, which is the unit a document
    bootstrap must resample.  The pooled ``ppl`` and ``ppl_per_seq`` values are
    unchanged (the new fields are a separate, deterministic aggregation).

    ``article_ids`` is an optional per-token ARTICLE mapping (same shape).
    WikiText's "document" sidecar is really one raw LINE; a line-level
    bootstrap resamples several paragraphs of the same article and is
    pseudoreplicated.  When supplied, each cell carries additional
    per-ARTICLE NLL/token counts/PPL alongside — never instead of — the
    line-level fields, so the frozen document contract is untouched and the
    article-level interval is additive evidence.
    """
    was = model.training
    model.eval()
    out = {}
    # Validate the requested geometry before any model work.  A fractional
    # length used to be truncated by int() and a non-positive max_pos turned
    # the last-token slice into a different (or even full) prefix, so the cell
    # silently measured something other than the declared length.
    try:
        _raw_lens = list(eval_lens)
    except TypeError:
        raise ValueError(
            f'eval_length_gen: eval_lens must be an iterable of lengths, got '
            f'{eval_lens!r}') from None
    eval_lens = []
    for _raw_len in _raw_lens:
        _n = L._exact_count(_raw_len, minimum=2)
        if _n is None:
            raise ValueError(
                f'eval_length_gen: every eval length must be an integer >= 2 '
                f'(one next-token target is the minimum), got {_raw_len!r}')
        eval_lens.append(int(_n))
    _n_seq = L._exact_count(n_seq, minimum=1)
    if _n_seq is None:
        raise ValueError(
            f'eval_length_gen: n_seq must be a positive integer, got {n_seq!r}')
    n_seq = int(_n_seq)
    _varr = np.asarray(val_ids)
    if _varr.ndim != 2:
        raise ValueError(
            f'eval_length_gen: val_ids must be a 2-D [n_seq, tokens] panel, '
            f'got shape {_varr.shape}')
    _doc2d = None
    if doc_ids is not None:
        _darr = np.asarray(doc_ids)
        if tuple(_darr.shape) != tuple(_varr.shape):
            raise ValueError(
                f'eval_length_gen: doc_ids must have the same shape as '
                f'val_ids {tuple(_varr.shape)}, got {tuple(_darr.shape)}')
        if _darr.dtype.kind not in 'iu':
            raise ValueError(
                'eval_length_gen: doc_ids must be integer document ids, got '
                f'dtype {_darr.dtype}')
        if _darr.size and int(_darr.min()) < 0:
            raise ValueError('eval_length_gen: doc_ids must be non-negative')
        _doc2d = _darr
    _art2d = None
    if article_ids is not None:
        _aarr = np.asarray(article_ids)
        if tuple(_aarr.shape) != tuple(_varr.shape):
            raise ValueError(
                f'eval_length_gen: article_ids must have the same shape as '
                f'val_ids {tuple(_varr.shape)}, got {tuple(_aarr.shape)}')
        if _aarr.dtype.kind not in 'iu':
            raise ValueError(
                'eval_length_gen: article_ids must be integer article ids, '
                f'got dtype {_aarr.dtype}')
        if _aarr.size and int(_aarr.min()) < 0:
            raise ValueError('eval_length_gen: article_ids must be non-negative')
        _art2d = _aarr
    if max_pos is not None:
        _mp = L._exact_count(max_pos, minimum=2)
        if _mp is None:
            raise ValueError(
                f'eval_length_gen: max_pos must be an integer >= 2 or None, '
                f'got {max_pos!r}')
        max_pos = int(_mp)
    # An absolute-position model cannot score beyond its learned table.  When
    # the caller passes max_pos=None (the "full context" request) a longer
    # length must be an explicit error, never a silent last-max_pos slice.
    _abs_pe = bool(getattr(model, 'use_abs_pe', True))
    _hard_max = int(getattr(model, 'max_seq', 1 << 62)) if _abs_pe else None
    # `max_pos` may only shorten the span; it can never exceed the absolute
    # position table.  Clamping here keeps every downstream slice inside the
    # table for both short and long lengths.
    _eff_max = max_pos
    if _abs_pe and _hard_max is not None:
        _eff_max = _hard_max if max_pos is None else min(int(max_pos), _hard_max)
    _over = [int(Ln) for Ln in eval_lens
             if max_pos is None and _hard_max is not None and int(Ln) > _hard_max]
    for Ln in _over:
        out[Ln] = {'error': (f'absolute position table covers {_hard_max} tokens; '
                             f'cannot evaluate L={Ln} with max_pos=None — use a RoPE arm '
                             'or pass max_pos to request the explicit truncated span'),
                   'truncated': False, 'eval_span': None}
    _within = [int(Ln) for Ln in eval_lens
               if (_eff_max is None or int(Ln) <= _eff_max) and int(Ln) not in _over]
    _trunc_lens = [int(Ln) for Ln in eval_lens
                   if _eff_max is not None and int(Ln) > _eff_max and int(Ln) not in _over]
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
        # A requested length is only measurable if the cached panel actually
        # holds that many tokens.  The old code sliced `val_ids` to the
        # available width and then allocated `nll_cols` for `_Lmax` targets,
        # so a RoPE arm evaluated at a longer length than the cache crashed
        # with a target/logit width mismatch (or silently scored a shorter
        # span on some backends).  Report the unavailable cells explicitly.
        _val_width_now = int(np.asarray(val_ids).shape[1]) if np.asarray(val_ids).ndim == 2 else 0
        # A truncated length also needs the full `Ln`-prefix, not merely
        # `_eff_max` tokens: the intended "last _eff_max tokens of the Ln
        # prefix" does not exist when the panel itself is shorter than Ln.
        # Checking only `_within` let two over-long lengths collapse onto the
        # same clamped (start,end) slice and report a different causal context
        # under the requested L labels.
        _too_short = [int(Ln) for Ln in (_within + _trunc_lens)
                      if int(Ln) > _val_width_now]
        for Ln in _too_short:
            out[Ln] = {'error': (f'val_ids has {_val_width_now} tokens; cannot '
                                 f'evaluate L={Ln} without a shorter, undeclared span'),
                       'truncated': False, 'eval_span': None}
        if _too_short:
            _too_short_set = set(_too_short)
            _within = [int(Ln) for Ln in _within
                       if int(Ln) not in _too_short_set]
            _trunc_lens = [int(Ln) for Ln in _trunc_lens
                           if int(Ln) not in _too_short_set]
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
            _doc_targets = None
            _doc_map = None
            if _doc2d is not None:
                _doc_targets = _doc2d[:int(ids.shape[0]), 1:_Lmax]
                _doc_map = L.document_index(_doc_targets)
            _art_targets = None
            _art_map = None
            if _art2d is not None:
                _art_targets = _art2d[:int(ids.shape[0]), 1:_Lmax]
                _art_map = L.document_index(_art_targets)
            for Ln in _within:
                _t = float(_cum[:, Ln - 2].sum()) if Ln >= 2 else 0.0
                _ntok = int(ids.shape[0]) * (Ln - 1)
                ppl = L._safe_exp_ppl(_t / max(_ntok, 1))
                out[Ln] = {'ppl': float(ppl), 'n_tok': _ntok, 'n_tokens': _ntok, 'truncated': False, 'eval_span': (Ln - 1),
                           'nll_sum': float(_t),
                           'nll_per_seq': [float(v) for v in _cum[:, Ln - 2].tolist()],
                           'n_seq': int(ids.shape[0])}
                # per-document rows: the pooled ppl above is
                # UNCHANGED (same reduction); these extra rows only make the
                # document-sampling uncertainty visible and stratifiable
                # (doc-level deltas vs seed-level variance).  Rows are the
                # panel's fixed eval rows, so the list order is stable.
                if Ln >= 2:
                    out[Ln]['ppl_per_seq'] = [float(L._safe_exp_ppl(float(v) / (Ln - 1))) for v in _cum[:, Ln - 2].tolist()]
                    out[Ln]['n_seq'] = int(ids.shape[0])
                if _doc_targets is not None:
                    # Same predicted tokens, grouped by their real document
                    # identity instead of by the arbitrary row window.
                    out[Ln].update(L.aggregate_nll_by_document(
                        nll_cols[:, :Ln - 1], _doc_targets[:, :Ln - 1],
                        document_index_map=(_doc_map[0],
                                            _doc_map[1][:, :Ln - 1])))
                else:
                    out[Ln]['doc_granularity'] = 'row'
                if _art_targets is not None:
                    # WikiText rows are lines; only the article is an
                    # independent sampling unit for the evaluation corpus.
                    # Additive: line-level fields above stay unchanged.
                    out[Ln].update(L.aggregate_nll_by_article(
                        nll_cols[:, :Ln - 1], _art_targets[:, :Ln - 1],
                        document_index_map=(_art_map[0],
                                            _art_map[1][:, :Ln - 1])))
            del x, nll_cols, _cum
        _trunc_cache = {}
        _val_width = int(np.asarray(val_ids).shape[1])
        for Ln in _trunc_lens:
            Ln = int(Ln)
            trunc = _eff_max is not None and Ln > _eff_max
            # Identify the actual token slice, not just its length: for
            # Ln > _eff_max the evaluation uses the LAST _eff_max tokens of the
            # Ln-prefix, so two lengths can share a span but different rows.
            # Only genuinely identical (start, end) slices are reused.
            _end = min(Ln, _val_width)
            _start = max(0, _end - int(_eff_max)) if trunc else 0
            _key = (_start, _end)
            _cached = _trunc_cache.get(_key)
            if _cached is not None:
                if 'error' in _cached:
                    out[Ln] = {'error': _cached['error']}
                else:
                    # Reuse the WHOLE cell: the per-document accounting and
                    # n_seq describe the same (start, end) slice, so dropping
                    # them here would make one length look measured but
                    # unverifiable simply because another length hit the cache
                    # first.
                    out[Ln] = dict(_cached)
                    out[Ln]['truncated'] = bool(trunc)
                continue
            ids = np.asarray(val_ids[:n_seq, :Ln], dtype=np.int64)
            if trunc:
                ids = ids[:, -int(_eff_max):]
            try:
                x = L._h2d_async(ids, torch.int64, device)
                nll, ntok = (torch.zeros((), dtype=torch.float64, device=device), 0)
                _row_nll = []
                _tok_nll = []
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
                    _row_nll.append(_nll_rows.sum(dim=1, dtype=torch.float64))
                    _tok_nll.append(_nll_rows.detach())
                    del _h, _nll_rows
                    _r += _chunk
                ppl = L._safe_exp_ppl(float(nll) / max(ntok, 1))
                # `eval_span` is the number of SCORED next-token targets per
                # row (input span - 1), matching n_tok and the by-length
                # audit.  It is not the input width: that off-by-one made the
                # cell metadata disagree with its own token denominator.
                _ev_span = max(0, int(x.shape[1]) - 1)
                _seq_nll = ([float(v) for v in torch.cat(_row_nll).tolist()]
                            if _row_nll else [])
                _seq_ppl = ([float(L._safe_exp_ppl(v / _ev_span)) for v in _seq_nll]
                            if _ev_span > 0 else [])
                out[Ln] = {'ppl': float(ppl), 'n_tok': ntok, 'truncated': trunc,
                           'eval_span': _ev_span, 'nll_sum': float(nll),
                           'nll_per_seq': _seq_nll, 'ppl_per_seq': _seq_ppl,
                           'n_seq': int(x.shape[0])}
                if _doc2d is not None:
                    # The slice is [start:end] of the original panel; targets
                    # are its positions 1..end-start-1.
                    _doc_targets = _doc2d[:int(x.shape[0]), _start + 1:_end]
                    _nll_all = torch.cat(_tok_nll).reshape(
                        int(x.shape[0]), _ev_span)
                    out[Ln].update(L.aggregate_nll_by_document(
                        _nll_all, _doc_targets))
                else:
                    out[Ln]['doc_granularity'] = 'row'
                if _art2d is not None:
                    _art_targets = _art2d[:int(x.shape[0]), _start + 1:_end]
                    _nll_all = torch.cat(_tok_nll).reshape(
                        int(x.shape[0]), _ev_span)
                    out[Ln].update(L.aggregate_nll_by_article(
                        _nll_all, _art_targets))
                _trunc_cache[_key] = dict(out[Ln])
                del x, _row_nll, _tok_nll
            except Exception as e:
                _err = f'{type(e).__name__}: {e}'
                out[Ln] = {'error': _err}
                _trunc_cache[_key] = {'error': _err}
    finally:
        for _a, _prev in zip(_reg_layers, _reg_prior):
            _a.need_reg = _prev
        model.train(was)
    return out
