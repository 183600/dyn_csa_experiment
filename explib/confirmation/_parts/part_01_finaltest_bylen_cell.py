

@torch.inference_mode()
def _finaltest_bylen_cell(cfg, manifest, v, seed, test_ids, split='test',
                          train_fp=None, frozen_vocab=None, env=None,
                          test_doc_ids=None, test_article_ids=None,
                          article_diag=None):
    """One (arm, seed) clean by-length evaluation on the TEST split.

    记录现在额外携带 manifest_sha256 与
    ckpt_sha256（绑定「哪份冻结计划、哪份权重」）；recipe 必须通过冻结
    清单的结构化配方签名（不再是自然语言说明）；resume 的 stale 判定
    把 manifest 哈希纳入——清单修订后，同一 outdir 的旧结果自动作废，
    不会被静默复用。评估记录同时带执行环境指纹（`exec_fp`）与紧凑的
    environment 快照：代码戳/配置/种子之外，机器与精度栈也被落盘，
    而不是只在训练记录里有。"""
    outdir = cfg['outdir']
    os.makedirs(outdir, exist_ok=True)
    spath = cfg['bylen_summary']
    _msha = manifest.get('manifest_sha256')
    _sig = _recipe_signature(manifest, v)
    out = {'status': 'ok', 'key': f'{v}::s{seed}', 'variant': v, 'seed': seed}
    _ft_run_id = L.new_experiment_id('p5t')
    _ft_started = time.strftime('%Y-%m-%dT%H:%M:%S')
    _ft_t0 = time.time()
    _ft_env = env if isinstance(env, dict) else L.environment_snapshot()
    _ft_exec_fp = L.execution_fingerprint(_ft_env)
    ck = os.path.join(cfg['ckpt_dir'], f'{v}_seed{seed}.pt')
    if not os.path.exists(ck):
        print(f'[p5t] SKIP {v} s{seed}: no checkpoint {ck} (run P3MT/P4MT/P5C first)')
        out['status'] = 'skipped_no_ckpt'
        return out
    try:
        d = V._load_ckpt(ck)
    except Exception as _cle:
        print(f'[p5t] SKIP {v} s{seed}: checkpoint unreadable ({type(_cle).__name__}: {_cle})')
        out['status'] = 'skipped_bad_ckpt'
        return out
    _ck_identity = _checkpoint_identity_problem(d, v, seed)
    if _ck_identity:
        print(f'[p5t] REFUSE {v} s{seed}: {_ck_identity} — a checkpoint from '
              'another arm/seed (or with different weights) cannot enter the '
              'confirmation panel')
        del d
        out['status'] = 'skipped_ckpt_identity'
        return out
    if d.get('code') != V.CKPT_CODE:
        print(f'[p5t] REFUSE {v} s{seed}: checkpoint predates the current code stamp (code={d.get("code")!r})')
        del d
        out['status'] = 'skipped_stale_ckpt'
        return out
    mech_summ = {}
    if os.path.exists(cfg['mech_summary']):
        mech_summ = json.load(open(cfg['mech_summary'], encoding='utf-8'))
    _tr = mech_summ.get(f'{v}::seed{seed}') or {}
    if not _tr or not L.result_is_current(_tr, V.CKPT_CODE, 'by_len') or _tr.get('synthesized'):
        print(f'[p5t] REFUSE {v} s{seed}: no current training record backs the checkpoint — the weights have no traceable provenance')
        del d
        out['status'] = 'skipped_no_provenance'
        return out
    # The summary key is not evidence: the record itself must name the same
    # arm/seed, otherwise a mis-keyed/swap-edited summary.json can pair the
    # checkpoint with another run's metadata.
    if _tr.get('variant') != v or _record_int(_tr.get('seed')) != int(seed):
        print(f'[p5t] REFUSE {v} s{seed}: training record identity is '
              f'variant={_tr.get("variant")!r}, seed={_tr.get("seed")!r} — '
              'the record does not describe the requested cell')
        del d
        out['status'] = 'skipped_record_identity'
        return out
    _tr_sd_sha = _tr.get('state_dict_sha256')
    if not _valid_sha256(_tr_sd_sha):
        print(f'[p5t] REFUSE {v} s{seed}: training record carries no valid '
              'state_dict_sha256; the record cannot be tied to the evaluated '
              'weights (regenerate the cell with the current writer)')
        del d
        out['status'] = 'skipped_record_digest'
        return out
    if _tr_sd_sha != d.get('state_dict_sha256'):
        print(f'[p5t] REFUSE {v} s{seed}: training-record weight digest '
              f'{str(_tr_sd_sha)[:12]}… != checkpoint weight digest '
              f'{str(d.get("state_dict_sha256"))[:12]}… — the record and the '
              'weights describe different runs')
        del d
        out['status'] = 'skipped_ckpt_digest'
        return out
    _health_refusal = _training_health_gate(cfg, v, seed, d, _tr)
    if _health_refusal:
        print(f'[p5t] REFUSE {_health_refusal}')
        out['status'] = 'skipped_unhealthy_training'
        out['health'] = (d.get('curve_health') or _tr.get('curve_health'))
        del d
        return out
    _ck_recipe = d.get('recipe')
    # structured recipe gate: the frozen plan's training
    # recipe is machine-checkable, not prose — a checkpoint trained under any
    # other protocol is REFUSED before it can enter the confirmation panel.
    if _sig is not None and not _recipe_matches_signature(_ck_recipe, _sig):
        print(f'[p5t] REFUSE {v} s{seed}: the checkpoint recipe does not match the frozen training_recipe signature ({_ck_recipe!r} vs {dict(_sig)!r}) — a different training protocol cannot enter this confirmation')
        del d
        out['status'] = 'skipped_recipe_mismatch'
        return out
    # Training-corpus binding: a recipe proves the hyper-parameters, not the
    # bytes the weights saw.  The checkpoint and its training record must name
    # the same non-null training fingerprint, and that fingerprint must match
    # the frozen canonical value when the manifest supplies one.
    _tr_train_fp = _tr.get('data_fp')
    _ck_train_fp = d.get('train_fp') or d.get('data_fp')
    # The caller passes the value derived BEFORE the first test read
    # and written into the freeze lock.  Recomputing it here from a null
    # manifest field would reintroduce the fail-open path and could accept a
    # record from a different corpus.
    _want_train_fp = train_fp if train_fp is not None else _frozen_train_fp(manifest)
    # Manifests that DECLARE binding.training_data_fingerprint require the
    # binding unconditionally, even while canonical_train_fp is still null.
    # Legacy manifests/records without either field stay admissible.
    _require_train_fp = _train_fp_required(manifest) or bool(_want_train_fp) or bool(_ck_train_fp) or bool(_tr_train_fp)
    if _require_train_fp and (not isinstance(_tr_train_fp, str) or not _tr_train_fp):
        print(f'[p5t] REFUSE {v} s{seed}: training record has no train_fp — the weights cannot be attributed to a specific training corpus')
        del d
        out['status'] = 'skipped_missing_train_fp'
        return out
    if _require_train_fp and (not isinstance(_ck_train_fp, str) or not _ck_train_fp):
        print(f'[p5t] REFUSE {v} s{seed}: checkpoint has no train_fp — regenerate it with the current writer so the training corpus is traceable')
        del d
        out['status'] = 'skipped_missing_train_fp'
        return out
    if _ck_train_fp and _tr_train_fp and _ck_train_fp != _tr_train_fp:
        print(f'[p5t] REFUSE {v} s{seed}: checkpoint train_fp {_ck_train_fp} != training-record train_fp {_tr_train_fp} — the weights and record describe different corpora')
        del d
        out['status'] = 'skipped_train_fp_mismatch'
        return out
    if _want_train_fp is not None and _tr_train_fp != _want_train_fp:
        print(f'[p5t] REFUSE {v} s{seed}: train_fp {_tr_train_fp} does not match the FROZEN canonical training fingerprint {_want_train_fp}')
        del d
        out['status'] = 'skipped_train_fp_mismatch'
        return out
    eval_lens = [int(x) for x in (manifest.get('eval') or {}).get('by_length_ppl', cfg['eval_lens'])]
    _dfp = L._ids_fp(np.ascontiguousarray(test_ids).reshape(-1))
    # The matched-initialization digest is the evidence that "same seed"
    # really means "same shared trunk".  Resolve it once (training summary
    # first, checkpoint metadata as a fallback) and require a resumed record
    # to carry the SAME value: a record whose provenance predates the field
    # is re-evaluated rather than silently reused as "unverified".
    _ck_init = d.get('init_core_fp')
    _tr_init = _tr.get('init_core_fp')
    if _ck_init and _tr_init and str(_ck_init) != str(_tr_init):
        print(f'[p5t] REFUSE {v} s{seed}: checkpoint init_core_fp {str(_ck_init)[:12]}… != training-record init_core_fp {str(_tr_init)[:12]}… — the weights and the training record describe different shared-trunk initializations')
        del d
        out['status'] = 'skipped_init_mismatch'
        return out
    _expect_init = _tr_init or _ck_init
    if not _expect_init:
        print(f'[p5t] WARNING: {v} s{seed}: no init_core_fp in the training record or checkpoint — the pair will be reported as initialization-UNVERIFIED and cannot be certified as matched')
    _cur = L.locked_json_read(spath).get(f'{v}::s{seed}')
    _ck_sha = None
    try:
        _ck_sig = _file_sig(ck)
    except Exception:
        _ck_sig = None

    def _ensure_ck_sha():
        nonlocal _ck_sha
        if _ck_sha is None:
            try:
                _ck_sha = _file_sha256(ck)
            except Exception as _he:
                print(f'[p5t] NOTE {v} s{seed}: could not hash the checkpoint file ({type(_he).__name__}: {_he}) — the record will carry ckpt_sha256=null (weaker binding)')
        return _ck_sha

    def _stale(rec):
        if not isinstance(rec, dict):
            return True
        # All metadata identity checks first: a stale record must not require
        # a full-file SHA-256 when it is going to be re-evaluated anyway.
        if rec.get('_code') != V.CKPT_CODE or rec.get('recipe') != _ck_recipe:
            return True
        if rec.get('data_fp') != _dfp:
            return True
        if rec.get('split') != split:
            return True
        if rec.get('train_fp') != _tr_train_fp:
            return True
        # A record written before the matched-initialization field existed is
        # not a reusable confirmation measurement when the training/checkpoint
        # provenance DOES carry the digest: re-evaluate it so the pair can be
        # verified instead of resuming on unverified metadata.
        if _expect_init and rec.get('init_core_fp') != _expect_init:
            return True
        # Same weights/data are not enough if the hardware/software/determinism
        # class changed: this record was measured on a different execution
        # environment.  Legacy records without exec_fp keep the old permissive
        # path; a present digest must match the current one.
        if rec.get('exec_fp') not in (None, _ft_exec_fp):
            return True
        # A record measured on a different row count is a different estimand.
        # n_seq is stored explicitly and a mismatch is treated as stale.  An
        # exact integer is required: int(16.5) == 16 would silently resume on
        # a different row count, and int(True) would accept a boolean flag.
        if L._exact_count(rec.get('n_seq'), minimum=0) != int(test_ids.shape[0]):
            return True
        if _msha is not None and rec.get('manifest_sha256') != _msha:
            return True  # the manifest changed after this record was written
        for Ln in eval_lens:
            _cells = L.by_len_cells([rec], Ln)
            # A usable PPL is not enough: a truncated cell is a different
            # endpoint and admission will reject it, so resuming on it would
            # skip the re-evaluation forever.
            if not _cells or not all(isinstance(c, dict) and c.get('truncated') is False
                                     for c in _cells):
                return True
        # The record must belong to the SAME weights now on disk.  A metadata
        # (size, mtime_ns, ctime_ns, inode) signature recorded at write time
        # settles the unchanged case without reading the checkpoint; only a
        # missing/different signature needs the content hash.
        if _ck_sig is not None and rec.get('ckpt_sig') == list(_ck_sig):
            return False
        _sha = _ensure_ck_sha()
        if _sha is not None:
            return rec.get('ckpt_sha256') != _sha
        # Neither the metadata signature nor a content hash is available, so
        # the record cannot be shown to belong to the weights on disk now.  Fail
        # closed (re-evaluate) rather than resuming on an unverified file.
        return True
    if not _stale(_cur):
        del d
        out['status'] = 'skipped_resume'
        return out
    _arch_missing = [k for k in ('d', 'n_layers', 'n_heads', 'd_head')
                     if d.get(k) is None]
    if _arch_missing:
        print(f'[p5t] REFUSE {v} s{seed}: checkpoint omits model geometry '
              f'{_arch_missing} — refusing to substitute a current-scale default')
        del d
        out['status'] = 'skipped_missing_geometry'
        return out
    _arch = (int(d['d']), int(d['n_layers']), int(d['n_heads']), int(d['d_head']))
    _ck_vocab_raw = d.get('vocab', frozen_vocab)
    if _ck_vocab_raw is None:
        print(f'[p5t] REFUSE {v} s{seed}: checkpoint and frozen plan declare no vocabulary width — the embedding shape cannot be reconstructed without a current-scale guess')
        del d
        out['status'] = 'skipped_missing_vocab'
        return out
    if isinstance(_ck_vocab_raw, (bool, np.bool_)):
        print(f'[p5t] REFUSE {v} s{seed}: checkpoint vocabulary width '
              f'{_ck_vocab_raw!r} is a boolean, not a positive integer')
        del d
        out['status'] = 'skipped_missing_vocab'
        return out
    try:
        _ck_vocab = int(_ck_vocab_raw)
    except (TypeError, ValueError, OverflowError):
        print(f'[p5t] REFUSE {v} s{seed}: checkpoint vocabulary width '
              f'{_ck_vocab_raw!r} is not an integer')
        del d
        out['status'] = 'skipped_missing_vocab'
        return out
    if _ck_vocab < 1 or (
            isinstance(_ck_vocab_raw, numbers.Real)
            and not isinstance(_ck_vocab_raw, numbers.Integral)
            and float(_ck_vocab_raw) != _ck_vocab):
        print(f'[p5t] REFUSE {v} s{seed}: checkpoint vocabulary width '
              f'{_ck_vocab_raw!r} is not a positive integer')
        del d
        out['status'] = 'skipped_missing_vocab'
        return out
    # The checkpoint must be the one the frozen recipe/test tokenizer names.
    # Using the checkpoint's width when it disagrees with the frozen plan
    # would evaluate that arm against a different id space (and could index
    # past the embedding) after the one-shot test access was already spent.
    if frozen_vocab is not None and _ck_vocab != int(frozen_vocab):
        print(f'[p5t] REFUSE {v} s{seed}: checkpoint vocabulary width '
              f'{_ck_vocab} != the frozen plan/test tokenizer width '
              f'{int(frozen_vocab)} — this arm cannot be evaluated on the '
              'frozen held-out slice')
        del d
        out['status'] = 'skipped_vocab_mismatch'
        return out
    _ck_train_len = d.get('train_len')
    if _ck_train_len is None:
        _bind_m = manifest.get('binding') or {}
        _sig_m = (_bind_m.get('training_recipe_signature')
                  if isinstance(_bind_m, dict) else None)
        _ck_train_len = (int(_sig_m.get('sl'))
                         if isinstance(_sig_m, dict) and _sig_m.get('sl') is not None
                         else None)
    if not _ck_train_len:
        print(f'[p5t] REFUSE {v} s{seed}: checkpoint and frozen plan declare no training context length — refusing to substitute a current-scale default')
        del d
        out['status'] = 'skipped_missing_train_len'
        return out
    model = L.SmallGPT(int(_ck_vocab), *_arch, int(_ck_train_len), [copy.deepcopy(c) for c in d['cfg']], mlp_ratio=d['mlp_ratio']).to(DEVICE)
    model.load_state_dict(d.pop('sd'))
    model.eval()
    # Keep the checkpoint's provenance fields on a separate binding so the
    # record writer never depends on a variable that this function deletes.
    _ck_prov = {k: d.get(k) for k in ('train_knobs', 'params', 'train_objective',
                                      'param_breakdown', 'rng_fp', 'init_core_fp')}
    # score EVERY loaded row: test_ids is the frozen slice already sliced to
    # the manifest's n_seqs_clean, and data_fp is computed over all of them.
    # eval_length_gen's default (n_seq=8) would measure only the
    # first 8 rows while the lock/fingerprint claim the full 48-row slice.
    # `test_doc_ids` is the per-token document mapping when the cache carries
    # it; it turns the "per-document" readout into a real document-level
    # statistic instead of treating arbitrary fixed rows as documents.
    r = V.eval_length_gen(model, test_ids, eval_lens, DEVICE, max_pos=None,
                          n_seq=int(test_ids.shape[0]),
                          doc_ids=test_doc_ids,
                          article_ids=test_article_ids)
    del model
    gc.collect()

    def _merge(summary):
        # by_len also carries ppl_per_seq (per-document rows)
        # straight from eval_length_gen; pooled ppl is unchanged.
        # The P5T record carries the SAME provenance descriptor as its
        # training record (`train_knobs` / `train_objective` /
        # `param_breakdown` / `rng_fp`) plus the evaluated row count.  Without
        # `train_knobs` the frozen implementation binding excluded ALL 48
        # records (admitted=0) after the one-shot test access had already been
        # burned.
        _tk = _tr.get('train_knobs') or _ck_prov.get('train_knobs')
        # _expect_init was resolved before the resume check; reuse the SAME
        # value here so the record and the staleness test cannot disagree.
        summary[f'{v}::s{seed}'] = {'variant': v, 'seed': seed, 'params': _tr.get('params') or _ck_prov.get('params'), 'by_len': r, 'split': split, 'data_fp': _dfp, 'train_fp': _tr_train_fp, '_code': V.CKPT_CODE, 'recipe': _ck_recipe, 'manifest_sha256': _msha, 'ckpt_sha256': _ensure_ck_sha(), 'ckpt_sig': (list(_ck_sig) if _ck_sig is not None else None),
                                    'n_seq': int(test_ids.shape[0]),
                                    'article_grouping': (dict(article_diag)
                                                         if isinstance(article_diag, dict)
                                                         else None),
                                    'train_knobs': _tk,
                                    'train_objective': _tr.get('train_objective') or _ck_prov.get('train_objective'),
                                    'param_breakdown': _tr.get('param_breakdown') or _ck_prov.get('param_breakdown'),
                                    'rng_fp': _tr.get('rng_fp') or _ck_prov.get('rng_fp'),
                                    'init_core_fp': _expect_init,
                                    'exec_fp': _ft_exec_fp,
                                    'environment': _ft_env,
                                    'amp_dtype': (_tk or {}).get('amp_dtype') if isinstance(_tk, dict) else None,
                                    'amp_grad_scaler': (_tk or {}).get('amp_grad_scaler') if isinstance(_tk, dict) else None,
                                    'eager_batch': (_tk or {}).get('eager_batch') if isinstance(_tk, dict) else None}
        print(f'  [p5t] {v:32s} s{seed} ' + '  '.join((f'L{k}={vv.get("ppl", float("nan")):.2f}' for k, vv in sorted(r.items()))))
        return summary
    L.locked_json_update(spath, _merge)
    L.log_experiment_event(outdir, {'kind': 'eval', 'tracking_schema': 2, 'run_id': _ft_run_id, 'status': 'ok', 'started_at': _ft_started, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'elapsed_s': time.time() - _ft_t0, 'key': f'{v}::s{seed}', 'variant': v, 'seed': int(seed), 'split': split, 'code': V.CKPT_CODE, 'manifest_sha256': _msha, 'recipe': _ck_recipe, 'ckpt_sha256': _ck_sha, 'data_fp': _dfp, 'train_fp': _tr_train_fp, 'params': _tr.get('params'), 'init_core_fp': _expect_init, 'exec_fp': _ft_exec_fp, 'environment': _ft_env, 'metrics': {'by_len': {str(k): (v2 or {}).get('ppl') for k, v2 in r.items()}}})
    del d
    return out


def _evaluation_order_design_seed(manifest):
    """Deterministic design seed for the P5T evaluation-order plan.

    A manifest-declared ``order.seed`` wins; otherwise the first 16 hex digits
    of the frozen manifest hash are used.  Tying the permutation to the frozen
    plan means a pre-access manifest revision changes the order, while a
    resumed run under the same manifest reuses exactly the same plan.
    """
    order = manifest.get('order') if isinstance(manifest.get('order'), dict) else {}
    raw = order.get('seed') if isinstance(order, dict) else None
    if isinstance(raw, numbers.Integral) and not isinstance(raw, bool):
        return int(raw)
    sha = manifest.get('manifest_sha256')
    if isinstance(sha, str) and len(sha) >= 16:
        try:
            return int(sha[:16], 16)
        except ValueError:
            pass
    return 0


def _test_access_consumed(outdir):
    """True once the one-shot test slice has been bound (or cannot be checked)."""
    if os.path.exists(os.path.join(outdir, 'bylen.json')):
        return True
    lp = _freeze_lock_path(outdir)
    if not os.path.exists(lp):
        return False
    try:
        with open(lp, encoding='utf-8') as f:
            lock = json.load(f)
    except Exception:
        return True
    if not isinstance(lock, dict):
        return True
    return bool(str(lock.get('stage') or '') == 'accessed'
                or lock.get('test_ids_fp') is not None)


def _ensure_evaluation_order_plan(outdir, manifest, task):
    """Write (or verify) the evaluation-order plan BEFORE the test read.

    The plan is derived only from the frozen (arm, seed) grid.  If a matching
    plan already exists it is reused byte-for-byte, so a resumed run keeps the
    original permutation and timestamp.  A mismatch is rewritten only while
    the test split is still unbound; after the first access it is a hard
    refusal, because reshuffling after seeing measurements would make the
    recorded order describe a different experiment.
    """
    cells = [(str(v), int(s)) for v in manifest['arms'] for s in manifest['seeds']]
    want = L.build_evaluation_order_plan(
        cells, task=task, seed=_evaluation_order_design_seed(manifest))
    path = os.path.join(outdir, f'{task}_order.json')
    stable = ('kind', 'schema', 'task', 'variants', 'seeds', 'blocks',
              'planned_cells', 'cell_blocks', 'cell_variants')
    if not os.path.exists(path) and _test_access_consumed(outdir):
        raise SystemExit(
            f'[v12] REFUSE: the one-shot test split is already bound but the '
            f'{task} evaluation-order plan {path} is missing — a plan written '
            'after measurement cannot establish that the order was '
            'precommitted; archive this results directory for a new round')
    if os.path.exists(path):
        try:
            with open(path, encoding='utf-8') as f:
                existing = json.load(f)
        except Exception as e:
            raise SystemExit(
                f'[v12] REFUSE: evaluation-order plan {path} exists but cannot '
                f'be parsed ({type(e).__name__}: {e}) — delete it only if you '
                'can prove no test-split measurement was written under it')
        if not isinstance(existing, dict):
            raise SystemExit(
                f'[v12] REFUSE: evaluation-order plan {path} is a '
                f'{type(existing).__name__}, not a JSON object')
        _same = (all(existing.get(k) == want.get(k) for k in stable)
                 and (existing.get('design') or {}).get('design_seed')
                 == want['design']['design_seed'])
        if not _same:
            if _test_access_consumed(outdir):
                raise SystemExit(
                    f'[v12] REFUSE: existing evaluation-order plan {path} does '
                    'not match the frozen (arm, seed) grid or design seed, and '
                    'the one-shot test split is already bound — refusing to '
                    'reshuffle after measurement; archive this results '
                    'directory for a genuinely new round')
            L.atomic_write_json(path, want, indent=2)
            return want
        _bal = L.run_order_balance(existing)
        if _bal.get('status') == 'fail':
            raise SystemExit(
                f'[v12] REFUSE: evaluation-order plan {path} is not a valid '
                f'complete-block design: {_bal.get("problems")[:3]}')
        return existing
    L.atomic_write_json(path, want, indent=2)
    return want


def _ordered_evaluation_cells(cells, plan, *, task):
    try:
        return L.evaluation_cells_in_plan_order(cells, plan)
    except Exception as e:
        raise SystemExit(
            f'[v12] REFUSE: {task} evaluation-order plan does not cover the '
            f'frozen cell grid ({type(e).__name__}: {e}) — refusing to '
            'measure an unplanned or incomplete panel')


def run_finaltest(cfg=None, guard=None):
    """Evaluate the FROZEN confirmation set on the held-out test split.

    只读冻结清单列出的臂/种子/终点；评估数据来自 `L.load_wikitest`（test
    split，其它 phase 不读）。产物：`results_lm_test/bylen.json`（干净
    by-length，含逐文档 ppl_per_seq）与 `results_lm_test/distractor.json`
    （uniform 腐坏探针）；确认性统计由 `python -m explib.confirmation analysis` 从这些产物
    现算。

    首次测试访问即写入冻结锁
    `.freeze_lock.json`（manifest sha256 + test 切片指纹）；此后清单或
    test 数据变更都会被拒绝（V12_REFREEZE=1 显式改冻结并降级前轮）。"""
    cfg = dict(FINALTEST_CFG) if cfg is None else dict(cfg)
    manifest = _load_manifest(cfg['manifest'])
    outdir = cfg['outdir']
    os.makedirs(outdir, exist_ok=True)
    need = max([int(x) for x in (manifest.get('eval') or {}).get('by_length_ppl', cfg['eval_lens'])]) + 1
    _nseq_clean = int((manifest.get('eval') or {}).get('n_seqs_clean') or cfg.get('n_seq', 48))
    test_ids = cfg.get('test_ids')
    _smoke = bool(cfg.get('smoke'))
    if not _smoke:
        # Direct `phase P5T` / finaltest must not bypass the same zero-test
        # preflight that protects P5C/P5MP.  plan_check() never reads test
        # data; it refuses an unconfirmable/misbound plan before the one-shot
        # access can be burned.
        plan_check(cfg.get('manifest') or FREEZE_MANIFEST)
        # The pre-test power gate is part of the frozen plan (v4).  Enforce it
        # on the direct `phase P5T` path too, not only in run_full's P5PW
        # phase: run_power() reads only the validation summary and refuses
        # before load_wikitest when the shipped plan is under-powered.
        run_power(cfg.get('manifest') or FREEZE_MANIFEST)
    if test_ids is not None:
        # A caller-supplied slice is ONLY a plumbing smoke input.  In normal
        # mode the test split must come from load_wikitest, otherwise a
        # validation slice could be relabelled as "test" and then frozen.
        if not _smoke:
            raise SystemExit('[v12] REFUSE: test_ids injection is allowed only with cfg["smoke"]=True; the confirmation test split must be read from exp_lib.load_wikitest')
        if 'smoke' not in os.path.basename(os.path.abspath(outdir)).lower():
            raise SystemExit(f'[v12] REFUSE: smoke test_ids require a smoke-marked outdir (got {outdir!r}) — injected data must never enter the confirmation results directory')
    # Derive/lock the canonical training-corpus fingerprint BEFORE the first
    # test read.  A manifest-level value wins; otherwise every frozen training
    # record must name the same non-null train_fp.
    _want_train_fp = _frozen_train_fp(manifest)
    _cand_train_fps = set()
    if os.path.exists(cfg.get('mech_summary') or ''):
        try:
            _ms = json.load(open(cfg['mech_summary'], encoding='utf-8'))
        except Exception:
            _ms = {}
        for _v in manifest['arms']:
            for _s in manifest['seeds']:
                _fp = (_ms.get(f'{_v}::seed{_s}') or {}).get('data_fp')
                if isinstance(_fp, str) and _fp:
                    _cand_train_fps.add(_fp)
    if _want_train_fp is not None and _cand_train_fps and _cand_train_fps != {_want_train_fp}:
        raise SystemExit(f'[v12] REFUSE: training records use train_fp {sorted(_cand_train_fps)}, but the frozen canonical value is {_want_train_fp}')
    if _want_train_fp is None and len(_cand_train_fps) == 1:
        _want_train_fp = next(iter(_cand_train_fps))
    elif _want_train_fp is None and len(_cand_train_fps) > 1:
        raise SystemExit(f'[v12] REFUSE: the frozen training records span {len(_cand_train_fps)} different train_fp values {sorted(_cand_train_fps)}; refusing to define a canonical training set after the fact')
    # A manifest that DECLARES training_data_fingerprint may not fall
    # through to a null canonical value.  Deriving it from the frozen training
    # records is the only legal path; if that is impossible, refuse BEFORE the
    # one-shot test read rather than discovering it per-cell afterwards.
    if not _smoke and _train_fp_required(manifest) and _want_train_fp is None:
        raise SystemExit('[v12] REFUSE: the frozen manifest declares a training-corpus binding but no training_fp can be derived from the frozen records — run P5C/P3MT/P4MT first and keep their summary.json; refusing to read the held-out split under an unpinned corpus')
    # Bind the LOCAL train/test cache pair to the training stream the frozen
    # checkpoints actually saw.  load_wikitest proves test==current train
    # tokenizer; this proves current train==frozen training corpus.
    if not _smoke and _want_train_fp is not None:
        _require_current_train_cache_fp(_want_train_fp)
    if not _smoke:
        # The held-out read is one-shot, so a missing checkpoint must be found
        # BEFORE it, not by silently skipping cells after the access has
        # already been consumed.  A genuinely partial round must say so
        # explicitly and is reported as incomplete, never as a confirmation.
        _missing_ck = [f'{_v}::s{_s}' for _v in manifest['arms']
                       for _s in manifest['seeds']
                       if not os.path.exists(os.path.join(cfg['ckpt_dir'],
                                                          f'{_v}_seed{_s}.pt'))]
        _allow_partial = os.environ.get('V12_ALLOW_PARTIAL_TEST', '').strip().lower() in ('1', 'true', 'yes', 'on')
        if _missing_ck and not _allow_partial:
            raise SystemExit(
                f'[v12] REFUSE: {len(_missing_ck)} frozen checkpoint(s) are '
                f'missing ({_missing_ck[:6]}…) — refusing to read the one-shot '
                'test split for a panel that cannot produce its declared '
                'cells.  Train the missing cells first, or set '
                'V12_ALLOW_PARTIAL_TEST=1 for an explicitly incomplete round.')
        if _missing_ck:
            print(f'[v12] WARNING: V12_ALLOW_PARTIAL_TEST=1 — {len(_missing_ck)} '
                  'declared checkpoint(s) are missing; this round is INCOMPLETE '
                  'and must not be reported as a confirmation')
    _lock_path = _freeze_lock_path(outdir)
    if os.path.exists(os.path.join(outdir, 'bylen.json')) and not os.path.exists(_lock_path):
        raise SystemExit(f'[v12] REFUSE: {outdir}/bylen.json exists but there is no pre-access freeze lock — the only honest interpretation is that a test-split measurement predates the lock; investigate before reusing it')
    # Training-health gate BEFORE the one-shot test read/lock: a run whose own
    # diagnostic curve is non-finite/overfit/stalled must be investigated, not
    # silently evaluated on test.  This never changes the frozen plan.
    if not _smoke:
        run_training_health({'manifest': cfg.get('manifest') or FREEZE_MANIFEST,
                             'ckpt_dir': cfg.get('ckpt_dir') or FINALTEST_CFG['ckpt_dir'],
                             'out': os.path.join(outdir, 'training_health.json')})
    # Write the pre-access attestation BEFORE reading the held-out split, so a
    # manifest mismatch is discovered before the one-shot access is used.
    _check_or_write_freeze_lock(outdir, manifest, None, stage='preaccess',
                                train_fp=_want_train_fp)
    # Precommit the measurement order BEFORE load_wikitest.  The order is
    # derived only from the frozen (arm, seed) grid, so writing it cannot leak
    # test information; it keeps a resume or a machine-drift episode from
    # silently reverting to arm-major evaluation.  The probe has its own plan
    # because it is a separate task (and a separate event kind).
    _clean_plan = _ensure_evaluation_order_plan(outdir, manifest, 'clean_by_length')
    _probe_plan = _ensure_evaluation_order_plan(outdir, manifest, 'distractor_probe')
    _frozen_vocab = _frozen_vocab_size(manifest, cfg)
    test_doc_ids = None
    test_article_ids = None
    article_diag = None
    if test_ids is None:
        _t0 = time.time()
        test_ids, vocab_t, _, _, test_doc_ids = L.load_wikitest(
            need, n_seqs=_nseq_clean, vocab_size=_frozen_vocab,
            purpose='final_evaluation', return_doc_ids=True)
        if guard is not None:
            guard.record_run(time.time() - _t0, 0, 0, 0, 0, 0)
        if test_doc_ids is None:
            print('[p5t] NOTE: the frozen test cache has no per-token document-id '
                  'sidecar (legacy cache); by-length records carry row-level '
                  'uncertainty only.  The pooled PPL is unaffected, but the '
                  'document bootstrap cannot be claimed as document-level for '
                  'this cache.')
        if _frozen_vocab is not None and int(vocab_t) != int(_frozen_vocab):
            raise SystemExit(
                f'[v12] REFUSE: loaded test vocabulary {vocab_t} != the frozen '
                f'{_frozen_vocab} — the held-out cache belongs to another '
                'tokenizer/scale protocol')
        if _frozen_vocab is None:
            # No binding declared the width, but the served test cache did:
            # use it as the binding so every checkpoint is still checked
            # against the tokenizer that actually produced the slice.
            _frozen_vocab = int(vocab_t)
    else:
        print('[p5t] SMOKE: test_ids supplied by the caller; records are stamped split="smoke" and are inadmissible to the confirmation statistics')
    _dfp = L._ids_fp(np.ascontiguousarray(test_ids).reshape(-1))
    if int(test_ids.shape[0]) != _nseq_clean or int(test_ids.shape[1]) != need:
        raise SystemExit(f'[v12] REFUSE: loaded test slice shape {tuple(test_ids.shape)} != the frozen ({_nseq_clean}, {need}) — the row count and sequence length are part of the pre-registration; re-freeze explicitly instead of shrinking the panel')
    # Bind the slice fingerprint IMMEDIATELY after the test read and shape
    # check, before any audit/evaluation.  If an audit then fails or the
    # process dies, the lock must still name the slice that was read; leaving
    # a null preaccess lock would let a later run bind a different slice.
    _check_or_write_freeze_lock(outdir, manifest, _dfp, stage='accessed',
                                train_fp=_want_train_fp)
    # The slice is now bound.  Recover the article clusters from the SAME
    # served runs so the evaluation corpus has the independent unit; a failure
    # is disclosed and falls back to the line map, never guessed.  Doing this
    # AFTER the lock keeps the one-shot fingerprint write as early as possible.
    if test_doc_ids is not None and _frozen_vocab is not None:
        test_article_ids, article_diag = _resolve_test_article_ids(
            test_ids, test_doc_ids, n_seqs=_nseq_clean, seq_len=need,
            vocab_size=int(_frozen_vocab))
        if test_article_ids is not None:
            print(f'[p5t] article unit: {article_diag.get("n_articles")} '
                  f'articles from {article_diag.get("n_lines")} served lines '
                  f'(pseudoreplication factor '
                  f'{article_diag.get("pseudoreplication_factor"):.2f}); '
                  'line-level fields are unchanged')
        else:
            print('[p5t] NOTE: article-level clustering is unavailable '
                  f'({article_diag.get("status")}: '
                  f'{article_diag.get("reason")}); the document readout stays '
                  'line-level and population-level uncertainty remains '
                  'pseudoreplicated')
    if not _smoke:
        # Run the leakage gate AFTER the slice is bound but BEFORE any
        # evaluation.  The binding must survive a failed audit (that is what
        # keeps the one-shot slice immutable); the audit still gates whether
        # any measurement is produced.  A gate that can never fire (a
        # Jaccard<=0.05 on a 4M-vs-65k shingle pair) is not a gate, and a
        # `not_verifiable_locally` result must not pass silently.  Both are
        # hard refusals; only a verifiable, clean audit unlocks the panel.
        _cache_dir = L._wt_cache_dir()
        _tf = ((manifest.get('binding') or {}).get('training_data_fingerprint') or {})
        _pref_sl = _tf.get('seq_len') if isinstance(_tf, dict) else None
        try:
            import audit_gates as _AG
            _ov = _AG.audit_leakage(cache_dir=_cache_dir,
                                    preferred_train_seq_len=_pref_sl)
        except Exception as _ove:
            raise SystemExit(f'[v12] REFUSE: the data-leakage audit could not run ({type(_ove).__name__}: {_ove}) — a confirmation claim requires a verifiable train/test isolation audit, not an untested one')
        if _ov.get('status') == 'not_verifiable_locally':
            raise SystemExit(f'[v12] REFUSE: the data-leakage audit is not verifiable locally ({_ov.get("note")}) — build/pair the train and test caches under one WT103_CACHE_DIR before using the one-shot test access; an unverifiable isolation claim is a hard refusal, not a warning')
        if _ov.get('status') == 'fail':
            _doc = (_ov.get('checks') or {}).get('doc_overlap') or {}
            raise SystemExit(f'[v12] REFUSE: the data-leakage audit reports a hard failure (doc_overlap={_doc if isinstance(_doc, dict) else _doc!r}); see audit_gates.py leakage output — confirmation claims are unavailable until the near-duplicate evaluation documents are excluded')
        _doc = (_ov.get('checks') or {}).get('doc_overlap')
        if not isinstance(_doc, dict):
            raise SystemExit(f'[v12] REFUSE: the leakage audit did not produce a document-overlap result ({_doc!r}) — the npy caches are missing or unreadable, so train/test isolation cannot be verified before the one-shot test access')
        if _doc.get('policy_violation'):
            raise SystemExit(f'[v12] REFUSE: train/test document overlap (test-shingle overlap {_doc.get("test_shingle_overlap")}, max row overlap {_doc.get("max_row_overlap")}) exceeds the frozen policy — confirmation claims are UNAVAILABLE until the near-duplicate evaluation documents are cleaned (rebuild the test cache with exp_lib.load_wikitest, which filters them)')
        # The leakage audit proves *provenance/overlap*; it does not re-check
        # the split contract itself.  Require the split-discipline gate too,
        # so exact train/test row overlap, document-group leakage, duplicate
        # holdout rows and a material train->test distribution shift are all
        # refused BEFORE the one-shot test access.  Warnings (e.g. a modest
        # natural domain shift, or a legacy metadata field) remain visible in
        # the audit output but do not by themselves invalidate the frozen
        # panel; a hard failure or an unverifiable cache does.
        try:
            _sd = _AG.audit_split_discipline(cache_dirs=[_cache_dir],
                                             preferred_train_seq_len=_pref_sl)
        except Exception as _sde:
            raise SystemExit(f'[v12] REFUSE: the split-discipline audit could not run ({type(_sde).__name__}: {_sde}) — a confirmation claim requires a verifiable split audit, not an untested one')
        if _sd.get('status') == 'not_verifiable_locally':
            raise SystemExit(f'[v12] REFUSE: the split-discipline audit found no auditable cache under {_cache_dir} — build the frozen train/validation/test caches before using the one-shot test access')
        if _sd.get('status') == 'fail':
            _why = list(_sd.get('warnings') or [])
            raise SystemExit(f'[v12] REFUSE: the split-discipline audit reports a hard failure ({"; ".join(_why[:4]) if _why else "see audit_gates.py split"}) — the held-out test split is not clean enough for a frozen confirmation')
    print(f'[p5t] manifest {cfg["manifest"]} frozen_at={manifest.get("frozen_at", "?")} sha256={manifest.get("manifest_sha256", "?")[:12]}… arms={manifest["arms"]} seeds={manifest["seeds"]}')
    print(f'[p5t] TEST split rows {test_ids.shape} (held out; no other phase reads this split; ids_fp={_dfp[:12]}…)')
    if _nseq_clean < 64:
        print(f'[p5t] NOTE: n_seqs_clean={_nseq_clean} — document-sampling uncertainty is NOT well covered at this row count; population-level claims need a larger row panel (per-doc losses are recorded in bylen.json for stratified analysis; see design_notes §20)')
    # 1) clean by-length per (arm, seed)
    _split_label = 'smoke' if _smoke else 'test'
    _ft_env_once = L.environment_snapshot()
    _cell_status = {}
    _clean_cells = [(str(v), int(s)) for v in manifest['arms']
                    for s in manifest['seeds']]
    for v, seed in _ordered_evaluation_cells(
            _clean_cells, _clean_plan, task='clean_by_length'):
        _cell = _finaltest_bylen_cell(
            cfg, manifest, v, seed, test_ids,
            split=_split_label, train_fp=_want_train_fp,
            frozen_vocab=_frozen_vocab,
            env=_ft_env_once,
            test_doc_ids=test_doc_ids,
            test_article_ids=test_article_ids,
            article_diag=article_diag)
        _cell_status[f'{v}::s{seed}'] = (_cell or {}).get('status')
    # 2) frozen uniform-corruption probe on the SAME test rows (secondary,
    #    descriptive-only per the manifest)
    _probe_eval = (manifest.get('eval') or {}).get('distractor_probe') or {}
    pcfg = dict(V10.PROBE, outdir=outdir, ckpt_dir=cfg['ckpt_dir'], summary=cfg['probe_summary'], variants=list(manifest['arms']), seeds=[int(s) for s in manifest['seeds']], eval_lens=[int(x) for x in (_probe_eval.get('eval_lens') or [2048, 4096])], rhos=[float(x) for x in (_probe_eval.get('rhos') or [0.0, 0.125, 0.25, 0.5])], modes={'uniform': [float(x) for x in (_probe_eval.get('rhos') or [0.0, 0.125, 0.25, 0.5])]}, niah=None, arm_overrides={v: ['learned'] for v in ('csa_fixed_rope_randidx_seeded', 'csa_fixed_rope_allblocks')}, target=int(_probe_eval.get('target', cfg.get('target', 512))), n_seq=int(_probe_eval.get('n_seq', cfg.get('n_seq', 48))), chunk=int(cfg.get('chunk', 4)))
    mech_summ = {}
    if os.path.exists(cfg['mech_summary']):
        mech_summ = json.load(open(cfg['mech_summary'], encoding='utf-8'))
    _probe_cells = [(str(v), int(s)) for v in pcfg['variants']
                    for s in pcfg['seeds']]
    for v, seed in _ordered_evaluation_cells(
            _probe_cells, _probe_plan, task='distractor_probe'):
        V10.run_probe_cell(pcfg, v, seed, guard=guard, val_ids=test_ids,
                           mech_summ=mech_summ)
    # 3) confirmation statistics (zero GPU, from disk)
    _stats = finaltest_stats(manifest=manifest, bylen_path=cfg['bylen_summary'],
                             probe_path=cfg['probe_summary'],
                             out=cfg.get('stats_out',
                                         'analysis_v12/test_stats.json'))
    # A normal return from every per-cell call is NOT completion: skipped
    # checkpoints/provenance/health leave the one-shot split already consumed
    # but bylen admission empty.  Raise so run_phase/run_full records P5T as
    # incomplete instead of printing ALL PHASES DONE.
    _allow_partial = os.environ.get('V12_ALLOW_PARTIAL_TEST', '').strip().lower() in ('1', 'true', 'yes', 'on')
    _admission_problem = _finaltest_admission_problem(
        _stats, manifest, smoke=_smoke, allow_partial=_allow_partial)
    if _admission_problem:
        _non_ok = {k: s for k, s in _cell_status.items() if s != 'ok'}
        raise RuntimeError(
            f'[v12] P5T INCOMPLETE: {_admission_problem}. '
            f'Non-ok cell statuses: {_non_ok if _non_ok else "none"}')
    return _stats

def _by_len_shape_reason(rec, n_seq):
    """Why a record's by_len cells do not match the frozen (n_seq, seq_len) grid.

    ``eval_length_gen`` predicts ``Ln - 1`` tokens per row, so a complete cell
    must carry ``n_tok == n_seq * (Ln - 1)``; a row-count mismatch or an empty
    slice is refused rather than admitted.
    """
    bl = rec.get('by_len') or {}
    if not isinstance(bl, dict) or not bl:
        return 'no by_len measurement'
    for _ln, _c in bl.items():
        if not isinstance(_c, dict):
            return f'by_len[{_ln!r}] is not a cell object'
        if isinstance(_ln, (bool, np.bool_)):
            return f'by_len key {_ln!r} is not an integer length'
        try:
            _ln_f = float(_ln)
            if not math.isfinite(_ln_f) or _ln_f != math.floor(_ln_f):
                raise ValueError
            _ln_i = int(_ln_f)
        except (TypeError, ValueError, OverflowError):
            return f'by_len key {_ln!r} is not an integer length'
        if _ln_i < 2:
            return f'by_len length {_ln_i} has no next-token prediction (needs Ln>=2); an empty cell is not admissible'
        _ntok = _c.get('n_tok')
        _want = int(n_seq) * (_ln_i - 1)
        if (isinstance(_ntok, (bool, np.bool_))
                or not isinstance(_ntok, (int, float, np.integer, np.floating))):
            return f'by_len[{_ln_i}].n_tok={_ntok!r} != the frozen {_want} (= {n_seq} rows x {_ln_i - 1} predictions) — truncated/empty cells are not admissible'
        try:
            _ntok_f = float(_ntok)
        except (TypeError, ValueError, OverflowError):
            _ntok_f = float('nan')
        if not math.isfinite(_ntok_f) or _ntok_f != math.floor(_ntok_f) or int(_ntok_f) != _want:
            return f'by_len[{_ln_i}].n_tok={_ntok!r} != the frozen {_want} (= {n_seq} rows x {_ln_i - 1} predictions) — truncated/empty cells are not admissible'
        # A cell that does not explicitly certify full length is unverifiable.
        # Matching n_tok alone cannot distinguish a full cell from a record
        # whose truncation marker was lost when it was serialised.
        if 'truncated' not in _c:
            return f'by_len[{_ln_i}] has no `truncated` flag — an unverifiable cell is not admissible'
        if not isinstance(_c.get('truncated'), (bool, np.bool_)):
            return f'by_len[{_ln_i}].truncated={_c.get("truncated")!r} is not a boolean — an unverifiable cell is not admissible'
        if bool(_c.get('truncated')):
            return f'by_len[{_ln_i}] is marked truncated — a partial cell is not admissible'
        # A per-document vector must have one value per declared/frozen row.
        # Otherwise population-claim gates can count 64 values for an n_seq=16
        # evaluation and admit a wider document sample than was measured.
        if 'ppl_per_seq' in _c:
            _pps = _c.get('ppl_per_seq')
            if not isinstance(_pps, list) or len(_pps) != int(n_seq):
                return (f'by_len[{_ln_i}].ppl_per_seq has '
                        f'{len(_pps) if isinstance(_pps, list) else type(_pps).__name__} '
                        f'entries, not the frozen n_seq={int(n_seq)}')
        # A document-level cell is a stronger claim than a row-level one: the
        # per-document token counts must partition exactly the scored targets,
        # and the document list must be complete.  Otherwise the downstream
        # "document bootstrap" would resample an inconsistent unit count.
        _gran = _c.get('doc_granularity')
        _has_doc_cell = any(k in _c for k in
                            ('nll_per_doc', 'n_tokens_per_doc', 'ppl_per_doc',
                             'doc_ids'))
        if _gran == 'document':
            for _k in ('nll_per_doc', 'n_tokens_per_doc', 'ppl_per_doc', 'doc_ids'):
                if not isinstance(_c.get(_k), list) or not _c.get(_k):
                    return (f'by_len[{_ln_i}].{_k} is missing while '
                            'doc_granularity=document — the document-level '
                            'claim is unverifiable')
            _dlen = {len(_c[_k]) for _k in
                     ('nll_per_doc', 'n_tokens_per_doc', 'ppl_per_doc', 'doc_ids')}
            if len(_dlen) != 1:
                return (f'by_len[{_ln_i}] document vectors disagree in length '
                        f'{sorted(_dlen)}')
            _dnt = _c.get('n_tokens_per_doc')
            _dsum = 0
            for _x in _dnt:
                if (isinstance(_x, (bool, np.bool_))
                        or not isinstance(_x, (int, float, np.integer, np.floating))):
                    return f'by_len[{_ln_i}].n_tokens_per_doc contains {_x!r}'
                _xf = float(_x)
                if not math.isfinite(_xf) or _xf != math.floor(_xf) or _xf <= 0:
                    return (f'by_len[{_ln_i}].n_tokens_per_doc contains {_x!r}; '
                            'every retained document needs at least one scored '
                            'target (zero-target documents are dropped, not '
                            'reported as evaluation units)')
                _dsum += int(_xf)
            if _dsum != int(_ntok_f):
                return (f'by_len[{_ln_i}].n_tokens_per_doc sums to {_dsum}, '
                        f'not the scored n_tok={int(_ntok_f)}')
            if _c.get('n_docs') is not None:
                _nd = _c.get('n_docs')
                if (isinstance(_nd, (bool, np.bool_))
                        or not isinstance(_nd, (int, float, np.integer, np.floating))):
                    return f'by_len[{_ln_i}].n_docs={_nd!r} is not an integer'
                try:
                    _ndf = float(_nd)
                except (TypeError, ValueError, OverflowError):
                    _ndf = float('nan')
                if (not math.isfinite(_ndf) or _ndf != math.floor(_ndf)
                        or int(_ndf) != len(_dnt)):
                    return (f'by_len[{_ln_i}].n_docs={_nd!r} != the '
                            f'{len(_dnt)} per-document entries')
            # A document-level NLL vector must actually reconstruct the cell's
            # own totals.  Otherwise a downstream pooled-Δ interval would be
            # computed from one grouping while the reported PPL came from
            # another, and the mismatch would be invisible.
            _dnll = _c.get('nll_per_doc')
            for _x in _dnll:
                if (isinstance(_x, (bool, np.bool_))
                        or not isinstance(_x, (int, float, np.integer, np.floating))):
                    return f'by_len[{_ln_i}].nll_per_doc contains {_x!r}'
                if not math.isfinite(float(_x)):
                    return f'by_len[{_ln_i}].nll_per_doc contains {_x!r}'
            if _c.get('nll_sum') is not None:
                _ns = _c.get('nll_sum')
                if (isinstance(_ns, (bool, np.bool_))
                        or not isinstance(_ns, (int, float, np.integer, np.floating))):
                    return f'by_len[{_ln_i}].nll_sum={_ns!r} is not numeric'
                _nsf = float(_ns)
                if (not math.isfinite(_nsf)
                        or abs(math.fsum(float(x) for x in _dnll) - _nsf)
                        > 1e-5 * max(1.0, abs(_nsf))):
                    return (f'by_len[{_ln_i}].nll_per_doc sums to '
                            f'{math.fsum(float(x) for x in _dnll)!r}, not the '
                            f'recorded nll_sum={_nsf!r}')
        elif _has_doc_cell:
            # A partially-written document cell is not evidence; do not let a
            # row-labelled record smuggle incomplete document fields into the
            # analysis.
            for _k in ('nll_per_doc', 'n_tokens_per_doc', 'ppl_per_doc', 'doc_ids'):
                if not isinstance(_c.get(_k), list) or not _c.get(_k):
                    return (f'by_len[{_ln_i}] carries partial document fields '
                            f'({_k} missing)')
        # Article-level cells are the independent-unit readout.  They are
        # additive (the line-level document fields above must stay present),
        # but a declared article cell must partition the SAME scored targets
        # and reconstruct the SAME totals; otherwise the article bootstrap
        # would resample a different grouping than the reported PPL.
        _agran = _c.get('article_granularity')
        _art_keys = ('nll_per_article', 'n_tokens_per_article',
                     'ppl_per_article', 'article_ids')
        _has_art_cell = any(k in _c for k in _art_keys)
        if _agran not in (None, 'article'):
            return (f'by_len[{_ln_i}].article_granularity={_agran!r} is not '
                    '"article"')
        if _agran == 'article':
            for _k in _art_keys:
                if not isinstance(_c.get(_k), list) or not _c.get(_k):
                    return (f'by_len[{_ln_i}].{_k} is missing while '
                            'article_granularity=article — the article-level '
                            'claim is unverifiable')
            _alen = {len(_c[_k]) for _k in _art_keys}
            if len(_alen) != 1:
                return (f'by_len[{_ln_i}] article vectors disagree in length '
                        f'{sorted(_alen)}')
            _ant = _c.get('n_tokens_per_article')
            _asum = 0
            for _x in _ant:
                if (isinstance(_x, (bool, np.bool_))
                        or not isinstance(_x, (int, float, np.integer, np.floating))):
                    return f'by_len[{_ln_i}].n_tokens_per_article contains {_x!r}'
                _xf = float(_x)
                if not math.isfinite(_xf) or _xf != math.floor(_xf) or _xf <= 0:
                    return (f'by_len[{_ln_i}].n_tokens_per_article contains '
                            f'{_x!r}; every retained article needs at least '
                            'one scored target')
                _asum += int(_xf)
            if _asum != int(_ntok_f):
                return (f'by_len[{_ln_i}].n_tokens_per_article sums to '
                        f'{_asum}, not the scored n_tok={int(_ntok_f)}')
            if _c.get('n_articles') is not None:
                _na = _c.get('n_articles')
                if (isinstance(_na, (bool, np.bool_))
                        or not isinstance(_na, (int, float, np.integer, np.floating))):
                    return f'by_len[{_ln_i}].n_articles={_na!r} is not an integer'
                try:
                    _naf = float(_na)
                except (TypeError, ValueError, OverflowError):
                    _naf = float('nan')
                if (not math.isfinite(_naf) or _naf != math.floor(_naf)
                        or int(_naf) != len(_ant)):
                    return (f'by_len[{_ln_i}].n_articles={_na!r} != the '
                            f'{len(_ant)} per-article entries')
            _aids = _c.get('article_ids')
            try:
                _aset = set(_aids)
            except TypeError:
                return (f'by_len[{_ln_i}].article_ids contains an unhashable '
                        'entry; the article list is malformed')
            if len(_aset) != len(_aids):
                return (f'by_len[{_ln_i}].article_ids contains duplicate ids; '
                        'an article cannot be counted twice')
            if _c.get('n_docs') is not None:
                _nd_cmp = _record_int(_c.get('n_docs'))
                if _nd_cmp is not None and len(_ant) > _nd_cmp:
                    return (f'by_len[{_ln_i}] has {len(_ant)} articles but only '
                            f'{_nd_cmp} line-level documents; article clusters '
                            'cannot be more numerous than their lines')
            _anll = _c.get('nll_per_article')
            for _x in _anll:
                if (isinstance(_x, (bool, np.bool_))
                        or not isinstance(_x, (int, float, np.integer, np.floating))):
                    return f'by_len[{_ln_i}].nll_per_article contains {_x!r}'
                if not math.isfinite(float(_x)):
                    return f'by_len[{_ln_i}].nll_per_article contains {_x!r}'
            if _c.get('nll_sum') is not None:
                _ns = _c.get('nll_sum')
                if (isinstance(_ns, (bool, np.bool_))
                        or not isinstance(_ns, (int, float, np.integer, np.floating))):
                    return f'by_len[{_ln_i}].nll_sum={_ns!r} is not numeric'
                _nsf = float(_ns)
                if (not math.isfinite(_nsf)
                        or abs(math.fsum(float(x) for x in _anll) - _nsf)
                        > 1e-5 * max(1.0, abs(_nsf))):
                    return (f'by_len[{_ln_i}].nll_per_article sums to '
                            f'{math.fsum(float(x) for x in _anll)!r}, not the '
                            f'recorded nll_sum={_nsf!r}')
            for _tot_key, _want in (('n_tokens_article_total', _asum),
                                    ('n_articles', len(_ant))):
                if _c.get(_tot_key) is not None:
                    _tv = _record_int(_c.get(_tot_key))
                    if _tv is None or _tv != _want:
                        return (f'by_len[{_ln_i}].{_tot_key}='
                                f'{_c.get(_tot_key)!r} does not match {_want}')
        elif _has_art_cell:
            for _k in _art_keys:
                if not isinstance(_c.get(_k), list) or not _c.get(_k):
                    return (f'by_len[{_ln_i}] carries partial article fields '
                            f'({_k} missing)')
        if _c.get('n_seq') is not None:
            _ns = _c.get('n_seq')
            if isinstance(_ns, (bool, np.bool_)) or not isinstance(
                    _ns, (int, float, np.integer, np.floating)):
                return f'by_len[{_ln_i}].n_seq={_ns!r} != the frozen n_seq={int(n_seq)}'
            try:
                _nsf = float(_ns)
            except (TypeError, ValueError, OverflowError):
                return f'by_len[{_ln_i}].n_seq={_ns!r} != the frozen n_seq={int(n_seq)}'
            if not math.isfinite(_nsf) or _nsf != int(n_seq):
                return f'by_len[{_ln_i}].n_seq={_ns!r} != the frozen n_seq={int(n_seq)}'
    return None


def _record_int(value):
    """Exact integer view of a recorded count, or None for anything else."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(f) or f != math.floor(f):
        return None
    return int(f)


def _valid_sha256(value):
    return (isinstance(value, str)
            and re.fullmatch(r'[0-9a-fA-F]{64}', value or '') is not None)


def _checkpoint_identity_problem(ckpt, variant, seed):
    """Return why a P5T checkpoint is not the requested arm/seed, or None.

    The lenphase recipe is deliberately arm-independent for several frozen
    controls (they share the same parameter count and hyper-parameters), and
    ``init_core_fp`` is a function of the seed, not the arm.  A copied file
    therefore passed every recipe/train-fp/init check.  Bind the payload to
    its (variant, seed) and verify that the declared weight digest is the one
    actually stored inside the file.
    """
    if not isinstance(ckpt, dict):
        return 'checkpoint payload is not an object'
    got_variant = ckpt.get('variant')
    if got_variant is None:
        return 'checkpoint carries no variant identity (legacy/pre-identity writer)'
    if str(got_variant) != str(variant):
        return f'checkpoint variant {got_variant!r} != requested {variant!r}'
    got_seed = _record_int(ckpt.get('seed'))
    if got_seed is None:
        return 'checkpoint carries no exact integer seed identity'
    if got_seed != int(seed):
        return f'checkpoint seed {got_seed!r} != requested {int(seed)!r}'
    declared = ckpt.get('state_dict_sha256')
    if not _valid_sha256(declared):
        return ('checkpoint carries no valid state_dict_sha256 (legacy/pre-'
                'identity writer); the stored weights cannot be tied to this '
                'arm/seed')
    sd = ckpt.get('sd')
    if not isinstance(sd, dict) or not sd:
        return 'checkpoint carries no state-dict mapping'
    try:
        actual = L.state_dict_sha256(sd)
    except Exception as e:
        return f'checkpoint state dict could not be hashed ({type(e).__name__}: {e})'
    if actual != declared:
        return ('checkpoint state-dict digest does not match its declared '
                'state_dict_sha256; the weights were edited or mis-copied')
    return None


def _manifest_requires_params(manifest):
    """Whether the frozen plan's own disclosure makes `params` mandatory.

    Parameter matching is part of the frozen comparison contract: the main
    dense reference is widened to match the CSA parameter total.  A plan that
    discloses `compute_budget_disclosure.parameter_matching` therefore cannot
    accept a confirmation record with no parameter count — the decision rule's
    `param_matched is not false` would otherwise be satisfied by *unknown*.
    Descriptive-only smoke plans and legacy manifests without the disclosure
    keep the permissive path (and still report `param_matched=None`).
    """
    if not isinstance(manifest, dict):
        return False
    if (manifest.get('statistics') or {}).get('descriptive_only') is True:
        return False
    _cbd = manifest.get('compute_budget_disclosure')
    return (isinstance(_cbd, dict)
            and bool(str(_cbd.get('parameter_matching') or '').strip()))


def _param_match_detail(recs_a, recs_b, seeds, a='a', b='b'):
    """Per-seed parameter identity for one confirmatory comparison.

    A single seed carrying `params` cannot certify an arm's capacity: the
    frozen decision rule uses the arm totals, and `_params_of_arm` in
    `exp_lib` already refuses a partial/mixed arm.  This helper reports the
    per-seed counts, requires every usable seed on BOTH sides, and requires
    each arm's own counts to be internally identical before returning a
    True/False verdict.  Anything missing/inconsistent is `None`
    (unverifiable), never silently treated as a match.
    """
    out = {'params_a': None, 'params_b': None, 'param_rel_diff': None,
           'param_matched': None, 'param_match_complete': False,
           'param_match_reason': None,
           'params_by_seed': {'a': {}, 'b': {}},
           'param_consistent': {'a': None, 'b': None}}
    seeds = list(seeds or [])
    side = {'a': recs_a, 'b': recs_b}
    missing = {'a': [], 'b': []}
    for label, recs in side.items():
        vals = {}
        for s in seeds:
            rec = recs.get(s)
            v = _record_int(rec.get('params')) if isinstance(rec, dict) else None
            if v is None or v <= 0:
                missing[label].append(int(s))
            else:
                vals[int(s)] = int(v)
        out['params_by_seed'][label] = {str(k): int(v) for k, v in sorted(vals.items())}
        out['param_consistent'][label] = (len(set(vals.values())) <= 1) if vals else None
    if not seeds:
        out['param_match_reason'] = 'no usable seed pairs'
        return out
    _bad = []
    _names = {'a': a, 'b': b}
    for label in ('a', 'b'):
        if missing[label]:
            _bad.append(f'{_names[label]} missing/invalid params for seed(s) '
                        f'{sorted(missing[label])}')
        elif out['param_consistent'][label] is False:
            _bad.append(f'{_names[label]} has different params across its own '
                        f"seeds ({sorted(set(out['params_by_seed'][label].values()))})")
    if _bad:
        out['param_match_reason'] = '; '.join(_bad)
        return out
    _pa = list(out['params_by_seed']['a'].values())
    _pb = list(out['params_by_seed']['b'].values())
    out['params_a'] = float(np.mean(_pa))
    out['params_b'] = float(np.mean(_pb))
    out['param_match_complete'] = True
    _am, _bm = out['params_a'], out['params_b']
    _rel = abs(_am - _bm) / max(_am, _bm, 1.0)
    out['param_rel_diff'] = float(_rel)
    out['param_matched'] = bool(_rel <= 0.01)
    return out


def _admit_confirmation_records(raw, manifest):
    """Strict admission gate for confirmation records.

    一条 bylen 记录要进入确认统计，必须同时满足：
      split==test、variant 在冻结臂内、seed 在冻结白名单内、_code 等于冻结
    清单声明的代码语义、manifest_sha256 等于当前清单哈希（同修订轮）、
    data_fp 非空、recipe 通过冻结配方签名、by_len 非空。清单披露了
    `compute_budget_disclosure.parameter_matching` 时，参数计数（params）
    也必须有有效正值——否则决策规则的「param_matched is not false」会把
    「未知」当成通过。
    不满足者被**点名剔除**（不入统计，也不静默消失）；同一 (variant, seed)
    出现两条可采纳记录 = 歧义，**拒绝运行**（而不是后者覆盖前者）；
    可采纳记录的 data_fp 必须全局唯一（同一冻结 test 切片），否则拒绝；
    若清单声明/记录携带 ckpt_sha256，则同一权重摘要不得冒充另一个
    (variant, seed) 的独立测量。
    返回 (admitted[(variant, seed)] -> record, unified data_fp)。"""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise SystemExit(
            '[p5t stats] REFUSE: the bylen payload must be a JSON object '
            f'mapping record keys to records, got {type(raw).__name__}')
    if not isinstance(manifest, dict):
        raise SystemExit(
            '[p5t stats] REFUSE: the frozen manifest must be a JSON object, '
            f'got {type(manifest).__name__}')
    arms = list(manifest['arms'])
    seed_whitelist = {int(s) for s in manifest['seeds']}
    exp_code = L.code_stamp()  # eager-only; kernel mode is refused at load
    impl = _frozen_implementation(manifest)
    msha = manifest.get('manifest_sha256')
    want_train_fp = _frozen_train_fp(manifest)
    _want_nseq = int(((manifest.get('eval') or {}).get('n_seqs_clean')) or 0)
    admitted = {}
    excluded = []
    duplicates = []
    fps = set()
    train_fps = set()
    ckpt_owners = {}
    _any_train_fp = any(isinstance(_r, dict) and bool(_r.get('train_fp'))
                        for _r in (raw or {}).values())
    _require_train_fp = _train_fp_required(manifest) or bool(want_train_fp) or _any_train_fp
    # A disclosed parameter-matching contract makes the parameter count part of
    # the measurement identity.  Missing counts are named exclusions, so the
    # decision rule cannot read an unverifiable `param_matched=None` as "not
    # false" and confirm a comparison that was never capacity-checked.
    _require_params = _manifest_requires_params(manifest)
    _bind = manifest.get('binding') or {}
    _any_ckpt = any(isinstance(_r, dict) and bool(_r.get('ckpt_sha256'))
                    for _r in (raw or {}).values())
    _require_ckpt = (bool((_bind.get('require_ckpt_sha256')
                           if isinstance(_bind, dict) else False))
                     or _any_ckpt)
    for key, r in (raw or {}).items():
        if not isinstance(r, dict):
            continue
        why = None
        s_i = None
        if r.get('synthesized'):
            # A reconstructed aggregate is provenance, not an independent run.
            why = 'synthesized reconstruction is not an independent measurement'
        elif r.get('split') != 'test':
            # Named exclusion, not a silent `continue`.  The
            # docstring promises "named, never silent"; a smoke/validation
            # record sitting in bylen.json must be visible in the audit trail.
            why = f"split {r.get('split')!r} is not 'test' (a non-test record cannot enter the confirmation)"
        elif r.get('variant') not in arms:
            why = f"variant {r.get('variant')!r} is not one of the frozen arms"
        else:
            s_i = _record_int(r.get('seed'))
            if s_i is None or s_i not in seed_whitelist:
                why = f"seed {r.get('seed')!r} is outside the frozen whitelist {sorted(seed_whitelist)} (the frozen sample size is part of the pre-registration)"
            elif exp_code is not None and r.get('_code') != exp_code:
                why = f"_code {r.get('_code')!r} does not match the frozen code semantics {exp_code!r}"
            elif msha is not None and r.get('manifest_sha256') != msha:
                why = f"manifest_sha256 {str(r.get('manifest_sha256'))[:12]}… does not match the current manifest {msha[:12]}… (record from a different freeze revision)"
            elif _require_ckpt and not _valid_sha256(r.get('ckpt_sha256')):
                why = ('missing/malformed checkpoint fingerprint (ckpt_sha256) — '
                       'the weights cannot be tied to this measurement')
            elif r.get('ckpt_sha256') is not None and not _valid_sha256(r.get('ckpt_sha256')):
                why = (f"ckpt_sha256 {str(r.get('ckpt_sha256'))[:12]}… is not a "
                       'SHA-256 digest')
            elif not isinstance(r.get('data_fp'), str) or not r.get('data_fp'):
                why = 'missing test-slice fingerprint (data_fp) — None==None pairing is not admissible'
            elif _require_train_fp and (not isinstance(r.get('train_fp'), str) or not r.get('train_fp')):
                why = 'missing training-corpus fingerprint (train_fp) — the weights cannot be tied to a training set'
            elif want_train_fp is not None and r.get('train_fp') != want_train_fp:
                why = f"train_fp {r.get('train_fp')} does not match the frozen canonical training fingerprint {want_train_fp}"
            elif _want_nseq and _record_int(r.get('n_seq')) != _want_nseq:
                why = f"n_seq {r.get('n_seq')!r} != the frozen {_want_nseq} — a different row count is a different estimand"
            elif _want_nseq and _by_len_shape_reason(r, _want_nseq) is not None:
                why = _by_len_shape_reason(r, _want_nseq)
            elif _recipe_signature(manifest, r.get('variant')) is not None and not _recipe_matches_signature(r.get('recipe'), _recipe_signature(manifest, r.get('variant'))):
                why = 'recipe does not match the frozen training_recipe signature (including the per-arm mr)'
            elif _implementation_reason(r, impl) is not None:
                why = _implementation_reason(r, impl)
            elif _require_params and ((_record_int(r.get('params')) or 0) <= 0):
                why = ('missing/invalid parameter count (params) — the frozen '
                       'plan discloses parameter matching, so capacity is part '
                       'of the measurement identity and cannot be left unknown')
            elif not isinstance(r.get('by_len'), dict) or not r.get('by_len'):
                why = 'no by_len measurement'
            else:
                v = r['variant']
                if (v, s_i) in admitted:
                    duplicates.append(f'{v}::s{s_i} appears twice ({admitted[(v, s_i)].get("key", "?")} and {key})')
                else:
                    _ck = r.get('ckpt_sha256')
                    if _ck:
                        _ck_key = str(_ck).lower()
                        _owner = ckpt_owners.get(_ck_key)
                        if _owner is not None and _owner != (v, s_i):
                            duplicates.append(
                                f'checkpoint digest {_ck[:12]}… is reused by '
                                f'{_owner[0]}::s{_owner[1]} and {v}::s{s_i} — '
                                'one set of weights is not two independent '
                                'measurements')
                        else:
                            ckpt_owners[_ck_key] = (v, s_i)
                    # Canonicalise the seed in the admitted record: string
                    # '0' and integer 0 passed the whitelist, but downstream
                    # pairing must not treat them as two different seeds.
                    admitted[(v, s_i)] = dict(r, key=key, seed=s_i)
                    fps.add(r['data_fp'])
                    if r.get('train_fp'):
                        train_fps.add(r['train_fp'])
        if why:
            excluded.append(f'{key}: {why}')
    if excluded:
        print('[p5t stats] EXCLUDED from the confirmation analysis (named, never silent):\n  - ' + '\n  - '.join(excluded[:20]) + ('' if len(excluded) <= 20 else f'\n  … {len(excluded) - 20} more'))
    if duplicates:
        raise SystemExit('[p5t stats] REFUSE: ambiguous duplicate records for the same (variant, seed): ' + '; '.join(duplicates) + ' — the paired statistics would depend on which one wins; resolve the bylen file first')
    if len(fps) > 1:
        raise SystemExit(f'[p5t stats] REFUSE: admissible records span {len(fps)} different test-slice fingerprints ({sorted(fps)}) — the whole panel must be measured on ONE frozen test slice, not merely pairwise-equal fingerprints')
    if len(train_fps) > 1:
        raise SystemExit(f'[p5t stats] REFUSE: admissible records were trained on {len(train_fps)} different corpora ({sorted(train_fps)}) — all frozen arms must share the canonical training split')
    return admitted, (sorted(fps)[0] if fps else None)


def _doc_nll_matrix(rows, n_docs, tokens, *, unit='document'):
    """Validate and stack per-unit NLL vectors for the pooled-Δ CI.

    ``unit`` is 'document' (the frozen line-level contract) or 'article'
    (the independent corpus unit).  Returns ``(matrix, None)`` or
    ``(None, reason)``.  Admission checks the same totals, but these checks
    are repeated at the analysis boundary: the CI must be refused when the
    unit vectors do not reconstruct the cell's own totals, because a wrong
    interval is worse than an explicitly unavailable one.
    """
    if unit not in ('document', 'article'):
        return None, f'unknown unit {unit!r}'
    _suffix = 'article' if unit == 'article' else 'doc'
    _nll_key = f'nll_per_{_suffix}'
    _tok_key = f'n_tokens_per_{_suffix}'
    _ppl_key = f'ppl_per_{_suffix}'
    if not isinstance(tokens, (list, tuple)) or len(tokens) != n_docs:
        return None, f'{_tok_key} is missing or misaligned'
    _tok_f = []
    for t in tokens:
        if isinstance(t, (bool, np.bool_)):
            return None, f'{_tok_key} contains a boolean'
        try:
            tf = float(t)
        except (TypeError, ValueError, OverflowError):
            return None, f'{_tok_key} contains a non-numeric value'
        if not math.isfinite(tf) or tf <= 0.0:
            return None, f'{_tok_key} contains a non-positive entry'
        _tok_f.append(tf)
    out = []
    for r in rows:
        nll = r.get(_nll_key)
        if not isinstance(nll, list) or len(nll) != n_docs:
            return None, f'{_nll_key} missing or misaligned with the {unit} vector'
        vals = []
        for x in nll:
            if isinstance(x, (bool, np.bool_)):
                return None, f'{_nll_key} contains a boolean'
            try:
                v = float(x)
            except (TypeError, ValueError, OverflowError):
                return None, f'{_nll_key} contains a non-numeric value'
            if not math.isfinite(v):
                return None, f'{_nll_key} contains a non-finite value'
            vals.append(v)
        _nll_sum = r.get('nll_sum')
        if _nll_sum is not None:
            if isinstance(_nll_sum, (bool, np.bool_)):
                return None, 'nll_sum is a boolean'
            try:
                _ns = float(_nll_sum)
            except (TypeError, ValueError, OverflowError):
                return None, 'nll_sum is not numeric'
            if not math.isfinite(_ns) or abs(math.fsum(vals) - _ns) > 1e-5 * max(1.0, abs(_ns)):
                return None, f'{_nll_key} does not partition the recorded nll_sum'
        _ppl = r.get(_ppl_key)
        if isinstance(_ppl, list) and len(_ppl) == n_docs:
            for v, t, p in zip(vals, _tok_f, _ppl):
                if isinstance(p, (bool, np.bool_)):
                    return None, f'{_ppl_key} contains a boolean'
                try:
                    pf = float(p)
                except (TypeError, ValueError, OverflowError):
                    return None, f'{_ppl_key} contains a non-numeric value'
                expected = L._safe_exp_ppl(v / t)
                if (expected is None or not math.isfinite(expected)
                        or not math.isfinite(pf) or pf <= 0.0
                        or abs(pf - expected) > 1e-6 * max(1.0, abs(expected))):
                    return None, (f'{_ppl_key} is inconsistent with '
                                  f'{_nll_key} / {_tok_key}')
        out.append(vals)
    return out, None


def _analysis_choice_block(out, primary_deltas=None):
    """Attach an explicit multi-specification readout to one comparison.

    The frozen test is the per-seed exact sign-flip on the token-pooled Δ.  The
    report also shows unit-weighted and robust summaries; this block records
    whether those choices agree in sign and in zero-exclusion, so a reader
    cannot see only the most favourable aggregation.  It adds no p-value and
    no Holm member.
    """
    if not isinstance(out, dict):
        return None
    deltas = []
    for x in (primary_deltas if primary_deltas is not None else []):
        f = L._finite_scalar(x)
        if f is not None:
            deltas.append(f)
    if not deltas:
        return None
    n = len(deltas)
    # A finite per-seed panel near the float limit overflows raw np.mean /
    # np.std; the guarded helpers keep the point estimate representable (and
    # leave ordinary panels bit-identical to the historical NumPy path).
    mean = float(L._scaled_mean_or_raw(deltas))
    seed_ci = None
    if n > 1 and math.isfinite(mean):
        _sd = L._scaled_std_or_raw(deltas)
        if _sd is not None and math.isfinite(_sd):
            _sem = _sd / math.sqrt(n)
            _tc = L._t_critical(0.95, n - 1)
            if math.isfinite(_tc) and math.isfinite(_sem):
                _lo, _hi = mean - _tc * _sem, mean + _tc * _sem
                if math.isfinite(_lo) and math.isfinite(_hi):
                    seed_ci = [_lo, _hi]
    unit = out.get('unit') or out.get('granularity') or 'unit'

    def _ci_or_none(ci):
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            lo = L._finite_scalar(ci[0])
            hi = L._finite_scalar(ci[1])
            if lo is not None and hi is not None:
                return [lo, hi]
        return None

    def _finite(key):
        return L._finite_scalar(out.get(key))

    primary = {
        'spec': 'seed_mean_ppl_delta',
        'estimand': 'mean of per-seed token-pooled PPL differences (a-b)',
        'unit': 'seed', 'scale': 'ppl_delta', 'point': mean, 'ci95': seed_ci,
    }
    alts = []

    def _add(spec, estimand, row_unit, scale, point, ci=None, reason=None):
        row = {'spec': spec, 'estimand': estimand, 'unit': row_unit,
               'scale': scale, 'point': point, 'ci95': ci}
        if point is None:
            row['available'] = False
            row['reason'] = reason or 'point estimate unavailable'
        alts.append(row)

    _add('seed_median_ppl_delta',
         'median of per-seed token-pooled PPL differences (a-b)',
         'seed', 'ppl_delta', float(L._scaled_median_or_raw(deltas)))
    if n >= 5:
        _k = int(math.floor(0.2 * n))
        _trimmed = sorted(deltas)[_k:n - _k]
        _add('seed_trimmed_mean_ppl_delta',
             '20% trimmed mean of per-seed token-pooled PPL differences (a-b)',
             'seed', 'ppl_delta', float(L._scaled_mean_or_raw(_trimmed)))
    else:
        alts.append({'spec': 'seed_trimmed_mean_ppl_delta',
                     'estimand': '20% trimmed mean of per-seed PPL differences',
                     'unit': 'seed', 'scale': 'ppl_delta', 'available': False,
                     'point': None, 'ci95': None,
                     'reason': 'requires at least 5 seeds'})
    _add('token_pooled_ppl_delta',
         'token-pooled PPL difference with unit-cluster interval (a-b)',
         unit, 'ppl_delta', _finite('pooled_delta_ppl'),
         _ci_or_none(out.get('pooled_delta_ppl_ci95')),
         out.get('pooled_ci_unavailable_reason'))
    _add('equal_unit_mean_ppl_delta',
         'equal-weighted mean of per-unit PPL differences (a-b)',
         unit, 'ppl_delta', _finite('per_doc_delta_mean'),
         _ci_or_none(out.get('per_doc_delta_ppl_ci95')))
    _add('unit_median_ppl_delta',
         'median of per-unit PPL differences (a-b)',
         unit, 'ppl_delta', _finite('per_doc_delta_median'))
    _add('unit_mean_log_ratio',
         'mean per-unit log(PPL_a / PPL_b)',
         unit, 'log_ratio', _finite('per_doc_log_ratio_mean'),
         _ci_or_none(out.get('per_doc_log_ratio_ci95')))
    _neg = _finite('doc_delta_negative')
    _pos = _finite('doc_delta_positive')
    _n_units = _finite('n_units')
    if (_neg is not None and _pos is not None and _n_units is not None
            and _n_units > 0 and _neg + _pos <= _n_units):
        _add('unit_win_balance_oriented',
             'fraction of units with a worse PPL minus fraction with a lower '
             'PPL (negative = arm a wins more often)',
             unit, 'win_balance', (_pos - _neg) / _n_units)
    else:
        alts.append({'spec': 'unit_win_balance_oriented',
                     'estimand': 'signed fraction of units where arm a has lower PPL',
                     'unit': unit, 'scale': 'win_balance', 'available': False,
                     'point': None, 'ci95': None,
                     'reason': 'win-balance summary unavailable'})
    return L.analysis_choice_concordance(primary, alts)
