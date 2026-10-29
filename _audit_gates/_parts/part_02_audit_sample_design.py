

def audit_sample_design(cache_dir=None, *, raw_dir=None, max_raw_articles=None):
    """Audit whether the cached WikiText slices are probability samples.

    The loaders have always selected a **source-order prefix** and capped it by
    tokens.  That is reproducible and leakage-safe, but it is not a random
    sample: later articles can be systematically absent, and a prefix of raw
    lines is not a prefix of the article population in any probabilistic
    sense.  This zero-GPU audit makes the distinction explicit:

      * it records the declared ``sampling`` block (new caches) or labels the
        slice ``prefix_inferred_legacy`` when the field is missing;
      * it refuses any block that claims representativeness/population validity
        without a validated ``document_stratified_plan`` (seed, without-
        replacement selection, stratum coverage);
      * when raw train/validation dumps are available, it regroups served line
        ids into whole articles, reports row/article coverage, the
        served-vs-source article-length KS distance and coverage by source
        size quartile; missing raw data is ``not_verifiable_locally``, never
        "clean";
      * it never reads ``wiki.test.raw``.  A test slice's coverage is reported
        from its build-time metadata only, because re-reading the held-out
        split would consume the one-shot confirmation access.

    Prefix slices are reported as ``warnings`` with
    ``representativeness_claim_allowed=False``; they are not silently relabelled
    as probability samples.  An invalid optimistic claim is a hard failure.
    """
    L = _import_exp_lib()
    import glob
    if cache_dir in (None, ''):
        cache_dir = os.environ.get('WT103_CACHE_DIR') or './wt103_cache'
    if max_raw_articles is not None:
        max_raw_articles = _audit_positive_int('max_raw_articles', max_raw_articles)
    if raw_dir in (None, ''):
        raw_dir = os.environ.get('WT103_RAW_TXT') or './wt103_raw'
    out = {'kind': 'sample_design_audit', 'status': 'ok', 'caches': {},
           'warnings': [], 'problems': [],
           'note': ('A source-order prefix is reproducible but not a '
                    'probability sample.  This audit never reads '
                    'wiki.test.raw; test coverage is metadata-only.')}
    if not os.path.isdir(cache_dir):
        out['status'] = 'not_verifiable_locally'
        out['note'] = f'{cache_dir!r} does not exist — no cache to audit'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    _metas = sorted(glob.glob(os.path.join(cache_dir, 'meta_*.json')))
    _metas += sorted(glob.glob(os.path.join(cache_dir, 'test_meta_*.json')))
    if not _metas:
        out['status'] = 'not_verifiable_locally'
        out['note'] = f'{cache_dir!r} contains no cache metadata'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    def _warn(msg):
        out['warnings'].append(msg)
        if out['status'] == 'ok':
            out['status'] = 'warnings'

    _raw_rows_cache = {}

    def _raw_rows(split):
        if split in _raw_rows_cache:
            return _raw_rows_cache[split]
        fname = {'train': 'wiki.train.raw', 'validation': 'wiki.valid.raw'}.get(split)
        if fname is None:
            return None
        path = os.path.join(raw_dir, fname)
        if not os.path.exists(path):
            _raw_rows_cache[split] = None
            return None
        try:
            rows = L._read_raw_rows(path)
        except Exception as e:
            _raw_rows_cache[split] = None
            _raw_rows_cache[split + '_error'] = f'{type(e).__name__}: {e}'
            return None
        _raw_rows_cache[split] = rows
        return rows

    def _raw_coverage(split, served_ids, sidecar):
        rec = {'status': 'ok'}
        if split == 'test':
            rec.update({
                'status': 'not_applicable',
                'reason': ('wiki.test.raw is never read: re-reading the frozen '
                           'held-out split would consume the one-shot access; '
                           'test coverage is reported from build metadata'),
            })
            return rec
        rows = _raw_rows(split)
        if rows is None:
            _err = _raw_rows_cache.get(split + '_error')
            rec.update({
                'status': 'not_verifiable_locally',
                'reason': (f'raw {split} dump is unavailable under {raw_dir!r}'
                           + (f' ({_err})' if _err else '')
                           + '; served-vs-source article coverage cannot be measured'),
            })
            return rec
        if served_ids is None:
            rec.update({
                'status': 'not_verifiable_locally',
                'reason': f'{sidecar} is missing; served line ids cannot be mapped to raw articles',
            })
            return rec
        try:
            groups = L.wikitext_article_groups(rows)
        except Exception as e:
            rec.update({
                'status': 'not_verifiable_locally',
                'reason': f'raw {split} dump cannot be segmented ({type(e).__name__}: {e})',
            })
            return rec
        if not groups.get('valid'):
            rec.update({
                'status': 'not_verifiable_locally',
                'reason': ('raw rows are not segmented by top-level headings; '
                           'article coverage is undefined (' + '; '.join(groups.get('problems') or []) + ')'),
            })
            return rec
        _groups = groups['groups']
        if max_raw_articles is not None:
            _groups = _groups[:int(max_raw_articles)]
        _sizes_all = []
        _served_flags = []
        _served_line_hits = 0
        for g in _groups:
            _hit = False
            for i in g:
                _row = rows[int(i)]
                _h = int(L._doc_id_u64(L._doc_exact_fp(_row)))
                if _h in served_ids:
                    _hit = True
                    _served_line_hits += 1
            _sizes_all.append(int(len(g)))
            _served_flags.append(bool(_hit))
        _raw_nonempty = int(sum(len(g) for g in _groups))
        _a = np.asarray(_sizes_all, dtype=np.int64)
        _flags = np.asarray(_served_flags, dtype=bool)
        _b = _a[_flags]
        _n_all = int(_a.size)
        _n_served = int(_b.size)
        _ks = _sample_ks(_a, _b)
        _ks_limit = (min(1.0, 1.36 * math.sqrt(
                        (float(_n_all) + float(_n_served))
                        / (float(_n_all) * float(_n_served))))
                     if _n_served and _n_all else None)
        rec.update({
            'raw_nonempty_rows': int(groups.get('n_nonempty_rows', 0)),
            'raw_articles': _n_all,
            'served_article_size_units': _n_served,
            'article_coverage_fraction': (float(_n_served) / float(_n_all)
                                          if _n_all else 0.0),
            'served_line_hits': int(_served_line_hits),
            'line_coverage_fraction': (float(_served_line_hits)
                                       / float(_raw_nonempty)
                                       if _raw_nonempty else 0.0),
            'article_size_ks': _ks,
            'article_size_ks_limit': _ks_limit,
            'article_size_shift_flag': bool(
                _ks is not None and _ks_limit is not None and _ks > _ks_limit),
        })
        if max_raw_articles is not None:
            rec['article_subset'] = f'first {int(max_raw_articles)} raw articles'
        # Coverage across source-size quartiles: a probability sample should
        # reach every stratum; a source prefix can miss the tail.
        try:
            _edges = np.quantile(_a, [0.25, 0.5, 0.75])
            _bin = np.digitize(_a, _edges, right=False)
            _qs = []
            for q in range(4):
                _mask = _bin == q
                _tot = int(_mask.sum())
                _srv = int((_mask & _flags).sum())
                _qs.append({'quartile': q + 1, 'articles': _tot,
                            'served_articles': _srv,
                            'coverage_fraction': (float(_srv) / float(_tot)
                                                  if _tot else None)})
            rec['by_source_size_quartile'] = _qs
            _uncovered = [q['quartile'] for q in _qs
                          if q['articles'] and q['served_articles'] == 0]
            rec['uncovered_quartiles'] = _uncovered
            _rw = []
            if _uncovered:
                rec['status'] = 'warnings'
                _rw.append(f'source-size quartile(s) {_uncovered} have no '
                           'served articles (prefix coverage is not whole-population)')
            if rec.get('article_size_shift_flag'):
                rec['status'] = 'warnings'
                _rw.append(f'served/articles article-length KS={_ks:.3f} '
                           f'exceeds the finite-sample limit {_ks_limit:.3f}; the '
                           'served length distribution is not representative')
            if _rw:
                rec['warnings'] = _rw
        except Exception as e:
            rec['quartile_error'] = f'{type(e).__name__}: {e}'
        return rec

    _n_blocks = 0
    for _mp in _metas:
        _meta = _load(_mp)
        if not isinstance(_meta, dict):
            _warn(f'{os.path.basename(_mp)} is not a readable metadata object')
            continue
        _is_test = os.path.basename(_mp).startswith('test_meta_')
        _tag = (os.path.basename(_mp)[len('test_meta_'):-len('.json')]
                if _is_test else
                os.path.basename(_mp)[len('meta_'):-len('.json')])
        _cachedir = os.path.abspath(cache_dir)
        _crec = out['caches'].setdefault(_cachedir, {'meta_files': []})
        _crec['meta_files'].append(os.path.basename(_mp))
        _sampling = _meta.get('sampling')
        if isinstance(_sampling, dict) and any(
                k in _sampling for k in ('train', 'validation', 'test')):
            _blocks = [(k, v) for k, v in _sampling.items()
                       if k in ('train', 'validation', 'test') and isinstance(v, dict)]
        elif isinstance(_sampling, dict):
            _blocks = [('test' if _is_test else 'train', _sampling)]
        elif _is_test:
            _blocks = [('test', {})]
        else:
            _blocks = [('train', {}), ('validation', {})]
        if not _is_test and not any(k == 'validation' for k, _ in _blocks):
            _blocks.append(('validation', {}))
        for _split, _blk in _blocks:
            _n_blocks += 1
            _srec = _crec.setdefault(_split, {'problems': [], 'warnings': []})
            _design = _blk.get('design') or 'prefix_inferred_legacy'
            _srec['design'] = _design
            _srec['declared_sampling_fields'] = bool(_blk)
            _srec.update({
                'source_rows': _blk.get('source_rows'),
                'rows_after_dedup': _blk.get('rows_after_dedup'),
                'rows_served': _blk.get('rows_served'),
                'tokens_served': _blk.get('tokens_served'),
                'token_cap': _blk.get('token_cap'),
                'cap_reached': _blk.get('cap_reached'),
            })
            _claim = _blk.get('representativeness_claim_allowed')
            _pop_claim = _blk.get('population_claim_allowed')
            _srec['declared_claims'] = {
                'representativeness_claim_allowed': _claim,
                'population_claim_allowed': _pop_claim,
            }
            _prefix = _design in ('source_order_prefix', 'prefix',
                                  'ordered_prefix', 'prefix_inferred_legacy')
            if _prefix:
                _srec['sampling_class'] = 'non_probability_prefix'
                _srec['representativeness_claim_allowed'] = False
                _srec['population_claim_allowed'] = False
                if _claim is True or _pop_claim is True:
                    _srec['problems'].append(
                        'optimistic claim: metadata claims representativeness/'
                        'population validity for a source-order prefix; that '
                        'claim is not supported')
                try:
                    _sr0 = int(_blk.get('source_rows'))
                    _rs0 = int(_blk.get('rows_served'))
                except (TypeError, ValueError):
                    _sr0 = _rs0 = None
                if _sr0 is not None and _rs0 is not None and _rs0 >= _sr0:
                    _srec['warnings'].append(
                        'source-order selection covers the full cached source '
                        'rows but is still not a probability sample; it is a '
                        'census of this source snapshot')
                else:
                    _srec['warnings'].append(
                        'source-order prefix, not a probability sample; later '
                        'articles are unserved by construction')
            elif _design == 'stratified_document':
                _plan = _blk.get('plan')
                _pp = L.sampling_plan_problems(_plan) if _plan is not None else [
                    'metadata declares a stratified design but records no plan']
                _srec['plan_fp'] = (_plan.get('plan_fp')
                                    if isinstance(_plan, dict) else None)
                if _pp:
                    _srec['problems'].extend(_pp)
                else:
                    _srec['sampling_class'] = 'probability_stratified_document'
                    _n_sel = int(_plan.get('n_select') or 0)
                    _all_repr = bool(_plan.get('all_strata_represented'))
                    _srec['representativeness_claim_allowed'] = bool(
                        _claim is True and _all_repr)
                    if _claim is True and not _all_repr:
                        _srec['problems'].append(
                            'claim=true but not every stratum is represented')
                    if _pop_claim is True and _claim is not True:
                        # A population claim is strictly stronger than a
                        # representativeness claim; accepting it while the
                        # weaker claim is false would let one contradictory
                        # metadata block authorise an extrapolation the
                        # sampling design does not support.
                        _srec['problems'].append(
                            'population claim requires '
                            'representativeness_claim_allowed=true; the '
                            'metadata contradicts itself')
                    if _pop_claim is True and _n_sel < 64:
                        _srec['problems'].append(
                            f'population claim with only {_n_sel} independent '
                            f'units (<64-unit floor)')
                    _srec['population_claim_allowed'] = bool(
                        _pop_claim is True and _claim is True
                        and _n_sel >= 64 and _all_repr and not _pp)
                    if _claim is not True:
                        _srec['warnings'].append(
                            'plan is a probability sample but the metadata does '
                            'not claim representativeness; claim stays false')
                if isinstance(_plan, dict) and _blk.get('source_fp') not in (None, _plan.get('source_fp')):
                    _srec['problems'].append(
                        'sampling source_fp disagrees with the plan source_fp')
            else:
                _srec['sampling_class'] = 'unknown'
                _srec['representativeness_claim_allowed'] = False
                _srec['population_claim_allowed'] = False
                _srec['problems'].append(f'unknown sampling design {_design!r}')
            # Token-budget honesty: a cap that was not reached means the source
            # ran dry and the advertised training/evaluation length is not the
            # length actually served.
            try:
                _ts = _blk.get('tokens_served')
                _tc = _blk.get('token_cap')
                if (_ts is not None and _tc is not None
                        and int(_ts) < int(_tc)
                        and _blk.get('cap_reached') is not True):
                    _srec['warnings'].append(
                        f'served {int(_ts)} tokens < cap {int(_tc)} and '
                        'cap_reached is not true: the source ran dry')
            except (TypeError, ValueError):
                _srec['problems'].append('tokens_served/token_cap are not integers')
            # Row coverage from metadata (cheap, works without raw dumps).
            try:
                _sr = int(_blk.get('source_rows'))
                _rs = int(_blk.get('rows_served'))
                if _sr < 0 or _rs < 0:
                    _srec['problems'].append(
                        'source_rows/rows_served must be non-negative')
                    _srec['row_coverage_fraction'] = None
                elif _rs > _sr:
                    # metadata that reports more served rows than the source
                    # contains is contradictory; a coverage > 1 must not be
                    # quoted as evidence of breadth.
                    _srec['problems'].append(
                        f'rows_served={_rs} exceeds source_rows={_sr}')
                    _srec['row_coverage_fraction'] = None
                elif _sr > 0:
                    _srec['row_coverage_fraction'] = float(_rs) / float(_sr)
                    _srec['row_coverage_incomplete'] = bool(_rs < _sr)
            except (TypeError, ValueError):
                _srec['row_coverage_fraction'] = None
            # Article-level coverage needs the raw source; test raw is never read.
            _sidecar = None
            if _split == 'train':
                _sidecar = os.path.join(cache_dir, f'train_doc_ids_{_tag}.npy')
            elif _split == 'validation':
                _sidecar = os.path.join(cache_dir, f'val_doc_ids_{_tag}.npy')
            elif _split == 'test':
                _sidecar = os.path.join(cache_dir, f'test_doc_ids_{_tag}.npy')
            _served_ids = None
            if _sidecar and os.path.exists(_sidecar):
                try:
                    _served_ids = set(int(x) for x in np.load(_sidecar).tolist())
                except Exception as e:
                    _srec['warnings'].append(
                        f'{os.path.basename(_sidecar)} unreadable ({type(e).__name__}: {e})')
            _raw = _raw_coverage(_split, _served_ids, _sidecar)
            _srec['raw_coverage'] = _raw
            if _raw.get('status') == 'not_verifiable_locally':
                _srec['warnings'].append(
                    _raw.get('reason') or 'raw coverage is not verifiable')
            for _rw in _raw.get('warnings') or []:
                _srec['warnings'].append(_rw)
            if _srec['problems']:
                _srec['status'] = 'fail'
            elif _srec['warnings']:
                _srec['status'] = 'warnings'
            else:
                _srec['status'] = 'ok'
            for _w in _srec['warnings']:
                _tagged_w = f'{_split}: {_w}'
                if _tagged_w not in out['warnings']:
                    out['warnings'].append(_tagged_w)
            for _p in _srec['problems']:
                _tagged_p = f'{_split}: {_p}'
                if _tagged_p not in out['problems']:
                    out['problems'].append(_tagged_p)
    if _n_blocks == 0:
        out['status'] = 'not_verifiable_locally'
        out['note'] = 'cache metadata contains no train/validation/test sampling blocks'
    elif out['problems']:
        out['status'] = 'fail'
    elif out['warnings']:
        out['status'] = 'warnings'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
# 2) causality：前缀不变性 / 标签移位 / 解码一致性（纯 CPU 微型模型）
# ---------------------------------------------------------------------------

def audit_causality():
    """自回归正确性单元审计（纯 CPU、微型形状、几秒完成——不是实验）。

    核验点（每项都是断言，失败即退出码非零）：
      a. 标签移位：eval PPL 的 cross_entropy 输入=ids[:, :-1] 目标=ids[:, 1:]
         ——用微型随机模型复算一行，确认「预测下一 token」；
      b. 前缀不变性（仓库 §1 的硬约束）：f(x)[:T'] == f(x[:T']) 对全部
         冻结相关变体（csa_fixed_rope / full_rope / randidx / allblocks）在
         微型形状上成立——稀疏臂的块读取门限若泄漏未来会在此暴露；
      c. 池化与选择器因果性：block_readable 的严格不等式（pos > last_tok）
         被全部读取路径遵守（经 b 的前缀不变性间接覆盖，另做直查）；
      d. 增量解码 vs 整段前向：对 RoPE 臂逐 token 前缀的 logits 末列与
         整段前向对应列一致（eval 的 by-length 评分的前提）。"""
    try:
        import torch
        import numpy as np
    except ImportError as e:
        print(f'[audit causality] SKIPPED: torch not importable ({e}); install CPU torch to run this unit audit')
        return {'status': 'skipped_no_torch'}
    L = _import_exp_lib()
    import explib.rope as _V7  # noqa: F401 — importing installs the *_rope variant patches (make_layer_cfgs_v7 + SmallGPTRoPE factory)
    import copy as _copy
    L.set_seed(0)
    dev = torch.device('cpu')
    V, T, vocab = 6, 128, 256
    variants = ['full_rope', 'csa_fixed_rope', 'csa_fixed_rope_randidx', 'csa_fixed_rope_allblocks']
    ids = torch.randint(0, vocab, (V, T + 1), dtype=torch.long)
    fails = []
    for v in variants:
        cfgs = L.make_layer_cfgs(4, v)
        model = L.SmallGPT(vocab, 64, 4, 4, 16, 128, _copy.deepcopy(cfgs)).to(dev)
        model.eval()
        with torch.inference_mode():
            full = model(ids[:, :-1])
            # a) label shift: model(x)[t] predicts x[t+1]; check column t against
            #    a re-run on the truncated input (also = prefix invariance b)
            for cut in (32, 64, 96):
                part = model(ids[:, :cut])
                if not torch.allclose(part, full[:, :cut], atol=1e-4, rtol=1e-4):
                    fails.append(f'{v}: prefix invariance broken at T\'={cut} (future leakage or T-dependent tail-block read)')
                    break
            # d) incremental decode vs full forward (logits tail column)
            inc = []
            ctx = ids[:, :1]
            for t in range(1, min(T, 48) + 1):
                out = model(ids[:, :t])
                inc.append(out[:, -1])
            inc = torch.stack(inc, dim=1)
            if not torch.allclose(inc, full[:, :inc.shape[1]], atol=1e-4, rtol=1e-4):
                fails.append(f'{v}: incremental decode disagrees with full forward (autoregressive inconsistency)')
        del model
    # c) strict readability gate, direct check (vectorised contract:
    #    READABLE(pos_i, block_j) == pos_i > last_tok[j], STRICTLY)
    _pos = torch.tensor([0, 1, 5, 7])
    _last = torch.tensor([0, 1, 5, 7])
    _m = L.block_readable(_pos, _last)
    _want = _pos[:, None] > _last[None, :]
    if not torch.equal(_m, _want):
        fails.append('block_readable violates the strict inequality pos > last_tok (future leakage into the tail block)')
    if bool(_m[0, 0]) or bool(_m[1, 1]):
        fails.append('block_readable must be strict: pos == last_tok must NOT be readable')
    # c2) the segmentation table itself must be prefix invariant for both the
    # no-cap and capped dynamic geometries.  A model-level prefix test can miss
    # this when the differing tail block happens to be unreadable.
    _rng = np.random.default_rng(0)
    for _mb in (None, 8, 32):
        _cuts = _rng.random(96) < 0.22
        _rep = L.segmentation_prefix_invariance(_cuts, 97, 2, _mb)
        if not _rep.get('ok'):
            fails.append(f'segmenter cut table is not prefix invariant (max_block={_mb}): '
                         f"{_rep.get('mismatches', [])[:1]}")
    if fails:
        print('[audit causality] FAIL:\n  - ' + '\n  - '.join(fails))
        out = {'status': 'fail', 'failures': fails}
    else:
        print(f'[audit causality] PASS: label shift + prefix invariance + strict gate + decode consistency over {variants}')
        out = {'status': 'pass', 'variants': variants}
    return out


# ---------------------------------------------------------------------------
# 3) attribution：动态/固定分块的块数、压缩率、计算量对齐读数
# ---------------------------------------------------------------------------

def audit_attribution(outdirs=None):
    """动态分块归因审计（零 GPU，读 results_*/summary.json + ckpt 元数据）。

    审计口径：固定块与动态块是否匹配实际块数、压缩率、可读摘要数量和
    计算量。本审计从落盘训练记录的 by_len / _stats / 参数量字段重算
    每臂的：参数量、（若启用块统计）块长分布与边界对齐读数，并输出
    「臂间块数是否对齐」的对照表。检查 `attn._stats`（enable_block_stats
    收集的块 id bincount，design_notes §14 同口径）是否在场。"""
    out = {'status': 'ok', 'panels': {}}
    L = _import_exp_lib()
    import numpy as np
    dirs = outdirs or list(_iter_result_dirs())
    for d in dirs:
        summ = _load(_audit_artifact(os.path.join(d, 'summary.json')))
        if not summ:
            continue
        panel = {}
        for key, r in summ.items():
            if not isinstance(r, dict):
                continue
            v = r.get('variant')
            if not isinstance(v, str) or not v:
                # The arm id is a dict key; a malformed/non-string id is not
                # an arm and must be excluded instead of crashing the audit.
                continue
            ent = panel.setdefault(v, {'n_records': 0, 'with_block_stats': 0, 'params': r.get('params'),
                                       'blocks_mean': [], 'block_len_mean': [], 'block_len_p95': []})
            ent['n_records'] += 1
            _layers = r.get('stats') if isinstance(r.get('stats'), dict) else {}
            _found = False
            for _lk, _lv in _layers.items():
                if not isinstance(_lv, dict) or 'blocks' not in _lv:
                    continue
                _found = True
                for _src, _dst in (('blocks', 'blocks_mean'), ('len_mean', 'block_len_mean'), ('len_p95', 'block_len_p95')):
                    _x = _lv.get(_src)
                    if isinstance(_x, (int, float)) and math.isfinite(float(_x)):
                        ent[_dst].append(float(_x))
            if _found or r.get('block_stats') or r.get('by_len_block_stats') or (isinstance(r.get('mech'), dict) and r['mech'].get('cuts')):
                ent['with_block_stats'] += 1
        if panel:
            missing = {v: e for v, e in panel.items() if e['with_block_stats'] == 0}
            # Turn the presence check into a block-count / block-length table
            # a fixed-vs-dynamic contrast that also changes
            # the block count or mean block length is not a clean blocking
            # contrast, and must be reported as such.
            for v, e in panel.items():
                for _src, _dst in (('blocks_mean', 'blocks_mean'), ('block_len_mean', 'block_len_mean'), ('block_len_p95', 'block_len_p95')):
                    _vals = e.pop(_src, [])
                    e[_dst] = (float(np.mean(_vals)) if _vals else None)
            _pairs = {'csa_fixed': 'csa_dynamic', 'hybrid_fixed': 'hybrid_dynamic'}
            for a, b in _pairs.items():
                if a in panel and b in panel:
                    _ba, _bb = panel[a].get('blocks_mean'), panel[b].get('blocks_mean')
                    _la, _lb = panel[a].get('block_len_mean'), panel[b].get('block_len_mean')
                    rec = {'a': a, 'b': b, 'blocks_a': _ba, 'blocks_b': _bb, 'block_len_a': _la, 'block_len_b': _lb}
                    if _ba and _bb:
                        _r = max(_ba, _bb) / min(_ba, _bb)
                        rec['block_count_ratio'] = float(_r)
                        rec['block_count_matched'] = bool(_r <= 1.10)
                    else:
                        rec['block_count_matched'] = None
                    if _la and _lb:
                        rec['block_len_ratio'] = float(max(_la, _lb) / min(_la, _lb))
                    out.setdefault('blocking_contrasts', {})[f'{a} vs {b}'] = rec
            out['panels'][d] = panel
            if missing:
                out['panels'][d]['note'] = f'no block-stat fields found for {sorted(missing)} — run evals with exp_lib.enable_block_stats(model, True) (or the kernel twin, same collection) before this audit can compare block counts/compression across arms'
    if not out['panels']:
        out['status'] = 'not_verifiable_locally'
        out['note'] = 'no results_*/summary.json on this machine — the attribution readout needs the experiment artifacts; the collection hooks are already wired (enable_block_stats / attn._stats per design_notes §14)'
    print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
    return out


# ---------------------------------------------------------------------------
# 4) boundary：边界对齐 vs 匹配块长分布的随机基线
# ---------------------------------------------------------------------------

def audit_boundary(outdirs=None):
    """边界意义审计（零 GPU）：对齐读数必须超过「匹配块长分布的随机切割」。

    审计口径：与标点/词边界的对齐是否超过匹配块长分布的随机基线——否则
    「语义边界」是块长分布的伪影。exp_lib 已实现该基线
    （`_random_cut_precision`，两种口径见 design_notes §10）；本审计把
    「对齐精度 − 随机基线」的差值与比值汇总成显式判读。"""
    import numpy as np
    out = {'status': 'ok', 'panels': {}, 'problems': []}
    L = _import_exp_lib()
    dirs = outdirs or list(_iter_result_dirs())

    def _num(value):
        # Alignment readings are measurements: booleans, strings and
        # non-finite values cannot support a "beats the random baseline" claim.
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            return None
        f = float(value)
        return f if math.isfinite(f) else None

    for d in dirs:
        summ = _load(_audit_artifact(os.path.join(d, 'summary.json')))
        if not summ:
            continue
        found = {}
        for key, r in summ.items():
            if not isinstance(r, dict):
                continue
            for field in ('boundary_alignment', 'bnd_alignment', 'mech'):
                b = r.get(field)
                if isinstance(b, dict) and ('precision' in b or 'random_precision' in b or 'excess' in b):
                    found[key] = b
                    break
        if found:
            agg = []
            for key, b in list(found.items())[:200]:
                p_raw, rp_raw = b.get('precision'), b.get('random_precision')
                if p_raw is None and rp_raw is None:
                    continue
                p, rp = _num(p_raw), _num(rp_raw)
                if p is None or rp is None:
                    out['problems'].append(
                        f'{d}/{key}: precision/random_precision must be finite real numbers '
                        f'(got {p_raw!r} / {rp_raw!r}); the alignment reading is unverifiable')
                    continue
                if not (0.0 <= p <= 1.0 and 0.0 <= rp <= 1.0):
                    # Precision and its matched random baseline are fractions;
                    # p=1.4 must not be reported as "exceeds the baseline".
                    out['problems'].append(
                        f'{d}/{key}: precision/random_precision outside [0,1] '
                        f'(got {p!r} / {rp!r}); the alignment reading is invalid')
                    continue
                agg.append({'key': key, 'precision': p, 'random_baseline': rp, 'excess': p - rp, 'ratio': (p / rp) if rp else None, 'verdict': 'exceeds random baseline' if p > rp else ('at baseline (block-length artifact)' if abs(p - rp) < 1e-9 else 'below baseline')})
            out['panels'][d] = agg
    if out['problems']:
        out['status'] = 'fail'
    if not out['panels'] or not any(v for v in out['panels'].values()):
        out['status'] = 'fail' if out['problems'] else 'not_verifiable_locally'
        out['note'] = 'no usable boundary-alignment fields in the local results; fuse_analysis.py recomputes per-layer alignment (tol=1) from summary.json and exp_lib._random_cut_precision provides the matched-length random baseline — run them on the experiment machine'
    print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
    return out


# ---------------------------------------------------------------------------
# 5) length：长度外推评分口径一致性
# ---------------------------------------------------------------------------

def audit_length(bylen_path='results_lm_test/bylen.json'):
    """长度外推一致性（零 GPU）：同长度跨臂评分同一组目标 token。

    核验点：同一 L 下全部记录的 n_tok 与 eval_span 必须一致（n_tok =
    n_seq × (L−1)，即同一组前缀目标 token 被评分）；truncated 必须为假；
    逐文档行数 n_seq 跨臂一致。若某臂对不同目标集评分，跨臂比较无效。"""
    raw = _load(_audit_artifact(bylen_path))
    if raw is None:
        out = {'status': 'not_verifiable_locally', 'note': f'{bylen_path} absent — P5T has not run; the same checks apply to results_v10_mech/summary.json by_len panels'}
        print(json.dumps(out, ensure_ascii=False))
        return out
    if not isinstance(raw, dict):
        out = {'status': 'fail', 'problems': [f'{bylen_path}: top-level JSON is {type(raw).__name__}, not an object — the by-length panel is unverifiable']}
        print(json.dumps(out, ensure_ascii=False))
        return out
    if not raw:
        # An existing but EMPTY artifact proves nothing; an empty dict must
        # not be reported as a passed length audit.
        out = {'status': 'not_verifiable_locally', 'note': f'{bylen_path} exists but contains no records — the length/target-set contract was not verified'}
        print(json.dumps(out, ensure_ascii=False))
        return out
    L = _import_exp_lib()
    import numpy as np

    def _exact_int(value):
        if value is None or isinstance(value, (bool, np.bool_)):
            return None
        if isinstance(value, numbers.Integral):
            return int(value)
        try:
            f = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return int(f) if math.isfinite(f) and f == math.floor(f) else None

    def _exact_value(value):
        # `_exact_int` stays permissive for by_len *keys* (JSON object keys
        # are strings); a count stored as a numeric string is not a
        # measurement and must not certify the target set.
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            return None
        f = float(value)
        return int(f) if math.isfinite(f) and f == math.floor(f) else None

    def _hashable(value):
        # Artifact cells are JSON, but a direct caller can supply a list/dict
        # where a count/flag belongs.  Such a value is unverifiable and must
        # become a recorded problem, not a TypeError from a set insert.
        try:
            hash(value)
        except TypeError:
            return ('__unhashable__', type(value).__name__, repr(value))
        return value

    by_len = {}
    problems = []
    for key, r in raw.items():
        if not isinstance(r, dict):
            problems.append(f'{key}: record is not a JSON object')
            continue
        _by_len = r.get('by_len')
        if _by_len is None:
            problems.append(f'{key}: record has no `by_len` panel — the target set is unverifiable')
            continue
        if not isinstance(_by_len, dict):
            problems.append(f'{key}: by_len is {type(_by_len).__name__}, not a JSON object')
            continue
        if not _by_len:
            problems.append(f'{key}: by_len is empty — no length cell can be verified')
            continue
        for k, c in _by_len.items():
            if isinstance(k, (bool, np.bool_)):
                problems.append(f'{key}: by_len key {k!r} is not an integer length')
                continue
            Ln = _exact_int(k)
            if Ln is None:
                problems.append(f'{key}: by_len key {k!r} is not an integer length')
                continue
            if not isinstance(c, dict):
                problems.append(f'{key}: L{Ln} cell is not a JSON object')
                continue
            e = by_len.setdefault(Ln, {'n_tok': set(), 'eval_span': set(), 'truncated': set(), 'n_seq': set()})
            e['n_tok'].add(_exact_value(c.get('n_tok')))
            e['eval_span'].add(_exact_value(c.get('eval_span')))
            e['truncated'].add(_hashable(c.get('truncated')))
            for _field in ('n_tok', 'eval_span'):
                if _exact_value(c.get(_field)) is None:
                    problems.append(
                        f'{key}: L{Ln} cell has no exact integer `{_field}` '
                        f'({c.get(_field)!r}) — the scored target set is unverifiable')
            _nseq_cell = None
            _seq_explicit = None
            if c.get('n_seq') is not None:
                _seq_explicit = _exact_value(c.get('n_seq'))
                if _seq_explicit is None:
                    problems.append(
                        f'{key}: L{Ln} cell has a non-integer `n_seq` '
                        f'({c.get("n_seq")!r}) — the per-document row count is unverifiable')
            if isinstance(c.get('ppl_per_seq'), list):
                _nseq_cell = len(c['ppl_per_seq'])
                e['n_seq'].add(_nseq_cell)
                if _seq_explicit is not None and _seq_explicit != _nseq_cell:
                    # The two row-count fields must describe the same rows;
                    # silently preferring the list length would hide an
                    # edited/truncated per-document vector.
                    problems.append(
                        f'{key}: L{Ln} n_seq={_seq_explicit} disagrees with '
                        f'len(ppl_per_seq)={_nseq_cell} — the per-document '
                        'accounting is not the same row set')
                    e['n_seq'].add(None)
            elif c.get('n_seq') is not None:
                _nseq_cell = _seq_explicit
                e['n_seq'].add(_nseq_cell)
            else:
                # Keep the absence in the same set as the observed counts so
                # one record with n_seq cannot mask another record without it.
                e['n_seq'].add(None)
            # The complete-cell identity is n_tok == n_seq * (Ln - 1).  Equal
            # (but equally wrong) counts across arms are not a valid target
            # set, so verify the formula whenever the cell claims to be a
            # full-length measurement.
            _nt = _exact_value(c.get('n_tok'))
            _ns = _exact_value(_nseq_cell)
            _trunc = c.get('truncated')
            if (isinstance(_trunc, (bool, np.bool_)) and not bool(_trunc)
                    and _ns is not None and _nt is not None):
                if Ln >= 2 and _nt != _ns * (Ln - 1):
                    problems.append(f'{key}: L{Ln} n_tok={_nt} != n_seq*(L-1)={_ns * (Ln - 1)} — the cell did not score one next-token target per row')
            if _nt is not None and _nt <= 0:
                problems.append(f'{key}: L{Ln} n_tok={_nt} is not positive — '
                                'a cell that scored no target tokens is not a '
                                'measurement')
            if _ns is not None and _ns <= 0:
                problems.append(f'{key}: L{Ln} n_seq={_ns} is not positive — '
                                'a cell with no document rows is not a '
                                'measurement')
            _es = _exact_int(c.get('eval_span'))
            if _es is not None and _es <= 0:
                problems.append(f'{key}: L{Ln} eval_span={_es} is not positive')
            # `eval_span` is the number of SCORED targets per row, so it must
            # agree with both the requested length and the token denominator.
            # Checking only cross-record consistency let every arm agree on a
            # wrong span while the n_tok formula happened to use L-1.
            if (isinstance(_trunc, (bool, np.bool_)) and not bool(_trunc)
                    and _es is not None and Ln >= 2 and _es != Ln - 1):
                problems.append(
                    f'{key}: L{Ln} eval_span={_es} != L-1={Ln - 1} — a '
                    'full-length cell must score one next-token target per '
                    'input position after the first')
            if (isinstance(_trunc, (bool, np.bool_)) and bool(_trunc)
                    and _es is not None):
                if _es > max(0, Ln - 1):
                    problems.append(
                        f'{key}: L{Ln} eval_span={_es} exceeds L-1={Ln - 1} — '
                        'a truncated cell cannot have scored more targets than '
                        'the requested length')
                if _ns is not None and _nt is not None and _es > 0:
                    _expected = _ns * _es
                    if _nt != _expected:
                        problems.append(
                            f'{key}: L{Ln} n_tok={_nt} != n_seq*eval_span='
                            f'{_expected} — the truncated cell did not score '
                            'the declared number of targets per row')
            if 'truncated' not in c:
                problems.append(f'{key}: L{Ln} cell has NO `truncated` flag — a missing truncation marker is treated as unverifiable, not as a full-length measurement')
            elif not isinstance(c.get('truncated'), (bool, np.bool_)):
                # An explicit null/string is just as unverifiable as a missing
                # flag; only a real boolean False certifies a full-length cell.
                problems.append(f'{key}: L{Ln} cell has a non-boolean `truncated` value {c.get("truncated")!r} — unverifiable, not a full-length measurement')
            elif c.get('truncated'):
                problems.append(f'{key}: L{Ln} cell is marked truncated — it must not enter cross-length means')
    out = {'status': 'ok', 'by_length': {}}
    if not by_len and not problems:
        problems.append('no length cells found in the artifact — the by-length panel is unverifiable')
    for Ln, e in sorted(by_len.items()):
        ok = (len(e['n_tok']) == 1 and None not in e['n_tok']
              and len(e['eval_span']) == 1 and None not in e['eval_span'])
        # Missing n_seq is unverifiable, not "consistent": without the
        # per-document row count we cannot show the arms scored the same
        # documents.  Fail closed exactly like a missing `truncated` flag.
        nseq_ok = len(e['n_seq']) == 1 and None not in e['n_seq']
        out['by_length'][str(Ln)] = {'n_tok_values': sorted((str(x) for x in e['n_tok'])), 'eval_span_values': sorted((str(x) for x in e['eval_span'])), 'n_seq_values': sorted((str(x) for x in e['n_seq'])), 'same_scored_targets_across_records': bool(ok), 'same_n_seq_across_records': bool(nseq_ok)}
        if not ok:
            problems.append(f'L{Ln}: records disagree on n_tok/eval_span — different target sets are being scored across arms; cross-arm comparison at this length is invalid')
        if not nseq_ok:
            problems.append(f'L{Ln}: records disagree on the per-document row count n_seq — the arms did not score the same documents')
    out['problems'] = problems
    out['status'] = 'fail' if problems else 'ok'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
# 6) fairness：冻结臂的基线公平审计
# ---------------------------------------------------------------------------

def audit_fairness(summary_path='results_v10_mech/summary.json', arms=None):
    """基线公平（零 GPU）：优化器/调度/训练量跨臂一致 + 调参机会审计。

    核验点：冻结四臂的 run_cfg（lr/wd/steps/token/warmup/形状）逐字段一致
    （exp_lib.pair_reason 口径）；train_knobs（优化器/betas/eps/裁剪/调度）
    一致；mlp_ratio 只在「参数量匹配」臂上有差异且被显式记录。任何臂间
    单独调参会在此被点名，而不是混进「方法效果」里。"""
    summ = _load(_audit_artifact(summary_path))
    if summ is None:
        out = {'status': 'not_verifiable_locally', 'note': f'{summary_path} absent — needs the training records of the frozen arms'}
        print(json.dumps(out, ensure_ascii=False))
        return out
    L = _import_exp_lib()
    arms = arms or ['csa_fixed_rope', 'full_rope', 'csa_fixed_rope_randidx', 'csa_fixed_rope_allblocks']
    per_arm = {}
    for key, r in summ.items():
        if not (isinstance(r, dict) and r.get('variant') in arms):
            continue
        per_arm.setdefault(r['variant'], []).append(r)
    out = {'status': 'ok', 'arms': {}, 'controlled_variables': None,
           'seed_coverage': {}}
    problems = []
    warnings = []
    arms_present = [a for a in arms if a in per_arm]

    def _setting_key(value):
        # Malformed JSON can carry a dict/list where a run_cfg string belongs;
        # a raw set insert would raise TypeError instead of reporting the
        # mixed protocol.  Keep hashable values unchanged and give unhashable
        # ones a stable canonical spelling.
        try:
            hash(value)
        except TypeError:
            return ('__unhashable__', json.dumps(value, sort_keys=True, default=str))
        return value

    for a, recs in per_arm.items():
        cfgs = {_setting_key(r.get('run_cfg')) for r in recs}
        knob_vals = {json.dumps(r.get('train_knobs'), sort_keys=True)
                     for r in recs if r.get('train_knobs') is not None}
        n_missing_knobs = sum(1 for r in recs if r.get('train_knobs') is None)
        # NaN / non-integral floats have no seed identity; int(nan) would
        # abort the whole audit instead of excluding the malformed record.
        seeds = sorted({_s for r in recs
                        if (_s := L._as_seed_int(r.get('seed'))) is not None})
        out['arms'][a] = {
            'n_records': len(recs), 'seeds': seeds, 'n_seeds': len(seeds),
            'n_distinct_run_cfg': len(cfgs),
            'n_distinct_train_knobs': len(knob_vals),
            'n_missing_train_knobs': n_missing_knobs,
            # A malformed JSON value (dict/list) is unhashable; canonicalise it
            # for the set instead of aborting the whole fairness audit.
            'params': sorted({_setting_key(r.get('params')) for r in recs},
                             key=lambda x: (
                                 x is None,
                                 0 if isinstance(x, numbers.Real)
                                 and not isinstance(x, (bool, np.bool_)) else 1,
                                 float(x) if isinstance(x, numbers.Real)
                                 and not isinstance(x, (bool, np.bool_)) else repr(x))),
        }
        if len(cfgs) > 1:
            problems.append(f'{a}: {len(cfgs)} distinct run_cfg values within one arm — mixed protocols inside an arm')
        if len(knob_vals) > 1:
            problems.append(f'{a}: {len(knob_vals)} distinct train_knobs values within one arm — mixed optimizer/schedule settings inside an arm')
        if 0 < n_missing_knobs < len(recs):
            # Some seeds carry the optimizer/schedule descriptor and some do
            # not.  The records that do carry it can agree while the arm as a
            # whole is still unverified; trusting the populated subset would
            # let one seed silently run under a different optimizer.
            problems.append(f'{a}: {n_missing_knobs}/{len(recs)} records omit train_knobs — the optimizer/schedule descriptor covers only part of the arm')
    # Parameter matching is part of a controlled baseline comparison, not
    # only of the contrast-level pair gate.  If every arm exposes one exact
    # positive parameter count, those totals must agree; otherwise a "gain"
    # can be a capacity gain.  Missing/malformed totals are unverifiable and
    # only warn (legacy panels), never a silent pass.
    _params_by_arm = {}
    for _a, _recs in per_arm.items():
        _raw_p = [r.get('params') for r in _recs]
        _parsed_p = [L._as_seed_int(v) for v in _raw_p]
        if (all(v is not None and v > 0 for v in _parsed_p)
                and len(set(_parsed_p)) == 1):
            _params_by_arm[_a] = int(_parsed_p[0])
    if len(per_arm) >= 2:
        if len(_params_by_arm) == len(per_arm):
            if len(set(_params_by_arm.values())) > 1:
                problems.append(
                    'parameter counts differ across arms (%s) — a gain from a '
                    'larger model is a capacity effect, not a controlled '
                    'baseline/method contrast'
                    % ', '.join('%s=%s' % (a, _params_by_arm[a])
                                for a in sorted(_params_by_arm)))
        else:
            _missing_params = sorted(set(per_arm) - set(_params_by_arm))
            warnings.append(
                'parameter counts are absent/non-unique for arm(s) %s — '
                'capacity matching cannot be verified from this summary'
                % _missing_params)
    _unverifiable = False
    if len(per_arm) >= 2:
        # The cross-arm check must use the SAME gate as the statistics
        # (`pair_reason`): identical run_cfg AND identical train_knobs AND the
        # same steps/tokens.
        recs_map, _amb = L._variant_seed_map(summ)
        _cv_arms = [a for a in arms if a in per_arm]
        if recs_map and len(_cv_arms) >= 1:
            cv = L.controlled_variables_report(recs_map, _cv_arms)
            out['controlled_variables'] = cv
            if cv.get('run_cfg_equal') is False:
                problems.append('run_cfg differs across arms — the training length / data / architecture knobs are not controlled; pair_reason would refuse every pairing')
            if cv.get('train_knobs_equal') is False:
                problems.append('train_knobs differ across arms (optimizer / betas / eps / lr-schedule / grad-clip / augmentation) — a gain from a different optimizer is not a gain from the method')
            if cv.get('hyperparams_equal') is False:
                problems.append('structured hyperparameters differ across arms: ' + ', '.join(cv.get('hyperparam_differences') or []) + ' — the comparison is not fully controlled')
            _obj_vals = cv.get('objective_values') or {}
            if cv.get('objective_equal') is False:
                warnings.append('training objectives differ across arms (' + ', '.join(
                    '%s=%d value(s)' % (a, len(v)) for a, v in sorted(_obj_vals.items())) +
                    ') — this is a system-level comparison unless the objective change is the declared intervention; do not describe it as an isolated mechanism')
            elif cv.get('objective_equal') is None and any(_obj_vals.values()):
                warnings.append('training-objective equality across arms is UNVERIFIABLE (some records omit the descriptor) — mechanism attribution needs the objective descriptor on every seed')
            for _arm in cv.get('within_arm_mixed') or []:
                problems.append(f'{_arm}: seeds mix DIFFERENT run_cfg/train_knobs/hyperparameter values inside one arm')
            for _arm, _defects in (cv.get('train_knobs_defects') or {}).items():
                _reasons = sorted({d.get('reason') for d in _defects if isinstance(d, dict)})
                problems.append(f'{_arm}: train_knobs descriptor is incomplete/malformed '
                                f'({"; ".join(_reasons) or "unknown"}) — the '
                                'optimizer/schedule/augmentation recipe cannot be '
                                'shown to be controlled')
            for _key in ('run_cfg', 'train_knobs', 'hyperparams'):
                _eq = cv.get(_key + '_equal')
                if _eq is None:
                    warnings.append(f'{_key} equality across arms is UNVERIFIABLE (some records omit it / differ in coverage) — do not read the arm means as a controlled contrast')
        ref_arm = arms[0] if arms[0] in per_arm else next(iter(per_arm))
        ref_cfg = {_setting_key(r.get('run_cfg')) for r in per_arm[ref_arm]}
        if not ref_cfg or None in ref_cfg:
            # Length-phase / checkpoint-backed records carry `by_len` and a
            # code stamp but no run_cfg fingerprint, so cross-arm recipe
            # equality is NOT verifiable from this summary.
            _unverifiable = True
            out['note'] = 'this panel\'s records carry no run_cfg fingerprint (e.g. length-phase records) — cross-arm training-recipe equality cannot be verified from this summary; run the fairness audit on the run-level summary that owns the run_cfg records'
        else:
            for a, recs in per_arm.items():
                if a == ref_arm:
                    continue
                for r in recs:
                    if r.get('run_cfg') is None or _setting_key(r.get('run_cfg')) not in ref_cfg:
                        problems.append(f'{a}: a record whose run_cfg does not equal the {ref_arm} reference (pair_reason would refuse this pairing)')
                        break
        # Seed coverage: arm means are only comparable if they average the
        # same seeds.  An arm that silently lost a seed (crash / cap / error)
        # must not be compared against one that kept it.
        if all(all(r.get('run_cfg') is not None for r in per_arm[a]) for a in per_arm):
            _cover = {}
            for (v, t, s) in recs_map:
                if v in per_arm:
                    _cover.setdefault(v, set()).add((str(t or ''), int(s)))
            out['seed_coverage'] = {a: sorted(_cover.get(a, set())) for a in arms_present}
            _sets = {a: frozenset(_cover.get(a, set())) for a in arms_present}
            if len(set(_sets.values())) > 1:
                _detail = '; '.join('%s={%s}' % (a, ','.join('%s:%s' % (t or 'base', s)
                                                              for t, s in sorted(_sets[a])))
                                    for a in arms_present)
                problems.append('arm seed/protocol coverage differs (%s) — arm means are over different seeds; pair only the shared seeds or restore the missing cells' % _detail)
    out['problems'] = problems
    out['tuning'] = L.baseline_tuning_audit(path=os.path.join(REPO, 'results_baseline_tuning', 'baseline_tuning.json'))
    out['simple_baseline'] = L.simple_baseline_audit(path=os.path.join(REPO, 'results_baseline_tuning', 'simple_baseline.json'))
    out['warnings'] = warnings
    if not out['tuning'].get('tuning_verified'):
        out['warnings'].append('no equal-search baseline tuning artifact — a win over these baselines is not yet evidence against a well-tuned baseline; run baseline_tune.py')
    elif out['tuning'].get('tuned_claim_allowed') is False:
        out['warnings'].append(
            'the equal-search tuning artifact is verified but does not support a '
            '"well-tuned baseline" claim (%s) — fix the search grid/horizon before '
            'calling the baseline tuned'
            % '; '.join(out['tuning'].get('tuned_claim_reasons') or []))
    if not out['simple_baseline'].get('baseline_verified'):
        out['warnings'].append(
            'no verified tuned count-model floor — a neural win has not yet been '
            'shown to beat the simplest baseline; run baseline_tune.py simple')
    elif out['simple_baseline'].get('tuned_floor_claim_allowed') is False:
        out['warnings'].append(
            'the count-model floor passed its structural audit but is not '
            'adequately tuned (%s); extend the smoothing grid before quoting a '
            'beats-floor claim'
            % '; '.join(out['simple_baseline'].get('tuned_floor_claim_reasons') or []))
    out['status'] = ('fail' if problems else
                     'warn' if out['warnings'] else
                     ('not_verifiable_locally' if _unverifiable else 'ok'))
    if not arms_present:
        # An existing summary with none of the requested arms cannot verify a
        # controlled comparison, even if the tuning artifact looks healthy.
        out['status'] = 'fail' if problems else 'not_verifiable_locally'
        out.setdefault('note', 'none of the requested arms is present in the summary — cross-arm fairness cannot be verified')
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
# 7) failures：失败/缺失的臂间集中度
# ---------------------------------------------------------------------------

def audit_failures(outdirs=None):
    """失败集中度（零 GPU）：error/NaN/截断记录按臂分布的读数。

    审计口径：崩溃/NaN/超时是否集中在某些臂；仅保留成功运行是否构成
    选择偏差。本审计把 error 记录按 (臂, 错误类型) 计数，并给出描述性
    集中度（不主张显著性——小样本下显著性检验本身功效不足）。"""
    L = _import_exp_lib()
    out = {'status': 'ok', 'panels': {}}
    dirs = outdirs or list(_iter_result_dirs())
    for d in dirs:
        summ = _load(_audit_artifact(os.path.join(d, 'summary.json')))
        if not summ:
            continue
        by_arm = {}
        n_ok = {}
        unmeasured = {}
        unmeasured_samples = {}
        for key, r in summ.items():
            if not isinstance(r, dict):
                continue
            v = r.get('variant')
            if not isinstance(v, str) or not v:
                continue
            if 'error' in r or r.get('status') in ('error', 'failed'):
                by_arm.setdefault(v, []).append(str(r.get('error') or r.get('status'))[:80])
            elif L.ppl_is_usable(r.get('ppl')) and not r.get('synthesized'):
                n_ok[v] = n_ok.get(v, 0) + 1
            else:
                # A record that neither errored nor produced a usable PPL is
                # still a missing measurement.  Dropping it from both sides
                # would make a panel with selectively failed seeds look like a
                # clean survivorship sample; count it explicitly.
                if r.get('synthesized'):
                    _why = 'synthesized record (excluded by protocol)'
                else:
                    _why = f'ppl not usable ({r.get("ppl")!r})'
                unmeasured.setdefault(v, []).append(_why)
                unmeasured_samples.setdefault(v, []).append(_why)
        if by_arm or unmeasured:
            panel = {'errors_by_arm': {v: len(e) for v, e in by_arm.items()},
                     'ok_by_arm': n_ok,
                     'error_samples': {v: e[:2] for v, e in by_arm.items()}}
            if unmeasured:
                panel['unmeasured_by_arm'] = {v: len(e) for v, e in unmeasured.items()}
                panel['unmeasured_samples'] = {v: e[:2] for v, e in unmeasured_samples.items()}
                out['status'] = 'warn'
            out['panels'][d] = panel
    if not out['panels']:
        out['status'] = 'not_verifiable_locally'
        out['note'] = 'no error records found in local results (or no results) — run after each phase on the experiment machine; failures are already written as error records by the drivers'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
# 7b) metrics：指标口径与不平衡处理
# ---------------------------------------------------------------------------

def audit_metrics(outdirs=None):
    """指标口径审计（零 GPU）：主指标/方向/聚合/不平衡声明 + 落盘记录核验。

    审计口径：类别不平衡不能只看 accuracy；多分类 F1 必须注明 macro/
    micro/weighted；极度不平衡优先 PR-AUC；指标必须对齐任务目标。本审计
    先检查 `exp_lib.METRIC_POLICY` 是否完整声明，再对本地面板检查边界
    指标是否同时给出 precision/recall/随机基线、NIAH 是否同时给出 answer
    NLL 与明确命名的 macro/micro/weighted F1、二分类是否声明决策阈值与
    来源、校准字段是否在合法范围，以及 F1 是否能由 per-class 计数复现。
    缺失只标
    warn/不可核验，不伪造一个「通过」。"""
    L = _import_exp_lib()
    out = {'status': 'ok', 'policy': L.metric_policy_audit(), 'panels': {}}
    dirs = outdirs or list(_iter_result_dirs())
    for d in dirs:
        summ = _load(_audit_artifact(os.path.join(d, 'summary.json')))
        if not summ:
            continue
        rep = L.metric_policy_audit(summ)
        out['panels'][d] = {'status': rep['status'], 'checks': rep['checks'], 'warnings': rep['warnings']}
        if rep['warnings']:
            out['status'] = 'warn'
    if out['policy']['warnings']:
        out['status'] = 'warn'
    if not out['panels']:
        out['note'] = 'no local summaries — policy declarations were checked; record-level checks run on the experiment machine'
        # A file that exists but holds no panels proves nothing about the
        # records.  Do not leave the top-level verdict at 'ok' (the same
        # fail-closed rule as an empty split cache).
        if out['status'] == 'ok':
            out['status'] = 'not_verifiable_locally'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
# 7c) tuning：基线调参范围对账
# ---------------------------------------------------------------------------

def audit_tuning(path=None):
    """基线调参审计（零 GPU）：主方法与简单基线是否用了同一搜索网格/范围。

    没有 `baseline_tuning.json` 时返回 `not_verifiable_locally`——未调参的
    基线只能证明复杂模型比稻草人强，不能写成「优于精心调优的基线」。"""
    L = _import_exp_lib()
    if path is None:
        path = os.path.join(REPO, 'results_baseline_tuning', 'baseline_tuning.json')
    out = L.baseline_tuning_audit(path=path)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def audit_simple_baseline(path=None):
    """简单基线地板审计（零 GPU）。

    复杂模型必须先战胜一个**调过参**的最简基线（这里是训练集计数的
    unigram/bigram 模型，平滑/插值在验证集上选）。没有
    `simple_baseline.json` 时返回 not_verifiable_locally，不能把「比稻草人
    强」写成「机制有效」。"""
    L = _import_exp_lib()
    if path is None:
        path = os.path.join(REPO, 'results_baseline_tuning', 'simple_baseline.json')
    out = L.simple_baseline_audit(path=path)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def audit_baseline_design():
    """基线设计预检（零 GPU）：调参臂组、搜索网格、种子分区与族功效。

    这是 `baseline_tune.py plan` 的审计入口：在花 GPU 之前检查声明的
    调参设计是否可启动、最终种子数是否足以对声明的 Holm 族做校正，
    以及计数地板的全因子网格是否完整。"""
    L = _import_exp_lib()
    out = L.baseline_plan()
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def audit_hypothesis_coverage():
    """假设覆盖审计（零 GPU）：每个声明对照族是否有预注册假设，是否有孤儿假设。"""
    L = _import_exp_lib()
    out = L.hypothesis_coverage_audit()
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def audit_sesoi_declaration(spec=None, design=None):
    """消融 SESOI 声明审计（零 GPU，设计期，不读结果）。

    `methodology_audit` 只在**每条对照都带预注册 margin** 时才能区分
    「模块可删（TOST 等价）」与「数据不足（不确定）」。这个入口在跑 GPU 前
    检查：
      * 每个 `ABLATION_DESIGN` 组件都有 margin，且单位唯一、有限、>0；
      * margin 能在一个正常 PPL 基线上换算成正的绝对 PPL 边界；
      * 没有为未声明组件写 margin（拼写错误会让整条对照静默失去 SESOI）；
      * 声明的留一对照本身在 `RUN_ABL` 里完整排程。
    缺 margin 或声明畸形是 fail-closed 的设计缺陷，不是事后调整理由。
    """
    L = _import_exp_lib()
    spec = L.ABLATION_SESOI if spec is None else spec
    design = L.ABLATION_DESIGN if design is None else design
    out = {'status': 'ok', 'problems': [], 'warnings': [],
           'components': {}, 'ablation_complete': None}
    try:
        norm = L._normalise_sesoi_design(spec)
    except Exception as e:
        out['status'] = 'fail'
        out['problems'].append(f'SESOI declaration is malformed: {type(e).__name__}: {e}')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    if not isinstance(design, dict):
        out['status'] = 'fail'
        out['problems'].append('ABLATION_DESIGN is not a mapping')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    for comp, contrasts in design.items():
        if not isinstance(comp, str) or not comp.strip():
            out['problems'].append(
                f'ablation component name {comp!r} is not a non-empty string')
            continue
        if not isinstance(contrasts, (list, tuple)):
            out['problems'].append(
                f'{comp}: declared contrasts must be a list/tuple, got '
                f'{type(contrasts).__name__}')
            continue
        bad_contrasts = [i for i, c in enumerate(contrasts)
                         if not (isinstance(c, (tuple, list)) and len(c) == 3
                                 and all(isinstance(x, str) and x.strip()
                                         for x in c[:2]))]
        if bad_contrasts:
            out['problems'].append(
                f'{comp}: malformed contrast declaration(s) at index '
                f'{bad_contrasts[:5]} (expected (with_arm, without_arm, note))')
        rec = {'declared': comp in norm, 'n_contrasts': len(contrasts)}
        if comp not in norm:
            out['problems'].append(f'{comp}: no pre-registered SESOI margin; '
                                   'a non-significant ablation cannot be called removable')
        else:
            rec['margin_spec'] = norm[comp]
            margin, why = L._sesoi_margin_ppl(norm[comp], 50.0)
            rec['probe_margin_ppl_at_baseline_50'] = margin
            if margin is None:
                out['problems'].append(f'{comp}: margin cannot be translated to PPL ({why})')
            if not str(norm[comp].get('rationale') or '').strip():
                out['warnings'].append(f'{comp}: margin has no recorded rationale')
        out['components'][comp] = rec
    extras = sorted(set(norm) - set(design), key=str)
    if extras:
        out['warnings'].append('SESOI declares component(s) not present in the '
                               'ablation design: ' + ', '.join(extras))
    try:
        abl = L.ablation_completeness_audit(design, L.RUN_ABL['variants'])
        out['ablation_complete'] = bool(abl.get('complete'))
        if not out['ablation_complete']:
            out['problems'].append('declared ablation is not complete under RUN_ABL: '
                                   + json.dumps(abl.get('missing_arms') or {}, ensure_ascii=False))
    except Exception as e:
        out['problems'].append(f'ablation completeness check failed: {type(e).__name__}: {e}')
    if out['problems']:
        out['status'] = 'fail'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def audit_baseline_sesoi(spec=None, design=None):
    """参数匹配基线的 SESOI 声明审计（零 GPU，设计期，不读结果）。

    `methodology_audit` 对方法 vs 参数匹配稠密基线同样需要预注册 margin，
    否则一个大 n 下显著、但实际只低 0.1% PPL 的结果会被误读为「方法有实际
    优势」。这个入口在跑 GPU 前检查：
      * 每个 `BASELINE_DESIGN` 对比都有唯一、有限、>0 的 margin；
      * margin 能在正常 PPL 基线上换算成正的绝对 PPL 边界；
      * 没有为未声明基线写 margin（拼写错误会让对照静默失去 SESOI）；
      * 声明的基线臂都在 `RUN['variants']` 中排程。
    """
    L = _import_exp_lib()
    spec = L.BASELINE_SESOI if spec is None else spec
    design = L.BASELINE_DESIGN if design is None else design
    out = {'kind': 'baseline_sesoi', 'status': 'ok', 'problems': [],
           'warnings': [], 'components': {}, 'all_arms_scheduled': None}
    try:
        norm = L._normalise_sesoi_design(spec)
    except Exception as e:
        out['status'] = 'fail'
        out['problems'].append(
            f'baseline SESOI declaration is malformed: {type(e).__name__}: {e}')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    if not isinstance(design, dict):
        out['status'] = 'fail'
        out['problems'].append('BASELINE_DESIGN is not a mapping')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    scheduled = set(L.RUN.get('variants') or ())
    all_scheduled = True
    for comp, contrasts in design.items():
        if not isinstance(comp, str) or not comp.strip():
            out['problems'].append(
                f'baseline component name {comp!r} is not a non-empty string')
            all_scheduled = False
            continue
        if not isinstance(contrasts, (list, tuple)):
            out['problems'].append(
                f'{comp}: declared contrasts must be a list/tuple, got '
                f'{type(contrasts).__name__}')
            all_scheduled = False
            continue
        bad_contrasts = [i for i, c in enumerate(contrasts)
                         if not (isinstance(c, (tuple, list)) and len(c) == 3
                                 and all(isinstance(x, str) and x.strip()
                                         for x in c[:2]))]
        if bad_contrasts:
            out['problems'].append(
                f'{comp}: malformed contrast declaration(s) at index '
                f'{bad_contrasts[:5]} (expected (with_arm, without_arm, note))')
        arms = sorted({str(x) for c in contrasts
                       if isinstance(c, (tuple, list)) and len(c) >= 2
                       for x in c[:2]})
        missing = [a for a in arms if a not in scheduled]
        if missing:
            all_scheduled = False
            out['problems'].append(
                f'{comp}: baseline arm(s) {missing} are not scheduled in '
                'RUN["variants"]')
        rec = {'declared': comp in norm, 'n_contrasts': len(contrasts),
               'arms': arms, 'arms_scheduled': not missing}
        if comp not in norm:
            out['problems'].append(
                f'{comp}: no pre-registered baseline SESOI margin; a '
                'non-significant baseline contrast cannot be called equivalent')
        else:
            rec['margin_spec'] = norm[comp]
            margin, why = L._sesoi_margin_ppl(norm[comp], 50.0)
            rec['probe_margin_ppl_at_baseline_50'] = margin
            if margin is None:
                out['problems'].append(
                    f'{comp}: baseline margin cannot be translated to PPL ({why})')
            if not str(norm[comp].get('rationale') or '').strip():
                out['warnings'].append(
                    f'{comp}: baseline margin has no recorded rationale')
        out['components'][comp] = rec
    extras = sorted(set(norm) - set(design), key=str)
    if extras:
        out['warnings'].append(
            'baseline SESOI declares component(s) not present in '
            'BASELINE_DESIGN: ' + ', '.join(extras))
    out['all_arms_scheduled'] = bool(all_scheduled)
    if out['problems']:
        out['status'] = 'fail'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def audit_interaction_sesoi(spec=None, design=None):
    """2x2 interaction SESOI declaration audit (zero-GPU, design-time).

    `factorial_interaction_audit` can only distinguish practical additivity
    from "not detected" when every declared interaction has a pre-registered
    margin.  This gate checks coverage, unit ambiguity, positivity and the
    directional hypothesis before any GPU is spent; it reads no result.
    """
    L = _import_exp_lib()
    out = L.interaction_sesoi_audit(design=design, sesoi=spec)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def audit_interaction_power(pilot_path=None, n_seeds=None, m_family=None,
                            require_power=False):
    """2x2 interaction power / minimum-detectable-effect design audit.

    Design-stage only: it reads an optional validation pilot artifact and the
    declared factorial/SESOI tables, never the held-out test split.  A missing
    pilot is ``not_verifiable_locally``; a family whose exact sign-flip floor
    cannot clear alpha (for example n=6 with m=5 interactions) is
    ``resolution_limited`` and is reported by name.  ``require_power=True``
    turns an unmeasured or underpowered interaction family into a hard
    failure; the default keeps it a warning so an exploratory panel is not
    silently promoted or silently deleted.
    """
    L = _import_exp_lib()
    pp = pilot_path or 'analysis_v12/factorial_power_pilot.json'
    if not os.path.isabs(str(pp)):
        pp = os.path.join(REPO, str(pp))
    pilot = None
    pilot_error = None
    if not os.path.exists(pp):
        pilot_error = f'no pilot artifact at {pp}'
    else:
        try:
            with open(pp, encoding='utf-8') as f:
                pilot = json.load(f)
        except Exception as exc:
            pilot_error = f'{pp} is not readable JSON ({type(exc).__name__}: {exc})'
    if pilot is not None and not isinstance(pilot, dict):
        pilot_error = pilot_error or (
            f'{pp} is not a JSON object ({type(pilot).__name__}); a pilot '
            'artifact must be a mapping')
    rep = L.factorial_power_audit(
        pilot, n_seeds=n_seeds, m_family=m_family,
        require_power=require_power)
    rep['pilot_path'] = pp
    # "Available" means a usable object artifact, not merely a file that
    # parsed to a scalar/list.
    rep['pilot_available'] = isinstance(pilot, dict)
    if pilot_error:
        rep['pilot_error'] = pilot_error
        rep.setdefault('warnings', []).append(
            pilot_error + '; only the resolution floor can be audited')
        if pilot is None and rep.get('status') == 'ok':
            rep['status'] = 'not_verifiable_locally'
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return rep


def audit_hypotheses():
    """预注册假设完整性 / 指标对齐审计（零 GPU）。

    每个声明式对照必须有 statement / mechanism / predicted_sign /
    primary_metric / falsifier；primary_metric 必须是 METRIC_POLICY 里
    真实定义的端点，predicted_sign=0 不能支撑方向性优势。"""
    L = _import_exp_lib()
    out = {'status': 'ok',
           'coverage': L.hypothesis_coverage_audit(),
           'component': L.hypothesis_audit(L.HYPOTHESIS_DESIGN, 'component'),
           'baseline': L.hypothesis_audit(L.BASELINE_HYPOTHESIS_DESIGN, 'baseline'),
           'tuned_baseline': L.hypothesis_audit(L.TUNED_BASELINE_HYPOTHESIS_DESIGN, 'tuned_baseline'),
           'floor': L.hypothesis_audit(L.FLOOR_HYPOTHESIS_DESIGN, 'floor'),
           'open_block': L.hypothesis_audit(L.OPEN_BLOCK_HYPOTHESIS_DESIGN, 'open_block'),
           'niah': L.hypothesis_audit(L.NIAH_HYPOTHESIS_DESIGN, 'niah'),
           'interaction': L.hypothesis_audit(L.INTERACTION_HYPOTHESIS_DESIGN, 'interaction')}
    if any(out[k]['status'] != 'ok' for k in ('coverage', 'component', 'baseline', 'tuned_baseline', 'floor', 'open_block', 'niah', 'interaction')):
        out['status'] = 'warn'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out
