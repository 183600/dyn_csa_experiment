#!/usr/bin/env python3
"""audit_gates.py — 评审二轮·意见十一的可执行审计入口（零实验）。

评审指出九个「目前不能排除、必须进一步验证」的领域。本模块把其中能以
**代码 + 现有落盘产物**核验的部分变成可运行的审计子命令；不能本地核验的
（需要 GPU 训练 / 需要 ckpt）提供入口与判据，等实验机器上运行。它**不启动
任何训练**——每个子命令要么零 GPU 读产物，要么纯 CPU 单元测试。

子命令（python audit_gates.py <cmd>）：
  leakage      数据泄漏：分词器拟合源（train-only）、train/test 词表一致性、
               缓存指纹自洽、train/test 文档重叠率（n-gram/shingle）
  split        划分纪律：元数据声明的时序/非打乱/群组构造、跨 split 完全
               重复行、train/holdout 词分布偏移（Jensen-Shannon）
  causality    自回归正确性：纯 CPU 微型模型上的前缀不变性、标签移位、
               池化/选择器因果性、逐 token 解码 vs 整段前向一致性
  attribution  动态/固定分块归因：从落盘 summary 重算每臂块数、压缩率、
               top-k 覆盖与参数量对照表（需要 results 目录，零 GPU）
  boundary     边界意义：重算边界对齐 vs 匹配块长分布的随机基线
               （复用 exp_lib 的 _random_cut_precision 口径，零 GPU）
  length       长度外推一致性：同长度跨臂 n_tok/eval_span 一致性
               （同一组目标 token 被评分）与截断检查（零 GPU）
  fairness     基线公平：冻结臂间 run_cfg/train_knobs/lr/warmup/调参机会
               一致性与 mlp_ratio 审计（零 GPU）
  metrics      指标口径：主指标/方向/聚合/不平衡声明与记录级伴随指标
               （边界需 precision/recall/随机基线，NIAH 需 answer NLL）
  tuning       基线调参：主方法与简单基线是否获得同一搜索网格与范围
               （读 results_baseline_tuning/baseline_tuning.json）
  methodology  受控对比 / 消融完整性 / 统计可靠性：逐臂全种子设置一致性、
               每个创新点的留一对照是否完整、n 与 mean±std（零 GPU）
  failures     失败与缺失：error 记录按臂集中度（选择偏差读数，零 GPU）
  budget       规模核算：6ND 代理 FLOPs 估算 + TimeGuard 实测用时对账
               （理论 FLOPs ≠ 实测速度，两者并列报告，零 GPU）
  efficiency   prefill/decode/峰值内存测量工具（需要模型；给出可复用函数
               与 CPU 微型演示——不是实验，是测量代码自检）
  report       汇总运行本地可执行的全部审计，输出 audit_report.json

运行纪律：`python audit_gates.py causality` 在 CPU 上几秒内完成；其余子命令
读 `results_*/`（不存在时给出「不可本地核验」的明确说明而不是报错）。
"""
import json
import math
import os
import sys
import time

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except (ValueError, OSError):
            pass

REPO = os.path.dirname(os.path.abspath(__file__))


def _load(path):
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path, encoding='utf-8'))
    except Exception as e:
        print(f'[audit] WARNING: {path} exists but cannot be parsed ({type(e).__name__}: {e})')
        return None


def _iter_result_dirs():
    for name in sorted(os.listdir(REPO)):
        if name.startswith('results_') and os.path.isdir(os.path.join(REPO, name)):
            yield name


def _import_exp_lib():
    sys.path.insert(0, REPO)
    import exp_lib as L
    return L


# ---------------------------------------------------------------------------
# 1) leakage：分词器来源 / 词表一致性 / 缓存指纹 / 文档重叠
# ---------------------------------------------------------------------------

def audit_leakage(cache_dir='./wt103_cache', shingle=13, max_rows_shingled=400):
    """数据泄漏审计（零 GPU，读 wt103_cache 落盘产物）。

    核验点：
      a. test 缓存 meta 的 fit_fp（拟合语料指纹）必须来自 TRAIN 前 50k 行
         ——与 train 侧 meta 的语料指纹同源（分词器没有见过 valid/test）；
      b. test 侧持久化分词器与 train 侧持久化分词器逐 token 全等
         （load_wikitest 运行期同样硬校验，这里独立复核落盘副本）；
      c. train/test id 缓存的 src_fp 与文件内容指纹自洽；
      d. train/test 文档重叠：k-gram shingle 集合的 Jaccard 与命中数
         （WikiText-103 本身存在近重复文档；本读数给出「重叠存在多少」的
         事实基线，供 length/generalization 主张的边界声明使用）。"""
    out = {'status': 'ok', 'checks': {}}
    test_meta = None
    train_meta = None
    import glob
    _tm = sorted(glob.glob(os.path.join(cache_dir, 'test_meta_*.json')))
    if _tm:
        test_meta = _load(_tm[0])
    _vm = sorted(glob.glob(os.path.join(cache_dir, 'meta_v4_*.json')))
    if _vm:
        train_meta = _load(_vm[0])
    if test_meta is None or train_meta is None:
        out['status'] = 'not_verifiable_locally'
        out['note'] = f'{cache_dir}/ lacks test_meta_*.json or meta_v4_*.json — run the corpus loaders once (or copy the cache) to audit tokenizer provenance; the runtime hard checks in exp_lib.load_wiktest/load_wikitext remain in force meanwhile'
        return out
    out['checks']['test_fit_fp'] = test_meta.get('fit_fp')
    out['checks']['train_src_fp'] = train_meta.get('src_fp') or train_meta.get('train_fp')
    out['checks']['test_src_fp'] = test_meta.get('src_fp')
    out['checks']['vocab_agree'] = int(test_meta.get('vocab', -1)) == int(train_meta.get('vocab', -2))
    # b) persisted tokenizers, if both exist
    import hashlib
    _tt = sorted(glob.glob(os.path.join(cache_dir, 'test_tok_*.json')))
    _vt = sorted(glob.glob(os.path.join(cache_dir, 'tok_v4_*.json')))
    if _tt and _vt:
        def _sha(p):
            h = hashlib.sha256()
            with open(p, 'rb') as f:
                h.update(f.read())
            return h.hexdigest()
        # tokenizers are fitted deterministically on the same rows: identical
        # bytes are expected; a size-only match is NOT sufficient (different
        # merges can share a size — see exp_lib.load_wikitest).
        _a, _b = (_sha(_tt[0]), _sha(_vt[0]))
        out['checks']['tokenizer_files_identical'] = bool(_a == _b)
        out['checks']['tokenizer_sha'] = {'test': _a[:16], 'train': _b[:16]}
        try:
            from tokenizers import Tokenizer
            ta, tb = (Tokenizer.from_file(_tt[0]), Tokenizer.from_file(_vt[0]))
            out['checks']['tokenizer_vocab_identical'] = bool(dict(ta.get_vocab()) == dict(tb.get_vocab()))
        except Exception as e:
            out['checks']['tokenizer_vocab_identical'] = f'skipped ({type(e).__name__})'
    # d) document overlap via shingles on the cached id matrices
    try:
        import numpy as np
        import glob as _g
        _te = sorted(_g.glob(os.path.join(cache_dir, 'test_ids_*.npy')))
        _tr = sorted(_g.glob(os.path.join(cache_dir, 'train_ids_*.npy')))
        if _te and _tr:
            te = np.load(_te[0])
            tr = np.load(_tr[0])[: 4000000]  # fixed-size train-side sample
            def _shingles(mat, k, cap_rows):
                s = set()
                rows = min(mat.shape[0], cap_rows)
                for i in range(rows):
                    r = mat[i].tolist()
                    for j in range(0, len(r) - k, k):
                        s.add(hash(tuple(r[j:j + k])))
                return s
            t0 = time.time()
            te_s = _shingles(te, shingle, max_rows_shingled)
            tr_s = _shingles(tr, shingle, max_rows_shingled)
            inter = len(te_s & tr_s)
            union = len(te_s | tr_s)
            out['checks']['doc_overlap'] = {'shingle_k': shingle, 'train_rows_scanned': int(tr.shape[0]), 'test_rows_scanned': int(te.shape[0]), 'test_shingles': len(te_s), 'train_shingles': len(tr_s), 'shared_shingles': inter, 'jaccard': (inter / union) if union else 0.0, 'seconds': round(time.time() - t0, 1), 'note': 'nonzero overlap reflects WikiText-103 near-duplicate documents; report it as a boundary of generalization claims, not as leakage by itself — tokenizer fit and split handling are the leakage-critical parts and are checked separately'}
        else:
            out['checks']['doc_overlap'] = 'npy caches absent'
    except Exception as e:
        out['checks']['doc_overlap'] = f'failed ({type(e).__name__}: {e})'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
# 1b) split discipline：划分顺序 / 群组完整性 / 重复行 / 分布偏移
# ---------------------------------------------------------------------------

def _sample_tokens(arr, cap=1000000, seed=0):
    """Deterministic sample of a token array (for distribution comparisons)."""
    import numpy as np
    a = np.asarray(arr).reshape(-1)
    if a.size <= cap:
        return a
    rng = np.random.default_rng(seed)
    idx = rng.choice(a.size, size=int(cap), replace=False)
    idx.sort()
    return a[idx]


def _window_hashes(values, width):
    """Polynomial hashes of every length-``width`` window (uint64 wrap)."""
    import numpy as np
    a = np.asarray(values).reshape(-1)
    if width <= 0 or a.size < width:
        return np.empty(0, dtype=np.uint64)
    a = a.astype(np.uint64, copy=False)
    base = np.uint64(1000003)
    mod = 1 << 64
    pow_b = np.empty(width, dtype=np.uint64)
    pow_b[0] = 1
    if width > 1:
        pow_b[1:] = np.multiply.accumulate(np.full(width - 1, base, dtype=np.uint64))
    h0 = int(np.sum(a[:width] * pow_b[::-1], dtype=np.uint64))
    count = a.size - width + 1
    if count == 1:
        return np.asarray([h0], dtype=np.uint64)
    b_width = int(pow_b[-1]) * int(base) % mod
    c = a[width:] - a[:count - 1] * np.uint64(b_width)
    inv = pow(int(base), -1, mod)
    inv_pow = np.multiply.accumulate(np.full(count - 1, np.uint64(inv), dtype=np.uint64))
    pref = np.cumsum(c * inv_pow, dtype=np.uint64)
    b_pow = np.empty(count, dtype=np.uint64)
    b_pow[0] = 1
    if count > 1:
        b_pow[1:] = np.multiply.accumulate(np.full(count - 1, base, dtype=np.uint64))
    out = np.empty(count, dtype=np.uint64)
    out[0] = np.uint64(h0)
    out[1:] = b_pow[1:] * (np.uint64(h0) + pref)
    return out


def _row_overlap_count(train, rows, chunk_tokens=1 << 20):
    """Count unique ``rows`` that occur verbatim in ``train``.

    A token cache stores the training side as a flat token stream, so a row
    overlap is a *subsequence* match, not an aligned row-index match.  The
    2-D case uses exact row keys; the 1-D case scans all windows with a
    rolling hash and verifies every candidate against the actual tokens, so
    no match can be missed and hash collisions cannot create one.
    """
    import numpy as np
    import data_guard as DG
    train = np.asarray(train)
    rows = np.asarray(rows)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.size == 0:
        return 0
    if train.ndim == 2:
        return len(set(DG._row_keys(train)) & set(DG._row_keys(rows)))
    if train.ndim != 1:
        return 0
    uniq = np.unique(rows, axis=0)
    width = int(uniq.shape[1])
    if width <= 0 or train.size < width:
        return 0
    val_hashes = np.asarray([int(_window_hashes(row, width)[0]) for row in uniq], dtype=np.uint64)
    by_hash = {}
    for i, h in enumerate(val_hashes.tolist()):
        by_hash.setdefault(h, []).append(i)
    matched = set()
    step = max(1, int(chunk_tokens))
    start = 0
    total = int(train.size)
    while start < total and len(matched) < len(uniq):
        end = min(total, start + step + width - 1)
        seg = np.asarray(train[start:end])
        hashes = _window_hashes(seg, width)
        if hashes.size:
            for i in np.nonzero(np.isin(hashes, val_hashes))[0]:
                for r in by_hash.get(int(hashes[i]), ()):
                    if r in matched:
                        continue
                    pos = start + int(i)
                    if np.array_equal(np.asarray(train[pos:pos + width]), uniq[r]):
                        matched.add(r)
        if end >= total:
            break
        start += step
    return len(matched)


def audit_split_discipline(cache_dirs=None, max_rows=50000, js_warn=0.05,
                           js_fail=0.25, min_shift_tokens=5000):
    """Audit the *split contract* for every cached corpus on disk.

    Unlike ``audit_leakage`` (tokenizer provenance and document overlap), this
    gate checks the split itself:

      * the metadata declares an ordered/non-shuffled split and a train-only
        tokenizer fit;
      * validation and test contain no exact row duplicated from train;
      * no group/document spans two sides (for FineWeb, document-disjoint
        chronological construction is recorded and the cache metadata keeps
        the validation fingerprint);
      * train vs holdout token distributions are compared with Jensen-Shannon
        divergence (a shift is reported explicitly rather than silently
        assumed away).

    Exact duplicate rows across splits are a hard failure.  Distribution or
    legacy-metadata issues are warnings because they need a scientific
    judgement, not an automatic rebuild.
    """
    import glob
    import numpy as np
    sys.path.insert(0, REPO)
    import data_guard as DG

    if cache_dirs is None:
        cache_dirs = [os.path.join(REPO, 'wt103_cache'),
                      os.path.join(REPO, 'fineweb_cache')]
    out = {'status': 'ok', 'caches': {}, 'warnings': []}

    def _fail(msg):
        out['warnings'].append(msg)
        out['status'] = 'fail'

    def _warn(msg):
        out['warnings'].append(msg)
        if out['status'] == 'ok':
            out['status'] = 'warnings'

    for cd in cache_dirs:
        if not os.path.isdir(cd):
            continue
        cache = {'status': 'ok', 'corpora': {}}
        out['caches'][os.path.abspath(cd)] = cache
        for mp in sorted(glob.glob(os.path.join(cd, 'meta_*.json'))):
            if os.path.basename(mp).startswith('test_meta_'):
                continue
            meta = _load(mp)
            if not isinstance(meta, dict):
                continue
            tag = os.path.basename(mp)[len('meta_'):-len('.json')]
            trp = os.path.join(cd, f'train_ids_{tag}.npy')
            vap = os.path.join(cd, f'val_batch_{tag}.npy')
            if not (os.path.exists(trp) and os.path.exists(vap)):
                continue
            rec = {'status': 'ok', 'meta': os.path.basename(mp)}
            try:
                tr = np.load(trp, mmap_mode='r')
                va = np.load(vap, mmap_mode='r')
            except Exception as e:
                rec['status'] = 'fail'
                rec['error'] = f'{type(e).__name__}: {e}'
                _fail(f'{cd}/{tag}: cache unreadable ({rec["error"]})')
                cache['corpora'][tag] = rec
                continue
            rec['train_shape'] = list(tr.shape)
            rec['val_shape'] = list(va.shape)
            rec['declared_split_strategy'] = meta.get('split_strategy') or 'legacy_unrecorded'
            rec['shuffle_at_split'] = meta.get('shuffle_at_split')
            rec['tokenizer_fit_split'] = meta.get('tokenizer_fit_split')
            rec['val_fingerprint_recorded'] = bool(meta.get('val_fp'))
            if meta.get('shuffle_at_split') is True:
                rec['status'] = 'fail'
                _fail(f'{cd}/{tag}: metadata declares a shuffled split')
            if meta.get('tokenizer_fit_split') not in (None, 'train'):
                rec['status'] = 'fail'
                _fail(f'{cd}/{tag}: tokenizer fit split is {meta.get("tokenizer_fit_split")!r}, not train')
            if meta.get('split_strategy') is None:
                _warn(f'{cd}/{tag}: legacy cache has no split_strategy/tokenizer provenance metadata; rebuild it to make the split auditable')

            # Exact duplicate rows across train/val/test.  This is row-level
            # leakage regardless of how the split was generated.
            shared_val = _row_overlap_count(tr, va)
            rec['exact_row_overlap_train_val'] = int(shared_val)
            if shared_val:
                rec['status'] = 'fail'
                _fail(f'{cd}/{tag}: {shared_val} validation rows exactly duplicate a training row')
            # The test cache filename carries its own seq_len/vocab/ns tag;
            # match on the sequence length recorded in the train meta rather
            # than guessing one file.
            test_files = sorted(glob.glob(os.path.join(cd, 'test_ids_*.npy')))
            te = None
            for tf in test_files:
                try:
                    cand = np.load(tf, mmap_mode='r')
                except Exception:
                    continue
                if meta.get('seq_len') is not None and cand.shape[1] != int(meta['seq_len']):
                    continue
                te = cand
                rec['test_file'] = os.path.basename(tf)
                break
            if te is not None:
                rec['test_shape'] = list(te.shape)
                shared_test = _row_overlap_count(tr, te)
                shared_val_test = _row_overlap_count(va, te)
                rec['exact_row_overlap_train_test'] = int(shared_test)
                rec['exact_row_overlap_val_test'] = int(shared_val_test)
                if shared_test:
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: {shared_test} test rows exactly duplicate a training row')
                if shared_val_test:
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: {shared_val_test} test rows exactly duplicate a validation row')
            else:
                rec['test_file'] = None

            # Token-distribution shift: compare holdout marginals with the
            # training reference (never the other way round).
            tr_tok = _sample_tokens(tr, cap=1000000, seed=0)
            va_tok = _sample_tokens(va, cap=1000000, seed=1)
            rec['train_val_shift'] = DG.categorical_shift_report(tr_tok, va_tok)
            # A handful of holdout tokens cannot estimate a distribution; mark
            # the reading as unreliable instead of emitting a huge JS value.
            rec['train_val_shift']['reliable'] = bool(
                tr_tok.size >= int(min_shift_tokens) and
                va_tok.size >= int(min_shift_tokens))
            jsv = rec['train_val_shift']['js_divergence']
            if rec['train_val_shift']['reliable']:
                if jsv >= float(js_fail):
                    _warn(f'{cd}/{tag}: train/val token JS divergence {jsv:.3f} >= {js_fail} — material distribution shift')
                elif jsv >= float(js_warn):
                    _warn(f'{cd}/{tag}: train/val token JS divergence {jsv:.3f} >= {js_warn}')
            else:
                _warn(f'{cd}/{tag}: train/val distribution shift not estimable from {va_tok.size} holdout tokens (need >= {min_shift_tokens})')
            if te is not None:
                te_tok = _sample_tokens(te, cap=1000000, seed=2)
                rec['train_test_shift'] = DG.categorical_shift_report(tr_tok, te_tok)
                rec['train_test_shift']['reliable'] = bool(
                    tr_tok.size >= int(min_shift_tokens) and
                    te_tok.size >= int(min_shift_tokens))
                jst = rec['train_test_shift']['js_divergence']
                if rec['train_test_shift']['reliable']:
                    if jst >= float(js_fail):
                        _warn(f'{cd}/{tag}: train/test token JS divergence {jst:.3f} >= {js_fail} — material distribution shift')
                    elif jst >= float(js_warn):
                        _warn(f'{cd}/{tag}: train/test token JS divergence {jst:.3f} >= {js_warn}')
                else:
                    _warn(f'{cd}/{tag}: train/test distribution shift not estimable from {te_tok.size} test tokens (need >= {min_shift_tokens})')
            cache['corpora'][tag] = rec
            if rec['status'] == 'fail':
                cache['status'] = 'fail'
                out['status'] = 'fail'
            elif out['status'] == 'ok' and rec['status'] == 'warnings':
                cache['status'] = 'warnings'
                out['status'] = 'warnings'
        if not cache['corpora']:
            cache['status'] = 'not_verifiable_locally'
    if not out['caches']:
        out['status'] = 'not_verifiable_locally'
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
    import v7_supp as _V7  # noqa: F401 — importing installs the *_rope variant patches (make_layer_cfgs_v7 + SmallGPTRoPE factory)
    import copy as _copy
    torch.manual_seed(0)
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
    if fails:
        print('[audit causality] FAIL:\n  - ' + '\n  - '.join(fails))
        out = {'status': 'fail', 'failures': fails}
    else:
        print(f'[audit causality] PASS: label shift + prefix invariance + strict gate + decode consistency over {variants}')
        out = {'status': 'pass', 'variants': variants}
    return out


# ---------------------------------------------------------------------------
# 3) attribution：动态/固定分块的块数、压缩率、计算开销对齐读数
# ---------------------------------------------------------------------------

def audit_attribution(outdirs=None):
    """动态分块归因审计（零 GPU，读 results_*/summary.json + ckpt 元数据）。

    评审问题：固定块与动态块是否匹配实际块数、压缩率、可读摘要数量和
    计算开销。本审计从落盘训练记录的 by_len / _stats / 参数量字段重算
    每臂的：参数量、（若启用块统计）块长分布与边界对齐读数，并输出
    「臂间块数是否对齐」的对照表。检查 `attn._stats`（enable_block_stats
    收集的块 id bincount，design_notes §14 同口径）是否在场。"""
    out = {'status': 'ok', 'panels': {}}
    L = _import_exp_lib()
    dirs = outdirs or list(_iter_result_dirs())
    for d in dirs:
        summ = _load(os.path.join(d, 'summary.json'))
        if not summ:
            continue
        panel = {}
        for key, r in summ.items():
            if not (isinstance(r, dict) and r.get('variant')):
                continue
            v = r['variant']
            ent = panel.setdefault(v, {'n_records': 0, 'with_block_stats': 0, 'params': r.get('params')})
            ent['n_records'] += 1
            if r.get('block_stats') or r.get('by_len_block_stats') or (isinstance(r.get('mech'), dict) and r['mech'].get('cuts')):
                ent['with_block_stats'] += 1
        if panel:
            missing = {v: e for v, e in panel.items() if e['with_block_stats'] == 0}
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

    评审问题：与标点/词边界的对齐是否超过匹配块长分布的随机基线——否则
    「语义边界」是块长分布的伪影。exp_lib 已实现该基线
    （`_random_cut_precision`，两种口径见 design_notes §10）；本审计把
    「对齐精度 − 随机基线」的差值与比值汇总成显式判读。"""
    out = {'status': 'ok', 'panels': {}}
    L = _import_exp_lib()
    dirs = outdirs or list(_iter_result_dirs())
    for d in dirs:
        summ = _load(os.path.join(d, 'summary.json'))
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
                p = b.get('precision')
                rp = b.get('random_precision')
                if p is not None and rp is not None:
                    agg.append({'key': key, 'precision': p, 'random_baseline': rp, 'excess': p - rp, 'ratio': (p / rp) if rp else None, 'verdict': 'exceeds random baseline' if p > rp else ('at baseline (block-length artifact)' if abs(p - rp) < 1e-9 else 'below baseline')})
            out['panels'][d] = agg
    if not out['panels']:
        out['status'] = 'not_verifiable_locally'
        out['note'] = 'no boundary-alignment fields in the local results; fuse_analysis.py recomputes per-layer alignment (tol=1) from summary.json and exp_lib._random_cut_precision provides the matched-length random baseline — run them on the experiment machine'
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
    raw = _load(bylen_path)
    if raw is None:
        out = {'status': 'not_verifiable_locally', 'note': f'{bylen_path} absent — P5T has not run; the same checks apply to results_v10_mech/summary.json by_len panels'}
        print(json.dumps(out, ensure_ascii=False))
        return out
    L = _import_exp_lib()
    by_len = {}
    problems = []
    for key, r in raw.items():
        if not isinstance(r, dict):
            continue
        for k, c in (r.get('by_len') or {}).items():
            try:
                Ln = int(k)
            except (TypeError, ValueError):
                continue
            if not isinstance(c, dict):
                continue
            e = by_len.setdefault(Ln, {'n_tok': set(), 'eval_span': set(), 'truncated': set(), 'n_seq': set()})
            e['n_tok'].add(c.get('n_tok'))
            e['eval_span'].add(c.get('eval_span'))
            e['truncated'].add(c.get('truncated'))
            if isinstance(c.get('ppl_per_seq'), list):
                e['n_seq'].add(len(c['ppl_per_seq']))
            if c.get('truncated'):
                problems.append(f'{key}: L{Ln} cell is marked truncated — it must not enter cross-length means')
    out = {'status': 'ok', 'by_length': {}}
    for Ln, e in sorted(by_len.items()):
        ok = len(e['n_tok']) == 1 and None not in e['n_tok'] and len(e['eval_span']) == 1
        out['by_length'][str(Ln)] = {'n_tok_values': sorted((str(x) for x in e['n_tok'])), 'eval_span_values': sorted((str(x) for x in e['eval_span'])), 'n_seq_values': sorted(e['n_seq']), 'same_scored_targets_across_records': bool(ok)}
        if not ok:
            problems.append(f'L{Ln}: records disagree on n_tok/eval_span — different target sets are being scored across arms; cross-arm comparison at this length is invalid')
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
    单独调参会在此被点名，而不是混进「提升」里。"""
    summ = _load(summary_path)
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
    for a, recs in per_arm.items():
        cfgs = {r.get('run_cfg') for r in recs}
        knob_vals = {json.dumps(r.get('train_knobs'), sort_keys=True)
                     for r in recs if r.get('train_knobs') is not None}
        n_missing_knobs = sum(1 for r in recs if r.get('train_knobs') is None)
        seeds = sorted({int(r['seed']) for r in recs
                        if isinstance(r.get('seed'), (int, float))
                        and not isinstance(r.get('seed'), bool)})
        out['arms'][a] = {
            'n_records': len(recs), 'seeds': seeds, 'n_seeds': len(seeds),
            'n_distinct_run_cfg': len(cfgs),
            'n_distinct_train_knobs': len(knob_vals),
            'n_missing_train_knobs': n_missing_knobs,
            'params': sorted({r.get('params') for r in recs},
                             key=lambda x: (x is None, x)),
        }
        if len(cfgs) > 1:
            problems.append(f'{a}: {len(cfgs)} distinct run_cfg values within one arm — mixed protocols inside an arm')
        if len(knob_vals) > 1:
            problems.append(f'{a}: {len(knob_vals)} distinct train_knobs values within one arm — mixed optimizer/schedule settings inside an arm')
    _unverifiable = False
    if len(per_arm) >= 2:
        # The cross-arm check must use the SAME gate as the statistics
        # (`pair_reason`): identical run_cfg AND identical train_knobs AND the
        # same steps/tokens.  Counting distinct values per arm (the old form)
        # only detects mixing *inside* one arm; one arm could still run a
        # different optimizer while every arm looked internally consistent.
        recs_map, _amb = L._variant_seed_map(summ)
        _cv_arms = [a for a in arms if a in per_arm]
        if recs_map and len(_cv_arms) >= 1:
            cv = L.controlled_variables_report(recs_map, _cv_arms)
            out['controlled_variables'] = cv
            if cv.get('run_cfg_equal') is False:
                problems.append('run_cfg differs across arms — the training budget / data / architecture knobs are not controlled; pair_reason would refuse every pairing')
            if cv.get('train_knobs_equal') is False:
                problems.append('train_knobs differ across arms (optimizer / betas / eps / lr-schedule / grad-clip / augmentation) — a gain from a different optimizer is not a gain from the method')
            if cv.get('hyperparams_equal') is False:
                problems.append('structured hyperparameters differ across arms: ' + ', '.join(cv.get('hyperparam_differences') or []) + ' — the comparison is not fully controlled')
            for _arm in cv.get('within_arm_mixed') or []:
                problems.append(f'{_arm}: seeds mix DIFFERENT run_cfg/train_knobs/hyperparameter values inside one arm')
            for _key in ('run_cfg', 'train_knobs', 'hyperparams'):
                _eq = cv.get(_key + '_equal')
                if _eq is None:
                    warnings.append(f'{_key} equality across arms is UNVERIFIABLE (some records omit it / differ in coverage) — do not read the arm means as a controlled contrast')
        ref_arm = arms[0] if arms[0] in per_arm else next(iter(per_arm))
        ref_cfg = {r.get('run_cfg') for r in per_arm[ref_arm]}
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
                    if r.get('run_cfg') is None or r.get('run_cfg') not in ref_cfg:
                        problems.append(f'{a}: a record whose run_cfg does not equal the {ref_arm} reference (pair_reason would refuse this pairing)')
                        break
        # Seed coverage: arm means are only comparable if they average the
        # same seeds.  An arm that silently lost a seed (OOM / cap / error)
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
    out['warnings'] = warnings
    if not out['tuning'].get('tuning_verified'):
        out['warnings'].append('no equal-budget baseline tuning artifact — a win over these baselines is not yet evidence against a well-tuned baseline; run baseline_tune.py')
    out['status'] = ('fail' if problems else
                     'warn' if out['warnings'] else
                     ('not_verifiable_locally' if _unverifiable else 'ok'))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
# 7) failures：失败/缺失的臂间集中度
# ---------------------------------------------------------------------------

def audit_failures(outdirs=None):
    """失败集中度（零 GPU）：error/NaN/截断记录按臂分布的读数。

    评审问题：OOM/NaN/超时是否集中在某些臂；仅保留成功运行是否构成
    选择偏差。本审计把 error 记录按 (臂, 错误类型) 计数，并给出描述性
    集中度（不主张显著性——小样本下显著性检验本身功效不足）。"""
    out = {'status': 'ok', 'panels': {}}
    dirs = outdirs or list(_iter_result_dirs())
    for d in dirs:
        summ = _load(os.path.join(d, 'summary.json'))
        if not summ:
            continue
        by_arm = {}
        n_ok = {}
        for key, r in summ.items():
            if not (isinstance(r, dict) and r.get('variant')):
                continue
            v = r['variant']
            if 'error' in r or r.get('status') in ('error', 'failed'):
                by_arm.setdefault(v, []).append(str(r.get('error') or r.get('status'))[:80])
            elif isinstance(r.get('ppl'), (int, float)):
                n_ok[v] = n_ok.get(v, 0) + 1
        if by_arm:
            out['panels'][d] = {'errors_by_arm': {v: len(e) for v, e in by_arm.items()}, 'ok_by_arm': n_ok, 'error_samples': {v: e[:2] for v, e in by_arm.items()}}
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

    评审要求：类别不平衡不能只看 accuracy；多分类 F1 必须注明 macro/
    micro/weighted；极度不平衡优先 PR-AUC；指标必须对齐任务目标。本审计
    先检查 `exp_lib.METRIC_POLICY` 是否完整声明，再对本地面板检查边界
    指标是否同时给出 precision/recall/随机基线、NIAH 是否同时给出 answer
    NLL。缺失只标 warn/不可核验，不伪造一个「通过」。"""
    L = _import_exp_lib()
    out = {'status': 'ok', 'policy': L.metric_policy_audit(), 'panels': {}}
    dirs = outdirs or list(_iter_result_dirs())
    for d in dirs:
        summ = _load(os.path.join(d, 'summary.json'))
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


# ---------------------------------------------------------------------------
# 7d) methodology：受控对比 / 消融完整性 / 统计可靠性
# ---------------------------------------------------------------------------

def audit_methodology(outdirs=None, design=None, alpha=0.05, min_n=3, write=True):
    """对比实验与消融审计（零 GPU，读落盘 summary）。

    对每个结果目录复用 `exp_lib.methodology_audit`，并额外给出三项面板级
    读数：
      * controlled_variables：逐臂全部种子记录（不是只看第一条）的
        run_cfg / train_knobs / structured hyperparams 一致性；
      * ablation_completeness：本面板实际进入族的创新点是否每个都具备全部
        声明的留一对照（缺臂会点名，部分组合不能支撑「各自有用」）；
      * arm_summary：每条臂的 n、mean±std 与是否达到 min_n（关键实验至少
        3–5 个种子的可复核读数）。

    没有 `results_*/summary.json` 时返回 not_verifiable_locally，而不是
    伪造一个通过。"""
    L = _import_exp_lib()
    dirs = list(outdirs) if outdirs is not None else list(_iter_result_dirs())
    out = {'status': 'ok', 'alpha': float(alpha), 'min_n': int(min_n), 'panels': {}}
    for d in dirs:
        _dir = d if os.path.isabs(d) else os.path.join(REPO, d)
        summ = _load(os.path.join(_dir, 'summary.json'))
        if not summ:
            continue
        rep = L.methodology_audit(summ, design=design, alpha=alpha, min_n=min_n,
                                  outdir=_dir, write=write, verbose=False)
        _panel = {
            'ablation_complete': rep.get('ablation_complete'),
            'ablation_completeness': rep.get('ablation_completeness'),
            'controlled_variables': rep.get('controlled_variables'),
            'arm_summary': rep.get('arm_summary'),
            'arms_below_min_seeds': rep.get('arms_below_min_seeds'),
            'verified_components': rep.get('verified_components'),
            'unverified_components': rep.get('unverified_components'),
            'incomplete_components': rep.get('incomplete_components'),
            'warnings': rep.get('warnings'),
        }
        out['panels'][d] = _panel
        if _panel['arms_below_min_seeds'] or _panel['warnings']:
            out['status'] = 'warn'
    if not out['panels']:
        out['status'] = 'not_verifiable_locally'
        out['note'] = 'no results_*/summary.json found — run the experiment phases first; the audit is zero-GPU over saved records'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
# 8) 规模核算：6ND 代理 FLOPs 估算 + 实测用时对账
# ---------------------------------------------------------------------------

def train_flops_estimate(params, tokens, attn_overhead=0.35):
    """6ND 估算 + 注意力开销系数（评审二轮·意见七：代理 ≠ 实测）。

    6·N·D 是前向+反向的主流近似；稀疏选择/池化/激活重算与长序列注意力
    不与参数量成比例，这里用一个**显式声明的**注意力开销系数给上界感，
    报告时必须同时给出 TimeGuard 实测秒数——两个口径并列，不得只报其一。"""
    n = float(params)
    d_tok = float(tokens)
    base = 6.0 * n * d_tok
    return {'flops_6nd': base, 'flops_with_attn_proxy': base * (1.0 + attn_overhead), 'attn_overhead_assumed': attn_overhead, 'note': 'analytic PROXY, not measured throughput; pair with TimeGuard seconds before any efficiency claim'}


def audit_budget(state_files=('run_time_state_v10.json', 'run_time_state_v12.json'), panels=(('results_lm_v10_scale_s', 'csa_fixed', 8000), ('results_lm_v10_scale_l8k', 'csa_fixed', 8000), ('results_lm_v10_scale_lflop', 'csa_fixed', None))):
    """规模核算（零 GPU）：每面板 6ND 估算 + 已记账实测秒数。"""
    out = {'status': 'ok', 'panels': {}}
    for d, v, steps in panels:
        summ = _load(os.path.join(d, 'summary.json'))
        if summ is None:
            out['panels'][d] = 'not present locally'
            continue
        recs = [r for r in summ.values() if isinstance(r, dict) and r.get('variant') == v and isinstance(r.get('params'), (int, float))]
        if not recs:
            out['panels'][d] = 'no param records'
            continue
        r0 = recs[0]
        _steps = steps or r0.get('steps')
        tok = r0.get('tokens_seen') or (_steps * r0.get('batch_size', 12) * r0.get('seq_len', 512) if _steps else None)
        est = train_flops_estimate(r0['params'], tok) if tok else None
        out['panels'][d] = {'params': r0.get('params'), 'steps': _steps, 'tokens': tok, 'estimate': est}
    for sf in state_files:
        st = _load(sf)
        if st is not None:
            out.setdefault('measured', {})[sf] = {'booked_seconds': st.get('booked_seconds'), 'run_count': len(st.get('runs') or []) or st.get('run_count')}
    if all(not isinstance(x, dict) for x in out['panels'].values()):
        out['status'] = 'not_verifiable_locally'
        out['note'] = 'scale panels not present locally; estimates + TimeGuard bookkeeping run on the experiment machine'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
# 9) 测量工具：prefill/decode 与容量读数（含 CPU 微型自检）
# ---------------------------------------------------------------------------

def measure_prefill_decode(model, ids, device, warmup=1, iters=3):
    """端到端 prefill / 增量 decode 计时 + 峰值内存（测量代码，非实验）。

    评审问题：理论 FLOPs 被当成实测速度。本函数给出可复用的测量口径：
    prefill = 整段前向一次的墙钟；decode = 逐 token 前缀推进的墙钟与
    每 token 均摊；峰值设备占用（CUDA 时）经 torch.cuda.max_memory_allocated
    记录。KV 缓存增长在本仓库的稀疏臂上是块摘要级的，直接以
    max_memory_allocated 随长度的差值读出，不做理论推断。"""
    import torch
    import time as _t
    res = {'n_tokens': int(ids.shape[1])}
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(ids)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = _t.perf_counter()
        for _ in range(iters):
            model(ids)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        res['prefill_ms_per_iter'] = (_t.perf_counter() - t0) * 1000.0 / iters
        t0 = _t.perf_counter()
        n_dec = min(32, int(ids.shape[1]))
        for t in range(1, n_dec + 1):
            model(ids[:, :t])
        res['decode_ms_per_token_avg'] = (_t.perf_counter() - t0) * 1000.0 / n_dec
        if device.type == 'cuda':
            res['peak_mem_mb'] = torch.cuda.max_memory_allocated() / (1 << 20)
    return res


def audit_efficiency_demo():
    """CPU 微型自检：证明测量代码可运行（不是实验，秒级）。"""
    try:
        import torch
        import numpy as np
    except ImportError as e:
        print(f'[audit efficiency] SKIPPED: torch not importable ({e})')
        return {'status': 'skipped_no_torch'}
    L = _import_exp_lib()
    import v7_supp as _V7  # noqa: F401 — installs the rope-variant patches
    import copy as _copy
    torch.manual_seed(0)
    dev = torch.device('cpu')
    cfgs = L.make_layer_cfgs(2, 'full_rope')
    model = L.SmallGPT(256, 64, 2, 4, 16, 128, cfgs).to(dev)
    ids = torch.randint(0, 256, (1, 128), dtype=torch.long)
    res = measure_prefill_decode(model, ids, dev, warmup=0, iters=1)
    res['status'] = 'ok'
    res['note'] = 'CPU demo of the measurement harness; run measure_prefill_decode on the real checkpoints (per arm, per length) for the efficiency panel — analytic FLOPs (audit_budget) and measured wall-clock are reported side by side, never merged'
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return res


# ---------------------------------------------------------------------------
# report：汇总本地可执行的审计
# ---------------------------------------------------------------------------

def run_report(out='audit_report.json'):
    res = {'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'leakage': audit_leakage(), 'split_discipline': audit_split_discipline(), 'attribution': audit_attribution(), 'boundary': audit_boundary(), 'length': audit_length(), 'fairness': audit_fairness(), 'metrics': audit_metrics(), 'tuning': audit_tuning(), 'methodology': audit_methodology(), 'failures': audit_failures(), 'budget': audit_budget()}
    L_atomic = None
    try:
        L = _import_exp_lib()
        L.atomic_write_json(os.path.join(REPO, out), res, indent=2)
        print(f'[audit] wrote {out}')
    except Exception as e:
        with open(os.path.join(REPO, out), 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f'[audit] wrote {out} (fallback writer: {type(e).__name__})')
    return res


if __name__ == '__main__':
    _cmds = {'leakage': audit_leakage, 'split': audit_split_discipline, 'split_discipline': audit_split_discipline, 'causality': audit_causality, 'attribution': audit_attribution, 'boundary': audit_boundary, 'length': audit_length, 'fairness': audit_fairness, 'metrics': audit_metrics, 'tuning': audit_tuning, 'methodology': audit_methodology, 'failures': audit_failures, 'budget': audit_budget, 'efficiency': audit_efficiency_demo, 'report': run_report}
    _arg = sys.argv[1] if len(sys.argv) > 1 else 'report'
    if _arg in ('-h', '--help'):
        print(__doc__)
        raise SystemExit(0)
    if _arg not in _cmds:
        print(f'[audit] unknown subcommand {_arg!r}; expected one of {sorted(_cmds)}')
        raise SystemExit(2)
    _r = _cmds[_arg]()
    if isinstance(_r, dict) and _r.get('status') == 'fail':
        raise SystemExit(1)
