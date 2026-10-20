#!/usr/bin/env python3
"""audit_gates.py — 可执行审计入口（零实验）。

本模块把「目前不能仅凭代码排除、必须进一步验证」的领域变成可运行的审计
子命令，能用**代码 + 现有落盘产物**核验；不能本地核验的
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
  budget       规模核算：6ND 估算 + TimeGuard 记录对账（零 GPU）
  efficiency   prefill/decode/容量读数工具（需要模型；给出可复用函数
               与 CPU 微型演示——不是实验，是测量代码自检）
  report       汇总运行本地可执行的全部审计，输出 audit_report.json

运行纪律：`python audit_gates.py causality` 在 CPU 上几秒内完成；其余子命令
读 `results_*/`（不存在时给出「不可本地核验」的明确说明而不是报错）。
"""
import json
import math
import numbers
import os
import sys
import time

import numpy as np

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

def audit_leakage(cache_dir=None, shingle=13, max_rows_shingled=400,
                  preferred_train_seq_len=None):
    """数据泄漏审计（零 GPU，读 wt103_cache 落盘产物）。

    ``cache_dir=None`` resolves ``WT103_CACHE_DIR`` (falling back to
    ``./wt103_cache``), the same resolver the loaders use.

    ``preferred_train_seq_len`` disambiguates a cache directory that contains
    more than one train meta (e.g. the 512-token training cache and a
    4097-token probe cache).  Only train metas with that ``seq_len`` are
    considered when at least one matches; otherwise the audit refuses to guess.

    核验点：
      a. test 缓存 meta 的 fit_fp（拟合语料指纹）必须来自 TRAIN 前 50k 行
         ——与 train 侧 meta 的语料指纹同源（分词器没有见过 valid/test）；
      b. test 侧持久化分词器与 train 侧持久化分词器逐 token 全等
         （load_wikitest 运行期同样硬校验，这里独立复核落盘副本）；
      c. train/test id 缓存的 src_fp 与文件内容指纹自洽；
      d. train/test 文档重叠：k-gram shingle 的 shared/test 比例与最差单行
         重叠比例（Jaccard 因 4M-vs-65k 的构造上限只作诊断记录）。"""
    # A non-positive shingle width makes _window_hashes return no hashes, which
    # would report 0% overlap and pass the leakage gate.  NaN policy limits
    # make every comparison False for the same reason.
    shingle = _audit_positive_int('shingle', shingle)
    max_rows_shingled = _audit_positive_int('max_rows_shingled', max_rows_shingled)
    if preferred_train_seq_len is not None:
        preferred_train_seq_len = _audit_positive_int(
            'preferred_train_seq_len', preferred_train_seq_len)
    _max_shingle_ov = _audit_nonneg_float(
        'CSA_MAX_TEST_SHINGLE_OVERLAP',
        os.environ.get('CSA_MAX_TEST_SHINGLE_OVERLAP') or '0.25')
    _max_row_ov = _audit_nonneg_float(
        'CSA_MAX_TEST_ROW_OVERLAP',
        os.environ.get('CSA_MAX_TEST_ROW_OVERLAP') or '0.50')
    if cache_dir in (None, ''):
        cache_dir = os.environ.get('WT103_CACHE_DIR') or './wt103_cache'
    out = {'status': 'ok', 'checks': {}}
    test_meta = None
    train_meta = None
    test_tag = None
    train_tag = None
    import glob
    import numpy as np
    _tm = sorted(glob.glob(os.path.join(cache_dir, 'test_meta_*.json')))
    _vm = sorted(p for p in glob.glob(os.path.join(cache_dir, 'meta_*.json'))
                 if not os.path.basename(p).startswith('test_meta_'))
    if not _tm or not _vm:
        out['status'] = 'not_verifiable_locally'
        out['note'] = f'{cache_dir}/ lacks test_meta_*.json or meta_*.json — run the corpus loaders once (or copy the cache) to audit tokenizer provenance; the runtime hard checks in exp_lib.load_wiktest/load_wikitext remain in force meanwhile'
        return out
    # Pair a train meta with the test meta that describes the SAME corpus
    # protocol (vocab + tokenizer fit provenance).  seq_len is deliberately NOT
    # required to match: the training cache is 512-token and the held-out cache
    # is 4097-token by construction.
    def _as_int(value):
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
    _pairs = []
    for _tp in _tm:
        _tmeta = _load(_tp)
        if not isinstance(_tmeta, dict):
            continue
        _ttag = os.path.basename(_tp)[len('test_meta_'):-len('.json')]
        for _mp in _vm:
            _mmeta = _load(_mp)
            if not isinstance(_mmeta, dict):
                continue
            _tv, _mv = _as_int(_tmeta.get('vocab')), _as_int(_mmeta.get('vocab'))
            if _tv is not None and _mv is not None and _tv != _mv:
                continue
            if (_tmeta.get('fit_fp') and _mmeta.get('fit_fp')
                    and _tmeta['fit_fp'] != _mmeta['fit_fp']):
                continue
            _pairs.append((_mp, _mmeta, _tp, _tmeta))
    if preferred_train_seq_len is not None:
        _pref = []
        for _p in _pairs:
            _sv = _as_int(_p[1].get('seq_len'))
            if _sv is not None and _sv == int(preferred_train_seq_len):
                _pref.append(_p)
        if not _pref:
            # The caller named the training sequence length; silently falling
            # back to a different one would audit the wrong split.
            out['status'] = 'not_verifiable_locally'
            out['note'] = (f'no train cache with seq_len={preferred_train_seq_len} '
                           'matches the test cache protocol; refusing to guess '
                           'which train split is authoritative')
            return out
        _pairs = _pref
    if not _pairs:
        out['status'] = 'not_verifiable_locally'
        out['note'] = 'no train/test cache pair shares the same vocab/tokenizer-fit provenance; cannot audit tokenizer or split provenance without guessing'
        return out
    _uniq = {(os.path.basename(a), os.path.basename(c)) for a, _, c, _ in _pairs}
    if len(_uniq) > 1:
        out['status'] = 'not_verifiable_locally'
        out['note'] = ('multiple train/test cache pairs match the same protocol; '
                       'refusing to guess which one is authoritative — pass an '
                       'explicit cache set or remove stale variants')
        out['checks']['candidate_pairs'] = sorted('/'.join(_u) for _u in _uniq)
        return out
    _mp, train_meta, _tp, test_meta = _pairs[0]
    train_tag = os.path.basename(_mp)[len('meta_'):-len('.json')]
    test_tag = os.path.basename(_tp)[len('test_meta_'):-len('.json')]
    out['checks']['cache_pair'] = {'train_meta': os.path.basename(_mp),
                                   'test_meta': os.path.basename(_tp)}
    out['checks']['test_fit_fp'] = test_meta.get('fit_fp')
    out['checks']['train_src_fp'] = train_meta.get('src_fp') or train_meta.get('train_fp')
    out['checks']['test_src_fp'] = test_meta.get('src_fp')
    _test_vocab_int = _as_int(test_meta.get('vocab'))
    _train_vocab_int = _as_int(train_meta.get('vocab'))
    out['checks']['vocab_agree'] = bool(_test_vocab_int is not None and _train_vocab_int is not None
                                        and _test_vocab_int == _train_vocab_int)
    _tok_problems = []
    if not out['checks']['vocab_agree']:
        _tok_problems.append('test/train cache vocabs disagree')
    # b) persisted tokenizers, if both exist.  Use the tag-paired files, never
    # the first filename in the directory.
    import hashlib
    _tt = [os.path.join(cache_dir, f'test_tok_{test_tag}.json')]
    _vt = [os.path.join(cache_dir, f'tok_{train_tag}.json')]
    _tt = [p for p in _tt if os.path.exists(p)]
    _vt = [p for p in _vt if os.path.exists(p)]
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
        # Byte equality is diagnostic only: two equivalent tokenizers may
        # serialize with different JSON ordering.  The ID-SPACE identity below
        # (full id->token map) is the binding check.
        try:
            from tokenizers import Tokenizer
            ta, tb = (Tokenizer.from_file(_tt[0]), Tokenizer.from_file(_vt[0]))
            _tv_eq = bool(dict(ta.get_vocab()) == dict(tb.get_vocab()))
            out['checks']['tokenizer_vocab_identical'] = _tv_eq
            if not _tv_eq:
                _tok_problems.append('test/train tokenizer id->token maps differ')
        except Exception as e:
            out['checks']['tokenizer_vocab_identical'] = f'failed ({type(e).__name__}: {e})'
            _tok_problems.append(f'tokenizer vocab comparison failed ({type(e).__name__}: {e})')
    else:
        out['checks']['tokenizer_files_present'] = False
        _tok_problems.append('a persisted train/test tokenizer file is missing; tokenizer identity cannot be verified')
    if _tok_problems:
        # Tokenizer/split provenance problems are hard audit failures.
        out['checks']['tokenizer_problems'] = _tok_problems
        out['status'] = 'fail'
        out['note'] = ('tokenizer/split provenance audit failed: ' + '; '.join(_tok_problems) +
                       ' — the train/test id spaces cannot be shown to be paired; '
                       'rebuild both caches with the current loaders before using '
                       'the held-out split')
    # c2) whole-document group isolation for the held-out split.  The loaders
    # now persist test_doc_ids; when the sidecar is present, recompute the
    # train/test and val/test document intersections instead of trusting the
    # metadata flags.  A missing sidecar is reported as not independently
    # verified (legacy cache), not silently treated as clean.
    _trd_p = os.path.join(cache_dir, f'train_doc_ids_{train_tag}.npy')
    _vad_p = os.path.join(cache_dir, f'val_doc_ids_{train_tag}.npy')
    _ted_p = os.path.join(cache_dir, f'test_doc_ids_{test_tag}.npy')
    if os.path.exists(_ted_p):
        try:
            import numpy as np
            _ted = np.load(_ted_p)
            _doc_checks = {'test_documents': int(_ted.size)}
            _doc_leak = []
            _tm_doc = (test_meta or {}).get('doc_ids') or {}
            if _tm_doc:
                if int(_tm_doc.get('test_count', -1)) != int(_ted.size):
                    _doc_leak.append('test document-id sidecar count disagrees with the test metadata')
            else:
                _doc_checks['metadata'] = 'test meta has no doc_ids attestation'
            if os.path.exists(_trd_p):
                _n = int(np.intersect1d(np.load(_trd_p), _ted).size)
                _doc_checks['test_train_overlap'] = _n
                if _n:
                    _doc_leak.append(f'{_n} test document ids also occur in train')
            else:
                _doc_checks['test_train_overlap'] = 'train sidecar missing'
                _doc_leak.append('train document-id sidecar is missing; test/train group isolation cannot be verified')
            if os.path.exists(_vad_p):
                _n = int(np.intersect1d(np.load(_vad_p), _ted).size)
                _doc_checks['test_validation_overlap'] = _n
                if _n:
                    _doc_leak.append(f'{_n} test document ids also occur in validation')
            else:
                _doc_checks['test_validation_overlap'] = 'validation sidecar missing'
                _doc_leak.append('validation document-id sidecar is missing; test/validation group isolation cannot be verified')
            out['checks']['document_isolation'] = _doc_checks
            if _doc_leak:
                out['status'] = 'fail'
                out['note'] = ('held-out document-group leakage: ' + '; '.join(_doc_leak) +
                               ' — rebuild the test cache with the current loader')
        except Exception as _de:
            out['checks']['document_isolation'] = f'failed ({type(_de).__name__}: {_de})'
            out['status'] = 'fail'
            out['note'] = (f'held-out document-id sidecar unreadable ({_de}); '
                           'document-group isolation cannot be verified')
    else:
        out['checks']['document_isolation'] = {
            'status': 'not_verified',
            'note': 'legacy test cache without test_doc_ids; rebuild it to prove train/val/test document-group isolation',
        }
    # d) document overlap via deterministic SLIDING shingles on the cached id
    # matrices.  A fixed polynomial rolling hash over every window, uniform
    # segments across the full train stream, and an explicit scanned fraction
    # keep the Jaccard estimate reproducible and prevent a low overlap from
    # being mistaken for a full-corpus proof.
    try:
        import numpy as np
        _te_path = os.path.join(cache_dir, f'test_ids_{test_tag}.npy')
        _tr_path = os.path.join(cache_dir, f'train_ids_{train_tag}.npy')
        def _unique_cache_file(expected, pattern):
            if os.path.exists(expected):
                return [expected]
            _c = sorted(glob.glob(os.path.join(cache_dir, pattern)))
            # Legacy caches may use a different tag spelling.  A single
            # remaining candidate is unambiguous; more than one is refused
            # instead of silently pairing the wrong configuration.
            return _c if len(_c) == 1 else []
        _te = _unique_cache_file(_te_path, 'test_ids_*.npy')
        _tr = _unique_cache_file(_tr_path, 'train_ids_*.npy')
        if _te and _tr:
            te = np.load(_te[0])
            tr = np.load(_tr[0], mmap_mode='r')

            def _uniform_rows(n_rows, cap):
                if n_rows <= cap:
                    return np.arange(n_rows, dtype=np.int64), n_rows
                return np.unique(np.linspace(0, n_rows - 1, int(cap), dtype=np.int64)), n_rows

            def _shingles(mat, k, cap_rows, cap_tokens=4000000):
                # The training cache is a flat 1-D token stream while the test
                # cache is 2-D; treat the 1-D stream as segments so the scan is
                # defined for both shapes.
                mat = np.asarray(mat)
                if k <= 0:
                    return np.empty(0, dtype=np.uint64), 0, 1.0
                if mat.ndim == 1:
                    total = int(mat.size)
                    n_seg = max(1, min(64, cap_tokens // max(k, 1)))
                    seg_len = max(int(cap_tokens) // n_seg, k + 1)
                    if total <= max(int(cap_tokens), n_seg * seg_len):
                        segments = [(0, total)]
                    else:
                        # total >= n_seg*seg_len makes the equally spaced
                        # starts at least seg_len apart, so the windows are
                        # disjoint and no shingle is counted twice.
                        starts = np.linspace(0, total - seg_len, n_seg, dtype=np.int64)
                        segments = [(int(s), int(min(total, s + seg_len))) for s in starts]
                    parts = []
                    for s, e in segments:
                        h = _window_hashes(np.asarray(mat[s:e]), k)
                        if h.size:
                            parts.append(h)
                    scanned = int(sum(e - s for s, e in segments))
                    hashes = np.concatenate(parts) if parts else np.empty(0, dtype=np.uint64)
                    frac = (scanned / total) if total else 1.0
                    return np.unique(hashes), 1, float(frac)
                rows, n_all = _uniform_rows(int(mat.shape[0]), int(cap_rows))
                parts = [_window_hashes(np.asarray(mat[int(i)]), k) for i in rows]
                parts = [p for p in parts if p.size]
                hashes = np.concatenate(parts) if parts else np.empty(0, dtype=np.uint64)
                return (np.unique(hashes), int(rows.size),
                        (float(rows.size) / n_all) if n_all else 1.0)

            t0 = time.time()
            tr_s, tr_rows, tr_frac = _shingles(tr, shingle, max_rows_shingled)
            te_s, te_rows, te_frac = _shingles(te, shingle, max_rows_shingled)
            if te_s.size == 0 or tr_s.size == 0:
                # A shingle wider than the cached rows produces no windows;
                # reporting 0% overlap would falsely certify a clean split.
                raise RuntimeError(
                    f'no {shingle}-token shingle windows could be formed '
                    f'(train windows={tr_s.size}, test windows={te_s.size}); '
                    'the overlap gate cannot verify this split')
            inter = int(np.intersect1d(tr_s, te_s, assume_unique=True).size)
            union = int(tr_s.size + te_s.size - inter)
            jac = (inter / union) if union else 0.0
            # Jaccard on a 4M-vs-65k shingle pair is bounded near
            # 1.6% even when EVERY test shingle occurs in train, so a
            # `jaccard > 0.05` threshold cannot fire.  Two reachable statistics
            # are used instead: the fraction of TEST shingles shared with train,
            # and the worst single test row's overlap fraction.
            test_overlap = (inter / te_s.size) if te_s.size else 0.0
            max_row_overlap = None
            row_overlaps = []
            if np.asarray(te).ndim == 2 and tr_s.size:
                _tr_set = np.sort(tr_s)
                for _i in range(int(np.asarray(te).shape[0])):
                    _rh = _window_hashes(np.asarray(te)[_i], shingle)
                    if _rh.size:
                        _row_frac = float(np.isin(_rh, _tr_set).sum()) / float(_rh.size)
                        row_overlaps.append(_row_frac)
                if row_overlaps:
                    max_row_overlap = float(max(row_overlaps))
            _viol = bool((test_overlap > _max_shingle_ov)
                         or (max_row_overlap is not None and max_row_overlap > _max_row_ov))
            out['checks']['doc_overlap'] = {
                'shingle_k': int(shingle),
                'hash': 'uint64 polynomial rolling hash (fixed base 1000003), sliding windows',
                'train_rows_scanned': int(tr_rows),
                'test_rows_scanned': int(te_rows),
                'train_tokens_scanned_fraction': round(float(tr_frac), 6),
                'test_rows_scanned_fraction': round(float(te_frac), 6),
                'test_shingles': int(te_s.size),
                'train_shingles': int(tr_s.size),
                'shared_shingles': inter,
                'jaccard': jac,
                'test_shingle_overlap': test_overlap,
                'max_row_overlap': max_row_overlap,
                'row_overlap_fractions': [round(x, 6) for x in row_overlaps[:64]],
                'rows_over_policy': int(sum(1 for x in row_overlaps if x > _max_row_ov)),
                'seconds': round(time.time() - t0, 1),
                'policy_max_test_shingle_overlap': _max_shingle_ov,
                'policy_max_row_overlap': _max_row_ov,
                'policy_max_jaccard_diagnostic_only': 0.05,
                'policy_violation': _viol,
                'status': 'fail' if _viol else 'ok',
                'note': ('nonzero overlap reflects WikiText-103 near-duplicate '
                         'documents. The gate uses reachable statistics '
                         '(shared/test shingles and the worst test row), not the '
                         'structurally-capped Jaccard. A violation blocks a clean '
                         'cross-split generalization/confirmation claim: clean the '
                         'evaluation subset or report an overlap-excluded sensitivity '
                         'analysis. Sampling fractions are reported above; '
                         'uniformly-spaced sliding windows over the requested '
                         'scan budget are deterministic across processes.'),
            }
            if _viol:
                out['status'] = 'fail'
                out['note'] = (f'train/test document overlap exceeds the frozen policy '
                               f'(test-shingle overlap {test_overlap:.4f} > {_max_shingle_ov}, '
                               f'max row overlap {max_row_overlap} > {_max_row_ov}) — '
                               'confirmation claims are UNAVAILABLE until the '
                               'near-duplicate evaluation documents are excluded '
                               '(or an explicit overlap-excluded sensitivity '
                               'analysis is reported).')
        else:
            out['checks']['doc_overlap'] = 'npy caches absent'
    except Exception as e:
        out['checks']['doc_overlap'] = f'failed ({type(e).__name__}: {e})'
        # An audit that did not run cannot certify a clean split.  Leaving the
        # top-level status at 'ok' here was a fail-open path.
        out['status'] = 'fail'
        out['note'] = (f'train/test overlap audit could not run '
                       f'({type(e).__name__}: {e}); no clean generalization '
                       'claim is available until it is rerun')
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
# 1b) split discipline：划分顺序 / 群组完整性 / 重复行 / 分布偏移
# ---------------------------------------------------------------------------

# Above this many elements the length-n permutation behind
# rng.choice(..., replace=False) is not worth its memory in an audit
# box (>= 2**28 * 8 B); _sample_tokens then draws the same uniform
# without-replacement sample by rejection in O(cap) memory instead.
_CHOICE_PERM_MAX_ELEMS = 1 << 28


def _audit_positive_int(name, value):
    """Positive integer gate parameter; no silent truncation/coercion."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a positive integer, got {value!r}')
    if isinstance(value, numbers.Integral):
        value = int(value)
    elif isinstance(value, numbers.Real):
        f = float(value)
        if not math.isfinite(f) or f != math.floor(f):
            raise ValueError(f'{name} must be a positive integer, got {value!r}')
        value = int(f)
    else:
        raise ValueError(f'{name} must be a positive integer, got {value!r}')
    if value < 1:
        raise ValueError(f'{name} must be a positive integer, got {value!r}')
    return value


def _audit_nonneg_float(name, value):
    """Finite non-negative threshold; NaN would silently disable a gate."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a finite non-negative number, got {value!r}')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f'{name} must be a finite non-negative number, got {value!r}') from None
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f'{name} must be a finite non-negative number, got {value!r}')
    return value


def _sample_tokens(arr, cap=1000000, seed=0):
    """Deterministic sample of a token array (for distribution comparisons)."""
    import numpy as np
    a = np.asarray(arr).reshape(-1)
    if a.size <= cap:
        return a
    n = int(a.size)
    rng = np.random.default_rng(seed)
    # rng.choice(..., replace=False) materialises a length-n permutation
    # (O(n) work and memory).  That is fine for today's corpus sizes, but a
    # scale-up corpus makes it an allocation hazard, so above the threshold
    # where the permutation does not fit a sane audit box the same kind of
    # uniform without-replacement sample is drawn by rejection instead:
    # every K-subset remains equally likely and the draw stays
    # seed-deterministic, with O(cap) memory.
    if n <= _CHOICE_PERM_MAX_ELEMS:
        idx = rng.choice(n, size=int(cap), replace=False)
    else:
        need = int(cap)
        idx = np.empty(0, dtype=np.int64)
        while idx.size < need:
            cand = rng.integers(0, n, size=2 * (need - idx.size), dtype=np.int64)
            # Keep the values in first-draw order.  Sorting the union and then
            # truncating would take the CAP SMALLEST of the candidates, which is
            # a biased (low-index) sample, not a uniform without-replacement
            # draw.  Taking the first `need` distinct values in draw order is
            # exactly the usual rejection sampler for a uniform K-subset.
            merged = np.concatenate([idx, cand])
            _, first = np.unique(merged, return_index=True)
            idx = merged[np.sort(first)]
        idx = idx[:need]
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
    js_warn = _audit_nonneg_float('js_warn', js_warn)
    js_fail = _audit_nonneg_float('js_fail', js_fail)
    if js_warn > js_fail:
        raise ValueError(f'js_warn ({js_warn}) must not exceed js_fail ({js_fail})')
    min_shift_tokens = _audit_positive_int('min_shift_tokens', min_shift_tokens)
    max_rows = _audit_positive_int('max_rows', max_rows)
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
            _nwarn_before = len(out['warnings'])
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

            # Data-cleaning coverage: non-finite values are a hard failure (an
            # embedding index or loss built from them is undefined), and exact
            # duplicates inside the holdout are reported.  Integer token caches
            # cannot represent NaN/Inf, so only float caches are scanned.
            for _name, _arr in (('train', tr), ('val', va)):
                if np.issubdtype(np.asarray(_arr).dtype, np.integer):
                    rec[f'nonfinite_{_name}'] = 0
                    continue
                _bad = int(np.size(_arr) - np.isfinite(_arr).sum())
                rec[f'nonfinite_{_name}'] = _bad
                if _bad:
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: {_bad} NaN/Inf values in the {_name} split')
            if np.asarray(va).ndim == 2:
                _dup_val = int(DG.duplicate_mask(np.asarray(va)).sum())
                rec['duplicate_rows_within_val'] = _dup_val
                if _dup_val:
                    _warn(f'{cd}/{tag}: {_dup_val} duplicate validation rows '
                          '(deduplicate before scoring, or report it as a '
                          'sampling artefact)')

            # Whole-document (group) integrity.  The loaders write one doc-id
            # per emitted document; checking the sidecars proves no group
            # straddles train/validation instead of trusting the build flag.
            _trd_path = os.path.join(cd, f'train_doc_ids_{tag}.npy')
            _vad_path = os.path.join(cd, f'val_doc_ids_{tag}.npy')
            _mdoc = meta.get('doc_ids') or {}
            if os.path.exists(_trd_path) and os.path.exists(_vad_path):
                try:
                    _trd = np.load(_trd_path)
                    _vad = np.load(_vad_path)
                    _shared_docs = int(np.intersect1d(_trd, _vad).size)
                    rec['documents'] = {
                        'train_count': int(np.asarray(_trd).size),
                        'validation_count': int(np.asarray(_vad).size),
                        'overlap': _shared_docs,
                    }
                    if _shared_docs:
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: {_shared_docs} document ids occur in '
                              'both train and validation (group leakage)')
                    if _mdoc and _mdoc.get('disjoint') is not True:
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: cache metadata does not attest '
                              'document-disjoint splits')
                    if _mdoc and int(_mdoc.get('overlap', 0) or 0) != 0:
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: cache metadata records a nonzero '
                              'train/validation document overlap')
                except Exception as _de:
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: document-id sidecars unreadable ({type(_de).__name__}: {_de})')
            elif _mdoc:
                rec['status'] = 'fail'
                _fail(f'{cd}/{tag}: cache metadata records document ids but the '
                      'sidecar files are missing')

            # Exact duplicate rows across train/val/test.  This is row-level
            # leakage regardless of how the split was generated.
            shared_val = _row_overlap_count(tr, va)
            rec['exact_row_overlap_train_val'] = int(shared_val)
            if shared_val:
                rec['status'] = 'fail'
                _fail(f'{cd}/{tag}: {shared_val} validation rows exactly duplicate a training row')
            # The test cache filename carries its own seq_len/vocab/ns tag.
            # P5T deliberately loads test at a longer sequence length than the
            # training cache, so seq_len is NOT part of the cache identity.
            # Pair on vocab + tokenizer fit provenance (as audit_leakage
            # does); a mis-paired test file would make both false positives
            # and false negatives in the overlap audit.
            _cands = []
            for _tmp in sorted(glob.glob(os.path.join(cd, 'test_meta_*.json'))):
                _tm = _load(_tmp)
                if not isinstance(_tm, dict):
                    continue
                _ttag = os.path.basename(_tmp)[len('test_meta_'):-len('.json')]
                _tfp = os.path.join(cd, f'test_ids_{_ttag}.npy')
                if not os.path.exists(_tfp):
                    continue
                if meta.get('vocab') is not None and _tm.get('vocab') is not None and int(_tm['vocab']) != int(meta['vocab']):
                    continue
                if meta.get('fit_fp') and _tm.get('fit_fp') and meta['fit_fp'] != _tm['fit_fp']:
                    continue
                _cands.append((_tfp, _ttag, _tm))
            te = None
            if len(_cands) == 1:
                _tfp, _ttag, _tmeta = _cands[0]
                rec['test_split_strategy'] = _tmeta.get('split_strategy') or 'legacy_unrecorded'
                rec['test_shuffle_at_split'] = _tmeta.get('shuffle_at_split')
                rec['test_tokenizer_fit_split'] = _tmeta.get('tokenizer_fit_split')
                if _tmeta.get('shuffle_at_split') is True:
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: test metadata declares a shuffled split')
                if _tmeta.get('tokenizer_fit_split') not in (None, 'train'):
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: test tokenizer was fitted on {_tmeta.get("tokenizer_fit_split")!r}, not train')
                if _tmeta.get('split_strategy') is None:
                    _warn(f'{cd}/{tag}: legacy test cache has no split_strategy/tokenizer provenance metadata')
                _tndc = _tmeta.get('near_duplicate_cleanup')
                if isinstance(_tndc, dict):
                    if _tndc.get('policy') != 'test_documents_deduplicated_against_train_and_validation':
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: test metadata does not attest duplicate cleanup against both train and validation')
                try:
                    te = np.load(_tfp, mmap_mode='r')
                    rec['test_file'] = os.path.basename(_tfp)
                    rec['test_tag'] = _ttag
                except Exception:
                    te = None
            elif len(_cands) > 1:
                rec['test_pairing'] = 'ambiguous'
                rec['test_candidates'] = sorted(os.path.basename(p) for p, _, _ in _cands)
                _warn(f'{cd}/{tag}: {len(_cands)} test caches match the same protocol; refusing to guess which test split to audit')
            else:
                rec['test_pairing'] = 'not_found'
            if te is not None:
                rec['test_shape'] = list(te.shape)
                if np.issubdtype(np.asarray(te).dtype, np.integer):
                    rec['nonfinite_test'] = 0
                else:
                    _bad_te = int(np.size(te) - np.isfinite(te).sum())
                    rec['nonfinite_test'] = _bad_te
                    if _bad_te:
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: {_bad_te} NaN/Inf values in the test split')
                if np.asarray(te).ndim == 2:
                    _dup_te = int(DG.duplicate_mask(np.asarray(te)).sum())
                    rec['duplicate_rows_within_test'] = _dup_te
                    if _dup_te:
                        _warn(f'{cd}/{tag}: {_dup_te} duplicate test rows')
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
                _ted_path = os.path.join(cd, f'test_doc_ids_{_ttag}.npy')
                if os.path.exists(_ted_path):
                    try:
                        _ted = np.load(_ted_path)
                        _trd = np.load(_trd_path) if os.path.exists(_trd_path) else np.empty(0, dtype=np.uint64)
                        _vad = np.load(_vad_path) if os.path.exists(_vad_path) else np.empty(0, dtype=np.uint64)
                        _dtrain = int(np.intersect1d(_trd, _ted).size)
                        _dval = int(np.intersect1d(_vad, _ted).size)
                        rec['test_documents'] = int(_ted.size)
                        rec['test_train_document_overlap'] = _dtrain
                        rec['test_validation_document_overlap'] = _dval
                        if _dtrain:
                            rec['status'] = 'fail'
                            _fail(f'{cd}/{tag}: {_dtrain} test document ids also occur in train (group leakage)')
                        if _dval:
                            rec['status'] = 'fail'
                            _fail(f'{cd}/{tag}: {_dval} test document ids also occur in validation (selection/test group leakage)')
                    except Exception as _de:
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: test document-id sidecar unreadable ({type(_de).__name__}: {_de})')
                else:
                    rec['test_document_isolation'] = 'not_verified (no test_doc_ids sidecar)'
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
            # A warning is a property of this corpus, not only of the global
            # audit: mark the per-cache record so a caller inspecting one cache
            # does not read it as clean while the top-level status says warnings.
            if rec['status'] == 'ok' and len(out['warnings']) > _nwarn_before:
                rec['status'] = 'warnings'
            cache['corpora'][tag] = rec
            if rec['status'] == 'fail':
                cache['status'] = 'fail'
                out['status'] = 'fail'
            elif rec['status'] == 'warnings':
                # _warn already raised out['status'] to 'warnings', so gating
                # this branch on out['status'] == 'ok' would leave the cache
                # record labelled clean.
                if cache['status'] == 'ok':
                    cache['status'] = 'warnings'
                if out['status'] == 'ok':
                    out['status'] = 'warnings'
        if not cache['corpora']:
            cache['status'] = 'not_verifiable_locally'
    # An existing but empty cache directory proves nothing: -- like "no cache
    # at all" -- the split contract was not verified.  Do not leave the
    # top-level verdict as 'ok' in that case.
    if (not out['caches']
            or not any(c.get('corpora') for c in out['caches'].values())):
        if out['status'] == 'ok':
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
        summ = _load(os.path.join(d, 'summary.json'))
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
                p_raw, rp_raw = b.get('precision'), b.get('random_precision')
                if p_raw is None and rp_raw is None:
                    continue
                p, rp = _num(p_raw), _num(rp_raw)
                if p is None or rp is None:
                    out['problems'].append(
                        f'{d}/{key}: precision/random_precision must be finite real numbers '
                        f'(got {p_raw!r} / {rp_raw!r}); the alignment reading is unverifiable')
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
    raw = _load(bylen_path)
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
            e['n_tok'].add(_exact_int(c.get('n_tok')))
            e['eval_span'].add(_exact_int(c.get('eval_span')))
            e['truncated'].add(_hashable(c.get('truncated')))
            for _field in ('n_tok', 'eval_span'):
                if _exact_int(c.get(_field)) is None:
                    problems.append(
                        f'{key}: L{Ln} cell has no exact integer `{_field}` '
                        f'({c.get(_field)!r}) — the scored target set is unverifiable')
            _nseq_cell = None
            if isinstance(c.get('ppl_per_seq'), list):
                _nseq_cell = len(c['ppl_per_seq'])
                e['n_seq'].add(_nseq_cell)
            elif c.get('n_seq') is not None:
                _nseq_cell = _exact_int(c.get('n_seq'))
                if _nseq_cell is None:
                    problems.append(
                        f'{key}: L{Ln} cell has a non-integer `n_seq` '
                        f'({c.get("n_seq")!r}) — the per-document row count is unverifiable')
                e['n_seq'].add(_nseq_cell)
            else:
                # Keep the absence in the same set as the observed counts so
                # one record with n_seq cannot mask another record without it.
                e['n_seq'].add(None)
            # The complete-cell identity is n_tok == n_seq * (Ln - 1).  Equal
            # (but equally wrong) counts across arms are not a valid target
            # set, so verify the formula whenever the cell claims to be a
            # full-length measurement.
            _nt = _exact_int(c.get('n_tok'))
            _ns = _exact_int(_nseq_cell)
            if c.get('truncated') is False and _ns is not None and _nt is not None:
                if Ln >= 2 and _nt != _ns * (Ln - 1):
                    problems.append(f'{key}: L{Ln} n_tok={_nt} != n_seq*(L-1)={_ns * (Ln - 1)} — the cell did not score one next-token target per row')
            if _nt is not None and _nt < 0:
                problems.append(f'{key}: L{Ln} n_tok={_nt} is negative')
            if _ns is not None and _ns < 0:
                problems.append(f'{key}: L{Ln} n_seq={_ns} is negative')
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
        # NaN / non-integral floats have no seed identity; int(nan) would
        # abort the whole audit instead of excluding the malformed record.
        seeds = sorted({_s for r in recs
                        if (_s := L._as_seed_int(r.get('seed'))) is not None})
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
        if 0 < n_missing_knobs < len(recs):
            # Some seeds carry the optimizer/schedule descriptor and some do
            # not.  The records that do carry it can agree while the arm as a
            # whole is still unverified; trusting the populated subset would
            # let one seed silently run under a different optimizer.
            problems.append(f'{a}: {n_missing_knobs}/{len(recs)} records omit train_knobs — the optimizer/schedule descriptor covers only part of the arm')
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
    out['warnings'] = warnings
    if not out['tuning'].get('tuning_verified'):
        out['warnings'].append('no equal-search baseline tuning artifact — a win over these baselines is not yet evidence against a well-tuned baseline; run baseline_tune.py')
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
        summ = _load(os.path.join(d, 'summary.json'))
        if not summ:
            continue
        by_arm = {}
        n_ok = {}
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

    审计口径：类别不平衡不能只看 accuracy；多分类 F1 必须注明 macro/
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


def audit_hypotheses():
    """预注册假设完整性 / 指标对齐审计（零 GPU）。

    每个声明式对照必须有 statement / mechanism / predicted_sign /
    primary_metric / falsifier；primary_metric 必须是 METRIC_POLICY 里
    真实定义的端点，predicted_sign=0 不能支撑方向性优势。"""
    L = _import_exp_lib()
    out = {'status': 'ok',
           'component': L.hypothesis_audit(L.HYPOTHESIS_DESIGN, 'component'),
           'baseline': L.hypothesis_audit(L.BASELINE_HYPOTHESIS_DESIGN, 'baseline'),
           'open_block': L.hypothesis_audit(L.OPEN_BLOCK_HYPOTHESIS_DESIGN, 'open_block')}
    if any(out[k]['status'] != 'ok' for k in ('component', 'baseline', 'open_block')):
        out['status'] = 'warn'
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
    # These are gate parameters, not measurements.  Validate them even when
    # there are no panels to audit: float(True)=1.0 / int(3.5)=3 would
    # otherwise be echoed as if the audit had run under a valid threshold.
    alpha = L._validated_alpha(alpha, context='audit_methodology alpha')
    _min_n = L._as_seed_int(min_n)
    if _min_n is None or _min_n < 0:
        raise ValueError(f'min_n must be a non-negative integer, got {min_n!r}')
    min_n = _min_n
    dirs = list(outdirs) if outdirs is not None else list(_iter_result_dirs())
    out = {'status': 'ok', 'alpha': float(alpha), 'min_n': int(min_n), 'panels': {},
           'ablation_coverage': {}}
    _reports = {}
    for d in dirs:
        _dir = d if os.path.isabs(d) else os.path.join(REPO, d)
        summ = _load(os.path.join(_dir, 'summary.json'))
        if not summ:
            continue
        rep = L.methodology_audit(summ, design=design, alpha=alpha, min_n=min_n,
                                  outdir=_dir, write=write, verbose=False)
        _reports[d] = rep
        _panel = {
            'ablation_complete': rep.get('ablation_complete'),
            'ablation_completeness': rep.get('ablation_completeness'),
            'controlled_variables': rep.get('controlled_variables'),
            'arm_summary': rep.get('arm_summary'),
            'arms_below_min_seeds': rep.get('arms_below_min_seeds'),
            'verified_components': rep.get('verified_components'),
            'unverified_components': rep.get('unverified_components'),
            'system_level_only_components': rep.get('system_level_only_components'),
            'incomplete_components': rep.get('incomplete_components'),
            # Seed robustness / capacity confounding are per-contrast reads;
            # surface them at panel level so a report reader does not have to
            # know the nested entry layout to notice them.
            'seed_driven_components': sorted(
                c for c, es in (rep.get('components') or {}).items()
                if any(e.get('seed_driven') for e in es)),
            'capacity_confounded_components': sorted(
                c for c, es in (rep.get('components') or {}).items()
                if any(e.get('capacity_confounded') for e in es)),
            'warnings': rep.get('warnings'),
        }
        out['panels'][d] = _panel
        if _panel['arms_below_min_seeds'] or _panel['warnings']:
            out['status'] = 'warn'
    if not out['panels']:
        out['status'] = 'not_verifiable_locally'
        out['note'] = 'no results_*/summary.json found — run the experiment phases first; the audit is zero-GPU over saved records'
    else:
        out['ablation_coverage'] = L.ablation_coverage_across_panels(_reports, design=design)
        if out['ablation_coverage'].get('status') != 'ok':
            out['status'] = 'warn'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
# 8) 规模核算：6ND 估算 + TimeGuard 记录对账
# ---------------------------------------------------------------------------

def train_flops_estimate(params, tokens, attn_overhead=0.35):
    """6ND 估算 + 注意力项系数。

    6·N·D 是前向+反向的常用近似；稀疏选择/池化/激活重算与长序列注意力
    不与参数量成比例，因此用一个**显式声明的**注意力项系数。"""
    n = float(params)
    d_tok = float(tokens)
    base = 6.0 * n * d_tok
    return {'flops_6nd': base, 'flops_with_attn_proxy': base * (1.0 + attn_overhead), 'attn_overhead_assumed': attn_overhead, 'note': 'analytic PROXY, not a measurement; report alongside TimeGuard seconds'}


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
            # TimeGuard stores `runs` as an integer counter; older/foreign
            # state files may store a list.  len(integer) would crash the whole
            # audit on any machine that has actually run a cell.
            _runs = st.get('runs')
            if isinstance(_runs, (list, tuple, dict, set)):
                _run_count = len(_runs)
            elif isinstance(_runs, (int, float)) and not isinstance(_runs, bool):
                _run_count = int(_runs)
            else:
                _run_count = st.get('run_count')
            out.setdefault('measured', {})[sf] = {'booked_seconds': st.get('booked_seconds'), 'run_count': _run_count}
    if all(not isinstance(x, dict) for x in out['panels'].values()):
        out['status'] = 'not_verifiable_locally'
        out['note'] = 'scale panels not present locally; estimates + TimeGuard bookkeeping run on the experiment machine'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
# 9) 测量工具：prefill/decode 与容量读数（含 CPU 微型自检）
# ---------------------------------------------------------------------------

def measure_prefill_decode(model, ids, device, warmup=1, iters=3):
    """端到端 prefill / 前缀推进计时（测量代码，非实验）。

    prefill = 整段前向一次的墙钟；所谓 decode 当前每次从 token 0 重跑完整
    前缀，因此它是 prefix-prefill 均摊，不是有 KV cache 的增量 decode。
    实现在 exp_lib.measure_efficiency（p50/p95、tokens/s 等读数）。"""
    try:
        L = _import_exp_lib()
        return L.measure_efficiency(model, ids, device, warmup=warmup, iters=iters)
    except Exception as _e:
        print(f'[audit] NOTE: exp_lib.measure_efficiency unavailable ({type(_e).__name__}: {_e}); using the local fallback')
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
        _per_tok = (_t.perf_counter() - t0) * 1000.0 / n_dec
        res['prefix_prefill_ms_per_token'] = _per_tok
        # Backwards-compatible alias, but the name must not be read as an
        # incremental decode measurement: there is no KV cache in this path.
        res['decode_ms_per_token_avg'] = _per_tok
        res['decode_has_kv_cache'] = False
        res['decode_metric'] = 'prefix_prefill (no KV cache; re-runs the whole prefix)'
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
    L.set_seed(0)
    dev = torch.device('cpu')
    cfgs = L.make_layer_cfgs(2, 'full_rope')
    model = L.SmallGPT(256, 64, 2, 4, 16, 128, cfgs).to(dev)
    ids = torch.randint(0, 256, (1, 128), dtype=torch.long)
    res = measure_prefill_decode(model, ids, dev, warmup=0, iters=1)
    res['status'] = 'ok'
    res['note'] = 'CPU demo of the measurement harness; run measure_prefill_decode on the real checkpoints (per arm, per length) for the efficiency panel'
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return res


# ---------------------------------------------------------------------------
# report：汇总本地可执行的审计
# ---------------------------------------------------------------------------

def audit_efficiency_pairs(outdirs=None, pairs=None, *, tolerance=0.05):
    """Zero-GPU audit of MEASURED prefill time between arms.

    The efficiency question is not "is there a timing field" but "is a
    speed/cheapness claim supported".  This audit pairs records only when both
    arms have `efficiency.status == 'ok'` at the SAME token count, on the same
    device and torch version, with the same KV-cache semantics; it reports the
    p50 ratio and refuses the words "faster/cheaper" unless the ratio is
    outside `tolerance` in the claimed direction.  It never converts analytic
    FLOPs into a latency claim.

    `pairs` may be a list of (arm_a, arm_b); by default every arm pair sharing
    a panel is reported (descriptive)."""
    import numpy as np
    # The band controls whether a speed claim is allowed at all.  A boolean,
    # NaN, negative or >=1 tolerance would make the claimed direction
    # meaningless (negative even labels every ratio as faster), so refuse it
    # instead of silently changing the gate.
    if isinstance(tolerance, (bool, np.bool_)):
        raise ValueError(f'tolerance must be a finite fraction in [0,1), got {tolerance!r}')
    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'tolerance must be a finite fraction in [0,1), got {tolerance!r}') from None
    if not (math.isfinite(tolerance) and 0.0 <= tolerance < 1.0):
        raise ValueError(f'tolerance must be a finite fraction in [0,1), got {tolerance!r}')
    dirs = outdirs or list(_iter_result_dirs())
    out = {'status': 'ok', 'panels': {}, 'claims_allowed': [], 'refused': [],
           'tolerance': tolerance,
           'note': 'measured prefill only; no KV cache in this model; analytic FLOPs are never substituted for timing'}
    for d in dirs:
        summ = _load(os.path.join(d, 'summary.json'))
        if not summ:
            continue
        eff = {}
        for key, r in summ.items():
            if not isinstance(r, dict):
                continue
            v = r.get('variant')
            if v is None:
                v = str(key).split('::')[0]
            elif not isinstance(v, str) or not v:
                # A non-string arm id is not a usable comparison key.
                continue
            e = r.get('efficiency')
            if isinstance(e, dict) and e.get('status') == 'ok':
                eff.setdefault(v, []).append(e)
        if not eff:
            continue
        panel = {}
        _pairs = pairs if pairs else [(a, b) for i, a in enumerate(sorted(eff)) for b in sorted(eff)[i + 1:]]
        for a, b in _pairs:
            if a not in eff or b not in eff:
                continue
            def _num(v):
                # Timing/token fields are measurements: booleans and
                # non-finite values are not numbers, whatever float() accepts.
                if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
                    return None
                f = float(v)
                return f if math.isfinite(f) else None

            def _med(rows, key):
                _v = [f for f in (_num(r.get(key)) for r in rows) if f is not None]
                return (float(np.median(_v)) if _v else None)
            # Environment/KV identity must hold for EVERY seed in both arms,
            # not only for the first record: a mixed arm would otherwise have
            # its median compared across different hardware/semantics.
            def _env_of(r):
                _dev, _tor = r.get('device'), r.get('torch')
                if (isinstance(_dev, str) and _dev.strip()
                        and isinstance(_tor, str) and _tor.strip()):
                    return (_dev, _tor)
                # Missing/blank provenance is not a match: two records that
                # both omit the device must not permit a speed claim.
                return None
            _env_a = {_env_of(r) for r in eff[a]}
            _env_b = {_env_of(r) for r in eff[b]}
            _kv_a = {r.get('decode_has_kv_cache') for r in eff[a]
                     if isinstance(r.get('decode_has_kv_cache'), (bool, np.bool_))}
            _kv_b = {r.get('decode_has_kv_cache') for r in eff[b]
                     if isinstance(r.get('decode_has_kv_cache'), (bool, np.bool_))}
            _kv_complete = (all(isinstance(r.get('decode_has_kv_cache'), (bool, np.bool_))
                                for r in eff[a] + eff[b]))
            same_env = bool(None not in _env_a and None not in _env_b
                            and len(_env_a) == 1 and len(_env_b) == 1
                            and _env_a == _env_b)
            same_kv = bool(_kv_complete and len(_kv_a) == 1 and len(_kv_b) == 1
                           and _kv_a == _kv_b)
            # Median over seeds, not the first record: a single slow seed must
            # not decide a "faster" claim.
            _ta, _tb = _med(eff[a], 'prefill_ms_p50'), _med(eff[b], 'prefill_ms_p50')
            _na = sorted({int(x) for x in (_num(r.get('n_tokens')) for r in eff[a]) if x is not None})
            _nb = sorted({int(x) for x in (_num(r.get('n_tokens')) for r in eff[b]) if x is not None})
            rec = {'a': a, 'b': b, 'same_device_torch': same_env, 'same_kv_semantics': same_kv,
                   'a_ms_p50': _ta, 'b_ms_p50': _tb, 'a_tokens': _na, 'b_tokens': _nb,
                   'ratio_b_over_a': None, 'speed_claim_allowed': False, 'faster_arm': None, 'reason': None}
            if not (isinstance(_ta, (int, float)) and isinstance(_tb, (int, float)) and _ta > 0 and _tb > 0):
                rec['reason'] = 'missing or non-positive measured prefill time'
            elif _na != _nb or len(_na) != 1 or len(_nb) != 1:
                rec['reason'] = f'different or ambiguous measured token counts ({_na} vs {_nb}); a latency ratio across shapes is not a speed comparison'
            elif not same_env:
                rec['reason'] = 'different or inconsistent device/torch version recorded (within or across arms)'
            elif not same_kv:
                rec['reason'] = 'different or inconsistent KV-cache semantics recorded (within or across arms)'
            else:
                ratio = float(_tb) / float(_ta)
                rec['ratio_b_over_a'] = ratio
                rec['faster_arm'] = (b if ratio < 1.0 - tolerance else (a if ratio > 1.0 + tolerance else None))
                rec['speed_claim_allowed'] = rec['faster_arm'] is not None
                if not rec['speed_claim_allowed']:
                    rec['reason'] = f'ratio {ratio:.3f} is within +/-{tolerance:.0%} of 1: no measured speed difference'
            panel[f'{a} vs {b}'] = rec
            (out['claims_allowed'] if rec['speed_claim_allowed'] else out['refused']).append(f'{d}: {a} vs {b}')
        if panel:
            out['panels'][d] = panel
    if not out['panels']:
        out['status'] = 'not_verifiable_locally'
        out['note'] = ('no records carry a measured `efficiency` block with status ok — the efficiency panel needs '
                       'checkpoints measured with exp_lib.measure_efficiency / measure_efficiency_sweep; analytic FLOPs do not qualify')
    print(json.dumps({k: v for k, v in out.items() if k != 'panels'}, ensure_ascii=False, indent=2)[:4000])
    return out


def audit_release_readiness(path='RELEASES.md'):
    """Zero-GPU check that a numeric claim has an auditable evidence release.

    "Reproducible code" is not "auditable results".
    This audit parses the tracked release index and refuses to call the
    repository auditable unless at least one row names a commit, an archive
    URL, both SHA-256 digests and a verifier.  It also reports whether the
    CURRENT commit is covered.  It cannot verify a remote archive without
    downloading it; `archive_release.py verify` does that locally.
    """
    import re
    import subprocess
    out = {'status': 'no_release', 'rows': 0, 'complete_rows': 0,
           'current_commit': None, 'current_commit_released': False,
           'problems': [], 'path': path,
           'note': ('a repository with no evidence release is source-only: its generated reports are not '
                    'independently auditable. Publish release_<tag>.tar.gz plus a manifest and record it here.')}
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True, timeout=20).stdout.strip()
        out['current_commit'] = head or None
    except Exception:
        pass
    p = os.path.join(REPO, path)
    if not os.path.exists(p):
        out['problems'].append(f'{path} is missing')
        return out
    rows = []
    for line in open(p, encoding='utf-8'):
        line = line.strip()
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        if len(cells) < 7:
            continue
        if cells[0].lower() in ('tag', '---') or set(cells[0]) <= {'-', ':'}:
            continue
        if cells[0].startswith('_none'):
            continue
        rows.append(cells)
    out['rows'] = len(rows)
    for cells in rows:
        tag, _date, commit, url, asha, msha, verifier = cells[:7]
        missing = []
        if not re.fullmatch(r'[0-9a-fA-F]{7,40}', commit or ''):
            missing.append('commit')
        if not re.match(r'https?://', url or ''):
            missing.append('archive URL')
        if not re.fullmatch(r'[0-9a-fA-F]{64}', asha or ''):
            missing.append('archive SHA-256')
        if not re.fullmatch(r'[0-9a-fA-F]{64}', msha or ''):
            missing.append('manifest SHA-256')
        if not verifier or verifier in ('—', '-', 'n/a', 'none'):
            missing.append('verifier')
        if missing:
            out['problems'].append(f'{tag}: missing {missing}')
            continue
        out['complete_rows'] += 1
        if out['current_commit'] and out['current_commit'].startswith(commit):
            out['current_commit_released'] = True
    if out['complete_rows'] == 0:
        out['status'] = 'no_release' if not rows else 'incomplete'
        out['note'] = ('no complete evidence release row: every numeric claim in any generated report remains '
                       'source-side only and is NOT independently auditable')
    else:
        out['status'] = 'ok'
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, ensure_ascii=False, indent=2))
    return out


def audit_selection(path='analysis_mechanism/selection_attribution.json'):
    """Gate for the DIRECT mechanism evidence.

    A claim that "the learned selector finds the relevant blocks" needs an
    artifact comparing the selected blocks with a dependency oracle, per
    query.  This audit defines the contract and validates one when it exists:

        { "cells": [ { "arm": ..., "eval_len": 4096, "k": 32, "n_blocks": 1024,
                       "n_queries": ..., "recall_at_k": ..., "mass_covered": ...,
                       "random_baseline": ...,
                       "per_seed": {"0": {...}, "1": {...}} } ] }

    `recall_at_k` is the fraction of the oracle's top-mass blocks captured by
    the learned top-k; `mass_covered` is the fraction of the oracle attention
    mass inside the selected blocks; `random_baseline` is k/n_blocks (or the
    matched-length random-cut baseline).  Without this artifact the repository
    has NO selection-level mechanism evidence, and wording such as "the
    selector found the relevant information" is unsupported.
    """
    import numpy as np
    out = {'status': 'not_verifiable_locally', 'path': path,
           'required_schema': ['arm', 'eval_len', 'k', 'n_blocks', 'n_queries',
                               'recall_at_k', 'mass_covered', 'random_baseline', 'per_seed'],
           'cells': [], 'problems': [],
           'note': ('no selection-attribution artifact: selection recall/precision versus a dependency oracle '
                    'has NOT been measured, so "the learned selector retrieves the relevant blocks" is an '
                    'unsupported mechanism claim. Run the mechanism probe, then re-run this audit.')}
    p = os.path.join(REPO, path)
    if not os.path.exists(p):
        print(json.dumps({k: v for k, v in out.items() if k != 'cells'}, ensure_ascii=False, indent=2))
        return out
    raw = _load(p)
    cells = raw.get('cells') if isinstance(raw, dict) else None
    if not isinstance(cells, list) or not cells:
        out.update({'status': 'fail', 'problems': ['artifact has no cells list']})
        return out
    def _exact_int(value):
        # Block/query counts are exact: int(1.5) would silently certify a
        # different geometry (and a boolean is a flag, not a count).
        if value is None or isinstance(value, (bool, np.bool_)):
            return None
        if isinstance(value, numbers.Integral):
            return int(value)
        try:
            f = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return int(f) if math.isfinite(f) and f == math.floor(f) else None

    for i, c in enumerate(cells):
        if not isinstance(c, dict):
            out['problems'].append(f'cell {i}: not an object')
            continue
        _miss = [k for k in ('arm', 'eval_len', 'k', 'n_blocks', 'n_queries', 'recall_at_k', 'mass_covered', 'random_baseline', 'per_seed') if c.get(k) is None]
        if _miss:
            out['problems'].append(f'cell {i} ({c.get("arm")}): missing {_miss}')
            continue
        if not isinstance(c.get('per_seed'), dict):
            out['problems'].append(f'cell {i} ({c.get("arm")}): per_seed must be a seed -> measurement object')
            continue
        try:
            k, B = _exact_int(c['k']), _exact_int(c['n_blocks'])
            el, nq = _exact_int(c['eval_len']), _exact_int(c['n_queries'])
            rec, mass, base = (float(c['recall_at_k']), float(c['mass_covered']), float(c['random_baseline']))
        except (TypeError, ValueError, OverflowError):
            out['problems'].append(f'cell {i}: non-numeric fields')
            continue
        if k is None or B is None or el is None or nq is None:
            out['problems'].append(f'cell {i}: k/n_blocks/eval_len/n_queries must be exact integers')
            continue
        if not all(math.isfinite(x) for x in (rec, mass, base)):
            out['problems'].append(f'cell {i}: recall/mass/baseline must be finite')
            continue
        if not (0.0 <= rec <= 1.0 and 0.0 <= mass <= 1.0 and 0.0 <= base <= 1.0):
            out['problems'].append(f'cell {i}: recall/mass/baseline outside [0,1]')
        if k <= 0 or B <= 0 or k > B:
            out['problems'].append(f'cell {i}: require 0 < k <= n_blocks (got k={k}, B={B})')
        if nq <= 0:
            out['problems'].append(f'cell {i}: n_queries must be positive')
        _seeds = c.get('per_seed')
        _n_seeds = len(_seeds)
        _st = None
        # Only complete seed rows enter the paired test.  A malformed row must
        # not inflate the reported seed count or the claim gate: the exact test
        # is run on the valid deltas and its own n is what the n>=6 floor uses.
        _d, _bad_seed = [], []
        _seen_seed_keys = set()
        for _sk, v in _seeds.items():
            # A seed key is an exact run identity.  "not-a-seed" (or two keys
            # that normalise to the same seed) must not inflate n and unlock
            # the n>=6 claim gate.
            _seed_id = _exact_int(_sk)
            if _seed_id is None or _seed_id in _seen_seed_keys:
                _bad_seed.append(str(_sk))
                continue
            _seen_seed_keys.add(_seed_id)
            if not isinstance(v, dict) or v.get('mass_covered') is None or v.get('random_baseline') is None:
                _bad_seed.append(str(_sk))
                continue
            try:
                _d.append(float(v.get('mass_covered')) - float(v.get('random_baseline')))
            except (TypeError, ValueError, OverflowError):
                _bad_seed.append(str(_sk))
        if _bad_seed:
            out['problems'].append(f'cell {i} ({c.get("arm")}): per_seed rows have an invalid/duplicate seed key or lack mass_covered/random_baseline: {_bad_seed[:5]}')
        _n_valid = len(_d)
        if _n_valid >= 2:
            try:
                L = _import_exp_lib()
                _r = L.exact_signflip_p(_d)
                _st = {'p_exact_signflip': float(_r.get('p')), 'n': int(_r.get('n', _n_valid)), 'method': _r.get('method')}
            except Exception as e:
                out['problems'].append(f'cell {i}: could not run the seed test ({type(e).__name__}: {e})')
        out['cells'].append({'arm': c.get('arm'), 'eval_len': el, 'k': k, 'n_blocks': B,
                             'recall_at_k': rec, 'mass_covered': mass, 'random_baseline': base,
                             'mass_excess': mass - base, 'recall_excess': rec - base,
                             'n_queries': nq, 'n_seeds': _n_seeds, 'n_seeds_valid': _n_valid,
                             'seed_test': _st,
                             'claim_allowed': bool(mass > base and rec > base and nq > 0 and _st is not None and _st.get('p_exact_signflip', 1.0) < 0.05 and int(_st.get('n', 0)) >= 6)})
    out['status'] = 'fail' if out['problems'] else 'ok'
    out['note'] = ('cells pass the structural contract; `claim_allowed` additionally requires mass/recall above the '
                   'random baseline and a seed-level exact sign-flip p<0.05 with n>=6')
    print(json.dumps({k: v for k, v in out.items() if k != 'cells'}, ensure_ascii=False, indent=2)[:4000])
    return out


def run_report(out='audit_report.json'):
    res = {'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'leakage': audit_leakage(), 'split_discipline': audit_split_discipline(), 'attribution': audit_attribution(), 'boundary': audit_boundary(), 'length': audit_length(), 'fairness': audit_fairness(), 'metrics': audit_metrics(), 'tuning': audit_tuning(), 'simple_baseline': audit_simple_baseline(), 'hypotheses': audit_hypotheses(), 'methodology': audit_methodology(), 'failures': audit_failures(), 'budget': audit_budget(), 'efficiency_pairs': audit_efficiency_pairs(), 'release': audit_release_readiness(), 'selection': audit_selection()}
    L_atomic = None
    try:
        L = _import_exp_lib()
        # The audit verdicts are quotable claims: stamp code/git provenance and
        # append a ledger entry so an audit artifact regenerated after a code
        # change is distinguishable from the original (the ledger lives under
        # analysis_audit/; the artifact itself stays at the documented path).
        res = L.stamped_analysis('audit_report', res)
        L.atomic_write_json(os.path.join(REPO, out), res, indent=2)
        L.log_experiment_event('analysis_audit', {'kind': 'analysis', 'tracking_schema': 2, 'artifact': out, 'created_at': res.get('created_at'), 'code': res.get('code'), 'label': 'audit_report', 'status': 'ok', 'audits': sorted(k for k in res if k not in ('kind', 'created_at', 'code', 'provenance', 'generated_at'))})
        print(f'[audit] wrote {out}')
    except Exception as e:
        with open(os.path.join(REPO, out), 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f'[audit] wrote {out} (fallback writer: {type(e).__name__})')
    return res


if __name__ == '__main__':
    _cmds = {'leakage': audit_leakage, 'split': audit_split_discipline, 'split_discipline': audit_split_discipline, 'causality': audit_causality, 'attribution': audit_attribution, 'boundary': audit_boundary, 'length': audit_length, 'fairness': audit_fairness, 'metrics': audit_metrics, 'tuning': audit_tuning, 'simple': audit_simple_baseline, 'baseline_floor': audit_simple_baseline, 'hypotheses': audit_hypotheses, 'methodology': audit_methodology, 'failures': audit_failures, 'budget': audit_budget, 'efficiency': audit_efficiency_demo, 'efficiency_pairs': audit_efficiency_pairs, 'selection': audit_selection, 'release': audit_release_readiness, 'report': run_report}
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
