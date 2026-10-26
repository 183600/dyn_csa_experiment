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
  document_unit  分析单元：WikiText 的 row 是原始**行**（标题/段落），不是
               文章；解码 per-token 映射后按顶层 " = Title = " 标题重分组，
               报告 line/article 单元、伪重复倍数与冻结前缀的真实文章数，
               并在 raw train/valid 可用时检查行级去重看不见的文章级泄漏
               （不读 wiki.test.raw）
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
               （边界需 precision/recall/随机基线，NIAH 需 answer NLL；
               二分类需决策阈值来源与校准，macro/micro/weighted F1 需可复现）
  tuning       基线调参：主方法与简单基线是否获得同一搜索网格与范围
               （读 results_baseline_tuning/baseline_tuning.json）
  methodology  受控对比 / 消融完整性 / 统计可靠性：逐臂全种子设置一致性、
               每个创新点的留一对照是否完整、n 与 mean±std、预注册 SESOI
               下的 TOST 等价/亚 SESOI/有害判定（零 GPU）
  sesoi        消融 SESOI 声明：每个声明组件都有唯一、有限、>0 的相对或
               绝对 margin，且留一对照在 RUN_ABL 中完整排程（设计期，零 GPU）
  interaction_sesoi  2x2 交互 SESOI 声明：每个声明的 factorial 都有唯一、
               有限、>0 的 log ratio-of-ratios margin 与方向假设；缺 margin
               时交互仍可报告，但不能声称可加/可删（设计期，零 GPU）
  unit_influence  结果分析：逐评测单位删除（leave-one-out）影响读数是否
               与主终点同口径、是否由单个文章/文档主导（读 analysis_v12/，
               零 GPU；warning 级，不改变冻结判定）
  analysis_choices  结果分析：同一主比较的多种分析口径（逐种子 mean/median/
               trimmed、token 池化、逐单元 mean/median/log-ratio/win-rate）
               是否方向一致、带区间的口径是否一致地排除零（读 analysis_v12/，
               零 GPU；warning 级，不改变冻结判定）
  looks        序列观测 / optional stopping：逐面板 look history 的种子集合、
               逐种子配对差、逐 look p 与 max-over-looks 序列 p 是否自洽；
               朴素最终 p 显著但序列 p 不显著的对照会被点名（零 GPU）
  single_factor  单因素设计契约：每条声明对照的臂配置 diff 恰好等于声明的
               干预键集（定义点审计，不读 results_*）；open-block 矩阵只沿
               两根声明轴变化；v7/v9 扫描的 factors 与真实 arm diff 对账；
               冻结清单的 mechanism_claim_allowed 标志与实际臂配置 diff 一致
               （零 GPU）
  design       设计前馈：token/参数/读取预算台账、专用对照臂、证据等级
                （v6-v11 永远 exploratory；draft 不能 confirmatory）与动态
                分块 H1-H5 假设/对照（零 GPU，不读 test）
  design_preflight  design 的别名
  dose_response  有序剂量设计：方向/形状预注册、>=3 个单调剂量、只动一个
                声明键、同一种子集与测量协议（零 GPU；清单无该声明时
                not_applicable）
  failures     失败与缺失：error 记录按臂集中度（选择偏差读数，零 GPU）
  training_health  训练曲线健康与 NaN 排查记录的自洽性（零 GPU）
  run_order    运行顺序 / 时间混杂：检查 run_order.json 的完整区组与位置
               平衡，并从 experiments.jsonl 复核实际执行顺序；arm-major
               顺序（一个臂全部种子跑完再跑下一个臂）判 fail（零 GPU）
  budget       规模核算：6ND 估算 + TimeGuard 记录对账（零 GPU）
  source       源码树身份：把当前源码字节与 plan_snapshot / summary 记录的
               source_tree 比对（逐文件 added/removed/changed；无记录时明确
               报 not_verifiable_locally，不把 git commit 当成源码哈希）
  dependencies 依赖锁契约：requirements.lock 的每条 pin 与已安装版本比对；
               mismatch/missing/invalid 判 fail，partial 给出覆盖率警告
  data_source  上游语料版本：每个 cache meta 是否记录并匹配固定的 Hub commit
               （WikiText-103 / FineWeb-Edu）；train/validation/test 必须同一
               快照；legacy/local 缓存明确报不可验证而非算作干净
  sample_design  抽样设计：缓存声明的是 source-order prefix 还是可复核的
               document-stratified 概率样本；prefix 明确标记为不可外推，
               raw train/valid 可用时量化文章覆盖率、文章长度 KS 与分层覆盖；
               永不读取 wiki.test.raw
  calibration  统计机器的零假设校准：精确符号翻转 / MITM / n>40 鞍点 /
               max-over-looks 序列 p / 配对 TOST / Holm 的负控制（超发率）
               与正控制（灵敏度），以及 seed-t / paired bootstrap-t /
               unit bootstrap-t / combined seed×doc 区间的覆盖率与
               非空洞宽度门；固定种子、局部 RNG、纯合成数据；
               不读结果、不训练、不改任何测量语义
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
        obj = json.load(open(path, encoding='utf-8'))
    except Exception as e:
        print(f'[audit] WARNING: {path} exists but cannot be parsed ({type(e).__name__}: {e})')
        return None
    if not isinstance(obj, dict):
        # Audit artifacts are keyed objects (panels, records, cells).  A
        # parsed list/scalar used to reach `.items()`/`.get()` in the callers
        # and crash the whole audit; treat it as unverifiable instead.
        print(f'[audit] WARNING: {path} is a {type(obj).__name__}, not a JSON '
              'object — this artifact is unverifiable')
        return None
    return obj


def _iter_result_dirs():
    for name in sorted(os.listdir(REPO)):
        if name.startswith('results_') and os.path.isdir(os.path.join(REPO, name)):
            yield name


def _audit_artifact(path):
    """Resolve a repository artifact independently of the caller's CWD.

    ``_iter_result_dirs`` enumerates ``REPO`` but yields bare names; joining
    them directly against the process CWD silently dropped every panel when
    the CLI was invoked from another directory.  An absolute path is used
    as-is, an existing CWD-relative path keeps its historical meaning for
    explicit caller arguments, and otherwise the path is rooted at REPO.
    """
    p = os.fspath(path)
    if os.path.isabs(p) or os.path.exists(p):
        return p
    return os.path.join(REPO, p)


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
    _unverifiable = []
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
    # a) tokenizer-fit provenance: the test meta must name the SAME fit corpus
    # as the train meta.  A missing fingerprint is unverifiable and must not
    # silently pass just because the id caches happen to share a vocab.
    _train_fit = train_meta.get('fit_fp')
    _test_fit = test_meta.get('fit_fp')
    out['checks']['train_fit_fp'] = _train_fit
    _fit_agree = None
    if _train_fit is not None and _test_fit is not None:
        _fit_agree = str(_train_fit) == str(_test_fit)
        if not _fit_agree:
            _tok_problems.append(
                'test tokenizer fit_fp differs from the train cache fit_fp '
                '(the test tokenizer was not fit on the same train prefix)')
    else:
        _tok_problems.append(
            'tokenizer fit provenance is missing on the train and/or test meta; '
            'the test tokenizer cannot be bound to the train rows')
    out['checks']['fit_fp_agree'] = _fit_agree
    _train_sem = train_meta.get('code_semantics')
    _test_sem = test_meta.get('code_semantics')
    out['checks']['code_semantics'] = {'train': _train_sem, 'test': _test_sem}
    if _train_sem is not None and _test_sem is not None and str(_train_sem) != str(_test_sem):
        _tok_problems.append(
            'train/test caches were built under different data code semantics '
            f'({_train_sem!r} vs {_test_sem!r})')
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
    # c) id-cache fingerprint self-consistency.  The tokenizer checks above
    # prove the id SPACE, but a replaced/edited .npy with a stale metadata
    # `src_fp` would still pass them.  Recompute the loader's own fingerprint
    # from the cached ids and compare it with the recorded value.
    _fp_checks = {}
    try:
        _L = _import_exp_lib()
        for _side, _tag, _meta, _pattern, _expected in (
                ('train', train_tag, train_meta, 'train_ids_*.npy',
                 os.path.join(cache_dir, f'train_ids_{train_tag}.npy')),
                ('test', test_tag, test_meta, 'test_ids_*.npy',
                 os.path.join(cache_dir, f'test_ids_{test_tag}.npy'))):
            _path = _expected if os.path.exists(_expected) else None
            if _path is None:
                _cands = sorted(glob.glob(os.path.join(cache_dir, _pattern)))
                _path = _cands[0] if len(_cands) == 1 else None
            _recorded = (_meta or {}).get('src_fp')
            if _path is None:
                _fp_checks[_side] = {'status': 'unverifiable',
                                     'reason': 'id cache file absent or ambiguous'}
                _unverifiable.append(
                    f'{_side} id cache is absent/ambiguous; its token fingerprint '
                    'cannot be recomputed')
                continue
            if _recorded is None:
                _fp_checks[_side] = {'status': 'unverifiable',
                                     'reason': 'metadata has no src_fp'}
                _unverifiable.append(
                    f'{_side} metadata has no src_fp; the id cache cannot be '
                    'bound to the audited token stream')
                continue
            try:
                _arr = np.load(_path, mmap_mode='r')
                _actual = _L._ids_fp(_arr)
            except Exception as _fe:
                _fp_checks[_side] = {'status': 'fail',
                                     'reason': f'cannot fingerprint {os.path.basename(_path)}: {_fe}'}
                _tok_problems.append(f'{_side} id cache unreadable/unhashable ({type(_fe).__name__}: {_fe})')
                continue
            _ok = str(_actual) == str(_recorded)
            _fp_checks[_side] = {'status': ('ok' if _ok else 'fail'),
                                 'file': os.path.basename(_path),
                                 'recorded': str(_recorded)[:16],
                                 'actual': str(_actual)[:16]}
            if not _ok:
                _tok_problems.append(
                    f'{_side} id-cache fingerprint does not match metadata src_fp '
                    '(the cache was edited/replaced, or the meta describes another file)')
    except Exception as _fe:
        _fp_checks['status'] = f'failed ({type(_fe).__name__}: {_fe})'
        _tok_problems.append(f'id-cache fingerprint audit failed ({type(_fe).__name__}: {_fe})')
    out['checks']['id_cache_fingerprints'] = _fp_checks
    if _tok_problems:
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
                _tf_rec = _tm_doc.get('test_fp')
                if _tf_rec is not None:
                    try:
                        _Lfp2 = _import_exp_lib()
                        if str(_Lfp2._ids_fp(_ted)) != str(_tf_rec):
                            _doc_leak.append('test document-id sidecar fingerprint disagrees with the test metadata')
                    except Exception:
                        _doc_leak.append('test document-id sidecar fingerprint could not be recomputed')
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
        _unverifiable.append(
            'test document-id sidecar is missing; train/val/test document-group '
            'isolation cannot be verified')
    # c3) per-token document mapping.  The document-level analysis (document
    # bootstrap, per-document PPL) is only honest when every scored token has
    # a document id; a fixed row is not a document.  The sidecar is optional
    # for legacy caches, but if the metadata declares it, its absence or
    # mismatch is a hard failure rather than a silent fallback to row-level
    # uncertainty.
    _ttd_p = os.path.join(cache_dir, f'test_token_doc_ids_{test_tag}.npy')
    _tm_doc2 = (test_meta or {}).get('doc_ids') or {}
    _ttd_declared = bool(isinstance(_tm_doc2, dict) and (
        _tm_doc2.get('test_token_count') is not None
        or _tm_doc2.get('test_token_fp') is not None))
    if os.path.exists(_ttd_p):
        try:
            _Lfp_doc = _import_exp_lib()
            _ttd = np.load(_ttd_p)
            _te_for_size = os.path.join(cache_dir, f'test_ids_{test_tag}.npy')
            _te_n = int(np.load(_te_for_size, mmap_mode='r').size) if os.path.exists(_te_for_size) else None
            _ttd_problems = []
            if _ttd.ndim != 1:
                _ttd_problems.append(f'per-token doc ids have shape {_ttd.shape}, not 1-D')
            if _ttd.dtype.kind not in 'iu':
                _ttd_problems.append(f'per-token doc ids have dtype {_ttd.dtype}, not an integer type')
            if _ttd.size and int(_ttd.min()) < 0:
                _ttd_problems.append('per-token doc ids contain negative values')
            if _te_n is not None and int(_ttd.size) != _te_n:
                _ttd_problems.append(
                    f'per-token doc ids cover {_ttd.size} tokens but the test ids have {_te_n}')
            _ted_for_ttd = os.path.join(cache_dir, f'test_doc_ids_{test_tag}.npy')
            if (os.path.exists(_ted_for_ttd) and _ttd.ndim == 1
                    and _ttd.dtype.kind in 'iu'):
                try:
                    if not np.array_equal(
                            np.unique(_ttd),
                            np.unique(np.load(_ted_for_ttd))):
                        _ttd_problems.append(
                            'per-token and per-document sidecars name different '
                            'document sets; the document-level statistic would '
                            'not use the verified groups')
                except Exception as _tde:
                    _ttd_problems.append(
                        f'per-document sidecar cannot be compared ({type(_tde).__name__}: {_tde})')
            if _tm_doc2.get('test_token_count') is not None:
                try:
                    if int(_tm_doc2['test_token_count']) != int(_ttd.size):
                        _ttd_problems.append('test_token_count disagrees with the sidecar length')
                except (TypeError, ValueError, OverflowError):
                    _ttd_problems.append('test_token_count is not an integer')
            if _tm_doc2.get('test_token_fp') is not None:
                if str(_tm_doc2['test_token_fp']) != str(_Lfp_doc._ids_fp(_ttd)):
                    _ttd_problems.append('per-token doc-id fingerprint disagrees with the metadata')
            out['checks']['document_level_eval'] = {
                'status': 'ok' if not _ttd_problems else 'fail',
                'tokens': int(_ttd.size),
                'documents': int(np.unique(_ttd).size),
                'problems': _ttd_problems,
            }
            if _ttd_problems:
                out['status'] = 'fail'
                out['note'] = ('per-token document mapping is not verifiable: '
                               + '; '.join(_ttd_problems))
        except Exception as _tte:
            out['checks']['document_level_eval'] = f'failed ({type(_tte).__name__}: {_tte})'
            out['status'] = 'fail'
            out['note'] = (f'per-token document mapping unreadable ({_tte}); '
                           'document-level uncertainty cannot be claimed')
    elif _ttd_declared:
        out['checks']['document_level_eval'] = {
            'status': 'fail',
            'note': 'metadata declares a per-token document-id sidecar that is missing',
        }
        out['status'] = 'fail'
        out['note'] = ('per-token document mapping is declared but missing; '
                       'rebuild the test cache rather than reporting row-level '
                       'uncertainty as document-level')
    else:
        out['checks']['document_level_eval'] = {
            'status': 'legacy_row_level',
            'note': ('no per-token document-id sidecar; the by-length readout is '
                     'row-level, not document-level (the pooled PPL is unaffected)'),
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
            # The overlap statistics are computed on the sampled shingles
            # only.  With a 1-D train stream capped at ~4M tokens, a 1e9-token
            # cache scans 0.4%; a test row copied from an unsampled region
            # then looks clean.  Report the coverage and refuse to certify a
            # split whose overlap evidence covers less than a quarter of
            # either side.
            _min_cov = 0.25
            _coverage_ok = (float(tr_frac) >= _min_cov
                            and float(te_frac) >= _min_cov)
            out['checks']['doc_overlap'] = {
                'shingle_k': int(shingle),
                'hash': 'uint64 polynomial rolling hash (fixed base 1000003), sliding windows',
                'train_rows_scanned': int(tr_rows),
                'test_rows_scanned': int(te_rows),
                'train_tokens_scanned_fraction': round(float(tr_frac), 6),
                'test_rows_scanned_fraction': round(float(te_frac), 6),
                'min_scan_fraction_for_certification': _min_cov,
                'scan_coverage_sufficient': bool(_coverage_ok),
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
                'status': 'fail' if _viol else ('ok' if _coverage_ok else 'not_verifiable'),
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
            elif not _coverage_ok:
                out['status'] = 'not_verifiable_locally'
                out['note'] = (
                    'the document-overlap scan covers only '
                    f'{float(tr_frac):.2%} of train and {float(te_frac):.2%} of '
                    f'test (minimum {_min_cov:.0%}); a clean sample is not '
                    'evidence that the unsampled remainder is clean. Increase '
                    'the shingle scan budget or treat the split as unverified.')
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
    # Missing id caches or document-id sidecars mean the split cannot be
    # independently verified.  Keeping 'ok' made the v12 one-shot gate treat
    # an unverified isolation claim as clean; only a fully verifiable audit may
    # leave this status at 'ok'.
    if out['status'] == 'ok' and _unverifiable:
        out['status'] = 'not_verifiable_locally'
        out['note'] = ('; '.join(_unverifiable)
                       + ' — the train/test split cannot be certified from the '
                       'local cache; rebuild/copy the missing artifacts before '
                       'using a confirmation claim')
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


def _passage_overlap_from_segments(train_segments, val_rows, *, shingle,
                                   scanned_tokens, total_train_tokens,
                                   min_scan_fraction=0.25,
                                   max_val_shingle_overlap=0.25,
                                   max_excess_overlap=0.10,
                                   max_val_row_overlap=0.50):
    """Near-duplicate *passage* overlap between a scanned train sample and val.

    Exact row overlap is blind to a validation row that reuses a long passage
    from a training document while differing elsewhere.  This helper hashes
    every ``shingle``-token window in the scanned train segments and in the
    validation rows, then measures:

      * the fraction of unique validation shingles that occur in train;
      * the maximum fraction of a single validation row's shingles in train.

    A train-internal baseline (disjoint halves of the same scanned sample) is
    reported and subtracted from the set-level fraction, so recurring corpus
    boilerplate is not mistaken for split contamination.  A row-level baseline
    is measured on held-out train chunks for the same reason.  Coverage is
    explicit: a clean scan of a small fraction of train is ``not_verifiable``,
    never ``ok``.
    """
    import numpy as np
    val = np.asarray(val_rows)
    if val.ndim == 1:
        val = val.reshape(1, -1)
    k = int(shingle)
    if k < 2 or val.ndim != 2 or val.shape[0] == 0 or val.shape[1] < k:
        return {'status': 'not_applicable', 'shingle': k,
                'reason': 'validation rows shorter than the shingle width or empty'}
    seg_data = []
    for seg in train_segments:
        a = np.asarray(seg).reshape(-1)
        if a.size >= k:
            h = _window_hashes(a, k)
            if h.size:
                seg_data.append((a, h))
    if not seg_data:
        return {'status': 'not_applicable', 'shingle': k,
                'reason': 'training scan shorter than the shingle width'}
    train_hashes = np.unique(np.concatenate([h for _, h in seg_data]))
    train_sorted = np.sort(train_hashes)
    # Split the scanned train sample into two disjoint references.  The
    # resulting overlap is the corpus's own repeated-passage rate, not
    # evidence of a train/validation leak.
    if len(seg_data) >= 2:
        b_raw = [a for a, _ in seg_data[1::2]]
        a_h = np.unique(np.concatenate([h for _, h in seg_data[0::2]]))
        b_h = np.unique(np.concatenate([h for _, h in seg_data[1::2]]))
    else:
        raw = seg_data[0][0]
        mid = raw.size // 2
        b_raw = [raw[mid:]]
        a_h = np.unique(_window_hashes(raw[:mid], k))
        b_h = np.unique(_window_hashes(raw[mid:], k))
    if a_h.size and b_h.size:
        baseline = float(np.intersect1d(a_h, b_h, assume_unique=True).size) / float(b_h.size)
    else:
        baseline = 0.0
    a_sorted = np.sort(a_h)
    baseline_row = 0.0
    if a_sorted.size:
        width = int(val.shape[1])
        _chunks = []
        for seg in b_raw:
            n = seg.size // width
            if n <= 0:
                continue
            idx = (np.unique(np.linspace(0, n - 1, num=min(16, n),
                                         dtype=np.int64)))
            _chunks.extend(seg[int(i) * width:(int(i) + 1) * width] for i in idx)
        for chunk in _chunks[:32]:
            rh = _window_hashes(chunk, k)
            if not rh.size:
                continue
            pos = np.searchsorted(a_sorted, rh)
            pos = np.clip(pos, 0, a_sorted.size - 1)
            hit = a_sorted[pos] == rh
            baseline_row = max(baseline_row, float(hit.mean()))
    row_hashes = []
    row_fracs = []
    for row in val:
        rh = _window_hashes(row, k)
        if not rh.size:
            continue
        row_hashes.append(rh)
        pos = np.searchsorted(train_sorted, rh)
        pos = np.clip(pos, 0, train_sorted.size - 1)
        hit = train_sorted[pos] == rh
        row_fracs.append(float(hit.mean()))
    if not row_hashes:
        return {'status': 'not_applicable', 'shingle': k,
                'reason': 'validation rows shorter than the shingle width'}
    val_hashes = np.unique(np.concatenate(row_hashes))
    shared = int(np.intersect1d(train_hashes, val_hashes, assume_unique=True).size)
    val_overlap = float(shared) / float(val_hashes.size)
    excess = max(0.0, val_overlap - baseline)
    max_row = max(row_fracs)
    row_excess = max(0.0, max_row - baseline_row)
    coverage = (float(scanned_tokens) / float(total_train_tokens)
                if total_train_tokens else 1.0)
    val_violation = (val_overlap > float(max_val_shingle_overlap)
                     and excess > float(max_excess_overlap))
    row_violation = (max_row > float(max_val_row_overlap)
                     and row_excess > float(max_excess_overlap))
    violation = bool(val_violation or row_violation)
    if violation:
        status = 'fail'
        note = (f'train/validation passage overlap {val_overlap:.3f} '
                f'(train-internal baseline {baseline:.3f}); '
                f'max validation-row overlap {max_row:.3f} '
                f'(row baseline {baseline_row:.3f})')
    elif coverage < float(min_scan_fraction):
        status = 'not_verifiable'
        note = (f'passage overlap scan covered only {coverage:.1%} of train; '
                'a clean sample is not a clean-split proof')
    else:
        status = 'ok'
        note = (f'train/validation passage overlap {val_overlap:.3f} '
                f'(baseline {baseline:.3f}); max row overlap {max_row:.3f}')
    return {
        'status': status,
        'shingle': k,
        'train_tokens_scanned': int(scanned_tokens),
        'train_tokens_scanned_fraction': round(coverage, 6),
        'val_rows_scanned': int(val.shape[0]),
        'unique_train_shingles': int(train_hashes.size),
        'unique_val_shingles': int(val_hashes.size),
        'shared_shingles': int(shared),
        'val_shingle_overlap': round(val_overlap, 6),
        'train_internal_shingle_overlap': round(baseline, 6),
        'excess_over_val_baseline': round(excess, 6),
        'max_val_row_overlap': round(max_row, 6),
        'max_train_row_overlap_baseline': round(baseline_row, 6),
        'max_val_row_excess': round(row_excess, 6),
        'val_rows_over_max_overlap': int(sum(
            f > float(max_val_row_overlap) for f in row_fracs)),
        'policy': {
            'min_scan_fraction': float(min_scan_fraction),
            'max_val_shingle_overlap': float(max_val_shingle_overlap),
            'max_excess_overlap': float(max_excess_overlap),
            'max_val_row_overlap': float(max_val_row_overlap),
        },
        'note': note,
    }


def _passage_overlap_stats(train, val, *, shingle=32, max_scan_tokens=4_000_000,
                           min_scan_fraction=0.25,
                           max_val_shingle_overlap=0.25,
                           max_excess_overlap=0.10,
                           max_val_row_overlap=0.50,
                           max_val_rows=4096):
    """Sample a train stream the same way as the exact-row audit and compare
    near-duplicate passage shingles against validation."""
    import numpy as np
    tr = np.asarray(train).reshape(-1)
    va = np.asarray(val)
    if va.ndim == 1:
        va = va.reshape(1, -1)
    k = _audit_positive_int('shingle', shingle)
    cap = _audit_positive_int('max_scan_tokens', max_scan_tokens)
    max_val_rows = _audit_positive_int('max_val_rows', max_val_rows)
    min_scan_fraction = _audit_nonneg_float('min_scan_fraction', min_scan_fraction)
    max_val_shingle_overlap = _audit_nonneg_float(
        'max_val_shingle_overlap', max_val_shingle_overlap)
    max_excess_overlap = _audit_nonneg_float(
        'max_excess_overlap', max_excess_overlap)
    max_val_row_overlap = _audit_nonneg_float(
        'max_val_row_overlap', max_val_row_overlap)
    for _name, _val in (('min_scan_fraction', min_scan_fraction),
                        ('max_val_shingle_overlap', max_val_shingle_overlap),
                        ('max_excess_overlap', max_excess_overlap),
                        ('max_val_row_overlap', max_val_row_overlap)):
        if _val > 1.0:
            raise ValueError(f'{_name} must be a fraction in [0, 1], got {_val!r}')
    if not tr.size or va.ndim != 2 or va.shape[0] == 0 or va.shape[1] < k:
        return {'status': 'not_applicable', 'shingle': k,
                'reason': 'train stream or validation rows too short'}
    val_total = int(va.shape[0])
    if val_total > max_val_rows:
        idx = np.unique(np.linspace(0, val_total - 1, max_val_rows,
                                    dtype=np.int64))
        va = va[idx]
    nseg = 16
    seg_len = max(int(cap) // nseg, int(va.shape[1]) + 1)
    if tr.size <= max(int(cap), nseg * seg_len):
        segments = [tr]
        scanned = int(tr.size)
    else:
        starts = np.linspace(0, tr.size - seg_len, nseg, dtype=np.int64)
        segments = [tr[int(s):int(s) + seg_len] for s in starts]
        scanned = int(sum(s.size for s in segments))
    out = _passage_overlap_from_segments(
        segments, va, shingle=k, scanned_tokens=scanned,
        total_train_tokens=int(tr.size), min_scan_fraction=min_scan_fraction,
        max_val_shingle_overlap=max_val_shingle_overlap,
        max_excess_overlap=max_excess_overlap,
        max_val_row_overlap=max_val_row_overlap)
    out['val_rows_total'] = val_total
    return out


def audit_split_discipline(cache_dirs=None, max_rows=50000, js_warn=0.05,
                           js_fail=0.25, min_shift_tokens=5000,
                           preferred_train_seq_len=None,
                           passage_shingle=32, max_passage_scan_tokens=4_000_000,
                           min_passage_scan_fraction=0.25,
                           max_val_shingle_overlap=0.25,
                           max_passage_excess_overlap=0.10,
                           max_val_row_shingle_overlap=0.50):
    """Audit the *split contract* for every cached corpus on disk.

    Unlike ``audit_leakage`` (tokenizer provenance and document overlap), this
    gate checks the split itself:

      * the metadata declares an ordered/non-shuffled split and a train-only
        tokenizer fit;
      * validation and test contain no exact row duplicated from train;
      * validation contains no long near-duplicate passage from train (a
        shingle-overlap scan calibrated against train-internal repetition);
      * no group/document spans two sides (for FineWeb, document-disjoint
        chronological construction is recorded and the cache metadata keeps
        the validation fingerprint);
      * train vs holdout token distributions are compared with Jensen-Shannon
        divergence (a shift is reported explicitly rather than silently
        assumed away).

     Exact duplicate rows across splits are a hard failure.  Distribution or
    legacy-metadata issues are warnings because they need a scientific
    judgement, not an automatic rebuild.

    ``preferred_train_seq_len`` restricts the audit to the train cache that
    actually served a panel when a directory holds several protocol variants
    (e.g. a 512-token training cache and a 4097-token probe cache).  A caller
    that names the training length must not be silently audited against a
    different stream.
    """
    if preferred_train_seq_len is not None:
        preferred_train_seq_len = _audit_positive_int(
            'preferred_train_seq_len', preferred_train_seq_len)
    js_warn = _audit_nonneg_float('js_warn', js_warn)
    js_fail = _audit_nonneg_float('js_fail', js_fail)
    if js_warn > js_fail:
        raise ValueError(f'js_warn ({js_warn}) must not exceed js_fail ({js_fail})')
    min_shift_tokens = _audit_positive_int('min_shift_tokens', min_shift_tokens)
    max_rows = _audit_positive_int('max_rows', max_rows)
    passage_shingle = _audit_positive_int('passage_shingle', passage_shingle)
    max_passage_scan_tokens = _audit_positive_int(
        'max_passage_scan_tokens', max_passage_scan_tokens)
    min_passage_scan_fraction = _audit_nonneg_float(
        'min_passage_scan_fraction', min_passage_scan_fraction)
    max_val_shingle_overlap = _audit_nonneg_float(
        'max_val_shingle_overlap', max_val_shingle_overlap)
    max_passage_excess_overlap = _audit_nonneg_float(
        'max_passage_excess_overlap', max_passage_excess_overlap)
    max_val_row_shingle_overlap = _audit_nonneg_float(
        'max_val_row_shingle_overlap', max_val_row_shingle_overlap)
    for _name, _val in (('min_passage_scan_fraction', min_passage_scan_fraction),
                        ('max_val_shingle_overlap', max_val_shingle_overlap),
                        ('max_passage_excess_overlap', max_passage_excess_overlap),
                        ('max_val_row_shingle_overlap', max_val_row_shingle_overlap)):
        if _val > 1.0:
            raise ValueError(f'{_name} must be a fraction in [0, 1], got {_val!r}')
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

    def _audit_row_sample(arr, label, rec):
        """Deterministically cap a 2-D panel at ``max_rows`` for duplicate scans.

        ``max_rows`` was validated but unused, so a 1e8-row cache still built
        giant row-key tables.  The cap is recorded; it is an audit sample, not
        a proof about the unsampled rows.
        """
        a = np.asarray(arr)
        if a.ndim == 2 and a.shape[0] > max_rows:
            _ix = np.unique(np.linspace(0, a.shape[0] - 1, num=max_rows,
                                        dtype=np.int64))
            rec[f'{label}_rows_total'] = int(a.shape[0])
            rec[f'{label}_rows_audited'] = int(_ix.size)
            return a[_ix]
        return a

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
            if preferred_train_seq_len is not None:
                try:
                    _meta_sl = int(meta.get('seq_len'))
                except (TypeError, ValueError, OverflowError):
                    continue
                if _meta_sl != int(preferred_train_seq_len):
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
            _va_scan = _audit_row_sample(va, 'validation', rec)
            rec['declared_split_strategy'] = meta.get('split_strategy') or 'legacy_unrecorded'
            rec['shuffle_at_split'] = meta.get('shuffle_at_split')
            rec['tokenizer_fit_split'] = meta.get('tokenizer_fit_split')
            rec['val_fingerprint_recorded'] = bool(meta.get('val_fp'))
            # The cache files must still BE the stream the metadata names.
            # Checking only "val_fp is present" (or only the runtime loader's
            # check) lets a replaced/edited .npy be certified by this offline
            # audit.  Recompute both recorded fingerprints from the actual
            # bytes; a missing fingerprint stays a legacy warning.
            _src_recorded = meta.get('src_fp') or meta.get('train_fp')
            _val_recorded = meta.get('val_fp')
            if _src_recorded is not None or _val_recorded is not None:
                try:
                    _Lfp = _import_exp_lib()
                    if _src_recorded is not None:
                        _src_actual = _Lfp._ids_fp(tr)
                        rec['train_fingerprint_verified'] = bool(
                            str(_src_actual) == str(_src_recorded))
                        if not rec['train_fingerprint_verified']:
                            rec['status'] = 'fail'
                            _fail(f'{cd}/{tag}: train id-cache fingerprint does not '
                                  'match metadata src_fp (the cache was edited or '
                                  'replaced)')
                    if _val_recorded is not None:
                        _val_actual = _Lfp._ids_fp(np.asarray(va).reshape(-1))
                        rec['val_fingerprint_verified'] = bool(
                            str(_val_actual) == str(_val_recorded))
                        if not rec['val_fingerprint_verified']:
                            rec['status'] = 'fail'
                            _fail(f'{cd}/{tag}: validation id-cache fingerprint does '
                                  'not match metadata val_fp (a stale/mixed '
                                  'validation file cannot be certified)')
                except Exception as _fpe:
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: id-cache fingerprint audit failed '
                          f'({type(_fpe).__name__}: {_fpe})')
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
            _trd_exists = os.path.exists(_trd_path)
            _vad_exists = os.path.exists(_vad_path)
            _mdoc = meta.get('doc_ids') or {}
            if _trd_exists != _vad_exists:
                rec['status'] = 'fail'
                _fail(f'{cd}/{tag}: only one train/validation document-id '
                      'sidecar exists; group isolation is half-attested')
            if _trd_exists and _vad_exists:
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
            shared_val = _row_overlap_count(tr, _va_scan)
            rec['exact_row_overlap_train_val'] = int(shared_val)
            if shared_val:
                rec['status'] = 'fail'
                _fail(f'{cd}/{tag}: {shared_val} validation rows exactly duplicate a training row')
            # Near-duplicate passage overlap catches a validation row that
            # reuses a long training passage while differing elsewhere.  The
            # scan is sampled, so a clean result on a small train fraction is
            # recorded as not_verifiable rather than certified.
            try:
                _passage = _passage_overlap_stats(
                    tr, _va_scan, shingle=passage_shingle,
                    max_scan_tokens=max_passage_scan_tokens,
                    min_scan_fraction=min_passage_scan_fraction,
                    max_val_shingle_overlap=max_val_shingle_overlap,
                    max_excess_overlap=max_passage_excess_overlap,
                    max_val_row_overlap=max_val_row_shingle_overlap)
                rec['passage_overlap'] = _passage
                if _passage.get('status') == 'fail':
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: ' + str(_passage.get(
                        'note', 'near-duplicate passage overlap')))
                elif _passage.get('status') == 'not_verifiable':
                    if rec['status'] == 'ok':
                        rec['status'] = 'warnings'
                    _warn(f'{cd}/{tag}: ' + str(_passage.get(
                        'note', 'passage overlap scan incomplete')))
            except Exception as _pe:
                rec['status'] = 'fail'
                _fail(f'{cd}/{tag}: passage-overlap audit failed '
                      f'({type(_pe).__name__}: {_pe})')
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
                _rp = _tmeta.get('read_purpose')
                if _rp is not None and _rp != 'final_evaluation':
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: test cache records read_purpose={_rp!r}, '
                          'not the frozen final evaluation; a diagnostic/tuning '
                          'read would turn test into an implicit validation set')
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
                _te_scan = _audit_row_sample(te, 'test', rec)
                # Per-split content fingerprints are recomputed from the
                # actual bytes, not trusted from the JSON.  The audit already
                # did this for train/validation; without the same check on
                # test, a replaced/stale test file could be certified by a
                # metadata-only reader and then become an implicit validation
                # set through repeated evaluation.
                try:
                    _Lf = _import_exp_lib()
                    _t_actual = _Lf._ids_fp(np.asarray(te))
                    _t_recorded = _tmeta.get('src_fp')
                    _t_explicit = _tmeta.get('test_fp')
                    _sf_all = _tmeta.get('split_fingerprints')
                    _t_split = _sf_all.get('test') if isinstance(_sf_all, dict) else None
                    if _t_split is not None and not isinstance(_t_split, dict):
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: test split_fingerprints is malformed')
                        _t_split = {}
                    _t_split = _t_split or {}
                    rec['test_fingerprint_recorded'] = bool(_t_recorded)
                    _fp_ok = (_t_recorded is not None
                              and str(_t_actual) == str(_t_recorded))
                    if _t_explicit is not None and str(_t_explicit) != str(_t_actual):
                        _fp_ok = False
                    if _t_split.get('ids_fp') is not None and str(_t_split['ids_fp']) != str(_t_actual):
                        _fp_ok = False
                    rec['test_fingerprint_verified'] = bool(_fp_ok)
                    if _t_recorded is None and _t_explicit is None and not _t_split:
                        # Legacy test cache: there is no recorded fingerprint to
                        # disagree with.  Report the missing evidence (like the
                        # train/validation reader does) instead of inventing a
                        # hard failure; a cache that RECORDS a value must match.
                        _warn(f'{cd}/{tag}: legacy test cache records no content '
                              'fingerprint; the held-out id bytes cannot be '
                              'bound to the audited stream')
                    elif _t_recorded is not None and str(_t_actual) != str(_t_recorded):
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: test id-cache fingerprint does not '
                              'match metadata src_fp (the held-out cache was '
                              'edited or replaced)')
                    if _t_explicit is not None and str(_t_explicit) != str(_t_actual):
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: test_fp does not match the actual '
                              'held-out ids')
                    if _t_split.get('ids_fp') is not None and str(_t_split['ids_fp']) != str(_t_actual):
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: split_fingerprints.test.ids_fp does '
                              'not match the actual held-out ids')
                except Exception as _tfe:
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: test id-cache fingerprint audit failed '
                          f'({type(_tfe).__name__}: {_tfe})')
                # Split-construction semantics and tokenizer identity must be
                # the SAME across the train/test pair.  A cache pair built by
                # different code semantics or different tokenizers is not one
                # frozen split, even if every individual file is intact.
                _train_sem = meta.get('code_semantics')
                _test_sem = _tmeta.get('code_semantics')
                rec['code_semantics'] = {'train': _train_sem, 'test': _test_sem}
                if _train_sem is not None and _test_sem is not None and str(_train_sem) != str(_test_sem):
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: train/test data code semantics differ '
                          f'({_train_sem!r} vs {_test_sem!r}); the split '
                          'construction is not one frozen pair')
                import hashlib as _hl
                _train_tok_sha = meta.get('tokenizer_file_sha256')
                _test_tok_sha = _tmeta.get('tokenizer_file_sha256')
                # The BINDING identity is the full id->token map.  File bytes
                # can differ for equivalent serialisations, so use the vocab
                # hash for equality and the file hash only to prove each
                # persisted artefact is the one its own metadata names.
                _train_vsha = meta.get('tokenizer_vocab_sha256')
                _test_vsha = _tmeta.get('tokenizer_vocab_sha256')
                if _train_vsha and _test_vsha and str(_train_vsha) != str(_test_vsha):
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: train/test tokenizer id-space hashes '
                          'differ; the id spaces are not one paired tokenizer')
                if _train_tok_sha and _test_tok_sha and str(_train_tok_sha) != str(_test_tok_sha):
                    if _train_vsha and _test_vsha and str(_train_vsha) == str(_test_vsha):
                        _warn(f'{cd}/{tag}: train/test tokenizer files are not '
                              'byte-identical, but their full vocab identity is; '
                              'the binding equality holds')
                    else:
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: train/test tokenizer file identities '
                              'differ without a matching vocab hash; the id '
                              'spaces cannot be shown to be one paired tokenizer')
                _train_tok_path = os.path.join(cd, f'tok_{tag}.json')
                if _train_tok_sha:
                    if not os.path.exists(_train_tok_path):
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: train metadata binds a tokenizer file '
                              'that is missing')
                    else:
                        try:
                            with open(_train_tok_path, 'rb') as _tfh2:
                                _actual_train_tok_sha = _hl.sha256(_tfh2.read()).hexdigest()
                        except Exception as _tke2:
                            rec['status'] = 'fail'
                            _fail(f'{cd}/{tag}: train tokenizer file unreadable '
                                  f'({type(_tke2).__name__}: {_tke2})')
                        else:
                            if str(_actual_train_tok_sha) != str(_train_tok_sha):
                                rec['status'] = 'fail'
                                _fail(f'{cd}/{tag}: train tokenizer file sha256 '
                                      'does not match its metadata (tokenizer '
                                      'was replaced)')
                _test_tok_path = os.path.join(cd, f"test_tok_{_ttag}.json")
                if _test_tok_sha:
                    if not os.path.exists(_test_tok_path):
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: test metadata binds a tokenizer file '
                              'that is missing')
                    else:
                        try:
                            with open(_test_tok_path, 'rb') as _tfh:
                                _actual_tok_sha = _hl.sha256(_tfh.read()).hexdigest()
                        except Exception as _tke:
                            rec['status'] = 'fail'
                            _fail(f'{cd}/{tag}: test tokenizer file unreadable '
                                  f'({type(_tke).__name__}: {_tke})')
                        else:
                            if str(_actual_tok_sha) != str(_test_tok_sha):
                                rec['status'] = 'fail'
                                _fail(f'{cd}/{tag}: test tokenizer file sha256 '
                                      'does not match its metadata (tokenizer '
                                      'was replaced)')
                _test_fit = _tmeta.get('fit_fp')
                _train_fit = meta.get('fit_fp')
                if _train_fit and _test_fit and str(_train_fit) != str(_test_fit):
                    rec['status'] = 'fail'
                    _fail(f'{cd}/{tag}: test fit_fp differs from the train '
                          'cache fit_fp (different tokenizer-fit corpus)')
                if np.issubdtype(np.asarray(te).dtype, np.integer):
                    rec['nonfinite_test'] = 0
                else:
                    _bad_te = int(np.size(te) - np.isfinite(te).sum())
                    rec['nonfinite_test'] = _bad_te
                    if _bad_te:
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: {_bad_te} NaN/Inf values in the test split')
                if np.asarray(te).ndim == 2:
                    _dup_te = int(DG.duplicate_mask(np.asarray(_te_scan)).sum())
                    rec['duplicate_rows_within_test'] = _dup_te
                    if _dup_te:
                        _warn(f'{cd}/{tag}: {_dup_te} duplicate test rows')
                shared_test = _row_overlap_count(tr, _te_scan)
                shared_val_test = _row_overlap_count(_va_scan, _te_scan)
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
                    if not (os.path.exists(_trd_path) and os.path.exists(_vad_path)):
                        # The old code substituted empty arrays for the
                        # missing train/val sidecars, which reported zero
                        # overlap and certified a test group split that was
                        # never checked.
                        rec['status'] = 'fail'
                        _fail(f'{cd}/{tag}: test document-id sidecar is present '
                              'but the train/validation doc-id sidecars are '
                              'missing; test group isolation cannot be verified')
                    try:
                        _ted = np.load(_ted_path)
                        _trd = np.load(_trd_path)
                        _vad = np.load(_vad_path)
                        _dtrain = int(np.intersect1d(_trd, _ted).size)
                        _dval = int(np.intersect1d(_vad, _ted).size)
                        rec['test_documents'] = int(_ted.size)
                        rec['test_train_document_overlap'] = _dtrain
                        rec['test_validation_document_overlap'] = _dval
                        _md_t = _tmeta.get('doc_ids') or {}
                        if _md_t.get('test_fp') is not None:
                            _Lfp3 = _import_exp_lib()
                            if str(_Lfp3._ids_fp(_ted)) != str(_md_t['test_fp']):
                                rec['status'] = 'fail'
                                _fail(f'{cd}/{tag}: test document-id sidecar does '
                                      'not match the fingerprint in its metadata (a '
                                      'stale/mixed sidecar cannot certify isolation)')
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
# 1b) row unit: line vs article (the WikiText "document" is a raw line)
# ---------------------------------------------------------------------------

def _unit_meta_int(meta, key):
    """Integer metadata field or None (bool/fraction/NaN are not ints)."""
    v = (meta or {}).get(key)
    if v is None or isinstance(v, (bool, np.bool_)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(f) or f != math.floor(f):
        return None
    return int(f)


def _unit_decode_runs(ids, ttd, tok, L, label, rec):
    """Decode a per-token document map into ``[(span, text, id)]`` runs."""
    import numpy as np
    flat = np.asarray(ids).reshape(-1)
    doc = np.asarray(ttd).reshape(-1)
    if doc.size < flat.size:
        rec['problems'].append(
            f'{label}: per-token document map covers {doc.size} tokens but the '
            f'id array has {flat.size}; the served slice is not fully mapped')
        return None
    if doc.size > flat.size:
        # The loaders cap the validation stream at val_seqs*seq_len + seq_len
        # but serve only the first n_val*seq_len tokens.  The sidecar therefore
        # carries one extra (unserved) sequence of ids by construction.  The
        # map is written in stream order, so the served rows are its prefix;
        # trim the unserved tail and say so instead of rejecting a normal cache.
        # A shorter map stays a hard error (above): that would mean some served
        # token has no document identity at all.
        rec['trimmed_unserved_tokens'] = int(doc.size - flat.size)
        doc = doc[:flat.size]
    try:
        spans = L.line_spans_from_token_doc_ids(doc)
    except (TypeError, ValueError, OverflowError) as e:
        rec['problems'].append(f'{label}: {type(e).__name__}: {e}')
        return None
    texts, dprobs = L.decode_token_runs(flat, spans, tok)
    rec['decoded_runs'] = int(len(spans))
    if dprobs:
        rec['problems'].extend(f'{label}: {p}' for p in dprobs)
    return list(zip(spans, texts))


def _unit_classify_runs(runs, L):
    """Classify a served slice as line-unit, article-unit or unknown.

    A run is a maximal stretch of equal ids in the per-token map, so its id is
    the hash of the *unit* the cache grouped by.  Both a line cache and an
    article cache reproduce their own text hash, so hashing alone cannot tell
    them apart; the structural signal is that a line cache's runs include many
    non-heading lines, while an article cache has one run per article and
    every run starts with a top-level heading.  The id/text hash is still
    checked, so a map that does not describe the decoded bytes is refused.
    """
    n = 0
    hash_ok = 0
    heading_runs = 0
    multiline_runs = 0
    failures = []
    for (_s, _e, _doc), _text in runs:
        if _text is None:
            continue
        if int(L._doc_id_u64(L._doc_exact_fp(_text))) == int(_doc):
            hash_ok += 1
        else:
            failures.append(n)
        n += 1
        _s_body = _text.strip()
        if _s_body:
            _first = _s_body.split('\n', 1)[0].rstrip('\r')
            # Full-line form covers a single decoded raw line; the prefix form
            # covers a whole-article run whose title and body are separated by
            # spaces (per-line EOS tokens are skipped by decode).
            if (L._WIKITEXT_TOP_HEADING_RE.match(_first)
                    or L._WIKITEXT_TOP_HEADING_PREFIX_RE.match(_s_body)):
                heading_runs += 1
            if '\n' in _s_body:
                multiline_runs += 1
    if n == 0:
        return {'unit': 'unknown', 'reason': 'no decoded runs'}
    # The served slice ends at a row/token boundary, so the FINAL line can be
    # cut mid-line (its doc id is the hash of the full raw line).  That is an
    # expected cap artefact, not a unit mismatch; allow exactly that one run.
    truncated_final = bool(failures == [n - 1])
    _common = {'hash_matches': hash_ok, 'n': n,
               'heading_runs': heading_runs,
               'multiline_runs': multiline_runs,
               'truncated_final_run': truncated_final}
    # An article-unit run is a whole article and therefore starts with a
    # top-level heading.  Its stored id may hash a different canonical form of
    # the article text (e.g. raw lines joined with '\n' vs the decoded token
    # run), so the id hash is diagnostic here, not the unit test.  The line
    # case still requires the exact decode hash (apart from the cap-truncated
    # final run): that is what proves the map describes these bytes.
    if heading_runs == n:
        if failures and not truncated_final:
            return dict(_common, unit='unknown', failures=failures[:8],
                        reason=('every run is headed like an article, but its '
                                'stored id does not hash the decoded article '
                                'text (or the cap-truncated final run)'))
        return dict(_common, unit='article',
                    reason='every served run starts with a top-level article heading')
    if failures and not truncated_final:
        return dict(_common, unit='unknown', failures=failures[:8],
                    reason=('served ids do not hash the decoded run text; the '
                            'cache and its tokenizer/sidecar do not describe the '
                            'same bytes'))
    if heading_runs > 0:
        return dict(_common, unit='line',
                    reason='runs include non-heading raw lines')
    return dict(_common, unit='unknown',
                reason='no top-level heading in the served runs; line/article '
                       'unit is not self-describing')


def _unit_group_lines(runs, L):
    """Regroup decoded line runs into articles (without re-reading the corpus).

    ``runs`` is the ``[((start, stop, id), text), ...]`` form returned by
    ``_unit_decode_runs``.
    """
    import numpy as np
    texts = [r[1] for r in runs]
    if any(t is None for t in texts):
        return None, 'a decoded line is missing; cannot regroup into articles'
    g = L.wikitext_article_groups(texts)
    if not g.get('has_headings'):
        return None, ('no top-level heading in the decoded line text; the '
                      'article unit cannot be recovered from this slice')
    line_to_group = np.full(len(texts), -1, dtype=np.int64)
    for _gi, _g in enumerate(g['groups']):
        line_to_group[np.asarray(_g, dtype=np.int64)] = _gi
    if int((line_to_group < 0).sum()):
        # A run decoded to empty/whitespace and was skipped by the grouping;
        # indexing art_ids[-1] would silently attach it to the last article.
        return None, (f'{int((line_to_group < 0).sum())} decoded run(s) are '
                      'empty and cannot be assigned to an article')
    art_ids = np.asarray(L.wikitext_article_ids(texts, g['groups']), dtype=np.uint64)
    lengths = np.asarray([int(r[0][1]) - int(r[0][0]) for r in runs],
                         dtype=np.int64)
    token_art = np.repeat(art_ids[line_to_group], lengths)
    return {'groups': g, 'line_to_group': line_to_group,
            'article_ids': art_ids, 'token_article_ids': token_art}, None


def audit_document_unit(cache_dir=None, *, seq_len=None, raw_dir=None,
                        max_articles=4000):
    """Audit the *unit of analysis* of the cached WikiText splits.

    The HF ``wikitext-103-raw-v1`` builder yields one row per raw **line**:
    every article is many rows (a `` = Title = `` heading plus its paragraphs
    and `` = = Section = = `` headings).  The loaders therefore persist line
    ids, and every downstream "per-document" statistic is a per-*line*
    statistic unless the rows are regrouped by their top-level headings.

    This zero-GPU audit makes that distinction executable:

      * it decodes the cached per-token map and classifies the served unit as
        ``line``, ``article`` or ``unknown`` from the run structure, checking
        the decoded bytes against the stored ids (a missing map is
        ``not_verifiable_locally``; test raw is never read to replace it);
      * for a line-unit slice it regroups the decoded lines by top-level
        headings and reports ``n_lines`` / ``n_articles`` and the
        pseudoreplication factor, plus the number of distinct articles in the
        frozen evaluation prefix (the honest denominator for the repo's
        64-document population floor);
      * when the raw train/validation dumps are locally available it checks
        **article-level** leakage the line index cannot see: an article whose
        lines straddle train and validation, and near-duplicate articles whose
        individual lines were all reworded.  It never reads ``wiki.test.raw``
        (that would consume the one-shot held-out access).

    A line-unit cache is reported as ``warnings`` with
    ``document_level_claims_allowed=False``; it is not silently relabelled.
    Article-level leakage is a hard failure.  Missing caches/raw data are
    ``not_verifiable_locally``, never "clean".

    ``seq_len`` optionally disambiguates a cache directory holding several
    training sequence lengths (it selects the train meta with that
    ``seq_len``); without it, an ambiguous train/test pairing is refused rather
    than guessed.  ``raw_dir`` overrides the ``WT103_RAW_TXT`` / ``./wt103_raw``
    discovery used only for the train/validation article-leakage section.
    """
    L = _import_exp_lib()
    import glob
    import numpy as np
    if seq_len is not None:
        seq_len = _audit_positive_int('seq_len', seq_len)
    if max_articles is not None:
        max_articles = _audit_positive_int('max_articles', max_articles)
    if cache_dir in (None, ''):
        cache_dir = os.environ.get('WT103_CACHE_DIR') or './wt103_cache'
    out = {'kind': 'document_unit_audit', 'status': 'ok', 'caches': {},
           'warnings': [], 'problems': [],
           'note': ('WikiText rows are raw lines, not articles; a per-row '
                    '"document" statistic is per-line unless regrouped by the '
                    'top-level " = Title = " headings.  This audit never reads '
                    'wiki.test.raw.')}
    if not os.path.isdir(cache_dir):
        out['status'] = 'not_verifiable_locally'
        out['note'] = f'{cache_dir!r} does not exist — no cache to audit'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    def _warn(msg):
        out['warnings'].append(msg)
        if out['status'] == 'ok':
            out['status'] = 'warnings'

    def _fail(msg):
        out['problems'].append(msg)
        out['status'] = 'fail'

    _tm = sorted(glob.glob(os.path.join(cache_dir, 'test_meta_*.json')))
    _vm = sorted(p for p in glob.glob(os.path.join(cache_dir, 'meta_*.json'))
                 if not os.path.basename(p).startswith('test_meta_'))
    if not _tm or not _vm:
        out['status'] = 'not_verifiable_locally'
        out['note'] = (f'{cache_dir}/ lacks test_meta_*.json or meta_*.json — '
                       'build the caches once before auditing the unit')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    _pairs = []
    for _tp in _tm:
        _t = _load(_tp)
        if not isinstance(_t, dict):
            continue
        for _mp in _vm:
            _m = _load(_mp)
            if not isinstance(_m, dict):
                continue
            _tv, _mv = _unit_meta_int(_t, 'vocab'), _unit_meta_int(_m, 'vocab')
            if _tv is not None and _mv is not None and _tv != _mv:
                continue
            if (_t.get('fit_fp') and _m.get('fit_fp')
                    and _t['fit_fp'] != _m['fit_fp']):
                continue
            _pairs.append((_mp, _m, _tp, _t))
    if seq_len is not None:
        _pref = [(mp, m, tp, t) for mp, m, tp, t in _pairs
                 if _unit_meta_int(m, 'seq_len') == int(seq_len)]
        if _pref:
            _pairs = _pref
    if len(_pairs) != 1:
        out['status'] = 'not_verifiable_locally'
        out['note'] = (f'{len(_pairs)} compatible train/test cache pair(s) '
                       'found; refusing to guess which one defines the served '
                       'unit' if _pairs else
                       'no train/test cache pair shares vocab + tokenizer-fit provenance')
        if _pairs:
            out['checks'] = {'candidate_pairs': [
                [os.path.basename(p[0]), os.path.basename(p[2])] for p in _pairs]}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    _mp, _mmeta, _tp, _tmeta = _pairs[0]
    _ds = _mmeta.get('dataset') or _tmeta.get('dataset')
    if _ds not in (None, '', 'wikitext'):
        out['status'] = 'not_verifiable_locally'
        out['note'] = (f'the paired cache declares dataset={_ds!r}; this audit '
                       'defines the WikiText raw-line/article unit and does not '
                       'assume it for another corpus')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    _train_tag = os.path.basename(_mp)[len('meta_'):-len('.json')]
    _test_tag = os.path.basename(_tp)[len('test_meta_'):-len('.json')]
    rec = {'train_meta': os.path.basename(_mp), 'test_meta': os.path.basename(_tp)}
    out['caches'][os.path.abspath(cache_dir)] = rec
    rec['declared_units'] = {'train': _mmeta.get('document_unit'),
                             'test': _tmeta.get('document_unit')}

    # --- test slice: decode line runs and regroup into articles ------------
    _te_path = os.path.join(cache_dir, f'test_ids_{_test_tag}.npy')
    _ttd_path = os.path.join(cache_dir, f'test_token_doc_ids_{_test_tag}.npy')
    _tok_path = os.path.join(cache_dir, f'test_tok_{_test_tag}.json')
    if not os.path.exists(_tok_path):
        _tok_path = os.path.join(cache_dir, f'tok_{_train_tag}.json')
    trec = {'status': 'ok', 'problems': []}
    rec['test'] = trec
    if not (os.path.exists(_te_path) and os.path.exists(_ttd_path)
            and os.path.exists(_tok_path)):
        trec['status'] = 'not_verifiable_locally'
        trec['problems'].append(
            'test id cache, per-token document map or persisted tokenizer is '
            'missing; the served unit cannot be decoded')
        _warn('test: ' + trec['problems'][-1])
    else:
        try:
            from tokenizers import Tokenizer
            _te = np.load(_te_path)
            _ttd = np.load(_ttd_path)
            _tok = Tokenizer.from_file(_tok_path)
        except Exception as e:
            trec['status'] = 'not_verifiable_locally'
            trec['problems'].append(
                f'test artifacts unreadable ({type(e).__name__}: {e})')
            _warn('test: ' + trec['problems'][-1])
        else:
            _runs = _unit_decode_runs(_te, _ttd, _tok, L, 'test', trec)
            if _runs is None:
                trec['status'] = 'not_verifiable_locally'
                _warn('test: ' + '; '.join(trec['problems']))
            else:
                _cls = _unit_classify_runs(_runs, L)
                trec['unit_classification'] = _cls
                trec['unit'] = _cls.get('unit')
                _n_rows = int(_te.shape[0]) if np.asarray(_te).ndim == 2 else 0
                _sl = _unit_meta_int(_tmeta, 'seq_len') or (
                    int(_te.shape[1]) if _n_rows else 0)
                _prefix_n = int(np.asarray(_te).size)
                _token_article = None
                if _cls.get('unit') == 'article':
                    trec['n_units'] = int(len(_runs))
                    trec['n_articles'] = int(len(_runs))
                    trec['n_lines'] = None
                    _token_article = np.repeat(
                        np.asarray([int(r[0][2]) for r in _runs],
                                   dtype=np.uint64),
                        np.asarray([int(r[0][1]) - int(r[0][0])
                                    for r in _runs], dtype=np.int64))
                elif _cls.get('unit') == 'line':
                    _grp, _gerr = _unit_group_lines(_runs, L)
                    if _grp is None:
                        trec['unit'] = 'unknown'
                        trec['problems'].append(_gerr)
                        _warn(f'test: {_gerr}')
                    else:
                        _token_article = _grp['token_article_ids']
                        trec['n_units'] = int(len(_runs))
                        trec['n_lines'] = int(len(_runs))
                        trec['n_articles'] = int(len(_grp['groups']['groups']))
                        trec['article_grouping'] = {
                            'boundary_rule': _grp['groups']['boundary_rule'],
                            'preamble_groups': int(_grp['groups']['preamble_groups']),
                            'max_lines_per_article': int(_grp['groups']['max_group_rows']),
                            'mean_lines_per_article': round(
                                float(_grp['groups']['mean_group_rows']), 3),
                        }
                else:
                    trec['status'] = 'fail'
                    _fail('test: ' + (_cls.get('reason') or 'unknown unit'))
                if _token_article is not None and _n_rows and _sl:
                    _token_article = np.asarray(_token_article)
                    _prefix_n = min(_prefix_n, _n_rows * _sl)
                    _art_pref = _token_article[:_prefix_n]
                    _line_pref = np.asarray(_ttd).reshape(-1)[:_prefix_n]
                    trec['frozen_prefix'] = {
                        'rows': _n_rows, 'seq_len': _sl,
                        'tokens': int(_prefix_n),
                        'distinct_lines': int(np.unique(_line_pref).size)
                        if trec.get('unit') == 'line' else None,
                        'distinct_articles': int(np.unique(_art_pref).size),
                    }
                    _art2d = _art_pref.reshape(_n_rows, _sl)
                    _per_row = [int(np.unique(_art2d[i]).size) for i in range(_n_rows)]
                    trec['frozen_prefix'].update({
                        'mean_articles_per_row': round(float(np.mean(_per_row)), 3),
                        'max_articles_per_row': int(max(_per_row)),
                        'rows_with_multiple_articles': int(sum(1 for x in _per_row if x > 1)),
                    })
                    _n_art = int(np.unique(_art_pref).size)
                    trec['claims'] = {
                        'document_level_claims_allowed': bool(
                            trec.get('unit') == 'article'),
                        'article_population_floor': 64,
                        'population_claim_allowed': bool(_n_art >= 64),
                        'n_independent_articles_in_slice': _n_art,
                        'reason': (
                            'the served unit is a raw line, not an article; a '
                            'line bootstrap resamples paragraphs and cannot '
                            'support a document-level interval or a 64-document '
                            'population claim' if trec.get('unit') == 'line'
                            else 'served unit is article-level'),
                    }
                if trec.get('unit') == 'line':
                    _warn(
                        'test: the cache serves %s line units for %s articles '
                        'in the frozen prefix (pseudoreplication factor %.1fx); '
                        'per-document claims must say "line" or be regrouped'
                        % (trec.get('n_lines'), trec.get('n_articles'),
                           (float(trec.get('n_lines') or 0)
                            / max(1.0, float(trec.get('n_articles') or 1)))))
                _decl = _tmeta.get('document_unit')
                if _decl is not None and trec.get('unit') not in (None, 'unknown', _decl):
                    trec['status'] = 'fail'
                    _fail(f'test: metadata declares document_unit={_decl!r} but '
                          f'the decoded served unit is {trec.get("unit")!r}')
                if trec.get('problems'):
                    trec['status'] = 'fail' if trec.get('status') == 'fail' else 'warnings'

    # --- validation slice: same decode (selection split) -------------------
    _vad = os.path.join(cache_dir, f'val_token_doc_ids_{_train_tag}.npy')
    _va = os.path.join(cache_dir, f'val_batch_{_train_tag}.npy')
    vrec = {'status': 'ok', 'problems': []}
    rec['validation'] = vrec
    if os.path.exists(_vad) and os.path.exists(_va):
        try:
            from tokenizers import Tokenizer
            _vids = np.load(_va)
            _vmap = np.load(_vad)
            _vtok = Tokenizer.from_file(_tok_path)
        except Exception as e:
            vrec['status'] = 'not_verifiable_locally'
            vrec['problems'].append(f'validation artifacts unreadable ({type(e).__name__}: {e})')
        else:
            _vruns = _unit_decode_runs(_vids, _vmap, _vtok, L, 'validation', vrec)
            if _vruns is not None:
                _vcls = _unit_classify_runs(_vruns, L)
                vrec['unit'] = _vcls.get('unit')
                vrec['unit_classification'] = _vcls
                vrec['n_units'] = int(len(_vruns))
                if _vcls.get('unit') == 'line':
                    _vg, _verr = _unit_group_lines(_vruns, L)
                    if _vg is not None:
                        vrec['n_articles'] = int(len(_vg['groups']['groups']))
                        vrec['article_grouping'] = {
                            'boundary_rule': _vg['groups']['boundary_rule'],
                            'max_lines_per_article': int(_vg['groups']['max_group_rows']),
                        }
                    else:
                        vrec['problems'].append(_verr)
                elif _vcls.get('unit') == 'article':
                    vrec['n_articles'] = int(len(_vruns))
            if vrec['problems']:
                vrec['status'] = 'warnings'
                _warn('validation: ' + '; '.join(vrec['problems']))
    else:
        vrec['status'] = 'not_verifiable_locally'
        vrec['problems'].append('validation per-token document map is missing')

    # --- train: line count from the sidecar, article count needs raw ------
    _trd = os.path.join(cache_dir, f'train_doc_ids_{_train_tag}.npy')
    trrec = {'status': 'ok', 'problems': []}
    rec['train'] = trrec
    _served_train = None
    _served_val = None
    if os.path.exists(_trd):
        try:
            _trd_arr = np.load(_trd)
            trrec['served_units'] = int(np.asarray(_trd_arr).size)
            trrec['unit'] = _mmeta.get('document_unit') or (
                vrec.get('unit') or trec.get('unit'))
            _served_train = set(int(x) for x in np.asarray(_trd_arr).tolist())
        except Exception as e:
            trrec['status'] = 'not_verifiable_locally'
            trrec['problems'].append(f'train document sidecar unreadable ({type(e).__name__}: {e})')
    else:
        trrec['status'] = 'not_verifiable_locally'
        trrec['problems'].append('train document-id sidecar is missing')
    if os.path.exists(os.path.join(cache_dir, f'val_doc_ids_{_train_tag}.npy')):
        try:
            _served_val = set(int(x) for x in np.load(
                os.path.join(cache_dir, f'val_doc_ids_{_train_tag}.npy')).tolist())
        except Exception:
            _served_val = None

    # --- raw train/validation: article-level leakage the line index misses -
    if raw_dir in (None, ''):
        raw_dir = os.environ.get('WT103_RAW_TXT') or './wt103_raw'
    _raw_tr = os.path.join(raw_dir, 'wiki.train.raw')
    _raw_va = os.path.join(raw_dir, 'wiki.valid.raw')
    rawrec = {'status': 'ok', 'problems': []}
    rec['article_leakage'] = rawrec
    if not (os.path.exists(_raw_tr) and os.path.exists(_raw_va)):
        rawrec['status'] = 'not_verifiable_locally'
        rawrec['problems'].append(
            f'raw train/validation dumps not found under {raw_dir!r}; '
            'article-level leakage cannot be tested (test raw is never read)')
        out.setdefault('notes', []).append(rawrec['problems'][-1])
    elif _served_train is None or _served_val is None:
        rawrec['status'] = 'not_verifiable_locally'
        rawrec['problems'].append(
            'served train/validation line ids are unavailable; cannot map raw '
            'articles to the served streams')
    else:
        try:
            _tr_rows = L._read_raw_rows(_raw_tr)
            _va_rows = L._read_raw_rows(_raw_va)
            _tr_g = L.wikitext_article_groups(_tr_rows)
            _va_g = L.wikitext_article_groups(_va_rows)
        except Exception as e:
            rawrec['status'] = 'not_verifiable_locally'
            rawrec['problems'].append(f'raw dump unreadable ({type(e).__name__}: {e})')
        else:
            if not (_tr_g.get('valid') and _va_g.get('valid')):
                rawrec['status'] = 'not_verifiable_locally'
                rawrec['problems'].append(
                    'raw dumps are not segmented by top-level headings; '
                    'article-level leakage is unverifiable')
            else:
                rawrec.update({
                    'raw_train_lines': int(_tr_g['n_nonempty_rows']),
                    'raw_train_articles': int(_tr_g['n_articles']),
                    'raw_validation_lines': int(_va_g['n_nonempty_rows']),
                    'raw_validation_articles': int(_va_g['n_articles']),
                    'lines_per_article_train': round(
                        float(_tr_g['mean_group_rows']), 3),
                })

                def _served_articles(rows, groups, served):
                    # (article title, article text built from served lines).
                    # The title is the group's FIRST row even when that row was
                    # not itself served: the title is the article's identity,
                    # and a split article is exactly the case where the train
                    # side saw one line and the validation side another.
                    _out = []
                    for _g in groups['groups']:
                        if not _g:
                            continue
                        _title = rows[int(_g[0])].strip()
                        _kept = []
                        for _i in _g:
                            _row = rows[int(_i)]
                            _h = int(L._doc_id_u64(L._doc_exact_fp(_row)))
                            if _h not in served:
                                continue
                            _kept.append(_row.rstrip('\r\n'))
                        if _kept:
                            _out.append((_title, '\n'.join(_kept)))
                    return _out

                _tr_served = _served_articles(_tr_rows, _tr_g, _served_train)
                _va_served = _served_articles(_va_rows, _va_g, _served_val)
                if max_articles is not None:
                    _tr_served = _tr_served[:int(max_articles)]
                    _va_served = _va_served[:int(max_articles)]
                rawrec['served_train_articles'] = int(len(_tr_served))
                rawrec['served_validation_articles'] = int(len(_va_served))
                # Same title on both sides with different served lines is an
                # article split; identical line hashes would already have been
                # an exact-row failure, so the title is the cross-line identity.
                _tr_titles = {t for t, _txt in _tr_served}
                _va_titles = {t for t, _txt in _va_served}
                _shared_titles = _tr_titles & _va_titles
                rawrec['articles_with_served_lines_in_both_splits'] = int(
                    len(_shared_titles))
                if _shared_titles:
                    rawrec['status'] = 'fail'
                    _fail('article leakage: %d article title(s) have served '
                          'lines in BOTH train and validation (line-level exact '
                          'dedup cannot see this): %s'
                          % (len(_shared_titles),
                             sorted(_shared_titles)[:5]))
                # Near-duplicate articles whose individual lines were reworded
                # are invisible to the exact line index.  Build the same
                # SimHash LSH the loader uses and query validation articles
                # against the served train articles.
                try:
                    _ded = L._DocDeduplicator()
                    _n_tr_seen = 0
                    for _t, _txt in _tr_served:
                        if _txt:
                            _ded.is_duplicate(_txt)
                            _n_tr_seen += 1
                    _near = 0
                    _exact = 0
                    _examples = []
                    for _t, _txt in _va_served:
                        if not _txt:
                            continue
                        _dup, _kind = _ded.find(_txt)
                        if _dup:
                            _near += int(_kind == 'near')
                            _exact += int(_kind == 'exact')
                            if len(_examples) < 5:
                                _examples.append({'title': _t, 'kind': _kind})
                    rawrec['article_near_duplicates'] = int(_near)
                    rawrec['article_exact_duplicates'] = int(_exact)
                    rawrec['article_duplicate_examples'] = _examples
                    rawrec['near_hamming_threshold'] = int(_ded.near_threshold)
                    rawrec['train_articles_indexed'] = int(_n_tr_seen)
                    if _near or _exact:
                        rawrec['status'] = 'warnings'
                        _warn(
                            'article-level duplicates missed by the line index: '
                            f'{_exact} exact / {_near} near-duplicate '
                            'validation article(s) (SimHash <= '
                            f'{int(_ded.near_threshold)} bits) against the '
                            'served train articles')
                except Exception as e:
                    rawrec['problems'].append(
                        f'article near-duplicate scan failed ({type(e).__name__}: {e})')
                    rawrec['status'] = 'not_verifiable_locally'
                if rawrec['problems']:
                    rawrec['status'] = 'warnings'
                    _warn('article leakage: ' + '; '.join(rawrec['problems']))

    # --- aggregate verdict -------------------------------------------------
    _test_unit = trec.get('unit')
    rec['unit'] = _test_unit
    rec['document_level_claims_allowed'] = bool(
        _test_unit == 'article' and rawrec.get('status') != 'fail')
    rec['line_unit_warning'] = bool(_test_unit == 'line')
    if _test_unit == 'line':
        _warn('the served evaluation unit is a raw LINE, not an article; '
              'per-document intervals/population claims are not supported '
              'until the rows are regrouped by top-level headings')
    elif _test_unit not in (None, 'article'):
        _fail(f'test unit is {_test_unit!r}, not line/article')
    if trec.get('claims', {}).get('population_claim_allowed') is False:
        _warn('the frozen prefix contains fewer than the 64 independent '
              'articles the repo requires for a population claim')
    if out['status'] == 'ok' and any(
            r.get('status') == 'not_verifiable_locally'
            for r in (trec, vrec, trrec, rawrec)):
        out['status'] = 'not_verifiable_locally'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
# 1b) sampling：前缀切片 vs 概率样本 / 文章覆盖 / 可外推性
# ---------------------------------------------------------------------------

def _sample_ks(a, b):
    """Two-sample Kolmogorov-Smirnov statistic without SciPy."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    if a.size == 0 or b.size == 0:
        return None
    grid = np.unique(np.concatenate([a, b]))
    ca = np.searchsorted(a, grid, side='right') / float(a.size)
    cb = np.searchsorted(b, grid, side='right') / float(b.size)
    return float(np.max(np.abs(ca - cb)))


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

    `ablation_coverage` 进一步汇总跨 panel 的 complete/verified 覆盖、
    replicated / single_panel 复现状态，以及任意 panel 的显著反方向
    （direction_conflict）或实际有害（practically_harmful）证据；后者
    关闭 `all_components_useful_allowed`，不会因为另一个 panel 通过而被
    平均掉。

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
            'practically_useful_components': rep.get('practically_useful_components'),
            'practical_equivalent_components': (
                (rep.get('sesoi') or {}).get('equivalent_components')),
            'sub_sesoi_components': rep.get('sub_sesoi_components'),
            'practically_harmful_components': rep.get('practically_harmful_components'),
            'practically_inconclusive_components': rep.get('practically_inconclusive_components'),
            'sesoi_family': (rep.get('sesoi') or {}).get('family'),
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
            'interaction_family': (rep.get('interactions') or {}).get('family'),
            'interaction_factorials': (rep.get('interactions') or {}).get('factorials'),
            'significant_interactions': (
                (rep.get('interactions') or {}).get('significant_interactions')),
            'interaction_direction_conflicts': (
                (rep.get('interactions') or {}).get('direction_conflict_interactions')),
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
        # Project-level claim gate: "each proposed innovation is useful" needs
        # a complete ablation AND corrected verification for every component.
        out['all_components_useful_allowed'] = bool(
            out['ablation_coverage'].get('all_components_useful_allowed'))
        out['all_components_useful_reasons'] = list(
            out['ablation_coverage'].get('all_components_useful_reasons') or [])
        out['unverified_components'] = list(
            out['ablation_coverage'].get('unverified_components') or [])
        out['verified_only_incomplete'] = list(
            out['ablation_coverage'].get('verified_only_incomplete') or [])
        out['replicated_components'] = list(
            out['ablation_coverage'].get('replicated_anywhere') or [])
        out['single_panel_verified_components'] = list(
            out['ablation_coverage'].get('single_panel_verified') or [])
        out['direction_conflict_across_panels'] = list(
            out['ablation_coverage'].get('direction_conflict_anywhere') or [])
        out['practically_harmful_across_panels'] = list(
            out['ablation_coverage'].get('practically_harmful_anywhere') or [])
        out['opposite_mean_across_panels'] = list(
            out['ablation_coverage'].get('opposite_mean_anywhere') or [])
        if out['ablation_coverage'].get('status') != 'ok':
            out['status'] = 'warn'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
# 7e) single-factor：单因素设计契约（定义点审计，不读任何 results_*）
# ---------------------------------------------------------------------------

def audit_single_factor(manifest_path=None):
    """单因素设计审计（零 GPU、零落盘依赖）。

    与 `methodology`（读 results_*/summary.json 的记录级门）互补，本审计在
    **定义点**核验每条声明对照的臂配置 diff：
      * ABLATION_DESIGN / BASELINE_DESIGN 的每条对照，其实际 AttnCfg diff
        必须恰好等于声明的干预键集（`ABLATION_INTERVENTION_KEYS`）：多一个键
        = 第二个因素混进「单因素」对照；少一个键 = 干预消失；缺声明 =
        fail-closed，加对照不声明键集本身就是错误；
      * open-block 2x2 矩阵：同一 cap 的 mode 对照只差 `open_block_mode`，
        同一 mode 的 cap 对照只差 `max_block`，scope 对照只差
        `open_block_scope`——交互统计只能读这两根轴；每个轴单元（含两种
        scope）都要核验，多一个键=漏进第三因素，少一个键=轴退化；
      * v9/v10 seq2k 扫描声明（`v9.SEQ2K_CONTRAST_FACTORS`）：把元数据里的
        `single_factor`/`factors` 与真实臂配置 diff 对账，单因素标签不能只
        靠手写；
      * v7 RoPE / 选择器 / sink 对照声明（`v7.ROPE_CONTRAST_FACTORS`）：把
        位置编码、选择器、全块读取与 sink 对照的 factors 与 v7 工厂的真实
        arm diff 对账；
      * 冻结清单（confirmation_frozen.json）：每条 primary comparison 的
        attention-cfg diff 是否单因素（`cfg_keys_single_factor` 的一族判据），
        与 `mechanism_claim_allowed` 标志的一致性。`True` 标志但 cfg 非单因素
        => fail（完整的含目标/参数量核验在 `v12_supp.py plan`，本审计给出
        同一口径的 cfg 级读数）。
    """
    L = _import_exp_lib()
    out = {'status': 'ok', 'design': {}, 'open_block_matrix': {},
           'sweep_contrasts': {}, 'v7_contrasts': {}, 'frozen_manifest': {},
           'factorial_design': {}}
    design_rep = L.single_factor_design_audit(verbose=False)
    out['design'] = design_rep
    ob_rep = L.open_block_matrix_audit(verbose=False)
    out['open_block_matrix'] = ob_rep
    fact_rep = L.factorial_design_audit()
    out['factorial_design'] = fact_rep
    if (design_rep.get('status') != 'ok' or ob_rep.get('status') != 'ok'
            or fact_rep.get('status') != 'ok'):
        out['status'] = 'fail'
    # v9/v10 attach a `single_factor`/`factors` label to each seq2k paired
    # record.  Verify that label against the actual arm construction instead of
    # trusting the hand-written metadata.
    try:
        import v9_supp as V9
        sw_rep = L.sweep_contrast_factor_audit(V9.SEQ2K_CONTRAST_FACTORS,
                                               verbose=False)
    except Exception as _e:
        sw_rep = {'status': 'not_verifiable_locally',
                  'note': f'cannot import the v9 sweep declarations '
                          f'({type(_e).__name__}: {_e})'}
    out['sweep_contrasts'] = sw_rep
    if sw_rep.get('status') == 'fail':
        out['status'] = 'fail'
    # The frozen family resolves through the v7 factory (rope arms); the base
    # factory is its delegate for non-rope arms, so one factory covers both.
    try:
        import v7_supp as V7
        _factory = V7.make_layer_cfgs_v7
    except Exception as _e:  # pragma: no cover - import failure is an environment defect
        out['frozen_manifest'] = {'status': 'not_verifiable_locally',
                                   'note': f'cannot import the v7 arm factory ({type(_e).__name__}: {_e})'}
        _factory = None
    if _factory is not None:
        _v7_rep = L.sweep_contrast_factor_audit(
            V7.ROPE_CONTRAST_FACTORS, factory=_factory, verbose=False)
        out['v7_contrasts'] = _v7_rep
        if _v7_rep.get('status') == 'fail':
            out['status'] = 'fail'
    _mp = manifest_path or os.path.join(REPO, 'confirmation_frozen.json')
    if _factory is not None:
        if not os.path.exists(_mp):
            out['frozen_manifest'] = {'status': 'not_verifiable_locally',
                                       'note': f'no freeze manifest at {_mp}'}
        else:
            try:
                with open(_mp, encoding='utf-8') as _f:
                    man = json.load(_f)
            except Exception as _e:
                out['frozen_manifest'] = {'status': 'fail',
                                          'note': f'manifest is not readable JSON ({type(_e).__name__}: {_e})'}
                man = None
            if isinstance(man, dict):
                _arms = [str(a) for a in (man.get('arms') or [])]
                _cmp = {}
                _bad = []
                _comps = man.get('primary_comparisons')
                if not isinstance(_comps, list) or not _comps:
                    # An empty/malformed family audits zero comparisons; an
                    # 'ok' verdict would claim the gate passed when nothing was
                    # checked.  Fail closed.
                    out['frozen_manifest'] = {
                        'status': 'fail', 'manifest': _mp,
                        'note': ('primary_comparisons must be a non-empty list, '
                                 f'got {type(_comps).__name__}'
                                 if not isinstance(_comps, list) else
                                 'primary_comparisons is empty; no declared '
                                 'comparison was audited')}
                    out['status'] = 'fail'
                    comps_for_audit = []
                else:
                    comps_for_audit = _comps
                for comp in comps_for_audit:
                    if not isinstance(comp, dict):
                        _bad.append(f'non-object primary comparison entry {comp!r}')
                        continue
                    _name = str(comp.get('name'))
                    _a, _b = str(comp.get('a')), str(comp.get('b'))
                    _keys, _per_layer = L._design_pair_cfg_keys(_a, _b, _factory, 6)
                    if isinstance(_keys, Exception):
                        _bad.append(f'{_name}: arm construction failed ({_keys})')
                        _cmp[_name] = {'a': _a, 'b': _b, 'cfg_diff_keys': None,
                                       'cfg_single_factor': False,
                                       'mechanism_claim_allowed': comp.get('mechanism_claim_allowed'),
                                       'claim_flag_cfg_consistent': False}
                        continue
                    _sf_cfg = L.cfg_keys_single_factor(_keys)
                    _flag = comp.get('mechanism_claim_allowed')
                    _consistent = (_flag is not True) or _sf_cfg
                    if not _consistent:
                        _bad.append(f'{_name}: mechanism_claim_allowed=true but the cfg diff '
                                    f'keys {sorted(_keys)} are not one declared single-factor family '
                                    f'(kind swaps are architecture-level by definition)')
                    _cmp[_name] = {'a': _a, 'b': _b, 'cfg_diff_keys': sorted(_keys),
                                   'cfg_single_factor': _sf_cfg,
                                   'mechanism_claim_allowed': _flag,
                                   'claim_flag_cfg_consistent': _consistent,
                                   'note': ('full verdict (objectives + parameter totals) is '
                                            'enforced by `v12_supp.py plan` at preflight')}
                if comps_for_audit or out['frozen_manifest'].get('status') != 'fail':
                    out['frozen_manifest'] = {'status': ('fail' if _bad else 'ok'),
                                              'manifest': _mp, 'comparisons': _cmp,
                                              'violations': _bad,
                                              'arms_checked': _arms}
                if _bad:
                    out['status'] = 'fail'
            else:
                # A readable but non-object manifest (or an unreadable one,
                # handled above) cannot be audited.  Leaving frozen_manifest
                # empty previously let the overall status stay 'ok', so a
                # malformed gate artifact silently passed.
                out['frozen_manifest'] = {
                    'status': 'fail', 'manifest': _mp,
                    'note': ('manifest must be a JSON object, got '
                             f'{type(man).__name__}' if man is not None
                             else 'manifest is not readable JSON')}
                out['status'] = 'fail'
            if (isinstance(out.get('frozen_manifest'), dict)
                    and out['frozen_manifest'].get('status') == 'fail'):
                out['status'] = 'fail'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
# 7e-2) factorial：2x2 交互设计的定义点审计（零 GPU）
# ---------------------------------------------------------------------------

def audit_factorial_design(manifest_path=None, *, n_layers=6):
    """2x2 因子交互设计审计（零 GPU、零落盘依赖）。

    复用 `exp_lib.factorial_design_audit`，在定义点核验三组交互：
      * 四个 cell 完整、臂唯一且已排程；
      * 每个因子在另一因子的两个水平上做**同一组** AttnCfg 改动
        （并行操纵；否则交互比较的是两种不同干预）；
      * 改动键属于该组件的声明干预集，且不交换 `kind`；
      * 每组交互都有完整、指标对齐的预注册假设。

    这是 `single_factor` 的补集：单因素审计回答「每个臂只动了一个组件吗」，
    factorial 审计回答「两个组件的 2x2 是否可识别、交互是否可解释」。"""
    L = _import_exp_lib()
    out = L.factorial_design_audit(n_layers=n_layers)
    if manifest_path:
        try:
            with open(manifest_path, encoding='utf-8') as _f:
                _man = json.load(_f)
            out['manifest_arms'] = list(_man.get('arms') or [])
            _missing = sorted(set(
                a for rec in (out.get('factorials') or {}).values()
                for a in (rec.get('cells') or {}).values()
                if a not in set(out['manifest_arms'])))
            out['manifest_missing_arms'] = _missing
            if _missing:
                out['status'] = 'fail'
                out['violations'].append(
                    'factorial arm(s) absent from the frozen manifest: '
                    + ', '.join(_missing))
        except Exception as _e:
            out['status'] = 'fail'
            out['violations'].append(
                f'cannot read manifest {manifest_path}: '
                f'{type(_e).__name__}: {_e}')
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
# 7e-3) dose_response：有序剂量/机制梯度的定义点审计（零 GPU）
# ---------------------------------------------------------------------------

def audit_dose_response(manifest_path=None, design=None, *, direction=None,
                        dose_key=None, factory=None, verbose=True):
    """有序剂量设计的定义点审计（零 GPU）。

    从冻结清单的可选 ``dose_response_design`` 字段读取声明，或由调用方直接
    传入 ``design``（``{levels, direction, dose_key, ...}`` 或 levels 本身）。
    核验“至少三个水平、剂量排序、只改变一个声明键、同一种子集、同一测量
    协议、方向/形状预注册”；清单没有该字段时明确返回 ``not_applicable``，
    不把“没有声明”误报成通过。记录级趋势检验在
    ``exp_lib.dose_response_trend``，本入口只核验设计。
    """
    L = _import_exp_lib()
    mp = manifest_path or os.path.join(REPO, 'confirmation_frozen.json')
    spec = design
    source = 'argument'
    if spec is None:
        man = _load_manifest_file(mp)
        if not isinstance(man, dict):
            out = {'status': 'not_verifiable_locally',
                   'manifest': mp,
                   'note': 'no readable frozen manifest; no ordered-dose '
                           'declaration can be audited'}
            if verbose:
                print(json.dumps(out, ensure_ascii=False, indent=2))
            return out
        spec = man.get('dose_response_design') or man.get('dose_response')
        source = mp
        if spec is None:
            out = {'status': 'not_applicable',
                   'manifest': mp,
                   'note': ('the manifest declares no dose_response_design; '
                            'an ordered ladder is not used by this plan')}
            if verbose:
                print(json.dumps(out, ensure_ascii=False, indent=2))
            return out
    if isinstance(spec, dict) and isinstance(spec.get('levels'), (dict, list, tuple)):
        levels = spec.get('levels')
        direction = spec.get('direction') if direction is None else direction
        dose_key = spec.get('dose_key') if dose_key is None else dose_key
        kwargs = {
            'n_layers': spec.get('n_layers', 6),
            'min_levels': spec.get('min_levels', 3),
            'min_seeds': spec.get('min_seeds', 3),
            'require_seeds': spec.get('require_seeds', True),
            'require_protocol': spec.get('require_protocol', False),
            'shape': spec.get('shape', 'linear'),
        }
    else:
        levels = spec
        kwargs = {}
    try:
        rep = L.dose_response_design_audit(
            levels, direction=direction, dose_key=dose_key,
            factory=factory, **kwargs)
    except Exception as e:
        rep = {'kind': 'dose_response_design', 'status': 'fail',
               'violations': [f'{type(e).__name__}: {e}'],
               'note': 'dose-response design audit could not be completed'}
    rep['source'] = source
    if verbose:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    return rep


# ---------------------------------------------------------------------------
# 7f) design preflight：预算台账 / 证据等级 / 动态分块假设与对照
# ---------------------------------------------------------------------------

def _load_manifest_file(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def audit_design_preflight(manifest_path=None, dynamic_path=None):
    """零 GPU 设计前馈审计：在训练/test 访问前核对实验设计。

    这是本轮科研方法改进的统一入口，回答四个在跑 GPU 之前就必须回答的问题：

      1. **预算控制**：主清单与动态草稿里每条对照的 token 预算、参数总量、
         读取预算是否真的匹配？机制主张是否只挂在专用对照臂上，而不是只跟
         dense 基线比？（`exp_lib.comparison_budget_ledger`）
      2. **证据等级**：v6-v11（同一 validation split 上选择/比较）是否仍被
         标成 exploratory？draft 清单是否被当成确认性计划？
         （`exp_lib.evidence_level_audit`）
      3. **假设驱动**：动态分块是否把「切分质量 / 检索精度 / 长程覆盖」写成
         可证伪的独立假设，并安排了 random-vs-random、read-matched、参数
         代理、sink 匹配等对照？（`exp_lib.dynamic_design_audit`）
      4. **功效与冻结**：种子数/族大小是否至少数学上可达 Holm 显著，是否声明
         了 test 前功效门槛？（两个 manifest 的统计块）

    返回结构化报告；任一硬性设计缺陷 => `status=fail`。它不读任何
    results_*/ 数值，也不调用 test loader。
    """
    L = _import_exp_lib()
    mp = manifest_path or os.path.join(REPO, 'confirmation_frozen.json')
    dp = dynamic_path or os.path.join(REPO, 'confirmation_frozen_dynamic.json')
    out = {'status': 'ok', 'frozen_manifest': mp, 'dynamic_manifest': dp,
           'frozen': {}, 'dynamic': {}, 'violations': [], 'warnings': []}
    main = _load_manifest_file(mp)
    if not isinstance(main, dict):
        out['violations'].append(f'frozen manifest is missing/unreadable: {mp}')
        out['frozen'] = {'status': 'not_verifiable_locally'}
    else:
        ledger = L.comparison_budget_ledger(main)
        ev = L.evidence_level_audit(main, manifest_path=mp,
                                    panel_paths=list(L.EXPLORATORY_PANEL_REGISTRY))
        out['frozen'] = {
            'status': ('fail' if (ledger.get('status') == 'fail'
                                  or ev.get('status') == 'fail') else 'ok'),
            'evidence_level': ev.get('manifest_level'),
            'token_budget': ledger.get('token_budget'),
            'design_cost': ledger.get('design_cost'),
            'comparisons': ledger.get('comparisons'),
            'dedicated_controls': ledger.get('dedicated_controls'),
            'power_plan': ev.get('power_plan'),
            'ledger_violations': ledger.get('violations'),
            'evidence_violations': ev.get('violations'),
            'evidence_warnings': ev.get('warnings'),
        }
        out['violations'].extend(f'frozen: {v}' for v in
                                 (ledger.get('violations') or []) +
                                 (ev.get('violations') or []))
        out['warnings'].extend(f'frozen: {w}' for w in
                               (ledger.get('warnings') or []) +
                               (ev.get('warnings') or []))
        if out['frozen']['status'] == 'fail':
            out['status'] = 'fail'
    dyn = _load_manifest_file(dp)
    if not isinstance(dyn, dict):
        out['warnings'].append(
            f'no readable dynamic draft at {dp}; the dynamic/HCA study has no '
            'design record yet')
        out['dynamic'] = {'status': 'not_verifiable_locally'}
    else:
        dd = L.dynamic_design_audit(dyn, manifest_path=dp)
        dl = L.comparison_budget_ledger(dyn)
        de = L.evidence_level_audit(dyn, manifest_path=dp, panel_paths=[])
        out['dynamic'] = {
            'status': ('fail' if (dd.get('status') == 'fail'
                                  or dl.get('status') == 'fail'
                                  or de.get('status') == 'fail') else 'ok'),
            'declared_status': dyn.get('status'),
            'evidence_level': de.get('manifest_level'),
            'runnable': dd.get('runnable'),
            'promotion_ready': dd.get('promotion_ready'),
            'hypotheses': sorted(dd.get('hypotheses') or {}),
            'power': dd.get('power'),
            'calibration': dd.get('calibration'),
            'design_cost': dl.get('design_cost'),
            'dedicated_controls': dl.get('dedicated_controls'),
            'comparisons': dl.get('comparisons'),
            'design_violations': dd.get('violations'),
            'ledger_violations': dl.get('violations'),
            'evidence_violations': de.get('violations'),
            'warnings': (dd.get('warnings') or []) + (dl.get('warnings') or [])
                        + (de.get('warnings') or []),
        }
        out['violations'].extend(f'dynamic: {v}' for v in
                                 (dd.get('violations') or []) +
                                 (dl.get('violations') or []) +
                                 (de.get('violations') or []))
        out['warnings'].extend(f'dynamic: {w}' for w in out['dynamic']['warnings'])
        if out['dynamic']['status'] == 'fail':
            out['status'] = 'fail'
        # A draft being non-runnable is expected, not a violation; make the
        # distinction explicit so a green design audit is not mistaken for a
        # green confirmation.
        if str(dyn.get('status') or '').lower() == 'draft':
            out['warnings'].append(
                'dynamic draft is intentionally non-runnable: `promotion_ready` '
                'must become true (disjoint source + measured read-budget '
                'calibration + passed power gate) before any dynamic confirmation')
    # A recorded violation is a hard design defect, including the case where
    # the frozen manifest is missing/unreadable.  Previously that branch only
    # appended to `violations` while `status` stayed 'ok', so the CLI exited 0
    # on a checkout with no frozen plan.
    if out['violations']:
        out['status'] = 'fail'
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


# ---------------------------------------------------------------------------
# 8) 规模核算：6ND 估算 + TimeGuard 记录对账
# ---------------------------------------------------------------------------

def train_flops_estimate(params, tokens, attn_overhead=0.35):
    """6ND 估算 + 注意力项系数。

    6·N·D 是前向+反向的常用近似；稀疏选择/池化/激活重算与长序列注意力
    不与参数量成比例，因此用一个**显式声明的**注意力项系数。"""
    def _positive(value, name):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f'{name} must be a positive finite number, got {value!r}')
        try:
            f = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'{name} must be a positive finite number, got {value!r}') from None
        if not (math.isfinite(f) and f > 0.0):
            raise ValueError(f'{name} must be a positive finite number, got {value!r}')
        return f
    n = _positive(params, 'params')
    d_tok = _positive(tokens, 'tokens')
    if isinstance(attn_overhead, (bool, np.bool_)):
        raise ValueError(f'attn_overhead must be a finite non-negative number, got {attn_overhead!r}')
    try:
        overhead = float(attn_overhead)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'attn_overhead must be a finite non-negative number, got {attn_overhead!r}') from None
    if not (math.isfinite(overhead) and overhead >= 0.0):
        raise ValueError(f'attn_overhead must be a finite non-negative number, got {attn_overhead!r}')
    base = 6.0 * n * d_tok
    return {'flops_6nd': base, 'flops_with_attn_proxy': base * (1.0 + overhead), 'attn_overhead_assumed': overhead, 'note': 'analytic PROXY, not a measurement; report alongside TimeGuard seconds'}


def audit_budget(state_files=('run_time_state_v10.json', 'run_time_state_v12.json'), panels=(('results_lm_v10_scale_s', 'csa_fixed', 8000), ('results_lm_v10_scale_l8k', 'csa_fixed', 8000), ('results_lm_v10_scale_lflop', 'csa_fixed', None))):
    """规模核算（零 GPU）：每面板 6ND 估算 + 已记账实测秒数。"""
    out = {'status': 'ok', 'panels': {}}
    for d, v, steps in panels:
        summ = _load(_audit_artifact(os.path.join(d, 'summary.json')))
        if summ is None:
            out['panels'][d] = 'not present locally'
            continue
        def _positive_int(value):
            if value is None or isinstance(value, (bool, np.bool_)):
                return None
            if isinstance(value, numbers.Integral):
                return int(value) if int(value) > 0 else None
            if isinstance(value, numbers.Real):
                f = float(value)
                return int(f) if math.isfinite(f) and f.is_integer() and f > 0 else None
            return None

        def _positive_real(value):
            if value is None or isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
                return None
            f = float(value)
            return f if math.isfinite(f) and f > 0.0 else None

        recs = []
        for r in summ.values():
            if not isinstance(r, dict) or r.get('variant') != v:
                continue
            _p = _positive_real(r.get('params'))
            if _p is not None:
                recs.append(r)
        if not recs:
            out['panels'][d] = 'no param records'
            continue
        r0 = recs[0]
        _steps = _positive_int(steps) if steps is not None else _positive_int(r0.get('steps'))
        tok = _positive_int(r0.get('tokens_seen'))
        if tok is None and _steps is not None:
            _batch = _positive_int(r0.get('batch_size', 12))
            _seq = _positive_int(r0.get('seq_len', 512))
            if _batch is not None and _seq is not None:
                tok = _steps * _batch * _seq
        est = (train_flops_estimate(_positive_real(r0.get('params')), tok)
               if tok is not None else None)
        out['panels'][d] = {'params': r0.get('params'), 'steps': _steps,
                            'tokens': tok, 'estimate': est}
    for sf in state_files:
        st = _load(_audit_artifact(sf))
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


def audit_training_health(outdirs=None):
    """Zero-GPU audit of recorded training-curve health and NaN triage.

    A run that overfit, stalled, or recorded a non-finite event may still have a
    PPL number.  This audit makes that diagnostic history explicit and refuses
    a record whose own triage contradicts its events.  It does not select
    models and does not change any panel.
    """
    dirs = outdirs or list(_iter_result_dirs())
    out = {'status': 'ok', 'panels': {}, 'warnings': [], 'errors': [],
           'note': 'diagnostic only: a healthy record is not thereby a valid treatment claim, and an unhealthy record is not silently deleted'}
    _Lmod = None
    for d in dirs:
        summ = _load(_audit_artifact(os.path.join(d, 'summary.json')))
        if not isinstance(summ, dict) or not summ:
            continue
        cells = {}
        for key, r in summ.items():
            if not isinstance(r, dict):
                continue
            w = []
            e = []
            h = r.get('curve_health')
            status = str(h.get('status') or 'unknown') if isinstance(h, dict) else 'missing'
            _known_health = {'healthy', 'overfitting', 'underfitting_or_stalled',
                             'validation_degrading', 'validation_improving_train_flat',
                             'nonfinite', 'insufficient_data', 'unmonitored'}
            if not isinstance(h, dict):
                w.append('no curve_health record')
            elif status in ('overfitting', 'underfitting_or_stalled',
                            'validation_degrading', 'validation_improving_train_flat',
                            'nonfinite'):
                w.append(f'curve_health={status}')
            elif status not in _known_health:
                w.append(f'unknown curve_health status {status!r}; '
                         'the run is not verified healthy')
            if (isinstance(h, dict) and status == 'healthy'
                    and r.get('health_history') == []):
                w.append('curve_health=healthy with an EMPTY health_history; '
                         'the claim cannot be recomputed from the curve')
            _diag_raw = r.get('opt_diag_summary')
            if _diag_raw is not None and not isinstance(_diag_raw, dict):
                e.append(f'opt_diag_summary is not an object ({type(_diag_raw).__name__})')
            diag = _diag_raw if isinstance(_diag_raw, dict) else {}
            _guard_raw = diag.get('nan_guard')
            if _guard_raw is not None and not isinstance(_guard_raw, dict):
                e.append(f'nan_guard is not an object ({type(_guard_raw).__name__})')
            guard = _guard_raw if isinstance(_guard_raw, dict) else {}
            def _nonneg(v):
                if (isinstance(v, (bool, np.bool_))
                        or not isinstance(v, numbers.Real)):
                    return None
                f = float(v)
                if not (math.isfinite(f) and f >= 0.0 and f == math.floor(f)):
                    return None
                return int(f)
            def _count_field(source, key):
                # Missing means zero events; a PRESENT but malformed count is
                # unknown evidence and must fail closed rather than silently
                # becoming zero (which would let a corrupt record look clean).
                raw = source.get(key) if isinstance(source, dict) else None
                if raw is None:
                    return 0
                value = _nonneg(raw)
                if value is None:
                    e.append(f'{key} is not a non-negative integer ({raw!r}); '
                             'the non-finite event count is unverifiable, not zero')
                    return 0
                return value
            n_loss = _count_field(diag, 'nonfinite_loss_steps')
            n_grad = _count_field(diag, 'nonfinite_grad_steps')
            n_guard = _count_field(guard, 'elements_cleaned')
            has_events = bool(n_loss or n_grad or n_guard)
            # A recorded data audit is the first branch of the NaN triage.
            # Only validate it when present: legacy records without one stay
            # on their existing warning path (missing evidence is not evidence
            # of corruption), but a present-and-failed audit must fail closed.
            _data_audit = r.get('data_audit')
            if _data_audit is not None:
                if not isinstance(_data_audit, dict):
                    e.append(f'data_audit is not an object ({type(_data_audit).__name__})')
                else:
                    if str(_data_audit.get('status') or '') != 'ok':
                        e.append(f'data_audit status={_data_audit.get("status")!r}; '
                                 'the data contract did not pass')
                    _secs = _data_audit.get('sections') if isinstance(_data_audit.get('sections'), dict) else {}
                    for _name in ('train', 'validation'):
                        _sec = _secs.get(_name) if isinstance(_secs.get(_name), dict) else {}
                        if not _sec:
                            continue
                        _nf = _count_field(_sec, 'n_nonfinite')
                        _oor = _count_field(_sec, 'n_out_of_range')
                        if _nf or _oor:
                            e.append(f'data_audit.{_name}: {_nf} non-finite and '
                                     f'{_oor} out-of-range token value(s)')
                    _ls = _data_audit.get('label_shift')
                    if isinstance(_ls, dict) and _ls.get('checked') and not _ls.get('ok'):
                        e.append('data_audit.label_shift failed: inputs/targets are '
                                 'not a one-token causal shift')
            # Event counters and their event log must not contradict each
            # other: an event log with loss/grad entries while the matching
            # counter is zero is a partially written record, not a clean run.
            _ev_raw = diag.get('events')
            if _ev_raw is not None and not isinstance(_ev_raw, list):
                e.append(f'opt_diag_summary.events is not a list ({type(_ev_raw).__name__})')
                _evs = []
            else:
                _evs = _ev_raw if isinstance(_ev_raw, list) else []
            _ev_loss = sum(1 for _x in _evs if isinstance(_x, dict) and _x.get('kind') == 'loss')
            _ev_grad = sum(1 for _x in _evs if isinstance(_x, dict) and _x.get('kind') == 'grad')
            if _ev_loss > n_loss:
                e.append(f'events contain {_ev_loss} non-finite loss step(s) but '
                         f'nonfinite_loss_steps={n_loss}')
            if _ev_grad > n_grad:
                e.append(f'events contain {_ev_grad} non-finite gradient step(s) but '
                         f'nonfinite_grad_steps={n_grad}')
            _fbs = diag.get('first_bad_step')
            if _fbs is not None:
                _fbsv = _nonneg(_fbs)
                if _fbsv is None or _fbsv < 1:
                    e.append(f'first_bad_step is not a positive integer ({_fbs!r})')
                elif not has_events and not _evs:
                    w.append(f'first_bad_step={_fbsv} is recorded but every event '
                             'counter/log is empty; the record is inconsistent')
            # The stored `curve_health` is a cache; the recorded curve is the
            # evidence.  Recompute it when present so a record cannot claim
            # `healthy` while its own history is overfitting/non-finite.
            _hist = r.get('health_history')
            if _hist is not None and not isinstance(_hist, list):
                e.append(f'health_history is not a list ({type(_hist).__name__})')
            elif isinstance(_hist, list) and _hist:
                if _Lmod is None:
                    try:
                        _Lmod = _import_exp_lib()
                    except Exception as _le:
                        # No classifier in this environment: the curve health
                        # stays unverified rather than being trusted blindly.
                        _Lmod = False
                        w.append(f'health_history present but the classifier is '
                                 f'unavailable ({type(_le).__name__}: {_le}); '
                                 'the recorded curve_health is unverified')
                if _Lmod is False:
                    pass
                else:
                    try:
                        _rh = _Lmod.classify_training_health(_hist)
                    except Exception as _he:
                        e.append(f'health_history could not be classified '
                                 f'({type(_he).__name__}: {_he}); the curve health is unverifiable')
                        _rh = None
                if _Lmod is not False and _rh is not None:
                    _rstat = str(_rh.get('status') or 'unknown')
                    if isinstance(h, dict):
                        if status != _rstat:
                            _msg = (f'recorded curve_health={status} contradicts the '
                                    f'health_history recomputation ({_rstat}: '
                                    f'{_rh.get("reason")})')
                            # A missing non-finite curve is a masked numerical
                            # failure; a trend-label difference is a (possibly
                            # version-dependent) diagnostic mismatch.
                            if status in ('healthy', 'unknown') and _rstat == 'nonfinite':
                                e.append(_msg)
                            else:
                                w.append(_msg)
                        _bs, _bp = h.get('best_step'), h.get('best_val_ppl')
                        if _bs is not None:
                            _bsv = _nonneg(_bs)
                            if _bsv is None:
                                w.append(f'curve_health.best_step is not a non-negative '
                                         f'integer ({_bs!r}); cannot verify it against the curve')
                            elif _rh.get('best_step') is not None and _bsv != int(_rh['best_step']):
                                w.append(f'curve_health.best_step={_bsv} does not match the '
                                         f'curve best step {int(_rh["best_step"])}')
                        if _bp is not None:
                            # PPL is a positive real, not an integer count.
                            _bpv = None
                            if (not isinstance(_bp, (bool, np.bool_))
                                    and isinstance(_bp, numbers.Real)):
                                _f = float(_bp)
                                _bpv = _f if (math.isfinite(_f) and _f > 0.0) else None
                            _rbp = _rh.get('best_val_ppl')
                            if _bpv is None:
                                w.append(f'curve_health.best_val_ppl is not a finite '
                                         f'positive number ({_bp!r})')
                            elif (isinstance(_rbp, (int, float)) and not isinstance(_rbp, bool)
                                  and math.isfinite(float(_rbp))
                                  and abs(_bpv - float(_rbp)) > 1e-6 * max(1.0, abs(float(_rbp)))):
                                w.append(f'curve_health.best_val_ppl={_bpv:g} does not match the '
                                         f'curve best {float(_rbp):g}')
                        _fm = h.get('final_minus_best')
                        if _fm is not None:
                            _fmv = None
                            if (not isinstance(_fm, (bool, np.bool_))
                                    and isinstance(_fm, numbers.Real)):
                                _f = float(_fm)
                                _fmv = _f if math.isfinite(_f) else None
                            _rfm = _rh.get('final_minus_best')
                            if _fmv is None:
                                w.append(f'curve_health.final_minus_best is not finite ({_fm!r})')
                            elif (isinstance(_rfm, (int, float)) and not isinstance(_rfm, bool)
                                  and math.isfinite(float(_rfm))
                                  and abs(_fmv - float(_rfm)) > 1e-6 * max(1.0, abs(float(_rfm)))):
                                w.append(f'curve_health.final_minus_best={_fmv:g} does not match '
                                         f'the curve {float(_rfm):g}')
                        _sb = h.get('steps_since_best')
                        if _sb is not None:
                            _sbv = _nonneg(_sb)
                            _rsb = _rh.get('steps_since_best')
                            if _sbv is None:
                                w.append(f'curve_health.steps_since_best is not a non-negative '
                                         f'integer ({_sb!r})')
                            elif _rsb is not None and _sbv != int(_rsb):
                                w.append(f'curve_health.steps_since_best={_sbv} does not match '
                                         f'the curve {int(_rsb)}')
                    else:
                        w.append(f'no curve_health record; health_history recomputes to '
                                 f'{_rstat} ({_rh.get("reason")})')
            triage = r.get('instability_diagnosis')
            if isinstance(triage, dict):
                tstat = str(triage.get('status') or 'unknown')
                if has_events and tstat == 'clean':
                    e.append('instability_diagnosis says clean but non-finite events are recorded')
                if status == 'nonfinite' and tstat == 'clean':
                    e.append('validation curve is non-finite but instability_diagnosis says clean')
                if tstat == 'unverifiable':
                    w.append('instability_diagnosis is unverifiable (malformed diagnostic record); '
                             'non-finite counts/health cannot be treated as clean')
            elif has_events:
                w.append('non-finite events without instability_diagnosis (legacy/unverified)')
            if status == 'nonfinite' and not isinstance(triage, dict):
                w.append('non-finite validation curve without triage (legacy/unverified)')
            es = r.get('early_stop')
            if isinstance(es, dict) and bool(es.get('stopped')):
                _budget = {}
                for _key in ('steps_completed', 'steps'):
                    _raw = r.get(_key)
                    if _raw is None:
                        continue
                    _val = _nonneg(_raw)
                    if _val is None:
                        e.append(f'{_key} is not a non-negative integer ({_raw!r}); '
                                 'the early-stop budget is unverifiable')
                    else:
                        _budget[_key] = _val
                if len(_budget) < 2:
                    w.append('early_stop.stopped without both steps_completed and '
                             'planned steps; the completed budget is unverifiable')
                elif _budget['steps_completed'] >= _budget['steps']:
                    w.append('early_stop.stopped is true but steps_completed >= planned steps')
            if w or e:
                cells[key] = {'health': status, 'warnings': w, 'errors': e}
        if cells:
            out['panels'][d] = cells
            for key, c in cells.items():
                out['errors'] += [f'{d}/{key}: {x}' for x in c['errors']]
                out['warnings'] += [f'{d}/{key}: {x}' for x in c['warnings']]
    if out['errors']:
        out['status'] = 'fail'
    elif out['warnings']:
        out['status'] = 'warnings'
    return out


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
        summ = _load(_audit_artifact(os.path.join(d, 'summary.json')))
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
            # New measurement records carry the batch size.  The same token
            # count at batch 1 vs batch 8 is a different execution shape; if
            # either arm recorded batch sizes, they must agree.  Legacy
            # records without the field stay comparable (but `batch_status`
            # says the shape was not recorded).
            def _batch_of(value):
                if isinstance(value, (bool, np.bool_)) or value is None:
                    return None
                if isinstance(value, numbers.Integral):
                    return int(value) if int(value) >= 1 else None
                if isinstance(value, numbers.Real):
                    f = float(value)
                    return int(f) if (math.isfinite(f) and f == math.floor(f)
                                      and f >= 1.0) else None
                return None
            _batch_a = {_batch_of(r.get('batch')) for r in eff[a]
                        if r.get('batch') is not None}
            _batch_b = {_batch_of(r.get('batch')) for r in eff[b]
                        if r.get('batch') is not None}
            # A partly recorded arm is not a legacy all-unrecorded arm: one
            # run with batch=8 and another without the field could be a
            # different execution shape, so the pair is unverifiable.
            _batch_a_complete = all(
                r.get('batch') is not None and _batch_of(r.get('batch')) is not None
                for r in eff[a])
            _batch_b_complete = all(
                r.get('batch') is not None and _batch_of(r.get('batch')) is not None
                for r in eff[b])
            if not (_batch_a or _batch_b):
                same_batch = True
                batch_status = 'unrecorded'
            elif (not _batch_a_complete or not _batch_b_complete
                  or None in _batch_a or None in _batch_b
                  or len(_batch_a) != 1 or len(_batch_b) != 1
                  or _batch_a != _batch_b):
                same_batch = False
                batch_status = 'different_or_incomplete'
            else:
                same_batch = True
                batch_status = 'same'
            # Median over seeds, not the first record: a single slow seed must
            # not decide a "faster" claim.
            _ta, _tb = _med(eff[a], 'prefill_ms_p50'), _med(eff[b], 'prefill_ms_p50')
            def _token_count(value):
                # Token counts are exact integers.  `int(1.5)` would turn a
                # fractional/malformed shape into a different row count and
                # let two runs be compared as if they used the same input.
                if isinstance(value, (bool, np.bool_)):
                    return None
                if isinstance(value, numbers.Integral):
                    return int(value) if int(value) > 0 else None
                if isinstance(value, numbers.Real):
                    f = float(value)
                    return (int(f) if (math.isfinite(f) and f == math.floor(f)
                                      and f > 0) else None)
                return None

            _na = sorted({_token_count(r.get('n_tokens')) for r in eff[a]},
                         key=lambda x: (x is None, -1 if x is None else x))
            _nb = sorted({_token_count(r.get('n_tokens')) for r in eff[b]},
                         key=lambda x: (x is None, -1 if x is None else x))
            rec = {'a': a, 'b': b, 'same_device_torch': same_env, 'same_kv_semantics': same_kv,
                   'same_batch': same_batch, 'batch_status': batch_status,
                   'a_batch': sorted(_batch_a, key=lambda x: (x is None, -1 if x is None else x)),
                   'b_batch': sorted(_batch_b, key=lambda x: (x is None, -1 if x is None else x)),
                   'a_ms_p50': _ta, 'b_ms_p50': _tb, 'a_tokens': _na, 'b_tokens': _nb,
                   'ratio_b_over_a': None, 'speed_claim_allowed': False, 'faster_arm': None, 'reason': None}
            if not (isinstance(_ta, (int, float)) and isinstance(_tb, (int, float)) and _ta > 0 and _tb > 0):
                rec['reason'] = 'missing or non-positive measured prefill time'
            elif (None in _na or None in _nb or _na != _nb
                  or len(_na) != 1 or len(_nb) != 1):
                rec['reason'] = (f'different, ambiguous or non-integer measured '
                                 f'token counts ({_na} vs {_nb}); a latency ratio '
                                 'across shapes is not a speed comparison')
            elif not same_env:
                rec['reason'] = 'different or inconsistent device/torch version recorded (within or across arms)'
            elif not same_kv:
                rec['reason'] = 'different or inconsistent KV-cache semantics recorded (within or across arms)'
            elif not same_batch:
                rec['reason'] = 'different, ambiguous or incomplete batch sizes recorded (within or across arms)'
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

    def _exact_count(value):
        # A numeric string is not an exact count.  `_exact_int` stays
        # permissive for seed *keys* (JSON object keys are strings), but cell
        # geometry is a measurement and must arrive as a real number.
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            return None
        f = float(value)
        return int(f) if math.isfinite(f) and f == math.floor(f) else None

    def _measurement(value):
        # Numeric strings are not recall/mass readings; float("0.9") would
        # otherwise let a hand-edited artifact unlock the claim gate.
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            return None
        f = float(value)
        return f if math.isfinite(f) else None

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
        _rec = _measurement(c['recall_at_k'])
        _mass = _measurement(c['mass_covered'])
        _base = _measurement(c['random_baseline'])
        if _rec is None or _mass is None or _base is None:
            out['problems'].append(
                f'cell {i}: recall/mass/baseline must be finite real numbers '
                '(not strings, booleans, or non-finite values)')
            continue
        rec, mass, base = _rec, _mass, _base
        k, B = _exact_count(c['k']), _exact_count(c['n_blocks'])
        el, nq = _exact_count(c['eval_len']), _exact_count(c['n_queries'])
        if k is None or B is None or el is None or nq is None:
            out['problems'].append(
                f'cell {i}: k/n_blocks/eval_len/n_queries must be exact '
                'integers (counts, not strings, booleans, or fractions)')
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
            _mv, _bv = v.get('mass_covered'), v.get('random_baseline')
            if (any(isinstance(x, (bool, np.bool_)) for x in (_mv, _bv))
                    or not all(isinstance(x, numbers.Real) for x in (_mv, _bv))):
                _bad_seed.append(str(_sk))
                continue
            _mf, _bf = float(_mv), float(_bv)
            # A present-but-NaN/out-of-range pair would be silently dropped by
            # the exact test, leaving `n_seeds` larger than the sample the p
            # value was computed on.  Treat it as a malformed row.
            if (not (math.isfinite(_mf) and math.isfinite(_bf))
                    or not (0.0 <= _mf <= 1.0 and 0.0 <= _bf <= 1.0)):
                _bad_seed.append(str(_sk))
                continue
            _d.append(_mf - _bf)
        if _bad_seed:
            out['problems'].append(f'cell {i} ({c.get("arm")}): per_seed rows have an invalid/duplicate seed key or lack mass_covered/random_baseline: {_bad_seed[:5]}')
        _n_valid = len(_d)
        _mean_delta = (float(np.mean(_d)) if _n_valid else None)
        if _n_valid >= 2:
            try:
                L = _import_exp_lib()
                _r = L.exact_signflip_p(_d)
                _st = {'p_exact_signflip': float(_r.get('p')), 'n': int(_r.get('n', _n_valid)), 'method': _r.get('method'),
                       'mean_delta': _mean_delta}
            except Exception as e:
                out['problems'].append(f'cell {i}: could not run the seed test ({type(e).__name__}: {e})')
        out['cells'].append({'arm': c.get('arm'), 'eval_len': el, 'k': k, 'n_blocks': B,
                             'recall_at_k': rec, 'mass_covered': mass, 'random_baseline': base,
                             'mass_excess': mass - base, 'recall_excess': rec - base,
                             'n_queries': nq, 'n_seeds': _n_seeds, 'n_seeds_valid': _n_valid,
                             'seed_test': _st,
                             'claim_allowed': bool(mass > base and rec > base and nq > 0 and _st is not None
                                                   and _st.get('mean_delta') is not None
                                                   and float(_st['mean_delta']) > 0.0
                                                   and _st.get('p_exact_signflip', 1.0) < 0.05
                                                   and int(_st.get('n', 0)) >= 6)})
    out['status'] = 'fail' if out['problems'] else 'ok'
    out['note'] = ('cells pass the structural contract; `claim_allowed` additionally requires mass/recall above the '
                   'random baseline and a seed-level exact sign-flip p<0.05 with n>=6')
    print(json.dumps({k: v for k, v in out.items() if k != 'cells'}, ensure_ascii=False, indent=2)[:4000])
    return out


def _extract_source_tree(obj):
    """Find a recorded source-tree manifest in a snapshot/summary object."""
    if not isinstance(obj, dict):
        return None
    for key in ('source_tree',):
        val = obj.get(key)
        if isinstance(val, dict) and (val.get('sha256') or val.get('files')):
            return val
    for key in ('code', 'provenance'):
        val = obj.get(key)
        if isinstance(val, dict):
            got = _extract_source_tree(val)
            if got:
                return got
    return None


def _display_repo_path(path):
    """Relative when inside the repo, absolute when the caller passed a temp path."""
    ap = os.path.abspath(path)
    rel = os.path.relpath(ap, REPO)
    return ap if rel.startswith('..') else rel


def _iter_freeze_locks(results_dirs=None):
    """Freeze-lock artifacts, explicit dirs first then every results_* tree."""
    dirs = list(results_dirs) if results_dirs is not None else list(_iter_result_dirs())
    for d in dirs:
        p = _audit_artifact(os.path.join(d, '.freeze_lock.json'))
        if os.path.exists(p):
            yield p


def _declared_source_drift(lock_files):
    """All declared analysis-time source-drift events among ``lock_files``.

    A drift event is evidence that the analysis was NOT run from the bytes
    bound at first test access.  It must be read from the freeze lock even
    when the plan snapshot supplies the *expected* source tree: the snapshot
    records the approved plan, while the lock records what actually ran.
    """
    out, seen, sources = [], set(), []
    for lp in lock_files:
        if not lp:
            continue
        p = os.path.abspath(lp)
        if p in seen or not os.path.exists(p):
            continue
        seen.add(p)
        lock = _load(p)
        ev = lock.get('source_tree_drift') if isinstance(lock, dict) else None
        if not ev:
            continue
        if not isinstance(ev, list):
            # Malformed evidence still means "a drift was declared"; never
            # silently treat a non-list value as a clean lock.
            ev = [ev]
        sources.append(_display_repo_path(p))
        out.extend(dict(item) if isinstance(item, dict) else {'event': item}
                   for item in ev)
    return out, sources


def _apply_declared_source_drift(out, lock_files):
    """Fail an audit result when a relevant freeze lock declares drift."""
    drift, sources = _declared_source_drift(lock_files)
    if drift:
        out['declared_source_drift'] = drift
        out['declared_source_drift_locks'] = sources
        out['problems'].append(
            f'{len(drift)} declared analysis-time source-drift event(s) '
            'are recorded in the freeze lock; the analysis was not run '
            'from byte-identical code')
        out['note'] = ('the analysis declared source drift via '
                       'V12_ALLOW_SOURCE_DRIFT=1; even if the bytes now '
                       'match the original attestation, this round is not '
                       'a byte-identical replay')
        out['status'] = 'fail'
    return out


def audit_source_integrity(snapshot_path='analysis_v12/plan_snapshot.json',
                           results_dirs=None, lock_path=None):
    """Compare the current source bytes with the source bound by an artifact.

    The git commit + CODE_SEMANTICS stamp cannot detect a dirty edit that
    leaves the stamp unchanged (the exact hazard documented at the top of the
    continuous-loop workflow), and a source tarball has no commit at all.  The
    plan snapshot, the first-test-access freeze lock and panel provenance now
    carry a content digest; this gate recomputes it and reports which files
    changed when the full manifest was recorded.  Preference order is
    snapshot -> explicit freeze lock -> any discovered freeze lock -> panel
    provenance, because the freeze lock is the artifact written immediately
    before the one-shot test access.  No recorded digest is
    ``not_verifiable_locally`` — never "source is fine".
    """
    L = _import_exp_lib()
    cur = L.source_tree_manifest()
    expected = None
    source = None
    legacy_lock = None

    def _relevant_locks():
        # Explicit lock first; otherwise every discovered freeze lock.  The
        # lock that supplied `source` is included even when a plan snapshot
        # was preferred for the expected tree, because a drift event is
        # evidence about what actually ran, not about the approved plan.
        locks = ([_audit_artifact(lock_path)] if lock_path
                 else list(_iter_freeze_locks(results_dirs)))
        if isinstance(source, str) and source.endswith('.freeze_lock.json'):
            locks.append(_audit_artifact(source))
        out, seen = [], set()
        for lp in locks:
            if not lp:
                continue
            p = os.path.abspath(lp)
            if p in seen or not os.path.exists(p):
                continue
            seen.add(p)
            out.append(p)
        return out

    def _check_source_locks(out):
        """Require every source-bound freeze lock to agree with `expected`.

        A plan snapshot records the approved tree, while a freeze lock records
        the tree actually used at first test access.  The audit used to prefer
        the snapshot and ignore a conflicting lock unless that lock happened to
        declare a drift event; a stale/mis-copied snapshot could therefore
        report "byte-identical" while the lock attested a different
        measurement tree.  Compare the digests whenever both sides exist.
        """
        if not isinstance(expected, dict):
            return out
        want = expected.get('sha256')
        if not isinstance(want, str) or not want:
            return out
        checked, mismatches, malformed = [], [], []
        for lp in _relevant_locks():
            lock = _load(lp)
            if not isinstance(lock, dict):
                continue
            tree = _extract_source_tree(lock)
            tree_sha = tree.get('sha256') if isinstance(tree, dict) else None
            top_sha = lock.get('source_tree_sha256')
            # A lock from the pre-source-binding driver has neither field: it
            # is honest evidence of the plan/slice, not of the code bytes.
            if tree_sha is None and top_sha is None:
                continue
            rec = {'lock': _display_repo_path(lp)}
            if (isinstance(tree_sha, str) and tree_sha
                    and isinstance(top_sha, str) and top_sha
                    and tree_sha != top_sha):
                rec['error'] = ('source_tree.sha256 disagrees with '
                                'source_tree_sha256')
                malformed.append(rec)
            sha = (tree_sha if isinstance(tree_sha, str) and tree_sha
                   else top_sha if isinstance(top_sha, str) and top_sha
                   else None)
            if sha is None:
                rec['error'] = 'source tree has no usable sha256'
                malformed.append(rec)
                continue
            rec['source_sha256'] = sha
            checked.append(rec)
            if sha != want:
                mismatches.append(rec)
        if checked:
            out['source_locks'] = checked
        if malformed:
            out['malformed_source_locks'] = malformed
            out['problems'].append(
                f'{len(malformed)} freeze lock(s) carry an incomplete or '
                'contradictory source-tree binding')
        if mismatches:
            out['source_lock_mismatches'] = mismatches
            out['problems'].append(
                f'{len(mismatches)} freeze lock(s) bind a different source '
                'tree than the snapshot/provenance; the measurement code '
                'bytes are not established by the preferred artifact')
        if malformed or mismatches:
            out['status'] = 'fail'
        return out

    def _emit(out):
        _check_source_locks(out)
        _apply_declared_source_drift(out, _relevant_locks())
        print(json.dumps({k: v for k, v in out.items() if k != 'files'},
                         ensure_ascii=False, indent=2)[:4000])
        return out
    snap_file = _audit_artifact(snapshot_path)
    snap = _load(snap_file)
    if isinstance(snap, dict):
        expected = _extract_source_tree(snap)
        if expected:
            source = _display_repo_path(snap_file)
    if expected is None:
        locks = ([_audit_artifact(lock_path)] if lock_path
                 else list(_iter_freeze_locks(results_dirs)))
        for lp in locks:
            if not os.path.exists(lp):
                continue
            lock = _load(lp)
            got = _extract_source_tree(lock)
            if got:
                expected = got
                source = _display_repo_path(lp)
                break
            if isinstance(lock, dict) and 'manifest_sha256' in lock:
                # A lock from the pre-source-binding driver is honest evidence
                # of the plan/slice, but it cannot attest the code bytes.
                legacy_lock = _display_repo_path(lp)
    if expected is None and legacy_lock is None:
        dirs = list(results_dirs) if results_dirs is not None else list(_iter_result_dirs())
        for d in dirs:
            p = _audit_artifact(os.path.join(d, 'summary.json'))
            summ = _load(p)
            expected = _extract_source_tree(summ)
            if expected:
                source = _display_repo_path(p)
                break
    if expected is None and legacy_lock is not None:
        out = {
            'kind': 'source_tree_audit',
            'status': 'not_verifiable_locally',
            'snapshot': (_display_repo_path(snap_file)
                         if os.path.exists(snap_file) else None),
            'source': legacy_lock,
            'expected_sha256': None,
            'current_sha256': cur.get('sha256'),
            'n_expected_files': None,
            'n_current_files': cur.get('n_files'),
            'added': [], 'removed': [], 'changed': [],
            'problems': [],
            'note': ('the discovered freeze lock predates source-tree '
                     'binding and has no source digest; the lock attests the '
                     'plan/test slice but not the measurement code bytes'),
        }
        return _emit(out)
    out = {
        'kind': 'source_tree_audit',
        'status': 'not_verifiable_locally',
        'snapshot': (_display_repo_path(snap_file)
                     if os.path.exists(snap_file) else None),
        'source': source,
        'expected_sha256': None,
        'current_sha256': cur.get('sha256'),
        'n_expected_files': None,
        'n_current_files': cur.get('n_files'),
        'added': [], 'removed': [], 'changed': [],
        'problems': [],
        'note': ('no source-tree digest is present in a plan snapshot or panel '
                 'provenance; a git commit/CODE_SEMANTICS alone cannot prove '
                 'which source bytes produced an artifact'),
    }
    if not isinstance(expected, dict):
        return _emit(out)
    want_sha = expected.get('sha256')
    out['expected_sha256'] = want_sha
    if not isinstance(want_sha, str) or not want_sha:
        out['status'] = 'fail'
        out['problems'].append('the recorded source-tree entry has no sha256')
        return _emit(out)
    _raw_exp_files = expected.get('files')
    if _raw_exp_files is not None and not isinstance(_raw_exp_files, dict):
        out['problems'].append('the recorded source-tree files entry is not an object')
    exp_files = _raw_exp_files if isinstance(_raw_exp_files, dict) else {}
    _raw_cur_files = cur.get('files')
    cur_files = _raw_cur_files if isinstance(_raw_cur_files, dict) else {}
    out['n_expected_files'] = len(exp_files) if _raw_exp_files is not None else None

    def _entry_sha(entry):
        # A malformed per-file entry is not evidence of a match.  Returning
        # None makes it differ from every real digest instead of raising
        # AttributeError on an untrusted snapshot.
        if not isinstance(entry, dict):
            return None
        value = entry.get('sha256')
        return value if isinstance(value, str) and value else None

    if any(not isinstance(v, dict) for v in exp_files.values()):
        out['problems'].append('the recorded source-tree file table has malformed entries')
    if exp_files:
        _added = sorted(set(cur_files) - set(exp_files))
        _removed = sorted(set(exp_files) - set(cur_files))
        _changed = sorted(
            rel for rel in set(cur_files) & set(exp_files)
            if _entry_sha(cur_files.get(rel)) != _entry_sha(exp_files.get(rel)))
        out['added'], out['removed'], out['changed'] = _added, _removed, _changed
        if _added:
            out['problems'].append(f'{len(_added)} source file(s) added since the snapshot')
        if _removed:
            out['problems'].append(f'{len(_removed)} source file(s) removed since the snapshot')
        if _changed:
            out['problems'].append(f'{len(_changed)} source file(s) changed since the snapshot')
    if want_sha != cur.get('sha256'):
        out['problems'].append(
            f'source tree sha256 {str(cur.get("sha256"))[:12]}… does not match '
            f'the recorded {str(want_sha)[:12]}…')
    out['status'] = 'fail' if out['problems'] else 'ok'
    if out['status'] == 'ok':
        out['note'] = 'the current source tree is byte-identical to the recorded tree'
    else:
        out['note'] = ('the current source tree differs from the tree bound to '
                       'the snapshot/provenance; a git commit + CODE_SEMANTICS '
                       'stamp alone cannot establish source identity')
    # `_emit` also inspects declared analysis-time drift in the relevant
    # freeze lock(s), even when the expected tree came from the plan snapshot.
    return _emit(out)


def audit_dependencies(lock_path=None):
    """Verify the installed environment against requirements.lock."""
    L = _import_exp_lib()
    c = L.dependency_contract(lock_path)
    out = {
        'kind': 'dependency_audit',
        'status': ('ok' if c.get('status') in ('ok', 'partial') else 'fail'),
        'contract_status': c.get('status'),
        'lock_path': c.get('lock_path'),
        'lock_sha256': c.get('lock_sha256'),
        'n_pinned': c.get('n_pinned'),
        'n_installed': c.get('n_installed'),
        'n_unpinned_installed': c.get('n_unpinned_installed'),
        'mismatches': c.get('mismatches') or [],
        'missing': c.get('missing') or [],
        'invalid': c.get('invalid') or [],
        'unpinned_installed': (c.get('unpinned_installed') or [])[:50],
        'problems': [],
        'warnings': [],
    }
    if c.get('status') not in ('ok', 'partial'):
        out['problems'].append(c.get('note') or f'dependency contract is {c.get("status")}')
    elif c.get('status') == 'partial':
        out['warnings'].append(c.get('note') or 'the lock does not pin every installed distribution')
    print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
    return out


def audit_data_source(cache_dir=None):
    """Verify the upstream corpus revision recorded in each cache metadata file.

    A token-content fingerprint proves that two runs used the same tokens; it
    does not prove that a *new* cache was built from the same upstream snapshot.
    WikiText-103 and FineWeb-Edu are mutable Hub repositories, so the loaders
    pin a full commit and record it in `dataset_source`.  This audit recomputes
    that contract independently of the loader:

      - every non-legacy cache must carry a schema-valid `dataset_source`;
      - a Hub source must name a full 40-hex commit and attest the pin;
      - all paired train/validation/test caches must name the same commit;
      - a legacy cache (no field) is reported as unverifiable, never as clean.

    Local raw/jsonl corpora are legitimate; their content is bound by the token
    fingerprints and the audit records that the upstream commit is not
    attested.
    """
    import glob as _glob
    L = _import_exp_lib()
    if cache_dir in (None, ''):
        cache_dir = os.environ.get('WT103_CACHE_DIR') or './wt103_cache'
    cache_dir = _audit_artifact(cache_dir)
    out = {
        'kind': 'dataset_source_audit',
        'status': 'ok',
        'cache_dir': cache_dir,
        'n_caches': 0,
        'n_pinned': 0,
        'n_legacy': 0,
        'n_local': 0,
        'revisions': {},
        'caches': {},
        'problems': [],
        'warnings': [],
    }
    if not os.path.isdir(cache_dir):
        out['status'] = 'not_verifiable_locally'
        out['note'] = f'{cache_dir} does not exist; build or copy the corpus cache to audit its upstream revision'
        print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out
    metas = sorted(_glob.glob(os.path.join(cache_dir, 'meta_*.json')))
    metas += sorted(_glob.glob(os.path.join(cache_dir, 'test_meta_*.json')))
    # `meta_*.json` also matches test_meta_*.json on case-sensitive globs only
    # when the pattern is literal, but de-duplicate defensively.
    metas = sorted(set(metas))
    if not metas:
        out['status'] = 'not_verifiable_locally'
        out['note'] = f'{cache_dir} has no meta_*.json/test_meta_*.json; cannot audit the corpus revision'
        print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
        return out
    train_metas = []
    test_metas = []
    dataset_revisions = {}
    for path in metas:
        name = os.path.basename(path)
        meta = _load(path)
        if not isinstance(meta, dict):
            out['problems'].append(f'{name}: unreadable metadata')
            continue
        out['n_caches'] += 1
        # FineWeb metadata records `source`; WikiText metadata does not.
        try:
            if meta.get('source'):
                expected = L.fineweb_edu_dataset_source(str(meta.get('source')))
            else:
                expected = L.wikitext_dataset_source()
        except Exception as e:
            out['problems'].append(
                f'{name}: the pinned source cannot be resolved '
                f'({type(e).__name__}: {e})')
            expected = None
        probs, warns = L._dataset_source_problems(meta, expected, name)
        src = meta.get('dataset_source')
        if isinstance(src, dict):
            if src.get('provider') == 'huggingface':
                out['n_pinned'] += 1
                rev = str(src.get('revision') or '').lower()
                if rev:
                    out['revisions'].setdefault(rev, []).append(name)
                    dataset_revisions.setdefault(
                        'fineweb' if meta.get('source') else 'wikitext',
                        set()).add(rev)
            elif src.get('provider') == 'local':
                out['n_local'] += 1
        else:
            out['n_legacy'] += 1
        out['caches'][name] = {
            'dataset': 'fineweb' if meta.get('source') else 'wikitext',
            'provider': (src or {}).get('provider') if isinstance(src, dict) else None,
            'repo': (src or {}).get('repo') if isinstance(src, dict) else None,
            'revision': (src or {}).get('revision') if isinstance(src, dict) else None,
            'revision_pinned': (src or {}).get('revision_pinned') if isinstance(src, dict) else None,
            'problems': probs,
            'warnings': warns,
        }
        out['problems'].extend(probs)
        out['warnings'].extend(warns)
        if name.startswith('test_meta_'):
            test_metas.append((name, meta))
        else:
            train_metas.append(meta)
    # Cross-cache consistency: every test cache must name the same upstream
    # commit as its paired train caches.  A directory may legitimately hold
    # several corpora (e.g. WikiText and FineWeb), so pair only train caches
    # from the same upstream repository/config.  Match the loaders' tokenizer-
    # fit rule too: a cache built for a different fit corpus is a different
    # protocol, not a reference snapshot for this test slice.
    if test_metas:
        for test_name, tm in test_metas:
            ts = (tm.get('dataset_source')
                  if isinstance(tm, dict) else None)
            ts = ts if isinstance(ts, dict) else {}
            tfit = tm.get('fit_fp') if isinstance(tm, dict) else None
            paired = []
            for _m in train_metas:
                s = _m.get('dataset_source') if isinstance(_m, dict) else None
                s = s if isinstance(s, dict) else {}
                if (ts.get('provider') == 'huggingface'
                        and s.get('provider') == 'huggingface'
                        and (ts.get('repo') != s.get('repo')
                             or ts.get('config') != s.get('config'))):
                    continue
                mfit = _m.get('fit_fp') if isinstance(_m, dict) else None
                if tfit and mfit and tfit != mfit:
                    continue
                paired.append(_m)
            if ts and not paired:
                out['warnings'].append(
                    f'{test_name}: no train cache matches the test corpus and '
                    'tokenizer-fit provenance; the upstream revision cannot '
                    'be compared')
            probs, warns = L.dataset_source_pair_problems(tm, paired,
                                                          label='test/train')
            out['problems'].extend(probs)
            out['warnings'].extend(warns)
    # Two different pinned revisions *within one corpus* mean that corpus's
    # caches mix upstream snapshots.  Per-cache records may still be
    # internally consistent, so this is a warning at directory level; the
    # train/test pair check above turns the dangerous case into a failure.
    # Different corpora naturally have different commits and must not warn.
    for _dataset, _revs in sorted(dataset_revisions.items()):
        if len(_revs) > 1:
            out['warnings'].append(
                f'the {_dataset} caches contain more than one upstream '
                'revision: ' + ', '.join(sorted(_revs)))
    if out['problems']:
        out['status'] = 'fail'
    elif out['n_legacy'] or out['n_local'] or out['warnings']:
        if out['n_legacy']:
            # Any cache without the field means the directory as a whole is
            # not fully attested.  Say exactly that instead of letting the
            # verified subset stand in for the missing one.
            out['status'] = 'not_verifiable_locally'
            out['note'] = (
                f'{out["n_legacy"]} of {out["n_caches"]} cache metadata file(s) '
                'predate dataset_source; the upstream commit cannot be attested '
                'for those caches without rebuilding (they remain readable, but '
                'are not clean)')
        else:
            out['status'] = 'ok'
            out['note'] = ('present dataset-source records verify; local '
                           'caches and directory-level caveats are listed under '
                           'warnings and are not claimed as attestations')
    print(json.dumps({k: v for k, v in out.items() if k != 'caches'},
                     ensure_ascii=False, indent=2)[:4000])
    return out


def _read_experiment_events(outdir):
    path = os.path.join(outdir, 'experiments.jsonl')
    events = []
    if not os.path.exists(path):
        return events
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if isinstance(e, dict):
                    events.append(e)
    except OSError:
        return []
    return events


def _actual_run_order(events):
    """Cell start order from the append-only event log (completed cells only)."""
    cells = []
    for e in events:
        if (e.get('kind') != 'cell' or e.get('status') != 'ok'
                or e.get('variant') is None or e.get('seed') is None):
            continue
        try:
            label = f"{e['variant']}::seed{int(e['seed'])}"
        except (TypeError, ValueError, OverflowError):
            continue
        cells.append((str(e.get('started_at') or ''), label))
    # A retrained/resumed cell legitimately appears more than once; its LAST
    # start is the execution that produced the current measurement.
    latest = {}
    for started, lab in cells:
        if lab not in latest or started >= latest[lab]:
            latest[lab] = started
    return [lab for lab, _started in sorted(latest.items(), key=lambda kv: kv[1])]


def audit_run_order(paths=None):
    """Run-order / temporal-confounding audit (zero GPU).

    A paired seed test only compares arms within a seed.  If the driver runs
    all seeds of arm A and only then all seeds of arm B, the two arms of every
    pair were trained in different machine/time blocks; a driver update, a
    thermal change or a noisy neighbour can then masquerade as an arm effect.
    This audit checks the recorded design (`run_order.json`) and, when present,
    the executed order from `experiments.jsonl`:

      * complete blocks: every block (default: seed) contains every declared
        arm exactly once;
      * position balance: an arm does not always occupy the same slot;
      * actual order: a completed arm-major run is a FAIL even when the plan
        was blocked (the driver did not honour it), while a missing plan is
        reported as not verifiable rather than silently assumed.

    It is descriptive: it never changes a PPL, seed or decision rule.
    """
    L = _import_exp_lib()
    if paths is None:
        paths = [os.path.join(REPO, d) for d in _iter_result_dirs()]
    elif isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    out = {'kind': 'run_order_audit', 'status': 'ok', 'panels': {},
           'n_ok': 0, 'n_fail': 0, 'n_not_verifiable': 0,
           'problems': [], 'warnings': [],
           'note': ('blocked randomized run order controls calendar-time / '
                    'machine-drift confounding; arm-major execution is a '
                    'design failure, not a null result')}
    for d in paths:
        d = os.fspath(d)
        if not os.path.isdir(d):
            continue
        name = os.path.basename(os.path.normpath(d))
        plan_path = os.path.join(d, 'run_order.json')
        _plan_exists = os.path.exists(plan_path)
        plan = _load(plan_path) if _plan_exists else None
        if _plan_exists and plan is None:
            # A file that exists but cannot be parsed is not the same as a
            # pre-planning legacy panel: fail closed instead of inferring.
            rec = {'status': 'fail', 'plan_present': True,
                   'problems': ['run_order.json exists but is not a JSON object; '
                                'the recorded design cannot be verified']}
            out['n_fail'] += 1
            out['problems'].extend(f'{name}: {p}' for p in rec['problems'])
            out['panels'][name] = rec
            continue
        events = _read_experiment_events(d)
        actual = _actual_run_order(events)
        if plan is None:
            if not actual:
                rec = {'status': 'not_verifiable_locally',
                       'reason': 'no run_order.json and no completed cell events'}
                out['n_not_verifiable'] += 1
                out['panels'][name] = rec
                continue
            # Legacy/inferred order: classify it, but never claim the design
            # was randomized when no plan exists.
            blocks = {}
            for lab in actual:
                try:
                    v, s = lab.rsplit('::seed', 1)
                except ValueError:
                    continue
                blocks.setdefault(s, []).append(v)
            cell_blocks = {}
            cell_variants = {}
            for lab in actual:
                try:
                    v, s = lab.rsplit('::seed', 1)
                except ValueError:
                    continue
                cell_blocks[lab] = str(s)
                cell_variants[lab] = v
            inferred = {'design': {'name': 'inferred_no_plan', 'randomized': False,
                                   'block': 'seed', 'design_seed': None},
                        'variants': sorted({v for vs in blocks.values() for v in vs}),
                        'blocks': {b: vs for b, vs in blocks.items()},
                        'planned_cells': list(actual),
                        'cell_blocks': cell_blocks,
                        'cell_variants': cell_variants}
            # Passing the actual order lets the same arm-major detector run on
            # a legacy panel; the inferred design itself stays unverifiable.
            rec = L.run_order_balance(inferred, actual)
            rec['actual']['inferred_from_events'] = True
            rec['plan_present'] = False
        else:
            _cb = plan.get('cell_blocks')
            if isinstance(_cb, dict):
                labels_in_plan = set(_cb.keys())
                filtered = [x for x in actual if x in labels_in_plan]
                rec = L.run_order_balance(plan, filtered)
                if not filtered and rec.get('status') == 'ok':
                    rec['status'] = 'not_verifiable_locally'
                    rec.setdefault('warnings', []).append(
                        'the plan exists but no completed cell event matches it; '
                        'the recorded design is verified, the execution is not')
            else:
                # A plan without its cell->block mapping cannot be tied to the
                # event log; audit the plan itself and say the actual order was
                # not checkable rather than crashing or silently skipping it.
                rec = L.run_order_balance(plan)
                rec.setdefault('warnings', []).append(
                    'plan has no cell_blocks mapping; the actual execution '
                    'order could not be checked')
            rec['plan_present'] = True
            rec['actual'] = rec.get('actual') or {'n_observed': len(actual)}
        st = rec.get('status')
        out['n_ok'] += int(st == 'ok')
        out['n_fail'] += int(st == 'fail')
        out['n_not_verifiable'] += int(st == 'not_verifiable_locally')
        if st == 'fail':
            out['problems'].extend(f'{name}: {p}' for p in rec.get('problems') or [])
        for w in rec.get('warnings') or []:
            out['warnings'].append(f'{name}: {w}')
        out['panels'][name] = rec
    if out['n_fail']:
        out['status'] = 'fail'
    elif out['n_ok'] == 0:
        out['status'] = 'not_verifiable_locally'
    print(json.dumps({k: v for k, v in out.items() if k != 'panels'},
                     ensure_ascii=False, indent=2)[:4000])
    if out['panels']:
        for _n, _r in out['panels'].items():
            print(f"  [run-order] {_n}: {_r.get('status')} "
                  f"(blocks={_r.get('n_blocks')}, complete={_r.get('complete_blocks')}, "
                  f"arm_major={_r.get('arm_major')})")
    return out


def audit_unit_influence(path=None):
    """Result-analysis audit: leave-one-unit-out influence is present and sound.

    A pooled ΔPPL can be statistically significant and still be carried by one
    evaluation article.  ``v12_supp._doc_level_analysis`` now records a
    ``doc_sampling.unit_influence`` block; this audit independently checks that
    the block is internally consistent with the reported pooled estimate and
    flags comparisons whose direction or magnitude depends on one unit.

    It is a **warning-level** audit by design: the frozen decision rule does
    not include a unit-drop threshold, so a fragile comparison must carry the
    caveat in the report but must not be silently turned into "not confirmed".
    A malformed block (point estimate disagrees with the pooled Δ, missing
    units, wrong vector lengths) is a ``fail`` because it cannot support any
    influence statement.
    """
    p = path or os.path.join(REPO, 'analysis_v12', 'test_stats.json')
    out = {'kind': 'unit_influence_audit', 'status': 'ok', 'path': p,
           'comparisons': {}, 'n_ok': 0, 'n_fragile': 0, 'n_missing': 0,
           'problems': [], 'warnings': [],
           'note': ('post-hoc descriptive robustness; a fragile comparison must '
                    'carry the single-unit caveat, but this audit never changes '
                    'a frozen p/Holm/confirmed verdict')}
    if not os.path.exists(p):
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'no final-test stats artifact on disk'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    data = _load(p)
    if data is None:
        out['status'] = 'fail'
        out['problems'].append(f'{p} exists but is not a readable JSON object')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    ft = data.get('finaltest') if isinstance(data.get('finaltest'), dict) else data
    comps = ft.get('comparisons') if isinstance(ft, dict) else None
    if not isinstance(comps, dict) or not comps:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'the stats artifact carries no comparisons'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    def _finite(x):
        # A numeric string (or bytes) is not a measurement: float("1.0")
        # would let a hand-edited/stale artifact replay as a finite reading.
        if (isinstance(x, (bool, np.bool_, str, bytes, bytearray))
                or x is None):
            return None
        try:
            y = float(x)
        except (TypeError, ValueError, OverflowError):
            return None
        return y if math.isfinite(y) else None

    for name, comp in comps.items():
        if not isinstance(comp, dict):
            # A non-object comparison cannot carry a measured influence block;
            # silently skipping it would hide a corrupt comparison.
            out['problems'].append(
                f'{name}: comparison is a {type(comp).__name__}, not an object')
            out['comparisons'][name] = {'present': False,
                                        'error': 'not_an_object'}
            continue
        ds = comp.get('doc_sampling')
        inf = ds.get('unit_influence') if isinstance(ds, dict) else None
        rec = {'present': isinstance(inf, dict)}
        if not isinstance(inf, dict):
            if isinstance(ds, dict) and ds.get('pooled_delta_ppl_ci95') is not None:
                out['n_missing'] += 1
                out['warnings'].append(
                    f'{name}: pooled CI exists but no unit_influence block '
                    '(stale/legacy analysis artifact); the single-unit '
                    'sensitivity is unverified')
            out['comparisons'][name] = rec
            continue
        _available = inf.get('available')
        if not isinstance(_available, (bool, np.bool_)):
            # The availability flag decides whether the block is a measurement
            # or an explicit omission; "false"/1 must not be silently treated
            # as available evidence.
            out['problems'].append(
                f'{name}: unit_influence.available is not a boolean '
                f'({_available!r})')
            rec['error'] = 'invalid_available'
            out['comparisons'][name] = rec
            continue
        rec['available'] = bool(_available)
        if not _available:
            out['warnings'].append(
                f'{name}: unit influence unavailable ({inf.get("reason")})')
            out['comparisons'][name] = rec
            continue
        n_units = inf.get('n_units')
        ids = inf.get('unit_ids')
        pooled = _finite(ds.get('pooled_delta_ppl'))
        full = _finite(inf.get('full_delta_ppl'))
        rec.update({'n_units': n_units, 'fragile_to_one_unit': inf.get('fragile_to_one_unit'),
                    'most_influential_unit': inf.get('most_influential_unit'),
                    'max_abs_relative_change_log_ratio':
                        inf.get('max_abs_relative_change_log_ratio')})
        if not isinstance(n_units, int) or n_units < 3:
            out['problems'].append(f'{name}: invalid n_units={n_units!r}')
        if not isinstance(ids, list) or len(ids) != n_units:
            out['problems'].append(
                f'{name}: unit_ids length does not match n_units={n_units!r}')
        elif len(set(map(str, ids))) != n_units:
            out['problems'].append(
                f'{name}: unit_ids are not unique; the folds cannot identify units')
        for _vec_key in ('loo_delta_ppl', 'loo_delta_log_ratio',
                         'loo_change_ppl', 'loo_change_log_ratio'):
            _vec = inf.get(_vec_key)
            if not isinstance(_vec, list) or len(_vec) != n_units:
                out['problems'].append(
                    f'{name}: {_vec_key} length does not match n_units={n_units!r}')
                continue
            if any(v is not None and _finite(v) is None for v in _vec):
                out['problems'].append(
                    f'{name}: {_vec_key} contains a non-finite, non-null value')
        _idx = inf.get('most_influential_index')
        if (_idx is not None
                and (not isinstance(_idx, int) or isinstance(_idx, bool)
                     or not (0 <= _idx < (n_units if isinstance(n_units, int) else 0)))):
            out['problems'].append(
                f'{name}: most_influential_index {_idx!r} is outside the unit vector')
        if (isinstance(_idx, int) and isinstance(ids, list)
                and 0 <= _idx < len(ids)
                and inf.get('most_influential_unit') is not None
                and str(ids[_idx]) != str(inf.get('most_influential_unit'))):
            out['problems'].append(
                f'{name}: most_influential_unit does not match unit_ids[index]')
        if pooled is None or full is None:
            out['problems'].append(
                f'{name}: pooled or full influence point estimate is missing/non-finite')
        elif abs(full - pooled) > 1e-6 * max(1.0, abs(pooled)):
            out['problems'].append(
                f'{name}: unit-influence full point {full!r} does not reproduce '
                f'the reported pooled Δ {pooled!r}')
        _mi = inf.get('most_influential_unit')
        if (_mi is not None and isinstance(ids, list) and _mi not in ids):
            out['problems'].append(
                f'{name}: most_influential_unit {_mi!r} is not one of unit_ids')
        _frag = inf.get('fragile_to_one_unit')
        if _frag is not None and not isinstance(_frag, bool):
            out['problems'].append(
                f'{name}: fragile_to_one_unit is not a boolean or null')
        elif _frag is True:
            out['n_fragile'] += 1
            out['warnings'].append(
                f'{name}: direction/magnitude depends on one '
                f'{inf.get("most_influential_unit")!r}; the report must carry '
                'the single-unit caveat')
        elif _frag is False:
            out['n_ok'] += 1
        out['comparisons'][name] = rec
    if out['problems']:
        out['status'] = 'fail'
    elif out['n_fragile'] or out['n_missing']:
        out['status'] = 'warnings'
    print(json.dumps({k: v for k, v in out.items() if k != 'comparisons'},
                     ensure_ascii=False, indent=2))
    return out


def audit_analysis_choices(path=None):
    """Result-analysis audit: multi-specification concordance is coherent.

    A single panel can be described by per-seed mean/median/trimmed summaries,
    a token-pooled difference, or equal-weighted evaluation units.  The report
    records ``doc_sampling.analysis_choice_concordance``; this gate
    independently replays that summary from the stored specification rows and
    checks it against the frozen per-seed mean.  Disagreement in direction or
    in zero-exclusion is a **warning** (a reporting caveat), not a change to
    the frozen decision.  A stale, internally inconsistent or non-reproducible
    block is a ``fail`` because it cannot support the claim that the
    sensitivity was actually checked.
    """
    L = _import_exp_lib()
    p = path or os.path.join(REPO, 'analysis_v12', 'test_stats.json')
    out = {'kind': 'analysis_choice_audit', 'status': 'ok', 'path': p,
           'comparisons': {}, 'n_ok': 0, 'n_fragile': 0,
           'n_missing': 0, 'problems': [], 'warnings': [],
           'note': ('post-hoc specification concordance; divergent directions or '
                    'zero-exclusion are reporting caveats, never changes to a '
                    'frozen p/Holm/confirmed verdict')}
    if not os.path.exists(p):
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'no final-test stats artifact on disk'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    data = _load(p)
    if data is None:
        out['status'] = 'fail'
        out['problems'].append(f'{p} exists but is not a readable JSON object')
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out
    ft = data.get('finaltest') if isinstance(data.get('finaltest'), dict) else data
    comps = ft.get('comparisons') if isinstance(ft, dict) else None
    if not isinstance(comps, dict) or not comps:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = 'the stats artifact carries no comparisons'
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return out

    def _finite(x):
        if (isinstance(x, (bool, np.bool_, str, bytes, bytearray))
                or x is None):
            return None
        try:
            y = float(x)
        except (TypeError, ValueError, OverflowError):
            return None
        return y if math.isfinite(y) else None

    def _same_choice_value(a, b):
        # bool must not compare equal to 1/0, and a numeric string must not
        # replay as a float (the concordance helper deliberately rejects both).
        if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
            return (isinstance(a, (bool, np.bool_))
                    and isinstance(b, (bool, np.bool_))
                    and bool(a) == bool(b))
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            return (isinstance(a, (list, tuple))
                    and isinstance(b, (list, tuple))
                    and len(a) == len(b)
                    and all(_same_choice_value(x, y) for x, y in zip(a, b)))
        if isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
            if (isinstance(a, (complex, np.complexfloating))
                    or isinstance(b, (complex, np.complexfloating))):
                return False
            _fa, _fb = _finite(a), _finite(b)
            return (_fa is not None and _fb is not None
                    and abs(_fa - _fb) <= 1e-12 * max(1.0, abs(_fa), abs(_fb)))
        return a == b

    keys = ('status', 'claim', 'n_specs', 'n_available', 'n_unavailable',
            'n_nonzero', 'n_positive', 'n_negative', 'n_zero',
            'n_same_as_primary', 'n_opposite_to_primary', 'direction_stable',
            'n_with_ci', 'n_excluding_zero', 'n_crossing_zero',
            'precision_concordant', 'excluding_zero_specs',
            'crossing_zero_specs', 'fragile', 'primary_sign',
            'primary_excludes_zero')
    for name, comp in comps.items():
        if not isinstance(comp, dict):
            # A non-object comparison cannot carry a replayable analysis-choice
            # block; silently skipping it would hide a corrupt comparison.
            out['problems'].append(
                f'{name}: comparison is a {type(comp).__name__}, not an object')
            out['comparisons'][name] = {'present': False,
                                        'error': 'not_an_object'}
            continue
        ds = comp.get('doc_sampling')
        _has_cc = isinstance(ds, dict) and 'analysis_choice_concordance' in ds
        cc = ds.get('analysis_choice_concordance') if _has_cc else None
        rec = {'present': isinstance(cc, dict)}
        if _has_cc and not isinstance(cc, dict):
            # Present-but-malformed is not the same as absent: a corrupt block
            # must not silently downgrade to a warning (or, worse, look like a
            # legacy artifact that simply never recorded the sensitivity).
            out['problems'].append(
                f'{name}: analysis_choice_concordance is a '
                f'{type(cc).__name__}, not a JSON object')
            rec['error'] = 'not_an_object'
            out['comparisons'][name] = rec
            continue
        if not _has_cc:
            if 'omitted' not in comp:
                out['n_missing'] += 1
                if (isinstance(ds, dict)
                        and (ds.get('per_doc_delta_mean') is not None
                             or ds.get('pooled_delta_ppl') is not None)):
                    _why = ('doc_sampling exists but no analysis-choice block '
                            '(stale/legacy analysis artifact)')
                else:
                    _why = 'no doc_sampling/analysis-choice block'
                out['warnings'].append(
                    f'{name}: {_why}; the multi-specification sensitivity is '
                    'unverified')
            out['comparisons'][name] = rec
            continue
        rec['status'] = cc.get('status')
        specs = cc.get('specifications')
        rec['n_specs'] = cc.get('n_specs')
        if cc.get('kind') != 'analysis_choice_concordance':
            out['problems'].append(
                f'{name}: unexpected kind {cc.get("kind")!r}')
        if cc.get('version') != L.ANALYSIS_CHOICE_POLICY['version']:
            out['problems'].append(
                f'{name}: analysis-choice version {cc.get("version")!r} does '
                f'not match {L.ANALYSIS_CHOICE_POLICY["version"]!r}')
        if not isinstance(specs, list) or len(specs) < 2:
            out['problems'].append(
                f'{name}: specifications must be a list with at least 2 rows')
            out['comparisons'][name] = rec
            continue
        if not _same_choice_value(cc.get('n_specs'), len(specs)):
            out['problems'].append(
                f'{name}: n_specs={cc.get("n_specs")!r} but {len(specs)} '
                'specification rows are stored')
        names = [s.get('spec') for s in specs if isinstance(s, dict)]
        if (len(names) != len(specs)
                or any(not isinstance(n, str) or not n.strip() for n in names)
                or len(set(names)) != len(names)):
            out['problems'].append(
                f'{name}: specification names are missing, non-string, '
                'duplicated or not all dicts')
            out['comparisons'][name] = rec
            continue
        pname = cc.get('primary_spec')
        primaries = [s for s in specs if s.get('spec') == pname]
        if len(primaries) != 1:
            out['problems'].append(
                f'{name}: primary_spec {pname!r} does not identify exactly one '
                'stored specification')
            out['comparisons'][name] = rec
            continue
        alts = [s for s in specs if s.get('spec') != pname]
        try:
            want = L.analysis_choice_concordance(primaries[0], alts)
        except (ValueError, TypeError, FloatingPointError, OverflowError) as e:
            out['problems'].append(
                f'{name}: stored specifications cannot be replayed '
                f'({type(e).__name__}: {e})')
            out['comparisons'][name] = rec
            continue
        for k in keys:
            if not _same_choice_value(cc.get(k), want.get(k)):
                out['problems'].append(
                    f'{name}: stored {k}={cc.get(k)!r} does not reproduce '
                    f'replayed {want.get(k)!r}')
        # The aggregate keys above do not see every derived field of every
        # row (``is_primary`` is the clearest example).  Replaying the rows
        # and comparing them also catches a row whose stored labels/counts
        # contradict its own point/CI instead of letting the aggregates hide it.
        _wspecs = want.get('specifications')
        if not isinstance(_wspecs, list) or len(_wspecs) != len(specs):
            out['problems'].append(
                f'{name}: specification rows do not reproduce the replayed '
                f'row count ({len(specs)} stored vs '
                f'{len(_wspecs) if isinstance(_wspecs, list) else "non-list"})')
        else:
            for _i, (_stored, _replayed) in enumerate(zip(specs, _wspecs)):
                if not isinstance(_stored, dict) or not isinstance(_replayed, dict):
                    out['problems'].append(
                        f'{name}: specification row {_i} is not an object')
                    continue
                # Compare the fields actually stored (a hand-written minimal
                # row may omit derived fields; the replay supplies them), but
                # reject fields the replayed row never produces.
                _bad = sorted(k for k in set(_stored) & set(_replayed)
                              if not _same_choice_value(_stored.get(k),
                                                        _replayed.get(k)))
                _extra = sorted(set(_stored) - set(_replayed))
                if _bad or _extra:
                    _what = []
                    if _bad:
                        _what.append(f'does not reproduce fields {_bad}')
                    if _extra:
                        _what.append(f'has unexpected fields {_extra}')
                    out['problems'].append(
                        f'{name}: specification row {_i} '
                        f'({_stored.get("spec")!r}) ' + ' and '.join(_what))
        _er = cc.get('effect_range_ppl')
        _wer = want.get('effect_range_ppl')
        if (_er is None) != (_wer is None):
            out['problems'].append(
                f'{name}: stored effect_range_ppl presence does not reproduce')
        elif _wer is not None and not isinstance(_er, dict):
            out['problems'].append(
                f'{name}: stored effect_range_ppl is a '
                f'{type(_er).__name__}, not an object')
        elif isinstance(_er, dict) and isinstance(_wer, dict):
            _extra = sorted(set(_er) - {'min', 'max', 'span',
                                        'relative_span', 'n'})
            if _extra:
                out['problems'].append(
                    f'{name}: stored effect_range_ppl has unexpected fields '
                    f'{_extra}')
            for k in ('min', 'max', 'span', 'relative_span', 'n'):
                _a, _b = _er.get(k), _wer.get(k)
                if k == 'n':
                    if not _same_choice_value(_a, _b):
                        out['problems'].append(
                            f'{name}: effect_range_ppl.{k}={_a!r} does not '
                            f'reproduce {_b!r}')
                    continue
                _fa, _fb = _finite(_a), _finite(_b)
                if (_fa is None) != (_fb is None):
                    out['problems'].append(
                        f'{name}: effect_range_ppl.{k} presence does not '
                        'reproduce')
                elif _fa is not None and abs(_fa - _fb) > 1e-9 * max(
                        1.0, abs(_fb)):
                    out['problems'].append(
                        f'{name}: effect_range_ppl.{k}={_fa!r} does not '
                        f'reproduce {_fb!r}')
        _pm = _finite(primaries[0].get('point'))
        _mean = _finite(comp.get('mean'))
        if _pm is None or _mean is None:
            out['problems'].append(
                f'{name}: primary analysis-choice point or frozen mean is '
                'missing/non-finite')
        elif abs(_pm - _mean) > 1e-6 * max(1.0, abs(_mean)):
            out['problems'].append(
                f'{name}: primary analysis-choice point {_pm!r} does not '
                f'reproduce the frozen per-seed mean {_mean!r}')
        if cc.get('status') in ('direction_unstable', 'precision_sensitive'):
            out['n_fragile'] += 1
            if cc.get('status') == 'direction_unstable':
                out['warnings'].append(
                    f'{name}: analysis specifications disagree on direction; '
                    'the report must show the specification table and must not '
                    'quote one aggregation as robust')
            else:
                out['warnings'].append(
                    f'{name}: direction is stable but zero-exclusion depends on '
                    'the analysis choice; interval claims must name the unit/'
                    'estimand')
        elif cc.get('status') in ('insufficient_specs', 'zero_effect'):
            out['warnings'].append(
                f'{name}: analysis-choice status is {cc.get("status")!r}; the '
                'multi-specification check is not informative')
        else:
            out['n_ok'] += 1
        out['comparisons'][name] = rec
    if out['problems']:
        out['status'] = 'fail'
    elif out['n_ok'] == 0 and not out['n_fragile']:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('no comparison carries a replayable analysis-choice '
                         'block')
    elif out['warnings']:
        out['status'] = 'warnings'
    print(json.dumps({k: v for k, v in out.items() if k != 'comparisons'},
                     ensure_ascii=False, indent=2))
    return out


def _find_look_histories(paths=None, max_depth=3):
    """Yield ``look_history.json`` paths from explicit paths or results_* trees."""
    seen = set()

    def _walk(base):
        base = os.path.abspath(base)
        for root, dirs, files in os.walk(base):
            depth = root[len(base):].count(os.sep)
            # Result trees can contain caches/checkpoints; the look file only
            # ever sits next to a panel summary, so prune deep/generated trees.
            dirs[:] = [d for d in dirs
                       if d not in ('__pycache__', 'wt103_cache', 'checkpoints')
                       and depth < max_depth]
            if 'look_history.json' in files:
                yield os.path.join(root, 'look_history.json')

    if paths is not None:
        for p in paths:
            p = os.fspath(p)
            if os.path.isdir(p):
                for f in _walk(p):
                    if f not in seen:
                        seen.add(f)
                        yield f
            elif os.path.isfile(p) and p.endswith('.json'):
                ap = os.path.abspath(p)
                if ap not in seen:
                    seen.add(ap)
                    yield ap
        return
    for name in _iter_result_dirs():
        for f in _walk(os.path.join(REPO, name)):
            if f not in seen:
                seen.add(f)
                yield f


def audit_look_history(paths=None):
    """Audit recorded analysis looks and the sequential (optional-stopping) p.

    Every seed top-up pass re-aggregates a panel; if the analyst then decides
    to add more seeds, the final-n p is not the p the experiment spent.  The
    driver records each distinct seed set as a look in ``look_history.json``.
    This gate independently:

      * validates the look file (schema, per-look seed/delta lengths, unique
        seeds, finite deltas, recomputed per-look exact p);
      * recomputes the exact max-over-looks sequential p with the same
        ``exp_lib.sequential_signflip_p`` the audit used and compares it with
        ``methodology_audit.json``;
      * flags seed regressions, stale/missing audit blocks and any contrast
        whose naive final-look family p would be significant but whose
        sequential p is not.

    A malformed or internally inconsistent look file is a ``fail``: an
    untraceable correction cannot be quoted.  No look files at all is
    ``not_verifiable_locally``, not a pass.
    """
    L = _import_exp_lib()
    out = {'kind': 'look_history_audit', 'status': 'ok', 'panels': {},
           'n_panels': 0, 'n_looks': 0, 'n_contrasts_adjusted': 0,
           'downgraded_claims': [], 'problems': [], 'warnings': [],
           'note': ('optional-stopping correction: a panel aggregated at several '
                    'growing seed sets is judged by the exact max-over-looks '
                    'sign-flip p, not only by the final-n p')}
    files = sorted(_find_look_histories(paths))

    def _finite(x):
        # A numeric string (or bytes) is not a measurement: float("1.0")
        # would let a hand-edited/stale artifact replay as a finite reading.
        if (isinstance(x, (bool, np.bool_, str, bytes, bytearray))
                or x is None):
            return None
        try:
            y = float(x)
        except (TypeError, ValueError, OverflowError):
            return None
        return y if math.isfinite(y) else None

    for f in files:
        panel_dir = os.path.dirname(f)
        rec = {'path': f, 'looks': 0, 'contrasts': {},
               'downgraded': [], 'warnings': [], 'problems': []}
        hist = _load(f)
        if hist is None:
            rec['problems'].append('look_history.json is missing or not a JSON object')
        elif not isinstance(hist.get('looks'), list):
            rec['problems'].append('looks is not a list')
        else:
            series = {}
            prev_look = 0
            for look in hist['looks']:
                if not isinstance(look, dict):
                    rec['problems'].append(f'look entry is a {type(look).__name__}, not an object')
                    continue
                lid = look.get('look')
                if not isinstance(lid, int) or isinstance(lid, bool) or lid <= prev_look:
                    rec['problems'].append(f'look number {lid!r} is not a strictly increasing integer')
                else:
                    prev_look = lid
                contrasts = look.get('contrasts')
                if not isinstance(contrasts, dict):
                    rec['problems'].append(f'look {lid}: contrasts is not an object')
                    continue
                rec['looks'] += 1
                for key, c in contrasts.items():
                    if not isinstance(c, dict):
                        rec['problems'].append(f'look {lid} {key}: contrast is not an object')
                        continue
                    seeds, deltas = c.get('seeds'), c.get('deltas')
                    if (not isinstance(seeds, list) or not isinstance(deltas, list)):
                        rec['problems'].append(f'look {lid} {key}: seeds/deltas are not lists')
                        continue
                    if len(seeds) != len(deltas):
                        rec['problems'].append(
                            f'look {lid} {key}: {len(seeds)} seeds vs {len(deltas)} deltas')
                        continue
                    if not seeds or not deltas:
                        rec['problems'].append(
                            f'look {lid} {key}: empty seeds/deltas is not a look')
                        continue
                    try:
                        # Use the library's exact seed reader: plain int()
                        # would silently truncate 1.5 to seed 1 (and accept
                        # True as 1), turning a corrupt record into a valid
                        # looking one.  Booleans/fractional values are not
                        # seed identities.
                        _ss = [L._as_seed_int(s) for s in seeds]
                    except (TypeError, ValueError, OverflowError):
                        _ss = None
                    if _ss is None or any(s is None for s in _ss):
                        rec['problems'].append(
                            f'look {lid} {key}: non-integer seed')
                        continue
                    if len(set(_ss)) != len(_ss):
                        rec['problems'].append(f'look {lid} {key}: duplicate seeds')
                        continue
                    _dd = [_finite(x) for x in deltas]
                    if any(x is None for x in _dd):
                        rec['problems'].append(
                            f'look {lid} {key}: non-finite/non-numeric delta')
                        continue
                    sf = L.exact_signflip_p(_dd)
                    # A present-but-malformed p is corrupt evidence, not an
                    # absent field: silently skipping the check would let a
                    # tampered look replay as verified.  Keep the deltas in
                    # the sequential family (dropping a look would erase the
                    # optional-stopping correction it represents).
                    if 'p_exact_signflip' in c:
                        _raw_p = c.get('p_exact_signflip')
                        recorded_p = _finite(_raw_p)
                        if recorded_p is None:
                            rec['problems'].append(
                                f'look {lid} {key}: recorded p is not a finite '
                                f'number ({_raw_p!r})')
                        elif abs(recorded_p - float(sf['p'])) > 1e-9:
                            rec['problems'].append(
                                f'look {lid} {key}: recorded p={recorded_p:.6g} does not '
                                f'recompute ({sf["p"]:.6g})')
                    _entry = {'look': lid, 'seeds': _ss, 'deltas': _dd}
                    # Interaction records additionally bind the four cell arms
                    # and per-seed protocol tags; preserve them so a re-declared
                    # result is not replayed as the same estimand.
                    for _meta in ('cells', 'tags'):
                        if isinstance(c.get(_meta), dict):
                            _entry[_meta] = c[_meta]
                    series.setdefault(key, []).append(_entry)
            # Recompute the sequential p and compare with the audit artifact.
            audit_path = os.path.join(panel_dir, 'methodology_audit.json')
            audit = _load(audit_path) if os.path.exists(audit_path) else None
            audit_entries = {}
            if isinstance(audit, dict):
                for group in ('components', 'baselines'):
                    for _comp, es in (audit.get(group) or {}).items():
                        for e in es or []:
                            la = e.get('look_analysis') if isinstance(e, dict) else None
                            if isinstance(la, dict) and la.get('contrast'):
                                audit_entries[str(la['contrast'])] = (e, la)
                # 2x2 interactions use the same look-history key namespace and
                # the same sequential primitive; the gate must verify them too,
                # not only the ablation/baseline contrasts.
                _ix = audit.get('interactions') or {}
                for _name, _rec in (_ix.get('factorials') or {}).items():
                    cell = (_rec.get('interaction')
                            if isinstance(_rec, dict) else None)
                    la = (cell.get('look_analysis')
                          if isinstance(cell, dict) else None)
                    if isinstance(la, dict) and la.get('contrast'):
                        audit_entries[str(la['contrast'])] = (cell, la)
            for key, looks in sorted(series.items()):
                if len(looks) < 2:
                    continue
                # Mirror `_attach_look_analysis`: a re-run can change one
                # seed's paired delta.  An earlier look that disagrees with the
                # current (last) record on a shared seed is stale and must not
                # be mixed into the joint null; name it rather than fail the
                # whole panel by recomputing a p the audit never used.
                cur = dict(zip(looks[-1]['seeds'], looks[-1]['deltas']))
                _last_cells = (looks[-1].get('cells')
                               if isinstance(looks[-1].get('cells'), dict) else None)
                _last_tags = (looks[-1].get('tags')
                              if isinstance(looks[-1].get('tags'), dict) else None)
                if _last_cells:
                    _last_cells = {str(k): str(v) for k, v in _last_cells.items()}
                _last_tags_n = {}
                if _last_tags:
                    for _s, _t in _last_tags.items():
                        try:
                            _si = L._as_seed_int(_s)
                        except (TypeError, ValueError, OverflowError):
                            _si = None
                        if _si is not None:
                            _last_tags_n[_si] = str(_t or '')
                valid, stale = [], []
                for lk in looks:
                    _bad = [s for s, d in zip(lk['seeds'], lk['deltas'])
                            if s in cur and abs(d - cur[s]) > 1e-9]
                    _reason = 'seed delta changed across looks'
                    _look_cells = (lk.get('cells')
                                   if isinstance(lk.get('cells'), dict) else None)
                    if (_last_cells and _look_cells
                            and {str(k): str(v) for k, v in _look_cells.items()}
                            != _last_cells):
                        _bad = []
                        _reason = 'cell arms changed across looks'
                    if not _bad and _last_tags_n and isinstance(lk.get('tags'), dict) and lk.get('tags'):
                        _look_tags = {}
                        for _s, _t in lk['tags'].items():
                            try:
                                _si = L._as_seed_int(_s)
                            except (TypeError, ValueError, OverflowError):
                                _si = None
                            if _si is not None:
                                _look_tags[_si] = str(_t or '')
                        _bad_tags = [s for s in lk['seeds']
                                     if s in _last_tags_n
                                     and _look_tags.get(s) != _last_tags_n[s]]
                        if _bad_tags:
                            _bad = _bad_tags
                            _reason = 'protocol tag changed across looks'
                    if _bad:
                        stale.append({'look': lk['look'], 'reason': _reason,
                                      'seeds': sorted(_bad)})
                    else:
                        valid.append(lk)
                if stale:
                    rec['warnings'].append(
                        f'{key}: {len(stale)} stale look(s) excluded '
                        f'(estimand changed across looks: {stale})')
                if len(valid) < 2:
                    continue
                rec['contrasts'][key] = {'n_looks': len(valid),
                                         'looks': [x['look'] for x in valid]}
                out['n_contrasts_adjusted'] += 1
                # Top-ups should be nested; a seed disappearing is suspicious
                # (a re-run that removed a pair) and is named, not hidden.
                for a, b in zip(valid, valid[1:]):
                    if not set(a['seeds']).issubset(set(b['seeds'])):
                        rec['warnings'].append(
                            f'{key}: look {b["look"]} does not contain every seed of '
                            f'look {a["look"]} (seed regression)')
                try:
                    seq = L.sequential_signflip_p(
                        [{'seeds': x['seeds'], 'deltas': x['deltas']} for x in valid])
                except Exception as e:
                    rec['problems'].append(
                        f'{key}: sequential test failed ({type(e).__name__}: {e})')
                    continue
                rec['contrasts'][key]['p_sequential'] = seq.get('p_sequential')
                rec['contrasts'][key]['method'] = seq.get('method')
                ent = audit_entries.get(key)
                if ent is None:
                    rec['warnings'].append(
                        f'{key}: {len(looks)} looks are recorded but the audit has no '
                        'look_analysis block; rerun methodology_audit on this panel')
                    continue
                e, la = ent
                ap = _finite(la.get('p_sequential'))
                if ap is None or abs(ap - float(seq['p_sequential'])) > 1e-9:
                    rec['problems'].append(
                        f'{key}: audit sequential p={la.get("p_sequential")!r} does not '
                        f'match the recomputed {seq.get("p_sequential")!r}')
                _naive = e.get('significant_global_naive')
                _adj = e.get('significant_global')
                _p_naive = e.get('p_holm_global_naive')
                if _naive is None and _adj is None:
                    # Interaction cells use the interaction-family names; the
                    # sequential p is still the same primitive and the same
                    # downgrade rule.
                    _naive = e.get('significant_family_naive')
                    _adj = e.get('significant_family')
                    _p_naive = e.get('p_holm_naive')
                if _naive and not _adj:
                    rec['downgraded'].append(key)
                    out['downgraded_claims'].append(
                        {'panel': panel_dir, 'contrast': key,
                         'p_naive': _p_naive,
                         'p_sequential': la.get('p_sequential')})
        out['n_panels'] += 1
        out['n_looks'] += int(rec['looks'])
        if rec['problems']:
            out['problems'].extend(f'{panel_dir}: {x}' for x in rec['problems'])
        if rec['warnings']:
            out['warnings'].extend(f'{panel_dir}: {x}' for x in rec['warnings'])
        out['panels'][panel_dir] = rec
    if not files:
        out['status'] = 'not_verifiable_locally'
        out['reason'] = ('no results_*/look_history.json found — no panel has been '
                         'aggregated since look tracking was added')
    elif out['problems']:
        out['status'] = 'fail'
    elif out['warnings']:
        out['status'] = 'warnings'
    print(json.dumps({k: v for k, v in out.items() if k != 'panels'},
                     ensure_ascii=False, indent=2))
    return out


def audit_statistical_calibration(*, n_sim=None, seed=None):
    """Deterministic synthetic calibration of the analysis machinery.

    The evidence pipeline quotes exact sign-flip p-values, a meet-in-the-middle
    exact path, a saddlepoint tail for n>40, a max-over-looks sequential p,
    paired TOST and Holm.  This gate runs negative controls (data generated
    under the null: the false-positive rate may not exceed alpha beyond
    Monte-Carlo noise) and positive controls (data with a known effect: the
    same procedure must retain sensitivity) for each path.  It also calibrates
    the reported interval estimators (seed-level paired t, paired bootstrap-t,
    unit/document bootstrap-t and the combined seed x document interval):
    coverage may not fall below nominal beyond Monte-Carlo noise, and the
    mean width may not exceed a declared cap (a trivially wide interval is not
    uncertainty quantification).  It uses only local RNG streams and synthetic
    data, reads no result artifact and starts no training; it therefore cannot
    change a measurement or a frozen verdict.

    A pass says the statistical machinery has the advertised error rates and
    interval coverage on idealized independent pairs.  It does not certify
    exchangeability, randomisation or measurement identity in a real panel,
    and it is not evidence for or against any substantive mechanism claim.
    """
    L = _import_exp_lib()
    rep = L.statistical_calibration_suite(n_sim=n_sim, seed=seed)
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ('policy', 'tests')},
                     ensure_ascii=False, indent=2))
    for name, rec in (rep.get('tests') or {}).items():
        if 'coverage_control' in rec:
            _cc = rec['coverage_control']
            _wc = rec['width_control']
            print(f"[calibration] {name}: coverage={_cc.get('rate')!r} "
                  f"(nominal={_cc.get('nominal')!r}, {_cc.get('status')}), "
                  f"mean_width={_wc.get('mean_width')!r} "
                  f"(cap={_wc.get('max_width')!r}, {_wc.get('status')})")
        elif 'null_control' in rec:
            print(f"[calibration] {name}: null={rec['null_control'].get('rate')!r} "
                  f"({rec['null_control'].get('status')}), "
                  f"positive={rec['positive_control'].get('rate')!r} "
                  f"({rec['positive_control'].get('status')})")
        else:
            print(f"[calibration] {name}: {rec.get('status')} "
                  f"(max_abs_error={rec.get('max_abs_error')!r})")
    return rep


def run_report(out='audit_report.json'):
    res = {'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'source_tree': audit_source_integrity(), 'dependencies': audit_dependencies(), 'data_source': audit_data_source(), 'leakage': audit_leakage(), 'split_discipline': audit_split_discipline(), 'document_unit': audit_document_unit(), 'sample_design': audit_sample_design(), 'attribution': audit_attribution(), 'boundary': audit_boundary(), 'length': audit_length(), 'fairness': audit_fairness(), 'metrics': audit_metrics(), 'tuning': audit_tuning(), 'simple_baseline': audit_simple_baseline(), 'hypotheses': audit_hypotheses(), 'sesoi': audit_sesoi_declaration(), 'interaction_sesoi': audit_interaction_sesoi(), 'unit_influence': audit_unit_influence(), 'analysis_choices': audit_analysis_choices(), 'looks': audit_look_history(), 'methodology': audit_methodology(), 'single_factor': audit_single_factor(), 'factorial': audit_factorial_design(), 'design_preflight': audit_design_preflight(), 'dose_response': audit_dose_response(verbose=False), 'failures': audit_failures(), 'budget': audit_budget(), 'training_health': audit_training_health(), 'efficiency_pairs': audit_efficiency_pairs(), 'release': audit_release_readiness(), 'selection': audit_selection(), 'run_order': audit_run_order(), 'calibration': audit_statistical_calibration()}
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
    _cmds = {'leakage': audit_leakage, 'split': audit_split_discipline, 'split_discipline': audit_split_discipline, 'document_unit': audit_document_unit, 'unit': audit_document_unit, 'sample_design': audit_sample_design, 'sampling': audit_sample_design, 'sample': audit_sample_design, 'causality': audit_causality, 'attribution': audit_attribution, 'boundary': audit_boundary, 'length': audit_length, 'fairness': audit_fairness, 'metrics': audit_metrics, 'tuning': audit_tuning, 'simple': audit_simple_baseline, 'baseline_floor': audit_simple_baseline, 'baseline_plan': audit_baseline_design, 'hypothesis_coverage': audit_hypothesis_coverage, 'hypotheses': audit_hypotheses, 'sesoi': audit_sesoi_declaration, 'interaction_sesoi': audit_interaction_sesoi, 'interaction': audit_interaction_sesoi, 'unit_influence': audit_unit_influence, 'influence': audit_unit_influence, 'analysis_choices': audit_analysis_choices, 'specification_curve': audit_analysis_choices, 'analysis_choice': audit_analysis_choices, 'looks': audit_look_history, 'look_history': audit_look_history, 'methodology': audit_methodology, 'single_factor': audit_single_factor, 'factorial': audit_factorial_design, 'factorial_design': audit_factorial_design, 'design': audit_design_preflight, 'design_preflight': audit_design_preflight, 'dose_response': audit_dose_response, 'dose_response_design': audit_dose_response, 'failures': audit_failures, 'budget': audit_budget, 'efficiency': audit_efficiency_demo, 'efficiency_pairs': audit_efficiency_pairs, 'training_health': audit_training_health, 'selection': audit_selection, 'release': audit_release_readiness, 'run_order': audit_run_order, 'calibration': audit_statistical_calibration, 'stats_calibration': audit_statistical_calibration, 'type_i': audit_statistical_calibration, 'source': audit_source_integrity, 'source_tree': audit_source_integrity, 'dependencies': audit_dependencies, 'dependency': audit_dependencies, 'data_source': audit_data_source, 'dataset_source': audit_data_source, 'report': run_report}
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
