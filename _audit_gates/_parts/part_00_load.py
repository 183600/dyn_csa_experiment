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
  baseline_sesoi  方法 vs 参数匹配基线的 SESOI 声明：每个声明基线都有
               唯一、有限、>0 的 margin，且基线臂在 RUN 中排程（设计期，零 GPU）
  interaction_sesoi  2x2 交互 SESOI 声明：每个声明的 factorial 都有唯一、
               有限、>0 的 log ratio-of-ratios margin 与方向假设；缺 margin
               时交互仍可报告，但不能声称可加/可删（设计期，零 GPU）
  interaction_power  2x2 交互设计功效：声明族大小下的精确 sign-flip 分辨率
               下限（n=6、m=5 时最小可达 Holm p=0.15625，任何效应量都
               无法显著）、交互 MDE 与达到 SESOI 所需种子数；pilot 缺失
               报 not_verifiable，不虚构 SD（设计期，零 GPU，不读 test）
               别名：factorial_power / power_2x2；可传一个 pilot JSON 路径
  unit_influence  结果分析：逐评测单位删除（leave-one-out）影响读数是否
               与主终点同口径、是否由单个文章/文档主导（读 analysis_v12/，
               零 GPU；warning 级，不改变冻结判定）
  verdicts     三向方向性判定：supported / refuted / inconclusive /
               not_testable；显著反向不是「未确认」而是反证（读
               analysis_v12/test_stats.json，零 GPU，不改变 p/Holm）
  pvalues      统计量重放：从 test_stats.json 的逐种子配对差重算 exact
               sign-flip p、n/nonzero/p_floor/mean 与整族 Holm；并存
               by-length 曲线逐项重放；缺 per_seed 报 not_verifiable，
               存值与重算不一致判 fail（零 GPU，只读，不改 p/Holm）
               别名：pvalue_replay / recompute_p；可传一个 artifact 路径
  missingness   缺失数据敏感性：按 run_manifest 的计划单元重算每个对照的
               观测/缺失/被控制门排除计数、声明范围内总体均值的 sharp
               bounds 与「抹掉符号/SESOI」的 tipping value；缺计划或旧
               artifact 报 not_verifiable，篡改判 fail。它不改变任何
               p/Holm/verified 或冻结判定（零 GPU；别名 missing /
               missing_data / attrition_sensitivity）
  trajectory    评测轨迹稳定性：从 methodology_audit.json 的配对逐评测步
               log-PPL 序列重算最终符号、late crossing、尾窗 gap trend、
               逐种子一致计数与端点覆盖；缺块或旧 artifact 报
               not_verifiable，篡改判 fail。它不产生 p 值、不改变
               p/Holm/verified 或冻结判定（零 GPU；别名 traj /
               trajectory_stability；可传面板目录或 artifact 路径）
  joint_correction  消融族的相关性感知联合校正：从逐种子配对差重放 exact
               sign-flip max-T（Westfall-Young free step-down）与同时
               区间；缺 family.maxt 报 not_verifiable，篡改 p/区间/排名/
               单调性判 fail；它不改变 Holm、verified 或任何冻结判定
               （零 GPU；别名 joint / maxt / max_t）
  analysis_choices  结果分析：同一主比较的多种分析口径（逐种子 mean/median/
               trimmed、token 池化、逐单元 mean/median/log-ratio/win-rate）
               是否方向一致、带区间的口径是否一致地排除零（读 analysis_v12/，
               零 GPU；warning 级，不改变冻结判定）
  analysis_integrity  分析产物溯源：重算 stamped_analysis 的 content_sha256，
               并核对它记录的全部输入文件哈希；正文被改或输入在分析后被
               改判 fail，输入不在本机则明确 warning，legacy 无摘要报
               not_verifiable（零 GPU，不重算统计量、不改任何判定）
  experiment_ledger  实验事件账本完整性：逐行重放 experiments.jsonl 的
               hash chain（seq/prev/sha256）与 tail anchor；编辑/删除/插入/
               重排/截断判 fail，缺 anchor 或 anchor 落后为 warning，全部
               事件早于链化的 legacy 账本报 not_verifiable（零 GPU，
               不改任何测量或冻结决定）
  disposition  运行处置预注册：把 run_disposition.json 的计划与
               experiments.jsonl 的逐次尝试对账——保留的测量必须是声明的
               那次尝试（默认第一次成功），未计划的种子/臂、成功后被改了
               run_cfg 的重跑、被声明为不可替换的失败类型仍重试、超过尝试
               预算、以及重试只集中在部分臂，分别判 fail/warn；缺计划或缺
               事件报 not_verifiable，账本哈希链失败判 fail。它不重算任何
               PPL/p/Holm，也不改变冻结判定（零 GPU；别名 attempts /
               retries / run_disposition；可传面板目录）
  reproduction  跨运行复现：显式给两份结果目录（或 summary.json），逐格比较
               源码字节、数据切片、配置、执行环境、匹配初始化与最终 PPL；
               缺指纹报 not_verifiable，环境/数据/配置/代码变化各报各的，
               不把「PPL 接近」当成复现（零 GPU，不重跑、不读 test split）
  looks        序列观测 / optional stopping：逐面板 look history 的种子集合、
               逐种子配对差、逐 look p 与 max-over-looks 序列 p 是否自洽；
               朴素最终 p 显著但序列 p 不显著的对照会被点名（零 GPU）
  error_case_inference  坏例桶推断重放：从 paired_errors.json 的 row-cluster
               聚合重算全样本均值/exact p/Holm、确定性发现/复现切分、
               discovery 选桶与 replication 符号/区间/复现判定；缺产物报
               not_verifiable，篡改判 fail（零 GPU，不读 test）
  single_factor  单因素设计契约：每条声明对照的臂配置 diff 恰好等于声明的
               干预键集（定义点审计，不读 results_*）；同 seed 下非干预参数
               必须逐值相同（可选模块构造不能移动共享初始化）；open-block
               矩阵只沿两根声明轴变化；v7/v9 扫描与 v10 NIAH 的 factors 与
               真实 arm diff/共享 override 对账；动态草稿的 mechanism-
               eligible 假设必须由单因素臂对检验；冻结清单的
               mechanism_claim_allowed 标志与实际臂配置 diff 一致
               （零 GPU）
  content_controls  消融对照阶梯：内容消融同时具备 zero（存在性）与
               shift（错位、形状/范数/参数量/读取预算匹配）两个对照；
               两者与 base 的 cfg diff 都必须恰好是 content_mode，且
               各自的留一对照已在 ABLATION_DESIGN 声明（零 GPU）
  sink_controls  消融对照阶梯：sink 消融同时具备 off（缺席）与
               frozen-random（同形状/同参数量/冻结、只破坏学习）两个
               对照；learned−random 隔离学习成分，learned−off 是总效应；
               参数量、冻结状态、cfg diff 与两条留一对照都必须是机器读数
               （零 GPU）
  fusion_controls  消融对照阶梯：local fusion 同时具备 identity（缺席）与
               frozen-random（同 conv 形状/参数量、冻结、不移动共享初始化）
               两个对照；learned−random 隔离学习成分，learned−identity 是
               总效应；别名 fusion_placebo / local_fusion（零 GPU）
  lesion       同 checkpoint 的推理期 lesion：声明组件可构造/可干预、
               权重不变的操作检查，以及逐种子 intact/lesion PPL 的
               necessary/harmful/redundant/inert 判定；缺 checkpoint 或
               操作检查不进 pass，缺产物报 not_verifiable（零 GPU，
               不重训、不读 test split）；别名 inference_lesion /
               lesion_panel / component_lesion
  design       设计前馈：token/参数/读取预算台账、专用对照臂、证据等级
                （v6-v11 永远 exploratory；draft 不能 confirmatory）与动态
                分块 H1-H5 假设/对照（零 GPU，不读 test）
  design_preflight  design 的别名
  sensitivity  设计前馈：功效曲线、target power 所需 q、近似 MDE、配对
               95% CI 半宽与（若有）SESOI 处的功效；缺 pilot/SESOI 时明确
               报不可验证，不虚构效应量（零 GPU，不读 test）
  evaluation   评测单位设计：冻结的文档/文章数、单位级 MDE/CI 半宽/SESOI
               功效，以及 population claim 所需的独立单位与抽样设计门；
               缺 unit pilot 时报 not_verifiable（零 GPU，不读 test）
  dose_response  有序剂量设计：方向/形状预注册、>=3 个单调剂量、只动一个
                声明键、同一种子集与测量协议（零 GPU；清单无该声明时
                not_applicable）
  failures     失败与缺失：error 记录按臂集中度（选择偏差读数，零 GPU）
  training_health  训练曲线健康与 NaN 排查记录的自洽性（零 GPU）
  run_order    运行顺序 / 时间混杂：检查 run_order.json 的完整区组与位置
               平衡，并从 experiments.jsonl 复核实际执行顺序；arm-major
               顺序（一个臂全部种子跑完再跑下一个臂）判 fail（零 GPU）
  run_allocation  运行分配：run_order.json 的 created_at 是否早于第一条
               cell 事件（事后补计划不是预注册）、声明的设备槽位是否覆盖
               并均衡全部 planned cells，以及 summary 记录里的
               provenance.environment 是否让臂与设备/主机完全共线
               （完全共线判 fail，缺环境身份报 not_verifiable；零 GPU，
               不改任何测量或冻结判定）
               别名：allocation / environment_balance
  evaluation_order  最终测试评估顺序 / 测量时间混杂：检查 P5T 在读取 test
               前写下的 clean_by_length_order.json / distractor_probe_order.json
               是否为完整区组、位置平衡、覆盖全部 (arm, seed)，并复核
               experiments.jsonl 的实际评估顺序；有测量但无预计划判 fail，
               arm-major 判 fail（零 GPU）
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
  data_processing  数据处理身份：把 exact/SimHash 近重复清理的 shingle 宽度、
               LSH band/Hamming 阈值、exact hash 与 `protocol_fp` 纳入缓存
               身份；同 vocab/fit_fp 的 train/val/test 必须共享同一协议；
               核对 source→served 行/文档/token 台账及 .npy 侧边长度；
               legacy 缓存明确降级，缺缓存报 not_verifiable
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
