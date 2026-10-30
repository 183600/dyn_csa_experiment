#!/usr/bin/env python3
"""v12 confirmation suite — 探索/确认分离的确认性实验。

前序版本（v7→v11）的全部面板都围绕同一个 validation split 展开，且多轮
追加种子/比较属于**探索性**证据。本驱动把「确认」独立出来：

  P5C   —— 按冻结清单的种子列表训练全部六个冻结臂（主臂 + 从头训练的
           选择器对照 csa_fixed_rope_randidx / csa_fixed_rope_allblocks；
           0-5 已由 P3MT/P4MT/P3MTC 覆盖，同 recipe 下 resume 自动跳过）；
  P5MP  —— 扩展探针在清单种子上重跑（uniform/natural/mask/delete +
           span/doc/topical 腐坏 + NIAH-natural + 从头训练
           对照臂）；
  P5S2  —— 等训练 token 的大模型规模臂（P3SL_TOK，8000 步）；
  P5S3  —— 6ND 参数量代理规模的大模型规模臂（P3SL_FLOP，参数量比例缩步）；
  P5T   —— **冻结后的独立最终测试**：按仓库根的冻结清单
           `confirmation_frozen.json`（预注册主比较、终点、种子数、检验与
           校正规则），在 WikiText-103 的 held-out test split（wiki.test.raw，
           其它 phase 不读）上一次性评估冻结臂，输出确认性统计。

加载/运行/统计三层的硬门槛：
  1. 功效门槛（意见一）：加载清单即校验「该方案是否数学上可能确认」
     （min attainable Holm p = m·2/2^n < α）。6 种子 × 双侧 × Holm/4 的
     原设计在该门槛下会被拒绝——清单 v2 已在**未读任何 test 数据**的
     前提下把种子修订为 12（见清单 revision 块）。
  2. 严格准入（意见二）：`finaltest_stats` 只接纳同时通过「冻结种子白名
     表、唯一记录、非空 data_fp、单一冻结 test 切片、_code 与 recipe 的
     机器校验、manifest 哈希绑定」的记录；歧义重复与混切片直接拒绝。
  3. 冻结绑定（意见三）：每条结果记录携带 manifest_sha256 与
     ckpt_sha256；首次 test 访问写入 `.freeze_lock.json`（清单哈希 +
     test 指纹 + 测量源码树摘要），此后清单/test/源码变更即拒绝
     （V12_REFREEZE=1 显式改冻结，前轮降级为探索性；分析阶段可用
     V12_ALLOW_SOURCE_DRIFT=1 披露式重跑，审计仍判 fail）；清单修订后
     同 outdir 的旧结果自动作废。
  4. 范围声明（意见四）：确认性主张只覆盖固定分块 CSA 机制
     （清单 scope 块）；动态分块/HCA 的优劣需要各自的预注册。

模式：
  python -m explib.confirmation smoke            # 端到端自检（微型 ckpt + 冻结清单副本）
  python -m explib.confirmation plan             # 零 GPU 预检：戳/实现绑定/准入/功效/预算台账/证据等级，不读 test；
                                      # 通过后自动把快照留档到 analysis_v12/plan_snapshot.json
  python -m explib.confirmation dynamic-plan     # 零 GPU 预检动态/HCA 草稿：假设、专用对照、token/参数/读取预算、
                                      # disjoint-source/校准/功效前置；通过也只说明设计自洽，不是确认
  python -m explib.confirmation power            # 零 GPU：用 P5C validation 差值做精确条件功效（同样不读 test）
  python -m explib.confirmation full             # P5C -> P5H -> P5PW -> P5T -> P5MP -> P5S2/P5S3 -> report
  python -m explib.confirmation auto             # 无人值守：与 full 同链，但把「plan 预检通过」的
                                      # 全部机器可查证据（清单哈希/代码戳/git 树/四个人工
                                      # 核对字段：arms/seeds/主比较/α）自动落盘为审计快照，
                                      # 人工核对变为事后异步，不再阻塞链的启动
  python -m explib.confirmation phase P5C        # 单阶段（可断点续跑）
  python -m explib.confirmation phase P5PW       # 仅功效预检
  python -m explib.confirmation analysis         # 零 GPU：确认性统计 -> analysis_v12/
  python -m explib.confirmation report           # 从落盘产物重建 REPORT_v12.md（零 GPU）

P5T 的防呆：代码语义戳与冻结清单不一致时拒绝运行（必须显式重冻结）；
manifest 不存在时拒绝运行（不许「先看数据再补注册」）；清单不可确认、
记录不可溯源、切片不一致、锁不匹配时全部拒绝运行。
"""
import copy
import gc
import hashlib
import json
import math
import numbers
import os
import re
import subprocess
import sys
import time
import traceback
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except (ValueError, OSError):
            pass
import numpy as np
import torch
import torch.nn.functional as F
import explib.corruption as V10
V9 = V10.V9
V8 = V10.V8
V = V10.V
L = V10.L
REPO = V10.REPO
DEVICE = L.DEVICE
RUN_CAP_V12 = dict(L.RUN_CAP)
RUN_CAP_V12.update(total_hours=float(os.environ.get('V12_CAP_HOURS', 30.0)), state_path='run_time_state_v12.json', already_hours=0.0)

def make_guard():
    g = L.TimeGuard(RUN_CAP_V12)
    if not g.state.get('sps_by_class'):
        for src in ('run_time_state_v11.json', 'run_time_state_v10.json', 'run_time_state_v9.json', 'run_time_state_v8.json', 'run_time_state_v7.json'):
            if os.path.exists(src):
                try:
                    st = json.load(open(src, encoding='utf-8'))
                    sbc = st.get('sps_by_class') or {}
                    norm = st.get('norm_sps')
                    if not sbc and norm is None:
                        print(f'[v12] NOTE: {src} holds no calibration data — leaving the guard uncalibrated rather than recording an empty calibration')
                        continue
                    g.state['sps_by_class'] = sbc
                    g.state['norm_sps'] = norm
                    g._save()
                    print(f'[v12] TimeGuard calibrated from {src}')
                    break
                except Exception as e:
                    print(f'[v12] WARNING: cannot read {src} ({type(e).__name__}: {e}) — this run starts UNCALIBRATED (the first admission uses the conservative default)')
    return g

# ---- 阶段配置 -------------------------------------------------------------
# 冻结清单把种子数从 6 修订为 12（修订发生在任何
# test-split 评估之前，见清单 revision 块）。P5C/P5MP 的
# 种子列表**不再是这里的硬编码**，运行期由 `_frozen_seeds()` 从冻结清单
# 读取（单一事实来源）；P5C 同时覆盖全部六个冻结臂（主臂 0-5 已由
# P3MT/P4MT 训练，resume 自动跳过；主臂 6-11 与对照臂统一在此补齐），
# 否则清单扩种子后没有任何驱动会训练主臂新种子。
P5C_CFG = dict(V10.P3MT_PAYLOAD, seeds=[0, 1, 2, 3, 4, 5],
               variants=['csa_fixed_rope', 'full_rope', 'full_rope_sink',
                         'csa_fixed_rope_nosink', 'csa_fixed_rope_randidx_seeded',
                         'csa_fixed_rope_allblocks'],
               # Diagnostic-only validation monitoring: never used for model
               # selection, but P5T refuses the one-shot test access when a
               # frozen arm trained through a non-finite/degrading tail (a
               # flat/plateaued tail is recorded as non-blocking).
               diag_eval_every=250)
P5MP_CFG = dict(V10.PROBE, seeds=[0, 1, 2, 3, 4, 5],
               variants=['csa_fixed_rope', 'full_rope', 'full_rope_sink',
                         'csa_fixed_rope_nosink', 'csa_fixed_rope_randidx_seeded',
                         'csa_fixed_rope_allblocks'],
               arm_overrides={v: ['learned'] for v in ('csa_fixed_rope_randidx_seeded', 'csa_fixed_rope_allblocks')})
P5S2_CFG = dict(V10.P3SL_TOK_CFG, seeds=[0, 1, 2])
P5S3_CFG = dict(V10.P3SL_FLOP_CFG, seeds=[0, 1, 2])
TEST_OUTDIR = 'results_lm_test'
FREEZE_MANIFEST = 'confirmation_frozen.json'
# The confirmatory phases run FIRST.  The exploratory scale/probe panels used
# to sit between P5C and P5T, so an exhausted run-time cap could stop the driver
# after all training and before the single test-split evaluation.  Order is now
# data-priority order: train -> health -> power preflight -> test -> exploratory.
PHASES = [('P5C', 'lenphase', P5C_CFG, None), ('P5H', 'health', None, None),
          ('P5PW', 'power', None, None), ('P5T', 'finaltest', None, None),
          ('P5MP', 'probe', P5MP_CFG, None), ('P5S2', 'run', P5S2_CFG, None),
          ('P5S3', 'run', P5S3_CFG, None)]
_PLAN_DONE = False

# Health-gate policy (review 5).  A fixed-budget run normally ends on a flat
# cosine tail; that is a converged/plateaued curve, not evidence that the
# weights are unusable.  Only a non-finite, actively overfitting/degrading or
# entirely unmonitored curve may block the one-shot test access.  Non-blocking
# statuses are still recorded and printed, never silently dropped.
_HEALTH_BLOCKING = frozenset({
    'nonfinite', 'overfitting', 'validation_degrading',
    'insufficient_data', 'unmonitored', 'unknown',
})
_HEALTH_NONBLOCKING = frozenset({
    'underfitting_or_stalled', 'validation_improving_train_flat',
})


def _health_blocks(status):
    """True unless the status is explicitly known to be safe.

    This is a one-shot test-access gate, so the check is a whitelist, not a
    blacklist: a future/renamed/typo'd status must fail closed instead of
    silently becoming "non-blocking" merely because it is absent from
    ``_HEALTH_BLOCKING``.
    """
    s = str(status if status is not None else 'unknown')
    return s != 'healthy' and s not in _HEALTH_NONBLOCKING

def run_training_health(cfg=None):
    """Pre-test health gate over the frozen arms/seeds.

    Reads the diagnostic curves stored in each checkpoint by P5C/P3MT.  The
    run is never allowed to change arms/seeds/endpoints; it only decides
    whether the one-shot test access is used or the run is stopped for
    investigation.
    """
    cfg = dict(cfg or {})
    manifest = _load_manifest(cfg.get('manifest') or FREEZE_MANIFEST)
    ckpt_dir = cfg.get('ckpt_dir') or FINALTEST_CFG['ckpt_dir']
    rows = {}
    unhealthy = []
    nonblocking = []
    missing = []
    for v in manifest['arms']:
        for seed in manifest['seeds']:
            key = f'{v}::s{seed}'
            ck = os.path.join(ckpt_dir, f'{v}_seed{seed}.pt')
            if not os.path.exists(ck):
                rows[key] = {'status': 'missing_checkpoint', 'path': ck}
                missing.append(key)
                continue
            try:
                d = V._load_ckpt(ck)
            except Exception as e:
                rows[key] = {'status': 'unreadable_checkpoint', 'error': f'{type(e).__name__}: {e}', 'path': ck}
                unhealthy.append(key)
                continue
            d = d if isinstance(d, dict) else {}
            h = d.get('curve_health') if isinstance(d.get('curve_health'), dict) else None
            status = (h or {}).get('status', 'unmonitored')
            rows[key] = {'status': status, 'health': h,
                         'best_step': d.get('best_step'),
                         'best_val_ppl': d.get('best_val_ppl'),
                         'final_ppl': d.get('final_ppl'),
                         'path': ck}
            if _health_blocks(status):
                unhealthy.append(key)
            elif status != 'healthy':
                nonblocking.append(key)
            del d
            try:
                gc.collect()
            except Exception:
                pass
    report = {'kind': 'training_health', 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
              'manifest': cfg.get('manifest') or FREEZE_MANIFEST,
              'n_arms': len(manifest['arms']), 'n_seeds': len(manifest['seeds']),
              'n_cells': len(rows), 'n_missing': len(missing), 'n_unhealthy': len(unhealthy),
              'n_nonblocking': len(nonblocking),
              'missing': missing, 'unhealthy': unhealthy, 'nonblocking': nonblocking,
              'blocking_statuses': sorted(_HEALTH_BLOCKING),
              'nonblocking_statuses': sorted(_HEALTH_NONBLOCKING),
              'status': ('pass' if not missing and not unhealthy else 'fail'),
              'note': ('diagnostic only: this gate never selects a checkpoint or changes the frozen plan. '
                       'A flat/plateaued tail (underfitting_or_stalled) is recorded as non-blocking; only '
                       'nonfinite/overfitting/validation_degrading/insufficient_data/unmonitored block P5T.'),
              'cells': rows}
    out = cfg.get('out') or os.path.join(TEST_OUTDIR, 'training_health.json')
    L.atomic_write_json(out, report, indent=2)
    print(f'[p5h] training health: {report["status"]} ({len(unhealthy)} blocking, '
          f'{len(nonblocking)} non-blocking, {len(missing)} missing) -> {out}')
    if nonblocking:
        print(f'[p5h] NOTE: {len(nonblocking)} cell(s) are plateaued/converged but non-blocking: {nonblocking[:6]}'
              + (' …' if len(nonblocking) > 6 else ''))
    allow = os.environ.get('V12_ALLOW_UNHEALTHY', '').strip().lower() in ('1', 'true', 'yes', 'on')
    if report['status'] != 'pass' and not allow:
        raise SystemExit(f'[p5h] REFUSE: frozen-arm training health is not clean ({len(unhealthy)} blocking, {len(missing)} missing); investigate before using the one-shot test access (V12_ALLOW_UNHEALTHY=1 is the explicit override)')
    if report['status'] != 'pass':
        print('[p5h] WARNING: V12_ALLOW_UNHEALTHY=1 — continuing despite unhealthy/missing training health')
    return report


def run_phase(name, guard, par=None):
    global _PLAN_DONE
    if name in ('P5C', 'P5MP') and not _PLAN_DONE:
        # A direct `phase P5C` must not start training under a plan that the
        # P5T admission gate would reject.  `run_full` runs this once at
        # startup; the cache avoids repeating it for each phase.
        plan_check()
    import par as _par
    _ctx = _par.resolve_ctx(par, label=f'v12 {name}')
    for pname, kind, payload, seeds in PHASES:
        if pname != name:
            continue
        cfg = payload() if callable(payload) else payload
        if pname in ('P5C', 'P5MP'):
            # the frozen seed list is the manifest's, not a hard-coded copy
            cfg = dict(cfg, seeds=_frozen_seeds())
        if isinstance(_ctx, dict):
            _ctx = dict(_ctx, module='explib.confirmation', ckpt_install=True, ckpt_flag=False)
        if kind == 'lenphase':
            return V.run_lenphase(cfg, guard=guard, label=f'v12 {pname}', par=_ctx)
        if kind == 'probe':
            return V10.run_probe(cfg, guard=guard, label=f'v12 {pname}', par=_ctx)
        if kind == 'power':
            # Zero-GPU pre-test power calibration; never reads the test split.
            return run_power(manifest_path=FREEZE_MANIFEST)
        if kind == 'health':
            return run_training_health({'manifest': FREEZE_MANIFEST, 'ckpt_dir': FINALTEST_CFG['ckpt_dir'], 'out': os.path.join(TEST_OUTDIR, 'training_health.json')})
        if kind == 'finaltest':
            print('[v12] NOTE: the final-test phase has no task-level parallel path, running it serially (evaluation only, ckpt per cell)')
            return run_finaltest(dict(FINALTEST_CFG), guard=guard)
        if cfg.get('steps') is None and pname == 'P5S3':
            cfg = dict(cfg, steps=V10._flop_matched_steps(V10.P3SS_CFG, cfg))
        if seeds:
            cfg = dict(cfg, seeds=list(seeds))
        s, _a = L.run(cfg, seeds=cfg.get('seeds'), guard=guard, label=f'v12 {pname}', par=_ctx)
        return s
    raise SystemExit(f'unknown phase {name}')

# ---- P5T：冻结后的独立最终测试 ----------------------------------------------
FINALTEST_CFG = dict(manifest=FREEZE_MANIFEST, outdir=TEST_OUTDIR, ckpt_dir='results_v10_mech/ckpt', mech_summary='results_v10_mech/summary.json', bylen_summary=os.path.join(TEST_OUTDIR, 'bylen.json'), probe_summary=os.path.join(TEST_OUTDIR, 'distractor.json'), eval_lens=[512, 1024, 2048, 4096], n_seq=48, target=512, chunk=4)

def _load_manifest(path):
    if not os.path.exists(path):
        raise SystemExit(f'[v12] REFUSE: freeze manifest {path} does not exist — a confirmation run requires a PRE-REGISTERED plan (arms, seeds, endpoints, statistics); write the manifest first, then run P5T')
    try:
        m = json.load(open(path, encoding='utf-8'))
    except Exception as e:
        raise SystemExit(f'[v12] REFUSE: freeze manifest {path} cannot be parsed ({type(e).__name__}: {e})')
    if not isinstance(m, dict):
        raise SystemExit(f'[v12] REFUSE: freeze manifest {path} is not a JSON object ({type(m).__name__})')
    if str(m.get('status') or '').strip().lower() == 'draft':
        # A draft is a design record, not a pre-registration.  It exists so the
        # dynamic/HCA study cannot be silently squeezed into the fixed v5
        # one-shot: it needs its own disjoint test source and freeze revision.
        _draft_msg = (f'[v12] REFUSE: {path} is status=draft — a draft is not a runnable '
                      'pre-registration. Reserve a disjoint test source (or a disjoint, '
                      'never-read document split), freeze the plan and record its revision '
                      'before using it. (V12_ALLOW_DRAFT=1 is only for inspecting the schema; '
                      'it does not make the result confirmatory.)')
        if os.environ.get('V12_ALLOW_DRAFT', '').strip().lower() not in ('1', 'true', 'yes', 'on'):
            raise SystemExit(_draft_msg)
        print(_draft_msg.replace('REFUSE:', 'WARNING (V12_ALLOW_DRAFT):'))
    for field in ('arms', 'seeds', 'primary_comparisons', 'statistics'):
        if field not in m:
            raise SystemExit(f'[v12] REFUSE: freeze manifest {path} lacks the required field {field!r} — refusing to guess the pre-registration')
    if not isinstance(m['arms'], list) or not m['arms'] or not all(isinstance(a, str) and a for a in m['arms']):
        raise SystemExit(f'[v12] REFUSE: arms must be a non-empty list of arm names ({m.get("arms")!r})')
    if len(set(m['arms'])) != len(m['arms']):
        raise SystemExit(f'[v12] REFUSE: duplicate arm names would collapse the frozen panel ({m.get("arms")!r})')
    if not isinstance(m['seeds'], list) or not m['seeds']:
        raise SystemExit(f'[v12] REFUSE: seeds must be a non-empty list ({m.get("seeds")!r})')
    if not isinstance(m['primary_comparisons'], list) or not m['primary_comparisons']:
        raise SystemExit(f'[v12] REFUSE: primary_comparisons must be a non-empty list ({m.get("primary_comparisons")!r})')
    for _ci, _comp in enumerate(m['primary_comparisons']):
        if not isinstance(_comp, dict):
            # The rest of this loader (and admission) reads fields with
            # .get(); a scalar/list entry would crash with AttributeError
            # instead of refusing the malformed pre-registration.
            raise SystemExit(
                f'[v12] REFUSE: primary_comparisons[{_ci}] must be an object, '
                f'got {type(_comp).__name__} ({_comp!r})')
    if not isinstance(m['statistics'], dict):
        raise SystemExit(f'[v12] REFUSE: statistics must be an object ({m.get("statistics")!r})')
    for _fld, _val in (('binding', m.get('binding')), ('eval', m.get('eval'))):
        if _val is not None and not isinstance(_val, dict):
            raise SystemExit(f'[v12] REFUSE: {_fld} must be an object ({_val!r})')

    def _int_field(value, label):
        if isinstance(value, (bool, np.bool_)):
            raise SystemExit(f'[v12] REFUSE: {label}={value!r} is not an integer')
        if isinstance(value, numbers.Integral):
            return int(value)
        try:
            f = float(value)
        except (TypeError, ValueError, OverflowError):
            raise SystemExit(f'[v12] REFUSE: {label}={value!r} is not an integer')
        if not math.isfinite(f) or f != math.floor(f):
            raise SystemExit(f'[v12] REFUSE: {label}={value!r} is not an integer')
        return int(f)

    _exp = m.get('code_semantics_expected')
    if L._kernel_stamp():
        # The frozen plan was pre-registered for the eager measurement
        # semantics.  Kernel mode appends '-tk1:<backend>' to `_code`, so the
        # load gate and the admission gate would disagree and P5T could burn
        # the one-shot test access before admitting zero records.
        raise SystemExit(f'[v12] REFUSE: CSA_KERNEL is set ({L._kernel_stamp()!r}) but the frozen confirmation plan is EAGER-only; unset CSA_KERNEL or re-freeze explicitly')
    if _exp is not None and str(_exp) != str(L.CODE_SEMANTICS):
        raise SystemExit(f'[v12] REFUSE: freeze manifest expects code semantics {_exp!r} but the current code is {L.CODE_SEMANTICS!r} — the code changed after freezing; re-freeze consciously (update {path} with a dated note) instead of silently testing new code under the old pre-registration')
    # The statistics block is a pre-registration contract: the analyzer must
    # implement exactly the frozen test/correction, so a mismatch is refused
    # rather than silently computed and mislabelled in the report.
    _stat = m.get('statistics') or {}
    # The held-out access budget is part of the pre-registration.  Validate it
    # before any training or test read so a typo cannot silently disable the
    # repeated-look accounting (the analysis would otherwise fall back to the
    # single-look default).
    try:
        L.test_access_policy(m)
    except ValueError as _ta_exc:
        raise SystemExit(
            '[v12] REFUSE: statistics.test_access_policy is invalid: '
            f'{_ta_exc} — the held-out access budget must be valid BEFORE the '
            'first test read')
    _ns = _stat.get('n_seeds')
    if _ns is not None and _int_field(_ns, 'statistics.n_seeds') != len(m['seeds']):
        raise SystemExit(f'[v12] REFUSE: freeze manifest declares n_seeds={_ns} but lists {len(m["seeds"])} seeds — the pre-registered resolution and the seed list disagree')
    _bind = m.get('binding') or {}
    _ts = (_bind.get('test_slice') or {}) if isinstance(_bind, dict) else {}
    if not isinstance(_ts, dict):
        raise SystemExit(f'[v12] REFUSE: binding.test_slice must be an object ({_ts!r})')
    _eval = m.get('eval') or {}
    if _eval:
        # The endpoint grid is part of the pre-registered plan.  int(4096.5)
        # would silently score a different sequence length, and a boolean is
        # a flag rather than an endpoint.
        _bl = _eval.get('by_length_ppl')
        if _bl is not None:
            if not isinstance(_bl, list):
                raise SystemExit(f'[v12] REFUSE: eval.by_length_ppl must be a list ({_bl!r})')
            for _i, _x in enumerate(_bl):
                _v = _int_field(_x, f'eval.by_length_ppl[{_i}]')
                if _v <= 0:
                    raise SystemExit(f'[v12] REFUSE: eval.by_length_ppl[{_i}]={_x!r} must be a positive length')
    if _ts and _eval:
        _ts_n = _ts.get('n_seqs')
        _ev_n = _eval.get('n_seqs_clean')
        if (_ts_n is not None and _ev_n is not None
                and _int_field(_ts_n, 'binding.test_slice.n_seqs') != _int_field(_ev_n, 'eval.n_seqs_clean')):
            raise SystemExit(f'[v12] REFUSE: binding.test_slice.n_seqs={_ts_n} disagrees with eval.n_seqs_clean={_ev_n} — the machine-checkable binding block and the evaluation plan must describe the same slice')
    if _eval and _int_field(_eval.get('n_seqs_clean') or 0, 'eval.n_seqs_clean') <= 0:
        raise SystemExit('[v12] REFUSE: eval.n_seqs_clean is missing/zero — the frozen row count is part of the held-out slice, not a runtime default')
    _tokb = (_bind.get('tokenizer') or {}) if isinstance(_bind, dict) else {}
    if not isinstance(_tokb, dict):
        raise SystemExit(
            f'[v12] REFUSE: binding.tokenizer must be an object ({_tokb!r})')
    if _tokb and str(_tokb.get('fit', '')).lower() and 'train' not in str(_tokb.get('fit', '')).lower():
        raise SystemExit(f'[v12] REFUSE: binding.tokenizer.fit={_tokb.get("fit")!r} is not the train-only fit required by the freeze')
    _test = str(_stat.get('test', 'exact two-sided sign-flip over paired per-seed deltas')).lower()
    if 'sign-flip' not in _test and 'sign flip' not in _test:
        raise SystemExit(f'[v12] REFUSE: freeze manifest pre-registers test {_stat.get("test")!r}, but this analyzer only implements the exact two-sided sign-flip test — update the code or re-freeze')
    _corr = str(_stat.get('correction', 'holm step-down')).lower()
    if 'holm' not in _corr:
        raise SystemExit(f'[v12] REFUSE: freeze manifest pre-registers correction {_stat.get("correction")!r}, but this analyzer only implements Holm step-down — update the code or re-freeze')
    for comp in m['primary_comparisons']:
        for _k in ('name', 'a', 'b'):
            if not comp.get(_k):
                raise SystemExit(f'[v12] REFUSE: a frozen primary comparison lacks {_k!r} ({comp!r})')
        _es = comp.get('expected_sign')
        if _es is not None and str(_es).lower() not in ('positive', 'negative'):
            raise SystemExit(f'[v12] REFUSE: comparison {comp.get("name")!r} has expected_sign={_es!r} (expected "positive", "negative", or null)')
        if comp.get('a') not in m['arms'] or comp.get('b') not in m['arms']:
            raise SystemExit(f'[v12] REFUSE: comparison {comp.get("name")!r} names arms {comp.get("a")!r}/{comp.get("b")!r} outside the frozen arms {m["arms"]}')
        if comp.get('a') == comp.get('b'):
            raise SystemExit(f'[v12] REFUSE: comparison {comp.get("name")!r} pairs an arm with itself ({comp.get("a")!r}) — a self-contrast is identically zero')
    _names = [str(_c.get('name')) for _c in m['primary_comparisons']]
    if len(set(_names)) != len(_names):
        raise SystemExit('[v12] REFUSE: duplicate primary-comparison names would collapse the Holm family — every declared contrast needs a unique name')
    # ---- a plan that CANNOT confirm is refused at load time.
    # 6 seeds x two-sided sign-flip x Holm/4 has a smallest attainable Holm p of
    # m * 2/2^n = 0.125 > 0.05: confirmation probability is ZERO BY CONSTRUCTION,
    # which is a resolution limit, not "conservatism".  Descriptive-only
    # manifests (smoke) may opt out explicitly.
    try:
        _alpha = float(_stat.get('alpha', 0.05))
    except (TypeError, ValueError, OverflowError):
        raise SystemExit(f'[v12] REFUSE: statistics.alpha={_stat.get("alpha")!r} is not numeric')
    if not (math.isfinite(_alpha) and 0.0 < _alpha < 1.0):
        raise SystemExit(f'[v12] REFUSE: statistics.alpha={_alpha!r} must be in (0, 1)')
    # `descriptive_only` is a pre-registration escape hatch.  A truthy string
    # such as "false" must not open it; only a real JSON boolean True does.
    _desc_flag = _stat.get('descriptive_only')
    if _desc_flag is not None and not isinstance(_desc_flag, (bool, np.bool_)):
        raise SystemExit(f'[v12] REFUSE: statistics.descriptive_only={_desc_flag!r} '
                         'must be a real boolean, not a truthy/falsy surrogate')
    _descriptive_only = _desc_flag is True
    _n = len(m['seeds'])
    _m_family = len(m['primary_comparisons'])
    _min_holm = L.holm_min_attainable_p(_n, _m_family, _alpha)
    if math.isfinite(_min_holm) and _min_holm > _alpha and not _descriptive_only:
        _kmin = L.signflip_min_consistency(_n, _m_family, _alpha)
        raise SystemExit(f'[v12] REFUSE: the frozen plan cannot confirm ANYTHING by construction — with n={_n} seeds, a two-sided exact sign-flip and a Holm family of m={_m_family}, the smallest attainable Holm p is {_min_holm:.4g} > alpha={_alpha} (all {_n} seeds one-sided still gives {_kmin is None and "insufficient" or f"k_min={_kmin} needed"}). This is a resolution ceiling, NOT a conservative design. Revise the plan BEFORE any test-split evaluation (add seeds / shrink the family / change the test) and record the revision in the manifest history. ({_stat.get("descriptive_only")!r})')
    if not isinstance(m.get('seeds'), list) or not all(isinstance(s, int) and (not isinstance(s, bool)) for s in m['seeds']) or len(set(m['seeds'])) != len(m['seeds']):
        raise SystemExit(f'[v12] REFUSE: seeds must be a list of unique integers ({m.get("seeds")!r}) — seed identity is the pairing unit of the whole protocol')
    # The declared n-floor is part of the decision rule.  Refuse a
    # floor that is below 2, above the frozen seed count, or mathematically
    # unable to reach Holm significance.
    _n_min = _stat.get('min_seeds_for_confirmation')
    if _n_min is not None:
        _n_min = _int_field(_n_min, 'statistics.min_seeds_for_confirmation')
        if _n_min < 2 or _n_min > _n:
            raise SystemExit(f'[v12] REFUSE: statistics.min_seeds_for_confirmation={_n_min} is outside [2, {_n}] — the declared confirmation floor must be reachable with the frozen seed list')
        _min_holm_floor = L.holm_min_attainable_p(_n_min, _m_family, _alpha)
        if math.isfinite(_min_holm_floor) and _min_holm_floor > _alpha and not _descriptive_only:
            raise SystemExit(f'[v12] REFUSE: the declared confirmation floor n={_n_min} cannot attain Holm significance (min p={_min_holm_floor:.4g} > alpha={_alpha}); raise the floor or add seeds BEFORE any test access')
    # ---- binding block.  Optional but validated when present;
    #     every record written under this manifest carries the manifest's sha256.
    _bind = m.get('binding') or {}
    if _bind:
        if not isinstance(_bind, dict):
            raise SystemExit(f'[v12] REFUSE: binding block must be an object ({_bind!r})')
        _sig = _recipe_signature({'binding': _bind})
        if (_bind.get('training_recipe_signature') is not None or _bind.get('training_recipe') is not None) and _sig is None:
            raise SystemExit(f'[v12] REFUSE: binding.training_recipe_signature must be an object of machine-checkable fields ({_bind.get("training_recipe_signature")!r})')
        _per = _bind.get('per_arm_recipe_signature')
        if _per is not None:
            if not isinstance(_per, dict) or set(_per) != set(m['arms']):
                raise SystemExit(f'[v12] REFUSE: binding.per_arm_recipe_signature must be an object keyed by exactly the frozen arms {sorted(m["arms"])} (got {_per!r})')
            for _arm, _asig in _per.items():
                if not isinstance(_asig, dict) or not _asig:
                    raise SystemExit(f'[v12] REFUSE: binding.per_arm_recipe_signature[{_arm!r}] must be a non-empty object')
                if 'mr' in _asig:
                    try:
                        _mr = float(_asig['mr'])
                    except (TypeError, ValueError, OverflowError):
                        raise SystemExit(f'[v12] REFUSE: per-arm mr for {_arm!r} is not numeric ({_asig["mr"]!r})')
                    if not math.isfinite(_mr) or _mr <= 1.0:
                        raise SystemExit(f'[v12] REFUSE: per-arm mr for {_arm!r} must be finite and > 1 ({_mr!r})')
    m['manifest_sha256'] = _file_sha256(path)
    # Keep the resolved source path on the loaded object.  The freeze lock
    # records which manifest actually granted the first test access; without
    # this it fell back to the default filename even when the caller loaded a
    # different (e.g. archived/renamed) manifest.
    m['_path'] = os.path.abspath(path)
    return m


_FILE_SHA_CACHE = {}


def _file_sig(path):
    """Metadata fingerprint for a checkpoint: (size, mtime_ns, ctime_ns, inode)."""
    st = os.stat(path)
    return (int(getattr(st, 'st_size', 0)),
            int(getattr(st, 'st_mtime_ns', int(st.st_mtime * 1e9))),
            int(getattr(st, 'st_ctime_ns', int(st.st_ctime * 1e9))),
            int(getattr(st, 'st_ino', 0)))


def _file_sha256(path, chunk=1 << 20):
    # Bind the digest to the full metadata identity (size, mtime, ctime, inode),
    # so an unchanged file is not re-read while any rewrite — even one that
    # preserves size and mtime — still forces a fresh hash.
    try:
        _key = (os.path.abspath(path),) + _file_sig(path)
        _hit = _FILE_SHA_CACHE.get(_key)
        if _hit is not None:
            return _hit
    except OSError:
        _key = None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    _sha = h.hexdigest()
    if _key is not None:
        if len(_FILE_SHA_CACHE) >= 64:
            _FILE_SHA_CACHE.pop(next(iter(_FILE_SHA_CACHE)))
        _FILE_SHA_CACHE[_key] = _sha
    return _sha


def _recipe_signature(manifest, arm=None):
    """The frozen structured training-recipe signature (binding block).

    键名 `training_recipe_signature`（v2 清单）；旧名 `training_recipe`
    仍被接受（smoke 清单与兼容路径）。

    An arm-specific signature
    (``binding.per_arm_recipe_signature[<arm>]``) is merged on top of the
    shared signature.  This is how the parameter-matching knob ``mr`` is
    pinned without pretending the four arms share one value: ``full_rope`` is
    trained at ``mr≈6.7635`` while the csa arms stay at ``mr=4``.
    """
    _bind = manifest.get('binding') or {}
    if not isinstance(_bind, dict):
        return None
    sig = _bind.get('training_recipe_signature')
    if sig is None:
        sig = _bind.get('training_recipe')
    if not isinstance(sig, dict) or not sig:
        sig = None
    if arm is not None:
        _per = _bind.get('per_arm_recipe_signature') or {}
        _extra = _per.get(arm) if isinstance(_per, dict) else None
        if isinstance(_extra, dict) and _extra:
            sig = dict(sig or {})
            sig.update(_extra)
    return sig if isinstance(sig, dict) and sig else None


def _train_fp_required(manifest):
    """Whether the frozen plan makes train_fp a hard admission requirement.

    A manifest that DECLARES ``binding.training_data_fingerprint`` is binding
    even while its ``canonical_train_fp`` is still null: the run must derive
    one value from the checkpoints before the first test read and lock it.
    A ``bool(want) or bool(ckpt) or bool(record)`` rule would be fail-open
    (all three empty -> no check at all), so it is not used.
    """
    _bind = manifest.get('binding') or {}
    if not isinstance(_bind, dict):
        return False
    return _bind.get('training_data_fingerprint') is not None or _bind.get('train_fp') is not None


def _frozen_train_fp(manifest):
    """Canonical training-corpus fingerprint declared by the frozen plan.

    Accepted shapes: ``binding.training_data_fingerprint`` (string or an
    object with ``canonical_train_fp`` / ``train_fp`` / ``value``),
    ``binding.train_fp``, or a top-level ``training_data_fingerprint``.
    ``None`` means the manifest froze the rule but not a literal value; in that
    case every admission/checkpoint must still agree on ONE non-null train_fp
    and the first P5T access locks it.
    """
    for _obj in (manifest.get('binding') or {}, manifest):
        if not isinstance(_obj, dict):
            continue
        for _k in ('training_data_fingerprint', 'canonical_train_fp', 'train_fp', 'training_fp'):
            _v = _obj.get(_k)
            if isinstance(_v, dict):
                _v = (_v.get('canonical_train_fp') or _v.get('train_fp')
                      or _v.get('value'))
            if isinstance(_v, str) and _v:
                return _v
    return None


def _frozen_vocab_size(manifest, cfg=None):
    """Vocabulary width declared by the frozen plan, or None if unspecified.

    The test cache tag contains the vocab width, so falling back to a
    repository constant (the historical 8192) would look up the wrong held-out
    slice after a scale-up.  The declared training recipe is authoritative;
    every per-arm signature must agree.
    """
    def _width(value, label):
        # A manifest is machine-checkable evidence: a boolean, a fractional
        # float or a non-finite value is not a vocabulary width.  Silently
        # truncating it would select a different test-cache tag than the one
        # the pre-registration names.
        if isinstance(value, (bool, np.bool_)):
            raise SystemExit(
                f'[v12] REFUSE: {label}={value!r} is not a positive integer '
                'vocabulary width')
        if isinstance(value, numbers.Integral):
            width = int(value)
        else:
            try:
                f = float(value)
            except (TypeError, ValueError, OverflowError):
                raise SystemExit(
                    f'[v12] REFUSE: {label}={value!r} is not an integer '
                    'vocabulary width')
            if not math.isfinite(f) or f != math.floor(f):
                raise SystemExit(
                    f'[v12] REFUSE: {label}={value!r} is not an integer '
                    'vocabulary width')
            width = int(f)
        if width < 1:
            raise SystemExit(
                f'[v12] REFUSE: {label}={value!r} must be a positive '
                'vocabulary width')
        return width

    _bind = manifest.get('binding') or {}
    _sig = _bind.get('training_recipe_signature') if isinstance(_bind, dict) else None
    # `_recipe_signature` accepts the pre-v2 spelling `training_recipe` as an
    # alias of `training_recipe_signature`.  Vocabulary binding must use the
    # same accepted shapes; reading only the new key made a legacy manifest
    # fall through to the local cache's width (and then burn the one-shot test
    # access before every checkpoint was refused as a vocab mismatch).
    _legacy_sig = (_bind.get('training_recipe')
                   if isinstance(_bind, dict) else None)
    _vals = set()
    for _label, _recipe in (
            ('binding.training_recipe_signature.v', _sig),
            ('binding.training_recipe.v', _legacy_sig)):
        if isinstance(_recipe, dict) and _recipe.get('v') is not None:
            _vals.add(_width(_recipe['v'], _label))
    _per = (_bind.get('per_arm_recipe_signature')
            if isinstance(_bind, dict) else None)
    if isinstance(_per, dict):
        for _arm, _arm_sig in _per.items():
            if isinstance(_arm_sig, dict) and _arm_sig.get('v') is not None:
                _vals.add(_width(
                    _arm_sig['v'],
                    f'binding.per_arm_recipe_signature[{_arm!r}].v'))
    if isinstance(cfg, dict) and cfg.get('vocab_size') is not None:
        _vals.add(_width(cfg['vocab_size'], 'cfg.vocab_size'))
    if len(_vals) > 1:
        raise SystemExit(
            f'[v12] REFUSE: the frozen plan declares conflicting vocabulary '
            f'widths {sorted(_vals)}; the held-out cache tag cannot be chosen '
            'without guessing')
    return next(iter(_vals)) if _vals else None


def _current_train_cache_fps(cache_dir=None):
    """Training-stream fingerprints declared by the LOCAL train caches.

    ``load_wikitest`` binds the test ids to whatever train cache is present,
    but the frozen plan binds them to the corpus the checkpoints actually saw
    (``data_fp`` / ``train_fp``).  Without joining the two, a rebuilt train
    cache with a different id stream could pair a fresh test split with
    checkpoints trained on the old corpus.
    """
    import glob as _glob
    d = L._wt_cache_dir(cache_dir)
    fps = set()
    for mp in _glob.glob(os.path.join(d, 'meta_v5_*.json')):
        try:
            m = json.load(open(mp, encoding='utf-8'))
        except Exception:
            continue
        if not isinstance(m, dict):
            continue
        fp = m.get('src_fp') or m.get('train_fp')
        if isinstance(fp, str) and fp:
            fps.add(fp)
    return fps


def _require_current_train_cache_fp(want, cache_dir=None):
    """Refuse when the current train cache is not the frozen training stream."""
    fps = _current_train_cache_fps(cache_dir)
    if not fps:
        raise SystemExit(
            'REFUSE: no train-cache meta with a source fingerprint is present '
            '— the test split cannot be bound to the frozen training corpus')
    if str(want) not in fps:
        raise SystemExit(
            f'REFUSE: the current train cache source fingerprint(s) {sorted(fps)} '
            f'do not include the frozen training fingerprint {want} — the test '
            'split would be paired with checkpoints trained on a different '
            'token stream; rebuild/restore the frozen cache before P5T')
    return True


def _frozen_implementation(manifest):
    """Implementation-mode bindings declared by the frozen plan, if any.

    `code_semantics_expected` only pins the source revision; the eager
    row-loop vs batched twin and the AMP dtype are machine/runtime choices.
    Newer manifests can pin them with ``eager_batch_expected`` /
    ``amp_dtype_expected`` (top level, or the same names without the suffix).
    Absent fields leave the permissive path unchanged, but when a
    field IS declared every admitted record must prove it ran that way through
    its own `train_knobs` (never by re-deriving the current box's CUDA state).
    """
    impl = {}
    for _k in ('eager_batch', 'amp_dtype', 'amp_grad_scaler'):
        _v = manifest.get(_k + '_expected', manifest.get(_k))
        if _v is not None:
            impl[_k] = _v
    return impl or None


def _implementation_reason(rec, impl):
    """Why ``rec`` fails the frozen implementation binding (None = passes)."""
    if not impl:
        return None
    tk = rec.get('train_knobs')
    if not isinstance(tk, dict):
        return 'record lacks train_knobs required by the frozen implementation binding'
    for _k, _want in impl.items():
        if _k not in tk:
            return f"train_knobs lacks {_k!r} required by the frozen implementation binding"
        _got = tk.get(_k)
        if _k == 'eager_batch':
            if bool(_got) != bool(_want):
                return f"train_knobs.eager_batch={_got!r} does not match the frozen {bool(_want)!r}"
        elif _k == 'amp_dtype':
            _g = _got or 'fp32'
            _w = _want or 'fp32'
            if str(_g).lower() != str(_w).lower():
                return f"train_knobs.amp_dtype={_got!r} does not match the frozen {_want!r}"
        elif bool(_got) != bool(_want):
            return f"train_knobs.{_k}={_got!r} does not match the frozen {bool(_want)!r}"
    return None


# The length-checkpoint writer (explib.rope.run_lenphase_cell — used by P3MT /
# P4MT / P5C, the phases whose checkpoints P5T evaluates) records a bespoke
# `p1l_...` recipe, NOT a run_cfg fingerprint.  It carries the same training
# knobs under different names (`tl` = train length, `v` = vocab, ...).  The
# architecture there is hard-wired (256 / 6L / 8H / 32Dh) and there is no
# in-training eval (val_batch=None), so `ee`/`es` are not part of that recipe.
_P1L_RECIPE_RE = re.compile(
    r'^p1l_steps(?P<steps>[^_]+)_tl(?P<sl>[^_]+)_v(?P<v>[^_]+)_bs(?P<bs>[^_]+)'
    r'_lr(?P<lr>[^_]+)_nt(?P<nt>[^_]+)_wu(?P<wu>[^_]+)_wd(?P<wd>[^_]+)'
    r'_cl(?P<cl>[^_]+)_dlm(?P<dlm>[^_]+)_mr(?P<mr>[^_-]+)')
_P1L_ARCH = {'d': '256', 'L': '6', 'H': '8', 'Dh': '32'}


def _recipe_knobs(recipe):
    """Decode either a `_run_fp` fingerprint or the P1L length-checkpoint
    recipe into the named knob dict.  `{}` means neither format."""
    knobs = L.run_cfg_knobs(recipe)
    if knobs:
        return knobs
    m = _P1L_RECIPE_RE.match(recipe) if isinstance(recipe, str) else None
    if not m:
        return {}
    out = dict(m.groupdict())
    out.update(_P1L_ARCH)
    return out


def _recipe_matches_signature(recipe, sig):
    """Check a run_cfg/recipe string against the frozen training-recipe
    signature (the stats entry must verify WHAT produced the
    records, not just that a record exists).

    sig 只声明要钉死的字段（键名与 run_cfg_knobs 的解码一致：steps/sl/bs/
    nt/lr/wd/d/L/H/Dh/v/wu/cl/dlm/ee/es）；mt/mr/det/df/cs 与变体/环境/数据
    绑定，不由签名钉死。recipe 不可解码或任一声明字段不匹配 → False。
    长度 checkpoint 的 `p1l_` 配方按同一组语义键解码（见 `_recipe_knobs`）；
    该路径不运行训练中验证，故 `ee`/`es`（只影响中间评测、不影响权重）不
    由它承载，不因此拒绝整条记录。"""
    if not isinstance(sig, dict) or not sig:
        return True
    if not isinstance(recipe, str):
        return False
    knobs = _recipe_knobs(recipe)
    if not isinstance(knobs, dict) or not knobs:
        return False
    _is_p1l = recipe.startswith('p1l_')
    for k, want in sig.items():
        got = knobs.get(k)
        if got is None:
            if _is_p1l and k in ('ee', 'es'):
                continue
            return False
        try:
            # Recipes format floats with limited precision (e.g. mr{:.8g});
            # compare numerically with a tolerance far tighter than any real
            # protocol difference but wider than the formatting round-trip.
            if not math.isclose(float(got), float(want), rel_tol=1e-6, abs_tol=1e-9):
                return False
        except (TypeError, ValueError, OverflowError):
            if str(got) != str(want):
                return False
    return True


def _frozen_seeds():
    """Seeds for the frozen-arm training phases (P5C/P5MP) — single source of
    truth is the manifest; hard-coded phase lists must not disagree with it.

    Run the FULL `_load_manifest` validation here, not just a JSON
    seed parse.  Training refuses before the first GPU-second if the manifest
    would be rejected at P5T load time (stamp mismatch, unresolvable power
    gate, binding inconsistency).
    """
    m = _load_manifest(FREEZE_MANIFEST)
    seeds = m.get('seeds')
    if not isinstance(seeds, list) or not seeds:
        raise SystemExit(f'[v12] REFUSE: {FREEZE_MANIFEST} has no usable seed list')
    return [int(s) for s in seeds]


_LOCK_NAME = '.freeze_lock.json'


def _freeze_lock_path(outdir):
    return os.path.join(outdir, _LOCK_NAME)


def _source_tree_for_lock(source_tree=None):
    """Return the measurement-source manifest bound by the freeze lock.

    ``source_tree`` is injectable so a test can exercise the drift guard
    without editing the working tree.  A caller-supplied manifest is validated
    with the same contract as a freshly computed one: a lock without a usable
    digest would attest nothing.
    """
    st = source_tree if source_tree is not None else L.source_tree_manifest()
    if not isinstance(st, dict) or not isinstance(st.get('sha256'), str) or not st['sha256']:
        raise SystemExit('[v12] REFUSE: cannot compute a source-tree digest for the freeze lock — the code that produced the test measurement would be unattested')
    if st.get('unreadable'):
        raise SystemExit(f'[v12] REFUSE: the source tree contains unreadable files ({st["unreadable"][:3]}) — fix the permissions or exclude the path explicitly before freezing; an incomplete source digest cannot attest the measurement code')
    return st


def _source_tree_delta(locked, current, limit=8):
    """Human-readable added/removed/changed source files (bounded)."""
    lf = locked.get('files') if isinstance(locked, dict) else None
    cf = current.get('files') if isinstance(current, dict) else None
    if not isinstance(lf, dict) or not isinstance(cf, dict):
        return {'added': [], 'removed': [], 'changed': []}
    def _h(entry):
        return entry.get('sha256') if isinstance(entry, dict) else None
    added = sorted(set(cf) - set(lf))
    removed = sorted(set(lf) - set(cf))
    changed = sorted(k for k in set(cf) & set(lf) if _h(cf[k]) != _h(lf[k]))
    return {'added': added[:limit], 'removed': removed[:limit],
            'changed': changed[:limit]}


def _check_or_write_freeze_lock(outdir, manifest, test_ids_fp, stage='access',
                                train_fp=None, source_tree=None):
    """Serialize the freeze-lock check-and-write across processes.

    The check (read the existing lock) and the write must be one critical
    section: two concurrent P5T/analysis processes could otherwise both see
    ``no lock`` and replace each other's manifest/test binding.  The guard is a
    separate stable file so the lock artifact itself is never the target of the
    read-modify-write.
    """
    os.makedirs(outdir, exist_ok=True)
    guard = _freeze_lock_path(outdir) + '.guard'
    with open(guard, 'a+', encoding='utf-8') as _gf:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-POSIX fallback
            fcntl = None
        try:
            if fcntl is not None:
                fcntl.flock(_gf.fileno(), fcntl.LOCK_EX)
            return _check_or_write_freeze_lock_locked(
                outdir, manifest, test_ids_fp, stage=stage, train_fp=train_fp,
                source_tree=source_tree)
        finally:
            if fcntl is not None:
                fcntl.flock(_gf.fileno(), fcntl.LOCK_UN)


def _check_or_write_freeze_lock_locked(outdir, manifest, test_ids_fp, stage='access',
                                       train_fp=None, source_tree=None):
    """Persistent first-test-access lock.

    首次测试访问后禁止修改计划：P5T 一旦在 outdir 落下首条记录（或本函数
    被调用），锁文件即记录 manifest sha256
    与 test 切片指纹。之后的 P5T/analysis 若带着**不同的** manifest 哈希
    或不同 test 指纹运行，将被拒绝——除非显式 V12_REFREEZE=1（并大声
    打印该轮降级为探索性的警告）。``train_fp`` 一旦记录，后续训练集
    指纹不一致同样拒绝。已经 accessed 的锁不再接受第二次 ``preaccess``：
    held-out split 是一次性评估资源，重读会把它变成隐式验证集；显式重跑
    必须同时设置 V12_REFREEZE=1 与 V12_ALLOW_TEST_REREAD=1，锁会记录
    ``test_reads`` 与 history。"""
    lp = _freeze_lock_path(outdir)
    _sha = manifest.get('manifest_sha256')
    _src = _source_tree_for_lock(source_tree)
    _src_sha = _src['sha256']
    if os.path.exists(lp):
        try:
            lock = json.load(open(lp, encoding='utf-8'))
        except Exception as e:
            raise SystemExit(f'[v12] REFUSE: freeze lock {lp} exists but cannot be parsed ({type(e).__name__}: {e}) — delete it only if you can prove no test-split record was ever written under it')
        if not isinstance(lock, dict):
            # A valid-JSON list/scalar has no lock fields.  Reading it with
            # .get() used to raise AttributeError after the first test access,
            # which is exactly when this integrity check must fail closed.
            raise SystemExit(
                f'[v12] REFUSE: freeze lock {lp} is a {type(lock).__name__}, '
                'not a JSON object — it cannot attest the frozen plan/test '
                'slice; delete it only if you can prove no test-split record '
                'was ever written under it')
        _manifest_changed = lock.get('manifest_sha256') != _sha
        _test_changed = (test_ids_fp is not None
                         and lock.get('test_ids_fp') not in (None, test_ids_fp))
        # Once the lock carries a training-corpus fingerprint, a later call
        # with no fingerprint is a provenance downgrade, not a match.  The old
        # condition made the mismatch disappear exactly when train_fp=None.
        _train_changed = (lock.get('train_fp') is not None
                          and train_fp != lock.get('train_fp'))
        # The manifest hash pins the plan, but not the bytes of the code that
        # turns it into measurements.  Bind the measurement-source digest on
        # first write and refuse a later test evaluation under a different
        # tree (the exact hazard the continuous-loop header warns about: an
        # edit can change semantics while CODE_SEMANTICS still matches).
        _locked_src = lock.get('source_tree')
        _locked_src_sha = (_locked_src.get('sha256')
                           if isinstance(_locked_src, dict) else None)
        _source_legacy = _locked_src_sha is None
        _source_changed = bool(_locked_src_sha and _locked_src_sha != _src_sha)
        # Initialize the write flag here, BEFORE the disclosed-drift branch
        # can set it.  It used to be reset unconditionally after that branch,
        # which discarded the event just recorded and left the on-disk lock
        # looking byte-identical to a clean run.
        _dirty = False
        _refreeze = os.environ.get('V12_REFREEZE', '').strip().lower() in ('1', 'true', 'yes', 'on')
        _source_override = (os.environ.get('V12_ALLOW_SOURCE_DRIFT', '').strip().lower()
                            in ('1', 'true', 'yes', 'on'))
        _allow_reread = os.environ.get('V12_ALLOW_TEST_REREAD', '').strip().lower() in ('1', 'true', 'yes', 'on')
        _reread_reason = os.environ.get('V12_TEST_REREAD_REASON', '').strip()
        _already_accessed = (str(lock.get('stage') or '') == 'accessed'
                             or lock.get('test_ids_fp') is not None)
        _forced_reread = False
        if stage == 'preaccess' and _already_accessed:
            # The held-out split is a one-shot evaluation resource.  A second
            # P5T run would re-score the same frozen slice, turning it into an
            # implicit validation set (new checkpoints/configs can be selected
            # by inspecting the change).  Resume a completed panel from its
            # records; a genuinely new round must state V12_REFREEZE=1 AND
            # V12_ALLOW_TEST_REREAD=1 explicitly.
            if not (_refreeze and _allow_reread):
                raise SystemExit(
                    f'[v12] REFUSE: this results directory already has a '
                    f'test-split access lock ({lp}, stage=accessed, '
                    f'test_ids_fp={str(lock.get("test_ids_fp"))[:12]}…). '
                    'The frozen held-out split may be read/evaluated only once; '
                    'a second evaluation makes it an implicit validation set. '
                    'Re-run the analysis from the recorded artifacts, or set '
                    'V12_REFREEZE=1 and V12_ALLOW_TEST_REREAD=1 for a consciously '
                    'declared new round (the prior round is then exploratory).')
            _forced_reread = True
        if _source_changed and not _refreeze:
            if _source_override and stage == 'analysis':
                _delta = _source_tree_delta(_locked_src, _src)
                print(f'[v12] WARNING: V12_ALLOW_SOURCE_DRIFT=1 — the analysis '
                      f'source differs from the tree bound at first test access '
                      f'({str(_locked_src_sha)[:12]}… -> {_src_sha[:12]}…; '
                      f'changed={_delta["changed"]} added={_delta["added"]} '
                      f'removed={_delta["removed"]}).  The lock keeps the '
                      'original attestation; this run is DISCLOSED DRIFT, not '
                      'a byte-identical replay, and the source gate will fail.')
                lock['source_tree_drift'] = list(lock.get('source_tree_drift') or []) + [{
                    'at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    'stage': stage,
                    'locked_sha256': _locked_src_sha,
                    'current_sha256': _src_sha,
                    'delta': _delta,
                    'code_semantics': L.code_stamp(),
                }]
                _dirty = True
            else:
                _delta = _source_tree_delta(_locked_src, _src)
                _hint = ('set V12_ALLOW_SOURCE_DRIFT=1 only to re-run the '
                         'ANALYSIS with a disclosed code difference' if stage == 'analysis'
                         else 'restore the tree that was frozen, or consciously set V12_REFREEZE=1 (which downgrades the earlier round)')
                raise SystemExit(
                    f'[v12] REFUSE: the measurement source changed after the '
                    f'first test access (lock {lp}: {str(_locked_src_sha)[:12]}… '
                    f'!= current {_src_sha[:12]}…; changed='
                    f'{_delta["changed"]} added={_delta["added"]} removed='
                    f'{_delta["removed"]}).  The frozen manifest binds the plan '
                    'and CODE_SEMANTICS stamps the semantics constant, but '
                    'neither hashes the code bytes; a dirty edit can therefore '
                    'change a measurement while both still match.  ' + _hint)
        if (_manifest_changed or _test_changed or _train_changed) and not _refreeze:
            if _manifest_changed:
                raise SystemExit(f'[v12] REFUSE: the test split was already accessed under a DIFFERENT frozen manifest (lock {lp}: {lock.get("manifest_sha256")} != current {_sha}, locked_at={lock.get("locked_at")}). Changing the plan after seeing test data is selection bias by construction. To consciously re-freeze, set V12_REFREEZE=1 — the earlier round must then be reported as exploratory in the plan history.')
            if _test_changed:
                raise SystemExit(f'[v12] REFUSE: the held-out test slice changed after the first access (lock has ids_fp {lock.get("test_ids_fp")}, current {test_ids_fp}) — the test data itself is part of the freeze; investigate before proceeding (V12_REFREEZE=1 consciously re-freezes to a new slice)')
            raise SystemExit(f'[v12] REFUSE: the training corpus changed after the first test access (lock has train_fp {lock.get("train_fp")}, current {train_fp}) — a different training set cannot enter the frozen confirmation (V12_REFREEZE=1 consciously re-freezes to a new corpus)')
        if _forced_reread:
            print(f'[v12] WARNING: V12_ALLOW_TEST_REREAD=1 — this is an '
                  f'ADDITIONAL read of the frozen held-out split (previous '
                  f'reads: {lock.get("test_reads", 1)}).  Re-evaluation can '
                  'turn test into an implicit validation set; record why the '
                  'new round is necessary and whether the decision rule changed.')
            try:
                _reread_policy = L.test_access_policy(manifest)
            except ValueError:
                _reread_policy = {'require_reason_for_reread': True}
            if (_reread_policy.get('require_reason_for_reread', True)
                    and not _reread_reason):
                print('[v12] WARNING: no V12_TEST_REREAD_REASON was supplied; '
                      'the reread history will carry reason=None and the '
                      'test-access audit will report an unexplained extra look.')
            lock['test_reads'] = int(lock.get('test_reads', 1)) + 1
            lock['history'] = list(lock.get('history') or []) + [{
                'stage': 'reread', 'at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'test_reads': lock['test_reads'],
                'code_semantics': L.code_stamp(),
                'reason': _reread_reason or None,
            }]
            _dirty = True
        if (_manifest_changed or _test_changed or _train_changed or _source_changed) and _refreeze:
            print(f'[v12] WARNING: V12_REFREEZE=1 — the locked plan/test/train/source identity changed (manifest {lock.get("manifest_sha256")} -> {_sha}, test_ids_fp {lock.get("test_ids_fp")} -> {test_ids_fp}, train_fp {lock.get("train_fp")} -> {train_fp}, source {str(_locked_src_sha)[:12]}… -> {_src_sha[:12]}…); the earlier round is DOWNGRADED to exploratory and this must be recorded in the plan history')
            # Persist the conscious re-freeze: under the new plan the lock must
            # carry the new identity so later runs do not refuse, and the
            # superseded plan is retained for the exploratory record.  The lock
            # is written ONCE below, after every check has passed, so a failure
            # can never leave a half-updated (new plan + stale fingerprints) file.
            prev = {_k: lock.get(_k) for _k in ('manifest_sha256', 'test_ids_fp', 'train_fp', 'locked_at', 'frozen_at')}
            prev['source_tree_sha256'] = _locked_src_sha
            lock['superseded'] = list(lock.get('superseded') or []) + [prev]
            if _source_changed:
                lock['history'] = list(lock.get('history') or []) + [{
                    'stage': 'refreeze-source-tree', 'at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    'source_tree_sha256': _src_sha,
                    'superseded_source_tree_sha256': _locked_src_sha,
                    'delta': _source_tree_delta(_locked_src, _src)}]
            lock.update({'manifest_sha256': _sha,
                         'manifest_path': os.path.abspath(manifest.get('_path') or FREEZE_MANIFEST),
                         'frozen_at': manifest.get('frozen_at'),
                         'locked_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                         'source_tree': _src,
                         'source_tree_sha256': _src_sha,
                         'source_tree_bound_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                         'code_semantics': L.CODE_SEMANTICS,
                         'note': 're-frozen under V12_REFREEZE=1: the previous plan is downgraded to exploratory (record it in the plan history)'})
            if _test_changed:
                lock['test_ids_fp'] = test_ids_fp
                lock['stage'] = 'accessed'
                lock['history'] = list(lock.get('history') or []) + [{'stage': 'refreeze-test-fp', 'at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'test_ids_fp': test_ids_fp, 'superseded_test_ids_fp': prev.get('test_ids_fp')}]
            if _train_changed:
                lock['train_fp'] = train_fp
                lock['history'] = list(lock.get('history') or []) + [{'stage': 'refreeze-train-fp', 'at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'train_fp': train_fp, 'superseded_train_fp': prev.get('train_fp')}]
            _dirty = True
        if test_ids_fp is not None and lock.get('test_ids_fp') is None:
            # two-phase lock: the pre-access record was written before the
            # first read; bind the exact slice immediately after it.
            lock['test_ids_fp'] = test_ids_fp
            lock['stage'] = 'accessed'
            lock['test_reads'] = max(1, int(lock.get('test_reads', 0)))
            lock['history'] = list(lock.get('history') or []) + [{'stage': 'accessed', 'at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'test_ids_fp': test_ids_fp}]
            _dirty = True
        if train_fp is not None and lock.get('train_fp') is None:
            lock['train_fp'] = train_fp
            lock['history'] = list(lock.get('history') or []) + [{'stage': 'train-fp', 'at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'train_fp': train_fp}]
            _dirty = True
        if _source_legacy and str(lock.get('stage') or '') == 'preaccess' and test_ids_fp is None:
            # A pre-access lock written by an older driver has no source
            # binding.  No test read has happened yet, so binding the current
            # tree here is not a retrospective attestation.  An already
            # accessed legacy lock stays visibly unbound (source_tree absent),
            # and the source audit reports not_verifiable instead of "clean".
            lock['source_tree'] = _src
            lock['source_tree_sha256'] = _src_sha
            lock['source_tree_bound_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
            lock['history'] = list(lock.get('history') or []) + [{
                'stage': 'source-tree-bind', 'at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'source_tree_sha256': _src_sha}]
            _dirty = True
        if _dirty:
            L.atomic_write_json(lp, lock, indent=2)
            if _manifest_changed or _test_changed or _train_changed or _source_changed:
                # A disclosed analysis-only drift is persisted without a
                # re-freeze, so `superseded` need not exist (it is created only
                # by V12_REFREEZE).  Indexing it unconditionally crashed after
                # the event had already been written.
                _n_sup = len(lock.get('superseded') or [])
                print(f'[v12] freeze lock UPDATED: {lp} (manifest '
                      f'{str(_sha)[:12]}…'
                      + (f'; {_n_sup} superseded plan(s) kept' if _n_sup else '')
                      + ')')
            else:
                print(f'[v12] freeze lock updated with the first-access slice/train fingerprint: {lp}')
        return lock
    lock = {'manifest_sha256': _sha, 'manifest_path': os.path.abspath(manifest.get('_path') or FREEZE_MANIFEST), 'frozen_at': manifest.get('frozen_at'), 'locked_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'test_ids_fp': test_ids_fp, 'train_fp': train_fp, 'stage': stage, 'test_reads': (0 if test_ids_fp is None else 1), 'source_tree': _src, 'source_tree_sha256': _src_sha, 'source_tree_bound_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'code_semantics': L.code_stamp(), 'history': [{'stage': stage, 'at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'test_ids_fp': test_ids_fp, 'train_fp': train_fp, 'source_tree_sha256': _src_sha}], 'note': 'first test-split access: the plan, test slice and measurement source may not change after this point (V12_REFREEZE=1 overrides consciously, downgrading the earlier round to exploratory)'}
    L.atomic_write_json(lp, lock, indent=2)
    print(f'[v12] freeze lock written: {lp} (manifest {_sha[:12]}…, stage {stage}, test_ids_fp {str(test_ids_fp)[:12]}…)')
    return lock

def _training_health_gate(cfg, v, seed, ckpt_meta, train_rec):
    """Refuse the one-shot test evaluation when training health is unknown or bad.

    This is deliberately NOT a model-selection rule: it does not change arms,
    seeds, steps or the endpoint.  It only prevents burning the frozen test
    access on a run whose own diagnostic curve says non-finite/overfitting/
    stalled.  `V12_ALLOW_UNHEALTHY=1` is the explicit, logged override.
    """
    if cfg.get('smoke'):
        return None
    health = (ckpt_meta or {}).get('curve_health') or (train_rec or {}).get('curve_health')
    allow = os.environ.get('V12_ALLOW_UNHEALTHY', '').strip().lower() in ('1', 'true', 'yes', 'on')
    if not isinstance(health, dict):
        msg = f'{v} s{seed}: no training curve/health evidence in the checkpoint or its record'
        if allow:
            print(f'[p5t] WARNING: {msg}; V12_ALLOW_UNHEALTHY=1 permits the test evaluation')
            return None
        return msg + ' — refusing to use the one-shot test access (set V12_ALLOW_UNHEALTHY=1 only with a written justification)'
    triage = (ckpt_meta or {}).get('instability_diagnosis') or (train_rec or {}).get('instability_diagnosis')
    if isinstance(triage, dict) and triage.get('status') == 'instability_observed':
        msg = (f'{v} s{seed}: non-finite training event recorded '
               f'(primary suspect {triage.get("primary_suspect")!r}: {triage.get("summary")})')
        if allow:
            print(f'[p5t] WARNING: {msg}; V12_ALLOW_UNHEALTHY=1 permits the test evaluation')
            return None
        return msg + ' — refusing to use the one-shot test access; diagnose the first bad step first'
    status = str(health.get('status') or 'unknown')
    if not _health_blocks(status):
        # `healthy` and the declared non-blocking plateau statuses both clear
        # the test gate.  A flat cosine tail is recorded, not treated as a
        # broken run; the status still reaches the P5H report and the cell row.
        if status != 'healthy':
            print(f'[p5t] NOTE: {v} s{seed}: training health is {status!r} '
                  f'({health.get("reason") or "no reason recorded"}) — plateau/converged, non-blocking')
        return None
    msg = f'{v} s{seed}: training health is {status!r} ({health.get("reason") or "no reason recorded"})'
    if allow:
        print(f'[p5t] WARNING: {msg}; V12_ALLOW_UNHEALTHY=1 permits the test evaluation')
        return None
    return msg + ' — refusing to use the one-shot test access'


def _finaltest_admission_problem(stats, manifest, *, smoke=False,
                                  allow_partial=False):
    """Return why a P5T pass cannot be reported as completed, or None.

    Running every evaluation call is not evidence: a stale/unreadable
    checkpoint, a missing training provenance record or the health gate can
    make every cell return ``skipped_*`` while the phase still exits normally.
    The one-shot test split has then been consumed with zero admitted cells,
    so the driver must not append P5T to its completed-phase list.
    """
    if smoke:
        return None
    admission = stats.get('admission') if isinstance(stats, dict) else None
    if not isinstance(admission, dict):
        return 'P5T produced no admission report'
    _raw_admitted = admission.get('n_admitted')
    if isinstance(_raw_admitted, (bool, np.bool_)):
        return f'P5T admission count is malformed: {_raw_admitted!r}'
    # Reuse the record-contract reader: int(2.5) == 2 would silently turn a
    # malformed count into a partial-but-plausible panel, and int(True) == 1
    # would turn a flag into an admitted cell.  `_record_int` rejects both.
    n_admitted = _record_int(_raw_admitted)
    if n_admitted is None:
        return f'P5T admission count is malformed: {_raw_admitted!r}'
    if n_admitted < 0:
        return f'P5T admission count is negative: {n_admitted!r}'
    n_expected = (len(manifest.get('arms') or [])
                  * len(manifest.get('seeds') or []))
    if n_expected <= 0:
        return 'the frozen manifest declares no (arm, seed) cells'
    if n_admitted <= 0:
        return (f'P5T admitted 0/{n_expected} frozen (arm, seed) cells — the '
                'one-shot test split produced no confirmatory evidence')
    if n_admitted < n_expected and not allow_partial:
        return (f'P5T admitted only {n_admitted}/{n_expected} frozen '
                '(arm, seed) cells; set V12_ALLOW_PARTIAL_TEST=1 only for an '
                'explicitly incomplete round')
    return None


def _resolve_test_article_ids(test_ids, test_doc_ids, *, n_seqs, seq_len,
                              vocab_size):
    """Recover article clusters for the frozen test slice, or explain why not.

    The test cache's per-token map groups raw WikiText LINES.  A line-level
    bootstrap resamples paragraphs of one article and understates uncertainty;
    the article is the independent unit.  Reconstruction decodes the already
    loaded line runs with the persisted test tokenizer, so it does NOT read
    ``wiki.test.raw`` again and does not rewrite a cache byte.

    Returns ``(article_ids_or_None, diagnostics)``.  Any failure is an
    explicit fallback reason, never a guessed article boundary.
    """
    diag = {'status': 'not_attempted', 'unit': 'article',
            'source': 'decoded_line_runs'}
    if test_doc_ids is None:
        diag.update(status='no_line_mapping',
                    reason='the test cache has no per-token line map')
        return None, diag
    if vocab_size is None:
        diag.update(status='no_vocab_binding',
                    reason='the frozen test vocabulary width is unknown')
        return None, diag
    try:
        tok = L.load_wikitext_test_tokenizer(
            n_seqs=int(n_seqs), seq_len=int(seq_len),
            vocab_size=int(vocab_size))
    except Exception as e:
        diag.update(status='tokenizer_unreadable',
                    reason=f'{type(e).__name__}: {e}')
        return None, diag
    if tok is None:
        diag.update(status='tokenizer_missing',
                    reason=('the test cache has no persisted tokenizer; '
                            'legacy caches need a rebuild for the article '
                            'unit'))
        return None, diag
    rep = L.wikitext_token_article_ids_from_line_map(
        test_ids, test_doc_ids, tok)
    for _k, _v in rep.items():
        if _k not in ('token_article_ids', 'line_article_ids', 'line_ids'):
            diag[_k] = _v
    if rep.get('status') != 'ok':
        diag.setdefault('reason', 'article grouping did not complete')
        return None, diag
    if not rep.get('valid_for_clustering'):
        diag.update(status='no_top_level_heading',
                    reason=('the decoded slice has no evidenced top-level '
                            '" = Title = " heading; article boundaries are '
                            'not identifiable'))
        return None, diag
    diag['status'] = 'ok'
    return rep['token_article_ids'], diag
