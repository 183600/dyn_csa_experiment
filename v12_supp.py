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
  python v12_supp.py smoke            # 端到端自检（微型 ckpt + 冻结清单副本）
  python v12_supp.py plan             # 零 GPU 预检：戳/实现绑定/准入/功效/预算台账/证据等级，不读 test；
                                      # 通过后自动把快照留档到 analysis_v12/plan_snapshot.json
  python v12_supp.py dynamic-plan     # 零 GPU 预检动态/HCA 草稿：假设、专用对照、token/参数/读取预算、
                                      # disjoint-source/校准/功效前置；通过也只说明设计自洽，不是确认
  python v12_supp.py power            # 零 GPU：用 P5C validation 差值做精确条件功效（同样不读 test）
  python v12_supp.py full             # P5C -> P5H -> P5PW -> P5T -> P5MP -> P5S2/P5S3 -> report
  python v12_supp.py auto             # 无人值守：与 full 同链，但把「plan 预检通过」的
                                      # 全部机器可查证据（清单哈希/代码戳/git 树/四个人工
                                      # 核对字段：arms/seeds/主比较/α）自动落盘为审计快照，
                                      # 人工核对变为事后异步，不再阻塞链的启动
  python v12_supp.py phase P5C        # 单阶段（可断点续跑）
  python v12_supp.py phase P5PW       # 仅功效预检
  python v12_supp.py analysis         # 零 GPU：确认性统计 -> analysis_v12/
  python v12_supp.py report           # 从落盘产物重建 REPORT_v12.md（零 GPU）

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
import v10_supp as V10
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
            _ctx = dict(_ctx, module='v12_supp', ckpt_install=True, ckpt_flag=False)
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


# The length-checkpoint writer (v7_supp.run_lenphase_cell — used by P3MT /
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
            lock['test_reads'] = int(lock.get('test_reads', 1)) + 1
            lock['history'] = list(lock.get('history') or []) + [{
                'stage': 'reread', 'at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'test_reads': lock['test_reads'],
                'code_semantics': L.code_stamp(),
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

def run_finaltest(cfg=None, guard=None):
    """Evaluate the FROZEN confirmation set on the held-out test split.

    只读冻结清单列出的臂/种子/终点；评估数据来自 `L.load_wikitest`（test
    split，其它 phase 不读）。产物：`results_lm_test/bylen.json`（干净
    by-length，含逐文档 ppl_per_seq）与 `results_lm_test/distractor.json`
    （uniform 腐坏探针）；确认性统计由 `v12_supp.py analysis` 从这些产物
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
    for v in manifest['arms']:
        for seed in manifest['seeds']:
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
    for v in pcfg['variants']:
        for seed in pcfg['seeds']:
            V10.run_probe_cell(pcfg, v, seed, guard=guard, val_ids=test_ids, mech_summ=mech_summ)
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
    mean = float(np.mean(deltas))
    seed_ci = None
    if n > 1:
        _sd = float(np.std(deltas, ddof=1))
        _sem = _sd / math.sqrt(n)
        _tc = L._t_critical(0.95, n - 1)
        if math.isfinite(_tc) and math.isfinite(_sem):
            seed_ci = [mean - _tc * _sem, mean + _tc * _sem]
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
         'seed', 'ppl_delta', float(np.median(deltas)))
    if n >= 5:
        _k = int(math.floor(0.2 * n))
        _trimmed = sorted(deltas)[_k:n - _k]
        _add('seed_trimmed_mean_ppl_delta',
             '20% trimmed mean of per-seed token-pooled PPL differences (a-b)',
             'seed', 'ppl_delta', float(np.mean(_trimmed)))
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


def _doc_level_analysis(recs_a, recs_b, common, Ln, primary_deltas=None):
    """Document-sampling readout for one comparison.

    逐文档配对差（对每个文档取「种子内臂差」的均值）与两层方差分解：
    between-doc（文档抽样不确定性）与 within-doc-across-seed（训练随机性）。
    48 条评测行仍不足以充分覆盖文档分布——本读数的目的就是把这个不确定性
    显式化，而不是宣称已消除。

    等权逐文档均值回答「平均文档是否更好」，主终点回答「token 池化 PPL
    是否更低」。文档长度不同时这是两个估计量；因此额外给出与主终点同口径
    的 token 池化 Δ 的文档 cluster-robust 区间（条件在已训练种子上），等权
    bootstrap 只保留为描述量。"""
    if not common:
        return None
    rows_a = [recs_a[s]['by_len'].get(Ln, recs_a[s]['by_len'].get(str(Ln))) or {} for s in common]
    rows_b = [recs_b[s]['by_len'].get(Ln, recs_b[s]['by_len'].get(str(Ln))) or {} for s in common]
    # A fixed eval ROW is an arbitrary window, not a document: it can split
    # one document over two rows (double-counting it) or merge two documents
    # into one row.  A WikiText "document" is itself one raw LINE, so even the
    # document bootstrap resamples paragraphs of the same article.  Prefer the
    # per-TOKEN ARTICLE aggregation when every record carries it (the
    # independent corpus unit); otherwise fall back to the line-level document
    # map, and only then to fixed rows.  Every fallback is named in the output.
    _art_a = [r.get('ppl_per_article') for r in rows_a]
    _art_b = [r.get('ppl_per_article') for r in rows_b]
    _has_art = [isinstance(x, list) and bool(x) for x in _art_a + _art_b]
    _doc_a = [r.get('ppl_per_doc') for r in rows_a]
    _doc_b = [r.get('ppl_per_doc') for r in rows_b]
    _has_doc = [isinstance(x, list) and bool(x) for x in _doc_a + _doc_b]
    _article_fallback = None
    _ref_ids = None
    if any(_has_art) and all(_has_art):
        _granularity = 'article'
        _ppl_a, _ppl_b = _art_a, _art_b
        _tok_key, _ids_key = 'n_tokens_per_article', 'article_ids'
    elif any(_has_art):
        if not (all(_has_doc) and any(_has_doc)):
            return {'n_docs': 0, 'n_seeds': len(common), 'comparable': False,
                    'reason': ('some records carry article-level cells and '
                               'others do not; the article and line units '
                               'cannot be pooled'),
                    'note': 'refused rather than mixing two estimands'}
        _article_fallback = ('article cells are missing for some arm/seed '
                             'records; reported at the line-level document '
                             'unit')
        _granularity = 'document'
        _ppl_a, _ppl_b = _doc_a, _doc_b
        _tok_key, _ids_key = 'n_tokens_per_doc', 'doc_ids'
    elif all(_has_doc) and any(_has_doc):
        _granularity = 'document'
        _ppl_a, _ppl_b = _doc_a, _doc_b
        _tok_key, _ids_key = 'n_tokens_per_doc', 'doc_ids'
    else:
        if any(_has_doc):
            return {'n_docs': 0, 'n_seeds': len(common), 'comparable': False,
                    'reason': ('some records carry document-level cells and '
                               'others do not; document- and row-level '
                               'uncertainty cannot be pooled'),
                    'note': 'refused rather than mixing two estimands'}
        _granularity = 'row'
        _ppl_a = [r.get('ppl_per_seq') for r in rows_a]
        _ppl_b = [r.get('ppl_per_seq') for r in rows_b]
        if not all(isinstance(x, list) and x for x in _ppl_a + _ppl_b):
            return None
    if _granularity in ('document', 'article'):
        _tok_a = [r.get(_tok_key) for r in rows_a]
        _tok_b = [r.get(_tok_key) for r in rows_b]
        if not all(isinstance(x, list) and x for x in _tok_a + _tok_b):
            return {'n_docs': 0, 'n_seeds': len(common), 'comparable': False,
                    'reason': (f'{_granularity}-level cells lack per-unit '
                               'token counts'),
                    'note': 'refused rather than treating unequal units as equal'}
        _lens = {len(x) for x in _ppl_a + _ppl_b + _tok_a + _tok_b}
        if len(_lens) != 1:
            return {'n_docs': int(min(_lens)), 'n_seeds': len(common),
                    'comparable': False,
                    'granularity': _granularity,
                    'reason': (f'{_granularity} counts differ across arm/seed '
                               'records'),
                    'row_counts': sorted(_lens),
                    'note': 'refused rather than silently truncating'}
        _ref_tok = list(_tok_a[0])
        if any(list(x) != _ref_tok for x in _tok_a + _tok_b):
            return {'n_docs': int(min(_lens)), 'n_seeds': len(common),
                    'comparable': False,
                    'granularity': _granularity,
                    'reason': (f'per-{_granularity} token counts differ across '
                               'arm/seed records; the units are not the same'),
                    'note': 'refused rather than pairing different units'}
        _ref_ids = rows_a[0].get(_ids_key)
        if _ref_ids is not None:
            if any(r.get(_ids_key) != _ref_ids for r in rows_a + rows_b):
                return {'n_docs': int(min(_lens)), 'n_seeds': len(common),
                        'comparable': False,
                        'granularity': _granularity,
                        'reason': (f'{_granularity} id order differs across '
                                   'arm/seed records; paired deltas would pair '
                                   'different units'),
                        'note': 'refused rather than pairing by position'}
        per_doc_a, per_doc_b = _ppl_a, _ppl_b
    else:
        per_doc_a, per_doc_b = _ppl_a, _ppl_b
    # Pair unit i across arms only when every record evaluated the SAME number
    # of units.  Silently taking the minimum would pair a truncated panel
    # with a full one and claim a larger sample than either scored.
    _lens = {len(x) for x in per_doc_a + per_doc_b}
    if len(_lens) != 1:
        return {'n_docs': int(min(_lens)), 'n_seeds': len(common), 'comparable': False,
                'granularity': _granularity,
                'reason': 'per-document row counts differ across arm/seed records; document-paired deltas are not defined',
                'row_counts': sorted(_lens), 'note': 'refused rather than silently truncating to the shortest record'}
    n_docs = min(_lens)
    if n_docs < 2:
        return None
    # Unit labels for the leave-one-unit-out influence readout.  The order was
    # already checked across arm/seed records above; only a unique, complete id
    # vector may label the folds, otherwise the readout falls back to indices
    # (and the report says so) rather than guessing which unit was dropped.
    _unit_ids = None
    if isinstance(_ref_ids, (list, tuple)) and len(_ref_ids) == n_docs:
        try:
            _ids_cand = [str(x) for x in _ref_ids]
            if len(set(_ids_cand)) == n_docs:
                _unit_ids = _ids_cand
        except (TypeError, ValueError):
            _unit_ids = None
    doc_deltas = []
    doc_log_ratios = []
    within = []
    incomplete = 0
    for i in range(n_docs):
        cell = []
        logs = []
        for k in range(len(common)):
            a, b = per_doc_a[k][i], per_doc_b[k][i]
            try:
                fa, fb = float(a), float(b)
            except (TypeError, ValueError, OverflowError):
                fa = fb = float('nan')
            if (not math.isfinite(fa)) or (not math.isfinite(fb)) or fa <= 0.0 or fb <= 0.0:
                incomplete += 1
                continue
            cell.append(fa - fb)
            logs.append(math.log(fa) - math.log(fb))
        if not cell:
            continue
        doc_deltas.append(float(np.mean(cell)))
        doc_log_ratios.append(float(np.mean(logs)))
        within.append(float(np.std(cell, ddof=1)) if len(cell) > 1 else 0.0)
    if len(doc_deltas) < 2:
        return None
    # The equal-weighted per-document mean above answers "is the average
    # document better?"; the primary endpoint asks "is the token-pooled PPL
    # lower?".  When document lengths differ these are different estimands, so
    # the interval shown next to the pooled Δ must resample/weight DOCUMENTS
    # with their token counts, not give every document equal weight.
    _pooled_ci = None
    _pooled_ci_reason = None
    _unit_influence = None
    _unit_influence_reason = None
    if _granularity in ('document', 'article'):
        _nll_a, _why_a = _doc_nll_matrix(rows_a, n_docs, _ref_tok,
                                         unit=_granularity)
        _nll_b, _why_b = _doc_nll_matrix(rows_b, n_docs, _ref_tok,
                                         unit=_granularity)
        if _nll_a is None or _nll_b is None:
            _pooled_ci_reason = (
                'token-pooled cluster CI not available: '
                + (_why_a or _why_b
                   or f'{_granularity} NLL vectors are unusable'))
        else:
            _pc = L._paired_nll_doc_ci(_nll_a, _nll_b, _ref_tok)
            if _pc is None:
                _pooled_ci_reason = (
                    'token-pooled cluster CI not available: the per-seed PPL '
                    'difference overflowed')
            else:
                # The interval is only meaningful if its point estimate is the
                # SAME per-seed pooled Δ the frozen comparison reports.
                _cell_a = [r.get('ppl') for r in rows_a]
                _cell_b = [r.get('ppl') for r in rows_b]
                _pt = []
                for _pa, _pb in zip(_cell_a, _cell_b):
                    try:
                        _fa, _fb = float(_pa), float(_pb)
                    except (TypeError, ValueError, OverflowError):
                        _pt = []
                        break
                    if (not math.isfinite(_fa)) or (not math.isfinite(_fb)):
                        _pt = []
                        break
                    _pt.append(_fa - _fb)
                if _pt:
                    _want = float(np.mean(_pt))
                    if abs(_pc['delta_ppl'] - _want) > 1e-5 * max(1.0, abs(_want)):
                        _pooled_ci_reason = (
                            f'token-pooled cluster CI refused: the '
                            f'{_granularity} vectors do not reproduce the '
                            'recorded per-seed pooled PPL difference')
                    else:
                        _pooled_ci = _pc
                        try:
                            _unit_influence = L.paired_nll_unit_influence(
                                _nll_a, _nll_b, _ref_tok, unit_ids=_unit_ids)
                        except (ValueError, TypeError, FloatingPointError,
                                OverflowError) as _ie:
                            _unit_influence = None
                            _unit_influence_reason = (
                                f'{type(_ie).__name__}: {_ie}')
                else:
                    # The frozen comparison has no finite per-seed pooled PPL
                    # for this cell, so the document vectors cannot be tied to
                    # the primary estimand.  Do not publish an interval whose
                    # point estimate could not be checked: `_pt` is empty for
                    # a missing/NaN `ppl`, not because the vectors are trusted.
                    _pooled_ci_reason = (
                        'token-pooled cluster CI refused: the recorded per-seed '
                        'pooled PPL is missing or non-finite, so the '
                        f'{_granularity} vectors cannot be tied to the primary '
                        'comparison')
    arr = np.asarray(doc_deltas, dtype=float)
    lr = np.asarray(doc_log_ratios, dtype=float)
    # Seed-level sign-flip remains the confirmatory test.  These intervals
    # condition on the trained seeds and quantify only DOCUMENT sampling:
    # with n_seqs=48 they are a property of the frozen slice, not of the
    # language population.  Resampling documents (not tokens) preserves the
    # within-document correlation.
    _ci_ppl = L.bootstrap_mean_ci(arr, seed=20261018)
    _ci_log = L.bootstrap_mean_ci(lr, seed=20261018)
    # Breadth diagnostics: a mean gain can be broad-based or carried by one
    # document.  Counts/fractions are descriptive (documents are not an
    # independent randomisation of the arm), but they expose an outlier-driven
    # "win" that a mean and a bootstrap CI can hide.
    _neg = int((arr < 0).sum())
    _pos = int((arr > 0).sum())
    _zero = int((arr == 0).sum())
    _abs_sum = float(np.abs(arr).sum())
    _top_share = (float(np.abs(arr).max() / _abs_sum) if _abs_sum > 0.0 else None)
    _unit_notes = {
        'article': ('article-sampling uncertainty only; seed-level exact '
                    'sign-flip remains the confirmatory test.  WikiText rows '
                    'are lines, so the article (top-level heading group) is '
                    'the independent corpus unit; article count is typically '
                    'far below the 64-unit population floor, so '
                    'population-level claims stay refused'),
        'document': ('line-level sampling uncertainty only; seed-level exact '
                     'sign-flip remains the confirmatory test.  A WikiText '
                     '"document" is one raw line, so an article spanning '
                     'several lines is pseudoreplicated here; report the '
                     'article unit when the cache/tokenizer can supply it '
                     '(design_notes section 54)'),
        'row': ('ROW-level uncertainty only: fixed eval rows can split or '
                'merge lines/articles, so this bootstrap is not a document '
                'bootstrap; rebuild/copy the test cache with per-token line '
                'ids and a byte-level tokenizer for the article-level readout'),
    }
    _ci_method = {
        'article': ('percentile bootstrap over ARTICLES (chunked, '
                    'deterministic); conditions on the trained seeds'),
        'document': ('percentile bootstrap over LINES (the cache calls them '
                     'documents; chunked, deterministic); conditions on the '
                     'trained seeds'),
        'row': ('percentile bootstrap over fixed eval ROWS (chunked, '
                'deterministic); rows are not documents'),
    }[_granularity]
    out = {'n_docs': int(len(doc_deltas)), 'n_units': int(len(doc_deltas)),
           'n_seeds': len(common),
           'granularity': _granularity,
           'unit': _granularity,
           'per_doc_delta_mean': float(arr.mean()),
           'per_doc_delta_median': float(np.median(arr)),
           'per_doc_log_ratio_mean': float(lr.mean()),
           'doc_delta_negative': _neg, 'doc_delta_positive': _pos,
           'doc_delta_zero': _zero,
           'doc_a_better_fraction': (float(_neg / arr.size) if arr.size else None),
           'top_doc_abs_share': _top_share,
           'between_doc_sd': float(arr.std(ddof=1)) if len(doc_deltas) > 1 else 0.0,
           'between_doc_log_ratio_sd': float(lr.std(ddof=1)) if len(doc_log_ratios) > 1 else 0.0,
           'mean_within_doc_seed_sd': float(np.mean(within)),
           'per_doc_delta_ppl_ci95': ([_ci_ppl['ci_low'], _ci_ppl['ci_high']] if _ci_ppl else None),
           'per_doc_log_ratio_ci95': ([_ci_log['ci_low'], _ci_log['ci_high']] if _ci_log else None),
           'incomplete_doc_seed_cells': int(incomplete),
           'ci_method': _ci_method,
           'breadth_note': ('doc_a_better_fraction counts units whose seed-mean Δ (a-b) is negative, i.e. lower PPL for arm a; '
                            'top_doc_abs_share is max|Δ_unit| / sum|Δ_unit| (a single unit carrying a large share means the '
                            'mean is outlier-sensitive).  Both are descriptive: units are not an arm randomisation.'),
           'article_fallback_reason': _article_fallback,
           'note': _unit_notes[_granularity]}
    if _granularity == 'article':
        out['n_articles'] = int(len(doc_deltas))
        _line_units = rows_a[0].get('ppl_per_doc') if rows_a else None
        out['n_line_docs'] = (len(_line_units)
                              if isinstance(_line_units, list) and _line_units
                              else None)
        if out['n_line_docs']:
            out['pseudoreplication_factor'] = (
                float(out['n_line_docs']) / float(out['n_articles']))
        if out['n_articles'] < 20:
            out['small_cluster_warning'] = (
                'fewer than 20 article clusters: the percentile interval is '
                'coarse and must be reported with its n; it is not a '
                'population interval')
    if _pooled_ci is not None:
        out.update({
            'pooled_delta_ppl': _pooled_ci['delta_ppl'],
            'pooled_delta_ppl_ci95': _pooled_ci['delta_ppl_ci95'],
            'pooled_delta_log_ratio': _pooled_ci['delta_log_ratio'],
            'pooled_delta_log_ratio_ci95': _pooled_ci['delta_log_ratio_ci95'],
            'pooled_ratio_ci95': _pooled_ci['ratio_ci95'],
            'pooled_ci_n_docs': _pooled_ci['n_docs'],
            'pooled_ci_n_units': _pooled_ci['n_docs'],
            'pooled_ci_unit': _granularity,
            'pooled_ci_df': _pooled_ci['df'],
            'pooled_ci_method': _pooled_ci['method'],
            'pooled_ci_note': ('interval for the token-pooled Δ, the same '
                               'estimand as the primary comparison; the '
                               f'{_granularity} units are clusters and the '
                               'trained seeds are conditioned on.  '
                               '`pooled_delta_log_ratio` is the seed-averaged '
                               'log PPL ratio, not the log of the ΔPPL.  This '
                               'is not a seed-level significance test.'),
        })
        if _unit_influence is not None:
            out['unit_influence'] = _unit_influence
        else:
            out['unit_influence_unavailable_reason'] = (
                _unit_influence_reason or
                'leave-one-unit-out influence not available for this cell')
    elif _granularity in ('document', 'article'):
        out['pooled_ci_unavailable_reason'] = (
            _pooled_ci_reason or
            'token-pooled cluster CI not available for this cell')
    _seed_deltas = primary_deltas
    if _seed_deltas is None:
        _seed_deltas = []
        for _ra, _rb in zip(rows_a, rows_b):
            _pa = L._finite_scalar((_ra or {}).get('ppl'))
            _pb = L._finite_scalar((_rb or {}).get('ppl'))
            if _pa is not None and _pb is not None:
                _seed_deltas.append(_pa - _pb)
    else:
        _seed_deltas = list(_seed_deltas)
    if _seed_deltas:
        _choice = _analysis_choice_block(out, _seed_deltas)
        if _choice is not None:
            out['analysis_choice_concordance'] = _choice
    return out


def _length_interaction_analysis(recs_a, recs_b, common, lengths, *, primary_len=None,
                                 alpha=0.05, min_n=6):
    """Paired effect modification of the arm contrast across evaluation lengths.

    PPL at different context lengths is not on one additive scale: the baseline
    PPL itself grows with length, so an absolute ΔPPL at L512 and at L4096 are
    not directly comparable.  The paired effect is therefore the per-seed
    ``log(PPL_a) - log(PPL_b)`` at each length (positive = arm a is worse),
    and the interaction is the within-seed change of that effect relative to a
    reference length (the training length 512 when present, otherwise the
    shortest evaluated length).

    This is deliberately **not** "compare the p-value at 512 with the p-value
    at 4096": a difference in significance is not a significant difference.
    The same seeds must be observed at every length; a seed missing one cell is
    dropped from the whole panel and named, never paired positionally.  The
    result is post-hoc/descriptive and never enters the frozen Holm family.
    """
    lens = sorted({int(x) for x in (lengths or [])})
    common = list(common or [])
    if len(lens) < 2 or len(common) < 2:
        return None
    effects = {}
    for Ln in lens:
        vals = []
        for s in common:
            pa = _ppl_at_len(recs_a.get(s), Ln)
            pb = _ppl_at_len(recs_b.get(s), Ln)
            if pa is None or pb is None:
                vals.append(float('nan'))
                continue
            try:
                vals.append(float(math.log(pa) - math.log(pb)))
            except (ValueError, OverflowError):
                vals.append(float('nan'))
        effects[str(Ln)] = vals
    reference = 512 if 512 in lens else lens[0]
    res = L.effect_modification(
        effects, reference=str(reference),
        condition_values={str(Ln): float(math.log2(Ln)) for Ln in lens},
        seeds=[int(s) for s in common], alpha=alpha, min_n=min_n)
    res.update({
        'lengths': [int(x) for x in lens],
        'reference_length': int(reference),
        'primary_length': (int(primary_len) if primary_len is not None else None),
        'effect_definition': ('log(PPL_a) - log(PPL_b) at each length; positive = arm a '
                              'has higher PPL (worse)'),
        'condition_axis': 'log2(eval length)',
        'role': ('post-hoc length effect-modification diagnostic; not in the frozen '
                 'Holm family and never changes `confirmed`'),
        'simple_effect_note': ('per-length simple effects are descriptive; compare the '
                               'interaction, not the significance of separate lengths'),
    })
    return res


def _test_probe_stats(probe_path, manifest, slice_fp):
    """Descriptive summary of the P5T test-split distractor probe.

    The probe is measured on the SAME held-out slice as the primary by-length
    panel (records carry `val_fp`), so admission here applies the same
    identity checks: frozen arm/seed whitelist, matching test fingerprint,
    matching code semantics and training recipe.  This is explicitly
    secondary/descriptive and never enters the Holm family or the
    confirmation decision."""
    out = {'path': probe_path, 'slice_fp': slice_fp, 'available': False,
           'n_records': 0, 'n_admitted': 0, 'n_excluded': 0,
           'cells': [], 'contrasts': {},
           'note': 'test-split descriptive secondary evidence; not part of the frozen primary family or the Holm correction'}
    if not slice_fp or not probe_path or not os.path.exists(probe_path):
        return out
    try:
        raw = json.load(open(probe_path, encoding='utf-8'))
    except Exception as e:
        out['error'] = f'{type(e).__name__}: {e}'
        return out
    if not isinstance(raw, dict):
        out['error'] = 'probe summary is not a JSON object'
        return out
    out['available'] = True
    out['n_records'] = len(raw)
    arms = set(manifest['arms'])
    seeds = {int(s) for s in manifest['seeds']}
    exp_code = L.code_stamp()
    impl = _frozen_implementation(manifest)
    want_train_fp = _frozen_train_fp(manifest)
    _require_train_fp = _train_fp_required(manifest) or any(
        isinstance(_r, dict) and bool(_r.get('train_fp')) for _r in raw.values())
    groups = {}
    excluded = []
    for key, r in raw.items():
        if not isinstance(r, dict):
            excluded.append(f'{key}: not a dict')
            continue
        why = None
        _seed_i = _record_int(r.get('seed'))
        _variant = r.get('variant')
        if r.get('synthesized'):
            # A reconstructed probe trajectory is provenance, not an
            # independent measurement.  The primary admission gate and the
            # v10 probe analysis already reject it; the test-probe path must
            # not become a back door into a descriptive contrast.
            why = 'synthesized reconstruction is not an independent measurement'
        elif not isinstance(_variant, str) or _variant not in arms:
            why = f"variant {_variant!r} is not one of the frozen arms"
        elif _seed_i is None or _seed_i not in seeds:
            why = f"seed {r.get('seed')!r} is outside the frozen whitelist"
        elif r.get('val_fp') != slice_fp:
            why = f"val_fp {str(r.get('val_fp'))[:12]}… does not match the frozen test slice {str(slice_fp)[:12]}…"
        elif exp_code is not None and r.get('_code') != exp_code:
            why = f"_code {r.get('_code')!r} does not match the frozen code semantics {exp_code!r}"
        elif _require_train_fp and not r.get('train_fp'):
            why = 'missing training-corpus fingerprint (train_fp) for the probe record'
        elif want_train_fp is not None and r.get('train_fp') != want_train_fp:
            why = f"train_fp {r.get('train_fp')} does not match the frozen canonical training fingerprint {want_train_fp}"
        elif _recipe_signature(manifest, r.get('variant')) is not None and not _recipe_matches_signature(r.get('recipe'), _recipe_signature(manifest, r.get('variant'))):
            why = 'recipe does not match the frozen training_recipe signature (including the per-arm mr)'
        elif _implementation_reason(r, impl) is not None:
            why = _implementation_reason(r, impl)
        elif not L.ppl_is_usable(r.get('ppl_mean')):
            why = 'ppl_mean is not a usable measurement'
        elif not isinstance(r.get('arm'), str) or not r.get('arm'):
            why = f"arm {r.get('arm')!r} is not a non-empty string"
        elif r.get('eval_len') is None or r.get('rho') is None:
            why = 'missing arm/eval_len/rho'
        elif _record_int(r.get('eval_len')) is None:
            why = f"eval_len {r.get('eval_len')!r} is not an integer"
        else:
            try:
                _rho_i = float(r.get('rho'))
            except (TypeError, ValueError, OverflowError):
                _rho_i = float('nan')
            if not math.isfinite(_rho_i):
                why = f"rho {r.get('rho')!r} is not numeric"
        if why:
            excluded.append(f'{key}: {why}')
            continue
        mode = str(r.get('mode', 'uniform'))
        ident = (r['variant'], r['arm'], _record_int(r['eval_len']), float(r['rho']), mode,
                 json.dumps(V10._fp_norm(r.get('probe_params')), sort_keys=True),
                 str(r.get('recipe')), str(r.get('_code')),
                 str(V10._recipe_norm(r.get('recipe'))))
        grp = groups.setdefault(ident, {})
        if _seed_i in grp:
            excluded.append(f'{key}: duplicate (cell, seed) identity — the pooled mean would be ambiguous')
            continue
        grp[_seed_i] = float(r['ppl_mean'])
    out['n_excluded'] = len(excluded)
    out['n_admitted'] = sum(len(g) for g in groups.values())
    by_visible = {}
    for ident, by_seed in groups.items():
        v, arm, Ln, rho, mode = ident[:5]
        by_visible.setdefault((v, arm, Ln, rho, mode), []).append((ident, by_seed))
    for (v, arm, Ln, rho, mode), entries in sorted(by_visible.items()):
        if len(entries) > 1:
            # multiple probe parameterisations/recipes for one readable cell:
            # report each separately, but never pool them.
            for ident, by_seed in entries:
                out['cells'].append({'variant': v, 'arm': arm, 'eval_len': Ln, 'rho': rho,
                                     'mode': mode, 'n': len(by_seed),
                                     'mean': float(np.mean(list(by_seed.values()))),
                                     'ppl_by_seed': {str(sd): p for sd, p in sorted(by_seed.items())},
                                     'ambiguous_cell': True})
            continue
        _ident, by_seed = entries[0]
        out['cells'].append({'variant': v, 'arm': arm, 'eval_len': Ln, 'rho': rho,
                             'mode': mode, 'n': len(by_seed),
                             'mean': float(np.mean(list(by_seed.values()))),
                             'ppl_by_seed': {str(sd): p for sd, p in sorted(by_seed.items())}})
    # Descriptive paired contrasts against the learned arm on the same slice.
    # A visible cell key deliberately omits probe_params/recipe/code so the
    # display can show one row per cell; a CONTRAST must not do the same.
    # Keep every full identity for a visible key: if it is ambiguous, or if
    # the two arms were probed under different parameters/recipes/code, their
    # difference would compare two different measurements.
    _idx = {}
    _contrast_exclusions = []
    for ident, by_seed in groups.items():
        _idx.setdefault(ident[:5], []).append((ident, by_seed))
    for (v, arm, Ln, rho, mode), learned_entries in sorted(_idx.items()):
        if arm != 'learned' or len(learned_entries) != 1:
            continue
        _learned_id, learned = learned_entries[0]
        for (_v, other_arm, _Ln, _rho, _mode), other_entries in sorted(_idx.items()):
            if (_v != v or _Ln != Ln or _rho != rho or _mode != mode
                    or other_arm == 'learned' or len(other_entries) != 1):
                continue
            _other_id, other = other_entries[0]
            # ident = (variant, arm, eval_len, rho, mode, probe_params,
            #          recipe, code, normalised recipe); compare the
            # measurement identity.  The normalised recipe ignores only the
            # parameter-matching `mr` field, exactly like v10's probe pairs.
            if _learned_id[5] != _other_id[5]:
                _contrast_exclusions.append(
                    f'contrast L{Ln} r{rho} [{mode}] {other_arm} - learned: '
                    'the two arms carry different probe_params — pairing them '
                    'would difference two different measurements')
                continue
            if _learned_id[7] != _other_id[7] or _learned_id[8] != _other_id[8]:
                _contrast_exclusions.append(
                    f'contrast L{Ln} r{rho} [{mode}] {other_arm} - learned: '
                    'the two arms carry different recipe/code stamps — '
                    'pairing them would mix two measurement semantics')
                continue
            common = sorted(set(learned) & set(other), key=float)
            if len(common) < 2:
                continue
            dl = [other[s] - learned[s] for s in common]
            res = V.exact_sign_permutation(dl)
            res['seeds'] = [int(s) for s in common]
            res['delta_definition'] = f'{other_arm} - learned (paired per seed, test split, descriptive)'
            out['contrasts'][f'L{Ln} r{rho} [{mode}] {other_arm} - learned'] = res
    out['contrast_exclusions'] = _contrast_exclusions
    if excluded:
        print('[p5t probe] EXCLUDED test-probe records (named, never silent):\n  - ' + '\n  - '.join(excluded[:20]) + ('' if len(excluded) <= 20 else f'\n  … {len(excluded) - 20} more'))
    if _contrast_exclusions:
        print('[p5t probe] EXCLUDED descriptive contrasts (named, never silent):\n  - ' + '\n  - '.join(_contrast_exclusions[:20]))
    return out


def _confirmation_robustness_block(stats, pairs, *, alpha, family_size, param_required,
                                   family_names=None):
    """Attach the post-hoc fragility/seed-drop readout to a stats dict.

    ``pairs`` maps each comparison name to ``{'deltas': [...], 'seeds': [...]}``
    — the same per-seed deltas that produced the primary p-values.  The
    frozen decision in ``stats['confirmed']`` is never recomputed or changed;
    this only adds a diagnosis of how much that decision depends on one seed.

    ``family_names`` is the DECLARED comparison list.  An omitted comparison
    has no deltas but still occupies a family slot, so the seed-drop jackknife
    must keep it at p=1 instead of shrinking m.
    """
    if not pairs:
        return None
    rob = L.confirmation_robustness(pairs, alpha=alpha,
                                    family_size=family_size,
                                    confirmed=stats.get('confirmed'),
                                    family_names=family_names)
    comps = stats.get('comparisons') or {}
    holm = stats.get('holm') or {}
    _rej = sorted(k for k, v in holm.items()
                  if v.get('p_holm') is not None
                  and float(v['p_holm']) < alpha)

    def _param_gate(k):
        if not param_required:
            return True
        return comps.get(k, {}).get('param_match_complete') is True

    # Replay the direction/parameter/minimum-n part of the frozen rule on each
    # seed-drop fold.  The complete-coverage gate is deliberately NOT
    # replayed: dropping a seed necessarily makes coverage false, so
    # including it would fail every fold by construction and say nothing
    # about sampling stability.
    full_nocov = sorted(
        k for k in _rej
        if comps.get(k, {}).get('direction_ok') is True
        and comps.get(k, {}).get('param_matched') is not False
        and _param_gate(k)
        and int(comps.get(k, {}).get('n_effective') or 0)
        >= int(comps.get(k, {}).get('min_seeds_for_confirmation')
               or len(comps.get(k, {}).get('seeds') or [])))
    fold_nocov = []
    for f in rob['family_jackknife']['folds']:
        keep = set(f['rejected'])
        row = []
        for k in full_nocov:
            if k not in keep:
                continue
            _comp = comps.get(k, {})
            _floor = int(_comp.get('min_seeds_for_confirmation')
                         or len(_comp.get('seeds') or []))
            if int(f.get('n_pairs', {}).get(k) or 0) < _floor:
                continue
            row.append(k)
        fold_nocov.append(sorted(row))
    rob['confirmed_without_coverage_full'] = full_nocov
    rob['confirmed_without_coverage_by_fold'] = {
        str(f['seed']): s for f, s in
        zip(rob['family_jackknife']['folds'], fold_nocov)}
    rob['confirmed_without_coverage_stable'] = bool(
        all(s == full_nocov for s in fold_nocov))
    rob['coverage_gate_note'] = (
        'the complete-coverage rule is not replayed inside the jackknife: '
        'dropping one seed always makes coverage_ok false, which would fail '
        'every fold by construction.  The jackknife asks only whether the '
        'statistical/direction/minimum-n verdict survives; the frozen decision still '
        'additionally requires complete coverage.')
    stats['robustness'] = rob
    return rob


def _doc_ci_markdown_cell(dsm):
    """Render the document/row sampling interval without mixing estimands.

    Prefer the token-pooled cluster-robust interval (same estimand as the
    primary Δ); fall back to the labelled equal-weighted unit bootstrap only
    when the pooled interval is unavailable.  Kept pure so the report wording
    is testable without a panel.
    """
    dsm = dsm if isinstance(dsm, dict) else {}
    _u = dsm.get('unit') or dsm.get('granularity')
    unit = {'article': '文章', 'document': '文档'}.get(_u, '行')

    def _fmt(ci):
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            try:
                lo, hi = float(ci[0]), float(ci[1])
            except (TypeError, ValueError, OverflowError):
                return None
            if math.isfinite(lo) and math.isfinite(hi):
                return f'[{lo:+.2f}, {hi:+.2f}]'
        return None

    pooled = _fmt(dsm.get('pooled_delta_ppl_ci95'))
    if pooled is not None:
        return f'{pooled}（{unit} cluster，token 池化）'
    unit_ci = _fmt(dsm.get('per_doc_delta_ppl_ci95'))
    if unit_ci is not None:
        return f'{unit_ci}（{unit} bootstrap，等权单元）'
    return '—'


def _doc_influence_markdown(ft):
    """Render the post-hoc leave-one-unit-out influence table.

    Pure function of the final-test stats dict, so the wording and the
    "does not change the frozen decision" boundary are testable without a
    panel.  It deliberately uses the scale-free log-ratio influence for the
    fragility flag: a PPL difference is expressed on the baseline's scale,
    while the log ratio is the multiplicative effect the arms are compared on.
    """
    comps = (ft or {}).get('comparisons') if isinstance(ft, dict) else None
    if not isinstance(comps, dict):
        return []
    rows = []
    missing = []
    for name, comp in comps.items():
        if not isinstance(comp, dict):
            continue
        dsm = comp.get('doc_sampling') or {}
        inf = dsm.get('unit_influence') if isinstance(dsm, dict) else None
        if not isinstance(inf, dict) or not inf.get('available'):
            if (isinstance(dsm, dict)
                    and dsm.get('pooled_delta_ppl_ci95') is not None):
                missing.append(
                    f'{name}（{dsm.get("unit_influence_unavailable_reason") or "缺少影响块"}）')
            continue
        unit = {'article': '文章', 'document': '文档'}.get(
            dsm.get('unit') or dsm.get('granularity'), '单元')
        full = inf.get('full_delta_ppl')
        idx = inf.get('most_influential_index')
        loo = inf.get('loo_delta_ppl') or []
        loo_delta = (loo[idx] if isinstance(idx, int) and 0 <= idx < len(loo)
                     else None)
        rel_lr = inf.get('max_abs_relative_change_log_ratio')
        rel_ppl = inf.get('max_abs_relative_change_ppl')
        rel_txt = '—'
        if isinstance(rel_lr, (int, float)) and math.isfinite(rel_lr):
            rel_txt = f'log {100.0 * rel_lr:.0f}%'
        if isinstance(rel_ppl, (int, float)) and math.isfinite(rel_ppl):
            rel_txt += (f' / PPL {100.0 * rel_ppl:.0f}%'
                        if rel_txt != '—' else f'PPL {100.0 * rel_ppl:.0f}%')
        same = inf.get('sign_stable_log_ratio')
        same_txt = ('—' if same is None else ('是' if same else '否（翻转）'))
        ci = inf.get('jackknife_ci95_ppl')
        ci_txt = '—'
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            try:
                lo, hi = float(ci[0]), float(ci[1])
                if math.isfinite(lo) and math.isfinite(hi):
                    ci_txt = f'[{lo:+.2f}, {hi:+.2f}]'
            except (TypeError, ValueError, OverflowError):
                pass
        _frag = inf.get('fragile_to_one_unit')
        if _frag is None:
            flag = '—（无方向）'
        elif _frag:
            flag = '⚠ 单单元主导'
        else:
            flag = '未见单单元翻转'
        uid = inf.get('most_influential_unit')
        uid_txt = (str(uid)[:16] if uid is not None else '—')
        if inf.get('label_source') == 'index_fallback':
            uid_txt += '（索引）'
        tied = inf.get('most_influential_tied_count')
        if isinstance(tied, int) and tied > 1:
            uid_txt += f'（并列 {tied}）'
        full_txt = f'{full:+.3f}' if isinstance(full, (int, float)) and math.isfinite(full) else '—'
        loo_txt = (f'{loo_delta:+.3f}'
                   if isinstance(loo_delta, (int, float)) and math.isfinite(loo_delta)
                   else '—')
        rows.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |'
                    % (name, unit, inf.get('n_units', '—'), full_txt, uid_txt,
                       loo_txt, rel_txt, same_txt, ci_txt, flag))
    if not rows and not missing:
        return []
    out = ['### 单元影响诊断（post-hoc；不改变上面的冻结判定）', '']
    if rows:
        out += ['| 比较 | 单位 | n | 全 panel ΔPPL | 最敏感单元 | 删除后 ΔPPL | 最大单单元变化 | 删除后同向 | jackknife 95% CI | 标记 |',
                '|---|---|---|---|---|---|---|---|---|---|']
        out.extend(rows)
        out += ['',
                '> 定义：对每个评测单位 i 删除它，再用**同一 token 池化估计量**重算 ΔPPL；'
                '`最敏感单元` 是使 log 效应变化最大的单位，`jackknife 95% CI` 是 delete-one-unit '
                'jackknife 区间（条件在已训练种子上）。`单单元主导` 指删除一个单位使 log 效应变化 ≥100%；'
                '方向翻转单独标记。该表是**对已发生数据的 post-hoc 描述**：不进入冻结 p/Holm/confirmed，'
                '不把评测单位当成随机化，也不能据此做总体推断；删除单位后描述的是「冻结切片减一个单位」。'
                '逐单元完整 LOO 向量与 delta-method influence 见 `analysis_v12/test_stats.json` 的 '
                '`doc_sampling.unit_influence`。', '']
    if missing:
        out += ['> 以下比较有 pooled CI 但没有可用的单元影响块，'
                '单单元敏感性**未验证**：' + '；'.join(missing) + '。', '']
    frag = [r.split('|')[1].strip() for r in rows if '单单元主导' in r]
    if frag:
        out += ['**解读提示**：' + '、'.join(frag) +
                ' 的方向/幅度依赖单个评测单位；引用时必须同时报告该敏感度，'
                '不能把结论描述为「对评测文档稳健」。', '']
    return out


def _analysis_choice_markdown(ft):
    """Render the post-hoc analysis-choice concordance table.

    Pure function of the final-test stats dict so the wording and the
    "does not change the frozen decision" boundary can be tested without a
    panel.  The block is descriptive: it neither adds a p-value nor selects a
    preferred specification.
    """
    comps = (ft or {}).get('comparisons') if isinstance(ft, dict) else None
    if not isinstance(comps, dict):
        return []
    labels = {
        'stable': '稳定',
        'precision_sensitive': '⚠ 精度口径敏感',
        'direction_unstable': '⚠ 方向翻转',
        'zero_effect': '零效应',
        'insufficient_specs': '口径不足',
    }
    rows = []
    details = []
    missing = []
    fragile = []
    for name, comp in comps.items():
        if not isinstance(comp, dict):
            continue
        dsm = comp.get('doc_sampling') or {}
        cc = dsm.get('analysis_choice_concordance') if isinstance(dsm, dict) else None
        if not isinstance(cc, dict):
            if (isinstance(dsm, dict)
                    and (dsm.get('per_doc_delta_mean') is not None
                         or dsm.get('pooled_delta_ppl') is not None)):
                missing.append(name)
            continue
        status = cc.get('status')
        if status not in labels:
            continue
        n_specs = cc.get('n_specs')
        n_avail = cc.get('n_available')
        n_same = cc.get('n_same_as_primary')
        n_opp = cc.get('n_opposite_to_primary')
        n_excl = cc.get('n_excluding_zero')
        n_ci = cc.get('n_with_ci')
        psign = cc.get('primary_sign')
        if psign is None or psign == 0:
            sign_txt = '零/无方向'
        else:
            sign_txt = '负（a 更低）' if float(psign) < 0 else '正（a 更高）'
        same_txt = f'{n_same}/{n_opp}' if n_same is not None else '—'
        ci_txt = (f'{n_excl}/{n_ci}' if n_ci is not None else '—')
        if cc.get('precision_concordant') is False:
            ci_txt += '（不一致）'
        er = cc.get('effect_range_ppl') or {}
        if all(isinstance(er.get(k), (int, float)) for k in ('min', 'max')):
            rng = f'[{er["min"]:+.3f}, {er["max"]:+.3f}]'
            if isinstance(er.get('span'), (int, float)):
                rng += f'（跨度 {er["span"]:.3f}）'
        else:
            rng = '—'
        flag = labels[status]
        if cc.get('fragile'):
            fragile.append(name)
        rows.append('| %s | %s/%s | %s | %s | %s | %s | %s |'
                    % (name, n_avail, n_specs, sign_txt, same_txt,
                       ci_txt, rng, flag))
        specs = cc.get('specifications') or []
        if status == 'direction_unstable':
            opp = [str(s.get('spec')) for s in specs
                   if isinstance(s, dict) and s.get('available')
                   and s.get('sign') not in (None, 0, psign)]
            if opp:
                details.append(f'{name}: 与主口径反向的规格为 {opp}')
        elif status == 'precision_sensitive':
            cross = cc.get('crossing_zero_specs') or []
            excl = cc.get('excluding_zero_specs') or []
            if cross or excl:
                details.append(
                    f'{name}: 排除零的规格 {excl or "无"}；跨零的规格 '
                    f'{cross or "无"}')
    if not rows and not missing:
        return []
    out = ['### 分析口径一致性（post-hoc；不改变上面的冻结判定）', '']
    if rows:
        out += ['| 比较 | 可用/总口径 | 主方向 | 同向/反向 | 区间排除零 | PPL Δ 范围 | 标记 |',
                '|---|---|---|---|---|---|---|']
        out.extend(rows)
        out += ['',
                '> 定义：同一 panel 的多种口径（逐种子 mean/median/trimmed mean、'
                'token 池化、逐评测单元 mean/median/log-ratio/win-balance）在方向与'
                '「区间是否排除零」上是否一致。`同向/反向` 相对主口径。'
                '方向一致不等于估计量相同：逐种子、逐单元与 token 池化回答不同问题，'
                '区间宽度也依赖单位。`精度口径敏感` 只说明带区间口径的零排除结论不一致，'
                '不是方向矛盾。该表是**对已发生数据的 post-hoc 描述**：不新增检验/多重校正，'
                '不进入冻结 p/Holm/confirmed，也不能用来挑选最有利口径；'
                '引用时不得只报单一聚合而隐去本表。', '']
    if details:
        out += ['**口径细节**：' + '；'.join(details) + '。', '']
    if missing:
        out += ['> 以下比较有 doc_sampling 但缺少分析口径一致性块'
                '（旧产物或未重新生成分析），该敏感性**未验证**：'
                + '、'.join(missing) + '。', '']
    if fragile:
        out += ['**解读提示**：' + '、'.join(fragile) +
                ' 的方向或精度结论依赖分析口径；引用时必须同时报告本表，'
                '不能把结论描述为「换一种聚合仍然稳健」。', '']
    return out


def _robustness_markdown(rob):
    """Render the post-hoc fragility/seed-drop table for the v12 report.

    Kept as a pure function of the JSON block so the wording and the
    "does not change the frozen decision" boundary can be unit-tested without
    a GPU panel or a full report build.
    """
    out = []
    if not (isinstance(rob, dict) and rob.get('per_comparison')):
        return out
    A = out.append
    A('### 结论稳健性（post-hoc 诊断；不改变上面的冻结判定）')
    A('')
    A('| 比较 | Δ mean [95% CI]（seed） | 同向种子 | 丢弃最弱种子仍显著 k | 翻转最小同向种子后失效 k | 逐种子删除的 Holm p 范围 | 最敏感种子 |')
    A('|---|---|---|---|---|---|---|')
    fjk = rob.get('family_jackknife') or {}
    fjc = fjk.get('per_comparison') or {}
    for rk, rr in rob['per_comparison'].items():
        n = rr.get('n')
        same = rr.get('same_sign_count')
        req = rr.get('min_same_sign_for_family_gate')
        sames = f'{same}/{n}' if same is not None and n else '—'
        if req is not None:
            sames += f'（族门等幅投影 ≥{req}）'
        _lo, _hi = rr.get('ci95_low'), rr.get('ci95_high')
        if (isinstance(rr.get('mean'), (int, float))
                and isinstance(_lo, (int, float)) and isinstance(_hi, (int, float))):
            citxt = f'{rr["mean"]:+.3f} [{_lo:+.3f}, {_hi:+.3f}]'
        else:
            citxt = '—'
        drop = rr.get('drop_fragility')
        flip = rr.get('flip_fragility')
        reason = {'p>=alpha': 'p≥α', 'direction reversed': '方向反转',
                  'not significant at the full panel': '整组已不显著',
                  'no mean direction': '无方向'}.get(
                      rr.get('flip_stop_reason'), rr.get('flip_stop_reason') or '—')
        fliptxt = f'{flip}（{reason}）' if isinstance(flip, int) else '—'
        fj = fjc.get(rk) or {}
        lo, hi = fj.get('p_holm_min'), fj.get('p_holm_max')
        rng = (f'{lo:.3g}–{hi:.3g}'
               if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) else '—')
        ms = rr.get('most_influential_seed')
        mp = rr.get('most_influential_p_shift')
        mstxt = ((f'种子 {ms}' + (f'（|Δp|={mp:.3g}）' if isinstance(mp, (int, float)) else ''))
                 if ms is not None else '—')
        _tied = rr.get('most_influential_tied_count')
        if isinstance(_tied, int) and _tied > 1 and mstxt != '—':
            mstxt += f'；并列 {_tied} 个'
        A(f'| {rk} | {citxt} | {sames} | {drop if isinstance(drop, int) else "—"} | '
          f'{fliptxt} | {rng} | {mstxt} |')
    A('')
    rejf = fjk.get('rejected_full') or []
    always = fjk.get('always_rejected') or []
    unstable = fjk.get('unstable_folds') or []
    A(f"- 整组 Holm 拒绝（全 panel）：{sorted(rejf) if rejf else '无'}。")
    if fjk.get('all_rejections_stable'):
        A('- 逐种子删除检验：**每一个**单种子删除后，Holm 拒绝集合都保持不变。')
    else:
        A(f"- 逐种子删除检验：**不稳定**。删除这些种子会改变 Holm 拒绝集合：{sorted(unstable)}；"
          f"始终被拒绝的比较：{sorted(always) if always else '无'}。")
    nocov_stable = rob.get('confirmed_without_coverage_stable')
    nocov_full = rob.get('confirmed_without_coverage_full') or []
    A(f"- 方向/参数/最小 n 判定（不含完整覆盖门）：全 panel = {sorted(nocov_full) if nocov_full else '无'}；"
      f"逐种子删除后保持不变 = {nocov_stable}。")
    A('')
    A('> 定义：`丢弃最弱种子仍显著 k` = 反复删除当前 |Δ| 最小的种子、精确符号翻转 p 仍 < α 的最大次数；'
      '`翻转最小同向种子后失效 k` = 按 |Δ| 从小到大翻转同向种子，直到 p ≥ α 或均值方向反转的次数（贪心，不是精确最小集）。'
      '两者都是**对已发生数据的 post-hoc 敏感度读数**，不进入冻结决策，也不改变上面的 p/Holm/confirmed。'
      '完整覆盖门不在 jackknife 内重放（删任何一个种子都会使覆盖失败），因此上一条只重放方向/参数/最小 n 判定。')
    A('')
    if fjk.get('all_rejections_stable') is False:
        A('**解读提示**：至少一个冻结判定在删除单个种子后不再成立。这不改变预注册的确认结论，'
          '但该结论**不能被描述为「对种子稳健」**；引用时必须同时报告这一敏感度。')
        A('')
    return out


def _length_interaction_markdown(name, li):
    """Render the post-hoc length effect-modification table.

    Pure formatting so the "interaction, not two significance verdicts" caveat
    and the table can be unit-tested without a GPU panel or a report build.
    """
    out = []
    if not (isinstance(li, dict) and li.get('conditions')):
        return out
    A = out.append

    def _fmt_ci(cell):
        lo, hi = cell.get('ci95_low'), cell.get('ci95_high')
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            if math.isfinite(float(lo)) and math.isfinite(float(hi)):
                return f'[{float(lo):+.4f}, {float(hi):+.4f}]'
        return '—'

    def _fmt_p(v):
        return f'{float(v):.4g}' if isinstance(v, (int, float)) and math.isfinite(float(v)) else '—'

    ref = str(li.get('reference'))
    A(f'**`{name}`** — 参考条件 L{li.get("reference_length", ref)}；'
      f'效应 = log(PPL_a/PPL_b)，正 = a 更差；交互 = 该长度的效应减参考长度的效应。')
    A('')
    A('| 条件 | n | 简单效应 mean [95% CI] | 与参考的交互 Δ [95% CI] | 效应比值变化 | 交互 p（原始） | 交互 p（Holm） |')
    A('|---|---|---|---|---|---|---|')
    _simples = li.get('simple_effects') or {}
    _inters = li.get('interactions') or {}
    for lab in li.get('conditions') or []:
        cell = _simples.get(lab) or {}
        mean = cell.get('mean')
        mtxt = f'{float(mean):+.4f} {_fmt_ci(cell)}' if isinstance(mean, (int, float)) else '—'
        if lab == ref:
            itxt, rtxt, praw, pholm = '—（参考）', '—', '—', '—'
        else:
            inter = _inters.get(lab) or {}
            dm = inter.get('delta_mean')
            itxt = (f'{float(dm):+.4f} {_fmt_ci(inter)}'
                    if isinstance(dm, (int, float)) else '—')
            ratio = inter.get('ratio')
            rtxt = (f'×{float(ratio):.3f}' if isinstance(ratio, (int, float))
                    and math.isfinite(float(ratio)) else '—')
            praw = _fmt_p(inter.get('p_exact_signflip'))
            pholm = _fmt_p(inter.get('p_holm'))
        A(f'| L{lab} | {cell.get("n", 0)} | {mtxt} | {itxt} | {rtxt} | {praw} | {pholm} |')
    A('')
    tr = li.get('trend') or {}
    if isinstance(tr, dict) and tr.get('n'):
        A(f"- 趋势（逐种子斜率 / {li.get('condition_axis', 'condition axis')}）："
          f"{float(tr.get('mean')):+.4f} {_fmt_ci(tr)}，p={_fmt_p(tr.get('p_exact_signflip'))}，"
          f"n={tr.get('n')}；这是 post-hoc 趋势读数，不进冻结族。")
    else:
        A(f"- 趋势：不可估（{li.get('trend_reason') or '—'}）。")
    _n, _min = li.get('n_complete'), li.get('min_n')
    if isinstance(_n, int) and isinstance(_min, int) and _n < _min:
        A(f"- 样本：只有 {_n} 个完整种子（< min_n={_min}），交互 p 仅作描述，不能称为已确认。")
    _dropped = li.get('dropped_seeds') or []
    if _dropped:
        A(f"- 完整种子集合：缺少任一长度可用单元的种子 {_dropped} 已从整个交互面板剔除（不是按长度各自配对）。")
    A('')
    A('> 这是 **post-hoc 效应修饰（interaction）诊断**，不属于冻结 Holm 族，也不改变 `confirmed`。'
      '不显著 ≠ 各长度效应相同；「L512 显著、L4096 不显著」也不能当作长度依赖证据——'
      '必须看本表的交互检验。简单效应与趋势仅作描述。')
    A('')
    return out


def finaltest_stats(manifest=None, bylen_path=None, probe_path=None, out='analysis_v12/test_stats.json'):
    """Zero-GPU: paired confirmation statistics under the frozen plan.

    统计入口现在是**严格准入制**——种子白名单、唯一
    记录、非空数据指纹、单一冻结 test 切片、_code/recipe 与冻结计划的机器
    校验、manifest 哈希绑定，缺一项即不入统计（点名剔除）或拒绝运行
    （歧义/混切片）。by_length 汇总同样只由可采纳记录构成。"""
    if manifest is None:
        manifest = _load_manifest(FREEZE_MANIFEST)
    bylen_path = bylen_path or os.path.join(TEST_OUTDIR, 'bylen.json')
    probe_path = probe_path or os.path.join(TEST_OUTDIR, 'distractor.json')
    raw = {}
    if os.path.exists(bylen_path):
        raw = json.load(open(bylen_path, encoding='utf-8'))
    admitted, unified_fp = _admit_confirmation_records(raw, manifest)
    # The analysis entry point is as much part of the frozen round as P5T:
    # reuse the same persistent lock so a manifest or test-slice change after
    # the first access is refused here too, not only during evaluation.  A
    # non-empty confirmation file without its first-access lock is refused:
    # a copied bylen.json alone cannot prove which plan/slice produced it.
    if unified_fp is not None:
        _lock_dir = os.path.dirname(os.path.abspath(bylen_path))
        if not os.path.exists(_freeze_lock_path(_lock_dir)):
            raise SystemExit(f'[v12] REFUSE: {bylen_path} has admissible confirmation records but no freeze lock in {_lock_dir} — refuse to analyse a copied result that cannot be tied to the first test-split access')
        _adm_train_fps = {r.get('train_fp') for r in admitted.values() if isinstance(r.get('train_fp'), str) and r.get('train_fp')}
        _check_or_write_freeze_lock(_lock_dir, manifest, unified_fp, stage='analysis',
                                    train_fp=(next(iter(_adm_train_fps)) if len(_adm_train_fps) == 1 else None))
    _stat = manifest.get('statistics') or {}
    _alpha = float(_stat.get('alpha', 0.05))
    _require_cov = bool(_stat.get('require_complete_coverage'))
    stats = {'alpha': float(_alpha), 'manifest': {'frozen_at': manifest.get('frozen_at'), 'manifest_sha256': manifest.get('manifest_sha256'), 'arms': manifest['arms'], 'seeds': manifest['seeds'], 'code_semantics_expected': manifest.get('code_semantics_expected'), 'statistics': manifest.get('statistics'), 'scope': manifest.get('scope'), 'revision': manifest.get('revision')}, 'admission': {'n_admitted': len(admitted), 'test_slice_fp': unified_fp, 'n_effective_seeds': len({int(s) for (_v, s) in admitted}), 'n_declared_seeds': len(manifest['seeds'])}, 'comparisons': {}, 'holm': {}, 'by_length': {}, 'probe': _test_probe_stats(probe_path, manifest, unified_fp)}
    # The exact sign-flip p is valid only under within-seed exchangeability.
    # admission binds the corpus/slice/row count/recipe/code, but the RNG stream
    # and batch order are not independently re-derivable from a record; report
    # them instead of silently assuming them.
    _rngs = sorted({str(r.get('rng_fp')) for r in admitted.values() if r.get('rng_fp')})
    # init_core_fp is a function of the SEED: a healthy panel has a different
    # digest for every seed and the SAME digest for the two arms within a
    # seed.  The old global "all values equal" test was both conceptually
    # wrong (it would fail a healthy panel) and vacuously true when no record
    # carried the field.  Summarise the per-seed two-arm contract instead.
    _init_by_seed = {}
    for (_v, _s), _r in admitted.items():
        _iv = _r.get('init_core_fp')
        # Same rule as the per-comparison audit: a malformed non-string value
        # is unverified, never stringified into a match.
        if isinstance(_iv, str) and _iv.strip():
            _init_by_seed.setdefault(int(_s), {}).setdefault(str(_v), set()).add(_iv.strip())
    _init_mismatch, _init_complete = [], True
    if set(_init_by_seed) != {int(s) for s in manifest['seeds']}:
        # A panel-level "matched initialization" statement needs evidence for
        # every declared seed, not only the ones that happened to be admitted.
        _init_complete = False
    for _s, _byarm in _init_by_seed.items():
        _vals = set().union(*_byarm.values())
        # More than one digest for the same seed is a definite inconsistency
        # whether it appears across arms or (via a duplicate record) within
        # one arm.  Admission refuses duplicate (arm, seed) records, so the
        # second case is defensive.
        if len(_vals) > 1 or any(len(_x) > 1 for _x in _byarm.values()):
            _init_mismatch.append(int(_s))
        if set(_byarm) != set(manifest['arms']):
            _init_complete = False
    _inits = sorted({x for _byarm in _init_by_seed.values() for _v in _byarm.values() for x in _v})
    _same_init = (None if not _init_by_seed
                  else (False if _init_mismatch
                        else (True if _init_complete else None)))
    stats['validity'] = {
        'rng_fp_values': _rngs,
        'same_rng_fp': len(_rngs) <= 1,
        'init_core_fp_values': _inits,
        'init_core_fp_recorded': bool(_init_by_seed),
        'init_core_fp_seeds': sorted(_init_by_seed),
        'init_core_fp_complete': bool(_init_complete),
        'init_core_fp_mismatch_seeds': _init_mismatch,
        'same_init_core_fp': _same_init,
        'n_seq_values': sorted({n for n in (_record_int(r.get('n_seq')) for r in admitted.values()) if n is not None}),
        'note': ('exact sign-flip requires within-seed exchangeability (same data order / eval rows / corruption grid). data_fp, train_fp, n_seq and recipe are machine-checked; the shared-trunk initialization digest (init_core_fp) is checked PER SEED between the two arms (a definite mismatch drops that pair; a one-sided/absent digest is UNVERIFIED, never read as a match), and rng_fp is reported but NOT enforced (architecture-induced RNG consumption is part of the arm). The exchangeability claim therefore remains a declared assumption, but the matched-initialization part of it is now measured and enforced rather than vacuously true.'),
    }
    # The frozen comparison is parameter-matched, not compute-matched.
    # Refuse a non-descriptive plan that does not disclose
    # both facts, so the caveat cannot be silently dropped from the report.
    _cbd = manifest.get('compute_budget_disclosure')
    _descriptive = (manifest.get('statistics') or {}).get('descriptive_only') is True
    if not _descriptive:
        _missing = [k for k in ('parameter_matching', 'flops', 'read_fraction')
                    if not (isinstance(_cbd, dict) and _cbd.get(k))]
        if _missing:
            raise SystemExit(f'[p5t stats] REFUSE: the frozen plan lacks compute-budget disclosure {_missing}. Parameter matching is not compute matching; the report must state the FLOPs/read-budget mismatch next to every primary comparison .')
    stats['compute_budget_disclosure'] = _cbd
    stats['parameter_binding'] = {
        'required_by_manifest': bool(_manifest_requires_params(manifest)),
        'confirmation_requires_complete': bool(_manifest_requires_params(manifest)),
        'n_admitted_records': len(admitted),
        'n_admitted_without_params': sum(
            1 for r in admitted.values()
            if (_record_int(r.get('params')) or 0) <= 0),
        'note': ('a disclosed parameter-matching contract makes params part of '
                 'the admission identity; a missing count is excluded rather '
                 'than read as param_matched=None'),
    }
    for v in manifest['arms']:
        rows = [r for (rv, s), r in admitted.items() if rv == v]
        _lens = sorted({n for n in (_record_int(k) for r in rows for k in (r.get('by_len') or {}))
                        if n is not None})
        for Ln in _lens:
            _by_seed = {}
            _nll_sum = 0.0
            _n_tok = 0
            _pool_ok = True
            for r in rows:
                bl = r.get('by_len') or {}
                c = bl.get(Ln, bl.get(str(Ln)))
                if isinstance(c, dict) and L.ppl_is_usable(c.get('ppl')) and isinstance(c.get('truncated'), (bool, np.bool_)) and not bool(c.get('truncated')):
                    _by_seed[r.get('seed')] = float(c['ppl'])
                    _nll_f = L._finite_scalar(c.get('nll_sum'))
                    _nt_i = _record_int(c.get('n_tok'))
                    if _nll_f is not None and _nt_i is not None and _nt_i > 0:
                        _nll_sum += _nll_f
                        _n_tok += _nt_i
                    else:
                        _pool_ok = False
            if _by_seed:
                _dl = list(_by_seed.values())
                _pooled = L._safe_exp_ppl(_nll_sum / _n_tok) if (_pool_ok and _n_tok > 0) else None
                stats['by_length'].setdefault(v, {})[str(Ln)] = {'mean': float(_pooled if _pooled is not None else np.mean(_dl)),
                                                                  'mean_arithmetic': float(np.mean(_dl)),
                                                                  'ppl_pooled': _pooled, 'n_tokens': _n_tok,
                                                                  'n': len(_dl), 'ppl_by_seed': {str(s): p for s, p in _by_seed.items()}, 'params': next((r.get('params') for r in rows if isinstance(r.get('params'), (int, float))), None), 'n_seqs': next((c.get('n_seq') for _r in rows for c in [((_r.get('by_len') or {}).get(Ln) or (_r.get('by_len') or {}).get(str(Ln)) or {})] if isinstance(c, dict) and c.get('n_seq') is not None), None)}
    for comp in manifest['primary_comparisons']:
        name = comp['name']
        a, b = (comp['a'], comp['b'])
        ep = comp.get('endpoint', 'by_len.ppl@4096')
        Ln = int(str(ep).rsplit('@', 1)[-1])
        exp_sign = comp.get('expected_sign')
        if exp_sign is not None:
            exp_sign = str(exp_sign).lower()
        recs_a = {s: r for (rv, s), r in admitted.items() if rv == a}
        recs_b = {s: r for (rv, s), r in admitted.items() if rv == b}
        # a paired delta is only valid on ONE held-out slice; the admission
        # gate has already enforced that EVERY admissible record shares the
        # same frozen fingerprint, so pairing cannot mix slices (and a missing
        # fingerprint could never enter here — None==None is rejected).
        common = sorted(set(recs_a) & set(recs_b))
        # Drop only the incomplete PAIR, not the whole comparison: one failed
        # L4096 cell would otherwise omit the entire comparison.  Use every
        # complete pair and report the dropped seeds.  Whether n<12 may still
        # confirm is a PRE-REGISTERED choice (statistics.min_seeds_for_confirmation,
        # manifest v3: 8, the mathematical minimum for Holm/4 with alpha=0.05).
        def _cell_ppl(rec):
            bl = rec.get('by_len') or {}
            c = bl.get(Ln, bl.get(str(Ln)))
            return float(c['ppl']) if isinstance(c, dict) and L.ppl_is_usable(c.get('ppl')) and isinstance(c.get('truncated'), (bool, np.bool_)) and not bool(c.get('truncated')) else None
        # A paired sign-flip test is only valid on pairs that started from the
        # same shared trunk.  Check it PER SEED (the digest is seed-dependent):
        # a definite mismatch is not a pair and is dropped by name; a legacy
        # one-sided/absent digest is reported as unverified, never as a match.
        _init_aud = L.paired_init_core_audit(recs_a, recs_b, common)
        _init_status = {e['seed']: e['status'] for e in _init_aud['per_seed']}
        dl = []
        usable = []
        dropped = []
        for s in common:
            if _init_status.get(int(s)) == 'mismatch':
                dropped.append({'seed': int(s), 'reason': 'init_core_fp mismatch: the two arms did not start from the same shared trunk, so the seed is not a matched pair'})
                continue
            pa, pb = (_cell_ppl(recs_a[s]), _cell_ppl(recs_b[s]))
            if pa is None or pb is None:
                dropped.append({'seed': int(s), 'reason': f'missing usable untruncated cell at L{Ln}'})
                continue
            dl.append(pa - pb)
            usable.append(s)
        if len(usable) < 2:
            stats['comparisons'][name] = {'omitted': f'<2 complete seed pairs at the frozen endpoint (common records: {common}; usable pairs: {usable}; whitelist {manifest["seeds"]})', 'n': 0, 'expected_sign': exp_sign, 'dropped_pairs': dropped, 'init_core_fp': _init_aud}
            continue
        res = V.exact_sign_permutation(dl)
        _n_eff = len(usable)
        _n_min = int((manifest.get('statistics') or {}).get('min_seeds_for_confirmation') or len(manifest['seeds']))
        res['seeds'] = usable
        res['n_effective'] = int(_n_eff)
        res['coverage_ok'] = bool(_n_eff == len(manifest['seeds']))
        res['missing_seeds'] = sorted(set(int(s) for s in manifest['seeds']) - set(int(s) for s in usable))
        res['dropped_pairs'] = dropped
        res['init_core_fp'] = _init_aud
        res['same_init_core_fp'] = _init_aud['pairing_ok']
        res['init_core_fp_recorded'] = any(('a' in e) or ('b' in e)
                                          for e in _init_aud['per_seed'])
        res['min_seeds_for_confirmation'] = _n_min
        # Diagnostics must use the n that actually entered THIS comparison,
        # so p_floor / min_attainable_p_holm / power describe the test that
        # really ran.
        # Effective resolution uses the number of NON-ZERO paired deltas;
        # exact ties cannot resolve anything.  This must describe the test
        # that actually ran, not the theoretical all-nonzero floor.
        _floor_n = int(res.get('n_nonzero', _n_eff))
        _fl_eff = (1.0 if _floor_n == 0 else L.signflip_p_floor(_floor_n))
        if math.isfinite(_fl_eff):
            res['p_floor_effective'] = _fl_eff
            res['min_attainable_p_holm_effective'] = L.holm_min_attainable_p(_floor_n, len(manifest['primary_comparisons']), _alpha)
            res['power_first_holm_gate_effective'] = {f'q={q:.2f}': L.signflip_power(_floor_n, len(manifest['primary_comparisons']), q, _alpha) for q in (0.8, 0.85, 0.9, 0.95)}
        _param = _param_match_detail(recs_a, recs_b, usable, a, b)
        res.update(_param)
        _act = []
        for _recs in (recs_a, recs_b):
            for s in usable:
                _pbd = _recs[s].get('param_breakdown')
                if isinstance(_pbd, dict):
                    _act_f = L._finite_scalar(_pbd.get('active'))
                    if _act_f is not None:
                        _act.append(_act_f)
        res['active_params_mean'] = (float(np.mean(_act)) if _act else None)
        res['data_fp'] = unified_fp
        res['manifest_sha256'] = manifest.get('manifest_sha256')
        res['delta_definition'] = f'{a} - {b} (paired per seed, test split)'
        res['expected_sign'] = exp_sign
        res['mechanism_claim_allowed'] = comp.get('mechanism_claim_allowed')
        res['claim_note'] = comp.get('claim_note')
        res['confirmation_coverage_required'] = bool(_require_cov)
        # The full by-length curve on the SAME admitted records.  Only the
        # frozen endpoint enters the Holm family / the confirmation decision;
        # the other lengths (notably L512, the training length) are
        # descriptive and are reported so an in-distribution reversal cannot be
        # hidden behind the L4096 extrapolation endpoint.
        res['by_length_curve'] = {}
        for _Ln2 in sorted({int(x) for x in ((manifest.get('eval') or {}).get('by_length_ppl') or [])}):
            _dl2, _used2 = _paired_len_deltas(recs_a, recs_b, usable, _Ln2)
            if len(_dl2) < 2:
                continue
            _o2 = L.exact_signflip_p(_dl2)
            res['by_length_curve'][str(_Ln2)] = {
                'n': len(_dl2), 'seeds': _used2, 'mean': float(np.mean(_dl2)),
                'p_exact_signflip': float(_o2['p']),
                'role': ('primary' if _Ln2 == Ln else 'descriptive'),
                'descriptive_only': bool(_Ln2 != Ln),
                'note': ('primary frozen endpoint' if _Ln2 == Ln else
                         'same records, different length; not in the frozen Holm family and never confirmatory'),
            }
        # Does the arm contrast CHANGE with length?  Separate per-length
        # sign-flip p-values cannot answer that (a difference in significance
        # is not a significant difference).  The interaction uses the same
        # seeds at every length, conditions on those seeds, and is post-hoc:
        # it is never added to the frozen Holm family or the decision rule.
        _li_lens = sorted(
            {int(x) for x in ((manifest.get('eval') or {}).get('by_length_ppl') or [])}
            | {int(Ln)})
        _li = _length_interaction_analysis(
            recs_a, recs_b, usable, _li_lens, primary_len=Ln,
            alpha=_alpha, min_n=6)
        if _li is not None and _li.get('n_complete', 0) >= 2:
            res['length_interaction'] = _li
        res['in_distribution_L512'] = res['by_length_curve'].get('512')
        if exp_sign in ('positive', 'negative') and math.isfinite(res.get('mean', float('nan'))):
            res['direction_ok'] = bool((res['mean'] > 0) == (exp_sign == 'positive'))
        else:
            res['direction_ok'] = None
        _doc = _doc_level_analysis(recs_a, recs_b, usable, Ln,
                                   primary_deltas=dl)
        if _doc is not None:
            res['doc_sampling'] = _doc
        stats['comparisons'][name] = res
    # Holm is corrected over the DECLARED frozen family, not over whichever
    # comparisons happened to produce a finite p: an omitted comparison enters
    # with p=1.0 so a missing arm can never shrink m and relax the correction.
    _fam = []
    for k, v in stats['comparisons'].items():
        p = v.get('p_exact_signflip')
        _fam.append((k, float(p) if isinstance(p, (int, float)) and math.isfinite(p) else 1.0))
    if _fam:
        stats['holm'] = L.holm_adjust(_fam, alpha=_alpha)
    _rej = [k for k, v in stats['holm'].items() if v.get('p_holm') is not None and float(v['p_holm']) < _alpha]
    # The pre-registered decision rule (manifest v4) requires p_holm < alpha
    # AND the observed direction AND parameter matching AND, when
    # statistics.require_complete_coverage is true, all frozen seeds.  A
    # complete-case subset can be selected by wall-clock skips, missing
    # checkpoints or the health gate; those are not random attrition, so the
    # shipped plan refuses to call such a subset "confirmed".  The 8-pair floor
    # remains the mathematical resolution floor and is still reported.
    # A disclosed parameter-matching contract is only satisfied by a COMPLETE
    # per-seed parameter record.  `param_matched is not false` alone would also
    # accept None (missing/inconsistent evidence), which is exactly the
    # unknown-as-pass path the admission gate was added to close.  Legacy
    # manifests without the disclosure keep the historical wording.
    _param_required = bool(_manifest_requires_params(manifest))

    # Matched initialization is a VALIDITY precondition of the paired test,
    # not an extra effect-size threshold: a pair whose two arms carry
    # different shared-trunk digests is dropped above, and a comparison that
    # still shows a definite mismatch (e.g. a legacy manifest without
    # complete coverage) cannot be called confirmed.  `None` (unverified
    # legacy evidence) keeps the historical permissive path, but it is never
    # printed as equality.
    def _init_gate(k):
        return stats['comparisons'].get(k, {}).get('same_init_core_fp') is not False

    def _param_gate(k):
        if not _param_required:
            return True
        return stats['comparisons'].get(k, {}).get('param_match_complete') is True

    _conf = [k for k in _rej if stats['comparisons'].get(k, {}).get('direction_ok') is True
             and stats['comparisons'].get(k, {}).get('param_matched') is not False
             and _param_gate(k)
             and _init_gate(k)
             and int(stats['comparisons'].get(k, {}).get('n_effective') or 0) >= int(stats['comparisons'].get(k, {}).get('min_seeds_for_confirmation') or len(manifest['seeds']))
             and (not _require_cov or stats['comparisons'].get(k, {}).get('coverage_ok') is True)]
    stats['coverage_requirement'] = {
        'require_complete_coverage': bool(_require_cov),
        'n_declared_seeds': len(manifest['seeds']),
        'note': ('complete coverage (every frozen seed) is required for a CONFIRMED verdict'
                 if _require_cov else
                 'legacy manifest: a complete-case subset with n_effective >= the declared floor may be confirmed'),
    }
    stats['confirmed'] = _conf
    # ---- post-hoc robustness: never part of the frozen decision -----------
    # A frozen verdict can be statistically valid and still hinge on one seed.
    # The diagnostic takes the SAME per-seed deltas that produced the primary
    # p-values and asks how many seeds the verdict can lose before the raw
    # test / the whole Holm family changes.  It is written next to the
    # decision, not into it: `confirmed` above is untouched.
    _rob_pairs = {}
    for _k, _v in stats['comparisons'].items():
        _dl = _v.get('per_seed')
        _sd = _v.get('seeds')
        if (isinstance(_dl, list) and isinstance(_sd, list)
                and len(_dl) == len(_sd) and len(_dl) >= 1
                and 'omitted' not in _v):
            _rob_pairs[_k] = {'deltas': _dl, 'seeds': _sd}
    _confirmation_robustness_block(
        stats, _rob_pairs, alpha=_alpha,
        family_size=len(manifest['primary_comparisons']),
        param_required=_param_required,
        family_names=[str(c.get('name')) for c in manifest['primary_comparisons']])
    stats['significant_nondirectional'] = [k for k in _rej if stats['comparisons'].get(k, {}).get('direction_ok') is None]
    stats['not_confirmed_reasons'] = {k: {'param_matched': stats['comparisons'].get(k, {}).get('param_matched'),
                                          'param_match_complete': stats['comparisons'].get(k, {}).get('param_match_complete'),
                                          'param_match_reason': stats['comparisons'].get(k, {}).get('param_match_reason'),
                                          'coverage_ok': stats['comparisons'].get(k, {}).get('coverage_ok'),
                                          'coverage_required': bool(_require_cov),
                                          'n_effective': stats['comparisons'].get(k, {}).get('n_effective'),
                                          'min_seeds_for_confirmation': stats['comparisons'].get(k, {}).get('min_seeds_for_confirmation'),
                                          'same_init_core_fp': stats['comparisons'].get(k, {}).get('same_init_core_fp'),
                                          'init_core_fp_mismatch_seeds': (stats['comparisons'].get(k, {}).get('init_core_fp') or {}).get('mismatch_seeds'),
                                          'direction_ok': stats['comparisons'].get(k, {}).get('direction_ok')}
                                      for k in _rej if k not in _conf}
    _m = len(manifest['primary_comparisons'])
    _n = len(manifest['seeds'])
    _fl = L.signflip_p_floor(_n)
    if math.isfinite(_fl):
        _min_holm = L.holm_min_attainable_p(_n, _m, _alpha)
        stats['min_attainable_p_holm'] = _min_holm
        stats['power_first_holm_gate'] = {f'q={q:.2f}': L.signflip_power(_n, _m, q, _alpha) for q in (0.8, 0.85, 0.9, 0.95)}
        if _min_holm > _alpha:
            # unreachable in practice: _load_manifest refuses such plans; keep
            # the belt-and-braces diagnostic in case a plan bypasses the gate.
            print(f'[p5t stats] WARNING: with n={_n} seeds and a Holm family of m={_m}, the smallest attainable Holm p is {_min_holm:.4g} > alpha={_alpha} — NO comparison can be confirmed regardless of the data. The load-time power gate should have refused this manifest; re-check which manifest/records produced this state.')
    if unified_fp is not None and admitted:
        # Read the recorded evaluation units from whichever frozen endpoint
        # carries them; smoking/alternate manifests need not contain L=4096.
        # A real article count is preferred when every admissible cell carries
        # it (the independent corpus unit); otherwise the line-level document
        # count; otherwise the row count.  Mixed article coverage fails closed
        # for a population claim rather than borrowing the larger line count.
        _n_art, _n_doc, _n_row = 0, 0, 0
        _art_cells = _doc_cells = _row_cells = 0
        _doc_len = None
        for _r in admitted.values():
            for _k, _c in (_r.get('by_len') or {}).items():
                if not isinstance(_c, dict):
                    continue
                _has_a = (isinstance(_c.get('ppl_per_article'), list)
                          and bool(_c['ppl_per_article']))
                _has_d = (isinstance(_c.get('ppl_per_doc'), list)
                          and bool(_c['ppl_per_doc']))
                if _has_a:
                    _art_cells += 1
                    if len(_c['ppl_per_article']) > _n_art:
                        _n_art = len(_c['ppl_per_article'])
                        _doc_len = _k
                if _has_d:
                    _doc_cells += 1
                    if len(_c['ppl_per_doc']) > _n_doc:
                        _n_doc = len(_c['ppl_per_doc'])
                        if not _has_a:
                            _doc_len = _k
                elif (isinstance(_c.get('ppl_per_seq'), list)
                      and _c['ppl_per_seq']):
                    _row_cells += 1
                    _n_row = max(_n_row, len(_c['ppl_per_seq']))
                    if not _has_a and not _has_d:
                        _doc_len = _k
        _mixed_article = bool(_art_cells and _doc_cells
                              and _art_cells != _doc_cells)
        if _art_cells:
            _gran = 'article'
            _n_units = _n_art
        elif _doc_cells:
            _gran = 'document'
            _n_units = _n_doc
        elif _row_cells:
            _gran = 'row'
            _n_units = _n_row
        else:
            _n_units = 0
            _gran = 'row'
        if _n_units:
            stats['doc_sampling_granularity'] = _gran
            stats['doc_sampling_unit'] = _gran
            stats['doc_sampling_endpoint'] = _doc_len
            stats['doc_sampling_n_units'] = int(_n_units)
            stats['doc_sampling_mixed_article_coverage'] = bool(_mixed_article)
            stats['population_claim_allowed'] = bool(
                _n_units >= L.MIN_DOCS_FOR_POPULATION_CLAIM
                and not _mixed_article)
            if (not stats['population_claim_allowed']):
                _unit = {'article': 'articles',
                         'document': 'lines (the cache calls them documents)',
                         'row': 'fixed eval rows'}[_gran]
                if _mixed_article:
                    stats['doc_sampling_note'] = (
                        'population-level claims are refused because article '
                        'coverage is mixed across admissible cells; the '
                        'larger line-level count is not substituted for the '
                        'independent unit (design_notes sections 20/43/54)')
                else:
                    stats['doc_sampling_note'] = (
                        f'{_unit} recorded (n={_n_units}); between-unit '
                        'variance is reported per comparison in '
                        f'doc_sampling — n<{L.MIN_DOCS_FOR_POPULATION_CLAIM} '
                        'does not cover the language-population distribution, '
                        'so population-level claims are refused; the '
                        'confirmed claims are about the frozen evaluation '
                        'slice (design_notes sections 20/43/54)'
                        + ('' if _gran == 'article' else
                           '; the article unit is the independent corpus '
                           'unit — report it when the cache/tokenizer can '
                           'supply it'))
    print(f'[p5t stats] admitted {len(admitted)}/{len(manifest["arms"]) * _n} frozen (arm, seed) cells on slice {str(unified_fp)[:12]}…; {len(_fam)} frozen primary comparison(s); {len(_rej)} survive Holm; {len(_conf)} also match the pre-registered direction')
    _rob_s = ((stats.get('robustness') or {}).get('summary') or {})
    if _rob_s:
        print(f'[p5t stats] robustness (post-hoc, does NOT change the verdict): '
              f'raw-significant={_rob_s.get("n_raw_significant")}/{_rob_s.get("n_comparisons")}; '
              f'Holm rejections stable under every one-seed drop='
              f'{_rob_s.get("all_holm_rejections_stable_under_one_seed_drop")}; '
              f'min drop/flip fragility={_rob_s.get("min_drop_fragility")}/'
              f'{_rob_s.get("min_flip_fragility")}; unstable seeds='
              f'{_rob_s.get("unstable_seeds")}')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    stats = L.stamped_analysis('finaltest_stats', stats, outdir=(os.path.dirname(out) or '.'), name=os.path.basename(out))
    print(f'[p5t stats] wrote {out}')
    return stats

def _ppl_at_len(rec, Ln):
    """Usable, untruncated PPL for one (record, length), else None."""
    bl = (rec or {}).get('by_len') or {}
    c = bl.get(Ln, bl.get(str(Ln)))
    if (isinstance(c, dict) and c.get('truncated') is False
            and L.ppl_is_usable(c.get('ppl'))):
        return float(c['ppl'])
    return None


def _paired_len_deltas(recs_a, recs_b, common, Ln):
    dl, used = [], []
    for s in common:
        pa, pb = _ppl_at_len(recs_a.get(s), Ln), _ppl_at_len(recs_b.get(s), Ln)
        if pa is None or pb is None:
            continue
        dl.append(pa - pb)
        used.append(int(s))
    return dl, used


def _signflip_power_table(abs_d, m_family, alpha, qs, expected_sign):
    """Exact conditional power of the sign-flip test for fixed |d_i|.

    Enumerates all 2**n sign patterns of the observed validation deltas.  The
    two-sided p of a candidate pattern is its rank in that same sign-flip null;
    Holm is screened conservatively as min(1, m*p).  Power at q is the
    q**k (1-q)**(n-k) mass of patterns that reject *and* point in the
    pre-registered direction.  This is the simulation the manifest asks for
    before the one-shot test; it is exact for n<=13 and refuses larger n.
    """
    abs_d = np.asarray([abs(float(x)) for x in abs_d], dtype=float)
    n = int(abs_d.size)
    if n < 2:
        raise ValueError('power needs at least two paired deltas')
    if n > 13:
        raise ValueError(f'power enumeration is capped at n<=13 (got {n}); split the panel or add a Monte-Carlo path consciously')
    total = 1 << n
    masks = np.arange(total, dtype=np.int64)[:, None]
    bits = (1 << np.arange(n, dtype=np.int64))[None, :]
    signs = np.where((masks & bits) != 0, 1.0, -1.0)
    sums = signs @ abs_d
    abs_sums = np.abs(sums)
    # p[j] = P_null(|sum| >= |sum_j|), the exact two-sided sign-flip p of the
    # pattern whose observed signed sum is sums[j].
    p = (abs_sums[None, :] >= abs_sums[:, None] - 1e-12).mean(axis=1)
    p_holm = np.minimum(1.0, p * float(m_family))
    k_plus = np.asarray([bin(i).count('1') for i in range(total)], dtype=int)
    if expected_sign == 'negative':
        sign_ok = sums < 0
        k_expected = n - k_plus
    elif expected_sign == 'positive':
        sign_ok = sums > 0
        k_expected = k_plus
    else:
        sign_ok = np.ones(total, dtype=bool)
        k_expected = np.maximum(k_plus, n - k_plus)
    power = {}
    for q in qs:
        q = float(q)
        prob = (q ** k_expected) * ((1.0 - q) ** (n - k_expected))
        power[f'q={q:.2f}'] = float((prob * ((p_holm < alpha) & sign_ok)).sum())
    min_same = None
    for s in range(n + 1):
        sel = (k_expected == s) & sign_ok
        if sel.any() and float(p_holm[sel].min()) < alpha:
            min_same = int(s)
            break
    return {'n': n, 'enumerated': total, 'm_family': int(m_family),
            'alpha': float(alpha), 'expected_sign': expected_sign,
            'power': power, 'min_same_sign_for_alpha': min_same,
            'min_attainable_p_holm': float(p_holm.min()),
            'note': ('exact enumeration over the 2**n sign patterns of the observed |d_i|; '
                     'Holm screened as min(1, m*p); power conditions on these magnitudes, '
                     'not on an assumed effect size')}


def run_power(manifest_path=FREEZE_MANIFEST, summary='results_v10_mech/summary.json',
              out='analysis_v12/power_validation.json'):
    """Zero-GPU pre-test power calibration on the P5C validation deltas.

    The manifest requires this BEFORE the one-shot test access: with the
    observed per-seed |delta| vector, how much probability mass actually lies
    in the reject region at the pre-registered direction probability q?  A
    plan whose power at `statistics.power_q_for_test` is below
    `statistics.min_power_for_test` must add seeds and record a revision
    *before* P5T (V12_ALLOW_LOW_POWER=1 is the explicit override).  Reads only
    the validation summary; never calls load_wikitest.
    """
    manifest = _load_manifest(manifest_path)
    if not os.path.exists(summary):
        raise SystemExit(f'[p5pw] REFUSE: {summary} does not exist — run P5C/P3MT first so the power '
                         'preflight has validation deltas; the test split must not be read to size the panel')
    raw = json.load(open(summary, encoding='utf-8'))
    if not isinstance(raw, dict):
        raise SystemExit(f'[p5pw] REFUSE: {summary} is not a JSON object')
    stat = manifest.get('statistics') or {}
    alpha = float(stat.get('alpha', 0.05))
    comps = list(manifest['primary_comparisons'])
    m_family = len(comps)
    qs = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    power_q = float(stat.get('power_q_for_test', 0.90))
    if f'q={power_q:.2f}' not in {f'q={q:.2f}' for q in qs}:
        qs.append(power_q)
    min_power = float(stat.get('min_power_for_test', 0.80))
    require = bool(stat.get('require_power_gate'))
    report = {'kind': 'power_validation', 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
              'manifest': manifest_path, 'manifest_sha256': manifest.get('manifest_sha256'),
              'summary': summary, 'alpha': alpha, 'm_family': m_family,
              'power_q_for_test': power_q, 'min_power_for_test': min_power,
              'require_power_gate': require, 'comparisons': {}}
    _seed_whitelist = {int(s) for s in manifest['seeds']}
    all_pass = True
    for comp in comps:
        name = comp.get('name')
        a, b = comp.get('a'), comp.get('b')
        Ln = int(str(comp.get('endpoint', 'by_len.ppl@4096')).rsplit('@', 1)[-1])
        exp = comp.get('expected_sign')
        exp = str(exp).lower() if exp is not None else None
        recs_a, recs_b = {}, {}
        skipped = 0
        for key, rec in raw.items():
            if not isinstance(rec, dict) or not isinstance(key, str) or '::' not in key:
                continue
            v, _sep, s = key.partition('::')
            if v not in (a, b):
                continue
            if not L.result_is_current(rec, V.CKPT_CODE, 'by_len'):
                skipped += 1
                continue
            try:
                seed = int(str(s).replace('seed', '').replace('s', ''))
            except (TypeError, ValueError):
                skipped += 1
                continue
            if seed not in _seed_whitelist:
                # The frozen sample size/seed set is part of the
                # pre-registration; an extra legacy seed must not enlarge the
                # power panel (and a duplicate key spelling must not silently
                # replace an existing record).
                skipped += 1
                continue
            _dst = recs_a if v == a else recs_b
            if seed in _dst:
                skipped += 1
                continue
            _dst[seed] = rec
        common = sorted(set(recs_a) & set(recs_b))
        dl, used = _paired_len_deltas(recs_a, recs_b, common, Ln)
        cell = {'endpoint': f'by_len.ppl@{Ln}', 'n': len(dl), 'seeds': used,
                'skipped_stale_records': skipped}
        if len(dl) >= 2 and exp in ('positive', 'negative'):
            obs = L.exact_signflip_p(dl)
            mean = float(np.mean(dl))
            tab = _signflip_power_table(np.abs(dl), m_family, alpha, qs, exp)
            sign_ok = (mean < 0) if exp == 'negative' else (mean > 0)
            key = f'q={power_q:.2f}'
            cell.update({'deltas': [float(x) for x in dl], 'mean': mean,
                         'p_exact_signflip': float(obs['p']),
                         'p_holm_screen': float(min(1.0, obs['p'] * m_family)),
                         'observed_sign_ok': sign_ok, 'power': tab})
            cell['power_ok'] = bool(cell['power']['power'].get(key, 0.0) >= min_power
                                    and sign_ok)
            all_pass &= cell['power_ok']
        elif len(dl) >= 2:
            # A non-directional mechanism readout has no "probability of the
            # pre-registered direction"; it is reported descriptively and must
            # not block the directional panel's power gate.
            obs = L.exact_signflip_p(dl)
            cell.update({'deltas': [float(x) for x in dl], 'mean': float(np.mean(dl)),
                         'p_exact_signflip': float(obs['p']),
                         'p_holm_screen': float(min(1.0, obs['p'] * m_family)),
                         'observed_sign_ok': None, 'power': None, 'power_ok': True,
                         'note': 'non-directional readout; excluded from the directional power gate'})
        else:
            cell['power_ok'] = False
            cell['error'] = f'fewer than two complete validation pairs at L{Ln}'
            all_pass = False
        # In-distribution readout on the SAME validation records.  This is the
        # 512-token contrast the frozen test never makes primary; it is
        # descriptive (m=1) and makes a reversal visible before test.
        dl512, used512 = _paired_len_deltas(recs_a, recs_b, common, 512)
        if len(dl512) >= 2:
            o512 = L.exact_signflip_p(dl512)
            cell['in_distribution_L512'] = {
                'n': len(dl512), 'seeds': used512, 'mean': float(np.mean(dl512)),
                'p_exact_signflip': float(o512['p']),
                'expected_sign_ok': (None if exp not in ('positive', 'negative')
                                     else bool((np.mean(dl512) < 0) if exp == 'negative' else (np.mean(dl512) > 0))),
                'descriptive_only': True, 'm_family': 1,
                'note': 'validation, in-distribution descriptive readout; not part of the frozen test family'}
        report['comparisons'][name] = cell
    report['all_pass'] = bool(all_pass)
    report['status'] = 'pass' if all_pass else 'fail'
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    L.stamped_analysis('power_validation', report, outdir=(os.path.dirname(out) or '.'),
                       name=os.path.basename(out))
    print(f'[p5pw] power preflight: {report["status"]} (q={power_q:.2f}, min={min_power:.2f}) -> {out}')
    for _k, _v in report['comparisons'].items():
        _pw = ((_v.get('power') or {}).get('power') or {}).get(f'q={power_q:.2f}')
        print(f'  [p5pw] {_k}: n={_v.get("n")} power={_pw if _pw is not None else "n/a"} '
              f'ok={_v.get("power_ok")}')
    allow = os.environ.get('V12_ALLOW_LOW_POWER', '').strip().lower() in ('1', 'true', 'yes', 'on')
    if require and not all_pass and not allow:
        raise SystemExit(f'[p5pw] REFUSE: the pre-test power calibration is below {min_power:.2f} at q={power_q:.2f}; '
                         'add seeds and record the revision in confirmation_frozen.json BEFORE touching test '
                         '(V12_ALLOW_LOW_POWER=1 is the explicit override)')
    if require and not all_pass:
        print('[p5pw] WARNING: V12_ALLOW_LOW_POWER=1 — running an under-powered frozen panel; '
              'the report must say so')
    return report


def v12_analysis(out='analysis_v12/stats.json'):
    """Zero-GPU aggregate: the shared v10 analysis over the extended probe +
    the frozen final-test statistics (when they exist on disk)."""
    V10.v10_analysis(out=out)
    st = json.load(open(out, encoding='utf-8'))
    if os.path.exists(os.path.join(TEST_OUTDIR, 'bylen.json')):
        st['finaltest'] = finaltest_stats()
        # this file's LAST writer is v12_analysis: re-stamp so the trace names
        # the code that produced the final content, not the earlier v10 pass
        st = L.stamped_analysis('v12_analysis', st, outdir=(os.path.dirname(out) or '.'), name=os.path.basename(out))
    return st

def _fmt_pm(cell):
    if isinstance(cell, dict):
        if cell.get('mean') is None:
            return '—'
        return f'{cell["mean"]:.2f} ±{cell.get("std", 0.0):.2f}'
    return f'{cell:.2f}'

def build_report(out='REPORT_v12.md', stats_p='analysis_v12/stats.json'):
    v12_analysis(out=stats_p)
    st = json.load(open(stats_p, encoding='utf-8'))
    ft = st.get('finaltest') or {}
    lines = []
    A = lines.append
    A('# CSA / HCA 受控机制研究 — v12 确认报告（冻结清单 + 独立 test split）')
    A('')
    A(f'> 生成时间：{time.strftime("%Y-%m-%d %H:%M:%S")}')
    A('> 参考论文：arXiv:2606.19348。**范围声明**：小规模 WikiText-103 上的受控研究，不是 DeepSeek-V4 的全量复现；结论不外推到其规模、语料、长上下文训练或生产系统。')
    A('> 说明：本报告全部数字由 `v12_supp.py report` 从 `results_lm_test/`、`results_v10_mech/`、`analysis_v12/` 的落盘产物计算得到，无手填数值。')
    A('')
    A('本驱动的确认性协议：v7→v11 的全部面板与 v10/v12 的机制扩展属于**探索性**证据；本报告只报告**冻结后**在 held-out test split（wiki.test.raw，其它 phase 不读）上的一次性确认评估。')
    A('')
    A('---')
    A('')
    A('## 冻结清单（预注册）')
    A('')
    m = ft.get('manifest') or {}
    if not m and os.path.exists(FREEZE_MANIFEST):
        try:
            _mc = json.load(open(FREEZE_MANIFEST, encoding='utf-8'))
            m = {'frozen_at': _mc.get('frozen_at'), 'manifest_sha256': None, 'arms': _mc.get('arms'), 'seeds': _mc.get('seeds'), 'code_semantics_expected': _mc.get('code_semantics_expected'), 'statistics': _mc.get('statistics'), 'scope': _mc.get('scope'), 'revision': _mc.get('revision')}
        except Exception:
            pass
    if m:
        _sha = m.get('manifest_sha256') or (_file_sha256(FREEZE_MANIFEST) if os.path.exists(FREEZE_MANIFEST) else None)
        A(f'- 冻结时间：{m.get("frozen_at", "?")}；清单 sha256：`{str(_sha)[:16]}…`（结果记录与冻结锁均绑定该哈希）；代码语义戳：`{m.get("code_semantics_expected", "?")}`（运行期校验一致才放行）。')
        A(f"- 冻结臂：{', '.join((f'`{v}`' for v in m.get('arms', [])))}；种子：{m.get('seeds')}。")
        _stat = m.get('statistics') or {}
        A(f"- 检验：{_stat.get('test', 'exact two-sided sign-flip')}；n={_stat.get('n_seeds', '?')}（p 值分辨率下限 {_stat.get('p_floor', float('nan'))}）；族 = primary_comparisons；校正 = {_stat.get('correction', 'holm step-down')}。")
        A(f"- 决策规则：{_stat.get('decision_rule', '—')}")
        _rev = m.get('revision') or {}
        if isinstance(_rev, dict) and _rev.get('history'):
            A(f"- 修订史：{'；'.join((f"{_r.get('date', '?')} {_r.get('change', '?')}" for _r in _rev['history']))}")
        _pw = ft.get('power_first_holm_gate') or {}
        if _pw:
            A(f"- 功效注记（第一道 Holm 门，逐种子同向概率 q）：{'；'.join((f"{k}→{v:.2f}" for k, v in _pw.items()))}——功效对 q 敏感，q 假设来源见清单 statistics.power_note；数学上可拒绝 ≠ 已足功效。")
        _scope = m.get('scope') or {}
        if isinstance(_scope, dict) and _scope:
            A(f"- 范围声明：{_scope.get('confirmatory_claims', '')} {_scope.get('not_covered', '')} {_scope.get('required_for', '')}")
        A('')
        A('| 冻结的主比较 |')
        A('|---|')
        try:
            _mc = json.load(open(FREEZE_MANIFEST, encoding='utf-8'))
            for c in _mc.get('primary_comparisons', []):
                A(f"| {c.get('name', '?')} — {c.get('a')} vs {c.get('b')} @ {c.get('endpoint', '?')} |")
        except Exception:
            A('| （冻结清单不可读——见运行日志） |')
        A('')
    else:
        A(f'**尚未运行 P5T**：`{TEST_OUTDIR}/bylen.json` 不存在（或为空）。前置：`python v12_supp.py phase P5C`（覆盖全部六个冻结臂、种子取自清单），再跑 `python v12_supp.py phase P5T`（首次评估会写入冻结锁 `.freeze_lock.json`，此后清单不可再改）。')
        A('')
    A('---')
    A('')
    A('## P5T 独立最终测试（test split，冻结终点）')
    A('')
    # The report must use the SAME frozen alpha as finaltest_stats, not a
    # hard-coded 0.05 that can disagree with the decision rule.
    try:
        _report_alpha = float(ft.get('alpha', (m.get('statistics') or {}).get('alpha', 0.05)))
    except (TypeError, ValueError, OverflowError):
        _report_alpha = 0.05
    bl = ft.get('by_length') or {}
    if bl:
        _lens = sorted({int(k) for v in bl.values() for k in v})
        A('| 臂 | params | n_seq | ' + ' | '.join((f'PPL@{Ln}' for Ln in _lens)) + ' |')
        A('|---|---|---|' + '---|' * len(_lens))
        for v in sorted(bl):
            row = []
            _p0 = None
            _n0 = None
            for Ln in _lens:
                e = bl[v].get(str(Ln))
                if isinstance(e, dict):
                    if _p0 is None:
                        _p0 = e.get('params')
                    if _n0 is None:
                        _n0 = e.get('n_seqs')
                row.append(f'{e["mean"]:.2f} (n={e["n"]})' if e else '—')
            A('| `%s` | %s | %s | ' % (v, _p0 if _p0 is not None else '—', _n0 if _n0 is not None else '—') + ' | '.join(row) + ' |')
        A('')
    else:
        A(f'**尚未运行 P5T**：`{TEST_OUTDIR}/bylen.json` 不存在（或为空）。前置：`python v12_supp.py phase P5C`（现覆盖全部六个冻结臂、种子数取自清单；主臂 0-5 已由 P3MT/P4MT 训练，resume 自动跳过），然后 `python v12_supp.py phase P5T`。冻结清单已入库（见上），随时可以执行；执行前请勿据任何 validation 读数调整冻结内容。')
        A('')
    comps = ft.get('comparisons') or {}
    if comps:
        A('| 冻结主比较（test split，配对） | n | 覆盖完整 | 参数匹配 | 初始化配对 | Δ mean±std | Δ 95% CI（文章/文档 token 池化 cluster / 行 bootstrap） | d_z | p (exact) | p 下限 | Holm p | Holm 拒绝 | 方向符合 |')
        A('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        holm = ft.get('holm') or {}
        for k, v in comps.items():
            if 'omitted' in v:
                A(f'| {k} | 0 | — | — | — | —（{v["omitted"]}） | — | — | — | — | — | — | — |')
                continue
            _h = holm.get(k) or {}
            _hp = f'{_h.get("p_holm", float("nan")):.3f}' if _h.get('p_holm') is not None else '—'
            _hr = '✔' if (_h.get('p_holm') is not None and float(_h['p_holm']) < _report_alpha) else '✘'
            _dir = v.get('direction_ok')
            _ds = '—（非方向性）' if _dir is None else ('✔' if _dir else '✘')
            _ipa = v.get('same_init_core_fp')
            _ipa_txt = '—（未验证）' if _ipa is None else ('✔' if _ipa else '✘')
            _dsm = v.get('doc_sampling') or {}
            _dcitxt = _doc_ci_markdown_cell(_dsm)
            _dwin = _dsm.get('doc_a_better_fraction')
            _du = _dsm.get('unit') or _dsm.get('granularity')
            _dunit = {'article': '文章', 'document': '文档'}.get(_du, '行')
            if isinstance(_dwin, (int, float)) and _dcitxt != '—':
                _dcitxt += f'；{100.0 * _dwin:.0f}% {_dunit} Δ<0'
            A(f'| {k} | {v["n"]} | {v.get("coverage_ok")} | {v.get("param_matched")} | {_ipa_txt} | {v["mean"]:+.2f} ± {v.get("std", 0):.2f} | {_dcitxt} | {v.get("cohens_dz", float("nan")):+.2f} | {v["p_exact_signflip"]:.3f} | {v.get("p_floor", float("nan")):.3f} | {_hp} | {_hr} | {_ds} |')
        A('')
        _init_dropped = {k: [d for d in (v.get('dropped_pairs') or [])
                             if 'init_core_fp' in str(d.get('reason', ''))]
                         for k, v in comps.items()}
        _init_dropped = {k: ds for k, ds in _init_dropped.items() if ds}
        if _init_dropped:
            A('（初始化配对：' + '；'.join(
                f'{k} 丢弃种子 {[d["seed"] for d in ds]}'
                for k, ds in sorted(_init_dropped.items()))
              + '。这些种子的两臂 shared-trunk 初始化摘要不同，已按「不是配对」剔除，'
                '不进入精确符号翻转检验；覆盖门因此会拒绝把它们当作完整面板。）')
            A('')
        if any(isinstance((_v.get('doc_sampling') or {}).get('pooled_delta_ppl_ci95'),
                          (list, tuple))
               for _v in comps.values()):
            A('（CI 口径：优先按**文章**（顶层 ` = Title = ` 分组）聚类的 token 池化 Δ '
              'cluster-robust 区间；没有文章映射时退回按行级「文档」聚类。WikiText 的 '
              'raw row 是一行，不是一篇文章，行级 bootstrap 会把同一文章的段落当作独立样本'
              '（伪重复），因此文章级区间是主口径，行列字段仅作冻结契约的兼容证据。'
              '点估计与冻结比较的 token 池化 Δ 相同；同表 `Δ mean±std` 的离散度是种子间'
              '训练随机性，两者不确定性来源不同。等权逐单元 bootstrap 仍保存在 '
              '`analysis_v12/test_stats.json` 的 `per_doc_delta_ppl_ci95`，只作描述，'
              '不再占据主表 CI 列。）')
            A('')
        _conf = ft.get('confirmed')
        if _conf is None:
            _conf = [k for k, v in holm.items() if v.get('reject') and (comps.get(k, {}).get('direction_ok') is True) and (comps.get(k, {}).get('param_matched') is not False) and (comps.get(k, {}).get('same_init_core_fp') is not False) and (comps.get(k, {}).get('coverage_ok') is True)]
        if _conf:
            A(f'**确认结果**：{len(_conf)}/{len(comps)} 个冻结主比较在 Holm 校正后成立**且方向符合预注册**（{sorted(_conf)}）——这是预注册规则下的确认性证据。')
            _frag_conf = []
            for _k in _conf:
                _dsm = comps.get(_k, {}).get('doc_sampling') or {}
                _ui = _dsm.get('unit_influence') or {}
                if _ui.get('fragile_to_one_unit') is True:
                    _u = {'article': '文章', 'document': '文档'}.get(
                        _dsm.get('unit') or _dsm.get('granularity'), '单元')
                    _frag_conf.append(f'{_k}（最敏感{_u} `{_ui.get("most_influential_unit")}`）')
            if _frag_conf:
                A('')
                A('**单单元敏感度提示**：' + '；'.join(_frag_conf) +
                  ' 的确认方向依赖删除单个评测单位；确认判定不变，但引用时必须同时报告'
                  '下方「单元影响诊断」，不能把结论描述为对评测文档稳健。')
            _choice_conf = []
            for _k in _conf:
                _cc = ((comps.get(_k, {}).get('doc_sampling') or {})
                       .get('analysis_choice_concordance') or {})
                if _cc.get('fragile') is True:
                    _choice_conf.append(
                        f'{_k}（{_cc.get("status")}）')
            if _choice_conf:
                A('')
                A('**分析口径敏感度提示**：' + '；'.join(_choice_conf) +
                  ' 的确认方向或精度结论依赖分析口径；确认判定不变，但引用时必须'
                  '同时报告下方「分析口径一致性」表，不能只报单一聚合。')
        else:
            _cov_req = bool((ft.get('coverage_requirement') or {}).get('require_complete_coverage'))
            _sig = [k for k, _h in holm.items()
                    if _h.get('p_holm') is not None and float(_h['p_holm']) < _report_alpha
                    and comps.get(k, {}).get('direction_ok') is True
                    and comps.get(k, {}).get('param_matched') is not False
                    and comps.get(k, {}).get('same_init_core_fp') is not False]
            _cov_blocked = [k for k in _sig if comps.get(k, {}).get('coverage_ok') is False]
            if _cov_req and _cov_blocked:
                A(f'**确认结果**：{sorted(_cov_blocked)} 达到「Holm p < {_report_alpha:g}」且方向符合，'
                  '但至少一个冻结种子没有进入该比较。按 v4 的 `require_complete_coverage`，缺失种子不是随机缺失，'
                  '不确认；该面板只能按实际 n 报告为 descriptive/unconfirmed。')
            else:
                A(f'**确认结果**：没有冻结主比较同时满足「Holm 校正后 p < {_report_alpha:g}」与「方向符合预注册」。按预注册的决策规则，相应主张**不获确认**；探索性面板的正向读数保持「探索性」标签，不得写成结论。')
        _rob_lines = _robustness_markdown(ft.get('robustness') or {})
        if _rob_lines:
            lines.extend(_rob_lines)
        _doc_inf_lines = _doc_influence_markdown(ft)
        if _doc_inf_lines:
            lines.extend(_doc_inf_lines)
        _choice_lines = _analysis_choice_markdown(ft)
        if _choice_lines:
            lines.extend(_choice_lines)
        _nd = ft.get('significant_nondirectional') or []
        if _nd:
            A('')
            A(f'**非方向性机制读数**：{sorted(_nd)} 在 Holm 校正后达到 p<{_report_alpha:g}，但冻结清单未给方向（expected_sign=null），只能作为机制读数，不能计入「方向符合预注册」的确认集合。')
        if ft.get('min_attainable_p_holm') is not None:
            A('')
            A(f"（诊断：n 个种子与 m 元 Holm 族下，最小可达 Holm p = {ft['min_attainable_p_holm']:.4f}。加载期功效门槛已拒绝「数学上不可能确认」的清单；若此处仍 > {_report_alpha:g}，说明有绕过门槛的旧状态，需回查清单与记录的来源。）")
        _dsn = ft.get('doc_sampling_note')
        if _dsn:
            A('')
            A(f"（文档抽样：{_dsn}）")
        A('')
        # Full by-length curve on the same admitted records.  Only the frozen
        # endpoint is confirmatory; L512 is the training length and makes an
        # in-distribution reversal visible instead of hiding it behind L4096.
        _lens_all = sorted({int(x) for _v in comps.values()
                            for x in ((_v.get('by_length_curve') or {}).keys())})
        if _lens_all:
            A('### 全长度曲线（同一批 test 记录；只有冻结终点属于确认族）')
            A('')
            A('| 比较 | ' + ' | '.join(f'L{x}' for x in _lens_all) + ' |')
            A('|---|' + '---|' * len(_lens_all))
            for _k, _v in comps.items():
                _curve = _v.get('by_length_curve') or {}
                _row = []
                for _x in _lens_all:
                    _c = _curve.get(str(_x))
                    if not _c:
                        _row.append('—')
                    else:
                        _row.append(f'{_c["mean"]:+.2f} (p={_c["p_exact_signflip"]:.3f})')
                A(f'| {_k} | ' + ' | '.join(_row) + ' |')
            A('')
            A('> L512 是训练长度，其余长度是外推；L512 与冻结终点方向相反时，冻结终点的“优势”只能读作长度外推鲁棒性，不能读作“CSA 更好”。')
            A('')
            _li_lines = []
            for _k, _v in comps.items():
                _one = _length_interaction_markdown(_k, _v.get('length_interaction'))
                if _one:
                    _li_lines.extend(_one)
            if _li_lines:
                A('### 长度效应修饰（post-hoc；不改变冻结判定）')
                A('')
                lines.extend(_li_lines)
        _claim_no = [k for k, v in comps.items() if v.get('mechanism_claim_allowed') is False]
        if _claim_no:
            A('**主张边界（v4）**：以下比较不自动构成机制证据（`mechanism_claim_allowed=false`）：')
            for _k in _claim_no:
                A(f"- `{_k}`：{comps[_k].get('claim_note') or 'multi-factor / not budget-matched'}")
            A('这些比较即使 Holm 显著，也只能作为 system-level 差异；不得写成“CSA 的 block selection 机制被证实”。')
            A('')
        _cov = ft.get('coverage_requirement') or {}
        if _cov:
            _cov_missing = [k for k, v in comps.items() if v.get('coverage_ok') is False]
            A(f"**覆盖要求**：require_complete_coverage={_cov.get('require_complete_coverage')}；"
              f"缺失完整种子的比较：{sorted(_cov_missing) if _cov_missing else '无'}。"
              + ('覆盖不足的比较不进入 CONFIRMED 集合（缺失不是随机缺失，不能当作较小的 n 使用）。'
                 if _cov.get('require_complete_coverage') else ''))
            A('')
    _pr = ft.get('probe') or {}
    if _pr.get('available') and _pr.get('cells'):
        A('### P5T test split 干扰探针（描述性二次证据，不进入确认性检验）')
        A('')
        A(f"> 与主表同一 held-out 切片（fp `{str(_pr.get('slice_fp'))[:12]}…`），按冻结臂/种子/配方/代码语义准入（{_pr.get('n_admitted')} 条记录，剔除 {_pr.get('n_excluded')} 条）。本表只作描述性二次证据，**不参与 Holm 校正，也不影响确认结论**；validation 探针的探索性读数见 `analysis_v12/stats.json` 的 v10 部分。")
        A('')
        A('| 臂 | arm | L | rho | mode | n | PPL mean |')
        A('|---|---|---|---|---|---|---|')
        for _c in _pr['cells']:
            _amb = '（歧义：多个探针参数化，未合并）' if _c.get('ambiguous_cell') else ''
            A(f"| `{_c['variant']}` | {_c['arm']} | {_c['eval_len']} | {_c['rho']} | {_c['mode']} | {_c['n']} | {_c['mean']:.2f}{_amb} |")
        A('')
        _pctr = _pr.get('contrasts') or {}
        if _pctr:
            A('| 描述性配对差（test split） | n | Δ mean | p (exact) |')
            A('|---|---|---|---|')
            for _k, _v in _pctr.items():
                A(f"| {_k} | {_v.get('n')} | {_v.get('mean', float('nan')):+.2f} | {_v.get('p_exact_signflip', float('nan')):.3f} |")
            A('')
    elif _pr:
        A('**P5T test split 干扰探针**：本次运行未找到可采纳的 test 探针记录，统计已显式标注为不可用。')
        A('')
    A('---')
    A('')
    A('## 产物清单')
    A('')
    A('| 路径 | 内容 |')
    A('|---|---|')
    A(f'| `{TEST_OUTDIR}/training_health.json` | P5H：冻结臂的训练曲线健康门禁（仅诊断，不参与选择） |')
    A(f'| `{TEST_OUTDIR}/bylen.json` | P5T：冻结臂在 test split 的干净 by-length 评估 |')
    A(f'| `{TEST_OUTDIR}/distractor.json` | P5T：test split 上的 uniform 腐坏探针（描述性） |')
    A('| `analysis_v12/stats.json` | v10 扩展探针统计 + 冻结确认性统计 |')
    A('| `analysis_v12/test_stats.json` | 冻结主比较 + Holm 校正（含准入记录、逐文档 doc_sampling 与功效注记） |')
    A('| `confirmation_frozen.json` | 预注册冻结清单（含修订史、范围声明与机器可校验的绑定块） |')
    A('| `results_lm_test/.freeze_lock.json` | 首次测试访问锁（清单哈希 + test 切片指纹；gitignore 的结果目录内，不入库） |')
    A('| `v12_supp.py` | 本驱动（smoke/phase/analysis/report，可断点续跑） |')
    A('')
    L.atomic_write_text(out, '\n'.join(lines) + '\n')
    print(f'[v12 report] wrote {out}')
    return out

def git_push(msg):
    if os.environ.get('V12_NO_PUSH'):
        print(f'[git] push skipped (V12_NO_PUSH): {msg}')
        return True
    return V.git_push(msg)

def schedule_shutdown(delay_s=120):
    if os.environ.get('V12_NO_SHUTDOWN'):
        print('[v12] shutdown suppressed (V12_NO_SHUTDOWN)')
        return
    subprocess.Popen(['bash', '-c', f'sleep {delay_s}; shutdown -h now'], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f'[v12] instance shuts down in {delay_s}s.')

def run_full():
    # Fail before the first GPU-second / before P5T can touch test.  This is
    # the same zero-GPU plan check CI runs; on a real box it also verifies that
    # the frozen manifest and this code agree.
    plan_check()
    guard = make_guard()
    guard.report()
    print(f'[v12] headroom {max(guard.remaining_hours(), 0.0):.2f} h left (booked {guard.state["booked_seconds"] / 3600:.2f} h)')
    git_push('v12: confirmation driver (frozen-manifest v3/revision 4: plan gate + coverage + pre-test power gate + freeze-lock binding)')
    import par
    _ctx = par.resolve_ctx(label='v12')
    all_ok = True
    completed = []
    for pname, _kind, _payload, _seeds in PHASES:
        rem = guard.remaining_hours()
        if rem < 0.4166666666666667:
            print(f'[v12] INCOMPLETE: stopping before {pname}: headroom exhausted ({rem:.2f} h)')
            all_ok = False
            break
        if not V.cuda_healthy():
            print(f'[v12] CUDA context poisoned before {pname} — aborting (re-run resumes).')
            all_ok = False
            break
        print(f'\n===== v12 phase {pname} =====')
        try:
            run_phase(pname, guard, par=_ctx)
        except SystemExit as _se:
            # a REFUSE (e.g. freeze-manifest gate) is a controlled stop, not a crash
            print(f'[v12] phase {pname} refused: {_se}')
            all_ok = False
            break
        except Exception:
            traceback.print_exc()
            all_ok = False
            break
        completed.append(pname)
        all_ok &= git_push(f'v12: phase {pname} results')
    incomplete = [p[0] for p in PHASES if p[0] not in completed]
    try:
        build_report()
    except Exception:
        traceback.print_exc()
        all_ok = False
    all_ok &= git_push('v12: stats + REPORT_v12.md (analysis_v12)')
    guard.report()
    if incomplete:
        print(f'\n[v12] INCOMPLETE — phases not completed: {incomplete}.')
        if 'P5T' in incomplete:
            print('[v12] The confirmatory test-split result is NOT available; do not describe this run as complete.')
    else:
        print('\n[v12] ALL PHASES DONE.')
    schedule_shutdown(120 if (all_ok and not incomplete) else 2400)

def _plan_recipe(manifest, arm, mr):
    """Build a P1L-format recipe string carrying the frozen signature.

    This is a SYNTHETIC descriptor used only by the zero-GPU plan check; it
    never enters a result file.  It mirrors the fields `run_lenphase_cell`
    writes so `_recipe_matches_signature` (including the per-arm `mr`) is
    exercised on the same code path as a real checkpoint.
    """
    _sig = _recipe_signature(manifest) or {}
    _nseq = int(((manifest.get('eval') or {}).get('n_seqs_clean')) or 0)
    return (f"p1l_steps{_sig.get('steps', 3000)}_tl{_sig.get('sl', 512)}"
            f"_v{_sig.get('v', 8192)}_bs{_sig.get('bs', 12)}"
            f"_lr{_sig.get('lr', 0.0003)}_nt{_sig.get('nt', 8000000)}"
            f"_wu{_sig.get('wu', 50)}_wd{_sig.get('wd', 0.1)}"
            f"_cl{_sig.get('cl', 0.05)}_dlm{_sig.get('dlm', 10.0)}"
            f"_mr{float(mr):.8g}-csa_fixed_rope_ns{_nseq}"
            f"_dfplan_gc1.0_b0.9,0.999_eps1e-08_tkplan_detplan_trfpplan")


def _archive_plan_snapshot(report, manifest, manifest_path=FREEZE_MANIFEST, outdir='analysis_v12'):
    """Persist what the plan gate approved, for post-hoc audit of an unattended run.

    The old protocol had a human pause between `plan` and the confirmation
    chain.  The automated gate replaces it with machine checks; this snapshot
    is the durable record of exactly those checks at approval time — the four
    fields a human would have eyeballed (arms / seeds / primary comparisons /
    alpha), the manifest sha256, the code stamp and the git tree — so the
    review can happen asynchronously (even after the one-shot P5T access)
    instead of blocking the run.  Local artifact by design (analysis_*/ is
    gitignored; the tamper-evident binding lives in the freeze lock and the
    per-record manifest_sha256).  A write failure must not fail an otherwise
    passed plan — the warning keeps the gap visible.
    """
    if os.path.abspath(str(manifest_path)) != os.path.abspath(FREEZE_MANIFEST):
        # A smoke/test manifest is a plumbing roundtrip, not a pre-registration;
        # only the real freeze gets the durable audit snapshot (otherwise a
        # smoke plan_check would silently overwrite the confirmation record).
        return
    try:
        _git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
        _head = _git.stdout.strip() if _git.returncode == 0 else None
        _stat = (manifest.get('statistics') or {})
        _comps = [{'name': c.get('name'), 'a': c.get('a'), 'b': c.get('b'), 'endpoint': c.get('endpoint')}
                  for c in (manifest.get('primary_comparisons') or []) if isinstance(c, dict)]
        snap = {
            'archived_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'archived_at_epoch': time.time(),
            'purpose': ('audit record of the automated plan gate: what the machine approved '
                        'before P5C training and the one-shot P5T test access'),
            'git_head': _head,
            'code_semantics': L.CODE_SEMANTICS,
            # The git commit alone cannot identify a dirty work tree, a source
            # tarball, or an edit that leaves CODE_SEMANTICS unchanged.  Bind
            # the exact source bytes and the dependency-lock contract that the
            # plan gate approved, so a later P5T/analysis artifact can be
            # compared against this record instead of trusting the stamp.
            'source_tree': L.source_tree_manifest(),
            'dependency_contract': L.dependency_contract(),
            'manifest': manifest_path,
            'manifest_sha256': report.get('manifest_sha256'),
            'human_review': {
                'arms': list(manifest.get('arms') or []),
                'seeds': list(manifest.get('seeds') or []),
                'primary_comparisons': _comps,
                'alpha': _stat.get('alpha', 0.05),
                'min_seeds_for_confirmation': _stat.get('min_seeds_for_confirmation'),
                'decision_rule': _stat.get('decision_rule'),
            },
            'plan': report,
        }
        os.makedirs(outdir, exist_ok=True)
        L.atomic_write_json(os.path.join(outdir, 'plan_snapshot.json'), snap, indent=2)
        print(f'[v12 plan] snapshot archived -> {outdir}/plan_snapshot.json (unattended-run audit record; the human review is now asynchronous and non-blocking)')
    except Exception as _e:
        print(f'[v12 plan] WARNING: plan snapshot not archived ({type(_e).__name__}: {_e}) — the checks passed, but keep the console log as the audit record')


def plan_check(manifest_path=FREEZE_MANIFEST):
    """Zero-GPU pre-flight for the frozen confirmation plan.

    A stamp-only fix would turn P5T from "fails fast" into "burns the
    one-shot test access, then admits 0/48 records".  This check closes that
    hole WITHOUT reading test data: it derives the implementation descriptor
    from the same
    `make_layer_cfgs` + `_uses_batched_eager` code the trainer uses, builds
    one synthetic record per (arm, seed) with the real recipe/admission
    schema, and runs `_admit_confirmation_records` + the power gate.

    It fails if:
      * the manifest stamp does not match this code;
      * any frozen arm's predicted train_knobs violates the implementation
        binding (e.g. a false eager_batch certification);
      * a per-arm `mr` in the manifest disagrees with `variant_mlp_ratio`;
      * any primary comparison's two arms cannot be admitted;
      * the pre-registered minimum n cannot attain Holm significance.

    Returns a report dict.  Raises `SystemExit` with all errors on failure.
    """
    m = _load_manifest(manifest_path)
    errs = []
    notes = []
    impl = _frozen_implementation(m)
    if not impl:
        errs.append('manifest declares no implementation binding (eager_batch/amp_dtype); manifest v3 must bind the implementation')
    sig = _recipe_signature(m) or {}
    arms = list(m['arms'])
    seeds = [int(s) for s in m['seeds']]
    _stat = m.get('statistics') or {}
    _alpha = float(_stat.get('alpha', 0.05))
    _n_min = int(_stat.get('min_seeds_for_confirmation') or len(seeds))
    _eval = m.get('eval') or {}
    _lens = [int(x) for x in (_eval.get('by_length_ppl') or [])]
    _nseq = int(_eval.get('n_seqs_clean') or 0)
    if len(set(arms)) != len(arms):
        errs.append(f'frozen arms are not unique: {arms}')
    if not _lens:
        errs.append('manifest eval.by_length_ppl is empty; P5T has no endpoint grid')
    if _nseq <= 0:
        errs.append('manifest eval.n_seqs_clean is missing/zero; the frozen row count is part of the slice')
    if _n_min < 2 or _n_min > len(seeds):
        errs.append(f'statistics.min_seeds_for_confirmation={_n_min} is outside [2, {len(seeds)}]')
    _min_holm = L.holm_min_attainable_p(_n_min, len(m['primary_comparisons']), _alpha) if _n_min >= 2 else float('inf')
    if not math.isfinite(_min_holm) or _min_holm > _alpha:
        errs.append(f'the pre-registered minimum n={_n_min} cannot attain Holm significance (min attainable Holm p={_min_holm:.4g} > alpha={_alpha})')
    # Predict the implementation descriptor exactly as the trainer will.
    _dev = type('_PlanCuda', (), {'type': 'cuda'})()
    _bs = int(sig.get('bs', 12))
    _n_layers = int(sig.get('L', 6))
    _amp_norm = L._normalise_amp_mode(impl.get('amp_dtype')) if impl and impl.get('amp_dtype') else None
    _per_arm = ((m.get('binding') or {}).get('per_arm_recipe_signature') or {})
    if _per_arm and set(_per_arm) != set(arms):
        errs.append(f'binding.per_arm_recipe_signature must name exactly the frozen arms {sorted(arms)} (got {sorted(_per_arm)})')
    _records = {}
    _pred = {}
    _arm_cfgs = {}
    for arm in arms:
        try:
            cfgs = L.make_layer_cfgs(_n_layers, arm)
        except Exception as _e:
            errs.append(f'{arm}: make_layer_cfgs failed ({type(_e).__name__}: {_e})')
            continue
        try:
            _eb = L._uses_batched_eager(cfgs, device=_dev, batch_size=_bs)
        except Exception as _e:
            errs.append(f'{arm}: _uses_batched_eager failed ({type(_e).__name__}: {_e})')
            _eb = None
        _tk = L.training_knobs(amp_dtype=_amp_norm, eager_batch=bool(_eb))
        _why = _implementation_reason({'train_knobs': _tk}, impl)
        if _why:
            errs.append(f'{arm}: implementation binding cannot be satisfied by the real dispatch ({_why}; predicted train_knobs={_tk})')
        if impl and 'eager_batch' in impl and _eb is not None and bool(_eb) != bool(impl['eager_batch']):
            errs.append(f'{arm}: predicted eager_batch={bool(_eb)} != frozen {bool(impl["eager_batch"])!r}')
        _expected_mr = None
        if isinstance(_per_arm.get(arm), dict):
            _expected_mr = _per_arm[arm].get('mr')
        if _expected_mr is not None:
            try:
                _actual_mr = L.variant_mlp_ratio(arm, int(sig.get('v', 8192)), d=int(sig.get('d', 256)),
                                                 n_layers=_n_layers, n_heads=int(sig.get('H', 8)),
                                                 d_head=int(sig.get('Dh', 32)), seq_len=int(sig.get('sl', 512)),
                                                 matched={arm}, ref_variant='csa_fixed_rope') if arm in ('full_rope', 'full_sw128_matched_rope', 'full_rope_sink') else 4.0
            except Exception as _e:
                errs.append(f'{arm}: variant_mlp_ratio failed ({type(_e).__name__}: {_e})')
                _actual_mr = None
            if _actual_mr is not None and not math.isclose(float(_actual_mr), float(_expected_mr), rel_tol=1e-6, abs_tol=0.0):
                errs.append(f'{arm}: frozen mr={_expected_mr!r} != computed mr={_actual_mr!r}; the parameter-matching knob is not pinned to this code')
        else:
            _actual_mr = 4.0
            notes.append(f'{arm}: no per-arm mr frozen; the recipe signature will not catch a mr mutation')
        _recipe = _plan_recipe(m, arm, _expected_mr if _expected_mr is not None else _actual_mr)
        _sig_arm = _recipe_signature(m, arm)
        if _sig_arm is None:
            errs.append(f'{arm}: no recipe signature (shared or per-arm) is declared')
        elif not _recipe_matches_signature(_recipe, _sig_arm):
            errs.append(f'{arm}: synthetic recipe does not satisfy the frozen signature {_sig_arm!r} (decoded {_recipe_knobs(_recipe)!r})')
        _pred[arm] = {'eager_batch': (bool(_eb) if _eb is not None else None),
                      'amp_dtype': _tk.get('amp_dtype'),
                      'train_knobs': _tk, 'recipe': _recipe}
        _arm_cfgs[arm] = cfgs
        # Parameter matching is part of the frozen comparison contract: the
        # analysis refuses a pair whose totals differ by >1%, but discovering
        # that only after P5T would waste the access.  Build the real shapes
        # here (CPU) and check it up front.
        try:
            _mr_for_shape = float(_expected_mr) if _expected_mr is not None else float(_actual_mr)
            _shape_model = L.SmallGPT(int(sig.get('v', 8192)), int(sig.get('d', 256)),
                                      _n_layers, int(sig.get('H', 8)), int(sig.get('Dh', 32)),
                                      int(sig.get('sl', 512)), cfgs, mlp_ratio=_mr_for_shape)
            _pred[arm]['params'] = int(L.count_params(_shape_model))
            # The ACTUAL optimisation objective of this arm (comp_reg / delta-LR
            # / boundary policy, read off the built model, not the arm name) —
            # the single-factor verdict below compares the two arms' objectives
            # through the same descriptor the record-level gate uses.
            _pred[arm]['train_objective'] = L.train_objective(
                _shape_model, comp_lambda=float(sig.get('cl', 0.05)),
                delta_lr_mult=float(sig.get('dlm', 10.0)))
            del _shape_model
        except Exception as _e:
            errs.append(f'{arm}: could not compute the parameter count for the frozen shape ({type(_e).__name__}: {_e})')
            _pred[arm]['params'] = None
        for s in seeds:
            _records[f'{arm}::s{s}'] = {
                'variant': arm, 'seed': int(s), 'split': 'test',
                'by_len': {str(Ln): {'ppl': 100.0, 'n_tok': int(_nseq) * int(Ln - 1),
                                     'n_seq': int(_nseq), 'ppl_per_seq': [100.0] * int(_nseq),
                                     'truncated': False}
                           for Ln in _lens},
                'data_fp': 'plan-test-slice-fp', 'train_fp': 'plan-train-fp',
                '_code': L.code_stamp(), 'manifest_sha256': m.get('manifest_sha256'),
                'recipe': _recipe, 'train_knobs': _tk, 'n_seq': int(_nseq),
                'params': 11089904,
            }
    try:
        admitted, _fp = _admit_confirmation_records(_records, m)
    except SystemExit as _e:
        errs.append(f'admission run raised: {_e}')
        admitted = {}
    if len(admitted) != len(arms) * len(seeds):
        errs.append(f'admission accepted {len(admitted)}/{len(arms) * len(seeds)} synthetic frozen cells — P5T would burn the test access and then admit only this many real records')
    _comp_report = {}
    for comp in m['primary_comparisons']:
        _a, _b = comp.get('a'), comp.get('b')
        try:
            _Ln = int(str(comp.get('endpoint', 'by_len.ppl@4096')).rsplit('@', 1)[-1])
        except (TypeError, ValueError, OverflowError):
            _Ln = None
        if _Ln is None:
            errs.append(f'comparison {comp.get("name")!r} has an unparseable endpoint {comp.get("endpoint")!r}; the frozen by-length endpoint must be `...@<integer>`')
        _missing = [x for x in (_a, _b) if x not in arms]
        if _missing:
            errs.append(f'comparison {comp.get("name")!r} names arms outside the frozen list: {_missing}')
        if _Ln is not None and _lens and _Ln not in _lens:
            errs.append(f'comparison {comp.get("name")!r} endpoint L{_Ln} is not in eval.by_length_ppl {_lens}')
        _low = [s for s in seeds if (_a, s) not in admitted or (_b, s) not in admitted]
        if _low:
            errs.append(f'comparison {comp.get("name")!r} cannot form all {len(seeds)} pairs; missing seeds {_low}')
        _pa = (_pred.get(_a) or {}).get('params')
        _pb = (_pred.get(_b) or {}).get('params')
        _rel = None
        if _pa and _pb:
            _rel = abs(_pa - _pb) / max(_pa, _pb, 1)
            if _rel > 0.01:
                errs.append(f'comparison {comp.get("name")!r} is NOT parameter-matched at the frozen shapes ({_pa} vs {_pb}, rel diff {_rel:.4%} > 1%) — the analysis would refuse it after the test access')
        # ---- single-factor verdict vs the mechanism-claim flag (review-7) ----
        # A frozen comparison may carry `mechanism_claim_allowed=true` only if
        # it is verifiably single-factor: the attention-cfg diff isolates ONE
        # declared intervention family (no `kind` swap), the two arms' ACTUAL
        # objectives match, and the parameter totals are matched.  A True flag
        # on a multi-factor contrast is a design defect and refuses the plan;
        # a False flag on a config-single-factor contrast stays allowed (the
        # caveat may rely on read-budget/interpretation constraints outside
        # the attention cfg — conservative direction is never an error).
        _diff_keys = None
        if _a in _arm_cfgs and _b in _arm_cfgs:
            _diff_keys = set()
            for _ca, _cb in zip(_arm_cfgs[_a], _arm_cfgs[_b]):
                _diff_keys.update(L.layer_cfg_diff(_ca, _cb).keys())
        _obj_reason = L._objective_diff({'train_objective': (_pred.get(_a) or {}).get('train_objective')},
                                        {'train_objective': (_pred.get(_b) or {}).get('train_objective')})[0] \
            if ((_pred.get(_a) or {}).get('train_objective') is not None
                and (_pred.get(_b) or {}).get('train_objective') is not None) else 'unverifiable train_objective (arm model not built)'
        _sf = bool((_diff_keys is not None) and L.cfg_keys_single_factor(_diff_keys)
                   and (_obj_reason is None)
                   and (_rel is None or _rel <= 0.01))
        _flag = comp.get('mechanism_claim_allowed')
        if _flag is True and not _sf:
            _why = (f'cfg diff keys {sorted(_diff_keys)}'
                    if _diff_keys is not None else 'cfg diff not computable')
            if _obj_reason is not None:
                _why += f'; objective mismatch ({_obj_reason})'
            if _rel is not None and _rel > 0.01:
                _why += f'; parameter totals differ by {_rel:.4%}'
            errs.append(f'comparison {comp.get("name")!r} declares mechanism_claim_allowed=true but is NOT single-factor ({_why}) — a mechanism claim requires the attention-cfg diff to isolate one declared intervention family, matching objectives and matched parameters; re-derive the flag or re-design the arms BEFORE any test access')
        if _flag is not True and _sf:
            notes.append(f'comparison {comp.get("name")!r} is config-single-factor ({sorted(_diff_keys)}) but carries a conservative mechanism_claim_allowed={_flag!r}; keeping the conservative flag (the caveat may rest on constraints outside the attention cfg, e.g. read budget or interpretation)')
        _comp_report[comp.get('name')] = {'a': _a, 'b': _b, 'endpoint_L': _Ln, 'pairs_planned': len(seeds) - len(_low), 'params_a': _pa, 'params_b': _pb, 'param_rel_diff': _rel,
                                         'cfg_diff_keys': (sorted(_diff_keys) if _diff_keys is not None else None),
                                         'objective_matched': (_obj_reason is None),
                                         'single_factor': _sf,
                                         'mechanism_claim_allowed': _flag,
                                         'claim_flag_consistent': (_flag is not True) or _sf}
    # ---- design preflight: budgets + evidence level (zero GPU) ----------
    # Single-factor / parameter / objective checks above answer "is the
    # attention cfg the only declared factor?".  They do NOT answer "do the
    # arms share a token and read/compute budget?" or "is this manifest allowed
    # to call the result confirmatory?".  Both are design questions and must
    # refuse the plan before any GPU spend or test access.
    _design = {}
    try:
        _ledger = L.comparison_budget_ledger(m)
        _ev = L.evidence_level_audit(m, manifest_path=manifest_path, panel_paths=[])
        _design = {'budget_ledger': _ledger, 'evidence_level': _ev}
        if _ledger.get('status') == 'fail':
            errs.extend(f'budget ledger: {v}' for v in (_ledger.get('violations') or []))
        if _ev.get('status') == 'fail':
            errs.extend(f'evidence level: {v}' for v in (_ev.get('violations') or []))
        notes.extend(f'budget ledger: {w}' for w in (_ledger.get('warnings') or []))
        notes.extend(f'evidence level: {w}' for w in (_ev.get('warnings') or []))
    except Exception as _e:
        errs.append(
            f'the design preflight could not run ({type(_e).__name__}: {_e}); a '
            'confirmation needs a verifiable token/parameter/read-budget ledger '
            'and evidence-level declaration')
    report = {'ok': not errs, 'manifest': manifest_path, 'manifest_sha256': m.get('manifest_sha256'),
              'code_semantics': L.CODE_SEMANTICS, 'n_arms': len(arms), 'n_seeds': len(seeds),
              'n_min_for_confirmation': _n_min, 'min_attainable_p_holm': _min_holm,
              'predicted': _pred, 'admitted': len(admitted), 'comparisons': _comp_report,
              'design_preflight': _design,
              'errors': errs, 'notes': notes}
    if errs:
        print('[v12 plan] REFUSED — the frozen plan is not runnable end-to-end:')
        for _e in errs:
            print(f'  - {_e}')
        raise SystemExit('[v12 plan] FAILED: fix the plan/records BEFORE any P5C training or P5T test access')
    print(f'[v12 plan] OK: {len(admitted)}/{len(arms) * len(seeds)} synthetic records admitted; {len(_comp_report)} comparisons form all pairs; min n={_n_min} (min Holm p={_min_holm:.4g} <= alpha={_alpha})')
    for _a, _p in _pred.items():
        print(f'  [plan] {_a:32s} eager_batch={_p["eager_batch"]} amp={_p["amp_dtype"]} mr={(_per_arm.get(_a) or {}).get("mr", 4.0)} params={_p.get("params")}')
    for _n in notes:
        print(f'  [plan][note] {_n}')
    _archive_plan_snapshot(report, m, manifest_path)
    globals()['_PLAN_DONE'] = True
    return report


def run_plan(manifest_path=FREEZE_MANIFEST):
    return plan_check(manifest_path)


DYNAMIC_MANIFEST = 'confirmation_frozen_dynamic.json'

def dynamic_plan_check(manifest_path=DYNAMIC_MANIFEST):
    """Zero-GPU design preflight for the DYNAMIC/HCA draft.

    Deliberately does NOT call `_load_manifest`: that loader refuses
    `status=draft`, which is exactly the state a design record must be in
    before its disjoint test source and power gate exist.  This check audits
    the draft as a DESIGN (hypotheses + dedicated controls + budget ledger +
    evidence level) and prints why it is not runnable yet.  Passing it never
    upgrades the draft to a confirmation.
    """
    if not os.path.exists(manifest_path):
        raise SystemExit(
            f'[v12 dynamic-plan] REFUSE: no dynamic design record at '
            f'{manifest_path!r}; write the hypotheses and dedicated controls '
            'before adding dynamic/HCA arms')
    try:
        with open(manifest_path, encoding='utf-8') as _f:
            m = json.load(_f)
    except Exception as _e:
        raise SystemExit(
            f'[v12 dynamic-plan] REFUSE: {manifest_path} is not readable JSON '
            f'({type(_e).__name__}: {_e})') from None
    d = L.dynamic_design_audit(m, manifest_path=manifest_path)
    led = L.comparison_budget_ledger(m)
    ev = L.evidence_level_audit(m, manifest_path=manifest_path, panel_paths=[])
    errs = list(d.get('violations') or []) + list(led.get('violations') or []) \
        + list(ev.get('violations') or [])
    report = {'ok': not errs, 'manifest': manifest_path,
              'declared_status': m.get('status'),
              'evidence_level': ev.get('manifest_level'),
              'runnable': d.get('runnable'),
              'promotion_ready': d.get('promotion_ready'),
              'power': d.get('power'),
              'calibration': d.get('calibration'),
              'design_cost': led.get('design_cost'),
              'design': d, 'budget_ledger': led, 'evidence_level_audit': ev,
              'errors': errs}
    if errs:
        print('[v12 dynamic-plan] REFUSED — the dynamic design is not coherent:')
        for _e in errs:
            print(f'  - {_e}')
        raise SystemExit(
            '[v12 dynamic-plan] FAILED: fix the dynamic design BEFORE any '
            'dynamic/HCA training or any disjoint-test access')
    print(f"[v12 dynamic-plan] OK: design coherent; evidence_level={ev.get('manifest_level')}; "
          f"hypotheses={sorted(d.get('hypotheses') or {})}; "
          f"runnable={d.get('runnable')}; promotion_ready={d.get('promotion_ready')}")
    _pw = d.get('power') or {}
    _cal = d.get('calibration') or {}
    _cost = led.get('design_cost') or {}
    if _pw:
        print(f"  [dynamic-plan] power: n={_pw.get('n_seeds')} m={_pw.get('family_m')} "
              f"projected@{_pw.get('power_q_for_test')}={_pw.get('projected_power')} "
              f"min={_pw.get('min_power_for_test')}")
    if _cal:
        print(f"  [dynamic-plan] read calibration: status={_cal.get('status')} "
              f"expected_topk={_cal.get('expected_topk')} arm={_cal.get('arm')}")
    if _cost:
        print(f"  [dynamic-plan] design cost: {_cost.get('n_runs')} runs, "
              f"{_cost.get('consumed_tokens_total')} consumed tokens (scheduling aid)")
    for _w in (d.get('warnings') or []) + (led.get('warnings') or []) + (ev.get('warnings') or []):
        print(f'  [dynamic-plan][note] {_w}')
    if not d.get('promotion_ready'):
        print('  [dynamic-plan] DRAFT: this is a design record, NOT a confirmation. '
              'Reserve a disjoint test source, measure the realized dynamic block '
              'length to calibrate the read-matched fixed top-k, and pass the '
              'pre-test power gate before freezing.')
    return report


def run_smoke():
    import copy
    import shutil
    # Smoke must never write into the frozen wt103_cache: a differently
    # tokenised tok_v5_*.json could make a later cold P5T build fail AFTER the
    # test read.  An isolated cache under the smoke outdir keeps the frozen
    # cache clean.
    os.environ.setdefault('WT103_CACHE_DIR', os.path.join('results_smoke_v12', 'wt103_cache'))
    print('[smoke] 1) tiny frozen-confirmation roundtrip on a synthetic local test split')
    L.set_seed(0)
    shutil.rmtree('results_smoke_v12', ignore_errors=True)
    payload = dict(V10.P3MT_PAYLOAD, outdir='results_smoke_v12', seeds=[0], steps=40, eval_lens=[512, 1024])
    V.run_lenphase(dict(payload, variants=['csa_fixed_rope', 'csa_fixed_rope_allblocks']), guard=None, label='v12 smoke', par=False)
    # a smoke freeze manifest: same schema, tiny arms/seeds, current code stamp.
    # statistics.descriptive_only=1 opts the smoke manifest out of the load-time
    # power gate (n=1 can never confirm — that is fine for a plumbing smoke);
    # binding.training_recipe_signature exercises the structured-recipe admission gate.
    _smoke_manifest = {'frozen_at': time.strftime('%Y-%m-%d'), 'code_semantics_expected': L.CODE_SEMANTICS, 'arms': ['csa_fixed_rope', 'csa_fixed_rope_allblocks'], 'seeds': [0], 'primary_comparisons': [{'name': 'smoke: allblocks_scratch - learned @ L1024', 'a': 'csa_fixed_rope_allblocks', 'b': 'csa_fixed_rope', 'endpoint': 'by_len.ppl@1024'}], 'statistics': {'test': 'exact two-sided sign-flip', 'n_seeds': 1, 'p_floor': 1.0, 'family': 'primary_comparisons', 'correction': 'holm step-down', 'decision_rule': 'smoke only', 'descriptive_only': True}, 'binding': {'training_recipe': {'steps': 40, 'sl': 512, 'bs': 12}}, 'eval': {'by_length_ppl': [512, 1024], 'n_seqs_clean': 4, 'distractor_probe': {'mode': 'uniform', 'rhos': [0.0, 0.5], 'eval_lens': [1024], 'target': 512, 'n_seq': 4}}}
    _mpath = 'results_smoke_v12/frozen.json'
    L.atomic_write_json(_mpath, _smoke_manifest, indent=2)
    # smoke test data: rows from the (tiny local dump or HF) validation split,
    # standing in for the test split so the smoke never depends on wiki.test.raw
    _, val_ids, _, _, _ = L.load_wikitext(1024, 4000000)
    cfg = dict(FINALTEST_CFG, manifest=_mpath, outdir='results_smoke_v12', ckpt_dir='results_smoke_v12/ckpt', mech_summary='results_smoke_v12/summary.json', bylen_summary='results_smoke_v12/bylen.json', probe_summary='results_smoke_v12/distractor.json', n_seq=4, chunk=2, stats_out='results_smoke_v12/test_stats.json', test_ids=np.asarray(val_ids[:4, :1025], dtype=np.int64), smoke=True)
    stats = run_finaltest(cfg)
    got = {r['variant'] for r in json.load(open('results_smoke_v12/bylen.json', encoding='utf-8')).values() if isinstance(r, dict)}
    assert {'csa_fixed_rope', 'csa_fixed_rope_allblocks'} <= got, f'finaltest bylen cells missing: {got}'
    _probe = {}
    if os.path.exists('results_smoke_v12/distractor.json'):
        _probe = json.load(open('results_smoke_v12/distractor.json', encoding='utf-8'))
    _pgot = {(r.get('mode', 'uniform'), r['arm'], float(r['rho'])) for r in _probe.values() if isinstance(r, dict)}
    _pwant = {('uniform', 'learned', 0.0), ('uniform', 'learned', 0.5), ('uniform', 'dense', 0.0), ('uniform', 'dense', 0.5)}
    _pwant = {w for w in _pwant if w[1] != 'dense'}  # full_rope arm is not in the smoke manifest
    assert _pwant <= _pgot, f'finaltest probe cells missing: {_pwant - _pgot}'
    assert 'comparisons' in stats and stats['comparisons'], 'finaltest stats produced no comparison record'
    print(f'  finaltest roundtrip OK ({len(got)} bylen arms, {len(_probe)} probe cells, {len(stats["comparisons"])} frozen comparison(s))')
    print('[smoke] 2) report builder on the smoke artifacts')
    build_report(out='results_smoke_v12/REPORT_v12.md', stats_p='results_smoke_v12/stats.json')
    shutil.rmtree('results_smoke_v12', ignore_errors=True)
    print('\n[smoke] PASSED')

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'full'
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    _MODES = ('full', 'smoke', 'plan', 'dynamic-plan', 'analysis', 'report', 'phase', 'finaltest', 'power', 'auto')
    if mode in ('-h', '--help', 'help'):
        print(__doc__)
        raise SystemExit(0)
    if mode not in _MODES:
        print(f'[v12] unknown mode {mode!r}; expected one of {_MODES} (no argument means "full").  Refusing to start a run.')
        raise SystemExit(2)
    if mode == 'phase' and len(sys.argv) < 3:
        print(f'[v12] mode "phase" needs a phase name, e.g. `python v12_supp.py phase P5C`.  Available: {[p[0] for p in PHASES]}')
        raise SystemExit(2)
    print(f'[v12] mode={mode} repo={REPO} device={DEVICE} ({(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")})')
    try:
        if mode == 'smoke':
            run_smoke()
        elif mode == 'plan':
            run_plan(sys.argv[2] if len(sys.argv) > 2 else FREEZE_MANIFEST)
        elif mode == 'dynamic-plan':
            dynamic_plan_check(sys.argv[2] if len(sys.argv) > 2 else DYNAMIC_MANIFEST)
        elif mode == 'analysis':
            v12_analysis()
        elif mode == 'report':
            build_report()
        elif mode == 'power':
            # zero-GPU: exact conditional power on the P5C validation deltas,
            # before the one-shot test access (never calls load_wikitest)
            run_power(sys.argv[2] if len(sys.argv) > 2 else FREEZE_MANIFEST)
        elif mode == 'phase':
            run_phase(sys.argv[2], make_guard())
            git_push(f'v12: phase {sys.argv[2]} results')
        elif mode == 'finaltest':
            run_finaltest(dict(FINALTEST_CFG))
        elif mode == 'auto':
            # Unattended entry: run_full already runs the zero-GPU plan gate
            # (fail-closed) before any GPU time and archives the approval
            # snapshot; `auto` is the explicit no-human-pause name for it.
            run_full()
        else:
            run_full()
    except Exception:
        traceback.print_exc()
        try:
            git_push('v12: PARTIAL — crashed, see log (re-run resumes)')
        except Exception:
            traceback.print_exc()
        if mode in ('full', 'auto'):
            schedule_shutdown(2400)
        sys.exit(1)
