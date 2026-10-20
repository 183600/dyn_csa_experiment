#!/usr/bin/env python3
"""v12 confirmation suite — 探索/确认分离的确认性实验。

前序版本（v7→v11）的全部面板都围绕同一个 validation split 展开，且多轮
追加种子/比较属于**探索性**证据。本驱动把「确认」独立出来：

  P5C   —— 按冻结清单的种子列表训练全部四个冻结臂（主臂 + 从头训练的
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
     test 指纹），此后清单/test 变更即拒绝（V12_REFREEZE=1 显式改冻结，
     前轮降级为探索性）；清单修订后同 outdir 的旧结果自动作废。
  4. 范围声明（意见四）：确认性主张只覆盖固定分块 CSA 机制
     （清单 scope 块）；动态分块/HCA 的优劣需要各自的预注册。

模式：
  python v12_supp.py smoke            # 端到端自检（微型 ckpt + 冻结清单副本）
  python v12_supp.py plan             # 零 GPU 预检：戳/实现绑定/准入/功效，不读 test
  python v12_supp.py full             # P5C -> P5MP -> P5S2 -> P5S3 -> P5T -> report
  python v12_supp.py phase P5C        # 单阶段（可断点续跑）
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
# 读取（单一事实来源）；P5C 同时覆盖全部四个冻结臂（主臂 0-5 已由
# P3MT/P4MT 训练，resume 自动跳过；主臂 6-11 与对照臂统一在此补齐），
# 否则清单扩种子后没有任何驱动会训练主臂新种子。
P5C_CFG = dict(V10.P3MT_PAYLOAD, seeds=[0, 1, 2, 3, 4, 5], variants=['csa_fixed_rope', 'full_rope'] + list(V10.SCRATCH_CONTROLS),
               # Diagnostic-only validation monitoring: never used for model
               # selection, but P5T refuses the one-shot test access when a
               # frozen arm trained through a non-finite/overfit tail.
               diag_eval_every=250)
P5MP_CFG = dict(V10.PROBE, seeds=[0, 1, 2, 3, 4, 5], variants=['csa_fixed_rope', 'full_rope'] + V10.SCRATCH_CONTROLS, arm_overrides={v: ['learned'] for v in V10.SCRATCH_CONTROLS})
P5S2_CFG = dict(V10.P3SL_TOK_CFG, seeds=[0, 1])
P5S3_CFG = dict(V10.P3SL_FLOP_CFG, seeds=[0, 1])
TEST_OUTDIR = 'results_lm_test'
FREEZE_MANIFEST = 'confirmation_frozen.json'
PHASES = [('P5C', 'lenphase', P5C_CFG, None), ('P5MP', 'probe', P5MP_CFG, None), ('P5S2', 'run', P5S2_CFG, None), ('P5S3', 'run', P5S3_CFG, None), ('P5H', 'health', None, None), ('P5T', 'finaltest', None, None)]
_PLAN_DONE = False

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
            if status != 'healthy':
                unhealthy.append(key)
            del d
            try:
                gc.collect()
            except Exception:
                pass
    report = {'kind': 'training_health', 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
              'manifest': cfg.get('manifest') or FREEZE_MANIFEST,
              'n_arms': len(manifest['arms']), 'n_seeds': len(manifest['seeds']),
              'n_cells': len(rows), 'n_missing': len(missing), 'n_unhealthy': len(unhealthy),
              'missing': missing, 'unhealthy': unhealthy,
              'status': ('pass' if not missing and not unhealthy else 'fail'),
              'note': 'diagnostic only: this gate never selects a checkpoint or changes the frozen plan',
              'cells': rows}
    out = cfg.get('out') or os.path.join(TEST_OUTDIR, 'training_health.json')
    L.atomic_write_json(out, report, indent=2)
    print(f'[p5h] training health: {report["status"]} ({len(unhealthy)} unhealthy, {len(missing)} missing) -> {out}')
    allow = os.environ.get('V12_ALLOW_UNHEALTHY', '').strip().lower() in ('1', 'true', 'yes', 'on')
    if report['status'] != 'pass' and not allow:
        raise SystemExit(f'[p5h] REFUSE: frozen-arm training health is not clean ({len(unhealthy)} unhealthy, {len(missing)} missing); investigate before using the one-shot test access (V12_ALLOW_UNHEALTHY=1 is the explicit override)')
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
FINALTEST_CFG = dict(manifest=FREEZE_MANIFEST, outdir=TEST_OUTDIR, ckpt_dir='results_v10_mech/ckpt', mech_summary='results_v10_mech/summary.json', bylen_summary=os.path.join(TEST_OUTDIR, 'bylen.json'), probe_summary=os.path.join(TEST_OUTDIR, 'distractor.json'), eval_lens=[512, 1024, 2048, 4096], n_seq=16, target=512, chunk=4)

def _load_manifest(path):
    if not os.path.exists(path):
        raise SystemExit(f'[v12] REFUSE: freeze manifest {path} does not exist — a confirmation run requires a PRE-REGISTERED plan (arms, seeds, endpoints, statistics); write the manifest first, then run P5T')
    try:
        m = json.load(open(path, encoding='utf-8'))
    except Exception as e:
        raise SystemExit(f'[v12] REFUSE: freeze manifest {path} cannot be parsed ({type(e).__name__}: {e})')
    if not isinstance(m, dict):
        raise SystemExit(f'[v12] REFUSE: freeze manifest {path} is not a JSON object ({type(m).__name__})')
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
    if _ts and _eval:
        _ts_n = _ts.get('n_seqs')
        _ev_n = _eval.get('n_seqs_clean')
        if (_ts_n is not None and _ev_n is not None
                and _int_field(_ts_n, 'binding.test_slice.n_seqs') != _int_field(_ev_n, 'eval.n_seqs_clean')):
            raise SystemExit(f'[v12] REFUSE: binding.test_slice.n_seqs={_ts_n} disagrees with eval.n_seqs_clean={_ev_n} — the machine-checkable binding block and the evaluation plan must describe the same slice')
    if _eval and _int_field(_eval.get('n_seqs_clean') or 0, 'eval.n_seqs_clean') <= 0:
        raise SystemExit('[v12] REFUSE: eval.n_seqs_clean is missing/zero — the frozen row count is part of the held-out slice, not a runtime default')
    _tokb = (_bind.get('tokenizer') or {}) if isinstance(_bind, dict) else {}
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
    _n = len(m['seeds'])
    _m_family = len(m['primary_comparisons'])
    _min_holm = L.holm_min_attainable_p(_n, _m_family, _alpha)
    if math.isfinite(_min_holm) and _min_holm > _alpha and not _stat.get('descriptive_only'):
        _kmin = L.signflip_min_consistency(_n, _m_family, _alpha)
        raise SystemExit(f'[v12] REFUSE: the frozen plan cannot confirm ANYTHING by construction — with n={_n} seeds, a two-sided exact sign-flip and a Holm family of m={_m_family}, the smallest attainable Holm p is {_min_holm:.4g} > alpha={_alpha} (all {_n} seeds one-sided still gives {_kmin is None and "insufficient" or f"k_min={_kmin} needed"}). This is a resolution ceiling, NOT a conservative design. Revise the plan BEFORE any test-split evaluation (add seeds / shrink the family / change the test) and record the revision in the manifest history. ({_stat.get("descriptive_only")!r})')
    if not isinstance(m.get('seeds'), list) or not all(isinstance(s, int) and (not isinstance(s, bool)) for s in m['seeds']) or len(set(m['seeds'])) != len(m['seeds']):
        raise SystemExit(f'[v12] REFUSE: seeds must be a list of unique integers ({m.get("seeds")!r}) — seed identity is the pairing unit of the whole protocol')
    # v11.128: the declared n-floor is part of the decision rule.  Refuse a
    # floor that is below 2, above the frozen seed count, or mathematically
    # unable to reach Holm significance.
    _n_min = _stat.get('min_seeds_for_confirmation')
    if _n_min is not None:
        _n_min = _int_field(_n_min, 'statistics.min_seeds_for_confirmation')
        if _n_min < 2 or _n_min > _n:
            raise SystemExit(f'[v12] REFUSE: statistics.min_seeds_for_confirmation={_n_min} is outside [2, {_n}] — the declared confirmation floor must be reachable with the frozen seed list')
        _min_holm_floor = L.holm_min_attainable_p(_n_min, _m_family, _alpha)
        if math.isfinite(_min_holm_floor) and _min_holm_floor > _alpha and not _stat.get('descriptive_only'):
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

    v11.128 / manifest v3: an arm-specific signature
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
    The historical ``bool(want) or bool(ckpt) or bool(record)`` rule was
    fail-open (all three empty -> no check at all), which the review flagged.
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


def _frozen_implementation(manifest):
    """Implementation-mode bindings declared by the frozen plan, if any.

    `code_semantics_expected` only pins the source revision; the eager
    row-loop vs batched twin and the AMP dtype are machine/runtime choices.
    Newer manifests can pin them with ``eager_batch_expected`` /
    ``amp_dtype_expected`` (top level, or the same names without the suffix).
    Absent fields leave the historical permissive path unchanged, but when a
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

    v11.128: run the FULL `_load_manifest` validation here, not just a JSON
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


def _check_or_write_freeze_lock(outdir, manifest, test_ids_fp, stage='access',
                                train_fp=None):
    """Persistent first-test-access lock.

    首次测试访问后禁止修改计划：P5T 一旦在 outdir 落下首条记录（或本函数
    被调用），锁文件即记录 manifest sha256
    与 test 切片指纹。之后的 P5T/analysis 若带着**不同的** manifest 哈希
    或不同 test 指纹运行，将被拒绝——除非显式 V12_REFREEZE=1（并大声
    打印该轮降级为探索性的警告）。``train_fp`` 一旦记录，后续训练集
    指纹不一致同样拒绝。"""
    lp = _freeze_lock_path(outdir)
    _sha = manifest.get('manifest_sha256')
    if os.path.exists(lp):
        try:
            lock = json.load(open(lp, encoding='utf-8'))
        except Exception as e:
            raise SystemExit(f'[v12] REFUSE: freeze lock {lp} exists but cannot be parsed ({type(e).__name__}: {e}) — delete it only if you can prove no test-split record was ever written under it')
        _manifest_changed = lock.get('manifest_sha256') != _sha
        _test_changed = (test_ids_fp is not None
                         and lock.get('test_ids_fp') not in (None, test_ids_fp))
        _train_changed = (train_fp is not None
                          and lock.get('train_fp') not in (None, train_fp))
        _refreeze = os.environ.get('V12_REFREEZE', '').strip().lower() in ('1', 'true', 'yes', 'on')
        if (_manifest_changed or _test_changed or _train_changed) and not _refreeze:
            if _manifest_changed:
                raise SystemExit(f'[v12] REFUSE: the test split was already accessed under a DIFFERENT frozen manifest (lock {lp}: {lock.get("manifest_sha256")} != current {_sha}, locked_at={lock.get("locked_at")}). Changing the plan after seeing test data is selection bias by construction. To consciously re-freeze, set V12_REFREEZE=1 — the earlier round must then be reported as exploratory in the plan history.')
            if _test_changed:
                raise SystemExit(f'[v12] REFUSE: the held-out test slice changed after the first access (lock has ids_fp {lock.get("test_ids_fp")}, current {test_ids_fp}) — the test data itself is part of the freeze; investigate before proceeding (V12_REFREEZE=1 consciously re-freezes to a new slice)')
            raise SystemExit(f'[v12] REFUSE: the training corpus changed after the first test access (lock has train_fp {lock.get("train_fp")}, current {train_fp}) — a different training set cannot enter the frozen confirmation (V12_REFREEZE=1 consciously re-freezes to a new corpus)')
        _dirty = False
        if _manifest_changed or _test_changed or _train_changed:
            print(f'[v12] WARNING: V12_REFREEZE=1 — the locked plan/test/train identity changed (manifest {lock.get("manifest_sha256")} -> {_sha}, test_ids_fp {lock.get("test_ids_fp")} -> {test_ids_fp}, train_fp {lock.get("train_fp")} -> {train_fp}); the earlier round is DOWNGRADED to exploratory and this must be recorded in the plan history')
            # Persist the conscious re-freeze: under the new plan the lock must
            # carry the new identity so later runs do not refuse, and the
            # superseded plan is retained for the exploratory record.  The lock
            # is written ONCE below, after every check has passed, so a failure
            # can never leave a half-updated (new plan + stale fingerprints) file.
            prev = {_k: lock.get(_k) for _k in ('manifest_sha256', 'test_ids_fp', 'train_fp', 'locked_at', 'frozen_at')}
            lock['superseded'] = list(lock.get('superseded') or []) + [prev]
            lock.update({'manifest_sha256': _sha,
                         'manifest_path': os.path.abspath(manifest.get('_path') or FREEZE_MANIFEST),
                         'frozen_at': manifest.get('frozen_at'),
                         'locked_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
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
            lock['history'] = list(lock.get('history') or []) + [{'stage': 'accessed', 'at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'test_ids_fp': test_ids_fp}]
            _dirty = True
        if train_fp is not None and lock.get('train_fp') is None:
            lock['train_fp'] = train_fp
            lock['history'] = list(lock.get('history') or []) + [{'stage': 'train-fp', 'at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'train_fp': train_fp}]
            _dirty = True
        if _dirty:
            L.atomic_write_json(lp, lock, indent=2)
            if _manifest_changed or _test_changed or _train_changed:
                print(f'[v12] freeze lock UPDATED: {lp} (manifest {str(_sha)[:12]}…; {len(lock["superseded"])} superseded plan(s) kept)')
            else:
                print(f'[v12] freeze lock updated with the first-access slice/train fingerprint: {lp}')
        return lock
    lock = {'manifest_sha256': _sha, 'manifest_path': os.path.abspath(manifest.get('_path') or FREEZE_MANIFEST), 'frozen_at': manifest.get('frozen_at'), 'locked_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'test_ids_fp': test_ids_fp, 'train_fp': train_fp, 'stage': stage, 'code_semantics': L.code_stamp(), 'history': [{'stage': stage, 'at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'test_ids_fp': test_ids_fp, 'train_fp': train_fp}], 'note': 'first test-split access: the plan may not change after this point (V12_REFREEZE=1 overrides consciously, downgrading the earlier round to exploratory)'}
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
    status = str(health.get('status') or 'unknown')
    if status == 'healthy':
        return None
    if status == 'unmonitored' and allow:
        print(f'[p5t] WARNING: {v} s{seed}: training health is unmonitored; V12_ALLOW_UNHEALTHY=1 permits the test evaluation')
        return None
    msg = f'{v} s{seed}: training health is {status!r} ({health.get("reason") or "no reason recorded"})'
    if allow:
        print(f'[p5t] WARNING: {msg}; V12_ALLOW_UNHEALTHY=1 permits the test evaluation')
        return None
    return msg + ' — refusing to use the one-shot test access'


@torch.inference_mode()
def _finaltest_bylen_cell(cfg, manifest, v, seed, test_ids, split='test', train_fp=None):
    """One (arm, seed) clean by-length evaluation on the TEST split.

    记录现在额外携带 manifest_sha256 与
    ckpt_sha256（绑定「哪份冻结计划、哪份权重」）；recipe 必须通过冻结
    清单的结构化配方签名（不再是自然语言说明）；resume 的 stale 判定
    把 manifest 哈希纳入——清单修订后，同一 outdir 的旧结果自动作废，
    不会被静默复用。"""
    outdir = cfg['outdir']
    os.makedirs(outdir, exist_ok=True)
    spath = cfg['bylen_summary']
    _msha = manifest.get('manifest_sha256')
    _sig = _recipe_signature(manifest, v)
    out = {'status': 'ok', 'key': f'{v}::s{seed}', 'variant': v, 'seed': seed}
    _ft_run_id = L.new_experiment_id('p5t')
    _ft_started = time.strftime('%Y-%m-%dT%H:%M:%S')
    _ft_t0 = time.time()
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
    # v11.128: the caller passes the value derived BEFORE the first test read
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
    eval_lens = [int(x) for x in manifest.get('eval', {}).get('by_length_ppl', cfg['eval_lens'])]
    _dfp = L._ids_fp(np.ascontiguousarray(test_ids).reshape(-1))
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
        # A record measured on a different row count is a different estimand.
        # v11.128 stores n_seq explicitly and treats a mismatch as stale.
        if int(rec.get('n_seq') or -1) != int(test_ids.shape[0]):
            return True
        if _msha is not None and rec.get('manifest_sha256') != _msha:
            return True  # the manifest changed after this record was written
        bl = rec.get('by_len') or {}
        if not all((L.by_len_cells([rec], Ln) for Ln in eval_lens)):
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
    _arch = (int(d.get('d', 256)), int(d.get('n_layers', 6)), int(d.get('n_heads', 8)), int(d.get('d_head', 32)))
    model = L.SmallGPT(int(d.get('vocab', 8192)), *_arch, d.get('train_len', 512), [copy.deepcopy(c) for c in d['cfg']], mlp_ratio=d['mlp_ratio']).to(DEVICE)
    model.load_state_dict(d.pop('sd'))
    model.eval()
    # Keep the checkpoint's provenance fields on a separate binding so the
    # record writer never depends on a variable that this function deletes.
    _ck_prov = {k: d.get(k) for k in ('train_knobs', 'params', 'train_objective',
                                      'param_breakdown', 'rng_fp')}
    # score EVERY loaded row: test_ids is the frozen slice already sliced to
    # the manifest's n_seqs_clean, and data_fp is computed over all of them.
    # eval_length_gen's default (n_seq=8) would measure only the
    # first 8 rows while the lock/fingerprint claim the full 16-row slice.
    r = V.eval_length_gen(model, test_ids, eval_lens, DEVICE, max_pos=None, n_seq=int(test_ids.shape[0]))
    del model
    gc.collect()

    def _merge(summary):
        # by_len now also carries ppl_per_seq (per-document rows,
        # issue 9b) straight from eval_length_gen; pooled ppl is unchanged.
        # v11.128: the P5T record carries the SAME provenance descriptor as its
        # training record (`train_knobs` / `train_objective` /
        # `param_breakdown` / `rng_fp`) plus the evaluated row count.  Without
        # `train_knobs` the frozen implementation binding excluded ALL 48
        # records (admitted=0) after the one-shot test access had already been
        # burned.
        _tk = _tr.get('train_knobs') or _ck_prov.get('train_knobs')
        summary[f'{v}::s{seed}'] = {'variant': v, 'seed': seed, 'params': _tr.get('params') or _ck_prov.get('params'), 'by_len': r, 'split': split, 'data_fp': _dfp, 'train_fp': _tr_train_fp, '_code': V.CKPT_CODE, 'recipe': _ck_recipe, 'manifest_sha256': _msha, 'ckpt_sha256': _ensure_ck_sha(), 'ckpt_sig': (list(_ck_sig) if _ck_sig is not None else None),
                                    'n_seq': int(test_ids.shape[0]),
                                    'train_knobs': _tk,
                                    'train_objective': _tr.get('train_objective') or _ck_prov.get('train_objective'),
                                    'param_breakdown': _tr.get('param_breakdown') or _ck_prov.get('param_breakdown'),
                                    'rng_fp': _tr.get('rng_fp') or _ck_prov.get('rng_fp'),
                                    'amp_dtype': (_tk or {}).get('amp_dtype') if isinstance(_tk, dict) else None,
                                    'amp_grad_scaler': (_tk or {}).get('amp_grad_scaler') if isinstance(_tk, dict) else None,
                                    'eager_batch': (_tk or {}).get('eager_batch') if isinstance(_tk, dict) else None}
        print(f'  [p5t] {v:32s} s{seed} ' + '  '.join((f'L{k}={vv.get("ppl", float("nan")):.2f}' for k, vv in sorted(r.items()))))
        return summary
    L.locked_json_update(spath, _merge)
    L.log_experiment_event(outdir, {'kind': 'eval', 'tracking_schema': 2, 'run_id': _ft_run_id, 'status': 'ok', 'started_at': _ft_started, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'elapsed_s': time.time() - _ft_t0, 'key': f'{v}::s{seed}', 'variant': v, 'seed': int(seed), 'split': split, 'code': V.CKPT_CODE, 'manifest_sha256': _msha, 'recipe': _ck_recipe, 'ckpt_sha256': _ck_sha, 'data_fp': _dfp, 'train_fp': _tr_train_fp, 'params': _tr.get('params'), 'metrics': {'by_len': {str(k): (v2 or {}).get('ppl') for k, v2 in r.items()}}})
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
    need = max([int(x) for x in manifest.get('eval', {}).get('by_length_ppl', cfg['eval_lens'])]) + 1
    _nseq_clean = int((manifest.get('eval') or {}).get('n_seqs_clean') or cfg.get('n_seq', 16))
    test_ids = cfg.get('test_ids')
    _smoke = bool(cfg.get('smoke'))
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
    # v11.128: a manifest that DECLARES training_data_fingerprint may not fall
    # through to a null canonical value.  Deriving it from the frozen training
    # records is the only legal path; if that is impossible, refuse BEFORE the
    # one-shot test read rather than discovering it per-cell afterwards.
    if not _smoke and _train_fp_required(manifest) and _want_train_fp is None:
        raise SystemExit('[v12] REFUSE: the frozen manifest declares a training-corpus binding but no training_fp can be derived from the frozen records — run P5C/P3MT/P4MT first and keep their summary.json; refusing to read the held-out split under an unpinned corpus')
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
    if test_ids is None:
        _t0 = time.time()
        test_ids, vocab_t, _, _ = L.load_wikitest(need, n_seqs=_nseq_clean)
        if guard is not None:
            guard.record_run(time.time() - _t0, 0, 0, 0, 0, 0)
    else:
        print('[p5t] SMOKE: test_ids supplied by the caller; records are stamped split="smoke" and are inadmissible to the confirmation statistics')
    _dfp = L._ids_fp(np.ascontiguousarray(test_ids).reshape(-1))
    if int(test_ids.shape[0]) != _nseq_clean or int(test_ids.shape[1]) != need:
        raise SystemExit(f'[v12] REFUSE: loaded test slice shape {tuple(test_ids.shape)} != the frozen ({_nseq_clean}, {need}) — the row count and sequence length are part of the pre-registration; re-freeze explicitly instead of shrinking the panel')
    if not _smoke:
        # v11.128: run the leakage gate BEFORE the `accessed` lock is written
        # and before any evaluation.  A gate that can never fire (a
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
    # first-test-access lock — the plan may not change after
    # the test split has been read, and the slice itself is part of the freeze.
    _check_or_write_freeze_lock(outdir, manifest, _dfp, stage='accessed',
                                train_fp=_want_train_fp)
    print(f'[p5t] manifest {cfg["manifest"]} frozen_at={manifest.get("frozen_at", "?")} sha256={manifest.get("manifest_sha256", "?")[:12]}… arms={manifest["arms"]} seeds={manifest["seeds"]}')
    print(f'[p5t] TEST split rows {test_ids.shape} (held out; no other phase reads this split; ids_fp={_dfp[:12]}…)')
    if _nseq_clean < 64:
        print(f'[p5t] NOTE: n_seqs_clean={_nseq_clean} — document-sampling uncertainty is NOT well covered at this row count; population-level claims need a larger row panel (per-doc losses are recorded in bylen.json for stratified analysis; see design_notes §20)')
    # 1) clean by-length per (arm, seed)
    _split_label = 'smoke' if _smoke else 'test'
    for v in manifest['arms']:
        for seed in manifest['seeds']:
            _finaltest_bylen_cell(cfg, manifest, v, seed, test_ids,
                                  split=_split_label, train_fp=_want_train_fp)
    # 2) frozen uniform-corruption probe on the SAME test rows (secondary,
    #    descriptive-only per the manifest)
    _probe_eval = manifest.get('eval', {}).get('distractor_probe') or {}
    pcfg = dict(V10.PROBE, outdir=outdir, ckpt_dir=cfg['ckpt_dir'], summary=cfg['probe_summary'], variants=list(manifest['arms']), seeds=[int(s) for s in manifest['seeds']], eval_lens=[int(x) for x in (_probe_eval.get('eval_lens') or [2048, 4096])], rhos=[float(x) for x in (_probe_eval.get('rhos') or [0.0, 0.125, 0.25, 0.5])], modes={'uniform': [float(x) for x in (_probe_eval.get('rhos') or [0.0, 0.125, 0.25, 0.5])]}, niah=None, arm_overrides={v: ['learned'] for v in V10.SCRATCH_CONTROLS}, target=int(_probe_eval.get('target', cfg.get('target', 512))), n_seq=int(_probe_eval.get('n_seq', cfg.get('n_seq', 16))), chunk=int(cfg.get('chunk', 4)))
    mech_summ = {}
    if os.path.exists(cfg['mech_summary']):
        mech_summ = json.load(open(cfg['mech_summary'], encoding='utf-8'))
    for v in pcfg['variants']:
        for seed in pcfg['seeds']:
            V10.run_probe_cell(pcfg, v, seed, guard=guard, val_ids=test_ids, mech_summ=mech_summ)
    # 3) confirmation statistics (zero GPU, from disk)
    return finaltest_stats(manifest=manifest, bylen_path=cfg['bylen_summary'], probe_path=cfg['probe_summary'], out=cfg.get('stats_out', 'analysis_v12/test_stats.json'))

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


def _admit_confirmation_records(raw, manifest):
    """Strict admission gate for confirmation records.

    一条 bylen 记录要进入确认统计，必须同时满足：
      split==test、variant 在冻结臂内、seed 在冻结白名单内、_code 等于冻结
    清单声明的代码语义、manifest_sha256 等于当前清单哈希（同修订轮）、
    data_fp 非空、recipe 通过冻结配方签名、by_len 非空。
    不满足者被**点名剔除**（不入统计，也不静默消失）；同一 (variant, seed)
    出现两条可采纳记录 = 歧义，**拒绝运行**（而不是后者覆盖前者）；
    可采纳记录的 data_fp 必须全局唯一（同一冻结 test 切片），否则拒绝。
    返回 (admitted[(variant, seed)] -> record, unified data_fp)。"""
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
    _any_train_fp = any(isinstance(_r, dict) and bool(_r.get('train_fp'))
                        for _r in (raw or {}).values())
    _require_train_fp = _train_fp_required(manifest) or bool(want_train_fp) or _any_train_fp
    for key, r in (raw or {}).items():
        if not isinstance(r, dict):
            continue
        why = None
        s_i = None
        if r.get('split') != 'test':
            # v11.128: named exclusion, not a silent `continue`.  The
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
            elif not isinstance(r.get('by_len'), dict) or not r.get('by_len'):
                why = 'no by_len measurement'
            else:
                v = r['variant']
                if (v, s_i) in admitted:
                    duplicates.append(f'{v}::s{s_i} appears twice ({admitted[(v, s_i)].get("key", "?")} and {key})')
                else:
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
        raise SystemExit('[p5t stats] REFUSE: ambiguous duplicate records for the same (variant, seed): ' + '; '.join(duplicates) + ' — the paired statistics would depend on which one wins; resolve the bylen file first (this is exactly the dict-overwrite ambiguity the review flagged)')
    if len(fps) > 1:
        raise SystemExit(f'[p5t stats] REFUSE: admissible records span {len(fps)} different test-slice fingerprints ({sorted(fps)}) — the whole panel must be measured on ONE frozen test slice, not merely pairwise-equal fingerprints')
    if len(train_fps) > 1:
        raise SystemExit(f'[p5t stats] REFUSE: admissible records were trained on {len(train_fps)} different corpora ({sorted(train_fps)}) — all frozen arms must share the canonical training split')
    return admitted, (sorted(fps)[0] if fps else None)


def _doc_level_analysis(recs_a, recs_b, common, Ln):
    """Document-sampling readout for one comparison.

    逐文档配对差（对每个文档取「种子内臂差」的均值）与两层方差分解：
    between-doc（文档抽样不确定性）与 within-doc-across-seed（训练随机性）。
    16 条评测行不足以充分覆盖文档分布——本读数的目的就是把这个不确定性
    显式化，而不是宣称已消除。"""
    rows_a = [recs_a[s]['by_len'].get(Ln, recs_a[s]['by_len'].get(str(Ln))) or {} for s in common]
    rows_b = [recs_b[s]['by_len'].get(Ln, recs_b[s]['by_len'].get(str(Ln))) or {} for s in common]
    per_seq_a = [r.get('ppl_per_seq') for r in rows_a]
    per_seq_b = [r.get('ppl_per_seq') for r in rows_b]
    if not all(isinstance(x, list) and x for x in per_seq_a + per_seq_b):
        return None
    # Pair row i across arms only when every record evaluated the SAME number
    # of documents.  Silently taking the minimum would pair a truncated panel
    # with a full one and claim a larger document sample than either scored.
    _lens = {len(x) for x in per_seq_a + per_seq_b}
    if len(_lens) != 1:
        return {'n_docs': int(min(_lens)), 'n_seeds': len(common), 'comparable': False,
                'reason': 'per-document row counts differ across arm/seed records; document-paired deltas are not defined',
                'row_counts': sorted(_lens), 'note': 'refused rather than silently truncating to the shortest record'}
    n_docs = min(_lens)
    if n_docs < 2:
        return None
    doc_deltas = []
    doc_log_ratios = []
    within = []
    incomplete = 0
    for i in range(n_docs):
        cell = []
        logs = []
        for k in range(len(common)):
            a, b = per_seq_a[k][i], per_seq_b[k][i]
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
    arr = np.asarray(doc_deltas, dtype=float)
    lr = np.asarray(doc_log_ratios, dtype=float)
    # Seed-level sign-flip remains the confirmatory test.  These intervals
    # condition on the trained seeds and quantify only DOCUMENT sampling:
    # with n_seqs=16 they are a property of the frozen slice, not of the
    # language population.  Resampling documents (not tokens) preserves the
    # within-document correlation.
    _ci_ppl = L.bootstrap_mean_ci(arr, seed=20261018)
    _ci_log = L.bootstrap_mean_ci(lr, seed=20261018)
    return {'n_docs': int(len(doc_deltas)), 'n_seeds': len(common),
            'per_doc_delta_mean': float(arr.mean()),
            'per_doc_log_ratio_mean': float(lr.mean()),
            'between_doc_sd': float(arr.std(ddof=1)) if len(doc_deltas) > 1 else 0.0,
            'between_doc_log_ratio_sd': float(lr.std(ddof=1)) if len(doc_log_ratios) > 1 else 0.0,
            'mean_within_doc_seed_sd': float(np.mean(within)),
            'per_doc_delta_ppl_ci95': ([_ci_ppl['ci_low'], _ci_ppl['ci_high']] if _ci_ppl else None),
            'per_doc_log_ratio_ci95': ([_ci_log['ci_low'], _ci_log['ci_high']] if _ci_log else None),
            'incomplete_doc_seed_cells': int(incomplete),
            'ci_method': 'percentile bootstrap over documents (chunked, deterministic); conditions on the trained seeds',
            'note': 'document-sampling uncertainty only; seed-level exact sign-flip remains the confirmatory test, and n_seqs<64 does not cover the language-population distribution — enlarge the row panel for population-level claims (design_notes section 20)'}


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
        if r.get('variant') not in arms:
            why = f"variant {r.get('variant')!r} is not one of the frozen arms"
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
        elif r.get('arm') is None or r.get('eval_len') is None or r.get('rho') is None:
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
                 str(r.get('recipe')), str(r.get('_code')))
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
    _idx = {}
    for c in out['cells']:
        if not c.get('ambiguous_cell'):
            _idx[(c['variant'], c['arm'], c['eval_len'], c['rho'], c['mode'])] = c['ppl_by_seed']
    for (v, arm, Ln, rho, mode), learned in sorted(_idx.items()):
        if arm != 'learned':
            continue
        for (_v, other_arm, _Ln, _rho, _mode), other in sorted(_idx.items()):
            if _v != v or _Ln != Ln or _rho != rho or _mode != mode or other_arm == 'learned':
                continue
            common = sorted(set(learned) & set(other), key=float)
            if len(common) < 2:
                continue
            dl = [other[s] - learned[s] for s in common]
            res = V.exact_sign_permutation(dl)
            res['seeds'] = [int(s) for s in common]
            res['delta_definition'] = f'{other_arm} - learned (paired per seed, test split, descriptive)'
            out['contrasts'][f'L{Ln} r{rho} [{mode}] {other_arm} - learned'] = res
    if excluded:
        print('[p5t probe] EXCLUDED test-probe records (named, never silent):\n  - ' + '\n  - '.join(excluded[:20]) + ('' if len(excluded) <= 20 else f'\n  … {len(excluded) - 20} more'))
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
        _check_or_write_freeze_lock(_lock_dir, manifest, unified_fp, stage='accessed',
                                    train_fp=(next(iter(_adm_train_fps)) if len(_adm_train_fps) == 1 else None))
    _alpha = float((manifest.get('statistics') or {}).get('alpha', 0.05))
    stats = {'manifest': {'frozen_at': manifest.get('frozen_at'), 'manifest_sha256': manifest.get('manifest_sha256'), 'arms': manifest['arms'], 'seeds': manifest['seeds'], 'code_semantics_expected': manifest.get('code_semantics_expected'), 'statistics': manifest.get('statistics'), 'scope': manifest.get('scope'), 'revision': manifest.get('revision')}, 'admission': {'n_admitted': len(admitted), 'test_slice_fp': unified_fp, 'n_effective_seeds': len({int(s) for (_v, s) in admitted}), 'n_declared_seeds': len(manifest['seeds'])}, 'comparisons': {}, 'holm': {}, 'by_length': {}, 'probe': _test_probe_stats(probe_path, manifest, unified_fp)}
    # The exact sign-flip p is valid only under within-seed exchangeability.
    # admission binds the corpus/slice/row count/recipe/code, but the RNG stream
    # and batch order are not independently re-derivable from a record; report
    # them instead of silently assuming them.
    _rngs = sorted({str(r.get('rng_fp')) for r in admitted.values() if r.get('rng_fp')})
    _inits = sorted({str(r.get('init_core_fp')) for r in admitted.values() if r.get('init_core_fp')})
    stats['validity'] = {
        'rng_fp_values': _rngs,
        'same_rng_fp': len(_rngs) <= 1,
        'init_core_fp_values': _inits,
        'same_init_core_fp': len(_inits) <= 1,
        'init_core_fp_recorded': bool(_inits),
        'n_seq_values': sorted({n for n in (_record_int(r.get('n_seq')) for r in admitted.values()) if n is not None}),
        'note': 'exact sign-flip requires within-seed exchangeability (same data order / eval rows / corruption grid). data_fp, train_fp, n_seq and recipe are machine-checked; the shared-trunk initialization digest (init_core_fp) is reported and compared (`same_init_core_fp`); rng_fp is reported but NOT enforced (architecture-induced RNG consumption is part of the arm), so the exchangeability claim remains a declared, machine-instrumented assumption.',
    }
    # Review issue 2: the frozen comparison is parameter-matched, not
    # compute-matched.  Refuse a non-descriptive plan that does not disclose
    # both facts, so the caveat cannot be silently dropped from the report.
    _cbd = manifest.get('compute_budget_disclosure')
    _descriptive = bool((manifest.get('statistics') or {}).get('descriptive_only'))
    if not _descriptive:
        _missing = [k for k in ('parameter_matching', 'flops', 'read_fraction')
                    if not (isinstance(_cbd, dict) and _cbd.get(k))]
        if _missing:
            raise SystemExit(f'[p5t stats] REFUSE: the frozen plan lacks compute-budget disclosure {_missing}. Parameter matching is not compute matching; the report must state the FLOPs/read-budget mismatch next to every primary comparison .')
    stats['compute_budget_disclosure'] = _cbd
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
                if isinstance(c, dict) and L.ppl_is_usable(c.get('ppl')) and (not c.get('truncated')):
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
        # v11.128: drop only the incomplete PAIR, not the whole comparison.
        # The review's scenario — one L4096 cell fails -> `dl=None` -> the
        # entire comparison is omitted — wasted 47 good cells.  Use every
        # complete pair and report the dropped seeds.  Whether n<12 may still
        # confirm is a PRE-REGISTERED choice (statistics.min_seeds_for_confirmation,
        # manifest v3: 8, the mathematical minimum for Holm/4 with alpha=0.05).
        def _cell_ppl(rec):
            bl = rec.get('by_len') or {}
            c = bl.get(Ln, bl.get(str(Ln)))
            return float(c['ppl']) if isinstance(c, dict) and L.ppl_is_usable(c.get('ppl')) and (not c.get('truncated')) else None
        dl = []
        usable = []
        dropped = []
        for s in common:
            pa, pb = (_cell_ppl(recs_a[s]), _cell_ppl(recs_b[s]))
            if pa is None or pb is None:
                dropped.append({'seed': int(s), 'reason': f'missing usable untruncated cell at L{Ln}'})
                continue
            dl.append(pa - pb)
            usable.append(s)
        if len(usable) < 2:
            stats['comparisons'][name] = {'omitted': f'<2 complete seed pairs at the frozen endpoint (common records: {common}; usable pairs: {usable}; whitelist {manifest["seeds"]})', 'n': 0, 'expected_sign': exp_sign, 'dropped_pairs': dropped}
            continue
        res = V.exact_sign_permutation(dl)
        _n_eff = len(usable)
        _n_min = int((manifest.get('statistics') or {}).get('min_seeds_for_confirmation') or len(manifest['seeds']))
        res['seeds'] = usable
        res['n_effective'] = int(_n_eff)
        res['coverage_ok'] = bool(_n_eff == len(manifest['seeds']))
        res['missing_seeds'] = sorted(set(int(s) for s in manifest['seeds']) - set(int(s) for s in usable))
        res['dropped_pairs'] = dropped
        res['min_seeds_for_confirmation'] = _n_min
        # Diagnostics must use the n that actually entered THIS comparison,
        # so p_floor / min_attainable_p_holm / power describe the test that
        # really ran.
        _fl_eff = L.signflip_p_floor(_n_eff)
        if math.isfinite(_fl_eff):
            res['p_floor_effective'] = _fl_eff
            res['min_attainable_p_holm_effective'] = L.holm_min_attainable_p(_n_eff, len(manifest['primary_comparisons']), _alpha)
            res['power_first_holm_gate_effective'] = {f'q={q:.2f}': L.signflip_power(_n_eff, len(manifest['primary_comparisons']), q, _alpha) for q in (0.8, 0.85, 0.9, 0.95)}
        _pa = [p for p in (L._finite_scalar(recs_a[s].get('params')) for s in usable) if p is not None]
        _pb = [p for p in (L._finite_scalar(recs_b[s].get('params')) for s in usable) if p is not None]
        if _pa and _pb:
            _am, _bm = float(np.mean(_pa)), float(np.mean(_pb))
            _rel = abs(_am - _bm) / max(_am, _bm, 1.0)
            res['param_matched'] = bool(_rel <= 0.01)
            res['param_rel_diff'] = float(_rel)
            res['params_a'] = _am
            res['params_b'] = _bm
        else:
            res['param_matched'] = None
            res['param_rel_diff'] = None
        _iv = sorted({str(recs_a[s].get('init_core_fp')) for s in usable if recs_a[s].get('init_core_fp')} |
                     {str(recs_b[s].get('init_core_fp')) for s in usable if recs_b[s].get('init_core_fp')})
        res['same_init_core_fp'] = (len(_iv) <= 1)
        res['init_core_fp_recorded'] = bool(_iv)
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
        if exp_sign in ('positive', 'negative') and math.isfinite(res.get('mean', float('nan'))):
            res['direction_ok'] = bool((res['mean'] > 0) == (exp_sign == 'positive'))
        else:
            res['direction_ok'] = None
        _doc = _doc_level_analysis(recs_a, recs_b, usable, Ln)
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
    # the pre-registered decision rule requires BOTH p_holm < alpha AND the
    # observed direction to match the validation-panel estimate (a null
    # expected_sign marks a non-directional mechanism readout).  The strict `<`
    # matches the manifest's `p_holm < 0.05` (the generic holm_adjust uses `<=`).
    # The pre-registered decision rule (manifest v3) requires p_holm < alpha
    # AND the observed direction AND parameter matching AND at least
    # `min_seeds_for_confirmation` complete pairs.  `coverage_ok` (all 12
    # seeds) is reported but is no longer an all-or-nothing veto: one failed
    # L4096 cell may not annihilate an otherwise complete comparison.
    _conf = [k for k in _rej if stats['comparisons'].get(k, {}).get('direction_ok') is True
             and stats['comparisons'].get(k, {}).get('param_matched') is not False
             and int(stats['comparisons'].get(k, {}).get('n_effective') or 0) >= int(stats['comparisons'].get(k, {}).get('min_seeds_for_confirmation') or len(manifest['seeds']))]
    stats['confirmed'] = _conf
    stats['significant_nondirectional'] = [k for k in _rej if stats['comparisons'].get(k, {}).get('direction_ok') is None]
    stats['not_confirmed_reasons'] = {k: {'param_matched': stats['comparisons'].get(k, {}).get('param_matched'),
                                          'coverage_ok': stats['comparisons'].get(k, {}).get('coverage_ok'),
                                          'n_effective': stats['comparisons'].get(k, {}).get('n_effective'),
                                          'min_seeds_for_confirmation': stats['comparisons'].get(k, {}).get('min_seeds_for_confirmation'),
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
        # Read the recorded document rows from whichever frozen endpoint
        # carries them; smoking/alternate manifests need not contain L=4096.
        _n_doc = next((len(c.get('ppl_per_seq') or [])
                       for r in admitted.values()
                       for c in (r.get('by_len') or {}).values()
                       if isinstance(c, dict) and c.get('ppl_per_seq')), 0)
        if _n_doc:
            stats['population_claim_allowed'] = bool(_n_doc >= L.MIN_DOCS_FOR_POPULATION_CLAIM)
            if _n_doc < L.MIN_DOCS_FOR_POPULATION_CLAIM:
                stats['doc_sampling_note'] = (f'per-document rows recorded (n_docs={_n_doc}); between-doc variance is reported per comparison in doc_sampling — '
                                              f'n_docs<{L.MIN_DOCS_FOR_POPULATION_CLAIM} does not cover the language-population distribution, so population-level claims are refused; '
                                              'the confirmed claims are about the frozen evaluation slice (design_notes section 20)')
    print(f'[p5t stats] admitted {len(admitted)}/{len(manifest["arms"]) * _n} frozen (arm, seed) cells on slice {str(unified_fp)[:12]}…; {len(_fam)} frozen primary comparison(s); {len(_rej)} survive Holm; {len(_conf)} also match the pre-registered direction')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    stats = L.stamped_analysis('finaltest_stats', stats, outdir=(os.path.dirname(out) or '.'), name=os.path.basename(out))
    print(f'[p5t stats] wrote {out}')
    return stats

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
        A(f'**尚未运行 P5T**：`{TEST_OUTDIR}/bylen.json` 不存在（或为空）。前置：`python v12_supp.py phase P5C`（覆盖全部四个冻结臂、种子取自清单），再跑 `python v12_supp.py phase P5T`（首次评估会写入冻结锁 `.freeze_lock.json`，此后清单不可再改）。')
        A('')
    A('---')
    A('')
    A('## P5T 独立最终测试（test split，冻结终点）')
    A('')
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
        A(f'**尚未运行 P5T**：`{TEST_OUTDIR}/bylen.json` 不存在（或为空）。前置：`python v12_supp.py phase P5C`（现覆盖全部四个冻结臂、种子数取自清单；主臂 0-5 已由 P3MT/P4MT 训练，resume 自动跳过），然后 `python v12_supp.py phase P5T`。冻结清单已入库（见上），随时可以执行；执行前请勿据任何 validation 读数调整冻结内容。')
        A('')
    comps = ft.get('comparisons') or {}
    if comps:
        A('| 冻结主比较（test split，配对） | n | 覆盖完整 | 参数匹配 | Δ mean±std | 文档 Δ 95% CI | d_z | p (exact) | p 下限 | Holm p | Holm 拒绝 | 方向符合 |')
        A('|---|---|---|---|---|---|---|---|---|---|---|---|')
        holm = ft.get('holm') or {}
        for k, v in comps.items():
            if 'omitted' in v:
                A(f'| {k} | 0 | — | — | —（{v["omitted"]}） | — | — | — | — | — | — | — |')
                continue
            _h = holm.get(k) or {}
            _hp = f'{_h.get("p_holm", float("nan")):.3f}' if _h.get('p_holm') is not None else '—'
            _hr = '✔' if (_h.get('p_holm') is not None and float(_h['p_holm']) < 0.05) else '✘'
            _dir = v.get('direction_ok')
            _ds = '—（非方向性）' if _dir is None else ('✔' if _dir else '✘')
            _dsm = v.get('doc_sampling') or {}
            _dci = _dsm.get('per_doc_delta_ppl_ci95')
            _dcitxt = f'[{_dci[0]:+.2f}, {_dci[1]:+.2f}]' if isinstance(_dci, (list, tuple)) and len(_dci) == 2 else '—'
            A(f'| {k} | {v["n"]} | {v.get("coverage_ok")} | {v.get("param_matched")} | {v["mean"]:+.2f} ± {v.get("std", 0):.2f} | {_dcitxt} | {v.get("cohens_dz", float("nan")):+.2f} | {v["p_exact_signflip"]:.3f} | {v.get("p_floor", float("nan")):.3f} | {_hp} | {_hr} | {_ds} |')
        A('')
        _conf = ft.get('confirmed')
        if _conf is None:
            _conf = [k for k, v in holm.items() if v.get('reject') and (comps.get(k, {}).get('direction_ok') is True) and (comps.get(k, {}).get('param_matched') is not False) and (comps.get(k, {}).get('coverage_ok') is True)]
        if _conf:
            A(f'**确认结果**：{len(_conf)}/{len(comps)} 个冻结主比较在 Holm 校正后成立**且方向符合预注册**（{sorted(_conf)}）——这是预注册规则下的确认性证据。')
        else:
            A('**确认结果**：没有冻结主比较同时满足「Holm 校正后 p < 0.05」与「方向符合预注册」。按预注册的决策规则，相应主张**不获确认**；探索性面板的正向读数保持「探索性」标签，不得写成结论。')
        _nd = ft.get('significant_nondirectional') or []
        if _nd:
            A('')
            A(f'**非方向性机制读数**：{sorted(_nd)} 在 Holm 校正后达到 p<0.05，但冻结清单未给方向（expected_sign=null），只能作为机制读数，不能计入「方向符合预注册」的确认集合。')
        if ft.get('min_attainable_p_holm') is not None:
            A('')
            A(f"（诊断：n 个种子与 m 元 Holm 族下，最小可达 Holm p = {ft['min_attainable_p_holm']:.4f}。加载期功效门槛已拒绝「数学上不可能确认」的清单；若此处仍 > 0.05，说明有绕过门槛的旧状态，需回查清单与记录的来源。）")
        _dsn = ft.get('doc_sampling_note')
        if _dsn:
            A('')
            A(f"（文档抽样：{_dsn}）")
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
    git_push('v12: confirmation driver (frozen-manifest v3: zero-GPU plan gate + strict record admission + freeze-lock binding; P5C covers all four frozen arms at manifest seeds)')
    import par
    _ctx = par.resolve_ctx(label='v12')
    all_ok = True
    for pname, _kind, _payload, _seeds in PHASES:
        rem = guard.remaining_hours()
        if rem < 0.4166666666666667:
            print(f'[v12] stopping before {pname}: headroom exhausted ({rem:.2f} h)')
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
        all_ok &= git_push(f'v12: phase {pname} results')
    try:
        build_report()
    except Exception:
        traceback.print_exc()
        all_ok = False
    all_ok &= git_push('v12: stats + REPORT_v12.md (analysis_v12)')
    guard.report()
    print('\n[v12] ALL PHASES DONE.')
    schedule_shutdown(120 if all_ok else 2400)

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


def plan_check(manifest_path=FREEZE_MANIFEST):
    """Zero-GPU pre-flight for the frozen confirmation plan.

    The review's most dangerous finding was that the obvious fix (align the
    code stamp) turns P5T from "fails fast" into "burns the one-shot test
    access, then admits 0/48 records".  This check closes that hole WITHOUT
    reading test data: it derives the implementation descriptor from the same
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
                                                 matched={arm}, ref_variant='csa_fixed_rope') if arm in ('full_rope', 'full_sw128_matched_rope') else 4.0
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
            del _shape_model
        except Exception as _e:
            errs.append(f'{arm}: could not compute the parameter count for the frozen shape ({type(_e).__name__}: {_e})')
            _pred[arm]['params'] = None
        for s in seeds:
            _records[f'{arm}::s{s}'] = {
                'variant': arm, 'seed': int(s), 'split': 'test',
                'by_len': {str(Ln): {'ppl': 100.0, 'n_tok': int(_nseq) * int(Ln - 1),
                                     'n_seq': int(_nseq), 'ppl_per_seq': [100.0] * int(_nseq)}
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
        _comp_report[comp.get('name')] = {'a': _a, 'b': _b, 'endpoint_L': _Ln, 'pairs_planned': len(seeds) - len(_low), 'params_a': _pa, 'params_b': _pb, 'param_rel_diff': _rel}
    report = {'ok': not errs, 'manifest': manifest_path, 'manifest_sha256': m.get('manifest_sha256'),
              'code_semantics': L.CODE_SEMANTICS, 'n_arms': len(arms), 'n_seeds': len(seeds),
              'n_min_for_confirmation': _n_min, 'min_attainable_p_holm': _min_holm,
              'predicted': _pred, 'admitted': len(admitted), 'comparisons': _comp_report,
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
    globals()['_PLAN_DONE'] = True
    return report


def run_plan(manifest_path=FREEZE_MANIFEST):
    return plan_check(manifest_path)


def run_smoke():
    import copy
    import shutil
    # v11.128: smoke must never write into the frozen wt103_cache.  It used a
    # 1M/4M-token fit cap and could leave a differently-tokenised
    # tok_v5_*.json that makes a later cold P5T build fail AFTER the test read.
    # An isolated cache under the smoke outdir keeps the frozen cache clean.
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
    _MODES = ('full', 'smoke', 'plan', 'analysis', 'report', 'phase', 'finaltest')
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
        elif mode == 'analysis':
            v12_analysis()
        elif mode == 'report':
            build_report()
        elif mode == 'phase':
            run_phase(sys.argv[2], make_guard())
            git_push(f'v12: phase {sys.argv[2]} results')
        elif mode == 'finaltest':
            run_finaltest(dict(FINALTEST_CFG))
        else:
            run_full()
    except Exception:
        traceback.print_exc()
        try:
            git_push('v12: PARTIAL — crashed, see log (re-run resumes)')
        except Exception:
            traceback.print_exc()
        if mode == 'full':
            schedule_shutdown(2400)
        sys.exit(1)
