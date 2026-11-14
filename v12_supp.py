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
  P5S2  —— 等训练 token 的大模型规模臂（P3SL_TOK）；
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
  python v12_supp.py power            # 零 GPU：访问前功效模拟（清单 power_note 义务）
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
# 种子列表在运行期由 `_frozen_seeds()` 从冻结清单读取（单一事实来源）；
# P5C 同时覆盖全部四个冻结臂（主臂 0-5 已由
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
    _require_tfp = _train_fp_required(manifest)
    _want_tfp = _frozen_train_fp(manifest)
    _seen_tfps = set()
    for v in manifest['arms']:
        for seed in manifest['seeds']:
            key = f'{v}::s{seed}'
            ck = os.path.join(ckpt_dir, L.direct_ckpt_name(v, seed))
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
            # The pre-test gate must reject a stale checkpoint BEFORE the
            # held-out split is read and locked.  A healthy curve alone proves
            # nothing about the code/recipe that produced these weights.
            _code_ok = d.get('code') == V.CKPT_CODE
            _ck_tfp = d.get('train_fp')
            if isinstance(_ck_tfp, str) and _ck_tfp:
                _seen_tfps.add(_ck_tfp)
            # A manifest that declares a training-corpus binding but has its
            # canonical value still null still requires every checkpoint to
            # carry a non-null fingerprint.  Otherwise P5H says healthy, P5T
            # reads/locks the test split, and only then refuses the checkpoint.
            _tfp_ok = ((_ck_tfp == _want_tfp) if _want_tfp is not None
                       else ((not _require_tfp)
                             or (isinstance(_ck_tfp, str) and bool(_ck_tfp))))
            # The recipe is part of the frozen protocol too.  check it here,
            # before P5T reads and locks the held-out split; otherwise a
            # healthy/code/train_fp-correct checkpoint trained under another
            # recipe reaches the one-shot access and is only refused per-cell
            # afterwards.
            _sig = _recipe_signature(manifest, v)
            _ck_recipe = d.get('recipe')
            _recipe_ok = (_sig is None
                          or _recipe_matches_signature(_ck_recipe, _sig))
            if not _code_ok:
                status = 'stale_code'
            elif not _tfp_ok:
                status = 'stale_train_fp'
            elif not _recipe_ok:
                status = 'stale_recipe'
            rows[key] = {'status': status, 'health': h,
                         'best_step': d.get('best_step'),
                         'best_val_ppl': d.get('best_val_ppl'),
                         'final_ppl': d.get('final_ppl'),
                         'code': d.get('code'), 'train_fp': d.get('train_fp'),
                         'recipe': _ck_recipe, 'recipe_matches': _recipe_ok,
                         'path': ck}
            if status != 'healthy':
                unhealthy.append(key)
            del d
            try:
                gc.collect()
            except Exception:
                pass
    if _require_tfp and len(_seen_tfps) > 1:
        # Checkpoints from different corpora cannot be pooled as one frozen
        # panel even if each individually has a fingerprint.
        for _key, _row in rows.items():
            if _row.get('status') == 'healthy' and _row.get('train_fp'):
                _row['status'] = 'mixed_train_fp'
                if _key not in unhealthy:
                    unhealthy.append(_key)
    report = {'kind': 'training_health', 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
              'manifest': cfg.get('manifest') or FREEZE_MANIFEST,
              'n_arms': len(manifest['arms']), 'n_seeds': len(manifest['seeds']),
              'n_cells': len(rows), 'n_missing': len(missing), 'n_unhealthy': len(unhealthy),
              'missing': missing, 'unhealthy': unhealthy,
              'status': ('pass' if not missing and not unhealthy else 'fail'),
              'note': 'diagnostic only: this gate never selects a checkpoint or changes the frozen plan',
              'cells': rows}
    out = cfg.get('out') or os.path.join(TEST_OUTDIR, 'training_health.json')
    allow = os.environ.get('V12_ALLOW_UNHEALTHY', '').strip().lower() in ('1', 'true', 'yes', 'on')
    # The override must be MACHINE-READABLE in the artifacts, not a stdout-only
    # warning: persist it on the health report itself so a later audit can see
    # that the one-shot test access was consumed while the panel was failing.
    report['v12_allow_unhealthy_used'] = bool(allow and report['status'] != 'pass')
    if report['v12_allow_unhealthy_used']:
        report['override_note'] = ('V12_ALLOW_UNHEALTHY=1 was set when this failing '
                                   'health panel was accepted; the subsequent P5T '
                                   'records carry allow_unhealthy_override=true')
    L.atomic_write_json(out, report, indent=2)
    print(f'[p5h] training health: {report["status"]} ({len(unhealthy)} unhealthy, {len(missing)} missing) -> {out}')
    if report['status'] != 'pass' and not allow:
        raise SystemExit(f'[p5h] REFUSE: frozen-arm training health is not clean ({len(unhealthy)} unhealthy, {len(missing)} missing); investigate before using the one-shot test access (V12_ALLOW_UNHEALTHY=1 is the explicit override)')
    if report['status'] != 'pass':
        print('[p5h] WARNING: V12_ALLOW_UNHEALTHY=1 — continuing despite unhealthy/missing training health (recorded in training_health.json and on every admitted P5T record)')
    return report


def _lock_proves_test_access(path):
    """True when a freeze-lock file shows a test-split access.

    A pre-access attestation (``stage='preaccess'`` and no bound test slice)
    is NOT evidence that the held-out split was read: the power gate can
    legitimately abort after writing it, and the protocol explicitly requires
    adding seeds / revising the manifest before any test access.  An
    unparseable or ambiguous lock still fails closed as an access."""
    if not os.path.exists(path):
        return False
    try:
        lock = json.load(open(path, encoding='utf-8'))
    except Exception:
        return True
    if not isinstance(lock, dict):
        return True
    if lock.get('test_ids_fp') is not None:
        return True
    _stage = str(lock.get('stage') or '').strip().lower()
    if _stage not in ('preaccess', 'pre-access', 'pre_access'):
        return True
    for _h in lock.get('history') or []:
        if not isinstance(_h, dict):
            return True
        _hs = str(_h.get('stage') or '').strip().lower()
        if _h.get('test_ids_fp') is not None or _hs in ('accessed', 'refreeze-test-fp'):
            return True
    return False


def _p5c_post_access_guard():
    """Refuse P5C (re)training once the test split has been accessed.

    The freeze lock binds the plan, the test slice and the training corpus —
    but NOT the checkpoint set.  Without this gate, one could see the test
    results, retrain a chosen (arm, seed) via P5C (which never checks the
    lock), and re-run P5T to overwrite that cell's measurement: selective
    re-measurement after seeing the held-out data is selection bias by
    construction, exactly like adding seeds after the fact.  A pre-access
    attestation is not an access and must not block the pre-access revision
    the protocol requires."""
    _refrz = os.environ.get('V12_REFREEZE', '').strip().lower() in ('1', 'true', 'yes', 'on')
    _lock_present = (_lock_proves_test_access(_freeze_lock_path(TEST_OUTDIR))
                     or _lock_proves_test_access(_freeze_lock_mirror_path()))
    if not _lock_present:
        return
    if _lock_present and not _refrz:
        raise SystemExit(
            '[v12] REFUSE: phase P5C would (re)train frozen arms, but the test '
            f'split was already accessed (freeze lock/mirror for {TEST_OUTDIR} exists). '
            'Retraining a chosen cell after seeing test results and re-measuring it '
            'would overwrite that cell\'s one-shot measurement — selective re-'
            'measurement is selection bias by construction. Investigate the failing '
            'cell and, if a conscious re-freeze is truly warranted, set V12_REFREEZE=1 '
            '(the earlier round is then downgraded to exploratory in the plan history).')
    if _lock_present:
        print('[v12] WARNING: V12_REFREEZE=1 — retraining frozen arms after the first test access; the earlier confirmation round is downgraded to exploratory')


def run_phase(name, guard, par=None):
    global _PLAN_DONE
    if name in ('P5C', 'P5MP', 'P5T') and not _PLAN_DONE:
        # A direct phase must not start under a plan that the P5T admission
        # gate would reject.  In particular `phase P5T` reads the held-out
        # split, so the zero-GPU plan preflight has to run before that access,
        # not only for the training phases.  `run_full` runs this once at startup.
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
        if pname == 'P5C':
            # Post-access selective-retraining guard (see the function doc).
            _p5c_post_access_guard()
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
        if _fld in m and _val is None:
            raise SystemExit(f'[v12] REFUSE: {_fld} is explicitly null; omit it or provide an object')
        if _val is not None and not isinstance(_val, dict):
            raise SystemExit(f'[v12] REFUSE: {_fld} must be an object ({_val!r})')

    def _int_field(value, label):
        if isinstance(value, (bool, np.bool_)):
            raise SystemExit(f'[v12] REFUSE: {label}={value!r} is not an integer')
        if isinstance(value, numbers.Integral):
            _iv = int(value)
            if abs(_iv) >= 2 ** 53:
                raise SystemExit(f'[v12] REFUSE: {label}={value!r} is outside the exact integer range (< 2**53)')
            return _iv
        if isinstance(value, (str, bytes)):
            # A manifest integer spelling must be an actual integer.  The
            # float path below would accept "512.0"/"1e3" and hand the raw
            # string to downstream int() calls after a test access.  Reject
            # non-canonical spellings (whitespace, leading '+', leading zeros,
            # underscores) at load time too: int() would silently accept them
            # as a different textual identity than the canonical JSON form.
            try:
                _text = value.decode('ascii') if isinstance(value, bytes) else value
            except (UnicodeDecodeError, AttributeError):
                raise SystemExit(f'[v12] REFUSE: {label}={value!r} is not an integer')
            if re.fullmatch(r'-?(?:0|[1-9][0-9]*)', _text) is None:
                raise SystemExit(f'[v12] REFUSE: {label}={value!r} is not an integer')
            try:
                _iv = int(_text)
            except (TypeError, ValueError, OverflowError):
                raise SystemExit(f'[v12] REFUSE: {label}={value!r} is not an integer')
            if abs(_iv) >= 2 ** 53:
                raise SystemExit(f'[v12] REFUSE: {label}={value!r} is outside the exact integer range (< 2**53)')
            return _iv
        try:
            f = float(value)
        except (TypeError, ValueError, OverflowError):
            raise SystemExit(f'[v12] REFUSE: {label}={value!r} is not an integer')
        # Exact integers only.
        if not math.isfinite(f) or f != math.floor(f) or abs(f) >= 2 ** 53:
            raise SystemExit(f'[v12] REFUSE: {label}={value!r} is not an exact integer')
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
    # rather than silently computed under a different contract.
    _stat = m.get('statistics') or {}
    _do = _stat.get('descriptive_only')
    if _do is not None and not isinstance(_do, (bool, np.bool_)):
        raise SystemExit(
            f'[v12] REFUSE: statistics.descriptive_only={_do!r} is not a '
            'boolean — a truthy string/number must not disable the power gate')
    for _flag in ('eager_batch', 'eager_batch_expected',
                  'amp_grad_scaler', 'amp_grad_scaler_expected'):
        _fv = m.get(_flag)
        if _fv is not None and not isinstance(_fv, (bool, np.bool_)):
            raise SystemExit(
                f'[v12] REFUSE: {_flag}={_fv!r} is not a boolean implementation '
                'flag')
    for _dtype_key in ('amp_dtype', 'amp_dtype_expected'):
        _dv = m.get(_dtype_key)
        if _dv is not None and (not isinstance(_dv, str) or not _dv):
            raise SystemExit(
                f'[v12] REFUSE: {_dtype_key}={_dv!r} is not a non-empty dtype name')
    _ns = _stat.get('n_seeds')
    if _ns is not None and _int_field(_ns, 'statistics.n_seeds') != len(m['seeds']):
        raise SystemExit(f'[v12] REFUSE: freeze manifest declares n_seeds={_ns} but lists {len(m["seeds"])} seeds — the pre-registered resolution and the seed list disagree')
    _bind = m.get('binding') or {}
    _ts = (_bind.get('test_slice') or {}) if isinstance(_bind, dict) else {}
    if not isinstance(_ts, dict):
        raise SystemExit(f'[v12] REFUSE: binding.test_slice must be an object ({_ts!r})')
    _eval = m.get('eval') or {}
    if isinstance(_eval, dict) and 'by_length_ppl' in _eval and _eval.get('by_length_ppl') is None:
        raise SystemExit('[v12] REFUSE: eval.by_length_ppl is explicitly null; omit it (to use the runtime default) or provide a non-empty list of frozen lengths')
    _bl = _eval.get('by_length_ppl') if isinstance(_eval, dict) else None
    _bl_ints = []
    if _bl is not None:
        if not isinstance(_bl, list) or not _bl:
            raise SystemExit(f'[v12] REFUSE: eval.by_length_ppl must be a non-empty list of positive integer lengths ({_bl!r})')
        _bl_ints = []
        for _x in _bl:
            _iv = _int_field(_x, 'eval.by_length_ppl entry')
            if _iv < 2:
                raise SystemExit(f'[v12] REFUSE: eval.by_length_ppl value {_x!r} has no next-token prediction (needs an integer length >= 2)')
            _bl_ints.append(_iv)
        if len(set(_bl_ints)) != len(_bl_ints):
            raise SystemExit(f'[v12] REFUSE: eval.by_length_ppl contains duplicate lengths {_bl_ints!r} — the frozen endpoint grid must name each length once')
    _ts_n = _ts.get('n_seqs') if isinstance(_ts, dict) else None
    if _ts_n is not None:
        _ev_n = _eval.get('n_seqs_clean')
        if _ev_n is None:
            raise SystemExit(f'[v12] REFUSE: binding.test_slice.n_seqs={_ts_n} is declared but eval.n_seqs_clean is missing — the evaluation plan must carry the same frozen row count')
        if _int_field(_ts_n, 'binding.test_slice.n_seqs') != _int_field(_ev_n, 'eval.n_seqs_clean'):
            raise SystemExit(f'[v12] REFUSE: binding.test_slice.n_seqs={_ts_n} disagrees with eval.n_seqs_clean={_ev_n} — the machine-checkable binding block and the evaluation plan must describe the same slice')
    if _eval and _int_field(_eval.get('n_seqs_clean') or 0, 'eval.n_seqs_clean') <= 0:
        raise SystemExit('[v12] REFUSE: eval.n_seqs_clean is missing/zero — the frozen row count is part of the held-out slice, not a runtime default')
    _probe = _eval.get('distractor_probe') if isinstance(_eval, dict) else None
    if _probe is not None:
        if not isinstance(_probe, dict):
            raise SystemExit(f'[v12] REFUSE: eval.distractor_probe must be an object ({_probe!r})')
        _pmode = _probe.get('mode')
        if _pmode is None:
            _pmode = 'uniform'
        if not isinstance(_pmode, str) or _pmode not in V10.PROBE_MODES:
            raise SystemExit(
                f'[v12] REFUSE: eval.distractor_probe.mode={_pmode!r} is not one '
                f'of the implemented corruption modes {sorted(V10.PROBE_MODES)}')
        for _pk in ('rhos', 'eval_lens'):
            if _pk not in _probe:
                continue
            _pv = _probe.get(_pk)
            if not isinstance(_pv, list) or not _pv:
                raise SystemExit(f'[v12] REFUSE: eval.distractor_probe.{_pk} must be a non-empty list ({_pv!r})')
            for _x in _pv:
                if isinstance(_x, (bool, np.bool_)):
                    raise SystemExit(f'[v12] REFUSE: eval.distractor_probe.{_pk} contains a boolean ({_x!r})')
                try:
                    _xf = float(_x)
                except (TypeError, ValueError, OverflowError):
                    raise SystemExit(f'[v12] REFUSE: eval.distractor_probe.{_pk} contains a non-numeric value ({_x!r})')
                if not math.isfinite(_xf):
                    raise SystemExit(f'[v12] REFUSE: eval.distractor_probe.{_pk} contains a non-finite value ({_x!r})')
                if _pk == 'rhos' and not (0.0 <= _xf <= 1.0):
                    raise SystemExit(f'[v12] REFUSE: eval.distractor_probe.rhos value {_xf!r} is outside [0,1]')
                if _pk == 'eval_lens':
                    if _int_field(_x, 'eval.distractor_probe.eval_lens entry') < 2:
                        raise SystemExit(f'[v12] REFUSE: eval.distractor_probe.eval_lens value {_x!r} has no next-token prediction (needs an integer length >= 2)')
        _eff_rhos = [float(x) for x in
                     (_probe.get('rhos') or V10.PROBE_MODES[_pmode])]
        if (_pmode in ('mask', 'delete', 'doc')
                and any(abs(x - 1.0) > 1e-9 for x in _eff_rhos)):
            raise SystemExit(
                f'[v12] REFUSE: eval.distractor_probe.mode={_pmode!r} is '
                f'all-or-nothing and only admits rho=1.0 (got {_eff_rhos})')
        for _pk in ('target', 'n_seq'):
            if _pk in _probe:
                _iv = _int_field(_probe.get(_pk), f'eval.distractor_probe.{_pk}')
                if _iv <= 0:
                    raise SystemExit(f'[v12] REFUSE: eval.distractor_probe.{_pk} must be positive ({_iv})')
        # The probe runs on the SAME frozen test slice as the by-length panel.
        # If it asks for a longer sequence or more rows, run_finaltest sizes
        # the slice from by_length_ppl/n_seqs_clean and the probe indexes out
        # of bounds AFTER the one-shot test access.  Validate the effective
        # defaults too, not only the fields the manifest happens to spell out.
        _plens = [_int_field(_x, 'eval.distractor_probe.eval_lens entry')
                  for _x in (_probe.get('eval_lens') or [])]
        _eff_plens = _plens or [2048, 4096]
        if _bl_ints and max(_eff_plens) > max(_bl_ints):
            raise SystemExit(f'[v12] REFUSE: eval.distractor_probe.eval_lens={_eff_plens} extends beyond the frozen test slice width max(by_length_ppl)={max(_bl_ints)} — enlarge by_length_ppl (and re-freeze) or shrink the probe')
        _nseq_frozen = None
        if _eval.get('n_seqs_clean') is not None:
            _nseq_frozen = _int_field(_eval.get('n_seqs_clean'), 'eval.n_seqs_clean')
            _eff_nseq = (_int_field(_probe.get('n_seq'), 'eval.distractor_probe.n_seq')
                         if 'n_seq' in _probe else _nseq_frozen)
            if _eff_nseq > _nseq_frozen:
                raise SystemExit(f'[v12] REFUSE: eval.distractor_probe.n_seq={_eff_nseq} exceeds the frozen test row count n_seqs_clean={_nseq_frozen}')
        if 'target' in _probe:
            _ptarget = _int_field(_probe.get('target'), 'eval.distractor_probe.target')
            if min(_eff_plens) <= _ptarget:
                raise SystemExit(f'[v12] REFUSE: eval.distractor_probe.target={_ptarget} must be smaller than every probe eval_len {sorted(_eff_plens)} (the far context would be empty/negative)')
    if _probe is None and _bl_ints and max([2048, 4096]) > max(_bl_ints):
        # No probe block at all: run_finaltest still uses its fixed default
        # probe lengths, so the default grid must fit the frozen slice before
        # the one-shot test access.
        raise SystemExit(f'[v12] REFUSE: the default distractor probe eval_lens=[2048, 4096] extends beyond the frozen test slice width max(by_length_ppl)={max(_bl_ints)} — declare a shorter eval.distractor_probe.eval_lens or enlarge by_length_ppl (and re-freeze)')
    _tokb = (_bind.get('tokenizer') or {}) if isinstance(_bind, dict) else {}
    if _tokb and not isinstance(_tokb, dict):
        raise SystemExit(f'[v12] REFUSE: binding.tokenizer must be an object ({_tokb!r})')
    if _tokb and str(_tokb.get('fit', '')).lower() and 'train' not in str(_tokb.get('fit', '')).lower():
        raise SystemExit(f'[v12] REFUSE: binding.tokenizer.fit={_tokb.get("fit")!r} is not the train-only fit required by the freeze')
    _test_raw = _stat.get('test', 'exact two-sided sign-flip over paired per-seed deltas')
    if not isinstance(_test_raw, str) or not _test_raw.strip():
        raise SystemExit(f'[v12] REFUSE: statistics.test={_test_raw!r} is not a non-empty text test identity')
    _test = _test_raw.lower()
    if ('sign-flip' not in _test and 'sign flip' not in _test) or 'two-sided' not in _test:
        raise SystemExit(f'[v12] REFUSE: freeze manifest pre-registers test {_stat.get("test")!r}, but this analyzer only implements the exact two-sided sign-flip test — update the code or re-freeze')
    _corr_raw = _stat.get('correction', 'holm step-down')
    if not isinstance(_corr_raw, str) or not _corr_raw.strip():
        raise SystemExit(f'[v12] REFUSE: statistics.correction={_corr_raw!r} is not a non-empty text correction identity')
    _corr = _corr_raw.lower()
    _neg_holm = any(_neg in _corr for _neg in (
        'no holm', 'no-holm', 'not holm', 'not-holm', 'non-holm',
        'without holm', 'without-holm'))
    if 'holm' not in _corr or _neg_holm:
        raise SystemExit(f'[v12] REFUSE: freeze manifest pre-registers correction {_stat.get("correction")!r}, but this analyzer only implements Holm step-down — update the code or re-freeze')
    for comp in m['primary_comparisons']:
        if not isinstance(comp, dict):
            raise SystemExit(f'[v12] REFUSE: every frozen primary comparison must be an object ({comp!r})')
        for _k in ('name', 'a', 'b'):
            if not comp.get(_k):
                raise SystemExit(f'[v12] REFUSE: a frozen primary comparison lacks {_k!r} ({comp!r})')
        if not isinstance(comp.get('name'), str):
            # Numeric/boolean names (1 vs 1.0 vs True) can be distinct after
            # str() but collapse as dict keys in the statistics family, which
            # would silently shrink the Holm correction.
            raise SystemExit(f'[v12] REFUSE: comparison name must be a string ({comp.get("name")!r})')
        _ep = comp.get('endpoint')
        if _ep is not None:
            if not isinstance(_ep, str) or '@' not in _ep:
                raise SystemExit(f'[v12] REFUSE: comparison {comp.get("name")!r} endpoint {_ep!r} is not a by-length endpoint (e.g. by_len.ppl@4096)')
            _ep_head, _, _tail = _ep.rpartition('@')
            # The analyzer implements exactly one endpoint family; accepting
            # another prefix would compute by_len.ppl numbers under a
            # different pre-registered label.
            if _ep_head != 'by_len.ppl':
                raise SystemExit(f'[v12] REFUSE: comparison {comp.get("name")!r} endpoint {_ep!r} is not a by_len.ppl endpoint (only by_len.ppl@<length> is implemented)')
            # Canonical decimal spelling only: `isdigit()` also accepts
            # non-ASCII digits, and int('04096') would silently alias the
            # canonical length 4096.  The endpoint length is part of the
            # pre-registered identity, not a loose numeric hint.
            if re.fullmatch(r'[1-9][0-9]*', _tail) is None:
                raise SystemExit(f'[v12] REFUSE: comparison {comp.get("name")!r} endpoint {_ep!r} does not end in a canonical positive integer length')
            try:
                _tail_i = _int_field(_tail, f'comparison {comp.get("name")!r} endpoint length')
            except SystemExit:
                raise SystemExit(f'[v12] REFUSE: comparison {comp.get("name")!r} endpoint {_ep!r} does not end in a canonical positive integer length') from None
            if _tail_i < 2:
                raise SystemExit(f'[v12] REFUSE: comparison {comp.get("name")!r} endpoint {_ep!r} has no next-token prediction (needs an integer length >= 2)')
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
    # The declared n-floor is part of the decision rule.  Refuse a
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
        if _sig is not None:
            for _sk, _sv in _sig.items():
                if isinstance(_sv, (bool, np.bool_)):
                    raise SystemExit(f'[v12] REFUSE: training_recipe_signature.{_sk}={_sv!r} is a boolean, not a numeric recipe knob')
                if isinstance(_sv, numbers.Real):
                    try:
                        _svf = float(_sv)
                    except (TypeError, ValueError, OverflowError):
                        _svf = float('nan')
                    if not math.isfinite(_svf):
                        raise SystemExit(f'[v12] REFUSE: training_recipe_signature.{_sk}={_sv!r} is not finite')
        _per = _bind.get('per_arm_recipe_signature')
        if _per is not None:
            if not isinstance(_per, dict) or set(_per) != set(m['arms']):
                raise SystemExit(f'[v12] REFUSE: binding.per_arm_recipe_signature must be an object keyed by exactly the frozen arms {sorted(m["arms"])} (got {_per!r})')
            for _arm, _asig in _per.items():
                if not isinstance(_asig, dict) or not _asig:
                    raise SystemExit(f'[v12] REFUSE: binding.per_arm_recipe_signature[{_arm!r}] must be a non-empty object')
                for _sk, _sv in _asig.items():
                    if isinstance(_sv, (bool, np.bool_)):
                        raise SystemExit(f'[v12] REFUSE: per-arm signature {_arm!r}.{_sk}={_sv!r} is a boolean, not a numeric recipe knob')
                    if isinstance(_sv, numbers.Real):
                        try:
                            _svf = float(_sv)
                        except (TypeError, ValueError, OverflowError):
                            _svf = float('nan')
                        if not math.isfinite(_svf):
                            raise SystemExit(f'[v12] REFUSE: per-arm signature {_arm!r}.{_sk}={_sv!r} is not finite')
                if 'mr' in _asig:
                    try:
                        _mr = float(_asig['mr'])
                    except (TypeError, ValueError, OverflowError):
                        raise SystemExit(f'[v12] REFUSE: per-arm mr for {_arm!r} is not numeric ({_asig["mr"]!r})')
                    if not math.isfinite(_mr) or _mr <= 1.0:
                        raise SystemExit(f'[v12] REFUSE: per-arm mr for {_arm!r} must be finite and > 1 ({_mr!r})')
    # Validate the training-corpus binding shape at load time too: a
    # malformed declaration must fail before any held-out access instead of
    # being treated as "no literal value" and paired with a derivable train_fp.
    _frozen_train_fp(m)
    # An implementation-mode binding declared under both its alias spellings
    # must name the same mode before any record is admitted.
    _frozen_implementation(m)
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
    # so any rewrite — even one that preserves size and mtime — still forces a
    # fresh hash.
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
    # The new and legacy spellings are aliases for ONE frozen recipe.  If both
    # are present with different content, the training recipe is ambiguous;
    # refuse the binding.
    _sig_new = _bind.get('training_recipe_signature')
    _sig_old = _bind.get('training_recipe')
    if (isinstance(_sig_new, dict) and _sig_new
            and isinstance(_sig_old, dict) and _sig_old):
        try:
            _same_sig = (json.dumps(_sig_new, sort_keys=True, default=str)
                         == json.dumps(_sig_old, sort_keys=True, default=str))
        except Exception:
            _same_sig = False
        if not _same_sig:
            raise SystemExit(
                '[v12] REFUSE: binding.training_recipe_signature and '
                'binding.training_recipe declare DIFFERENT frozen recipes — '
                'the training recipe is ambiguous')
    sig = _sig_new
    if sig is None:
        sig = _sig_old
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
    # Inspect the same locations/keys as _frozen_train_fp: a top-level
    # training_data_fingerprint (or its aliases) is just as binding as one
    # nested under `binding`.
    _keys = ('training_data_fingerprint', 'canonical_train_fp', 'train_fp', 'training_fp')
    for _obj in (manifest.get('binding') or {}, manifest):
        if not isinstance(_obj, dict):
            continue
        if any(_obj.get(_k) is not None for _k in _keys):
            return True
    return False


def _frozen_train_fp(manifest):
    """Canonical training-corpus fingerprint declared by the frozen plan.

    Accepted shapes: ``binding.training_data_fingerprint`` (string or an
    object with ``canonical_train_fp`` / ``train_fp`` / ``value``),
    ``binding.train_fp``, or a top-level ``training_data_fingerprint``.
    ``None`` means the manifest froze the rule but not a literal value; in that
    case every admission/checkpoint must still agree on ONE non-null train_fp
    and the first P5T access locks it.
    """
    _fp_keys = ('training_data_fingerprint', 'canonical_train_fp',
                'train_fp', 'training_fp')

    def _take(label, value):
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (str, dict)):
            raise SystemExit(
                f'[v12] REFUSE: {label}={value!r} is not a training-corpus '
                'fingerprint (expected a non-empty string or an object)')
        if isinstance(value, str):
            if not value:
                raise SystemExit(
                    f'[v12] REFUSE: {label} is an empty training-corpus '
                    'fingerprint — an empty/absent identity cannot bind a corpus')
            return value
        # Collect every declared literal first: two different non-null aliases
        # in one object cannot both bind the corpus.
        _found = []
        for _sub in ('canonical_train_fp', 'train_fp', 'value'):
            if _sub not in value:
                continue
            _sv = value[_sub]
            if _sv is None:
                continue
            if (isinstance(_sv, (bool, np.bool_))
                    or not isinstance(_sv, str) or not _sv):
                raise SystemExit(
                    f'[v12] REFUSE: {label}.{_sub}={_sv!r} is not a '
                    'non-empty text fingerprint')
            _found.append((f'{label}.{_sub}', _sv))
        if len({_v for _n, _v in _found}) > 1:
            raise SystemExit(
                f'[v12] REFUSE: {label} declares conflicting training-corpus '
                f'fingerprints ({", ".join(f"{_n}={_v!r}" for _n, _v in _found)}) '
                '— an ambiguous binding cannot identify the frozen corpus')
        # Declared geometry fields are part of the binding, so validate them
        # here rather than at P5T, including when a literal fingerprint is
        # present.
        for _cnt, _minimum in (('seq_len', 2), ('n_train_tokens', 1)):
            if _cnt not in value or value[_cnt] is None:
                continue
            _iv = L._as_seed_int(value[_cnt])
            if _iv is None or _iv < _minimum:
                raise SystemExit(
                    f'[v12] REFUSE: {label}.{_cnt}={value[_cnt]!r} is not an '
                    f'exact integer >= {_minimum}')
        return _found[0][1] if _found else None

    _found = []
    for _obj_name, _obj in (('binding', manifest.get('binding') or {}),
                            ('manifest', manifest)):
        if not isinstance(_obj, dict):
            continue
        for _k in _fp_keys:
            if _k not in _obj:
                continue
            _v = _take(f'{_obj_name}.{_k}', _obj.get(_k))
            if _v is not None:
                _found.append((f'{_obj_name}.{_k}', _v))
    if len({_v for _n, _v in _found}) > 1:
        raise SystemExit(
            '[v12] REFUSE: the frozen plan declares conflicting '
            'training-corpus fingerprints across aliases '
            f'({", ".join(f"{_n}={_v!r}" for _n, _v in _found)}) — '
            'the corpus identity is ambiguous')
    return _found[0][1] if _found else None


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
        _suffix = _k + '_expected'
        _e = manifest.get(_suffix)
        _b = manifest.get(_k)
        if _e is not None and _b is not None:
            try:
                _same = bool(_e == _b)
            except Exception:
                _same = False
            if not _same:
                raise SystemExit(
                    f'[v12] REFUSE: implementation binding {_k!r} is declared '
                    f'twice with different values ({_suffix}={_e!r}, {_k}={_b!r}) '
                    '— the frozen run mode is ambiguous')
        # A present-but-null `*_expected` key carries no override; use the
        # non-null alias when present.
        _v = _e if _e is not None else _b
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
        if _k == 'amp_dtype':
            if _got is None:
                _got = 'fp32'
            if _want is None:
                _want = 'fp32'
            if (not isinstance(_got, str) or not _got
                    or not isinstance(_want, str) or not _want):
                return (f"train_knobs.amp_dtype={tk.get(_k)!r} / frozen "
                        f"{impl.get(_k)!r} is not a text dtype name")
            if _got.lower() != _want.lower():
                return f"train_knobs.amp_dtype={_got!r} does not match the frozen {_want!r}"
        else:
            # A truthy string/int is not a boolean declaration; bool('false')
            # would otherwise match a frozen True and admit the wrong run mode.
            if (isinstance(_got, (bool, np.bool_))
                    and isinstance(_want, (bool, np.bool_))):
                if bool(_got) != bool(_want):
                    return f"train_knobs.{_k}={_got!r} does not match the frozen {_want!r}"
            else:
                return (f"train_knobs.{_k}={_got!r} / frozen {_want!r} "
                        'is not a boolean implementation flag')
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
    该配方的字符串格式不携带 `ee`/`es`，故缺少这两个键时不拒绝整条记录。
    理由：`ee`/`es` 只门控**诊断性**的固定批训练侧评测（`diag_eval_every`：
    健康监控专用，绝不参与模型选择，也不改动权重或冻结端点——端点在测试
    切片上重评），因此它们不是权重的训练配方；P5C 自身以
    `diag_eval_every=250` 训练，该事实由阶段配置与 `run_length_extrapolation`
    的调用点约束，不由 p1l 配方字符串承载。manifest **v4** 已据此把
    `ee`/`es` 从可机检签名中移除（预访问修订，见其 revision.history）；
    对 v3 时代的旧清单，本函数对 p1l 配方跳过这两个键，而不是拒绝整条
    记录（v3 的 `ee=250`/`es=128` 对 p1l 记录本就是不可机检的死字母）。"""
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
        if isinstance(got, (bool, np.bool_)) or isinstance(want, (bool, np.bool_)):
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
    seed parse.  Training refuses before it starts if the manifest
    would be rejected at P5T load time (stamp mismatch, unresolvable power
    gate, binding inconsistency).
    """
    m = _load_manifest(FREEZE_MANIFEST)
    seeds = m.get('seeds')
    if not isinstance(seeds, list) or not seeds:
        raise SystemExit(f'[v12] REFUSE: {FREEZE_MANIFEST} has no usable seed list')
    return [int(s) for s in seeds]


_LOCK_NAME = '.freeze_lock.json'

# (variant, seed) -> health-status string for cells whose one-shot test
# evaluation was permitted by V12_ALLOW_UNHEALTHY=1 despite a failing/missing
# health panel.  Populated by _training_health_gate, read by the P5T record
# writer so the override is machine-visible on every admitted record (the
# override must not live only in stdout, which is not an artifact).
_UNHEALTHY_OVERRIDE_CELLS = {}


def _freeze_lock_path(outdir):
    return os.path.join(outdir, _LOCK_NAME)


def _freeze_lock_mirror_path():
    """Git-tracked mirror of the freeze lock (lives next to the manifest).

    The outdir lock lives inside a gitignored results directory, so renaming or
    deleting that directory resets the freeze with no trace — exactly the
    one-shot-access promise the lock is supposed to enforce.  The mirror is
    meant to be COMMITTED (it is whitelisted in .gitignore), so a results-dir
    rename cannot erase the fact that a first test access happened."""
    return os.path.join(os.path.dirname(os.path.abspath(FREEZE_MANIFEST)) or '.',
                        'confirmation_freeze_lock.json')


def _is_smoke_outdir(outdir):
    return 'smoke' in os.path.basename(os.path.abspath(outdir)).lower()


def _mirror_active(outdir):
    """The mirror is maintained only for the CANONICAL confirmation outdir.

    Ad-hoc callers (tests, alternate result directories) must not create or
    consult the repo-level mirror: it exists solely to make renaming/deleting
    the ONE frozen results directory non-reversible."""
    if _is_smoke_outdir(outdir):
        return False
    try:
        _canon = os.path.abspath(TEST_OUTDIR)
        _here = os.path.abspath(outdir)
    except Exception:
        return False
    return _here == _canon


def _write_lock_mirror(outdir, lock):
    """Mirror the lock into the git-tracked mirror file (canonical outdir only)."""
    if not _mirror_active(outdir):
        return
    try:
        mirror = dict(lock)
        mirror['mirror_of_outdir'] = os.path.abspath(outdir)
        L.atomic_write_json(_freeze_lock_mirror_path(), mirror, indent=2)
    except Exception as _me:
        # A failed mirror write must not kill a legitimate run, but it must be
        # visible: without the mirror, directory-rename tamper-evidence is off.
        print(f'[v12] WARNING: could not write the freeze-lock mirror {_freeze_lock_mirror_path()} ({type(_me).__name__}: {_me}) — the directory-level tamper evidence is NOT active for this round; commit the mirror once writable')


def _verify_lock_mirror(outdir, lock=None):
    """Cross-check the local outdir lock against the git-tracked mirror.

    Returns the mirror dict when one exists (None otherwise).  Refuses when
    the two disagree on identity (manifest / test slice / train corpus), or
    when the mirror proves a first access happened but the outdir lock is
    gone (renamed/deleted results directory) — unless V12_REFREEZE=1."""
    if not _mirror_active(outdir):
        return None
    mp = _freeze_lock_mirror_path()
    if not os.path.exists(mp):
        return None
    try:
        mirror = json.load(open(mp, encoding='utf-8'))
    except Exception as e:
        raise SystemExit(f'[v12] REFUSE: freeze-lock mirror {mp} exists but cannot be parsed ({type(e).__name__}: {e}) — the mirror is part of the one-shot-access evidence; restore it from git history or re-freeze consciously')
    if not isinstance(mirror, dict):
        raise SystemExit(f'[v12] REFUSE: freeze-lock mirror {mp} is valid JSON but not an object ({type(mirror).__name__}) — refusing to interpret it')
    lp = _freeze_lock_path(outdir)
    _refreeze = os.environ.get('V12_REFREEZE', '').strip().lower() in ('1', 'true', 'yes', 'on')
    if not os.path.exists(lp):
        if not _refreeze:
            raise SystemExit(
                f'[v12] REFUSE: the git-tracked freeze-lock mirror {mp} records a first '
                f'test-split access (manifest {str(mirror.get("manifest_sha256"))[:12]}…, '
                f'locked_at={mirror.get("locked_at")}, outdir={mirror.get("mirror_of_outdir")}), but '
                f'there is no lock in {outdir} — the results directory was renamed, deleted or '
                f'never copied here. Renaming the results directory must not reset the freeze; '
                f'restore the original directory or set V12_REFREEZE=1 (the earlier round is then '
                f'downgraded to exploratory in the plan history)')
        print('[v12] WARNING: V12_REFREEZE=1 with a mirror but no local lock — treating the mirrored round as superseded and writing a fresh lock')
        return mirror
    if lock is None:
        try:
            lock = json.load(open(lp, encoding='utf-8'))
        except Exception:
            return mirror
    for _k in ('manifest_sha256', 'test_ids_fp', 'train_fp'):
        _mv = mirror.get(_k)
        _lv = lock.get(_k)
        if _mv is None:
            # The mirror predates the two-phase binding (or a failed mirror
            # write left the key absent); a more specific local value can be
            # backfilled, but it may never contradict a mirrored identity.
            continue
        if _lv != _mv:
            if _refreeze:
                print(f'[v12] WARNING: V12_REFREEZE=1 — the local lock disagrees with '
                      f'the git-tracked mirror on {_k} ({_lv!r} vs {_mv!r}); the '
                      'mirrored round must be reported as exploratory')
                continue
            raise SystemExit(
                f'[v12] REFUSE: the freeze lock in {outdir} disagrees with the git-tracked '
                f'mirror on {_k} ({_lv!r} vs mirror {_mv!r}) — one of the two was modified '
                f'independently (a missing local value is NOT a reset). Investigate before '
                f'any further test access (V12_REFREEZE=1 re-freezes consciously).')
    return mirror


def _check_or_write_freeze_lock(outdir, manifest, test_ids_fp, stage='access',
                                train_fp=None):
    """Persistent first-test-access lock.

    首次测试访问后禁止修改计划：P5T 一旦在 outdir 落下首条记录（或本函数
    被调用），锁文件即记录 manifest sha256
    与 test 切片指纹。之后的 P5T/analysis 若带着**不同的** manifest 哈希
    或不同 test 指纹运行，将被拒绝——除非显式 V12_REFREEZE=1（并大声
    打印该轮降级为探索性的警告）。``train_fp`` 一旦记录，后续训练集
    指纹不一致同样拒绝。

    锁同时镜像到 git 管理的 ``confirmation_freeze_lock.json``（见
    ``_freeze_lock_mirror_path``）：结果目录被改名/删除时，镜像证明首次
    访问发生过，冻结不会被静默重置。"""
    lp = _freeze_lock_path(outdir)
    _sha = manifest.get('manifest_sha256')
    _mirror_prev = _verify_lock_mirror(outdir)
    if os.path.exists(lp):
        try:
            lock = json.load(open(lp, encoding='utf-8'))
        except Exception as e:
            raise SystemExit(f'[v12] REFUSE: freeze lock {lp} exists but cannot be parsed ({type(e).__name__}: {e}) — delete it only if you can prove no test-split record was ever written under it')
        if not isinstance(lock, dict):
            raise SystemExit(f'[v12] REFUSE: freeze lock {lp} is valid JSON but not an object ({type(lock).__name__}) — refusing to interpret it')
        _manifest_changed = lock.get('manifest_sha256') != _sha
        _test_changed = (test_ids_fp is not None
                         and lock.get('test_ids_fp') not in (None, test_ids_fp))
        _train_changed = (train_fp is not None
                          and lock.get('train_fp') not in (None, train_fp))
        _refreeze = os.environ.get('V12_REFREEZE', '').strip().lower() in ('1', 'true', 'yes', 'on')
        if train_fp is None and lock.get('train_fp') is not None and not _refreeze:
            raise SystemExit(f'[v12] REFUSE: the freeze lock pins a training corpus (train_fp={lock.get("train_fp")}) but the current records do not prove one — a confirmation run cannot drop the training-set binding')
        if (_manifest_changed or _test_changed or _train_changed) and not _refreeze:
            if _manifest_changed:
                raise SystemExit(f'[v12] REFUSE: the test split was already accessed under a DIFFERENT frozen manifest (lock {lp}: {lock.get("manifest_sha256")} != current {_sha}, locked_at={lock.get("locked_at")}). Changing the plan after seeing test data is selection bias by construction. To consciously re-freeze, set V12_REFREEZE=1 — the earlier round must then be reported as exploratory in the plan history.')
            if _test_changed:
                raise SystemExit(f'[v12] REFUSE: the held-out test slice changed after the first access (lock has ids_fp {lock.get("test_ids_fp")}, current {test_ids_fp}) — the test data itself is part of the freeze; investigate before proceeding (V12_REFREEZE=1 consciously re-freezes to a new slice)')
            raise SystemExit(f'[v12] REFUSE: the training corpus changed after the first test access (lock has train_fp {lock.get("train_fp")}, current {train_fp}) — a different training set cannot enter the frozen confirmation (V12_REFREEZE=1 consciously re-freezes to a new corpus)')
        _dirty = False
        if _manifest_changed or _test_changed or _train_changed:
            print(f'[v12] WARNING: V12_REFREEZE=1 — the locked plan/test/train identity changed (manifest {lock.get("manifest_sha256")} -> {_sha}, test_ids_fp {lock.get("test_ids_fp")} -> {test_ids_fp}, train_fp {lock.get("train_fp")} -> {train_fp}); the earlier round is DOWNGRADED to exploratory and this must be recorded in the plan history')
            # Persist the conscious re-freeze: the lock carries the current
            # identity so later runs do not refuse, and the superseded plan is
            # retained for the exploratory record.  The lock is written ONCE
            # below, after every check has passed, so a failure can never leave
            # a half-updated (changed plan + stale fingerprints) file.
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
        # Backfill the git-tracked mirror when the lock predates it (the
        # mirror is the newer half of the tamper-evidence pair).
        if _mirror_active(outdir) and not os.path.exists(_freeze_lock_mirror_path()):
            lock['history'] = list(lock.get('history') or []) + [{'stage': 'mirror-backfilled', 'at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                                                                  'note': 'lock predates the mirror; mirroring now'}]
            _dirty = True
        if _dirty:
            L.atomic_write_json(lp, lock, indent=2)
            _write_lock_mirror(outdir, lock)
            if _manifest_changed or _test_changed or _train_changed:
                print(f'[v12] freeze lock UPDATED: {lp} (manifest {str(_sha)[:12]}…; {len(lock["superseded"])} superseded plan(s) kept)')
            else:
                print(f'[v12] freeze lock updated with the first-access slice/train fingerprint: {lp}')
        return lock
    lock = {'manifest_sha256': _sha, 'manifest_path': os.path.abspath(manifest.get('_path') or FREEZE_MANIFEST), 'frozen_at': manifest.get('frozen_at'), 'locked_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'test_ids_fp': test_ids_fp, 'train_fp': train_fp, 'stage': stage, 'code_semantics': L.code_stamp(), 'history': [{'stage': stage, 'at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'test_ids_fp': test_ids_fp, 'train_fp': train_fp}], 'note': 'first test-split access: the plan may not change after this point (V12_REFREEZE=1 overrides consciously, downgrading the earlier round to exploratory)'}
    if _mirror_prev is not None:
        # V12_REFREEZE after a directory rename/delete: keep the superseded
        # identity from the mirror so the exploratory history survives.
        prev = {_k: _mirror_prev.get(_k) for _k in ('manifest_sha256', 'test_ids_fp', 'train_fp', 'locked_at', 'frozen_at')}
        lock['superseded'] = [_mirror_prev.get('superseded'), prev] if _mirror_prev.get('superseded') else [prev]
        lock['history'] = lock['history'] + [{'stage': 'refreeze-mirror', 'at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                                              'note': 'local lock was absent but the git-tracked mirror proved an earlier access; that round is downgraded to exploratory',
                                              'superseded': prev}]
        lock['note'] = 're-frozen under V12_REFREEZE=1 after a missing local lock (mirror evidence): the earlier round is downgraded to exploratory'
    L.atomic_write_json(lp, lock, indent=2)
    _write_lock_mirror(outdir, lock)
    print(f'[v12] freeze lock written: {lp} (manifest {_sha[:12]}…, stage {stage}, test_ids_fp {str(test_ids_fp)[:12]}…)')
    return lock

def _training_health_gate(cfg, v, seed, ckpt_meta, train_rec):
    """Refuse the one-shot test evaluation when training health is unknown or bad.

    This is deliberately NOT a model-selection rule: it does not change arms,
    seeds, steps or the endpoint.  It only prevents burning the frozen test
    access on a run whose own diagnostic curve says non-finite/overfitting/
    stalled.  `V12_ALLOW_UNHEALTHY=1` is the explicit, logged override; every
    cell admitted through the override is recorded in
    ``_UNHEALTHY_OVERRIDE_CELLS`` and stamped on its P5T record as
    ``allow_unhealthy_override=True`` (plus the health status), so the
    override is auditable from the artifacts alone.
    """
    if L._as_bool_flag(cfg.get('smoke')) is True:
        return None
    health = (ckpt_meta or {}).get('curve_health') or (train_rec or {}).get('curve_health')
    allow = os.environ.get('V12_ALLOW_UNHEALTHY', '').strip().lower() in ('1', 'true', 'yes', 'on')
    if not isinstance(health, dict):
        msg = f'{v} s{seed}: no training curve/health evidence in the checkpoint or its record'
        if allow:
            print(f'[p5t] WARNING: {msg}; V12_ALLOW_UNHEALTHY=1 permits the test evaluation')
            _UNHEALTHY_OVERRIDE_CELLS[(v, L._as_seed_int(seed) if L._as_seed_int(seed) is not None else seed)] = 'unknown'
            return None
        return msg + ' — refusing to use the one-shot test access (set V12_ALLOW_UNHEALTHY=1 only with a written justification)'
    status = str(health.get('status') or 'unknown')
    if status == 'healthy':
        return None
    if status == 'unmonitored' and allow:
        print(f'[p5t] WARNING: {v} s{seed}: training health is unmonitored; V12_ALLOW_UNHEALTHY=1 permits the test evaluation')
        _UNHEALTHY_OVERRIDE_CELLS[(v, L._as_seed_int(seed) if L._as_seed_int(seed) is not None else seed)] = status
        return None
    msg = f'{v} s{seed}: training health is {status!r} ({health.get("reason") or "no reason recorded"})'
    if allow:
        print(f'[p5t] WARNING: {msg}; V12_ALLOW_UNHEALTHY=1 permits the test evaluation')
        _UNHEALTHY_OVERRIDE_CELLS[(v, L._as_seed_int(seed) if L._as_seed_int(seed) is not None else seed)] = status
        return None
    return msg + ' — refusing to use the one-shot test access'


@torch.inference_mode()
def _finaltest_bylen_cell(cfg, manifest, v, seed, test_ids, split='test', train_fp=None):
    """One (arm, seed) clean by-length evaluation on the TEST split.

    记录现在额外携带 manifest_sha256 与
    ckpt_sha256（绑定「哪份冻结计划、哪份权重」）；recipe 必须通过冻结
    清单的结构化配方签名；resume 的 stale 判定
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
    ck = os.path.join(cfg['ckpt_dir'], L.direct_ckpt_name(v, seed))
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
    # The caller passes the value derived BEFORE the first test read
    # and written into the freeze lock.  Recomputing it here from a null
    # manifest field would reintroduce the fail-open path and could accept a
    # record from a different corpus.
    _want_train_fp = train_fp if train_fp is not None else _frozen_train_fp(manifest)
    # Manifests that DECLARE binding.training_data_fingerprint require the
    # binding unconditionally, even while canonical_train_fp is still null.
    # Manifests/records without either field stay admissible.
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
    eval_lens = [int(x) for x in ((manifest.get('eval') or {}).get('by_length_ppl') or cfg['eval_lens'])]
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
        # The record's own identity must agree with the summary key used to
        # fetch it; otherwise a mismatched or merged record could make this
        # cell look complete without ever measuring the requested (variant, seed).
        if (rec.get('variant') != v
                or L._as_seed_int(rec.get('seed')) != int(seed)):
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
        # n_seq is stored explicitly and a mismatch is treated as stale.
        if _record_int(rec.get('n_seq')) != int(test_ids.shape[0]):
            return True
        if _msha is not None and rec.get('manifest_sha256') != _msha:
            return True  # the manifest changed after this record was written
        # by_len_cells() only checks that a usable ppl exists; it does not
        # reject a truncated/partial cell.  Use the same full-shape predicate
        # as admission.
        if (not all((L.by_len_cells([rec], Ln) for Ln in eval_lens))
                or _by_len_shape_reason(rec, test_ids.shape[0]) is not None):
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
    # Post-access selective-remeasurement guard.  A stale record whose key
    # already carries a measurement, evaluated now on DIFFERENT weights, would
    # silently OVERWRITE that cell's one-shot test measurement (resume's _merge
    # writes by key).  The freeze lock pins the plan/slice/corpus but not the
    # checkpoint set, so this is the last line of defence against
    # retrain-a-chosen-cell-and-remeasure after the test split was read.
    if (split != 'smoke' and isinstance(_cur, dict)
            and (os.path.exists(_freeze_lock_path(outdir))
                 or os.path.exists(_freeze_lock_mirror_path()))):
        _prior_sha = _cur.get('ckpt_sha256')
        _now_sha = _ensure_ck_sha()
        _refrz = os.environ.get('V12_REFREEZE', '').strip().lower() in ('1', 'true', 'yes', 'on')
        if (_now_sha is None or _prior_sha is None or _prior_sha != _now_sha) and not _refrz:
            del d
            raise SystemExit(
                f'[v12] REFUSE {v} s{seed}: this cell already has a test measurement '
                f'(ckpt_sha256={str(_prior_sha)[:12]}…) but the checkpoint on disk now hashes to '
                f'{str(_now_sha)[:12]}… — re-measuring the same (arm, seed) on different weights '
                'after the first test access would replace a one-shot measurement selectively '
                '(selection bias by construction). Restore the original checkpoint, or set '
                'V12_REFREEZE=1 to re-freeze consciously (the earlier round is downgraded to '
                'exploratory in the plan history).')
        if _prior_sha is not None and _now_sha is not None and _prior_sha != _now_sha:
            print(f'[v12] WARNING: V12_REFREEZE=1 — {v} s{seed} is re-measured on different weights; the prior measurement (ckpt {str(_prior_sha)[:12]}…) is replaced and the earlier round is downgraded to exploratory')
            out['replaced_prior_ckpt_sha256'] = _prior_sha
    _arch = (int(d.get('d', 256)), int(d.get('n_layers', 6)), int(d.get('n_heads', 8)), int(d.get('d_head', 32)))
    # Explicit architecture / MLP-width verification against the frozen
    # binding.  The recipe string is decoded from injected constants on the
    # p1l path, so a matching recipe alone cannot prove the checkpoint's real
    # shape; the probe path already hard-refuses architecture mismatches —
    # mirror that check for the confirmatory by-length cells.
    _bind = manifest.get('binding') or {}
    _arch_sig = _bind.get('training_recipe_signature') or {}
    if all(isinstance(_arch_sig.get(_k), (int, float)) and not isinstance(_arch_sig.get(_k), bool)
           for _k in ('d', 'L', 'H', 'Dh')):
        _want_arch = (int(_arch_sig['d']), int(_arch_sig['L']), int(_arch_sig['H']), int(_arch_sig['Dh']))
        if _arch != _want_arch:
            del d
            raise SystemExit(
                f'[v12] REFUSE {v} s{seed}: checkpoint architecture {_arch} != the frozen '
                f'training_recipe_signature {_want_arch} — a differently shaped model cannot '
                'enter the confirmation panel even when its recipe string matches (the p1l '
                'recipe is decoded from injected constants, not from the weights).')
    _want_mr = ((_bind.get('per_arm_recipe_signature') or {}).get(v) or {}).get('mr')
    if _want_mr is not None:
        try:
            _want_mr = float(_want_mr)
        except (TypeError, ValueError, OverflowError):
            _want_mr = None
    if _want_mr is not None:
        _got_mr = d.get('mlp_ratio')
        try:
            _got_mr = float(_got_mr)
        except (TypeError, ValueError, OverflowError):
            _got_mr = None
        if _got_mr is None or abs(_got_mr - _want_mr) > 1e-6:
            del d
            raise SystemExit(
                f'[v12] REFUSE {v} s{seed}: checkpoint mlp_ratio {_got_mr!r} != the frozen '
                f'per_arm_recipe_signature mr={_want_mr!r} — the parameter-matched arm contract '
                'is part of the pre-registration and is verified from the weights, not from '
                'the recipe string.')
    model = L.SmallGPT(int(d.get('vocab', 8192)), *_arch, d.get('train_len', 512), [copy.deepcopy(c) for c in d['cfg']], mlp_ratio=d['mlp_ratio']).to(DEVICE)
    model.load_state_dict(d.pop('sd'))
    model.eval()
    # The shared-trunk initialisation digest is part of the matched-arm
    # contract.  Recover it from the loaded weights when the checkpoint does
    # not carry the field.
    _ck_init = d.get('init_core_fp')
    if not _ck_init:
        try:
            _ck_init = L.init_core_fingerprint(model)
        except Exception:
            _ck_init = None
    # Keep the checkpoint's provenance fields on a separate binding so the
    # record writer never depends on a variable that this function deletes.
    _ck_prov = {k: d.get(k) for k in ('train_knobs', 'params', 'train_objective',
                                      'param_breakdown', 'rng_fp', 'init_core_fp')}
    if _ck_prov.get('init_core_fp') is None:
        _ck_prov['init_core_fp'] = _ck_init
    # score EVERY loaded row: test_ids is the frozen slice already sliced to
    # the manifest's n_seqs_clean, and data_fp is computed over all of them.
    # eval_length_gen's default (n_seq=8) would measure only the
    # first 8 rows while the lock/fingerprint claim the full 16-row slice.
    r = V.eval_length_gen(model, test_ids, eval_lens, DEVICE, max_pos=None, n_seq=int(test_ids.shape[0]))
    del model
    gc.collect()

    def _merge(summary):
        # by_len also carries ppl_per_seq (per-document rows)
        # straight from eval_length_gen; pooled ppl is unchanged.
        # The P5T record carries the SAME provenance descriptor as its
        # training record (`train_knobs` / `train_objective` /
        # `param_breakdown` / `rng_fp`) plus the evaluated row count.  The
        # frozen implementation binding requires `train_knobs` on every
        # record.
        _tk = _tr.get('train_knobs') or _ck_prov.get('train_knobs')
        _ov_key = (v, L._as_seed_int(seed) if L._as_seed_int(seed) is not None else seed)
        _ov_status = _UNHEALTHY_OVERRIDE_CELLS.get(_ov_key)
        summary[f'{v}::s{seed}'] = {'variant': v, 'seed': seed, 'params': _tr.get('params') or _ck_prov.get('params'), 'by_len': r, 'split': split, 'data_fp': _dfp, 'train_fp': _tr_train_fp, '_code': V.CKPT_CODE, 'recipe': _ck_recipe, 'manifest_sha256': _msha, 'ckpt_sha256': _ensure_ck_sha(), 'ckpt_sig': (list(_ck_sig) if _ck_sig is not None else None),
                                    'n_seq': int(test_ids.shape[0]),
                                    'train_knobs': _tk,
                                    'train_objective': _tr.get('train_objective') or _ck_prov.get('train_objective'),
                                    'param_breakdown': _tr.get('param_breakdown') or _ck_prov.get('param_breakdown'),
                                    'rng_fp': _tr.get('rng_fp') or _ck_prov.get('rng_fp'),
                                    'init_core_fp': _tr.get('init_core_fp') or _ck_prov.get('init_core_fp'),
                                    'amp_dtype': (_tk or {}).get('amp_dtype') if isinstance(_tk, dict) else None,
                                    'amp_grad_scaler': (_tk or {}).get('amp_grad_scaler') if isinstance(_tk, dict) else None,
                                    'eager_batch': (_tk or {}).get('eager_batch') if isinstance(_tk, dict) else None,
                                    # Machine-readable V12_ALLOW_UNHEALTHY override
                                    # audit trail: False on every honestly healthy
                                    # cell, True + the failing status on overridden
                                    # ones (see _training_health_gate).
                                    'allow_unhealthy_override': _ov_status is not None,
                                    'health_status_override': _ov_status,
                                    # Post-access replacement audit trail (see the
                                    # selective-remeasurement guard above).
                                    'post_access_replacement': out.get('replaced_prior_ckpt_sha256') is not None,
                                    'replaced_prior_ckpt_sha256': out.get('replaced_prior_ckpt_sha256')}
        print(f'  [p5t] {v:32s} s{seed} ' + '  '.join((f'L{k}={vv.get("ppl", float("nan")):.2f}' for k, vv in sorted(r.items()))))
        return summary
    L.locked_json_update(spath, _merge)
    _ov_key_e = (v, L._as_seed_int(seed) if L._as_seed_int(seed) is not None else seed)
    L.log_experiment_event(outdir, {'kind': 'eval', 'tracking_schema': 2, 'run_id': _ft_run_id, 'status': 'ok', 'started_at': _ft_started, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'elapsed_s': time.time() - _ft_t0, 'key': f'{v}::s{seed}', 'variant': v, 'seed': int(seed), 'split': split, 'code': V.CKPT_CODE, 'manifest_sha256': _msha, 'recipe': _ck_recipe, 'ckpt_sha256': _ck_sha, 'data_fp': _dfp, 'train_fp': _tr_train_fp, 'params': _tr.get('params'), 'allow_unhealthy_override': _ov_key_e in _UNHEALTHY_OVERRIDE_CELLS, 'health_status_override': _UNHEALTHY_OVERRIDE_CELLS.get(_ov_key_e), 'metrics': {'by_len': {str(k): (v2 or {}).get('ppl') for k, v2 in r.items()}}})
    del d
    return out

def _power_enumeration(deltas, expected_sign, alpha, m_comparisons):
    """Exact sign-flip power enumeration on FIXED magnitudes (zero GPU).

    Implements the manifest's statistics.power_note obligation: treat the
    observed |d_i| as fixed, enumerate every 2^n sign pattern as a candidate
    observation, compute the exact two-sided sign-flip p for each (the null
    distribution of |sum s_i |d_i|| is the same multiset for every sign
    convention), and report (a) power under the q-model — each seed's delta
    falls in the pre-registered direction with probability q — using the
    conservative Holm threshold p <= alpha/m for the smallest-p slot, and
    (b) the minimum same-sign count among rejecting patterns.

    This is a design aid computed on the VALIDATION panel before the one-shot
    test access; it is not a guarantee and not a bound (unequal magnitudes can
    make the exact test more or less powerful than these numbers).
    """
    import bisect as _bisect
    import itertools as _it
    n = len(deltas)
    mags = [abs(float(x)) for x in deltas]
    if n == 0 or all(mm == 0.0 for mm in mags):
        return {'n': n, 'status': 'no_data'}
    if n > 16:
        return {'n': n, 'status': 'too_many_seeds_for_enumeration',
                'note': 'n>16: enumeration skipped (2^n patterns); use the symbol-test projection table instead'}
    # Null distribution of |sum s_i m_i| over all 2^n flips (same multiset for
    # any sign convention of the magnitudes).
    _sums = sorted(abs(sum(s_i * mags[i] for i, s_i in enumerate(signs)))
                   for signs in _it.product((1.0, -1.0), repeat=n))
    _total = float(len(_sums))
    _tol = 1e-12 * max(1.0, max(_sums) if _sums else 1.0)
    _thr_holm = float(alpha) / max(1, int(m_comparisons))
    _thr_marginal = float(alpha)
    exp_dir = None
    if expected_sign == 'positive':
        exp_dir = 1.0
    elif expected_sign == 'negative':
        exp_dir = -1.0
    powers = {f'q={q:.2f}': 0.0 for q in (0.8, 0.85, 0.9, 0.95)}
    _min_same = None
    _n_reject_holm = 0
    _n_reject_marginal = 0
    # A non-directional comparison has no pre-registered sign, so its power is
    # the average over the two possible true directions.  The rejecting set is
    # symmetric under flipping every sign, so computing it with the positive
    # reference and accepting either observed direction is that average
    # (flipping all signs maps the set onto itself and swaps the q-weights).
    _power_dir = 1.0 if exp_dir is None else exp_dir
    for signs in _it.product((1.0, -1.0), repeat=n):
        _signed = sum(s_i * mags[i] for i, s_i in enumerate(signs))
        stat = abs(_signed)
        # exact p: #null sums >= observed statistic (ties included, as in the
        # exact test's "at least as extreme" rule).  _sums is sorted, so use
        # binary search: the previous linear scan made this O(4**n) and would
        # hang long before the advertised n<=16 ceiling (2**16 patterns).
        _p = (len(_sums) - _bisect.bisect_left(_sums, stat - _tol)) / _total
        # A two-sided rejection is only a CONFIRMATION when the observed
        # direction also matches the pre-registered sign (decision_rule's
        # direction_ok): the all-opposite pattern rejects just as hard but in
        # the wrong direction.  Direction is the SIGN OF THE WEIGHTED SUM, not
        # a same-sign majority: with unequal |d_i| a pattern can have a
        # positive majority and a negative mean (or vice versa), and the real
        # decision rule looks at the mean.  For a non-directional comparison
        # every rejection counts.
        _same = sum(1 for s_i in signs if s_i == _power_dir)
        _dir_ok = (exp_dir is None) or (_signed * exp_dir > 0.0)
        _rej_h = _p <= _thr_holm and _dir_ok
        _rej_m = _p <= _thr_marginal and _dir_ok
        if _rej_h:
            _n_reject_holm += 1
        if _rej_m:
            _n_reject_marginal += 1
        if _rej_h and exp_dir is not None and (_min_same is None or _same < _min_same):
            _min_same = _same
        if _rej_h:
            for q in powers:
                _qv = float(q.split('=')[1])
                powers[q] += (_qv ** _same) * ((1.0 - _qv) ** (n - _same))
    return {
        'n': n,
        'status': 'ok',
        'method': ('exact enumeration of 2^n sign patterns on fixed |d_i|; exact two-sided '
                   'sign-flip p per pattern; a pattern counts as CONFIRMING only when it also '
                   'matches the pre-registered direction (direction_ok); rejection at the '
                   'conservative Holm slot p <= alpha/m; power(q) sums the q-model pattern '
                   'weights of confirming patterns'),
        'threshold_holm_conservative': _thr_holm,
        'threshold_marginal_alpha': _thr_marginal,
        'power_holm_conservative': {k: round(v, 6) for k, v in powers.items()},
        'fraction_patterns_confirming_holm': _n_reject_holm / _total,
        'fraction_patterns_confirming_marginal': _n_reject_marginal / _total,
        'min_same_sign_for_holm_confirmation': _min_same,
        'expected_sign': expected_sign,
        'note': ('per-comparison marginal power only: cross-comparison correlation from shared '
                 'arms/seeds is not modelled; the Holm threshold used is the conservative '
                 'smallest-p slot alpha/m. Design aid, not a guarantee or a bound.'),
    }


def run_power_check(cfg=None):
    """Pre-access power simulation on the frozen arms' VALIDATION deltas.

    Fulfils the frozen manifest's statistics.power_note: BEFORE the one-shot
    test access, enumerate sign-flips on the per-seed validation delta vector
    (fixed |d_i|) to obtain the real power / minimum same-sign count for THIS
    panel.  If the power is inadequate, seeds must be added and the revision
    recorded in the manifest BEFORE touching test — discovering inadequate
    power after the access is irreversible.

    Zero GPU: reads by-length VALIDATION records from the mech/P5C summary
    (the frozen arms trained by P3MT/P4MT/P3MTC/P5C all write there).  Writes
    ``<outdir>/p5c_power.json`` and returns the report."""
    cfg = dict(FINALTEST_CFG) if cfg is None else dict(cfg)
    manifest = _load_manifest(cfg.get('manifest') or FREEZE_MANIFEST)
    outdir = cfg['outdir']
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, 'p5c_power.json')
    _alpha = float((manifest.get('statistics') or {}).get('alpha', 0.05))
    _m = len(manifest.get('primary_comparisons') or [])
    summary_path = cfg.get('mech_summary') or 'results_v10_mech/summary.json'
    summary = {}
    if os.path.exists(summary_path):
        try:
            summary = json.load(open(summary_path, encoding='utf-8'))
        except Exception as _e:
            raise SystemExit(f'[v12 power] REFUSE: {summary_path} exists but cannot be parsed ({type(_e).__name__}: {_e})')
    else:
        print(f'[v12 power] NOTE: {summary_path} not found — no validation panel yet; the artifact records a data gap (run P3MT/P4MT/P5C first for a real power read)')
    comps = {}
    for comp in manifest.get('primary_comparisons') or []:
        name = comp.get('name')
        a, b = comp.get('a'), comp.get('b')
        try:
            Ln = int(str(comp.get('endpoint', 'by_len.ppl@4096')).rsplit('@', 1)[-1])
        except (TypeError, ValueError, OverflowError):
            Ln = 4096
        deltas = {}
        for s in manifest.get('seeds') or []:
            ra = summary.get(f'{a}::seed{s}') or {}
            rb = summary.get(f'{b}::seed{s}') or {}
            if ra.get('synthesized') or rb.get('synthesized'):
                continue
            pa = ((ra.get('by_len') or {}).get(Ln) or (ra.get('by_len') or {}).get(str(Ln)) or {}).get('ppl')
            pb = ((rb.get('by_len') or {}).get(Ln) or (rb.get('by_len') or {}).get(str(Ln)) or {}).get('ppl')
            if L.ppl_is_usable(pa) and L.ppl_is_usable(pb):
                deltas[int(s)] = float(pa) - float(pb)
        ordered = [deltas[s] for s in sorted(deltas)]
        entry = {'a': a, 'b': b, 'endpoint': f'by_len.ppl@{Ln}', 'split': 'validation',
                 'expected_sign': comp.get('expected_sign'),
                 'n_paired_seeds': len(ordered),
                 'seeds': sorted(deltas),
                 'deltas': ordered}
        entry.update(_power_enumeration(ordered, comp.get('expected_sign'), _alpha, _m))
        if entry.get('status') == 'no_data':
            entry['note'] = ('no paired validation by-length records for this comparison yet — '
                             'run the frozen-arm training phases (P3MT/P4MT/P3MTC/P5C) first')
        comps[name] = entry
    report = {'kind': 'p5c_power', 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
              'manifest': manifest.get('manifest_sha256'),
              'source_summary': summary_path,
              'alpha': _alpha, 'm_comparisons': _m,
              'status': 'ok', 'comparisons': comps,
              'note': ('Pre-access power simulation per the manifest statistics.power_note: '
                       'sign-flip enumeration on fixed VALIDATION |d_i| magnitudes. Inadequate '
                       'power must be remedied by adding seeds and recording the manifest '
                       'revision BEFORE the first test-split access.')}
    low = [k for k, v in comps.items()
           if v.get('power_holm_conservative') and v['power_holm_conservative'].get('q=0.80') is not None
           and float(v['power_holm_conservative']['q=0.80']) < 0.5]
    if low:
        report['low_power_warnings'] = low
    L.atomic_write_json(out_path, report, indent=2)
    print(f'[v12 power] pre-access power artifact written: {out_path}')
    for k, v in comps.items():
        _pw = (v.get('power_holm_conservative') or {}).get('q=0.80')
        print(f'  {k}: n={v.get("n_paired_seeds")} status={v.get("status")} power(q=0.80, Holm-conservative)={_pw if _pw is None else round(float(_pw), 4)} min_same_sign={v.get("min_same_sign_for_holm_confirmation")}')
    if low:
        print(f'[v12 power] WARNING: low power (<0.5 at q=0.80) for {low} — consider adding seeds and recording the manifest revision BEFORE the first test access')
    return report


def _power_artifact_problem(manifest, power):
    """Why a ``p5c_power.json`` artifact cannot certify THIS frozen panel.

    The artifact is the evidence that the pre-access power simulation was run
    for the exact manifest and seed list that is about to consume the one-shot
    test slice.  A leftover artifact from an earlier revision, an empty/failed
    enumeration, or a different comparison family satisfies nothing."""
    if not isinstance(power, dict):
        return f'the artifact is not a JSON object ({type(power).__name__})'
    _status = power.get('status')
    _skip_now = os.environ.get('V12_SKIP_POWER', '').strip().lower() in ('1', 'true', 'yes', 'on')
    if _status == 'skipped':
        if not _skip_now:
            return ('the artifact records a V12_SKIP_POWER=1 override, but the '
                    'override is not active for this run')
        return None
    if _status != 'ok':
        return f"artifact status is {_status!r}, not 'ok' (the power simulation did not complete)"
    _sha = manifest.get('manifest_sha256')
    if power.get('manifest') != _sha:
        return (f"manifest sha256 {str(power.get('manifest'))[:12]}… does not match "
                f"the frozen manifest {str(_sha)[:12]}… (stale artifact from another plan)")
    _stat = manifest.get('statistics') or {}
    try:
        _alpha = float(_stat.get('alpha', 0.05))
        _m = len(manifest.get('primary_comparisons') or [])
        _p_alpha = float(power.get('alpha'))
        _p_m = int(power.get('m_comparisons'))
    except (TypeError, ValueError, OverflowError):
        return 'artifact alpha/m_comparisons is malformed'
    if not (math.isfinite(_p_alpha) and abs(_p_alpha - _alpha) <= 1e-12):
        return f'artifact alpha {_p_alpha!r} differs from the frozen alpha {_alpha!r}'
    if _p_m != _m:
        return f'artifact family size {_p_m} differs from the frozen {_m}'
    _comps = power.get('comparisons')
    if not isinstance(_comps, dict) or not _comps:
        return 'artifact has no comparison entries'
    _names = [str(c.get('name')) for c in (manifest.get('primary_comparisons') or [])]
    if set(_comps) != set(_names):
        return (f'artifact comparisons {sorted(_comps)} do not match the frozen '
                f'family {sorted(_names)}')
    _seeds = sorted({int(s) for s in (manifest.get('seeds') or [])})
    for _name in _names:
        _entry = _comps.get(_name)
        if not isinstance(_entry, dict):
            return f'{_name}: entry is not an object'
        if _entry.get('status') != 'ok':
            return f"{_name}: power status is {_entry.get('status')!r}, not 'ok'"
        try:
            _eseeds = sorted({int(s) for s in (_entry.get('seeds') or [])})
        except (TypeError, ValueError, OverflowError):
            return f'{_name}: artifact seed list is malformed'
        if _eseeds != _seeds:
            return (f'{_name}: artifact seeds {_eseeds} do not match the frozen '
                    f'seed list {_seeds}')
        try:
            _n = int(_entry.get('n_paired_seeds'))
        except (TypeError, ValueError, OverflowError):
            return f'{_name}: n_paired_seeds is malformed'
        if _n != len(_seeds):
            return (f'{_name}: artifact has {_n} paired seed(s), the frozen panel '
                    f'has {len(_seeds)}')
    return None


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
    # `phase P5T` performs this zero-GPU preflight; the direct
    # `finaltest` CLI mode must run it before any test access so a plan that
    # plan_check would reject cannot consume the one-shot test.  Smoke runs use injected
    # data and are not the frozen confirmation, so they skip it.
    _smoke = L._as_bool_flag(cfg.get('smoke')) is True
    if not _PLAN_DONE and not _smoke:
        plan_check(cfg.get('manifest') or FREEZE_MANIFEST)
    manifest = _load_manifest(cfg['manifest'])
    outdir = cfg['outdir']
    os.makedirs(outdir, exist_ok=True)
    need = max([int(x) for x in ((manifest.get('eval') or {}).get('by_length_ppl') or cfg['eval_lens'])]) + 1
    _nseq_clean = int((manifest.get('eval') or {}).get('n_seqs_clean') or cfg.get('n_seq', 16))
    test_ids = cfg.get('test_ids')
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
    if not _smoke:
        # Pre-access POWER gate (the manifest statistics.power_note obligation):
        # the sign-flip enumeration on the frozen arms' validation deltas must
        # have been run (python v12_supp.py power, zero GPU) BEFORE the one-shot
        # read, so inadequate power is discovered while seeds can still be added
        # with a recorded revision — not after the access.  V12_SKIP_POWER=1 is
        # the explicit, artifact-recorded override.
        #
        # This gate runs BEFORE the pre-access lock is written: a failed power
        # check legitimately requires a manifest revision and added seeds, and
        # writing the lock first would falsely record a test access (and block
        # the P5C retraining that revision needs).
        _power_path = os.path.join(outdir, 'p5c_power.json')
        if not os.path.exists(_power_path):
            if os.environ.get('V12_SKIP_POWER', '').strip().lower() in ('1', 'true', 'yes', 'on'):
                print('[v12] WARNING: V12_SKIP_POWER=1 — consuming the one-shot test access WITHOUT the pre-access power artifact; the override is recorded in p5c_power.json and must be justified in the decision log')
                L.atomic_write_json(_power_path, {'kind': 'p5c_power', 'status': 'skipped',
                                                  'v12_skip_power_used': True,
                                                  'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                                                  'manifest': manifest.get('manifest_sha256'),
                                                  'note': 'V12_SKIP_POWER=1: the pre-access power simulation was consciously skipped before the one-shot test access'})
            else:
                raise SystemExit(
                    f'[v12] REFUSE: no pre-access power artifact at {_power_path} — the frozen '
                    'manifest (statistics.power_note) requires the sign-flip enumeration on the '
                    'validation deltas to be run BEFORE the first test-split access. Run '
                    '`python v12_supp.py power` (zero GPU) first; if the power is inadequate, '
                    'add seeds and record the manifest revision before touching test. '
                    'V12_SKIP_POWER=1 is the explicit override (recorded in the artifact).')
        else:
            try:
                _pow = json.load(open(_power_path, encoding='utf-8'))
            except Exception as _pe:
                raise SystemExit(
                    f'[v12] REFUSE: the pre-access power artifact {_power_path} exists but '
                    f'cannot be read ({type(_pe).__name__}: {_pe}) — a confirmation run cannot '
                    'proceed on an unreadable power attestation (fix it or set the explicit '
                    'V12_SKIP_POWER=1 override)')
            _pow_why = _power_artifact_problem(manifest, _pow)
            if _pow_why is not None:
                raise SystemExit(
                    f'[v12] REFUSE: the pre-access power artifact {_power_path} does not '
                    f'certify this frozen panel: {_pow_why}. Re-run `python v12_supp.py power` '
                    'for the current manifest/seed list before touching the one-shot test '
                    'split (V12_SKIP_POWER=1 is the explicit, recorded override).')
            if _pow.get('status') == 'skipped':
                print('[v12] WARNING: the pre-access power artifact was SKIPPED (V12_SKIP_POWER=1) — this test access proceeds without the power simulation the manifest asks for')
            else:
                for _k, _v in (_pow.get('comparisons') or {}).items():
                    _pw = (_v.get('power_holm_conservative') or {}).get('q=0.80')
                    print(f'  [p5t power] {_k}: n={_v.get("n_paired_seeds")} power(q=0.80)={_pw if _pw is None else round(float(_pw), 4)} min_same_sign={_v.get("min_same_sign_for_holm_confirmation")}')
                if _pow.get('low_power_warnings'):
                    print(f'[v12] WARNING: the pre-access power artifact flags LOW POWER for {_pow["low_power_warnings"]} — the confirmation may be resolution-limited; this was known BEFORE the access')
    # Write the pre-access attestation only AFTER every zero-GPU gate passed,
    # and still BEFORE reading the held-out split, so a manifest mismatch is
    # discovered before the one-shot access is used.
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
        # Run the leakage gate BEFORE the `accessed` lock is written
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
        if _ov.get('status') not in (None, 'ok'):
            raise SystemExit(f'[v12] REFUSE: the data-leakage audit returned status {_ov.get("status")!r} ({_ov.get("note") or _ov.get("warnings")}) — a confirmation run unlocks only on a verifiably clean isolation audit, not on warnings')
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
    _bylen_cell_status = {}
    for v in manifest['arms']:
        for seed in manifest['seeds']:
            _cell_out = _finaltest_bylen_cell(cfg, manifest, v, seed, test_ids,
                                              split=_split_label, train_fp=_want_train_fp)
            _bylen_cell_status[(v, L._as_seed_int(seed) if L._as_seed_int(seed) is not None else seed)] = ((_cell_out or {}).get('status') if isinstance(_cell_out, dict) else None)
    # 2) frozen uniform-corruption probe on the SAME test rows (secondary,
    #    descriptive-only per the manifest)
    _probe_eval = (manifest.get('eval') or {}).get('distractor_probe') or {}
    _probe_mode = str(_probe_eval.get('mode') or 'uniform')
    if _probe_mode not in V10.PROBE_MODES:
        raise SystemExit(f"[v12] REFUSE: eval.distractor_probe.mode={_probe_mode!r} is not one of {sorted(V10.PROBE_MODES)}")
    _probe_rhos = [float(x) for x in (_probe_eval.get('rhos') or V10.PROBE_MODES[_probe_mode])]
    if _probe_mode in ('mask', 'delete', 'doc') and any(abs(x - 1.0) > 1e-9 for x in _probe_rhos):
        raise SystemExit(f"[v12] REFUSE: eval.distractor_probe.mode={_probe_mode!r} is all-or-nothing and only admits rho=1.0 (got {_probe_rhos})")
    # The probe is measured on the SAME frozen rows as the by-length panel, so
    # an omitted probe n_seq must default to that frozen row count — not to a
    # runtime config value that could exceed it.
    pcfg = dict(V10.PROBE, outdir=outdir, ckpt_dir=cfg['ckpt_dir'], summary=cfg['probe_summary'], variants=list(manifest['arms']), seeds=[int(s) for s in manifest['seeds']], eval_lens=[int(x) for x in (_probe_eval.get('eval_lens') or [2048, 4096])], rhos=_probe_rhos, modes={_probe_mode: _probe_rhos}, niah=None, arm_overrides={v: ['learned'] for v in V10.SCRATCH_CONTROLS}, target=int(_probe_eval.get('target', cfg.get('target', 512))), n_seq=int(_probe_eval.get('n_seq', _nseq_clean)), chunk=int(cfg.get('chunk', 4)), manifest_sha256=manifest.get('manifest_sha256'))
    mech_summ = {}
    if os.path.exists(cfg['mech_summary']):
        mech_summ = json.load(open(cfg['mech_summary'], encoding='utf-8'))
    for v in pcfg['variants']:
        for seed in pcfg['seeds']:
            _cell_status = _bylen_cell_status.get((v, int(seed)))
            if _cell_status == 'skipped_unhealthy_training':
                # The one-shot access is not spent on weights whose own curve
                # failed the health gate: the by-length cell was refused, so a
                # probe measurement on the same weights would describe a
                # checkpoint that is not part of the confirmatory panel.
                print(f'[p5t] SKIP probe {v} s{seed}: its by-length cell was refused by the training-health gate — the probe budget is not spent on excluded weights')
                continue
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
    # The frozen row count is an exact positive integer; fractional, boolean
    # or non-positive values are refused.
    _nseq_i = _record_int(n_seq)
    if _nseq_i is None or _nseq_i <= 0:
        return f'n_seq={n_seq!r} is not a positive exact integer row count'
    # The frozen grid names each evaluated length once.  ``512`` and ``"512"``
    # are two Python keys for ONE length; a record carrying both is ambiguous.
    _seen_lens = {}
    for _ln, _c in bl.items():
        if not isinstance(_c, dict):
            return f'by_len[{_ln!r}] is not a cell object'
        if isinstance(_ln, (bool, np.bool_)):
            return f'by_len key {_ln!r} is not an integer length'
        # Exact integer length only.
        _ln_i = _record_int(_ln)
        if _ln_i is None:
            return f'by_len key {_ln!r} is not an integer length'
        if _ln_i < 2:
            return f'by_len length {_ln_i} has no next-token prediction (needs Ln>=2); an empty cell is not admissible'
        if _ln_i in _seen_lens:
            return (f'by_len declares length {_ln_i} more than once '
                    f'(keys {_seen_lens[_ln_i]!r} and {_ln!r}) — the frozen '
                    'endpoint would be ambiguous')
        _seen_lens[_ln_i] = _ln
        _ntok = _c.get('n_tok')
        _want = _nseq_i * (_ln_i - 1)
        if (isinstance(_ntok, (bool, np.bool_))
                or not isinstance(_ntok, (int, float, np.integer, np.floating))):
            return f'by_len[{_ln_i}].n_tok={_ntok!r} != the frozen {_want} (= {_nseq_i} rows x {_ln_i - 1} predictions) — truncated/empty cells are not admissible'
        # Exact integer contract: _record_int keeps integral floats but
        # rejects lossy/fractional values.
        _ntok_i = _record_int(_ntok)
        if _ntok_i is None or _ntok_i != _want:
            return f'by_len[{_ln_i}].n_tok={_ntok!r} != the frozen {_want} (= {_nseq_i} rows x {_ln_i - 1} predictions) — truncated/empty cells are not admissible'
        # The cell's own row count is part of the estimand: the by-length
        # summary quotes the first cell's n_seq, so each cell must state the
        # frozen row count.
        _cnseq_raw = _c.get('n_seq')
        if _cnseq_raw is not None:
            _cnseq = _record_int(_cnseq_raw)
            if _cnseq is None or _cnseq != _nseq_i:
                return (f'by_len[{_ln_i}].n_seq={_cnseq_raw!r} != the frozen '
                        f'{_nseq_i} — a different row count is a different estimand')
        # A cell that does not explicitly certify full length is unverifiable.
        # Matching n_tok alone cannot distinguish a full cell from a record
        # whose truncation marker was lost when it was serialised.
        if 'truncated' not in _c:
            return f'by_len[{_ln_i}] has no `truncated` flag — an unverifiable cell is not admissible'
        if not isinstance(_c.get('truncated'), (bool, np.bool_)):
            return f'by_len[{_ln_i}].truncated={_c.get("truncated")!r} is not a boolean — an unverifiable cell is not admissible'
        if bool(_c.get('truncated')):
            return f'by_len[{_ln_i}] is marked truncated — a partial cell is not admissible'
        # A non-truncated cell must cover exactly the requested span:
        # eval_span is the cell's own record of how many positions it
        # actually ran (the same rule as audit_length).
        _es = _c.get('eval_span')
        # eval_span is REQUIRED, not optional: without it a record cannot prove
        # it scored the full length, and an optional check lets a hand-crafted
        # record omit the field to dodge the span gate.  Honest writers always
        # record it (run_lenphase / eval_length_gen).
        if _es is None:
            return (f'by_len[{_ln_i}] carries no eval_span — the cell does not '
                    'prove it scored the requested length; only records from '
                    'the current writer (eval_length_gen) are admissible')
        _es_i = _record_int(_es)
        if _es_i is None or _es_i != _ln_i:
            return (f'by_len[{_ln_i}].eval_span={_es!r} != the requested '
                    f'length — a shorter span under a full-length label is '
                    'not admissible')
    return None


def _record_int(value):
    """Exact integer view of a recorded count, or None for anything else.

    Use the shared seed/count contract so a value admitted here is exactly the
    value the plan/pairing gates admit.  In particular a decimal string such
    as ``"4.0"`` or ``"1e3"`` is not an exact integer spelling: accepting it
    would map a recorded seed/count to a different integer than the frozen
    whitelist, while a genuine integer string ``"4"`` stays accepted.
    """
    return L._as_seed_int(value)


def _manifest_seed_ids(manifest):
    """Exact, unique frozen seed ids (also for an in-memory manifest)."""
    if not isinstance(manifest, dict):
        raise SystemExit(f'[p5t stats] REFUSE: frozen manifest is {type(manifest).__name__}, not an object')
    _seeds_raw = manifest.get('seeds')
    if not isinstance(_seeds_raw, (list, tuple)) or not _seeds_raw:
        # A string/bytes container would iterate characters or byte values and
        # silently build a whitelist of different seed identities.
        raise SystemExit(
            f'[p5t stats] REFUSE: frozen seeds must be a non-empty list '
            f'({manifest.get("seeds")!r})')
    _ids = []
    for _s in _seeds_raw:
        _si = L._as_seed_int(_s)
        if _si is None:
            raise SystemExit(f'[p5t stats] REFUSE: frozen seed {_s!r} is not an exact integer')
        _ids.append(_si)
    if len(set(_ids)) != len(_ids):
        raise SystemExit(f'[p5t stats] REFUSE: frozen seeds are not unique ({manifest.get("seeds")!r})')
    return _ids


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
    _arms_raw = manifest.get('arms')
    if (not isinstance(_arms_raw, (list, tuple)) or not _arms_raw
            or not all(isinstance(a, str) and a for a in _arms_raw)):
        raise SystemExit(
            f'[p5t stats] REFUSE: frozen arms must be a non-empty list of '
            f'arm names ({manifest.get("arms")!r})')
    arms = list(_arms_raw)
    # This gate is also callable with an in-memory manifest (finaltest_stats /
    # tests).  Apply the loader's exact seed identity here too: int(True)==1
    # or int(4.9)==4 would admit a record under a different planned seed.
    seed_whitelist = set(_manifest_seed_ids(manifest))
    exp_code = L.code_stamp()  # eager-only; kernel mode is refused at load
    impl = _frozen_implementation(manifest)
    msha = manifest.get('manifest_sha256')
    want_train_fp = _frozen_train_fp(manifest)
    _eval_obj = manifest.get('eval')
    if _eval_obj is not None and not isinstance(_eval_obj, dict):
        raise SystemExit(
            f'[p5t stats] REFUSE: manifest.eval is {type(_eval_obj).__name__}, '
            'not an object — the frozen row count cannot be read')
    _nseq_raw = (_eval_obj or {}).get('n_seqs_clean')
    _want_nseq = _record_int(_nseq_raw) if _nseq_raw is not None else 0
    if _nseq_raw is not None and (_want_nseq is None or _want_nseq <= 0):
        raise SystemExit(f'[p5t stats] REFUSE: eval.n_seqs_clean={_nseq_raw!r} is not a positive integer row count')
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
        if r.get('synthesized'):
            # A reconstructed record has no weights and is not an independent
            # measurement; it must never enter the one-shot confirmation.
            why = 'synthesized reconstruction — not an independent measured record'
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
            elif not any(isinstance(_c, dict) and L.ppl_is_usable(_c.get('ppl'))
                         for _c in r['by_len'].values()):
                # A record whose cells are all shape-complete but carry no
                # usable PPL is not a measurement at all.  (A single failed
                # endpoint is handled per pair later; this guard only refuses
                # a record that cannot contribute ANY endpoint.)
                why = 'by_len contains no usable PPL measurement'
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
        raise SystemExit('[p5t stats] REFUSE: ambiguous duplicate records for the same (variant, seed): ' + '; '.join(duplicates) + ' — the paired statistics would depend on which one wins; resolve the bylen file first')
    if len(fps) > 1:
        raise SystemExit(f'[p5t stats] REFUSE: admissible records span {len(fps)} different test-slice fingerprints ({sorted(fps)}) — the whole panel must be measured on ONE frozen test slice, not merely pairwise-equal fingerprints')
    if len(train_fps) > 1:
        raise SystemExit(f'[p5t stats] REFUSE: admissible records were trained on {len(train_fps)} different corpora ({sorted(train_fps)}) — all frozen arms must share the canonical training split')
    if not _want_nseq and admitted:
        # A descriptive manifest may omit the eval block, but if the
        # records DO carry a row count it is still part of the estimand.  Do
        # not let a missing manifest field disable the check and pair two
        # different n_seq panels.
        _nseq_by = {k: _record_int(r.get('n_seq')) for k, r in admitted.items()}
        _present = {v for v in _nseq_by.values() if v is not None}
        if len(_present) > 1:
            raise SystemExit(
                f'[p5t stats] REFUSE: admissible records disagree on n_seq '
                f'({sorted(_present)}) and the manifest has no frozen eval.n_seqs_clean '
                '— different row counts are different estimands')
        if _present:
            _missing = [k for k, v in _nseq_by.items() if v is None]
            if _missing:
                raise SystemExit(
                    '[p5t stats] REFUSE: some admissible records carry n_seq and '
                    f'others do not ({_missing[:5]}) — the row count cannot be verified')
            _nseq = next(iter(_present))
            for _k, _r in admitted.items():
                _why = _by_len_shape_reason(_r, _nseq)
                if _why is not None:
                    raise SystemExit(
                        f'[p5t stats] REFUSE: {_k}: {_why} (no frozen row count '
                        'to fall back on)')
    return admitted, (sorted(fps)[0] if fps else None)


def _by_len_cell(rec, Ln):
    """The one cell for a frozen length, or None when it is ambiguous.

    JSON records carry string keys while an in-memory record may carry the
    integer spelling too.  If both spellings are present the endpoint is
    ambiguous; report no cell so every consumer treats the record as
    unverifiable at that length.
    """
    if not isinstance(rec, dict):
        return None
    bl = rec.get('by_len')
    if not isinstance(bl, dict):
        return None
    hits = [k for k in (Ln, str(Ln)) if k in bl]
    if len(hits) != 1:
        return None
    cell = bl[hits[0]]
    return cell if isinstance(cell, dict) else None


def _doc_level_analysis(recs_a, recs_b, common, Ln):
    """Document-sampling readout for one comparison.

    逐文档配对差（对每个文档取「种子内臂差」的均值）与两层方差分解：
    between-doc（文档抽样不确定性）与 within-doc-across-seed（训练随机性）。
    16 条评测行不足以充分覆盖文档分布——本读数的目的就是把这个不确定性
    显式化，而不是宣称已消除。"""
    rows_a = [_by_len_cell(recs_a[s], Ln) or {} for s in common]
    rows_b = [_by_len_cell(recs_b[s], Ln) or {} for s in common]
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
            # Booleans/strings are flags/metadata, not per-document PPL
            # measurements; float(True) would silently become PPL 1.0.
            fa, fb = L._finite_scalar(a), L._finite_scalar(b)
            if fa is None or fb is None or fa <= 0.0 or fb <= 0.0:
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
    # Canonicalise the frozen slice identity with the same rule the rest of
    # the probe gate uses.
    _slice_text = L._content_fp_text(slice_fp)
    if _slice_text is None:
        out['error'] = f'slice_fp {slice_fp!r} is not a non-empty text fingerprint'
        return out
    slice_fp = _slice_text
    out['slice_fp'] = _slice_text
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
    seeds = set(_manifest_seed_ids(manifest))
    exp_code = L.code_stamp()
    impl = _frozen_implementation(manifest)
    want_train_fp = _frozen_train_fp(manifest)
    _want_probe_mode = str((((manifest.get('eval') or {}).get('distractor_probe')
                            or {}).get('mode') or 'uniform'))
    _require_train_fp = _train_fp_required(manifest) or any(
        # A malformed truthy/falsey value still declares an intended binding;
        # presence, not truthiness, decides whether a descriptor is required.
        isinstance(_r, dict) and _r.get('train_fp') is not None
        for _r in raw.values())
    groups = {}
    excluded = []
    _probe_train_fps = set()
    _ambiguous_idents = set()
    for key, r in raw.items():
        if not isinstance(r, dict):
            excluded.append(f'{key}: not a dict')
            continue
        why = None
        _seed_i = _record_int(r.get('seed'))
        _tfp_text = L._content_fp_text(r.get('train_fp'))
        _geom_why = V10._probe_record_problem(r)
        if r.get('synthesized'):
            why = 'synthesized reconstruction — not an independent probe measurement'
        elif _geom_why is not None:
            # Text names/geometry are checked before any set/dict key is built:
            # an unhashable variant/arm becomes a named exclusion.
            why = _geom_why
        elif r.get('variant') not in arms:
            why = f"variant {r.get('variant')!r} is not one of the frozen arms"
        elif str(r.get('mode', 'uniform')) != _want_probe_mode:
            why = (f"probe mode {r.get('mode', 'uniform')!r} is not the frozen "
                   f"{_want_probe_mode!r} corruption mode")
        elif _seed_i is None or _seed_i not in seeds:
            why = f"seed {r.get('seed')!r} is outside the frozen whitelist"
        elif L._content_fp_text(r.get('val_fp')) != slice_fp:
            why = f"val_fp {str(r.get('val_fp'))[:12]}… does not match the frozen test slice {str(slice_fp)[:12]}…"
        elif exp_code is not None and r.get('_code') != exp_code:
            why = f"_code {r.get('_code')!r} does not match the frozen code semantics {exp_code!r}"
        elif _require_train_fp and _tfp_text is None:
            why = (f'missing/invalid training-corpus fingerprint (train_fp={r.get("train_fp")!r}) '
                   'for the probe record')
        elif r.get('train_fp') is not None and _tfp_text is None:
            why = (f'train_fp {r.get("train_fp")!r} is not a text fingerprint — '
                   'a boolean/empty/non-text value is not a corpus identity')
        elif want_train_fp is not None and _tfp_text != want_train_fp:
            why = f"train_fp {_tfp_text} does not match the frozen canonical training fingerprint {want_train_fp}"
        elif _recipe_signature(manifest, r.get('variant')) is not None and not _recipe_matches_signature(r.get('recipe'), _recipe_signature(manifest, r.get('variant'))):
            why = 'recipe does not match the frozen training_recipe signature (including the per-arm mr)'
        elif (manifest.get('manifest_sha256') is not None
              and r.get('manifest_sha256') != manifest.get('manifest_sha256')):
            # A missing hash is not a match: legacy/hand-edited probe rows
            # must not enter evidence for this freeze revision.
            why = (f"manifest_sha256 {str(r.get('manifest_sha256'))[:12]}… does not match the "
                   f"frozen manifest {str(manifest.get('manifest_sha256'))[:12]}… — the probe was "
                   'measured under a different plan (or carries no plan hash)')
        elif _implementation_reason(r, impl) is not None:
            why = _implementation_reason(r, impl)
        elif not L.ppl_is_usable(r.get('ppl_mean')):
            why = 'ppl_mean is not a usable measurement'
        if why:
            excluded.append(f'{key}: {why}')
            continue
        mode = r.get('mode', 'uniform')
        ident = (r['variant'], r['arm'], _record_int(r['eval_len']), float(r['rho']), mode,
                 json.dumps(V10._fp_norm(r.get('probe_params')), sort_keys=True),
                 str(r.get('recipe')), str(r.get('_code')))
        grp = groups.setdefault(ident, {})
        if _seed_i in grp:
            excluded.append(f'{key}: duplicate (cell, seed) identity — the pooled mean would be ambiguous')
            # Do not keep the first-seen value: which one is "first" depends on
            # JSON object order, so the cell has no unambiguous measurement.
            _ambiguous_idents.add(ident)
            continue
        grp[_seed_i] = float(r['ppl_mean'])
        if _tfp_text is not None:
            _probe_train_fps.add(_tfp_text)
    for _ident in _ambiguous_idents:
        groups.pop(_ident, None)
    if len(_probe_train_fps) > 1:
        # The probe is descriptive, but pooling/contrasting records trained on
        # different corpora would still misattribute the effect to this slice.
        out['error'] = (f'probe records span {len(_probe_train_fps)} different training corpora '
                        f'({sorted(_probe_train_fps)}) — refusing to pool or contrast them')
        out['n_excluded'] = len(excluded)
        out['n_admitted'] = 0
        out['train_fps'] = sorted(_probe_train_fps)
        return out
    out['n_excluded'] = len(excluded)
    out['n_admitted'] = sum(len(g) for g in groups.values())
    if _probe_train_fps:
        out['train_fps'] = sorted(_probe_train_fps)
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
    # The full cell identity (including probe_params, recipe and code stamp)
    # must match on both sides: same length/rho/mode is not enough when the
    # two arms were probed under different sequence counts or evaluation
    # geometries.  A visible cell with several parameterisations is ambiguous
    # and contributes no contrast.
    for ident, learned in sorted(groups.items()):
        v, arm, Ln, rho, mode = ident[:5]
        if arm != 'learned' or len(by_visible.get((v, arm, Ln, rho, mode), [])) > 1:
            continue
        for oident, other in sorted(groups.items()):
            _v, other_arm, _Ln, _rho, _mode = oident[:5]
            if (_v != v or _Ln != Ln or _rho != rho or _mode != mode
                    or other_arm == 'learned'
                    or len(by_visible.get((_v, other_arm, _Ln, _rho, _mode), [])) > 1):
                continue
            if oident[5:] != ident[5:]:
                # Different probe_params / recipe / code stamp.
                continue
            common = sorted(set(learned) & set(other))
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


def _frozen_rule_confirms(holm_entry, comp, alpha, n_seeds_fallback=8):
    """The manifest-v3 decision rule for ONE frozen comparison.

    CONFIRMED iff p_holm < alpha (strict — the manifest text says
    `p_holm < 0.05`, while the generic holm_adjust `reject` uses `<=`) AND
    direction_ok is True AND `param_matched is not false` (True or the
    unverifiable None; only a measured >1% mismatch blocks) AND the effective
    pair count reaches the declared min_seeds_for_confirmation floor.
    `coverage_ok` and the init-core digests are reported instrumentation,
    never decision inputs.  Both `finaltest_stats` and the report's legacy
    fallback call this ONE implementation so they cannot drift apart again.
    """
    _ph = (holm_entry or {}).get('p_holm') if isinstance(holm_entry, dict) else None
    # A boolean is a flag, not a probability (float(False) == 0.0 would be the
    # strongest possible p); alpha is likewise a real level, not a flag.
    if isinstance(_ph, (bool, np.bool_)) or isinstance(alpha, (bool, np.bool_)):
        return False
    try:
        _a = float(alpha)
        _p = None if _ph is None else float(_ph)
    except (TypeError, ValueError, OverflowError):
        return False
    if not (math.isfinite(_a) and 0.0 < _a < 1.0):
        return False
    if _p is None or not math.isfinite(_p) or _p >= _a:
        return False
    comp = comp if isinstance(comp, dict) else {}
    _dir = comp.get('direction_ok')
    if not isinstance(_dir, (bool, np.bool_)) or not bool(_dir):
        return False
    _pm = comp.get('param_matched')
    # `is not False` in the manifest means True or unverifiable None; a
    # measured False blocks.  Accept NumPy booleans (the other statistics
    # helpers do, and `np.bool_ is False` is False), but refuse any other type
    # instead of letting a truthy string through.
    if _pm is not None:
        if not isinstance(_pm, (bool, np.bool_)) or not bool(_pm):
            return False
    # An explicitly measured pair count is used verbatim; only a MISSING field
    # falls back.  `or` would turn a measured 0 (or False) into a different
    # count, and int(12.9) would round a fractional count UP to a valid seed
    # floor.  Exact integer contract, matching the rest of the statistics.
    _n_raw = comp.get('n_effective') if comp.get('n_effective') is not None else comp.get('n')
    _floor_raw = (comp.get('min_seeds_for_confirmation')
                  if comp.get('min_seeds_for_confirmation') is not None
                  else n_seeds_fallback)
    _n = L._as_seed_int(_n_raw)
    _floor = L._as_seed_int(_floor_raw)
    if _n is None or _floor is None or _floor < 1 or _n < 1:
        return False
    return _n >= _floor


def finaltest_stats(manifest=None, bylen_path=None, probe_path=None, out='analysis_v12/test_stats.json'):
    """Zero-GPU: paired confirmation statistics under the frozen plan.

    统计入口现在是**严格准入制**——种子白名单、唯一
    记录、非空数据指纹、单一冻结 test 切片、_code/recipe 与冻结计划的机器
    校验、manifest 哈希绑定，缺一项即不入统计（点名剔除）或拒绝运行
    （歧义/混切片）。by_length 汇总同样只由可采纳记录构成。"""
    if manifest is None:
        manifest = _load_manifest(FREEZE_MANIFEST)
    # finaltest_stats is also callable with an in-memory manifest; the nested
    # plan objects are validated here.
    for _key in ('statistics', 'eval', 'binding'):
        _obj = manifest.get(_key)
        if _obj is not None and not isinstance(_obj, dict):
            raise SystemExit(
                f'[p5t stats] REFUSE: manifest.{_key} is '
                f'{type(_obj).__name__}, not an object')
    _comps = manifest.get('primary_comparisons')
    if _comps is not None:
        if not isinstance(_comps, (list, tuple)):
            raise SystemExit(
                f'[p5t stats] REFUSE: manifest.primary_comparisons is '
                f'{type(_comps).__name__}, not a list')
        if not all(isinstance(c, dict) for c in _comps):
            raise SystemExit(
                '[p5t stats] REFUSE: every manifest.primary_comparisons entry '
                'must be an object')
        _comp_names = []
        for _c in _comps:
            _name = _c.get('name')
            if not isinstance(_name, str) or not _name:
                raise SystemExit(
                    '[p5t stats] REFUSE: every manifest.primary_comparisons '
                    'entry must carry a non-empty string name')
            _comp_names.append(_name)
        if len(set(_comp_names)) != len(_comp_names):
            raise SystemExit(
                '[p5t stats] REFUSE: duplicate primary-comparison names would '
                'collapse the Holm family — every declared contrast needs a '
                'unique name')
    bylen_path = bylen_path or os.path.join(TEST_OUTDIR, 'bylen.json')
    probe_path = probe_path or os.path.join(TEST_OUTDIR, 'distractor.json')
    raw = {}
    if os.path.exists(bylen_path):
        raw = json.load(open(bylen_path, encoding='utf-8'))
        if not isinstance(raw, dict):
            raise SystemExit(f'[v12] REFUSE: {bylen_path} is valid JSON but not an object ({type(raw).__name__}) — the confirmation records cannot be admitted')
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
    _alpha_raw = (manifest.get('statistics') or {}).get('alpha', 0.05)
    if isinstance(_alpha_raw, (bool, np.bool_)):
        raise SystemExit(f'[p5t stats] REFUSE: statistics.alpha={_alpha_raw!r} is a boolean, not a significance level')
    try:
        _alpha = float(_alpha_raw)
    except (TypeError, ValueError, OverflowError):
        raise SystemExit(f'[p5t stats] REFUSE: statistics.alpha={_alpha_raw!r} is not numeric') from None
    if not (math.isfinite(_alpha) and 0.0 < _alpha < 1.0):
        raise SystemExit(f'[p5t stats] REFUSE: statistics.alpha={_alpha!r} must be in (0,1)')
    stats = {'manifest': {'frozen_at': manifest.get('frozen_at'), 'manifest_sha256': manifest.get('manifest_sha256'), 'arms': manifest['arms'], 'seeds': manifest['seeds'], 'code_semantics_expected': manifest.get('code_semantics_expected'), 'statistics': manifest.get('statistics'), 'scope': manifest.get('scope'), 'revision': manifest.get('revision')}, 'admission': {'n_admitted': len(admitted), 'test_slice_fp': unified_fp, 'n_effective_seeds': len({int(s) for (_v, s) in admitted}), 'n_declared_seeds': len(manifest['seeds'])}, 'comparisons': {}, 'holm': {}, 'by_length': {}, 'probe': _test_probe_stats(probe_path, manifest, unified_fp)}
    # The exact sign-flip p is valid only under within-seed exchangeability.
    # admission binds the corpus/slice/row count/recipe/code, but the RNG stream
    # and batch order are not independently re-derivable from a record; report
    # them instead of silently assuming them.
    _rngs = sorted({_t for _t in (L._content_fp_text(r.get('rng_fp'))
                                  for r in admitted.values()) if _t is not None})
    _inits = sorted({_t for _t in (L._content_fp_text(r.get('init_core_fp'))
                                  for r in admitted.values()) if _t is not None})
    stats['validity'] = {
        'rng_fp_values': _rngs,
        'same_rng_fp': len(_rngs) <= 1,
        'init_core_fp_values': _inits,
        'same_init_core_fp': len(_inits) <= 1,
        'init_core_fp_recorded': bool(_inits),
        'n_seq_values': sorted({n for n in (_record_int(r.get('n_seq')) for r in admitted.values()) if n is not None}),
        'note': 'exact sign-flip requires within-seed exchangeability (same data order / eval rows / corruption grid). data_fp, train_fp, n_seq and recipe are machine-checked; the shared-trunk initialization digest (init_core_fp) is reported and compared (`same_init_core_fp`); rng_fp is reported but NOT enforced (architecture-induced RNG consumption is part of the arm), so the exchangeability claim remains a declared, machine-instrumented assumption.',
    }
    # The frozen comparison is parameter-matched, not compute-matched.
    # Refuse a non-descriptive plan that does not disclose
    # both facts, so the caveat cannot be silently dropped from the report.
    _cbd = manifest.get('compute_budget_disclosure')
    _descriptive = bool((manifest.get('statistics') or {}).get('descriptive_only'))
    if not _descriptive:
        _missing = [k for k in ('parameter_matching', 'flops', 'read_fraction')
                    if not (isinstance(_cbd, dict) and _cbd.get(k))]
        if _missing:
            raise SystemExit(f'[p5t stats] REFUSE: the frozen plan lacks compute-allocation disclosure {_missing}. Parameter matching is not compute matching; the report must state the FLOPs/read-allocation mismatch next to every primary comparison .')
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
                c = _by_len_cell(r, Ln)
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
                if _pool_ok and _n_tok > 0:
                    try:
                        _pooled_ce = _nll_sum / _n_tok
                    except OverflowError:
                        _pooled_ce = 0.0
                    _pooled = L._safe_exp_ppl(_pooled_ce)
                else:
                    _pooled = None
                stats['by_length'].setdefault(v, {})[str(Ln)] = {'mean': float(_pooled if _pooled is not None else np.mean(_dl)),
                                                                  'mean_arithmetic': float(np.mean(_dl)),
                                                                  'ppl_pooled': _pooled, 'n_tokens': _n_tok,
                                                                  'n': len(_dl), 'ppl_by_seed': {str(s): p for s, p in _by_seed.items()}, 'params': next((L._as_seed_int(r.get('params')) for r in rows
                                                                     if L._as_seed_int(r.get('params')) is not None
                                                                     and L._as_seed_int(r.get('params')) > 0), None), 'n_seqs': next((_record_int(c.get('n_seq')) for _r in rows for c in [_by_len_cell(_r, Ln)] if isinstance(c, dict) and _record_int(c.get('n_seq')) is not None and _record_int(c.get('n_seq')) > 0), None)}
    for comp in manifest['primary_comparisons']:
        name = comp['name']
        a, b = (comp['a'], comp['b'])
        ep = comp.get('endpoint', 'by_len.ppl@4096')
        # An in-memory manifest (the tests and some callers bypass
        # _load_manifest) must also fail closed here.
        _ep_tail = str(ep).rsplit('@', 1)[-1].strip() if '@' in str(ep) else ''
        Ln = _record_int(_ep_tail)
        if Ln is None or Ln < 2:
            raise SystemExit(
                f'[p5t stats] REFUSE: comparison {name!r} endpoint={ep!r} '
                'does not end in an exact by-length integer >= 2')
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
            c = _by_len_cell(rec, Ln)
            return float(c['ppl']) if isinstance(c, dict) and L.ppl_is_usable(c.get('ppl')) and isinstance(c.get('truncated'), (bool, np.bool_)) and not bool(c.get('truncated')) else None
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
        # Zero paired deltas do not change the sign-flip null distribution, so
        # the resolution floor/power describe the NONZERO pairs only.
        _n_eff_nonzero = int(sum(1 for _d in dl if _d != 0.0))
        _n_min_raw = (manifest.get('statistics') or {}).get('min_seeds_for_confirmation')
        if _n_min_raw is None:
            _n_min = len(_manifest_seed_ids(manifest))
        else:
            _n_min = _record_int(_n_min_raw)
            if _n_min is None or _n_min < 2 or _n_min > len(_manifest_seed_ids(manifest)):
                raise SystemExit(
                    f'[p5t stats] REFUSE: statistics.min_seeds_for_confirmation={_n_min_raw!r} '
                    'is not an exact integer in [2, number of frozen seeds]')
        res['seeds'] = usable
        res['n_effective'] = int(_n_eff)
        res['n_nonzero_effective'] = int(_n_eff_nonzero)
        res['coverage_ok'] = bool(_n_eff == len(manifest['seeds']))
        res['missing_seeds'] = sorted(set(int(s) for s in manifest['seeds']) - set(int(s) for s in usable))
        res['dropped_pairs'] = dropped
        res['min_seeds_for_confirmation'] = _n_min
        # Diagnostics must use the n that actually entered THIS comparison,
        # so p_floor / min_attainable_p_holm / power describe the test that
        # really ran.
        _fl_eff = L.signflip_p_floor(_n_eff_nonzero)
        if math.isfinite(_fl_eff):
            res['p_floor_effective'] = _fl_eff
            res['min_attainable_p_holm_effective'] = L.holm_min_attainable_p(_n_eff_nonzero, len(manifest['primary_comparisons']), _alpha)
            res['power_first_holm_gate_effective'] = {f'q={q:.2f}': L.signflip_power(_n_eff_nonzero, len(manifest['primary_comparisons']), q, _alpha) for q in (0.8, 0.85, 0.9, 0.95)}
        # Parameter counts are exact positive integers; bool/fractional or
        # lossy large values are malformed provenance, not a matched arm.
        # Every usable PAIR must expose BOTH counts so the parameter
        # comparison is verifiable.
        _params_a = [L._as_seed_int(recs_a[s].get('params')) for s in usable]
        _params_b = [L._as_seed_int(recs_b[s].get('params')) for s in usable]
        _param_complete = bool(usable) and all(
            p is not None and p > 0 for p in _params_a + _params_b)
        if _param_complete:
            _am, _bm = float(np.mean(_params_a)), float(np.mean(_params_b))
            _rel = abs(_am - _bm) / max(_am, _bm, 1.0)
            res['param_matched'] = bool(_rel <= 0.01)
            res['param_rel_diff'] = float(_rel)
            res['params_a'] = _am
            res['params_b'] = _bm
        else:
            res['param_matched'] = None
            res['param_rel_diff'] = None
            res['params_unverifiable_seeds'] = [
                int(s) for s, pa, pb in zip(usable, _params_a, _params_b)
                if pa is None or pa <= 0 or pb is None or pb <= 0]
        # Every usable PAIR must carry a digest for both arms.  A union over
        # the non-null values would let one missing digest disappear while the
        # remaining 23 records still attest the same init, confirming a
        # comparison whose initialization was not actually verified per seed.
        _iv = []
        _init_complete = bool(usable)
        for _s in usable:
            _ia = L._content_fp_text(recs_a[_s].get('init_core_fp'))
            _ib = L._content_fp_text(recs_b[_s].get('init_core_fp'))
            if _ia is None or _ib is None:
                # A boolean/empty/non-text digest is not a shared-trunk
                # initialization identity; both arms must carry a text digest.
                _init_complete = False
                continue
            _iv.extend((_ia, _ib))
        res['init_core_fp_recorded'] = _init_complete
        res['same_init_core_fp'] = bool(_init_complete and len(set(_iv)) <= 1)
        _act = []
        for _recs in (recs_a, recs_b):
            for s in usable:
                _pbd = _recs[s].get('param_breakdown')
                if isinstance(_pbd, dict):
                    _act_i = L._as_seed_int(_pbd.get('active'))
                    if _act_i is not None and _act_i > 0:
                        _act.append(_act_i)
        res['active_params_mean'] = (float(np.mean(_act)) if _act else None)
        res['data_fp'] = unified_fp
        res['manifest_sha256'] = manifest.get('manifest_sha256')
        res['delta_definition'] = f'{a} - {b} (paired per seed, test split)'
        res['expected_sign'] = exp_sign
        if exp_sign in ('positive', 'negative') and math.isfinite(res.get('mean', float('nan'))):
            # A mean of exactly 0 has no direction: treat it as NOT matching
            # the pre-registered sign (the old `(mean > 0) == (exp == 'positive')`
            # encoding made mean==0 spuriously satisfy expected_sign='negative').
            if res['mean'] == 0.0:
                res['direction_ok'] = False
                res['direction_note'] = 'mean delta is exactly zero — no observed direction, cannot match any pre-registered sign'
            else:
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
        # Booleans are flags, not probabilities.  float(False) == 0.0 would
        # become the strongest possible p and let a malformed comparison
        # shrink the family while appearing significant.
        if isinstance(p, (bool, np.bool_)):
            p = float('nan')
        try:
            p = float(p)
        except (TypeError, ValueError, OverflowError):
            p = float('nan')
        _fam.append((k, p if math.isfinite(p) and 0.0 <= p <= 1.0 else 1.0))
    if _fam:
        stats['holm'] = L.holm_adjust(_fam, alpha=_alpha)
    _rej = [k for k, v in stats['holm'].items() if v.get('p_holm') is not None and float(v['p_holm']) < _alpha]
    # the pre-registered decision rule requires BOTH p_holm < alpha AND the
    # observed direction to match the validation-panel estimate (a null
    # expected_sign marks a non-directional mechanism readout).  The strict `<`
    # matches the manifest's `p_holm < 0.05` (the generic holm_adjust uses `<=`).
    # The pre-registered decision rule (manifest v3) requires p_holm < alpha
    # AND the observed direction AND `param_matched is not false` (True or
    # unverifiable — None; only a measured >1% mismatch blocks) AND at least
    # `min_seeds_for_confirmation` complete pairs.  `coverage_ok` (all 12
    # seeds) is reported but is not an all-or-nothing veto: one failed
    # L4096 cell may not annihilate an otherwise complete comparison.
    # `same_init_core_fp` / `init_core_fp_recorded` are NOT decision inputs:
    # the manifest records the init digest as reported instrumentation
    # ("reported and compared"), so they stay in `not_confirmed_reasons`
    # for transparency but never gate the CONFIRMED set.
    _conf = [k for k in _rej if _frozen_rule_confirms(stats['holm'].get(k, {}),
                                                       stats['comparisons'].get(k, {}),
                                                       _alpha,
                                                       len(manifest['seeds']))]
    stats['confirmed'] = _conf
    stats['significant_nondirectional'] = [k for k in _rej if stats['comparisons'].get(k, {}).get('direction_ok') is None]
    stats['not_confirmed_reasons'] = {k: {'param_matched': stats['comparisons'].get(k, {}).get('param_matched'),
                                          'coverage_ok': stats['comparisons'].get(k, {}).get('coverage_ok'),
                                          'n_effective': stats['comparisons'].get(k, {}).get('n_effective'),
                                          'min_seeds_for_confirmation': stats['comparisons'].get(k, {}).get('min_seeds_for_confirmation'),
                                          'direction_ok': stats['comparisons'].get(k, {}).get('direction_ok'),
                                          'same_init_core_fp': stats['comparisons'].get(k, {}).get('same_init_core_fp'),
                                          'init_core_fp_recorded': stats['comparisons'].get(k, {}).get('init_core_fp_recorded')}
                                      for k in _rej if k not in _conf}
    # DISCLOSURE: the implemented decision rule is STRICTER than the manifest
    # The implemented rule is EXACTLY the manifest decision_rule (v3/v4
    # wording, unchanged): "p_holm < alpha AND direction matches AND
    # param_matched is not false AND >= min_seeds_for_confirmation complete
    # pairs".  `param_matched is not false` means True or the unverifiable
    # None both pass — only a measured >1% mismatch blocks — and the
    # init-core digests are reported instrumentation, never decision inputs.
    # An earlier implementation was silently stricter (param_matched is True
    # + init-digest gates + a coverage_ok veto in the report fallback); that
    # drift was reverted so the confirmatory label means exactly what the
    # pre-registration says.  The disclosure block stays so any future
    # divergence is visible next to every 'confirmed' list.
    stats['decision_rule_applied'] = {
        'manifest_rule': (manifest.get('statistics') or {}).get('decision_rule'),
        'implementation_gates': ['p_holm < alpha (strict <)', 'direction_ok is True',
                                 'param_matched is not False (True or unverifiable None; a measured >1% mismatch blocks)',
                                 'n_effective >= min_seeds_for_confirmation'],
        'excluded_gates': ['coverage_ok (reported, not required — v3 removed the all-seeds veto)',
                           'same_init_core_fp / init_core_fp_recorded (reported instrumentation, never decision inputs)'],
        'note': 'The implemented rule matches the pre-registered manifest wording exactly (one shared helper, v12_supp._frozen_rule_confirms, is used by finaltest_stats and the report fallback).',
    }
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
        try:
            _alpha = float(((ft.get('manifest') or {}).get('statistics') or {}).get('alpha', 0.05))
        except (TypeError, ValueError, OverflowError):
            _alpha = 0.05
        A('| 冻结主比较（test split，配对） | n | 覆盖完整 | 参数匹配 | Δ mean±std | 文档 Δ 95% CI | d_z | p (exact) | p 下限 | Holm p | Holm 拒绝 | 方向符合 |')
        A('|---|---|---|---|---|---|---|---|---|---|---|---|')
        holm = ft.get('holm') or {}
        for k, v in comps.items():
            if 'omitted' in v:
                A(f'| {k} | 0 | — | — | —（{v["omitted"]}） | — | — | — | — | — | — | — |')
                continue
            _h = holm.get(k) or {}
            _hp = f'{_h.get("p_holm", float("nan")):.3f}' if _h.get('p_holm') is not None else '—'
            _hr = '✔' if (_h.get('p_holm') is not None and float(_h['p_holm']) < _alpha) else '✘'
            _dir = v.get('direction_ok')
            _ds = '—（非方向性）' if _dir is None else ('✔' if _dir else '✘')
            _dsm = v.get('doc_sampling') or {}
            _dci = _dsm.get('per_doc_delta_ppl_ci95')
            _dcitxt = f'[{_dci[0]:+.2f}, {_dci[1]:+.2f}]' if isinstance(_dci, (list, tuple)) and len(_dci) == 2 else '—'
            A(f'| {k} | {v["n"]} | {v.get("coverage_ok")} | {v.get("param_matched")} | {v["mean"]:+.2f} ± {v.get("std", 0):.2f} | {_dcitxt} | {v.get("cohens_dz", float("nan")):+.2f} | {v["p_exact_signflip"]:.3f} | {v.get("p_floor", float("nan")):.3f} | {_hp} | {_hr} | {_ds} |')
        A('')
        _conf = ft.get('confirmed')
        if _conf is None:
            # Legacy stats files without a `confirmed` key: re-derive the
            # pre-registered decision rule EXACTLY as `finaltest_stats` does
            # (manifest v3): strict p_holm < alpha (not holm_adjust's `<=`
            # reject flag), direction_ok is True, `param_matched is not false`
            # (None = unverifiable does not block), and n >= the declared
            # min_seeds_for_confirmation floor.  `coverage_ok` is reported,
            # NOT required — v3 explicitly removed the all-12-seeds veto —
            # and the init-core digests are instrumentation, not decision
            # inputs.
            _conf = [k for k, v in holm.items()
                     if _frozen_rule_confirms(v, comps.get(k, {}), _alpha)]
        if _conf:
            A(f'**确认结果**：{len(_conf)}/{len(comps)} 个冻结主比较在 Holm 校正后成立**且方向符合预注册**（{sorted(_conf)}）——这是预注册规则下的确认性证据。')
        else:
            A(f'**确认结果**：没有冻结主比较同时满足「Holm 校正后 p < {_alpha:g}」与「方向符合预注册」。按预注册的决策规则，相应主张**不获确认**；探索性面板的正向读数保持「探索性」标签，不得写成结论。')
        _nd = ft.get('significant_nondirectional') or []
        if _nd:
            A('')
            A(f'**非方向性机制读数**：{sorted(_nd)} 在 Holm 校正后达到 p<{_alpha:g}，但冻结清单未给方向（expected_sign=null），只能作为机制读数，不能计入「方向符合预注册」的确认集合。')
        if ft.get('min_attainable_p_holm') is not None:
            A('')
            A(f"（诊断：n 个种子与 m 元 Holm 族下，最小可达 Holm p = {ft['min_attainable_p_holm']:.4f}。加载期功效门槛已拒绝「数学上不可能确认」的清单；若此处仍 > {_alpha:g}，说明有绕过门槛的旧状态，需回查清单与记录的来源。）")
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
    # Fail before any training starts and before P5T can touch test.  This is
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
            # a REFUSE (e.g. freeze-manifest gate) is a controlled stop
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

    A plan that only updates the stamp would let P5T burn the
    one-shot test access, then admit 0/48 records.  This check closes that
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
    # P5T fills a missing distractor-probe field from FINALTEST_CFG, so the
    # pre-flight must validate those DEFAULTS too.  Otherwise a manifest that
    # omits probe eval_lens/target/n_seq can pass here and only fail when the
    # probe indexes outside the frozen slice.
    _probe_plan = _eval.get('distractor_probe') or {}
    _probe_mode = str(_probe_plan.get('mode') or 'uniform')
    if _probe_mode not in V10.PROBE_MODES:
        errs.append(f'eval.distractor_probe.mode={_probe_mode!r} is not one of {sorted(V10.PROBE_MODES)}')
    _probe_rhos = [float(x) for x in
                   (_probe_plan.get('rhos') or V10.PROBE_MODES.get(_probe_mode, []))]
    if _probe_mode in ('mask', 'delete', 'doc') and any(abs(x - 1.0) > 1e-9 for x in _probe_rhos):
        errs.append(f'eval.distractor_probe.mode={_probe_mode!r} is all-or-nothing and only admits rho=1.0 (got {_probe_rhos})')
    _probe_lens = [int(x) for x in (_probe_plan.get('eval_lens') or [2048, 4096])]
    _probe_target = int(_probe_plan.get('target', FINALTEST_CFG.get('target', 512)))
    # run_finaltest now defaults an omitted probe n_seq to the FROZEN row
    # count (the probe must score the same rows as the primary panel); the
    # pre-flight must validate that same effective value.
    _probe_nseq = int(_probe_plan.get(
        'n_seq', _nseq if _nseq > 0 else FINALTEST_CFG.get('n_seq', 16)))
    if _probe_nseq <= 0:
        errs.append(f'distractor probe n_seq={_probe_nseq} must be positive')
    elif _nseq > 0 and _probe_nseq > _nseq:
        errs.append(f'distractor probe n_seq={_probe_nseq} exceeds the frozen test row count {_nseq} — the probe would index outside the frozen slice')
    if not _probe_lens:
        errs.append('distractor probe eval_lens is empty; the probe has no evaluation length')
    else:
        if _lens and max(_probe_lens) > max(_lens):
            errs.append(f'distractor probe eval_lens={_probe_lens} extends beyond the frozen test slice width max(by_length_ppl)={max(_lens)}')
        if _probe_target < 0 or _probe_target >= min(_probe_lens):
            errs.append(f'distractor probe target={_probe_target} must be non-negative and smaller than every probe eval_len {sorted(_probe_lens)}')
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
                                     'n_seq': int(_nseq), 'ppl_per_seq': [100.0] * int(_nseq),
                                     'truncated': False, 'eval_span': int(Ln)}
                           for Ln in _lens},
                'data_fp': 'plan-test-slice-fp',
                'train_fp': (_frozen_train_fp(m) or 'plan-train-fp'),
                '_code': L.code_stamp(), 'manifest_sha256': m.get('manifest_sha256'),
                'recipe': _recipe, 'train_knobs': _tk, 'n_seq': int(_nseq),
                # the real parameter count computed for this arm's frozen shape
                # above (never a hard-coded constant): admission does not check
                # params, but the param-matched contract reads them from here.
                'params': (_pred[arm].get('params') if isinstance(_pred[arm].get('params'), int) else None),
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
        if _Ln is None:
            errs.append(f'comparison {comp.get("name")!r} has a malformed endpoint {comp.get("endpoint")!r} — the analysis cannot resolve a length')
        elif _lens and _Ln not in _lens:
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
        raise SystemExit('[v12 plan] FAILED: ' + '; '.join(errs)
                         + ' — correct the plan/records BEFORE any P5C training or P5T test access')
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
    # Smoke must never write into the frozen wt103_cache: a differently
    # tokenised tok_v5_*.json could make a later cold P5T build fail AFTER the
    # test read.  An isolated cache under the smoke outdir keeps the frozen
    # cache clean.  The override is UNCONDITIONAL: `setdefault` would let an
    # exported WT103_CACHE_DIR (e.g. pointing at the frozen cache) silently
    # redirect the smoke's cold build into exactly the directory this guard
    # exists to protect.
    os.environ['WT103_CACHE_DIR'] = os.path.join('results_smoke_v12', 'wt103_cache')
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
    _MODES = ('full', 'smoke', 'plan', 'power', 'analysis', 'report', 'phase', 'finaltest')
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
        elif mode == 'power':
            run_power_check(dict(FINALTEST_CFG, manifest=(sys.argv[2] if len(sys.argv) > 2 else FREEZE_MANIFEST)))
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
            git_push('v12: PARTIAL — interrupted, see log (re-run resumes)')
        except Exception:
            traceback.print_exc()
        if mode == 'full':
            schedule_shutdown(2400)
        sys.exit(1)
